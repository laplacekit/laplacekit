r"""Function-space Laplace: a GP over the model's outputs with kernel $J \Sigma_0 J^\top$.

$J$ is the Jacobian of the outputs at the fitted parameters, and $\Sigma_0$ the prior's
covariance.

`fs.laplace(model, data, likelihood=..., method=fs.SubsetOfData(m))` runs the same probe,
passes and generators as `lk.laplace` and returns a function-space posterior. `gp.at(x)` gives the
Gaussian over the outputs at `x`, and `gp.log_marglik()` the GP evidence of the subset.
"""

import copy
import dataclasses
import math
from abc import abstractmethod
from collections.abc import Iterable, Mapping
from typing import (
    Any,
    Generic,
    Literal,
    Protocol,
    Self,
    TypeVar,
    cast,
    get_args,
    runtime_checkable,
)

import torch
from torch import Tensor, nn
from torch.func import jacrev, vmap

import laplacekit.core as core
from laplacekit import priors, subsets
from laplacekit.core import (
    Batch,
    Capabilities,
    Done,
    FitContext,
    FunctionPosterior,
    HasMask,
    Inputs,
    Likelihood,
    ModelFn,
    OutputGaussian,
    Params,
    Pass,
    PerInputCov,
    PriorWithCovariance,
    Requirements,
    Selection,
    StructureError,
    Subset,
    Targets,
    as_pytree,
    chain,
    curvature_scale,
    fit_curvature,
    output_hessian,
    pytree,
    spawn_generator,
    stat_dtype,
    static,
    tree_map,
)
from laplacekit.data import Data, as_data
from laplacekit.likelihoods import Gaussian, _can_mask

__all__ = [
    "MAX_SUBSET_ENTRIES",
    "MAX_SUBSET_JACOBIAN_ENTRIES",
    "FunctionSpaceMethod",
    "SubsetGP",
    "SubsetOfData",
    "cov_rows",
    "is_shared",
    "jacobian",
    "laplace",
    "rows_of",
]

MAX_SUBSET_ENTRIES = 16384
r"""Most output entries $m E$ a subset GP conditions on: its Cholesky is $m E \times m E$.

For $m$ inputs with $E$ output entries each.
"""

MAX_SUBSET_JACOBIAN_ENTRIES = 2**28
r"""Most entries $m E P$ of the Jacobian a subset GP keeps (1 GiB in float32).

For $m$ inputs with $E$ output entries each and $P$ parameters.
"""

# The jitter in rᵀ (K + jitter I)⁻¹ r, laplace-torch's `FunctionalLaplace.scatter(eps)`.
_SCATTER_JITTER = 1e-5

_Structure = Literal["joint", "diagonal", "independent"]

_B_co = TypeVar("_B_co", covariant=True)
_Pr = TypeVar("_Pr", bound=PriorWithCovariance)

# Explicit variance, as for `LaplacePosterior`: the likelihood's type, covariant, so a GP
# with a Gaussian is assignable to `SubsetGP[Pr]`.
_L_co = TypeVar("_L_co", bound=Likelihood, covariant=True, default=Likelihood)


@runtime_checkable
class FunctionSpaceMethod(Protocol[_B_co]):  # explicit variance, as for `Staged`
    r"""A function-space method: passes over the data that end in the posterior itself.

    The posterior is a GP whose kernel contains the prior's covariance ($J \Sigma_0 J^\top$),
    so `plan` takes the prior. Passes, statistics and generators are those of a weight-space
    `Staged`. `fs.laplace` swaps the real likelihood in with `copy.replace`, so the posterior
    must be a dataclass with a `likelihood` field.
    """

    @property
    @abstractmethod
    def requires(self) -> Requirements:
        """What the method needs from the model and the likelihood."""
        ...

    @abstractmethod
    def plan(self, params: Params, ctx: FitContext, prior: PriorWithCovariance, /) -> Pass[_B_co]:
        r"""The method's passes, before any data.

        Args:
            params: The fitted params, where the model is linearized.
            ctx: The fit's context.
            prior: The prior over `params`. Its covariance $\Sigma_0$ enters the kernel.

        Returns:
            The first `Pass` of a chain built with `chain(...)`, ending in `Done` with the
            posterior.

        Raises:
            CapabilityError: Implementations may raise it when the model or likelihood
                lacks something the method needs.
            StructureError: Implementations may raise it when the params or layout don't
                fit the method.
        """
        ...


# --- helpers for function-space methods ------------------------------------------------


def is_shared(leaf: object, /) -> bool:
    """Whether an inputs leaf is shared by every example: a non-tensor or a 0-dim tensor.

    The other leaves carry one entry per example along their leading axis.

    Args:
        leaf: A leaf of an inputs pytree.

    Returns:
        Whether `leaf` is shared.
    """

    return not isinstance(leaf, Tensor) or leaf.ndim == 0


def rows_of(inputs: Inputs, index: Tensor, /) -> Inputs:
    """The examples `index` of `inputs`.

    Args:
        inputs: Inputs with one entry per example along the leading axis of each leaf that
            is not shared (`is_shared`).
        index: The indices of the examples to keep, moved to each leaf's device.

    Returns:
        The inputs at `index`. Shared leaves stay as they are.
    """

    def pick(t: Any) -> Any:
        return t if is_shared(t) else t[index.to(t.device)]

    return tree_map(pick, inputs)


def jacobian(model: ModelFn, params: Params, inputs: Inputs, dtype: torch.dtype, /) -> Tensor:
    r"""The Jacobian of the flattened outputs with respect to `params`.

    One `jacrev` per example, vmapped over the examples.

    Args:
        model: The model as a function of the params.
        params: The parameters. Their order sets the order of the columns.
        inputs: $N$ examples. Shared leaves (`is_shared`) go to every example as they are.
        dtype: The dtype of the result.

    Returns:
        The Jacobian, $[N E, P]$, for $E$ output entries per example and $P$ parameters.
        Row $n E + e$ is output entry $e$ of example $n$.
    """

    names = list(params)
    shared = tree_map(is_shared, inputs)
    in_dims = tree_map(lambda sh: None if sh else 0, shared)

    def unsq(t: Any, sh: bool) -> Any:
        return t if sh else t.unsqueeze(0)

    def one(x: Inputs) -> Tensor:
        xb = tree_map(unsq, x, shared)

        def fwd(p: Params) -> Tensor:
            return cast(Tensor, model(p, xb)).reshape(-1)

        jac = cast(Mapping[str, Tensor], jacrev(fwd)(dict(params)))

        return torch.cat([jac[k].reshape(jac[k].shape[0], -1) for k in names], dim=1)

    jac = vmap(one, in_dims=(in_dims,))(inputs)

    return jac.reshape(-1, jac.shape[-1]).to(dtype)


def cov_rows(prior: PriorWithCovariance, like: Params, rows: Tensor, /) -> Tensor:
    r"""$\Sigma_0 r$ for each row $r$ of `rows`, through `prior.cov_mv` under `vmap`.

    Args:
        prior: The prior. $\Sigma_0$ is its covariance.
        like: Gives the parameter names, shapes and order of the $P$ columns.
        rows: The rows, $[k, P]$.

    Returns:
        The products as rows, $[k, P]$, in the dtype of `rows`.

    Raises:
        StructureError: `prior.cov_mv` refuses the parameters (a `Layerwise` prior whose
            blocks differ from theirs, say).
    """

    k = rows.shape[0]
    sizes = [p.numel() for p in like.values()]
    parts = torch.split(rows, sizes, dim=1)
    v = {n: x.reshape(k, *p.shape) for (n, p), x in zip(like.items(), parts, strict=True)}

    out = vmap(prior.cov_mv)(v)

    return torch.cat([out[n].reshape(k, -1).to(rows.dtype) for n in like], dim=1)


# --- the posterior -----------------------------------------------------------------------


@pytree
class SubsetGP(Generic[_Pr, _L_co]):  # noqa: UP046
    r"""A GP over the outputs, conditioned on the outputs at a subset of $m$ training inputs.

    The kernel is

    $$k(x, x') = J(x) \, \Sigma_0 \, J(x')^\top,$$

    with $J$ the model's Jacobian at the MAP $\theta_*$ and $\Sigma_0$ the prior's
    covariance. The noise precision at the subset is the likelihood's output Hessian there.
    With every training input this is the linearized Laplace posterior with the full dense
    GGN. The output Hessian is not rescaled by $N / m$, so a smaller subset gives a wider
    posterior than the full data would.

    The mean is $f(x; \theta_*)$: the Laplace mean is the MAP for any prior location, and
    the GP view changes only the covariance.

    `jac` and `root` are fixed at fit time and depend on the model and the MAP only. `at`
    and `log_marglik` read the prior and the likelihood's `curvature_scale` on every call,
    so tuning them never refits. Both raise if one of the module's fixed parameters changed
    in place since the fit (see `check_versions`).

    Args:
        model: The model as a function of the params.
        likelihood: The real likelihood. Its tensors (`sigma`) are leaves, so they can be
            tuned.
        mean: The MAP $\theta_*$, a parameter mapping.
        prior: The prior. It gives $\Sigma_0$ and the location $\mu_0$.
        caps: The model's capabilities.
        jac: The Jacobian at the subset, $[m E, P]$, for $E$ output entries per input and
            $P$ parameters.
        root: $A$ with $A A^\top$ the output Hessian of the unit likelihood, $[m, E, E]$.
        n_subset: The number of subset rows $m$.
        inputs: The subset's rows. Shared leaves (0-dim tensors and non-tensors) are kept
            by reference, and `check_versions` does not cover them.
        targets: The subset's targets, or `None` when the data had none.
        n_seen: The number of rows the pass saw.
        structure: That of `SubsetOfData`. With `"independent"` the kernel is block diagonal
            over the outputs, so `at` gives diagonal $E \times E$ blocks.
        versions: Version counters of the module's fixed parameters at fit time. The mean
            is the posterior's own copy, so it has none.
    """

    model: ModelFn = static()
    likelihood: _L_co  # a child: its tensors (sigma) are leaves to tune
    mean: Params
    prior: _Pr
    caps: Capabilities = static()
    jac: Tensor
    root: Tensor
    n_subset: int = static()
    inputs: Inputs
    targets: Targets | None
    n_seen: int = static()
    structure: _Structure = static(default="joint")
    versions: Mapping[str, int] = static(default_factory=dict[str, int])

    def __repr__(self) -> str:
        size = sum(p.numel() for p in self.mean.values())

        return (
            f"SubsetGP(n_subset={self.n_subset}, structure={self.structure!r}, parameters={size}, "
            f"prior={self.prior!r})"
        )

    @property
    def requires(self) -> Requirements:
        """Needs vjp and vmap, for the Jacobian at new inputs."""

        return Requirements(vjp=True, vmap=True)

    def check_versions(self) -> None:
        """Raise `RuntimeError` if a fixed parameter of the module changed in place since the fit.

        `jac` and `root` used its old value, so the GP no longer describes the model. `at`
        calls this first.
        """

        module = self.model.module
        live = dict(module.named_parameters()) if module is not None else {}

        for name, version in self.versions.items():
            now = live[name]._version if name in live else None

            if now != version:
                raise RuntimeError(
                    f"fixed parameter {name!r} changed in place since the fit (version "
                    f"{version}, now {now}); the Jacobian and the output Hessians used its old "
                    "value; refit the posterior"
                )

    def at(self, inputs: Inputs, /) -> OutputGaussian:
        r"""The Gaussian over the outputs at `inputs`, with one $E \times E$ block per input.

        Per input $x$, the covariance is

        $$K_{xx} - B^\top B, \qquad B = L^{-1} W^\top K_{mx}, \qquad
        L L^\top = I + W^\top K_{mm} W,$$

        where $K_{ab}$ is the kernel between $a$ and $b$, $m$ stands for the subset, and $W$
        is block diagonal with the roots $A$ of the output Hessians. This form needs no
        inverse of the output Hessian, so a singular one (softmax) is fine. The algebra runs
        in `stat_dtype` of the outputs. With a `HasMask` likelihood the masked entries of
        the subset drop out, as in `log_marglik`, also after a likelihood swap.

        Args:
            inputs: A batch of inputs.

        Returns:
            The output Gaussian of the batch.

        Raises:
            CapabilityError: The model lacks vjp or vmap.
            NotImplementedError: The model does not return a single tensor.
            StructureError: The likelihood's `curvature_scale` has more than one element,
                its mask is not bool or not shaped like the leading axes of the outputs, or
                `prior.cov_mv` refuses the parameters.
        """

        self.caps.check(self.requires, "SubsetGP.at")
        self.check_versions()

        inputs = as_pytree(inputs)  # a `BatchEncoding` as a dict, so `vmap` sees its tensors
        f = self.model(self.mean, inputs)

        if not isinstance(f, Tensor):
            raise NotImplementedError("SubsetGP supports models returning a single tensor")

        dtype = stat_dtype(f.dtype)
        n, event = f.shape[0], tuple(f.shape[1:])
        e = math.prod(event)

        jm = self.jac.to(dtype)
        js = jacobian(self.model, self.mean, inputs, dtype)
        sm = cov_rows(self.prior, self.mean, jm)  # rows of Σ₀ Jₘᵀ
        ss = cov_rows(self.prior, self.mean, js)

        root = self._masked(self._root(dtype))
        m, em = root.shape[0], root.shape[1]

        k_mm = self._kernel(jm @ sm.mT, m, m, em)
        k_ms = self._kernel(sm @ js.mT, m, n, e)
        k_ss = torch.einsum("nep,nfp->nef", js.reshape(n, e, -1), ss.reshape(n, e, -1))

        if self.structure == "independent":
            k_ss = k_ss * torch.eye(e, dtype=dtype, device=k_ss.device)

        chol = _chol_wkw(k_mm, root)

        wk = torch.einsum("iea,ien->ian", root, k_ms.reshape(m, em, n * e))
        b = torch.linalg.solve_triangular(chol, wk.reshape(m * em, n * e), upper=False)
        b = b.reshape(m * em, n, e)

        cov = k_ss - torch.einsum("kne,knf->nef", b, b)

        return OutputGaussian(f, PerInputCov(0.5 * (cov + cov.mT), event))

    def log_marglik(self) -> Tensor:
        r"""The GP evidence of the subset, after laplace-torch's `FunctionalLaplace`.

        Let $K = J \Sigma_0 J^\top$ at the subset (through `prior.cov_mv`), $f$ the outputs
        at the MAP $\theta_*$, and $r = J (\theta_* - \mu_0)$ with $\mu_0$ the prior's
        location.

        - `lk.Gaussian`: the exact evidence of the linearized model,
          $\log \mathcal{N}(y; f - r, K + \sigma^2 I)$.
        - Any other likelihood: R&W (3.44),

          $$\sum_i \log p(y_i \mid f_i) - \tfrac{1}{2} \Big(r^\top (K + 10^{-5} I)^{-1} r
          + \log \det (I + W^\top K W)\Big),$$

          with $W$ the roots of the output Hessians of the real likelihood. The $10^{-5}$
          is laplace-torch's `scatter(eps)`.

        With `structure="independent"` $K$ is block diagonal over the outputs, so both are
        sums over the outputs. With a `HasMask` likelihood the masked output entries drop
        out of every term: the evidence is that of the observed entries only.
        Differentiable in the prior's and the likelihood's tensors.

        Returns:
            The log evidence, a 0-dim tensor in `stat_dtype` of the outputs.

        Raises:
            CapabilityError: The model lacks vjp or vmap.
            ValueError: The pass saw more rows than it kept (the evidence of a subset is
                not the data's), or the data had no targets.
            RuntimeError: A fixed parameter changed in place since the fit
                (`check_versions`).
            StructureError: The model's outputs are not one tensor, the `Gaussian`'s
                targets are not shaped like $f$, the likelihood's `curvature_scale` has more
                than one element, its mask or `log_prob` is badly shaped, or
                `prior.cov_mv` refuses the parameters.
        """

        self.caps.check(self.requires, "SubsetGP.log_marglik")

        if self.n_seen != self.n_subset:
            raise ValueError(
                "SubsetGP.log_marglik: the evidence of a subset of m < N rows is not the "
                f"data's evidence (seen {self.n_seen} rows, kept n_subset={self.n_subset}); "
                f"fit with SubsetOfData(n_subset={self.n_seen}) or more"
            )

        if self.targets is None:
            raise ValueError("SubsetGP.log_marglik needs targets, but the data had none")

        self.check_versions()

        inputs = as_pytree(self.inputs)
        f = self.model(self.mean, inputs)

        if not isinstance(f, Tensor):
            raise StructureError(
                f"SubsetGP.log_marglik supports models returning a single tensor, got "
                f"{type(f).__name__}"
            )

        dtype = stat_dtype(f.dtype)
        root = self._root(dtype)
        m, e = root.shape[0], root.shape[1]

        jm = self.jac.to(dtype)
        k = self._kernel(jm @ cov_rows(self.prior, self.mean, jm).mT, m, m, e)

        loc = self.prior.location(self.mean)
        delta = torch.cat([(self.mean[n] - loc[n]).reshape(-1) for n in self.mean])
        r = jm @ delta.to(dtype)

        # Masked entries leave every term: K and r keep the observed entries only, and the
        # roots lose their masked rows (W Wᵀ is then H with those rows and columns zero).
        observed = _observed_entries(self.likelihood, f, inputs)
        k_obs, r_obs = k, r

        if observed is not None:
            idx = observed.nonzero().squeeze(1).to(k.device)
            k_obs, r_obs = k[idx][:, idx], r[idx]
            root = root * observed.reshape(m, e, 1).to(root)

        if isinstance(self.likelihood, Gaussian):
            y = cast(Tensor, self.targets)

            return _gaussian_evidence(self.likelihood, f, y, r, k_obs, observed)

        log_lik = core.log_prob(self.likelihood, f, self.targets, inputs=inputs).sum()

        jittered = k_obs.clone()
        jittered.diagonal().add_(_SCATTER_JITTER)
        z = torch.linalg.solve_triangular(
            torch.linalg.cholesky(jittered), r_obs.unsqueeze(1), upper=False
        )

        chol = _chol_wkw(k, root)
        logdet = 2 * chol.diagonal().log().sum()

        return log_lik.to(dtype) - 0.5 * (z.square().sum() + logdet)

    def _kernel(self, k: Tensor, rows: int, cols: int, e: int) -> Tensor:
        """`k` ([rows E, cols E]) as the structure's kernel.

        Block diagonal over the outputs with `"independent"`, else as it is.
        """

        if self.structure != "independent":
            return k

        eye = torch.eye(e, dtype=k.dtype, device=k.device).reshape(1, e, 1, e)

        return (k.reshape(rows, e, cols, e) * eye).reshape(k.shape)

    def _root(self, dtype: torch.dtype) -> Tensor:
        """The roots of the output Hessians of the real likelihood: sqrt(scale) A."""

        root = self.root.to(dtype)
        scale = curvature_scale(self.likelihood)

        if scale is None:
            return root

        # Scaling the columns of A is right for one factor only.
        if scale.numel() != 1:
            raise StructureError(
                f"SubsetGP needs a one-element curvature_scale, but "
                f"{type(self.likelihood).__name__} gives shape {tuple(scale.shape)}"
            )

        return root * scale.reshape(()).sqrt().to(dtype)

    def _masked(self, root: Tensor) -> Tensor:
        """`root` without the rows of the subset's masked entries.

        Masking is idempotent, so roots already masked at fit time are unchanged.
        """

        # A likelihood that never masks (a built-in without `mask_fn`): skip the forward pass.
        if not _can_mask(self.likelihood):
            return root

        inputs = as_pytree(self.inputs)
        f = cast(Tensor, self.model(self.mean, inputs))
        observed = _observed_entries(self.likelihood, f, inputs)

        if observed is None:
            return root

        m, e = root.shape[0], root.shape[1]

        return root * observed.reshape(m, e, 1).to(root)


def _chol_wkw(k: Tensor, root: Tensor) -> Tensor:
    """The Cholesky factor of I + Wᵀ K W, with W block diagonal with the roots ([m, E, E])."""

    m, e = root.shape[0], root.shape[1]

    # Wᵀ K W, block by block: A_iᵀ K_ij A_j.
    left = torch.einsum("iea,iejf->iajf", root, k.reshape(m, e, m, e))
    wkw = torch.einsum("iajf,jfb->iajb", left, root).reshape(m * e, m * e)

    # I + (Wᵀ K W + its transpose) / 2, with one [m E, m E] temporary.
    sym = wkw + wkw.mT
    sym.mul_(0.5).diagonal().add_(1)

    return torch.linalg.cholesky(sym)


def _observed_entries(lik: Likelihood, f: Tensor, inputs: Inputs) -> Tensor | None:
    """The output entries a `HasMask` likelihood counts, flat over [m E] (bool).

    None when every entry counts. The mask covers the leading axes of f, as in the fit.
    """

    mask = lik.mask(f, inputs) if isinstance(lik, HasMask) else None

    if mask is None:
        return None

    if mask.dtype != torch.bool or mask.ndim > f.ndim or mask.shape != f.shape[: mask.ndim]:
        raise StructureError(
            f"{type(lik).__name__}.mask must be a bool tensor shaped like the leading axes of "
            f"f {tuple(f.shape)}, got {mask.dtype} of shape {tuple(mask.shape)}"
        )

    mask = mask.reshape(*mask.shape, *([1] * (f.ndim - mask.ndim)))

    return mask.expand_as(f).reshape(-1)


def _gaussian_evidence(
    lik: Gaussian, f: Tensor, y: Tensor, r: Tensor, k: Tensor, observed: Tensor | None
) -> Tensor:
    """The Gaussian evidence log N(y; f - r, K + σ² I), over [m E], by its Cholesky factor.

    With `observed`, over those entries only (`k` is already restricted to them).
    """

    if y.shape != f.shape:
        raise StructureError(
            f"Gaussian needs targets shaped like f {tuple(f.shape)}, got {tuple(y.shape)}"
        )

    var = lik.distribution(f).variance.reshape(-1).to(k.dtype)
    resid = (y - f).reshape(-1).to(k.dtype) + r

    if observed is not None:
        var, resid = var[observed], resid[observed]

    chol = torch.linalg.cholesky(k + torch.diag(var))
    z = torch.linalg.solve_triangular(chol, resid.unsqueeze(1), upper=False)
    logdet = 2 * chol.diagonal().log().sum()

    return -0.5 * (z.square().sum() + logdet + resid.numel() * math.log(2 * math.pi))


# --- a subset of the data ---------------------------------------------------------------


@pytree
class _Rows:
    inputs: Inputs  # leading dim k
    targets: Targets | None  # leading dim k, or None when the batch had none
    root: Tensor  # [k, E, E]
    priority: Tensor  # [k], ascending


@pytree
class _Picked:
    """The examples with the smallest priorities so far, at most `m` of them.

    A monoid: `+` keeps the `m` smallest of both, sorted by priority, and the identity holds
    no rows. It is not a sum, so `scaled` leaves it as it is.
    """

    rows: tuple[_Rows, ...]  # empty, or one block of rows
    seen: Tensor  # examples seen
    m: int = static()

    @property
    def total_weight(self) -> Tensor:
        return self.seen

    def __add__(self, other: Self, /) -> Self:
        parts = self.rows + other.rows
        seen = self.seen + other.seen

        if not parts:
            return type(self)((), seen, self.m)

        priority = torch.cat([r.priority for r in parts])
        keep = torch.argsort(priority, stable=True)[: self.m]

        def take(*xs: Any) -> Any:
            if is_shared(xs[0]):
                return xs[0]

            return torch.cat(xs)[keep.to(xs[0].device)]

        inputs = tree_map(take, *(r.inputs for r in parts))
        root = take(*(r.root for r in parts))

        given = [r.targets for r in parts]
        missing = [t is None for t in given]

        if any(missing) and not all(missing):
            raise ValueError(
                "SubsetOfData: some batches have targets and others don't; give every batch "
                "targets, or none"
            )

        targets = None if all(missing) else tree_map(take, *given)

        return type(self)((_Rows(inputs, targets, root, priority[keep]),), seen, self.m)

    def scaled(self, s: Tensor, /) -> Self:
        return self


def _requirements() -> Requirements:
    # Batch weights would rescale the output Hessians, which a subset does not average.
    return Requirements(
        vjp=True, vmap=True, needs_output_hessian=True, additive_stat=False, weights=False
    )


def _dense_root(hess: core.OutputHessian, f: Tensor) -> Tensor:
    """A [N, E, E] with A Aᵀ = H per example, column j from `sqrt_mv` of the basis vector."""

    n, e = f.shape[0], f[0].numel()
    eye = torch.eye(e, dtype=f.dtype, device=f.device)
    cols = [hess.sqrt_mv(eye[j].expand(n, e).reshape(f.shape)).reshape(n, e) for j in range(e)]

    return torch.stack(cols, dim=2)


def _diagonal_root(hess: core.OutputHessian, f: Tensor) -> Tensor:
    """diag(sqrt(diag H)) per example, [N, E, E]: the root of H's diagonal."""

    n = f.shape[0]

    return torch.diag_embed(hess.diagonal().reshape(n, -1).sqrt())


@pytree
class _Subset:
    """The pass of `SubsetOfData`: the m examples of smallest uniform priority.

    Priorities are the first draws of each batch's generator, which the fit seeds with the
    process rank under `torch.distributed`, so ranks with equal shards draw differently.
    Each kept example carries its targets and the root of its output Hessian (of the unit
    likelihood): the dense root with `"joint"`, the root of its diagonal otherwise. Roots
    are built only for the batch's m smallest, which are the only rows that can survive any
    merge.
    """

    m: int = static()
    structure: _Structure = static(default="joint")

    def __repr__(self) -> str:
        return f"SubsetOfData(n_subset={self.m}, structure={self.structure!r})"

    @property
    def requires(self) -> Requirements:
        return _requirements()

    def init(self, params: Params, ctx: FitContext, /) -> _Picked:
        p = next(iter(params.values()))

        return _Picked((), torch.zeros((), dtype=torch.int64, device=p.device), self.m)

    def update(
        self,
        stat: _Picked,
        params: Params,
        batch: Batch,
        generator: torch.Generator,
        ctx: FitContext,
        /,
    ) -> _Picked:
        f = ctx.model(params, batch.inputs)

        if not isinstance(f, Tensor):
            raise NotImplementedError("SubsetOfData supports models returning a single tensor")

        n = batch.size
        gen = generator
        priority = torch.rand(n, generator=gen, dtype=torch.float64, device=gen.device)
        keep = torch.argsort(priority, stable=True)[: self.m]

        # Only the batch's m smallest can be kept, whatever the merges: roots for those only.
        inputs = rows_of(batch.inputs, keep)
        targets = None if batch.targets is None else rows_of(batch.targets, keep)
        fk = f[keep.to(f.device)]
        hess = output_hessian(ctx.likelihood, fk, generator=generator, inputs=inputs)
        dense = self.structure == "joint"
        root = (_dense_root if dense else _diagonal_root)(hess, fk).to(stat_dtype(f.dtype))

        count = torch.tensor(n, dtype=torch.int64, device=stat.seen.device)
        rows = _Rows(inputs, targets, root, priority[keep].to(f.device))

        return stat + _Picked((rows,), count, self.m)

    def finalize(self, stat: _Picked, /) -> _Picked:
        return stat


def _check_budget(m: int, e: int, p: int) -> None:
    what = f"SubsetOfData(n_subset={m})"
    sizes = f"m={m}, E={e}, P={p}"

    if m * e > MAX_SUBSET_ENTRIES:
        raise StructureError(
            f"{what} conditions on m*E = {m * e} output entries ({sizes}), above "
            f"MAX_SUBSET_ENTRIES={MAX_SUBSET_ENTRIES}; use a smaller m"
        )

    if m * e * p > MAX_SUBSET_JACOBIAN_ENTRIES:
        raise StructureError(
            f"{what} keeps a Jacobian of m*E*P = {m * e * p} entries ({sizes}), above "
            f"MAX_SUBSET_JACOBIAN_ENTRIES={MAX_SUBSET_JACOBIAN_ENTRIES}; use a smaller m or "
            "fewer parameters (subset=...)"
        )


@pytree
class SubsetOfData:
    r"""A subset-of-data GP: condition on $m$ training inputs drawn uniformly without replacement.

    It takes one pass, and keeps every input when there are fewer than $m$. Each example
    gets a uniform priority drawn from its batch's generator, and the $m$ smallest are
    kept, so the subset depends on the fit's seed only. Needs vjp and vmap (the Jacobian at
    the subset) and the likelihood's output Hessian. Weighted batches are refused. `plan`
    checks `MAX_SUBSET_ENTRIES` and `MAX_SUBSET_JACOBIAN_ENTRIES` against $m$, the probe's
    event size and the parameter count, before any data.

    Args:
        n_subset: The subset size $m$, at least 1.
        structure: How the outputs of one input interact. `"joint"`: the output Hessian as
            it is and the joint kernel over the outputs. `"diagonal"`: the output Hessian's
            diagonal, with the joint kernel; this is the joint `FunctionalLaplace` of
            laplace-torch. `"independent"`: the diagonal, and one GP per output
            (`independent_outputs=True` in laplace-torch), so `at` gives diagonal
            $E \times E$ blocks. A Gaussian's output Hessian is already diagonal, so
            `"joint"` and `"diagonal"` agree.

    Raises:
        ValueError: `n_subset` is below 1, or `structure` is not `"joint"`, `"diagonal"` or
            `"independent"`.
    """

    n_subset: int = static()
    structure: _Structure = static(default="joint")

    def __post_init__(self) -> None:
        if self.n_subset < 1:
            raise ValueError(f"SubsetOfData needs n_subset >= 1, got {self.n_subset}")

        if self.structure not in get_args(_Structure):
            raise ValueError(
                'SubsetOfData structure must be "joint", "diagonal" or "independent", got '
                f"{self.structure!r}"
            )

    def __repr__(self) -> str:
        return f"SubsetOfData(n_subset={self.n_subset}, structure={self.structure!r})"

    @property
    def requires(self) -> Requirements:
        """Needs vjp, vmap and the output Hessian, and refuses weighted batches."""

        return _requirements()

    def plan(
        self, params: Params, ctx: FitContext, prior: PriorWithCovariance, /
    ) -> Pass[SubsetGP[Any]]:
        r"""The subset pass, ending in the `SubsetGP`.

        Returns:
            The pass. When run, it raises `ValueError` when the data yields no examples or
            some batches have targets and others don't, `NotImplementedError` when the
            model does not return a single tensor, and `StructureError` when the rows it
            kept break the budgets below.

        Args:
            params: The fitted params, where the model is linearized.
            ctx: The fit's context.
            prior: The prior over `params`. Its covariance $\Sigma_0$ enters the kernel.

        Raises:
            StructureError: With the probe's event size $E$ and $P$ parameters, $m E$
                exceeds `MAX_SUBSET_ENTRIES` or $m E P$ exceeds
                `MAX_SUBSET_JACOBIAN_ENTRIES`.
        """

        p = sum(v.numel() for v in params.values())

        if ctx.caps.event_size is not None:
            _check_budget(self.n_subset, ctx.caps.event_size, p)

        def then(picked: _Picked) -> Done[SubsetGP[Any]]:
            if not picked.rows:
                raise ValueError("SubsetOfData: the data yielded no examples")

            (rows,) = picked.rows
            m, e = rows.root.shape[0], rows.root.shape[1]
            _check_budget(m, e, p)

            jac = jacobian(ctx.model, params, rows.inputs, rows.root.dtype)
            fixed = {
                k: v for k, v in core.param_versions(ctx.model, params).items() if k not in params
            }
            gp = SubsetGP(
                ctx.model,
                ctx.likelihood,
                params,
                prior,
                ctx.caps,
                jac,
                rows.root,
                m,
                rows.inputs,
                rows.targets,
                int(picked.seen),
                self.structure,
                fixed,
            )

            return Done(gp)

        return chain(_Subset(self.n_subset, self.structure), then=then)


# --- the front door -----------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class _WithPrior[B]:
    """A function-space method with its prior bound: a `Staged` the core fit loop runs."""

    approx: FunctionSpaceMethod[B]
    prior: PriorWithCovariance

    def __repr__(self) -> str:
        return repr(self.approx)

    @property
    def requires(self) -> Requirements:
        return self.approx.requires

    def plan(self, params: Params, ctx: FitContext, /) -> Pass[B]:
        return self.approx.plan(params, ctx, self.prior)


def laplace[B: FunctionPosterior](
    model: nn.Module | ModelFn,
    data: Data | Iterable[Any],
    likelihood: Likelihood,
    *,
    method: FunctionSpaceMethod[B],
    subset: Subset | Params | Selection = subsets.last_layer,
    prior: PriorWithCovariance = priors.Isotropic(),  # noqa: B008  # immutable
    generator: torch.Generator | None = None,
) -> B:
    """Function-space Laplace: probe, pick a subset of parameters, run the method's passes.

    The probe is that of `lk.laplace`. The method's requirements, the prior's location and
    the method's own budgets are checked before any data past the probe's batch.

    ```python
    gp = lk.function_space.laplace(model, loader, lk.Gaussian(), method=lk.gp.ELLA(points, rank=20))
    ```

    The result's type `B` comes from `method`. Python's types cannot bind the likelihood's
    type inside it, so `gp.likelihood` is typed `Likelihood`; narrow it to reach a field:

    ```python
    lik = gp.likelihood
    assert isinstance(lik, lk.Gaussian)
    ```

    Args:
        model: The model. A module is wrapped with `ModelFn.from_module`.
        data: A `Data` source, or a re-iterable of `Batch`es or `(inputs, targets)` pairs,
            which `lk.data.from_loader` wraps.
        likelihood: The likelihood. The method's needs are checked against it before any
            data is read.
        method: The function-space method, such as `SubsetOfData(n_subset)` or
            `lk.gp.ELLA(points, rank)`.
        subset: The parameters to fit, as for `lk.laplace`: a subset function, called with
            the model and the probe's capabilities, a parameter mapping, or a `Selection`.
        prior: The prior. A fitted weight-space posterior works too, if it is over the same
            parameters.
        generator: Seeds the data passes and the method's draws, as in `lk.laplace`. `None`
            means torch's global RNG, so `torch.manual_seed` makes the fit repeatable.

    Returns:
        The function-space posterior. It holds a copy of the fitted parameters as its mean,
        the prior as given, and the real likelihood (the passes see its unit version, so
        `sigma` stays tunable). Moving one of the module's other parameters in place
        afterwards makes `at` raise.

    Raises:
        CapabilityError: The likelihood or the model lacks what the method needs, a batch
            fails the method's batch requirements (`SubsetOfData` refuses weights), or
            `last_layer` finds no parameterized module.
        StructureError: The prior's location doesn't match the fitted parameters, the
            method's budgets fail, the subset refuses the model, or the probe or a pass
            finds badly shaped outputs or targets, or a mask that is not bool or not shaped
            like the leading axes of the outputs.
        ValueError: The data yields no batches or no examples, some batches have targets
            and others don't, or the likelihood finds a bad target value.
        TypeError: `data` is a one-shot iterator, or an item is neither a `Batch` nor an
            `(inputs, targets)` pair.
        NotImplementedError: The model does not return a single tensor.
    """

    gen = spawn_generator(generator)
    fn = ModelFn.from_module(model) if isinstance(model, nn.Module) else model
    src = as_data(data)
    approx = method

    core.check_likelihood(approx.requires, likelihood, repr(approx))

    first = next(iter(src(spawn_generator(gen))), None)

    if first is None:
        raise ValueError("laplace: the data yielded no batches")

    order = core.probe(fn, likelihood, first, params={})
    picked = (
        cast(Params | Selection, subset)
        if isinstance(subset, Selection | Mapping)
        else subset(fn, order)
    )

    # A `Selection` fits its entries as parameters of their own (see `lk.laplace`).
    fn, picked_params = core.select(fn, picked)
    chosen = {k: v.detach() for k, v in picked_params.items()}

    caps = core.probe(fn, likelihood, first, params=chosen)
    caps.check(approx.requires, repr(approx))
    approx.requires.check_batch(first, repr(approx))

    chosen_prior: PriorWithCovariance = prior
    # A prior whose location does not match the parameters raises before any pass.
    chosen_prior.location(chosen)

    # `start` runs the method's `plan`, which checks its budgets before the data.
    staged = _WithPrior(approx, chosen_prior)
    fit = fit_curvature(fn, likelihood, chosen, staged, generator=gen, capabilities=caps)
    step = core.run(fit, src, generator=gen)

    # The posterior owns a copy of the fitted parameters (the tensors the method was given),
    # so training the module later leaves it in place. The prior's tensors come back as
    # given, as in `lk.laplace`.
    fitted = {id(v) for v in chosen.values()}
    given = {id(t) for t in core.tree_leaves(chosen_prior) if isinstance(t, Tensor)}

    def own(t: Any) -> Any:
        if not isinstance(t, Tensor) or id(t) in given:
            return t

        return t.clone() if id(t) in fitted else t.detach()

    result = tree_map(own, step.result)

    return copy.replace(result, likelihood=likelihood)
