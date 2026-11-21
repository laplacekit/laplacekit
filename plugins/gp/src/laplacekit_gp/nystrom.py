r"""ELLA (Deng et al. 2022), the linearized Laplace GP on a Nyström basis.

`ELLA(points, rank)` takes Jacobian rows at a few points and keeps the top `rank`
directions of their kernel as a basis $V$, `[P, r]`. A second pass over the data sums the
GGN projected onto $V$. The fit returns a `NystromGP`, a GP over the outputs with
covariance $\Phi (s c G + \lambda I)^{-1} \Phi^\top$ for $\Phi = J V$.
"""

import dataclasses
import math
from collections.abc import Mapping
from typing import Generic, Literal, TypeVar, get_args

import torch
from torch import Tensor

import laplacekit as lk
from laplacekit.core import (
    Capabilities,
    Done,
    FitContext,
    Inputs,
    Likelihood,
    ModelFn,
    OutputGaussian,
    Params,
    Pass,
    PerInputCov,
    PriorWithCovariance,
    Requirements,
    Strategy,
    StructureError,
    as_pytree,
    chain,
    curvature_scale,
    param_versions,
    pytree,
    stat_dtype,
    static,
)
from laplacekit_gp._gram import GramStat, ProjectedGram, check_event
from laplacekit_gp._jacobian import choose_via, features, rows
from laplacekit_gp._versions import check_versions
from laplacekit_gp.points import Drawn, Fixed, Picked, Points, check_coords, point_count

__all__ = ["MAX_BASIS_ENTRIES", "ELLA", "NystromGP"]

MAX_BASIS_ENTRIES = 2**28
"""The largest array `ELLA` builds, in entries: $2^{28}$, which is 1 GiB in float32.

The cap applies to the basis ($P r$ entries), to the picked Jacobian rows ($m k P$) and, in
reverse mode, to the Jacobian of one input ($E P$).
"""

_Outputs = Literal["random", "all"]

# Explicit variance, as for `LaplacePosterior`: the likelihood's type, covariant, so a
# posterior with a Gaussian is assignable to a bare `NystromGP`.
_L_co = TypeVar("_L_co", bound=Likelihood, covariant=True, default=Likelihood)


# --- the posterior -----------------------------------------------------------------------


@pytree
class NystromGP(Generic[_L_co]):  # noqa: UP046
    r"""The linearized Laplace posterior on a Nyström basis, as a GP over the outputs.

    At inputs $x$, let $\Phi = J(x) V$, `[N, E, r]`, where $J(x)$ is the Jacobian at the
    mean parameters and $V$ is the basis. The covariance of each input's outputs is

    $$\Phi \, (s c G + \lambda I)^{-1} \, \Phi^\top,$$

    where $c$ is the likelihood's `curvature_scale` (1 if it has none) and $\lambda$ is the
    prior's precision. The mean is $f(x)$ at the mean parameters. `at` reads the prior and
    the likelihood on every call, so tuning $\lambda$ or `sigma` needs no refit.

    Args:
        model: The model as a function of the parameters.
        likelihood: The likelihood of the fit. Its tensors, such as `sigma`, are leaves
            that tuning can change.
        mean: The fitted parameters.
        prior: The prior, an `Isotropic` with one precision $\lambda$.
        caps: The model's capabilities.
        basis: The Nyström basis $V$, `[P, r]`.
        gram: $G$, the per-example average of $\Phi_n^\top H_n \Phi_n$ over the data,
            `[r, r]`. $H_n$ is the output Hessian without the likelihood's
            `curvature_scale`.
        n_data: $s$, the total weight of the data.
        strategy: Whether `at` forms $\Phi$ in forward mode (`"forward-mode"`) or reverse mode
            (`"reverse-mode"`).
        versions: The version counters of the module's fixed parameters at fit time.
            `at` raises `RuntimeError` when one has changed in place since then (see
            `check_versions`).
    """

    model: ModelFn = static()
    likelihood: _L_co  # a child: its tensors (sigma) are leaves to tune
    mean: Params
    prior: lk.priors.Isotropic
    caps: Capabilities = static()
    basis: Tensor
    gram: Tensor
    n_data: Tensor
    strategy: Strategy = static()
    versions: Mapping[str, int] = static(default_factory=dict[str, int])

    def __repr__(self) -> str:
        size = sum(p.numel() for p in self.mean.values())

        return (
            f"NystromGP(rank={self.basis.shape[1]}, parameters={size}, strategy={self.strategy!r}, "
            f"prior={self.prior!r})"
        )

    @property
    def requires(self) -> Requirements:
        r"""What `at` needs to form $\Phi$ at new inputs: vmap, and jvp or vjp by `strategy`."""

        return Requirements(
            jvp=self.strategy == "forward-mode", vjp=self.strategy == "reverse-mode", vmap=True
        )

    def check_versions(self) -> None:
        """Raise `RuntimeError` if a fixed parameter of the module changed in place since the fit.

        The basis and $G$ were computed with the old value. `at` calls this first.
        """

        check_versions(self.model, self.versions, "the basis and the Gram used its old value")

    def at(self, inputs: Inputs, /) -> OutputGaussian:
        r"""The Gaussian over the outputs at `inputs`, with one $E \times E$ block per input.

        The solve uses a Cholesky factor of $s c G + \lambda I$, in the dtype of `gram`.

        Args:
            inputs: The inputs, with the examples along the leading axis.

        Returns:
            An `OutputGaussian` whose mean is the network's outputs at `inputs` and whose covariance
            holds one $E \times E$ block per input.

        Raises:
            CapabilityError: The model does not support vmap or the mode that `strategy` names.
            RuntimeError: A fixed parameter of the module changed in place since the fit.
            StructureError: The model's outputs are not one tensor, the likelihood's
                `curvature_scale` has more than one element, or the prior is no longer an
                `Isotropic` with one precision.
        """

        self.caps.check(self.requires, "NystromGP.at")
        self.check_versions()

        inputs = as_pytree(inputs)  # a `BatchEncoding` as a dict, so `vmap` sees its tensors
        dtype = self.gram.dtype
        f, phi = features(self.model, self.mean, inputs, self.basis, self.strategy, dtype)
        n, e, r = phi.shape

        chol = torch.linalg.cholesky(self._precision(dtype))
        b = torch.linalg.solve_triangular(chol, phi.reshape(n * e, r).T, upper=False)
        b = b.reshape(r, n, e)
        cov = torch.einsum("rne,rnf->nef", b, b)

        return OutputGaussian(f, PerInputCov(0.5 * (cov + cov.mT), tuple(f.shape[1:])))

    def _precision(self, dtype: torch.dtype) -> Tensor:
        r"""The precision $s c G + \lambda I$, in `dtype`.

        The prior must still be an `Isotropic` with one precision.
        """

        c = curvature_scale(self.likelihood)

        if c is not None and c.numel() != 1:
            raise StructureError(
                f"NystromGP needs a one-element curvature_scale, but "
                f"{type(self.likelihood).__name__} gives shape {tuple(c.shape)}"
            )

        prec = _check_prior(self.prior).prior_precision
        s = self.n_data.to(dtype)
        weight = s if c is None else s * c.reshape(()).to(dtype)
        eye = torch.eye(self.gram.shape[0], dtype=dtype, device=self.gram.device)

        return weight * self.gram + prec.reshape(()).to(dtype) * eye


# --- the method -----------------------------------------------------------------------


def _check_prior(prior: PriorWithCovariance) -> lk.priors.Isotropic:
    """The prior as an `Isotropic` with one precision, else `StructureError`."""

    if isinstance(prior, lk.priors.Isotropic) and prior.prior_precision.numel() == 1:
        return prior

    what = (
        "an Isotropic prior with a vector prior_precision"
        if isinstance(prior, lk.priors.Isotropic)
        else f"a {type(prior).__name__} prior"
    )

    raise StructureError(
        f"ELLA needs an Isotropic prior with one precision, which it can apply after the "
        f"fit, got {what}; fs.SubsetOfData and lk.gp.VaLLA take any pushable prior"
    )


def _check_rows(m: int, k: int, n_params: int) -> None:
    if m * k * n_params > MAX_BASIS_ENTRIES:
        raise StructureError(
            f"ELLA holds m*k*P = {m * k * n_params} Jacobian entries for its points "
            f"(m={m}, k={k}, P={n_params}), above MAX_BASIS_ENTRIES={MAX_BASIS_ENTRIES}; use "
            'fewer points, outputs="random" or fewer parameters (subset=...)'
        )


@pytree
class ELLA:
    r"""ELLA (Deng et al. 2022), the linearized Laplace GP on a Nyström basis of rank `rank`.

    `points` selects $m$ inputs. At each one, `outputs` decides which $k$ Jacobian rows to
    take, and the rows stack into $R$, `[m k, P]`. The top `rank` eigenpairs
    $(U, \Lambda)$ of the kernel $K = R R^\top$, computed in float64, give the basis

    $$V = R^\top U \Lambda^{-1/2}.$$

    A second pass sums the GGN projected onto $V$, and `fs.laplace` returns a `NystromGP`.

    The prior must be an `Isotropic` with one precision, because `NystromGP` applies it
    after the fit, as it does the likelihood's `curvature_scale`. `plan` refuses any other
    prior before it reads data.

    The kernel needs `rank` eigenvalues above ELLA's tolerance,
    $\epsilon \max(m k, P) \max_i |\lambda_i|$, where $\lambda_i$ are the eigenvalues of
    $K$ and $\epsilon$ is the float64 machine epsilon. With fewer, the fit raises
    `ValueError` and names the usable rank. Every built-in `Points` knows $m$ in advance,
    so $m k < \text{rank}$ is caught before any data. Duplicate points and a Jacobian of
    lower rank show up once the points are picked, before the second pass. k-means can
    snap two centres to the same data row, and such duplicates lower the usable rank.

    An output coordinate outside $[0, E)$ raises `StructureError`. Given coordinates are
    checked before any data, and picked ones as soon as the points are picked.

    `MAX_BASIS_ENTRIES` caps the basis ($P \cdot \text{rank}$ entries), the picked rows
    ($m k P$) and, in reverse mode, the Jacobian of one input ($E P$). Going over raises
    `StructureError` with the sizes. Each cap is checked before any data, except $m k P$
    for points picked from the data, which is checked once they are picked. A model whose
    outputs are not one tensor is refused before any data when the likelihood is a
    mapping, and before the second pass otherwise.

    The model must support vjp and vmap to form the rows, and jvp as well with
    `strategy="forward-mode"`. The likelihood needs an output Hessian, or sampling for the Monte
    Carlo Fisher. Points picked from the data refuse weighted batches. With `fixed` points, the
    second pass uses the weights.

    Args:
        points: Selects the $m$ inputs, for example `fixed`, `from_data` or `kmeans`.
        rank: The rank $r$ of the basis.
        outputs: Which Jacobian rows each point gives. `"random"` gives one row
            ($k = 1$), at the coordinate the point carries (`fixed(..., coords)`) or at
            $\lfloor u E \rfloor$ for its `Picked.u`. `"all"` gives all $k = E$ rows.
        strategy: How the second pass and `NystromGP.at` form $\Phi$. `None` uses forward
            mode when the probe found jvp and reverse mode otherwise, as laplace-torch
            does. `"forward-mode"` or `"reverse-mode"` sets the mode, and `fs.laplace` refuses
            before any data when the probe did not find it.

    Raises:
        ValueError: `rank` is below 1, `outputs` is neither `"random"` nor `"all"`, or
            `strategy` is not `"forward-mode"`, `"reverse-mode"` or `None`.
    """

    points: Points
    rank: int = static()
    _: dataclasses.KW_ONLY
    outputs: _Outputs = static(default="random")
    strategy: Strategy | None = static(default=None)

    def __post_init__(self) -> None:
        if self.rank < 1:
            raise ValueError(f"ELLA needs rank >= 1, got {self.rank}")

        if self.outputs not in get_args(_Outputs):
            raise ValueError(f'ELLA outputs must be "random" or "all", got {self.outputs!r}')

        if self.strategy is not None and self.strategy not in get_args(Strategy.__value__):
            raise ValueError(
                'ELLA strategy must be "forward-mode", "reverse-mode" or None, '
                f"got {self.strategy!r}"
            )

    def __repr__(self) -> str:
        return (
            f"ELLA(rank={self.rank}, points={self.points!r}, outputs={self.outputs!r}, "
            f"strategy={self.strategy!r})"
        )

    @property
    def requires(self) -> Requirements:
        """What the passes need: vjp, vmap, an output Hessian, and jvp in forward mode.

        A likelihood that can sample may replace the output Hessian, through the Monte
        Carlo Fisher. Points picked from the data refuse weights.
        """

        # Every pass, declared up front: the rows need vjp and vmap, the Gram the output
        # Hessian (and jvp when asked for), and a pick pass ignores weights.
        picks = not isinstance(self.points, Drawn)

        return Requirements(
            vjp=True,
            jvp=self.strategy == "forward-mode",
            vmap=True,
            needs_output_hessian=True,
            additive_stat=not picks,
            weights=not picks,
        )

    def plan(
        self, params: Params, ctx: FitContext, prior: PriorWithCovariance, /
    ) -> Pass[NystromGP]:
        """Plan the passes: pick the points, build the basis, then sum the projected GGN.

        `Drawn` points need no data, so `plan` builds the basis right away and raises its
        errors itself. Points picked from the data build the basis after the pass that
        picks them.

        Returns:
            The first pass. Running it raises `ValueError` when the data has no
            examples, the total weight is zero, a weight is negative or the kernel has
            fewer than `rank` usable directions, and `StructureError` when the model's
            outputs are not one tensor, a picked coordinate lies outside $[0, E)$ or
            $m k P$ exceeds `MAX_BASIS_ENTRIES`.

        Args:
            params: The parameters to fit, as the fit's subset gives them.
            ctx: The fit's context: the model, the likelihood, the capabilities the probe found, and
                the layout of the parameters.
            prior: The prior over the parameters.

        Raises:
            StructureError: The prior is not an `Isotropic` with one precision, the
                likelihood is a mapping, a budget known before data fails, a given
                coordinate lies outside $[0, E)$, or the MC Fisher (a likelihood without
                `HasOutputHessian`) would form dense blocks above `MAX_DENSE_EVENT`. With
                `Drawn` points, also when the model's outputs are not one tensor, a
                picked coordinate lies outside $[0, E)$ or $m k P$ exceeds
                `MAX_BASIS_ENTRIES`.
            ValueError: The points give fewer than `rank` Jacobian rows, or, with `Drawn`
                points, the kernel has fewer than `rank` usable directions.
            TypeError: `points.pick` returns `Done` instead of handing its points on.
        """

        iso = _check_prior(prior)

        if isinstance(ctx.likelihood, Mapping):
            raise StructureError(
                "ELLA needs a model returning one tensor, but the likelihood is a mapping"
            )

        n_params = sum(v.numel() for v in params.values())
        event = ctx.caps.event_size

        if self.strategy is None:
            strategy = choose_via(ctx.caps)
        else:
            strategy = self.strategy

        self._check_budgets(n_params, event, strategy)
        check_event(ctx, repr(self))

        # Rows known before data: m for every built-in pick, k = E only with "all".
        m = point_count(self.points)
        k = 1 if self.outputs == "random" else event

        given = self.points.coords if isinstance(self.points, Fixed) else None

        if given is not None and self.outputs == "random" and event is not None:
            check_coords(given, event, repr(self))

        if m is not None and k is not None:
            self._check_reachable(m * k)

            if isinstance(self.points, Fixed):
                _check_rows(m, k, n_params)

        def basis_step(picked: Picked) -> Pass[NystromGP]:
            basis = self._basis(params, ctx, picked, n_params)
            fixed = {k: v for k, v in param_versions(ctx.model, params).items() if k not in params}

            def done(stat: GramStat) -> Done[NystromGP]:
                gp = NystromGP(
                    ctx.model,
                    ctx.likelihood,
                    params,
                    iso,
                    ctx.caps,
                    basis,
                    stat.gram,
                    stat.weight,
                    strategy,
                    fixed,
                )

                return Done(gp)

            return chain(ProjectedGram(basis, strategy), then=done)

        out = self.points.pick(params, ctx, basis_step)

        if isinstance(out, Done):
            raise TypeError(f"{self.points!r}.pick must hand its points to the next step")

        return out

    def _check_reachable(self, n_rows: int) -> None:
        """`ValueError` when the points give fewer Jacobian rows than `rank`."""

        if n_rows < self.rank:
            raise ValueError(
                f"ELLA(rank={self.rank}): the points give at most {n_rows} Jacobian rows, "
                f"so the usable rank is at most {n_rows}; use a lower rank or more points"
            )

    def _check_budgets(self, n_params: int, event: int | None, strategy: Strategy) -> None:
        if n_params * self.rank > MAX_BASIS_ENTRIES:
            raise StructureError(
                f"ELLA holds a basis of P*rank = {n_params * self.rank} entries "
                f"(P={n_params}, rank={self.rank}), above MAX_BASIS_ENTRIES="
                f"{MAX_BASIS_ENTRIES}; use a lower rank or fewer parameters (subset=...)"
            )

        if (
            strategy == "reverse-mode"
            and event is not None
            and event * n_params > MAX_BASIS_ENTRIES
        ):
            raise StructureError(
                f"ELLA in reverse mode holds one input's Jacobian, E*P = "
                f"{event * n_params} entries (E={event}, P={n_params}), above "
                f"MAX_BASIS_ENTRIES={MAX_BASIS_ENTRIES}; "
                'use strategy="forward-mode" or fewer parameters'
            )

    def _basis(self, params: Params, ctx: FitContext, picked: Picked, n_params: int) -> Tensor:
        """V = Rᵀ U Λ^{-1/2} from the picked rows, in the statistics' dtype."""

        inputs = as_pytree(picked.inputs)
        f = ctx.model(params, inputs)

        if not isinstance(f, Tensor):
            raise StructureError(f"ELLA needs a model returning one tensor, got {type(f).__name__}")

        m, e = f.shape[0], math.prod(f.shape[1:])
        k = e if self.outputs == "all" else 1
        _check_rows(m, k, n_params)

        if self.outputs == "all":
            coords = None
        elif picked.coords is not None:
            check_coords(picked.coords, e, repr(self))
            coords = picked.coords.to(f.device)
        else:
            coords = (picked.u * e).floor().long().clamp(max=e - 1).to(f.device)

        r = rows(ctx.model, params, inputs, coords, torch.float64)

        # ELLA's truncation: the top `rank` eigenpairs, each above the tolerance.
        evals, evecs = torch.linalg.eigh(r @ r.T)
        top = torch.argsort(evals, descending=True)[: self.rank]
        values = evals[top]
        tol = torch.finfo(torch.float64).eps * max(r.shape) * evals.abs().max()

        if len(top) < self.rank or bool((values <= tol).any()):
            usable = int((evals > tol).sum())

            raise ValueError(
                f"ELLA(rank={self.rank}): the kernel of the {r.shape[0]} picked Jacobian "
                f"rows has {usable} directions above the tolerance, so the usable rank is "
                f"{usable}; use a lower rank, more points, or another seed (duplicate "
                "points lower the rank)"
            )

        basis = r.T @ (evecs[:, top] / values.sqrt())
        p = next(iter(params.values()))

        return basis.to(stat_dtype(p.dtype))
