"""FSP-Laplace (Cinquin et al. 2024), a Laplace approximation under a GP prior on the outputs.

`FunctionPrior` holds the GP prior: a kernel, a mean and a jitter. Training adds
`rkhs_norm`, the prior's term, to the loss. After training, `fsp_laplace` fits
`FSPPosterior`, the posterior of Algorithm 2. It is a low-rank Gaussian over the weights,
supported on the directions that the context points constrain.
"""

import copy
import math
from collections.abc import Callable, Iterable, Mapping
from typing import Any, Generic, Self, TypeVar, cast

import torch
from torch import Tensor, nn

import laplacekit as lk
from laplacekit.core import (
    Capabilities,
    Done,
    FitContext,
    Layout,
    Likelihood,
    ModelFn,
    Operator,
    Params,
    Pass,
    Requirements,
    Selection,
    Strategy,
    StructureError,
    Subset,
    chain,
    check_likelihood,
    check_location,
    curvature_scale,
    fit_curvature,
    param_versions,
    pytree,
    run,
    select,
    stat_dtype,
    static,
    tree_map,
)
from laplacekit_gp._gram import GramStat, ProjectedGram, check_event
from laplacekit_gp._jacobian import choose_via, rows
from laplacekit_gp._lanczos import lanczos, pinv_root
from laplacekit_gp._versions import check_versions
from laplacekit_gp.kernels import Kernel
from laplacekit_gp.points import Drawn, Picked, Points, point_count

__all__ = [
    "MAX_CONTEXT_JACOBIAN_ENTRIES",
    "FSPPosterior",
    "FunctionPrior",
    "RangePrecision",
    "fsp_laplace",
    "rkhs_norm",
]

MAX_CONTEXT_JACOBIAN_ENTRIES = 2**28
"""The most entries $n_C E P$ of the Jacobian at the context points: $2^{28}$, which is 2 GiB in
float64, the dtype the Jacobian is built in."""


@pytree
class FunctionPrior:
    r"""A GP prior over the outputs, $f \sim \mathcal{GP}(\mu, k)$, shared by every output entry.

    Args:
        kernel: The kernel $k$.
        mean: The mean $\mu$, a function from inputs `[n, ...]` to outputs shaped like the
            network's. `None` means zero.
        jitter: The relative jitter $\delta \ge 0$ that `cov` adds.

    Raises:
        ValueError: `jitter` is negative.
    """

    kernel: Kernel
    mean: Callable[[Tensor], Tensor] | None = static(default=None)
    jitter: float = static(default=1e-6)

    def __post_init__(self) -> None:
        if self.jitter < 0:
            raise ValueError(f"FunctionPrior needs jitter >= 0, got {self.jitter}")

    def cov(self, x: Tensor, /) -> Tensor:
        r"""The prior covariance at `x`, $K + \delta \, \overline{\operatorname{diag} K} \, I$.

        Here $K = k(x, x)$, and $\overline{\operatorname{diag} K}$ is the mean of its
        diagonal.

        Args:
            x: Inputs `[n, ...]`.

        Returns:
            The covariance, `[n, n]`.

        Raises:
            StructureError: The kernel refuses `x`.
        """

        k = self.kernel(x, x)
        eye = torch.eye(k.shape[0], dtype=k.dtype, device=k.device)

        return k + self.jitter * k.diagonal().mean() * eye


def rkhs_norm(prior: FunctionPrior, inputs: Tensor, f: Tensor, /) -> Tensor:
    r"""The estimate of $\lVert f - \mu \rVert^2_{\mathcal{H}}$ in Eq. 3.7 of Cinquin et al. (2024).

    $$\sum_e (f_e - \mu_e)^\top \, C^{-1} \, (f_e - \mu_e),$$

    where $C$ is `prior.cov(inputs)`, and $f_e$ and $\mu_e$ are output entry $e$ of $f$ and
    of the prior's mean at the $n$ inputs. Output entries are flattened after the first
    axis. The estimate is the norm of the smallest interpolant of $f$ at `inputs`, so it
    underestimates $\lVert f - \mu \rVert^2_{\mathcal{H}}$. It is differentiable in `f`
    and in the kernel's leaves.

    In the MAP loss of Algorithm 1, a minibatch holds $b$ of the $N$ examples, and each
    step draws new context points:

    ```python
    context = lk.gp.uniform(low, high, m)
    for x, y in loader:
        c = context.draw()  # new points from torch's global RNG
        loss = -(N / b) * lik.log_prob(model(x), y).sum() + 0.5 * lk.gp.rkhs_norm(
            prior, c, model(c)
        )
    ```

    Args:
        prior: The prior, with the kernel and the mean $\mu$.
        inputs: The $n$ points, one tensor `[n, ...]`.
        f: The network's outputs at `inputs`, `[n, ...]`.

    Returns:
        A 0-dim tensor.

    Raises:
        StructureError: `inputs` or `f` is not a tensor with a leading example axis, the
            leading axis of `f` is not $n$, the prior's mean has a shape other than that
            of `f`, or the kernel refuses `inputs`.
        ValueError: `prior.cov(inputs)` is not positive definite.
    """

    if not isinstance(inputs, Tensor) or inputs.ndim == 0:
        raise StructureError(
            "rkhs_norm needs inputs as one tensor with a leading example axis, got "
            f"{type(inputs).__name__ if not isinstance(inputs, Tensor) else 'a 0-dim tensor'}"
        )

    n = inputs.shape[0]

    if not isinstance(f, Tensor) or f.ndim == 0 or f.shape[0] != n:
        got = f"shape {list(f.shape)}" if isinstance(f, Tensor) else type(f).__name__

        raise StructureError(f"rkhs_norm needs f as a tensor [{n}, ...] at the inputs, got {got}")

    d = f

    if prior.mean is not None:
        mu = prior.mean(inputs)

        if mu.shape != f.shape:
            raise StructureError(
                f"the prior's mean gives shape {list(mu.shape)} at the inputs; f has "
                f"{list(f.shape)}"
            )

        d = f - mu

    d = d.reshape(n, -1)
    chol, info = torch.linalg.cholesky_ex(prior.cov(inputs))

    if int(info) != 0:
        raise ValueError(
            "rkhs_norm: the prior covariance at the inputs is not positive definite; use a "
            "larger jitter or fewer coincident inputs"
        )

    return (d * torch.cholesky_solve(d, chol)).sum()


# --- the posterior (Algorithm 2) ------------------------------------------------------------


@pytree
class FSPBasis:
    """What `fsp_basis` returns.

    U_M `[P, r]`, D_M² `[r]`, J(C) U_M `[n_C E, r]`, and the jittered prior variance at
    the context points repeated over E, `[n_C E]`.
    """

    basis: Tensor
    prior_evals: Tensor
    context_jac: Tensor
    prior_var: Tensor


def fsp_basis(
    jac_c: Tensor, prior: FunctionPrior, inputs: Tensor, events: int, rank: int
) -> FSPBasis:
    """Steps 1 to 4 of Algorithm 2: the basis of the prior precision J(C)ᵀ Σ(C, C)⁻¹ J(C).

    `jac_c` is the dense Jacobian at the n_C context points `inputs`, `[n_C E, P]`, with
    the rows ordered by input (row n E + e). Lanczos runs for `min(rank, n_C)` steps on
    K_CC = `prior.cov(inputs)`. It starts from the row sums of J(C) added over the E
    outputs, which is J(C)·1 for E = 1 as in the paper. It gives L_C L_Cᵀ ≈ K_CC⁻¹. With
    L = L_C ⊗ I_E, the matrix M = J(C)ᵀ L has up to rank·E columns. The thin SVD
    M = U_M D_M V_Mᵀ drops singular values at most eps · max(shape) · max D_M.

    Lanczos runs on K_CC instead of Σ(C, C) = K_CC ⊗ I_E. Each eigenvalue of K_CC ⊗ I_E
    has multiplicity E, so the Krylov space of one start vector spans at most n_C
    directions. Everything is computed in float64. Raises `StructureError` when `jac_c`
    does not have n_C E rows, and `ValueError` when `rank` is below 1 or the start vector
    is zero.
    """

    if rank < 1:
        raise ValueError(f"fsp_basis needs rank >= 1, got {rank}")

    k_cc = prior.cov(inputs).to(torch.float64)
    n_c = k_cc.shape[0]

    if jac_c.ndim != 2 or jac_c.shape[0] != n_c * events:
        raise StructureError(
            f"fsp_basis needs J(C) with n_C*E = {n_c}*{events} = {n_c * events} rows, got "
            f"shape {tuple(jac_c.shape)}"
        )

    jac = jac_c.to(torch.float64)
    n_params = jac.shape[1]
    v0 = jac.sum(1).reshape(n_c, events).sum(1)

    if not bool(v0.abs().max() > 0):
        raise ValueError(
            "fsp_basis: the start vector J(C)·1 is zero (the outputs at the context points "
            "don't move with the parameters along the all-ones direction)"
        )

    q, t = lanczos(lambda v: k_cc @ v, v0, min(rank, n_c))
    root = pinv_root(q, t)  # [n_C, k]

    # M = J(C)ᵀ (L_C ⊗ I_E), column j E + e, without forming the Kronecker product.
    m = torch.einsum("nep,nj->pje", jac.reshape(n_c, events, n_params), root)
    m = m.reshape(n_params, -1)

    u, d, _ = torch.linalg.svd(m, full_matrices=False)
    keep = d > torch.finfo(d.dtype).eps * max(m.shape) * d.max()
    u, d = u[:, keep], d[keep]

    return FSPBasis(
        basis=u,
        prior_evals=d.square(),
        context_jac=jac @ u,
        prior_var=k_cc.diagonal().repeat_interleave(events),
    )


def _degenerate(rank: int, size: int) -> str:
    return f"degenerate precision of rank {rank} < {size}"


@pytree
class RangePrecision:
    r"""The precision of an `FSPPosterior`, $V \operatorname{diag}(d) V^\top$.

    The orthonormal columns of $V$ span the range of the covariance. Vectors are flat, in
    the order of the posterior's mean. The matrix has rank $r' < P$. `mv`, `diagonal` and
    `scaled` work, while `solve`, `inverse_diagonal`, `logdet`, `cov_sqrt_mv` and the
    `add_*` methods raise `StructureError`.

    Args:
        vecs: $V$, `[P, r']`.
        evals: $d$, `[r']`.
    """

    vecs: Tensor
    evals: Tensor

    def _refuse(self, what: str) -> StructureError:
        rank, size = self.vecs.shape[1], self.vecs.shape[0]

        return StructureError(
            f"RangePrecision.{what}: {_degenerate(rank, size)}; FSP-Laplace's posterior "
            "has no full-rank precision"
        )

    def mv(self, v: Tensor, /) -> Tensor:
        r"""$V \operatorname{diag}(d) V^\top v$.

        Args:
            v: A flat vector `[P]`, in the order of the posterior's mean.

        Returns:
            $V \operatorname{diag}(d) V^\top v$, `[P]`.
        """

        return ((v @ self.vecs) * self.evals) @ self.vecs.T

    def diagonal(self) -> Tensor:
        """The diagonal, `[P]`.

        Returns:
            The diagonal, `[P]`.
        """

        return self.vecs.square() @ self.evals

    def inverse_diagonal(self) -> Tensor:
        """Not available, because the precision is singular.

        Returns:
            It never returns, because it always raises `StructureError`.

        Raises:
            StructureError: Always.
        """

        raise self._refuse("inverse_diagonal")

    def logdet(self) -> Tensor:
        """Not available, because the precision is singular.

        Returns:
            It never returns, because it always raises `StructureError`.

        Raises:
            StructureError: Always.
        """

        raise self._refuse("logdet")

    def solve(self, v: Tensor, /) -> Tensor:
        """Not available, because the precision is singular.

        Args:
            v: A flat vector `[P]`, in the order of the posterior's mean.

        Returns:
            It never returns, because it always raises `StructureError`.

        Raises:
            StructureError: Always.
        """

        raise self._refuse("solve")

    def cov_sqrt_mv(self, v: Tensor, /) -> Tensor:
        """Not available, because the precision is singular. Use `FSPPosterior.cov_sqrt_mv`.

        Args:
            v: A flat vector `[P]`, in the order of the posterior's mean.

        Returns:
            It never returns, because it always raises `StructureError`.

        Raises:
            StructureError: Always.
        """

        raise self._refuse("cov_sqrt_mv")

    def scaled(self, s: Tensor, /) -> Self:
        """The precision with $d$ multiplied by `s`.

        Args:
            s: The factor, a 0-dim tensor.

        Returns:
            A `RangePrecision` with the same $V$ and $s d$.
        """

        return copy.replace(self, evals=s * self.evals)

    def add_isotropic(self, precision: Tensor, /) -> Self:
        """Not available, because FSP-Laplace's posterior has no full-rank precision.

        Args:
            precision: The precision to add.

        Returns:
            It never returns, because it always raises `StructureError`.

        Raises:
            StructureError: Always.
        """

        raise self._refuse("add_isotropic")

    def add_diagonal(self, d: Tensor, /) -> Self:
        """Not available, because FSP-Laplace's posterior has no full-rank precision.

        Args:
            d: The diagonal to add.

        Returns:
            It never returns, because it always raises `StructureError`.

        Raises:
            StructureError: Always.
        """

        raise self._refuse("add_diagonal")

    def add_same(self, other: Operator[Tensor], /) -> Self:
        """Not available, because FSP-Laplace's posterior has no full-rank precision.

        Args:
            other: The operator to add.

        Returns:
            It never returns, because it always raises `StructureError`.

        Raises:
            StructureError: Always.
        """

        raise self._refuse("add_same")


# Explicit variance, as for `LaplacePosterior`: the likelihood's type, covariant, so a
# posterior with a Gaussian is assignable to a bare `FSPPosterior`.
_L_co = TypeVar("_L_co", bound=Likelihood, covariant=True, default=Likelihood)


@pytree
class FSPPosterior(Generic[_L_co]):  # noqa: UP046
    r"""FSP-Laplace's posterior over the weights, $\mathcal{N}(\theta_*, S S^\top)$.

    $\theta_*$ is `mean`, and $S$ has rank $r' \le r$. Let $c$ be the likelihood's
    curvature scale (1 if it has none). Every use rebuilds $S$ (steps 5 to 9 of
    Algorithm 2), so it reads the current $\sigma$ from `likelihood`:

    - Diagonalize $A = \operatorname{diag}(D_M^2) + s c G = U_A D_A U_A^\top$, with the
      eigenvalues in ascending order.
    - Drop the eigenvalues at most $\epsilon r \max_i (D_A)_{ii}$, where $\epsilon$ is the
      machine epsilon of the dtype of `gram`. This takes the pseudo-inverse of $A$, since
      rounding gives $A$ eigenvalues at or below zero when $D_M^2$ spans many decades.
    - Set $S = U_M U_A D_A^{-1/2}$ on the remaining eigenvalues.
    - Drop the first $k$ columns, which carry the most variance. $k$ is the smallest
      number for which the variance at every context point,
      $\operatorname{diag}\big(J(C) S S^\top J(C)^\top\big)$, is at most
      $v_C (1 + 100 \epsilon)$. Without data near a context point, the variance there
      equals the prior's in exact arithmetic, and the slack keeps rounding from dropping
      the column with the most variance.

    `lk.core.mean_outputs`, `lk.predict_outputs` and `lk.sample_outputs` call `check_versions` first
    (`HasVersions`).

    Predict with `Linearized`, the default of `lk.predict(post, x)`. `WeightSamples` and
    `sample_outputs(kind="network")` also run, but the covariance is large along the
    directions that the context points barely constrain, so draws of the network can be
    far from the linearized predictive.

    The posterior is degenerate, with rank $r' < P$. `log_prob` and `add_precision_to`
    raise `StructureError`, and `precision()` returns a `RangePrecision`.

    Args:
        model: The model as a function of the parameters.
        likelihood: The likelihood of the fit. Its `sigma` is read on every use.
        mean: The fitted parameters $\theta_*$.
        caps: The model's capabilities.
        layout: The block layout of the parameters.
        basis: $U_M$, `[P, r]`.
        prior_evals: $D_M^2$, `[r]`.
        gram: $G = U_M^\top \bar{G} U_M$, `[r, r]`, where $\bar{G}$ is the per-example
            average of the GGN without the likelihood's curvature scale.
        n_data: $s$, the total weight of the data.
        context_jac: $J(C) U_M$, `[n_C E, r]`, where $J(C)$ is the Jacobian at the context
            points $C$.
        prior_var: $v_C$, the jittered diagonal of the prior covariance at $C$, repeated
            over the $E$ outputs, `[n_C E]`.
        kernel: The name of the prior's kernel, for display only. The kernel itself was
            used at fit time.
        strategy: How the fit formed $J(x) U_M$ for the Gram matrix.
        versions: The version counters of the module's fixed parameters at fit time (see
            `check_versions`).
    """

    model: ModelFn = static()
    likelihood: _L_co
    mean: Params
    caps: Capabilities = static()
    layout: Layout = static()
    basis: Tensor
    prior_evals: Tensor
    gram: Tensor
    n_data: Tensor
    context_jac: Tensor
    prior_var: Tensor
    kernel: str = static()
    strategy: Strategy = static()
    versions: Mapping[str, int] = static(default_factory=dict[str, int])

    def __repr__(self) -> str:
        kept = self._range()[1].shape[0]
        size = sum(p.numel() for p in self.mean.values())

        return (
            f"FSPPosterior(rank={kept} of {self.basis.shape[1]}, parameters={size}, "
            f"kernel={self.kernel}, strategy={self.strategy!r})"
        )

    def check_versions(self) -> None:
        """Raise `RuntimeError` if a fixed parameter of the module changed in place since the fit.

        The basis, the context Jacobian and the Gram matrix were computed with the old
        value, so the posterior no longer describes the model. `cov_mv`, `cov_sqrt_mv`,
        `sample_centered`, `sample`, `lk.core.mean_outputs`, `lk.predict_outputs` and
        `lk.sample_outputs` call this first.
        """

        check_versions(
            self.model,
            self.versions,
            "the basis, the context Jacobian and the Gram matrix used its old value",
        )

    # the factor

    def _range(self) -> tuple[Tensor, Tensor]:
        """V = U_M U_A and D_A on the kept columns, so that S = V D_A^{-1/2}."""

        c = curvature_scale(self.likelihood)
        data = self.n_data * self.gram

        if c is not None:
            data = c.reshape(()).to(data.dtype) * data

        a = torch.diag(self.prior_evals) + data
        a = (a + a.mT) / 2
        d, w = torch.linalg.eigh(a)

        # A's pseudo-inverse: drop eigenvalues at most eps r max.
        keep = d > torch.finfo(d.dtype).eps * d.shape[0] * d.max()
        d, w = d[keep], w[:, keep]

        # Truncation: tail[:, k] is the variance at C of the columns k, k+1, ...
        js = self.context_jac.to(w.dtype) @ w / d.sqrt()
        tail = js.square().flip(1).cumsum(1).flip(1)
        bound = self.prior_var.to(w.dtype) * (1 + 100 * torch.finfo(w.dtype).eps)
        ok = (tail <= bound.unsqueeze(1)).all(0)
        hits = ok.nonzero()
        k = int(hits[0, 0]) if hits.numel() > 0 else ok.shape[0]

        return self.basis @ w[:, k:], d[k:]

    def _factor(self) -> Tensor:
        vecs, evals = self._range()

        return vecs / evals.sqrt()

    # params <-> flat vectors, in the order of the mean

    def _vec(self, v: Params) -> Tensor:
        parts = [v[k].reshape(*v[k].shape[: v[k].ndim - m.ndim], -1) for k, m in self.mean.items()]

        return torch.cat(parts, dim=-1).to(self.basis.dtype)

    def _unvec(self, x: Tensor, *, cast: bool = True) -> Params:
        sizes = [m.numel() for m in self.mean.values()]
        parts = torch.split(x, sizes, dim=-1)
        out = {
            k: p.reshape(*x.shape[:-1], *m.shape)
            for (k, m), p in zip(self.mean.items(), parts, strict=True)
        }

        return {k: v.to(self.mean[k].dtype) for k, v in out.items()} if cast else out

    # the Gaussian

    def precision(self) -> RangePrecision:
        r"""The pseudo-inverse of $S S^\top$, which is $V D_A V^\top$ with $V = U_M U_A$.

        Both use only the kept columns.

        Returns:
            The pseudo-inverse precision, as a `RangePrecision`.
        """

        vecs, evals = self._range()

        return RangePrecision(vecs, evals)

    def cov_mv(self, v: Params, /) -> Params:
        r"""$S S^\top v$.

        Args:
            v: A parameter mapping, with the names and shapes of the mean.

        Returns:
            $S S^\top v$, as a parameter mapping like `v`.

        Raises:
            RuntimeError: A fixed parameter of the module changed in place since the fit.
        """

        self.check_versions()
        s = self._factor()

        return self._unvec((self._vec(v) @ s) @ s.T)

    def cov_sqrt_mv(self, v: Params, /) -> Params:
        r"""$L v$ with the square root $L = [S, 0]$, so $L L^\top = S S^\top$.

        Only the first $r'$ entries of $v$ matter.

        Args:
            v: A parameter mapping, with the names and shapes of the mean.

        Returns:
            $L v$, as a parameter mapping like `v`.

        Raises:
            RuntimeError: A fixed parameter of the module changed in place since the fit.
        """

        self.check_versions()
        s = self._factor()

        return self._unvec(self._vec(v)[..., : s.shape[1]] @ s.T)

    def sample_centered(self, n: int, /, *, generator: torch.Generator | None = None) -> Params:
        r"""`n` draws $S z$ with $z \sim \mathcal{N}(0, I)$, in the basis's dtype.

        Args:
            n: The number of draws.
            generator: The generator to draw with, on any device: the draws move to
                the basis's. `None` means torch's global RNG.

        Returns:
            A parameter mapping whose tensors have a leading axis of `n` draws.

        Raises:
            RuntimeError: A fixed parameter of the module changed in place since the fit.
        """

        self.check_versions()
        s = self._factor()

        home = generator.device if generator is not None else torch.device("cpu")
        z = torch.randn(n, s.shape[1], generator=generator, dtype=s.dtype, device=home)

        return self._unvec(z.to(s.device) @ s.T, cast=False)

    def sample(self, n: int, /, *, generator: torch.Generator | None = None) -> Params:
        r"""`n` draws from $\mathcal{N}(\theta_*, S S^\top)$.

        Args:
            n: The number of draws.
            generator: The generator to draw with, on any device: the draws move to
                the basis's. `None` means torch's global RNG.

        Returns:
            A parameter mapping whose tensors have a leading axis of `n` draws.

        Raises:
            RuntimeError: A fixed parameter of the module changed in place since the fit.
        """

        self.check_versions()
        noise = self.sample_centered(n, generator=generator)

        return {k: self.mean[k] + noise[k].to(self.mean[k].dtype) for k in self.mean}

    @property
    def variance(self) -> Params:
        r"""$\operatorname{diag}(S S^\top)$.

        Returns:
            A parameter mapping with the variance of every entry, shaped like the mean.
        """

        return self._unvec(self._factor().square().sum(1))

    # not a prior

    def log_prob(self, params: Params, /) -> Tensor:
        """Not available, because the posterior has no density over the weights.

        Args:
            params: The parameters at which to evaluate the density.

        Returns:
            It never returns, because it always raises `StructureError`.

        Raises:
            StructureError: Always.
        """

        raise StructureError(self._not_a_prior("log_prob"))

    def location(self, like: Params, /) -> Params:
        r"""The mean, after checking that it has the names, shapes, dtypes and devices of `like`.

        Args:
            like: A parameter mapping whose names, shapes, dtypes and devices the mean must match.

        Returns:
            The mean, $\theta_*$.
        """

        check_location(self.mean, like)

        return self.mean

    def add_precision_to[Op: Operator[Any]](self, op: Op, /) -> Op:
        """Not available, because the posterior cannot be the prior of a Laplace update.

        Args:
            op: The operator that would receive the precision.

        Returns:
            It never returns, because it always raises `StructureError`.

        Raises:
            StructureError: Always.
        """

        raise StructureError(self._not_a_prior("add_precision_to"))

    def _not_a_prior(self, what: str) -> str:
        kept = self._range()[1].shape[0]
        size = sum(p.numel() for p in self.mean.values())

        return (
            f"FSPPosterior.{what}: {_degenerate(kept, size)}; it has no density over the "
            "weights and can't be the prior of a Laplace update"
        )


# --- the method and the front door ----------------------------------------------------------


def _requirements(context: Points, strategy: Strategy) -> Requirements:
    # The context Jacobian needs vjp and vmap, the Gram pass the output Hessian (and jvp
    # when it pushes forward). A pick pass from the data ignores weights.
    picks = not isinstance(context, Drawn)

    return Requirements(
        vjp=True,
        jvp=strategy == "forward-mode",
        vmap=True,
        needs_output_hessian=True,
        additive_stat=not picks,
        weights=not picks,
    )


def _check_budget(n_c: int, event: int, n_params: int) -> None:
    entries = n_c * event * n_params

    if entries > MAX_CONTEXT_JACOBIAN_ENTRIES:
        raise StructureError(
            f"fsp_laplace holds the Jacobian at the context points, n_C*E*P = {entries} "
            f"entries (n_C={n_c}, E={event}, P={n_params}), above "
            f"MAX_CONTEXT_JACOBIAN_ENTRIES={MAX_CONTEXT_JACOBIAN_ENTRIES}; use fewer context "
            "points or fewer parameters (subset=...)"
        )


@pytree
class _FSP:
    """FSP-Laplace's passes: pick the context points, build the basis, sum the projected GGN."""

    prior: FunctionPrior
    context: Points
    rank: int = static()
    strategy: Strategy = static()

    def __repr__(self) -> str:
        return (
            f"fsp_laplace(context={self.context!r}, rank={self.rank}, strategy={self.strategy!r})"
        )

    @property
    def requires(self) -> Requirements:
        return _requirements(self.context, self.strategy)

    def plan(self, params: Params, ctx: FitContext, /) -> Pass[FSPPosterior]:
        if isinstance(ctx.likelihood, Mapping):
            raise StructureError(
                "fsp_laplace needs a model returning one tensor, but the likelihood is a mapping"
            )

        n_params = sum(v.numel() for v in params.values())
        n_c, event = point_count(self.context), ctx.caps.event_size

        if n_c is not None and event is not None:
            _check_budget(n_c, event, n_params)

        check_event(ctx, "fsp_laplace")

        out = self.context.pick(params, ctx, lambda picked: self._basis(params, ctx, picked))

        if isinstance(out, Done):
            raise TypeError(f"{self.context!r}.pick must hand its points to the next step")

        return out

    def _basis(self, params: Params, ctx: FitContext, picked: Picked) -> Pass[FSPPosterior]:
        """Form J(C) and the basis U_M, then plan the pass that sums the GGN projected onto U_M."""

        inputs = picked.inputs

        if not isinstance(inputs, Tensor) or inputs.ndim == 0:
            raise StructureError(
                "fsp_laplace needs the context points as one tensor [n_C, ...], which the "
                "kernel takes, got "
                f"{type(inputs).__name__ if not isinstance(inputs, Tensor) else 'a 0-dim tensor'}"
            )

        f = ctx.model(params, inputs)

        if not isinstance(f, Tensor):
            raise StructureError(
                f"fsp_laplace needs a model returning one tensor, got {type(f).__name__}"
            )

        n_c, event = f.shape[0], math.prod(f.shape[1:])
        _check_budget(n_c, event, sum(v.numel() for v in params.values()))

        jac_c = rows(ctx.model, params, inputs, None, torch.float64)
        fb = fsp_basis(jac_c, self.prior, inputs, event, self.rank)

        dtype = stat_dtype(next(iter(params.values())).dtype)
        basis = fb.basis.to(dtype)
        fixed = {k: v for k, v in param_versions(ctx.model, params).items() if k not in params}

        def done(stat: GramStat) -> Done[FSPPosterior]:
            post = FSPPosterior(
                model=ctx.model,
                likelihood=ctx.likelihood,
                mean=params,
                caps=ctx.caps,
                layout=ctx.layout,
                basis=basis,
                prior_evals=fb.prior_evals.to(dtype),
                gram=stat.gram,
                n_data=stat.weight,
                context_jac=fb.context_jac.to(dtype),
                prior_var=fb.prior_var.to(dtype),
                kernel=type(self.prior.kernel).__name__,
                strategy=self.strategy,
                versions=fixed,
            )

            return Done(post)

        return chain(ProjectedGram(basis, self.strategy), then=done)


def fsp_laplace[L: Likelihood](
    model: nn.Module | ModelFn,
    data: lk.data.Data | Iterable[Any],
    likelihood: L,
    *,
    prior: FunctionPrior,
    context: Points,
    rank: int,
    subset: Subset | Params | Selection = lk.subsets.last_layer,
    strategy: Strategy | None = None,
    generator: torch.Generator | None = None,
) -> FSPPosterior[L]:
    r"""FSP-Laplace (Cinquin et al. 2024, Algorithm 2): Laplace under a GP prior on the outputs.

    Fit it at a MAP trained with `rkhs_norm` in the loss. The probe and the parameter
    subset work as in `lk.function_space.laplace`. `context.pick` gives the $n_C$ context points
    $C$. The fit forms the dense Jacobian $J(C)$, `[n_C E, P]`, and builds the basis $U_M$ of the
    prior precision

    $$J(C)^\top \Sigma(C, C)^{-1} J(C), \qquad \Sigma(C, C) = K_{CC} \otimes I_E,$$

    where $K_{CC}$ is `prior.cov(C)`. It then sums the GGN projected onto $U_M$ in one pass
    over the data, or two when the context points are picked from the data.

    To build the basis, Lanczos runs on $K_{CC}$ rather than on $K_{CC} \otimes I_E$, whose
    eigenvalues each have multiplicity $E$. It gives $L_C$ with
    $L_C L_C^\top \approx K_{CC}^{-1}$, and the thin SVD

    $$J(C)^\top (L_C \otimes I_E) = U_M D_M V_M^\top$$

    gives $U_M$ and $D_M^2$. The SVD drops singular values at most $\epsilon$ times the
    larger side of that matrix times the largest singular value. All of this runs in
    float64, and $\epsilon$ is its machine epsilon.

    Predict with `Linearized`, the default of `lk.predict(post, x)`. `WeightSamples` also
    runs, but the covariance is large along the directions that the context points barely
    constrain, so draws of the network can be far from the linearized predictive.

    `rank`, the model's capabilities and the budget are checked before any data beyond
    the probe's batch. With `Drawn` context points (`grid`, `uniform`, `fixed`), the
    checks that the points and the model's outputs are each one tensor run early as well,
    and so do the kernel's checks of the points.

    Args:
        model: The model. A module is wrapped with `ModelFn.from_module`.
        data: A `Data` source, or a re-iterable of `Batch`es or `(inputs, targets)` pairs,
            which `lk.data.from_loader` wraps.
        likelihood: The likelihood. The posterior reads its `sigma` on every use, so
            tuning can still change it.
        prior: The function prior. The fit uses its kernel once, and the posterior keeps
            only the kernel's name, so `lk.tune` cannot change a kernel hyperparameter.
            Fit again to use another kernel.
        context: Picks the context points $C$ as one tensor that the kernel accepts:
            `grid`, `uniform`, `fixed`, `from_data` or `kmeans`.
        rank: The number of Lanczos steps on the $n_C \times n_C$ kernel matrix. At most
            $n_C$ of them run. The basis has up to $\text{rank} \cdot E$ columns.
        subset: The parameters to fit, as for `lk.function_space.laplace`: a subset
            function, a parameter mapping or a `Selection`.
        strategy: How the pass over the data forms $J(x) U_M$. `None` uses forward mode when the
            probe found it and vjp otherwise. A mode given here must be one the probe
            found.
        generator: Seeds the data passes and the fit's draws. One seed is drawn from it,
            or from torch's global RNG when `None`, so `torch.manual_seed` makes the fit
            repeatable.

    Returns:
        The posterior. It holds a copy of the fitted parameters and the likelihood.
        Changing another parameter of the module in place afterwards makes `cov_mv`
        raise.

    Raises:
        ValueError: `rank` is below 1, `strategy` is not `"forward-mode"`, `"reverse-mode"` or
            `None`, the data has no batches or no examples, a weight is negative, the total weight
            is zero, the Lanczos start vector (the row sums of $J(C)$ added over the $E$
            outputs) is zero, or the likelihood finds a bad target value.
        CapabilityError: The model does not support vjp, vmap, or jvp with
            `strategy="forward-mode"`. The likelihood has neither an output Hessian nor sampling for
            the Monte Carlo Fisher. A batch carries weights while the context points are
            picked from the data, or `last_layer` finds no module with parameters.
        StructureError: $n_C E P$ exceeds `MAX_CONTEXT_JACOBIAN_ENTRIES` (the built-in
            points know $n_C$ before the data), the context points are not one tensor,
            the model's outputs are not one tensor, the likelihood is a mapping, the MC
            Fisher (a likelihood without `HasOutputHessian`) would form dense blocks above
            `MAX_DENSE_EVENT`, the kernel refuses the context points, `kmeans` holds too
            many entries, the subset refuses the model, or the probe or a pass finds badly
            shaped outputs or targets.
        TypeError: `data` is a one-shot iterator, an item is neither a `Batch` nor an
            `(inputs, targets)` pair, or `context.pick` returns `Done` instead of handing
            its points on.
    """

    if rank < 1:
        raise ValueError(f"fsp_laplace needs rank >= 1, got {rank}")

    if strategy is not None and strategy not in ("forward-mode", "reverse-mode"):
        raise ValueError(
            f'fsp_laplace strategy must be "forward-mode", "reverse-mode" or None, got {strategy!r}'
        )

    gen = lk.core.spawn_generator(generator)
    fn = ModelFn.from_module(model) if isinstance(model, nn.Module) else model
    src = lk.data.as_data(data)

    # The likelihood needs don't depend on the mode. (`lk.core.probe`: importing the name
    # `probe` reads as the submodule to import-linter.)
    check_likelihood(_requirements(context, "reverse-mode"), likelihood, "fsp_laplace")

    first = next(iter(src(gen)), None)

    if first is None:
        raise ValueError("fsp_laplace: the data yielded no batches")

    order = lk.core.probe(fn, likelihood, first, params={})
    picked = (
        cast(Params | Selection, subset)
        if isinstance(subset, Selection | Mapping)
        else subset(fn, order)
    )

    # A `Selection` fits its entries as parameters of their own (see `lk.laplace`).
    fn, picked_params = select(fn, picked)
    chosen = {k: v.detach() for k, v in picked_params.items()}

    caps = lk.core.probe(fn, likelihood, first, params=chosen)
    method = _FSP(prior, context, rank, choose_via(caps) if strategy is None else strategy)
    caps.check(method.requires, repr(method))
    method.requires.check_batch(first, repr(method))

    # `start` runs `plan`, which checks the budget (and given context points) before data.
    fit = fit_curvature(fn, likelihood, chosen, method, generator=gen, capabilities=caps)
    step = run(fit, src, generator=gen)

    # The posterior owns a copy of the fitted parameters, so training the module later leaves
    # it in place.
    fitted = {id(v) for v in chosen.values()}

    def own(t: Any) -> Any:
        if not isinstance(t, Tensor):
            return t

        return t.clone() if id(t) in fitted else t.detach()

    result = tree_map(own, step.result)

    # The fit's posterior is typed with any likelihood; with `likelihood` swapped in, it is L.
    return cast(FSPPosterior[L], copy.replace(result, likelihood=likelihood))
