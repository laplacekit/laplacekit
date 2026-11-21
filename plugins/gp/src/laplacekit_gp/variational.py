r"""VaLLA (Ortega et al. 2024), the linearized Laplace GP with variational inducing outputs.

`VaLLA(points)` picks $M$ inducing inputs $Z$, gives each one output coordinate, and
counts the data. The fit returns a `VariationalGP`: a GP over the outputs with kernel
$K = J \Sigma_0 J^\top$, conditioned on the inducing outputs $f_z$ through a variational
factor $L$ that starts at the identity. `fit_variational` then trains $L$ by Adam on
`variational_loss`, together with the prior's precision, $\sigma$ and the inducing inputs.
"""

import copy
import dataclasses
import math
from collections.abc import Callable, Iterable, Iterator, Mapping
from typing import Any, Generic, Literal, Self, TypeVar, cast, get_args

import torch
from torch import Tensor

import laplacekit as lk
import laplacekit_gp._jacobian as _jacobian
import laplacekit_gp.nystrom as _nystrom
from laplacekit.core import (
    Batch,
    Capabilities,
    CapabilityError,
    Done,
    FitContext,
    HasMask,
    HasTargetCheck,
    Inputs,
    Likelihood,
    ModelFn,
    OutputGaussian,
    Params,
    Pass,
    PerInputCov,
    PriorWithCovariance,
    Requirements,
    StructureError,
    TunablePrior,
    as_pytree,
    chain,
    detach_leaves,
    log_prob,
    param_versions,
    pytree,
    replace_leaves,
    stat_dtype,
    static,
    tree_map,
)
from laplacekit.function_space import cov_rows, rows_of
from laplacekit_gp._jacobian import rows
from laplacekit_gp._versions import check_versions
from laplacekit_gp.points import (
    Drawn,
    Fixed,
    Picked,
    Points,
    check_coords,
    point_count,
)

__all__ = [
    "VaLLA",
    "VariationalGP",
    "fit_variational",
    "variational_loss",
]

_Coords = Literal["predicted", "random"]

_WORK = torch.float64


def _outputs(model: ModelFn, params: Params, inputs: Inputs, what: str) -> Tensor:
    f = model(params, inputs)

    if not isinstance(f, Tensor):
        raise StructureError(f"{what} needs a model returning one tensor, got {type(f).__name__}")

    return f


def _check_budgets(m: int | None, event: int | None, n_params: int) -> None:
    """Check M P and E P against `MAX_BASIS_ENTRIES`, for whichever sizes are known."""

    limit = _nystrom.MAX_BASIS_ENTRIES

    if m is not None and m * n_params > limit:
        raise StructureError(
            f"VaLLA holds the Jacobian rows of its inducing points, M*P = "
            f"{m * n_params} entries (M={m}, P={n_params}), above MAX_BASIS_ENTRIES={limit}; "
            "use fewer points or fewer parameters (subset=...)"
        )

    if event is not None and event * n_params > limit:
        raise StructureError(
            f"VariationalGP.at holds one input's Jacobian, E*P = {event * n_params} entries "
            f"(E={event}, P={n_params}), above MAX_BASIS_ENTRIES={limit}; use fewer "
            "parameters (subset=...)"
        )


# --- the posterior -----------------------------------------------------------------------


# Explicit variance, as for `LaplacePosterior`: the likelihood's type, covariant, so a
# posterior with a Gaussian is assignable to a bare `VariationalGP`.
_L_co = TypeVar("_L_co", bound=Likelihood, covariant=True, default=Likelihood)


@pytree
class VariationalGP(Generic[_L_co]):  # noqa: UP046
    r"""The linearized Laplace prior conditioned on $M$ inducing outputs, as a GP over the outputs.

    Let $J_a$ be the Jacobian at the mean for inputs $a$, $\Sigma_0$ the prior's covariance
    (applied through `cov_mv`), and $K_{ab} = J_a \Sigma_0 J_b^\top$. For the inducing
    inputs $z$, $J_z$ holds one row per point, at the output that `coords` names. With
    $L = \operatorname{tril}(\text{factor})$, the covariance at an input $x$ is

    $$K_{xx} - (L^\top K_{zx})^\top \, (I + L^\top K_{zz} L)^{-1} \, (L^\top K_{zx}).$$

    This is the prior conditioned on pseudo-observations of $f_z$ with noise covariance
    $(L L^\top)^{-1}$. The mean is $f(x)$ at the mean parameters.

    `at` and `kl` read the prior on every call. Both are differentiable in `factor`, in
    the prior's tensors and in the inducing inputs. They compute in float64 and return
    results in the `stat_dtype` of the outputs.

    Args:
        model: The model as a function of the parameters.
        likelihood: The likelihood of the fit. Its tensors, such as `sigma`, are leaves
            that tuning can change.
        mean: The fitted parameters.
        prior: The prior, whose covariance is $\Sigma_0$.
        caps: The model's capabilities.
        inducing: The $M$ inducing inputs $z$, along the leading axis.
        coords: The flattened output coordinate of each inducing point, `[M]` int64.
        factor: The variational factor, `[M, M]`. Only its lower triangle $L$ is used.
        n_data: The total weight of the data, $N$ in `variational_loss`.
        versions: The version counters of the module's fixed parameters at fit time. `at`
            and `kl` raise `RuntimeError` when one has changed in place since then (see
            `check_versions`).
    """

    model: ModelFn = static()
    likelihood: _L_co  # a child: its tensors (sigma) are leaves to tune
    mean: Params
    prior: PriorWithCovariance
    caps: Capabilities = static()
    inducing: Inputs
    coords: Tensor
    factor: Tensor
    n_data: Tensor
    versions: Mapping[str, int] = static(default_factory=dict[str, int])

    def __repr__(self) -> str:
        size = sum(p.numel() for p in self.mean.values())

        return f"VariationalGP(m={self.coords.shape[0]}, parameters={size}, prior={self.prior!r})"

    @property
    def requires(self) -> Requirements:
        """What the Jacobian rows need: vjp and vmap."""

        return Requirements(vjp=True, vmap=True)

    def check_versions(self) -> None:
        """Raise `RuntimeError` if a fixed parameter of the module changed in place since the fit.

        The GP linearizes the model at the old value. `at` and `kl` call this first.
        """

        check_versions(self.model, self.versions, "the GP linearizes the model at its old value")

    def at(self, inputs: Inputs, /) -> OutputGaussian:
        r"""The Gaussian over the outputs at `inputs`, with one $E \times E$ block per input.

        The Jacobian at `inputs` is formed a few inputs at a time, in blocks of at most
        $2^{24}$ entries (inputs times $E$ times $P$) or of one input.

        Args:
            inputs: The inputs, with the examples along the leading axis.

        Returns:
            An `OutputGaussian` whose mean is the network's outputs at `inputs` and whose covariance
            holds one $E \times E$ block per input.

        Raises:
            CapabilityError: The model does not support vjp or vmap.
            RuntimeError: A fixed parameter of the module changed in place since the fit.
            StructureError: The model's outputs are not one tensor, or `prior.cov_mv`
                refuses the parameters.
        """

        self.caps.check(self.requires, "VariationalGP.at")
        self.check_versions()

        return self._at(inputs, self._inducing())

    def _at(
        self, inputs: Inputs, inducing: tuple[Tensor, Tensor, Tensor, Tensor]
    ) -> OutputGaussian:
        inputs = as_pytree(inputs)  # a `BatchEncoding` as a dict, so `vmap` sees its tensors
        f = _outputs(self.model, self.mean, inputs, "VariationalGP.at")
        n, event = f.shape[0], tuple(f.shape[1:])
        e = math.prod(event)

        sz, low, _, chol = inducing
        step = max(1, _jacobian.VJP_CHUNK_ENTRIES // max(1, e * sz.shape[1]))
        blocks = []

        # The covariance is per input, so the query Jacobian comes a few inputs at a time.
        for s in range(0, n, step):
            part = rows_of(inputs, torch.arange(s, min(s + step, n)))
            blocks.append(self._cov_block(part, sz, low, chol, e))

        cov = torch.cat(blocks).to(stat_dtype(f.dtype))

        return OutputGaussian(f, PerInputCov(cov, event))

    def _cov_block(self, inputs: Inputs, sz: Tensor, low: Tensor, chol: Tensor, e: int) -> Tensor:
        """The `[n, E, E]` covariance blocks of a few inputs, in float64."""

        jx = rows(self.model, self.mean, inputs, None, _WORK)  # [n E, P]
        sx = cov_rows(self.prior, self.mean, jx)
        n = jx.shape[0] // e

        k_xx = torch.einsum("nep,nfp->nef", jx.reshape(n, e, -1), sx.reshape(n, e, -1))

        b = low.T @ (sz @ jx.T)  # Lᵀ K_zx, [M, n E]
        w = torch.linalg.solve_triangular(chol, b, upper=False).reshape(-1, n, e)

        cov = k_xx - torch.einsum("mne,mnf->nef", w, w)

        return 0.5 * (cov + cov.mT)

    def kl(self) -> Tensor:
        r"""The KL divergence $\mathrm{KL}(q(f_z) \,\Vert\, p(f_z))$ at the inducing outputs.

        $$\mathrm{KL} = \tfrac{1}{2} \big(\log\det(I + A) - \operatorname{tr}((I + A)^{-1} A)\big),
        \qquad A = L^\top K_{zz} L.$$

        Both Gaussians have mean zero. The covariance of $q$ is
        $(K_{zz}^{-1} + L L^\top)^{-1}$, and that of the prior is $K_{zz}$.

        Returns:
            A 0-dim tensor, in `stat_dtype` of the mean.

        Raises:
            CapabilityError: The model does not support vjp or vmap.
            RuntimeError: A fixed parameter of the module changed in place since the fit.
            StructureError: `prior.cov_mv` refuses the parameters.
        """

        self.caps.check(self.requires, "VariationalGP.kl")
        self.check_versions()

        return self._kl(self._inducing())

    def _kl(self, inducing: tuple[Tensor, Tensor, Tensor, Tensor]) -> Tensor:
        _, _, a, chol = inducing
        half = torch.linalg.solve_triangular(chol, a, upper=False)
        trace = torch.linalg.solve_triangular(chol, half.T, upper=False).diagonal().sum()
        logdet = 2 * chol.diagonal().log().sum()
        dtype = stat_dtype(next(iter(self.mean.values())).dtype)

        return (0.5 * (logdet - trace)).to(dtype)

    def _at_and_kl(self, inputs: Inputs) -> tuple[OutputGaussian, Tensor]:
        """`at(inputs)` and `kl()`, computed with one inducing Jacobian and one Cholesky factor."""

        self.caps.check(self.requires, "variational_loss")
        self.check_versions()
        inducing = self._inducing()

        return self._at(inputs, inducing), self._kl(inducing)

    def _inducing(self) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Σ₀ J_zᵀ (as rows), L, A = Lᵀ K_zz L and the Cholesky factor of I + A, in float64."""

        jz = rows(self.model, self.mean, as_pytree(self.inducing), self.coords, _WORK)
        sz = cov_rows(self.prior, self.mean, jz)
        low = self.factor.to(_WORK).tril()

        a = low.T @ (jz @ sz.T) @ low
        a = 0.5 * (a + a.T)
        eye = torch.eye(a.shape[0], dtype=_WORK, device=a.device)

        return sz, low, a, torch.linalg.cholesky(eye + a)


# --- the count pass -------------------------------------------------------------------


@pytree
class CountStat:
    """The total weight of the data seen.

    Args:
        weight: The total weight, a 0-dim tensor.
    """

    weight: Tensor

    @property
    def total_weight(self) -> Tensor:
        """The total weight."""

        return self.weight

    def __add__(self, other: Self, /) -> Self:
        return type(self)(self.weight + other.weight)

    def scaled(self, s: Tensor, /) -> Self:
        """The statistic with the weight multiplied by `s`."""

        return type(self)(s * self.weight)


@pytree
class Count:
    """A pass that sums the weights of the data, counting 1 per example without weights.

    It runs no forward pass.
    """

    def __repr__(self) -> str:
        return "Count()"

    @property
    def requires(self) -> Requirements:
        """No requirements."""

        return Requirements()

    def init(self, params: Params, ctx: FitContext, /) -> CountStat:
        """A zero count, in `stat_dtype` of the first parameter."""

        p = next(iter(params.values()))

        return CountStat(torch.zeros((), dtype=stat_dtype(p.dtype), device=p.device))

    def update(
        self,
        stat: CountStat,
        params: Params,
        batch: Batch,
        generator: torch.Generator,
        ctx: FitContext,
        /,
    ) -> CountStat:
        """Add the batch's total weight to the count.

        Raises:
            ValueError: A weight is negative.
        """

        if batch.weights is None:
            seen = torch.tensor(float(batch.size), dtype=stat.weight.dtype)
        elif bool((batch.weights < 0).any()):
            raise ValueError("Batch.weights must be non-negative")
        else:
            seen = batch.weights.sum()

        return copy.replace(stat, weight=stat.weight + seen.to(stat.weight))

    def finalize(self, stat: CountStat, /) -> CountStat:
        """Return the count unchanged.

        Raises:
            ValueError: The total weight is not positive.
        """

        if not bool(stat.weight > 0):
            raise ValueError("Count saw no data with positive weight")

        return stat


# --- the method -----------------------------------------------------------------------


@pytree
class VaLLA:
    r"""VaLLA (Ortega et al. 2024), the linearized Laplace GP with variational inducing outputs.

    `points` picks the $M$ inducing inputs, and each gets one output coordinate. A second
    pass sums the weights of the data without running the model. `fs.laplace` then
    returns a `VariationalGP` with factor $I$ and that total as its `scale`.

    laplace-torch uses the training labels of points picked from the data as their
    coordinates, where `"predicted"` uses the argmax of the outputs. `fixed(x, coords=y)`
    gives the label coordinates. In regression with $E > 1$, the argmax depends on the
    scales of the outputs, and laplace-torch takes output 0.

    Any pushable prior works. The model must support vjp and vmap. Points picked from the
    data refuse weighted batches, and with `fixed` points the count uses the weights. A
    model whose outputs are not one tensor is refused before any data when the likelihood
    is a mapping, and otherwise when the coordinates are set, after the points are picked.

    Some checks run before any data and raise `StructureError`: coordinates must lie in
    $[0, E)$, and both $M P$ (the inducing Jacobian rows) and $E P$ (the Jacobian of one
    input, the smallest block of `at`) must stay within `MAX_BASIS_ENTRIES`. Every
    built-in `Points` knows $M$ before the data. For other points, $M$ and the coordinates
    they set are checked when the points are handed over.

    Args:
        points: Selects the inducing inputs, for example `fixed`, `from_data` or `kmeans`.
        coords: How a point gets its output coordinate when `fixed(..., coords)` gives
            none. `"predicted"` takes the argmax of the outputs at the point, and
            `"random"` takes $\lfloor u E \rfloor$ for its `Picked.u`.

    Raises:
        ValueError: `coords` is neither `"predicted"` nor `"random"`.
    """

    points: Points
    _: dataclasses.KW_ONLY
    coords: _Coords = static(default="predicted")

    def __post_init__(self) -> None:
        if self.coords not in get_args(_Coords):
            raise ValueError(f'VaLLA coords must be "predicted" or "random", got {self.coords!r}')

    def __repr__(self) -> str:
        return f"VaLLA(points={self.points!r}, coords={self.coords!r})"

    @property
    def requires(self) -> Requirements:
        """What the passes need: vjp and vmap. Points picked from the data refuse weights."""

        # Every pass, declared up front: a pick pass ignores weights; the count honours them.
        picks = not isinstance(self.points, Drawn)

        return Requirements(vjp=True, vmap=True, additive_stat=not picks, weights=not picks)

    def plan(
        self, params: Params, ctx: FitContext, prior: PriorWithCovariance, /
    ) -> Pass[VariationalGP]:
        """Plan the passes: pick the points, then count the data.

        `Drawn` points need no data, so `plan` sets their coordinates right away and
        raises those errors itself. Points picked from the data get their coordinates
        after the pass that picks them.

        Returns:
            The first pass. Running it raises `ValueError` when the data has no
            examples, a weight is negative or the total weight is not positive, and
            `StructureError` when the model's outputs are not one tensor, a picked
            coordinate lies outside $[0, E)$, or $M P$ exceeds `MAX_BASIS_ENTRIES`.

        Args:
            params: The parameters to fit, as the fit's subset gives them.
            ctx: The fit's context: the model, the likelihood, the capabilities the probe found, and
                the layout of the parameters.
            prior: The prior over the parameters.

        Raises:
            StructureError: The likelihood is a mapping, $M P$ or $E P$ exceeds
                `MAX_BASIS_ENTRIES` for sizes known before data, or a given coordinate
                lies outside $[0, E)$. With `Drawn` points, also when the model's outputs
                are not one tensor, a picked coordinate lies outside $[0, E)$, or $M P$
                or $E P$ exceeds `MAX_BASIS_ENTRIES`.
            TypeError: `points.pick` returns `Done` instead of handing its points on.
        """

        if isinstance(ctx.likelihood, Mapping):
            raise StructureError(
                "VaLLA needs a model returning one tensor, but the likelihood is a mapping"
            )

        n_params = sum(v.numel() for v in params.values())
        event = ctx.caps.event_size
        given = self.points.coords if isinstance(self.points, Fixed) else None
        _check_budgets(point_count(self.points), event, n_params)

        if given is not None and event is not None:
            check_coords(given, event, repr(self))

        def init(picked: Picked) -> Pass[VariationalGP]:
            coords = self._coords(params, ctx, picked, n_params)
            m = coords.shape[0]
            p = next(iter(params.values()))
            dtype = stat_dtype(p.dtype)
            fixed = {k: v for k, v in param_versions(ctx.model, params).items() if k not in params}

            def done(stat: CountStat) -> Done[VariationalGP]:
                gp = VariationalGP(
                    ctx.model,
                    ctx.likelihood,
                    params,
                    prior,
                    ctx.caps,
                    picked.inputs,
                    coords,
                    torch.eye(m, dtype=dtype, device=p.device),
                    stat.weight,
                    fixed,
                )

                return Done(gp)

            return chain(Count(), then=done)

        out = self.points.pick(params, ctx, init)

        if isinstance(out, Done):
            raise TypeError(f"{self.points!r}.pick must hand its points to the next step")

        return out

    def _coords(self, params: Params, ctx: FitContext, picked: Picked, n_params: int) -> Tensor:
        """One flattened output coordinate per point, `[M]` int64, after checking the budgets."""

        f = _outputs(ctx.model, params, as_pytree(picked.inputs), "VaLLA")
        m, e = f.shape[0], math.prod(f.shape[1:])
        _check_budgets(m, e, n_params)

        if picked.coords is not None:
            check_coords(picked.coords, e, repr(self))

            return picked.coords.to(f.device)

        if self.coords == "predicted":
            return f.reshape(m, e).argmax(-1)

        return (picked.u * e).floor().long().clamp(max=e - 1).to(f.device)


# --- the objective and the fit --------------------------------------------------------


def _check_options(alpha: float, n_samples: int | None) -> None:
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must lie in [0, 1], got {alpha}")

    if n_samples is not None and (isinstance(n_samples, bool) or not isinstance(n_samples, int)):
        raise ValueError(f"samples must be an int or None, got {type(n_samples).__name__}")

    if n_samples is not None and n_samples < 2:
        raise ValueError(f"n_samples must be 2 or more (None for the closed form), got {n_samples}")


def _check_likelihood(lik: Likelihood, alpha: float, n_samples: int | None) -> None:
    """Refuse `n_samples=None` when the likelihood has no closed-form data term."""

    if n_samples is not None or isinstance(lik, lk.Gaussian):
        return

    name = type(lik).__name__

    if isinstance(lik, lk.Categorical | lk.Bernoulli):
        if alpha != 1:
            raise ValueError(
                f"the closed-form data term for {name} is the probit approximation, the same "
                f"for every alpha, so alpha={alpha} needs Monte Carlo: pass n_samples="
            )

        return

    raise CapabilityError(
        f"variational_loss has a closed-form data term for Gaussian, Categorical and "
        f"Bernoulli only, not {name}; pass n_samples= for a Monte Carlo one"
    )


def _gaussian_term(lik: lk.Gaussian, f: Tensor, y: Tensor, v: Tensor, alpha: float) -> Tensor:
    """laplace-torch's alpha energy per example, which is the ELBO at α = 0.

    −½ Σ [(y − f)²/(σ² + αv) + log(1 + αv/σ²)/α + log 2πσ²] over the entries.
    """

    if y.shape != f.shape:
        raise StructureError(
            f"Gaussian needs targets shaped like f {tuple(f.shape)}, got {tuple(y.shape)}"
        )

    if lik.sigma is None:
        var = torch.ones((), dtype=f.dtype, device=f.device)
    elif lik.per_output:
        lik.check_outputs(f)
        var = lik.sigma.to(f.device) ** 2
    else:
        var = lik.sigma.to(f.device).reshape(()) ** 2

    sq = (y - f) ** 2

    if alpha == 0:
        energy = (sq + v) / var
    else:
        energy = sq / (var + alpha * v) + torch.log1p(alpha * v / var) / alpha

    term = -0.5 * (energy + torch.log(2 * math.pi * var))

    return term.reshape(f.shape[0], -1).sum(1)


def _repeat(t: Any, s: int) -> Any:
    """Repeat `t` `s` times along its batch axis, one whole copy after another."""

    if not isinstance(t, Tensor):
        return t

    return t.unsqueeze(0).expand(s, *t.shape).reshape(s * t.shape[0], *t.shape[1:])


JITTERS = (1.0, 10.0, 100.0)
"""The relative jitters for the Monte Carlo draws, in units of eps(dtype) · max diag(block)."""


def _draws(out: OutputGaussian, generator: torch.Generator | None, s: int) -> Tensor:
    """Draw `s` samples μ + C ε, `[S, N, *event]`.

    C is the Cholesky factor of each `[E, E]` block plus a jitter. Each block takes the
    smallest jitter j · eps(dtype) · max diag(block), with j from `JITTERS`, for which the
    Cholesky factorization succeeds. Its gradient is finite for any positive definite
    block, even with repeated eigenvalues, which is not true of `eigh`. A block that fails
    at every jitter raises `StructureError`.
    """

    f = cast(Tensor, out.mean)
    blocks = out.cov.per_input()
    n, e = blocks.shape[0], blocks.shape[-1]
    eye = torch.eye(e, dtype=blocks.dtype, device=blocks.device)
    eps = torch.finfo(blocks.dtype).eps
    top = blocks.diagonal(dim1=-2, dim2=-1).amax(-1).detach()
    top = top.clamp_min(torch.finfo(blocks.dtype).tiny)

    # The jitter of each block is picked without gradients; one Cholesky then carries them.
    with torch.no_grad():
        level = torch.full((n,), math.nan, dtype=blocks.dtype, device=blocks.device)

        for j in JITTERS:
            _, info = torch.linalg.cholesky_ex(
                blocks.detach() + (j * eps * top)[:, None, None] * eye
            )
            level = torch.where(level.isnan() & (info == 0), j, level)

    if bool(level.isnan().any()):
        raise StructureError(
            f"variational_loss can't draw from {int(level.isnan().sum())} output covariance "
            f"blocks: their Cholesky fails at every jitter {JITTERS} x eps x max diag; the "
            "covariance is not positive semi-definite"
        )

    chol = torch.linalg.cholesky(blocks + (level * eps * top)[:, None, None] * eye)

    # Drawn on the generator's device, which may differ from the blocks'.
    device = generator.device if generator is not None else blocks.device
    z = torch.randn(s, n, e, generator=generator, dtype=blocks.dtype, device=device)
    z = z.to(blocks.device)
    noise = torch.einsum("nef,snf->sne", chol, z).reshape(s, *f.shape)

    return f + noise.to(f.dtype)


def _mc_term(
    lik: Likelihood,
    out: OutputGaussian,
    batch: Batch,
    alpha: float,
    s: int,
    generator: torch.Generator | None,
) -> Tensor:
    """(1/α) log (1/S) Σₛ p(yₙ | fₛ)^α per example, or the mean log-likelihood at α = 0."""

    f = cast(Tensor, out.mean)
    n = f.shape[0]
    draws = _draws(out, generator, s)  # [S, N, *event]

    y = tree_map(lambda t: _repeat(t, s), batch.targets)
    inputs = tree_map(lambda t: _repeat(t, s), batch.inputs) if isinstance(lik, HasMask) else None
    ll = log_prob(lik, draws.reshape(s * n, *f.shape[1:]), y, inputs).reshape(s, n)

    if alpha == 0:
        return ll.mean(0)

    return (torch.logsumexp(alpha * ll, 0) - math.log(s)) / alpha


def variational_loss(
    gp: VariationalGP,
    batch: Batch,
    /,
    *,
    alpha: float = 1.0,
    n_samples: int | None = None,
    generator: torch.Generator | None = None,
) -> Tensor:
    r"""VaLLA's objective on one batch (Ortega et al. 2024).

    $$\mathcal{L} = -\frac{N}{B} D + \mathrm{KL}\big(q(f_z) \,\|\, p(f_z)\big), \qquad
    D = \sum_n w_n \frac{1}{\alpha} \log \mathbb{E}_q\big[p(y_n \mid f_n)^\alpha\big],$$

    where $N$ is `gp.n_data`, $B$ is the batch's total weight (its size when it has no
    weights), $w_n$ are the weights (1 when there are none), and $D$ is the batch's
    alpha-divergence data term. $\alpha = 0$ gives the ELBO,
    $D = \sum_n w_n \mathbb{E}_q[\log p(y_n \mid f_n)]$. Let $\mu$ and $v$ be the mean and
    variance of an output entry under $q$. The data term follows laplace-torch:

    - With `n_samples=None` and `lk.Gaussian`, it has a closed form, exact for
      $\alpha \in [0, 1]$, with $\sigma$ from the likelihood (1 when `sigma` is `None`).
      Per output entry it is
      $-\tfrac{1}{2} \big[(y - \mu)^2 / (\sigma^2 + \alpha v)
      + \tfrac{1}{\alpha} \log(1 + \alpha v / \sigma^2) + \log 2 \pi \sigma^2\big]$, and
      $-\tfrac{1}{2} \big[((y - \mu)^2 + v) / \sigma^2 + \log 2 \pi \sigma^2\big]$ at
      $\alpha = 0$.
    - With `n_samples=None` and `lk.Categorical` or `lk.Bernoulli`, it is the probit
      approximation $\log p\big(y \mid \mu / \sqrt{1 + \pi v / 8}\big)$. That does not
      depend on $\alpha$, so $\alpha \ne 1$ raises `ValueError`.
    - With `n_samples=S`, it is the Monte Carlo estimate
      $\frac{1}{\alpha} \log \frac{1}{S} \sum_s p(y_n \mid f_s)^\alpha$, or the mean of
      $\log p(y_n \mid f_s)$ at $\alpha = 0$. Each input gets $S$ draws from its
      $E \times E$ Gaussian, scored with `log_prob`. The draws use a Cholesky factor of
      each block, jittered by $j \epsilon \max \operatorname{diag}$ of the block, where
      $\epsilon$ is the machine epsilon and $j$ is the smallest of $(1, 10, 100)$ for
      which the factorization succeeds.

    Args:
        gp: The variational GP.
        batch: One batch, with targets.
        alpha: $\alpha \in [0, 1]$.
        n_samples: $S$, the number of Monte Carlo draws per input, at least 2. `None` uses
            the closed form.
        generator: The generator of the Monte Carlo draws, on any device. `None` means
            torch's global RNG.

    Returns:
        $\mathcal{L}$, a 0-dim tensor. It is differentiable in `factor`, in the tensors of
        the prior and the likelihood, and in the inducing inputs.

    Raises:
        ValueError: `alpha` lies outside $[0, 1]$, `n_samples` is neither an int nor `None`
            or is below 2, $\alpha \ne 1$ with the closed form for `lk.Categorical` or
            `lk.Bernoulli`, the batch has no targets, or the likelihood finds a bad target
            value.
        CapabilityError: `n_samples=None` with a likelihood other than `lk.Gaussian`,
            `lk.Categorical` and `lk.Bernoulli`, or the model does not support vjp or
            vmap.
        RuntimeError: A fixed parameter of the module changed in place since the fit.
        StructureError: The model's outputs are not one tensor, the targets are not shaped
            like the outputs in the Gaussian closed form, `prior.cov_mv` refuses the
            parameters, a covariance block's Cholesky fails at every jitter, or the
            likelihood refuses the outputs or targets.
    """

    _check_options(alpha, n_samples)
    _check_likelihood(gp.likelihood, alpha, n_samples)

    if batch.targets is None:
        raise ValueError("variational_loss needs a batch with targets")

    lik = gp.likelihood
    y = batch.targets

    if isinstance(lik, HasTargetCheck):
        lik.check_targets(y)

    out, kl = gp._at_and_kl(batch.inputs)
    f = cast(Tensor, out.mean)

    if n_samples is not None:
        per = _mc_term(lik, out, batch, alpha, n_samples, generator)
    elif isinstance(lik, lk.Gaussian):
        per = _gaussian_term(lik, f, cast(Tensor, y), out.cov.diagonal(), alpha)
    else:
        scaled = f / torch.sqrt(1 + math.pi / 8 * out.cov.diagonal())
        per = log_prob(lik, scaled, y, batch.inputs)

    if batch.weights is None:
        data, total = per.sum(), torch.tensor(float(batch.size), dtype=per.dtype)
    else:
        w = batch.weights.to(per)
        data, total = (w * per).sum(), w.sum()

    return -(gp.n_data.to(per) / total.to(per.device)) * data + kl.to(per)


@dataclasses.dataclass(frozen=True)
class _Trained:
    """One trained leaf: the posterior's tensor, the free tensor Adam moves, and the map back."""

    old: Tensor
    free: Tensor
    value: Callable[[Tensor], Tensor]


def _trained(
    gp: VariationalGP, learn_inducing: bool, learn_prior: bool, learn_noise: bool
) -> list[_Trained]:
    """The leaves `fit_variational` trains, each with its unconstrained parameterization."""

    def ident(t: Tensor) -> Tensor:
        return t

    def half_exp(t: Tensor) -> Tensor:
        return (0.5 * t).exp()

    out = [_Trained(gp.factor, gp.factor.clone(), ident)]

    if learn_prior and isinstance(gp.prior, TunablePrior):
        hyper = gp.prior.tunable()
        named = dict(hyper) if isinstance(hyper, Mapping) else {"tunable()": hyper}

        for name, t in named.items():
            if not bool((t > 0).all()):
                raise ValueError(
                    f"fit_variational trains the prior's {name} through its log, but it has "
                    "non-positive entries"
                )

            out.append(_Trained(t, t.log(), torch.exp))

    lik = gp.likelihood

    if (
        learn_noise
        and isinstance(lik, lk.Gaussian)
        and lik.sigma is not None
        and not lik.per_output
    ):
        out.append(_Trained(lik.sigma, 2 * lik.sigma.log(), half_exp))

    z = gp.inducing

    if learn_inducing and isinstance(z, Tensor) and z.is_floating_point():
        out.append(_Trained(z, z.clone(), ident))

    for t in out:
        t.free.requires_grad_()

    return out


def fit_variational[L: Likelihood](
    gp: VariationalGP[L],
    data: lk.data.Data | Iterable[Any],
    /,
    *,
    n_steps: int,
    lr: float = 1e-3,
    alpha: float = 1.0,
    n_samples: int | None = None,
    learn_inducing: bool = True,
    learn_prior: bool = True,
    learn_noise: bool = True,
    generator: torch.Generator | None = None,
) -> VariationalGP[L]:
    r"""Fit VaLLA's variational posterior with `n_steps` Adam steps on `variational_loss`.

    As in laplace-torch, Adam trains the factor (of which only the lower triangle is
    used), the log of each of the prior's `tunable()` leaves, $\log \sigma^2$ for an
    `lk.Gaussian` with a one-element $\sigma$, and the inducing inputs when they are a
    floating-point tensor and `learn_inducing` is true. It leaves alone a prior without
    `tunable`, a per-output $\sigma$ and inducing inputs given as a mapping.
    `learn_prior=False` fixes the prior's leaves and `learn_noise=False` fixes $\sigma$,
    for example at the values from a marginal-likelihood fit. With the mean fixed, the
    ELBO ($\alpha = 0$) has no minimum in the prior's precision $\lambda$ (Ortega et al.
    2024, §3.4). The model's parameters never change: the mean is detached, gradients
    flow to the trained leaves only, and grad is enabled only inside the loop.

    Each step takes the next batch, and the batches cycle through the data, one pass per
    epoch. One seed is drawn from `generator`, or from torch's global RNG when it is
    `None`, and seeds a CPU generator. Each pass and each step's Monte Carlo draws take
    their turn on it. $N$ in the objective is `gp.n_data`, the total weight
    that the `fs.laplace` pass of `VaLLA` counted. `Data.size` is never read, so
    passing data of another length here changes nothing.

    `n_steps`, `lr`, `alpha`, `n_samples`, the likelihood, the model's capabilities, the
    versions and the prior's leaves are all checked before any data is read. Each batch's
    targets are checked at its step.

    Args:
        gp: The fitted posterior. It is left unchanged.
        data: A `Data` source, or a re-iterable of `Batch`es or `(inputs, targets)` pairs,
            which `lk.data.from_loader` wraps.
        n_steps: The number of Adam steps, at least 1.
        lr: Adam's learning rate, positive and finite.
        alpha: $\alpha$ of `variational_loss`.
        n_samples: The number of Monte Carlo draws, as in `variational_loss`. `None` uses
            the closed form.
        learn_inducing: Whether to train the inducing inputs.
        learn_prior: Whether to train the prior's `tunable()` leaves.
        learn_noise: Whether to train $\sigma$.
        generator: Seeds the batches and the draws, as above. `torch.manual_seed` makes
            the fit repeatable when it is `None`.

    Returns:
        A new posterior with the trained leaves detached.

    Raises:
        ValueError: `n_steps` is not an int of at least 1, `lr` is not positive and finite,
            `alpha` or `n_samples` is refused as in `variational_loss`, a trained leaf of
            the prior has non-positive entries, the data has no batches, or a batch fails
            `variational_loss` (no targets, a bad target value).
        CapabilityError: `variational_loss` has no closed form for the likelihood, or the
            model does not support vjp or vmap.
        RuntimeError: A fixed parameter of the module changed in place since the fit.
        StructureError: A batch fails `variational_loss` (see there).
        TypeError: `data` is a one-shot iterator, or an item is neither a `Batch` nor an
            `(inputs, targets)` pair.
    """

    if isinstance(n_steps, bool) or not isinstance(n_steps, int) or n_steps < 1:
        raise ValueError(f"fit_variational needs n_steps >= 1, got {n_steps!r}")

    if not (math.isfinite(lr) and lr > 0):
        raise ValueError(f"fit_variational needs a positive finite lr, got {lr}")

    _check_options(alpha, n_samples)
    _check_likelihood(gp.likelihood, alpha, n_samples)

    gp.caps.check(gp.requires, "fit_variational")
    gp.check_versions()

    gen = lk.core.spawn_generator(generator)
    src = data if isinstance(data, lk.data.Data) else lk.data.from_loader(data)
    gp = detach_leaves(gp)
    leaves = _trained(gp, learn_inducing, learn_prior, learn_noise)
    old = tuple(t.old for t in leaves)
    free = [t.free for t in leaves]
    opt = torch.optim.Adam(free, lr=lr)

    def current() -> VariationalGP[L]:
        return replace_leaves(gp, old, tuple(t.value(t.free) for t in leaves))

    batches: Iterator[Batch] = iter(())

    for _ in range(n_steps):
        batch = next(batches, None)

        if batch is None:
            batches = iter(src(gen))
            batch = next(batches, None)

            if batch is None:
                raise ValueError("fit_variational got data without batches")

        with torch.enable_grad():
            loss = variational_loss(
                current(), batch, alpha=alpha, n_samples=n_samples, generator=gen
            )
            grads = torch.autograd.grad(loss, free, materialize_grads=True)

        for x, g in zip(free, grads, strict=True):
            x.grad = g

        opt.step()

    with torch.no_grad():
        return detach_leaves(current())
