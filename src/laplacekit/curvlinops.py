r"""KFAC through Curvlinops.

Curvlinops builds the Kronecker factors per batch, on a thin wrapper around the
user's module so that the original `Linear` / `Conv2d` hooks fire. The factors
are turned into per-example sums, which add across batches. `finalize` returns
a `KronEigenbasis` per block.

Normalization in Curvlinops 3.0.1 with `reduction="sum"` and `num_data=B` (the batch
size), for $T$ positions per example (spatial, or along a sequence): the input factor is
$\frac{1}{B T} \sum_{n,t} x_{n,t} x_{n,t}^\top$ and the gradient factor is
$\sum_{n,t} d_{n,t} d_{n,t}^\top$. So `a` $= B \cdot$ (input factor) and `g` = (gradient
factor) are per-batch sums, and over $N$ examples Curvlinops returns
$g \otimes (a / N)$ for the full data.
"""

import math
from collections.abc import Iterable, Mapping
from typing import Any, Literal, Self, cast

import torch
from curvlinops import KFACLinearOperator
from torch import Tensor, nn

from laplacekit.core import (
    Batch,
    BlockDiag,
    Capabilities,
    CapabilityError,
    FitContext,
    KronEigenbasis,
    Likelihood,
    Params,
    Requirements,
    check_likelihood,
    pytree,
    spawn_generator,
    stat_dtype,
    static,
    tied_names,
)
from laplacekit.likelihoods import Bernoulli, Categorical, Gaussian

__all__ = ["KFAC", "KronStat"]

type Fisher = Literal["exact", "mc", "empirical"]
r"""Which output Hessian $H_n$ a GGN method uses. `"exact"`: the likelihood's closed form
(`HasOutputHessian`); a likelihood without one raises `CapabilityError`. `"mc"`: the
Monte Carlo Fisher, from targets drawn from the model (`mc_samples` draws). `"empirical"`:
the empirical Fisher, from the batch's own targets."""


_MODULE = "module."


class _Wrapped(nn.Module):
    """Calls the model function with the module's own, live parameters.

    For a Categorical likelihood the class axis (last) moves to dim 1, where
    `CrossEntropyLoss` and Curvlinops expect it for outputs like `[N, T, C]`. For a
    Gaussian the outputs are whitened, f / sigma, so the unit-noise `MSELoss` sees the
    noise of each output (a fixed sigma is applied at fit time).
    """

    def __init__(self, ctx: FitContext, names: tuple[str, ...]) -> None:
        super().__init__()
        module = cast(nn.Module, ctx.model.module)
        self.module = module
        self._model = ctx.model
        self._names = names
        self._classes_to_dim1 = isinstance(ctx.likelihood, Categorical)
        self._whitened = _whitened(ctx.likelihood)

    def forward(self, x: Any) -> Any:
        live = dict(self.module.named_parameters(remove_duplicate=False))
        f = cast(Tensor, self._model({n: live[n] for n in self._names}, x))

        if self._whitened is not None:
            return f / _sigma_like(self._whitened, f)

        return f.movedim(-1, 1) if self._classes_to_dim1 else f


def _whitened(lik: Likelihood) -> Gaussian | None:
    """`lik` when it is a Gaussian with a sigma (fixed, or the unit one of a fit), else None.

    Its outputs and targets are divided by sigma, so the unit-noise `MSELoss` sees the
    noise of each output.
    """

    return lik if isinstance(lik, Gaussian) and lik.sigma is not None else None


def _sigma_like(lik: Gaussian, f: Tensor) -> Tensor:
    """The sigma, checked against f and broadcast to its shape, detached."""

    lik.check_outputs(f)

    return torch.broadcast_to(lik.sigma.detach().to(f.device), f.shape)


class _HookIds:
    """The forward hooks on each submodule, so the ones added later can be removed.

    Curvlinops removes its hooks only when `compute_kronecker_factors` returns; when a
    forward raises, they would stay on the user's module. Its handles are local to that
    method, so the hooks are found through the module's own hook dicts instead (torch
    module internals, not Curvlinops'). A `RemovableHandle` removes an id from these same
    dicts.
    """

    def __init__(self, module: nn.Module) -> None:
        self._before = {
            m: (set(m._forward_pre_hooks), set(m._forward_hooks)) for m in module.modules()
        }

    def remove_new(self) -> None:
        for m, (pre, post) in self._before.items():
            for i in set(m._forward_pre_hooks) - pre:
                del m._forward_pre_hooks[i]
                m._forward_pre_hooks_with_kwargs.pop(i, None)

            for i in set(m._forward_hooks) - post:
                del m._forward_hooks[i]
                m._forward_hooks_with_kwargs.pop(i, None)
                m._forward_hooks_always_called.pop(i, None)


@pytree
class KronStat:
    r"""Per-block sums of the input factor `a` and gradient factor `g`, and the example count.

    After $N$ examples, `a` holds $N A$ and `g` holds $N G$ per block, for the factors
    $A$ and $G$ of `KFAC`.

    Args:
        a: The input factor's sum per block, `[in (+1), in (+1)]`.
        g: The gradient factor's sum per block, `[out, out]`.
        weight: The example count $N$, 0-dim.
        shapes: Per block, `(name, has_bias, weight_shape)`, for `finalize`.
    """

    a: Mapping[str, Tensor]
    g: Mapping[str, Tensor]
    weight: Tensor
    shapes: tuple[tuple[str, bool, tuple[int, ...]], ...] = static()

    @property
    def total_weight(self) -> Tensor:
        """The example count."""

        return self.weight

    def __add__(self, other: Self, /) -> Self:
        return type(self)(
            {k: v + other.a[k] for k, v in self.a.items()},
            {k: v + other.g[k] for k, v in self.g.items()},
            self.weight + other.weight,
            self.shapes,
        )

    def scaled(self, s: Tensor, /) -> Self:
        """Both sums and the count, times `s`."""

        return type(self)(
            {k: s * v for k, v in self.a.items()},
            {k: s * v for k, v in self.g.items()},
            s * self.weight,
            self.shapes,
        )


def _unsupported(lik: Likelihood) -> str | None:
    """Why KFAC can't fit `lik`, or None when it can."""

    if isinstance(lik, Categorical):
        if lik.mask_fn is not None:
            # Masking the output only zeroes the gradient factor's rows; its normalisation and the
            # whole input factor still count every position, and Curvlinops 3.0.1 can't drop any.
            return (
                "KFAC can't fit a Categorical with mask_fn: Curvlinops 3.0.1 sums the input "
                "factor over every position, padding included, and has no option to leave "
                "positions out; use DenseGGN or DiagGGN, or a Categorical without mask_fn"
            )

        return None if lik.class_dim == -1 else "KFAC needs Categorical with class_dim=-1"

    # BCE is elementwise, so the factors don't depend on how many axes form the event.
    if isinstance(lik, Gaussian | Bernoulli):
        return None

    return (
        f"KFAC supports Categorical, Gaussian and Bernoulli likelihoods, got "
        f"{type(lik).__name__}; use DiagGGN or DenseGGN"
    )


def _same(live: Tensor, passed: Tensor) -> bool:
    """Whether `passed` holds the live parameter's values.

    A view of the same memory (the usual case: `p.detach()`) is decided without reading
    the values.
    """

    view = (
        live.data_ptr() == passed.data_ptr()
        and live.shape == passed.shape
        and live.stride() == passed.stride()
        and live.dtype == passed.dtype
    )

    return view or torch.equal(live, passed)


def _aliased(names: Iterable[str], tied: Mapping[str, tuple[str, ...]]) -> str | None:
    """Why KFAC can't cover `names` because one is a tied tensor, or None."""

    for n in names:
        if n in tied:
            return (
                f"KFAC sees only the hooked layer's use of a tensor, but {n!r} is tied: it is "
                f"also {', '.join(repr(o) for o in tied[n] if o != n)}; use a method that "
                "differentiates every use (DenseGGN, DiagGGN, DiagGGN(fisher='mc'))"
            )

    return None


_FISHER_TYPE = {"exact": "type-2", "mc": "mc", "empirical": "empirical"}


def _eigh(m: Tensor) -> tuple[Tensor, Tensor]:
    evals, evecs = torch.linalg.eigh(m)

    return evals.clamp(min=0), evecs


def _shapes(ctx: FitContext) -> tuple[tuple[str, bool, tuple[int, ...]], ...]:
    """Per block: its name, whether it has a bias, and its weight's shape."""

    shapes: list[tuple[str, bool, tuple[int, ...]]] = []

    for b, names in ctx.layout.blocks.items():
        if f"{b}.weight" not in names:
            raise CapabilityError(f"KFAC needs the weight of block {b!r} in the subset")

        shapes.append((b, f"{b}.bias" in names, tuple(ctx.layout.shapes[f"{b}.weight"])))

    return tuple(shapes)


def _first(params: Params) -> Tensor:
    if not params:
        raise CapabilityError("KFAC needs at least one parameter in the subset")

    return next(iter(params.values()))


def _d_in(bias: bool, w_shape: tuple[int, ...]) -> int:
    return math.prod(w_shape[1:]) + int(bias)


@pytree
class KFAC:
    r"""Kronecker-factored GGN per Linear / Conv2d layer, computed by Curvlinops.

    Per layer, the per-example average curvature is approximated by $G \otimes A$, on the
    weight and bias folded into one matrix (row-major, as in `KronEigenbasis`):

    $$A = \frac{1}{N} \sum_n \frac{1}{T} \sum_t x_{n,t} x_{n,t}^\top, \qquad
    G = \frac{1}{N} \sum_n \sum_t \sum_c d_{n,t,c} \, d_{n,t,c}^\top.$$

    Here $x_{n,t}$ is the layer's input at position $t$ of example $n$ (with a 1 appended
    when the layer has a bias), over $T$ positions (spatial, or along a sequence), and the
    $d_{n,t,c}$ are vectors backpropagated to the layer's output at that position. With
    `"exact"` they come from the columns $c$ of a square root of the output Hessian. With
    `"mc"` and `"empirical"` there is one per position, from the score
    $\nabla_f \log p(y \mid f)$ at a target $y$ drawn from the model or at the batch's own
    target.

    The empirical Fisher needs targets and can't apply a late $\sigma$, so it takes
    a fixed $\sigma$ such as `Gaussian(sigma=0.1)`, but not `Gaussian()`, which learns it.
    On sequence outputs `[N, T, C]` it sums the per-position $d d^\top$ (KFAC-expand), so
    it does not factor the same matrix as `DenseGGN(fisher="empirical")`, whose gradient
    is per sequence.

    Per-example weights are not supported. Masked likelihoods
    (`Categorical(mask_fn=...)`) are refused: Curvlinops builds the input factor from
    every position's layer input and can't leave padding out. Tied tensors are refused,
    since the hooks see only the layer's own use.

    Args:
        fisher: `"exact"` uses the exact output Hessian, `"mc"` sampled targets,
            seeded from the batch's generator, and `"empirical"` the batch's own targets.
        mc_samples: Sampled targets per example for `fisher="mc"`, at least 1 (1 when
            `None`). Refused with another `fisher`.

    Raises:
        ValueError: `fisher` is not `"exact"`, `"mc"` or `"empirical"`, or `mc_samples`
            comes without `fisher="mc"` or is below 1.
    """

    fisher: Fisher = static(default="exact")
    mc_samples: int | None = static(default=None)

    def __post_init__(self) -> None:
        if self.fisher not in ("exact", "mc", "empirical"):
            raise ValueError(f'KFAC takes fisher="exact", "mc" or "empirical", got {self.fisher!r}')

        if self.mc_samples is not None and self.fisher != "mc":
            raise ValueError(
                f'KFAC: mc_samples only applies to fisher="mc", got fisher={self.fisher!r}'
            )

        if self.mc_samples is not None and self.mc_samples < 1:
            raise ValueError(f"KFAC needs mc_samples >= 1, got {self.mc_samples}")

    @staticmethod
    def supports(
        likelihood: Likelihood | None,
        /,
        caps: Capabilities | None = None,
        names: Iterable[str] = (),
    ) -> bool:
        """Whether KFAC can fit `likelihood` on the parameters `names`.

        `init` raises for everything refused here.

        Args:
            likelihood: The likelihood, or None to skip its check. KFAC takes `Gaussian`,
                `Bernoulli`, and `Categorical` with `class_dim=-1` and no `mask_fn`.
            caps: The probe's capabilities, read for the tied tensors `caps.tied`. None
                skips the tie check.
            names: Parameter names. A name of a tied tensor is refused: the hooks see only
                one of its uses.

        Returns:
            True unless the likelihood or a tied tensor rules KFAC out.
        """

        if likelihood is not None and _unsupported(likelihood) is not None:
            return False

        return _aliased(names, caps.tied if caps is not None else {}) is None

    @property
    def requires(self) -> Requirements:
        """Needs hooks on `nn.Linear` and `nn.Conv2d`, and no per-example weights.

        With `"empirical"`, also targets and a likelihood without late hyperparameters.
        """

        if self.fisher == "empirical":
            return Requirements(
                hooks=(nn.Linear, nn.Conv2d), weights=False, needs_targets=True, late_scale=False
            )

        return Requirements(hooks=(nn.Linear, nn.Conv2d), weights=False)

    def init(self, params: Params, ctx: FitContext, /) -> KronStat:
        """Zero factor sums per block, `[in (+1), in (+1)]` and `[out, out]`.

        Args:
            params: The params the curvature is taken at.
            ctx: The fit's context.

        Returns:
            A statistic $e$ with $e + s = s$ for every statistic $s$ of this fit.

        Raises:
            CapabilityError: The likelihood is not one KFAC takes (see `supports`), the
                model was not built from an `nn.Module`, the capabilities miss `requires`
                (`Capabilities.check`, which also refuses blocks not reached by hooks or of
                another type, and parameters that were `selected`, since hooks see whole
                tensors), the likelihood has a late hyperparameter with
                `"empirical"`, a parameter is a tied tensor, there are no parameters, or a
                block has no weight in the subset.
        """

        unsupported = _unsupported(ctx.likelihood)

        if unsupported is not None:
            raise CapabilityError(unsupported)

        if ctx.model.module is None:
            raise CapabilityError("KFAC needs a model built from an nn.Module")

        blocks = ctx.layout.blocks
        ctx.caps.check(self.requires, "KFAC", blocks=blocks)
        check_likelihood(self.requires, ctx.likelihood, "KFAC")

        aliased = _aliased((n for ns in blocks.values() for n in ns), tied_names(ctx.model.module))

        if aliased is not None:
            raise CapabilityError(aliased)

        p = _first(params)
        dtype = stat_dtype(p.dtype)
        a: dict[str, Tensor] = {}
        g: dict[str, Tensor] = {}
        shapes = _shapes(ctx)

        for b, bias, w_shape in shapes:
            d_in = _d_in(bias, w_shape)
            a[b] = torch.zeros(d_in, d_in, dtype=dtype, device=p.device)
            g[b] = torch.zeros(w_shape[0], w_shape[0], dtype=dtype, device=p.device)

        return KronStat(a, g, torch.zeros((), dtype=dtype, device=p.device), shapes)

    def update(
        self,
        stat: KronStat,
        params: Params,
        batch: Batch,
        generator: torch.Generator,
        ctx: FitContext,
        /,
    ) -> KronStat:
        r"""Add the batch's factor sums, computed by Curvlinops at the module's parameters.

        Curvlinops differentiates the module's live parameters, so `params` must hold
        their values.

        Args:
            stat: The statistic so far.
            params: The params the curvature is taken at.
            batch: The batch to add.
            generator: The batch's generator. With `fisher="mc"`, Curvlinops draws the
                targets from a seed drawn from it (`lk.core.spawn_generator`).
            ctx: The fit's context.

        Returns:
            `stat` with the batch added. Updating with batches $a$, then $b$, equals adding
            the statistics of $a$ and of $b$.

        Raises:
            CapabilityError: The batch has per-example weights, it has no targets with
                `"empirical"`, or `params` differ from the module's live parameters.
            ValueError: A `Bernoulli` target lies outside $[0, 1]$ with `"empirical"`.
            StructureError: A per-output $\sigma$ doesn't fit the outputs' event shape.
        """

        if batch.weights is not None:
            raise CapabilityError("KFAC does not support per-example weights")

        empirical = self.fisher == "empirical"

        if empirical and batch.targets is None:
            raise CapabilityError("KFAC(fisher='empirical') needs targets, but the batch has none")

        lik = ctx.likelihood
        names = tuple(n for ns in ctx.layout.blocks.values() for n in ns)
        wrapped = _Wrapped(ctx, names)
        live = dict(wrapped.module.named_parameters(remove_duplicate=False))

        stale = [n for n in names if not _same(live[n], params[n])]

        if stale:
            raise CapabilityError(
                "KFAC evaluates at the module's live parameters (Curvlinops differentiates "
                f"them), but the passed params differ for: {', '.join(stale)}"
            )

        # A mapping becomes a dict, the mutable mapping Curvlinops expects. Curvlinops types
        # its inputs as a tensor or a mutable mapping, but passes any other input (a pair of
        # inputs, say) to the model as it is, hence the cast.
        x = batch.inputs
        data_x = dict(cast(Mapping[str, Any], x)) if isinstance(x, Mapping) else cast(Any, x)
        size = batch.size

        if isinstance(lik, Categorical):
            loss: nn.Module = nn.CrossEntropyLoss(reduction="sum")
            out_scale = 1.0
        elif isinstance(lik, Bernoulli):
            # Its output Hessian, and the square of its score sigmoid(f) - y, are ours.
            loss = nn.BCEWithLogitsLoss(reduction="sum")
            out_scale = 1.0
        elif empirical:
            loss = nn.MSELoss(reduction="sum")
            # The loss gradient is 2 (f - y) on the whitened f and y, so the score is
            # (y - f) / sigma^2. A late sigma can't reach here: `init` refuses it
            # (`requires.late_scale` is False).
            out_scale = 0.25
        else:
            loss = nn.MSELoss(reduction="sum")
            # Curvlinops' output Hessian is 2 I on the whitened outputs; ours is 1 / sigma^2,
            # which the whitening applies (sigma is 1 for the unit Gaussian of a late fit).
            out_scale = 0.5

        if empirical:
            # The real targets: class indices as long for a Categorical (whose class axis
            # `_Wrapped` moves to dim 1), floats for a Bernoulli (BCE takes soft labels),
            # the targets as given for a Gaussian, whitened like f for a per-output sigma.
            y = cast(Tensor, batch.targets)
            whitened = _whitened(lik)

            if isinstance(lik, Categorical):
                y = y.long()
            elif isinstance(lik, Bernoulli):
                lik.check_targets(y)  # `update` may be called outside the fit's check
                y = y.to(live[names[0]].dtype)
            elif whitened is not None:
                y = y / _sigma_like(whitened, y)
        else:
            # The exact and MC Fisher never read the targets.
            with torch.no_grad():
                f = cast(Tensor, ctx.model(params, x))

            y = (
                torch.zeros(f.shape[:-1], dtype=torch.long, device=f.device)
                if isinstance(lik, Categorical)
                else torch.zeros_like(f)
            )

        saved = {n: live[n].requires_grad for n in names}
        hooks = _HookIds(wrapped)
        seed = spawn_generator(generator).initial_seed()

        try:
            with torch.enable_grad():
                for n in names:
                    live[n].requires_grad_(True)

                op = KFACLinearOperator(
                    wrapped,
                    loss,
                    [live[n] for n in names],
                    [(data_x, y)],
                    check_deterministic=False,
                    seed=seed,
                    fisher_type=_FISHER_TYPE[self.fisher],
                    mc_samples=self.mc_samples or 1,
                    separate_weight_and_bias=False,
                    num_data=size,
                    batch_size_fn=lambda _: size,
                )
                op.compute_kronecker_factors()
        finally:
            hooks.remove_new()

            for n, r in saved.items():
                live[n].requires_grad_(r)

        a_cov = op._input_covariances
        g_cov = op._gradient_covariances
        dtype = stat.weight.dtype
        a = {b: stat.a[b] + size * a_cov[_MODULE + b].detach().to(dtype) for b in stat.a}
        g = {b: stat.g[b] + out_scale * g_cov[_MODULE + b].detach().to(dtype) for b in stat.g}
        weight = stat.weight + torch.tensor(
            float(size), dtype=stat.weight.dtype, device=stat.weight.device
        )

        return KronStat(a, g, weight, stat.shapes)

    def finalize(self, stat: KronStat, /) -> BlockDiag[KronEigenbasis]:
        r"""$G \otimes A$ per block, as a `KronEigenbasis`.

        It holds the eigenvectors of $A$ and $G$ and the eigenvalues
        $\lambda_G \lambda_A^\top$, with each factor's negative eigenvalues (round-off)
        clamped to zero. The Kronecker factors are kept in `factors`, with count 1.

        Args:
            stat: The statistic over all the data, after `all_reduce`.

        Returns:
            The per-example average $\bar{C} = \sum_n w_n C_n / \sum_n w_n$. The posterior
            multiplies it by `scale`.
        """

        if bool(stat.weight == 0):
            raise ValueError("cannot finalize: total weight is zero")

        blocks: dict[str, KronEigenbasis] = {}

        for b, has_bias, w_shape in stat.shapes:
            la, qa = _eigh(stat.a[b] / stat.weight)
            lg, qg = _eigh(stat.g[b] / stat.weight)
            count = torch.ones((), dtype=la.dtype, device=la.device)
            blocks[b] = KronEigenbasis(
                qa, qg, torch.outer(lg, la), has_bias, w_shape, (la, lg, count)
            )

        return BlockDiag(blocks)

    def template(self, params: Params, ctx: FitContext, /) -> BlockDiag[KronEigenbasis]:
        """Identity eigenbases with zero eigenvalues and zero Kronecker factors.

        Args:
            params: The params the curvature is taken at.
            ctx: The fit's context.

        Returns:
            A zero operator with the operator types, blocks, static fields, leaf shapes
            and dtypes that `finalize` returns.

        Raises:
            CapabilityError: There are no parameters, or a block has no weight in the
                subset.
        """

        p = _first(params)
        dtype = stat_dtype(p.dtype)
        blocks: dict[str, KronEigenbasis] = {}

        for b, bias, w_shape in _shapes(ctx):
            d_in, out = _d_in(bias, w_shape), w_shape[0]
            zeros = torch.zeros(out, d_in, dtype=dtype, device=p.device)
            blocks[b] = KronEigenbasis(
                torch.eye(d_in, dtype=dtype, device=p.device),
                torch.eye(out, dtype=dtype, device=p.device),
                zeros,
                bias,
                w_shape,
                (zeros[0], zeros[:, 0], zeros.new_zeros(())),
            )

        return BlockDiag(blocks)
