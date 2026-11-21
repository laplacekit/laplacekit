r"""Native curvature methods.

They are the exact diagonal GGN, the dense GGN, the sampled and empirical Fisher
diagonals, the dense empirical Fisher, the exact Hessian, and a low-rank GGN.

Each returns a `BlockDiag` with one block per module (see `Layout.by_module`). With
`joint=True`, the dense ones return a `Joint` over all blocks instead; `LowRankGGN` always
does. `default` picks one of them, or KFAC, per block.

Notation: example $n$ has outputs $f_n = f(x_n; \theta)$, the Jacobian
$J_n = \partial f_n / \partial \theta$, the output Hessian $H_n$ and the weight $w_n$
(`Batch.weights`, 1 without them). Each method sums a curvature $C_n$ per example, and
`finalize` returns the weighted average

$$\bar{C} = \frac{1}{W} \sum_n w_n C_n, \qquad W = \sum_n w_n.$$

Weights must be non-negative; `update` raises `ValueError` otherwise.
"""

import copy
import math
from collections.abc import Callable, Mapping, Sequence
from functools import lru_cache, partial
from typing import Any, Literal, Self, cast

import torch
from torch import Tensor
from torch.func import grad, jacrev, jvp, vjp, vmap

from laplacekit import core
from laplacekit.core import (
    Batch,
    BlockDiag,
    Capabilities,
    CapabilityError,
    Curvature,
    FitContext,
    HasOutputHessian,
    Inputs,
    Joint,
    Likelihood,
    OutputHessian,
    Params,
    PerBlockCurvature,
    Requirements,
    SumStat,
    Targets,
    can_sample,
    log_prob,
    observed_output_hessian,
    output_hessian,
    pytree,
    sample,
    static,
    tree_map,
)
from laplacekit.curvlinops import KFAC, Fisher

__all__ = [
    "DENSE_MAX_PARAMS",
    "DenseGGN",
    "DenseHessian",
    "DiagGGN",
    "Fisher",
    "KFAC",
    "LowRankGGN",
    "default",
]

DENSE_MAX_PARAMS = 4096

type _OutputHessianKind = Literal["exact", "mc", "observed"]
r"""`"exact"`: the likelihood's closed form. `"mc"`: the MC Fisher with `mc_samples` draws
from the batch's generator. `"observed"`: the Hessian of $-\log p(y \mid f)$ at the batch's targets,
which can be indefinite (a heteroscedastic Gaussian), so the curvature can be too."""


def _weights(batch: Batch, like: Tensor) -> Tensor:
    """Per-example weights in `like`'s dtype (the statistic's)."""

    if batch.weights is None:
        return torch.ones(batch.size, dtype=like.dtype, device=like.device)

    if bool((batch.weights < 0).any()):
        raise ValueError("Batch.weights must be non-negative")

    return batch.weights.to(like.dtype)


def _forward(ctx: FitContext, params: Params, batch: Batch) -> Tensor:
    return cast(Tensor, ctx.model(params, batch.inputs))


def _lead(x: Any) -> Any:
    """Add a leading batch axis of size 1 to every leaf."""

    def add(t: Tensor) -> Tensor:
        return t.unsqueeze(0)

    return tree_map(add, x)


def _requires(kind: _OutputHessianKind) -> Requirements:
    return Requirements(
        vjp=True,
        vmap=True,
        needs_targets=kind == "observed",
        needs_closed_form_hessian=kind == "exact",
        needs_sample=kind == "mc",
    )


def _check_options(method: object, mc_samples: int) -> None:
    if mc_samples < 1:
        raise ValueError(f"{type(method).__name__} needs mc_samples >= 1, got {mc_samples}")


def _check_mc(method: object, fisher: str, mc_samples: int | None) -> None:
    name = type(method).__name__

    if mc_samples is not None and fisher != "mc":
        raise ValueError(f'{name}: mc_samples only applies to fisher="mc", got fisher={fisher!r}')

    if mc_samples is not None and mc_samples < 1:
        raise ValueError(f"{name} needs mc_samples >= 1, got {mc_samples}")


def _check_fisher(method: object, fisher: str, observed: bool) -> None:
    name = type(method).__name__

    if fisher not in ("exact", "mc", "empirical"):
        raise ValueError(f'{name} takes fisher="exact", "mc" or "empirical", got {fisher!r}')

    if observed and fisher != "exact":
        raise ValueError(f'{name}: observed=True replaces the Fisher, so it needs fisher="exact"')


def _check_event(kind: _OutputHessianKind, ctx: FitContext, what: str) -> None:
    """Before data, refuse an event size above `MAX_DENSE_EVENT` for a dense output Hessian.

    The MC Fisher and the observed Hessian form a dense [E, E] block per example.
    """

    event = ctx.caps.event_size
    dense = kind != "exact"

    if event is None or not dense or event <= core.MAX_DENSE_EVENT:
        return

    source = "observed=True" if kind == "observed" else 'fisher="mc"'
    raise core.StructureError(
        f"{what} with {source} forms a dense {event}x{event} matrix per example; give "
        f"{type(ctx.likelihood).__name__} an output_hessian method (HasOutputHessian)"
    )


def _output_hessian(
    kind: _OutputHessianKind,
    mc_samples: int,
    ctx: FitContext,
    f: Tensor,
    batch: Batch,
    generator: torch.Generator,
) -> OutputHessian:
    if kind != "observed":
        return output_hessian(
            ctx.likelihood,
            f,
            generator=generator,
            inputs=batch.inputs,
            mc_samples=mc_samples,
            fisher=kind,
        )

    if batch.targets is None:
        raise CapabilityError("observed=True needs targets, but the batch has none")

    return observed_output_hessian(ctx.likelihood, f, batch.targets, inputs=batch.inputs)


def _vec_shape(ctx: FitContext, b: str) -> tuple[int, ...]:
    return (ctx.layout.size(b),)


def _mat_shape(ctx: FitContext, b: str) -> tuple[int, ...]:
    return (ctx.layout.size(b),) * 2


def _zero_stat(
    params: Params, ctx: FitContext, shape: Callable[[FitContext, str], tuple[int, ...]]
) -> SumStat:
    p = next(iter(params.values()))
    dtype = core.stat_dtype(p.dtype)

    return SumStat(
        {b: torch.zeros(shape(ctx, b), dtype=dtype, device=p.device) for b in ctx.layout.blocks},
        torch.zeros((), dtype=dtype, device=p.device),
    )


def _check_total(weight: Tensor) -> None:
    if bool(weight == 0):
        raise ValueError("cannot finalize: total weight is zero")


def _finalize_diag(stat: SumStat) -> BlockDiag[core.Diag]:
    _check_total(stat.weight)
    return BlockDiag({b: core.Diag(v / stat.weight) for b, v in stat.sum.items()})


def _diag_template(params: Params, ctx: FitContext) -> BlockDiag[core.Diag]:
    zero = _zero_stat(params, ctx, _vec_shape)

    return BlockDiag({b: core.Diag(v) for b, v in zero.sum.items()})


def _flat_blocks(ctx: FitContext, per_param: Mapping[str, Tensor], n: int) -> dict[str, Tensor]:
    """Per-example, per-param tensors [N, ...] to per-block [N, P_b] in layout order."""

    return {
        b: torch.cat([per_param[name].reshape(n, -1) for name in names], dim=1)
        for b, names in ctx.layout.blocks.items()
    }


_JOINT = "<joint>"
"""The key of a joint statistic's one sum: [P, P], or [P, k] for `LowRankGGN`."""


@pytree
class _JointStat:
    """The weighted sum of a joint curvature, and the total weight.

    The sum is `{_JOINT: [P, P]}` (or `LowRankGGN`'s [P, k] sketch) with P over all blocks.
    `sizes` holds each block's name and flat size, in layout order.
    """

    sum: Mapping[str, Tensor]
    weight: Tensor
    sizes: tuple[tuple[str, int], ...] = static()
    seed: int = static(default=0)  # `LowRankGGN`'s test matrix

    @property
    def total_weight(self) -> Tensor:
        return self.weight

    def __add__(self, other: Self, /) -> Self:
        if other.sizes != self.sizes or other.seed != self.seed:
            raise core.StructureError(
                f"joint statistics have different blocks or sketches: {list(self.sizes)} vs "
                f"{list(other.sizes)}"
            )

        return copy.replace(
            self,
            sum={k: v + other.sum[k] for k, v in self.sum.items()},
            weight=self.weight + other.weight,
        )

    def scaled(self, s: Tensor, /) -> Self:
        return copy.replace(
            self, sum={k: s * v for k, v in self.sum.items()}, weight=s * self.weight
        )


type _DenseStat = SumStat | _JointStat
"""A dense method's statistic: per block, or one joint sum."""


def _init_dense(params: Params, ctx: FitContext, joint: bool) -> _DenseStat:
    if not joint:
        return _zero_stat(params, ctx, _mat_shape)

    sizes = tuple((b, ctx.layout.size(b)) for b in ctx.layout.blocks)
    total = sum(n for _, n in sizes)
    zero = _zero_stat(params, ctx, _vec_shape).weight

    return _JointStat({_JOINT: zero.new_zeros(total, total)}, zero, sizes)


def _joined(per_block: Mapping[str, Tensor], stat: _DenseStat) -> Mapping[str, Tensor]:
    """Per-block tensors [..., P_b] keyed like `stat`.

    As they are, or concatenated in layout order under `_JOINT`.
    """

    if isinstance(stat, _JointStat):
        return {_JOINT: torch.cat([per_block[b] for b, _ in stat.sizes], dim=-1)}

    return per_block


def _finalize_dense(stat: _DenseStat) -> BlockDiag[core.Dense] | Joint[core.Dense]:
    _check_total(stat.weight)

    if isinstance(stat, _JointStat):
        return Joint(core.Dense(stat.sum[_JOINT] / stat.weight), stat.sizes)

    return BlockDiag({b: core.Dense(v / stat.weight) for b, v in stat.sum.items()})


def _dense_template(
    params: Params, ctx: FitContext, joint: bool
) -> BlockDiag[core.Dense] | Joint[core.Dense]:
    zero = _init_dense(params, ctx, joint)

    if isinstance(zero, _JointStat):
        return Joint(core.Dense(zero.sum[_JOINT]), zero.sizes)

    return BlockDiag({b: core.Dense(v) for b, v in zero.sum.items()})


@pytree
class _DiagJacobian:
    r"""Exact diagonal GGN, $C_n = \operatorname{diag}(J_n^\top H_n J_n)$.

    Per example, $E$ VJPs (one per entry $c$ of $f_n$) with cotangents $A_n e_c$, where
    $A_n A_n^\top = H_n$, squared and summed:

    $$\operatorname{diag}(J_n^\top H_n J_n) = \sum_{c=1}^{E} (J_n^\top A_n e_c)^{\odot 2}.$$

    With `"observed"`, $H_n$ may be indefinite, so each entry pairs the VJPs of $e_c$ and
    $H_n e_c$ instead: $\sum_c (J_n^\top e_c) \odot (J_n^\top H_n e_c)$.

    Args:
        output_hessian: Picks $H_n$ (see `_OutputHessianKind`).
        mc_samples: Draws for the MC Fisher. Only the MC Fisher uses it.

    Raises:
        ValueError: `mc_samples` is below 1.
    """

    output_hessian: _OutputHessianKind = static(default="exact")
    mc_samples: int = static(default=1)

    def __post_init__(self) -> None:
        _check_options(self, self.mc_samples)

    @property
    def requires(self) -> Requirements:
        """Needs vjp and vmap, and an output Hessian, or targets with `"observed"`."""

        return _requires(self.output_hessian)

    def init(self, params: Params, ctx: FitContext, /) -> SumStat:
        """Zero diagonals per block.

        Args:
            params: The params the curvature is taken at.
            ctx: The fit's context.

        Returns:
            A statistic $e$ with $e + s = s$ for every statistic $s$ of this fit.

        Raises:
            StructureError: $H_n$ is dense (the MC Fisher, or `"observed"`) and the probe's
                event size exceeds `MAX_DENSE_EVENT`.
        """

        _check_event(self.output_hessian, ctx, "DiagGGN")
        return _zero_stat(params, ctx, _vec_shape)

    def update(
        self,
        stat: SumStat,
        params: Params,
        batch: Batch,
        generator: torch.Generator,
        ctx: FitContext,
        /,
    ) -> SumStat:
        r"""Add $\sum_n w_n \operatorname{diag}(J_n^\top H_n J_n)$ over the batch.

        Args:
            stat: The statistic so far.
            params: The params the curvature is taken at.
            batch: The batch to add.
            generator: The batch's generator, for sampled methods.
            ctx: The fit's context.

        Returns:
            `stat` with the batch added. Updating with batches $a$, then $b$, equals adding
            the statistics of $a$ and of $b$.

        Raises:
            ValueError: `batch.weights` has a negative entry.
            CapabilityError: `"observed"` and the batch has no targets.
            StructureError: The likelihood refuses the outputs or targets, or its mask is
                not bool or not shaped like the leading axes of $f$, or $H_n$ is dense (the
                MC Fisher, or `"observed"`) and the event size exceeds `MAX_DENSE_EVENT`
                (checked here when the probe did not record it).
        """

        p = dict(params)
        n = batch.size
        f = _forward(ctx, p, batch)
        w = _weights(batch, stat.weight)
        hess = _output_hessian(self.output_hessian, self.mc_samples, ctx, f, batch, generator)
        event = f[0].numel()
        observed = self.output_hessian == "observed"

        def one(x: Any, ct: Tensor) -> Mapping[str, Tensor]:
            xb = _lead(x)

            def fwd(q: Params) -> Any:
                return ctx.model(q, xb)

            pull = cast(Any, vjp(fwd, p))[1]

            return pull(ct.unsqueeze(0))[0]

        sums = {b: s for b, s in stat.sum.items()}

        for c in range(event):
            e = torch.zeros(n, event, dtype=f.dtype, device=f.device)
            e[:, c] = 1
            e = e.reshape(f.shape)

            # diag(Jᵀ H J) = sum_c (Jᵀ A e_c)^2 for H = A Aᵀ, or sum_c (Jᵀ e_c) (Jᵀ H e_c).
            if observed:
                g = _flat_blocks(ctx, vmap(one)(batch.inputs, e), n)
                gh = _flat_blocks(ctx, vmap(one)(batch.inputs, hess.mv(e)), n)
            else:
                g = gh = _flat_blocks(ctx, vmap(one)(batch.inputs, hess.sqrt_mv(e)), n)

            for b in sums:
                prod = g[b].to(w.dtype) * gh[b].to(w.dtype)
                sums[b] = sums[b] + (w.unsqueeze(1) * prod).sum(0)

        return SumStat(sums, stat.weight + w.sum())

    def finalize(self, stat: SumStat, /) -> BlockDiag[core.Diag]:
        r"""$\bar{C}$ per block, as `Diag` operators.

        Args:
            stat: The statistic over all the data, after `all_reduce`.

        Returns:
            The per-example average $\bar{C} = \sum_n w_n C_n / \sum_n w_n$. The posterior
            multiplies it by `scale`.
        """

        return _finalize_diag(stat)

    def template(self, params: Params, ctx: FitContext, /) -> BlockDiag[core.Diag]:
        """Zero `Diag` operators per block.

        Args:
            params: The params the curvature is taken at.
            ctx: The fit's context.

        Returns:
            A zero operator with the operator types, blocks, static fields, leaf shapes
            and dtypes that `finalize` returns.
        """

        return _diag_template(params, ctx)


@pytree
class _DenseJacobian:
    r"""Dense GGN, $C_n = J_n^\top H_n J_n$, with $J_n$ from `jacrev`.

    So $\bar{C} = \frac{1}{W} \sum_n w_n J_n^\top H_n J_n$. Per block by default:
    cross-block entries are not stored.

    Args:
        output_hessian: Picks $H_n$ (see `_OutputHessianKind`).
        mc_samples: Draws for the MC Fisher. Only the MC Fisher uses it.
        joint: Keep one matrix over all blocks, in layout order, as a `Joint`
            (laplace-torch's `hessian_structure="full"` over several layers).

    Raises:
        ValueError: `mc_samples` is below 1.
    """

    output_hessian: _OutputHessianKind = static(default="exact")
    mc_samples: int = static(default=1)
    joint: bool = static(default=False)

    def __post_init__(self) -> None:
        _check_options(self, self.mc_samples)

    @property
    def requires(self) -> Requirements:
        """Needs vjp and vmap, and an output Hessian, or targets with `"observed"`."""

        return _requires(self.output_hessian)

    def init(self, params: Params, ctx: FitContext, /) -> _DenseStat:
        """Zero matrices per block, or one over all blocks with `joint=True`.

        Args:
            params: The params the curvature is taken at.
            ctx: The fit's context.

        Returns:
            A statistic $e$ with $e + s = s$ for every statistic $s$ of this fit.

        Raises:
            StructureError: $H_n$ is dense (the MC Fisher, or `"observed"`) and the probe's
                event size exceeds `MAX_DENSE_EVENT`.
        """

        _check_event(self.output_hessian, ctx, "DenseGGN")
        return _init_dense(params, ctx, self.joint)

    def update(
        self,
        stat: _DenseStat,
        params: Params,
        batch: Batch,
        generator: torch.Generator,
        ctx: FitContext,
        /,
    ) -> _DenseStat:
        r"""Add $\sum_n w_n J_n^\top H_n J_n$ over the batch.

        It forms $H_n$ densely, also from a closed form.

        Args:
            stat: The statistic so far.
            params: The params the curvature is taken at.
            batch: The batch to add.
            generator: The batch's generator, for sampled methods.
            ctx: The fit's context.

        Returns:
            `stat` with the batch added. Updating with batches $a$, then $b$, equals adding
            the statistics of $a$ and of $b$.

        Raises:
            ValueError: `batch.weights` has a negative entry.
            CapabilityError: `"observed"` and the batch has no targets.
            StructureError: The event size exceeds `MAX_DENSE_EVENT` (a closed-form $H_n$
                made dense), the likelihood refuses the outputs or targets, or its mask is
                not bool or not shaped like the leading axes of $f$.
        """

        p = dict(params)
        n = batch.size
        f = _forward(ctx, p, batch)
        w = _weights(batch, stat.weight)
        hess = _output_hessian(self.output_hessian, self.mc_samples, ctx, f, batch, generator)
        h = hess.dense().reshape(n, f[0].numel(), -1).to(w.dtype)

        def one(x: Any) -> Mapping[str, Tensor]:
            xb = _lead(x)

            def fwd(q: Params) -> Tensor:
                return cast(Tensor, ctx.model(q, xb)).reshape(-1)

            return cast(Mapping[str, Tensor], jacrev(fwd)(p))

        jac = vmap(one)(batch.inputs)  # name -> [N, E, *shape]
        event = f[0].numel()
        flat = {
            b: torch.cat([jac[name].reshape(n, event, -1) for name in names], dim=2).to(w.dtype)
            for b, names in ctx.layout.blocks.items()
        }
        sums = {
            k: stat.sum[k] + torch.einsum("n,nep,nef,nfq->pq", w, j, h, j)
            for k, j in _joined(flat, stat).items()
        }

        return copy.replace(stat, sum=sums, weight=stat.weight + w.sum())

    def finalize(self, stat: _DenseStat, /) -> BlockDiag[core.Dense] | Joint[core.Dense]:
        r"""$\bar{C}$ as `Dense` operators per block, or one `Joint` with `joint=True`.

        Args:
            stat: The statistic over all the data, after `all_reduce`.

        Returns:
            The per-example average $\bar{C} = \sum_n w_n C_n / \sum_n w_n$. The posterior
            multiplies it by `scale`.
        """

        return _finalize_dense(stat)

    def template(
        self, params: Params, ctx: FitContext, /
    ) -> BlockDiag[core.Dense] | Joint[core.Dense]:
        """Zero `Dense` operators, per block or one `Joint`.

        Args:
            params: The params the curvature is taken at.
            ctx: The fit's context.

        Returns:
            A zero operator with the operator types, blocks, static fields, leaf shapes
            and dtypes that `finalize` returns.
        """

        return _dense_template(params, ctx, self.joint)


@pytree
class _DiagSampled:
    r"""Diagonal sampled Fisher from plain backward passes. Works on any model.

    Targets are always drawn from the model; targets in the batch are ignored. For each of
    the $S$ draws $\tilde{y}_n \sim p(y \mid f_n)$, one backward gives

    $$\tilde{g} = \nabla_\theta \sum_n \sqrt{w_n} \log p(\tilde{y}_n \mid f_n),$$

    and the batch adds $\tilde{g}^{\odot 2} / S$. Cross terms have zero mean under
    model-drawn targets, so $\mathbb{E}[\tilde{g}^{\odot 2}] = \sum_n w_n
    \mathbb{E}[g_n^{\odot 2}]$ with $g_n = \nabla_\theta \log p(\tilde{y}_n \mid f_n)$.

    Args:
        mc_samples: Number of draws $S$ per batch.
    """

    mc_samples: int = static(default=1)

    @property
    def requires(self) -> Requirements:
        """Needs a likelihood that can sample targets, and nothing from the model."""

        return Requirements(needs_sample=True)

    def init(self, params: Params, ctx: FitContext, /) -> SumStat:
        """Zero diagonals per block.

        Args:
            params: The params the curvature is taken at.
            ctx: The fit's context.

        Returns:
            A statistic $e$ with $e + s = s$ for every statistic $s$ of this fit.
        """

        return _zero_stat(params, ctx, _vec_shape)

    def update(
        self,
        stat: SumStat,
        params: Params,
        batch: Batch,
        generator: torch.Generator,
        ctx: FitContext,
        /,
    ) -> SumStat:
        r"""Add $\frac{1}{S} \sum_s \tilde{g}_s^{\odot 2}$ over the batch.

        It detaches `params`, so no gradient flows back to them.

        Args:
            stat: The statistic so far.
            params: The params the curvature is taken at.
            batch: The batch to add.
            generator: The batch's generator, for sampled methods.
            ctx: The fit's context.

        Returns:
            `stat` with the batch added. Updating with batches $a$, then $b$, equals adding
            the statistics of $a$ and of $b$.

        Raises:
            ValueError: `batch.weights` has a negative entry.
            StructureError: The likelihood refuses the outputs or the drawn targets, or its
                mask is not bool or not shaped like its `log_prob`.
        """

        p = {k: v.detach().requires_grad_() for k, v in params.items()}

        with torch.enable_grad():
            f = _forward(ctx, p, batch)
            w = _weights(batch, stat.weight)
            sqrt_w = w.sqrt()
            sums = {b: s for b, s in stat.sum.items()}

            for _ in range(self.mc_samples):
                y = sample(ctx.likelihood, f.detach(), generator=generator)
                per = log_prob(ctx.likelihood, f, y, inputs=batch.inputs)
                obj = (sqrt_w.to(per.dtype) * per).sum()

                names = list(p)
                grads = cast(
                    Sequence[Tensor | None],
                    torch.autograd.grad(
                        obj, [p[k] for k in names], retain_graph=True, allow_unused=True
                    ),
                )
                g = {
                    k: (torch.zeros_like(p[k]) if gk is None else gk)
                    for k, gk in zip(names, grads, strict=True)
                }

                for b in sums:
                    g_b = ctx.layout.flatten(g, b).detach().to(w.dtype)
                    sums[b] = sums[b] + g_b**2 / self.mc_samples

        return SumStat(sums, stat.weight + w.sum().detach())

    def finalize(self, stat: SumStat, /) -> BlockDiag[core.Diag]:
        r"""$\bar{C}$ per block, as `Diag` operators.

        Args:
            stat: The statistic over all the data, after `all_reduce`.

        Returns:
            The per-example average $\bar{C} = \sum_n w_n C_n / \sum_n w_n$. The posterior
            multiplies it by `scale`.
        """

        return _finalize_diag(stat)

    def template(self, params: Params, ctx: FitContext, /) -> BlockDiag[core.Diag]:
        """Zero `Diag` operators per block.

        Args:
            params: The params the curvature is taken at.
            ctx: The fit's context.

        Returns:
            A zero operator with the operator types, blocks, static fields, leaf shapes
            and dtypes that `finalize` returns.
        """

        return _diag_template(params, ctx)


@pytree
class _DiagEmpirical:
    r"""Diagonal empirical Fisher at the batch's targets.

    $$\bar{C} = \frac{1}{W} \sum_n w_n g_n^{\odot 2}, \qquad
    g_n = \nabla_\theta \log p(y_n \mid f_n).$$

    It needs only `log_prob`, so it fits any likelihood. Per-example gradients come from
    `vmap(grad)` when the probe found vmap, else from a forward and backward per example.
    It matches the Fisher only when the targets follow the model.

    It is quadratic in the score, so it refuses a `LateLikelihood` with a late
    hyperparameter: $\sigma$ can't be applied after the fit. That includes `Gaussian()`,
    which learns $\sigma$. A fixed $\sigma$, `Gaussian(sigma=0.1)`, fits. Replacing its
    `sigma` on the posterior afterwards is unsupported and gives a wrong curvature.
    """

    @property
    def requires(self) -> Requirements:
        """Needs targets, and a likelihood without late hyperparameters."""

        return Requirements(needs_targets=True, late_scale=False)

    def init(self, params: Params, ctx: FitContext, /) -> SumStat:
        """Zero diagonals per block.

        Args:
            params: The params the curvature is taken at.
            ctx: The fit's context.

        Returns:
            A statistic $e$ with $e + s = s$ for every statistic $s$ of this fit.
        """

        return _zero_stat(params, ctx, _vec_shape)

    def update(
        self,
        stat: SumStat,
        params: Params,
        batch: Batch,
        generator: torch.Generator,
        ctx: FitContext,
        /,
    ) -> SumStat:
        r"""Add $\sum_n w_n g_n^{\odot 2}$ over the batch.

        Args:
            stat: The statistic so far.
            params: The params the curvature is taken at.
            batch: The batch to add.
            generator: The batch's generator, for sampled methods.
            ctx: The fit's context.

        Returns:
            `stat` with the batch added. Updating with batches $a$, then $b$, equals adding
            the statistics of $a$ and of $b$.

        Raises:
            CapabilityError: The batch has no targets.
            ValueError: `batch.weights` has a negative entry.
            StructureError: The likelihood refuses the outputs or targets, or its mask is
                not bool or not shaped like its `log_prob`.
        """

        y = batch.targets

        if y is None:
            raise CapabilityError(
                'DiagGGN(fisher="empirical") needs targets, but the batch has none'
            )

        n = batch.size
        w = _weights(batch, stat.weight)

        if ctx.caps.vmap:
            g = _flat_blocks(ctx, _vmap_grads(ctx, dict(params), batch.inputs, y), n)
        else:
            g = _loop_grads(ctx, params, batch, y)

        sums = {b: s + (w.unsqueeze(1) * g[b].to(w.dtype) ** 2).sum(0) for b, s in stat.sum.items()}

        return SumStat(sums, stat.weight + w.sum())

    def finalize(self, stat: SumStat, /) -> BlockDiag[core.Diag]:
        r"""$\bar{C}$ per block, as `Diag` operators.

        Args:
            stat: The statistic over all the data, after `all_reduce`.

        Returns:
            The per-example average $\bar{C} = \sum_n w_n C_n / \sum_n w_n$. The posterior
            multiplies it by `scale`.
        """

        return _finalize_diag(stat)

    def template(self, params: Params, ctx: FitContext, /) -> BlockDiag[core.Diag]:
        """Zero `Diag` operators per block.

        Args:
            params: The params the curvature is taken at.
            ctx: The fit's context.

        Returns:
            A zero operator with the operator types, blocks, static fields, leaf shapes
            and dtypes that `finalize` returns.
        """

        return _diag_template(params, ctx)


@pytree
class _DenseEmpirical:
    r"""Dense empirical Fisher at the batch's targets.

    $$\bar{C} = \frac{1}{W} \sum_n w_n g_n g_n^\top, \qquad
    g_n = \nabla_\theta \log p(y_n \mid f_n).$$

    Per block by default. Per-example gradients come as in `DiagGGN(fisher="empirical")`, which
    also lists the likelihoods it refuses.

    Args:
        joint: Keep one matrix over all blocks, in layout order, as a `Joint`.
    """

    joint: bool = static(default=False)

    @property
    def requires(self) -> Requirements:
        """Needs targets, and a likelihood without late hyperparameters."""

        return Requirements(needs_targets=True, late_scale=False)

    def init(self, params: Params, ctx: FitContext, /) -> _DenseStat:
        """Zero matrices per block, or one over all blocks with `joint=True`.

        Args:
            params: The params the curvature is taken at.
            ctx: The fit's context.

        Returns:
            A statistic $e$ with $e + s = s$ for every statistic $s$ of this fit.
        """

        return _init_dense(params, ctx, self.joint)

    def update(
        self,
        stat: _DenseStat,
        params: Params,
        batch: Batch,
        generator: torch.Generator,
        ctx: FitContext,
        /,
    ) -> _DenseStat:
        r"""Add $\sum_n w_n g_n g_n^\top$ over the batch.

        Args:
            stat: The statistic so far.
            params: The params the curvature is taken at.
            batch: The batch to add.
            generator: The batch's generator, for sampled methods.
            ctx: The fit's context.

        Returns:
            `stat` with the batch added. Updating with batches $a$, then $b$, equals adding
            the statistics of $a$ and of $b$.

        Raises:
            CapabilityError: The batch has no targets.
            ValueError: `batch.weights` has a negative entry.
            StructureError: The likelihood refuses the outputs or targets, or its mask is
                not bool or not shaped like its `log_prob`.
        """

        y = batch.targets

        if y is None:
            raise CapabilityError(
                'DenseGGN(fisher="empirical") needs targets, but the batch has none'
            )

        n = batch.size
        w = _weights(batch, stat.weight)

        if ctx.caps.vmap:
            g = _flat_blocks(ctx, _vmap_grads(ctx, dict(params), batch.inputs, y), n)
        else:
            g = _loop_grads(ctx, params, batch, y)

        sums = {
            k: stat.sum[k] + torch.einsum("n,np,nq->pq", w, gk.to(w.dtype), gk.to(w.dtype))
            for k, gk in _joined(g, stat).items()
        }

        return copy.replace(stat, sum=sums, weight=stat.weight + w.sum())

    def finalize(self, stat: _DenseStat, /) -> BlockDiag[core.Dense] | Joint[core.Dense]:
        r"""$\bar{C}$ as `Dense` operators per block, or one `Joint` with `joint=True`.

        Args:
            stat: The statistic over all the data, after `all_reduce`.

        Returns:
            The per-example average $\bar{C} = \sum_n w_n C_n / \sum_n w_n$. The posterior
            multiplies it by `scale`.
        """

        return _finalize_dense(stat)

    def template(
        self, params: Params, ctx: FitContext, /
    ) -> BlockDiag[core.Dense] | Joint[core.Dense]:
        """Zero `Dense` operators, per block or one `Joint`.

        Args:
            params: The params the curvature is taken at.
            ctx: The fit's context.

        Returns:
            A zero operator with the operator types, blocks, static fields, leaf shapes
            and dtypes that `finalize` returns.
        """

        return _dense_template(params, ctx, self.joint)


class _Delegating[O]:
    """Forwards a curvature method's steps to the implementation its options pick."""

    def _impl(self) -> Any:
        raise NotImplementedError

    @property
    def requires(self) -> Requirements:
        """What the chosen implementation needs."""

        return self._impl().requires

    def init(self, params: Params, ctx: FitContext, /) -> Any:
        """The zero statistic of the chosen implementation.

        Args:
            params: The params the curvature is taken at.
            ctx: The fit's context.

        Returns:
            A statistic $e$ with $e + s = s$ for every statistic $s$ of this fit.
        """

        return self._impl().init(params, ctx)

    def update(
        self,
        stat: Any,
        params: Params,
        batch: Batch,
        generator: torch.Generator,
        ctx: FitContext,
        /,
    ) -> Any:
        """Add the batch, as the chosen implementation does.

        Args:
            stat: The statistic so far.
            params: The params the curvature is taken at.
            batch: The batch to add.
            generator: The batch's generator, for sampled methods.
            ctx: The fit's context.

        Returns:
            `stat` with the batch added.
        """

        return self._impl().update(stat, params, batch, generator, ctx)

    def finalize(self, stat: Any, /) -> O:
        r"""$\bar{C}$, the per-example average.

        Args:
            stat: The statistic summed over all batches.

        Returns:
            The curvature operator.
        """

        return self._impl().finalize(stat)

    def template(self, params: Params, ctx: FitContext, /) -> Any:
        """A zero operator of the result's structure, before any data.

        Args:
            params: The params the curvature is taken at.
            ctx: The fit's context.

        Returns:
            The operator `finalize` will return, with zeros.
        """

        return self._impl().template(params, ctx)


@pytree
class DiagGGN(_Delegating[BlockDiag[core.Diag]]):
    r"""Diagonal of the GGN, $C_n = \operatorname{diag}(J_n^\top H_n J_n)$.

    `fisher` picks $H_n$ (see `Fisher`):

    - `"exact"`: the closed form, through $E$ VJPs per example (needs vjp and vmap).
    - `"mc"`: the Monte Carlo Fisher from plain backward passes: for each of the
      $S$ draws $\tilde{y}_n \sim p(y \mid f_n)$, one backward of
      $\sum_n \sqrt{w_n} \log p(\tilde{y}_n \mid f_n)$, squared. Needs nothing from the
      model, so it works on any model (a language model with fused attention, say).
    - `"empirical"`: the squared per-example gradients at the batch's targets.

    `observed=True` takes the Hessian of $-\log p(y \mid f)$ at the batch's targets
    instead of the Fisher, which can be indefinite (a heteroscedastic Gaussian).

    Args:
        fisher: Which Fisher, `"exact"`, `"mc"` or `"empirical"`.
        mc_samples: Draws $S$ for `fisher="mc"`, at least 1 (1 when `None`). Refused with
            another `fisher`.
        observed: Use the observed output Hessian at the targets. Needs `fisher="exact"`.

    Raises:
        ValueError: `fisher` is unknown, `mc_samples` comes without `fisher="mc"` or is
            below 1, or `observed` comes without `fisher="exact"`.
    """

    fisher: Fisher = static(default="exact")
    mc_samples: int | None = static(default=None)
    observed: bool = static(default=False)

    def __post_init__(self) -> None:
        _check_fisher(self, self.fisher, self.observed)
        _check_mc(self, self.fisher, self.mc_samples)

    def _impl(self) -> Any:
        if self.fisher == "mc":
            return _DiagSampled(mc_samples=self.mc_samples or 1)

        if self.fisher == "empirical":
            return _DiagEmpirical()

        return _DiagJacobian(output_hessian="observed" if self.observed else "exact")


@pytree
class DenseGGN(_Delegating[BlockDiag[core.Dense] | Joint[core.Dense]]):
    r"""Dense GGN, $C_n = J_n^\top H_n J_n$, with $J_n$ from `jacrev`.

    So $\bar{C} = \frac{1}{W} \sum_n w_n J_n^\top H_n J_n$. Per block by default:
    cross-block entries are not stored. `fisher` picks $H_n$ (see `Fisher`); with
    `"empirical"` it is the dense empirical Fisher of the per-example gradients.
    `observed=True` takes the Hessian of $-\log p(y \mid f)$ at the batch's targets.

    Args:
        fisher: Which Fisher, `"exact"`, `"mc"` or `"empirical"`.
        mc_samples: Draws for `fisher="mc"`, at least 1 (1 when `None`). Refused with
            another `fisher`.
        observed: Use the observed output Hessian at the targets. Needs `fisher="exact"`.
        joint: Keep one matrix over all blocks, in layout order, as a `Joint`
            (laplace-torch's `hessian_structure="full"` over several layers).

    Raises:
        ValueError: `fisher` is unknown, `mc_samples` comes without `fisher="mc"` or is
            below 1, or `observed` comes without `fisher="exact"`.
    """

    fisher: Fisher = static(default="exact")
    mc_samples: int | None = static(default=None)
    observed: bool = static(default=False)
    joint: bool = static(default=False)

    def __post_init__(self) -> None:
        _check_fisher(self, self.fisher, self.observed)
        _check_mc(self, self.fisher, self.mc_samples)

    def _impl(self) -> Any:
        if self.fisher == "empirical":
            return _DenseEmpirical(joint=self.joint)

        kind = "observed" if self.observed else self.fisher

        return _DenseJacobian(
            output_hessian=kind, mc_samples=self.mc_samples or 1, joint=self.joint
        )


@pytree
class DenseHessian:
    r"""Exact Hessian of the negative log-likelihood at the batch's targets.

    $$C_n = -\nabla_\theta^2 \log p(y_n \mid f(x_n; \theta)),$$

    with masked positions left out, so $\bar{C} = \frac{1}{W} \sum_n w_n C_n$.

    Per block by default: each block's Hessian, with the other blocks held at their
    values. Built with `jacrev(jacrev(...))`, which vmaps a double backward over $P$ basis
    vectors, for $P$ parameters in the block (memory about $P$ times the activations), so
    it needs vjp_twice and vmap from the probe. That refuses some models `jacrev` could
    handle; the GGN curvatures still fit them. The probe checks the double backward in the
    cotangent only, so a custom backward that detaches its saved inputs passes it and gives
    a wrong Hessian.

    The Hessian can be indefinite away from a minimum of the loss. The posterior precision
    can then be indefinite too, and its Cholesky raises `StructureError`. A larger prior
    precision makes it positive definite; the GGN and Fisher curvatures are always PSD.

    Args:
        joint: Keep one matrix over all blocks, in layout order, as a `Joint`. $P$ is then
            the total parameter count.
    """

    joint: bool = static(default=False)

    @property
    def requires(self) -> Requirements:
        """Needs vjp, vjp_twice and vmap, and targets."""

        return Requirements(vjp=True, vjp_twice=True, vmap=True, needs_targets=True)

    def init(self, params: Params, ctx: FitContext, /) -> _DenseStat:
        """Zero matrices per block, or one over all blocks with `joint=True`.

        Args:
            params: The params the curvature is taken at.
            ctx: The fit's context.

        Returns:
            A statistic $e$ with $e + s = s$ for every statistic $s$ of this fit.
        """

        return _init_dense(params, ctx, self.joint)

    def update(
        self,
        stat: _DenseStat,
        params: Params,
        batch: Batch,
        generator: torch.Generator,
        ctx: FitContext,
        /,
    ) -> _DenseStat:
        r"""Add $\sum_n w_n C_n$, the Hessian of the batch's weighted negative log-likelihood.

        Args:
            stat: The statistic so far.
            params: The params the curvature is taken at.
            batch: The batch to add.
            generator: The batch's generator, for sampled methods.
            ctx: The fit's context.

        Returns:
            `stat` with the batch added. Updating with batches $a$, then $b$, equals adding
            the statistics of $a$ and of $b$.

        Raises:
            CapabilityError: The batch has no targets.
            ValueError: `batch.weights` has a negative entry.
            StructureError: The likelihood refuses the outputs or targets, or its mask is
                not bool or not shaped like its `log_prob`.
        """

        y = batch.targets

        if y is None:
            raise CapabilityError("DenseHessian needs targets, but the batch has none")

        p = dict(params)
        w = _weights(batch, stat.weight)

        # One Hessian per group of blocks: all of them (joint), or each alone.
        if isinstance(stat, _JointStat):
            groups = {_JOINT: list(ctx.layout.blocks)}
        else:
            groups = {b: [b] for b in ctx.layout.blocks}

        sums = dict(stat.sum)

        for k, blocks in groups.items():
            sizes = [ctx.layout.size(b) for b in blocks]

            def nll(v: Tensor, blocks: list[str] = blocks, sizes: list[int] = sizes) -> Tensor:
                q = dict(p)

                for b, part in zip(blocks, torch.split(v, sizes), strict=True):
                    q.update(ctx.layout.unflatten(part, b))

                lp = log_prob(ctx.likelihood, ctx.model(q, batch.inputs), y, inputs=batch.inputs)

                return -(w.to(lp.dtype) * lp).sum()

            v = torch.cat([ctx.layout.flatten(p, b) for b in blocks])
            h = cast(Tensor, jacrev(jacrev(nll))(v))
            sums[k] = sums[k] + h.to(w.dtype)

        return copy.replace(stat, sum=sums, weight=stat.weight + w.sum())

    def finalize(self, stat: _DenseStat, /) -> BlockDiag[core.Dense] | Joint[core.Dense]:
        r"""$\bar{C}$ as `Dense` operators per block, or one `Joint` with `joint=True`.

        Args:
            stat: The statistic over all the data, after `all_reduce`.

        Returns:
            The per-example average $\bar{C} = \sum_n w_n C_n / \sum_n w_n$. The posterior
            multiplies it by `scale`.
        """

        return _finalize_dense(stat)

    def template(
        self, params: Params, ctx: FitContext, /
    ) -> BlockDiag[core.Dense] | Joint[core.Dense]:
        """Zero `Dense` operators, per block or one `Joint`.

        Args:
            params: The params the curvature is taken at.
            ctx: The fit's context.

        Returns:
            A zero operator with the operator types, blocks, static fields, leaf shapes
            and dtypes that `finalize` returns.
        """

        return _dense_template(params, ctx, self.joint)


def _draw_test_matrix(seed: int, p: int, k: int) -> Tensor:
    """The Q of a QR of a [p, k] Gaussian draw from `seed`, in float64 on the CPU."""

    draw = torch.randn(p, k, generator=torch.Generator().manual_seed(seed % (1 << 63)))
    q, _ = torch.linalg.qr(draw.double(), mode="reduced")

    return q


@lru_cache(maxsize=4)
def _test_matrix(seed: int, p: int, k: int) -> Tensor:
    """`_draw_test_matrix`, cached: the QR is the cost, and one sketch size repeats."""

    with torch.no_grad():
        return _draw_test_matrix(seed, p, k)


@pytree
class LowRankGGN:
    r"""Low-rank GGN over all fitted blocks jointly, from a Nyström sketch.

    It keeps the top `rank` eigenpairs of the GGN
    $\bar{C} = \frac{1}{W} \sum_n w_n J_n^\top H_n J_n$, by Tropp et al. 2017, Alg. 3.

    Each batch adds $Y = \sum_n w_n J_n^\top H_n J_n \Omega$ for a fixed $P \times k$ test
    matrix $\Omega$ with orthonormal columns, for $P$ fitted parameters and
    $k$ = `rank + oversample`. $\Omega$ comes from the fit's seed (`FitContext.seed`,
    drawn from the fit's `generator`), so it is the same on every batch and every rank
    that passes the same generator, and the sketches add. A batch costs $k$ JVPs (a VJP of a VJP
    each when the probe found no jvp) and $k$ VJPs; the statistic holds $P k$ entries.
    $\Omega$ is kept in a process-wide cache of the last four $(\text{seed}, P, k)$, as
    float64 $P \times k$ matrices that outlive the fit.

    `finalize` divides $Y$ by $W$ and shifts it to $Y + \nu \Omega$, with
    $\nu = \sqrt{P} \, \epsilon \lVert Y \rVert_2$ for the machine epsilon $\epsilon$,
    before its Cholesky and SVD. It clamps eigenvalues that round-off pushes below zero,
    so a rank-deficient $\bar{C}$ stays finite, also in float32 and over many examples.
    Its null directions get eigenvalues of about zero.

    The result is a `Joint[LowRankPlusDiag]` with a zero diagonal; a prior makes it
    invertible. The sketch needs a positive semi-definite $\bar{C}$, so $H_n$ is always the
    Fisher (the closed form, or the MC Fisher); the observed Hessian can be indefinite and
    is not offered. Gradients through `finalize` (D8) need distinct sketch singular values,
    so they can be NaN when $\bar{C}$ has rank below $k$. Models that only support plain
    autograd (checkpointing) need `DiagGGN(fisher="mc")` instead.

    Args:
        rank: Number of eigenpairs kept, at least 1.
        oversample: Extra sketch columns, at least 0.
        fisher: `"exact"` takes the likelihood's closed-form output Hessian, `"mc"` the
            Monte Carlo Fisher. The empirical Fisher is not offered here.
        mc_samples: Draws for the MC Fisher, at least 1; only with `fisher="mc"` (1 when
            `None`).

    Raises:
        ValueError: `rank` is below 1, `oversample` is below 0, `fisher` is not
            `"exact"` or `"mc"`, or `mc_samples` is given without `fisher="mc"` or below 1.
    """

    rank: int = static()
    oversample: int = static(default=10)
    fisher: Literal["exact", "mc"] = static(default="exact")
    mc_samples: int | None = static(default=None)

    def __post_init__(self) -> None:
        if self.fisher not in ("exact", "mc"):
            raise ValueError(f'LowRankGGN takes fisher="exact" or "mc", got {self.fisher!r}')

        _check_mc(self, self.fisher, self.mc_samples)

        if self.rank < 1:
            raise ValueError(f"LowRankGGN needs rank >= 1, got {self.rank}")

        if self.oversample < 0:
            raise ValueError(f"LowRankGGN needs oversample >= 0, got {self.oversample}")

    @property
    def requires(self) -> Requirements:
        """Needs vjp, an output Hessian, and jvp or vjp_twice."""

        return Requirements(
            vjp=True,
            needs_closed_form_hessian=self.fisher == "exact",
            needs_sample=self.fisher == "mc",
            one_of=(Requirements(jvp=True), Requirements(vjp_twice=True)),
        )

    def init(self, params: Params, ctx: FitContext, /) -> _JointStat:
        r"""A zero $P \times k$ sketch over all blocks.

        Args:
            params: The params the curvature is taken at.
            ctx: The fit's context.

        Returns:
            A statistic $e$ with $e + s = s$ for every statistic $s$ of this fit.

        Raises:
            ValueError: `rank + oversample` exceeds the number of fitted parameters $P$.
            StructureError: $H_n$ is the dense MC Fisher and the probe's event size exceeds
                `MAX_DENSE_EVENT`.
        """

        _check_event(self.fisher, ctx, "LowRankGGN")
        sizes = tuple((b, ctx.layout.size(b)) for b in ctx.layout.blocks)
        total = sum(n for _, n in sizes)
        k = self.rank + self.oversample

        if k > total:
            raise ValueError(
                f"LowRankGGN(rank={self.rank}, oversample={self.oversample}) sketches {k} "
                f"columns, more than the P = {total} fitted parameters; rank + oversample "
                f"must be at most {total}"
            )

        zero = _zero_stat(params, ctx, _vec_shape).weight

        return _JointStat({_JOINT: zero.new_zeros(total, k)}, zero, sizes, ctx.seed)

    def _omega(self, seed: int, p: int, k: int, dtype: torch.dtype, device: torch.device) -> Tensor:
        """The [p, k] test matrix: orthonormal, from the fit's seed (drawn on the CPU).

        The float64 CPU matrix is cached per (seed, p, k). Each call returns a copy in
        `dtype` on `device`, so callers can't write into the cached one.
        """

        return _test_matrix(seed, p, k).to(dtype=dtype, device=device, copy=True)

    def update(
        self,
        stat: _JointStat,
        params: Params,
        batch: Batch,
        generator: torch.Generator,
        ctx: FitContext,
        /,
    ) -> _JointStat:
        r"""Add $\sum_n w_n J_n^\top H_n J_n \Omega$ over the batch.

        Args:
            stat: The statistic so far.
            params: The params the curvature is taken at.
            batch: The batch to add.
            generator: The batch's generator, for sampled methods.
            ctx: The fit's context.

        Returns:
            `stat` with the batch added. Updating with batches $a$, then $b$, equals adding
            the statistics of $a$ and of $b$.

        Raises:
            ValueError: `batch.weights` has a negative entry.
            StructureError: The likelihood refuses the outputs, or its mask is not bool or
                not shaped like the leading axes of $f$, or $H_n$ is the dense MC Fisher and
                the event size exceeds `MAX_DENSE_EVENT` (checked here when the probe did
                not record it).
        """

        p = dict(params)
        f = _forward(ctx, p, batch)
        w = _weights(batch, stat.weight)
        hess = _output_hessian(self.fisher, self.mc_samples or 1, ctx, f, batch, generator)

        blocks = [b for b, _ in stat.sizes]
        fitted = {n: p[n] for b in blocks for n in ctx.layout.blocks[b]}
        total, k = stat.sum[_JOINT].shape
        omega = self._omega(stat.seed, total, k, w.dtype, w.device)

        def fwd(q: Params) -> Tensor:
            return cast(Tensor, ctx.model({**p, **q}, batch.inputs))

        pull = cast(Any, vjp(fwd, fitted))[1]

        if ctx.caps.jvp:

            def push(t: Params) -> Tensor:
                return cast(Any, jvp(fwd, (fitted,), (t,)))[1]

        else:
            # J t is the VJP of the linear map u -> J^T u at any u, with cotangent t.
            pull_t = cast(Any, vjp(lambda u: pull(u)[0], torch.zeros_like(f)))[1]

            def push(t: Params) -> Tensor:
                return pull_t(t)[0]

        weight = w.reshape(-1, *[1] * (f.ndim - 1)).to(f.dtype)
        cols: list[Tensor] = []

        for j in range(k):
            parts = torch.split(omega[:, j], [n for _, n in stat.sizes])
            t = {
                name: v.to(fitted[name].dtype)
                for b, part in zip(blocks, parts, strict=True)
                for name, v in ctx.layout.unflatten(part, b).items()
            }

            # J^T (w H (J omega_j)), one VJP over the whole batch.
            g = pull((weight * hess.mv(push(t))).to(f.dtype))[0]
            cols.append(torch.cat([ctx.layout.flatten(g, b).to(w.dtype) for b in blocks]))

        sums = {_JOINT: stat.sum[_JOINT] + torch.stack(cols, dim=1)}

        return copy.replace(stat, sum=sums, weight=stat.weight + w.sum())

    def finalize(self, stat: _JointStat, /) -> Joint[core.LowRankPlusDiag]:
        r"""The top `rank` eigenpairs of $\bar{C}$, in a `Joint[LowRankPlusDiag]`.

        The diagonal is zero. A zero sketch gives the first `rank` columns of $\Omega$
        with zero eigenvalues.

        Args:
            stat: The statistic over all the data, after `all_reduce`.

        Returns:
            The per-example average $\bar{C} = \sum_n w_n C_n / \sum_n w_n$. The posterior
            multiplies it by `scale`.
        """

        _check_total(stat.weight)

        y = stat.sum[_JOINT] / stat.weight
        total, k = y.shape
        omega = self._omega(stat.seed, total, k, y.dtype, y.device)
        zeros = y.new_zeros(total)
        nu = math.sqrt(total) * torch.finfo(y.dtype).eps * torch.linalg.matrix_norm(y, 2)

        # A zero sketch (G = 0) has nothing to factor.
        if bool(nu == 0):
            return Joint(
                core.LowRankPlusDiag(omega[:, : self.rank], zeros[: self.rank], zeros), stat.sizes
            )

        y_nu = y + nu * omega
        m = (omega.T @ y_nu + y_nu.T @ omega) / 2

        # Tropp et al. 2017 (Alg. 3; "Fixed-rank approximation of a positive-semidefinite
        # matrix from streaming data"). G is PSD, so the shifted core has eigenvalues >= nu.
        # Round-off in Y, a sum over many examples, can push some below that; they are
        # clamped up to nu. The lift is a constant, so gradients flow through `m` alone.
        e, v = torch.linalg.eigh(m.detach())
        m = m + (v * (nu.detach() - e).clamp(min=0)) @ v.T

        c = torch.linalg.cholesky(m)
        b = torch.linalg.solve_triangular(c, y_nu.T, upper=False).T
        u, sigma, _ = torch.linalg.svd(b, full_matrices=False)
        lam = (sigma.square() - nu).clamp(min=0)

        return Joint(core.LowRankPlusDiag(u[:, : self.rank], lam[: self.rank], zeros), stat.sizes)

    def template(self, params: Params, ctx: FitContext, /) -> Joint[core.LowRankPlusDiag]:
        """A zero `Joint[LowRankPlusDiag]` of width `rank`.

        Args:
            params: The params the curvature is taken at.
            ctx: The fit's context.

        Returns:
            A zero operator with the operator types, blocks, static fields, leaf shapes
            and dtypes that `finalize` returns.
        """

        sizes = tuple((b, ctx.layout.size(b)) for b in ctx.layout.blocks)
        total = sum(n for _, n in sizes)
        zero = _zero_stat(params, ctx, _vec_shape).weight
        op = core.LowRankPlusDiag(
            zero.new_zeros(total, self.rank), zero.new_zeros(self.rank), zero.new_zeros(total)
        )

        return Joint(op, sizes)


def _vmap_grads(ctx: FitContext, p: Params, inputs: Inputs, y: Targets) -> Mapping[str, Tensor]:
    """Per-example gradients of the masked log-likelihood, name -> [N, *shape]."""

    def one(x: Any, y: Any) -> Mapping[str, Tensor]:
        xb, yb = _lead(x), _lead(y)

        def lp(q: Params) -> Tensor:
            return log_prob(ctx.likelihood, ctx.model(q, xb), yb, inputs=xb).sum()

        return cast(Mapping[str, Tensor], grad(lp)(p))

    return vmap(one)(inputs, y)


def _loop_grads(ctx: FitContext, params: Params, batch: Batch, y: Targets) -> dict[str, Tensor]:
    """Per-example gradients, block -> [N, P_b], from one forward and backward per example.

    Each example runs alone, so the batch costs N single-example passes. The graph to
    `params` is kept when they require grad (D8).
    """

    keep = torch.is_grad_enabled() and any(v.requires_grad for v in params.values())
    p = {
        k: v if keep and v.requires_grad else v.detach().requires_grad_() for k, v in params.items()
    }
    names = list(p)
    rows: dict[str, list[Tensor]] = {b: [] for b in ctx.layout.blocks}

    for i in range(batch.size):
        xi = tree_map(partial(_row, i=i), batch.inputs)
        yi = tree_map(partial(_row, i=i), y)

        with torch.enable_grad():
            per = log_prob(ctx.likelihood, ctx.model(p, xi), yi, inputs=xi)
            grads = cast(
                Sequence[Tensor | None],
                torch.autograd.grad(
                    per.sum(), [p[k] for k in names], create_graph=keep, allow_unused=True
                ),
            )

        g = {
            k: (torch.zeros_like(p[k]) if gk is None else gk)
            for k, gk in zip(names, grads, strict=True)
        }

        for b in rows:
            rows[b].append(ctx.layout.flatten(g, b))

    return {b: torch.stack(r) if keep else torch.stack(r).detach() for b, r in rows.items()}


def _row(t: Tensor, i: int) -> Tensor:
    """Example `i` of a batched leaf, keeping the batch axis."""

    return t[i : i + 1]


def _choose(
    block: str,
    layout: core.Layout,
    caps: Capabilities,
    likelihood: Likelihood | None,
    weighted: bool,
) -> Curvature[Any, Any]:
    exact = caps.vjp and caps.vmap
    names = layout.blocks[block]
    # KFAC needs the layer's weight; a bias-only block goes to the next rule. `check`
    # refuses hook methods on every block once anything is selected, so none gets KFAC then.
    kfac = (
        not weighted
        and not caps.selected
        and KFAC.supports(likelihood, caps, names)
        and block in caps.hooked
        and f"{block}.weight" in names
    )

    # The exact methods cost one pass per output entry, or a dense [N, E, E] output Hessian.
    if caps.event_size is not None and caps.event_size > core.MAX_DENSE_EVENT:
        return KFAC(fisher="mc") if kfac else DiagGGN(fisher="mc")

    # Without a closed-form output Hessian, the GGN methods take the Monte Carlo Fisher.
    closed = likelihood is None or isinstance(likelihood, HasOutputHessian)

    if exact and layout.size(block) <= DENSE_MAX_PARAMS:
        return DenseGGN() if closed else DenseGGN(fisher="mc")

    if kfac:
        return KFAC()

    if exact and closed:
        return DiagGGN()

    return DiagGGN(fisher="mc")


def default(
    params: Params,
    caps: Capabilities,
    /,
    *,
    likelihood: Likelihood | None = None,
    weighted: bool = False,
) -> PerBlockCurvature:
    """A method per block, from the probe.

    The first that fits, in a fixed order: the dense GGN for blocks of at most
    `DENSE_MAX_PARAMS` parameters, KFAC where module hooks reach the layer, the exact
    diagonal, then the sampled Fisher diagonal. The first and third need vjp and vmap.
    Plugins never change the choice.

    Outputs of more than `MAX_DENSE_EVENT` entries per example (`caps.event_size`, e.g.
    tokens times vocabulary) get only sampled methods: `KFAC(fisher="mc")` where KFAC
    fits, else `DiagGGN(fisher="mc")`.

    KFAC is skipped when `KFAC.supports` refuses the block (the likelihood, or a tied
    tensor) or the batches carry per-example weights. It is skipped on every block once
    the probe recorded index selections (`caps.selected`), since `check` then refuses hook
    methods for the whole fit. KFAC also needs the layer's weight in the block, so a
    bias-only block goes to the next method.

    A likelihood without a closed-form output Hessian gets the GGN methods with
    `fisher="mc"`. One with neither an output Hessian nor sampling (`from_log_prob`) gets
    `DiagGGN(fisher="empirical")` on every block, which needs targets.

    Args:
        params: The parameters to fit. Their blocks are those of `Layout.by_module`.
        caps: The probe's capabilities.
        likelihood: The likelihood. With None, KFAC checks only the tie.
        weighted: Whether the batches carry per-example weights.

    Returns:
        The methods as a `PerBlockCurvature`. Blocks with the same method share one rule (`"*"` when
        there is a single method, else a tuple of block names), so each method runs once
        per batch over all its blocks.
    """

    layout = core.Layout.by_module(params)

    # Without an output Hessian or sampling, only the empirical Fisher applies.
    if likelihood is not None and not (
        isinstance(likelihood, HasOutputHessian) or can_sample(likelihood)
    ):
        return core.per_block({"*": DiagGGN(fisher="empirical")})

    # One member per method, over all its blocks: the native methods are block-separable,
    # so this equals a member per block, with one pass per batch instead of one per block.
    groups: list[tuple[Curvature[Any, Any], list[str]]] = []

    for b in layout.blocks:
        method = _choose(b, layout, caps, likelihood, weighted)
        group = next((g for m, g in groups if m == method), None)

        if group is None:
            groups.append((method, [b]))
        else:
            group.append(b)

    if len(groups) == 1:
        return core.per_block({"*": groups[0][0]})

    return core.per_block({tuple(g): m for m, g in groups})
