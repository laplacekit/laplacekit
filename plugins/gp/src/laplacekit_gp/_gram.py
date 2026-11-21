"""The GGN projected onto a basis, Vᵀ (Σ wₙ Jₙᵀ Hₙ Jₙ) V, as a monoid."""

import copy
from typing import Self, get_args

import torch
from torch import Tensor
from torch.func import vmap

from laplacekit.core import (
    MAX_DENSE_EVENT,
    Batch,
    FitContext,
    HasOutputHessian,
    Params,
    Requirements,
    Strategy,
    StructureError,
    output_hessian,
    pytree,
    stat_dtype,
    static,
)
from laplacekit_gp._jacobian import features

__all__ = ["GramStat", "ProjectedGram", "check_event"]


def check_event(ctx: FitContext, what: str, /) -> None:
    """Raise `StructureError` when the MC Fisher would form dense blocks too large.

    The MC Fisher stands in for the output Hessian of a likelihood without
    `HasOutputHessian`. Its blocks are `[E, E]`, with E from the probe, and E may not
    exceed `MAX_DENSE_EVENT`.
    """

    event = ctx.caps.event_size

    if (
        event is not None
        and event > MAX_DENSE_EVENT
        and not isinstance(ctx.likelihood, HasOutputHessian)
    ):
        raise StructureError(
            f"{what} with the MC Fisher forms a dense {event}x{event} matrix per example; "
            f"give {type(ctx.likelihood).__name__} an output_hessian method (HasOutputHessian)"
        )


@pytree
class GramStat:
    """A weighted sum of `[r, r]` projected GGN terms, with the total weight."""

    gram: Tensor
    weight: Tensor

    @property
    def total_weight(self) -> Tensor:
        return self.weight

    def __add__(self, other: Self, /) -> Self:
        return type(self)(self.gram + other.gram, self.weight + other.weight)

    def scaled(self, s: Tensor, /) -> Self:
        return type(self)(s * self.gram, s * self.weight)


@pytree
class ProjectedGram:
    """Σₙ wₙ Φₙᵀ Hₙ Φₙ, where Φₙ = Jₙ `basis` and `basis` is `[P, r]` in the order of the params.

    H is the output Hessian of the fit's likelihood without its curvature scale, or the
    MC Fisher when the likelihood has no closed-form Hessian. Each term is Φₙᵀ (H Φₙ),
    with `mv` applying H to the r columns of Φ, and is then symmetrized. `strategy` sets how Φ
    is formed (see `features`). `finalize` returns the per-example average with the total
    weight, and refuses a total weight of zero.
    """

    basis: Tensor
    strategy: Strategy = static()

    def __post_init__(self) -> None:
        if self.strategy not in get_args(Strategy.__value__):
            raise ValueError(
                'ProjectedGram strategy must be "forward-mode" or "reverse-mode", '
                f"got {self.strategy!r}"
            )

    @property
    def requires(self) -> Requirements:
        return Requirements(
            jvp=self.strategy == "forward-mode",
            vjp=self.strategy == "reverse-mode",
            vmap=True,
            needs_output_hessian=True,
        )

    def init(self, params: Params, ctx: FitContext, /) -> GramStat:
        n_params = sum(p.numel() for p in params.values())

        if self.basis.ndim != 2 or self.basis.shape[0] != n_params:
            raise StructureError(
                f"ProjectedGram needs a [P, r] basis with P = {n_params} (the parameter "
                f"count), got shape {tuple(self.basis.shape)}"
            )

        check_event(ctx, "ProjectedGram")

        p = next(iter(params.values()))
        dtype = stat_dtype(p.dtype)
        r = self.basis.shape[1]

        return GramStat(
            torch.zeros(r, r, dtype=dtype, device=p.device),
            torch.zeros((), dtype=dtype, device=p.device),
        )

    def update(
        self,
        stat: GramStat,
        params: Params,
        batch: Batch,
        generator: torch.Generator,
        ctx: FitContext,
        /,
    ) -> GramStat:
        dtype = stat.gram.dtype
        f, phi = features(ctx.model, params, batch.inputs, self.basis, self.strategy, dtype)
        hess = output_hessian(ctx.likelihood, f, generator=generator, inputs=batch.inputs)
        n, e = phi.shape[0], phi.shape[1]

        if batch.weights is None:
            w = torch.ones(n, dtype=dtype, device=phi.device)
        elif bool((batch.weights < 0).any()):
            raise ValueError("Batch.weights must be non-negative")
        else:
            w = batch.weights.to(dtype)

        # H applied to Φ's r columns: r `mv` calls under one vmap, none per output entry.
        cols = phi.to(f.dtype).reshape(n, *f.shape[1:], -1)
        h_phi = vmap(hess.mv, in_dims=-1, out_dims=-1)(cols).reshape(n, e, -1).to(dtype)
        term = torch.einsum("n,ner,nes->rs", w, phi, h_phi)
        gram = stat.gram + (term + term.T) / 2

        return copy.replace(stat, gram=gram, weight=stat.weight + w.sum())

    def finalize(self, stat: GramStat, /) -> GramStat:
        if not bool(stat.weight > 0):
            raise ValueError("ProjectedGram saw no data with positive weight")

        return copy.replace(stat, gram=stat.gram / stat.weight)
