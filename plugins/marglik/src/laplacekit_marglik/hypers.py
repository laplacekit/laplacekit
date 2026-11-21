r"""The hyperparameters that `train` fits: the prior precision and the likelihood's noise.

`PriorStructure` names how the weights share the precision (the prior's class). `train` keeps the
precision and a one-element $\sigma$ as float64 logs, and Adam moves them on the negative log
marginal likelihood. A per-output $\sigma$ stays fixed during a fit, and MacKay's
closed-form update (`_noise_update`) sets it after each estimation.
"""

import copy
import dataclasses
import math
from collections.abc import Mapping
from typing import Any, Literal, cast

import torch
from torch import Tensor

import laplacekit as lk
from laplacekit.core import (
    LaplacePosterior,
    Layout,
    Likelihood,
    Params,
    TunablePrior,
)

__all__: list[str] = []


type PriorStructure = Literal["scalar", "layerwise", "diagonal"]
"""How the weights share the prior precision in `train`.

`"scalar"` gives one precision for all weights (`lk.priors.Isotropic`), `"layerwise"` one
per parameter block (`lk.priors.Layerwise`), and `"diagonal"` one per weight
(`lk.priors.Diagonal`).
"""


@dataclasses.dataclass(frozen=True)
class _Hypers:
    """The hyperparameters, as float64 logs.

    They are the prior's precisions, in the prior's leaf order, and the one-element σ of
    an `lk.Gaussian` when it is learned.
    """

    structure: PriorStructure
    blocks: tuple[str, ...]
    prior: tuple[Tensor, ...]
    sigma: Tensor | None

    @property
    def logs(self) -> list[Tensor]:
        return [*self.prior, *(() if self.sigma is None else (self.sigma,))]


def _init(
    structure: PriorStructure, params: Params, prior_init: float, sigma: Tensor | None
) -> _Hypers:
    log = math.log(prior_init)
    blocks = tuple(Layout.by_module(params).blocks)
    size = sum(p.numel() for p in params.values())

    if structure == "scalar":
        prior = (torch.tensor(log, dtype=torch.float64),)
    elif structure == "layerwise":
        prior = tuple(torch.tensor(log, dtype=torch.float64) for _ in blocks)
    else:
        prior = (torch.full((size,), log, dtype=torch.float64),)

    log_sigma = None if sigma is None else sigma.detach().to(torch.float64).log()

    for t in (*prior, *(() if log_sigma is None else (log_sigma,))):
        t.requires_grad_()

    return _Hypers(structure, blocks, prior, log_sigma)


def _prior(hypers: _Hypers, values: tuple[Tensor, ...]) -> TunablePrior:
    """The prior with the given precision leaves."""

    if hypers.structure == "scalar":
        return lk.priors.Isotropic(values[0])

    if hypers.structure == "layerwise":
        return lk.priors.Layerwise(dict(zip(hypers.blocks, values, strict=True)))

    return lk.priors.Diagonal(values[0])


def _prior_leaves(post: LaplacePosterior[Any, Any], hypers: _Hypers) -> tuple[Tensor, ...]:
    """The precision leaves of the posterior's prior, in the order of `hypers.prior`."""

    sel = cast(TunablePrior, post.prior).tunable()

    if isinstance(sel, Mapping):
        return tuple(sel[b] for b in hypers.blocks)

    return (sel,)


def _penalty(hypers: _Hypers, prec: tuple[Tensor, ...], params: Params) -> Tensor:
    """½ Σ_b λ_b ‖θ_b‖², the negative log prior up to a constant."""

    if hypers.structure == "diagonal":
        flat = torch.cat([p.reshape(-1) for p in params.values()])

        return 0.5 * (prec[0] * flat.square()).sum()

    blocks = Layout.by_module(params).blocks
    squares = {
        b: torch.stack([params[n].square().sum() for n in ns]).sum() for b, ns in blocks.items()
    }

    if hypers.structure == "scalar":
        return 0.5 * prec[0] * torch.stack(list(squares.values())).sum()

    terms = [lam * squares[b] for b, lam in zip(hypers.blocks, prec, strict=True)]

    return 0.5 * torch.stack(terms).sum()


def _with_sigma[L: Likelihood](lik: L, sigma: Tensor | None) -> L:
    """`lik` with `sigma` replaced. Only an `lk.Gaussian` is ever given a σ."""

    return lik if sigma is None else cast(L, copy.replace(cast(Any, lik), sigma=sigma))


def _noise_update(
    post: LaplacePosterior[Any, Any], src: lk.data.Data, generator: torch.Generator, sigma: Tensor
) -> Tensor:
    """MacKay's update of a per-output σ, from one pass over `src`.

    σ² = Σ_n w_n [(y_n - f_n)² + Var f(x_n)] / Σ_n w_n, summed down to the shape of σ.
    """

    total: Tensor | None = None
    count = 0.0

    for batch in src(generator):
        # Training has passed over the same batches and raised on one without targets. This
        # still raises for a loader that yields different batches on another pass.
        if batch.targets is None:
            raise ValueError("train needs data with targets, but a batch has none")

        with torch.no_grad():
            out = lk.predict_outputs(post, batch.inputs, n_samples="exact")

        f = cast(Tensor, out.mean)
        terms = (cast(Tensor, batch.targets).reshape(f.shape) - f).square() + out.cov.diagonal()
        weights = batch.weights

        if weights is not None:
            terms = weights.reshape(-1, *[1] * (f.ndim - 1)) * terms
            n = weights.sum().item()
        else:
            n = float(f.shape[0])

        part = terms.sum(0).sum_to_size(sigma.shape)
        total = part if total is None else total + part
        count += n * f[0].numel() / sigma.numel()

    if total is None:
        raise ValueError("train: the data yielded no batches")

    return (total / count).sqrt().to(sigma.dtype)
