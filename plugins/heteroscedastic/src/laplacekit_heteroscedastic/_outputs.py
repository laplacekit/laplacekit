"""Helpers that both likelihoods use.

`_halves` splits the last axis of the outputs in two, `_check_targets` checks the targets
against the outputs, and `_mixture` turns Normal draws into one Normal for `average`.
"""

from torch import Tensor
from torch.distributions import Distribution, Normal

from laplacekit.core import (
    StructureError,
)


def _halves(
    f: Tensor, what: str = "NaturalGaussian", parts: str = "the natural parameters [η₁, η₂]"
) -> tuple[Tensor, Tensor]:
    """The two halves of the last axis of f. Raises `StructureError` when its length is odd."""

    if f.ndim < 2 or f.shape[-1] % 2:
        raise StructureError(
            f"{what} needs outputs [N, ..., 2K], {parts} joined on an even last axis; "
            f"got {tuple(f.shape)}"
        )

    first, second = f.chunk(2, -1)

    return first, second


def _check_targets(what: str, f: Tensor, half: Tensor, y: Tensor) -> None:
    if y.shape != half.shape:
        raise StructureError(
            f"{what} targets must be [..., K] = {tuple(half.shape)} for outputs "
            f"{tuple(f.shape)}, got {tuple(y.shape)}"
        )


def _mixture(mean: Tensor, var: Tensor) -> Distribution:
    """One Normal from the draws N(μ_s, σ²_s), with `mean` and `var` shaped `[S, N, ..., K]`.

    The mean is the mean of the μ_s. The variance is the sample variance of the μ_s
    (divided by S − 1, as for the built-in `Gaussian`) plus the mean of the σ²_s. It is an
    unbiased estimate of Var_q[μ] + E_q[σ²].
    """

    mu = mean.mean(0)

    return Normal(mu, (mean.var(0) + var.mean(0)).sqrt())
