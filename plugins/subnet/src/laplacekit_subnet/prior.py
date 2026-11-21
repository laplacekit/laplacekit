"""The prior precision that Daxberger et al. 2021 use for a subnetwork."""

import torch
from torch import Tensor, nn

import laplacekit as lk
from laplacekit_subnet.selection import _check_n

__all__ = ["rescaled_prior"]


def rescaled_prior(
    prior_precision: float | Tensor, n: int, model: nn.Module, /
) -> lk.priors.Isotropic:
    r"""An isotropic prior over a subnetwork of $S$ entries, with precision $\lambda S / D$.

    Daxberger et al. 2021 use $\lambda_S = \lambda S / D$ for a subnetwork of $S$ out of
    $D$ entries. It keeps the prior's total variance, $S / \lambda_S = D / \lambda$.

    Args:
        prior_precision: The full network's prior precision $\lambda$. A float takes the dtype and
            device of `model`'s first parameter.
        n: The subnetwork size $S$, from 1 to $D$.
        model: The network. $D$ counts every parameter entry of `model`, including frozen
            ones.

    Returns:
        `lk.priors.Isotropic` with precision $\lambda S / D$.

    Raises:
        ValueError: `n` is not in $[1, D]$.
    """

    total = sum(p.numel() for p in model.parameters())
    _check_n(n, total, "rescaled_prior")

    if isinstance(prior_precision, Tensor):
        value = prior_precision
    else:
        first = next(model.parameters())
        value = torch.tensor(prior_precision, dtype=first.dtype, device=first.device)

    return lk.priors.Isotropic(value * n / total)
