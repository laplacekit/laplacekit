"""The predictive of a heteroscedastic Gaussian, for either likelihood.

`moments` takes the linearized outputs of a fit and returns the predictive mean, the
epistemic variance from the weight posterior and the aleatoric variance from the
likelihood's noise. `NoisyMean`, the default predictive of both likelihoods, is a Normal
with that mean and the sum of the two variances.
"""

import math
from typing import cast

import torch
from torch import Tensor
from torch.distributions import Distribution, Normal

from laplacekit.core import (
    CapabilityError,
    Likelihood,
    OutputGaussian,
    pytree,
)
from laplacekit_heteroscedastic.mean_variance import MeanVarianceGaussian
from laplacekit_heteroscedastic.natural import NaturalGaussian, _moments

__all__ = ["Moments", "NoisyMean", "moments"]


@pytree
class Moments:
    """The predictive mean and its epistemic and aleatoric variances, each `[N, ..., K]`.

    `moments` returns it. The total predictive variance is `epistemic + aleatoric`.

    Args:
        mean: The predictive mean.
        epistemic: The variance from the weight posterior.
        aleatoric: The noise variance of the likelihood.
    """

    mean: Tensor
    epistemic: Tensor
    aleatoric: Tensor


def _pair_blocks(out: OutputGaussian) -> Tensor:
    """The 2×2 covariance blocks on (k, K + k) at each position, `[N, ..., K, 2, 2]`."""

    f = cast(Tensor, out.mean)
    n, k = f.shape[0], f.shape[-1] // 2
    p = math.prod(f.shape[1:-1])

    # Per-input blocks [N, E, E] in f's row-major order: E = P · 2 · K.
    blocks = out.cov.per_input().reshape(n, p, 2, k, p, 2, k)
    blocks = blocks.diagonal(dim1=1, dim2=4)  # [N, 2, K, 2, K, P]
    blocks = blocks.diagonal(dim1=2, dim2=4)  # [N, 2, 2, P, K]

    return blocks.permute(0, 3, 4, 1, 2).reshape(*f.shape[:-1], k, 2, 2)


def moments(lik: Likelihood, out: OutputGaussian, /) -> Moments:
    r"""Split the predictive into its mean, epistemic variance and aleatoric variance.

    For `NaturalGaussian`, the mean is $\mu(\hat\eta)$ and the aleatoric variance is
    $\sigma^2(\hat\eta)$, both at the output mean $\hat\eta$. The epistemic variance is

    $$g^\top \Sigma_k \, g, \qquad g = \frac{\partial \mu}{\partial \eta}
    = (\sigma^2, 2 \mu \sigma^2),$$

    where $\Sigma_k$ is the $2 \times 2$ output covariance on $(\eta_{1k}, \eta_{2k})$,
    read from `out.cov.per_input()`. This is the same as linearizing $\mu(x; \theta)$ in
    $\theta$.

    For `MeanVarianceGaussian`, the mean and the clamped variance come from the output mean, and
    the epistemic variance is the output variance of the mean.

    Args:
        lik: The fit's likelihood, a `NaturalGaussian` or a `MeanVarianceGaussian`.
        out: The linearized outputs, `lk.predict_outputs(post, x)`. `NaturalGaussian` allows at
            most `MAX_DENSE_EVENT` output entries per input. `MeanVarianceGaussian` reads only
            `out.cov.diagonal()`.

    Returns:
        The `Moments`.

    Raises:
        CapabilityError: `lik` is neither a `NaturalGaussian` nor a `MeanVarianceGaussian`.
        StructureError: The output mean has fewer than 2 dims or an odd last axis, or
            `out.cov.per_input()` refuses an input with too many output entries (as
            `SampleCov` does above `MAX_DENSE_EVENT`).
        ValueError: With `NaturalGaussian`, an entry of $\eta_2$ at the output mean is
            not negative.
    """

    f = cast(Tensor, out.mean)

    if isinstance(lik, NaturalGaussian):
        mu, var = _moments(f)
        g = torch.stack([var, 2 * mu * var], -1)
        epistemic = torch.einsum("...i,...ij,...j->...", g, _pair_blocks(out), g)

        return Moments(mu, epistemic, var)

    if isinstance(lik, MeanVarianceGaussian):
        mean, var = lik._split(f)
        epistemic, _ = out.cov.diagonal().reshape(f.shape).chunk(2, -1)

        return Moments(mean, epistemic, var)

    raise CapabilityError(
        "moments needs a NaturalGaussian or MeanVarianceGaussian likelihood, "
        f"got {type(lik).__name__}"
    )


@pytree
class NoisyMean:
    """The default predictive of `NaturalGaussian` and `MeanVarianceGaussian`.

    `lk.predict(post, x)` uses it for both likelihoods. It returns a Normal over the
    targets, with the mean from `moments` and the sum of its epistemic and aleatoric
    variances.
    """

    def supports(self, lik: Likelihood, f_mu: Tensor, /) -> bool:
        """True for a `NaturalGaussian` or a `MeanVarianceGaussian`.

        Args:
            lik: The fit's likelihood.
            f_mu: The output mean. It is not used.

        Returns:
            Whether `lik` is a `NaturalGaussian` or a `MeanVarianceGaussian`.
        """

        return isinstance(lik, NaturalGaussian | MeanVarianceGaussian)

    def __call__(
        self,
        lik: Likelihood,
        out: OutputGaussian,
        /,
        *,
        generator: torch.Generator | None = None,
    ) -> Distribution:
        """The Normal from `moments(lik, out)`.

        Args:
            lik: A likelihood the link `supports`.
            out: Gaussian posterior over the outputs.
            generator: Ignored: the link draws nothing.

        Returns:
            A distribution over targets.

        Raises:
            CapabilityError: `lik` is neither a `NaturalGaussian` nor a `MeanVarianceGaussian`.
            StructureError: As in `moments`.
            ValueError: As in `moments`.
        """

        if not isinstance(lik, NaturalGaussian | MeanVarianceGaussian):
            raise CapabilityError(
                "NoisyMean needs a NaturalGaussian or MeanVarianceGaussian likelihood, got "
                f"{type(lik).__name__}"
            )

        m = moments(lik, out)

        return Normal(m.mean, (m.epistemic + m.aleatoric).sqrt())
