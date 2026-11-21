r"""The `MeanVarianceGaussian` likelihood, which puts the Laplace approximation over the mean only.

The outputs hold the mean $m$ and the variance $v$, `[..., K]` each, joined on the last
axis. The variance is clamped from below. The output Hessian is $1 / v$ on the mean and
zero on the variance, so the curvature covers the mean only, as in laplace-torch's
`HeteroscedasticGaussianLikelihood`.
"""

from typing import Literal, cast

import torch
from torch import Tensor
from torch.distributions import Distribution, Normal

from laplacekit.core import (
    MAX_DENSE_EVENT,
    LinkApprox,
    OutputHessian,
    Outputs,
    StructureError,
    Targets,
    pytree,
    static,
)
from laplacekit_heteroscedastic._outputs import _check_targets, _halves, _mixture

__all__ = ["MeanVarianceGaussian"]


def _mean_var(f: Tensor) -> tuple[Tensor, Tensor]:
    """Split the outputs of `MeanVarianceGaussian` into the mean and the raw variance."""

    return _halves(f, "MeanVarianceGaussian", "the mean and the variance")


@pytree
class _DiagonalHessian:
    """A diagonal output Hessian H, whose diagonal `prec` is shaped like f."""

    precision: Tensor

    def mv(self, v: Tensor, /) -> Tensor:
        return v * self.precision

    def sqrt_mv(self, v: Tensor, /) -> Tensor:
        return v * self.precision.sqrt()

    def diagonal(self) -> Tensor:
        return self.precision

    def dense(self) -> Tensor:
        n = self.precision.shape[0]
        event = self.precision[0].numel()

        if event > MAX_DENSE_EVENT:
            raise StructureError(
                f"dense MeanVarianceGaussian Hessian with event size {event} is too large"
            )

        return torch.diag_embed(self.precision.reshape(n, event))


@pytree
class MeanVarianceGaussian:
    r"""A Gaussian over targets `[..., K]` from outputs $f = [m, v]$, `[..., 2K]`.

    $m$ is the mean and $v$ the variance, clamped from below at $\epsilon$ to
    $\bar{v} = \max(v, \epsilon)$. For each target,

    $$p(y \mid f) = \mathcal{N}(y \mid m, \bar{v}), \qquad
    H = \operatorname{diag}(1 / \bar{v}, 0),$$

    where $H$ is the output Hessian, $1 / \bar{v}$ on the mean and zero on the variance.
    `log_prob` detaches the variance, so the Laplace approximation covers the mean only,
    as in laplace-torch's `HeteroscedasticGaussianLikelihood`.

    The fitted parameters must include the head that outputs the mean. Use
    `subset=lk.subsets.all` or a subset that holds that head. `last_layer` takes only the
    head that runs last, and if that is the variance head, its curvature is zero.

    The probe calls `check_outputs` (`HasOutputCheck`) on the first batch, with or
    without targets, so an odd last axis fails early. KFAC refuses this likelihood, so
    use `DiagGGN` or `DenseGGN`. The default predictive is `NoisyMean`.

    Args:
        eps: The lower clamp $\epsilon$ on the variance, as in `nn.GaussianNLLLoss`.
    """

    eps: float = static(default=1e-6)

    def check_outputs(self, f: Outputs, /) -> None:
        """Raise `StructureError` unless $f$ is `[N, ..., 2K]`.

        Args:
            f: The outputs $[m, v]$, `[N, ..., 2K]`.
        """

        _mean_var(cast(Tensor, f))

    def _split(self, f: Outputs) -> tuple[Tensor, Tensor]:
        """The mean and the clamped variance."""

        mean, var = _mean_var(cast(Tensor, f))

        return mean, var.clamp_min(self.eps)

    def log_prob(self, f: Outputs, y: Targets, /) -> Tensor:
        r"""$\log \mathcal{N}(y \mid m, \bar{v})$ summed over the $K$ targets.

        $\bar{v}$ is detached. The result is shaped `f.shape[:-1]`.

        Args:
            f: The outputs $[m, v]$, `[N, ..., 2K]`.
            y: The targets, `[N, ..., K]`.

        Returns:
            The log-likelihood per example, shaped `f.shape[:-1]`.

        Raises:
            StructureError: $f$ has fewer than 2 dims or an odd last axis, or `y` is not
                `[..., K]`, shaped like $f$'s halves.
        """

        ft, yt = cast(Tensor, f), cast(Tensor, y)
        mean, var = _mean_var(ft)
        _check_targets("MeanVarianceGaussian", ft, mean, yt)

        scale = var.detach().clamp_min(self.eps).sqrt()

        return Normal(mean, scale).log_prob(yt.to(ft.dtype)).sum(-1)

    def distribution(self, f: Outputs, /) -> Distribution:
        r"""A Normal with mean $m$ and variance $\bar{v}$.

        Args:
            f: The outputs $[m, v]$, `[N, ..., 2K]`.

        Returns:
            A `torch.distributions.Normal` over the targets, `[N, ..., K]`.

        Raises:
            StructureError: $f$ has fewer than 2 dims or an odd last axis.
        """

        mean, var = self._split(f)

        return Normal(mean, var.sqrt())

    def output_hessian(self, f: Outputs, /) -> OutputHessian:
        r"""The diagonal $H = \operatorname{diag}(1 / \bar{v}, 0)$.

        Args:
            f: The outputs $[m, v]$, `[N, ..., 2K]`.

        Returns:
            The output Hessian as an operator over vectors shaped like $f$.

        Raises:
            StructureError: $f$ has fewer than 2 dims or an odd last axis.
        """

        _, var = self._split(f)

        return _DiagonalHessian(torch.cat([1 / var, torch.zeros_like(var)], -1))

    def default_link(self, n_samples: int | Literal["exact"], /) -> LinkApprox:
        """`NoisyMean`, on an exact or a sampled output covariance.

        Args:
            n_samples: The number of Monte Carlo draws that `lk.predict` was given, or `None` for an
                exact output covariance.

        Returns:
            `NoisyMean()`.
        """

        # Imported here: `NoisyMean` checks for this likelihood, so the modules import each other.
        from laplacekit_heteroscedastic.predictive import NoisyMean

        return NoisyMean()

    def average(self, f: Tensor, /) -> Distribution:
        r"""A Normal from draws $f$, `[S, N, ..., 2K]`.

        Its mean is the mean of the $m_s$. Its variance is the sample variance of the
        $m_s$ (divided by $S - 1$) plus the mean of the $\bar{v}_s$.

        Args:
            f: The drawn outputs, `[S, N, ..., 2K]`.

        Returns:
            A `torch.distributions.Normal` over the targets, `[N, ..., K]`.

        Raises:
            StructureError: $f$ has an odd last axis.
        """

        return _mixture(*self._split(f))
