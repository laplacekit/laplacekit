r"""The `NaturalGaussian` likelihood (Immer et al. 2023) and `NaturalHead`, the layer that feeds it.

The model's outputs are $f = [\eta_1, \eta_2]$, `[..., 2K]`, with $\eta_2 < 0$, so that
$\mu = -\eta_1 / (2 \eta_2)$ and $\sigma^2 = -1 / (2 \eta_2)$. The output Hessian is the
Hessian of the log-partition, one $2 \times 2$ block per Gaussian. It stays
differentiable in $f$, which marginal-likelihood training needs.

`NaturalHead` has no parameters. As the model's last layer, it maps $[h_1, h_2]$ to
$[\eta_1, \eta_2]$ with $\eta_2 < 0$.
"""

import math
from typing import Literal, cast, get_args

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.distributions import Distribution, Normal
from torch.func import vmap

from laplacekit.core import (
    MAX_DENSE_EVENT,
    LinkApprox,
    OutputHessian,
    Outputs,
    StructureError,
    Targets,
    pytree,
)
from laplacekit_heteroscedastic._outputs import _check_targets, _halves, _mixture

__all__ = ["NaturalGaussian", "NaturalHead"]


type Link = Literal["exp", "softplus"]


class NaturalHead(nn.Module):
    r"""A last layer that maps $[h_1, h_2]$ to the natural parameters $[\eta_1, \eta_2]$.

    Use it to end a model for `NaturalGaussian`. The halves of $h$, `[N, ..., 2K]`, map to

    $$\eta_1 = h_1, \qquad \eta_2 = -\tfrac{1}{2} \exp(h_2) \;\text{ or }\;
    \eta_2 = -\tfrac{1}{2} \big(\operatorname{softplus}(h_2) + 10^{-8}\big),$$

    so $\eta_2 < 0$. It has no parameters.

    Args:
        link: `"exp"`, as in Immer et al., or `"softplus"`. With `"softplus"`, $\eta_2$
            stays negative in floating point even when $h_2$ is very negative.

    Raises:
        ValueError: `link` is neither `"exp"` nor `"softplus"`.
    """

    def __init__(self, link: Link = "exp") -> None:
        super().__init__()

        if link not in get_args(Link.__value__):
            raise ValueError(f"NaturalHead link must be 'exp' or 'softplus', got {link!r}")

        self.link = link

    def forward(self, h: Tensor) -> Tensor:
        r"""The natural parameters $[\eta_1, \eta_2]$.

        Args:
            h: The previous layer's output, `[N, ..., 2K]`.

        Returns:
            $[\eta_1, \eta_2]$, shaped like `h`.

        Raises:
            StructureError: `h` has fewer than 2 dims or an odd last axis.
        """

        h1, h2 = _halves(h)
        scale = h2.exp() if self.link == "exp" else F.softplus(h2) + 1e-8

        return torch.cat([h1, -0.5 * scale], -1)

    def extra_repr(self) -> str:
        """Show the link in the module's `repr`.

        Returns:
            For example `"link='exp'"`.
        """

        return f"link={self.link!r}"


def _moments(f: Tensor) -> tuple[Tensor, Tensor]:
    """μ = −η₁ / (2η₂) and σ² = −1 / (2η₂), after checking that `η₂ < 0`."""

    eta1, eta2 = _halves(f)
    bad = int((~(eta2 < 0)).sum())

    if bad:
        raise ValueError(
            f"NaturalGaussian needs η₂ < 0; {bad} entries are not; end the model with NaturalHead()"
        )

    return -eta1 / (2 * eta2), -1 / (2 * eta2)


@pytree
class _NaturalHessian:
    """∇²A(η), with the 2×2 block [[σ², 2μσ²], [2μσ², 4μ²σ² + 2σ⁴]] on each (η₁ₖ, η₂ₖ).

    `mu` and `var` are `[..., K]`, and vectors are shaped like f, `[..., 2K]`.
    """

    mu: Tensor
    var: Tensor

    def _blocks(self) -> tuple[Tensor, Tensor, Tensor]:
        """The entries (a, b, c) of [[a, b], [b, c]]."""

        mu, var = self.mu, self.var

        return var, 2 * mu * var, 4 * mu**2 * var + 2 * var**2

    def mv(self, v: Tensor, /) -> Tensor:
        a, b, c = self._blocks()
        v1, v2 = v.chunk(2, -1)

        return torch.cat([a * v1 + b * v2, b * v1 + c * v2], -1)

    def sqrt_mv(self, v: Tensor, /) -> Tensor:
        """L v with the Cholesky factor L = [[σ, 0], [2μσ, √2 σ²]], so that L Lᵀ = H."""

        sigma = self.var.sqrt()
        v1, v2 = v.chunk(2, -1)

        return torch.cat([sigma * v1, 2 * self.mu * sigma * v1 + math.sqrt(2) * self.var * v2], -1)

    def diagonal(self) -> Tensor:
        a, _, c = self._blocks()

        return torch.cat([a, c], -1)

    def dense(self) -> Tensor:
        """H as `[N, E, E]` over each example's flattened f, block diagonal across positions."""

        shape = (*self.mu.shape[:-1], 2 * self.mu.shape[-1])
        n = shape[0]
        event = math.prod(shape[1:])

        if event > MAX_DENSE_EVENT:
            raise StructureError(
                f"dense NaturalGaussian Hessian with event size {event} is too large"
            )

        # Column i of H is H e_i. vmap over the basis axis keeps f's own shape inside `mv`.
        basis = torch.eye(event, dtype=self.mu.dtype, device=self.mu.device)
        basis = basis.reshape(1, event, *shape[1:]).expand(n, event, *shape[1:])
        cols = vmap(self.mv, in_dims=1, out_dims=1)(basis)

        return cols.reshape(n, event, event)  # symmetric, so rows and columns agree


@pytree
class NaturalGaussian:
    r"""A Gaussian over targets `[..., K]` in its natural parameters $f = [\eta_1, \eta_2]$.

    $f$ is `[..., 2K]`. Per Gaussian,

    $$\mu = -\frac{\eta_1}{2 \eta_2}, \qquad \sigma^2 = -\frac{1}{2 \eta_2}, \qquad
    \log p(y \mid f) = \eta_1 y + \eta_2 y^2 - A(\eta) - \tfrac{1}{2} \log 2\pi,$$

    with the log-partition $A(\eta) = -\eta_1^2 / (4 \eta_2) - \tfrac{1}{2} \log(-2 \eta_2)$.
    This needs $\eta_2 < 0$, which `NaturalHead` guarantees as the model's last layer. The
    output Hessian is the Fisher $\nabla^2 A(\eta)$, which here equals the Hessian of
    $-\log p(y \mid f)$ at any $y$. On $(\eta_{1k}, \eta_{2k})$ it is the block

    $$H_k = \begin{pmatrix} \sigma^2 & 2 \mu \sigma^2 \\
    2 \mu \sigma^2 & 4 \mu^2 \sigma^2 + 2 \sigma^4 \end{pmatrix}.$$

    The probe calls `check_outputs` (`HasOutputCheck`) on the first batch, with or
    without targets, so an odd last axis or an $\eta_2$ that is not negative fails early.
    The empirical Fisher differentiates `log_prob` and would get NaN otherwise.

    KFAC refuses this likelihood, so use `DiagGGN` or `DenseGGN`. The default curvature never
    picks KFAC for it. The default predictive is `NoisyMean`, and `WeightSamples` averages
    its draws as a mixture.
    """

    def check_outputs(self, f: Outputs, /) -> None:
        r"""Check that $f$ is `[N, ..., 2K]` with $\eta_2 < 0$.

        Args:
            f: The outputs $[\eta_1, \eta_2]$, `[N, ..., 2K]`.

        Raises:
            StructureError: $f$ has fewer than 2 dims or an odd last axis.
            ValueError: An entry of $\eta_2$ is not negative.
        """

        _moments(cast(Tensor, f))

    def log_prob(self, f: Outputs, y: Targets, /) -> Tensor:
        r"""$\log p(y \mid f)$ summed over the $K$ targets, shaped `f.shape[:-1]`.

        It does not check $\eta_2 < 0$, so that it works under `torch.func` transforms.

        Args:
            f: The outputs $[\eta_1, \eta_2]$, `[N, ..., 2K]`.
            y: The targets, `[N, ..., K]`.

        Returns:
            The log-likelihood per example, shaped `f.shape[:-1]`.

        Raises:
            StructureError: $f$ has fewer than 2 dims or an odd last axis, or `y` is not
                `[..., K]`, shaped like $f$'s halves.
        """

        ft, yt = cast(Tensor, f), cast(Tensor, y)
        eta1, eta2 = _halves(ft)
        _check_targets("NaturalGaussian", ft, eta1, yt)

        yt = yt.to(ft.dtype)
        lp = (
            eta1 * yt
            + eta2 * yt**2
            + eta1**2 / (4 * eta2)
            + 0.5 * torch.log(-2 * eta2)
            - 0.5 * math.log(2 * math.pi)
        )

        return lp.sum(-1)

    def distribution(self, f: Outputs, /) -> Distribution:
        r"""A Normal with mean $\mu$ and variance $\sigma^2$.

        Args:
            f: The outputs $[\eta_1, \eta_2]$, `[N, ..., 2K]`.

        Returns:
            A `torch.distributions.Normal` over the targets, `[N, ..., K]`.

        Raises:
            StructureError: $f$ has fewer than 2 dims or an odd last axis.
            ValueError: An entry of $\eta_2$ is not negative.
        """

        mu, var = _moments(cast(Tensor, f))

        return Normal(mu, var.sqrt())

    def output_hessian(self, f: Outputs, /) -> OutputHessian:
        r"""The blocks $H_k = \nabla^2 A(\eta)$ on $(\eta_{1k}, \eta_{2k})$.

        It stays differentiable in $f$.

        Args:
            f: The outputs $[\eta_1, \eta_2]$, `[N, ..., 2K]`.

        Returns:
            The output Hessian as an operator over vectors shaped like $f$.

        Raises:
            StructureError: $f$ has fewer than 2 dims or an odd last axis.
            ValueError: An entry of $\eta_2$ is not negative.
        """

        mu, var = _moments(cast(Tensor, f))

        return _NaturalHessian(mu, var)

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

        Its mean is the mean of the $\mu_s$. Its variance is the sample variance of the
        $\mu_s$ (divided by $S - 1$) plus the mean of the $\sigma^2_s$.

        Args:
            f: The drawn outputs, `[S, N, ..., 2K]`.

        Returns:
            A `torch.distributions.Normal` over the targets, `[N, ..., K]`.

        Raises:
            StructureError: $f$ has an odd last axis.
            ValueError: An entry of $\eta_2$ is not negative.
        """

        return _mixture(*_moments(f))
