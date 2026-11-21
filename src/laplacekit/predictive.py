r"""Built-in predictives and links.

The predictives are `Linearized` and `WeightSamples`. The links `Probit`, `Bridge`, `MC` and
`Exact` turn the Gaussian over the outputs that `Linearized` gives into a distribution over
targets.

Notation for the links: per input, the outputs (the logits of a classifier) are
$f \sim \mathcal{N}(\bar f, V)$, with $K$ classes for a `Categorical`.
"""

import dataclasses
import math
from typing import Any, Literal, cast

import torch
from torch import Tensor
from torch.distributions import Distribution, Independent, Normal

from laplacekit.core import (
    DEFAULT_OUTPUT_SAMPLES,
    Capabilities,
    CapabilityError,
    FactorCov,
    FunctionPosterior,
    HasDefaultLink,
    HasDistribution,
    HasSampleAverage,
    Inputs,
    LaplacePosterior,
    Likelihood,
    LinkApprox,
    OutputGaussian,
    Posterior,
    Requirements,
    SampleCov,
    Strategy,
    WeightPosterior,
    check_function_args,
    check_outputs_args,
    exact_fits,
    is_transformed,
    mean_outputs,
    predict_outputs,
    pytree,
    resolve_strategy,
    spawn_generator,
    static,
    tree_leaves,
    tree_map,
)
from laplacekit.likelihoods import Bernoulli, Categorical, Gaussian

__all__ = ["MC", "Bridge", "Exact", "Linearized", "Probit", "WeightSamples"]


def _noise(lik: Gaussian, like: Tensor) -> Tensor:
    """Sigma broadcast to `like`'s shape on its device (sigma may sit on the CPU).

    A per-output sigma that doesn't fit `like`'s event shape raises `StructureError`.
    """

    if lik.sigma is None:
        return torch.ones_like(like)

    lik.check_outputs(like)

    return lik.sigma.to(device=like.device).expand_as(like)


def _average(lik: Likelihood, f: Tensor) -> Distribution:
    """Predictive from output n_samples `f` shaped [S, N, ...].

    Categorical: mean of the softmax over samples. Gaussian: moment-matched Normal
    with variance Var_s(f) + sigma^2. Other likelihoods (`Bernoulli` among them): their
    `HasSampleAverage`.
    """

    if isinstance(lik, Categorical):
        _check_classes(lik, f[0])
        dim = lik.class_dim + 1 if lik.class_dim >= 0 else lik.class_dim  # skip sample axis
        return lik.distribution(torch.softmax(f, dim=dim).mean(0).log())

    if isinstance(lik, Gaussian):
        mu = f.mean(0)
        d = Normal(mu, torch.sqrt(f.var(0) + _noise(lik, mu) ** 2))

        return Independent(d, lik.event_dims) if lik.event_dims > 0 else d

    if isinstance(lik, HasSampleAverage):
        return lik.average(f)

    raise CapabilityError(
        f"sampling predictives do not support {type(lik).__name__}: it needs an average "
        "method (HasSampleAverage)"
    )


def _check_distribution(lik: Likelihood, what: str) -> None:
    """A predictive over targets needs the likelihood's distribution."""

    if not isinstance(lik, HasDistribution):
        raise CapabilityError(
            f"{what} gives a distribution over targets, but {type(lik).__name__} has none "
            "(HasDistribution); use lk.predict_outputs(posterior, inputs) for the Gaussian over f"
        )


def _supports_sampling(lik: Likelihood) -> bool:
    return isinstance(lik, Categorical | Gaussian | HasSampleAverage)


_ONE_CLASS_MESSAGE = (
    "a Categorical over one class gives it probability 1; for a scalar output such as a "
    "reward, use lk.predict_outputs(posterior, inputs)"
)
"""The refusal for a `Categorical` with K = 1, shared by the links that see one."""


def _check_classes(lik: Likelihood, f: Tensor, what: str = "a Categorical", /) -> int | None:
    """Refuse a `Categorical` whose outputs `f` [N, ...] have no class axis off the batch axis.

    Scalar outputs [N] have none, say; one class is refused too. Returns the class axis, or
    None for any other likelihood.
    """

    if not isinstance(lik, Categorical):
        return None

    cd = lik.class_dim % f.ndim if f.ndim > 0 else 0

    if f.ndim < 2 or cd == 0:
        raise CapabilityError(
            f"{what} needs outputs [N, ...] with class_dim off the batch axis, got "
            f"class_dim={lik.class_dim} for outputs shaped {tuple(f.shape)}; for a scalar "
            "output such as a reward, use lk.predict_outputs(posterior, inputs)"
        )

    if f.shape[cd] == 1:
        raise CapabilityError(_ONE_CLASS_MESSAGE)

    return cd


def _check_event_size(lik: Likelihood, event_size: int | None, /) -> None:
    """Refuse a `Categorical` whose probed outputs have one entry: one class, or none."""

    if event_size == 1 and isinstance(lik, Categorical):
        raise CapabilityError(_ONE_CLASS_MESSAGE)


def _class_axis(lik: Categorical, mu: Tensor, what: str, /) -> int:
    """The class axis of logits `mu` [N, ...], off the batch axis, with K >= 2."""

    return cast(int, _check_classes(lik, mu, what))


def _classes_last(t: Tensor, cd: int, n: int, k: int, /) -> Tensor:
    """Move the class axis `cd` of `t` [N, ...] last and flatten the rest: [N, P, K]."""

    return t.movedim(cd, -1).reshape(n, -1, k)


def _position_blocks(out: OutputGaussian, mu: Tensor, cd: int, /) -> Tensor:
    """Each position's own [K, K] block of the per-input covariance: [N, P, K, K]."""

    n, k = mu.shape[0], mu.shape[cd]
    e = mu.ndim - 1
    p = math.prod(mu.shape[1:]) // k

    # Per-input blocks [N, E, E] in mu's row-major event order.
    blocks = out.cov.per_input().reshape(n, *mu.shape[1:], *mu.shape[1:])
    blocks = blocks.movedim((cd, e + cd), (e, 2 * e)).reshape(n, p, k, p, k)

    return blocks.diagonal(dim1=1, dim2=3).movedim(-1, 1)


def _per_position(
    out: OutputGaussian, mu: Tensor, cd: int, /
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Per-position moments of logits `mu` [N, ...] with classes on axis `cd`.

    The other event axes are flattened into P positions, and Sigma is each position's
    [K, K] covariance block. Returns the mean [N, P, K], Sigma 1 [N, P, K], 1^T Sigma 1
    [N, P] and diag Sigma [N, P, K]. A `SampleCov` or `FactorCov` is read from its draws
    or factor, with no [K, K] block; any other covariance through `per_input()`.
    """

    n, k = mu.shape[0], mu.shape[cd]
    m = _classes_last(mu, cd, n, k)

    if isinstance(out.cov, SampleCov):
        draws = out.cov.draws  # [S, N, *event]
        eps = draws.movedim(cd + 1, -1).reshape(draws.shape[0], n, -1, k)
        total = eps.sum(-1)

        return m, (eps * total.unsqueeze(-1)).mean(0), (total**2).mean(0), (eps**2).mean(0)

    if isinstance(out.cov, FactorCov):
        factor = out.cov.factor  # [N, *event, R]: the class axis is cd here too
        f = factor.movedim(cd, -2).reshape(n, -1, k, factor.shape[-1])
        total = f.sum(-2)  # [N, P, R]

        return m, (f * total.unsqueeze(-2)).sum(-1), (total**2).sum(-1), (f**2).sum(-1)

    blocks = _position_blocks(out, mu, cd)

    return m, blocks.sum(-1), blocks.sum((-2, -1)), blocks.diagonal(dim1=-2, dim2=-1)


@pytree
class Probit:
    r"""Moderated softmax, or moderated sigmoid, by the probit approximation.

    With logits $f \sim \mathcal{N}(\bar f, V)$ over $K$ classes, the sigmoid $s$ and

    $$\kappa(v) = (1 + \pi v / 8)^{-1/2},$$

    the class probabilities are:

    - $K \ge 3$: the softmax of the centred logits $g = f - \frac{1}{K} \sum_k f_k$, each
      scaled by its own variance,
      $p = \operatorname{softmax}_k\big(\kappa(\operatorname{Var} g_k) \, \mathbb{E}[g_k]\big)$.
    - $K = 2$: the probit of the logit difference $d = f_1 - f_2$, $p = [s(z), s(-z)]$
      with $z = \kappa(\operatorname{Var} d) \, \mathbb{E}[d]$.
    - A `Bernoulli`: the moderated sigmoid $s\big(\kappa(V_{ii}) \, \bar f_i\big)$ per
      entry $i$.

    Softmax ignores a shift shared by all logits, but the variance along that shift is
    often huge (the GGN has a null direction there), and using it would flatten the result
    toward uniform. Centring, or the difference for two classes, removes it.

    Outputs may have extra axes (`[N, T, K]`, or `[N, K, T]` with `class_dim=1`): each
    position uses its own $K \times K$ block of the per-input covariance. A `SampleCov` and
    a `FactorCov` are read through `diagonal()` only: their draws or factor are centred over
    the classes (or differenced) first, so large vocabularies never need a $K \times K$
    block. A `Bernoulli` needs only the variances, read through `diagonal()` from any
    covariance.

    Known limitation: on $K \times K$ blocks in float32, a large shared-shift variance
    cancels the small differences, and the centred or difference variance can come out
    zero or negative. Float64 or `Linearized(strategy="forward-mode")` (whose `FactorCov` is centred
    before the variance is formed) avoid it.
    """

    def supports(self, lik: Likelihood, f_mu: Tensor, /) -> bool:
        """True for a `Categorical` or a `Bernoulli`.

        Args:
            lik: The likelihood.
            f_mu: The output mean.

        Returns:
            True when `__call__` can turn outputs like `f_mu` into a distribution.
        """

        return isinstance(lik, Categorical | Bernoulli)

    def __call__(
        self,
        lik: Likelihood,
        out: OutputGaussian,
        /,
        *,
        generator: torch.Generator | None = None,
    ) -> Distribution:
        """The moderated class probabilities, or the moderated logits of a `Bernoulli`.

        Args:
            lik: A likelihood the link `supports`.
            out: Gaussian posterior over the outputs.
            generator: Ignored: the link draws nothing.

        Returns:
            A distribution over targets.

        Raises:
            CapabilityError: `lik` is neither a `Categorical` nor a `Bernoulli`, or it is a
                `Categorical` and the outputs have no class axis off the batch axis (scalar
                outputs `[N]`, say) or one class.
            ValueError: A centred or difference variance is negative (the float32
                cancellation above). Under a `torch.func` transform that wraps the
                variances, this check is skipped.
            StructureError: A covariance other than `SampleCov` and `FactorCov` refuses
                `per_input()` (too many output entries).
        """

        if isinstance(lik, Bernoulli):
            mu = cast(Tensor, out.mean)
            var = out.cov.diagonal().reshape(mu.shape)

            # The moderated logits themselves: no sigmoid round trip to saturate.
            return lik.distribution(mu / torch.sqrt(1 + math.pi / 8 * var))

        if not isinstance(lik, Categorical):
            raise CapabilityError(
                f"Probit needs a Categorical or Bernoulli likelihood, got {type(lik).__name__}"
            )

        mu = cast(Tensor, out.mean)
        cd = _class_axis(lik, mu, "Probit")

        # Classes last, the other event axes flattened into P positions: [N, P, K].
        n, k = mu.shape[0], mu.shape[cd]
        moved = mu.movedim(cd, -1)
        g = moved.reshape(n, -1, k)

        if k == 2:
            var_d = _nonnegative(self._difference_variance(out, mu, cd))
            z = (g[..., 0] - g[..., 1]) / torch.sqrt(1 + math.pi / 8 * var_d)

            # Logits [z/2, -z/2]: softmax gives [sigmoid(z), sigmoid(-z)] through a
            # log-sum-exp, so confident logits stay finite.
            logits = torch.stack([z / 2, -z / 2], -1)
        else:
            var = _nonnegative(self._centred_variance(out, mu, cd))
            g = g - g.mean(-1, keepdim=True)
            logits = g / torch.sqrt(1 + math.pi / 8 * var)

        return lik.distribution(logits.reshape(moved.shape).movedim(-1, cd))

    @staticmethod
    def _difference_variance(out: OutputGaussian, mu: Tensor, cd: int) -> Tensor:
        """Var(f_1 - f_2) for two classes on axis `cd`, shaped [N, P]."""

        n = mu.shape[0]

        if isinstance(out.cov, SampleCov):
            draws = out.cov.draws
            diff = SampleCov(draws.select(cd + 1, 0) - draws.select(cd + 1, 1))

            return diff.diagonal().reshape(n, -1)

        if isinstance(out.cov, FactorCov):
            factor = out.cov.factor
            diff = FactorCov(factor.select(cd, 0) - factor.select(cd, 1))

            return diff.diagonal().reshape(n, -1)

        # Sigma_11 + Sigma_22 - 2 Sigma_12 = 2 tr(Sigma) - 1^T Sigma 1.
        _, _, total, diag = _per_position(out, mu, cd)

        return 2 * diag.sum(-1) - total

    @staticmethod
    def _centred_variance(out: OutputGaussian, mu: Tensor, cd: int) -> Tensor:
        """Var(f_k - mean_j f_j), shaped [N, P, K] like the centred logits."""

        n, k = mu.shape[0], mu.shape[cd]

        if isinstance(out.cov, SampleCov):
            draws = out.cov.draws
            centred = SampleCov(draws - draws.mean(cd + 1, keepdim=True))

            return _classes_last(centred.diagonal(), cd, n, k)

        if isinstance(out.cov, FactorCov):
            factor = out.cov.factor  # [N, *event, R]: the class axis is cd here too
            centred = FactorCov(factor - factor.mean(cd, keepdim=True))

            return _classes_last(centred.diagonal(), cd, n, k)

        _, row, total, diag = _per_position(out, mu, cd)

        return diag - 2 / k * row + total.unsqueeze(-1) / k**2


def _nonnegative(var: Tensor) -> Tensor:
    """`var` itself; `ValueError` when any entry is negative, which only cancellation gives.

    Under a `torch.func` transform that wraps `var` (`vmap`, `grad`, `jvp`), the check is
    skipped.
    """

    if is_transformed(var):
        return var

    bad = int((var < 0).sum())

    if bad:
        raise ValueError(
            f"Probit got {bad} negative logit variances (down to {float(var.min()):.3g}): "
            f"cancellation in the {var.dtype} [K, K] blocks, typical of float32 under a large "
            'shared-shift variance. Use float64, or Linearized(strategy="forward-mode"), whose '
            "factor is "
            "centred before the variance is formed"
        )

    return var


def _to_float64(t: object) -> object:
    """A floating tensor in float64; anything else unchanged."""

    if isinstance(t, Tensor) and t.is_floating_point():
        return t.to(torch.float64)

    return t


@pytree
class Bridge:
    r"""The Laplace bridge: a Dirichlet over the class probabilities, matched to the logits.

    After Hobbhahn et al. (2022). The predictive is the Dirichlet's mean. Per input (and
    per position), from logits $f \sim \mathcal{N}(\bar f, V)$ over $K$ classes, with
    $\mathbf{1}$ the vector of ones:

    1. The shift shared by all logits is removed first:

       $$\bar f' = \bar f - r \, \mathbf{1}^\top \bar f, \qquad
       v'_k = V_{kk} - \frac{(V \mathbf{1})_k^2}{\mathbf{1}^\top V \mathbf{1}}, \qquad
       r = \frac{V \mathbf{1}}{\mathbf{1}^\top V \mathbf{1}}.$$

       When $\mathbf{1}^\top V \mathbf{1}$ is zero to rounding, $\bar f' = \bar f$ and
       $v'_k = V_{kk}$. "Zero to rounding" is
       $|\mathbf{1}^\top V \mathbf{1}| \le \epsilon \big(\sum_k \sqrt{V_{kk}}\big)^2$, with
       $\epsilon$ the machine epsilon of the covariance's dtype. That is a bound on
       $\sum_{ij} |V_{ij}|$ from the diagonal, up to $K$ times looser than laplace-torch's
       $\epsilon \sum_{ij} |V_{ij}|$, that every covariance type can compute.
    2. The Dirichlet parameters, computed in float64 and in log space, are

       $$\alpha_k \propto \frac{1}{v'_k} \Big(1 - \frac{2}{K}
       + \frac{e^{\bar f'_k}}{K^2} \sum_j e^{-\bar f'_j}\Big),$$

       and the predictive is $p_k = \alpha_k / \sum_j \alpha_j$.

    At $K = 2$ the two corrected variances are equal, so the plain bridge is the softmax
    of the corrected mean whatever the covariance. For two classes use `Probit` (or
    `normalized=True`).

    A row is invalid when a corrected variance is $\le 0$ (a zero output covariance, for
    one) or an $\alpha_k$ is not finite. `fallback` says what happens to it.

    Needs only $V \mathbf{1}$, $\mathbf{1}^\top V \mathbf{1}$ and $\operatorname{diag} V$
    per position, so a `SampleCov` or a `FactorCov` never builds a $K \times K$ block.
    Outputs may have extra axes, as for `Probit`.

    Args:
        normalized: Divide $\bar f'$ by $\sqrt{c}$ before step 2, with
            $c = \big(\frac{1}{K} \sum_k v'_k\big) / \sqrt{K / 2}$.
        fallback: `"raise"` raises on invalid rows. `"probit"` gives them
            $\operatorname{softmax}\big(\tilde f / \sqrt{1 + \pi \max(v', 0) / 8}\big)$ on
            the shift-corrected moments instead, with $\tilde f$ the $\bar f'$ of step 1
            (not divided by $\sqrt{c}$) centred over the classes, as in laplace-torch.
    """

    normalized: bool = static(default=False)
    fallback: Literal["raise", "probit"] = static(default="raise")

    def supports(self, lik: Likelihood, f_mu: Tensor, /) -> bool:
        """True for a `Categorical`.

        Args:
            lik: The likelihood.
            f_mu: The output mean.

        Returns:
            True when `__call__` can turn outputs like `f_mu` into a distribution.
        """

        return isinstance(lik, Categorical)

    def __call__(
        self,
        lik: Likelihood,
        out: OutputGaussian,
        /,
        *,
        generator: torch.Generator | None = None,
    ) -> Distribution:
        r"""The class probabilities $p$, the Dirichlet's mean.

        Args:
            lik: A likelihood the link `supports`.
            out: Gaussian posterior over the outputs.
            generator: Ignored: the link draws nothing.

        Returns:
            A distribution over targets.

        Raises:
            CapabilityError: `lik` is not a `Categorical`, or the outputs have no class axis
                off the batch axis (scalar outputs `[N]`, say) or one class.
            ValueError: `fallback="raise"` and a row is invalid: a corrected variance is
                $\le 0$ or an $\alpha_k$ is not finite.
            StructureError: A covariance other than `SampleCov` and `FactorCov` refuses
                `per_input()` (too many output entries).
        """

        if not isinstance(lik, Categorical):
            raise CapabilityError(
                f"Bridge needs a Categorical likelihood, got {type(lik).__name__}"
            )

        mu = cast(Tensor, out.mean)
        cd = _class_axis(lik, mu, "Bridge")
        k = mu.shape[cd]

        # The moments in float64: float32 sums overflow, and v' cancels. Rounding is the
        # covariance's, as in laplace-torch.
        leaves = [t for t in tree_leaves(out.cov) if isinstance(t, Tensor)]
        eps = torch.finfo(leaves[0].dtype if leaves else mu.dtype).eps
        wide = cast(OutputGaussian, tree_map(_to_float64, out))
        moments = _per_position(wide, cast(Tensor, wide.mean), cd)

        # A covariance that `tree_map` can't enter keeps its dtype up to here.
        m, row, total, diag = (t.to(torch.float64) for t in moments)

        # Remove the shared shift, unless the sum of the logits is certain. The threshold
        # bounds sum |Sigma_ij| by (sum_k sqrt(Sigma_kk))^2, which needs only the diagonal.
        centred = total.abs() <= eps * diag.clamp_min(0).sqrt().sum(-1) ** 2
        safe = torch.where(centred, 1, total).unsqueeze(-1)
        r = torch.where(centred.unsqueeze(-1), 0, row / safe)  # [N, P, K]

        m = m - r * m.sum(-1, keepdim=True)
        v = diag - r * row
        valid = (v > 0).all(-1)

        logits = m

        if self.normalized:
            c = v.sum(-1) / k / math.sqrt(k / 2)
            valid = valid & (c > 0)
            logits = m / torch.where(valid, c, 1).sqrt().unsqueeze(-1)

        # log alpha, less the common log sum_j e^{-mu'_j}.
        log_sum_negative = torch.logsumexp(-logits, -1, keepdim=True)
        log_base = math.log(1 - 2 / k) if k > 2 else -math.inf
        log_alpha = torch.logaddexp(
            torch.full_like(logits, log_base) - log_sum_negative, logits - 2 * math.log(k)
        )

        # At K = 2 both corrected variances are equal and cancel.
        if k > 2:
            log_alpha = log_alpha - v.clamp_min(torch.finfo(torch.float64).tiny).log()

        log_probs = torch.log_softmax(log_alpha, -1)
        valid = valid & torch.isfinite(log_probs).all(-1)

        if not valid.all():
            log_probs = self._fallback(log_probs, valid, m, v)

        moved = mu.movedim(cd, -1)
        log_probs = log_probs.to(mu.dtype).reshape(moved.shape).movedim(-1, cd)

        return lik.distribution(log_probs)

    def _fallback(self, log_probs: Tensor, valid: Tensor, m: Tensor, v: Tensor) -> Tensor:
        """Replace the invalid rows by the probit on the shift-corrected moments, or raise."""

        bad = int((~valid).sum())

        if self.fallback == "raise":
            raise ValueError(
                f"Bridge: {bad} of {valid.numel()} rows have a corrected variance <= 0 or "
                'non-finite Dirichlet parameters; pass Bridge(fallback="probit") to use the '
                "probit on those rows"
            )

        # Centred, so a shift shared by all logits never matters: when the correction was
        # skipped, m is the raw mean. On corrected rows 1^T m is 0 already.
        m = m - m.mean(-1, keepdim=True)
        probit = torch.log_softmax(m / torch.sqrt(1 + math.pi / 8 * v.clamp_min(0)), -1)

        return torch.where(valid.unsqueeze(-1), log_probs, probit)


@pytree
class MC:
    r"""Average over Gaussian draws of the outputs.

    It takes $S$ draws $f_s = \bar f + \varepsilon_s$, with the noise $\varepsilon_s$ from
    the output covariance's `sample`, and averages them into a predictive. A `Categorical`
    gets the mean of $\operatorname{softmax}(f_s)$, and a `Gaussian` the moment-matched
    Normal with variance $\widehat{\operatorname{Var}}_s(f_s) + \sigma^2$. Any other
    likelihood with `HasSampleAverage` (`Bernoulli` among them) averages them itself.

    Args:
        n_samples: The number of draws $S$. A `SampleCov` must hold at least that many.
    """

    n_samples: int = static(default=100)

    def supports(self, lik: Likelihood, f_mu: Tensor, /) -> bool:
        """True for a `Categorical`, a `Gaussian` or a `HasSampleAverage` likelihood.

        Args:
            lik: The likelihood.
            f_mu: The output mean.

        Returns:
            True when `__call__` can turn outputs like `f_mu` into a distribution.
        """

        return _supports_sampling(lik)

    def __call__(
        self,
        lik: Likelihood,
        out: OutputGaussian,
        /,
        *,
        generator: torch.Generator | None = None,
    ) -> Distribution:
        r"""The average of the predictive over the draws.

        Args:
            lik: A likelihood the link `supports`.
            out: Gaussian posterior over the outputs.
            generator: Seeds the draws: one seed is drawn from it, on any device. `None`
                means torch's global RNG. `lk.predict` passes it through `Linearized`.

        Returns:
            A distribution over targets.

        Raises:
            CapabilityError: `lik` is not one that `supports` accepts, or it is a
                `Categorical` and the outputs have no class axis off the batch axis or one
                class.
            ValueError: The covariance is a `SampleCov` with fewer than `n_samples` draws.
            StructureError: A `Gaussian`'s per-output $\sigma$ doesn't fit the outputs'
                event shape.
        """

        if not _supports_sampling(lik):
            raise CapabilityError(
                f"MC does not support {type(lik).__name__}: it needs an average method "
                "(HasSampleAverage)"
            )

        mu = cast(Tensor, out.mean)
        _check_classes(lik, mu)

        # A fresh generator on the outputs' device, so any generator works.
        gen = spawn_generator(generator, device=mu.device)
        noise = out.cov.sample(self.n_samples, generator=gen)
        f = mu.unsqueeze(0) + noise.reshape(self.n_samples, *mu.shape)

        return _average(lik, f)


@pytree
class Exact:
    r"""Gaussian outputs plus observation noise: $\mathcal{N}(\bar f_i, V_{ii} + \sigma^2)$.

    The Normal is per output entry $i$. Only the output variances $V_{ii}$ enter: the
    entries are independent. $\sigma$ is the `Gaussian`'s `sigma`, or 1 when it is `None`.
    """

    def supports(self, lik: Likelihood, f_mu: Tensor, /) -> bool:
        """True for a `Gaussian`.

        Args:
            lik: The likelihood.
            f_mu: The output mean.

        Returns:
            True when `__call__` can turn outputs like `f_mu` into a distribution.
        """

        return isinstance(lik, Gaussian)

    def __call__(
        self,
        lik: Likelihood,
        out: OutputGaussian,
        /,
        *,
        generator: torch.Generator | None = None,
    ) -> Distribution:
        r"""The Normal over targets, with the likelihood's last `event_dims` dims as the event.

        Args:
            lik: A likelihood the link `supports`.
            out: Gaussian posterior over the outputs.
            generator: Ignored: the link draws nothing.

        Returns:
            A distribution over targets.

        Raises:
            CapabilityError: `lik` is not a `Gaussian`.
            StructureError: A per-output $\sigma$ doesn't fit the outputs' event shape.
        """

        if not isinstance(lik, Gaussian):
            raise CapabilityError(f"Exact needs a Gaussian likelihood, got {type(lik).__name__}")

        mu = cast(Tensor, out.mean)
        var = out.cov.diagonal().reshape(mu.shape)
        d = Normal(mu, torch.sqrt(var + _noise(lik, mu) ** 2))

        return Independent(d, lik.event_dims) if lik.event_dims > 0 else d


@pytree
class Linearized:
    r"""Linearize the model at the mean, push the weight-space posterior to outputs, apply a link.

    For a weight-space posterior $\mathcal{N}(\mu, \Sigma)$, the model is linearized at its mean,

    $$f(x; \theta) \approx f(x; \mu) + J(x) (\theta - \mu),$$

    with $J(x)$ the Jacobian of the outputs with respect to the parameters at $\mu$. The
    outputs at $x$ are then Gaussian, $\mathcal{N}\big(f(x; \mu), J(x) \Sigma J(x)^\top\big)$
    (see `outputs`), and the link turns them into a distribution over targets.

    `None` fields are chosen by `resolve` from the posterior's capabilities and
    likelihood.

    On a function-space posterior the link is applied to `posterior.at(inputs)`, which is
    already the Gaussian over the outputs; `strategy`, `n_samples` and `chunk_size` must
    stay unset.

    Args:
        link: Turns the output Gaussian into the predictive. `None` takes it from the
            likelihood: for a `Categorical` or `Bernoulli`, `Probit` on the exact covariance
            and `MC` over all the draws of a `SampleCov`; for a `Gaussian`, `Exact`;
            otherwise the likelihood's `default_link` (`HasDefaultLink`).
        strategy: How the Jacobian products run, `"forward-mode"` (JVPs) or
            `"reverse-mode"` (VJPs), as in `torch.autograd.functional.jacobian`. `None`
            picks: for exact outputs, the cheaper path that fits; for sampled outputs,
            reverse mode (a VJP of a VJP) when the probe found it works, else forward
            mode. The VJP of a VJP keeps the batch's forward and backward graph until the
            last draw; `strategy="forward-mode"` runs the forward per draw instead.
        n_samples: `"exact"` for the exact output covariance. An int (at least 2, and at
            least an `MC` link's `n_samples`) gives a `SampleCov` of that many draws; see
            `lk.predict_outputs`. `None` is exact when the exact path fits its budget,
            else 64 draws, or the `MC` link's `n_samples` if larger.
        chunk_size: Output entries (reverse mode) or parameter directions (forward mode)
            per Jacobian chunk of the exact covariance. `None` means 256.
    """

    link: LinkApprox | None = static(default=None)
    strategy: Strategy | None = static(default=None)
    n_samples: int | Literal["exact"] | None = static(default=None)
    chunk_size: int | None = static(default=None)

    def __repr__(self) -> str:
        fields = (
            ("link", self.link),
            ("strategy", self.strategy),
            ("n_samples", self.n_samples),
            ("chunk_size", self.chunk_size),
        )
        shown = ", ".join(f"{k}={v!r}" for k, v in fields if v is not None)

        return f"Linearized({shown})"

    @property
    def requires(self) -> Requirements:
        """What the pushforward needs, by `strategy` and `n_samples`.

        Exact covariances (`n_samples="exact"`) vmap a JVP over parameter directions or a
        VJP over inputs and output entries; sampled ones push one draw at a time, by a JVP
        or by a VJP of a VJP (`vjp_twice`). An unresolved `strategy` or `n_samples` asks for
        one of the combinations it can resolve to.
        """

        exact = (Requirements(jvp=True, vmap=True), Requirements(vjp=True, vmap=True))
        sampled = (Requirements(jvp=True), Requirements(vjp=True, vjp_twice=True))

        if self.strategy is None:
            if self.n_samples == "exact":
                return Requirements(one_of=exact)

            if self.n_samples is None:
                return Requirements(one_of=exact + sampled)

            return Requirements(one_of=sampled)

        if self.strategy == "forward-mode":
            return Requirements(jvp=True, vmap=self.n_samples == "exact")

        if self.n_samples == "exact":
            return Requirements(vjp=True, vmap=True)

        if self.n_samples is None:
            either = (Requirements(vmap=True), Requirements(vjp_twice=True))
            return Requirements(vjp=True, one_of=either)

        return Requirements(vjp=True, vjp_twice=True)

    def resolve(self, posterior: Posterior, /) -> "Linearized":
        """Fill the `None` fields from the posterior's capabilities and likelihood.

        The event size is the probe's (`caps.event_size`). A function-space posterior gives
        its output Gaussian through `at`, so only `link` is resolved, as for exact outputs.

        Args:
            posterior: A weight-space or function-space posterior.

        Returns:
            A copy with `link`, `strategy` and `n_samples` resolved and `chunk_size` as
            given. For a function-space posterior, a copy with `link` resolved and
            `n_samples="exact"`.

        Raises:
            TypeError: `posterior` is neither a weight-space nor a function-space posterior.
            CapabilityError: The likelihood has no distribution (`lk.predict_outputs` still
                gives the Gaussian over $f$), or it needs a default link and has none (it
                is not a `Categorical` or a `Gaussian` and has no `HasDefaultLink`). For a
                weight-space posterior, also: the likelihood is a `Categorical` and the
                probed outputs have one entry per input (one class, or none), or the model
                lacks what `requires` asks for.
            ValueError: For a function-space posterior, an explicit `strategy`,
                `n_samples` or `chunk_size`. For a weight-space posterior, an int
                `n_samples` below 2, a `chunk_size` below 1, or an int `n_samples` (given,
                or the default number of draws) below the `n_samples` of the `MC` link,
                given or from the likelihood's default link.
        """

        if isinstance(posterior, FunctionPosterior):
            return self._resolve_function(posterior)

        if not isinstance(posterior, WeightPosterior):
            raise TypeError(
                "Linearized needs a weight-space or a function-space posterior, got "
                f"{type(posterior).__name__}"
            )

        return self._resolve(posterior, posterior.caps.event_size)

    def _resolve_function(self, posterior: FunctionPosterior, /) -> "Linearized":
        what = f"Linearized on {type(posterior).__name__}"
        check_function_args(
            what, strategy=self.strategy, n_samples=self.n_samples, chunk_size=self.chunk_size
        )
        _check_distribution(posterior.likelihood, "Linearized")

        given = self.link
        link = _default_link(posterior.likelihood, "exact") if given is None else given

        return dataclasses.replace(self, link=link, n_samples="exact")

    def _resolve(self, posterior: WeightPosterior, event_size: int | None, /) -> "Linearized":
        check_outputs_args(self.n_samples, self.chunk_size, self.strategy)
        _check_distribution(posterior.likelihood, "Linearized")

        # One output entry is one class, or none.
        _check_event_size(posterior.likelihood, event_size)

        caps: Capabilities = posterior.caps
        n_params = posterior.layout.n_params

        if self.strategy is None:
            strategy = resolve_strategy(caps, event_size, n_params, n_samples=self.n_samples)
        else:
            strategy = self.strategy

        given = self.link
        n_samples: int | Literal["exact"]

        if self.n_samples is None:
            draws = max(DEFAULT_OUTPUT_SAMPLES, given.n_samples if isinstance(given, MC) else 0)
            fits = exact_fits(caps, event_size, strategy=strategy, n_params=n_params)
            n_samples = "exact" if fits else draws
        else:
            n_samples = self.n_samples

            if isinstance(n_samples, int):
                _check_samples(n_samples, given)

        link = _default_link(posterior.likelihood, n_samples) if given is None else given

        # A likelihood's default link may ask for more draws than `n_samples` gives.
        if given is None and isinstance(n_samples, int):
            _check_samples(n_samples, link)

        me = dataclasses.replace(self, link=link, strategy=strategy, n_samples=n_samples)

        try:
            caps.check(me.requires, f"Linearized(strategy={strategy!r}, n_samples={n_samples!r})")
        except CapabilityError as e:
            twice = (
                " The sampled pushforward differentiates the model's backward, which fused "
                "attention kernels (scaled_dot_product_attention) can't; a Hugging Face model "
                'loaded with attn_implementation="eager" can.'
                if "vjp_twice" in me.requires.missing(caps) and caps.was_probed("vjp_twice")
                else ""
            )
            raise CapabilityError(
                f"{e}.{twice} Or use the WeightSamples predictive, which needs only forward "
                "passes: lk.predict(post, x, method=lk.WeightSamples())"
            ) from None

        return me

    def __call__(
        self, posterior: Posterior, inputs: Inputs, /, *, generator: torch.Generator | None = None
    ) -> Distribution:
        r"""The linearized predictive at `inputs`.

        `resolve` runs first, before any forward pass. On a weight-space posterior, one forward
        pass then gives the mean outputs, and the link is checked against the likelihood
        before the Jacobian work. The fields are resolved again from the event size of
        these outputs, which may differ from the probe's (`select`), so the default link
        follows it too. On a function-space posterior, `at` gives both moments, and what it
        raises propagates (`CapabilityError`, `StructureError`, `RuntimeError`).

        Args:
            posterior: The posterior, of a type this predictive accepts.
            inputs: Inputs, as the model's `forward` takes them.
            generator: Seeds the weight noise of sampled outputs and the draws of an `MC`
                link. `None` means torch's global RNG. Exact outputs with a closed-form
                link draw nothing.

        Returns:
            A distribution over targets, one per example.

        Raises:
            TypeError: As in `resolve`.
            CapabilityError: As in `resolve`, also for the event size of the outputs; or
                the likelihood is a `Categorical` and the outputs have no class axis off the
                batch axis or one class; or the link doesn't `support` the likelihood.
            ValueError: As in `resolve`, also for the event size of the outputs, or the link
                refuses its numbers (`Probit` with negative logit variances, `Bridge` with
                invalid rows).
            RuntimeError: A parameter changed in place since the fit (`check_versions`).
            NotImplementedError: The model does not return a single tensor.
            StructureError: The posterior's covariance products fail (see `outputs`), or the
                link refuses the outputs (a `Gaussian` whose per-output $\sigma$ doesn't
                fit them, say).
        """

        me = self.resolve(posterior)  # fails before any forward pass
        link = cast(LinkApprox, me.link)

        if isinstance(posterior, FunctionPosterior):
            # One call to `at` gives both moments.
            out = posterior.at(inputs)
            _check_classes(posterior.likelihood, cast(Tensor, out.mean))

            if not link.supports(posterior.likelihood, cast(Tensor, out.mean)):
                raise CapabilityError(
                    f"{type(link).__name__} does not support {type(posterior.likelihood).__name__}"
                )

            return link(posterior.likelihood, out, generator=generator)

        mu = cast(Tensor, mean_outputs(posterior, inputs))
        _check_classes(posterior.likelihood, mu)

        # A forward pass only: an unsupported link fails before the Jacobian work.
        if not link.supports(posterior.likelihood, mu):
            raise CapabilityError(
                f"{type(link).__name__} does not support {type(posterior.likelihood).__name__}"
            )

        # The event size of these outputs, which may differ from the probe's (`select`).
        me = self._resolve(posterior, math.prod(mu.shape[1:]))
        out = predict_outputs(
            posterior,
            inputs,
            strategy=me.strategy,
            n_samples=me.n_samples,
            chunk_size=self.chunk_size,
            generator=generator,
        )

        # The default link follows `n_samples`, which the selected event size can change.
        return cast(LinkApprox, me.link)(posterior.likelihood, out, generator=generator)


def _check_samples(n_samples: int, link: LinkApprox | None, /) -> None:
    """Refuse an int `n_samples` that gives a `SampleCov` the link can't use.

    `check_outputs_args` checks that it is at least 2.
    """

    if isinstance(link, MC) and n_samples < link.n_samples:
        raise ValueError(
            f"MC(n_samples={link.n_samples}) needs at least {link.n_samples} draws, but "
            f"Linearized has n_samples={n_samples}; raise its n_samples or lower MC's"
        )


def _default_link(lik: Likelihood, n_samples: int | Literal["exact"]) -> LinkApprox:
    """The link `Linearized(link=None)` uses for `lik`.

    Categorical: `Probit` on the exact covariance; `MC` over the draws of a `SampleCov`,
    which keeps the correlations between logits that `Probit` (diagonal only) drops.
    Gaussian: `Exact`. Other likelihoods (`Bernoulli` among them): their `HasDefaultLink`.
    """

    if isinstance(lik, Categorical):
        return Probit() if n_samples == "exact" else MC(n_samples=n_samples)

    if isinstance(lik, Gaussian):
        return Exact()

    if isinstance(lik, HasDefaultLink):
        return lik.default_link(n_samples)

    raise CapabilityError(
        f"no default link for {type(lik).__name__}: give it a default_link method "
        "(HasDefaultLink), or pass Linearized(link=...)"
    )


@pytree
class WeightSamples:
    r"""Sample weights from the posterior and average the predictive over the samples.

    For a weight-space posterior $\mathcal{N}(\mu, \Sigma)$ it draws weights
    $\theta_s \sim \mathcal{N}(\mu, \Sigma)$, $s = 1, \dots, S$, runs the model at each,
    and averages the outputs $f(x; \theta_s)$ into a predictive as `MC` does. It needs
    forward passes only.

    Needs a weight-space posterior (`WeightPosterior`): a function-space posterior has no weights
    to sample.

    Args:
        n_samples: The number of weight draws $S$.
    """

    n_samples: int = static(default=100)

    @property
    def requires(self) -> Requirements:
        """Nothing beyond forward passes."""

        return Requirements()

    def __call__(
        self,
        posterior: WeightPosterior,
        inputs: Inputs,
        /,
        *,
        generator: torch.Generator | None = None,
    ) -> Distribution:
        r"""The average of the predictive over the weight draws.

        Args:
            posterior: The posterior, of a type this predictive accepts.
            inputs: Inputs, as the model's `forward` takes them.
            generator: Seeds the weight draws: one seed is drawn from it, on any device.
                `None` means torch's global RNG.

        Returns:
            A distribution over targets, one per example.

        Raises:
            TypeError: `posterior` is not a weight-space posterior (a function-space one has
                no weights to sample).
            CapabilityError: The likelihood has no distribution or is not a `Categorical`,
                a `Gaussian` or a `HasSampleAverage`. Or it is a `Categorical` and the
                probed outputs have one entry per input (refused before any sampling), or
                the first draw's outputs have no class axis off the batch axis or one class
                (refused before the other forward passes).
            RuntimeError: The posterior is a `LaplacePosterior` and a parameter changed in place
                since the fit (`check_versions`).
            StructureError: The posterior's covariance products fail, or a `Gaussian`'s
                per-output $\sigma$ doesn't fit the outputs' event shape.
        """

        if not isinstance(posterior, WeightPosterior):
            raise TypeError(
                f"WeightSamples samples weights, but {type(posterior).__name__} is not a weight "
                "Gaussian; function-space posteriors have no weights to sample: use "
                "lk.Linearized()"
            )

        lik = posterior.likelihood
        _check_distribution(lik, "WeightSamples")

        if not _supports_sampling(lik):
            raise CapabilityError(
                f"WeightSamples does not support {type(lik).__name__}: it needs an average "
                "method (HasSampleAverage)"
            )

        # One output entry is one class, or none: refused before any sampling.
        _check_event_size(lik, posterior.caps.event_size)

        if isinstance(posterior, LaplacePosterior):
            cast(LaplacePosterior[Any, Any], posterior).check_versions()

        # A fresh generator on the mean's device, so any generator works.
        mean = next(iter(posterior.mean.values()), None)
        device = mean.device if mean is not None else None
        gen = spawn_generator(generator, device=device)
        draws = posterior.sample(self.n_samples, generator=gen)  # leading dim n

        def forward(i: int) -> Tensor:
            return cast(Tensor, posterior.model({k: v[i] for k, v in draws.items()}, inputs))

        # The first output's shape is checked before the other forward passes.
        first = forward(0)
        _check_classes(lik, first)

        f = torch.stack([first, *(forward(i) for i in range(1, self.n_samples))])  # [S, N, ...]

        return _average(lik, f)
