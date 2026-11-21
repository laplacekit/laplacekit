"""Built-in likelihoods: `Categorical`, `Gaussian`, `Bernoulli` and `from_log_prob`.

All four work on outputs $f$ shaped `[N, ...]`, `N` examples along the leading axis.
"""

import copy
import dataclasses
import math
from collections.abc import Callable
from typing import Literal, Self, cast

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import Bernoulli as _TorchBernoulli
from torch.distributions import Categorical as _TorchCategorical
from torch.distributions import Distribution, Independent, Normal
from torch.func import vmap

from laplacekit.core import (
    MAX_DENSE_EVENT,
    HasMask,
    Inputs,
    Likelihood,
    LinkApprox,
    OutputHessian,
    Outputs,
    StructureError,
    Targets,
    default_log_prob,
    pytree,
    static,
)

__all__ = ["Bernoulli", "Categorical", "Gaussian", "LogProb", "from_log_prob"]


@pytree
class _CategoricalHessian:
    """H = diag(p) - p pᵀ per example, with the class axis at `dim`."""

    p: Tensor
    dim: int = static()

    def mv(self, v: Tensor, /) -> Tensor:
        p = self.p
        return p * v - p * (p * v).sum(self.dim, keepdim=True)

    def sqrt_mv(self, v: Tensor, /) -> Tensor:
        """A v with A = diag(sqrt p) - p sqrt(p)ᵀ, so A Aᵀ = H."""

        p = self.p
        s = p.sqrt()

        return s * v - p * (s * v).sum(self.dim, keepdim=True)

    def diagonal(self) -> Tensor:
        return self.p * (1 - self.p)

    def dense(self) -> Tensor:
        """[N, E, E] over each example's flattened f, block diagonal across positions."""

        n = self.p.shape[0]
        event = self.p[0].numel()

        if event > MAX_DENSE_EVENT:
            raise StructureError(f"dense Categorical Hessian with event size {event} is too large")

        # Column i of H is H e_i. vmap over the basis axis keeps f's own shape inside `mv`.
        basis = torch.eye(event, dtype=self.p.dtype, device=self.p.device)
        basis = basis.reshape(1, event, *self.p.shape[1:]).expand(n, event, *self.p.shape[1:])
        cols = vmap(self.mv, in_dims=1, out_dims=1)(basis)

        return cols.reshape(n, event, event)  # symmetric, so rows and columns agree


@pytree
class Categorical:
    r"""Softmax likelihood over a class axis of $f$.

    At each position, with $q = \operatorname{softmax}(f)$ over the classes,

    $$p(y \mid f) = q_y, \qquad H = \operatorname{diag}(q) - q q^\top,$$

    where $H$ is the output Hessian.

    Targets are class indices, shaped like $f$ without the class axis. `log_prob` is per
    position, shaped like the targets. With a mask the class axis must be last.

    Args:
        class_dim: The class axis of $f$. The default is the last axis.
        mask_fn: `mask_fn(f, inputs)` returns a bool tensor shaped like the targets, False
            at positions that don't count (padding, say), or None when every position
            counts. Without it, every position counts.

    Raises:
        StructureError: `class_dim` is 0, the batch axis.
    """

    class_dim: int = static(default=-1)
    mask_fn: Callable[[Outputs, Inputs], Tensor | None] | None = static(default=None)

    def __post_init__(self) -> None:
        if self.class_dim == 0:
            raise StructureError(
                "Categorical(class_dim=0) puts the classes on the batch axis; outputs are "
                "[N, ...], so use a class axis of 1 or more, or a negative one"
            )

    def check_targets(self, y: Targets, /) -> None:
        """Raise `ValueError` unless the targets are integer class indices.

        A torch `Categorical` casts float targets to integers, so `1.7` would count as
        class 1.

        Args:
            y: The batch's targets.
        """

        t = cast(Tensor, y)

        if t.is_floating_point() or t.is_complex() or t.dtype == torch.bool:
            raise ValueError(
                f"Categorical needs class indices as targets (an integer tensor), got "
                f"{t.dtype}; pass y.long(), or use Bernoulli for 0/1 labels"
            )

    def log_prob(self, f: Outputs, y: Targets, /) -> Tensor:
        r"""$\log q_y$ at each position.

        Args:
            f: Model outputs for a batch.
            y: Targets for the same batch.

        Returns:
            A tensor shaped `[N]`, or `[N, *positions]` (per token, say).

        Raises:
            StructureError: `y` is not shaped like $f$ without the class axis.
        """

        return default_log_prob(self, f, y)

    def distribution(self, f: Outputs, /) -> Distribution:
        """A torch `Categorical` with logits $f$, the class axis moved last.

        Args:
            f: Model outputs for a batch.

        Returns:
            A torch distribution over the batch's targets.
        """

        return _TorchCategorical(logits=cast(Tensor, f).movedim(self.class_dim, -1))

    def mask(self, f: Outputs, inputs: Inputs, /) -> Tensor | None:
        """`mask_fn(f, inputs)`, or None without `mask_fn`.

        Args:
            f: Model outputs for a batch.
            inputs: The batch's inputs.

        Returns:
            A bool tensor shaped like the output of `log_prob`, `[N, *positions]`, True
            where a position counts, or `None` when every position counts.
        """

        return None if self.mask_fn is None else self.mask_fn(f, inputs)

    def output_hessian(self, f: Outputs, /) -> OutputHessian:
        r"""$\operatorname{diag}(q) - q q^\top$ at each position.

        Args:
            f: Model outputs for a batch.

        Returns:
            The per-example output Hessian.
        """

        p = torch.softmax(cast(Tensor, f), dim=self.class_dim)
        return _CategoricalHessian(p, self.class_dim)


@pytree
class _DiagonalHessian:
    """A diagonal H, with its diagonal `precision` shaped like f."""

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
            raise StructureError(f"dense diagonal Hessian with event size {event} is too large")

        return torch.diag_embed(self.precision.reshape(n, event))


def _like(t: Tensor, f: Tensor) -> Tensor:
    """`t` broadcast to f's shape on f's device and in f's dtype when f is floating.

    A float sigma is stored in double precision; at f's dtype it neither promotes f nor
    loses precision in a float64 fit.
    """

    dtype = f.dtype if f.is_floating_point() else t.dtype

    return t.to(device=f.device, dtype=dtype).expand_as(f)


def _fits(shape: tuple[int, ...], event: tuple[int, ...]) -> bool:
    """Whether `shape` broadcasts to `event` by trailing alignment without adding dims."""

    if len(shape) > len(event):
        return False

    return all(s in (1, e) for s, e in zip(reversed(shape), reversed(event), strict=False))


@pytree
class Gaussian:
    r"""Independent Gaussian noise over the last `event_dims` dims of $f$.

    Over the entries $e$ of $f$,

    $$p(y \mid f) = \prod_e \mathcal{N}(y_e \mid f_e, \sigma_e^2), \qquad
    H = \operatorname{diag}(1 / \sigma_e^2),$$

    where $H$ is the output Hessian and $\sigma_e = \sigma$ unless $\sigma$ is per output.

    `Gaussian()` learns $\sigma$: the hyper tuner of `lk.laplace` fits it together with
    the prior precision. A learned $\sigma$ is applied after the fit (`LateLikelihood`): the
    fit uses unit noise and keeps the weighted sum of squared residuals and the count of
    entries, so tuning $\sigma$ never refits.

    `Gaussian(sigma=0.1)` fixes $\sigma$. A fixed $\sigma$ is applied at fit time: the
    curvature holds it, and no tuner changes it. A $\sigma$ with several elements is per
    output and also applied at fit time. The fit then keeps the sums of squared residuals
    and the counts per entry of the event. `lk.marglik.train` learns a sigma per output
    from `sigma_init=` (MacKay's update); the hyper tuners leave one alone.

    The predictive keeps the event shape of $f$. `log_prob` is per position: $f$'s shape
    without the last `event_dims`.

    Args:
        sigma: The noise scale $\sigma$, a float or a tensor of positive entries, kept
            fixed. A float stays in double precision until the fit casts it to its
            statistic's dtype. `None` (the default) learns it, starting at 1 or at the tuner's
            `sigma_init`. A tensor with several elements must
            broadcast to the event shape `f.shape[1:]` without adding dims (`[C]` for
            outputs `[N, C]`, `[C]` or `[T, C]` for `[N, T, C]`); otherwise the first call
            that sees $f$ raises `StructureError`.
        event_dims: How many trailing dims of $f$ form one event.

    Raises:
        ValueError: `sigma` has an entry that is not positive.
    """

    sigma: Tensor
    event_dims: int = static(default=0)
    learned: bool = static(default=False)

    def __init__(self, sigma: float | Tensor | None = None, *, event_dims: int = 0) -> None:
        # A Python number keeps its double precision until the fit casts it to the
        # statistic's dtype, so Gaussian(0.3) is 0.3 in a float64 fit.
        if sigma is None or isinstance(sigma, int | float):
            value = torch.tensor(1.0 if sigma is None else float(sigma), dtype=torch.float64)
        else:
            value = torch.as_tensor(sigma)

        if not value.is_floating_point():
            value = value.to(torch.get_default_dtype())

        if not bool((value > 0).all()):
            raise ValueError(f"Gaussian sigma must be positive, got {value}")

        object.__setattr__(self, "sigma", value)
        object.__setattr__(self, "event_dims", event_dims)
        object.__setattr__(self, "learned", sigma is None)

    def __replace__(self, /, **changes: object) -> Self:
        """A copy with `changes`; a learned sigma stays learned when its value changes."""

        sigma = changes.get("sigma")

        if isinstance(sigma, Tensor) and not bool((sigma > 0).all()):
            raise ValueError(f"Gaussian sigma must be positive, got {sigma}")

        out = copy.copy(self)

        for name, value in changes.items():
            if name not in {f.name for f in dataclasses.fields(self)}:
                raise TypeError(f"Gaussian has no field {name!r}")

            object.__setattr__(out, name, value)

        return out

    def __repr__(self) -> str:
        sigma = (
            f"{self.sigma.item():.4g}" if self.sigma.numel() == 1 else str(tuple(self.sigma.shape))
        )
        extra = f", event_dims={self.event_dims}" if self.event_dims else ""
        state = "learned" if self.learned else "fixed"

        return f"Gaussian(sigma={sigma} ({state}){extra})"

    @property
    def per_output(self) -> bool:
        r"""Whether $\sigma$ has several elements, one per output entry."""

        return self.sigma.numel() > 1

    def tunable(self) -> dict[str, Tensor]:
        r"""The hyperparameters a hyper tuner may change.

        Returns:
            `{"sigma": sigma}` when $\sigma$ is learned and has one element, else an empty
            dict: a per-output $\sigma$ is applied at fit time.
        """

        return {"sigma": self.sigma} if self.learned and not self.per_output else {}

    def check_outputs(self, f: Outputs, /) -> None:
        r"""Raise `StructureError` when a per-output $\sigma$ doesn't fit $f$'s event shape.

        As a `HasOutputCheck`, the probe calls it on its own forward, before any data is
        fitted.

        Args:
            f: Outputs of the probe's plain forward.
        """

        ft = cast(Tensor, f)
        sigma = self.sigma

        if sigma.numel() > 1 and not _fits(sigma.shape, ft.shape[1:]):
            raise StructureError(
                f"Gaussian sigma of shape {sigma.shape} doesn't fit outputs with event shape "
                f"{ft.shape[1:]}"
            )

    def _sigma_like(self, f: Tensor) -> Tensor:
        """Sigma broadcast to f's shape on f's device."""

        if self.per_output:
            self.check_outputs(f)
            return _like(self.sigma, f)

        return _like(self.sigma.reshape(()), f)

    def unit(self) -> Self:
        """The likelihood the fit uses: unit noise for a learned sigma, else itself.

        The copy keeps the class and every other field, so a subclass keeps its hooks (a
        mask, say).

        Returns:
            A copy with a fixed unit sigma when sigma is learned, else `self` (by
            identity): a fixed sigma is applied at fit time.
        """

        if not self.learned or self.per_output:
            return self

        return copy.replace(self, sigma=torch.ones_like(self.sigma), learned=False)

    def curvature_scale(self) -> Tensor:
        r"""$1 / \sigma^2$ for a learned $\sigma$; 1 for a fixed one, which the fit applied.

        Returns:
            A tensor computed from the current hyperparameters.
        """

        if not self.learned or self.per_output:
            return torch.ones((), dtype=self.sigma.dtype, device=self.sigma.device)

        return 1.0 / self.sigma.reshape(()) ** 2

    def fit_stat(self, f: Outputs, y: Targets, weights: Tensor | None, /) -> Tensor:
        r"""The statistic $[r, m]$ for $N$ examples with $E$ entries each.

        $$r = \sum_{n=1}^N w_n \lVert y_n - f_n \rVert^2, \qquad m = \sum_{n=1}^N w_n E,$$

        with $w_n = 1$ when `weights` is None. With a per-output $\sigma$, the same two
        sums per entry of the event, shaped `[2, *f.shape[1:]]`.

        Args:
            f: Model outputs for a batch.
            y: Targets for the same batch.
            weights: Per-example weights, or `None` for all ones.

        Returns:
            The batch's statistic.

        Raises:
            StructureError: `y` is not shaped like `f`, or a per-output $\sigma$ doesn't
                fit $f$'s event shape.
        """

        ft, yt = cast(Tensor, f), cast(Tensor, y)

        if yt.shape != ft.shape:
            raise StructureError(
                f"Gaussian needs targets shaped like f {tuple(ft.shape)}, got {tuple(yt.shape)}"
            )

        if self.per_output:
            self.check_outputs(ft)
            sq = (yt - ft) ** 2
            entries = torch.ones_like(sq)

            if weights is not None:
                w = weights.reshape(-1, *([1] * (ft.ndim - 1)))
                sq, entries = w * sq, w * entries

            return torch.stack([sq.sum(0), entries.sum(0)])

        sq = ((yt - ft) ** 2).reshape(ft.shape[0], -1).sum(1)
        entries = torch.full_like(sq, float(ft[0].numel()))

        if weights is not None:
            sq, entries = weights * sq, weights * entries

        return torch.stack([sq.sum(), entries.sum()])

    def log_lik(self, stat: Tensor, /) -> Tensor:
        r"""The log-likelihood from the statistic $[r, m]$ of `fit_stat`.

        $$\log p = -\frac{1}{2} \Big(m \log(2 \pi \sigma^2) + \frac{r}{\sigma^2}\Big).$$

        For a per-output $\sigma$, the same per entry of the event, summed.

        Args:
            stat: `fit_stat` summed over all batches.

        Returns:
            The summed log-likelihood at the current hyperparameters.

        Raises:
            StructureError: `stat` doesn't fit the kind of $\sigma$: shape `[2]` for a
                one-element $\sigma$, `[2, *event]` (broadcast from $\sigma$'s shape) for a
                per-output one. A per-output $\sigma$ is fixed at fit time, so a fit with
                one kind of $\sigma$ can't take the other.
        """

        sigma = self.sigma
        fits = stat.shape == (2,) if not self.per_output else _fits(sigma.shape, stat.shape[1:])

        if not fits:
            raise StructureError(
                f"Gaussian log_lik got a statistic of shape {tuple(stat.shape)} for sigma "
                f"{tuple(sigma.shape)}; a per-output sigma is fixed at fit time, so a fit "
                "with one kind of sigma can't take the other"
            )

        if self.per_output:
            var = sigma.to(stat.device) ** 2
            var = var.expand(stat.shape[1:])

            return -0.5 * (stat[1] * torch.log(2 * math.pi * var) + stat[0] / var).sum()

        var = sigma.reshape(()).to(stat.device) ** 2

        return -0.5 * (stat[1] * torch.log(2 * math.pi * var) + stat[0] / var)

    def log_prob(self, f: Outputs, y: Targets, /) -> Tensor:
        r"""Gaussian log density per position, summed over the event.

        Args:
            f: Model outputs for a batch.
            y: Targets for the same batch.

        Returns:
            A tensor shaped `[N]`, or `[N, *positions]` (per token, say).

        Raises:
            StructureError: `y` is not shaped like `f`, or a per-output $\sigma$ doesn't
                fit $f$'s event shape.
        """

        return default_log_prob(self, f, y)

    def distribution(self, f: Outputs, /) -> Distribution:
        r"""`Normal(f, sigma)`, with the last `event_dims` dims as the event.

        Args:
            f: Model outputs for a batch.

        Returns:
            A torch distribution over the batch's targets.

        Raises:
            StructureError: A per-output $\sigma$ doesn't fit $f$'s event shape.
        """

        ft = cast(Tensor, f)
        d = Normal(ft, self._sigma_like(ft))

        return Independent(d, self.event_dims) if self.event_dims > 0 else d

    def output_hessian(self, f: Outputs, /) -> OutputHessian:
        r"""$\operatorname{diag}(1 / \sigma_e^2)$.

        Args:
            f: Model outputs for a batch.

        Returns:
            The per-example output Hessian.

        Raises:
            StructureError: A per-output $\sigma$ doesn't fit $f$'s event shape.
        """

        ft = cast(Tensor, f)

        return _DiagonalHessian(1.0 / self._sigma_like(ft) ** 2)


@pytree
class Bernoulli:
    r"""Sigmoid likelihood: each entry of $f$ is the logit of an independent 0/1 target.

    Per entry, with $s(f) = 1 / (1 + e^{-f})$ the sigmoid,

    $$\log p(y \mid f) = y \log s(f) + (1 - y) \log s(-f), \qquad
    H = \operatorname{diag}\big(s(f) \, s(-f)\big),$$

    where $H$ is the output Hessian.

    Targets are shaped like $f$: 0/1 labels (bool or int are cast to $f$'s dtype) or soft
    labels in $[0, 1]$. The fit refuses other values (`check_targets`), and so does the
    predictive's `log_prob`, which takes only 0/1. `log_prob` is per position: $f$'s shape
    without the last `event_dims`, over which it sums. The default link is the sigmoid
    `Probit` on an exact output covariance and `MC` over the draws of a sampled one.

    `BCEWithLogitsLoss`'s `pos_weight` has no counterpart: it reweights the positive
    targets, so the output Hessian would depend on the targets. Its `weight` is left out
    on purpose; `Batch.weights` gives per-example weights.

    Args:
        event_dims: How many trailing dims of $f$ form one event.

    Raises:
        StructureError: `event_dims` is negative.
    """

    event_dims: int = static(default=0)

    def __post_init__(self) -> None:
        if self.event_dims < 0:
            raise StructureError(f"Bernoulli event_dims must be 0 or more, got {self.event_dims}")

    def check_targets(self, y: Targets, /) -> None:
        """Raise `ValueError` unless every target lies in $[0, 1]$.

        Args:
            y: The batch's targets.
        """

        t = cast(Tensor, y).to(torch.float64)

        if t.numel() and not bool(((t >= 0) & (t <= 1)).all()):
            raise ValueError(
                f"Bernoulli targets must lie in [0, 1], got values in "
                f"[{float(t.min())}, {float(t.max())}]; class indices need Categorical"
            )

    def log_prob(self, f: Outputs, y: Targets, /) -> Tensor:
        """Log-likelihood per position, summed over the last `event_dims` dims.

        Soft labels are taken as they are; `check_targets` is what refuses values outside
        $[0, 1]$.

        Args:
            f: Model outputs for a batch.
            y: Targets for the same batch.

        Returns:
            A tensor shaped `[N]`, or `[N, *positions]` (per token, say).

        Raises:
            StructureError: `y` is not shaped like `f`.
        """

        ft, yt = cast(Tensor, f), cast(Tensor, y)

        if yt.shape != ft.shape:
            raise StructureError(
                f"Bernoulli targets must be shaped like f {tuple(ft.shape)}, got {tuple(yt.shape)}"
            )

        lp = -F.binary_cross_entropy_with_logits(ft, yt.to(ft.dtype), reduction="none")

        return lp.sum(tuple(range(-self.event_dims, 0))) if self.event_dims > 0 else lp

    def distribution(self, f: Outputs, /) -> Distribution:
        """A torch `Bernoulli` with logits $f$, the last `event_dims` dims as the event.

        Args:
            f: Model outputs for a batch.

        Returns:
            A torch distribution over the batch's targets.
        """

        d = _TorchBernoulli(logits=cast(Tensor, f))

        return Independent(d, self.event_dims) if self.event_dims > 0 else d

    def output_hessian(self, f: Outputs, /) -> OutputHessian:
        r"""$\operatorname{diag}\big(s(f) \, s(-f)\big)$, shaped like $f$.

        Args:
            f: Model outputs for a batch.

        Returns:
            The per-example output Hessian.
        """

        ft = cast(Tensor, f)

        return _DiagonalHessian(torch.sigmoid(ft) * torch.sigmoid(-ft))

    def default_link(self, n_samples: int | Literal["exact"], /) -> LinkApprox:
        """`Probit` for an exact output covariance (`n_samples=None`), else `MC` over all draws.

        Args:
            n_samples: The resolved `Linearized.n_samples`: `"exact"` for an exact output
                covariance, an int for a `SampleCov` with that many draws. An `MC` link must not
                ask for more.

        Returns:
            A link that `supports` this likelihood.
        """

        from laplacekit.predictive import MC, Probit  # predictive imports this module

        return Probit() if n_samples == "exact" else MC(n_samples=n_samples)

    def average(self, f: Tensor, /) -> Distribution:
        r"""The mean of $s(f)$ over the $S$ draws, as logits computed in log space.

        $$\log \frac{1}{S} \sum_{i=1}^S s(f_i) - \log \frac{1}{S} \sum_{i=1}^S s(-f_i),$$

        which stays finite when saturated.

        Args:
            f: Output draws shaped `[S, N, ...]`: `S` draws for `N` examples.

        Returns:
            A distribution over targets, one per example.
        """

        logits = torch.logsumexp(F.logsigmoid(f), 0) - torch.logsumexp(F.logsigmoid(-f), 0)

        return self.distribution(logits)


def _name(fn: Callable[..., object]) -> str:
    return getattr(fn, "__qualname__", type(fn).__name__)


@pytree
class LogProb:
    r"""A likelihood given by its log-probability alone, built by `from_log_prob`.

    It has no distribution, no sampling and no output Hessian, so `curvature.default` picks
    the empirical Fisher, and predictives over targets refuse it.

    Args:
        fn: `fn(f, y)` returns $\log p(y \mid f)$ (see `from_log_prob`).
        mask_fn: `mask_fn(f, inputs)` returns the mask (see `from_log_prob`), or None.
    """

    fn: Callable[[Outputs, Targets], Tensor] = static()
    mask_fn: Callable[[Outputs, Inputs], Tensor | None] | None = static(default=None)

    def __repr__(self) -> str:
        if self.mask_fn is None:
            return f"from_log_prob({_name(self.fn)})"

        return f"from_log_prob({_name(self.fn)}, mask_fn={_name(self.mask_fn)})"

    def log_prob(self, f: Outputs, y: Targets, /) -> Tensor:
        """`fn(f, y)`.

        Args:
            f: Model outputs for a batch.
            y: Targets for the same batch.

        Returns:
            A tensor shaped `[N]`, or `[N, *positions]` (per token, say).
        """

        return self.fn(f, y)

    def mask(self, f: Outputs, inputs: Inputs, /) -> Tensor | None:
        """`mask_fn(f, inputs)`, or None without `mask_fn`.

        Args:
            f: Model outputs for a batch.
            inputs: The batch's inputs.

        Returns:
            A bool tensor shaped like the output of `log_prob`, `[N, *positions]`, True
            where a position counts, or `None` when every position counts.
        """

        return None if self.mask_fn is None else self.mask_fn(f, inputs)


def from_log_prob(
    fn: Callable[[Outputs, Targets], Tensor],
    /,
    mask_fn: Callable[[Outputs, Inputs], Tensor | None] | None = None,
) -> LogProb:
    r"""Wrap a per-example log-likelihood `fn(f, y)` (contrastive, ranking, a custom energy).

    Losses that couple examples (pairwise or listwise across the batch) don't fit this
    contract.

    Args:
        fn: `fn(f, y)` returns $\log p(y \mid f)$ shaped `[N]` or `[N, *positions]`.
        mask_fn: `mask_fn(f, inputs)` returns a bool tensor shaped like `fn`'s output,
            False at positions that don't count (padding, say), or None when every position
            counts. Without it, every position counts.

    Returns:
        The likelihood, a `LogProb`. It has only `log_prob` and `mask`.
    """

    return LogProb(fn, mask_fn)


def _can_mask(lik: Likelihood, /) -> bool:
    """False for a likelihood whose mask is always None, known without a forward pass.

    That is: no `HasMask`, or the built-in `Categorical` or `LogProb` without `mask_fn`.
    Exactly those classes: a subclass may define its own `mask`.
    """

    if not isinstance(lik, HasMask):
        return False

    if type(lik) is Categorical or type(lik) is LogProb:
        return lik.mask_fn is not None

    return True
