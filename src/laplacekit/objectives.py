"""Built-in objectives: marginal likelihood, predictive NLL, Brier score and ECE.

An objective is a function of a posterior to a 0-dim tensor, lower is better. The hyper
tuners take them: `lk.GridSearch(lk.objectives.predictive_nll(val_loader), values=...)`.
"""

from collections.abc import Iterable
from typing import Any, cast

import torch
from torch import Tensor
from torch.distributions import Bernoulli, Categorical, Distribution, Independent

from laplacekit.core import (
    Batch,
    CapabilityError,
    FunctionPosterior,
    HasMask,
    Inputs,
    LaplacePosterior,
    Objective,
    Posterior,
    Predictive,
    StructureError,
    WeightPosterior,
)
from laplacekit.data import Data, as_data
from laplacekit.likelihoods import _can_mask
from laplacekit.predictive import Linearized

__all__ = ["brier", "ece", "neg_log_marglik", "predictive_nll"]


def neg_log_marglik(posterior: LaplacePosterior[Any, Any], /) -> Tensor:
    r"""Laplace negative log marginal likelihood of the fit data, `-post.log_marglik()`.

    For a posterior $\mathcal{N}(\mu, \Lambda^{-1})$ with prior precision $\Lambda_0$ and
    prior location $\mu_0$, it is

    $$
    -\log p(\mathcal{D} \mid \mu)
        + \tfrac{1}{2} \left( \log \det \Lambda - \log \det \Lambda_0 \right)
        + \tfrac{1}{2} (\mu - \mu_0)^\top \Lambda_0 (\mu - \mu_0).
    $$

    `lk.MarginalLikelihood()` minimizes it; `lk.Minimize(neg_log_marglik)` is the same.

    Args:
        posterior: A posterior fitted on data with targets.

    Returns:
        A 0-dim tensor.

    Raises:
        ValueError: The posterior has no data-fit term (a fit without targets).
        StructureError: As `LaplacePosterior.log_marglik`.
    """

    return -posterior.log_marglik()


def _batches(data: object, what: str, /) -> list[Batch]:
    """The batches of `data`, any form `lk.laplace` takes, read once."""

    src = as_data(cast(Any, data), what=f"{what}'s data")

    return list(src(torch.Generator().manual_seed(0)))


def predictive_nll(
    data: Data | Iterable[Any], /, predictive: Predictive[Posterior] | None = None
) -> Objective[Posterior]:
    r"""Mean negative log predictive density of the batches' targets, per example.

    With $N$ the examples over all batches,

    $$
    \frac{1}{N} \sum_{n=1}^{N} -\log p(y_n \mid x_n, \mathcal{D}),
    $$

    where $p(\cdot \mid x_n, \mathcal{D})$ is the predictive distribution.

    With a single `HasMask` likelihood the masked positions drop out, as in `core.log_prob`
    with inputs. The mask reads the outputs at the mean, which costs one extra model call
    per batch for a weight-space posterior and one extra `at(inputs)` for a function posterior. The
    built-in `Categorical` and `from_log_prob` without `mask_fn` skip it. The mask covers
    the leading axes of the log-probabilities of the distribution under any `Independent`
    wrappers.

    Args:
        data: Data with targets, in any form `lk.laplace` takes: a list or `DataLoader`
            of `(inputs, targets)` pairs, or a `Data` source. It is read once, into a list. Targets
            must be shaped like the predictive distribution's batch and event shapes together.
        predictive: The predictive method. `None` means `Linearized()`, resolved against
            the posterior.

    Returns:
        The objective, a function of a posterior to a 0-dim tensor. When called, it raises
        `ValueError` for a batch without targets, and `StructureError` for targets of
        another shape or a mask that is not bool or doesn't cover the leading axes. It
        also raises what the predictive raises.
    """

    data = _batches(data, "predictive_nll")
    pred: Predictive[Posterior] = Linearized() if predictive is None else predictive

    def objective(posterior: Posterior) -> torch.Tensor:
        total = torch.zeros(())
        n = 0

        for b in data:
            if b.targets is None:
                raise ValueError("predictive_nll needs batches with targets")

            d = pred(posterior, b.inputs)
            y = cast(torch.Tensor, b.targets)
            want = d.batch_shape + d.event_shape

            if y.shape != want:
                raise StructureError(
                    f"predictive_nll needs targets shaped {tuple(want)} (the predictive's "
                    f"batch and event shapes), got {tuple(y.shape)}"
                )

            mask = _mask(posterior, b.inputs)

            if mask is None:
                total = total - d.log_prob(y).sum()
            else:
                total = total - _masked_log_prob(d, y, mask).sum()

            n += b.size

        return total / n

    return objective


def _mask(posterior: Posterior, inputs: Inputs) -> Tensor | None:
    """The likelihood's mask at the outputs at the mean, or None without one."""

    if not isinstance(posterior, WeightPosterior | FunctionPosterior):
        return None

    lik = posterior.likelihood

    if not isinstance(lik, HasMask) or not _can_mask(lik):
        return None

    with torch.no_grad():
        if isinstance(posterior, FunctionPosterior):
            f = posterior.at(inputs).mean
        else:
            f = posterior.model(posterior.mean, inputs)

    mask = lik.mask(f, inputs)

    if mask is not None and mask.dtype != torch.bool:
        raise StructureError(
            f"{type(lik).__name__}.mask returned a {mask.dtype} tensor; the mask must be bool"
        )

    return mask


def _masked_log_prob(d: Distribution, y: Tensor, mask: Tensor) -> Tensor:
    """The log-probability under the `Independent` wrappers, zero where `mask` is False.

    `mask` covers the leading axes.
    """

    while isinstance(d, Independent):
        d = d.base_dist

    lp = d.log_prob(y)

    if mask.ndim > lp.ndim or mask.shape != lp.shape[: mask.ndim]:
        raise StructureError(
            f"predictive_nll: the likelihood's mask is shaped {tuple(mask.shape)}, but it must "
            f"cover the leading axes of the log-probabilities {tuple(lp.shape)}"
        )

    mask = mask.reshape(*mask.shape, *([1] * (lp.ndim - mask.ndim))).to(lp.device)

    return torch.where(mask, lp, 0)


def _probs(name: str, d: Distribution) -> tuple[Tensor, bool]:
    """The probabilities under any `Independent` wrappers, and whether it is a Categorical.

    Only a Categorical or Bernoulli has them; other distributions raise `CapabilityError`.
    """

    while isinstance(d, Independent):
        d = d.base_dist

    if not isinstance(d, (Categorical, Bernoulli)):
        raise CapabilityError(
            f"{name} needs a Categorical or Bernoulli predictive, got {type(d).__name__}"
        )

    return cast(Tensor, d.probs), isinstance(d, Categorical)  # a lazy_property in the stubs


def _targets(name: str, b: Batch) -> Tensor:
    if b.targets is None:
        raise ValueError(f"{name} needs batches with targets")

    return cast(Tensor, b.targets)


def _check_targets(name: str, p: Tensor, classes: bool, y: Tensor) -> None:
    """Check targets against the probabilities `p`.

    `StructureError` unless `y` is shaped like the probabilities (without the class axis for
    a Categorical); `ValueError` for class indices that are not integers in range, or
    Bernoulli targets outside [0, 1].
    """

    if classes:
        if y.shape != p.shape[:-1]:
            raise StructureError(
                f"{name} needs class-index targets shaped {tuple(p.shape[:-1])} (the "
                f"probabilities {tuple(p.shape)} without the class axis), got {tuple(y.shape)}"
            )

        if y.is_floating_point() or y.is_complex() or y.dtype == torch.bool:
            raise ValueError(f"{name} needs integer class indices, got {y.dtype}")

        k = p.shape[-1]

        if y.numel() and not bool(((y >= 0) & (y < k)).all()):
            raise ValueError(
                f"{name} needs class indices in [0, {k}), got values in "
                f"[{int(y.min())}, {int(y.max())}]"
            )

        return

    if y.shape != p.shape:
        raise StructureError(
            f"{name} needs Bernoulli targets shaped like the probabilities "
            f"{tuple(p.shape)}, got {tuple(y.shape)}"
        )

    t = y.to(torch.float64)

    if t.numel() and not bool(((t >= 0) & (t <= 1)).all()):
        raise ValueError(
            f"{name} needs Bernoulli targets in [0, 1], got values in "
            f"[{float(t.min())}, {float(t.max())}]"
        )


def brier(
    data: Data | Iterable[Any], /, predictive: Predictive[Posterior] | None = None
) -> Objective[Posterior]:
    r"""Brier score: the squared error between predictive probabilities and targets, per example.

    With $N$ the examples over all batches,

    $$
    \frac{1}{N} \sum_{n=1}^{N} \sum_{j} (p_{nj} - t_{nj})^2,
    $$

    where $j$ runs over classes (against one-hot class indices $t$) for a `Categorical`
    predictive, or over outputs for a `Bernoulli` one, and over positions. A two-class
    `Categorical` therefore gives twice the Brier score of the matching single-output
    `Bernoulli`. Every position counts: a likelihood's mask is not applied.

    Args:
        data: Data with targets, in any form `lk.laplace` takes: a list or `DataLoader`
            of `(inputs, targets)` pairs, or a `Data` source. It is read once, into a list. Targets
            are class indices shaped like the probabilities without the class axis (`Categorical`),
            or values in $[0, 1]$ shaped like the probabilities (`Bernoulli`).
        predictive: The predictive method. `None` means `Linearized()`, resolved against
            the posterior.

    Returns:
        The objective, a function of a posterior to a 0-dim tensor. When called, it raises
        `CapabilityError` for a predictive distribution other than a `Categorical` or a
        `Bernoulli` (under any `Independent` wrappers), `StructureError` for targets of
        another shape, and `ValueError` for a batch without targets, non-integer or
        out-of-range class indices, or Bernoulli targets outside $[0, 1]$.
    """

    data = _batches(data, "brier")
    pred: Predictive[Posterior] = Linearized() if predictive is None else predictive

    def objective(posterior: Posterior) -> torch.Tensor:
        total = torch.zeros(())
        n = 0

        for b in data:
            y = _targets("brier", b)
            p, classes = _probs("brier", pred(posterior, b.inputs))
            _check_targets("brier", p, classes, y)

            if classes:
                target = torch.nn.functional.one_hot(y.long(), p.shape[-1]).to(p)
            else:
                target = y.to(p)

            total = total + ((p - target) ** 2).sum()
            n += b.size

        return total / n

    return objective


def ece(
    data: Data | Iterable[Any],
    /,
    predictive: Predictive[Posterior] | None = None,
    n_bins: int = 15,
) -> Objective[Posterior]:
    r"""Expected calibration error with equal-width bins, positions pooled.

    A `Categorical` predictive compares the top-label confidence with accuracy; a
    `Bernoulli` one compares $p(y = 1)$ with the frequency of $y = 1$ over all entries. With
    $M$ entries and $S_b$ the entries in bin $b$,

    $$
    \mathrm{ECE} = \sum_{b} \frac{|S_b|}{M} \left| \operatorname{acc}(S_b)
        - \operatorname{conf}(S_b) \right|.
    $$

    Binning follows `torchmetrics`: a confidence of exactly 1.0 forms its own bin. Every
    position counts: a likelihood's mask is not applied. The value is detached, so tune it
    with `lk.GridSearch`, not `lk.Minimize`.

    Args:
        data: Data with targets, in any form `lk.laplace` takes, checked as in `brier`.
            It is read once, into a
            list.
        predictive: The predictive method. `None` means `Linearized()`, resolved against
            the posterior.
        n_bins: The number of equal-width bins on $[0, 1]$, at least 1.

    Returns:
        The objective, a function of a posterior to a detached 0-dim tensor. When called, it
        raises as `brier` does, and `ValueError` when there are no batches.

    Raises:
        ValueError: `n_bins < 1`.
    """

    if n_bins < 1:
        raise ValueError(f"ece needs n_bins >= 1, got {n_bins}")

    data = _batches(data, "ece")
    pred: Predictive[Posterior] = Linearized() if predictive is None else predictive

    def objective(posterior: Posterior) -> torch.Tensor:
        sums: Tensor | None = None

        with torch.no_grad():
            for b in data:
                y = _targets("ece", b)
                p, classes = _probs("ece", pred(posterior, b.inputs))
                _check_targets("ece", p, classes, y)

                if classes:
                    conf, label = p.max(-1)
                    hit = (label == y).to(conf)
                else:
                    conf, hit = p, y.to(p)

                conf, hit = conf.reshape(-1).contiguous(), hit.reshape(-1).contiguous()
                edges = torch.linspace(0, 1, n_bins + 1, dtype=conf.dtype, device=conf.device)
                idx = torch.bucketize(conf, edges, right=True) - 1

                # Per bin: count, summed confidence, summed accuracy. Bin n_bins holds 1.0.
                part = torch.zeros(3, n_bins + 1, dtype=conf.dtype, device=conf.device)
                part[0].scatter_add_(0, idx, torch.ones_like(conf))
                part[1].scatter_add_(0, idx, conf)
                part[2].scatter_add_(0, idx, hit)

                sums = part if sums is None else sums + part

            if sums is None:
                raise ValueError("ece needs at least one batch")

            # Within a bin, |acc - conf| * count / total = |sum acc - sum conf| / total.
            return (sums[2] - sums[1]).abs().sum() / sums[0].sum()

    return objective
