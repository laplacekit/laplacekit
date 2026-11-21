"""Hyper tuners: tune a posterior's prior precision and learned noise without refitting.

`lk.laplace` runs `MarginalLikelihood()` after the fit by default. Pass another tuner as
`hyper_tuner=`, or `None` to skip the tune. `lk.tune(post, tuner)` (or `tuner(post)`)
tunes a posterior that is already fitted:

```python
post = lk.laplace(model, loader, lk.Categorical(), hyper_tuner=None)
post = lk.tune(post, lk.GridSearch(lk.objectives.predictive_nll(val_loader), values=lams))
```

A tuner changes the posterior's hyperparameters: the prior precision of a
`TunablePrior` (the built-in priors) and the likelihood's learned hyperparameters (the
`sigma` of `lk.Gaussian()`). Both are applied after the fit, so no tune refits.
"""

import copy
import math
from abc import abstractmethod
from collections.abc import Iterable, Mapping
from typing import Any, Protocol, cast, runtime_checkable

import torch
from torch import Tensor

from laplacekit import core
from laplacekit.core import (
    Batch,
    HyperTuner,
    Likelihood,
    Objective,
    Prior,
    StructureError,
    TunablePrior,
    pytree,
    static,
)

__all__ = ["GridSearch", "HyperTuner", "MarginalLikelihood", "Minimize", "tune"]


@runtime_checkable
class _HasLogMarglik(Protocol):
    @abstractmethod
    def log_marglik(self) -> Tensor: ...


def _neg_log_marglik(posterior: Any) -> Tensor:
    if not isinstance(posterior, _HasLogMarglik):
        raise TypeError(
            f"MarginalLikelihood needs a posterior with log_marglik(), got "
            f"{type(posterior).__name__}"
        )

    return -posterior.log_marglik()


def _start_sigma[P](what: str, posterior: P, sigma_init: float | None) -> P:
    """The posterior with its learned sigma set to `sigma_init`, when given."""

    if sigma_init is None:
        return posterior

    found = core.hyperparameters(posterior)

    if "sigma" not in found:
        raise ValueError(
            f"{what}(sigma_init=...) needs a learned sigma, but the likelihood has none; "
            "pass lk.Gaussian() to learn it"
        )

    return core.fill_leaves(posterior, {"sigma": found["sigma"]}, sigma_init)


def _check_sigma_init(what: str, sigma_init: float | None) -> None:
    if sigma_init is not None and not (math.isfinite(sigma_init) and sigma_init > 0):
        raise ValueError(f"{what} needs a positive finite sigma_init, got {sigma_init}")


def _record[P](posterior: P, tuner: object) -> P:
    """The posterior with its printed `hyper_tuner` row set to `tuner`."""

    if not isinstance(posterior, core.LaplacePosterior):
        return posterior

    row = ("hyper_tuner", repr(tuner))
    rows = [row if k == "hyper_tuner" else (k, v) for k, v in posterior.provenance]

    if row not in rows:
        rows.append(row)

    return cast(P, copy.replace(posterior, provenance=tuple(rows)))


def _tuned[P](what: str, posterior: P, leaves: Mapping[str, Tensor]) -> dict[str, Tensor]:
    """Check the leaves (fit-time ones raise) and refuse an empty set."""

    if not leaves:
        raise StructureError(
            f"{what}: {type(posterior).__name__} has no hyperparameters to tune (its prior "
            "is not a TunablePrior, and its likelihood learns nothing)"
        )

    core.check_fit_time(what, posterior, tuple(leaves), tuple(leaves.values()))

    return dict(leaves)


@pytree
class Minimize:
    """Minimize an objective over every hyperparameter, with L-BFGS on their logs.

    It tunes the prior precision and the likelihood's learned hyperparameters jointly, and
    keeps them positive. The objective needs a gradient with respect to them; use
    `GridSearch` for one without (such as `lk.objectives.ece`).

    ```python
    post = lk.tune(post, lk.Minimize(lk.objectives.predictive_nll(val_loader)))
    ```

    Args:
        objective: The loss over a posterior, lower is better.
        n_steps: Cap on the L-BFGS iterations. L-BFGS stops earlier once the gradient or
            the change in the logs is negligible.
        lr: Initial step of the strong-Wolfe line search each iteration runs.
        sigma_init: Where a learned sigma (`lk.Gaussian()`) starts, as laplace-torch's
            `sigma_noise_init`. `None` keeps its current value.
    """

    objective: Objective[Any] = static()
    n_steps: int = static(default=100)
    lr: float = static(default=1.0)
    sigma_init: float | None = static(default=None)

    def __post_init__(self) -> None:
        _check_sigma_init("Minimize", self.sigma_init)

    def __repr__(self) -> str:
        name = getattr(self.objective, "__qualname__", type(self.objective).__name__)

        return f"Minimize({name.split('.<locals>')[0]})"

    def check(self, prior: Prior, likelihood: Likelihood, batch: Batch, /) -> None:
        """Check the prior's `tunable()` before the fit.

        Args:
            prior: The prior of the fit.
            likelihood: The likelihood of the fit.
            batch: The first batch of the data.

        Raises:
            StructureError: The prior's `tunable()` breaks its rules, or there is nothing
                to tune: the prior is not a `TunablePrior` and the likelihood learns
                nothing.
        """

        core.check_tunable(prior)

        if not isinstance(prior, TunablePrior) and not core.likelihood_hyperparameters(likelihood):
            raise StructureError(
                f"Minimize has nothing to tune: {type(prior).__name__} is not a TunablePrior, "
                f"and {type(likelihood).__name__} learns nothing; pass hyper_tuner=None"
            )

    def __call__[P](self, posterior: P, /) -> P:
        """Tune the posterior's hyperparameters.

        Args:
            posterior: The posterior. It is left unchanged.

        Returns:
            A new posterior with the hyperparameters that minimize the objective.

        Raises:
            StructureError: The posterior has no hyperparameters, or one is applied at fit
                time (a fixed `sigma`).
            ValueError: The objective has no gradient with respect to the hyperparameters.
        """

        posterior = _start_sigma("Minimize", core.detach_leaves(posterior), self.sigma_init)
        leaves = _tuned("Minimize", posterior, core.hyperparameters(posterior))

        tuned = core.minimize_on_logs(
            posterior, self.objective, leaves, n_steps=self.n_steps, lr=self.lr
        )

        return _record(tuned, self)


@pytree
class MarginalLikelihood:
    r"""Maximize the Laplace marginal likelihood of the fit data. The default tuner.

    It tunes the prior precision and the likelihood's learned hyperparameters (the `sigma`
    of `lk.Gaussian()`) jointly, with L-BFGS on their logs, as laplace-torch's
    `optimize_prior_precision(method="marglik")`. The marginal likelihood needs the data
    fit term, so the data must have targets.

    Args:
        n_steps: Cap on the L-BFGS iterations.
        lr: Initial step of the strong-Wolfe line search each iteration runs.
        sigma_init: Where a learned sigma (`lk.Gaussian()`) starts, as laplace-torch's
            `sigma_noise_init`. `None` keeps its current value (1 after `lk.laplace`).
    """

    n_steps: int = static(default=100)
    lr: float = static(default=1.0)
    sigma_init: float | None = static(default=None)

    def __post_init__(self) -> None:
        _check_sigma_init("MarginalLikelihood", self.sigma_init)

    def check(self, prior: Prior, likelihood: Likelihood, batch: Batch, /) -> None:
        """Check that the data has targets and the prior's `tunable()` keeps its rules.

        Args:
            prior: The prior of the fit.
            likelihood: The likelihood of the fit.
            batch: The first batch of the data.

        Raises:
            ValueError: The batch has no targets, so there is no marginal likelihood.
            StructureError: The prior's `tunable()` breaks its rules.
        """

        if batch.targets is None:
            raise ValueError(
                "MarginalLikelihood needs targets for the data-fit term, but the data has "
                "none; pass hyper_tuner=None, or a tuner on held-out data (lk.GridSearch)"
            )

        core.check_tunable(prior)

    def __call__[P](self, posterior: P, /) -> P:
        """Tune the posterior's hyperparameters.

        Args:
            posterior: The posterior. It is left unchanged.

        Returns:
            A new posterior at the maximum of the marginal likelihood.

        Raises:
            StructureError: The posterior has no hyperparameters, or one is applied at fit
                time (a fixed `sigma`).
            ValueError: The posterior has no data-fit term (a fit without targets).
        """

        posterior = core.detach_leaves(posterior)
        posterior = _start_sigma("MarginalLikelihood", posterior, self.sigma_init)
        leaves = _tuned("MarginalLikelihood", posterior, core.hyperparameters(posterior))

        tuned = core.minimize_on_logs(
            posterior,
            _neg_log_marglik,
            leaves,
            n_steps=self.n_steps,
            lr=self.lr,
            what="MarginalLikelihood",
        )

        return _record(tuned, self)


@pytree
class GridSearch:
    """Set the prior precision to the value from a grid that minimizes an objective.

    Every entry of the prior precision (each block of `Layerwise`, each weight of
    `Diagonal`) takes the same value. Other hyperparameters, such as a learned `sigma`,
    stay as they are. The objective runs under `torch.no_grad`, so it needs no gradient.

    ```python
    nll = lk.objectives.predictive_nll(val_loader)
    post = lk.tune(post, lk.GridSearch(nll, values=torch.logspace(-2, 3, 21)))
    ```

    Args:
        objective: The loss over a posterior, lower is better.
        values: The candidate precisions.
    """

    objective: Objective[Any] = static()
    values: tuple[float, ...] = static()

    def __init__(self, objective: Objective[Any], values: Iterable[float | Tensor]) -> None:
        grid = tuple(float(v) for v in values)

        if not grid:
            raise ValueError("GridSearch needs at least one value")

        object.__setattr__(self, "objective", objective)
        object.__setattr__(self, "values", grid)

    def __repr__(self) -> str:
        name = getattr(self.objective, "__qualname__", type(self.objective).__name__)

        return f"GridSearch({name.split('.<locals>')[0]}, {len(self.values)} values)"

    def check(self, prior: Prior, likelihood: Likelihood, batch: Batch, /) -> None:
        """Check that the prior has a precision to tune.

        Args:
            prior: The prior of the fit.
            likelihood: The likelihood of the fit.
            batch: The first batch of the data.

        Raises:
            StructureError: The prior is not a `TunablePrior`, or its `tunable()` breaks
                its rules.
        """

        if not isinstance(prior, TunablePrior):
            raise StructureError(
                f"GridSearch tunes the prior precision, but {type(prior).__name__} is not a "
                "TunablePrior; pass hyper_tuner=None"
            )

        core.check_tunable(prior)

    def __call__[P](self, posterior: P, /) -> P:
        """Tune the posterior's prior precision.

        Args:
            posterior: The posterior. It is left unchanged.

        Returns:
            A new posterior whose prior precision is the best value.

        Raises:
            StructureError: The posterior has no prior precision to tune.
        """

        posterior = core.detach_leaves(posterior)
        prior = posterior.prior if isinstance(posterior, core.HasPrior) else None
        leaves = _tuned("GridSearch", posterior, core.prior_hyperparameters(prior))
        best: tuple[Tensor, float] | None = None

        with torch.no_grad():
            for v in self.values:
                score = self.objective(core.fill_leaves(posterior, leaves, v))

                if best is None or bool(score < best[0]):
                    best = (score, v)

        assert best is not None

        return _record(core.fill_leaves(posterior, leaves, best[1]), self)


def tune[P](posterior: P, hyper_tuner: HyperTuner = MarginalLikelihood()) -> P:  # noqa: B008
    """Tune a fitted posterior's hyperparameters, without refitting.

    ```python
    post = lk.tune(post)  # the marginal likelihood
    post = lk.tune(post, lk.GridSearch(nll, values=torch.logspace(-2, 3, 21)))
    ```

    Args:
        posterior: The posterior. It is left unchanged.
        hyper_tuner: The tuner.

    Returns:
        The tuned posterior, `hyper_tuner(posterior)`.

    Raises:
        TypeError: `hyper_tuner` is not a `HyperTuner` (a string, a class, or `None`).
    """

    if not isinstance(hyper_tuner, HyperTuner) or isinstance(hyper_tuner, type):
        raise TypeError(_tuner_hint(hyper_tuner))

    return _record(hyper_tuner(posterior), hyper_tuner)


def _tuner_hint(got: object) -> str:
    if isinstance(got, type):
        return (
            f"hyper_tuner takes an instance, got the class {got.__name__}; pass "
            f"{got.__name__}(...), such as lk.MarginalLikelihood()"
        )

    return (
        f"hyper_tuner takes a HyperTuner, got {type(got).__name__}; pass "
        "lk.MarginalLikelihood(), lk.GridSearch(objective, values) or lk.Minimize(objective)"
    )
