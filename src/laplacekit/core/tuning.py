"""Hyperparameter tuning: the `HyperTuner` protocol and the shared machinery.

A posterior's hyperparameters are the prior's precision (`TunablePrior.tunable`) and the
likelihood's learned hyperparameters (`TunableLikelihood.tunable`, such as a learned
`sigma`). `hyperparameters` finds them, and tuners such as `lk.MarginalLikelihood` change
them without refitting. The helpers `detach_leaves`, `replace_leaves` and
`minimize_on_logs` are public so other code (`laplacekit.testing`, plugins) evaluates and
tunes the same way.
"""

from abc import abstractmethod
from collections.abc import Callable, Mapping
from typing import Any, Protocol, cast, runtime_checkable

import torch
from torch import Tensor

from laplacekit.core._pytree import tree_leaves, tree_leaves_with_path, tree_map
from laplacekit.core.likelihoods import Likelihood, LikelihoodTree, unit
from laplacekit.core.posteriors import Prior, TunablePrior
from laplacekit.core.types import Batch, StructureError

type Objective[P] = Callable[[P], Tensor]
"""A loss over a posterior, lower is better, such as `lk.objectives.predictive_nll(data)`."""


@runtime_checkable
class TunableLikelihood(Protocol):
    """A likelihood with hyperparameters a tuner may change, such as a learned `sigma`."""

    @abstractmethod
    def tunable(self) -> Mapping[str, Tensor]:
        """The hyperparameters to tune, by name.

        Returns:
            A mapping of positive tensors, each a leaf of the likelihood itself (a
            `@pytree`), returned as it is. Empty when nothing is learned.
        """
        ...


@runtime_checkable
class HasPrior(Protocol):
    """A posterior that keeps its prior, so a tuner can reach the prior precision."""

    @property
    @abstractmethod
    def prior(self) -> object:
        """The prior of the fit."""
        ...


@runtime_checkable
class HasLikelihood(Protocol):
    """A posterior that keeps its likelihood, so a tuner can reach its hyperparameters."""

    @property
    @abstractmethod
    def likelihood(self) -> LikelihoodTree:
        """The likelihood of the fit."""
        ...


@runtime_checkable
class HyperTuner(Protocol):
    """Tunes the hyperparameters of a posterior: its prior precision, a learned `sigma`.

    `lk.laplace(..., hyper_tuner=...)` runs it after the fit, and `lk.tune` on any
    posterior. A tuner returns a new posterior and leaves its input unchanged.
    """

    @abstractmethod
    def check(self, prior: Prior, likelihood: Likelihood, batch: Batch, /) -> None:
        """Raise before the fit when this tuner can't tune what the fit will give.

        `lk.laplace` calls it once, with the first batch, before it reads the others.

        Args:
            prior: The prior of the fit.
            likelihood: The likelihood of the fit.
            batch: The first batch of the data.

        Raises:
            StructureError: The prior or the likelihood has nothing this tuner can tune.
            ValueError: The data can't support the tune (no targets, say).
        """
        ...

    @abstractmethod
    def __call__[P](self, posterior: P, /) -> P:
        """Tune the posterior's hyperparameters.

        Args:
            posterior: The posterior to tune. It is left unchanged.

        Returns:
            A new posterior with tuned hyperparameters.
        """
        ...


def prior_hyperparameters(prior: object, /) -> dict[str, Tensor]:
    """The prior's tunable precisions, named for messages.

    Args:
        prior: Any prior.

    Returns:
        `{"prior_precision": t}`, or one `"prior_precision['block']"` entry per block, for
        a `TunablePrior`; empty for any other prior.
    """

    if not isinstance(prior, TunablePrior):
        return {}

    sel = prior.tunable()

    if isinstance(sel, Mapping):
        return {f"prior_precision[{k!r}]": v for k, v in sel.items()}

    return {"prior_precision": sel}


def likelihood_hyperparameters(likelihood: LikelihoodTree, /) -> dict[str, Tensor]:
    """The likelihood's learned hyperparameters, named for messages.

    Args:
        likelihood: A likelihood, or a mapping of them.

    Returns:
        Each `TunableLikelihood.tunable()` entry, prefixed with its key in a mapping.
    """

    if isinstance(likelihood, Mapping):
        return {
            f"{k}.{name}": t
            for k, sub in likelihood.items()
            for name, t in likelihood_hyperparameters(sub).items()
        }

    if isinstance(likelihood, TunableLikelihood):
        return dict(likelihood.tunable())

    return {}


def hyperparameters(posterior: object, /) -> dict[str, Tensor]:
    """Every hyperparameter of a posterior a tuner may change.

    Args:
        posterior: Any posterior.

    Returns:
        The prior's (`prior_hyperparameters`) and then the likelihood's
        (`likelihood_hyperparameters`), each a leaf of `posterior`.
    """

    out: dict[str, Tensor] = {}

    if isinstance(posterior, HasPrior):
        out.update(prior_hyperparameters(posterior.prior))

    if isinstance(posterior, HasLikelihood):
        out.update(likelihood_hyperparameters(posterior.likelihood))

    return out


def check_tunable(prior: Prior, /) -> None:
    """Check the rules of `prior.tunable()`: positive tensors that are leaves of the prior.

    Args:
        prior: The prior. Anything but a `TunablePrior` passes.

    Raises:
        StructureError: `tunable()` returns something other than tensors, a tensor that
            is not a leaf of the prior (a copy?), or one with non-positive entries.
    """

    if not isinstance(prior, TunablePrior):
        return

    name = type(prior).__name__
    sel = prior.tunable()
    picked = tuple(sel.values()) if isinstance(sel, Mapping) else (sel,)
    leaves = tree_leaves(prior)

    for t in picked:
        if not isinstance(t, Tensor):
            raise StructureError(
                f"{name}.tunable() must return a tensor or a mapping of tensors, "
                f"got {type(t).__name__}"
            )

        if not any(t is leaf for leaf in leaves):
            raise StructureError(
                f"{name}.tunable() returned a tensor that is not a leaf of {name} (a copy?); "
                "return the prior's own tensors, or pass hyper_tuner=None"
            )

        if not bool((t > 0).all()):
            raise StructureError(
                f"{name}.tunable() must return positive tensors, since the tune works on their logs"
            )


def check_fit_time(
    what: str, posterior: object, names: tuple[str, ...], leaves: tuple[Tensor, ...]
) -> None:
    """Raise `StructureError` when a hyperparameter is applied at fit time.

    Such a leaf is a leaf of the likelihood's `unit()` (the likelihood itself without late
    hyperparameters), so the curvature and the fit's statistic hold its old value.

    Args:
        what: The tuner, for the message.
        posterior: The posterior.
        names: The hyperparameters' names.
        leaves: The hyperparameters, leaves of `posterior`.

    Raises:
        StructureError: A leaf is shared with the likelihood's `unit()`.
    """

    if not isinstance(posterior, HasLikelihood):
        return

    def walk(lik: LikelihoodTree, where: str) -> None:
        if isinstance(lik, Mapping):
            for k, sub in lik.items():
                walk(sub, f"{where}[{k!r}]")

            return

        fit_time = tree_leaves(unit(lik))

        for path, leaf in tree_leaves_with_path(lik):
            if not any(leaf is u for u in fit_time):
                continue

            for selected, t in zip(names, leaves, strict=True):
                if t is not leaf:
                    continue

                raise StructureError(
                    f"{what} can't change {selected} ({where}{path}): {type(lik).__name__} "
                    "applies it at fit time, so the curvature holds the old value; refit with "
                    "the new value, or make the hyperparameter late (leave it out of unit(), "
                    "as a LateLikelihood whose unit() is another object)"
                )

    walk(posterior.likelihood, "likelihood")


def detach_leaves[P](tree: P) -> P:
    """Detach every tensor leaf of a pytree.

    Args:
        tree: A posterior, or any pytree.

    Returns:
        The tree with every tensor leaf detached, so changes to it never reach the
        caller's autograd graph.
    """

    def detach(t: Any) -> Any:
        return t.detach() if isinstance(t, Tensor) else t

    return cast(P, tree_map(detach, tree))


def _cast_like(new: Tensor, like: Tensor) -> Tensor:
    """`new`, detached and copied, in the dtype and on the device of `like`."""

    return new.detach().to(dtype=like.dtype, device=like.device, copy=True)


def replace_leaves[P](tree: P, old: tuple[Tensor, ...], new: tuple[Tensor, ...]) -> P:
    """Swap leaves of a pytree by identity.

    Args:
        tree: A posterior, or any pytree.
        old: Leaves of `tree` to replace.
        new: Their replacements, in the same order.

    Returns:
        The tree with each leaf that is `old[i]` replaced by `new[i]`.
    """

    def swap(leaf: Any) -> Any:
        for o, n in zip(old, new, strict=True):
            if leaf is o:
                return n

        return leaf

    return cast(P, tree_map(swap, tree))


def minimize_on_logs[P](
    posterior: P,
    objective: Objective[P],
    leaves: Mapping[str, Tensor],
    /,
    *,
    n_steps: int = 100,
    lr: float = 1.0,
    what: str = "Minimize",
) -> P:
    """Minimize `objective` over positive leaves of `posterior`, with L-BFGS on their logs.

    The logs and the optimizer state are float64. The objective sees 0-dim leaves in
    float64 and larger ones in their own dtype. Terms that mix a 0-dim leaf with a scalar
    statistic, like the data fit of the marginal likelihood, then compute in float64, so
    a float32 posterior whose objective is in the millions still resolves small steps. A
    larger leaf in float64 would turn its operator float64 and break products with the
    float32 posterior.

    Args:
        posterior: The posterior, already detached (`detach_leaves`).
        objective: The loss to minimize. It needs a gradient with respect to the leaves.
        leaves: The leaves to tune, by name, each a positive leaf of `posterior`.
        n_steps: Cap on the L-BFGS iterations. L-BFGS stops earlier once the gradient or
            the change in the logs is negligible.
        lr: Initial step of the strong-Wolfe line search each iteration runs.
        what: The tuner, for messages.

    Returns:
        A new posterior. The tuned leaves come back in their original dtype and device.

    Raises:
        ValueError: A leaf has non-positive entries, or the objective has no gradient
            with respect to the leaves (a detached one, such as ece).
    """

    names, old = tuple(leaves), tuple(leaves.values())

    for name, t in leaves.items():
        if not bool((t > 0).all()):
            raise ValueError(f"{what} needs positive hyperparameters, but {name} is not")

    logs = [t.to(torch.float64).log().requires_grad_() for t in old]
    # Hyperparameter objectives are smooth and low-dimensional; a first-order method with a
    # fixed step stalls far from the optimum when started a few decades away.
    opt = torch.optim.LBFGS(
        logs,
        lr=lr,
        max_iter=n_steps,
        tolerance_grad=1e-9,
        tolerance_change=1e-12,
        line_search_fn="strong_wolfe",
    )

    def closure() -> Tensor:
        opt.zero_grad()
        # A 0-dim float64 leaf leaves the operator's dtype alone under torch's promotion; a
        # dimensioned one does not.
        current = tuple(
            x.exp() if t.ndim == 0 else x.exp().to(t.dtype) for x, t in zip(logs, old, strict=True)
        )
        loss = objective(replace_leaves(posterior, old, current))

        if not loss.requires_grad:
            raise ValueError(
                f"the objective has no gradient with respect to {', '.join(names)}; a "
                "detached objective (such as ece) goes with lk.GridSearch"
            )

        grads = torch.autograd.grad(loss, logs)

        for x, g in zip(logs, grads, strict=True):
            x.grad = g

        return loss.detach()

    opt.step(closure)

    tuned = tuple(_cast_like(x.exp(), t) for x, t in zip(logs, old, strict=True))

    return replace_leaves(posterior, old, tuned)


def fill_leaves[P](posterior: P, leaves: Mapping[str, Tensor], value: Tensor | float, /) -> P:
    """Set every entry of each leaf to `value`, keeping its dtype, device and shape.

    Args:
        posterior: The posterior.
        leaves: Leaves of `posterior`.
        value: The new value.

    Returns:
        A new posterior.
    """

    old = tuple(leaves.values())
    new = tuple(
        torch.full_like(t, float(value) if not isinstance(value, Tensor) else value.item())
        for t in old
    )

    return replace_leaves(posterior, old, new)
