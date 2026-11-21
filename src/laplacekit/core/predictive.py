"""Predictives.

A Predictive turns a posterior and inputs into a distribution over targets; a LinkApprox
turns Gaussian outputs into one.

`Predictive[B]` says which posteriors it accepts: the linearized predictive takes any
posterior, weight sampling only a WeightPosterior. For uncertainty over outputs
without observation noise, use `outputs`.

A likelihood outside the built-ins names its own default link (`HasDefaultLink`) and
how to average output draws (`HasSampleAverage`).
"""

from abc import abstractmethod
from typing import Literal, Protocol, runtime_checkable

import torch
from torch import Tensor
from torch.distributions import Distribution

from laplacekit.core.likelihoods import Likelihood
from laplacekit.core.posteriors import OutputGaussian
from laplacekit.core.types import Inputs, Requirements


@runtime_checkable
class Predictive[B](Protocol):
    """Turns a posterior `B` and inputs into a distribution over targets."""

    @property
    @abstractmethod
    def requires(self) -> Requirements:
        """What the predictive needs from the model and the likelihood."""
        ...

    @abstractmethod
    def __call__(
        self, posterior: B, inputs: Inputs, /, *, generator: torch.Generator | None = None
    ) -> Distribution:
        """The predictive distribution at `inputs`.

        Args:
            posterior: The posterior, of a type this predictive accepts.
            inputs: Inputs, as the model's `forward` takes them.
            generator: The generator for sampled predictives to draw with. `None` means
                torch's global RNG. `lk.predict` passes a fresh one on the inputs' device.

        Returns:
            A distribution over targets, one per example.
        """
        ...


@runtime_checkable
class LinkApprox(Protocol):
    """Turns Gaussian outputs into a distribution over targets.

    It works only for the likelihoods it `supports`.
    """

    @abstractmethod
    def supports(self, lik: Likelihood, f_mu: Tensor, /) -> bool:
        """Whether this link handles `lik` at these outputs.

        For a weight-space posterior, `Linearized` calls it after the forward pass and before the
        Jacobian work; for a function-space posterior, after `at`.

        Args:
            lik: The likelihood.
            f_mu: The output mean.

        Returns:
            True when `__call__` can turn outputs like `f_mu` into a distribution.
        """
        ...

    @abstractmethod
    def __call__(
        self,
        lik: Likelihood,
        out: OutputGaussian,
        /,
        *,
        generator: torch.Generator | None = None,
    ) -> Distribution:
        """The distribution over targets.

        Args:
            lik: A likelihood the link `supports`.
            out: Gaussian posterior over the outputs.
            generator: The generator for a sampled link (`MC`) to draw with. `None` means
                torch's global RNG. Links that draw nothing ignore it.

        Returns:
            A distribution over targets.
        """
        ...


@runtime_checkable
class HasDefaultLink(Protocol):
    """The link a likelihood uses when `Linearized` has `link=None`.

    `Categorical` and `Gaussian`, and their subclasses, keep their own links.
    """

    @abstractmethod
    def default_link(self, n_samples: int | Literal["exact"], /) -> LinkApprox:
        """The link to use for this likelihood.

        Args:
            n_samples: The resolved `Linearized.n_samples`: `"exact"` for an exact output
                covariance, an int for a `SampleCov` with that many draws. An `MC` link
                must not ask for more.

        Returns:
            A link that `supports` this likelihood.
        """
        ...


@runtime_checkable
class HasSampleAverage(Protocol):
    """The predictive from output draws.

    `WeightSamples` and the `MC` link call it. `Categorical` and `Gaussian`, and their
    subclasses, keep their own averages.
    """

    @abstractmethod
    def average(self, f: Tensor, /) -> Distribution:
        """Average output draws into a predictive distribution.

        Args:
            f: Output draws shaped `[S, N, ...]`: `S` draws for `N` examples.

        Returns:
            A distribution over targets, one per example.
        """
        ...
