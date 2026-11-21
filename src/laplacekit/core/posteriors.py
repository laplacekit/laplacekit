"""Posteriors over a model's weights or over its outputs.

"Prior" and "posterior" are roles a posterior plays relative to some data, not kinds of
thing: today's posterior is tomorrow's prior. What differs is the space a posterior lives in.

- A weight-space posterior (`Prior`) can be the prior of a Laplace update. A
  fitted one (`WeightPosterior`) also has a mean and can be sampled.
- A function-space posterior (`FunctionPosterior`) gives Gaussian moments of the model's
  outputs at given inputs. It has no weights to sample.
"""

import copy
import dataclasses
import math
from abc import abstractmethod
from collections.abc import Mapping
from functools import partial
from typing import Any, Generic, Literal, Protocol, Self, TypeVar, cast, runtime_checkable

import torch
from torch import Tensor
from torch.func import jvp, vjp, vmap

from laplacekit.core._pytree import as_pytree, pytree, static, tree_map
from laplacekit.core._seed import _Seed
from laplacekit.core.curvature import stat_dtype
from laplacekit.core.layout import Layout
from laplacekit.core.likelihoods import MAX_DENSE_EVENT, Likelihood, curvature_scale, log_lik
from laplacekit.core.model import ModelFn
from laplacekit.core.ops import BlockDiag, HasBlocks, Joint, Operator
from laplacekit.core.types import (
    Capabilities,
    Inputs,
    Outputs,
    Params,
    Requirements,
    StructureError,
)

type Strategy = Literal["forward-mode", "reverse-mode"]


def _home(generator: torch.Generator | None) -> torch.device:
    """Where `generator` draws: its own device, or the CPU for torch's global RNG."""

    return generator.device if generator is not None else torch.device("cpu")


# --- output-space Gaussians ----------------------------------------------------------


class OutputCov(Protocol):
    r"""Covariance of model outputs for a batch of inputs, never formed as an $NE \times NE$ matrix.

    Shapes follow the mean, for example `[N, C]` for $N$ inputs, and $E$ is the flattened
    event size. `per_input()` holds no cross-input
    blocks; `sample` may draw jointly across the batch (`SampleCov`, `FactorCov`: one
    weight draw per sample).
    """

    @abstractmethod
    def diagonal(self) -> Tensor:
        """The marginal variances.

        Returns:
            A tensor shaped like the mean.
        """
        ...

    @abstractmethod
    def sample(self, n_samples: int, /, *, generator: torch.Generator | None = None) -> Tensor:
        """Zero-mean draws from the covariance.

        Args:
            n_samples: Number of draws.
            generator: The generator to draw with, on any device: the draws move
                to the covariance's. `None` means torch's global RNG.

        Returns:
            The draws, `[n_samples, *mean.shape]`.

        Raises:
            ValueError: Implementations may raise it when they can't give `n_samples` draws, as
                `SampleCov` does.
        """
        ...

    @abstractmethod
    def per_input(self) -> Tensor:
        """The covariance block of each input, without the cross-input blocks.

        Returns:
            The blocks, `[N, E, E]`.

        Raises:
            StructureError: Implementations may raise it when $E$ is too large for dense
                blocks, as `SampleCov` and `FactorCov` do above `MAX_DENSE_EVENT`.
        """
        ...

    @abstractmethod
    def scaled(self, s: Tensor, /) -> Self:
        r"""The covariance times $s$.

        Args:
            s: The factor $s \ge 0$, a 0-dim tensor.

        Returns:
            A covariance of the same type.
        """
        ...


@pytree
class PerInputCov:
    """Dense `[N, E, E]` blocks, one per input. Fine for classifiers with small C.

    Args:
        blocks: The blocks, `[N, E, E]`.
        event_shape: The shape of one input's outputs; $E$ is its product.
    """

    blocks: Tensor
    event_shape: tuple[int, ...] = static()

    def diagonal(self) -> Tensor:
        """The diagonals of the blocks, `[N, *event_shape]`.

        Returns:
            A tensor shaped like the mean.
        """

        return self.blocks.diagonal(dim1=-2, dim2=-1).reshape(-1, *self.event_shape)

    def sample(self, n_samples: int, /, *, generator: torch.Generator | None = None) -> Tensor:
        """`n_samples` zero-mean draws through a symmetric square root of each block.

        The draws are independent across inputs. Negative eigenvalues of a block count as
        zero.

        Args:
            n_samples: Number of draws.
            generator: The generator to draw with, on any device: the draws move
                to the blocks'. `None` means torch's global RNG.

        Returns:
            The draws, `[n_samples, *mean.shape]`.
        """

        evals, evecs = cast(tuple[Tensor, Tensor], torch.linalg.eigh(self.blocks))
        root = evecs * evals.clamp_min(0).sqrt().unsqueeze(-2)  # root @ root^T = blocks
        m, e = self.blocks.shape[0], self.blocks.shape[-1]

        b = self.blocks
        z = torch.randn(
            n_samples, m, e, generator=generator, dtype=b.dtype, device=_home(generator)
        )
        z = z.to(b.device)

        return torch.einsum("nef,snf->sne", root, z).reshape(n_samples, m, *self.event_shape)

    def per_input(self) -> Tensor:
        """`blocks`.

        Returns:
            The blocks, `[N, E, E]`.
        """

        return self.blocks

    def scaled(self, s: Tensor, /) -> Self:
        r"""The blocks times $s$.

        Args:
            s: The factor $s \ge 0$, a 0-dim tensor.

        Returns:
            A covariance of the same type.
        """

        return type(self)(s * self.blocks, self.event_shape)


@pytree
class SampleCov:
    r"""Zero-mean draws $J z_s$, $z_s \sim \mathcal{N}(0, \Sigma)$. For large output events.

    $J$ is the Jacobian of the outputs and $\Sigma$ the weight covariance. The moments are
    empirical, computed from the stored draws.

    Args:
        draws: The $S$ draws, `[S, N, *event]`.
    """

    draws: Tensor

    def diagonal(self) -> Tensor:
        """The mean of the squared draws, `[N, *event]`.

        Returns:
            A tensor shaped like the mean.
        """

        return (self.draws**2).mean(0)

    def sample(self, n_samples: int, /, *, generator: torch.Generator | None = None) -> Tensor:
        """`n_samples` of the stored draws, chosen at random without replacement.

        Args:
            n_samples: Number of draws.
            generator: The generator that picks them, on any device. `None` means
                torch's global RNG.

        Returns:
            The draws, `[n_samples, *mean.shape]`.

        Raises:
            ValueError: `n_samples` is larger than the number of stored draws $S$.
        """

        s = self.draws.shape[0]

        if n_samples > s:
            raise ValueError(
                f"SampleCov holds S={s} draws, so it cannot give {n_samples}; ask outputs for "
                f"n_samples >= {n_samples}"
            )

        idx = torch.randperm(s, generator=generator, device=_home(generator))[:n_samples]
        idx = idx.to(self.draws.device)

        return self.draws[idx]

    def per_input(self) -> Tensor:
        """The mean outer product of the draws, per input.

        Returns:
            The blocks, `[N, E, E]`.

        Raises:
            StructureError: An input has more than `MAX_DENSE_EVENT` output entries.
        """

        s, n = self.draws.shape[:2]
        e = math.prod(self.draws.shape[2:])

        if e > MAX_DENSE_EVENT:
            raise StructureError(
                f"SampleCov.per_input needs at most MAX_DENSE_EVENT={MAX_DENSE_EVENT} output "
                f"entries per input, got {e}; use diagonal() or sample()"
            )

        flat = self.draws.reshape(s, n, e)

        return torch.einsum("sne,snf->nef", flat, flat) / s

    def scaled(self, s: Tensor, /) -> Self:
        r"""The draws times $\sqrt{s}$.

        Args:
            s: The factor $s \ge 0$, a 0-dim tensor.

        Returns:
            A covariance of the same type.
        """

        return type(self)(self.draws * s.sqrt())


@pytree
class FactorCov:
    r"""Low-rank covariance $F F^\top$ per input.

    The draws of `sample` are joint across the batch, as with `SampleCov`.

    Args:
        factor: The factor $F$, `[N, *event, R]`.
    """

    factor: Tensor

    def diagonal(self) -> Tensor:
        r"""$\sum_r F_{\cdot r}^2$, `[N, *event]`.

        Returns:
            A tensor shaped like the mean.
        """

        return (self.factor**2).sum(-1)

    def sample(self, n_samples: int, /, *, generator: torch.Generator | None = None) -> Tensor:
        r"""`n_samples` draws $F z$, $z \sim \mathcal{N}(0, I_R)$, one $z$ per draw for all inputs.

        Args:
            n_samples: Number of draws.
            generator: The generator to draw with, on any device: the draws move
                to the factor's. `None` means torch's global RNG.

        Returns:
            The draws, `[n_samples, *mean.shape]`.
        """

        r = self.factor.shape[-1]

        fac = self.factor
        z = torch.randn(n_samples, r, generator=generator, dtype=fac.dtype, device=_home(generator))
        z = z.to(fac.device)

        return torch.einsum("...r,sr->s...", self.factor, z)

    def per_input(self) -> Tensor:
        r"""$F F^\top$ per input, `[N, E, E]`.

        Returns:
            The blocks, `[N, E, E]`.

        Raises:
            StructureError: An input has more than `MAX_DENSE_EVENT` output entries.
        """

        n, r = self.factor.shape[0], self.factor.shape[-1]
        e = math.prod(self.factor.shape[1:-1])

        if e > MAX_DENSE_EVENT:
            raise StructureError(
                f"FactorCov.per_input needs at most MAX_DENSE_EVENT={MAX_DENSE_EVENT} output "
                f"entries per input, got {e}; use diagonal() or sample()"
            )

        flat = self.factor.reshape(n, e, r)

        return flat @ flat.mT

    def scaled(self, s: Tensor, /) -> Self:
        r"""The factor times $\sqrt{s}$.

        Args:
            s: The factor $s \ge 0$, a 0-dim tensor.

        Returns:
            A covariance of the same type.
        """

        return type(self)(self.factor * s.sqrt())


@pytree
class OutputGaussian:
    """Gaussian over model outputs (not targets: no observation noise).

    Args:
        mean: The mean of the outputs.
        cov: Their covariance.
    """

    mean: Outputs
    cov: OutputCov

    @property
    def variance(self) -> Tensor:
        """The marginal variance of each output entry, shaped like `mean`."""

        return self.cov.diagonal()

    @property
    def stddev(self) -> Tensor:
        """The marginal standard deviation of each output entry, shaped like `mean`."""

        return self.variance.sqrt()

    @property
    def covariance_matrix(self) -> Tensor:
        """The covariance of each input's output entries, `[N, E, E]` for `E` entries.

        Inputs are independent, so these blocks are the whole covariance.
        """

        return self.cov.per_input()

    def __repr__(self) -> str:
        shape = tuple(cast(Tensor, self.mean).shape)

        return f"OutputGaussian(shape={shape}, cov={type(self.cov).__name__})"


# --- posteriors -------------------------------------------------------------------------


@runtime_checkable
class Prior(Protocol):
    """Anything that can be the prior of a Laplace update."""

    def log_prob(self, params: Params, /) -> Tensor:
        r"""Log density of the posterior at `params`.

        Args:
            params: Parameter mapping with the posterior's keys and shapes.

        Returns:
            A 0-dim tensor, $\log q(\theta)$.
        """

    @abstractmethod
    def location(self, like: Params, /) -> Params:
        """The posterior's mean, shaped like `like`.

        Args:
            like: Parameter mapping whose names, shapes, dtypes and devices the mean must
                have.

        Returns:
            The mean, a parameter mapping.

        Raises:
            StructureError: Implementations raise it when their mean doesn't match `like`
                (see `check_location`).
        """
        ...

    @abstractmethod
    def add_precision_to[Op: Operator[Any]](self, op: Op, /) -> Op:
        r"""The operator plus the posterior's precision $\Lambda_0$.

        A Laplace update calls it on the scaled curvature to form the posterior precision
        (see `LaplacePosterior`).

        Args:
            op: A precision operator.

        Returns:
            An operator of the same type, $\text{op} + \Lambda_0$.

        Raises:
            StructureError: Implementations raise it when the operator's structure can't
                hold their precision.
        """
        ...


@runtime_checkable
class PriorWithCovariance(Prior, Protocol):
    """A weight posterior that can be pushed forward: it multiplies by its covariance."""

    @abstractmethod
    def cov_mv(self, v: Params, /) -> Params:
        r"""The product $\Sigma v$ with the posterior's covariance $\Sigma$.

        Args:
            v: Parameter mapping with the posterior's keys and shapes.

        Returns:
            $\Sigma v$, a parameter mapping.
        """
        ...


@runtime_checkable
class HasExpand(Protocol):
    """A prior with one shared precision that can expand to one per block or per weight.

    `lk.laplace` expands such a prior to the fitted parameters before the fit, so the
    hyper tuner tunes each block (`Layerwise`) or weight (`Diagonal`) on its own.
    """

    @abstractmethod
    def expand(self, params: Params, /) -> Self:
        """The prior with its precision expanded to `params`.

        Args:
            params: The parameters the prior is over.

        Returns:
            The expanded prior, or `self` when it already is.
        """
        ...


@runtime_checkable
class TunablePrior(Prior, Protocol):
    """A prior whose precision the hyper tuners tune (`lk.MarginalLikelihood`, say).

    `lk.laplace` checks the rules of `tunable` before it reads data and raises
    `StructureError` when they fail. A prior without `tunable` has no hyperparameters
    for the tuners.
    """

    @abstractmethod
    def tunable(self) -> Tensor | Mapping[str, Tensor]:
        """The hyperparameters to tune, usually the precision.

        Returns:
            A tensor or a mapping of tensors. Each must be a leaf of the prior itself (a
            `@pytree`), returned as it is, and positive, since the tune works on its log.
        """
        ...


@runtime_checkable
class WeightPosterior(Prior, Protocol):
    r"""A fitted weight-space Gaussian $\mathcal{N}(\mu, \Sigma)$ over the parameters in `mean`.

    A posterior whose covariance products batch can also define `cov_many` and `cov_sqrt_many`
    and so satisfy `HasCovMany`. For the exact output covariance, `outputs` (and the
    predictives built on it) then passes each chunk of vectors to them in one call instead
    of vmapping `cov_mv` and `cov_sqrt_mv`. This is the fast path for many outputs.
    """

    @property
    @abstractmethod
    def mean(self) -> Params:
        r"""The mean $\mu$, a parameter mapping."""
        ...

    @property
    @abstractmethod
    def model(self) -> ModelFn:
        """The model the posterior was fitted for."""
        ...

    @property
    @abstractmethod
    def likelihood(self) -> Likelihood:
        """The likelihood of the fit."""
        ...

    @property
    @abstractmethod
    def layout(self) -> Layout:
        """The block layout of the parameters."""
        ...

    @property
    @abstractmethod
    def caps(self) -> Capabilities:
        """The model's capabilities, which `outputs` and `sample_outputs` check."""
        ...

    @abstractmethod
    def precision(self) -> Operator[Any]:
        r"""The precision $\Sigma^{-1}$.

        Returns:
            A precision operator.
        """
        ...

    @abstractmethod
    def cov_mv(self, v: Params, /) -> Params:
        r"""The product $\Sigma v$.

        Args:
            v: Parameter mapping with the mean's keys and shapes.

        Returns:
            $\Sigma v$, a parameter mapping.

        Raises:
            StructureError: Implementations may raise it when the precision is not positive
                definite or `v` doesn't fit the operator's structure.
        """
        ...

    @abstractmethod
    def cov_sqrt_mv(self, v: Params, /) -> Params:
        r"""The product $L v$ with a square root of the covariance, $L L^\top = \Sigma$.

        Args:
            v: Parameter mapping with the mean's keys and shapes.

        Returns:
            $L v$, a parameter mapping.

        Raises:
            StructureError: Implementations may raise it when the precision is not positive
                definite or `v` doesn't fit the operator's structure.
        """
        ...

    @abstractmethod
    def sample(self, n_samples: int, /, *, generator: torch.Generator | None = None) -> Params:
        r"""`n_samples` draws from $\mathcal{N}(\mu, \Sigma)$.

        Args:
            n_samples: Number of draws.
            generator: The generator to draw with, on any device: the draws move
                to the mean's. `None` means torch's global RNG.

        Returns:
            A parameter mapping with a leading axis of `n_samples` on every leaf.

        Raises:
            StructureError: Implementations may raise it when the precision is not positive
                definite.
        """
        ...

    @abstractmethod
    def sample_centered(
        self, n_samples: int, /, *, generator: torch.Generator | None = None
    ) -> Params:
        r"""`n_samples` draws from $\mathcal{N}(0, \Sigma)$.

        Args:
            n_samples: Number of draws.
            generator: The generator to draw with, on any device: the draws move
                to the mean's. `None` means torch's global RNG.

        Returns:
            A parameter mapping with a leading axis of `n_samples` on every leaf, in at least
            float32 (`stat_dtype`).

        Raises:
            StructureError: Implementations may raise it when the precision is not positive
                definite.
        """
        ...

    @property
    @abstractmethod
    def variance(self) -> Params:
        r"""The diagonal of $\Sigma$.

        Returns:
            A parameter mapping shaped like `mean`.

        Raises:
            StructureError: Implementations may raise it when the precision is not positive
                definite.
        """
        ...


@runtime_checkable
class HasCovMany(Protocol):
    """A weight-space posterior whose covariance products take many vectors in one call.

    `outputs` uses these methods when a posterior has them and vmaps `cov_mv` and
    `cov_sqrt_mv` otherwise. A posterior implements them only when its products batch.
    """

    @abstractmethod
    def cov_many(self, v: Params, /) -> Params:
        r"""$\Sigma v_i$ for each of the `n` vectors $v_i$.

        Args:
            v: Parameter mapping whose leaves have a leading axis of `n` vectors.

        Returns:
            A parameter mapping shaped like `v`, in the mean's dtypes.
        """
        ...

    @abstractmethod
    def cov_sqrt_many(self, v: Params, /) -> Params:
        r"""$L v_i$ for each of the `n` vectors $v_i$, with $L L^\top = \Sigma$.

        Args:
            v: Parameter mapping whose leaves have a leading axis of `n` vectors.

        Returns:
            A parameter mapping shaped like `v`, in the mean's dtypes.
        """
        ...


@runtime_checkable
class HasVersions(Protocol):
    """A posterior that can tell when it no longer describes the model.

    For a weight-space posterior, `mean_outputs`, `outputs` and `sample_outputs` call
    `check_versions` before the forward. A function-space posterior calls it itself, in `at`.
    It is optional: a posterior without it is not checked.
    """

    @abstractmethod
    def check_versions(self) -> None:
        """Raise if a parameter the posterior was built from changed in place since the fit.

        An implementation can compare the version counters taken at fit time
        (`param_versions`) with the current ones.

        Raises:
            RuntimeError: A parameter changed in place since the fit.
        """
        ...


@runtime_checkable
class FunctionPosterior(Protocol):
    """A posterior over the model's outputs, evaluated at inputs."""

    @property
    @abstractmethod
    def model(self) -> ModelFn:
        """The model the posterior describes."""
        ...

    @property
    @abstractmethod
    def likelihood(self) -> Likelihood:
        """The likelihood of the fit."""
        ...

    @property
    @abstractmethod
    def requires(self) -> Requirements:
        """What the posterior needs from the model."""
        ...

    @abstractmethod
    def at(self, inputs: Inputs, /) -> OutputGaussian:
        """The Gaussian over the model's outputs at `inputs`.

        Args:
            inputs: A batch of inputs.

        Returns:
            The output Gaussian of the batch.

        Raises:
            RuntimeError: Implementations with `check_versions` raise it when a parameter
                changed in place since the fit.
        """
        ...


type Posterior = WeightPosterior | FunctionPosterior
"""What a predictive accepts: anything with a distribution over outputs."""


# --- the fitted Gaussian -------------------------------------------------------------


_Op = TypeVar("_Op", bound=Operator[Any])
_Pr = TypeVar("_Pr", bound=Prior)

# Explicit variance, as for `Pass`: inferred, it would be invariant (the dataclass's
# `__replace__` takes a likelihood), so a Gaussian posterior would not be assignable to
# `LaplacePosterior[Op, Pr]`.
_L_co = TypeVar("_L_co", bound=Likelihood, covariant=True, default=Likelihood)


@pytree
class LaplacePosterior(Generic[_Op, _Pr, _L_co]):  # noqa: UP046
    r"""The result of a Laplace update: the prior precision plus the scaled curvature.

    The posterior is $\mathcal{N}(\mu, \Lambda^{-1})$ with the precision

    $$\Lambda = \Lambda_0 + c N \bar C,$$

    where $\Lambda_0$ is the prior's precision, $N$ the `n_data`, $\bar C$ the
    `curvature`, and $c$ the likelihood's `curvature_scale()` for a `LateLikelihood` (a
    Gaussian with `sigma`), else 1. For a `LateLikelihood`, `curvature` is the unit one,
    so replacing `likelihood` (or tuning its leaves) never refits. The posterior is itself a
    `Prior`, so it can be the prior of the next update.

    The third type parameter is the likelihood's type, so `post.likelihood` keeps it
    (`lk.Gaussian` for a posterior fitted with one). `LaplacePosterior[Op, Pr]` means any
    likelihood, and a posterior with a narrower one is assignable to it.

    A per-block curvature (`HasBlocks`: `BlockDiag`, `Joint`) takes vectors keyed by block.
    Any other curvature needs a layout with exactly one block: otherwise the methods that
    turn parameters into vectors raise `StructureError`.

    Args:
        mean: The mean $\mu$, a parameter mapping.
        curvature: The curvature $\bar C$, a per-example average.
        prior: The prior posterior, which adds $\Lambda_0$.
        n_data: The effective data size $N$.
        model: The model.
        likelihood: The likelihood. Its tensors (such as `sigma`) are leaves, so they can
            be tuned.
        layout: The block layout of `mean`.
        caps: The model's capabilities. The default has nothing probed.
        versions: Version counters at fit time: of each `mean` tensor (a view of the
            module's parameter shares its counter; a copy has its own) and of the module's
            other, fixed parameters. `check_versions` compares them with the current ones.
        provenance: Resolved front-door choices, shown before the curvature in the repr.
            Display only, never dispatch.
        data_fit: The data-fit term of the fit data at the mean (`FitStat.data_fit`), as a
            one-element tuple, or empty when unknown (a tuple keeps None out of the
            leaves).
        index: For a `Selection`, each selected name (`"fc.weight[idx]"`) mapped to its
            flat indices (`Selection.selected_index`); empty for whole tensors.
            `lk.laplace` sets it. By hand: `copy.replace(post, index=sel.selected_index)`.
            It identifies the entries only; nothing computes with it.

    Raises:
        StructureError: `layout` is not `Layout.by_module(mean)`. Custom layouts are not
            supported in 0.1a.
    """

    mean: Params
    curvature: _Op
    prior: _Pr
    n_data: Tensor  # effective data size N
    model: ModelFn = static()
    likelihood: _L_co  # a child: its tensors (sigma) are leaves to tune
    layout: Layout = static()
    caps: Capabilities = static(default=Capabilities())
    versions: Mapping[str, int] = static(default_factory=dict[str, int])
    provenance: tuple[tuple[str, str], ...] = static(default=())
    data_fit: tuple[Tensor, ...] = ()
    index: Mapping[str, Tensor] = dataclasses.field(default_factory=dict[str, Tensor])

    def __post_init__(self) -> None:
        if self.layout != Layout.by_module(self.mean):
            raise StructureError(
                "LaplacePosterior needs layout == Layout.by_module(mean); "
                "custom layouts are not supported in 0.1a"
            )

    def __repr__(self) -> str:
        size = sum(p.numel() for p in self.mean.values())

        if isinstance(self.curvature, BlockDiag):
            blocks = cast(Mapping[str, Operator[Any]], self.curvature.blocks)
            kinds = ", ".join(sorted({type(b).__name__ for b in blocks.values()}))
            n = len(blocks)
            curvature = f"BlockDiag[{kinds}] ({n} block{'s' * (n != 1)}, {size} parameters)"
        elif isinstance(self.curvature, Joint):
            inner = type(cast(Joint[Any], self.curvature).op).__name__
            n = len(self.curvature.sizes)
            curvature = f"Joint[{inner}] ({n} block{'s' * (n != 1)}, {size} parameters)"
        else:
            curvature = f"{type(self.curvature).__name__} ({size} parameters)"

        if self.n_data.numel() == 1:
            n_data = f"{self.n_data.item():.6g}"
        else:
            n_data = str(tuple(self.n_data.shape))

        given = dict(self.provenance)
        rows = [
            *((k, given[k]) for k in ("subset", "hessian_approx") if k in given),
            ("curvature", curvature),
            ("prior", repr(self.prior)),
            ("likelihood", repr(self.likelihood)),
            *((k, v) for k, v in self.provenance if k not in ("subset", "hessian_approx")),
            ("n_data", n_data),
        ]
        width = max(len(k) for k, _ in rows)
        # A posterior as the prior prints on several lines: indent them under its row.
        pad = "\n" + " " * (width + 5)
        body = "\n".join(f"  {k:<{width}} = {v.replace(chr(10), pad)}" for k, v in rows)

        return f"LaplacePosterior(\n{body}\n)"

    def precision(self) -> _Op:
        r"""$\Lambda = \Lambda_0 + c N \bar C$: the prior's precision added to the scaled curvature.

        Returns:
            A precision operator.

        Raises:
            StructureError: The prior can't add its precision to the curvature's operator.
        """

        factor = curvature_scale(self.likelihood)
        scale = self.n_data if factor is None else self.n_data * factor

        return self.prior.add_precision_to(self.curvature.scaled(scale))

    @property
    def prior_precision(self) -> Tensor | Mapping[str, Tensor]:
        """The prior's precision hyperparameter, `prior.tunable()`.

        For the built-in priors that is their `prior_precision`.

        Raises:
            StructureError: The prior is not a `TunablePrior` (a previous posterior, say);
                its precision is an operator, `prior.precision()`.
        """

        if not isinstance(self.prior, TunablePrior):
            raise StructureError(
                f"the prior {type(self.prior).__name__} has no prior_precision hyperparameter "
                "(it is not a TunablePrior); use post.prior.precision() for its operator"
            )

        return self.prior.tunable()

    def snapshot(self) -> Self:
        """The same posterior with its own copy of the mean.

        Training the module afterwards then moves neither the mean nor a later fit's prior.
        The copy is a `clone`, so it stays differentiable with respect to the old mean. The
        fixed parameters keep their fit-time counters, so the guard still fires when they
        move later.

        Returns:
            A `LaplacePosterior` with the cloned mean, and with `versions` updated to the
            clones' counters.

        Raises:
            RuntimeError: The posterior is stale (see `check_versions`).
        """

        self.check_versions()

        mean = {k: v.clone() for k, v in self.mean.items()}
        versions = {**self.versions, **{k: v._version for k, v in mean.items()}}

        return copy.replace(self, mean=mean, versions=versions)

    def check_versions(self) -> None:
        """Raise `RuntimeError` if a parameter changed in place since the fit.

        A fixed parameter: the curvature was computed with its old value. A posterior
        parameter: only a mean that shares memory with the module (a by-hand posterior over
        views) or a mean edited in place moves; `lk.laplace` keeps a copy. Either way the
        posterior no longer describes the model. Predictives and `outputs` call this first.

        A posterior parameter whose counter went down is another tensor than the recorded
        one (counters only go up): its mean was replaced by a copy, as `lk.load` does into a
        template whose mean views the module.

        Raises:
            RuntimeError: A counter in `versions` differs from the current one, or a fixed
                parameter is gone from the module. When a posterior parameter's counter
                went down, the message points to `snapshot()`.
        """

        module = self.model.module
        live = dict(module.named_parameters()) if module is not None else {}

        for name, version in self.versions.items():
            if name in self.mean and self.mean[name]._version < version:
                raise RuntimeError(
                    f"posterior parameter {name!r} is another tensor than at the fit (version "
                    f"{version}, now {self.mean[name]._version}; counters only go up): the "
                    "mean was replaced by a copy, for example by `lk.load` into a template "
                    "whose mean views the module's parameters. Build the posterior from "
                    "`posterior.snapshot()` before replacing its mean (for `lk.load`, load into "
                    "`template.snapshot()`)"
                )

            if name in self.mean:
                now, what = self.mean[name]._version, "posterior parameter"
                why = (
                    "the mean moved (it shares memory with the module's parameter, or was "
                    "edited in place)"
                )
            else:
                now = live[name]._version if name in live else None
                what, why = "fixed parameter", "the curvature used its old value"

            if now != version:
                raise RuntimeError(
                    f"{what} {name!r} changed in place since the fit (version {version}, now "
                    f"{now}); {why}; refit the posterior"
                )

    # params <-> operator vectors

    def _single_block(self) -> str:
        if len(self.layout.blocks) != 1:
            raise StructureError(
                f"{type(self.curvature).__name__} needs a layout with exactly one block, "
                f"got {sorted(self.layout.blocks)}"
            )

        return next(iter(self.layout.blocks))

    def _vec(self, params: Params, lead: int = 0) -> Any:
        """Flat vectors in the operator's dtype (at least float32, see `stat_dtype`).

        With `lead=1`, params with a leading axis of n vectors give [n, P] each.
        """

        up = {k: v.to(stat_dtype(v.dtype)) for k, v in params.items()}

        if isinstance(self.curvature, HasBlocks):
            return self.layout.flatten_all(up, lead=lead)

        return self.layout.flatten(up, self._single_block(), lead=lead)

    def _unvec(self, vec: Any) -> Params:
        """Back to params, keeping the vectors' dtype and leading axes."""

        if isinstance(self.curvature, HasBlocks):
            out: dict[str, Tensor] = {}

            for b, x in cast(Mapping[str, Tensor], vec).items():
                out.update(self.layout.unflatten(x, b))

            return out

        return dict(self.layout.unflatten(cast(Tensor, vec), self._single_block()))

    def _leaves(self, vec: Any) -> list[Tensor]:
        return list(cast(Mapping[str, Tensor], vec).values()) if isinstance(vec, Mapping) else [vec]

    # Gaussian operations

    def cov_mv(self, v: Params, /) -> Params:
        r"""$\Lambda^{-1} v$ by the operator's `solve`, in the mean's dtypes.

        Args:
            v: Parameter mapping with the posterior's keys and shapes.

        Returns:
            $\Sigma v$, a parameter mapping.
        """

        return self._to_mean_dtype(self._unvec(self.precision().solve(self._vec(v))))

    def cov_sqrt_mv(self, v: Params, /) -> Params:
        r"""$L v$ with $L L^\top = \Lambda^{-1}$, the root the operator uses for sampling.

        Args:
            v: Parameter mapping with the mean's keys and shapes.

        Returns:
            $L v$, a parameter mapping.
        """

        return self._to_mean_dtype(self._unvec(self.precision().cov_sqrt_mv(self._vec(v))))

    def cov_many(self, v: Params, /) -> Params:
        """`cov_mv` of `n` vectors at once (a leading axis on every leaf); see `HasCovMany`.

        Args:
            v: Parameter mapping whose leaves have a leading axis of `n` vectors.

        Returns:
            A parameter mapping shaped like `v`, in the mean's dtypes.
        """

        return self._to_mean_dtype(self._unvec(self.precision().solve(self._vec(v, lead=1))))

    def cov_sqrt_many(self, v: Params, /) -> Params:
        """`cov_sqrt_mv` of `n` vectors at once (a leading axis on every leaf).

        Args:
            v: Parameter mapping whose leaves have a leading axis of `n` vectors.

        Returns:
            A parameter mapping shaped like `v`, in the mean's dtypes.
        """

        return self._to_mean_dtype(self._unvec(self.precision().cov_sqrt_mv(self._vec(v, lead=1))))

    def _to_mean_dtype(self, params: Params) -> Params:
        return {k: v.to(self.mean[k].dtype) for k, v in params.items()}

    def sample(self, n_samples: int, /, *, generator: torch.Generator | None = None) -> Params:
        r"""`n_samples` draws from $\mathcal{N}(\mu, \Lambda^{-1})$.

        Each is the mean plus a draw of `sample_centered`, cast to the mean's dtypes.

        Args:
            n_samples: Number of draws.
            generator: The generator to draw with, on any device: the draws move
                to the mean's. `None` means torch's global RNG.

        Returns:
            A parameter mapping with a leading axis of `n_samples` on every leaf.
        """

        noise = self.sample_centered(n_samples, generator=generator)

        return {k: self.mean[k] + noise[k].to(self.mean[k].dtype) for k in self.mean}

    def sample_centered(
        self, n_samples: int, /, *, generator: torch.Generator | None = None
    ) -> Params:
        r"""`n_samples` draws from $\mathcal{N}(0, \Lambda^{-1})$, in at least float32.

        Each is $L \varepsilon$ with $\varepsilon \sim \mathcal{N}(0, I)$ and $L$ the root of
        `cov_sqrt_mv`. They are the draws in `sample` minus the mean, without rounding at
        the mean's precision.

        Args:
            n_samples: Number of draws.
            generator: The generator to draw with, on any device. `None` means torch's
                global RNG. The draws move to the device of each leaf.

        Returns:
            A parameter mapping with a leading axis of `n_samples` on every leaf, in at least
            float32.
        """

        ref: Tensor | Mapping[str, Tensor] = self._vec(self.mean)

        # One generator for all leaves, which may live on several devices: draw on its
        # device (the CPU for torch's global RNG), then move each draw to its leaf.
        home = _home(generator)

        def draw(x: Tensor) -> Tensor:
            z = torch.randn(n_samples, *x.shape, generator=generator, dtype=x.dtype, device=home)
            return z.to(x.device)

        eps = {k: draw(x) for k, x in ref.items()} if isinstance(ref, Mapping) else draw(ref)
        return self._unvec(self.precision().cov_sqrt_mv(eps))

    @property
    def variance(self) -> Params:
        r"""$\operatorname{diag}(\Lambda^{-1})$ by `inverse_diagonal`, in the mean's dtypes.

        Returns:
            A parameter mapping shaped like `mean`.
        """

        return self._to_mean_dtype(self._unvec(self.precision().inverse_diagonal()))

    def log_marglik(self) -> Tensor:
        r"""Laplace log marginal likelihood of the fit data.

        $$
        \log p(\mathcal{D}) \approx \log p(\mathcal{D} \mid \mu)
            - \tfrac{1}{2} \left( \log \det \Lambda - \log \det \Lambda_0 \right)
            - \tfrac{1}{2} (\mu - \mu_0)^\top \Lambda_0 (\mu - \mu_0),
        $$

        the data fit minus half of `log_det_ratio` and `prior_scatter`. The
        log-likelihood $\log p(\mathcal{D} \mid \mu)$ is read from `data_fit` with the
        current `likelihood`. `lk.laplace` keeps `data_fit` when the data had targets.

        Returns:
            A 0-dim tensor for a 0-dim `scale`.

        Raises:
            ValueError: `data_fit` is empty (a fit without targets).
            StructureError: The prior can't add its precision to the curvature's operator,
                the operator finds a precision that is not positive definite (as `Dense`
                does), the prior's location doesn't match `mean`, or the curvature is not
                `HasBlocks` and the layout has more than one block.
        """

        if not self.data_fit:
            raise ValueError(
                "log_marglik needs the data-fit term, but data_fit is empty (fit without targets)"
            )

        ll = log_lik(self.likelihood, self.data_fit[0])

        return ll - 0.5 * (self.log_det_ratio() + self.prior_scatter())

    def log_det_ratio(self) -> Tensor:
        r"""The Occam term of `log_marglik`, $\log \det \Lambda - \log \det \Lambda_0$.

        The prior precision $\Lambda_0$ is the prior added to the curvature scaled by zero,
        so it has the curvature's operator structure.

        Returns:
            A 0-dim tensor.

        Raises:
            StructureError: The prior can't add its precision to the curvature's operator,
                or the operator finds a precision that is not positive definite (as `Dense`
                does).
        """

        zero = self.curvature.scaled(torch.zeros_like(self.n_data))

        return self.precision().logdet() - self.prior.add_precision_to(zero).logdet()

    def prior_scatter(self) -> Tensor:
        r"""The prior scatter $(\mu - \mu_0)^\top \Lambda_0 (\mu - \mu_0)$.

        $\mu_0$ is the prior's location and $\Lambda_0$ its precision, formed as in
        `log_det_ratio`.

        Returns:
            A 0-dim tensor for a 0-dim `scale`.

        Raises:
            StructureError: The prior's location doesn't match `mean`, the prior can't add
                its precision to the curvature's operator, or the curvature is not
                `HasBlocks` and the layout has more than one block.
        """

        mu0 = self.prior.location(self.mean)
        delta = self._vec({k: self.mean[k] - mu0[k] for k in self.mean})
        zero = self.curvature.scaled(torch.zeros_like(self.n_data))
        p0_delta = self.prior.add_precision_to(zero).mv(delta)

        return sum(
            (
                (d * p).sum()
                for d, p in zip(self._leaves(delta), self._leaves(p0_delta), strict=True)
            ),
            torch.zeros_like(self.n_data),
        )

    # as a prior for the next update

    def log_prob(self, params: Params, /) -> Tensor:
        r"""$\log \mathcal{N}(\theta; \mu, \Lambda^{-1})$ by the operator's `mv` and `logdet`.

        Args:
            params: Parameter mapping with the posterior's keys and shapes.

        Returns:
            A 0-dim tensor, $\log q(\theta)$.
        """

        delta = self._vec({k: params[k] - self.mean[k] for k in self.mean})
        prec = self.precision()
        prec_delta = prec.mv(delta)
        quad = sum(
            (d * p).sum()
            for d, p in zip(self._leaves(delta), self._leaves(prec_delta), strict=True)
        )
        dim = sum(d.numel() for d in self._leaves(delta))

        return 0.5 * prec.logdet() - 0.5 * dim * math.log(2 * math.pi) - 0.5 * quad

    def location(self, like: Params, /) -> Params:
        """The mean, after `check_location`: a posterior over other parameters can't be a prior.

        Args:
            like: Parameter mapping whose names, shapes, dtypes and devices the mean must have.

        Returns:
            The mean, a parameter mapping.
        """

        check_location(self.mean, like)

        return self.mean

    def add_precision_to[O: Operator[Any]](self, op: O, /) -> O:
        r"""`op` plus $\Lambda$ by `op.add_same`, so `op` must have the curvature's type.

        Args:
            op: A precision operator.

        Returns:
            An operator of the same type, $\text{op} + \Lambda_0$.
        """

        return op.add_same(self.precision())


def check_location(mean: Params, like: Params, /) -> None:
    """Check that a prior mean matches the parameters, for a prior's `location`.

    Args:
        mean: The prior's mean.
        like: The parameters.

    Raises:
        StructureError: `mean` differs from `like` in names, shapes, dtypes or devices.
    """

    if mean.keys() != like.keys():
        raise StructureError(
            f"prior mean has keys {sorted(mean)}, the parameters are {sorted(like)}"
        )

    wrong = {
        k: (tuple(mean[k].shape), tuple(x.shape))
        for k, x in like.items()
        if mean[k].shape != x.shape
    }

    if wrong:
        detail = ", ".join(f"{k}: {a} vs {b}" for k, (a, b) in wrong.items())
        raise StructureError(f"prior mean shapes differ from the parameters ({detail})")

    for k, x in like.items():
        m = mean[k]

        if m.dtype != x.dtype or m.device != x.device:
            raise StructureError(
                f"prior mean {k!r} is {m.dtype} on {m.device}, "
                f"the parameter is {x.dtype} on {x.device}"
            )


def param_versions(model: ModelFn, mean: Params, /) -> Mapping[str, int]:
    """Version counters of the `mean` tensors and of the module's other parameters.

    Args:
        model: The model. Without a module, only `mean` is recorded.
        mean: The posterior mean.

    Returns:
        A mapping from parameter name to its version counter.
    """

    out = {n: p._version for n, p in mean.items()}

    if model.module is not None:
        out.update({n: p._version for n, p in model.module.named_parameters() if n not in mean})

    return out


def _check_versions(posterior: WeightPosterior) -> None:
    if isinstance(posterior, HasVersions):
        posterior.check_versions()


# --- pushforward to output space -----------------------------------------------------


def mean_outputs(posterior: Posterior, inputs: Inputs, /) -> Outputs:
    """The model's outputs at the posterior's mean.

    For a function-space posterior this is `posterior.at(inputs).mean`, which pays for the
    covariance too.

    Args:
        posterior: A weight-space posterior or a function-space posterior.
        inputs: A batch of inputs.

    Returns:
        The outputs, as the model returns them.

    Raises:
        RuntimeError: The posterior's `check_versions` finds a parameter changed in place
            since the fit.
    """

    inputs = as_pytree(inputs)

    if isinstance(posterior, FunctionPosterior):
        return posterior.at(inputs).mean

    _check_versions(posterior)

    return posterior.model(posterior.mean, inputs)


_OUTPUT_CHUNK = 256
"""Output entries per Jacobian chunk in `predict_outputs` when `chunk_size` is `None`."""


DEFAULT_OUTPUT_SAMPLES = 64
"""Draws of a `SampleCov` when `n_samples` is `None` and the exact path does not apply."""


_MAX_FACTOR_ENTRIES = MAX_DENSE_EVENT**2
"""Most entries E * P per input of the factor of the exact jvp pushforward (`FactorCov`)
when `n_samples` is `None`: the budget of the dense [E, E] blocks of the vjp one."""


def exact_fits(
    caps: Capabilities,
    event_size: int | None,
    /,
    *,
    strategy: Strategy = "reverse-mode",
    n_params: int | None = None,
) -> bool:
    r"""Whether `n_samples=None` gives the exact output covariance by `strategy`.

    The budgets steer the default only: an explicit `n_samples=None` is always exact.

    Args:
        caps: The model's capabilities.
        event_size: Output entries per input $E$. `None` (not known yet) checks the
            capabilities only.
        strategy: The pushforward. `"reverse-mode"` needs vjp and vmap and $E \le$
            `MAX_DENSE_EVENT`. `"forward-mode"` needs jvp and vmap and $E P \le$ `MAX_DENSE_EVENT`
            squared, the entry budget of the vjp path's dense $[E, E]$ blocks.
        n_params: The number of parameters $P$.

    Returns:
        Whether `caps` and the budget allow the exact path.

    Raises:
        ValueError: `strategy="forward-mode"` with a known `event_size` and no `n_params`.
    """

    if strategy == "forward-mode":
        if event_size is not None and n_params is None:
            raise ValueError(
                "exact_fits(strategy='forward-mode') needs n_params when the event size is known"
            )

        small = event_size is None or event_size * cast(int, n_params) <= _MAX_FACTOR_ENTRIES

        return caps.jvp and caps.vmap and small

    small = event_size is None or event_size <= MAX_DENSE_EVENT

    return caps.vjp and caps.vmap and small


def resolve_strategy(
    caps: Capabilities,
    event_size: int | None,
    n_params: int,
    /,
    *,
    n_samples: int | Literal["exact"] | None = None,
) -> Strategy:
    """The pushforward mode `strategy=None` picks, for $E$ output entries and $P$ parameters.

    With `n_samples=None` (chosen): when both exact paths fit (`exact_fits`), forward
    mode if $P < E$ (a factor of $P$ columns instead of `[E, E]` blocks), else reverse
    mode; when one fits, that one. When neither does (sampled outputs), reverse mode (a
    VJP of a VJP) when the probe found vjp_twice, else forward mode when it found jvp.
    `event_size=None` picks reverse mode when both fit.

    The VJP of a VJP records the forward once, and its graph and the backward's stay in
    memory until the last draw. `strategy="forward-mode"` runs the forward per draw
    instead.

    `n_samples="exact"` ignores the budgets: it picks among the exact paths the
    capabilities allow, with the same $P < E$ rule when both are possible. When neither
    is, the requirement check that follows refuses the result.

    An int `n_samples` (sampled outputs) picks as sampled outputs do.

    Args:
        caps: The model's capabilities.
        event_size: Output entries per input $E$, or `None` when not known yet.
        n_params: The number of parameters $P$.
        n_samples: `n_samples` of `predict_outputs`.

    Returns:
        `"forward-mode"` or `"reverse-mode"`.
    """

    if isinstance(n_samples, int):
        return _sampled_strategy(caps)

    if n_samples == "exact":
        exact_jvp = exact_fits(caps, None, strategy="forward-mode")
        exact_vjp = exact_fits(caps, None, strategy="reverse-mode")
    else:
        exact_jvp = exact_fits(caps, event_size, strategy="forward-mode", n_params=n_params)
        exact_vjp = exact_fits(caps, event_size, strategy="reverse-mode")

    if exact_jvp and exact_vjp:
        return (
            "forward-mode" if event_size is not None and n_params < event_size else "reverse-mode"
        )

    if exact_jvp:
        return "forward-mode"

    if exact_vjp:
        return "reverse-mode"

    if n_samples == "exact":
        return "forward-mode" if caps.jvp else "reverse-mode"

    return _sampled_strategy(caps)


def _sampled_strategy(caps: Capabilities, /) -> Strategy:
    """The pushforward of sampled outputs.

    It is vjp (a VJP of a VJP) when the probe found vjp_twice, else jvp when it found jvp,
    else vjp (refused by the requirement check). The VJP of a VJP records the forward once
    and runs only the double backward per draw; a JVP runs the whole forward per draw.
    """

    if caps.vjp and caps.vjp_twice:
        return "reverse-mode"

    return "forward-mode" if caps.jvp else "reverse-mode"


def check_outputs_args(n_samples: object, chunk_size: object, strategy: object = None, /) -> None:
    """Refuse what `predict_outputs` refuses before any forward.

    Args:
        n_samples: `n_samples` of `predict_outputs`.
        chunk_size: `chunk_size` of `predict_outputs`.
        strategy: `strategy` of `predict_outputs`.

    Raises:
        ValueError: `n_samples` is not an int of at least 2 (sampled outputs are
            centred), `"exact"` or `None`; `chunk_size` is not a positive int or `None`;
            or `strategy` is not `"forward-mode"`, `"reverse-mode"` or `None`.
    """

    if strategy is not None and strategy not in ("forward-mode", "reverse-mode"):
        raise ValueError(
            f'strategy must be "forward-mode", "reverse-mode" or None, got {strategy!r}'
        )

    if n_samples is not None and n_samples != "exact":
        if isinstance(n_samples, bool) or not isinstance(n_samples, int):
            raise ValueError(
                f'n_samples must be an int (at least 2), "exact" or None, got {n_samples!r}'
            )

        if n_samples < 2:
            raise ValueError(
                f"sampled outputs need n_samples >= 2 (they are centred), got {n_samples}"
            )

    if chunk_size is not None and (
        isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size < 1
    ):
        raise ValueError(f"chunk_size must be an int of at least 1, got {chunk_size!r}")


def check_function_args(
    what: str,
    /,
    *,
    strategy: Strategy | None = None,
    n_samples: int | Literal["exact"] | None = None,
    chunk_size: int | None = None,
) -> None:
    """Refuse `strategy`, `n_samples` and `chunk_size` for a function-space posterior.

    Its `at` gives the output Gaussian itself, so there is no pushforward to choose.

    Args:
        what: What the error message names, such as the posterior's type.
        strategy: `strategy` of `predict_outputs`.
        n_samples: `n_samples` of `predict_outputs`.
        chunk_size: `chunk_size` of `predict_outputs`.

    Raises:
        ValueError: `strategy`, `n_samples` or `chunk_size` is not `None`.
    """

    given = [
        f"{name}={value!r}"
        for name, value in (
            ("strategy", strategy),
            ("n_samples", n_samples),
            ("chunk_size", chunk_size),
        )
        if value is not None
    ]

    if given:
        raise ValueError(
            f"{what} is a function-space posterior: its outputs come from `at`, so "
            f"{', '.join(given)} has no effect; leave them unset"
        )


def predict_outputs(
    posterior: Posterior,
    inputs: Inputs,
    *,
    strategy: Strategy | None = None,
    n_samples: int | Literal["exact"] | None = None,
    chunk_size: int | None = None,
    generator: torch.Generator | None = None,
) -> OutputGaussian:
    r"""Linearized pushforward of a posterior, $\mathcal{N}(f(x; \mu), J \Sigma J^\top)$.

    With $J$ the Jacobian of the outputs with respect to the parameters at the mean $\mu$,
    the linearization $f(x; \theta) \approx f(x; \mu) + J (\theta - \mu)$ maps
    $\theta \sim \mathcal{N}(\mu, \Sigma)$ to this Gaussian over the outputs, per input.

    `n_samples="exact"` is exact. With `strategy="reverse-mode"` it gives a `PerInputCov`
    and needs vjp and vmap. Its Jacobian is streamed in chunks of `chunk_size` output
    entries, so without autograd at most $N \cdot \text{chunk} \cdot P$ Jacobian entries are
    held at once (with autograd recording, the graph keeps more). Smaller chunks use less
    memory and take more VJPs.
    Each chunk's rows are recomputed for every later chunk, so with $E$ output entries the
    cost is $(E / \text{chunk} + 1) / 2$ times one Jacobian.

    With `strategy="forward-mode"` it gives a `FactorCov` with factor $J L$
    (`[N, *event, P]`, $L L^\top = \Sigma$, from `cov_sqrt_mv`) and needs jvp and vmap.
    Here `chunk_size` counts parameter directions: one JVP on the whole batch per
    direction, `chunk_size` vmapped at a time, from a `[chunk, P]` block of the identity.
    The factor holds $N E P$ entries in `stat_dtype` of the outputs. Each $L e_p$ is cast
    to the model's dtype before its JVP, so under bfloat16 the factor is a bfloat16
    product, kept in float32 afterwards.

    An int `n_samples` gives a `SampleCov` of that many draws $J z_s$, with $z_s$ the
    posterior's weight noise (`sample_centered`, at least float32) centred and rescaled so
    the variance is unbiased. They are kept in at least float32. They are pushed one at a
    time, so no vmap: `strategy="forward-mode"` uses a JVP per draw (needs jvp),
    `"reverse-mode"` a VJP of a VJP (needs vjp and vjp_twice, which fused attention
    kernels lack).

    `strategy=None` is `resolve_strategy` of the probe's event size (`caps.event_size`),
    resolved again from the outputs' event size if that differs. For exact outputs it is
    the cheaper path that fits (forward mode when $P < E$); models without jvp (attention)
    get the dense reverse-mode path when $E \le$ `MAX_DENSE_EVENT`. `n_samples=None` is
    exact when the chosen strategy's exact path fits (`exact_fits`), else 64 draws.
    `n_samples="exact"` picks the cheaper exact path the capabilities allow, whatever the
    budgets. Sampled outputs (an int `n_samples`, or `None` over budget) take reverse
    mode when the probe found vjp_twice, else forward mode. The VJP of a VJP records the
    forward once for all draws, so the batch's forward and backward graph stays in memory
    until the last draw; `strategy="forward-mode"` runs the forward per draw instead.

    Args:
        posterior: A weight-space posterior. A function-space one gives `posterior.at(inputs)`.
        inputs: A batch of independent inputs (the model acts per example). A mapping
            that is not a torch pytree node (a tokenizer's `BatchEncoding`) is passed on as
            a `dict`. 0-dim tensors and Python scalars in `inputs` are shared by every
            input (e.g. a diffusion timestep in a sampling loop); fitting data carries such
            values per example (`prepare` expands them).
        strategy: How the Jacobian products run, `"forward-mode"` or `"reverse-mode"`.
            `None` picks one, as above.
        n_samples: `"exact"` for the exact covariance, an int (at least 2) for that many
            draws, or `None` to choose, as above.
        chunk_size: Output entries (reverse mode) or parameter directions (forward mode)
            per chunk, at least 1. `None` means 256.
        generator: Seeds sampled outputs. One seed is drawn from it, or from torch's
            global RNG when `None`, so `torch.manual_seed` makes the draws repeatable.
            Exact outputs draw nothing.

    Returns:
        The `OutputGaussian` with mean $f(x; \mu)$ and a `PerInputCov`, `FactorCov` or
        `SampleCov`; for a function-space posterior, the output Gaussian of its `at`.

    Raises:
        ValueError: An int `n_samples` below 2 or an int `chunk_size` below 1. For a
            function-space posterior, any explicit `strategy`, `n_samples` or `chunk_size`: it has
            no pushforward to choose.
        RuntimeError: The posterior's `check_versions` finds a parameter changed in place
            since the fit.
        CapabilityError: The model lacks what the chosen path needs. Checked before the
            forward, and again once the outputs' event size is known.
        NotImplementedError: The model does not return a single tensor.
        StructureError: The posterior's covariance products fail, for example on a precision
            that is not positive definite.
    """

    check_outputs_args(n_samples, chunk_size, strategy)
    size = _OUTPUT_CHUNK if chunk_size is None else chunk_size
    inputs = as_pytree(inputs)

    if isinstance(posterior, FunctionPosterior):
        check_function_args(
            type(posterior).__name__, strategy=strategy, n_samples=n_samples, chunk_size=chunk_size
        )

        return posterior.at(inputs)

    _check_versions(posterior)

    caps = posterior.caps
    n_params = posterior.layout.n_params
    mode: Strategy = (
        resolve_strategy(caps, caps.event_size, n_params, n_samples=n_samples)
        if strategy is None
        else strategy
    )

    def exact(m: Strategy) -> Requirements:
        return Requirements(jvp=m == "forward-mode", vjp=m == "reverse-mode", vmap=True)

    # Check the capabilities before the forward. The event size budgets wait for the
    # outputs: their event size may differ from the probe's. Without vmap the default is
    # sampled whatever the event size.
    if n_samples == "exact":
        caps.check(exact(mode), f"outputs strategy {mode}")
    elif n_samples is not None or not exact_fits(caps, None, strategy=mode):
        caps.check(_sampled_requires(mode), f"sampled outputs strategy {mode}")

    mean = posterior.mean
    f = posterior.model(mean, inputs)

    if not isinstance(f, Tensor):
        raise NotImplementedError("outputs supports models returning a single tensor")

    n, event = f.shape[0], f.shape[1:]
    e = math.prod(event)

    if strategy is None and e != caps.event_size:
        mode = resolve_strategy(caps, e, n_params, n_samples=n_samples)

    if n_samples is None:
        fits = exact_fits(caps, e, strategy=mode, n_params=n_params)
        n_samples = "exact" if fits else DEFAULT_OUTPUT_SAMPLES

    if n_samples != "exact":
        s = n_samples
        caps.check(_sampled_requires(mode), f"sampled outputs strategy {mode}")
        gen = _Seed.draw(generator).generator(_mean_device(posterior))
        draws = _push_samples(posterior, inputs, f, mode, s, gen)

        return OutputGaussian(f, SampleCov(draws))

    caps.check(exact(mode), f"outputs strategy {mode}")

    if mode == "forward-mode":
        factor = _push_factor(posterior, inputs, n_params, size)

        return OutputGaussian(f, FactorCov(factor.to(stat_dtype(f.dtype))))

    starts = range(0, e, size)

    def basis(a: int) -> Tensor:
        """Rows a : a + size of the [E, E] identity, without forming it."""

        index = torch.arange(a, min(a + size, e), device=f.device)

        return torch.nn.functional.one_hot(index, e).to(f.dtype)

    def fwd(p: Params, xb: Inputs) -> Tensor:
        return cast(Tensor, posterior.model(p, xb)).reshape(-1)

    # 0-dim tensors and non-tensor leaves (Python scalars) are shared by every input: not
    # vmapped, passed as they are.
    shared = tree_map(lambda t: not isinstance(t, Tensor) or t.ndim == 0, inputs)
    in_dims = tree_map(lambda sh: None if sh else 0, shared)

    def unsq(t: Any, sh: bool) -> Any:
        return t if sh else t.unsqueeze(0)

    def rows(a: int) -> Params:
        """Jacobian rows a : a + size of every input: name -> [N, c, *shape]."""

        def jac(x: Inputs) -> Params:
            xb = tree_map(unsq, x, shared)
            pull = cast(Any, vjp(partial(fwd, xb=xb), dict(mean)))[1]

            def row(u: Tensor) -> Params:
                return pull(u)[0]

            return vmap(row)(basis(a))

        return vmap(jac, in_dims=(in_dims,))(inputs)

    def cov_rows(j: Params) -> Params:
        """Sigma J^T for Jacobian rows `j`, shaped like `j`."""

        c = next(iter(j.values())).shape[1]
        flat = {k: v.reshape(n * c, *v.shape[2:]) for k, v in j.items()}
        sj = (
            posterior.cov_many(flat)
            if isinstance(posterior, HasCovMany)
            else vmap(posterior.cov_mv)(flat)
        )

        return {k: v.reshape(n, c, *v.shape[1:]) for k, v in sj.items()}

    def tile(j: Params, sj: Params) -> Tensor:
        """J_a Sigma J_b^T per input: [N, c_a, c_b]."""

        return cast(
            Tensor,
            sum(torch.einsum("icp,idp->icd", j[k].flatten(2), sj[k].flatten(2)) for k in j),
        )

    # Tile (a, b) of each [E, E] block, for b <= a; the upper ones are transposes.
    tiles: dict[tuple[int, int], Tensor] = {}

    for a in starts:
        ja = rows(a)
        sa = cov_rows(ja)
        tiles[a, a] = tile(ja, sa)
        del ja

        for b in range(0, a, size):
            tiles[a, b] = tile(rows(b), sa).mT

    blocks = torch.cat(
        [
            torch.cat([tiles[a, b] if b <= a else tiles[b, a].mT for b in starts], dim=2)
            for a in starts
        ],
        dim=1,
    )

    return OutputGaussian(f, PerInputCov(blocks, tuple(event)))


def sample_outputs(
    posterior: Posterior,
    inputs: Inputs,
    n_samples: int,
    /,
    *,
    generator: torch.Generator | None = None,
    kind: Literal["linearized", "network"] = "linearized",
    strategy: Strategy | None = None,
) -> Tensor:
    r"""`n_samples` draws of the model's outputs at `inputs`, stacked along a new leading axis.

    For a weight-space posterior, draw $s$ uses one weight noise
    $\varepsilon_s \sim \mathcal{N}(0, \Sigma)$ (`posterior.noise`) for every input, so the
    draws are joint across the batch. The draws are in `stat_dtype` of the outputs. All
    `n_samples` weight draws are held at once ($n_samples P$ entries).

    A function-space posterior gives `posterior.at(inputs)`'s mean plus `cov.sample(n_samples)`.
    Those are joint across the batch for a `FactorCov` or a `SampleCov`, and independent
    per input for a `PerInputCov` (see `OutputCov`).

    The outputs have no observation noise: the predictive samples of a classifier are a
    link on top, e.g. `torch.softmax(lk.sample_outputs(post, x, 100), dim=-1)`. These are
    laplace-torch's `functional_samples` (`pred_type="glm"` for linearized, `"nn"` for
    network).

    Args:
        posterior: A weight-space posterior or a function-space posterior.
        inputs: A batch of inputs.
        n_samples: Number of draws, at least 1.
        generator: Seeds the draws. One seed is drawn from it, or from torch's global
            RNG when `None`, so `torch.manual_seed` makes the draws repeatable.
        kind: `"linearized"` draws $f(x; \mu) + J \varepsilon_s$, with $J$ the Jacobian at
            the mean $\mu$. `"network"` draws $f(x; \mu + \varepsilon_s)$, one forward per
            draw; it needs a weight-space posterior.
        strategy: The pushforward of a linearized draw on a weight-space posterior, no vmap: one
            JVP per draw (`"forward-mode"`, needs jvp) or a VJP of a VJP (`"reverse-mode"`, needs
            vjp and vjp_twice). `None` picks vjp when the probe found vjp_twice (the forward is
            recorded once for all draws), else jvp. The capabilities are checked before
            the forward. Leave it unset for `kind="network"` and for a function-space
            posterior.

    Returns:
        The draws, `[n_samples, N, *event]`: the model must return a single tensor, as for
        `outputs`.

    Raises:
        ValueError: `n_samples` is below 1; `kind` is neither `"linearized"` nor `"network"`; an
            explicit `strategy` with `kind="network"` or a function-space posterior; or, for a
            function-space posterior whose covariance is a `SampleCov`, `n_samples` above its number
            of draws.
        TypeError: `kind="network"` with a function-space posterior, which has no weights.
        RuntimeError: The posterior's `check_versions` finds a parameter changed in place
            since the fit.
        CapabilityError: For `kind="linearized"` on a weight-space posterior, the model lacks
            what `strategy` needs.
        StructureError: For a weight-space posterior, the model does not return one tensor
            (raised on the first forward), or the posterior's covariance products fail, for
            example on a precision that is not positive definite.
    """

    if n_samples < 1:
        raise ValueError(f"sample_outputs needs n_samples >= 1, got {n_samples}")

    if kind not in ("linearized", "network"):
        raise ValueError(f"sample_outputs kind must be 'linearized' or 'network', got {kind!r}")

    seed = _Seed.draw(generator)
    inputs = as_pytree(inputs)

    if isinstance(posterior, FunctionPosterior):
        if kind == "network":
            raise TypeError(
                f"sample_outputs(kind='network') samples weights, but {type(posterior).__name__} "
                "is not a weight-space posterior; function-space posteriors have no weights to "
                "sample: "
                "use kind='linearized'"
            )

        check_function_args(f"sample_outputs on {type(posterior).__name__}", strategy=strategy)
        out = posterior.at(inputs)
        mu = cast(Tensor, out.mean)

        gen = seed.generator(mu.device)

        return mu.unsqueeze(0) + out.cov.sample(n_samples, generator=gen).reshape(
            n_samples, *mu.shape
        )

    _check_versions(posterior)

    if kind == "network":
        if strategy is not None:
            raise ValueError(
                f"sample_outputs(kind='network') runs the network at weight draws, so "
                f"strategy={strategy!r} has no effect; leave it unset"
            )

        draws = posterior.sample(n_samples, generator=seed.generator(_mean_device(posterior)))
        f = [
            _one_tensor(posterior.model({k: v[s] for k, v in draws.items()}, inputs))
            for s in range(n_samples)
        ]

        return torch.stack(f).to(stat_dtype(f[0].dtype))

    caps = posterior.caps
    mode: Strategy = (
        resolve_strategy(caps, caps.event_size, posterior.layout.n_params, n_samples=n_samples)
        if strategy is None
        else strategy
    )
    caps.check(_sampled_requires(mode), f"sample_outputs strategy {mode}")

    mean = posterior.mean
    f = _one_tensor(posterior.model(mean, inputs))
    noise = posterior.sample_centered(n_samples, generator=seed.generator(_mean_device(posterior)))
    z = {k: v.to(mean[k].dtype) for k, v in noise.items()}
    draws = _push_draws(posterior, inputs, f, mode, z)

    return f.to(draws.dtype).unsqueeze(0) + draws


def _mean_device(posterior: WeightPosterior) -> torch.device:
    """The device of the posterior's mean, where its draws are made."""

    m = next(iter(posterior.mean.values()), None)

    return m.device if m is not None else torch.device("cpu")


def _one_tensor(f: Outputs, /) -> Tensor:
    """`f` if it is a tensor, else a `StructureError` naming what the model returned."""

    if not isinstance(f, Tensor):
        raise StructureError(
            f"sample_outputs needs a model that returns one tensor, got {type(f).__name__}"
        )

    return f


def _push_factor(posterior: WeightPosterior, inputs: Inputs, n_params: int, size: int) -> Tensor:
    """The factor J L, [N, *event, P], in the model's dtype.

    One JVP on the whole batch per column L e_p of the flattened parameter basis. Each
    chunk holds `size` rows of the [P, P] identity ([size, P] entries, in each leaf's
    dtype) and vmaps `size` forwards at a time.
    """

    mean = dict(posterior.mean)

    def fwd(p: Params) -> Tensor:
        return cast(Tensor, posterior.model(p, inputs))

    def push(t: Params) -> Tensor:
        return cast(Any, jvp(fwd, (mean,), (t,)))[1]

    def basis(a: int, b: int) -> Params:
        """Rows a : b of the [P, P] identity, split by leaf and shaped like it."""

        rows: dict[str, Tensor] = {}
        offset = 0

        for k, v in mean.items():
            block = torch.zeros(b - a, v.numel(), dtype=v.dtype, device=v.device)
            lo, hi = max(a, offset), min(b, offset + v.numel())

            if lo < hi:
                i = torch.arange(lo, hi, device=v.device)
                block[i - a, i - offset] = 1

            rows[k] = block.reshape(b - a, *v.shape)
            offset += v.numel()

        return rows

    cols: list[Tensor] = []

    for a in range(0, n_params, size):
        rows = basis(a, min(a + size, n_params))
        roots = (
            posterior.cov_sqrt_many(rows)
            if isinstance(posterior, HasCovMany)
            else vmap(posterior.cov_sqrt_mv)(rows)
        )
        cols.append(vmap(push)({k: roots[k].to(v.dtype) for k, v in mean.items()}))

    return torch.cat(cols).movedim(0, -1)


def _push_samples(
    posterior: WeightPosterior,
    inputs: Inputs,
    f: Tensor,
    strategy: Strategy,
    s: int,
    generator: torch.Generator,
) -> Tensor:
    """Zero-mean draws J z_s, [S, N, *event].

    z_s is the posterior's noise, centred and rescaled (unbiased variance); see `_push_draws`.
    """

    # Centre and rescale the noise in its own dtype (at least float32); cast to each
    # parameter's dtype only where it enters the model.
    mean = posterior.mean
    noise = posterior.sample_centered(s, generator=generator)
    z = {k: ((v - v.mean(0)) * math.sqrt(s / (s - 1))).to(mean[k].dtype) for k, v in noise.items()}

    return _push_draws(posterior, inputs, f, strategy, z)


def _sampled_requires(strategy: Strategy, /) -> Requirements:
    """What pushing one draw needs: a JVP, or a VJP of a VJP."""

    return (
        Requirements(jvp=True)
        if strategy == "forward-mode"
        else Requirements(vjp=True, vjp_twice=True)
    )


def _push_draws(
    posterior: WeightPosterior, inputs: Inputs, f: Tensor, strategy: Strategy, z: Params
) -> Tensor:
    """J z_s for each weight draw z_s, [S, N, *event], one pushforward per draw.

    The draws z have a leading dim S and the parameters' dtypes. The JVP or VJP runs in
    the model's dtype; the draws are kept in `stat_dtype` of it.
    """

    mean = dict(posterior.mean)
    s = next(iter(z.values())).shape[0]

    def fwd(p: Params) -> Tensor:
        return cast(Tensor, posterior.model(p, inputs))

    if strategy == "forward-mode":

        def push(t: Params) -> Tensor:
            return cast(Any, jvp(fwd, (mean,), (t,)))[1]

    else:
        # J t is the VJP of the linear map u -> J^T u at any u, with cotangent t.
        pull = cast(Any, vjp(fwd, mean))[1]
        pull_t = cast(Any, vjp(lambda u: pull(u)[0], torch.zeros_like(f)))[1]

        def push(t: Params) -> Tensor:
            return pull_t(t)[0]

    out = [push({k: v[i] for k, v in z.items()}) for i in range(s)]

    return torch.stack(out).to(stat_dtype(f.dtype))
