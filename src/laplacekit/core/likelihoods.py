"""Likelihood contract: one required method, `log_prob(f, y)`.

Optional hooks are separate one-method protocols, checked with `isinstance`:
`HasDistribution`, `HasSample`, `HasOutputHessian` and `HasMask`. The helpers in this
module call a hook when present and derive what they can from the others.

`LateLikelihood` is the protocol of a likelihood whose hyperparameters are applied after
the fit (the Gaussian's `sigma`), so tuning them never refits.
"""

from abc import abstractmethod
from collections.abc import Mapping
from typing import Literal, Protocol, cast, runtime_checkable

import torch
from torch import Tensor
from torch.distributions import Distribution
from torch.func import grad, hessian, vmap

from laplacekit.core._pytree import pytree, static, tree_leaves
from laplacekit.core._seed import _Seed
from laplacekit.core.types import (
    CapabilityError,
    Inputs,
    Outputs,
    Requirements,
    StructureError,
    Targets,
)

MAX_DENSE_EVENT = 4096


@runtime_checkable
class Likelihood(Protocol):
    r"""A likelihood $p(y \mid f)$ over targets given model outputs.

    The core sums positions and weights examples.
    """

    @abstractmethod
    def log_prob(self, f: Outputs, y: Targets, /) -> Tensor:
        r"""Log-likelihood $\log p(y \mid f)$ summed over the event.

        Args:
            f: Model outputs for a batch.
            y: Targets for the same batch.

        Returns:
            A tensor shaped `[N]`, or `[N, *positions]` (per token, say).
        """
        ...


@runtime_checkable
class HasDistribution(Protocol):
    """The distribution over targets. `default_log_prob` and `default_sample` derive from it."""

    @abstractmethod
    def distribution(self, f: Outputs, /) -> Distribution:
        r"""The distribution $p(y \mid f)$.

        Args:
            f: Model outputs for a batch.

        Returns:
            A torch distribution over the batch's targets.
        """
        ...


@runtime_checkable
class HasMask(Protocol):
    """Which output positions count, e.g. non-padding tokens.

    False positions change neither the data fit nor the curvature.
    """

    @abstractmethod
    def mask(self, f: Outputs, inputs: Inputs, /) -> Tensor | None:
        """The positions that count.

        Args:
            f: Model outputs for a batch.
            inputs: The batch's inputs.

        Returns:
            A bool tensor shaped like `log_prob`'s output, `[N, *positions]`, True where a
            position counts. `None` when every position counts.
        """
        ...


@runtime_checkable
class HasTargetCheck(Protocol):
    """Checks a batch's targets before the fit uses them, e.g. values outside the support.

    The fit calls it eagerly on each batch with targets, before the curvature's `update`.
    It runs under a `torch.func` transform over the parameters (`grad`), where the targets
    are plain tensors, and is skipped when a target leaf is transformed (`vmap` over
    batches).
    """

    @abstractmethod
    def check_targets(self, y: Targets, /) -> None:
        """Check the targets of one batch.

        It reads only `y`: under a transform over the likelihood's own fields those fields
        are wrapped.

        Args:
            y: The batch's targets.

        Raises:
            ValueError: A target value is invalid, e.g. outside the support.
            StructureError: The targets are badly shaped.
        """
        ...


@runtime_checkable
class HasOutputCheck(Protocol):
    """Checks the model's outputs before any data is fitted.

    For example, a shaped parameter that must fit the event shape. The probe calls it once
    on the outputs of its plain forward (the first two examples of the first batch, under
    `no_grad`, outside any `torch.func` transform), with or without targets, before it
    tries any transform. Whatever it raises propagates from the probe, never recorded as a
    missing capability.
    """

    @abstractmethod
    def check_outputs(self, f: Outputs, /) -> None:
        """Check the model's outputs.

        Args:
            f: Outputs of the probe's plain forward.

        Raises:
            StructureError: The likelihood can't take outputs like `f`.
        """
        ...


@runtime_checkable
class HasSample(Protocol):
    """Sampling of targets, used by the sampled and MC Fishers."""

    @abstractmethod
    def sample(self, f: Outputs, /, *, generator: torch.Generator | None = None) -> Targets:
        r"""Draw targets $y \sim p(y \mid f)$.

        Args:
            f: Model outputs for a batch.
            generator: The generator to draw with, on $f$'s device. `None` means torch's
                global RNG.

        Returns:
            One draw of targets for the batch, shaped like real targets.
        """
        ...


class OutputHessian(Protocol):
    r"""Hessian $H = -\nabla_f^2 \log p(y \mid f)$ with respect to $f$, per example.

    Vectors are shaped like $f$.
    """

    @abstractmethod
    def mv(self, v: Tensor, /) -> Tensor:
        """Product $H v$, per example.

        Args:
            v: A tensor shaped like $f$.

        Returns:
            $H v$, shaped like $f$.
        """
        ...

    @abstractmethod
    def sqrt_mv(self, v: Tensor, /) -> Tensor:
        r"""Product $A v$ with a square root $A$, $A A^\top = H$, per example.

        Args:
            v: A tensor shaped like $f$.

        Returns:
            $A v$, shaped like $f$.
        """
        ...

    @abstractmethod
    def diagonal(self) -> Tensor:
        """The diagonal of $H$, per example.

        Returns:
            A tensor shaped like $f$.
        """
        ...

    @abstractmethod
    def dense(self) -> Tensor:
        """$H$ as a dense matrix per example.

        Returns:
            A tensor shaped `[N, E, E]`, for $E$ entries per example.

        Raises:
            StructureError: The event size $E$ is too large.
        """
        ...


@runtime_checkable
class HasOutputHessian(Protocol):
    """Closed-form output Hessian. Without it, `output_hessian` uses the MC Fisher."""

    @abstractmethod
    def output_hessian(self, f: Outputs, /) -> OutputHessian:
        """The output Hessian (the Fisher) at `f`, in closed form.

        Args:
            f: Model outputs for a batch.

        Returns:
            The per-example output Hessian.
        """
        ...


@runtime_checkable
class LateLikelihood(Protocol):
    """A likelihood whose hyperparameters are applied after the fit.

    The fit uses `unit()` for curvature and sampling, and keeps the additive statistic
    `fit_stat` in place of the log-likelihood. The posterior multiplies the unit curvature by
    `curvature_scale()` and turns the summed statistic into the log-likelihood with
    `log_lik`. So both read the hyperparameters at the time they are called.

    `unit()` returns the likelihood itself (`unit() is lik`) exactly when there is no late
    hyperparameter, as for a `Gaussian` with a fixed sigma. Methods that can't apply hyperparameters
    late (`Requirements.late_scale=False`) accept those and refuse the rest. A likelihood
    whose hyperparameter shapes the curvature, such as a per-output sigma, returns itself
    from `unit()`.

    A leaf that `unit()` shares (by identity) is applied at fit time; the hyper tuners
    refuse to change it. A likelihood that is not a `LateLikelihood` is its own
    `unit()`, so all its leaves are refused. A leaf left out of `unit()` is late and tunable,
    so it must be read only by `curvature_scale` and `log_lik`.

    The hyper tuners change the leaves a likelihood lists in `tunable()`
    (`TunableLikelihood`), such as the learned `sigma` of the built-in `Gaussian`.
    """

    @abstractmethod
    def unit(self) -> Likelihood:
        """The likelihood the fit uses for curvature and sampling.

        Returns:
            The likelihood without its late hyperparameters, or `self` (by identity) when
            there are none.
        """
        ...

    @abstractmethod
    def curvature_scale(self) -> Tensor:
        """The factor the posterior multiplies the unit curvature by.

        Returns:
            A tensor computed from the current hyperparameters.
        """
        ...

    @abstractmethod
    def fit_stat(self, f: Outputs, y: Targets, weights: Tensor | None, /) -> Tensor:
        """The additive statistic the fit keeps in place of the log-likelihood.

        The fit sums it over batches.

        Args:
            f: Model outputs for a batch.
            y: Targets for the same batch.
            weights: Per-example weights, or `None` for all ones.

        Returns:
            The batch's statistic.
        """
        ...

    @abstractmethod
    def log_lik(self, stat: Tensor, /) -> Tensor:
        """The log-likelihood from the summed statistic.

        Args:
            stat: `fit_stat` summed over all batches.

        Returns:
            The summed log-likelihood at the current hyperparameters.
        """
        ...


def unit(lik: Likelihood, /) -> Likelihood:
    """The likelihood the fit uses.

    Args:
        lik: Any likelihood.

    Returns:
        `lik.unit()` for a `LateLikelihood`, else `lik`.
    """

    return lik.unit() if isinstance(lik, LateLikelihood) else lik


def _has_late_hyperparameters(lik: Likelihood, /) -> bool:
    """Whether `lik` is a `LateLikelihood` whose `unit()` is a different object."""

    return isinstance(lik, LateLikelihood) and lik.unit() is not lik


def curvature_scale(lik: Likelihood, /) -> Tensor | None:
    """The factor on the unit curvature.

    Args:
        lik: Any likelihood.

    Returns:
        `lik.curvature_scale()` for a `LateLikelihood`, else `None` (no factor).
    """

    return lik.curvature_scale() if isinstance(lik, LateLikelihood) else None


def log_lik(lik: Likelihood, data_fit: Tensor, /) -> Tensor:
    """The summed log-likelihood from the fit's data-fit term.

    Args:
        lik: The likelihood of the fit (not its `unit()`).
        data_fit: `FitStat.data_fit`.

    Returns:
        `lik.log_lik(data_fit)` for a `LateLikelihood`. Otherwise `data_fit` itself, which
        already is the summed log-likelihood.
    """

    return lik.log_lik(data_fit) if isinstance(lik, LateLikelihood) else data_fit


type LikelihoodTree = Likelihood | Mapping[str, LikelihoodTree]
"""A model returning a dict of outputs gets a dict of likelihoods."""


@pytree
class DenseOutputHessian:
    """Per-example dense output Hessian.

    Build it with `from_dense`.

    Args:
        h: The Hessian, shaped `[N, E, E]`.
        root: The symmetric square root of `h`, computed once by `from_dense`. Negative
            eigenvalues are clamped to zero there, so `sqrt_mv` holds only for a PSD `h`.
        shape: The shape of the outputs, `[N, ...]`.
    """

    h: Tensor
    root: Tensor
    shape: torch.Size = static()

    @staticmethod
    def from_dense(h: Tensor, shape: torch.Size) -> "DenseOutputHessian":
        """Wrap a dense Hessian and compute its symmetric square root.

        Args:
            h: The Hessian, shaped `[N, E, E]`.
            shape: The shape of the outputs, `[N, ...]`, with $E$ entries per example.

        Returns:
            The output Hessian, with negative eigenvalues of `h` clamped to zero in `root`.
        """

        eigvals, eigvecs = cast(tuple[Tensor, Tensor], torch.linalg.eigh(h))
        root = eigvecs @ (eigvals.clamp_min(0).sqrt().unsqueeze(-1) * eigvecs.mT)

        return DenseOutputHessian(h, root, shape)

    def mv(self, v: Tensor, /) -> Tensor:
        """Product $H v$, per example."""

        return (self.h @ v.reshape(v.shape[0], -1, 1)).reshape(self.shape)

    def sqrt_mv(self, v: Tensor, /) -> Tensor:
        """Product of the symmetric square root `root` with `v`, per example."""

        return (self.root @ v.reshape(v.shape[0], -1, 1)).reshape(self.shape)

    def diagonal(self) -> Tensor:
        """The diagonal of $H$, per example."""

        return self.h.diagonal(dim1=-2, dim2=-1).reshape(self.shape)

    def dense(self) -> Tensor:
        """`h` itself."""

        return self.h


@pytree
class MaskedOutputHessian:
    r"""The output Hessian $M H M$ for the 0/1 diagonal $M$ of `mask`.

    Masked entries of $f$ carry no curvature.

    Args:
        inner: The unmasked output Hessian $H$.
        mask: Bool, shaped like $f$'s leading axes (the positions). It is broadcast over
            the trailing event axes.
    """

    inner: OutputHessian
    mask: Tensor

    def _m(self, like: Tensor) -> Tensor:
        m = self.mask.reshape(*self.mask.shape, *([1] * (like.ndim - self.mask.ndim)))
        return m.expand_as(like)

    def mv(self, v: Tensor, /) -> Tensor:
        """Product $M H M v$, per example.

        Args:
            v: A tensor shaped like $f$.

        Returns:
            $H v$, shaped like $f$.
        """

        m = self._m(v)
        return torch.where(m, self.inner.mv(torch.where(m, v, 0)), 0)

    def sqrt_mv(self, v: Tensor, /) -> Tensor:
        r"""Product $M A v$, per example, with $A A^\top = H$ from `inner.sqrt_mv`.

        Args:
            v: A tensor shaped like $f$.

        Returns:
            $A v$, shaped like $f$.
        """

        out = self.inner.sqrt_mv(v)
        return torch.where(self._m(out), out, 0)

    def diagonal(self) -> Tensor:
        """The diagonal of $H$, zero at masked entries.

        Returns:
            A tensor shaped like $f$.
        """

        d = self.inner.diagonal()
        return torch.where(self._m(d), d, 0)

    def dense(self) -> Tensor:
        """$M H M$ as a dense matrix per example, from `inner.dense()`.

        Returns:
            A tensor shaped `[N, E, E]`, for $E$ entries per example.
        """

        h = self.inner.dense()
        n, event = h.shape[0], h.shape[1]

        # f flattens position-major, so each position's flag repeats over its event entries.
        m = self.mask.reshape(n, -1, 1)
        m = m.expand(n, m.shape[1], event // m.shape[1]).reshape(n, event)
        keep = m.unsqueeze(2) & m.unsqueeze(1)

        return torch.where(keep, h, 0)


def _mask(lik: Likelihood, f: Outputs, inputs: Inputs | None) -> Tensor | None:
    """The likelihood's mask, checked to be bool. None without `HasMask` or inputs."""

    if inputs is None or not isinstance(lik, HasMask):
        return None

    mask = lik.mask(f, inputs)

    if mask is None:
        return None

    if mask.dtype != torch.bool:
        raise StructureError(
            f"{type(lik).__name__}.mask returned a {mask.dtype} tensor; the mask must be bool "
            "(True where a position counts), e.g. `attention_mask.bool()`"
        )

    return mask


def _masked(
    hess: OutputHessian, lik: Likelihood, f: Outputs, inputs: Inputs | None
) -> OutputHessian:
    mask = _mask(lik, f, inputs)

    if mask is None:
        return hess

    shape = cast(Tensor, f).shape

    if mask.ndim > len(shape) or mask.shape != shape[: mask.ndim]:
        raise StructureError(
            f"{type(lik).__name__}.mask is shaped {tuple(mask.shape)}, but it must be the "
            f"leading axes of f, shaped {tuple(shape)}: the event axes must be trailing"
        )

    return MaskedOutputHessian(hess, mask)


def log_prob(lik: Likelihood, f: Outputs, y: Targets, /, inputs: Inputs | None = None) -> Tensor:
    r"""Log-likelihood $\log p(y \mid f)$ per example: `lik.log_prob` summed over positions.

    Args:
        lik: The likelihood.
        f: Model outputs for a batch.
        y: Targets for the same batch.
        inputs: The batch's inputs. With `inputs` and a `HasMask` likelihood, masked
            positions are zeroed before the sum. Without `inputs`, no mask is applied.

    Returns:
        A tensor shaped `[N]`.

    Raises:
        StructureError: `lik.log_prob` returns a scalar, the mask is not bool, or the mask
            is not shaped like `lik.log_prob`'s output.
    """

    lp = lik.log_prob(f, y)

    if lp.ndim == 0:
        raise StructureError(
            f"{type(lik).__name__}.log_prob returned a scalar; it must be shaped [N] or "
            "[N, *positions], one value per example"
        )

    mask = _mask(lik, f, inputs)

    if mask is not None:
        if mask.shape != lp.shape:
            raise StructureError(
                f"{type(lik).__name__}.mask is shaped {tuple(mask.shape)}, but log_prob is "
                f"shaped {tuple(lp.shape)}; they must match"
            )

        lp = torch.where(mask, lp, 0)

    return lp.reshape(lp.shape[0], -1).sum(1)


def default_log_prob(lik: HasDistribution, f: Outputs, y: Targets, /) -> Tensor:
    r"""Log-likelihood from the distribution: `distribution(f).log_prob(y)`.

    Args:
        lik: A likelihood with `HasDistribution`.
        f: Model outputs for a batch.
        y: Targets, shaped like the distribution's batch and event shapes together.

    Returns:
        $\log p(y \mid f)$ per example and position, summed over the distribution's
        event.

    Raises:
        StructureError: `y` has another shape. It is not broadcast.
    """

    d = lik.distribution(f)
    yt = cast(Tensor, y)
    want = d.batch_shape + d.event_shape

    if yt.shape != want:
        raise StructureError(
            f"{type(lik).__name__} needs targets shaped {tuple(want)} (the distribution's "
            f"batch and event shapes), got {tuple(yt.shape)}"
        )

    return d.log_prob(yt)


def default_sample(
    lik: HasDistribution, f: Outputs, /, *, generator: torch.Generator | None = None
) -> Targets:
    """Targets drawn from `distribution(f).sample()`.

    `Distribution.sample()` takes no generator. With one, a seed drawn from it seeds
    torch's global RNG on $f$'s device for the draw, and the old state comes back after.

    Args:
        lik: A likelihood with `HasDistribution`.
        f: Model outputs for a batch.
        generator: The generator to draw with. `None` means torch's global RNG.

    Returns:
        One draw of targets.
    """

    if generator is None:
        return lik.distribution(f).sample()

    device = tree_leaves(f)[0].device

    with _Seed.draw(generator).torch_rng(device):
        return lik.distribution(f).sample()


def can_sample(lik: Likelihood, /) -> bool:
    """Whether `sample` works.

    Args:
        lik: Any likelihood.

    Returns:
        True when `lik` has `HasSample` or `HasDistribution`.
    """

    return isinstance(lik, HasSample | HasDistribution)


def distribution(lik: Likelihood, f: Outputs, /) -> Distribution:
    r"""The distribution $p(y \mid f)$ of a likelihood.

    Args:
        lik: A likelihood with `HasDistribution`.
        f: Model outputs for a batch.

    Returns:
        `lik.distribution(f)`.

    Raises:
        CapabilityError: `lik` has no `HasDistribution`.
    """

    if not isinstance(lik, HasDistribution):
        raise CapabilityError(f"{type(lik).__name__} has no distribution method (HasDistribution)")

    return lik.distribution(f)


def sample(lik: Likelihood, f: Outputs, /, *, generator: torch.Generator | None = None) -> Targets:
    r"""Targets drawn from $p(y \mid f)$.

    The `HasSample` hook when `lik` has one, else a draw from its distribution
    (`default_sample`).

    Args:
        lik: A likelihood with `HasSample` or `HasDistribution`.
        f: Model outputs for a batch.
        generator: The generator to draw with. `None` means torch's global RNG.

    Returns:
        One draw of targets.

    Raises:
        CapabilityError: `lik` has neither `HasSample` nor `HasDistribution`.
    """

    if isinstance(lik, HasSample):
        return lik.sample(f, generator=generator)

    if isinstance(lik, HasDistribution):
        return default_sample(lik, f, generator=generator)

    raise CapabilityError(
        f"{type(lik).__name__} can't sample targets: it needs HasSample or HasDistribution"
    )


def _dense_event(lik: Likelihood, f: Tensor, what: str) -> int:
    event = f[0].numel()

    if event > MAX_DENSE_EVENT:
        raise StructureError(
            f"{what} forms a dense {event}x{event} matrix per example; give "
            f"{type(lik).__name__} an output_hessian method (HasOutputHessian)"
        )

    return event


def output_hessian(
    lik: Likelihood,
    f: Outputs,
    /,
    *,
    generator: torch.Generator | None = None,
    inputs: Inputs | None = None,
    mc_samples: int = 1,
    fisher: Literal["exact", "mc"] | None = None,
) -> OutputHessian:
    r"""The Fisher of the likelihood with respect to $f$.

    It is the closed form when `lik` has `HasOutputHessian`. Otherwise it is the MC Fisher

    $$\frac{1}{S} \sum_{s=1}^S g_s g_s^\top, \qquad
    g_s = \nabla_f \log p(y_s \mid f),$$

    with the $y_s$ drawn one after another from `generator`. The MC Fisher is dense per
    example, so the event size is capped at `MAX_DENSE_EVENT`.

    Args:
        lik: The likelihood.
        f: Model outputs for a batch.
        generator: The generator of the MC Fisher's draws. `None` means torch's global
            RNG. The closed form draws nothing.
        inputs: The batch's inputs. With `inputs` and a `HasMask` likelihood, the result
            is a `MaskedOutputHessian`.
        mc_samples: Number of draws $S$ for the MC Fisher.
        fisher: `"exact"` takes the closed form and raises without one, `"mc"` always
            draws. `None` takes the closed form when there is one, else draws (as the
            function-space methods do).

    Returns:
        The per-example output Hessian.

    Raises:
        CapabilityError: `lik` has none of `HasOutputHessian`, `HasSample` and
            `HasDistribution`.
        StructureError: The MC Fisher's event size exceeds `MAX_DENSE_EVENT`, or the mask
            is not bool or not shaped like the leading axes of $f$.
    """

    if fisher == "exact" and not isinstance(lik, HasOutputHessian):
        raise CapabilityError(
            f'{type(lik).__name__} has no closed-form output Hessian; pass fisher="mc" for the '
            "Monte Carlo Fisher"
        )

    if fisher != "mc" and isinstance(lik, HasOutputHessian):
        return _masked(lik.output_hessian(f), lik, f, inputs)

    if not can_sample(lik):
        raise CapabilityError(
            f"{type(lik).__name__} has no output Hessian: it needs HasOutputHessian, or "
            "HasSample or HasDistribution for the MC Fisher"
        )

    ft = cast(Tensor, f)
    n = ft.shape[0]
    event = _dense_event(lik, ft, "the MC Fisher")

    # The S draws stacked along the batch axis: one gradient gives every g_s.
    with torch.no_grad():
        ys = [
            cast(Tensor, sample(lik, ft.detach(), generator=generator)) for _ in range(mc_samples)
        ]

    y = torch.cat(ys)

    def total(fr: Tensor) -> Tensor:
        return log_prob(lik, fr, y).sum()

    g = grad(total)(ft.repeat(mc_samples, *([1] * (ft.ndim - 1))))
    g = g.reshape(mc_samples, n, event)
    h = torch.einsum("sne,snf->nef", g, g) / mc_samples

    return _masked(DenseOutputHessian.from_dense(h, ft.shape), lik, f, inputs)


def observed_output_hessian(
    lik: Likelihood, f: Outputs, y: Targets, /, inputs: Inputs | None = None
) -> OutputHessian:
    r"""Hessian of $-\log p(y \mid f)$ with respect to $f$ at the given targets, per example.

    Computed by autograd. It may be indefinite (a heteroscedastic Gaussian, a Student-t),
    so a curvature built from it may be too. It equals the Fisher for exponential families
    in their natural parameters.

    Args:
        lik: The likelihood.
        f: Model outputs for a batch.
        y: Targets for the same batch.
        inputs: The batch's inputs. With `inputs` and a `HasMask` likelihood, the result
            is a `MaskedOutputHessian`.

    Returns:
        The per-example observed output Hessian, dense.

    Raises:
        StructureError: The event size exceeds `MAX_DENSE_EVENT`, the mask is not bool or
            not shaped like the leading axes of $f$, or `lik.log_prob` returns a scalar.
    """

    ft = cast(Tensor, f)
    event = _dense_event(lik, ft, "observed_output_hessian")

    def nll(fi: Tensor, yi: Tensor) -> Tensor:
        return -log_prob(lik, fi.unsqueeze(0), yi.unsqueeze(0)).sum()

    def hess_i(fi: Tensor, yi: Tensor) -> Tensor:
        return hessian(nll)(fi, yi).reshape(event, event)

    h = vmap(hess_i)(ft, cast(Tensor, y))

    return _masked(DenseOutputHessian.from_dense(h, ft.shape), lik, f, inputs)


def check_likelihood(req: Requirements, lik: LikelihoodTree, what: str, /) -> None:
    """Check that a likelihood has what a method needs.

    Not `lk.testing.check_likelihood`, which checks a likelihood's own laws for plugin
    tests.

    Args:
        req: The method's requirements.
        lik: The likelihood, or a tree of them; each leaf is checked.
        what: The method's name, for the error message.

    Raises:
        CapabilityError: `lik` (or a leaf) lacks a protocol `req` needs:
            `needs_distribution`, `needs_sample` or `needs_output_hessian`. Or `lik` has
            late hyperparameters (its `unit()` is another object) and `req.late_scale` is
            False.
    """

    if isinstance(lik, Mapping):
        for k, sub in lik.items():
            check_likelihood(req, sub, f"{what} (likelihood {k!r})")

        return

    if not req.late_scale and _has_late_hyperparameters(lik):
        name = type(lik).__name__

        raise CapabilityError(
            f"{what} can't fit {name}: {name} has a late hyperparameter (such as sigma), and "
            "the empirical Fisher is quadratic in it, so it can't be applied after the fit. "
            "Either fix the noise (lk.Gaussian(sigma=0.1), say: the default Gaussian() learns "
            "sigma; or from_log_prob with a fixed-sigma log-prob), or use a curvature that "
            "applies sigma after the fit: DiagGGN, DenseGGN or DiagGGN(fisher='mc')"
        )

    missing: list[str] = []

    if req.needs_distribution and not isinstance(lik, HasDistribution):
        missing.append("a distribution (HasDistribution)")

    if req.needs_sample and not can_sample(lik):
        missing.append("sampling (HasSample or HasDistribution)")

    hint = ""

    if req.needs_closed_form_hessian and not isinstance(lik, HasOutputHessian):
        missing.append("a closed-form output Hessian (HasOutputHessian)")
        hint = (
            '; pass fisher="mc" for the Monte Carlo Fisher'
            if can_sample(lik)
            else '; it can\'t sample either, so pass fisher="empirical" or observed=True'
        )

    if req.needs_output_hessian and not (isinstance(lik, HasOutputHessian) or can_sample(lik)):
        missing.append("an output Hessian (HasOutputHessian, or HasSample or HasDistribution)")

    if missing:
        raise CapabilityError(
            f"{what} needs a likelihood with {' and '.join(missing)}; "
            f"{type(lik).__name__} has none{hint}"
        )
