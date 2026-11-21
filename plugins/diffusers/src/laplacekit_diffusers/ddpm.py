"""Laplace on a denoiser trained with a DDPM-family scheduler, and the checks of that scheduler."""

from abc import abstractmethod
from collections.abc import Callable, Iterable, Mapping
from typing import Any, Literal, Protocol, TypeIs, get_args, runtime_checkable

import torch
from torch import Tensor, nn

import laplacekit as lk
from laplacekit.core import Params, Selection, Subset
from laplacekit_diffusers.denoiser import denoiser

__all__ = ["laplace"]


_Prediction = Literal["epsilon", "v_prediction", "sample"]


_PREDICTIONS: tuple[_Prediction, ...] = get_args(_Prediction)


@runtime_checkable
class _HasConfig(Protocol):
    @property
    @abstractmethod
    def config(self) -> Any: ...


@runtime_checkable
class _HasAddNoise(Protocol):
    @abstractmethod
    def add_noise(self, original_samples: Any, noise: Any, timesteps: Any, /) -> Any: ...


@runtime_checkable
class _HasGetVelocity(Protocol):
    @abstractmethod
    def get_velocity(self, sample: Any, noise: Any, timesteps: Any, /) -> Any: ...


@runtime_checkable
class _NoiseScheduler(_HasConfig, _HasAddNoise, Protocol):
    """A DDPM-family scheduler, which has a config and `add_noise`."""


_MEMBERS: tuple[tuple[str, type], ...] = (
    ("config", _HasConfig),
    ("add_noise", _HasAddNoise),
)


_Velocity = Callable[[Tensor, Tensor, Tensor], Tensor]


def laplace(
    unet: nn.Module,
    scheduler: object,
    data: Iterable[Tensor | Mapping[str, Tensor]],
    *,
    n_data: int | None = None,
    subset: Subset | Params | Selection = lk.subsets.last_layer,
    hessian_approx: lk.core.HessianApprox[Any] | None = None,
    prior: lk.core.Prior = lk.priors.Isotropic(),  # noqa: B008  # immutable
    hyper_tuner: lk.core.HyperTuner | None = lk.MarginalLikelihood(),  # noqa: B008  # immutable
    generator: torch.Generator | None = None,
) -> lk.LaplacePosterior[Any, Any]:
    r"""Fit a Laplace approximation to a denoiser trained with a DDPM-family scheduler.

    DDPM-family schedulers include `DDPMScheduler` and `DDIMScheduler`. For each batch,
    the timestep $t$ is uniform over the training timesteps,
    $\epsilon \sim \mathcal{N}(0, I)$, and $x_t$ is `scheduler.add_noise(x0, eps, t)`.
    All of these are drawn from the batch's generator. The target depends on
    `scheduler.config.prediction_type`. It is $\epsilon$ for `"epsilon"`,
    `scheduler.get_velocity(x0, eps, t)` for `"v_prediction"`, and $x_0$ for `"sample"`.

    The fit is `lk.laplace` with `denoiser(unet)`, that preparation of the batches, and
    `Gaussian(sigma=1.0, event_dims=len(S))`, a Gaussian with fixed unit noise.

    The scheduler, its `prediction_type` and its noise are checked before any data is
    read. The noise must be variance-preserving,
    $x_t = \sqrt{\bar\alpha_t}\, x_0 + \sqrt{1 - \bar\alpha_t}\, \epsilon$, which is
    checked at three timesteps. One batch then goes through `unet` to check that the
    output has the target's shape, so `data` is iterated once more than by `lk.laplace`.

    Args:
        unet: The denoiser, a `UNet2DModel` or any `diffusers` model with the same call.
        scheduler: The noise scheduler. It needs `config` and `add_noise`, and also
            `get_velocity` for `"v_prediction"`.
        data: Batches of clean samples $x_0$, `[N, *S]`, or mappings that hold the clean
            `"sample"` and conditioning entries (`class_labels`,
            `encoder_hidden_states`, ...), which reach the model unchanged. A mapping
            may not hold a `"timestep"`, since `laplace` draws it.
        n_data: The number of examples, which is the size of the training set when `data`
            is a subset of it.
        subset: As in `lk.laplace`.
        hessian_approx: As in `lk.laplace`.
        prior: As in `lk.laplace`.
        hyper_tuner: As in `lk.laplace`. The noise is fixed, so it tunes the prior
            precision only.
        generator: As in `lk.laplace`. It seeds the timesteps and the noise too.

    Returns:
        The posterior, as from `lk.laplace`.

    Raises:
        StructureError: Before any data, when the scheduler lacks `config` or
            `add_noise` (as a flow-matching scheduler does), its `prediction_type` is
            not `"epsilon"`, `"v_prediction"` or `"sample"`, it has `"v_prediction"`
            without `get_velocity`, `num_train_timesteps` is not an int, or its noise is
            not variance-preserving (as in `EulerDiscreteScheduler`). On the first
            batch, when the output is not a tensor of the target's shape (as with a
            model that predicts a variance). On any batch, when the batch is neither a
            tensor nor a mapping with a tensor `"sample"`, it holds a `"timestep"`, or
            `unet` does not return a `BaseOutput` with a `sample`. Otherwise as in
            `lk.laplace`.
        CapabilityError: As in `lk.laplace`.
        ValueError: The data has no batches, or as in `lk.laplace`.
        TypeError: `data` is a one-shot iterator, or as in `lk.laplace`.
    """

    sched, prediction, steps, velocity = _check_scheduler(scheduler)
    fn = denoiser(unet)

    src = lk.data.from_loader(
        data, prepare=_prepare(sched, prediction, steps, velocity), n_data=n_data
    )
    # Only the shapes of this batch matter, so a fixed generator draws it.
    first = next(iter(src(torch.Generator())), None)

    if first is None:
        raise ValueError("lk.diffusers.laplace: the data yielded no batches")

    with torch.no_grad():
        out = fn({}, first.inputs)

    target = first.targets

    if not isinstance(out, Tensor) or not isinstance(target, Tensor):
        raise lk.StructureError("lk.diffusers.laplace needs tensor outputs and targets")

    if out.shape != target.shape:
        raise lk.StructureError(
            f"lk.diffusers.laplace: the model's output {tuple(out.shape)} does not match the "
            f"target {tuple(target.shape)}; models with learned variance "
            f"(`out_channels == 2 * in_channels`) are not supported"
        )

    return lk.laplace(
        fn,
        src,
        lk.Gaussian(sigma=1.0, event_dims=target.ndim - 1),
        subset=subset,
        hessian_approx=hessian_approx,
        prior=prior,
        hyper_tuner=hyper_tuner,
        generator=generator,
    )


def _check_scheduler(
    scheduler: object,
) -> tuple[_NoiseScheduler, _Prediction, int, _Velocity | None]:
    if not isinstance(scheduler, _NoiseScheduler):
        missing = [name for name, member in _MEMBERS if not isinstance(scheduler, member)]
        # A scheduler without add_noise is most likely flow matching (`scale_noise`).
        flow = "" if isinstance(scheduler, _HasAddNoise) else "; flow matching is not supported yet"

        raise lk.StructureError(
            f"lk.diffusers.laplace needs a DDPM-family scheduler with config and add_noise "
            f"(DDPMScheduler, DDIMScheduler, ...); {type(scheduler).__name__} has no "
            f"{', '.join(missing)}{flow}"
        )

    config = scheduler.config
    prediction = config.get("prediction_type") if isinstance(config, Mapping) else None
    steps = config.get("num_train_timesteps") if isinstance(config, Mapping) else None

    if not _is_prediction(prediction):
        allowed = ", ".join(repr(p) for p in _PREDICTIONS)

        raise lk.StructureError(
            f"lk.diffusers.laplace needs scheduler.config.prediction_type in {allowed}, got "
            f"{prediction!r}"
        )

    velocity: _Velocity | None = None

    if prediction == "v_prediction":
        if not isinstance(scheduler, _HasGetVelocity):
            raise lk.StructureError(
                f"lk.diffusers.laplace: prediction_type 'v_prediction' needs "
                f"scheduler.get_velocity; {type(scheduler).__name__} has none"
            )

        velocity = scheduler.get_velocity

    if not isinstance(steps, int):
        raise lk.StructureError(
            f"lk.diffusers.laplace needs an int scheduler.config.num_train_timesteps, got {steps!r}"
        )

    # DDPM-family noise is variance-preserving: x_t = sqrt(abar_t) x0 + sqrt(1 - abar_t) eps.
    # A scheduler with the same methods but another forward process (Euler's x0 + sigma eps)
    # would give a fit against the wrong inputs.
    t = torch.tensor([0, steps // 2, steps - 1])
    one, zero = torch.ones(3, dtype=torch.float64), torch.zeros(3, dtype=torch.float64)
    total = scheduler.add_noise(one, zero, t) ** 2 + scheduler.add_noise(zero, one, t) ** 2

    if not torch.allclose(total, torch.ones_like(total), rtol=0, atol=1e-6):
        raise lk.StructureError(
            f"lk.diffusers.laplace needs a variance-preserving (DDPM-family) scheduler, where "
            f"add_noise gives sqrt(abar_t) x0 + sqrt(1 - abar_t) eps; "
            f"{type(scheduler).__name__} is not"
        )

    return scheduler, prediction, steps, velocity


def _prepare(
    scheduler: _NoiseScheduler, prediction: _Prediction, steps: int, velocity: _Velocity | None
) -> Callable[[Any, torch.Generator], lk.Batch]:
    def prepare(raw: Any, generator: torch.Generator, /) -> lk.Batch:
        x0, cond = _split(raw)

        g = generator
        t = torch.randint(0, steps, (len(x0),), generator=g, device=g.device).to(x0.device)
        eps = torch.randn(x0.shape, generator=g, dtype=x0.dtype, device=g.device).to(x0.device)
        x_t = scheduler.add_noise(x0, eps, t)

        match prediction:
            case "epsilon":
                target = eps
            case "v_prediction":
                # `_check_scheduler` gives a velocity for every "v_prediction" scheduler.
                assert velocity is not None
                target = velocity(x0, eps, t)
            case "sample":
                target = x0

        return lk.Batch({"sample": x_t, "timestep": t, **cond}, target)

    return prepare


def _split(raw: Any) -> tuple[Tensor, dict[str, Any]]:
    if isinstance(raw, Tensor):
        return raw, {}

    if isinstance(raw, Mapping):
        if "sample" not in raw or not isinstance(raw["sample"], Tensor):
            raise lk.StructureError(
                f"lk.diffusers.laplace needs a tensor under 'sample' in each batch, got the "
                f"keys {sorted(raw)}"
            )

        # The timestep is drawn from the batch's generator; a given one would replace it.
        if "timestep" in raw:
            raise lk.StructureError(
                "lk.diffusers.laplace draws each batch's 'timestep' itself; remove the "
                "'timestep' entry from the batch"
            )

        return raw["sample"], {k: v for k, v in raw.items() if k != "sample"}

    raise lk.StructureError(
        f"lk.diffusers.laplace needs batches of clean samples (a tensor) or mappings with a "
        f"'sample', got {type(raw).__name__}"
    )


def _is_prediction(x: object, /) -> TypeIs[_Prediction]:
    return x in _PREDICTIONS
