"""laplacekit-diffusers on tiny denoisers built from configs (no downloads), in float64.

Each fit is compared with a by-hand `lk.laplace` written out here: a `ModelFn`, a `prepare`
that draws `t` and `eps` from the batch's generator and computes `x_t` and the target from
the noise schedule by hand, and a generator with the same seed.
"""

from importlib import metadata
from typing import Any

import laplacekit_diffusers
import pytest
import torch
from diffusers.models.transformers.dit_transformer_2d import DiTTransformer2DModel
from diffusers.models.unets.unet_2d import UNet2DModel
from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_dpmsolver_multistep import DPMSolverMultistepScheduler
from diffusers.schedulers.scheduling_euler_discrete import EulerDiscreteScheduler
from diffusers.schedulers.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler,
)
from torch import Tensor, nn
from torch.func import functional_call

import laplacekit as lk

F64 = torch.float64
STEPS = 50  # num_train_timesteps of every scheduler here
BETA_START, BETA_END = 1e-4, 0.02  # the DDPM defaults: a linear beta schedule

# functorch has no batching rule for the fused CPU attention kernel (the probe's vmap on the
# attention models); vmap still runs, through the slow fallback.
ATTENTION_VMAP = pytest.mark.filterwarnings(
    "ignore:There is a performance drop because we have not yet implemented the batching "
    "rule:UserWarning"
)


# --- tiny models -----------------------------------------------------------------------------


def _unet(out_channels: int = 1, num_class_embeds: int | None = None) -> UNet2DModel:
    torch.manual_seed(0)

    return (
        UNet2DModel(
            sample_size=8,
            in_channels=1,
            out_channels=out_channels,
            layers_per_block=1,
            block_out_channels=(8, 16),
            norm_num_groups=4,
            down_block_types=("DownBlock2D", "DownBlock2D"),
            up_block_types=("UpBlock2D", "UpBlock2D"),
            add_attention=False,
            num_class_embeds=num_class_embeds,
        )
        .double()
        .eval()
    )


def _text_unet() -> UNet2DConditionModel:
    torch.manual_seed(0)

    return (
        UNet2DConditionModel(
            sample_size=8,
            in_channels=1,
            out_channels=1,
            layers_per_block=1,
            block_out_channels=(8, 16),
            norm_num_groups=4,
            down_block_types=("CrossAttnDownBlock2D", "DownBlock2D"),
            up_block_types=("UpBlock2D", "CrossAttnUpBlock2D"),
            cross_attention_dim=8,
            attention_head_dim=2,
        )
        .double()
        .eval()
    )


def _dit() -> DiTTransformer2DModel:
    torch.manual_seed(0)

    return (
        DiTTransformer2DModel(
            num_attention_heads=2,
            attention_head_dim=4,
            in_channels=1,
            out_channels=1,
            num_layers=1,
            sample_size=4,
            patch_size=2,
            num_embeds_ada_norm=STEPS,
            norm_num_groups=1,
        )
        .double()
        .eval()
    )


def _images(size: int = 8, batches: int = 2, n: int = 4) -> list[Tensor]:
    g = torch.Generator().manual_seed(1)

    return [torch.randn(n, 1, size, size, generator=g, dtype=F64) for _ in range(batches)]


def _labelled(images: list[Tensor], key: str, make: Any) -> list[dict[str, Tensor]]:
    g = torch.Generator().manual_seed(3)

    return [{"sample": x, key: make(len(x), g)} for x in images]


# --- the by-hand reference -------------------------------------------------------------------


def _abar() -> Tensor:
    """alpha-bar_t = prod_{s <= t} (1 - beta_s), from the linear schedule (float32, as built)."""

    betas = torch.linspace(BETA_START, BETA_END, STEPS, dtype=torch.float32)

    return torch.cumprod(1.0 - betas, dim=0).to(F64)


def _bcast(a: Tensor, x: Tensor) -> Tensor:
    return a.reshape(-1, *([1] * (x.ndim - 1)))


def _eps_target(x0: Tensor, eps: Tensor, t: Tensor) -> Tensor:
    del x0, t
    return eps


def _v_target(x0: Tensor, eps: Tensor, t: Tensor) -> Tensor:
    a = _bcast(_abar()[t], x0)
    return a.sqrt() * eps - (1 - a).sqrt() * x0


def _x0_target(x0: Tensor, eps: Tensor, t: Tensor) -> Tensor:
    del eps, t
    return x0


def _gen() -> torch.Generator:
    """The seed both sides of a comparison fit with."""

    return torch.Generator().manual_seed(0)


def _by_hand(net: nn.Module, data: list[Any], target: Any, **kwargs: Any) -> lk.LaplacePosterior:
    """`lk.laplace` with a hand-written ModelFn and prepare; `kwargs` go to `lk.laplace`."""

    def forward(params: lk.core.Params, inp: Any) -> Tensor:
        rest = {k: v for k, v in inp.items() if k not in ("sample", "timestep")}
        return functional_call(net, dict(params), (inp["sample"], inp["timestep"]), rest).sample

    def prepare(raw: Any, g: torch.Generator, /) -> lk.Batch:
        if isinstance(raw, Tensor):
            x0, cond = raw, {}
        else:
            x0, cond = raw["sample"], {k: v for k, v in raw.items() if k != "sample"}

        t = torch.randint(0, STEPS, (len(x0),), generator=g)
        eps = torch.randn(x0.shape, generator=g, dtype=x0.dtype)
        a = _bcast(_abar()[t], x0)
        x_t = a.sqrt() * x0 + (1 - a).sqrt() * eps

        return lk.Batch({"sample": x_t, "timestep": t, **cond}, target(x0, eps, t))

    first = data[0] if isinstance(data[0], Tensor) else data[0]["sample"]
    size = kwargs.pop("n_data", None)
    kwargs.setdefault("generator", _gen())

    return lk.laplace(
        lk.core.ModelFn(forward, module=net),
        lk.data.from_loader(data, prepare=prepare, n_data=size),
        likelihood=lk.Gaussian(sigma=1.0, event_dims=first.ndim - 1),
        **kwargs,
    )


def _assert_same(got: lk.LaplacePosterior, want: lk.LaplacePosterior) -> None:
    assert set(got.mean) == set(want.mean)
    assert got.likelihood == want.likelihood

    g = torch.Generator().manual_seed(4)
    v = {k: torch.randn(p.shape, generator=g, dtype=p.dtype) for k, p in want.mean.items()}

    for k in want.mean:
        torch.testing.assert_close(got.mean[k], want.mean[k], rtol=0, atol=0)

    torch.testing.assert_close(got.n_data, want.n_data)
    torch.testing.assert_close(got.prior.prior_precision, want.prior.prior_precision)
    torch.testing.assert_close(got.data_fit, want.data_fit, rtol=1e-10, atol=0)

    got_cov, want_cov = got.cov_mv(v), want.cov_mv(v)

    for k in want.mean:
        torch.testing.assert_close(got_cov[k], want_cov[k], rtol=1e-10, atol=1e-12)


# --- model -----------------------------------------------------------------------------------


def test_model_returns_the_sample_of_the_output():
    net = _unet()
    x, t = _images()[0], torch.tensor([3, 4, 5, 6])
    fn = lk.diffusers.denoiser(net)
    params = dict(net.named_parameters())

    with torch.no_grad():
        torch.testing.assert_close(
            fn(params, {"sample": x, "timestep": t}), net(x, t).sample, rtol=0, atol=0
        )


def test_model_passes_sample_and_timestep_positionally():
    """A DiT's first argument is `hidden_states`, not `sample`."""

    net = _dit()
    x, t, y = torch.randn(2, 1, 4, 4, dtype=F64), torch.tensor([3, 40]), torch.tensor([0, 1])
    fn = lk.diffusers.denoiser(net)

    with torch.no_grad():
        got = fn({}, {"sample": x, "timestep": t, "class_labels": y})
        torch.testing.assert_close(got, net(x, t, class_labels=y).sample, rtol=0, atol=0)


class _Plain(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(2, 2)

    def forward(self, sample: Tensor, timestep: Tensor) -> Tensor:
        del timestep
        return self.lin(sample)


def test_model_needs_a_diffusers_output():
    fn = lk.diffusers.denoiser(_Plain())

    with pytest.raises(lk.StructureError, match="BaseOutput.*Tensor"):
        fn({}, {"sample": torch.randn(3, 2), "timestep": torch.tensor(1)})


def test_model_needs_sample_and_timestep():
    fn = lk.diffusers.denoiser(_unet())

    with pytest.raises(lk.StructureError, match="timestep"):
        fn({}, {"sample": torch.randn(1, 1, 8, 8, dtype=F64)})


# --- laplace against the by-hand fit ---------------------------------------------------------


@pytest.mark.parametrize("scheduler", [DDPMScheduler, DDIMScheduler])
def test_epsilon_matches_by_hand_fit(scheduler: Any):
    net, data = _unet(), _images()
    post = lk.diffusers.laplace(net, scheduler(num_train_timesteps=STEPS), data, generator=_gen())
    want = _by_hand(net, data, _eps_target, subset=lk.subsets.last_layer)

    assert set(post.mean) == {"conv_out.weight", "conv_out.bias"}
    assert post.likelihood == lk.Gaussian(sigma=1.0, event_dims=3)
    _assert_same(post, want)


def test_v_prediction_target_by_hand():
    """The target is sqrt(abar_t) eps - sqrt(1 - abar_t) x0 (only the data-fit term sees it)."""

    net, data = _unet(), _images()
    sched = DDPMScheduler(num_train_timesteps=STEPS, prediction_type="v_prediction")
    post = lk.diffusers.laplace(net, sched, data, generator=_gen())

    _assert_same(post, _by_hand(net, data, _v_target))

    eps = _by_hand(net, data, _eps_target)
    assert not torch.allclose(post.data_fit[0], eps.data_fit[0])


def test_sample_prediction_target_is_x0():
    net, data = _unet(), _images()
    sched = DDPMScheduler(num_train_timesteps=STEPS, prediction_type="sample")
    post = lk.diffusers.laplace(net, sched, data, generator=_gen())

    _assert_same(post, _by_hand(net, data, _x0_target))

    eps = _by_hand(net, data, _eps_target)
    assert not torch.allclose(post.data_fit[0], eps.data_fit[0])


class _NoVelocity:
    """A variance-preserving scheduler with only `config` and `add_noise` (no `get_velocity`)."""

    def __init__(self, prediction: str) -> None:
        self._inner = DDPMScheduler(num_train_timesteps=STEPS, prediction_type=prediction)

    @property
    def config(self) -> Any:
        return self._inner.config

    def add_noise(self, original_samples: Any, noise: Any, timesteps: Any, /) -> Any:
        return self._inner.add_noise(original_samples, noise, timesteps)


def test_epsilon_scheduler_without_get_velocity_fits():
    net, data = _unet(), _images()
    post = lk.diffusers.laplace(net, _NoVelocity("epsilon"), data, generator=_gen())

    _assert_same(post, _by_hand(net, data, _eps_target, subset=lk.subsets.last_layer))


def test_sample_scheduler_without_get_velocity_fits():
    net, data = _unet(), _images()
    post = lk.diffusers.laplace(net, _NoVelocity("sample"), data, generator=_gen())

    _assert_same(post, _by_hand(net, data, _x0_target, subset=lk.subsets.last_layer))


def test_class_conditional_passes_labels():
    net = _unet(num_class_embeds=3)
    data = _labelled(_images(), "class_labels", lambda n, g: torch.randint(0, 3, (n,), generator=g))
    post = lk.diffusers.laplace(
        net, DDPMScheduler(num_train_timesteps=STEPS), data, generator=_gen()
    )

    _assert_same(post, _by_hand(net, data, _eps_target))

    # The labels reach the model: other labels give another fit.
    other = [{**d, "class_labels": (d["class_labels"] + 1) % 3} for d in data]
    moved = lk.diffusers.laplace(
        net, DDPMScheduler(num_train_timesteps=STEPS), other, generator=_gen()
    )
    assert not torch.allclose(moved.data_fit[0], post.data_fit[0])


@ATTENTION_VMAP
def test_text_conditional_passes_encoder_hidden_states():
    net = _text_unet()
    data = _labelled(
        _images(),
        "encoder_hidden_states",
        lambda n, g: torch.randn(n, 3, 8, generator=g, dtype=F64),
    )
    post = lk.diffusers.laplace(
        net, DDPMScheduler(num_train_timesteps=STEPS), data, generator=_gen()
    )

    assert set(post.mean) == {"conv_out.weight", "conv_out.bias"}
    _assert_same(post, _by_hand(net, data, _eps_target))


@ATTENTION_VMAP
def test_dit_fixed_variance():
    net = _dit()
    data = _labelled(
        _images(size=4), "class_labels", lambda n, g: torch.randint(0, 2, (n,), generator=g)
    )
    post = lk.diffusers.laplace(
        net, DDPMScheduler(num_train_timesteps=STEPS), data, generator=_gen()
    )

    assert set(post.mean) == {"proj_out_2.weight", "proj_out_2.bias"}
    _assert_same(post, _by_hand(net, data, _eps_target))


def test_size_generator_and_options_pass_through():
    net, data = _unet(), _images()
    options: dict[str, Any] = {
        "n_data": 100,
        "subset": {"conv_out.bias": net.conv_out.bias},
        "hessian_approx": lk.DenseGGN(),
        "prior": lk.priors.Isotropic(2.0),
        "hyper_tuner": None,
        "generator": torch.Generator().manual_seed(7),
    }
    post = lk.diffusers.laplace(net, DDPMScheduler(num_train_timesteps=STEPS), data, **options)
    options["generator"] = torch.Generator().manual_seed(7)  # the first fit advanced it

    assert set(post.mean) == {"conv_out.bias"}
    torch.testing.assert_close(post.n_data, torch.tensor(100.0, dtype=F64))
    torch.testing.assert_close(post.prior.prior_precision, torch.tensor(2.0, dtype=F64))
    _assert_same(post, _by_hand(net, data, _eps_target, **options))


# --- checks before the fit -------------------------------------------------------------------


class _Counting:
    """A loader that counts the batches it hands out."""

    def __init__(self, batches: list[Any]) -> None:
        self.batches = batches
        self.read = 0

    def __iter__(self):
        for b in self.batches:
            self.read += 1
            yield b


class _Untouchable:
    def __iter__(self):
        raise AssertionError("data was read")


def test_learned_variance_raises_before_fit():
    """`out_channels = 2 * in_channels` predicts noise and variance; only one batch is read."""

    data = _Counting(_images())

    with pytest.raises(lk.StructureError) as e:
        lk.diffusers.laplace(_unet(out_channels=2), DDPMScheduler(num_train_timesteps=STEPS), data)

    assert "(4, 2, 8, 8)" in str(e.value) and "(4, 1, 8, 8)" in str(e.value)
    assert "learned variance (`out_channels == 2 * in_channels`)" in str(e.value)
    assert data.read == 1


def test_flow_matching_scheduler_raises_before_data():
    with pytest.raises(lk.StructureError, match="flow matching is not supported yet"):
        lk.diffusers.laplace(_unet(), FlowMatchEulerDiscreteScheduler(), _Untouchable())


def test_scheduler_without_get_velocity_names_it_before_data():
    """DPM-Solver has add_noise but no get_velocity; it is not flow matching."""

    sched = DPMSolverMultistepScheduler(num_train_timesteps=STEPS, prediction_type="v_prediction")

    with pytest.raises(lk.StructureError) as e:
        lk.diffusers.laplace(_unet(), sched, _Untouchable())

    assert (
        "prediction_type 'v_prediction' needs scheduler.get_velocity; "
        "DPMSolverMultistepScheduler has none"
    ) in str(e.value)
    assert "flow matching" not in str(e.value)


def test_v_prediction_without_get_velocity_names_it_before_data():
    with pytest.raises(lk.StructureError) as e:
        lk.diffusers.laplace(_unet(), _NoVelocity("v_prediction"), _Untouchable())

    assert (
        "prediction_type 'v_prediction' needs scheduler.get_velocity; _NoVelocity has none"
    ) in str(e.value)


def test_variance_exploding_scheduler_raises_before_data():
    """Euler has add_noise and get_velocity, but its add_noise is x0 + sigma * eps."""

    sched = EulerDiscreteScheduler(num_train_timesteps=STEPS)

    with pytest.raises(lk.StructureError) as e:
        lk.diffusers.laplace(_unet(), sched, _Untouchable())

    assert "variance-preserving (DDPM-family)" in str(e.value)
    assert "EulerDiscreteScheduler" in str(e.value)


@pytest.mark.parametrize("prediction", ["flow_prediction", "learned"])
def test_unknown_prediction_type_raises_before_data(prediction: str):
    sched = DDPMScheduler(num_train_timesteps=STEPS, prediction_type=prediction)

    with pytest.raises(lk.StructureError) as e:
        lk.diffusers.laplace(_unet(), sched, _Untouchable())

    assert prediction in str(e.value)
    assert all(p in str(e.value) for p in ("'epsilon'", "'v_prediction'", "'sample'"))


@pytest.mark.parametrize(
    "batch, match",
    [
        ({"class_labels": torch.zeros(4, dtype=torch.long)}, "'sample'"),
        ({"sample": torch.zeros(4, 1, 8, 8, dtype=F64), "timestep": torch.zeros(4)}, "'timestep'"),
        ((torch.zeros(4, 1, 8, 8), torch.zeros(4)), "tuple"),
    ],
)
def test_batch_needs_samples(batch: Any, match: str):
    with pytest.raises(lk.StructureError, match=match):
        lk.diffusers.laplace(_unet(), DDPMScheduler(num_train_timesteps=STEPS), [batch])


def test_empty_data_raises():
    with pytest.raises(ValueError, match="no batches"):
        lk.diffusers.laplace(_unet(), DDPMScheduler(num_train_timesteps=STEPS), [])


# --- in the sampling loop --------------------------------------------------------------------


def test_outputs_in_sampling_loop_is_factor():
    net = _unet()
    sched = DDPMScheduler(num_train_timesteps=STEPS)
    post = lk.diffusers.laplace(net, sched, _images(), generator=_gen())
    sched.set_timesteps(3)
    g = torch.Generator().manual_seed(2)
    x = torch.randn(3, 1, 8, 8, generator=g, dtype=F64)

    for t in sched.timesteps:
        assert t.ndim == 0
        out = lk.predict_outputs(post, {"sample": x, "timestep": t}, strategy="forward-mode")

        assert isinstance(out.cov, lk.core.FactorCov)
        assert out.cov.diagonal().shape == x.shape
        assert (out.cov.diagonal() >= 0).all()

        with torch.no_grad():
            torch.testing.assert_close(out.mean, net(x, t).sample)

        x = sched.step(out.mean, t, x, generator=g).prev_sample


# --- plugin record and health ----------------------------------------------------------------


def test_reached_as_sl_diffusers():
    assert lk.diffusers is laplacekit_diffusers
    assert "plugin" not in laplacekit_diffusers.__all__
    assert laplacekit_diffusers.plugin.health is laplacekit_diffusers.health


def test_health_all_installed_is_ok():
    (check,) = laplacekit_diffusers.health()

    assert check.name == "diffusers" and check.status == "ok"
    assert metadata.version("diffusers") in check.detail


def test_health_diffusers_missing_is_error(monkeypatch):
    real = metadata.version

    def version(dist):
        if dist == "diffusers":
            raise metadata.PackageNotFoundError(dist)

        return real(dist)

    monkeypatch.setattr(metadata, "version", version)
    (check,) = laplacekit_diffusers.health()

    assert check.status == "error"
    assert "not installed" in check.detail
