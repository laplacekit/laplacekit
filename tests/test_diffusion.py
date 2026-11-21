"""Scenario 14: last-layer Laplace on a tiny diffusers UNet, inside a sampling loop.

The UNet and the scheduler are built from configs, never downloaded. The last layer is
`conv_out` (8 -> 1 channels, 3x3): P = 73 parameters against E = 64 outputs per image.
"""

from typing import Any

import pytest
import torch
from diffusers.models.unets.unet_2d import UNet2DModel
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from torch import Tensor
from torch.func import functional_call, jacrev

import laplacekit as lk
from laplacekit.core import Strategy

F64 = torch.float64
N_TRAIN, N_SAMPLE, STEPS = 8, 3, 5


def _unet(add_attention: bool = False) -> UNet2DModel:
    torch.manual_seed(0)

    return UNet2DModel(
        sample_size=8,
        in_channels=1,
        out_channels=1,
        layers_per_block=1,
        block_out_channels=(8, 16),
        norm_num_groups=4,
        down_block_types=("DownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "UpBlock2D"),
        add_attention=add_attention,
    ).eval()


def _fit(unet: Any, dtype: torch.dtype = torch.float32) -> tuple[lk.LaplacePosterior, Any]:
    """The scenario's fit, verbatim up to the toy images."""

    sched: Any = DDPMScheduler(num_train_timesteps=50)

    def forward(params: lk.core.Params, inp: Any) -> Tensor:
        return functional_call(unet, dict(params), (), inp).sample

    fn = lk.core.ModelFn(forward, module=unet)

    def prepare(images: Tensor, g: torch.Generator, /) -> lk.Batch:
        t = torch.randint(0, sched.config.num_train_timesteps, (len(images),), generator=g)
        noise = torch.randn(images.shape, generator=g, dtype=dtype)
        return lk.Batch(
            inputs={"sample": sched.add_noise(images, noise, t), "timestep": t}, targets=noise
        )

    g = torch.Generator().manual_seed(1)
    image_loader = [torch.randn(4, 1, 8, 8, generator=g, dtype=dtype) for _ in range(2)]

    post = lk.laplace(
        fn,
        lk.data.from_loader(image_loader, prepare=prepare),
        likelihood=lk.Gaussian(sigma=1.0, event_dims=3),
        subset=lk.subsets.last_layer,
    )

    return post, sched


def test_last_layer_is_conv_out() -> None:
    post, _ = _fit(_unet())

    assert set(post.mean) == {"conv_out.weight", "conv_out.bias"}
    assert sum(p.numel() for p in post.mean.values()) == 73


@pytest.mark.parametrize("strategy", [None, "forward-mode"])
def test_scenario_14_runs(strategy: Strategy | None) -> None:
    unet = _unet()
    post, sched = _fit(unet)
    sched.set_timesteps(STEPS)
    g = torch.Generator().manual_seed(2)
    x = torch.randn(N_SAMPLE, 1, 8, 8, generator=g)
    kind = lk.core.FactorCov if strategy == "forward-mode" else lk.core.PerInputCov

    assert len(sched.timesteps) == STEPS

    for t in sched.timesteps:
        assert t.ndim == 0
        out = lk.predict_outputs(post, {"sample": x, "timestep": t}, strategy=strategy)
        eps_var = out.cov.diagonal()

        assert isinstance(out.cov, kind)
        assert isinstance(out.mean, Tensor)
        assert eps_var.shape == x.shape
        assert torch.isfinite(eps_var).all()
        assert (eps_var >= 0).all()

        with torch.no_grad():
            torch.testing.assert_close(out.mean, unet(x, t).sample)

        other = lk.predict_outputs(
            post,
            {"sample": x, "timestep": t},
            strategy="reverse-mode" if strategy == "forward-mode" else "forward-mode",
        )
        torch.testing.assert_close(other.cov.diagonal(), eps_var, rtol=0, atol=1e-6)

        x = sched.step(out.mean, t, x, generator=g).prev_sample

    pred = lk.predict(post, {"sample": x, "timestep": sched.timesteps[-1]})

    assert pred.event_shape == (1, 8, 8)


def _dense_cov(post: lk.LaplacePosterior, names: list[str]) -> Tensor:
    """Sigma as a dense [P, P] matrix, one `cov_mv` per unit vector."""

    sizes = [post.mean[n].numel() for n in names]
    cols = []

    for i in range(sum(sizes)):
        e = torch.zeros(sum(sizes), dtype=F64)
        e[i] = 1.0
        parts = torch.split(e, sizes)
        v = {n: p.reshape(post.mean[n].shape) for n, p in zip(names, parts, strict=True)}
        cols.append(torch.cat([post.cov_mv(v)[n].reshape(-1) for n in names]))

    return torch.stack(cols, 1)


@pytest.mark.parametrize("strategy", ["forward-mode", "reverse-mode"])
def test_unet_outputs_equal_dense_reference(strategy: Strategy) -> None:
    unet = _unet().to(F64)
    post, _ = _fit(unet, F64)
    names = list(post.mean)
    shapes = [post.mean[n].shape for n in names]
    sizes = [post.mean[n].numel() for n in names]
    x = torch.randn(N_SAMPLE, 1, 8, 8, generator=torch.Generator().manual_seed(2), dtype=F64)
    t = torch.tensor(17)

    def f(flat: Tensor, xi: Tensor) -> Tensor:
        parts = torch.split(flat, sizes)
        p = {n: q.reshape(s) for n, q, s in zip(names, parts, shapes, strict=True)}
        return functional_call(unet, p, (xi.unsqueeze(0), t)).sample.reshape(-1)

    w = torch.cat([post.mean[n].reshape(-1) for n in names])
    jac = torch.stack([jacrev(f)(w, xi) for xi in x])  # [N, E, P]
    sigma = _dense_cov(post, names)

    assert jac.shape == (N_SAMPLE, 64, 73)

    out = lk.predict_outputs(post, {"sample": x, "timestep": t}, strategy=strategy)

    torch.testing.assert_close(out.cov.per_input(), jac @ sigma @ jac.mT, rtol=0, atol=1e-8)


# functorch has no batching rule for the fused CPU attention kernel (the probe's vmap on the
# attention UNet); vmap still runs, through the slow fallback.
@pytest.mark.filterwarnings(
    "ignore:There is a performance drop because we have not yet implemented the batching "
    "rule:UserWarning"
)
@pytest.mark.parametrize("attention", [False, True])
def test_unet_probe_jvp_needs_no_attention(attention: bool) -> None:
    unet = _unet(add_attention=attention)
    fn = lk.core.ModelFn(lambda p, inp: functional_call(unet, dict(p), (), inp).sample, module=unet)
    inputs = {"sample": torch.randn(2, 1, 8, 8), "timestep": torch.tensor([3, 40])}
    batch = lk.Batch(inputs, torch.randn(2, 1, 8, 8))
    caps = lk.core.probe(fn, lk.Gaussian(sigma=1.0, event_dims=3), batch)

    assert caps.jvp is not attention

    if attention:
        assert "forward AD" in caps.failures["jvp"]


@pytest.mark.parametrize("strategy", ["forward-mode", "reverse-mode"])
def test_int_input_leaf_is_shared(strategy: Strategy) -> None:
    """A Python int timestep, which diffusers accepts, acts like a 0-dim tensor."""

    post, _ = _fit(_unet())
    x = torch.randn(N_SAMPLE, 1, 8, 8, generator=torch.Generator().manual_seed(2))
    want = lk.predict_outputs(
        post, {"sample": x, "timestep": torch.tensor(17)}, strategy=strategy, n_samples="exact"
    )
    got = lk.predict_outputs(
        post, {"sample": x, "timestep": 17}, strategy=strategy, n_samples="exact"
    )

    torch.testing.assert_close(got.mean, want.mean, rtol=0, atol=0)
    torch.testing.assert_close(got.cov.per_input(), want.cov.per_input(), rtol=0, atol=0)
