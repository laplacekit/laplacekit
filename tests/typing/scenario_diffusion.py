"""Last-layer Laplace on a diffusers UNet, used inside the user's sampling loop."""

from typing import Any

import torch
from _fixtures import image_loader
from diffusers.models.unets.unet_2d import UNet2DModel
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from torch import Tensor
from torch.func import functional_call

import laplacekit as lk

unet: Any = UNet2DModel.from_pretrained("google/ddpm-cifar10-32")
sched: Any = DDPMScheduler.from_pretrained("google/ddpm-cifar10-32")


def forward(params: lk.core.Params, inp: Any) -> Tensor:
    return functional_call(unet, dict(params), (), inp).sample


fn = lk.core.ModelFn(forward, module=unet)


def prepare(images: Tensor, g: torch.Generator, /) -> lk.Batch:
    t = torch.randint(0, sched.config.num_train_timesteps, (len(images),), generator=g)
    noise = torch.randn(images.shape, generator=g)
    return lk.Batch(
        inputs={"sample": sched.add_noise(images, noise, t), "timestep": t}, targets=noise
    )


post = lk.laplace(
    fn,
    lk.data.from_loader(image_loader, prepare=prepare),
    likelihood=lk.Gaussian(sigma=1.0, event_dims=3),
    subset=lk.subsets.last_layer,
)

# The same fit through the preset, which also covers v- and x0-prediction schedulers.
post = lk.diffusers.laplace(unet, sched, image_loader)
post = lk.diffusers.laplace(
    unet,
    sched,
    image_loader,
    n_data=50_000,
    hessian_approx=lk.DenseGGN(),
    hyper_tuner=None,
    generator=torch.Generator().manual_seed(1),
)
labelled = [
    {"sample": x0, "class_labels": torch.zeros(len(x0), dtype=torch.long)} for x0 in image_loader
]
post = lk.diffusers.laplace(unet, sched, labelled, subset=lk.subsets.last_layer)
fn = lk.diffusers.denoiser(unet)

x = torch.randn(16, 3, 32, 32)
for t in sched.timesteps:
    out = lk.predict_outputs(
        post, {"sample": x, "timestep": t}
    )  # Gaussian over the predicted noise, no σ²
    eps_var: Tensor = out.cov.diagonal()  # shaped like the image
    jvp_cov = lk.predict_outputs(post, {"sample": x, "timestep": t}, strategy="forward-mode").cov
    assert isinstance(jvp_cov, lk.core.FactorCov)
    as_cov: lk.core.OutputCov = jvp_cov  # a FactorCov is an OutputCov
    assert isinstance(out.mean, Tensor)
    x = sched.step(out.mean, t, x).prev_sample
