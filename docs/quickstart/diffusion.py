# %% [markdown]
# ---
# title: Diffusion models
# jupyter: python3
# ---
#
# We take a pretrained DDPM on MNIST, fit a Laplace approximation to the last
# layer of its UNet, and carry a variance for every pixel through the sampling
# loop.
#
# ::: {.callout-note title="Plugin"}
# This page uses the
# [`laplacekit-diffusers`](https://github.com/laplacekit/laplacekit/tree/main/plugins/diffusers)
# plugin. Install it with
#
# ```bash
# pip install "laplacekit[diffusers]" torchvision
# ```
# :::

# %%
import warnings
from pathlib import Path

import huggingface_hub
import matplotlib.pyplot as plt
import torch
from diffusers import DDPMScheduler, UNet2DModel
from torchvision.datasets import MNIST

import laplacekit as lk

huggingface_hub.utils.disable_progress_bars()
huggingface_hub.logging.set_verbosity_error()
warnings.filterwarnings(
    "ignore", category=UserWarning
)  # speed and deprecation notices

REPO, REVISION = (
    "1aurent/ddpm-mnist",
    "e6ee37a720cd5c9b48fa5be809e5926f49f38246",
)
unet = UNet2DModel.from_pretrained(REPO, revision=REVISION).eval()
scheduler = DDPMScheduler.from_pretrained(REPO, revision=REVISION)

# %% [markdown]
# `torchvision` downloads MNIST once.

# %%
#| output: false
mnist = MNIST(Path.home() / ".cache" / "laplacekit" / "mnist", download=True)

# %% [markdown]
# The data are clean images, scaled to $[-1, 1]$. `lk.diffusers.laplace` adds
# the noise itself, as in training. We fit on 1,000 images and tell it that the
# training set has 60,000.

# %%
images = mnist.data[:1000].unsqueeze(1).float() / 127.5 - 1.0
loader = torch.utils.data.DataLoader(images, batch_size=250)

post = lk.diffusers.laplace(unet, scheduler, loader, n_data=60_000)
print(post)

# %% [markdown]
# We sample with 50 deterministic DDIM steps. At each step, `lk.predict_outputs` gives
# the predicted noise and its per-pixel variance, and we push the variance
# through the step.

# %%
alpha_bar = scheduler.alphas_cumprod
timesteps = list(range(980, -1, -20))

torch.manual_seed(0)
x = torch.randn(16, 1, 28, 28)
var = torch.zeros_like(x)

with torch.no_grad():
    for i, t in enumerate(timesteps):
        ab = alpha_bar[t]
        ab_next = (
            alpha_bar[timesteps[i + 1]]
            if i + 1 < len(timesteps)
            else torch.tensor(1.0)
        )

        eps = lk.predict_outputs(
            post, {"sample": x, "timestep": torch.tensor(t)}
        )
        x0 = (x - (1 - ab).sqrt() * eps.mean) / ab.sqrt()
        inside = (x0.abs() <= 1).float()

        dx = inside * (ab_next / ab).sqrt()
        de = (1 - ab_next).sqrt() - dx * (1 - ab).sqrt()

        x = ab_next.sqrt() * x0.clamp(-1, 1) + (1 - ab_next).sqrt() * eps.mean
        var = dx**2 * var + de**2 * eps.cov.diagonal()

# %%
fig, axes = plt.subplots(2, 8, figsize=(9, 2.6))

for k in range(8):
    axes[0, k].imshow(x[k, 0], cmap="gray", vmin=-1, vmax=1)
    axes[1, k].imshow(var[k, 0], cmap="magma")

for ax in axes.flat:
    ax.set_axis_off()

plt.tight_layout()
plt.show()

# %% [markdown]
# The top row shows the samples and the bottom row their variance maps, each
# on its own color scale. The maps look like speckle. Most of the variance
# comes from the first steps, where many pixels of $\hat x_0$ are clipped, and
# the full tutorial explains why.
#
# ::: {.callout-note collapse="true" title="The math"}
# The UNet $\epsilon_\theta(x_t, t)$ predicts the noise in $x_t$. Linearized
# around the fitted weights, the Laplace posterior makes the predicted noise
# Gaussian, $\mathcal{N}(\epsilon_{\theta_*}(x_t, t), J \Sigma J^\top)$, and we
# keep the diagonal of its covariance. One DDIM step from $t$ to $t'$ is
#
# $$
#   \hat x_0 = \frac{x_t - \sqrt{1 - \bar\alpha_t} \, \hat\epsilon}{\sqrt{\bar\alpha_t}} , \qquad
#   x_{t'} = \sqrt{\bar\alpha_{t'}} \, \mathrm{clip}(\hat x_0) + \sqrt{1 - \bar\alpha_{t'}} \, \hat\epsilon .
# $$
#
# Let $m$ be $1$ where $\hat x_0$ lies in $[-1, 1]$ and $0$ where it is
# clipped. Linearizing the step pixel by pixel,
#
# $$
#   \mathrm{Var}(x_{t'}) \approx \Big(m \sqrt{\tfrac{\bar\alpha_{t'}}{\bar\alpha_t}}\Big)^2 \mathrm{Var}(x_t) + \Big(\sqrt{1 - \bar\alpha_{t'}} - m \sqrt{\tfrac{\bar\alpha_{t'}}{\bar\alpha_t}} \sqrt{1 - \bar\alpha_t}\Big)^2 \mathrm{Var}(\hat\epsilon) ,
# $$
#
# which drops the covariance between $x_t$ and $\hat\epsilon$. This is the
# idea of BayesDiff ([Kou et al., 2024](https://arxiv.org/abs/2310.11142)).
# :::
#
# The full tutorial: [Diffusion models](../examples/diffusion.py)
