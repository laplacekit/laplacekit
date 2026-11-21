# %% [markdown]
# ---
# title: Diffusion models
# jupyter: python3
# ---
#
# ::: {.callout-tip title="In short"}
# ```{.python}
# post = lk.diffusers.laplace(unet, scheduler, loader, n_data=60_000)  # last layer, unit noise
# eps = lk.predict_outputs(post, {"sample": x_t, "timestep": torch.tensor(t)})
# eps.mean, eps.cov.diagonal()  # predicted noise and its variance, per pixel
# ```
# :::
#
# The goal of this page is to put a variance on every pixel of an image that a
# diffusion model generates. We take a pretrained denoising diffusion model for
# MNIST digits, fit a Laplace approximation to its last layer, and carry the
# uncertainty of the predicted noise through the sampling loop. This is the
# idea of BayesDiff, from Kou et al. (2024)[^kou2024]. Nothing is trained on
# this page.
#
# Recall how a DDPM (Ho et al., 2020)[^ho2020] works. For an image $x_0$, the
# forward process adds Gaussian noise in $T$ steps, and has a closed form at
# every step $t$,
#
# $$
#   x_t = \sqrt{\bar\alpha_t} \, x_0 + \sqrt{1 - \bar\alpha_t} \, \epsilon , \qquad \epsilon \sim \mathcal{N}(0, I) ,
# $$
#
# with $\bar\alpha_t$ decreasing from almost $1$ to almost $0$. A network
# $\epsilon_\theta(x_t, t)$ learns to predict the noise $\epsilon$ from $x_t$
# and $t$, by least squares. To generate an image, we start from pure noise
# $x_T$ and remove the predicted noise step by step.
#
# The Laplace approximation gives a Gaussian over $\theta$, $\mathcal{N}(\theta^*,
# \Sigma)$. Linearizing the network around $\theta^*$ then makes the predicted
# noise Gaussian too,
#
# $$
#   \epsilon_t \sim \mathcal{N}\big(\epsilon_{\theta^*}(x_t, t), \; J \Sigma J^\top\big) ,
# $$
#
# where $J$ is the Jacobian of the network output with respect to $\theta$, at
# $(x_t, t)$. In what follows, only the diagonal of $J \Sigma J^\top$ matters,
# i.e. one variance per pixel.

# %%
import time
import warnings
from pathlib import Path

import huggingface_hub
import matplotlib.pyplot as plt
import torch
from diffusers import DDPMScheduler, UNet2DModel
from torch.func import functional_call
from torchvision.datasets import MNIST

import laplacekit as lk

huggingface_hub.utils.disable_progress_bars()
huggingface_hub.logging.set_verbosity_error()

# torch.func's vmap has no batching rule for the fused attention kernel on CPU, so the fit
# falls back to a slower loop there. The result is the same.
warnings.filterwarnings(
    "ignore",
    message=r"There is a performance drop.*_scaled_dot_product",
    category=UserWarning,
)

# %% [markdown]
# ## Model and data
#
# The model is `1aurent/ddpm-mnist` from the Hugging Face Hub, pinned to one
# revision. It is a `diffusers` UNet on $28 \times 28$ grayscale images, with
# three resolutions ($28 \to 14 \to 7$), 32, 64 and 32 channels, and
# self-attention at $14 \times 14$ and $7 \times 7$. Its
# scheduler is the usual DDPM one: $T = 1000$ steps, $\beta_t$ linear from
# $10^{-4}$ to $0.02$, and the network predicts the noise $\epsilon$.

# %%
REPO, REVISION = (
    "1aurent/ddpm-mnist",
    "e6ee37a720cd5c9b48fa5be809e5926f49f38246",
)

with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore", message="The `local_dir_use_symlinks`", category=UserWarning
    )
    unet = UNet2DModel.from_pretrained(REPO, revision=REVISION).eval()
    scheduler = DDPMScheduler.from_pretrained(REPO, revision=REVISION)

T = scheduler.config.num_train_timesteps
print("parameters:", sum(p.numel() for p in unet.parameters()))
print(
    "prediction:",
    scheduler.config.prediction_type,
    "| clips x0:",
    scheduler.config.clip_sample,
)

# %% [markdown]
# The data are the 60,000 training images of MNIST[^lecun1998], which
# `torchvision` downloads once into a local cache. We scale the pixels to $[-1,
# 1]$, as in the model's training.

# %%
#| output: false
mnist = MNIST(
    Path.home() / ".cache" / "laplacekit" / "mnist", train=True, download=True
)

# %%
images = mnist.data.unsqueeze(1).float() / 127.5 - 1.0
print(images.shape)

# %% [markdown]
# ## Fitting the last layer
#
# The training loss is the squared error between $\epsilon$ and
# $\epsilon_\theta(x_t, t)$, i.e. a Gaussian log-likelihood on the noise, with
# unit variance and the $28 \times 28$ pixels of one image as one event. We fit
# the Laplace approximation to the last layer of the UNet only, `conv_out`, a $3
# \times 3$ convolution from 32 channels to 1, with $P = 289$ parameters.
#
# `lk.diffusers.laplace` takes the UNet, its scheduler and a loader of clean
# images. For every batch, it draws a timestep and a noise per image, noises the
# images as in training, and uses the noise as the target. The curvature is a
# per-example average, so we estimate it on 1,000 images instead of all 60,000,
# which only makes the estimate noisier. The data size of the posterior, `n_data` $= N$, is
# still the size of the training set: we pass it as `n_data`. The marginal
# likelihood picks the prior precision $\lambda$. A seeded `generator` fixes the
# timesteps and noises, so that we can draw the same ones again below.

# %%
loader = torch.utils.data.DataLoader(images[:1000], batch_size=250)

start = time.perf_counter()
post = lk.diffusers.laplace(
    unet,
    scheduler,
    loader,
    n_data=len(images),
    generator=torch.Generator().manual_seed(0),
)
fit_time = time.perf_counter() - start

print(post)

# %% [markdown]
# `lk.curvature.default` picked the dense GGN, i.e. the full $289 \times 289$
# matrix, since the block has at most 4096 parameters. The marginal likelihood
# chose $\lambda \approx 650$.
#
# ### Under the hood
#
# `lk.diffusers.laplace` is `lk.laplace` with three things filled in. The model
# function calls the UNet with the noisy sample and the timestep and returns
# the `.sample` of its output. The `prepare` function turns a batch of clean
# images into pairs of a noisy image and its noise, drawn from the batch's own
# `torch.Generator`, and puts the timestep of each image next to it in the inputs. The
# likelihood is `lk.Gaussian(sigma=1.0, event_dims=3)`: fixed unit noise, with
# channel, height and width as one event. Written out, the fit above is the
# following.


# %%
def forward(params, inputs):
    return functional_call(
        unet, dict(params), (inputs["sample"], inputs["timestep"])
    ).sample


def prepare(x0, g, /):
    t = torch.randint(0, T, (len(x0),), generator=g)
    eps = torch.randn(x0.shape, generator=g)

    return lk.Batch(
        {"sample": scheduler.add_noise(x0, eps, t), "timestep": t}, eps
    )


by_hand = lk.laplace(
    lk.core.ModelFn(forward, unet),
    lk.data.from_loader(loader, prepare=prepare, n_data=len(images)),
    likelihood=lk.Gaussian(sigma=1.0, event_dims=3),
    subset=lk.subsets.last_layer,
    generator=torch.Generator().manual_seed(0),
)
print(
    "prior precision:",
    float(by_hand.prior.prior_precision),
    float(post.prior.prior_precision),
)

# %% [markdown]
# Both fits get a generator with the same seed, so they draw the same
# timesteps and noises and agree. On top of this, `lk.diffusers.laplace` checks
# the scheduler before it reads any data: its `prediction_type` must be one
# that it knows (`"epsilon"` here; for `"v_prediction"` and `"sample"`, the
# target changes), and its forward process must be the variance-preserving one
# above. It also runs one batch through the UNet to check that the output has
# the shape of the target.
#
# ## The predicted noise and its variance
#
# For one step of the sampling loop, we need the diagonal of $J \Sigma J^\top$
# for a batch of images $x_t$ at one timestep $t$. Here $J$ is $E \times P$ per
# image, with $E = 784$ pixels and $P = 289$ parameters. Let $L$ be a square
# root of the covariance, $L L^\top = \Sigma$. Then
#
# $$
#   J \Sigma J^\top = (J L) (J L)^\top ,
# $$
#
# and the $P$ columns of $J L$ are Jacobian-vector products, $J L e_p$ for the
# unit vectors $e_p$. Forward-mode autodiff computes all $P$ of them in one
# batched pass through the network. Only the last layer has a tangent, so that
# pass costs about one forward pass plus a batched last convolution. The
# variance of pixel $k$ is then the squared norm of row $k$ of $J L$.
#
# `lk.predict_outputs` does this. Its result is a Gaussian whose `cov` is a
# `lk.core.FactorCov`, holding the factor $J L$ of shape $[B, 1, 28, 28, P]$ for a
# batch of $B$ images, and `cov.diagonal()` has the shape of the images. In a
# sampling loop, $t$ is the same for all images, so we pass it as a 0-dim
# tensor in the inputs and **laplacekit** shares it across the batch.
#
# ### Factor or dense blocks
#
# There are two exact paths, and `strategy=` picks one of them. With `strategy="reverse-mode"`, the
# Jacobian comes row by row from vector-Jacobian products, and the result is a
# `lk.core.PerInputCov` with one dense $E \times E$ block per input. That costs $E$
# VJPs and $E^2$ entries per input. With `strategy="forward-mode"`, the result is the factor
# above, with $P$ JVPs and $E P$ entries per input. Both give the same
# covariance. The default, `lk.core.resolve_strategy`, takes forward mode when the
# probe found it and $P < E$, as here.
#
# The probe tries forward mode with respect to the fitted parameters only. This
# matters for this UNet: its attention uses PyTorch's fused kernel, which has no
# forward-mode derivative. A tangent on all the weights would have to pass
# through it and fail, but a tangent on `conv_out` starts after the last
# attention layer, so the probe finds forward mode.
#
# A `FactorCov` gives `diagonal()` and `sample()` for any $E$. The dense blocks,
# `per_input()`, need $E \le 4096$, which is $3 \times 64 \times 64$ for a color
# image. Above that, `per_input()` raises a `StructureError`.

# %%
x = torch.randn(4, 1, 28, 28)
out = lk.predict_outputs(post, {"sample": x, "timestep": torch.tensor(500)})
print(
    type(out.cov).__name__,
    tuple(out.cov.factor.shape),
    tuple(out.cov.diagonal().shape),
)

# %% [markdown]
# ## Sampling with a variance
#
# We sample with DDIM (Song et al., 2021)[^song2021], deterministic and with 50
# timesteps $t = 980, 960, \dots, 0$ instead of 1000. A step from $t$ to the
# next timestep $t'$ first estimates the clean image from the predicted noise,
#
# $$
#   \hat x_0 = \frac{x_t - \sqrt{1 - \bar\alpha_t} \, \hat\epsilon}{\sqrt{\bar\alpha_t}} ,
# $$
#
# clips it to the pixel range $[-1, 1]$, as the model's own scheduler does
# (`clip_sample` above), and then noises it again to level $t'$,
#
# $$
#   x_{t'} = \sqrt{\bar\alpha_{t'}} \, \mathrm{clip}(\hat x_0) + \sqrt{1 - \bar\alpha_{t'}} \, \hat\epsilon ,
# $$
#
# with $\bar\alpha_{t'} = 1$ after the last step. At $t = 980$,
# $\sqrt{\bar\alpha_t} \approx 0.008$, so a small error in $\hat\epsilon$
# becomes a large error in $\hat x_0$, and the clip keeps it in range.
#
# The variance follows from the same step, linearized. Let $m$ be $1$ at the
# pixels where $\hat x_0$ lies inside $[-1, 1]$ and $0$ where it is clipped.
# Then, pixel by pixel,
#
# $$
#   \frac{\partial x_{t'}}{\partial x_t} = m \sqrt{\frac{\bar\alpha_{t'}}{\bar\alpha_t}} , \qquad
#   \frac{\partial x_{t'}}{\partial \hat\epsilon} = \sqrt{1 - \bar\alpha_{t'}} - m \sqrt{\frac{\bar\alpha_{t'}}{\bar\alpha_t}} \sqrt{1 - \bar\alpha_t} ,
# $$
#
# and
#
# $$
#   \mathrm{Var}(x_{t'}) \approx \Big(\frac{\partial x_{t'}}{\partial x_t}\Big)^2 \mathrm{Var}(x_t) + \Big(\frac{\partial x_{t'}}{\partial \hat\epsilon}\Big)^2 \mathrm{Var}(\epsilon_t) .
# $$
#
# A clipped pixel carries no variance forward from $x_t$: its $\hat x_0$ sits at
# the boundary whatever $x_t$ is, so only the noise term remains.
#
# Two more approximations are in there. First, we drop the covariance term
# between $x_t$ and $\epsilon_t$. Second, we take $\mathrm{Var}(\epsilon_t)$ at
# the mean path $x_t$ only, while $x_t$ is itself uncertain. BayesDiff puts both
# back by Monte Carlo, with samples of $x_t$ at every step. We run the loop
# under `torch.no_grad()`, since we need no gradients here. Besides the
# variance, the loop keeps the share of unclipped pixels per step, the last
# step at which each pixel was clipped, and the variance that only the last
# five steps add.

# %%
alpha_bar = scheduler.alphas_cumprod
timesteps = list(range(980, -1, -20))

torch.manual_seed(1)
x = torch.randn(16, 1, 28, 28)
var = torch.zeros_like(x)
late = torch.zeros_like(x)
clipped_at = torch.full_like(x, -1)
inside = {}

start = time.perf_counter()

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
        m = (x0.abs() <= 1).float()
        inside[t] = float(m.mean())
        clipped_at[m == 0] = t

        dx = m * (ab_next / ab).sqrt()
        de = (1 - ab_next).sqrt() - dx * (1 - ab).sqrt()

        x = ab_next.sqrt() * x0.clamp(-1, 1) + (1 - ab_next).sqrt() * eps.mean
        var = dx**2 * var + de**2 * eps.cov.diagonal()

        if t <= 80:
            late = dx**2 * late + de**2 * eps.cov.diagonal()

        if t == 20:
            eps_var_20 = eps.cov.diagonal()

sample_time = time.perf_counter() - start

# %% [markdown]
# Below are the first eight samples (top row) and their variance maps (middle
# row). The bottom row is the variance that only the last five steps add, i.e.
# the same recursion started from zero at $t = 80$. Each row has its own color
# scale.

# %%
fig, axes = plt.subplots(3, 8, figsize=(9, 3.6))

for k in range(8):
    axes[0, k].imshow(x[k, 0], cmap="gray", vmin=-1, vmax=1)
    axes[1, k].imshow(
        var[k, 0], cmap="magma", vmin=0, vmax=float(var[:8].max())
    )
    axes[2, k].imshow(
        late[k, 0], cmap="magma", vmin=0, vmax=float(late[:8].max())
    )

for ax in axes.flat:
    ax.set_axis_off()

plt.tight_layout()
plt.show()

# %% [markdown]
# We start with the middle row. The lines below give the weight of
# $\mathrm{Var}(\epsilon_t)$ at the first step, the telescoping factor of the
# next paragraph, the share of unclipped pixels at three steps, and the mean
# final variance of the pixels, grouped by the last step at which each was
# clipped.

# %%
a, a_next = alpha_bar[980], alpha_bar[960]
w_inside = float(
    ((1 - a_next).sqrt() - (a_next / a).sqrt() * (1 - a).sqrt()) ** 2
)
w_clipped = float(1 - a_next)

print(
    f"weight of Var(eps) at t = 980: unclipped {w_inside:.2f}, clipped "
    f"{w_clipped:.2f}"
)
print(f"1 / alpha_bar at t = 980: {float(1 / alpha_bar[980]):,.0f}")
print(
    "unclipped pixels:",
    ", ".join(f"t = {t} {inside[t]:.0%}" for t in (980, 780, 500)),
)

groups = {
    "never": clipped_at < 0,
    "t >= 900": clipped_at >= 900,
    "500 <= t < 900": (clipped_at >= 500) & (clipped_at < 900),
    "t < 500": (clipped_at >= 0) & (clipped_at < 500),
}

for name, where in groups.items():
    share, mean = float(where.float().mean()), float(var[where].mean())
    print(
        f"last clipped {name:>14}: {share:4.0%} of pixels, mean variance "
        f"{mean:.1e}"
    )

# %% [markdown]
# The full variance looks like speckle and shows no digit. It is set in the
# first steps, and clipping decides where it lands. $\mathrm{Var}(\epsilon_t)$
# enters with the weight $(\partial x_{t'} / \partial \hat\epsilon)^2$. For an
# unclipped pixel, the two terms of that derivative nearly cancel, and at the
# first step the weight is $0.05$. For a clipped pixel, $m = 0$ and the weight
# is $1 - \bar\alpha_{t'}$, i.e. $1$ (the first line above). Whatever variance
# a pixel takes in then grows while it stays unclipped, since the factors
# $\sqrt{\bar\alpha_{t'} / \bar\alpha_t}$ telescope: from $t = 980$ to the
# end, they multiply a variance by $1 / \bar\alpha_{980} \approx 17{,}000$.
#
# So a clip does two things to a pixel. It drops the variance the pixel
# carried, and it takes in the noise with about 20 times the weight an
# unclipped pixel would. A pixel clipped for the last time at an early step
# then grows this larger variance over almost all the remaining steps. At $t =
# 980$, only a third of the pixels have $\hat x_0$ inside $[-1, 1]$, and by $t =
# 780$, three quarters do. The pixels last clipped at $t \ge 900$ end with a
# mean variance of $0.03$, five times that of the pixels that were never
# clipped. Those last clipped after $t = 500$ have few steps left to grow and
# end with about $2 \times 10^{-5}$. Which pixels are clipped at the first
# steps depends on the starting noise, hence the speckle.
#
# Now the last five steps, in the bottom row. We split the pixels of the
# samples into strokes ($x > -0.8$) and background, and look separately at a
# border two pixels wide.

# %%
stroke = x > -0.8
border = torch.ones_like(stroke)
border[..., 2:-2, 2:-2] = False
last = clipped_at == 0


def ratio(v, a, b):
    return float(v[a].mean() / v[b].mean())


print(
    f"mean pixel variance: all steps {float(var.mean()):.1e}, last five "
    f"{float(late.mean()):.1e}"
)
print(
    f"strokes / background, last five steps: {ratio(late, stroke, ~stroke):.2f}"
)
print(
    "strokes / background, Var(eps) at t = 20: "
    f"{ratio(eps_var_20, stroke, ~stroke):.2f}"
)
print(
    "background, border / inside: "
    f"{ratio(late, border & ~stroke, ~border & ~stroke):.2f}"
)
print(
    f"clipped at the last step: {float(last.float().mean()):.1%} of pixels,",
    end=" ",
)
print(f"{float(stroke[last].float().mean()):.0%} of them on strokes")

# %% [markdown]
# The last five steps add about $3 \times 10^{-7}$ per pixel, against about
# $0.01$ in total, so they need their own color scale. On that scale, the maps
# follow the digits: their variance is $1.4$ times larger on the strokes than
# on the background. The predicted noise shows the same ratio: at $t = 20$, its
# variance is also about $1.4$ times larger on the strokes, so the model is
# less sure of the noise where the digit is. Within the background, a border
# two pixels wide gets about $1.6$ times the variance of the inside. We have not
# looked into why; the zero padding of the convolutions is one candidate. The
# black dots are the pixels clipped at the last step, $0.5\%$ of them. There
# $\bar\alpha_{t'} = 1$, so a clipped pixel gets no variance at all. Most of
# them ($87\%$) sit on strokes, clipped at the bright end of the range.
#
# ## Ranking the samples
#
# BayesDiff uses the summed pixel variance of each image to filter out bad
# samples. The full sum here mostly records the clipping of the first steps,
# as above, so we rank our 16 samples by the variance of the last five steps
# instead. Below are the four most and the four least uncertain ones, with
# their summed variance in units of $10^{-5}$.

# %%
total = late.sum((1, 2, 3)) * 1e5
order = total.argsort(descending=True)
picks = [*order[:4], *order[-4:]]
ink = stroke.sum((1, 2, 3)).float()
corr = float(torch.corrcoef(torch.stack([total, ink]))[0, 1])

print(
    "summed variance, most to least:",
    [round(float(v), 1) for v in total[order]],
)
print(f"correlation with the number of stroke pixels: {corr:.2f}")

fig, axes = plt.subplots(2, 8, figsize=(9, 2.6))

for col, k in enumerate(picks):
    axes[0, col].imshow(x[k, 0], cmap="gray", vmin=-1, vmax=1)
    axes[1, col].imshow(
        late[k, 0], cmap="magma", vmin=0, vmax=float(late.max())
    )
    axes[0, col].set_title(f"{float(total[k]):.1f}", fontsize=9)

for ax in axes.flat:
    ax.set_axis_off()

fig.text(0.27, 0.0, "most uncertain", ha="center")
fig.text(0.75, 0.0, "least uncertain", ha="center")
plt.tight_layout()
plt.show()

# %% [markdown]
# The most uncertain samples are the boldest digits, and the least uncertain
# ones are thin. The summed variance mostly counts ink: its correlation with
# the number of stroke pixels is $0.97$, which follows from the higher
# variance on the strokes. Quality does not line up with the ranking: the
# least uncertain group holds two samples that are hard to read as digits
# (both 19.3). The totals also lie close together, from about $19$
# to $26$. This page leaves out the Monte Carlo correction of BayesDiff and
# draws only 16 n_samples, so we can't say more about the filter from here.
#
# The times below are for the fit and for the 50 sampling steps, which compute
# the variances of the 16 images along the way.

# %%
print(f"fit:    {fit_time:5.1f} s")
print(f"sample: {sample_time:5.1f} s")

# %% [markdown]
# [^kou2024]: Kou, S., Gan, L., Wang, D., Li, C., & Deng, Z. (2024). BayesDiff: Estimating pixel-wise uncertainty in diffusion via Bayesian inference. *ICLR*. <https://arxiv.org/abs/2310.11142>
#
# [^ho2020]: Ho, J., Jain, A., & Abbeel, P. (2020). Denoising diffusion probabilistic models. *NeurIPS*. <https://arxiv.org/abs/2006.11239>
#
# [^song2021]: Song, J., Meng, C., & Ermon, S. (2021). Denoising diffusion implicit models. *ICLR*. <https://arxiv.org/abs/2010.02502>
#
# [^lecun1998]: LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). Gradient-based learning applied to document recognition. *Proceedings of the IEEE*, 86(11), 2278-2324. <https://doi.org/10.1109/5.726791>
