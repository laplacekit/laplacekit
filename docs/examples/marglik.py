# %% [markdown]
# ---
# title: Marginal-likelihood training
# jupyter: python3
# ---
#
# ::: {.callout-tip title="In short"}
# ```{.python}
# trained = lk.marglik.train(
#     model, loader, lk.Gaussian(), n_data=len(dataset),
#     optimizer=torch.optim.Adam(model.parameters(), lr=1e-2),  # no weight decay
#     epochs=300, marglik_frequency=5,
# )
# post = trained.posterior  # the Laplace fit of the kept epoch, at the learned prior and noise
# ```
# :::
#
# A network has hyperparameters that its training loss can't pick: the weight
# decay, which is a prior precision $\lambda$ in disguise, and in regression the
# noise $\sigma$. We usually pick them by a grid search, with one training run
# per value, each scored on held-out data. The marginal likelihood picks them
# from the training data alone. For hyperparameters $h$, it is
#
# $$
#   p(\mathcal{D} \mid h) = \int p(\mathcal{D} \mid \theta, h) \, p(\theta \mid h) \, \mathrm{d}\theta .
# $$
#
# It rewards a good fit of the data and penalizes a prior that spreads its mass
# over weights the data rule out. For a network the integral is intractable, but
# the Laplace approximation around weights $\theta_*$ gives it in closed form:
#
# $$
#   \log p(\mathcal{D} \mid h) \approx \log p(\mathcal{D} \mid \theta_*, h) + \log p(\theta_* \mid h) - \frac{1}{2} \log \det \frac{\Lambda(h) + N G}{2 \pi} ,
# $$
#
# with $\Lambda(h)$ the prior precision and $G$ the GGN averaged over the $N$
# data points. This is what `post.log_marglik()` computes and what the default
# tune of `lk.laplace` maximizes. After training, though, it can only pick the
# $\lambda$ of the posterior. The network itself was already trained with some
# other weight decay. Immer et al. (2021)[^immer2021] learn both during the
# training, with their Algorithm 1:
#
# 1. One epoch of gradient steps on $\theta$ for the negative log joint
#    $-\log p(\mathcal{D} \mid \theta, h) - \log p(\theta \mid h)$, at the current $h$.
# 2. Every $F$ epochs, a Laplace fit at the current $\theta$, then $K$ gradient
#    steps on $\log h$ for the negative log marginal likelihood above.
#
# Step 1 is ordinary training, with the prior as the weight decay. Step 2 moves
# $\lambda$ and $\sigma$ with $\theta$ fixed. At the end, they keep the
# network of the epoch with the highest evidence. `lk.marglik.train` (the
# `laplacekit-marglik` plugin, installed with the `marglik` extra) runs this
# loop, with $F$ as `every` and $K$ as `hypersteps` (10 by default).

# %%
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import laplacekit as lk

# %% [markdown]
# ## Data and model
#
# We use the sine of the [regression page](regression.py): 150 points with $x$
# uniform in $[0, 8]$ and noise $\sigma = 0.3$. The network has one hidden
# layer of 50 tanh units, 151 weights. The test set has 200 noisy points on
# $[-2, 10]$, and we report the predictive negative log-likelihood (NLL) per
# point on the data range and past it, on $[-2, 0)$ and $(8, 10]$, where the
# network has to extrapolate. `mlp(seed)` draws the initial weights from a seed,
# so every method can start from the same ones.

# %%
torch.set_default_dtype(torch.float64)
torch.manual_seed(711)

x = torch.rand(150, 1) * 8.0
y = torch.sin(x) + 0.3 * torch.randn(150, 1)
x_test = torch.linspace(-2.0, 10.0, 200).reshape(-1, 1)
y_test = torch.sin(x_test) + 0.3 * torch.randn_like(x_test)

t = x_test.squeeze()
on_data = (t >= 0) & (t <= 8)

dataset = TensorDataset(x, y)
loader = DataLoader(dataset, batch_size=50, shuffle=True)


def mlp(seed: int) -> nn.Module:
    torch.manual_seed(seed)

    return nn.Sequential(nn.Linear(1, 50), nn.Tanh(), nn.Linear(50, 1))


PRIORS = {"layerwise": lk.priors.Layerwise, "scalar": lk.priors.Isotropic}


def marglik(seed: int, prior: str) -> lk.marglik.Trained:
    model = mlp(seed)

    return lk.marglik.train(
        model,
        loader,
        n_data=len(dataset),
        likelihood=lk.Gaussian(),
        optimizer=torch.optim.Adam(model.parameters(), lr=1e-2),
        epochs=300,
        marglik_frequency=5,
        prior=PRIORS[prior],
        generator=torch.Generator().manual_seed(0),
    )


# %% [markdown]
# ## Training with the marginal likelihood
#
# The optimizer has no weight decay, since $\lambda$ sets it, and `train`
# refuses one that has. We train for 300 epochs and estimate the
# hyperparameters after every fifth (`marglik_frequency=5`). The prior is
# `lk.priors.Layerwise` by default, with one $\lambda$ per layer, and $\sigma$ starts at the 1 of `lk.Gaussian()`.

# %%
trained = marglik(0, "layerwise")
print(trained.posterior)
print("sigma:", round(float(trained.posterior.likelihood.sigma), 3))

# %% [markdown]
# `train` kept epoch 270. The learned $\sigma$ is 0.304, and the first layer
# gets the weaker prior, with $\lambda = 0.226$ against 0.621 for the last.
# `prior=lk.priors.Isotropic` would share one $\lambda$ over all weights
# instead, and `lk.priors.Diagonal` gives each weight its own.
#
# ## Against weight decay
#
# The usual alternative trains with a weight decay picked by hand. We train
# the same network with the mean squared error and Adam's weight decay of
# $10^{-4}$, $10^{-3}$ and $10^{-2}$. After each run, `lk.laplace` over all
# weights tunes one $\lambda$ and $\sigma$ for it by the evidence. Marglik
# training runs once with `Layerwise` and once with `Isotropic`, the same
# prior structure as these baselines. We repeat everything for three
# initializations and print the mean and the standard deviation over them of
# the log evidence $\log Z$ of the final posterior and of the test NLL.


# %%
def decay(seed: int, value: float):
    model = mlp(seed)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=value)

    for _ in range(300):
        for xb, yb in loader:
            opt.zero_grad()
            nn.functional.mse_loss(model(xb), yb).backward()
            opt.step()

    return lk.laplace(
        model,
        loader,
        likelihood=lk.Gaussian(),
        subset=lk.subsets.all,
        generator=torch.Generator().manual_seed(0),
    )


def scores(post) -> list[float]:
    v = -lk.predict(post, x_test).log_prob(y_test).squeeze()

    return [float(post.log_marglik()), float(v[on_data].mean()), float(v[~on_data].mean())]


runs = {(0, "layerwise"): trained}
rows: dict[str, list[list[float]]] = {}

for seed in range(3):
    for prior in ("layerwise", "scalar"):
        if (seed, prior) not in runs:
            runs[seed, prior] = marglik(seed, prior)

        rows.setdefault(f"marglik, {prior}", []).append(scores(runs[seed, prior].posterior))

    for value in (1e-4, 1e-3, 1e-2):
        rows.setdefault(f"decay {value:g}", []).append(scores(decay(seed, value)))

print(f"{'':18} {'log Z':>16} {'NLL on data':>16} {'past data':>16}")

for name, values in rows.items():
    a = torch.tensor(values)
    cells = " ".join(f"{m:8.3f} ± {s:.3f}" for m, s in zip(a.mean(0), a.std(0), strict=True))
    print(f"{name:18} {cells}")

# %% [markdown]
# On the data, marglik training is slightly better than the best decay: an
# NLL of 0.273 with one $\lambda$ per layer and 0.274 with a shared one,
# against 0.278 for $10^{-3}$, and 0.282 and 0.308 for the other two decays.
# With one $\lambda$ per layer, it also reaches the highest evidence,
# $-75.863$, against $-76.477$ for $10^{-3}$, the best decay. With a shared
# $\lambda$ it is close behind, at $-76.722$. This is what Immer et al. are
# after: the evidence ranks the networks without validation data, and a
# single marglik run lands at or near the top of that ranking, where the
# decays need a run per value.
#
# Past the data, the order turns around. The decay of $10^{-2}$ has the lowest
# evidence ($-81.704$) and the best NLL (2.876), and the two marglik runs have
# the worst NLL (3.749 and 3.766). The spread over the
# three initializations is large there, 0.295 to 0.615, so the trend is
# clearer than the order of neighbouring rows. No network predicts the sine
# past the data (an NLL around 3, against 0.27 on it). The evidence measures
# how well a network explains the training data, so it has nothing to say
# about the bands where there are no data.
#
# The plot shows the log marginal likelihood at every hyperparameter step of
# the first initialization: 10 per estimation, 60 estimations.

# %%
scalar = runs[0, "scalar"]
fig, ax = plt.subplots(figsize=(7, 4))

for name, run in [("layerwise", trained), ("scalar", scalar)]:
    ax.plot([-v for v in run.margliks], label=name)

ax.set_xlabel("hyperparameter step")
ax.set_ylabel("log marginal likelihood")
ax.set_ylim(-210, -60)
ax.legend()
plt.show()

# %% [markdown]
# For both priors, the evidence climbs during about the first 20 estimations
# (100 epochs) and then levels off. Within one estimation it rises step by
# step. The jumps between estimations come from the five epochs of training in
# between, and the two dips, near step 150 and between steps 360 and 390, are
# such jumps. That is why `train` keeps the best epoch instead of the last.
#
# ## Heteroscedastic regression
#
# Immer et al. (2023)[^immer2023] train heteroscedastic networks, which predict
# the noise too, the same way. The network's two outputs go through
# `lk.heteroscedastic.NaturalHead` into the natural parameters of a Gaussian,
# $\eta = (\mu / \sigma^2, -1 / (2 \sigma^2))$, and
# `lk.heteroscedastic.NaturalGaussian` reads them (see
# [Heteroscedastic regression](heteroscedastic.py)). This likelihood has no
# hyperparameters, so `train` learns $\lambda$ alone. Our data are the sine of
# that page, with noise growing away from the origin, $\sigma(x) = 0.05 + 0.15
# |x|$, and a gap with no data on $(-0.7, 0.7)$. The network has two hidden
# layers of 16 units. We start $\lambda$ at 0.01 with `prior_init`.

# %%
torch.manual_seed(2023)

side = torch.randint(0, 2, (150, 1)) * 2 - 1
x_h = side * (0.7 + 2.3 * torch.rand(150, 1))  # in [-3, -0.7] and [0.7, 3]
y_h = torch.sin(2 * x_h) + (0.05 + 0.15 * x_h.abs()) * torch.randn(150, 1)
loader_h = DataLoader(TensorDataset(x_h, y_h), batch_size=50, shuffle=True)

torch.manual_seed(0)
model = nn.Sequential(
    nn.Linear(1, 16),
    nn.Tanh(),
    nn.Linear(16, 16),
    nn.Tanh(),
    nn.Linear(16, 2),
    lk.heteroscedastic.NaturalHead(),
)
het = lk.marglik.train(
    model,
    loader_h,
    n_data=150,
    likelihood=lk.heteroscedastic.NaturalGaussian(),
    optimizer=torch.optim.Adam(model.parameters(), lr=1e-2),
    epochs=600,
    marglik_frequency=10,
    prior=lk.priors.Layerwise(0.01),
    generator=torch.Generator().manual_seed(0),
)
print(het.posterior.prior, "best epoch", het.best_epoch)

grid = torch.linspace(-4.0, 4.0, 201).reshape(-1, 1)
parts = lk.heteroscedastic.moments(
    het.posterior.likelihood, lk.predict_outputs(het.posterior, grid)
)
noise = parts.aleatoric.sqrt().squeeze()

for at in [1.0, 2.0, 3.0, 4.0]:
    i = int((grid.squeeze() - at).abs().argmin())
    print(f"x = {at}: noise std {float(noise[i]):.3f} (true {0.05 + 0.15 * at:.3f})")

# %%
g = grid.squeeze()

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(g, noise, label="learned noise std")
ax.plot(g, 0.05 + 0.15 * g.abs(), "k--", lw=0.8, label="true noise std")
ax.axvspan(-0.7, 0.7, color="0.9", label="no data")
ax.set_xlabel("x")
ax.legend()
plt.show()

# %% [markdown]
# At $x = 2$ and $x = 3$ the learned noise is close to the truth: 0.370
# against 0.350, and 0.521 against 0.500. At $x = 1$ it is too large, 0.287
# against 0.200. Past the data it falls behind the truth on both sides. On the
# right it stops growing, at 0.527 for $x = 4$ against a true 0.650, and on the
# left it still rises, but more slowly than the truth, as the plot shows. Across
# the gap it stays flat.

# %% [markdown]
# ::: {.callout-note title="Differences from `laplace-torch`"}
# `laplace-torch`'s `marglik_training` runs the same algorithm, with these
# differences:
#
# - The training loss of a Gaussian. `train` takes the negative log joint per
#   example at the current $\sigma$. `laplace-torch` multiplies its loss by
#   $2 \sigma^2 T$, with $T$ the temperature, so with a learned $\sigma$ the two
#   trajectories differ.
# - `lk.priors.Layerwise` is one $\lambda$ per module, so a layer's weight and bias
#   share it. `laplace-torch` has one per parameter tensor.
# - The default curvature is the default of `lk.laplace`, here a dense GGN per
#   layer. `laplace-torch` defaults to KFAC.
# - An optimizer with weight decay raises `ValueError`. `laplace-torch` warns
#   and sets the decay to zero.
# - `train` leaves the module in train or eval mode, as it found it.
#   `laplace-torch` leaves it in eval mode.
# - There is no temperature.
# :::
#
# [^immer2021]: Immer, A., Bauer, M., Fortuin, V., Rätsch, G., & Khan, M. E. (2021). Scalable marginal likelihood estimation for model selection in deep learning. *ICML*. <https://arxiv.org/abs/2104.04975>
#
# [^immer2023]: Immer, A., Palumbo, E., Marx, A., & Vogt, J. E. (2023). Effective Bayesian heteroscedastic regression with deep neural networks. *NeurIPS*. <https://arxiv.org/abs/2306.16717>
