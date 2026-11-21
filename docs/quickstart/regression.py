# %% [markdown]
# ---
# title: Regression
# jupyter: python3
# ---
#
# We want error bars for a small regression network. The data are noisy
# samples of a sine on $[0, 8]$, and we predict on $[-2, 10]$, past both ends.

# %%
import matplotlib.pyplot as plt
import torch
from torch import nn

import laplacekit as lk

torch.set_default_dtype(torch.float64)
torch.manual_seed(711)

x = torch.rand(150, 1) * 8.0
y = torch.sin(x) + 0.3 * torch.randn(150, 1)
x_test = torch.linspace(-2.0, 10.0, 200).reshape(-1, 1)

model = nn.Sequential(nn.Linear(1, 50), nn.Tanh(), nn.Linear(50, 1))
opt = torch.optim.Adam(model.parameters(), lr=1e-2)

for _ in range(1000):
    opt.zero_grad()
    nn.functional.mse_loss(model(x), y).backward()
    opt.step()

# %% [markdown]
# `lk.Gaussian()` tunes the noise $\sigma$ together with the prior, so we don't
# need to know it.

# %%
post = lk.laplace(
    model, [(x, y)], likelihood=lk.Gaussian(), generator=torch.Generator().manual_seed(0)
)
print("sigma:", float(post.likelihood.sigma))

pred = lk.predict(post, x_test)
mean, std = pred.mean.squeeze(), pred.stddev.squeeze()

# %%
t = x_test.squeeze()

fig, ax = plt.subplots(figsize=(7, 4))
ax.scatter(x.squeeze(), y.squeeze(), s=8, color="0.5", label="data")
ax.plot(t, mean, label="mean")
ax.fill_between(t, mean - 2 * std, mean + 2 * std, alpha=0.3, label="mean ± 2 std")
ax.legend()
plt.show()

# %% [markdown]
# The tuned $\sigma$ is close to the $0.3$ we used to make the data. The band
# is about $\pm 2 \sigma$ on the data and widens past both ends, most on the
# left.
#
# ## Noise per output
#
# A network with several outputs can have a different noise level on each. Our
# second toy predicts a sine and a cosine, with noise 0.1 on the first and 0.5
# on the second. We train with the squared error weighted by those levels.

# %%
sigma = torch.tensor([0.1, 0.5])
y2 = torch.cat([torch.sin(x), torch.cos(x)], 1) + sigma * torch.randn(150, 2)

model2 = nn.Sequential(nn.Linear(1, 50), nn.Tanh(), nn.Linear(50, 2))
opt = torch.optim.Adam(model2.parameters(), lr=1e-2)

for _ in range(1000):
    opt.zero_grad()
    (((model2(x) - y2) / sigma) ** 2).mean().backward()
    opt.step()

# %% [markdown]
# `lk.Gaussian` takes one $\sigma$ per output.

# %%
post2 = lk.laplace(
    model2,
    [(x, y2)],
    likelihood=lk.Gaussian(sigma=sigma),
    generator=torch.Generator().manual_seed(0),
)
std2 = lk.predict(post2, x_test).stddev

inside = (x_test.squeeze() >= 0) & (x_test.squeeze() <= 8)
print("sigma:", post2.likelihood.sigma.tolist())
print("predictive std on [0, 8]:", [round(v, 3) for v in std2[inside].mean(0).tolist()])
print("predictive std at x = 10:", [round(v, 3) for v in std2[-1].tolist()])

# %% [markdown]
# Averaged over the data range, the predictive standard deviations are 0.103
# and 0.511, close to the two noise levels. At x = 10 they have grown to 0.15
# and 0.567.
#
# A $\sigma$ with several entries is fixed at fit time. The curvature weights
# each output by its own inverse noise variance, so another $\sigma$ would
# need a new fit. A single learned $\sigma$ (`lk.Gaussian()`) only scales the
# whole data term, which is why the hyper tuner can change it without one. Here
# the tuner fits the prior alone and leaves this $\sigma$ as given.
#
# ::: {.callout-note collapse="true" title="The math"}
# The likelihood is $p(y \mid f) = \mathcal{N}(y \mid f, \sigma^2)$. With the
# Laplace posterior $\mathcal{N}(\theta_*, \Sigma)$ on the last layer and the
# network linearized around $\theta_*$, the predictive is Gaussian with no
# further approximation,
#
# $$
#   p(y \mid x, \mathcal{D}) \approx \mathcal{N}\big(y \mid f_{\theta_*}(x), \; J(x) \Sigma J(x)^\top + \sigma^2\big) ,
# $$
#
# where $J(x)$ is the Jacobian of the output with respect to the last layer.
# The prior precision and $\sigma$ are picked together by maximizing the
# marginal likelihood, without a refit.
#
# With one $\sigma_k$ per output, the GGN is $J^\top \mathrm{diag}(\sigma^{-2}) J$,
# summed over the data, and the noise term of the predictive is
# $\mathrm{diag}(\sigma^2)$. A single $\sigma$ factors out of the GGN as
# $\sigma^{-2} J^\top J$; a vector doesn't.
# :::
#
# The full tutorial: [Regression](../examples/regression.py)
