# %% [markdown]
# ---
# title: Function space
# jupyter: python3
# ---
#
# The linearized Laplace approximation is also a Gaussian process over the
# network's outputs. As a GP, it can condition on a random subset of the
# training points instead of all of them. We use the sine data of the
# [regression page](regression.py).

# %%
import matplotlib.pyplot as plt
import torch
from torch import nn

import laplacekit as lk
from laplacekit import function_space as fs

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
# We fit the usual weight-space posterior first, to get a tuned prior and noise.
# Then `fs.laplace` builds a GP on 30 of the 150 points, with the same prior and
# likelihood.

# %%
post = lk.laplace(model, [(x, y)], likelihood=lk.Gaussian())
gp = fs.laplace(
    model,
    [(x, y)],
    likelihood=post.likelihood,
    method=fs.SubsetOfData(n_subset=30),
    prior=post.prior,
)
print(gp)

# %% [markdown]
# A GP predicts with the same `lk.predict` as a weight-space posterior.

# %%
t = x_test.squeeze()
fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

for ax, posterior, name in zip(
    axes, [post, gp], ["Weight space", "GP on 30 points"], strict=True
):
    pred = lk.predict(posterior, x_test)
    mean, std = pred.mean.squeeze(), pred.stddev.squeeze()
    ax.scatter(x.squeeze(), y.squeeze(), s=8, color="0.5")
    ax.plot(t, mean)
    ax.fill_between(t, mean - 2 * std, mean + 2 * std, alpha=0.3)
    ax.set_title(name)

plt.show()

# %% [markdown]
# The GP sees only a fifth of the data, so its band is wider, most of all on
# the left. With `n_subset=150`, it would match the weight-space posterior up to
# rounding.
#
# ::: {.callout-note collapse="true" title="The math"}
# Let $J(x)$ be the Jacobian of the outputs with respect to the parameters at
# $\theta_*$, and $\Sigma_0 = \lambda^{-1} I$ the prior covariance. The
# linearized network is a GP with kernel $k(x, x') = J(x) \Sigma_0 J(x')^\top$.
# Stack the Jacobians of the training inputs into $J_N$ and the output
# Hessians of the negative log-likelihood into the block-diagonal $\Lambda$
# (here $1/\sigma^2$). The Woodbury identity turns the weight-space covariance
# into the GP posterior,
#
# $$
#   J(x_*) \Sigma J(x_*)^\top = K_{**} - K_{N*}^\top \Lambda^{1/2} \big( I + \Lambda^{1/2} K_{NN} \Lambda^{1/2} \big)^{-1} \Lambda^{1/2} K_{N*} ,
# $$
#
# with $K_{NN} = J_N \Sigma_0 J_N^\top$, $K_{N*} = J_N \Sigma_0 J(x_*)^\top$ and
# $K_{**} = k(x_*, x_*)$. A subset of data keeps $m$ of the $N$ training
# inputs, so the matrix to factorize is $m E \times m E$ for $E$ outputs.
# :::
#
# For methods that use all the data at scale, see [GP methods](gp.py), the
# `lk.gp` plugin.
#
# The full tutorial: [Function space](../examples/function_space.py)
