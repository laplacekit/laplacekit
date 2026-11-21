# %% [markdown]
# ---
# title: GP methods
# jupyter: python3
# ---
#
# A GP on a subset of the data throws the other points away. The methods in
# `lk.gp` use all of them and stay cheap. We use the sine data and the network of the
# [regression page](regression.py).
#
# ::: {.callout-note title="Plugin"}
# This page uses the
# [`laplacekit-gp`](https://github.com/laplacekit/laplacekit/tree/main/plugins/gp)
# plugin. Install it with
#
# ```bash
# pip install "laplacekit[gp]"
# ```
# :::

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
# ## Nyström (ELLA)
#
# `lk.gp.ELLA` approximates the network's kernel with a rank-15 basis
# built from 100 training points. We fit all 151 weights and set the noise to
# the $0.3$ we used to make the data.

# %%
gp = fs.laplace(
    model,
    [(x, y)],
    likelihood=lk.Gaussian(sigma=0.3),
    method=lk.gp.ELLA(lk.gp.from_data(100), rank=15),
    subset=lk.subsets.all,
    generator=torch.Generator().manual_seed(0),
)
print(gp)

# %% [markdown]
# The prior precision is applied after the fit, so `lk.GridSearch` can pick it
# without refitting. We score it by the predictive NLL of 50 validation
# points on $[-2, 10]$, so the NLL also sees the inputs past both ends of the
# data.

# %%
x_val = torch.rand(50, 1) * 12.0 - 2.0
y_val = torch.sin(x_val) + 0.3 * torch.randn(50, 1)
nll = lk.objectives.predictive_nll([(x_val, y_val)])

gp = lk.tune(gp, lk.GridSearch(nll, values=torch.logspace(-3, 3, 13)))
print("prior precision:", f"{float(gp.prior.prior_precision):.3g}")
print("validation NLL: ", f"{float(nll(gp)):.3f}")

# %% [markdown]
# A GP predicts with the same `lk.predict` as a weight-space posterior.

# %%
pred = lk.predict(gp, x_test)
mean, std = pred.mean.squeeze(), pred.stddev.squeeze()
t = x_test.squeeze()

fig, ax = plt.subplots(figsize=(7, 4))
ax.scatter(x.squeeze(), y.squeeze(), s=8, color="0.5", label="data")
ax.plot(t, mean, label="mean")
ax.fill_between(
    t, mean - 2 * std, mean + 2 * std, alpha=0.3, label="mean ± 2 std"
)
ax.legend()
plt.show()

# %% [markdown]
# On the data, the band is about twice the noise $0.3$ on each side. Past
# both ends it widens, most of all on the left.
#
# ## Variational (VaLLA)
#
# `lk.gp.VaLLA` conditions the GP on 20 inducing points, here placed
# by k-means on the inputs. `fs.laplace` only places them and counts the
# data. `lk.gp.fit_variational` then trains the posterior and the inducing
# inputs with Adam. We keep the noise at $0.3$ and the prior precision at
# the one the grid picked. `fit_variational` could learn both, but with the
# mean fixed at the network its objective shrinks the error bars when it
# does ([details](../examples/scalable_gp.py)).

# %%
nystrom = gp
gp = fs.laplace(
    model,
    [(x, y)],
    likelihood=lk.Gaussian(sigma=0.3),
    method=lk.gp.VaLLA(lk.gp.kmeans(20)),
    prior=nystrom.prior,
    subset=lk.subsets.all,
    generator=torch.Generator().manual_seed(0),
)
gp = lk.gp.fit_variational(
    gp,
    [(x, y)],
    n_steps=1000,
    lr=1e-2,
    learn_prior=False,
    learn_noise=False,
    generator=torch.Generator().manual_seed(0),
)
print(gp)

grid = torch.tensor([[-2.0], [4.0], [10.0]])

for name, posterior in [("Nyström", nystrom), ("VaLLA", gp)]:
    std = lk.predict(posterior, grid).stddev.squeeze()
    print(f"{name:<8} std at x = -2, 4, 10:", [round(float(s), 3) for s in std])

# %%
pred = lk.predict(gp, x_test)
mean, std = pred.mean.squeeze(), pred.stddev.squeeze()
z = gp.inducing.squeeze()

fig, ax = plt.subplots(figsize=(7, 4))
ax.scatter(x.squeeze(), y.squeeze(), s=8, color="0.5", label="data")
ax.plot(t, mean, label="mean")
ax.fill_between(
    t, mean - 2 * std, mean + 2 * std, alpha=0.3, label="mean ± 2 std"
)
ax.plot(
    z,
    torch.full_like(z, -2.0),
    "|",
    color="k",
    markersize=12,
    label="inducing inputs",
)
ax.legend(loc="lower right")
plt.show()

# %% [markdown]
# On the data, the two bands are close (0.31 and 0.327 at $x = 4$). Past
# both ends, VaLLA's band widens like Nyström's, and more: 4.999 against
# 3.439 at $x = -2$, and 1.67 against 1.128 at $x = 10$.
#
# ## Function-space priors (FSP-Laplace)
#
# The methods above keep the isotropic prior on the weights. FSP-Laplace puts
# a GP prior on the network's outputs instead. We switch to the 1-D toy of
# its paper: $\sin(2 \pi x)$ plus noise $0.1$, on two intervals with a gap
# between them, and a network with two hidden layers of 50 units.

# %%
x = torch.cat([torch.rand(50, 1) * 0.5 - 1.0, torch.rand(50, 1) * 0.5 + 0.5])
y = torch.sin(2 * torch.pi * x) + 0.1 * torch.randn(100, 1)
x_test = torch.linspace(-2.0, 2.0, 201).reshape(-1, 1)

# %% [markdown]
# The prior is a zero-mean GP with an RBF kernel of lengthscale $0.3$. The
# MAP adds the prior's RKHS norm to the loss, estimated at 100 context points
# drawn fresh at every step.

# %%
prior = lk.gp.FunctionPrior(lk.gp.RBF(torch.tensor(0.3)))
lik = lk.Gaussian(sigma=0.1)
context = lk.gp.uniform(
    -2.0, 2.0, 100, generator=torch.Generator().manual_seed(0)
)

net = nn.Sequential(
    nn.Linear(1, 50), nn.Tanh(), nn.Linear(50, 50), nn.Tanh(), nn.Linear(50, 1)
)
opt = torch.optim.Adam(net.parameters(), lr=1e-2)

for _ in range(5000):
    opt.zero_grad()
    c = context.draw()
    nll = -lik.log_prob(net(x), y).sum()
    loss = nll + 0.5 * lk.gp.rkhs_norm(prior, c, net(c))
    loss.backward()
    opt.step()

# %% [markdown]
# `lk.gp.fsp_laplace` fits the posterior, with 100 grid points as its
# context points.

# %%
post = lk.gp.fsp_laplace(
    net,
    [(x, y)],
    likelihood=lik,
    prior=prior,
    context=lk.gp.grid(-2.0, 2.0, 100),
    rank=100,
    subset=lk.subsets.all,
    generator=torch.Generator().manual_seed(0),
)
print(post)

pred = lk.predict(post, x_test)
mean, std = pred.mean.squeeze(), pred.stddev.squeeze()
t = x_test.squeeze()

fig, ax = plt.subplots(figsize=(7, 4))
ax.scatter(x.squeeze(), y.squeeze(), s=8, color="0.5", label="data")
ax.plot(t, mean, label="mean")
ax.fill_between(
    t, mean - 2 * std, mean + 2 * std, alpha=0.3, label="mean ± 2 std"
)
ax.legend(loc="lower right")
plt.show()

print("std at x = -2, 0, 2:", [round(float(std[i]), 3) for i in (0, 100, 200)])

# %% [markdown]
# Away from the data, the posterior goes back to the prior. The mean returns
# to zero at both ends, and the standard deviation to the prior's 1: 0.97 at
# $x = -2$ and 0.943 at $x = 2$. In the gap, at $x = 0$, it is 0.815.
#
# ::: {.callout-note collapse="true" title="The math"}
# **The GP view.** Let $J(x)$ be the Jacobian of the outputs with respect to
# the parameters at $\theta_*$ and $\Sigma_0 = \lambda^{-1} I$ the prior
# covariance. The linearized network is a GP with kernel
# $k(x, x') = \lambda^{-1} J(x) J(x')^\top$. Its exact posterior needs an
# $NE \times NE$ solve for $N$ points and $E$ outputs.
#
# **Nyström.** Stack one Jacobian row per picked point into $R$, an
# $m \times P$ matrix, and keep the top $r$ eigenpairs $(U, \Lambda)$ of
# $R R^\top$. The features
#
# $$
#   \varphi(x) = \Lambda^{-1/2} U^\top R \, J(x)^\top \in \mathbb{R}^{r \times E}
# $$
#
# give $J(x) J(x')^\top \approx \varphi(x)^\top \varphi(x')$, a linear model
# with $r$ weights. With $H_i$ the output Hessian of the negative
# log-likelihood at $x_i$, its posterior covariance is the $r \times r$ matrix
#
# $$
#   \Big( \lambda I + \sum_{i=1}^N \varphi(x_i) H_i \varphi(x_i)^\top \Big)^{-1} ,
# $$
#
# and the output covariance at $x_*$ is $\varphi(x_*)^\top$ times it times
# $\varphi(x_*)$. The sum doesn't depend on $\lambda$, so a new $\lambda$
# costs one $r \times r$ solve.
#
# **Variational.** Take $M$ inducing inputs $z_j$, one output each, and let
# $K_{zz}$ and $K_{zx}$ be the kernel among them and between them and $x$.
# VaLLA keeps the mean at $f(x; \theta_*)$ and, with a lower-triangular
# $M \times M$ matrix $L$, sets the output covariance to
#
# $$
#   K_{xx} - K_{xz} L \big( I + L^\top K_{zz} L \big)^{-1} L^\top K_{zx} .
# $$
#
# The fit moves $L$ and the $z_j$ (and $\lambda$ and $\sigma$, if asked) to minimize
#
# $$
#   -\sum_{i=1}^N \frac{1}{\alpha} \log \mathbb{E}_q \big[ p(y_i \mid f_i)^\alpha \big]
#   + \mathrm{KL}\big( q(f_z) \,\|\, p(f_z) \big) ,
# $$
#
# with $\alpha = 1$ by default. As $\alpha \to 0$, this is the negative ELBO.
#
# **Function-space priors.** With the prior $f \sim \mathcal{GP}(\mu, k)$
# and $K_{CC}$ the kernel matrix at the $n_C$ context points $C$, the MAP
# minimizes
#
# $$
#   -\sum_{i=1}^N \log p(y_i \mid f(x_i; \theta))
#   + \tfrac{1}{2} \big(f(C) - \mu(C)\big)^\top K_{CC}^{-1} \big(f(C) - \mu(C)\big) ,
# $$
#
# where the quadratic form is an estimate of the RKHS norm
# $\lVert f - \mu \rVert_{\mathcal{H}}^2$, and never above it. Linearized
# at $\theta_*$, the prior pulls back to the precision
# $J(C)^\top K_{CC}^{-1} J(C)$ over the weights, of rank at most $n_C E$.
# Write it as $M M^\top$, with $L L^\top \approx K_{CC}^{-1}$ from
# Lanczos and $M = J(C)^\top L = U_M D_M V_M^\top$. On the range of $U_M$,
# the posterior precision is $A = D_M^2 + U_M^\top G \, U_M$, with $G$ the
# GGN. With $A = U_A D_A U_A^\top$, the weight covariance is $S S^\top$ with
# $S = U_M U_A D_A^{-1/2}$. The columns of $S$ with the most variance are
# dropped until the variance at every context point is at most the prior's.
# :::
#
# The full tutorials: [Nyström and variational GPs](../examples/scalable_gp.py)
# and [Function-space priors](../examples/fsp.py)
