# %% [markdown]
# ---
# title: Regression
# jupyter: python3
# ---
#
# ::: {.callout-tip title="In short"}
# ```{.python}
# post = lk.laplace(model, [(x, y)], likelihood=lk.Gaussian())  # tunes prior and sigma
# pred = lk.predict(post, x_test)  # pred.mean, pred.stddev: noise included
# f = lk.predict_outputs(post, x_test)  # the function alone
# ```
# :::
#
# The goal of this page is to get error bars for a small regression network. We
# fit the network to noisy samples of a sine, $y = \sin(x) + \varepsilon$ with
# $\varepsilon \sim \mathcal{N}(0, \sigma^2)$ and $\sigma = 0.3$, and then fit a
# Laplace approximation to it. We also see how to tune the prior and the noise
# together, and how to separate the uncertainty about the function from the
# observation noise.

# %%
import matplotlib.pyplot as plt
import torch
from torch import nn

import laplacekit as lk

# %% [markdown]
# ## Data and model
#
# We draw 150 points with $x$ uniform in $[0, 8]$, and test on $[-2, 10]$, so
# that both ends lie outside the data. The network has one hidden layer of 50
# tanh units.

# %%
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
# ## Fit
#
# The likelihood carries the noise: $p(y \mid f) = \mathcal{N}(y \mid f,
# \sigma^2)$, which is `lk.Gaussian`. Here we fix $\sigma = 0.3$. As on
# the classification page, `lk.laplace` fits the dense GGN of the last layer,
# here 51 parameters. For a Gaussian likelihood, the output Hessian is $H =
# 1/\sigma^2$, so the GGN is $G = \frac{1}{N \sigma^2} \sum_i J_i^\top J_i$. We
# keep the prior precision at $\lambda = 1$ and the noise at $\sigma = 0.3$ for now,
# with `hyper_tuner=None`.

# %%
lik = lk.Gaussian(sigma=0.3)
post = lk.laplace(
    model, [(x, y)], likelihood=lik, hyper_tuner=None, generator=torch.Generator().manual_seed(0)
)
print(post)

# %% [markdown]
# ## Predictive mean and two standard deviations
#
# Since the likelihood is Gaussian, the linearized predictive needs no further
# approximation. Linearizing the network around $\theta_\text{MAP}$ gives
#
# $$
#   p(y \mid x, \mathcal{D}) \approx \mathcal{N}\big( y \mid f_{\theta_\text{MAP}}(x), \; J(x) \Lambda^{-1} J(x)^\top + \sigma^2 \big) ,
# $$
#
# where $J(x)$ is the Jacobian of the output with respect to the last layer's
# parameters. The band below is the mean plus and minus two standard deviations.

# %%
pred = lk.predict(post, x_test)
mean = pred.mean.squeeze()
std = pred.stddev.squeeze()
t = x_test.squeeze()

fig, ax = plt.subplots(figsize=(7, 4))
ax.scatter(x.squeeze(), y.squeeze(), s=8, color="0.5", label="data")
ax.plot(t, mean, label="mean")
ax.fill_between(t, mean - 2 * std, mean + 2 * std, alpha=0.3, label="mean ± 2 std")
ax.legend()
plt.show()

# %% [markdown]
# ## Tuning the prior and the noise
#
# We passed `hyper_tuner=None`, so $\lambda$ is still $1$. Now we leave
# $\sigma$ open with `lk.Gaussian()` and `hyper_tuner` at its default,
# `lk.MarginalLikelihood()`, which picks $\lambda$ and $\sigma$ together by
# maximizing the log marginal likelihood. Neither needs a refit: the fit keeps
# the GGN at unit noise, $\frac{1}{N} \sum_i J_i^\top J_i$, and the sum of
# squared residuals, and $1/\sigma^2$ is applied afterwards, as in
# `laplace-torch`.

# %%
tuned = lk.laplace(
    model, [(x, y)], likelihood=lk.Gaussian(), generator=torch.Generator().manual_seed(0)
)
sigma = float(tuned.likelihood.sigma)
print("prior precision before:", float(post.prior.prior_precision))
print("prior precision after: ", float(tuned.prior.prior_precision))
print("sigma after:           ", sigma)

# %% [markdown]
# The tuned $\sigma$ is close to the $0.3$ we used to generate the data.
# The tuner starts $\sigma$ at 1; `lk.MarginalLikelihood(sigma_init=0.3)` would
# start it elsewhere.
#
# On a posterior that is already fitted, the call is
# `lk.tune(post, lk.MarginalLikelihood())`. [Step by step](by_hand.py) shows it
# in context, for the prior alone.
#
# ## Epistemic and total variance
#
# The predictive variance above has two parts. The first, $J(x) \Lambda^{-1}
# J(x)^\top$, is the uncertainty about the function $f$ (epistemic). The second,
# $\sigma^2$, is the observation noise. `lk.predict_outputs` gives the distribution of
# $f(x)$ alone, and `lk.predict` the distribution of a new target $y$. So, the
# difference between their variances should be exactly the tuned $\sigma^2$,
# which the last line checks.

# %%
f = lk.predict_outputs(tuned, x_test)
epistemic = f.cov.diagonal().squeeze()
total = lk.predict(tuned, x_test).variance.squeeze()

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(t, epistemic, label="epistemic (f)")
ax.plot(t, total, label="total (y)")
ax.axhline(sigma**2, color="0.6", linestyle=":", label="sigma squared")
ax.set_ylabel("variance")
ax.legend()
plt.show()

print("max |total - epistemic - sigma^2|:", float((total - epistemic - sigma**2).abs().max()))

# %% [markdown]
# ## Noise per output
#
# A network with $C$ outputs can have a different noise level on each,
# $p(y \mid f) = \mathcal{N}(y \mid f, \mathrm{diag}(\sigma^2))$ with $\sigma =
# (\sigma_1, \dots, \sigma_C)$. The output Hessian of the negative
# log-likelihood is then $H = \mathrm{diag}(\sigma^{-2})$, and the GGN becomes
#
# $$
#   G = \frac{1}{N} \sum_{i=1}^N J_i^\top \mathrm{diag}(\sigma^{-2}) J_i
#     = \frac{1}{N} \sum_{i=1}^N \sum_{k=1}^C \frac{1}{\sigma_k^2} \, j_{ik}^\top j_{ik} ,
# $$
#
# where $j_{ik}$ is the $k$-th row of $J_i$. With a single $\sigma$, the factor
# $1/\sigma^2$ comes out of both sums, $G = \sigma^{-2} \cdot \frac{1}{N}
# \sum_i J_i^\top J_i$. This is why the tuning above needed no refit: the fit
# keeps the sum at unit noise, and $\sigma$ is applied late. With one
# $\sigma_k$ per output, each row of $J_i$ has its own weight inside the sum.
# The fit only keeps the weighted sum, so there is no factor left to change
# afterwards: the noise levels are fixed when we fit, and new ones need a new
# fit.
#
# Our toy predicts a sine and a cosine, with noise 0.1 on the first and 0.5 on
# the second. We train with the squared error weighted by the inverse noise
# variances, which is the negative log-likelihood up to a constant.

# %%
sigma2 = torch.tensor([0.1, 0.5])
y2 = torch.cat([torch.sin(x), torch.cos(x)], 1) + sigma2 * torch.randn(150, 2)

model2 = nn.Sequential(nn.Linear(1, 50), nn.Tanh(), nn.Linear(50, 2))
opt = torch.optim.Adam(model2.parameters(), lr=1e-2)
for _ in range(1000):
    opt.zero_grad()
    (((model2(x) - y2) / sigma2) ** 2).mean().backward()
    opt.step()

# %% [markdown]
# `lk.Gaussian` takes the vector as its `sigma`. A vector is fixed, so the
# hyper tuner picks the prior precision alone.

# %%
post2 = lk.laplace(
    model2,
    [(x, y2)],
    likelihood=lk.Gaussian(sigma=sigma2),
    generator=torch.Generator().manual_seed(0),
)
print(post2)

# %% [markdown]
# We check the precision $\Lambda = \lambda I + N G$ of this fit against
# the formula above. We compute the Jacobians of the last layer with
# `torch.func.jacrev`, build $\Lambda$ as a dense $102 \times 102$ matrix, and
# compare log-determinants, which don't depend on how the parameters are
# ordered.

# %%
last = {name: p.detach() for name, p in model2[2].named_parameters()}

with torch.no_grad():
    phi = model2[:2](x)  # the fixed features under the last layer


def last_layer(params, features):
    return torch.func.functional_call(model2[2], params, (features,))


jac = torch.func.jacrev(last_layer)(last, phi)  # per name: [150, 2, *shape]
J = torch.cat([j.reshape(150, 2, -1) for j in jac.values()], -1)  # [150, 2, 102]
G = torch.einsum("nkp,k,nkq->pq", J, sigma2**-2, J)
Lam = post2.prior.prior_precision * torch.eye(J.shape[-1]) + G

print("log det, lk.laplace:", f"{float(post2.precision().logdet()):.4f}")
print("log det, by hand:   ", f"{float(torch.linalg.slogdet(Lam).logabsdet):.4f}")

# %% [markdown]
# The two agree. The predictive adds $\mathrm{diag}(\sigma^2)$ to the epistemic
# covariance $J(x) \Lambda^{-1} J(x)^\top$, so each output gets its own noise.
# We average the predictive standard deviations over the data range and print
# them at $x = 10$, past the data.

# %%
std2 = lk.predict(post2, x_test).stddev
inside = (t >= 0) & (t <= 8)
print("predictive std on [0, 8]:", [round(v, 3) for v in std2[inside].mean(0).tolist()])
print("predictive std at x = 10:", [round(v, 3) for v in std2[-1].tolist()])

# %% [markdown]
# On the data, the standard deviations are 0.103 and 0.511, close to the
# noise levels of 0.1 and 0.5. At $x = 10$ they have grown to 0.15 and 0.567.
# Since the noise is part of the fit, it is not a hyperparameter of the
# posterior, and no tuner changes it:

# %%
print(list(lk.core.hyperparameters(post2)))

# %% [markdown]
# When the data arrive in parts, the posterior after one part can be the
# prior for the next. [Continual learning](continual.py) does this with the network of this
# page.
