# %% [markdown]
# ---
# title: Heteroscedastic regression
# jupyter: python3
# ---
#
# When the noise changes with the input, the network should predict it too. Our
# data are a sine with noise that grows away from the origin, and a gap in the
# middle with no data at all.
#
# ::: {.callout-note title="Plugin"}
# This page uses the
# [`laplacekit-heteroscedastic`](https://github.com/laplacekit/laplacekit/tree/main/plugins/heteroscedastic)
# plugin. Install it with
#
# ```bash
# pip install "laplacekit[heteroscedastic]"
# ```
# :::

# %%
import matplotlib.pyplot as plt
import torch
from torch import nn

import laplacekit as lk

torch.set_default_dtype(torch.float64)
torch.manual_seed(2023)

side = torch.randint(0, 2, (150, 1)) * 2 - 1
x = side * (0.7 + 2.3 * torch.rand(150, 1))  # in [-3, -0.7] and [0.7, 3]
y = torch.sin(2 * x) + (0.05 + 0.15 * x.abs()) * torch.randn(150, 1)
x_test = torch.linspace(-4.0, 4.0, 201).reshape(-1, 1)

# %% [markdown]
# The network has two outputs, which `lk.heteroscedastic.NaturalHead` turns
# into the natural parameters of a Gaussian. The likelihood,
# `lk.heteroscedastic.NaturalGaussian`, reads them, and we train with its
# negative log-likelihood.

# %%
lik = lk.heteroscedastic.NaturalGaussian()
model = nn.Sequential(
    nn.Linear(1, 32),
    nn.Tanh(),
    nn.Linear(32, 32),
    nn.Tanh(),
    nn.Linear(32, 2),
    lk.heteroscedastic.NaturalHead(),
)
opt = torch.optim.Adam(model.parameters(), lr=1e-2)

for _ in range(1000):
    opt.zero_grad()
    (-lik.log_prob(model(x), y).mean()).backward()
    opt.step()

# %% [markdown]
# The fit is the usual call. The network is small, so we take all of its
# weights and a dense GGN for each layer.

# %%
post = lk.laplace(
    model,
    [(x, y)],
    likelihood=lik,
    subset=lk.subsets.all,
    hessian_approx=lk.DenseGGN(),
)
print(post)

# %% [markdown]
# `lk.predict` gives a Normal whose variance is the sum of two parts, and
# `lk.heteroscedastic.moments` returns them one by one: the epistemic part,
# from our uncertainty over the weights, and the aleatoric part, the noise the
# network predicts.

# %%
pred = lk.predict(post, x_test)
parts = lk.heteroscedastic.moments(
    post.likelihood, lk.predict_outputs(post, x_test)
)

mean = pred.mean.squeeze()
epistemic = parts.epistemic.sqrt().squeeze()
total = pred.stddev.squeeze()

for at in [0.0, 1.0, 3.0, 4.0]:
    i = int((x_test.squeeze() - at).abs().argmin())
    print(
        f"x = {at}: noise std {float(parts.aleatoric[i].sqrt()):.3f} "
        f"(true {0.05 + 0.15 * at:.3f}), epistemic std "
        f"{float(epistemic[i]):.3f}"
    )

# %%
t = x_test.squeeze()

fig, ax = plt.subplots(figsize=(7, 4))
ax.scatter(x.squeeze(), y.squeeze(), s=8, color="0.5", label="data")
ax.plot(t, mean, label="mean")
ax.fill_between(t, mean - 2 * total, mean + 2 * total, alpha=0.2, label="total")
ax.fill_between(
    t, mean - 2 * epistemic, mean + 2 * epistemic, alpha=0.4, label="epistemic"
)
ax.set_title("mean ± 2 std")
ax.set_ylim(-3, 3)
ax.legend()
plt.show()

# %% [markdown]
# On the data, the network gets the noise roughly right: a standard deviation
# of 0.238 at x = 1, where the truth is 0.200, and 0.380 at x = 3, where it is
# 0.500. Past the data the predicted noise drops, to 0.209 at x = 4
# against a true 0.650. The epistemic part grows there, from 0.240 at x = 3 to
# 0.334 at x = 4, and makes up only some of the difference. Between the two
# clusters the network has seen nothing, and the epistemic standard deviation
# at x = 0 is 42.7, far off the plot.
#
# ::: {.callout-note collapse="true" title="The math"}
# A Gaussian $\mathcal{N}(y \mid \mu, \sigma^2)$ in its natural parameters
# $\eta = (\eta_1, \eta_2) = (\mu / \sigma^2, -1 / (2 \sigma^2))$ is
#
# $$
#   p(y \mid \eta) = \exp\big(\eta_1 y + \eta_2 y^2 - A(\eta)\big) / \sqrt{2 \pi} ,
#   \qquad A(\eta) = -\frac{\eta_1^2}{4 \eta_2} - \frac{1}{2} \log(-2 \eta_2) ,
# $$
#
# so $\mu = -\eta_1 / (2 \eta_2)$ and $\sigma^2 = -1 / (2 \eta_2)$, and the head
# keeps $\eta_2 < 0$. The Hessian of $-\log p(y \mid \eta)$ in $\eta$ is
# $\nabla^2 A(\eta)$ whatever $y$ is. It is the Fisher, and it is positive
# semi-definite, so the GGN $J^\top \nabla^2 A(\eta) J$ is a valid curvature
# (Immer et al., 2023).
#
# With the Gaussian $\mathcal{N}(\hat\eta(x), \Sigma_\eta(x))$ over the
# outputs from the linearized network, the predictive is
#
# $$
#   p(y \mid x, \mathcal{D}) \approx \mathcal{N}\big(y \mid \mu(\hat\eta), \; g^\top \Sigma_\eta g + \sigma^2(\hat\eta)\big) ,
#   \qquad g = \nabla_\eta \mu = (\sigma^2, 2 \mu \sigma^2) ,
# $$
#
# the epistemic variance $g^\top \Sigma_\eta g$ plus the aleatoric one.
# :::
#
# The full tutorial: [Heteroscedastic regression](../examples/heteroscedastic.py)
