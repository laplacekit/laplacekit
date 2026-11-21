# %% [markdown]
# ---
# title: Heteroscedastic regression
# jupyter: python3
# ---
#
# ::: {.callout-tip title="In short"}
# ```{.python}
# lik = lk.heteroscedastic.NaturalGaussian()  # the model ends with lk.heteroscedastic.NaturalHead()
# post = lk.laplace(model, data, likelihood=lik, subset=lk.subsets.all, hessian_approx=lk.DenseGGN())
# parts = lk.heteroscedastic.moments(lik, lk.predict_outputs(post, x_test))  # mean, epistemic, aleatoric
# ```
# :::
#
# On the [Regression](regression.py) page the noise $\sigma$ was one number for
# all inputs. Often it is not: a sensor gets worse at the ends of its range, or
# some inputs are simply harder to predict. The goal of this page is a Laplace
# approximation for a network that predicts the noise along with the mean, and
# a predictive that tells the two kinds of uncertainty apart. We follow Immer et
# al. (2023)[^immer2023], who show that the right parameterization of the
# likelihood makes the GGN exact at the output.
#
# The likelihood is $p(y \mid x) = \mathcal{N}(y \mid \mu(x), \sigma^2(x))$,
# where the network gives both $\mu(x)$ and $\sigma^2(x)$. The obvious
# choice is to let it output the pair $(\mu, \sigma^2)$. The negative
# log-likelihood is then
#
# $$
#   \ell(\mu, \sigma^2; y) = \frac{1}{2} \log \sigma^2 + \frac{(y - \mu)^2}{2 \sigma^2} + \text{const} ,
# $$
#
# with the Hessian
#
# $$
#   \nabla^2 \ell =
#   \begin{pmatrix}
#     \frac{1}{\sigma^2} & \frac{y - \mu}{\sigma^4} \\
#     \frac{y - \mu}{\sigma^4} & \frac{(y - \mu)^2}{\sigma^6} - \frac{1}{2 \sigma^4}
#   \end{pmatrix} .
# $$
#
# It depends on $y$, and it is not positive semi-definite: at $y = \mu$ its
# second diagonal entry is $-1 / (2 \sigma^4)$. A GGN $J^\top \nabla^2 \ell \, J$
# built on it can be indefinite, so it is no use as a posterior precision. The
# common fix, which `laplace-torch`[^daxberger2021] takes, is to keep only the
# mean block $1 / \sigma^2$ and to treat $\sigma^2$ as fixed. The Laplace
# approximation then covers the mean alone. Taking the expectation over $y$
# would also give a positive semi-definite matrix, the Fisher, but it is no
# longer the Hessian of the loss.
#
# ## Natural parameters
#
# Immer et al. write the Gaussian as an exponential family instead. With the
# natural parameters $\eta_1 = \mu / \sigma^2$ and $\eta_2 = -1 / (2 \sigma^2)$
# and the sufficient statistics $T(y) = (y, y^2)$,
#
# $$
#   p(y \mid \eta) = \frac{1}{\sqrt{2 \pi}} \exp\big( \eta^\top T(y) - A(\eta) \big) ,
#   \qquad
#   A(\eta) = -\frac{\eta_1^2}{4 \eta_2} - \frac{1}{2} \log(-2 \eta_2) .
# $$
#
# The negative log-likelihood $A(\eta) - \eta^\top T(y)$ is linear in $\eta$
# apart from the log-partition $A$, so its Hessian in $\eta$ is $\nabla^2
# A(\eta)$, whatever $y$ is. This is the covariance of $T(y)$, and thus the
# Fisher, and it is positive semi-definite:
#
# $$
#   \nabla^2 A(\eta) = \mathrm{Cov}[T(y)] =
#   \begin{pmatrix}
#     \sigma^2 & 2 \mu \sigma^2 \\
#     2 \mu \sigma^2 & 4 \mu^2 \sigma^2 + 2 \sigma^4
#   \end{pmatrix} .
# $$
#
# So in these parameters the Hessian and the Fisher at the output are the same
# matrix, and the GGN $G = \frac{1}{N} \sum_i J_i^\top \nabla^2 A(\eta_i) J_i$,
# with $J_i$ the Jacobian of $\eta$ with respect to the weights, drops only the
# second derivatives of the network. The network outputs two numbers $(h_1,
# h_2)$, and its last layer maps them to $\eta_1 = h_1$ and $\eta_2 = -\frac{1}{2}
# \exp(h_2)$, which keeps $\eta_2 < 0$.
#
# For the predictive, the linearized network gives a Gaussian $\mathcal{N}(\eta
# \mid \hat\eta(x), \Sigma_\eta(x))$ over the outputs. The mean $\mu = -\eta_1 /
# (2 \eta_2)$ is not linear in $\eta$, so we linearize it as well, with the
# gradient $g = \nabla_\eta \mu = (\sigma^2, 2 \mu \sigma^2)$. This gives
#
# $$
#   p(y \mid x, \mathcal{D}) \approx \mathcal{N}\big( y \mid \mu(\hat\eta), \;
#   \underbrace{g^\top \Sigma_\eta(x) \, g}_{\text{epistemic}} + \underbrace{\sigma^2(\hat\eta)}_{\text{aleatoric}} \big) ,
# $$
#
# the uncertainty about the function plus the noise the network predicts.

# %%
import matplotlib.pyplot as plt
import torch
from torch import nn

import laplacekit as lk

# %% [markdown]
# ## Data and model
#
# The data are the toy of the Quickstart: a sine with noise whose standard
# deviation $0.05 + 0.15 \lvert x \rvert$ grows away from the origin. The
# inputs lie in $[-3, -0.7]$ and $[0.7, 3]$, so there is a gap with no data in
# the middle. We predict on $[-4, 4]$.

# %%
torch.set_default_dtype(torch.float64)
torch.manual_seed(2023)


def draw(n):
    side = torch.randint(0, 2, (n, 1)) * 2 - 1
    x = side * (0.7 + 2.3 * torch.rand(n, 1))  # in [-3, -0.7] and [0.7, 3]
    y = torch.sin(2 * x) + (0.05 + 0.15 * x.abs()) * torch.randn(n, 1)

    return x, y


x, y = draw(150)
x_test = torch.linspace(-4.0, 4.0, 201).reshape(-1, 1)
t = x_test.squeeze()

# %% [markdown]
# The network has two tanh layers and two outputs, followed by
# `lk.heteroscedastic.NaturalHead`, the map from $(h_1, h_2)$ to $(\eta_1,
# \eta_2)$ above. It has no parameters. We train with the negative
# log-likelihood of `lk.heteroscedastic.NaturalGaussian`, the likelihood in
# natural parameters.

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
# ## Fit
#
# The output Hessian of `NaturalGaussian` is $\nabla^2 A(\eta)$, so the dense
# GGN is the one above. The network has 1186 weights, few enough for a dense
# GGN per layer over all of them, so we take all weights instead of the last
# layer. We come back to this choice below.

# %%
post = lk.laplace(
    model,
    [(x, y)],
    likelihood=lik,
    subset=lk.subsets.all,
    hessian_approx=lk.DenseGGN(),
    generator=torch.Generator().manual_seed(0),
)
print(post)

# %% [markdown]
# `lk.heteroscedastic.moments` takes the likelihood and the Gaussian over the
# outputs from `lk.predict_outputs`, and returns the mean $\mu(\hat\eta)$, the
# epistemic variance $g^\top \Sigma_\eta g$ and the aleatoric variance
# $\sigma^2(\hat\eta)$. Their sum is the variance of `lk.predict`, whose
# default predictive for this likelihood is `lk.heteroscedastic.NoisyMean`.
# We print the two standard deviations at a few points, next to the true
# noise.

# %%
parts = lk.heteroscedastic.moments(post.likelihood, lk.predict_outputs(post, x_test))
pred = lk.predict(post, x_test)
print(
    "max |predictive var - sum of parts|:",
    f"{float((pred.variance - parts.epistemic - parts.aleatoric).abs().max()):.1g}",
)


def at(m, where):
    for point in where:
        i = int((t - point).abs().argmin())
        print(
            f"  x = {point}: epistemic {float(m.epistemic[i].sqrt()):.3f}, "
            f"aleatoric {float(m.aleatoric[i].sqrt()):.3f} (true {0.05 + 0.15 * point:.3f})"
        )


print("all weights:")
at(parts, [0.0, 1.0, 3.0, 4.0])

# %% [markdown]
# On the data, the predicted noise follows the truth roughly: 0.238 at $x = 1$
# against a true 0.200, and 0.380 at $x = 3$ against 0.500. Past the data it
# drops, to 0.209 at $x = 4$, while the true noise grows to 0.650. The
# epistemic part grows there, from 0.240 at $x = 3$ to 0.334 at $x = 4$, and
# makes up some of the difference. In the gap, at $x = 0$, the network has
# seen nothing, and the epistemic standard deviation is 42.7.
#
# The default fit takes the last layer only. Its epistemic part at the same
# points shows why we didn't use it here.

# %%
last = lk.laplace(
    model, [(x, y)], likelihood=lik, generator=torch.Generator().manual_seed(0)
)  # the default: the last layer
print(last)
print("last layer:")
at(
    lk.heteroscedastic.moments(last.likelihood, lk.predict_outputs(last, x_test)),
    [0.0, 1.0, 3.0, 4.0],
)

with torch.no_grad():
    phi = model[:4](torch.tensor([[2.0], [3.0], [4.0]]))  # the features under the last layer

print(f"feature change from x = 2 to 3: {float((phi[1] - phi[0]).norm()):.2f}")
print(f"feature change from x = 3 to 4: {float((phi[2] - phi[1]).norm()):.2f}")

# %% [markdown]
# On the last layer, the epistemic standard deviation goes from 0.126 at $x =
# 3$ down to 0.113 at $x = 4$, so the band doesn't grow past the data. It is
# 11.7 in the gap, a quarter of the 42.7 over all weights. The last-layer
# Laplace keeps the features of the body fixed, and those flatten out past the
# data: from $x = 3$ to $4$ they move by 1.03 in norm, against 3.97 from
# $x = 2$ to $3$. Over all weights, the features are uncertain as well.
#
# ## The mean-only Laplace
#
# For comparison, we fit the mean-only form on the same data. The likelihood is
# `lk.heteroscedastic.MeanVarianceGaussian`, whose output Hessian is $1 / \sigma^2$ on
# the mean and zero on the variance. The model gives $(\mu, \sigma^2)$ from two
# heads on a shared body, with a softplus to keep $\sigma^2$ positive, and we
# train it with `nn.GaussianNLLLoss`. We again fit all weights. With
# `subsets.last_layer`, the fit would take the head that runs last, here the
# variance head, which gets no curvature at all.


# %%
class MeanVarianceGaussian(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(1, 32), nn.Tanh(), nn.Linear(32, 32), nn.Tanh())
        self.mean = nn.Linear(32, 1)
        self.var = nn.Linear(32, 1)

    def forward(self, x):
        h = self.body(x)

        return torch.cat([self.mean(h), nn.functional.softplus(self.var(h))], -1)


lik_mv = lk.heteroscedastic.MeanVarianceGaussian()
model_mv = MeanVarianceGaussian()
opt = torch.optim.Adam(model_mv.parameters(), lr=1e-2)
gaussian_nll = nn.GaussianNLLLoss()

for _ in range(1000):
    opt.zero_grad()
    f = model_mv(x)
    gaussian_nll(f[:, :1], y, f[:, 1:]).backward()
    opt.step()

post_mv = lk.laplace(
    model_mv,
    [(x, y)],
    likelihood=lik_mv,
    subset=lk.subsets.all,
    hessian_approx=lk.DenseGGN(),
    generator=torch.Generator().manual_seed(0),
)
parts_mv = lk.heteroscedastic.moments(post_mv.likelihood, lk.predict_outputs(post_mv, x_test))
print("mean-only, all weights:")
at(parts_mv, [0.0, 1.0, 3.0, 4.0])

# %% [markdown]
# We compare the two on 500 new points from the same distribution, by the
# negative log predictive density per point. Next to each Laplace predictive,
# we put its MAP network, with the predicted noise alone.

# %%
x_val, y_val = draw(500)

for name, p, net in [("natural", post, model), ("mean-only", post_mv, model_mv)]:
    with torch.no_grad():
        map_nll = -p.likelihood.distribution(net(x_val)).log_prob(y_val).mean()

    lap_nll = -lk.predict(p, x_val).log_prob(y_val).mean()
    print(f"{name:<10} test NLL: MAP {float(map_nll):.3f}, Laplace {float(lap_nll):.3f}")

# %% [markdown]
# Both Laplace predictives improve on their MAP networks: the natural one from
# 0.460 to 0.408, the mean-only one from 0.932 to 0.493. The natural one has
# the lower NLL. Part of the gap is in the MAP networks themselves, which are
# trained differently: the mean-only network predicts less noise than the
# truth, 0.274 at $x = 3$ against 0.500. The plots show both predictives as
# the mean plus and minus two standard deviations, for the total and for the
# epistemic part alone.

# %%
fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

for ax, m, name in zip(axes, [parts, parts_mv], ["Natural", "Mean-only"], strict=True):
    mean = m.mean.squeeze()
    total = (m.epistemic + m.aleatoric).sqrt().squeeze()
    epistemic = m.epistemic.sqrt().squeeze()

    ax.scatter(x.squeeze(), y.squeeze(), s=8, color="0.5", label="data")
    ax.plot(t, mean, label="mean")
    ax.fill_between(t, mean - 2 * total, mean + 2 * total, alpha=0.2, label="total")
    ax.fill_between(t, mean - 2 * epistemic, mean + 2 * epistemic, alpha=0.4, label="epistemic")
    ax.set_title(name)
    ax.set_ylim(-3, 3)

axes[0].legend()
plt.show()

# %% [markdown]
# Both bands are narrow on the data and blow up in the gap, the natural one far
# more: 42.7 at $x = 0$ against 2.59. Past the data, it is the other way
# round. The mean-only epistemic part grows from 0.298 at $x = 3$ to 1.92 at
# $x = 4$, the natural one only from 0.240 to 0.334. The two approximations
# put their uncertainty in different places: the natural one over both
# $\eta_1$ and $\eta_2$, the mean-only one over $\mu$ alone, with $\sigma^2$
# read off the output mean.
#
# We trained both networks by MAP and fitted the Laplace afterwards. Immer et
# al. also train the network with the marginal likelihood, which picks the
# prior during training, and this page doesn't do that.

# %% [markdown]
# [^immer2023]: Immer, A., Palumbo, E., Marx, A., & Vogt, J. E. (2023). Effective Bayesian heteroscedastic regression with deep neural networks. *NeurIPS*. <https://arxiv.org/abs/2306.16717>
#
# [^daxberger2021]: Daxberger, E., Kristiadi, A., Immer, A., Eschenhagen, R., Bauer, M., & Hennig, P. (2021). Laplace redux: Effortless Bayesian deep learning. *NeurIPS*. <https://arxiv.org/abs/2106.14806>
