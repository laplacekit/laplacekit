# %% [markdown]
# ---
# title: Continual learning
# jupyter: python3
# ---
#
# ::: {.callout-tip title="In short"}
# ```{.python}
# post_1 = lk.laplace(model, d1, likelihood=lik, prior=prior, hyper_tuner=None)
# # train on d2 with -post_1.log_prob(params) as the regularizer, then:
# post_2 = lk.laplace(model, d2, likelihood=lik, prior=post_1, hyper_tuner=None)
# # with KFAC in both fits, opt into summing the Kronecker factors:
# kfac_2 = lk.laplace(model, d2, likelihood=lik, prior=lk.priors.KronSum.of(kfac_1), hessian_approx=lk.KFAC())
# ```
# :::
#
# Suppose the data arrive in two parts, $\mathcal{D}_1$ and then
# $\mathcal{D}_2$. Bayes' rule lets us process them one after the other, with
# the posterior after $\mathcal{D}_1$ as the prior for $\mathcal{D}_2$:
#
# $$
#   p(\theta \mid \mathcal{D}_1, \mathcal{D}_2) \propto p(\mathcal{D}_2 \mid \theta) \, p(\theta \mid \mathcal{D}_1) .
# $$
#
# Laplace for continual learning (Ritter et al., 2018)[^ritter2018] replaces
# $p(\theta \mid \mathcal{D}_1)$ by its Laplace approximation
# $\mathcal{N}(\theta \mid \theta_1, \Lambda_1^{-1})$. We train on
# $\mathcal{D}_2$ with the negative log-density of that Gaussian as the
# regularizer, which gives $\theta_2$, and fit a Laplace approximation with
# $\Lambda_2 = \Lambda_1 + N_2 G_2$. In general, this loses information at
# each step. For a linear-Gaussian model, the log posterior is exactly
# quadratic, so each Laplace approximation is exact, and the continual posterior
# equals the one fitted on all the data at once.
#
# We use the sine data and the network of the [regression
# page](regression.py). Its last layer, with the hidden layer kept fixed, is
# such a linear-Gaussian model, so we can check the continual posterior against
# the batch one.

# %%
import matplotlib.pyplot as plt
import torch
from torch import nn

import laplacekit as lk

# %% [markdown]
# ## Data and model
#
# As on the regression page, we draw 150 points with $x$ uniform in $[0, 8]$
# and $y = \sin(x) + \varepsilon$, $\varepsilon \sim \mathcal{N}(0, 0.3^2)$,
# and train a network with one hidden layer of 50 tanh units. We split the
# data at $x = 4$ into $\mathcal{D}_1$ (left) and $\mathcal{D}_2$ (right).

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

left = x.squeeze() < 4
d1, d2 = [(x[left], y[left])], [(x[~left], y[~left])]

# %% [markdown]
# ## Two fits in a row
#
# Training on a part minimizes the negative log joint of the last layer, the
# squared error divided by $2 \sigma^2$ minus the log prior. The prior is
# either the first one, $\mathcal{N}(0, I)$, or the previous posterior. Both
# have a `log_prob` of the parameters, so `train` takes either.
#
# We fix $\sigma = 0.3$ and $\lambda = 1$ and pass `hyper_tuner=None`. Each Laplace
# approximation then sits at the mode of the objective that `train`
# minimized. With a posterior as the prior, `lk.laplace` has nothing to tune,
# but the first fit would otherwise move $\lambda$ and $\sigma$ away from the
# values that `train` used.
#
# The prior of a fit can be an earlier posterior: `lk.laplace` reads its mean
# and precision. It also keeps its own copy of the fitted mean, so training the
# module afterwards leaves `post_1` as it was.

# %%
last = {"2.weight": model[2].weight, "2.bias": model[2].bias}


def train(inputs, targets, prior):
    """Minimize the negative log joint of the last layer, with sigma = 0.3."""

    opt = torch.optim.LBFGS(
        list(last.values()),
        max_iter=1000,
        tolerance_grad=1e-12,
        tolerance_change=1e-14,
        line_search_fn="strong_wolfe",
    )

    def closure():
        opt.zero_grad()
        loss = ((targets - model(inputs)) ** 2).sum() / (2 * 0.3**2) - prior.log_prob(last)
        loss.backward()
        return loss

    opt.step(closure)


lik = lk.Gaussian(sigma=0.3)
prior = lk.priors.Isotropic(1.0)

train(*d1[0], prior)
post_1 = lk.laplace(model, d1, likelihood=lik, prior=prior, hyper_tuner=None)

train(*d2[0], post_1)
post_2 = lk.laplace(model, d2, likelihood=lik, prior=post_1, hyper_tuner=None)

# %% [markdown]
# For comparison, we also train on all the data at once, from the first
# prior, and fit the batch posterior.

# %%
train(x, y, prior)
batch = lk.laplace(model, [(x, y)], likelihood=lik, prior=prior, hyper_tuner=None)

# %% [markdown]
# ## Continual against batch
#
# Below, the left panel shows the band after $\mathcal{D}_1$ only, and the
# right one the band after both parts, with the batch fit's band as dashed
# lines.

# %%
t = x_test.squeeze()
fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

for ax, posterior, name, part in zip(
    axes, [post_1, post_2], ["after D1", "after D1, then D2"], [left, ~left], strict=True
):
    p = lk.predict(posterior, x_test)
    m, s = p.mean.squeeze(), p.stddev.squeeze()
    ax.scatter(x[part].squeeze(), y[part].squeeze(), s=8, color="0.5", label="new data")
    ax.plot(t, m, label="mean")
    ax.fill_between(t, m - 2 * s, m + 2 * s, alpha=0.3, label="mean ± 2 std")
    ax.set_title(name)

b = lk.predict(batch, x_test)

for sign in (-1, 1):
    axes[1].plot(t, b.mean.squeeze() + sign * 2 * b.stddev.squeeze(), "k--", lw=1)

axes[1].plot([], [], "k--", lw=1, label="batch fit ± 2 std")
axes[1].legend()
plt.show()

gap = max(float((post_2.mean[k] - batch.mean[k]).abs().max()) for k in batch.mean)
cov_gap = (
    lk.predict_outputs(post_2, x_test).cov.per_input()
    - lk.predict_outputs(batch, x_test).cov.per_input()
)
print("max |mean difference|:      ", f"{gap:.1g}")
print("max |covariance difference|:", f"{float(cov_gap.abs().max()):.1g}")

# %% [markdown]
# The two posteriors agree. The covariances match to rounding, and the means
# to the tolerance of the optimizer. After $\mathcal{D}_1$ alone, the band
# widens over $[4, 8]$, where there are no data yet. After $\mathcal{D}_2$, it
# narrows there as in the batch fit.
#
# ## Continual learning with KFAC
#
# KFAC stores each layer's curvature as a Kronecker product $B \otimes A$ (see
# the [curvature page](curvature.py)), where $A$ averages the layer's inputs
# $a a^\top$ and $B$ the gradients sent back to its output. After the two parts,
# a layer's precision should be
#
# $$
#   \Lambda_2 = \lambda I + c_1 \, B_1 \otimes A_1 + c_2 \, B_2 \otimes A_2 ,
# $$
#
# where $c_t = N_t / \sigma^2$ for our Gaussian likelihood, and $A_t$, $B_t$ are
# the per-example averages over $\mathcal{D}_t$. The sum of two Kronecker
# products is not a Kronecker product, so $\Lambda_2$ loses the cheap
# eigendecomposition that KFAC relies on. Ritter et al. (2018)[^ritter2018] sum
# the factors instead, and `laplace-torch` does the same:
#
# $$
#   c_1 \, B_1 \otimes A_1 + c_2 \, B_2 \otimes A_2
#     \approx (c_1 B_1 + c_2 B_2) \otimes \frac{c_1 A_1 + c_2 A_2}{c_1 + c_2} .
# $$
#
# The two sides are equal when $B_1 = B_2 = B$, since the right-hand side is
# then $B \otimes (c_1 A_1 + c_2 A_2)$. On our last layer, this holds. The
# output is $s = W a$ itself, and the output Hessian of the Gaussian is
# $I / \sigma^2$, which is already in $c_t$, so $B = I$ for every example. There
# KFAC is even the GGN, $I \otimes A$, so its continual posterior should match the
# dense one above.
#
# In **laplacekit**, a KFAC posterior as the prior of a KFAC fit raises
# `StructureError`, since the sum is approximate in general. We opt into the
# approximation with `lk.priors.KronSum.of(post)`. We run the same two steps as
# before with KFAC, and compare the predictive standard deviation with that of
# `post_2`.

# %%
std_dense = lk.predict(post_2, x_test).stddev

train(*d1[0], prior)
kfac_1 = lk.laplace(
    model, d1, likelihood=lik, prior=prior, hessian_approx=lk.KFAC(), hyper_tuner=None
)

train(*d2[0], lk.priors.KronSum.of(kfac_1))
kfac_2 = lk.laplace(
    model,
    d2,
    likelihood=lik,
    prior=lk.priors.KronSum.of(kfac_1),
    hessian_approx=lk.KFAC(),
    hyper_tuner=None,
)

gap = (lk.predict(kfac_2, x_test).stddev - std_dense).abs().max()
print(f"largest gap in the standard deviation: {gap:.1e}")

# %% [markdown]
# The gap is rounding error. On the hidden layer, $B$ depends on the inputs,
# through the derivative of tanh at the hidden units, and the two parts have
# their inputs on different halves of $[0, 8]$. So $B_1 \neq B_2$ there, and
# the sum is approximate. To see by how much, we fit KFAC on all layers, once
# on $\mathcal{D}_1$ and once on $\mathcal{D}_2$ with the first as the prior,
# and compare each layer's precision with the exact sum. We get the exact sum
# from a third fit on $\mathcal{D}_2$ alone: its precision is $\lambda I + c_2
# B_2 \otimes A_2$. We don't retrain the hidden layer, since only the sum
# matters here.


# %%
def block(post, name):
    """One block of the posterior precision, as a matrix."""

    op = post.precision().blocks[name]
    eye = torch.eye(op.diagonal().numel())

    return torch.stack([op.mv(e) for e in eye], dim=1)


every = {
    "likelihood": lik,
    "subset": lk.subsets.all,
    "hessian_approx": lk.KFAC(),
    "hyper_tuner": None,
}
all_1 = lk.laplace(model, d1, prior=prior, **every)
all_2 = lk.laplace(model, d2, prior=lk.priors.KronSum.of(all_1), **every)
alone_2 = lk.laplace(model, d2, prior=prior, **every)

for name, layer in [("0", "hidden"), ("2", "last")]:
    first, alone, summed = (block(post, name) for post in (all_1, alone_2, all_2))
    exact = first + alone - prior.prior_precision * torch.eye(len(first))
    err = torch.linalg.matrix_norm(summed - exact) / torch.linalg.matrix_norm(exact)

    print(f"{layer} layer: relative error of KronSum {err:.2g}")

# %% [markdown]
# On the last layer, the sum is exact to rounding. On the hidden layer, its
# relative error is 0.4. A classifier is like the hidden layer even on its last
# layer, since its output Hessian $\mathrm{diag}(p) - p p^\top$ depends on the
# predictions $p$.
#
# ::: {.callout-note}
# The prior's precision and the new curvature must add. With the diagonal
# curvature, `lk.DiagGGN()`, in every fit, or the dense GGN on small blocks as
# here, the sums are exact and cheap. A KFAC posterior goes in as the prior of a
# KFAC fit through `lk.priors.KronSum.of(post)`, as above. A diagonal posterior
# can't be the prior of a KFAC fit at all, since KFAC takes only one prior
# precision per layer, and `lk.laplace` raises `StructureError`.
# :::

# %% [markdown]
# [^ritter2018]: Ritter, H., Botev, A., & Barber, D. (2018). Online structured Laplace approximations for overcoming catastrophic forgetting. *NeurIPS*. <https://arxiv.org/abs/1805.07810>
