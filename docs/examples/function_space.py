# %% [markdown]
# ---
# title: Function space
# jupyter: python3
# ---
#
# ::: {.callout-tip title="In short"}
# ```{.python}
# from laplacekit import function_space as fs
#
# gp = fs.laplace(model, loader, likelihood=lik, method=fs.SubsetOfData(n_subset=50), prior=prior)
# probs = lk.predict(gp, x_test).probs  # the same calls as for a weight-space posterior
# ```
# :::
#
# The goal of this page is to look at the linearized Laplace approximation as a
# Gaussian process over the network's outputs, following Immer et al.
# (2021)[^immer2021]. In this view, the posterior is a GP conditioned on the
# training inputs, and we can condition on a random subset of them instead.
# We use the toy data and the network of the [classification
# page](classification.py), with last-layer Laplace (Kristiadi et al.,
# 2020)[^kristiadi2020].
#
# ## From weights to functions
#
# Let $\theta_*$ be the MAP estimate and $J(x)$ the $E \times P$ Jacobian of
# the network's $E$ outputs with respect to its $P$ parameters at $\theta_*$.
# The linearized network is
#
# $$
#   f_\text{lin}(x; \theta) = f(x; \theta_*) + J(x) (\theta - \theta_*) .
# $$
#
# With a Gaussian prior $\mathcal{N}(\theta \mid \mu_0, \Sigma_0)$, the
# Laplace posterior over the weights is $\mathcal{N}(\theta \mid \theta_*,
# \Sigma)$ with
#
# $$
#   \Sigma = \Big( \Sigma_0^{-1} + \sum_{i=1}^N J_i^\top \Lambda_i J_i \Big)^{-1} ,
# $$
#
# where $J_i = J(x_i)$ and $\Lambda_i = -\nabla^2_f \log p(y_i \mid f)$ is the
# $E \times E$ Hessian of the negative log-likelihood with respect to the
# outputs, evaluated at $f(x_i; \theta_*)$. The sum is the GGN, here without the
# $1/N$ average. Pushed through $f_\text{lin}$, it gives a Gaussian over the
# outputs, with mean $f(x; \theta_*)$ and covariance $J(x) \Sigma J(x)^\top$.
#
# Now stack the $N$ Jacobians of the training inputs into $J_N$, an $NE
# \times P$ matrix, and let $\Lambda$ be the block-diagonal matrix with the
# blocks $\Lambda_i$. Define the kernel
#
# $$
#   k(x, x') = J(x) \, \Sigma_0 \, J(x')^\top ,
# $$
#
# and write $K_{NN} = J_N \Sigma_0 J_N^\top$, $K_{N*} = J_N \Sigma_0
# J(x_*)^\top$, and $K_{**} = k(x_*, x_*)$. The Woodbury identity turns the
# weight-space covariance into
#
# $$
#   J(x_*) \Sigma J(x_*)^\top = K_{**} - K_{N*}^\top \Lambda^{1/2} \big( I + \Lambda^{1/2} K_{NN} \Lambda^{1/2} \big)^{-1} \Lambda^{1/2} K_{N*} .
# $$
#
# This is the posterior covariance of a GP with kernel $k$ after observing
# the $N$ training inputs with noise precision $\Lambda$. We use the form with
# $\Lambda^{1/2}$ since it never inverts $\Lambda$. For the softmax,
# $\Lambda_i$ is singular, so $\Lambda^{-1}$ does not exist, while
# $I + \Lambda^{1/2} K_{NN} \Lambda^{1/2}$ is always invertible.
#
# In weight space, we factorize a $P \times P$ matrix. In function space, we
# factorize an $NE \times NE$ one. When $P$ is large, the function-space form
# can be the cheaper one, as long as $NE$ is small. When $NE$ is large too, we
# can keep only $m$ random training inputs, a subset of data (SoD), and
# factorize an $mE \times mE$ matrix.
#
# For the last layer, $f(x) = W \phi(x)$, the kernel has a simple form. With
# $\Sigma_0 = \lambda^{-1} I$, it is $k(x, x') = \lambda^{-1} \phi(x)^\top
# \phi(x') \, I_E$, a linear kernel on the features. The last layer has $P = 80$
# parameters and the data $NE = 2000$ outputs, so here the function-space form
# is the more expensive one. We use it to look at the equivalence and at the
# effect of a subset.

# %%
import copy

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.datasets import make_blobs
from torch import nn

import laplacekit as lk
from laplacekit import function_space as fs

# %% [markdown]
# ## Data and model
#
# The data and the training are those of the classification page: $N = 500$
# points in 4 classes, and a ReLU network with batch norm trained with weight
# decay $5 \times 10^{-4}$. We also draw 200 validation points from the same
# four clusters, which we use later to tune the prior. After training, we
# switch the model to float64. With $\lambda = 5 \times 10^{-4}$, the prior
# variance is $2000$ and the covariance entries reach the hundreds of
# thousands, so in float32 the comparison below would mostly show rounding.

# %%
x_np, y_np, centers = make_blobs(
    n_samples=500,
    centers=4,
    cluster_std=1.2,
    center_box=(-10, 10),
    random_state=37,
    return_centers=True,
)
xv_np, yv_np = make_blobs(
    n_samples=200, centers=centers, cluster_std=1.2, random_state=38
)
x = torch.tensor(x_np, dtype=torch.float32)
y = torch.tensor(y_np)

torch.manual_seed(99999)
model = nn.Sequential(
    nn.Linear(2, 20),
    nn.BatchNorm1d(20),
    nn.ReLU(),
    nn.Linear(20, 20),
    nn.BatchNorm1d(20),
    nn.ReLU(),
    nn.Linear(20, 4, bias=False),
)
opt = torch.optim.SGD(
    model.parameters(), lr=1e-3, momentum=0.9, weight_decay=5e-4
)
for _ in range(5000):
    opt.zero_grad()
    loss = nn.functional.cross_entropy(model(x), y)
    loss.backward()
    opt.step()

model.eval().double()
x = x.double()
loader = [(x, y)]
val = [lk.Batch(torch.tensor(xv_np), torch.tensor(yv_np))]

# %% [markdown]
# ## Weight space and function space
#
# The weight-space posterior is the one of the classification page: the dense
# GGN of the last layer, with $\lambda$ set to the weight decay. The
# function-space version is `fs.laplace`, with `fs.SubsetOfData(n_subset=...)` as the
# method. It draws $m$ training inputs uniformly without replacement and
# returns a GP conditioned on them. We take $m \in \{50, 200, 500\}$, where $m =
# 500$ is the whole training set.

# %%
lik = lk.Categorical()
prior = lk.priors.Isotropic(torch.tensor(5e-4, dtype=torch.float64))

post = lk.laplace(
    model,
    loader,
    likelihood=lik,
    prior=prior,
    hyper_tuner=None,
    generator=torch.Generator().manual_seed(0),
)
gps = {
    m: fs.laplace(
        model,
        loader,
        likelihood=lik,
        method=fs.SubsetOfData(n_subset=m),
        prior=prior,
        generator=torch.Generator().manual_seed(0),
    )
    for m in (50, 200, 500)
}
print(gps[50])

# %% [markdown]
# A GP works with the same `lk.predict_outputs` and `lk.predict` as a weight-space
# posterior. Its mean is $f(x; \theta_*)$ and its covariance is the formula
# above, one $E \times E$ block per input. With `lk.Linearized()`, the
# predictive is the probit approximation of the classification page, applied
# to this Gaussian.
#
# At $m = N$, the GP should be the weight-space posterior. We compare the
# output covariances on a grid of test inputs, and the predictive
# probabilities.

# %%
lin = np.linspace(-15, 15, 50)
gx, gy = np.meshgrid(lin, lin)
grid = torch.tensor(np.stack([gx.ravel(), gy.ravel()], axis=1))

cov_w = lk.predict_outputs(post, grid).cov.per_input()
cov_f = lk.predict_outputs(gps[500], grid).cov.per_input()
probs_w = lk.predict(post, grid, lk.Linearized()).probs
probs_f = lk.predict(gps[500], grid, lk.Linearized()).probs

print("largest covariance entry:      ", f"{float(cov_w.abs().max()):.4g}")
print(
    "max |covariance difference|:   ",
    f"{float((cov_f - cov_w).abs().max()):.2g}",
)
print(
    "max |probability difference|:  ",
    f"{float((probs_f - probs_w).abs().max()):.2g}",
)

# %% [markdown]
# The two agree up to rounding. Below are the confidence maps, the highest
# class probability, for the three subsets and the weight-space posterior,
# with the color levels of the classification page.

# %%
levels = np.arange(0.3, 1.01, 0.1)
colors = ["coral", "yellow", "yellowgreen", "violet"]


def confidence(posterior, inputs):
    return lk.predict(posterior, inputs, lk.Linearized()).probs.max(-1).values


def show(posteriors, names):
    fig, axes = plt.subplots(2, 2, figsize=(9, 8), sharex=True, sharey=True)
    for ax, posterior, name in zip(
        axes.ravel(), posteriors, names, strict=True
    ):
        conf = confidence(posterior, grid)
        im = ax.contourf(
            gx,
            gy,
            conf.numpy().reshape(gx.shape),
            levels=levels,
            cmap="Blues",
            alpha=0.7,
        )
        for k, c in enumerate(colors):
            ax.scatter(
                x_np[y_np == k, 0],
                x_np[y_np == k, 1],
                c=c,
                edgecolors="k",
                linewidths=0.5,
                s=10,
            )
        ax.set_title(name)
        ax.set_aspect("equal")
    fig.colorbar(im, ax=list(axes.ravel()), label="max class probability")
    return fig


fig = show(
    [gps[50], gps[200], gps[500], post],
    ["SoD, m = 50", "SoD, m = 200", "SoD, m = 500 (= N)", "Weight space"],
)
plt.show()

# %% [markdown]
# The two bottom panels are the same picture. With fewer points, the
# predictive is less confident everywhere, even on the data. The reason is in
# the formula: a subset conditions on $m$ of the $N$ terms $\Lambda_i$, and
# `fs.SubsetOfData` does not scale them up by $N / m$. So the GP sees less data
# than there is, and its covariance is larger than the full posterior's. The
# mean confidences on the training points and on the grid points more than 5
# units away from all of them show the same.

# %%
far = torch.tensor([bool((torch.cdist(g[None], x) > 5).all()) for g in grid])
nll = lk.objectives.predictive_nll(val)


def summarize(name, posterior):
    on, conf = confidence(posterior, x), confidence(posterior, grid)
    print(
        f"{name:<16} on data {on.mean():.2f}, far away {conf[far].mean():.2f}, "
        f"validation NLL {float(nll(posterior)):.3f}"
    )


for m, gp in gps.items():
    summarize(f"SoD, m = {m}", gp)
summarize("Weight space", post)

# %% [markdown]
# ## Tuning the prior on validation data
#
# A common way to pick $\lambda$ is to minimize the negative log predictive
# density (NLL) of validation points. We do this with `lk.tune` and
# `lk.Minimize`, the same call as for a weight-space posterior. As on the classification page, the prior
# and the likelihood enter only when we predict, so tuning never refits.
#
# Note what $\lambda$ does here. The kernel is $\lambda^{-1} \phi(x)^\top
# \phi(x') \, I_E$, so $\lambda$ scales the whole kernel. A larger $\lambda$
# shrinks the prior covariance at every input, near the data and far from it
# alike, and the GP's covariance follows. Using all $N$ points instead of $m$
# would shrink the covariance only where the extra data are.
#
# We start the search at $\lambda = 1$. For large $\lambda$, the predictive
# is the MAP's and the validation NLL is almost flat. For $m = 50$, a search
# started at $5 \times 10^{-4}$ stops on that plateau, at $\lambda \approx
# 10^{12}$.

# %%
start = lk.priors.Isotropic(torch.tensor(1.0, dtype=torch.float64))
tuned_gp = lk.tune(copy.replace(gps[50], prior=start), lk.Minimize(nll))
tuned_post = lk.tune(copy.replace(post, prior=start), lk.Minimize(nll))

print(
    "tuned prior precision, SoD m = 50:",
    f"{float(tuned_gp.prior.prior_precision):.3g}",
)
print(
    "tuned prior precision, weights:   ",
    f"{float(tuned_post.prior.prior_precision):.3g}",
)
summarize("SoD, m = 50", tuned_gp)
summarize("Weight space", tuned_post)

low = lk.priors.Isotropic(torch.tensor(5e-4, dtype=torch.float64))
plateau = lk.tune(copy.replace(gps[50], prior=low), lk.Minimize(nll))
print(
    "SoD, m = 50, search from 5e-4:   ",
    f"{float(plateau.prior.prior_precision):.3g}",
)

with torch.no_grad():
    map_nll = nn.functional.cross_entropy(model(val[0].inputs), val[0].targets)
print(f"MAP              validation NLL {float(map_nll):.4f}")
print(f"SoD, m = 50      validation NLL {float(nll(tuned_gp)):.4f} (tuned)")

# %% [markdown]
# The tuned GP on 50 points matches the tuned weight posterior on the
# validation data. Both pick a $\lambda$ between 10 and 20, far above the
# weight decay, and the mean confidence far from the data goes from $0.28$
# (the GP) and $0.33$ (the weights) to $0.89$. That is, far away, both are now close to the MAP. The validation
# points come from the four clusters, so the NLL rewards confidence and
# can't see what happens away from them. It barely tells the tuned GP from
# the MAP itself: $0.1346$ against $0.1360$.
#
# So this is not how we recommend picking $\lambda$ when the uncertainty away
# from the data matters, which is the point of last-layer Laplace. For that,
# use the marginal likelihood of a weight-space posterior (the default of
# `lk.laplace`), or validation data that include such inputs.

# %%
fig2, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=True)
for ax, posterior, name in zip(
    axes,
    [tuned_gp, tuned_post],
    ["SoD, m = 50, tuned", "Weight space, tuned"],
    strict=True,
):
    conf = confidence(posterior, grid)
    im = ax.contourf(
        gx,
        gy,
        conf.numpy().reshape(gx.shape),
        levels=levels,
        cmap="Blues",
        alpha=0.7,
    )
    for k, c in enumerate(colors):
        ax.scatter(
            x_np[y_np == k, 0],
            x_np[y_np == k, 1],
            c=c,
            edgecolors="k",
            linewidths=0.5,
            s=10,
        )
    ax.set_title(name)
    ax.set_aspect("equal")
fig2.colorbar(im, ax=list(axes), label="max class probability")
plt.show()

# %% [markdown]
# ## Tuning with the GP evidence
#
# The GP has its own marginal likelihood, `gp.log_marglik()`, the evidence
# of the points it conditions on. For the softmax it is the Laplace
# approximation of Rasmussen and Williams (2006, eq. 3.44)[^rw2006], as in
# `laplace-torch`. `fs.SubsetOfData(n_subset, structure=...)` also takes
# `"diagonal"` (the diagonal of each $\Lambda_i$) and `"independent"` (one GP
# per output), and the evidence follows the structure. We tune $\lambda$ on
# it for the GP on all 500 points, from the same start as before. Next to it
# is the default hyper tuner of `lk.laplace`, the weight-space marginal likelihood.

# %%
evidence_gp = lk.tune(
    copy.replace(gps[500], prior=start), lk.MarginalLikelihood()
)
evidence_post = lk.laplace(
    model,
    loader,
    likelihood=lik,
    prior=start,
    generator=torch.Generator().manual_seed(0),
)

print(
    "prior precision, GP evidence:     ",
    f"{float(evidence_gp.prior.prior_precision):.4g}",
)
print(
    "prior precision, weight evidence: ",
    f"{float(evidence_post.prior.prior_precision):.4g}",
)
summarize("SoD, m = 500", evidence_gp)

# %% [markdown]
# The two evidences pick the same $\lambda$. They should here, at $m = N$
# with $NE \geq P$ and the same curvature. It is more than ten times smaller
# than the 16 to 18 that the validation NLL picked. The validation NLL is
# $0.138$, higher than the $0.135$ after the validation tune and the MAP's
# $0.1360$, and the mean confidence far from the data is $0.76$ instead of
# $0.89$. So the evidence keeps more of the uncertainty away from the data,
# at a small cost on the validation points.
#
# On a subset of $m < N$ points, the evidence refuses. It is the evidence of
# those $m$ points only, and tuning on it would fit $\lambda$ to a fraction
# of the data.

# %%
try:
    gps[50].log_marglik()
except ValueError as err:
    print(err)

# %% [markdown]
# :::{.callout-note}
# `fs.laplace` takes the same `subset` as `lk.laplace`, and a fitted
# weight-space posterior over the same parameters works as its prior. The
# kernel is then $J(x) \Sigma J(x')^\top$ with the earlier posterior's
# covariance $\Sigma$.
# :::

# %% [markdown]
# [^immer2021]: Immer, A., Korzepa, M., & Bauer, M. (2021). Improving predictions of Bayesian neural nets via local linearization. *AISTATS*. <https://arxiv.org/abs/2008.08400>
#
# [^rw2006]: Rasmussen, C. E., & Williams, C. K. I. (2006). *Gaussian Processes for Machine Learning*. MIT Press. <https://gaussianprocess.org/gpml/>
#
# [^kristiadi2020]: Kristiadi, A., Hein, M., & Hennig, P. (2020). Being Bayesian, even just a bit, fixes overconfidence in ReLU networks. *ICML*. <https://arxiv.org/abs/2002.10118>
