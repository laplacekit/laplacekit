# %% [markdown]
# ---
# title: Classification
# jupyter: python3
# ---
#
# ::: {.callout-tip title="In short"}
# ```{.python}
# post = lk.laplace(model, loader, likelihood=lk.Categorical())  # last layer, dense GGN, tuned prior
# probs = lk.predict(post, x_test).probs  # linearized, probit
# ```
# :::
#
# The goal of this page is to show that a cheap, post-hoc Bayesian treatment of
# a trained classifier fixes its overconfidence away from the data. We reproduce
# the toy example of last-layer Laplace (Kristiadi et al., 2020)[^kristiadi2020]. The idea is as
# follows. Suppose we have an $L$-layer network trained to its MAP estimate. We
# treat the first $L - 1$ layers as a fixed feature map $\phi$, and apply a
# Laplace approximation only to the last layer's weight matrix $W$. That is,
#
# $$
#   p(W \mid \mathcal{D}) \approx \mathcal{N}(\mathrm{vec}(W) \mid \mathrm{vec}(W_\text{MAP}), \Lambda^{-1}) , \qquad \Lambda = \lambda I + N G ,
# $$
#
# where $\lambda$ is the prior precision, $N$ the number of training points, and
# $G$ the generalized Gauss-Newton matrix of the last layer, averaged over the
# data. Since the last layer is small, we can afford all of $G$.

# %%
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.datasets import make_blobs
from torch import nn

import laplacekit as lk

# %% [markdown]
# ## Data and model
#
# First, we create a toy 4-class dataset $\mathcal{D}$ with $N = 500$ points.
# The network is a two-layer ReLU feature extractor with batch norm, followed by
# a linear last layer without bias. We train it full batch with weight decay $5
# \times 10^{-4}$. Note that the weight decay can be seen as the precision of a
# Gaussian prior. We use it as $\lambda$ for the Laplace approximation below.

# %%
x_np, y_np = make_blobs(
    n_samples=500, centers=4, cluster_std=1.2, center_box=(-10, 10), random_state=37
)
x = torch.tensor(x_np, dtype=torch.float32)
y = torch.tensor(y_np)
loader = [(x, y)]

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
opt = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=5e-4)
for _ in range(5000):
    opt.zero_grad()
    loss = nn.functional.cross_entropy(model(x), y)
    loss.backward()
    opt.step()

model.eval()  # batch norm uses its running statistics from here on
print("final loss:", round(loss.item(), 3))

# %% [markdown]
# ## One line
#
# `lk.laplace` takes the trained model, an iterable of `(inputs, targets)`
# batches, and a likelihood. We set $\lambda$ to the weight decay and switch off
# its tuning, which we come back to later.

# %%
post = lk.laplace(
    model,
    loader,
    likelihood=lk.Categorical(),
    prior=lk.priors.Isotropic(5e-4),
    hyper_tuner=None,
)
print(post)

# %% [markdown]
# Printing the posterior shows what `lk.laplace` chose. By default, it takes the
# last layer (`params`) and the dense GGN for it (`approx`). Here $W$ is $4
# \times 20$, so $G$ is an $80 \times 80$ matrix, and `n_data` is $N$. Each
# choice can be passed explicitly, e.g. `hessian_approx=lk.DiagGGN()` or
# `subset=lk.subsets.all`.
#
# ## Confidence away from the data
#
# To predict, we want to marginalize the posterior out:
#
# $$
#   p(y \mid x, \mathcal{D}) = \int \mathrm{softmax}(W \phi(x)) \, p(W \mid \mathcal{D}) \, dW .
# $$
#
# This integral is intractable. However, notice that the logits $f = W \phi(x)$
# are linear in $W$, so under the Laplace posterior they are Gaussian,
# $\mathcal{N}(f \mid m(x), V(x))$ with $m(x) = W_\text{MAP} \phi(x)$ and $V(x)
# = J(x) \Lambda^{-1} J(x)^\top$, where $J(x)$ is the Jacobian of the logits
# with respect to $\mathrm{vec}(W)$. The default predictive then uses the probit
# approximation,
#
# $$
#   p(y = k \mid x, \mathcal{D}) \approx \mathrm{softmax}_k \left( \frac{m(x)}{\sqrt{1 + \frac{\pi}{8} \, \mathrm{diag} \, V(x)}} \right) ,
# $$
#
# with the division taken elementwise, after subtracting the mean over the
# classes from the logits.
#
# The plots show the confidence, i.e. the highest class probability, on a grid
# that reaches well past the data. Both panels share the same color levels.

# %%
lin = np.linspace(-15, 15, 50)
gx, gy = np.meshgrid(lin, lin)
grid = torch.tensor(np.stack([gx.ravel(), gy.ravel()], axis=1), dtype=torch.float32)
levels = np.arange(0.3, 1.01, 0.1)

with torch.no_grad():
    map_conf = model(grid).softmax(-1).max(-1).values
    map_on = model(x).softmax(-1).max(-1).values
lap_conf = lk.predict(post, grid).probs.max(-1).values
lap_on = lk.predict(post, x).probs.max(-1).values

colors = ["coral", "yellow", "yellowgreen", "violet"]


def show(confs, names):
    fig, axes = plt.subplots(1, len(confs), figsize=(4.5 * len(confs), 4), sharey=True)
    for ax, conf, name in zip(axes, confs, names, strict=True):
        im = ax.contourf(
            gx, gy, conf.numpy().reshape(gx.shape), levels=levels, cmap="Blues", alpha=0.7
        )
        for k, c in enumerate(colors):
            ax.scatter(
                x_np[y_np == k, 0], x_np[y_np == k, 1], c=c, edgecolors="k", linewidths=0.5, s=14
            )
        ax.set_title(name)
        ax.set_aspect("equal")
    fig.colorbar(im, ax=list(axes), label="max class probability")
    return fig


fig = show([map_conf, lap_conf], ["MAP", "Last-layer Laplace"])
plt.show()

# %% [markdown]
# Notice how the MAP network is confident almost everywhere, even far from the
# data. The Laplace predictive is confident only near the training points and
# falls toward chance, $0.25$, away from them. This is because, for a ReLU
# network, $\lVert \phi(x) \rVert$ grows as $x$ moves away from the data, and
# with it the variance $V(x)$. Below are the mean confidences on the training
# points and on the grid points more than 5 units away from every training
# point.

# %%
far = torch.tensor([bool((torch.cdist(g[None], x) > 5).all()) for g in grid])


def summarize(name, on, conf):
    print(f"{name:<22} on data {on.mean():.2f}, far away {conf[far].mean():.2f}")


summarize("MAP", map_on, map_conf)
summarize("Laplace", lap_on, lap_conf)

# %% [markdown]
# ## The prior
#
# So far, $\lambda$ was the weight decay. By default, `lk.laplace` instead picks
# $\lambda$ by maximizing the log marginal likelihood (see [Choosing a
# curvature](curvature.py) for its formula). Here, it picks a $\lambda$ much
# larger than the weight decay, printed below. A larger $\lambda$ shrinks
# $\Lambda^{-1}$, and with it $V(x)$, so the predictive is confident over a much
# wider region, and the mean confidence far away goes up. That is, the prior
# decides how fast the confidence falls off away from the data.

# %%
tuned = lk.laplace(model, loader, likelihood=lk.Categorical())
print("tuned prior precision:", float(tuned.prior.prior_precision))
tuned_conf = lk.predict(tuned, grid).probs.max(-1).values
tuned_on = lk.predict(tuned, x).probs.max(-1).values
summarize("Laplace (tuned prior)", tuned_on, tuned_conf)
fig2 = show([lap_conf, tuned_conf], ["Prior = weight decay", "Tuned prior"])
plt.show()

# %% [markdown]
# The tuned $\lambda$ is about 1.05, and the mean confidence far away goes from
# 0.33 to 0.76. The marginal likelihood only looks at the training data, so
# nothing in it asks for low confidence far from the data. When we want the
# faster fall-off, we keep a small $\lambda$ of our own with `prior=` and
# `hyper_tuner=None`, as above.
#
# ## Choosing λ by calibration
#
# Instead of the marginal likelihood, we can pick $\lambda$ by how well the
# predictive does on held-out data $\mathcal{D}_\text{val} = \{(x_m, y_m)\}_{m=1}^M$.
# Writing $p_m = p(y \mid x_m, \mathcal{D})$ for the predictive probabilities at
# $x_m$, the three objectives in `lk.objectives` are the negative log
# predictive density, the Brier score, and the expected calibration error (ECE):
#
# $$
#   \mathrm{NLL} = -\frac{1}{M} \sum_{m=1}^M \log p_{m, y_m} , \qquad
#   \mathrm{Brier} = \frac{1}{M} \sum_{m=1}^M \lVert p_m - e_{y_m} \rVert^2 , \qquad
#   \mathrm{ECE} = \sum_{b=1}^{B} \frac{\lvert I_b \rvert}{M} \, \big\lvert \mathrm{acc}(I_b) - \mathrm{conf}(I_b) \big\rvert ,
# $$
#
# where $e_y$ is the one-hot vector of class $y$, and $I_b$ holds the points
# whose confidence $\max_k p_{m,k}$ falls into the $b$-th of $B = 15$ equal-width
# bins. In each bin, $\mathrm{acc}$ is the fraction of correct predictions and
# $\mathrm{conf}$ the mean confidence. The ECE counts points in bins, so it has
# no gradient in $\lambda$. Hence, we pick $\lambda$ with all four objectives,
# the marginal likelihood included, by `lk.GridSearch` over the same candidates,
# from $10^{-4}$ to $10^7$. An objective is any function from a posterior to a
# scalar, so the marginal likelihood is `lambda p: -p.log_marglik()`.
#
# The validation split has 200 new points drawn from the same four blobs. For
# each choice, we print $\lambda$, the three validation scores, and the mean
# confidence far from the data.

# %%
_, _, centers = make_blobs(
    n_samples=500, centers=4, center_box=(-10, 10), random_state=37, return_centers=True
)
xv_np, yv_np = make_blobs(n_samples=200, centers=centers, cluster_std=1.2, random_state=1)
val = [(torch.tensor(xv_np, dtype=torch.float32), torch.tensor(yv_np))]

nll = lk.objectives.predictive_nll(val)
brier = lk.objectives.brier(val)
ece = lk.objectives.ece(val)
objectives = {
    "marg. lik.": lambda p: -p.log_marglik(),
    "val. NLL": nll,
    "Brier": brier,
    "ECE": ece,
}

lams = torch.logspace(-4, 7, 45)
picked = {
    name: lk.tune(post, lk.GridSearch(objective, values=lams))
    for name, objective in objectives.items()
}

print(f"{'λ from':<14} {'λ':>8} {'NLL':>6} {'Brier':>6} {'ECE':>6} {'far away':>9}")

for name, p in {"weight decay": post, **picked}.items():
    far_conf = lk.predict(p, grid).probs.max(-1).values[far].mean()
    scores = " ".join(f"{float(objective(p)):6.3f}" for objective in (nll, brier, ece))
    print(f"{name:<14} {float(p.prior.prior_precision):8.3g} {scores} {float(far_conf):9.2f}")

# %% [markdown]
# On the grid, the marginal likelihood picks $\lambda = 1$, next to the 1.05
# that the default tuner found above. All three held-out objectives go for more: the
# NLL picks $\lambda \approx 178$, the ECE $\lambda = 10$, and the Brier score
# $\lambda = 10^6$, where the posterior has almost no variance left and the
# predictive is the MAP network's. With `lk.Minimize` instead of the grid, the
# Brier score keeps raising $\lambda$ to about $1.2 \times 10^6$. The scores of
# the three choices barely differ: the NLL is 0.082 or 0.083 and the Brier
# score 0.036 or 0.037. Against the weight decay, which gives an NLL of 0.170 and
# an ECE of 0.089, every tuned $\lambda$ does better on the validation points.
#
# Far from the data, the picture is different. The validation points come
# from the same blobs as the training points, so none of these objectives sees
# that region. There, the marginal likelihood keeps a mean confidence of 0.75,
# the ECE 0.88, the NLL 0.92 and the Brier score 0.93, as high as the MAP
# network. Good calibration on data like the training data says nothing about
# the confidence away from it. For low confidence far away, we need validation
# data that reaches that far, or a small $\lambda$ of our own.
#
# ## Laplace bridge
#
# The probit approximation is not the only way to integrate the softmax
# against the Gaussian $\mathcal{N}(f \mid \mu, \Sigma)$ over the logits. The
# Laplace bridge (Hobbhahn et al., 2022)[^hobbhahn2022] maps it to a Dirichlet
# distribution $\mathrm{Dir}(\pi \mid \alpha)$ over the class probabilities
# $\pi$ instead. The Dirichlet has a Laplace approximation of its own in the
# softmax basis, a Gaussian over the logits, and the bridge inverts that map.
# Given the logit means $\mu_k$ and variances $v_k$ of $K$ classes, it gives
#
# $$
#   \alpha_k = \frac{1}{v_k} \left( 1 - \frac{2}{K} + \frac{e^{\mu_k}}{K^2} \sum_{j=1}^K e^{-\mu_j} \right) ,
#   \qquad
#   p(y = k \mid x, \mathcal{D}) \approx \mathbb{E}_{\mathrm{Dir}(\pi \mid \alpha)}[\pi_k] = \frac{\alpha_k}{\sum_j \alpha_j} .
# $$
#
# The softmax ignores a shift shared by all logits, and the bridge assumes the
# logits carry none. So before the formula we condition the Gaussian on
# $\mathbf{1}^\top f = 0$:
#
# $$
#   \mu' = \mu - \frac{\Sigma \mathbf{1} \, \mathbf{1}^\top \mu}{\mathbf{1}^\top \Sigma \mathbf{1}} ,
#   \qquad
#   v'_k = \Sigma_{kk} - \frac{(\Sigma \mathbf{1})_k^2}{\mathbf{1}^\top \Sigma \mathbf{1}} ,
# $$
#
# and use $\mu'$ and $v'$ in place of $\mu$ and $v$. Notice that multiplying
# every $v'_k$ by the same factor multiplies every $\alpha_k$ by its inverse,
# which cancels in $\alpha_k / \sum_j \alpha_j$. The mean of the plain bridge
# thus sees the variances only through their ratios across classes. The
# normalized bridge puts their size back in. It divides $\mu'$ by $\sqrt{c}$,
# with $c = \bar{v}' / \sqrt{K / 2}$ and $\bar{v}'$ the average of the $v'_k$.
#
# The bridge is a link, like the probit, so we pass it to `lk.Linearized`. We
# use the posterior with the weight decay as the prior again and compare the
# three links on the confidence plots from before.

# %%
bridge = lk.Linearized(link=lk.Bridge())
normalized = lk.Linearized(link=lk.Bridge(normalized=True))

bridge_conf = lk.predict(post, grid, bridge).probs.max(-1).values
bridge_on = lk.predict(post, x, bridge).probs.max(-1).values
normalized_conf = lk.predict(post, grid, normalized).probs.max(-1).values
normalized_on = lk.predict(post, x, normalized).probs.max(-1).values

summarize("Laplace, probit", lap_on, lap_conf)
summarize("Laplace, bridge", bridge_on, bridge_conf)
summarize("Laplace, normalized", normalized_on, normalized_conf)

fig3 = show([lap_conf, bridge_conf, normalized_conf], ["Probit", "Bridge", "Normalized bridge"])
plt.show()

# %% [markdown]
# The plain bridge is as confident as the MAP network, both on the data and
# far away. To see why, we compute the shift-corrected variances $v'$ on the
# grid points far from the data.

# %%
cov = lk.predict_outputs(post, grid[far]).cov.per_input()  # [points, 4, 4]
row = cov.sum(-1)
v = cov.diagonal(dim1=-2, dim2=-1) - row**2 / row.sum(-1, keepdim=True)
ratio = v.max(-1).values / v.min(-1).values
print(f"far away: mean v' {float(v.mean()):.0f}, median max/min {float(ratio.median()):.1f}")

# %% [markdown]
# Far away, the corrected variances are huge, 4165 on average, but they are
# huge together: the largest is a median 2.1 times the smallest. Their size
# cancels in the mean of the plain bridge, and what is left is close to the
# softmax of $\mu'$. The
# normalized bridge divides the logits by the root of that size, and its
# confidence far away drops to 0.32, close to the probit's 0.33. On the data,
# though, it is the least confident of the three, at 0.73.
#
# With two classes, $v'_1 = v'_2$ always, since $\mathbf{1}^\top f = 0$ ties
# $f_2$ to $-f_1$. Then $\alpha_k \propto e^{\mu'_k}$, and the plain bridge is
# the softmax of $\mu'$, whatever the covariance. For $K = 2$ we therefore keep
# the probit, which is the default.
#
# ## Under the hood
#
# `lk.laplace` is a thin composition of public steps. It wraps the model as a
# function of its parameters, probes what it supports, picks the parameters
# and the curvature, folds the batches into a statistic, and combines the
# result with the prior into a posterior. Then it tunes $\lambda$. [Step by
# step](by_hand.py) runs each of these by hand on the model of this page, so
# that we can replace any of them with our own code.

# %% [markdown]
# [^kristiadi2020]: Kristiadi, A., Hein, M., & Hennig, P. (2020). Being Bayesian, even just a bit, fixes overconfidence in ReLU networks. *ICML*. <https://arxiv.org/abs/2002.10118>
#
# [^hobbhahn2022]: Hobbhahn, M., Kristiadi, A., & Hennig, P. (2022). Fast predictive uncertainty for classification with Bayesian deep networks. *UAI*. <https://arxiv.org/abs/2003.01227>
