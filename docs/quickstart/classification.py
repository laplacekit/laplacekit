# %% [markdown]
# ---
# title: Classification
# jupyter: python3
# ---
#
# A ReLU classifier is confident far away from its training data. A Laplace
# approximation on its last layer makes it much less so. We train a small network on four
# 2D blobs.

# %%
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.datasets import make_blobs
from torch import nn

import laplacekit as lk

x_np, y_np = make_blobs(
    n_samples=500,
    centers=4,
    cluster_std=1.2,
    center_box=(-10, 10),
    random_state=37,
)
x, y = torch.tensor(x_np, dtype=torch.float32), torch.tensor(y_np)

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
    nn.functional.cross_entropy(model(x), y).backward()
    opt.step()

# %% [markdown]
# The fit is one call. We take the weight decay as the prior precision and keep
# it fixed. We also print the prior precision that the default tuning picks.

# %%
model.eval()
post = lk.laplace(
    model,
    [(x, y)],
    likelihood=lk.Categorical(),
    prior=lk.priors.Isotropic(5e-4),
    hyper_tuner=None,
)
print(post)

tuned = lk.laplace(
    model, [(x, y)], likelihood=lk.Categorical()
)  # the default, for comparison
print("tuned prior precision:", float(tuned.prior.prior_precision))

# %% [markdown]
# We plot the confidence, the highest class probability, on a grid that
# reaches well past the data.

# %%
lin = np.linspace(-15, 15, 50)
gx, gy = np.meshgrid(lin, lin)
grid = torch.tensor(
    np.stack([gx.ravel(), gy.ravel()], axis=1), dtype=torch.float32
)

with torch.no_grad():
    conf_map = model(grid).softmax(-1).max(-1).values

conf_lap = lk.predict(post, grid).probs.max(-1).values

fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=True)

for ax, conf, name in zip(
    axes, [conf_map, conf_lap], ["MAP", "Laplace"], strict=True
):
    im = ax.contourf(
        gx,
        gy,
        conf.reshape(gx.shape),
        levels=np.arange(0.3, 1.01, 0.1),
        cmap="Blues",
    )
    ax.scatter(
        x_np[:, 0],
        x_np[:, 1],
        c=y_np,
        cmap="Set2",
        edgecolors="k",
        linewidths=0.5,
        s=12,
    )
    ax.set_title(name)
    ax.set_aspect("equal")

fig.colorbar(im, ax=list(axes), label="max class probability")
plt.show()

# %% [markdown]
# The MAP network is sure of itself almost everywhere. The Laplace predictive
# is sure near the data, and its confidence falls off away from it in every
# direction. How fast it falls off depends on the prior. By default,
# `lk.laplace` tunes the prior precision on the marginal likelihood. Here that
# gives $\lambda \approx 1.05$ instead of the weight decay $5
# \times 10^{-4}$, and with it the predictive stays confident far away in more
# than half of the directions. The full tutorial compares the two.
#
# To use another $\lambda$, we fix it, or pick it by the NLL on held-out data
# (a loader or a list of `(inputs, targets)` pairs), with L-BFGS (`lk.Minimize`)
# or on a grid (`lk.GridSearch`):
#
# ```python
# post = lk.laplace(model, data, lik, prior=lk.priors.Isotropic(λ), hyper_tuner=None)
# post = lk.tune(post, lk.Minimize(lk.objectives.predictive_nll(val_loader)))
# post = lk.tune(post, lk.GridSearch(lk.objectives.predictive_nll(val_loader), values=lams))
# ```
#
# ## Tuning for calibration
#
# The held-out objective doesn't have to be the NLL. `lk.objectives.brier` is
# the Brier score and `lk.objectives.ece` the expected calibration error. We
# draw 200 validation points from the same four blobs and pick $\lambda$ with
# each. The ECE counts points in bins, so it has no gradient: `lk.Minimize`
# refuses it, and we search a grid with `lk.GridSearch` instead. The Brier score
# is smooth, so `lk.Minimize` works.

# %%
_, _, centers = make_blobs(
    n_samples=500,
    centers=4,
    center_box=(-10, 10),
    random_state=37,
    return_centers=True,
)
xv_np, yv_np = make_blobs(
    n_samples=200, centers=centers, cluster_std=1.2, random_state=1
)
val = [(torch.tensor(xv_np, dtype=torch.float32), torch.tensor(yv_np))]

ece, brier = lk.objectives.ece(val), lk.objectives.brier(val)

by_ece = lk.tune(post, lk.GridSearch(ece, values=torch.logspace(-4, 2, 25)))
by_brier = lk.tune(post, lk.Minimize(brier))

print(f"{'λ from':<16} {'λ':>8} {'ECE':>6} {'Brier':>6}")

for name, p in [
    ("weight decay", post),
    ("marg. lik.", tuned),
    ("grid on ECE", by_ece),
    ("tune on Brier", by_brier),
]:
    print(
        f"{name:<16} {float(p.prior.prior_precision):8.3g} "
        f"{float(ece(p)):6.3f} {float(brier(p)):6.3f}"
    )

# %% [markdown]
# With the weight decay as the prior, the ECE is 0.089. The
# marginal-likelihood $\lambda \approx 1.05$ brings it down to 0.023, and the
# grid on the ECE picks $\lambda = 10$, with 0.020. The Brier score falls from
# 0.060 to 0.037 with either. `lk.Minimize` on it keeps raising $\lambda$, to about
# $1.2 \times 10^6$, where the posterior has almost no variance left, and gets
# 0.036. On validation data like the training data, the Brier score prefers the
# MAP network.
#
# ## Laplace bridge
#
# The probit is one way to turn the Gaussian over the logits into class
# probabilities. The Laplace bridge is another: it matches a Dirichlet to the
# Gaussian and predicts its mean. It is a link, like the probit, so we pass it
# to `lk.Linearized`.

# %%
bridge = (
    lk.predict(post, grid, lk.Linearized(link=lk.Bridge())).probs.max(-1).values
)
normed = (
    lk.predict(post, grid, lk.Linearized(link=lk.Bridge(normalized=True)))
    .probs.max(-1)
    .values
)

links = {
    "MAP": conf_map,
    "probit": conf_lap,
    "bridge": bridge,
    "normalized": normed,
}

for name, conf in links.items():
    print(f"{name:<11} mean confidence on the grid {float(conf.mean()):.3f}")

cov = lk.predict_outputs(post, grid).cov.per_input()  # [2500, 4, 4]
row = cov.sum(-1)
var = cov.diagonal(dim1=-2, dim2=-1) - row**2 / row.sum(
    -1, keepdim=True
)  # shared shift removed
ratio = var.max(-1).values / var.min(-1).values
print(
    f"logit variance: mean {float(var.mean()):.0f}, median max/min "
    f"{float(ratio.median()):.1f}"
)

# %% [markdown]
# The probit gives 0.446, but the plain bridge is about as confident as the
# MAP network: 0.933 on average over the grid, against 0.936. Its mean depends
# on the logit variances (after removing the shift shared by all logits) only
# through their ratios across classes. Here they are large together, 2229 on
# average with a median max/min of 2.3, so their size mostly cancels. The
# normalized bridge, `lk.Bridge(normalized=True)`, folds the size back in by
# dividing the logit mean by a factor that grows with the average variance, and
# gives 0.390.
#
# The bridge is for three classes or more. At two classes it ignores the
# variances, so there we keep the probit, which is the default.
#
# ## Binary and multi-label
#
# When each input can have several labels at once, every output is its own yes
# or no. `lk.Bernoulli()` is the sigmoid likelihood for that, with targets of 0
# and 1 shaped like the outputs. Our toy has three labels on the plane: "right
# of the origin", "above it" and "within distance 2 of it".

# %%
torch.manual_seed(0)
xb = 6 * torch.rand(300, 2) - 3
yb = torch.stack(
    [xb[:, 0] > 0, xb[:, 1] > 0, xb.norm(dim=1) < 2], dim=1
).float()

net = nn.Sequential(
    nn.Linear(2, 32), nn.ReLU(), nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 3)
)
opt = torch.optim.Adam(net.parameters(), lr=1e-2)

for _ in range(500):
    opt.zero_grad()
    nn.functional.binary_cross_entropy_with_logits(net(xb), yb).backward()
    opt.step()

post_b = lk.laplace(net, [(xb, yb)], likelihood=lk.Bernoulli())
print(post_b)

# %% [markdown]
# `lk.predict` gives one probability per label. We ask at a point inside the
# data and at one far outside it.

# %%
points = torch.tensor([[1.0, 1.0], [20.0, 20.0]])

with torch.no_grad():
    p_map = net(points).sigmoid()

p_lap = lk.predict(post_b, points).probs

for pt, a, b in zip(
    points.tolist(), p_map.tolist(), p_lap.tolist(), strict=True
):
    print(
        f"x = {pt}:  MAP {[round(v, 2) for v in a]},  Laplace "
        f"{[round(v, 2) for v in b]}"
    )

# %% [markdown]
# The point (1, 1) has all three labels, and both are sure of them. The point
# (20, 20) is right of and above the origin but far from it, and the MAP
# network gives 1.0, 1.0 and 0.0. The Laplace probabilities there, 0.95, 0.93
# and 0.01, are only a little less sure.
#
# ::: {.callout-note collapse="true" title="The math"}
# Write the network as $f(x) = W \phi(x)$, with the last layer $W$ on top of the
# features $\phi$. The Laplace approximation keeps $\phi$ fixed and puts a
# Gaussian on $W$,
#
# $$
#   p(W \mid \mathcal{D}) \approx \mathcal{N}\big(\mathrm{vec}(W) \mid \mathrm{vec}(W_*), (\lambda I + N G)^{-1}\big) ,
# $$
#
# with $G$ the generalized Gauss-Newton matrix of the last layer and $\lambda$
# the prior precision, here the weight decay. The logits are linear in $W$, so they are
# Gaussian with mean $W_* \phi(x)$ and a covariance $V(x)$, and the probit
# approximation gives
#
# $$
#   p(y \mid x, \mathcal{D}) \approx \mathrm{softmax}\Big(\frac{W_* \phi(x)}{\sqrt{1 + \frac{\pi}{8} \mathrm{diag} V(x)}}\Big) .
# $$
#
# For a ReLU network, $\lVert \phi(x) \rVert$ grows away from the data, and so
# does $V(x)$, which pulls the probabilities toward uniform. With two classes,
# the probit is applied to the difference of the two logits, $f_1 - f_2$, with
# its variance $v_d = V_{11} + V_{22} - 2 V_{12}$.
#
# A Bernoulli output is one logit $f_k$ per label, and each gets its own
# sigmoid probit, $\sigma\big(\mu_k / \sqrt{1 + \frac{\pi}{8} V_{kk}(x)}\big)$.
#
# The Laplace bridge (Hobbhahn et al., 2022) maps the logit mean $\mu$ and
# variances $v$ (after removing the shift shared by all logits) to a Dirichlet
# with
#
# $$
#   \alpha_k = \frac{1}{v_k} \Big(1 - \frac{2}{K} + \frac{e^{\mu_k}}{K^2} \sum_j e^{-\mu_j}\Big) ,
# $$
#
# and predicts its mean $\alpha / \sum_j \alpha_j$. At $K = 2$ the two
# variances are equal, so they cancel in that mean.
# :::
#
# The full tutorial: [Classification](../examples/classification.py)
