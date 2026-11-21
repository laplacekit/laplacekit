# %% [markdown]
# ---
# title: Subnetworks
# jupyter: python3
# ---
#
# A full covariance over every weight is too big for most networks, but a full
# covariance over a few hundred of them is cheap. A subnetwork Laplace picks
# those weights and keeps the rest fixed. We use the classifier of the
# [classification page](classification.py).

# %%
import torch
from sklearn.datasets import make_blobs
from torch import nn

import laplacekit as lk

x_np, y_np = make_blobs(
    n_samples=500, centers=4, cluster_std=1.2, center_box=(-10, 10), random_state=37
)
x, y = torch.tensor(x_np, dtype=torch.float32), torch.tensor(y_np)
data = [(x, y)]

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
    nn.functional.cross_entropy(model(x), y).backward()
    opt.step()

model.eval()

# %% [markdown]
# The strategies live in `lk.subnet` (the `laplacekit-subnet` plugin, installed
# with the `subnet` extra). `lk.subnet.largest_magnitude(100)` takes the 100 weights
# with the largest $|\theta|$, out of 640. `DenseGGN(joint=True)` gives them one
# dense covariance, across layers.

# %%
post_mag = lk.laplace(
    model,
    data,
    likelihood=lk.Categorical(),
    subset=lk.subnet.largest_magnitude(100),
    hessian_approx=lk.DenseGGN(joint=True),
)
print(post_mag)

# %% [markdown]
# `lk.subnet.largest_variance(diag, 100)` takes the 100 weights with the largest
# marginal variance under a cheaper posterior, usually a `Diag` over all
# weights. It also takes the SWAG variances of
# `lk.subnet.swag_variance(model, data, likelihood=lk.Categorical())`, which
# skip weights that the data never move ([Subnetworks](../examples/subnetwork.py)
# compares the two).

# %%
diag = lk.laplace(
    model, data, likelihood=lk.Categorical(), subset=lk.subsets.all, hessian_approx=lk.DiagGGN()
)
post_var = lk.laplace(
    model,
    data,
    likelihood=lk.Categorical(),
    subset=lk.subnet.largest_variance(diag, 100),
    hessian_approx=lk.DenseGGN(joint=True),
)
print(post_var)

# %% [markdown]
# The two pick different weights. By magnitude, the 100 come from the first
# and last layers and from the scales of both batch norms. By variance, 98 of
# them are in the middle layer, `3.weight`. Both predict like any other
# posterior. We look at the four corners of $[-15, 15]^2$, far from the data.
# The MAP network puts probability 1.0 on one class in three of them. The
# subnetworks stay sure in the first corner (0.98 and 0.97) and drop to
# between 0.30 and 0.66 in the other three.

# %%
far = torch.tensor([[-15.0, 15.0], [15.0, -15.0], [15.0, 15.0], [-15.0, -15.0]])

with torch.no_grad():
    print("MAP      ", model(far).softmax(-1).max(-1).values)

print("magnitude", lk.predict(post_mag, far).probs.max(-1).values)
print("variance ", lk.predict(post_var, far).probs.max(-1).values)

# %% [markdown]
# Three more ways to choose, one line each: `lk.subnet.random(n)` draws $n$
# weights at random, `lk.subsets.from_mask` takes a boolean mask per
# parameter, and `lk.subsets.named` takes whole parameters by name. The mask
# below keeps 20 weights of the last layer, and the two names hold 500.

# %%
mask = torch.zeros(4, 20, dtype=torch.bool)
mask[:, :5] = True  # the weights from the first five features

subnets = [
    lk.subnet.random(100),
    lk.subsets.from_mask({"6.weight": mask}),
    lk.subsets.named("3.*", "6.*"),
]

for subnet in subnets:
    post = lk.laplace(
        model,
        data,
        likelihood=lk.Categorical(),
        subset=subnet,
        hessian_approx=lk.DenseGGN(joint=True),
    )
    print(sum(m.numel() for m in post.mean.values()), "weights")

# %% [markdown]
# KFAC reaches whole weight matrices through module hooks, so it refuses a
# subnetwork before it reads any data:

# %%
try:
    lk.laplace(
        model,
        data,
        likelihood=lk.Categorical(),
        subset=lk.subnet.largest_magnitude(100),
        hessian_approx=lk.KFAC(),
    )
except lk.CapabilityError as e:
    print("CapabilityError:", e)

# %% [markdown]
# ::: {.callout-note collapse="true" title="The math"}
# Split the weights into the chosen ones $\theta_S$ and the rest $\theta_R$.
# Subnetwork Laplace (Daxberger et al. 2021) keeps $\theta_R$ fixed at the MAP
# and puts a Gaussian on $\theta_S$ alone,
#
# $$
#   p(\theta_S \mid \mathcal{D}) \approx \mathcal{N}\big(\theta_S \mid \theta_{S*}, (\lambda I + N G_{SS})^{-1}\big) ,
# $$
#
# where $G_{SS}$ is the block of the generalized Gauss-Newton matrix for the
# chosen entries. For $|S| = 100$, that is a $100 \times 100$ matrix, whatever
# the size of the network. The predictive linearizes the network in
# $\theta_S$ only. Choosing by variance follows the paper: it keeps the
# entries whose marginal variance, under a diagonal Laplace over all weights,
# is largest.
# :::
#
# The full tutorial: [Subnetworks](../examples/subnetwork.py)
