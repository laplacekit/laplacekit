# %% [markdown]
# ---
# title: Using a plugin
# jupyter: python3
# ---
#
# A plugin is a separate package that adds methods to **laplacekit**.
# We install [`laplacekit-subnet`](https://github.com/laplacekit/laplacekit/tree/main/plugins/subnet), which picks a subnetwork of weights for the Laplace approximation.
#
# ```bash
# pip install "laplacekit[subnet]"
# ```
#
# A plugin's method goes wherever a built-in one goes.
# Here, the subnetwork goes in as `subset=`: the 100 weights with the largest magnitude.

# %%
import torch
from sklearn.datasets import make_blobs
from torch import nn

import laplacekit as lk

x_np, y_np = make_blobs(
    n_samples=500, centers=4, cluster_std=1.2, random_state=37
)
x, y = torch.tensor(x_np, dtype=torch.float32), torch.tensor(y_np)

torch.manual_seed(0)
model = nn.Sequential(
    nn.Linear(2, 20), nn.ReLU(), nn.Linear(20, 20), nn.ReLU(), nn.Linear(20, 4)
)
opt = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=5e-4)

for _ in range(300):
    opt.zero_grad()
    nn.functional.cross_entropy(model(x), y).backward()
    opt.step()

model.eval()
post = lk.laplace(
    model,
    [(x, y)],
    likelihood=lk.Categorical(),
    subset=lk.subnet.largest_magnitude(100),
)
print(post)

# %% [markdown]
# Installing a plugin changes no defaults.
# Without `subset=`, `lk.laplace` still takes the last layer.
# The plugin is reached as `lk.subnet`, which imports it on first use.
#
# ## What is installed
#
# `lk.plugins()` lists the installed plugins and what each one provides.

# %%
print(lk.plugins())

# %% [markdown]
# `laplacekit-example` is the docs' own example from [Your first plugin](../plugins/first.py), installed with the docs and never published.
#
# `lk.health()` checks the install: the core first, then each plugin with its own checks.

# %%
print(lk.health())

# %% [markdown]
# ::: {.callout-note collapse="true" title="The math"}
# Subnetwork Laplace ([Daxberger et al., 2021](https://arxiv.org/abs/2010.14689)) keeps a Gaussian over $S$ chosen weights $\theta_S$ and fixes the others at their trained values,
#
# $$
#   p(\theta_S \mid \mathcal{D}) \approx \mathcal{N}\big(\hat\theta_S, (\lambda I + G_S)^{-1}\big) ,
# $$
#
# where $G_S$ is the GGN restricted to those weights.
# With $S$ small, $G_S$ fits as a dense matrix.
# :::
#
# The full tutorial: [Subnetworks](subnetwork.py)
