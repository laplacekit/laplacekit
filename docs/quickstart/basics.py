# %% [markdown]
# ---
# title: Basics
# jupyter: python3
# ---
#
# Train a model as usual, then fit a Laplace approximation to it and predict.
# We use a small classifier on four 2D blobs.

# %%
import tempfile
from pathlib import Path

import torch
from sklearn.datasets import make_blobs
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import laplacekit as lk

x_np, y_np = make_blobs(
    n_samples=500, centers=4, cluster_std=1.2, random_state=37
)
x, y = torch.tensor(x_np, dtype=torch.float32), torch.tensor(y_np)
loader = DataLoader(TensorDataset(x, y), batch_size=100)

torch.manual_seed(0)
model = nn.Sequential(
    nn.Linear(2, 20),
    nn.ReLU(),
    nn.Linear(20, 20),
    nn.ReLU(),
    nn.Dropout(0.1),
    nn.Linear(20, 4),
)
opt = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=5e-4)

for _ in range(300):
    opt.zero_grad()
    nn.functional.cross_entropy(model(x), y).backward()
    opt.step()

# %% [markdown]
# The fit is one line, and so is the prediction.

# %%
model.eval()  # dropout off
post = lk.laplace(model, loader, lk.Categorical())
probs = lk.predict(post, x[:3]).probs
print(probs)

# %% [markdown]
# ## What it chose
#
# `print(post)` shows every choice `lk.laplace` made for us.

# %%
print(post)

# %% [markdown]
# Line by line:
#
# - `subset`: which weights get a distribution. The default is the last layer,
#   here its weight and bias.
# - `hessian_approx`: how the Hessian of the loss is approximated. The last
#   layer is small, so it gets the full GGN matrix (`DenseGGN`).
# - `curvature`: what was fitted, one dense block of 84 parameters.
# - `prior`: the prior precision $\lambda$ after tuning.
# - `likelihood`: the likelihood we passed.
# - `hyper_tuner`: how $\lambda$ was picked. `MarginalLikelihood()` maximizes
#   the marginal likelihood of the training data.
# - `n_data`: the number of training examples $N$, read from the loader.
#
# ## Changing a default
#
# Each choice is one keyword argument.

# %%
post_all = lk.laplace(model, loader, lk.Categorical(), subset=lk.subsets.all)
post_kfac = lk.laplace(
    model, loader, lk.Categorical(), hessian_approx=lk.KFAC()
)
post_fixed = lk.laplace(
    model,
    loader,
    lk.Categorical(),
    prior=lk.priors.Isotropic(1.0),
    hyper_tuner=None,
)
print(post_fixed)

# %% [markdown]
# `subset=lk.subsets.all` puts the distribution on every weight,
# `hessian_approx=lk.KFAC()` uses a Kronecker-factored curvature, and `prior=`
# with `hyper_tuner=None` fixes $\lambda$ instead of tuning it. [Classification](classification.py) shows
# when the tuned $\lambda$ isn't the one we want, and how to pick another.
#
# ## Samples of the outputs
#
# `lk.sample_outputs(post, x, 16)` draws 16 outputs of the linearized network
# at `x` and stacks them along a new first axis. For a
# classifier, those are logits, and `softmax` turns them into probabilities.

# %%
logits = lk.sample_outputs(
    post, x[:3], 16, generator=torch.Generator().manual_seed(0)
)
print(logits.shape)
print(
    logits.softmax(-1)[:4, 0].round(decimals=3)
)  # four draws for the first input

# %% [markdown]
# Three of the four draws put the first input in class 3, each with a
# probability of at least 0.997. The first puts it in class 0.
#
# ## Save and load
#
# `lk.save(post, "post.pt")` writes the posterior without the model, as
# `torch.save` writes a `state_dict`. `lk.load("post.pt", model)` reads it back
# onto the model, so in a new session we restore the model's weights first.
# Here we save into a temporary directory.

# %%
with tempfile.TemporaryDirectory() as tmp:
    lk.save(post, Path(tmp) / "post.pt")
    loaded = lk.load(Path(tmp) / "post.pt", model)

print(torch.equal(lk.predict(loaded, x[:3]).probs, probs))

# %% [markdown]
# The file holds tensors and plain data, never code, so `lk.load` runs nothing
# from it. Functions are saved by name. A lambda, say as `mask_fn` of
# `lk.Categorical`, has none, so `lk.save` leaves it out and `lk.load` asks for
# the likelihood back: `lk.load("post.pt", model, likelihood=lik)`. A
# module-level function decorated with `@lk.core.serializable` is saved by name
# and found again on load.

# %% [markdown]
# ## Pitfalls
#
# - Call `model.eval()` before the fit. In training mode, dropout makes the
#   model random. The default then skips the methods that need a deterministic
#   model, and asking for one of them raises (the first error below).
# - The data is a loader (or a list) of `(inputs, targets)` pairs, as in
#   training. One pair on its own, or a `Dataset`, raises a `TypeError` that
#   says what to pass.
# - `lk.Categorical()` is for class labels, with logits as the model's outputs.
#   `lk.Gaussian()` is for real-valued targets of the same shape as the outputs.
# - A `CapabilityError` means the method needs something the model or the
#   likelihood can't do. A `StructureError` means the request doesn't fit the
#   model or the data. Both come before the fit: **laplacekit** runs the model on
#   the first batch, checks the request, and only then goes through the data.

# %%
model.train()  # dropout is on again

try:
    lk.laplace(model, loader, lk.Categorical(), hessian_approx=lk.DiagGGN())
except lk.CapabilityError as e:
    print("CapabilityError:", e)

model.eval()

try:
    lk.laplace(model, loader, lk.Categorical(), subset=lk.subsets.named("fc.*"))
except lk.StructureError as e:
    print("StructureError:", e)

# %% [markdown]
# ::: {.callout-note collapse="true" title="The math"}
# Let $\theta_*$ be the trained weights. The Laplace approximation is a Gaussian
# around them,
#
# $$
#   p(\theta \mid \mathcal{D}) \approx \mathcal{N}\big(\theta \mid \theta_*, (\lambda I + N G)^{-1}\big) ,
# $$
#
# where $\lambda$ is the prior precision, $N$ the number of training examples,
# and $G$ a curvature of the loss averaged over the data, here the generalized
# Gauss-Newton matrix. To predict, `lk.predict` linearizes the network around
# $\theta_*$, so its outputs are Gaussian, $\mathcal{N}(f_{\theta_*}(x), J
# \Sigma J^\top)$ with $J$ the Jacobian and $\Sigma$ the covariance above. For
# a classifier, the probit approximation then turns that Gaussian into class
# probabilities.
# :::
#
# The full tutorial: [Classification](../examples/classification.py)
