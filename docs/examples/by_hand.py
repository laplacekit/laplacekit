# %% [markdown]
# ---
# title: Step by step
# jupyter: python3
# ---
#
# ::: {.callout-tip title="In short"}
# ```{.python}
# fn = lk.core.ModelFn.from_module(model)
# caps = lk.core.probe(fn, lik, batch)
# fit = lk.core.fit_curvature(fn, lik, lk.subsets.last_layer(fn, caps), lk.DenseGGN(), capabilities=caps)
# stat = fit.update(fit.init(), batch)
# post = fit.posterior(fit.finalize(stat), prior=prior, n_data=stat.count)
# ```
# :::
#
# `lk.laplace` is a thin composition of public steps. On this page, we run
# them one by one on the classifier of the [classification
# page](classification.py), so that we can replace any of them with our own
# code. Recall its Laplace approximation,
#
# $$
#   p(\theta \mid \mathcal{D}) \approx \mathcal{N}(\theta \mid \theta_\text{MAP}, \Lambda^{-1}) , \qquad
#   \Lambda = \lambda I + N G , \qquad
#   G = \frac{1}{N} \sum_{i=1}^N J_i^\top H_i J_i ,
# $$
#
# where $J_i$ is the Jacobian of the network's output at $x_i$ and $H_i$ the
# Hessian of the negative log-likelihood with respect to that output. The sum
# over the examples splits over any partition of the data into batches. Let
# $s(B) = \sum_{i \in B} J_i^\top H_i J_i$ for a batch $B$. Then
#
# $$
#   s(B_1 \cup B_2) = s(B_1) + s(B_2) , \qquad s(\emptyset) = 0 , \qquad
#   G = \frac{1}{N} \, s(\mathcal{D}) .
# $$
#
# So a fit is a fold. We start from the empty statistic, add one batch at a
# time, and divide by the number of examples at the end. The prior precision
# $\lambda$ and the data size $N$ enter only after the fold, which is why
# changing them, e.g. when tuning $\lambda$, never touches the data again.

# %%
import copy

import matplotlib.pyplot as plt
import torch
from sklearn.datasets import make_blobs
from torch import nn

import laplacekit as lk

# %% [markdown]
# ## Data and model
#
# The data and the training are those of the classification page: $N = 500$
# points in 4 classes, and a ReLU network with batch norm trained with weight
# decay $5 \times 10^{-4}$. After training, we switch the model to float64.
# With $\lambda = 5 \times 10^{-4}$, the prior variance is $2000$, and in
# float32 the comparisons below would mostly show rounding. We also cut the
# data into five batches of 100 points.

# %%
x_np, y_np = make_blobs(
    n_samples=500, centers=4, cluster_std=1.2, center_box=(-10, 10), random_state=37
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
opt = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=5e-4)

for _ in range(5000):
    opt.zero_grad()
    nn.functional.cross_entropy(model(x), y).backward()
    opt.step()

model.eval().double()
x = x.double()
batches = [lk.Batch(xb, yb) for xb, yb in zip(x.split(100), y.split(100), strict=True)]

# %% [markdown]
# ## The model as a function
#
# The Jacobians $J_i$ are with respect to the parameters, so we first turn the
# module into a function of them. `lk.core.ModelFn.from_module` gives
# `fn(params, inputs)`, where `params` maps parameter names to tensors.
#
# Then `lk.core.probe` runs the model once on a batch and records what works on it
# through autograd: vector-Jacobian products (`vjp`), forward-mode products
# (`jvp`), `vmap`, which layers module hooks reach, and the order in which the
# parameterized modules ran. A method states what it needs, and a mismatch
# raises `CapabilityError` before any data are used.

# %%
fn = lk.core.ModelFn.from_module(model)
lik = lk.Categorical()
caps = lk.core.probe(fn, lik, batches[0])
print(caps)

# %% [markdown]
# A subset picks the parameters that get a posterior. `lk.subsets.last_layer`
# takes the parameters of the last module in `module_order`, here the $4
# \times 20$ weight matrix $W$. It returns a plain dict, so any dict of the
# model's parameters works too, e.g. `{"6.weight": model[6].weight}`.

# %%
params = lk.subsets.last_layer(fn, caps)
print({name: tuple(p.shape) for name, p in params.items()})

# %% [markdown]
# ## The curvature fit
#
# `lk.core.fit_curvature` binds the model, the likelihood, the parameters, and a
# curvature method into a fit. We take the method `lk.laplace` would pick,
# `lk.curvature.default`. It is a map from blocks of parameters to methods, and
# here every block (`'*'`) gets the dense GGN. A keyword `generator=` would
# seed methods that sample. Without it, the fit draws its seed from torch's
# global RNG, so `torch.manual_seed` makes it repeatable. Passing
# `caps` checks the method's requirements against the probe here, before any
# data. With `caps=None`, the checks are skipped.

# %%
approx = lk.curvature.default(params, caps)
fit = lk.core.fit_curvature(fn, lik, params, approx, capabilities=caps)
print(approx)

# %% [markdown]
# ## Folding the batches
#
# The fold is three calls. `fit.init()` is the empty statistic,
# `fit.update(stat, batch)` adds a batch, and `fit.finalize(stat)` turns the
# sum into the average $G$, as a curvature operator. Along the way, the
# statistic also counts the examples and sums the log-likelihood of the
# targets at $\theta_\text{MAP}$, the data-fit term of the marginal likelihood
# below.

# %%
stat = fit.init()

for batch in batches:
    stat = fit.update(stat, batch)

curv = fit.finalize(stat)
print("examples:", int(stat.count))
print("log-likelihood:", f"{float(stat.data_fit):.4f}")

# %% [markdown]
# Statistics add, so the grouping of the data doesn't matter. We can fold
# batch by batch, fold on several processes and add the results
# (`lk.distributed.all_reduce` does this), or take the whole data at once. To check,
# we fit the whole data as one batch, and also fit the two halves separately
# and add their statistics. Nothing here runs under `torch.no_grad`, so every
# result keeps its autograd graph (more on that at the end). We use `no_grad`
# where we only want the numbers.

# %%
one = fit.update(fit.init(), lk.Batch(x, y))
first, second = lk.Batch(x[:250], y[:250]), lk.Batch(x[250:], y[250:])
halves = fit.update(fit.init(), first) + fit.update(fit.init(), second)
prior = lk.priors.Isotropic(torch.tensor(5e-4, dtype=torch.float64))


def probs(s):
    posterior = fit.posterior(fit.finalize(s), prior=prior, n_data=s.count)

    with torch.no_grad():
        return lk.predict(posterior, x).probs


gap_one = float((probs(stat) - probs(one)).abs().max())
gap_halves = float((probs(stat) - probs(halves)).abs().max())

print("five batches vs one:   ", f"{gap_one:.1g}")
print("five batches vs halves:", f"{gap_halves:.1g}")

# %% [markdown]
# The predictive probabilities agree to rounding.

# %% [markdown]
# ## The posterior
#
# `fit.posterior` combines the curvature with a prior and the data size $N$ into
# $\Lambda = \lambda I + N G$. It fills in the model, the likelihood, and the
# parameter layout from the fit. Here, `n_data` is the number of
# examples the fold saw. When the batches are only part of the data, e.g. a
# subsample, it should be the size of the whole data set.

# %%
post = fit.posterior(curv, prior=prior, n_data=stat.count)
print(post)

# %% [markdown]
# This is the posterior that `lk.laplace` returns for the same prior with
# `hyper_tuner=None`:

# %%
ref = lk.laplace(model, batches, likelihood=lik, prior=prior, hyper_tuner=None)

with torch.no_grad():
    gap = (lk.predict(post, x).probs - lk.predict(ref, x).probs).abs().max()

print("max |probability difference|:", f"{float(gap):.1g}")

# %% [markdown]
# ## Tuning the prior
#
# By default, `lk.laplace` picks $\lambda$ by minimizing the negative log
# marginal likelihood (see [Choosing a curvature](curvature.py) for its
# formula). By hand, that is `lk.tune(post, lk.MarginalLikelihood())`. The
# marginal likelihood needs the data-fit term, which the statistic holds, so we
# copy it into the posterior first (`lk.laplace` does this for us). The tuner
# finds the hyperparameter itself, the prior's precision, runs L-BFGS on its
# logarithm and returns a new posterior, leaving `post` as it was. Since only
# $\lambda$ changes, tuning never refits.

# %%
post = copy.replace(post, data_fit=(stat.data_fit,))
tuned = lk.tune(post, lk.MarginalLikelihood())
default = lk.laplace(model, batches, likelihood=lik)

print("prior precision, by hand:   ", f"{float(tuned.prior.prior_precision):.4f}")
print("prior precision, lk.laplace:", f"{float(default.prior.prior_precision):.4f}")

# %% [markdown]
# `lk.GridSearch` takes an objective, here `lk.objectives.neg_log_marglik`
# (what `lk.MarginalLikelihood` minimizes), and a list of candidate values
# instead of an optimizer. It returns the posterior with the best one. Since the
# objective is a plain function of the posterior, we can also plot it. Below, it is evaluated on 25 values of $\lambda$ between $10^{-4}$ and
# $10^2$, with the grid's choice and the tuned value marked.

# %%
objective = lk.objectives.neg_log_marglik
values = torch.logspace(-4, 2, 25, dtype=torch.float64)
best = lk.tune(post, lk.GridSearch(objective, values=values))
print("prior precision, grid:      ", f"{float(best.prior.prior_precision):.4f}")

with torch.no_grad():
    curve = [float(objective(copy.replace(post, prior=lk.priors.Isotropic(v)))) for v in values]

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(values, curve, marker="o", markersize=3)
ax.axvline(float(best.prior.prior_precision), color="0.6", linestyle=":", label="grid")
ax.axvline(float(tuned.prior.prior_precision), color="k", linestyle="--", label="tune")
ax.set_xscale("log")
ax.set_xlabel(r"prior precision $\lambda$")
ax.set_ylabel("negative log marginal likelihood")
ax.legend()
plt.show()

# %% [markdown]
# The two lines almost coincide. The grid can only return one of its values,
# and $\lambda = 1$ is the one closest to the tuned $1.0541$.

# %% [markdown]
# ## Methods with several passes
#
# Some methods need more than one pass over the data: the first pass estimates
# something, such as an eigenbasis, that the next pass needs. Such a method is
# `Staged` (see [Curvatures](../plugins/curvature.py) for how to write one),
# and for it `lk.core.fit_curvature` returns a fit with `start()` only, since the
# number of passes is the method's business.
#
# `start()` returns a `lk.core.Running` step. Each step has the same `init` and
# `update` as before. Its `finish(stat)` returns either the next
# `lk.core.Running` step or, after the last pass, a `lk.core.Finished` with the result
# and one statistic per pass. The loop below is the one `lk.laplace` runs. A
# single-pass fit has `start()` too, so the same loop works for every
# method. We run it with KFAC, which takes one pass.

# %%
stepped = lk.core.fit_curvature(fn, lik, params, lk.KFAC(), capabilities=caps)
step = stepped.start()
passes = 0

while isinstance(step, lk.core.Running):
    stat_k = step.init()

    for batch in batches:
        stat_k = step.update(stat_k, batch)

    step = step.finish(stat_k)
    passes += 1

print("passes:", passes)
print(stepped.posterior(step.result, prior=prior, n_data=step.count))

# %% [markdown]
# `step.count` is the number of examples of the last pass, and `step.stats[-1]`
# is the statistic to pass to `lk.objectives.neg_log_marglik`.
#
# ## Gradients
#
# `lk.laplace` detaches the parameters and folds under `torch.no_grad`. By
# hand, the parameters are not detached, so the curvature part of the
# statistic, the posterior, and the log-determinant and prior terms of the
# objective are differentiable with respect to `params`. The data-fit term is
# the exception: the fold computes the log-likelihood under `no_grad`, so
# `stat.data_fit` is a value without a graph.
#
# Marginal-likelihood training of the network (Immer et al.,
# 2021)[^immer2021] needs the gradient of the whole negative log marginal
# likelihood with respect to $W$. So we put the log-likelihood back from a
# forward of our own, in place of the value. The objective keeps its value,
# and the gradient takes one call to `torch.autograd.grad`.

# %%
print("data-fit term has a graph:", stat.data_fit.requires_grad)

log_lik = lk.core.log_prob(lik, fn(params, x), y).sum()
loss = objective(post) + stat.data_fit - log_lik
print("same value:", bool(torch.isclose(loss, objective(post))))

(grad,) = torch.autograd.grad(loss, params["6.weight"])
print("gradient shape:", tuple(grad.shape))

# %% [markdown]
# [^immer2021]: Immer, A., Bauer, M., Fortuin, V., Rätsch, G., & Khan, M. E. (2021). Scalable marginal likelihood estimation for model selection in deep learning. *ICML*. <https://arxiv.org/abs/2104.04975>
