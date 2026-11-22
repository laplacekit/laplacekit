# %% [markdown]
# ---
# title: Tuning objectives
# jupyter: python3
# ---
#
# ::: {.callout-tip}
# ## New to protocols?
# The contract below is a function signature.
# Any function with that signature works, and your editor checks it where you pass it in.
# [Protocols in one page](protocols.py) explains protocols in general.
# :::
#
# A tuning objective is a function from a posterior to a 0-dim tensor.
# Lower is better:
#
# ```python
# type Objective[P] = Callable[[P], Tensor]
# ```
#
# A hyper tuner takes it.
# For example, `lk.laplace(..., hyper_tuner=lk.Minimize(objective))` minimizes it over the prior precision (and a learned noise) after the fit.
# `lk.GridSearch(objective, values)` searches a grid instead.
# `lk.tune` runs the same tuners on a posterior that is already fitted.
# **laplacekit** ships objectives such as the negative log marginal likelihood (what the default `lk.MarginalLikelihood()` minimizes) and the predictive NLL on held-out data.
# We add another, the Brier score on held-out data.

# %%
import copy

import torch
import torch.nn.functional as F
from sklearn.datasets import make_blobs
from torch import Tensor, nn

import laplacekit as lk
from laplacekit.core import Objective, Posterior


def brier(data: list[tuple[Tensor, Tensor]]) -> Objective[Posterior]:
    """Mean squared distance of the probabilities from the one-hot targets."""

    def objective(posterior: Posterior) -> Tensor:
        total, n = torch.zeros(()), 0

        for inputs, targets in data:
            probs = lk.predict(posterior, inputs).probs
            onehot = F.one_hot(targets, probs.shape[-1])
            total = total + (probs - onehot).pow(2).sum()
            n += len(targets)

        return total / n

    return objective


# %% [markdown]
# Like the built-ins, `brier` is a factory: it takes the held-out batches and returns the objective.
# We split four overlapping blobs into a training set and a validation set, train an MLP, and fit three posteriors, each with a different objective for the prior precision.

# %%
x_np, y_np = make_blobs(
    n_samples=600, centers=4, cluster_std=2.0, random_state=1
)
x, y = torch.tensor(x_np, dtype=torch.float32), torch.tensor(y_np)
x_train, y_train, x_val, y_val = x[:400], y[:400], x[400:], y[400:]
val = [(x_val, y_val)]

torch.manual_seed(0)
model = nn.Sequential(
    nn.Linear(2, 32), nn.Tanh(), nn.Linear(32, 32), nn.Tanh(), nn.Linear(32, 4)
)
opt = torch.optim.Adam(model.parameters(), lr=1e-2)

for _ in range(500):
    opt.zero_grad()
    nn.functional.cross_entropy(model(x_train), y_train).backward()
    opt.step()

tuners = {
    "marginal likelihood": lk.MarginalLikelihood(),
    "validation NLL": lk.Minimize(lk.objectives.predictive_nll(val)),
    "validation Brier": lk.Minimize(brier(val)),
}

for name, tuner in tuners.items():
    post = lk.laplace(
        model, [(x_train, y_train)], lk.Categorical(), hyper_tuner=tuner
    )
    nll, score = lk.objectives.predictive_nll(val)(post), brier(val)(post)
    lam = post.prior.prior_precision
    print(
        f"{name:>19}: prior precision {lam:.3f}, val NLL {nll:.4f}, val Brier "
        f"{score:.4f}"
    )

# %% [markdown]
# The validation NLL picks the smallest precision and gets the lowest NLL.
# The Brier objective picks a precision close to the marginal likelihood's, and the two tie on the Brier score.
# On this problem, the scores differ only in the third decimal, so the choice of objective matters little.
# The `hyper_tuner` line of `print(post)` names the objective.
# For a closure, it is the name of the factory, here `brier`.
#
# `lk.tune` runs a tuner on a posterior that is already fitted.
# The tuner finds what to tune by itself: the prior precision, and the noise of a learned `lk.Gaussian()`.

# %%
post = lk.laplace(
    model, [(x_train, y_train)], lk.Categorical(), hyper_tuner=None
)
tuned = lk.tune(post, lk.Minimize(brier(val)))
print(f"prior precision {tuned.prior.prior_precision:.3f}")

# %% [markdown]
# ## The math
#
# For $K$ classes, with $p(x)$ the predictive probabilities and $e_y$ the one-hot vector of the label, the Brier score on $M$ held-out pairs is
#
# $$
#   B(\lambda) = \frac{1}{M} \sum_{j=1}^M \big\lVert p_\lambda(x_j) - e_{y_j} \big\rVert^2 .
# $$
#
# Like the NLL, it is a proper scoring rule.
# Unlike the NLL, it is bounded by 2, so a few confident mistakes weigh less.
# `lk.Minimize` minimizes it over $\log \lambda$ with L-BFGS, so the objective has to be [differentiable](gradients.qmd) in the hyperparameters.
# Here the gradient flows through `lk.predict`: the linearized predictive's covariance depends on $\lambda$ through $(\lambda I + N G)^{-1}$.
#
# ## The test
#
# `lk.testing.check_objective` calls the objective on a copy of the posterior whose selected leaves require grad.
# It checks that the result is a finite 0-dim tensor and that a finite gradient reaches every selected leaf.
# For an independent reference, we use a huge precision.
# It leaves almost no posterior variance, so the score must equal the Brier score of the network's own softmax.


# %%
def test_brier():
    post = lk.laplace(
        model, [(x_train, y_train)], lk.Categorical(), hyper_tuner=None
    )

    lk.testing.check_objective(brier(val), post)

    sharp = copy.replace(post, prior=lk.priors.Isotropic(1e8))

    with torch.no_grad():
        probs = model(x_val).softmax(-1)

    want = (probs - F.one_hot(y_val, 4)).pow(2).sum(-1).mean()

    torch.testing.assert_close(brier(val)(sharp), want, rtol=1e-4, atol=1e-5)


test_brier()
