# %% [markdown]
# ---
# title: Priors
# jupyter: python3
# ---
#
# ::: {.callout-tip}
# ## New to protocols?
# The contract below is a protocol. Subclass it, as in
# `class MyClass(lk.core.Prior):`, and your editor marks a missing method in red.
# [Protocols in one page](protocols.py) shows how.
# :::
#
# A prior is any `Prior`: it has a log density, a location, and it can
# add its precision to a curvature operator. The built-in priors put one
# precision on everything (`Isotropic`), one per entry (`Diagonal`), or one per
# layer (`Layerwise`). We write a prior with two precisions, one for the weights
# and one for the biases, and let `lk.laplace` tune both.
#
# ```python
# class Prior(Protocol):
#     def log_prob(self, params: Params, /) -> Tensor: ...
#
#     def location(self, like: Params, /) -> Params: ...
#
#     def add_precision_to[Op: Operator](self, op: Op, /) -> Op: ...
#
#
# class PriorWithCovariance(Prior, Protocol):
#     def cov_mv(self, v: Params, /) -> Params: ...
#
#
# class TunablePrior(Prior, Protocol):
#     def tunable(self) -> Tensor | Mapping[str, Tensor]: ...
# ```
#
# `cov_mv` lets the predictives push the prior forward, and `tunable` tells the
# hyper tuners which hyperparameters to tune.

# %%
from collections.abc import Mapping

import torch
from sklearn.datasets import make_blobs
from torch import Tensor, nn

import laplacekit as lk
from laplacekit.core import Operator, Params


@lk.core.pytree
class WeightBias:
    """Zero-mean Gaussian prior, one precision for weights, one for biases."""

    prior_precision: Mapping[str, Tensor]  # {"weight": ..., "bias": ...}
    # 1.0 at each bias entry, 0.0 elsewhere, flat over the parameters
    is_bias: Tensor

    def __repr__(self) -> str:
        lam = self.prior_precision
        return f"WeightBias(weight={lam['weight']:.3g}, bias={lam['bias']:.3g})"

    def _diagonal(self) -> lk.priors.Diagonal:
        w, b = self.prior_precision["weight"], self.prior_precision["bias"]

        return lk.priors.Diagonal(w * (1 - self.is_bias) + b * self.is_bias)

    def log_prob(self, params: Params, /) -> Tensor:
        return self._diagonal().log_prob(params)

    def location(self, like: Params, /) -> Params:
        return self._diagonal().location(like)

    def add_precision_to[Op: Operator](self, op: Op, /) -> Op:
        return self._diagonal().add_precision_to(op)

    def cov_mv(self, v: Params, /) -> Params:
        return self._diagonal().cov_mv(v)

    def tunable(self) -> Mapping[str, Tensor]:
        return self.prior_precision


def weight_bias(
    params: Params, weight: float = 1.0, bias: float = 1.0
) -> WeightBias:
    """A `WeightBias` prior over `params`; the 1-d tensors are the biases."""

    parts = [
        torch.full((p.numel(),), float(p.ndim == 1), dtype=p.dtype)
        for p in params.values()
    ]
    prior_precision = {
        "weight": torch.tensor(weight),
        "bias": torch.tensor(bias),
    }

    return WeightBias(prior_precision, torch.cat(parts))


# %% [markdown]
# The prior is a [`@lk.core.pytree`](pytrees.qmd), so a tuner can find its tensors, and every
# method hands the work to a built-in `Diagonal` with the right entries. We fit
# an MLP on four blobs and put the posterior on all its parameters, once with
# our prior and once with the default `Isotropic`.

# %%
x_np, y_np = make_blobs(
    n_samples=400, centers=4, cluster_std=1.2, random_state=0
)
x, y = torch.tensor(x_np, dtype=torch.float32), torch.tensor(y_np)

torch.manual_seed(0)
model = nn.Sequential(
    nn.Linear(2, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 4)
)
opt = torch.optim.Adam(model.parameters(), lr=1e-2)

for _ in range(300):
    opt.zero_grad()
    nn.functional.cross_entropy(model(x), y).backward()
    opt.step()

params = dict(model.named_parameters())

for prior in (weight_bias(params), lk.priors.Isotropic(1.0)):
    post = lk.laplace(
        model,
        [(x, y)],
        likelihood=lk.Categorical(),
        subset=lk.subsets.all,
        hessian_approx=lk.DiagGGN(),
        prior=prior,
    )
    print(post)
    print(f"log marginal likelihood: {post.log_marglik():.1f}\n")

# %% [markdown]
# For both, `lk.MarginalLikelihood()` maximized the marginal likelihood over the
# prior's precisions, both of them for our prior, because it is a
# `TunablePrior`. A prior without `tunable` has nothing to tune, so the tuner
# leaves it as it is.
#
# The weights end up with about the precision `Isotropic` picks, and the biases
# get a looser prior. The marginal likelihood goes up by a few nats. Since
# `Isotropic` is the special case of equal precisions, the best `WeightBias`
# can't do worse, and here the second precision buys only a little.
#
# We used `lk.DiagGGN()` on purpose. A vector precision has to be added to the
# curvature entry by entry, and a Kronecker-factored curvature (KFAC or EKFAC)
# has no room for that: it refuses any vector precision, even a constant one,
# with `StructureError`. It only takes one scalar per block, as `Isotropic` or a
# `Layerwise` with a scalar per block gives it.
#
# ## The math
#
# With $b_i = 1$ when entry $i$ of $\theta$ is a bias, the prior is
#
# $$
#   p(\theta) = \mathcal{N}\big(0, \Lambda_0^{-1}\big) , \qquad \Lambda_0 = \mathrm{diag}\big(\lambda_w (1 - b) + \lambda_b \, b\big) ,
# $$
#
# and the Laplace precision is $\Lambda_0 + N G$, with $G$ the curvature. The
# default hyper tuner maximizes the Laplace marginal likelihood over $\log \lambda_w$
# and $\log \lambda_b$ without refitting $G$. That is why `tunable` must return
# the prior's own positive leaves: the tune replaces them in the posterior by
# identity and optimizes their logs. `lk.laplace` checks this before it reads
# any data, so a `tunable` that returns a copy raises `StructureError` right
# away instead of after the fit.
#
# ## The test
#
# `lk.testing.check_prior` checks the contract in float64: a finite `log_prob`,
# `add_precision_to` keeping each operator's type, `cov_mv` being the inverse
# of the precision, and, since our prior is a `TunablePrior`, `tunable`
# returning its own positive leaves. We compare the density and the precision
# with a reference built from `torch.distributions`.


# %%
def test_weight_bias():
    p64 = {k: v.detach().double() for k, v in params.items()}
    prior = weight_bias(p64, weight=2.0, bias=0.5)

    lk.testing.check_prior(prior, p64)

    assert isinstance(prior, lk.core.TunablePrior)

    std = {k: (0.5 if p.ndim == 1 else 2.0) ** -0.5 for k, p in p64.items()}
    want = sum(
        torch.distributions.Normal(0.0, std[k]).log_prob(p).sum()
        for k, p in p64.items()
    )

    torch.testing.assert_close(prior.log_prob(p64), want)

    flat = torch.cat([p.reshape(-1) for p in p64.values()])
    zero = lk.core.Diag(torch.zeros_like(flat))
    is_bias = torch.cat(
        [torch.full((p.numel(),), p.ndim == 1) for p in p64.values()]
    )

    torch.testing.assert_close(
        prior.add_precision_to(zero).diagonal(),
        torch.where(is_bias, 0.5, 2.0).double(),
    )


test_weight_bias()
