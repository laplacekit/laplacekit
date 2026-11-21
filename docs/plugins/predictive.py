# %% [markdown]
# ---
# title: Predictives
# jupyter: python3
# ---
#
# ::: {.callout-tip}
# ## New to protocols?
# The contract below is a protocol. Subclass it, as in
# `class MyClass(lk.core.Predictive):`, and your editor marks a missing method in red.
# [Protocols in one page](protocols.py) shows how.
# :::
#
# A predictive turns a posterior and inputs into a distribution over targets. It
# says what it needs from the model in `requires`, which `lk.predict` checks
# against [the probe](requirements.qmd) before it runs:
#
# ```python
# class Predictive[B](Protocol):
#     @property
#     def requires(self) -> Requirements: ...
#
#     def __call__(
#         self, posterior: B, inputs: Inputs, /, *, generator: torch.Generator | None = None
#     ) -> Distribution: ...
# ```
#
# `requires` is a read-only property: `lk.predict` reads `predictive.requires`
# without a call, and a property or a field of a frozen dataclass both satisfy
# it. `B` is the kind of posterior it accepts. `lk.predict` passes a fresh
# `generator`, which a predictive that samples draws from. We write the Laplace bridge[^bridge] for
# classifiers. It matches a Dirichlet to the Gaussian over the logits, in closed
# form and without sampling, and predicts with the Dirichlet's mean.

# %%
import copy

import torch
from sklearn.datasets import make_blobs
from torch import Tensor, nn
from torch.distributions import Categorical, Distribution

import laplacekit as lk
from laplacekit.core import CapabilityError, Inputs, Posterior, Requirements


def bridge(mean: Tensor, var: Tensor) -> Tensor:
    """Dirichlet concentrations from Gaussian logits with means `mean` and variances `var`."""

    k = mean.shape[-1]
    spread = torch.exp(-mean).sum(-1, keepdim=True)

    return (1 - 2 / k + mean.exp() * spread / k**2) / var


@lk.core.pytree
class LaplaceBridge:
    """The mean of the Laplace-bridge Dirichlet, for a `Categorical` likelihood."""

    @property
    def requires(self) -> Requirements:
        # What the exact output covariance needs: a JVP or a VJP, vmapped.
        exact = (Requirements(jvp=True, vmap=True), Requirements(vjp=True, vmap=True))

        return Requirements(one_of=exact)

    def __call__(
        self, posterior: Posterior, inputs: Inputs, /, *, generator: torch.Generator | None = None
    ) -> Distribution:
        if not isinstance(posterior.likelihood, lk.Categorical):
            raise CapabilityError(
                f"LaplaceBridge needs a Categorical likelihood, got {type(posterior.likelihood).__name__}"
            )

        out = lk.predict_outputs(posterior, inputs, n_samples="exact")
        alpha = bridge(out.mean, out.cov.diagonal())

        return Categorical(probs=alpha / alpha.sum(-1, keepdim=True))


# %% [markdown]
# `lk.predict_outputs` does the linearization and gives the Gaussian over the logits,
# so the predictive only maps its means and variances to probabilities. It is
# exact, so it ignores the generator. It goes
# into `lk.predict` as the method. We compare it with the default, the probit
# approximation, and with a Monte Carlo estimate from 2,000 logit n_samples, on the
# training data and on points far from it. A seeded generator fixes the Monte
# Carlo draws, so we can compare with them later.

# %%
x_np, y_np = make_blobs(n_samples=400, centers=4, cluster_std=1.2, random_state=0)
x, y = torch.tensor(x_np, dtype=torch.float32), torch.tensor(y_np)

torch.manual_seed(0)
model = nn.Sequential(nn.Linear(2, 32), nn.Tanh(), nn.Linear(32, 32), nn.Tanh(), nn.Linear(32, 4))
opt = torch.optim.Adam(model.parameters(), lr=1e-2)

for _ in range(300):
    opt.zero_grad()
    nn.functional.cross_entropy(model(x), y).backward()
    opt.step()

post = lk.laplace(
    model, [(x, y)], likelihood=lk.Categorical(), generator=torch.Generator().manual_seed(0)
)
far = 5 * torch.randn(400, 2) + 25

methods = {
    "probit (default)": None,
    "Laplace bridge": LaplaceBridge(),
    "Monte Carlo": lk.Linearized(link=lk.MC(n_samples=2_000)),
}

for name, method in methods.items():
    conf = [
        lk.predict(post, z, method, generator=torch.Generator().manual_seed(0))
        .probs.max(-1)
        .values.mean()
        for z in (x, far)
    ]
    print(f"{name:>16}: mean confidence {conf[0]:.3f} on the data, {conf[1]:.3f} far away")

# %% [markdown]
# On the data, the three agree to about a percent. Far away, the bridge is the
# most confident and the Monte Carlo estimate the least, with probit in between.
# On this problem, the bridge is the coarsest of the three approximations. What
# it gives in exchange is the Dirichlet itself, a closed-form distribution over
# the class probabilities, which this predictive reduces to its mean.
#
# ## The math
#
# With $K$ classes and logits $f \sim \mathcal{N}(\mu, \Sigma)$, the Laplace
# bridge gives the Dirichlet with concentrations
#
# $$
#   \alpha_k = \frac{1}{\Sigma_{kk}} \Big( 1 - \frac{2}{K} + \frac{e^{\mu_k}}{K^2} \sum_{l=1}^K e^{-\mu_l} \Big) .
# $$
#
# It is the inverse of the Laplace approximation of a Dirichlet in the softmax
# basis, which maps $\alpha$ to
#
# $$
#   \mu_k = \log \alpha_k - \frac{1}{K} \sum_{l=1}^K \log \alpha_l , \qquad \Sigma_{kk} = \frac{1}{\alpha_k} \Big( 1 - \frac{2}{K} \Big) + \frac{1}{K^2} \sum_{l=1}^K \frac{1}{\alpha_l} .
# $$
#
# The predictive probabilities are the Dirichlet's mean, $\alpha_k / \sum_l
# \alpha_l$.
#
# ## The test
#
# `lk.testing.check_predictive` checks that `requires` is a `Requirements` and
# that the result is a `Distribution` with one batch entry per input. As an
# independent reference for the numbers, the bridge must undo the second map
# above, written out separately.


# %%
def test_laplace_bridge():
    lk.testing.check_predictive(LaplaceBridge(), post, x[:7])

    alpha = torch.rand(5, 4, dtype=torch.float64) * 10 + 0.1
    k = alpha.shape[-1]
    mean = alpha.log() - alpha.log().mean(-1, keepdim=True)
    var = (1 - 2 / k) / alpha + (1 / alpha).sum(-1, keepdim=True) / k**2

    torch.testing.assert_close(bridge(mean, var), alpha)

    probs = lk.predict(post, x[:7], LaplaceBridge()).probs
    torch.testing.assert_close(probs.sum(-1), torch.ones(7))


test_laplace_bridge()

# %% [markdown]
# ## A likelihood of your own
#
# The built-in predictives know the built-in likelihoods. For any other, they
# ask the likelihood itself, through two hooks:
#
# ```python
# class HasSampleAverage(Protocol):  # WeightSamples and the MC link
#     def average(self, f: Tensor, /) -> Distribution: ...
#
#
# class HasDefaultLink(Protocol):  # Linearized without link=
#     def default_link(self, n_samples: int | Literal["exact"], /) -> LinkApprox: ...
# ```
#
# To see them at work, we write the categorical likelihood ourselves and swap
# it into the posterior from above with `copy.replace`. That keeps the
# curvature fitted under the built-in `Categorical`, which is right here only
# because `Softmax` is the same likelihood, and it lets us check the numbers
# against the built-in.


# %%
@lk.core.pytree
class Softmax:
    """A categorical likelihood outside the built-ins."""

    def log_prob(self, f, y):
        return lk.core.default_log_prob(self, f, y)

    def distribution(self, f):
        return Categorical(logits=f)


try:
    lk.predict(copy.replace(post, likelihood=Softmax()), far)
except CapabilityError as e:
    print(e)

# %% [markdown]
# `average` is the smaller contract. It gets draws of the outputs, shaped `[S,
# N, K]` here, and returns the predictive. For a categorical, that is the mean
# of the softmax over the draws. With it, `WeightSamples` and the `MC` link work.


# %%
@lk.core.pytree
class SoftmaxAverage(Softmax):
    def average(self, f):
        return Categorical(probs=f.softmax(-1).mean(0))


mc = lk.Linearized(link=lk.MC(n_samples=2_000))
seed = torch.Generator().manual_seed(0)
conf = lk.predict(copy.replace(post, likelihood=SoftmaxAverage()), far, mc, generator=seed)
conf = conf.probs.max(-1).values
print(f"MC link: mean confidence {conf.mean():.3f} far away")

# %% [markdown]
# That is the Monte Carlo row from the start of the page. `default_link` then
# names the link that `Linearized` uses when it is given none. `n_samples` is
# `None` when the output covariance is exact, and the number of draws when it is
# sampled. In that case an `MC` link must not ask for more draws than there are.


# %%
@lk.core.pytree
class SoftmaxLinked(SoftmaxAverage):
    def default_link(self, n_samples):
        return lk.MC(n_samples=2_000 if n_samples == "exact" else n_samples)


seed = torch.Generator().manual_seed(0)
conf = lk.predict(copy.replace(post, likelihood=SoftmaxLinked()), far, generator=seed)
conf = conf.probs.max(-1).values
print(f"default link: mean confidence {conf.mean():.3f} far away")

# %% [markdown]
# Plain `lk.predict` now gives the same number, with the same seed.
#
# [^bridge]: Hobbhahn, M., Kristiadi, A., & Hennig, P. (2022). Fast predictive uncertainty for classification with Bayesian deep networks. *UAI*. <https://arxiv.org/abs/2003.01227>
