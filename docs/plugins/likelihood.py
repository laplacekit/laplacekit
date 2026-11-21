# %% [markdown]
# ---
# title: Likelihoods
# jupyter: python3
# ---
#
# ::: {.callout-tip}
# ## New to protocols?
# The contract below is a protocol. Subclass it, as in
# `class MyClass(lk.core.Likelihood):`, and your editor marks a missing method in red.
# [Protocols in one page](protocols.py) shows how.
# :::
#
# A likelihood needs one method, `log_prob(f, y)`, which returns $\log p(y \mid
# f)$ per example, shaped `[N]`. Each optional hook unlocks more methods:
#
# ```python
# class Likelihood(Protocol):
#     def log_prob(self, f: Outputs, y: Targets, /) -> Tensor: ...
#
#
# class HasDistribution(Protocol):  # the GGN-type curvatures, the predictives
#     def distribution(self, f: Outputs, /) -> Distribution: ...
#
#
# class HasOutputHessian(Protocol):  # the exact GGN, without sampling
#     def output_hessian(self, f: Outputs, /) -> OutputHessian: ...
#
#
# class HasSample(Protocol):  # sampling without a distribution
#     def sample(self, f: Outputs, /, *, generator: torch.Generator | None = None) -> Targets: ...
#
#
# class HasMask(Protocol):  # output positions that don't count, such as padding
#     def mask(self, f: Outputs, inputs: Inputs, /) -> Tensor | None: ...
#
#
# class HasTargetCheck(Protocol):  # refuse bad targets before the fit uses them
#     def check_targets(self, y: Targets, /) -> None: ...
#
#
# class HasOutputCheck(Protocol):  # refuse outputs of the wrong shape, before any data
#     def check_outputs(self, f: Outputs, /) -> None: ...
#
#
# class HasDefaultLink(Protocol):  # lk.predict without link=
#     def default_link(self, n_samples: int | Literal["exact"], /) -> LinkApprox: ...
#
#
# class HasSampleAverage(Protocol):  # the predictives that average output draws
#     def average(self, f: Tensor, /) -> Distribution: ...
# ```
#
# Our example is the Bradley–Terry model for pairwise preferences, as in reward
# models. **laplacekit** ships it in `lk.reward`, which the [Reward
# models](../quickstart/reward.py) page uses. Here we build it from scratch: we
# start from nothing but its log-likelihood, see what the core can and can't do
# with it, and then add the hooks that unlock the rest.

# %%
import math

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Bernoulli

import laplacekit as lk

# %% [markdown]
# ## Data and model
#
# A network $g_\theta$ gives each item $x$ a score. An example is a pair of
# items $(a, b)$ with a label $y = 1$ if $a$ is preferred over $b$ and $y = 0$
# otherwise. The items are points $x \in [-2, 2]$ with a hidden utility $u(x) =
# \sin(2x)$. We draw 400 random pairs and label each with $y \sim
# \mathrm{Bernoulli}(\sigma(3 (u(a) - u(b))))$. The inputs are shaped `[N, 2,
# 1]`: two items of one feature each. The scorer is a small ReLU MLP applied to
# each item, so the model outputs `[N, 2]`, the two scores.

# %%
torch.set_default_dtype(torch.float64)
torch.manual_seed(3)


def utility(x):
    return torch.sin(2 * x)


x = 4 * torch.rand(400, 2, 1) - 2
u = utility(x).squeeze(-1)
y = torch.bernoulli(torch.sigmoid(3 * (u[:, 0] - u[:, 1])))
loader = list(zip(x.split(100), y.split(100), strict=True))

model = nn.Sequential(
    nn.Linear(1, 32), nn.ReLU(), nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 1)
)
pairs = nn.Sequential(model, nn.Flatten())  # [N, 2, 1] -> [N, 2]
opt = torch.optim.Adam(pairs.parameters(), lr=1e-2, weight_decay=1e-4)

for _ in range(500):
    f = pairs(x)
    loss = F.binary_cross_entropy_with_logits(f[:, 0] - f[:, 1], y)
    opt.zero_grad()
    loss.backward()
    opt.step()

print("final loss:", round(loss.item(), 3))

# %% [markdown]
# ## Only the log-likelihood
#
# `lk.likelihoods.from_log_prob` turns a plain function into a likelihood. Ours
# is per pair, so it fits the contract: each example's log-likelihood depends on
# its own outputs only. The network is small, so we put the Laplace
# approximation on all its 1,153 parameters (`subset=lk.subsets.all`).


# %%
def bradley_terry(f, y):
    return -F.binary_cross_entropy_with_logits(
        f[:, 0] - f[:, 1], y, reduction="none"
    )


lik = lk.likelihoods.from_log_prob(bradley_terry)
post = lk.laplace(
    pairs,
    loader,
    likelihood=lik,
    subset=lk.subsets.all,
    generator=torch.Generator().manual_seed(0),
)
print(post)

# %% [markdown]
# Writing $f = (g_\theta(a), g_\theta(b))$ for the two scores, this is the
# Bradley–Terry likelihood,
#
# $$
#   p(y = 1 \mid f) = \sigma(f_1 - f_2) , \qquad \log p(y \mid f) = y \log \sigma(f_1 - f_2) + (1 - y) \log \sigma(f_2 - f_1) ,
# $$
#
# where $\sigma$ is the logistic sigmoid. Note that adding a constant to every
# score leaves the likelihood unchanged.
#
# `lk.laplace` picked `DiagGGN(fisher="empirical")` for every layer. The reason is
# that the GGN, $G = \frac{1}{N} \sum_i J_i^\top H_i J_i$, needs the output
# Hessian $H_i$, either in closed form or as a Monte Carlo estimate from samples
# $y \sim p(y \mid f_i)$. A bare `log_prob` gives neither a closed form nor
# samples. It does give the observed Hessian, by autograd at the observed
# labels, which we come back to at the end. By default, though, the core uses
# the gradient at the observed label, that is, the empirical Fisher,
#
# $$
#   F_\text{emp} = \frac{1}{N} \sum_{i=1}^N \mathrm{diag}(g_i g_i^\top) , \qquad g_i = \nabla_\theta \log p(y_i \mid f_\theta(x_i)) .
# $$
#
# :::{.callout-note}
# The empirical Fisher equals the Fisher only when the labels follow the model,
# and it can be a poor curvature estimate otherwise (Kunstner et al., 2019)[^kunstner2019]. It
# also needs the targets, so it can't fit on unlabelled data.
# :::
#
# The prior precision was tuned by the marginal likelihood. It needs $\log p(y
# \mid f)$ at the fitted parameters, which we have.
#
# What doesn't work is a predictive over $y$. The predictives need a
# distribution over targets, and we haven't given one:

# %%
try:
    lk.predict(post, x[:5])
except lk.CapabilityError as e:
    print(e)

# %% [markdown]
# The error points to `lk.predict_outputs`, which gives the Gaussian over the scores $f$
# under the linearized model, $\mathcal{N}(f \mid m, V)$ with $m =
# f_{\hat\theta}(x)$ and $V = J \Sigma J^\top$. It needs only the model and the
# posterior. From there we can do the last step by hand. The score difference
# $d = f_1 - f_2$ is Gaussian with mean $m_1 - m_2$ and variance $V_{11} + V_{22}
# - 2 V_{12}$, and the probit approximation gives
#
# $$
#   p(y = 1 \mid a, b, \mathcal{D}) \approx \sigma\left( \frac{m_1 - m_2}{\sqrt{1 + \frac{\pi}{8} (V_{11} + V_{22} - 2 V_{12})}} \right) .
# $$
#
# Note that the shift direction drops out of $d$, so its large variance doesn't
# matter. The variance is large because the likelihood doesn't change when we
# add the same constant to both scores. The data then say nothing about that
# direction, so its posterior variance is close to the prior's.


# %%
def win_prob(out):
    m = out.mean
    v = out.cov.per_input()  # [N, 2, 2]
    var = v[:, 0, 0] + v[:, 1, 1] - 2 * v[:, 0, 1]

    return torch.sigmoid(
        (m[:, 0] - m[:, 1]) / torch.sqrt(1 + math.pi / 8 * var)
    )


# %% [markdown]
# We compare each item $x \in [-6, 6]$ with the reference item $b = 0$, and plot
# the probability that $x$ is preferred. Outside $[-2, 2]$, there is no data, so
# the true curve is drawn only inside.

# %%
t = torch.linspace(-6, 6, 300)
grid = torch.stack([t, torch.zeros_like(t)], dim=1).unsqueeze(
    -1
)  # pairs (x, 0)

with torch.no_grad():
    f_map = pairs(grid)

p_map = torch.sigmoid(f_map[:, 0] - f_map[:, 1])
p_lap = win_prob(lk.predict_outputs(post, grid))
inside = t.abs() <= 2
p_true = torch.sigmoid(3 * (utility(t[inside]) - utility(torch.zeros(()))))

fig, ax = plt.subplots(figsize=(7, 4))
ax.axvspan(-2, 2, color="0.93", label="data range")
ax.plot(t[inside], p_true, color="0.5", linestyle=":", label="true")
ax.plot(t, p_map, label="MAP")
ax.plot(t, p_lap, label="Laplace (empirical Fisher)")
ax.set_xlabel("item x")
ax.set_ylabel("P(x preferred over 0)")
ax.legend()
plt.show()

# %% [markdown]
# Inside the data range, the two curves mostly agree. Outside, the MAP model
# extrapolates its scores and goes to $0$ or $1$, while the Laplace predictive is
# less sure. On the left, for instance, it levels off near $0.9$.
#
# ## Adding a distribution
#
# The Bradley–Terry label is a Bernoulli variable with logit $f_1 - f_2$. We can
# say so with a `distribution(f)` method (the `HasDistribution` hook). Then the
# core can sample labels, so the GGN with a Monte Carlo output
# Hessian, $H_i \approx \frac{1}{S} \sum_s \nabla_f \log p(y_s \mid f_i) \nabla_f
# \log p(y_s \mid f_i)^\top$ with $y_s \sim p(y \mid f_i)$, becomes available.
# And the predictives get a distribution to put on top of $f$.
#
# The likelihood is now a small class instead of a function. `log_prob` can be
# derived from the distribution with `lk.core.default_log_prob`.


# %%
@lk.core.pytree
class BradleyTerry:
    def log_prob(self, f, y):
        return lk.core.default_log_prob(self, f, y)

    def distribution(self, f):
        return Bernoulli(logits=f[:, 0] - f[:, 1])


post_mc = lk.laplace(
    pairs,
    loader,
    likelihood=BradleyTerry(),
    subset=lk.subsets.all,
    generator=torch.Generator().manual_seed(0),
)
print(post_mc)

# %% [markdown]
# The default is now the dense GGN of every layer. The built-in links know
# only the built-in likelihoods, so for the predictive we give `Linearized`
# our own link: an object with `supports` and a call that turns the Gaussian
# over $f$ into a distribution over $y$. Its body is `win_prob` from above. The
# call also takes a `generator`, which a link that samples draws from; ours
# draws nothing.


# %%
@lk.core.pytree
class PairProbit:
    def supports(self, lik, f_mu):
        return isinstance(lik, BradleyTerry)

    def __call__(self, lik, out, *, generator=None):
        return Bernoulli(probs=win_prob(out))


p_ggn = lk.predict(post_mc, grid, lk.Linearized(link=PairProbit())).probs

fig, ax = plt.subplots(figsize=(7, 4))
ax.axvspan(-2, 2, color="0.93", label="data range")
ax.plot(t, p_map, label="MAP")
ax.plot(t, p_lap, label="Laplace (empirical Fisher)")
ax.plot(t, p_ggn, label="Laplace (GGN)")
ax.set_xlabel("item x")
ax.set_ylabel("P(x preferred over 0)")
ax.legend()
plt.show()

# %% [markdown]
# Far from the data, the GGN version is less confident still. Part of the
# difference is the prior: the marginal likelihood picks a different $\lambda$
# for each curvature, as the two printouts show. The jumps come from the ReLUs.
# The Jacobian jumps whenever a unit switches on or off, since the ReLU's
# derivative jumps from 0 to 1, so the variance jumps too.
#
# ## Adding a default link
#
# Passing `link=PairProbit()` on every call gets old. A `default_link` method
# (the `HasDefaultLink` hook) names the link that `Linearized` uses when we
# don't give one, so plain `lk.predict(post, x)` works. Its `n_samples` is `None`
# when the output covariance is exact and the number of draws when it is
# sampled. `PairProbit` only reads the per-input covariance, so it serves both.


# %%
@lk.core.pytree
class LinkedBradleyTerry(BradleyTerry):
    def default_link(self, n_samples):
        return PairProbit()


post_linked = lk.laplace(
    pairs,
    loader,
    likelihood=LinkedBradleyTerry(),
    subset=lk.subsets.all,
    generator=torch.Generator().manual_seed(0),
)
p_default = lk.predict(post_linked, grid).probs

print(
    "largest difference from link=PairProbit(): "
    f"{float((p_default - p_ggn).abs().max()):.1e}"
)

# %% [markdown]
# With the same seed, the Monte Carlo output Hessian draws the same labels, so the
# fit is the same as before, and so are the probabilities.
#
# ## Adding the output Hessian
#
# For Bradley–Terry, we know the output Hessian in closed form. With $p =
# \sigma(f_1 - f_2)$ and $u = (1, -1)^\top$, the gradient of $-\log p(y \mid f)$
# is $(p - y) \, u$, so
#
# $$
#   H = p (1 - p) \, u u^\top = p (1 - p) \begin{pmatrix} 1 & -1 \\ -1 & 1 \end{pmatrix} .
# $$
#
# An `output_hessian(f)` method (the `HasOutputHessian` hook) returns it as an
# object with `mv`, `sqrt_mv`, `diagonal` and `dense`, all per example. For
# `sqrt_mv`, we need some $A$ with $A A^\top = H$. Since $u^\top u = 2$, $A =
# \sqrt{p (1 - p) / 2} \; u u^\top$ works.

# %%
U = torch.tensor([1.0, -1.0])


@lk.core.pytree
class PairHessian:
    s: torch.Tensor  # p (1 - p), shaped [N]

    def mv(self, v):
        return self.s[:, None] * U * (v @ U)[:, None]

    def sqrt_mv(self, v):
        return (self.s / 2).sqrt()[:, None] * U * (v @ U)[:, None]

    def diagonal(self):
        return self.s[:, None].expand(-1, 2)

    def dense(self):
        return self.s[:, None, None] * torch.outer(U, U)


@lk.core.pytree
class BradleyTerryExact(BradleyTerry):
    def output_hessian(self, f):
        p = torch.sigmoid(f[:, 0] - f[:, 1])
        return PairHessian(p * (1 - p))


post_exact = lk.laplace(
    pairs,
    loader,
    likelihood=BradleyTerryExact(),
    subset=lk.subsets.all,
    hyper_tuner=None,
    generator=torch.Generator().manual_seed(0),
)

# %% [markdown]
# The Monte Carlo estimate with one sample is $(p - y_s)^2 \, u u^\top$, which is
# either $(1 - p)^2 u u^\top$ or $p^2 u u^\top$ instead of $p (1 - p) \, u u^\top$.
# It is right on average, and the GGN averages it over the 400 pairs. To see how
# much noise is left, we compare the GGN from the Monte Carlo Hessian with $S = 1$
# and $S = 100$ samples (`lk.DenseGGN(fisher="mc", mc_samples=...)`) against the closed form,
# by the relative error in the Frobenius norm over all blocks.


# %%
def ggn(post):
    blocks = post.curvature.blocks.values()
    return [op.mv(torch.eye(op.diagonal().numel())) for op in blocks]


exact = ggn(post_exact)


def rel_err(post):
    diff = sum(
        torch.linalg.norm(a - b) ** 2
        for a, b in zip(ggn(post), exact, strict=True)
    )
    return (diff / sum(torch.linalg.norm(b) ** 2 for b in exact)).sqrt()


for s in [1, 100]:
    approx = lk.DenseGGN(fisher="mc", mc_samples=s)
    mc = lk.laplace(
        pairs,
        loader,
        likelihood=BradleyTerry(),
        subset=lk.subsets.all,
        hessian_approx=approx,
        hyper_tuner=None,
        generator=torch.Generator().manual_seed(0),
    )
    print(f"S = {s:3d}: relative error {rel_err(mc):.3f}")

# %% [markdown]
# The error is about 7% with one sample and shrinks with more, at $S$ times
# the sampling cost. The closed form has no error and needs no samples.
#
# There is one more route, which needs no hook at all. `observed=True`
# makes `Diag` and `DenseGGN` use the Hessian of $-\log p(y \mid f)$ at the
# observed labels, computed by autograd from `log_prob`. In general, it differs
# from the Fisher and can be indefinite. For the Bradley–Terry model, it is $p (1 -
# p) \, u u^\top$ whatever $y$ is, so even our bare `from_log_prob` likelihood
# gets the exact GGN this way:

# %%
observed = lk.laplace(
    pairs,
    loader,
    likelihood=lik,
    subset=lk.subsets.all,
    hessian_approx=lk.DenseGGN(observed=True),
    hyper_tuner=None,
    generator=torch.Generator().manual_seed(0),
)
print(f"observed: relative error {rel_err(observed):.1e}")

# %% [markdown]
# ## Which hooks to add
#
# | The likelihood has | Curvatures | Predictive over $y$ |
# | --- | --- | --- |
# | `log_prob` only (`from_log_prob`) | `DiagGGN(fisher="empirical")`, and `DiagGGN` or `DenseGGN` with `observed=True` (all need targets) | none; `lk.predict_outputs` for $f$ |
# | + `distribution` (`HasDistribution`) | also `DenseGGN(fisher="mc")` and `DiagGGN(fisher="mc")`, with the Monte Carlo output Hessian (the default picks them) | `Linearized` with a link that knows it |
# | + `default_link` (`HasDefaultLink`) | the same | `lk.predict(post, x)` with no `link=` |
# | + `output_hessian` (`HasOutputHessian`) | the same, and `fisher="exact"` (the default) with the exact output Hessian | the same |
# | is a built-in `Categorical` (`class_dim=-1`, no `mask_fn`), `Gaussian` or `Bernoulli` | also `KFAC`, which refuses every other likelihood | the same |
#
# Start with `from_log_prob` when the log-likelihood is all there is, for example
# a contrastive or energy-based objective. Add `distribution` as soon as the
# targets have a known distribution. It is usually one line, and it makes the
# GGN-type curvatures and the predictives available. Add `output_hessian` when the Hessian has a closed
# form and the outputs are small or structured, so the GGN is exact and cheap.
# The likelihood's `sample` (`HasSample`) and `mask` (`HasMask`) hooks are
# optional too: the first replaces sampling from the distribution, the second
# drops output positions such as padding tokens. A bare log-likelihood gets the
# mask from `from_log_prob(fn, mask_fn=...)`, where `mask_fn(f, inputs)` returns
# False at the positions that don't count.
#
# Two more hooks catch mistakes early. `check_targets` (`HasTargetCheck`) runs
# on each batch's targets before the fit uses them and raises on bad values,
# such as a label outside $\{0, 1\}$ for ours. `check_outputs`
# (`HasOutputCheck`) runs once on the probe's forward pass, before any data is
# fitted, and raises `StructureError` on outputs the likelihood can't take, such
# as scores that aren't shaped `[N, 2]`. The predictives that average output
# draws, `WeightSamples` and the `MC` link, need `average` (`HasSampleAverage`),
# which the [Predictives](predictive.py) guide adds to a likelihood of its own.
#
# :::{.callout-note}
# The contract is per example. A loss that couples the examples of a batch, such
# as a listwise ranking loss or InfoNCE with in-batch negatives, doesn't fit it,
# since the curvature statistics are sums over examples.
# :::

# %% [markdown]
# ## The test
#
# `lk.testing.check_likelihood` checks the laws the core relies on, such as a
# `log_prob` shaped `[N]` and a symmetric PSD output Hessian. A wrong Hessian can
# keep those laws, so we also test the hooks against independent references: the
# closed-form output Hessian against `torch.func.hessian` of $-\log p(y \mid
# f)$, which for Bradley–Terry doesn't depend on $y$, and `sqrt_mv` against $A
# A^\top = H$. The link has a checker, `lk.testing.check_link`. It checks that
# `supports` agrees with the call and that the result is a valid distribution.


# %%
def test_bradley_terry():
    f = torch.randn(6, 2)
    lk.testing.check_likelihood(BradleyTerryExact(), f, torch.ones(6))

    got = BradleyTerryExact().output_hessian(f)

    def nll(fi, yi):
        return -bradley_terry(fi[None], yi[None]).sum()

    want = torch.stack(
        [torch.func.hessian(nll)(fi, torch.tensor(1.0)) for fi in f]
    )

    torch.testing.assert_close(got.dense(), want)
    torch.testing.assert_close(
        got.diagonal(), torch.diagonal(want, dim1=1, dim2=2)
    )

    eye = torch.eye(2).expand(6, 2, 2)  # columns of the identity, per example
    a = torch.stack([got.sqrt_mv(eye[:, :, j]) for j in range(2)], dim=-1)

    torch.testing.assert_close(a @ a.transpose(1, 2), want)

    lk.testing.check_link(
        PairProbit(), BradleyTerry(), lk.predict_outputs(post_mc, x[:5])
    )


test_bradley_terry()

# %% [markdown]
# [^kunstner2019]: Kunstner, F., Balles, L., & Hennig, P. (2019). Limitations of the empirical Fisher approximation for natural gradient descent. *NeurIPS*. <https://arxiv.org/abs/1905.12558>
