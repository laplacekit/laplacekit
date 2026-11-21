# %% [markdown]
# ---
# title: Continual learning
# jupyter: python3
# ---
#
# When the data arrive in parts, the posterior after one part is the prior for
# the next. In **laplacekit**, that is `prior=post`. We split the sine data of
# the [regression page](regression.py) at $x = 4$ and update only the last
# layer.

# %%
import matplotlib.pyplot as plt
import torch
from torch import nn

import laplacekit as lk

torch.set_default_dtype(torch.float64)
torch.manual_seed(711)

x = torch.rand(150, 1) * 8.0
y = torch.sin(x) + 0.3 * torch.randn(150, 1)
x_test = torch.linspace(-2.0, 10.0, 200).reshape(-1, 1)

model = nn.Sequential(nn.Linear(1, 50), nn.Tanh(), nn.Linear(50, 1))
opt = torch.optim.Adam(model.parameters(), lr=1e-2)

for _ in range(1000):
    opt.zero_grad()
    nn.functional.mse_loss(model(x), y).backward()
    opt.step()

# %% [markdown]
# Training on a part minimizes the negative log-likelihood minus the log
# prior, where the prior is either the first one or the previous posterior.
# Both have a `log_prob` of the parameters.

# %%
last = {"2.weight": model[2].weight, "2.bias": model[2].bias}


def train(inputs, targets, prior):
    opt = torch.optim.LBFGS(
        list(last.values()), max_iter=500, line_search_fn="strong_wolfe"
    )

    def closure():
        opt.zero_grad()
        loss = ((targets - model(inputs)) ** 2).sum() / (
            2 * 0.3**2
        ) - prior.log_prob(last)
        loss.backward()
        return loss

    opt.step(closure)


lik = lk.Gaussian(sigma=0.3)
prior = lk.priors.Isotropic(1.0)
left = x.squeeze() < 4
d1, d2 = [(x[left], y[left])], [(x[~left], y[~left])]

train(*d1[0], prior)
post_1 = lk.laplace(model, d1, likelihood=lik, prior=prior, hyper_tuner=None)

train(*d2[0], post_1)
post_2 = lk.laplace(model, d2, likelihood=lik, prior=post_1, hyper_tuner=None)

# %%
t = x_test.squeeze()
fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
panels = [
    (post_1, left, "after part 1"),
    (post_2, ~left, "after part 1, then part 2"),
]

for ax, (posterior, part, name) in zip(axes, panels, strict=True):
    pred = lk.predict(posterior, x_test)
    mean, std = pred.mean.squeeze(), pred.stddev.squeeze()
    ax.scatter(
        x[part].squeeze(), y[part].squeeze(), s=8, color="0.5", label="new data"
    )
    ax.plot(t, mean)
    ax.fill_between(t, mean - 2 * std, mean + 2 * std, alpha=0.3)
    ax.set_title(name)

axes[0].legend()
plt.show()

# %% [markdown]
# After the first part, the band is wide over $[4, 8]$, where there are no data
# yet. The second fit sees only the right half, and the band narrows there. Over
# the first part's data, $[0, 4]$, the fit barely moves, because the first
# posterior is now the prior. Left of all data, on $[-2, 0]$, nothing holds the
# mean in place, and it moves.
#
# ## With KFAC
#
# For a KFAC posterior, `prior=post` raises a `StructureError`, because the sum of two
# Kronecker products has no Kronecker form. `lk.priors.KronSum.of(post)` sums the
# factors instead, as `laplace-torch` does. The result only approximates the
# sum, which is why it is opt-in. We run the same two steps with KFAC.

# %%
std_dense = lk.predict(post_2, x_test).stddev

train(*d1[0], prior)
kfac_1 = lk.laplace(
    model,
    d1,
    likelihood=lik,
    prior=prior,
    hessian_approx=lk.KFAC(),
    hyper_tuner=None,
)

train(*d2[0], lk.priors.KronSum.of(kfac_1))
kfac_2 = lk.laplace(
    model,
    d2,
    likelihood=lik,
    prior=lk.priors.KronSum.of(kfac_1),
    hessian_approx=lk.KFAC(),
    hyper_tuner=None,
)

gap = (lk.predict(kfac_2, x_test).stddev - std_dense).abs().max()
print(f"largest gap in the standard deviation: {gap:.1e}")

# %% [markdown]
# The gap is rounding error, so here the sum is exact. That holds whenever the
# output-side factor of KFAC is the same in both fits, as for a Gaussian
# likelihood on the last layer, where it doesn't depend on the data. For a
# classifier, that factor depends on the predictions, so it changes between the
# fits and the sum is approximate.
#
# ::: {.callout-note collapse="true" title="The math"}
# Bayes' rule processes the parts one after the other,
#
# $$
#   p(\theta \mid \mathcal{D}_1, \mathcal{D}_2) \propto p(\mathcal{D}_2 \mid \theta) \, p(\theta \mid \mathcal{D}_1) .
# $$
#
# Laplace for continual learning replaces $p(\theta \mid \mathcal{D}_1)$ by its Laplace
# approximation $\mathcal{N}(\theta_1, \Lambda_1^{-1})$. Training on
# $\mathcal{D}_2$ with that Gaussian as the prior gives $\theta_2$, and the new
# precision is $\Lambda_2 = \Lambda_1 + N_2 G_2$. For the last layer of a
# network with Gaussian noise, the log posterior is quadratic, so every step is
# exact and the result equals a single fit on all the data.
# :::
#
# The full tutorial: [Continual learning](../examples/continual.py)
