# %% [markdown]
# ---
# title: Custom likelihood
# jupyter: python3
# ---
#
# ::: {.callout-tip}
# ## New to protocols?
# The contract below is a protocol. Subclass it, as in
# `class MyClass(lk.core.Likelihood):`, and your editor marks a missing method in red.
# [Protocols in one page](../plugins/protocols.py) shows how.
# :::
#
# A likelihood that **laplacekit** doesn't ship needs one function, `log_prob(f,
# y)`, returning $\log p(y \mid f)$ per example. Our example is the
# Bradley–Terry model for pairwise preferences, as in reward models: a network
# scores two items, and the label says whether the first one wins. For real
# reward models, `lk.reward` ships it ready to use: see [Reward
# models](reward.py).

# %%
import math

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch import nn

import laplacekit as lk

torch.set_default_dtype(torch.float64)
torch.manual_seed(3)

x = 4 * torch.rand(400, 2, 1) - 2  # 400 pairs of items in [-2, 2]
u = torch.sin(2 * x).squeeze(-1)  # hidden utility of each item
y = torch.bernoulli(torch.sigmoid(3 * (u[:, 0] - u[:, 1])))
loader = list(zip(x.split(100), y.split(100), strict=True))

scorer = nn.Sequential(nn.Linear(1, 32), nn.ReLU(), nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 1))
model = nn.Sequential(scorer, nn.Flatten())  # [N, 2, 1] -> [N, 2] scores
opt = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-4)

for _ in range(500):
    f = model(x)
    loss = F.binary_cross_entropy_with_logits(f[:, 0] - f[:, 1], y)
    opt.zero_grad()
    loss.backward()
    opt.step()

# %% [markdown]
# `lk.likelihoods.from_log_prob` turns the function into a likelihood.


# %%
def bradley_terry(f, y):
    return -F.binary_cross_entropy_with_logits(f[:, 0] - f[:, 1], y, reduction="none")


lik = lk.likelihoods.from_log_prob(bradley_terry)
post = lk.laplace(model, loader, likelihood=lik, subset=lk.subsets.all)
print(post)

# %% [markdown]
# A bare `log_prob` gives no distribution over $y$, so `lk.predict` refuses.
# `lk.predict_outputs` still gives the Gaussian over the two scores, and we finish the
# prediction by hand.

# %%
t = torch.linspace(-6, 6, 300)
grid = torch.stack([t, torch.zeros_like(t)], dim=1).unsqueeze(-1)  # pairs (x, 0)

try:
    lk.predict(post, grid)
except lk.CapabilityError as e:
    print(e)

out = lk.predict_outputs(post, grid)
v = out.cov.per_input()  # [N, 2, 2]
var = v[:, 0, 0] + v[:, 1, 1] - 2 * v[:, 0, 1]
p_lap = torch.sigmoid((out.mean[:, 0] - out.mean[:, 1]) / torch.sqrt(1 + math.pi / 8 * var))

with torch.no_grad():
    f_map = model(grid)

fig, ax = plt.subplots(figsize=(7, 4))
ax.axvspan(-2, 2, color="0.93", label="data range")
ax.plot(t, torch.sigmoid(f_map[:, 0] - f_map[:, 1]), label="MAP")
ax.plot(t, p_lap, label="Laplace")
ax.set_xlabel("item x")
ax.set_ylabel("P(x preferred over 0)")
ax.legend()
plt.show()

# %% [markdown]
# Outside the data range, the MAP model goes to $0$ or $1$, and the Laplace
# predictive stays less sure.
#
# ::: {.callout-note collapse="true" title="The math"}
# With scores $f = (f_1, f_2)$, the Bradley–Terry likelihood is $p(y = 1 \mid
# f) = \sigma(f_1 - f_2)$. Without the output Hessian or samples of $y$, the
# only curvature left is the empirical Fisher, the outer product of the
# per-example gradients at the observed labels, so `lk.laplace` picks
# `DiagGGN(fisher="empirical")`. The linearized scores are Gaussian with mean $m$ and
# covariance $V$, so the difference $f_1 - f_2$ has mean $m_1 - m_2$ and
# variance $V_{11} + V_{22} - 2 V_{12}$, and the probit approximation gives
#
# $$
#   p(y = 1 \mid x, \mathcal{D}) \approx \sigma\left( \frac{m_1 - m_2}{\sqrt{1 + \frac{\pi}{8} (V_{11} + V_{22} - 2 V_{12})}} \right) .
# $$
# :::
#
# The full tutorial: [Likelihoods](../plugins/likelihood.py)
