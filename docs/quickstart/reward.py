# %% [markdown]
# ---
# title: Reward models
# jupyter: python3
# ---
#
# A reward model gives one number per item, and it learns from pairs: two items and which one a person preferred.
# Laplace gives each reward an error bar.
# Our items are points in the cube $[-2, 2]^3$, and a hidden utility of the first two coordinates decides the labels.
#
# ::: {.callout-note title="Plugin"}
# This page uses the [`laplacekit-reward`](https://github.com/laplacekit/laplacekit/tree/main/plugins/reward) plugin.
# Install it with:
#
# ```bash
# pip install "laplacekit[reward]"
# ```
# :::

# %%
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import laplacekit as lk

torch.set_default_dtype(torch.float64)
torch.manual_seed(2027)


def utility(x):
    return torch.sin(2 * x[:, 0]) + x[:, 1]


first, second = 4 * torch.rand(300, 3) - 2, 4 * torch.rand(300, 3) - 2
p_first = torch.sigmoid(3 * (utility(first) - utility(second)))
label = (
    torch.rand(300) > p_first
).long()  # the index of the preferred item, 0 = first

# %% [markdown]
# The network scores one item.
# `lk.reward.pairwise` turns it into a model that scores both items of a pair.
# The Bradley–Terry loss is then the cross-entropy over the two rewards.
# The labels only see the difference of two rewards, so the data can't pin down a shift of all of them.
# A bias in the last layer would be such a shift, so we leave it out.

# %%
net = nn.Sequential(nn.Linear(3, 32), nn.ReLU(), nn.Linear(32, 1, bias=False))
model = lk.reward.pairwise(net)
opt = torch.optim.Adam(net.parameters(), lr=1e-2, weight_decay=1e-2)

for _ in range(300):
    rewards = torch.cat([net(first), net(second)], -1)  # [300, 2]
    opt.zero_grad()
    F.cross_entropy(rewards, label).backward()
    opt.step()

# %% [markdown]
# `lk.reward.prepare_pairs` turns the `(first, second, label)` rows of the loader into batches of pairs.
# `lk.reward.bradley_terry()` is the likelihood.

# %%
loader = DataLoader(TensorDataset(first, second, label), batch_size=100)
data = lk.data.from_loader(loader, prepare=lk.reward.prepare_pairs)

post = lk.laplace(model, data, likelihood=lk.reward.bradley_terry())
print(post)

# %% [markdown]
# On single items, `lk.predict_outputs` gives the reward with its variance.
# We move the first coordinate from $-4$ to $4$, past the data on both sides, and keep the other two at zero.

# %%
t = torch.linspace(-4.0, 4.0, 201)
items = torch.stack([t, torch.zeros_like(t), torch.zeros_like(t)], 1)

out = lk.predict_outputs(post, items)
reward = out.mean.squeeze()
std = out.cov.per_input().squeeze().sqrt()

fig, ax = plt.subplots(figsize=(7, 4))
ax.axvspan(-2, 2, color="0.93", label="data range")
ax.plot(t, utility(items), "k--", label="utility")
ax.plot(t, reward, label="reward")
ax.fill_between(
    t, reward - 2 * std, reward + 2 * std, alpha=0.3, label="± 2 std"
)
ax.set_xlabel("first coordinate")
ax.legend()
plt.show()

print(f"reward std: {float(std[100]):.3f} at 0, {float(std[-1]):.3f} at 4")

# %% [markdown]
# Inside the data range, the reward roughly follows the shape of the utility.
# Pairs say nothing about the level of the rewards, so the curve sits wherever training left it.
# The reward's standard deviation is 0.527 at zero and grows to 1.415 at 4, past the data.
#
# On a `Pair`, `lk.predict` gives the preference probabilities.

# %%
a = torch.tensor([[0.75, 0.0, 0.0]])
b = torch.tensor([[-0.75, 0.0, 0.0]])

probs = lk.predict(post, lk.reward.Pair(a, b)).probs.squeeze()

print(f"utility: a {float(utility(a)):.3f}, b {float(utility(b)):.3f}")
print(f"P(a wins) {float(probs[0]):.3f}, P(b wins) {float(probs[1]):.3f}")

# %% [markdown]
# The utilities are 0.997 and $-0.997$, so with the labels' factor of 3, the true probability that `a` wins is $\sigma(3 \times 1.994) \approx 0.997$.
# The Laplace predictive gives 0.869: it picks the same winner but is less sure.
#
# ::: {.callout-note collapse="true" title="The math"}
# Bradley–Terry says that a person prefers item $a$ to item $b$ with
#
# $$
#   p(a \succ b) = \sigma\big(r(a) - r(b)\big) ,
# $$
#
# which is the softmax over the two rewards.
# Laplace puts a Gaussian on the weights, and the linearized network turns it into a Gaussian over the two rewards.
# Their difference $d = r(a) - r(b)$ is then Gaussian too, with mean $\mu_d$ and variance $v_d = V_{aa} + V_{bb} - 2 V_{ab}$.
# The probit approximation then gives
#
# $$
#   p(a \succ b \mid \mathcal{D}) \approx \sigma\Big(\frac{\mu_d}{\sqrt{1 + \frac{\pi}{8} v_d}}\Big) .
# $$
# :::
#
# The full tutorial: [Reward models](../examples/reward.py)
