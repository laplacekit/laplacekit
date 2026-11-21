# %% [markdown]
# ---
# title: Marglik training
# jupyter: python3
# ---
#
# Weight decay and noise are usually picked by hand, or by a grid search over
# several training runs. Marginal-likelihood training learns them during the one
# run. We use the data and the network of the [regression page](regression.py).

# %%
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import laplacekit as lk

torch.set_default_dtype(torch.float64)
torch.manual_seed(711)

x = torch.rand(150, 1) * 8.0
y = torch.sin(x) + 0.3 * torch.randn(150, 1)
x_test = torch.linspace(-2.0, 10.0, 200).reshape(-1, 1)

dataset = TensorDataset(x, y)
loader = DataLoader(dataset, batch_size=50, shuffle=True)
model = nn.Sequential(nn.Linear(1, 50), nn.Tanh(), nn.Linear(50, 1))

# %% [markdown]
# `lk.marglik.train` (the `laplacekit-marglik` plugin, installed with the
# `marglik` extra) replaces the training loop. The optimizer has no weight
# decay: the prior precision $\lambda$ sets it, one per layer by default.

# %%
trained = lk.marglik.train(
    model,
    loader,
    n_data=len(dataset),
    likelihood=lk.Gaussian(),
    optimizer=torch.optim.Adam(model.parameters(), lr=1e-2),
    epochs=200,
    generator=torch.Generator().manual_seed(0),
)
post = trained.posterior
print(post)

print("lambda:", {k: round(float(v), 3) for k, v in post.prior.prior_precision.items()})
print("sigma: ", round(float(post.likelihood.sigma), 3))
print("best epoch:", trained.best_epoch)

# %%
pred = lk.predict(post, x_test)
mean, std = pred.mean.squeeze(), pred.stddev.squeeze()
t = x_test.squeeze()

fig, ax = plt.subplots(figsize=(7, 4))
ax.scatter(x.squeeze(), y.squeeze(), s=8, color="0.5", label="data")
ax.plot(t, mean, label="mean")
ax.fill_between(t, mean - 2 * std, mean + 2 * std, alpha=0.3, label="mean ± 2 std")
ax.legend()
plt.show()

# %% [markdown]
# `train` kept epoch 196 of 200, the one with the highest evidence. The learned
# $\sigma$ is 0.307, close to the 0.3 we used to make the data, and the first
# layer gets a weaker prior than the last ($\lambda$ of 0.201 against 0.716).
# `trained.posterior` is a Laplace posterior like any other, fit at the kept epoch
# with the learned $\lambda$ and $\sigma$, so `lk.predict` takes it as it is.
#
# ::: {.callout-note collapse="true" title="The math"}
# With hyperparameters $h$ (here $\lambda$ per layer and $\sigma$), the
# Laplace approximation of the log marginal likelihood at the weights
# $\theta_*$ is
#
# $$
#   \log p(\mathcal{D} \mid h) \approx \log p(\mathcal{D} \mid \theta_*, h) + \log p(\theta_* \mid h) - \tfrac{1}{2} \log \det \tfrac{1}{2 \pi} \big(\Lambda(h) + N G\big) ,
# $$
#
# with $\Lambda(h)$ the prior precision and $G$ the GGN. Immer et al.
# (2021)[^immer2021] alternate between fitting $\theta$ and $h$:
#
# 1. One epoch of gradient steps on $\theta$ for $-\log p(\mathcal{D} \mid \theta, h) - \log p(\theta \mid h)$.
# 2. After every `marglik_frequency` epochs (1 by default), a Laplace fit at the current $\theta$.
# 3. A few gradient steps on $\log h$ for the negative log marginal likelihood above.
#
# The network kept at the end is the one from the epoch with the highest
# evidence.
# :::
#
# The full tutorial: [Marginal-likelihood training](../examples/marglik.py)
#
# [^immer2021]: Immer, A., Bauer, M., Fortuin, V., Rätsch, G., & Khan, M. E. (2021). Scalable marginal likelihood estimation for model selection in deep learning. *ICML*. <https://arxiv.org/abs/2104.04975>
