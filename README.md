<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/assets/wordmark-dark.svg">
    <img alt="laplacekit" src="docs/assets/wordmark-light.svg" width="330">
  </picture>
</p>

<p align="center">
An extensible, modern <b>Laplace-approximation library</b> for PyTorch, built on a <b>plugin system</b>.
Modern models like HuggingFace LLMs, diffusion models, reward models are supported.
</p>

<p align="center">
  <a href="https://laplacekit.org"><img alt="Docs" src="https://img.shields.io/badge/docs-laplacekit.org-1f6feb?style=for-the-badge"></a>
  <a href="https://laplacekit.org/quickstart/basics.html"><img alt="Quickstart" src="https://img.shields.io/badge/quickstart-1f6feb?style=for-the-badge"></a>
  <a href="https://laplacekit.org/examples/classification.html"><img alt="In depth" src="https://img.shields.io/badge/in%20depth-1f6feb?style=for-the-badge"></a>
  <a href="https://laplacekit.org/plugins/overview.html"><img alt="Plugins" src="https://img.shields.io/badge/plugins-1f6feb?style=for-the-badge"></a>
  <br>
  <img alt="Python 3.13+" src="https://img.shields.io/badge/python-3.13%2B-3776ab?style=for-the-badge&logo=python&logoColor=white">
  <img alt="PyTorch 2.15+" src="https://img.shields.io/badge/pytorch-2.15%2B-ee4c2c?style=for-the-badge&logo=pytorch&logoColor=white">
  <a href="LICENSE"><img alt="License: MIT" src="https://img.shields.io/badge/license-MIT-555555?style=for-the-badge"></a>
</p>


## Install

```bash
pip install laplacekit
```

## First-party plugins

Available first party plugins are: HuggingFace's `transformers`, `diffusers` models, reward models (Bradley-Terry likelihood), subnetwork Laplace, heteroscedastic Laplace, function-space GP-Laplace, and marginal-likelihood-based Laplace.

Install all of them:

```bash
pip install "laplacekit[all]"
```

Or individually:

```bash
pip install "laplacekit[transformers]"
pip install "laplacekit[diffusers]"
pip install "laplacekit[subnet]"
pip install "laplacekit[heteroscedastic]"
pip install "laplacekit[reward]"
pip install "laplacekit[gp]"
pip install "laplacekit[marglik]"
```

## Usage

Train the model as usual, then fit and predict:

```python
import laplacekit as lk

# Multi-class classification, last layer, tuned prior
posterior = lk.laplace(model, train_loader, lk.Categorical())

# Predicted probabilities
probs = lk.predict(posterior, x_test).probs
```

Various arguments, including serialization:

```python
# Fit the posterior
post = lk.laplace(
    model,
    train_loader,
    lk.Gaussian(),  # Gaussian likelihood for regression
    hessian_approx=lk.KFAC(),
    subset=lk.subsets.all,
    prior=lk.priors.Layerwise(),
    hyper_tuner=lk.GridSearch(lk.objectives.predictive_nll(val_loader), values=lams),
)

# Saving the posterior
lk.save(post, "post.pt")

# Load the posterior
post = lk.load("post.pt", model)
```

## What's new?

**Plugin system.** Think Neovim or VS Code. A new Laplace method is a package:
install it with `pip` and pass it to `lk.laplace`. The built-in methods
are written the exact same way, against the same public interfaces, with no
special access. So a third-party method is a first-class citizen, and several
plugins can be mixed in one fit. Extending `laplace-torch` means subclassing
`BaseLaplace` (a few thousand lines) and sending a PR. `laplax` and
`posteriors` are libraries of functions: they can be composed, or a new method
written from scratch, but there is no plugin host.

## How it compares

| | <code><b>laplacekit</b></code> | [`laplace-torch`](https://github.com/aleximmer/Laplace) | [`laplax`](https://github.com/laplax-org/laplax) | [`posteriors`](https://github.com/normal-computing/posteriors) |
| --- | :---: | :---: | :---: | :---: |
| Framework | PyTorch | PyTorch | JAX | PyTorch |
| LLMs (Hugging Face, LoRA) without special code | ✅ | 🟡 | 🟡 | ✅ |
| Diffusion models | ✅ | ❌ | 🟡 | ❓ |
| Fit from your own training loop | ✅ | ❌ | ❓ | ✅ |
| Fully type-checked, with typed extension points | ✅ | ❌ | ❓ | ❓ |
| Options as typed objects and checked literals | ✅ | ❌ | ❌ | ✅ |
| Differentiate through the whole fit | ✅ | ❌ | ✅ | ✅ |
| Diagonal and full curvature | ✅ | ✅ | ✅ | ✅ |
| KFAC | ✅ | ✅ | ❌ | ❌ |
| Low-rank curvature | ✅ | ✅ | ✅ | ❌ |
| Last-layer Laplace | ✅ | ✅ | ❓ | ✅ |
| Subnetwork Laplace | ✅ | ✅ | ❓ | ✅ |
| Function-space (GP) Laplace | ✅ | ✅ | ❓ | ❌ |
| Linearized predictive | ✅ | ✅ | ✅ | 🟡 |
| Marginal-likelihood tuning | ✅ | ✅ | ✅ | ❌ |
| Marglik training | ✅ | ✅ | ❓ | ❓ |

✅ yes · 🟡 partly · ❌ no · ❓ not checked. Details on the [Why page](https://laplacekit.org/why.html#how-it-compares).

> [!TIP]
> Coming from `laplace-torch`? The [Why page](https://laplacekit.org/why.html#coming-from-laplace-torch)
> maps its API to <code><b>laplacekit</b></code>.

## Agent workflow

<code><b>laplacekit</b></code> is designed so that a coding agent can write a new Laplace
method without reading the core:

- Each extension point is a small typed interface in its own file, and `ty`
  tells the agent it got something wrong in seconds.
- Errors say what's missing _and_ what would work instead, so the agent can fix
  its own mistakes. A wrong argument type, such as `hessian_approx="kron"` or
  the class `lk.KFAC` for an instance, raises a `TypeError` that names what to
  pass.
- `print(post)` shows every choice a fit made, and `torch.manual_seed` (or a
  `generator=` argument) makes reruns identical, as in torch.
- `lk.testing` checks a new plugin against its interface and the laws it must
  satisfy.
- `print(lk.health())` reports whether the install works: the torch version,
  the `curvlinops` pin, autograd on a small model, float64, and each plugin's own
  checks.
