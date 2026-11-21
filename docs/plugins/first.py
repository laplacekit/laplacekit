# %% [markdown]
# ---
# title: Your first plugin
# jupyter: python3
# ---
#
# The smallest contract in **laplacekit** is a subset: a function that gets the
# model and what [the probe](requirements.qmd) found, and returns the parameters that get a
# posterior. We write one, `last_layers(n)`, which picks the last $n$ layers,
# and ship it as a package with a test and an entry point. The package is
# [`laplacekit-plugin-example`](https://github.com/laplacekit/laplacekit-plugin-example),
# a repository of its own that doubles as a template: GitHub's **Use this
# template** button starts a new plugin from it.
#
# ## The code
#
# A plugin is an import package, here `laplacekit_example`. All of it fits in
# its `__init__.py`:

# %%
#| echo: false
#| output: asis
from pathlib import Path

import laplacekit_example

code = Path(laplacekit_example.__file__).read_text()
print(
    f'```{{.python filename="src/laplacekit_example/__init__.py"}}\n{code}```'
)

# %% [markdown]
# A subset is called after the probe has run the model on the first batch, and
# before any data goes into the fit. `caps.module_order` lists the modules that
# own parameters, in the order their forward finished, so the last $n$ entries
# are the last $n$ layers. The root module comes first as `ROOT_BLOCK` when it
# owns parameters itself. Asking for more layers than the model has raises
# `CapabilityError` with the count, rather than returning fewer layers.
#
# The subset returns the module's own parameter tensors under their names. A
# copy fails `check_subset`.
#
# ## Using it
#
# The function goes where a built-in subset would. We fit a small classifier on
# four blobs and put a posterior on its last two layers.

# %%
import torch
from laplacekit_example import last_layers
from sklearn.datasets import make_blobs
from torch import nn

import laplacekit as lk

x_np, y_np = make_blobs(
    n_samples=400, centers=4, cluster_std=1.2, random_state=0
)
x, y = torch.tensor(x_np, dtype=torch.float32), torch.tensor(y_np)

torch.manual_seed(0)
model = nn.Sequential(
    nn.Linear(2, 16), nn.Tanh(), nn.Linear(16, 16), nn.Tanh(), nn.Linear(16, 4)
)
opt = torch.optim.Adam(model.parameters(), lr=1e-2)

for _ in range(300):
    opt.zero_grad()
    nn.functional.cross_entropy(model(x), y).backward()
    opt.step()

post = lk.laplace(
    model, [(x, y)], likelihood=lk.Categorical(), subset=last_layers(2)
)
print(post)

# %% [markdown]
# The posterior covers layers `2` and `4`, the last two `Linear` layers. Layer
# `0` stays a point estimate.
#
# ## The test
#
# `lk.testing.check_subset` checks what every subset must do: return a
# non-empty mapping of the module's own parameters, by their names, and the same
# selection on a second call. On top of that, we compare the result with names
# we write out by hand:
#
# ```{.python filename="tests/test_last_layers.py"}
# import laplacekit as lk
# import pytest
# import torch
# from torch import nn
#
# from laplacekit_example import last_layers
#
#
# def mlp() -> nn.Sequential:
#     torch.manual_seed(0)
#
#     return nn.Sequential(nn.Linear(2, 8), nn.Tanh(), nn.Linear(8, 8), nn.Tanh(), nn.Linear(8, 3))
#
#
# def probed(model: nn.Module) -> lk.core.Capabilities:
#     batch = lk.Batch(torch.randn(4, 2), torch.zeros(4, dtype=torch.long))
#
#     return lk.core.probe(lk.core.ModelFn.from_module(model), lk.Categorical(), batch, params={})
#
#
# def select(subset: lk.core.Subset, model: nn.Module) -> lk.core.Params:
#     chosen = subset(lk.core.ModelFn.from_module(model), probed(model))
#     assert not isinstance(chosen, lk.core.Selection)
#
#     return chosen
#
#
# def test_passes_the_subset_checks():
#     model = mlp()
#
#     for n in (1, 2, 3):
#         lk.testing.check_subset(last_layers(n), model, probed(model))
#
#
# def test_selects_the_last_two_layers():
#     model = mlp()
#     params = select(last_layers(2), model)
#
#     assert list(params) == ["2.weight", "2.bias", "4.weight", "4.bias"]
#     assert params["4.weight"] is model[4].weight
# ```
#
# The full file also checks the forward order (a layer defined first but run
# last is the last layer), the root module's own parameters, each error, and
# that `lk.plugins()` finds the package. From a clone of the repository, run it
# with
#
# ```bash
# uv sync
# uv run pytest
# ```
#
# [Testing](testing.qmd) lists the checkers for the other kinds of plugin.
#
# ## The package
#
# The package is a `pyproject.toml` next to `src/` and `tests/`. Its first
# part:
#
# ```{.toml filename="pyproject.toml"}
# [project]
# name = "laplacekit-example"
# version = "0.1a0"
# description = "The docs' example plugin: the last n layers as a subset"
# authors = [{ name = "Agustinus Kristiadi", email = "agustinus@kristia.de" }]
# license = "MIT"
# requires-python = ">=3.13"
# dependencies = ["laplacekit>=0.1a0,<0.2"]
#
# [project.entry-points."laplacekit.plugins"]
# laplacekit_example = "laplacekit_example:plugin"
#
# [build-system]
# requires = ["uv_build>=0.13.0,<0.14"]
# build-backend = "uv_build"
#
# [dependency-groups]
# dev = [
#   # laplacekit from GitHub until it is on PyPI. Delete this line once it is.
#   "laplacekit @ git+https://github.com/laplacekit/laplacekit",
#   "pytest>=8",
#   "ruff>=0.14",
#   "ty>=0.0.104",
#   "import-linter>=2.1",
# ]
# ```
#
# The rest of the file configures `ruff`, `import-linter` (see
# [Testing](testing.qmd#imports)) and `pytest`.
#
# The convention is `laplacekit-<name>` for the distribution and
# `laplacekit_<name>` for the import package. The entry point in the group
# `laplacekit.plugins` names the import package and points at its `plugin`, the
# `lk.Plugin` record at the bottom of `__init__.py`. Its `supports` says which
# **laplacekit** versions the plugin works with. The `dev` group installs
# `laplacekit` from GitHub while it is not on PyPI; the group is for working on
# the plugin only, so a project that installs the plugin keeps its own
# `laplacekit`.
#
# `uv sync` in a clone makes an editable install, so the package and its entry
# point are visible to Python.
#
# ## Listing it
#
# `lk.plugins()` reads the entry points of everything installed. Each plugin's
# summary is the `description` from its `pyproject.toml`, and under it come the
# names in its `__all__`, each with the first line of its docstring.

# %%
print(lk.plugins())

# %% [markdown]
# `laplacekit-example` is this page's own package, installed with the docs and
# never published.
#
# Next: [Plugin record and health](record.qmd) adds checks to `lk.health()`, and
# [Publishing](publishing.qmd) puts the package on PyPI.
