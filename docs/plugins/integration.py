# %% [markdown]
# ---
# title: Library integrations
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
# A library integration gives another library's models a front door. It uses
# the same protocols as the other guides, plus a `ModelFn` that tells the core
# how to call the model. We go through `laplacekit-diffusers`, which fits a
# Laplace approximation to the denoiser of a diffusion model:
#
# ```python
# post = lk.diffusers.laplace(unet, scheduler, loader)
# ```
#
# It has four parts: a `ModelFn` wrapper, a `prepare` function that turns raw
# batches into inputs and targets, checks that run before any data, and the
# plugin record. The excerpts below are the package's own source, with the
# docstrings left out.
#
# ## The model

# %%
#| echo: false
#| output: asis
import ast
import inspect
import textwrap
from pathlib import Path

import laplacekit_diffusers
import laplacekit_diffusers.ddpm

FOLDER = "plugins/diffusers/src/laplacekit_diffusers"


def file(obj: object) -> str:
    """The path of the file that defines `obj`, from the repository root."""

    return f"{FOLDER}/{Path(inspect.getsourcefile(obj)).name}"


def show(*objs: object) -> None:
    """Print the source of each function as one fenced block, without its docstring."""

    parts = []

    for obj in objs:
        src = textwrap.dedent(inspect.getsource(obj))
        body = ast.parse(src).body[0].body
        lines = src.splitlines()

        if isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
            del lines[body[0].lineno - 1 : body[0].end_lineno + 1]  # and the blank line after

        parts.append("\n".join(lines))

    print(f'```{{.python filename="{file(objs[0])}"}}\n' + "\n\n\n".join(parts) + "\n```")


show(laplacekit_diffusers.denoiser)

# %% [markdown]
# The core calls every model as `forward(params, inputs)`, a function of the
# parameters it fits. `lk.core.ModelFn.from_module` covers `module(x)` for a tensor
# and `module(**x)` for a mapping, and returns what the module returns. A
# `diffusers` denoiser takes the noisy sample and the timestep positionally,
# plus conditioning such as class labels by name, and returns a `BaseOutput`.
# So the wrapper takes a mapping of inputs, calls the
# network with `torch.func.functional_call`, and returns the output's
# `.sample`. Parameters that are not fitted are passed detached, so no gradient
# flows into them. Passing the module as the second argument of `lk.core.ModelFn`
# lets the probe find its layers, and subsets such as `last_layer` read them.
#
# ## The data
#
# A diffusion model is trained on clean samples $x_0$, but its loss is a
# regression: at a random timestep $t$, it predicts the noise $\epsilon$ in
# $x_t = \sqrt{\bar\alpha_t} \, x_0 + \sqrt{1 - \bar\alpha_t} \, \epsilon$. The
# Laplace approximation needs that regression's inputs and targets, so
# `prepare` builds them from each raw batch:

# %%
#| echo: false
#| output: asis
show(laplacekit_diffusers.ddpm._prepare)

# %% [markdown]
# `lk.data.from_loader(data, prepare=...)` calls `prepare(raw, generator)` on
# every batch. The generator is the batch's own, seeded from the pass and the
# examples before the batch, so $t$ and $\epsilon$ are the same each time the fit
# sees the batch, and the curvature's statistic stays a [monoid](statistics.qmd).
# `prediction` is the scheduler's `prediction_type`, a value from a third-party
# config. `_check_scheduler` has already checked it against an allow-list, so
# the `match` covers every case.
#
# ## The front door
#
# `laplace` puts the parts together and hands the rest to `lk.laplace`:

# %%
#| echo: false
#| output: asis
show(laplacekit_diffusers.laplace)

# %% [markdown]
# The checks come first. `_check_scheduler` runs on the scheduler alone,
# before any data: it checks the scheduler against runtime-checkable
# protocols, one per member it needs, checks `prediction_type` against the allow-list, and checks on three
# timesteps that the noise is variance-preserving. Then one batch goes through
# the network, to check that the output has the target's shape, before the fit
# starts. Each check raises `StructureError` naming what is wrong. Here `EulerDiscreteScheduler`, whose noise is not
# variance-preserving, is refused, and the data is never touched:

# %%
from diffusers import EulerDiscreteScheduler, UNet2DModel

import laplacekit as lk


def untouched():
    raise AssertionError("the data was read")
    yield


unet = UNet2DModel(
    sample_size=8,
    in_channels=1,
    out_channels=1,
    layers_per_block=1,
    block_out_channels=(8, 16),
    norm_num_groups=4,
    down_block_types=("DownBlock2D", "DownBlock2D"),
    up_block_types=("UpBlock2D", "UpBlock2D"),
    add_attention=False,
)

try:
    lk.diffusers.laplace(unet, EulerDiscreteScheduler(), untouched())
except lk.StructureError as e:
    print(e)

# %% [markdown]
# ## The record
#
# The package ends with its health check and the record its entry point names,
# as in [Plugin record and health](record.qmd):

# %%
#| echo: false
#| output: asis
lines = inspect.getsource(laplacekit_diffusers).splitlines()
record = next(line for line in lines if line.startswith("plugin = "))

show(laplacekit_diffusers.health)
print(f'```{{.python filename="{file(laplacekit_diffusers)}"}}\n{record}\n```')

# %% [markdown]
# `laplacekit-diffusers` is first-party, so `laplacekit` also has a
# `diffusers` extra and the `lk.diffusers` alias, which imports the package on
# first use. A third-party integration is the same package without those two:
# users import it by name. The [Overview](overview.qmd) has the comparison.
#
# ## The test
#
# There is no checker for a whole integration, so its tests compare with a
# reference written by hand. The `diffusers` tests fit the same tiny
# `UNet2DModel` twice: once with `lk.diffusers.laplace`, and once with
# `lk.laplace` on a hand-written model function and a `prepare` that builds
# $x_t$ from the scheduler's $\bar\alpha_t$ directly. The two posteriors must
# agree in the mean, the prior, the data fit and the covariance, for each
# `prediction_type`. Other tests check each refusal before data, as above, and
# that conditioning entries such as `class_labels` reach the network.
