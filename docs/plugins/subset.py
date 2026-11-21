# %% [markdown]
# ---
# title: Subsets
# jupyter: python3
# ---
#
# ::: {.callout-tip}
# ## New to protocols?
# The contract below is a function signature: any function with it works, and
# your editor checks it where you pass it in.
# [Protocols in one page](protocols.py) explains protocols in general.
# :::
#
# A subset picks the parameters that get a posterior. It is a function of the
# model and of what [the probe](requirements.qmd) found:
#
# ```python
# class Subset(Protocol):
#     def __call__(self, model: ModelFn, caps: Capabilities, /) -> Params | Selection: ...
# ```
#
# It returns either whole parameters, as `Params`, or some of their entries, as a
# `Selection`. We start with whole parameters.
#
# [Your first plugin](first.py) wrote `last_layers(n)`, which counts layers. Here
# we count parameters instead. `budget(max_params)` takes as many layers from the
# end of the network as fit into `max_params` parameters, so that a dense
# curvature over them stays affordable.

# %%
import pytest
import torch
from sklearn.datasets import make_blobs
from torch import nn

import laplacekit as lk
from laplacekit.core import (
    ROOT_BLOCK,
    Capabilities,
    ModelFn,
    Params,
    Selection,
    StructureError,
    Subset,
)


def budget(max_params: int) -> Subset:
    """The most layers from the end whose own parameters fit into `max_params`."""

    def subset(model: ModelFn, caps: Capabilities, /) -> Params:
        if model.module is None:
            raise StructureError("budget needs a module, but ModelFn.module is None")

        chosen: Params = {}

        for name in reversed(caps.module_order):
            module = model.module if name == ROOT_BLOCK else model.module.get_submodule(name)
            prefix = "" if name == ROOT_BLOCK else f"{name}."
            own = {prefix + k: p for k, p in module.named_parameters(recurse=False)}

            if sum(p.numel() for p in [*chosen.values(), *own.values()]) > max_params:
                break

            chosen = own | chosen  # keep the forward order

        if not chosen:
            raise StructureError(f"budget({max_params}): the last layer alone is larger")

        return chosen

    return subset


# %% [markdown]
# `caps.module_order` lists the modules that own parameters, in the order their
# forward finished, so we walk it backwards. Each layer contributes its own
# parameters, under the names the module gives them, and we stop at the first
# layer that would go over the budget.
#
# We try it on an MLP with 96, 1,056 and 132 parameters in its three layers. A
# budget of 500 gets the last layer only, and 1,200 gets the last two.

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

for max_params in (500, 1_200):
    post = lk.laplace(
        model,
        [(x, y)],
        likelihood=lk.Categorical(),
        subset=budget(max_params),
        hessian_approx=lk.DenseGGN(),
    )
    print(post)

# %% [markdown]
# The subset runs after the probe has run the model on the first batch, and
# before any data goes into the fit. So a subset that can't select anything
# should raise there, with `StructureError` or `CapabilityError`, and the user
# finds out before a long pass over the data:

# %%
try:
    lk.laplace(model, [(x, y)], likelihood=lk.Categorical(), subset=budget(100))
except lk.StructureError as e:
    print(e)

# %% [markdown]
# ## Some entries only
#
# A budget of 500 leaves 368 parameters unused, since the next layer doesn't
# fit whole. A subset can also pick single entries of a parameter, as the
# subnetwork Laplace of Daxberger et al. (2021)[^daxberger2021] does. It then
# returns a `Selection`, which holds the full tensors in `base` and the chosen
# flat indices in `index`. The easiest way to build one is
# `Selection.from_masks(params, masks)`, with one bool mask per parameter,
# shaped like it.
#
# `fill(max_params)` takes the whole layers from `budget`, and tops them up
# with the largest-magnitude entries of the other parameters, so that exactly
# `max_params` entries get a posterior.


# %%
def fill(max_params: int) -> Subset:
    """`budget(max_params)`, topped up with the largest-magnitude entries of the rest."""

    def subset(model: ModelFn, caps: Capabilities, /) -> Selection:
        whole = budget(max_params)(model, caps)
        params = lk.subsets.all(model, caps)
        rest = [n for n in params if n not in whole]
        left = max_params - sum(p.numel() for p in whole.values())
        masks = {n: torch.ones_like(p, dtype=torch.bool) for n, p in whole.items()}

        # The whole layers already fill the budget, or nothing else is left.
        if not rest or left == 0:
            return Selection.from_masks(params, masks)

        flat = torch.cat([params[n].detach().abs().reshape(-1) for n in rest])
        top = torch.zeros(flat.numel(), dtype=torch.bool)
        top[flat.topk(left).indices] = True
        parts = top.split([params[n].numel() for n in rest])
        masks |= {n: m.reshape(params[n].shape) for n, m in zip(rest, parts, strict=True)}

        return Selection.from_masks(params, masks)

    return subset


post = lk.laplace(
    model, [(x, y)], likelihood=lk.Categorical(), subset=fill(500), hessian_approx=lk.DenseGGN()
)
print(post)
print("entries:", sum(v.numel() for v in post.mean.values()))

# %% [markdown]
# The fit sees one 1-D parameter per source tensor, named `"2.weight[idx]"` and
# so on, and the rest of the module stays fixed at its trained values. The 368
# extra entries come from both earlier layers, and the fit keeps one block per
# module, three in all.
#
# ## What a subset may read
#
# A subset gets two things. `model` is the `ModelFn`, and its `module` is the
# `nn.Module` when there is one. `caps` is what the probe recorded: besides
# `module_order`, it has `hooked`, the `Linear` and `Conv2d` modules whose
# forward hooks fire (the ones KFAC can reach), and `tied`, the names of tied
# weights. A subset should decide from these, never from the module's class or
# its attribute names, so that it works on models it has never seen.
#
# It must return the module's own parameter tensors, not copies, or a
# `Selection` whose `base` holds them. A view such as `p.detach()` counts, and
# `from_masks` stores exactly that. The fit reads the live tensors, and the
# posterior's mean is taken from them. And it must return the same selection
# every time it is called, since `lk.laplace` calls it once and `check_subset`
# twice. For a `Selection`, that means the same entries too.
#
# ## The test
#
# `lk.testing.check_subset` checks both rules, for `Params` and for a
# `Selection`. It needs the forward order, which
# `lk.core.probe` with `subset={}` records without differentiating anything. On top
# of it, we compare with names we write out by hand, and check the error.


# %%
def probed(model: nn.Module) -> Capabilities:
    batch = lk.Batch(x[:4], y[:4])

    return lk.core.probe(ModelFn.from_module(model), lk.Categorical(), batch, params={})


def test_budget():
    caps = probed(model)

    for max_params in (300, 1_200, 10_000):
        lk.testing.check_subset(budget(max_params), model, caps)

    fn = ModelFn.from_module(model)

    assert list(budget(300)(fn, caps)) == ["4.weight", "4.bias"]
    assert list(budget(1_188)(fn, caps)) == ["2.weight", "2.bias", "4.weight", "4.bias"]
    assert len(budget(10_000)(fn, caps)) == 6

    with pytest.raises(StructureError, match="last layer alone"):
        budget(131)(fn, caps)


test_budget()


def test_fill():
    caps = probed(model)

    for max_params in (300, 1_200):
        lk.testing.check_subset(fill(max_params), model, caps)

    chosen = fill(500)(ModelFn.from_module(model), caps)

    assert sum(idx.numel() for idx in chosen.index.values()) == 500
    assert torch.equal(chosen.index["4.weight"], torch.arange(128))

    # Whole layers that use up the budget, or every parameter, leave nothing to top up.
    total = sum(p.numel() for p in model.parameters())

    for max_params in (1_188, total, 10_000):
        chosen = fill(max_params)(ModelFn.from_module(model), caps)
        whole = budget(max_params)(ModelFn.from_module(model), caps)

        assert {n: idx.numel() for n, idx in chosen.index.items()} == {
            n: p.numel() for n, p in whole.items()
        }


test_fill()

# %% [markdown]
# Two of the budgets sit at the edges: 1,188 is exactly the last two layers,
# and 131 is one short of the last layer's 132. In a plugin package, the test
# goes into `tests/` and runs with `pytest`; [Your first plugin](first.py) shows
# the layout.
#
# [^daxberger2021]: Daxberger, E., Nalisnick, E., Allingham, J. U., Antorán, J., & Hernández-Lobato, J. M. (2021). Bayesian deep learning via subnetwork inference. *ICML*. <https://arxiv.org/abs/2010.14689>
