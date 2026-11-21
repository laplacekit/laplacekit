"""Scenario: an index-set subnetwork through the front doors."""

import copy

import torch
from _fixtures import train_loader, train_my_resnet, x_test
from torch import Tensor

import laplacekit as lk
from laplacekit import function_space as fs

model = train_my_resnet()
lik = lk.Categorical()


# a subset that keeps a tenth of the last layer's weight
def some_entries(
    fn: lk.core.ModelFn, caps: lk.core.Capabilities, /
) -> lk.core.Params | lk.core.Selection:
    params = lk.subsets.last_layer(fn, caps)
    assert not isinstance(params, lk.core.Selection)
    masks = {name: torch.rand(p.shape) < 0.1 for name, p in params.items()}

    return lk.core.Selection.from_masks(params, masks)


post = lk.laplace(model, train_loader, likelihood=lik, subset=some_entries)
index: dict[str, Tensor] = dict(post.index)
pred = lk.predict(post, x_test)

# a selection built by hand, given directly
fn = lk.core.ModelFn.from_module(model)
sel = lk.core.Selection.from_masks(
    dict(model.named_parameters()), {"fc.weight": torch.ones(2, 2) > 0}
)
post_2 = lk.laplace(model, train_loader, likelihood=lik, subset=sel, hessian_approx=lk.DiagGGN())
gp = fs.laplace(
    model, train_loader, likelihood=lik, method=fs.SubsetOfData(n_subset=100), subset=sel
)

# by hand, the posterior is told which entries it covers
sub_fn, sub_params = lk.core.select(fn, sel)
fit = lk.core.fit_curvature(sub_fn, lik, sub_params, lk.DiagGGN())
caps = lk.core.probe(sub_fn, lik, lk.Batch(x_test), params=sub_params)
blocks: frozenset[str] = caps.selected_blocks
tagged = copy.replace(post_2, index=sel.selected_index)

# built-in masks; the subnet plugin's random, largest magnitude, largest diag-Laplace and
# SWAG variance, and the rescaled prior
masks = {"fc.weight": torch.ones(2, 2) > 0}
by_mask: lk.core.Subset = lk.subsets.from_mask(masks)
by_chance: lk.core.Subset = lk.subnet.random(
    100, generator=torch.Generator().manual_seed(1), subset=lk.subsets.last_layer
)
by_size: lk.core.Subset = lk.subnet.largest_magnitude(100, subset=lk.subsets.all)
diag = lk.laplace(
    model, train_loader, likelihood=lik, subset=lk.subsets.all, hessian_approx=lk.DiagGGN()
)
by_variance: lk.core.Subset = lk.subnet.largest_variance(diag, 100)
post_3 = lk.laplace(
    model, train_loader, likelihood=lik, subset=by_variance, hessian_approx=lk.DenseGGN()
)
var: lk.core.Params = lk.subnet.swag_variance(model, train_loader, likelihood=lik, n_snapshots=2)
post_4 = lk.laplace(
    model,
    train_loader,
    likelihood=lik,
    subset=lk.subnet.largest_variance(var, 30),
    prior=lk.subnet.rescaled_prior(1.0, 30, model),
)
