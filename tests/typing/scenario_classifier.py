"""Scenario 1: a classifier in a few lines, and by hand."""

import functools
from typing import assert_type

import torch
from _fixtures import train_loader, train_my_resnet, val_batches, x_test, y_test

import laplacekit as lk

model = train_my_resnet()
post = lk.laplace(model, train_loader, likelihood=lk.Categorical())
assert_type(post.likelihood, lk.Categorical)
pred = lk.predict(post, x_test)
pred = lk.predict(post, x_test, lk.WeightSamples())

# joint output n_samples, linearized or through the network, and a softmax on top
f_lin: torch.Tensor = lk.sample_outputs(
    post, x_test, 32, generator=torch.Generator().manual_seed(1)
)
f_net = lk.sample_outputs(post, x_test, 32, kind="network")
f_jvp = lk.sample_outputs(post, x_test, 32, strategy="forward-mode")
probs = torch.softmax(f_lin, dim=-1).mean(0)
_ = pred.log_prob(y_test)

# overriding defaults, and a home-made preset
post = lk.laplace(
    model, train_loader, likelihood=lk.Categorical(), hessian_approx=lk.DiagGGN(), hyper_tuner=None
)
post = lk.laplace(
    model, train_loader, likelihood=lk.Categorical(), hessian_approx=lk.DenseGGN(fisher="empirical")
)
post = lk.laplace(
    model, train_loader, likelihood=lk.Categorical(), hessian_approx=lk.DenseHessian()
)
post = lk.laplace(
    model,
    train_loader,
    likelihood=lk.Categorical(),
    subset=lk.subsets.all,
    hessian_approx=lk.LowRankGGN(20),
)
post = lk.laplace(
    model, train_loader, likelihood=lk.Categorical(), hessian_approx=lk.KFAC(fisher="empirical")
)
post = lk.laplace(
    model,
    train_loader,
    likelihood=lk.Categorical(),
    hessian_approx=lk.DenseGGN(joint=True),
    hyper_tuner=None,
)
my_laplace = functools.partial(lk.laplace, hessian_approx=lk.DiagGGN(), hyper_tuner=None)
post = my_laplace(model, train_loader, likelihood=lk.Categorical())

# by hand
fn = lk.core.ModelFn.from_module(model)
lik = lk.Categorical()
caps = lk.core.probe(fn, lik, lk.Batch.from_pair(next(iter(train_loader))))
params = lk.subsets.last_layer(fn, caps)
fit = lk.core.fit_curvature(fn, lik, params, lk.curvature.default(params, caps))

stat = fit.init()

for pair in train_loader:
    stat = fit.update(stat, lk.Batch.from_pair(pair))

post2 = fit.posterior(
    fit.finalize(stat), prior=lk.priors.Isotropic(torch.tensor(1.0)), n_data=stat.count
)
post2 = lk.tune(post2, lk.Minimize(lk.objectives.neg_log_marglik))
post2 = lk.tune(post2, lk.Minimize(lk.objectives.brier(val_batches)))
post2 = lk.tune(
    post2,
    lk.GridSearch(
        lk.objectives.ece(val_batches, lk.Linearized(lk.Probit()), n_bins=10),
        values=torch.logspace(-2, 2, 9),
    ),
)
pred2 = lk.predict(post2, x_test, lk.Linearized(lk.Probit()))
pred_bridge = lk.predict(post2, x_test, lk.Linearized(lk.Bridge(normalized=True)))
pred_bridge = lk.predict(post2, x_test, lk.Linearized(lk.Bridge(fallback="probit")))

# an index-set subnetwork, by hand: some entries of the last layer get a posterior
mask = {name: torch.rand(p.shape) < 0.1 for name, p in params.items()}
sel = lk.core.Selection.from_masks(params, mask)
sub_fn, sub_params = lk.core.select(fn, sel)
fit3 = lk.core.fit_curvature(sub_fn, lik, sub_params, lk.DiagGGN())
fit4 = lk.core.fit_curvature(sub_fn, lik, sub_params, lk.DenseGGN(fisher="empirical", joint=True))
fit5 = lk.core.fit_curvature(sub_fn, lik, sub_params, lk.DenseHessian(joint=True))
fit6 = lk.core.fit_curvature(sub_fn, lik, sub_params, lk.LowRankGGN(rank=8, oversample=4))
_masks: dict[str, torch.Tensor] = sel.masks()

# multi-label: one sigmoid per output, the sigmoid probit by default
post_ml = lk.laplace(model, train_loader, likelihood=lk.Bernoulli())
pred_ml = lk.predict(post_ml, x_test, lk.Linearized(lk.Probit()))
post_ml = lk.laplace(
    model,
    train_loader,
    likelihood=lk.Bernoulli(event_dims=1),
    hessian_approx=lk.KFAC(),
    hyper_tuner=None,
)
