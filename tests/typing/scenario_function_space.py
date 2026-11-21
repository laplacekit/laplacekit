"""Scenario 6: function-space Laplace with a subset of data, predict, outputs and tune."""

import torch
from _fixtures import train_loader, train_my_resnet, val_batches, x_test
from torch.distributions import Distribution

import laplacekit as lk
from laplacekit import function_space as fs

model = train_my_resnet()
lik = lk.Categorical()
gp = fs.laplace(
    model,
    train_loader,
    likelihood=lik,
    method=fs.SubsetOfData(n_subset=1000),
    prior=lk.priors.Isotropic(torch.tensor(1.0)),
)
at: lk.core.OutputGaussian = gp.at(x_test)
n_subset: int = gp.n_subset
lml: torch.Tensor = gp.log_marglik()
pred: Distribution = lk.predict(gp, x_test, lk.Linearized())
out: lk.core.OutputGaussian = lk.predict_outputs(gp, x_test)
mu = lk.core.mean_outputs(gp, x_test)
gp = lk.tune(gp, lk.Minimize(lk.objectives.predictive_nll(val_batches)))

# a fitted weight-space posterior can be the prior of a function-space update too
post = lk.laplace(model, train_loader, likelihood=lik)
gp2 = fs.laplace(
    model, train_loader, likelihood=lik, method=fs.SubsetOfData(n_subset=500), prior=post
)
cov = gp2.at(x_test).cov.per_input()

# one GP per output; its evidence tunes the prior
per_output = fs.laplace(
    model,
    train_loader,
    likelihood=lik,
    method=fs.SubsetOfData(n_subset=200, structure="independent"),
)
per_output = lk.tune(per_output, lk.Minimize(lambda g: -g.log_marglik()))
