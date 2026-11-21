"""Scenario: marginal-likelihood training, then the posterior it returns."""

from typing import Any, assert_type

import torch
from _fixtures import train_loader, train_my_resnet, x_test
from torch.distributions import Distribution

import laplacekit as lk

model = train_my_resnet()
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

trained = lk.marglik.train(
    model,
    train_loader,
    likelihood=lk.Categorical(),
    optimizer=opt,
    epochs=100,
    n_data=50_000,
    hessian_approx=lk.KFAC(),
    prior=lk.priors.Layerwise,
    marglik_frequency=5,
    n_epochs_burnin=10,
)
post: lk.LaplacePosterior[Any, Any] = trained.posterior
assert_type(trained.posterior.likelihood, lk.Categorical)
bare: lk.marglik.Trained = trained  # L defaults to Likelihood
margliks: tuple[float, ...] = trained.margliks
best: int | None = trained.best_epoch
pred: Distribution = lk.predict(trained.posterior, x_test)

# a regression net with a learned noise and a cosine schedule
reg = train_my_resnet()
sgd = torch.optim.SGD(reg.parameters(), lr=0.1)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(sgd, T_max=1000)
data = lk.data.from_loader(train_loader, n_data=50_000)
fitted = lk.marglik.train(
    reg, data, likelihood=lk.Gaussian(), optimizer=sgd, epochs=10, scheduler=sched
)
assert_type(fitted.posterior.likelihood, lk.Gaussian)
