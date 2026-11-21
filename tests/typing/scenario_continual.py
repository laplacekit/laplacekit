"""Scenario 4: continual learning. Today's posterior is tomorrow's prior."""

from collections.abc import Mapping
from typing import Any, assert_type

import torch
from _fixtures import train_loader, train_my_resnet, x_test

import laplacekit as lk
from laplacekit.core import Dense, HasBlocks, Joint, LowRankPlusDiag

model = train_my_resnet()
lik = lk.Categorical()
post_1 = lk.laplace(model, train_loader, likelihood=lik)
post_2 = lk.laplace(model, train_loader, likelihood=lik, prior=post_1)
post_3 = lk.laplace(model, train_loader, likelihood=lik, prior=post_2)

# each posterior owns its mean, and is itself a posterior to predict with
mean_1: lk.core.Params = post_1.location(post_3.mean)
pred = lk.predict(post_3, x_test)
_ = post_3.log_marglik() + post_3.prior_scatter()

# one operator across all blocks: a per-block operator, so per-block priors add to it
joint = Joint(Dense(torch.eye(3)), (("fc", 3),))
blocked: HasBlocks = joint
joint_2: Joint[Dense] = lk.priors.Layerwise({"fc": torch.tensor(1.0)}).add_precision_to(joint)
sizes: Mapping[str, int] = joint_2.add_same(joint).block_sizes()

# a low-rank curvature over all blocks; a per-block prior keeps the structure
low_rank = Joint(LowRankPlusDiag(torch.ones(3, 1), torch.ones(1), torch.zeros(3)), (("fc", 3),))
low_rank_2: Joint[LowRankPlusDiag] = lk.priors.Layerwise(
    {"fc": torch.tensor(1.0)}
).add_precision_to(low_rank)

# continual learning with KFAC: the explicit approximate opt-in sums the Kronecker factors
kfac_1 = lk.laplace(model, train_loader, likelihood=lik, hessian_approx=lk.KFAC(), hyper_tuner=None)
kfac_prior: lk.core.Prior = lk.priors.KronSum.of(kfac_1)
kfac_2 = lk.laplace(
    model,
    train_loader,
    likelihood=lik,
    hessian_approx=lk.KFAC(),
    prior=kfac_prior,
    hyper_tuner=None,
)
kfac_3 = lk.laplace(
    model,
    train_loader,
    likelihood=lik,
    hessian_approx=lk.KFAC(),
    prior=lk.priors.KronSum.of(kfac_2),
)

# continual learning with LowRankGGN: the sum is truncated back to a fixed rank
lr_1 = lk.laplace(
    model, train_loader, likelihood=lik, hessian_approx=lk.LowRankGGN(4), hyper_tuner=None
)
lr_prior: lk.core.Prior = lk.priors.LowRankSum.of(lr_1, rank=4)
lr_2 = lk.laplace(
    model, train_loader, likelihood=lik, hessian_approx=lk.LowRankGGN(4), prior=lr_prior
)
lr_cut: LowRankPlusDiag = LowRankPlusDiag(
    torch.ones(3, 1), torch.ones(1), torch.zeros(3)
).truncated(1)

# checkpoints: save a posterior, load it with the model
lk.save(kfac_3, "kfac_3.pt")
kfac_loaded: lk.LaplacePosterior = lk.load("kfac_3.pt", model)

# a per-output sigma is fixed at fit time; it saves and loads with the posterior
per_output = lk.Gaussian(sigma=torch.tensor([0.1, 0.5]))
reg = lk.laplace(
    model, train_loader, likelihood=per_output, hessian_approx=lk.KFAC(), hyper_tuner=None
)
lk.save(reg, "reg.pt")
reg_loaded: lk.LaplacePosterior = lk.load("reg.pt", model)
fixed: bool = per_output.per_output
per_output.check_outputs(torch.zeros(4, 2))

# the posterior's likelihood keeps its type: tune a scalar sigma through a typed lens
gauss = lk.laplace(model, train_loader, likelihood=lk.Gaussian(sigma=torch.tensor(0.1)))
assert_type(gauss.likelihood, lk.Gaussian)


def noise(p: lk.LaplacePosterior[Any, Any, lk.Gaussian]) -> torch.Tensor:
    assert p.likelihood.sigma is not None

    return p.likelihood.sigma


gauss = lk.tune(gauss, lk.Minimize(lk.objectives.predictive_nll([])))
