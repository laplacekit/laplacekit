"""ELLA: a Nyström-basis GP from `fs.laplace`, then predict, outputs and tune."""

from typing import assert_type

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
    method=lk.gp.ELLA(lk.gp.from_data(100), rank=20),
    subset=lk.subsets.all,
    prior=lk.priors.Isotropic(torch.tensor(1.0)),
)
at: lk.core.OutputGaussian = gp.at(x_test)
basis: torch.Tensor = gp.basis
pred: Distribution = lk.predict(gp, x_test)
out: lk.core.OutputGaussian = lk.predict_outputs(gp, x_test)
gp = lk.tune(gp, lk.Minimize(lk.objectives.predictive_nll(val_batches)))

# every output of k-means points, in reverse mode
coords = torch.zeros(10, dtype=torch.int64)
every = lk.gp.ELLA(lk.gp.kmeans(50), 40, outputs="all", strategy="reverse-mode")
given = lk.gp.ELLA(lk.gp.fixed(x_test, coords), 5)
gp2: lk.gp.NystromGP = fs.laplace(model, train_loader, likelihood=lik, method=every)

# VaLLA: inducing points with predicted coordinates, then the KL and the moments
valla = lk.gp.VaLLA(lk.gp.from_data(50))
approx: fs.FunctionSpaceMethod[lk.gp.VariationalGP] = valla
vgp = fs.laplace(model, train_loader, likelihood=lik, method=valla)
kl: torch.Tensor = vgp.kl()
factor: torch.Tensor = vgp.factor
vout: lk.core.OutputGaussian = lk.predict_outputs(vgp, x_test)
random = lk.gp.VaLLA(lk.gp.kmeans(20), coords="random")

# train the factor, the prior precision and the inducing inputs, then a Monte Carlo loss
vgp = lk.gp.fit_variational(vgp, train_loader, n_steps=100, lr=1e-2)
trained: lk.gp.VariationalGP = vgp
loss: torch.Tensor = lk.gp.variational_loss(vgp, val_batches[0], alpha=0.5, n_samples=16)
mc = lk.gp.fit_variational(
    vgp, train_loader, n_steps=10, n_samples=8, generator=torch.Generator().manual_seed(1)
)

# FSP-Laplace's prior: a kernel built from parts, its covariance and the RKHS norm
kernel: lk.gp.Kernel = lk.gp.Matern("5/2", torch.tensor(0.5)) * lk.gp.RBF(torch.tensor(1.0))
summed: lk.gp.Sum = lk.gp.Periodic(torch.tensor(1.0), torch.tensor(0.5)) + kernel
fsp_prior = lk.gp.FunctionPrior(summed, jitter=1e-5)
context = lk.gp.uniform(torch.zeros(2), 1.0, 50)
c: torch.Tensor = context.draw(generator=torch.Generator().manual_seed(3))
k_cc: torch.Tensor = fsp_prior.cov(c)
norm: torch.Tensor = lk.gp.rkhs_norm(fsp_prior, c, model(c))
square: lk.gp.Points = lk.gp.grid(torch.full((2,), -3.0), 3.0, [10, 10])

# FSP-Laplace: the posterior under that prior, at 50 context points, then predict and tune σ
gauss = lk.Gaussian(sigma=torch.tensor(0.1))
fsp = lk.gp.fsp_laplace(
    model, train_loader, likelihood=gauss, prior=fsp_prior, context=context, rank=20
)
on_grid = lk.gp.fsp_laplace(
    model,
    train_loader,
    likelihood=gauss,
    prior=fsp_prior,
    context=square,
    rank=10,
    subset=lk.subsets.all,
    strategy="reverse-mode",
    generator=torch.Generator().manual_seed(0),
)
fsp_pred: Distribution = lk.predict(fsp, x_test)
fsp_out: lk.core.OutputGaussian = lk.predict_outputs(fsp, x_test)


# the posterior's likelihood keeps its type, so a lens reaches sigma without a narrowing
assert_type(fsp.likelihood, lk.Gaussian)
fsp_bare: lk.gp.FSPPosterior = fsp  # L defaults to Likelihood


def noise(p: lk.gp.FSPPosterior[lk.Gaussian]) -> torch.Tensor:
    assert p.likelihood.sigma is not None

    return p.likelihood.sigma


fsp = lk.tune(fsp, lk.Minimize(lk.objectives.predictive_nll(val_batches)))
