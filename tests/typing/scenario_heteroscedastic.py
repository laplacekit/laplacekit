"""Scenario: heteroscedastic regression in natural parameters, through `lk.heteroscedastic`."""

from _fixtures import train_loader, x_test
from torch import nn

import laplacekit as lk

het = lk.heteroscedastic
model = nn.Sequential(nn.Linear(1, 32), nn.Tanh(), nn.Linear(32, 2), het.NaturalHead())
soft = het.NaturalHead(link="softplus")

post = lk.laplace(model, train_loader, likelihood=lk.heteroscedastic.NaturalGaussian())
diag = lk.laplace(
    model, train_loader, likelihood=het.NaturalGaussian(), hessian_approx=lk.DiagGGN()
)
out = lk.predict_outputs(post, x_test)

# The mean and the variance, and the predictive moments of either likelihood.
mean_var = het.MeanVarianceGaussian(eps=1e-4)
mv = lk.laplace(model, train_loader, likelihood=mean_var, subset=lk.subsets.all)
m: het.Moments = het.moments(mean_var, lk.predict_outputs(mv, x_test))
noisy = lk.predict(post, x_test, lk.Linearized(link=het.NoisyMean()))
from_post = het.moments(post.likelihood, out)
