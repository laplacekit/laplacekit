"""A likelihood with only `log_prob`, fitted with the observed output Hessian; and one
that names its default link and sample average."""

from typing import Literal

from _fixtures import train_loader, train_my_resnet, x_test
from torch import Tensor
from torch.distributions import Distribution, Independent, Poisson

import laplacekit as lk
from laplacekit.core import Outputs, Targets


class SquaredError:
    def log_prob(self, f: Outputs, y: Targets, /) -> Tensor:
        assert isinstance(f, Tensor) and isinstance(y, Tensor)
        return -((f - y) ** 2).sum(-1)


model = train_my_resnet()
post = lk.laplace(
    model,
    train_loader,
    likelihood=SquaredError(),
    hessian_approx=lk.DenseGGN(observed=True),
)
post = lk.laplace(
    model,
    train_loader,
    likelihood=SquaredError(),
    hessian_approx=lk.DiagGGN(fisher="mc", mc_samples=8),
)


class Counts:
    """Poisson counts: names its own default link and averages output draws."""

    def log_prob(self, f: Outputs, y: Targets, /) -> Tensor:
        assert isinstance(f, Tensor) and isinstance(y, Tensor)
        return Poisson(f.exp()).log_prob(y).sum(-1)

    def distribution(self, f: Outputs, /) -> Distribution:
        assert isinstance(f, Tensor)
        return Independent(Poisson(f.exp()), 1)

    def default_link(self, n_samples: int | Literal["exact"], /) -> lk.core.LinkApprox:
        return lk.MC(n_samples=100 if n_samples == "exact" else n_samples)

    def average(self, f: Tensor, /) -> Distribution:
        return Independent(Poisson(f.exp().mean(0)), 1)

    def check_targets(self, y: Targets, /) -> None:
        assert isinstance(y, Tensor)

        if bool((y < 0).any()):
            raise ValueError("counts are never negative")


counts: lk.core.HasDefaultLink = Counts()
averaged: lk.core.HasSampleAverage = Counts()
checked: lk.core.HasTargetCheck = Counts()
shaped: lk.core.HasOutputCheck = lk.Gaussian(sigma=1.0)
bernoulli: lk.core.HasDefaultLink = lk.Bernoulli()
post = lk.laplace(
    model, train_loader, likelihood=Counts(), hessian_approx=lk.DiagGGN(fisher="mc", mc_samples=8)
)
d: Distribution = lk.predict(post, x_test)
d = lk.predict(post, x_test, lk.WeightSamples())
