"""`lk.marglik.train` with `NaturalGaussian` (Immer et al. 2023): the first hyperstep
against a dense marginal likelihood built here, the training loss, and the vmap paths.

`NaturalGaussian` has no σ, so only the prior precision is learned. The parity of
`MeanVarianceGaussian` with laplace-torch is in `tests/parity/test_parity_marglik.py`.
"""

import math

import pytest
import torch
from laplacekit_heteroscedastic import NaturalGaussian, NaturalHead
from torch import Tensor, nn
from torch.distributions import Normal
from torch.func import functional_call, jacrev

import laplacekit as lk

DT = torch.float64


def _data(n: int = 20) -> tuple[Tensor, Tensor]:
    g = torch.Generator().manual_seed(7)
    x = torch.randn(n, 2, generator=g, dtype=DT)
    y = torch.sin(x[:, :1]) + (0.1 + 0.2 * x[:, 1:].abs()) * torch.randn(
        n, 1, generator=g, dtype=DT
    )

    return x, y


def _model(seed: int = 0) -> nn.Module:
    torch.manual_seed(seed)

    return nn.Sequential(nn.Linear(2, 4), nn.Tanh(), nn.Linear(4, 2), NaturalHead()).to(DT)


def _moments(eta: Tensor) -> tuple[Tensor, Tensor]:
    eta1, eta2 = eta.chunk(2, -1)

    return -eta1 / (2 * eta2), -1 / (2 * eta2)


def _hessian(eta: Tensor) -> Tensor:
    """∇²A(η) for K = 1, [N, 2, 2]: [[σ², 2μσ²], [2μσ², 4μ²σ² + 2σ⁴]]."""

    mu, var = (t[:, 0] for t in _moments(eta))
    a, b, c = var, 2 * mu * var, 4 * mu**2 * var + 2 * var**2

    return torch.stack([torch.stack([a, b], -1), torch.stack([b, c], -1)], -2)


def _dense_neg_log_marglik(model: nn.Module, x: Tensor, y: Tensor, lam: Tensor) -> Tensor:
    """-log Z of the full GGN Laplace with an isotropic prior, from `jacrev` Jacobians."""

    params = {k: v.detach() for k, v in model.named_parameters()}
    shapes = {k: v.shape for k, v in params.items()}
    sizes = [v.numel() for v in params.values()]
    theta = torch.cat([v.reshape(-1) for v in params.values()])

    def f(flat: Tensor) -> Tensor:
        parts = torch.split(flat, sizes)
        ps = {k: p.reshape(shapes[k]) for k, p in zip(params, parts, strict=True)}

        return functional_call(model, ps, (x,))

    eta = f(theta)
    jac = jacrev(f)(theta)
    ggn = torch.einsum("nkp,nkl,nlq->pq", jac, _hessian(eta), jac)
    prec = lam * torch.eye(theta.numel(), dtype=DT) + ggn

    mu, var = _moments(eta)
    log_lik = Normal(mu, var.sqrt()).log_prob(y).sum()
    occam = torch.logdet(prec) - theta.numel() * torch.log(lam) + lam * theta @ theta

    return -(log_lik - 0.5 * occam)


def test_natural_first_step_matches_dense():
    x, y = _data()
    model = _model()

    # Two estimations, one hyperstep each; lr 0 keeps the network and its GGN fixed.
    trained = lk.marglik.train(
        model,
        [(x, y)],
        n_data=len(x),
        likelihood=NaturalGaussian(),
        optimizer=torch.optim.SGD(model.parameters(), lr=0.0),
        epochs=2,
        hessian_approx=lk.DenseGGN(joint=True),
        prior=lk.priors.Isotropic,
        n_hypersteps=1,
    )

    log_lam = torch.zeros((), dtype=DT, requires_grad=True)
    first = _dense_neg_log_marglik(model, x, y, log_lam.exp())
    (g_lam,) = torch.autograd.grad(first, (log_lam,))

    # Adam's first step is lr * sign(g), up to its eps.
    lam_1 = torch.exp(-0.1 * g_lam.sign())
    second = _dense_neg_log_marglik(model, x, y, lam_1)

    assert len(trained.margliks) == 2
    assert trained.margliks[0] == pytest.approx(first.item(), rel=1e-6)
    assert trained.margliks[1] == pytest.approx(second.item(), rel=1e-6)

    assert trained.best_epoch == 2
    torch.testing.assert_close(trained.posterior.prior.prior_precision, lam_1, rtol=1e-6, atol=0)
    assert isinstance(trained.posterior.likelihood, NaturalGaussian)


def test_natural_training_loss():
    x, y = _data()
    model = _model()
    lam = 2.0
    theta = torch.cat([p.detach().reshape(-1) for p in model.parameters()])

    with torch.no_grad():
        mu, var = _moments(model(x))

    trained = lk.marglik.train(
        model,
        [(x, y)],
        n_data=len(x),
        likelihood=NaturalGaussian(),
        optimizer=torch.optim.Adam(model.parameters(), lr=1e-2),
        epochs=1,
        marglik_frequency=2,
        prior=lk.priors.Isotropic(lam),
    )

    nll = -Normal(mu, var.sqrt()).log_prob(y).sum(-1).mean()
    expected = nll + 0.5 * lam * theta @ theta / len(x)

    assert trained.losses[0] == pytest.approx(expected.item(), rel=1e-12)

    # The Adam step moved the network.
    after = torch.cat([p.detach().reshape(-1) for p in model.parameters()])

    assert not torch.equal(after, theta)


@pytest.mark.parametrize("approx", [lk.DiagGGN(), lk.DenseGGN()], ids=["diag", "dense"])
def test_natural_under_vmap(approx):
    x, y = _data()
    model = _model()

    trained = lk.marglik.train(
        model,
        [(x[:10], y[:10]), (x[10:], y[10:])],
        n_data=len(x),
        likelihood=NaturalGaussian(),
        optimizer=torch.optim.Adam(model.parameters(), lr=1e-2),
        epochs=2,
        hessian_approx=approx,
        n_hypersteps=3,
    )

    assert len(trained.margliks) == 6
    assert all(math.isfinite(m) for m in trained.margliks)
    assert trained.best_epoch in (1, 2)
    assert math.isfinite(trained.posterior.log_marglik().item())
