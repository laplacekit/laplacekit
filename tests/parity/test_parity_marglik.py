"""Match laplace-torch's `marglik_training` (Immer et al. 2021, Alg. 1).

The loaders are full-batch and unshuffled, as in `make_reference.py`. Classification runs
the whole trajectory: on a `Categorical` the network losses agree. Regression runs SGD at
lr 0, so only the hyperparameters move: laplace-torch scales a Gaussian's network loss by
2σ², which changes Adam's trajectory once σ is learned (a known difference).
"""

import json
from pathlib import Path

import pytest
import torch
from torch import nn

import laplacekit as lk

REF = json.loads(Path(__file__).with_name("reference.json").read_text())
DT = torch.float64


def _tensor(x: object) -> torch.Tensor:
    return torch.tensor(x, dtype=DT)


def _load(model: nn.Module, state: dict) -> nn.Module:
    model.to(DT).load_state_dict({k: _tensor(v) for k, v in state.items()})

    return model


def _flat(prec: object) -> torch.Tensor:
    """A prior's precision as one flat tensor (a `Layerwise` mapping in block order)."""

    if isinstance(prec, dict):
        return torch.stack([p.reshape(()) for p in prec.values()])

    return torch.as_tensor(prec).reshape(-1)


@pytest.mark.parametrize(
    ("approx", "prior", "key"),
    [
        (lk.KFAC(), lk.priors.Layerwise, "kron_layerwise"),
        (lk.DenseGGN(joint=True), lk.priors.Isotropic, "full_scalar"),
        (lk.DiagGGN(), lk.priors.Diagonal, "diag_diag"),
    ],
    ids=["kron-layerwise", "full-scalar", "diag-diagonal"],
)
def test_classification(approx, prior, key):
    ref = REF["marglik"]["classification"][key]
    data = REF["functional"]["classification"]
    x, y = _tensor(data["X"]), torch.tensor(data["Y"])
    model = _load(
        nn.Sequential(nn.Linear(2, 16, bias=False), nn.Tanh(), nn.Linear(16, 3, bias=False)),
        REF["marglik"]["classification"]["state_dict"],
    )

    trained = lk.marglik.train(
        model,
        [(x, y)],
        n_data=len(x),
        likelihood=lk.Categorical(),
        optimizer=torch.optim.Adam(model.parameters(), lr=1e-2),
        epochs=20,
        hessian_approx=approx,
        prior=prior,
        marglik_frequency=5,
        n_epochs_burnin=5,
    )

    torch.testing.assert_close(
        _tensor(trained.margliks), _tensor(ref["margliks"]), rtol=1e-4, atol=0
    )
    torch.testing.assert_close(_tensor(trained.losses), _tensor(ref["losses"]), rtol=1e-4, atol=0)
    torch.testing.assert_close(
        _flat(trained.posterior.prior.prior_precision),
        _tensor(ref["prior_precision"]).reshape(-1),
        rtol=1e-4,
        atol=0,
    )

    for k, v in ref["state_dict"].items():
        torch.testing.assert_close(trained.posterior.mean[k], _tensor(v), rtol=1e-4, atol=1e-8)

    assert trained.posterior.log_marglik().item() == pytest.approx(ref["log_marglik"], rel=1e-4)


@pytest.mark.parametrize(
    ("approx", "key", "bias"),
    [(lk.DenseGGN(joint=True), "full", True), (lk.KFAC(), "nobias_kron", False)],
    ids=["full", "kron"],
)
def test_regression_hyper_phase(approx, key, bias):
    ref = REF["marglik"]["regression"][key]
    model = _load(
        nn.Sequential(nn.Linear(1, 50, bias=bias), nn.Tanh(), nn.Linear(50, 1, bias=bias)),
        REF["state_dict"] if bias else REF["nobias_state_dict"],
    )
    x, y = _tensor(REF["X"]), _tensor(REF["Y"])

    trained = lk.marglik.train(
        model,
        [(x, y)],
        n_data=len(x),
        likelihood=lk.Gaussian(),
        optimizer=torch.optim.SGD(model.parameters(), lr=0.0),
        epochs=3,
        hessian_approx=approx,
        prior=lk.priors.Isotropic,
    )

    torch.testing.assert_close(
        _tensor(trained.margliks), _tensor(ref["margliks"]), rtol=1e-4, atol=0
    )
    assert trained.posterior.prior.prior_precision.item() == pytest.approx(
        _tensor(ref["prior_precision"]).item(), rel=1e-4
    )
    assert trained.posterior.likelihood.sigma.item() == pytest.approx(ref["sigma_noise"], rel=1e-4)


def test_per_output():
    ref = REF["marglik"]["per_output"]
    data = REF["functional"]["regression2"]
    x, y = _tensor(data["X"]), _tensor(data["Y"])
    model = _load(nn.Sequential(nn.Linear(1, 30), nn.Tanh(), nn.Linear(30, 2)), data["state_dict"])

    trained = lk.marglik.train(
        model,
        [(x, y)],
        n_data=len(x),
        likelihood=lk.Gaussian(),
        sigma_init=_tensor([1.0, 1.0]),
        optimizer=torch.optim.SGD(model.parameters(), lr=0.0),
        epochs=3,
        hessian_approx=lk.DenseGGN(joint=True),
        prior=lk.priors.Isotropic,
    )

    torch.testing.assert_close(
        _tensor(trained.margliks), _tensor(ref["margliks"]), rtol=1e-4, atol=0
    )
    torch.testing.assert_close(
        _flat(trained.posterior.prior.prior_precision),
        _tensor(ref["prior_precision"]),
        rtol=1e-4,
        atol=0,
    )
    torch.testing.assert_close(
        trained.posterior.likelihood.sigma, _tensor(ref["sigma_noise"]), rtol=1e-4, atol=0
    )


class MeanVarianceNet(nn.Module):
    """The reference's network, returning `[mean, var]` joined on the last axis."""

    def __init__(self) -> None:
        super().__init__()

        self.body = nn.Sequential(nn.Linear(1, 32), nn.Tanh(), nn.Linear(32, 32), nn.Tanh())
        self.mean = nn.Linear(32, 1)
        self.var = nn.Sequential(nn.Linear(32, 1), nn.Softplus())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.body(x)

        return torch.cat([self.mean(h), self.var(h) + 1e-4], -1)


def test_heteroscedastic():
    """`MeanVarianceGaussian` against `HeteroscedasticGaussianLikelihood(full=True)`, SGD at lr 0
    for two epochs. The Laplace is over the mean only and there is no σ: only λ moves."""

    ref = REF["marglik"]["heteroscedastic"]
    data = REF["heteroscedastic"]
    x, y = _tensor(data["X"]), _tensor(data["Y"])
    model = _load(MeanVarianceNet(), data["mean_variance"]["state_dict"])

    trained = lk.marglik.train(
        model,
        [(x, y)],
        n_data=len(x),
        likelihood=lk.heteroscedastic.MeanVarianceGaussian(),
        optimizer=torch.optim.SGD(model.parameters(), lr=0.0),
        epochs=2,
        hessian_approx=lk.DenseGGN(joint=True),
        prior=lk.priors.Isotropic,
    )

    torch.testing.assert_close(
        _tensor(trained.margliks), _tensor(ref["margliks"]), rtol=1e-4, atol=0
    )
    torch.testing.assert_close(
        _flat(trained.posterior.prior.prior_precision),
        _tensor(ref["prior_precision"]),
        rtol=1e-4,
        atol=0,
    )
