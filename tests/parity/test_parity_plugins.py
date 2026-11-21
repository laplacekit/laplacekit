"""Match laplace-torch on the first-party plugins' likelihoods (see `make_reference.py`).

Heteroscedastic, natural parameters: the reference's network ends with `NaturalHead()`
(η₁ = h₁, η₂ = −½ exp h₂) and is fitted over all weights with prior precision 1 and no
tuning, full and diagonal. Its mean and variances are the delta method on η → (μ, σ²).

Heteroscedastic, mean and variance: laplace-torch's `HeteroscedasticGaussianLikelihood` on a
network returning `[mean, var]`, fitted the same way.

Reward: laplace-torch's `BradleyTerryLikelihood` on 300 pairs, with prior precision 1 and no
tuning, last-layer full, all-weights diagonal and bias-free KFAC. The rewards of single items
and their variances are what it predicts.
"""

import json
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import laplacekit as lk

REF = json.loads(Path(__file__).with_name("reference.json").read_text())
DT = torch.float64


def _tensor(x: object) -> torch.Tensor:
    return torch.tensor(x, dtype=DT)


def _x_test() -> torch.Tensor:
    return _tensor(REF["heteroscedastic"]["X_test"])


def _natural(approx: lk.core.Curvature) -> lk.LaplacePosterior:
    ref = REF["heteroscedastic"]
    natural = lk.heteroscedastic
    model = nn.Sequential(
        nn.Linear(1, 32),
        nn.Tanh(),
        nn.Linear(32, 32),
        nn.Tanh(),
        nn.Linear(32, 2),
        natural.NaturalHead(),
    ).to(DT)
    model.load_state_dict({k: _tensor(v) for k, v in ref["natural"]["state_dict"].items()})
    x, y = _tensor(ref["X"]), _tensor(ref["Y"])

    return lk.laplace(
        model.eval(),
        [(x, y)],
        likelihood=natural.NaturalGaussian(),
        hessian_approx=approx,
        subset=lk.subsets.all,
        prior=lk.priors.Isotropic(_tensor(1.0)),
        hyper_tuner=None,
    )


@pytest.mark.parametrize(
    ("approx", "key"),
    [(lk.DenseGGN(joint=True), "all_full"), (lk.DiagGGN(), "all_diag")],
    ids=["all_full", "all_diag"],
)
def test_natural(approx, key):
    ref = REF["heteroscedastic"]["natural"][key]
    post = _natural(approx)
    out = lk.predict_outputs(post, _tensor(REF["heteroscedastic"]["X_test"]))

    torch.testing.assert_close(out.mean, _tensor(ref["f_mu"]), rtol=1e-4, atol=0)
    torch.testing.assert_close(out.cov.per_input(), _tensor(ref["f_var"]), rtol=1e-4, atol=1e-12)
    assert float(post.log_marglik()) == pytest.approx(ref["log_marglik"], rel=1e-4)


@pytest.mark.parametrize(
    ("approx", "key"),
    [(lk.DenseGGN(joint=True), "all_full"), (lk.DiagGGN(), "all_diag")],
    ids=["all_full", "all_diag"],
)
def test_natural_predictive(approx, key):
    ref = REF["heteroscedastic"]["natural"][key]
    post = _natural(approx)
    m = lk.heteroscedastic.moments(post.likelihood, lk.predict_outputs(post, _x_test()))

    torch.testing.assert_close(m.mean, _tensor(ref["mean"]), rtol=1e-4, atol=0)
    torch.testing.assert_close(m.epistemic, _tensor(ref["epistemic"]), rtol=1e-4, atol=1e-12)
    torch.testing.assert_close(m.aleatoric, _tensor(ref["aleatoric"]), rtol=1e-4, atol=0)


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


@pytest.mark.parametrize(
    ("approx", "key"),
    [(lk.DenseGGN(joint=True), "all_full"), (lk.DiagGGN(), "all_diag")],
    ids=["all_full", "all_diag"],
)
def test_mean_variance(approx, key):
    ref = REF["heteroscedastic"]
    want = ref["mean_variance"][key]
    model = MeanVarianceNet().to(DT)
    model.load_state_dict({k: _tensor(v) for k, v in ref["mean_variance"]["state_dict"].items()})

    post = lk.laplace(
        model.eval(),
        [(_tensor(ref["X"]), _tensor(ref["Y"]))],
        likelihood=lk.heteroscedastic.MeanVarianceGaussian(),
        hessian_approx=approx,
        subset=lk.subsets.all,
        prior=lk.priors.Isotropic(_tensor(1.0)),
        hyper_tuner=None,
    )
    m = lk.heteroscedastic.moments(post.likelihood, lk.predict_outputs(post, _x_test()))
    epistemic = _tensor(want["f_epis"]).diagonal(dim1=-2, dim2=-1)

    assert float(post.log_marglik()) == pytest.approx(want["log_marglik"], rel=1e-4)
    torch.testing.assert_close(m.mean, _tensor(want["f_mean"]), rtol=1e-4, atol=0)
    torch.testing.assert_close(m.aleatoric, _tensor(want["f_alea"]), rtol=1e-4, atol=0)
    torch.testing.assert_close(m.epistemic, epistemic, rtol=1e-4, atol=1e-12)


class RewardNet(nn.Module):
    """The reference's reward network on single items, `[N, 3] -> [N, 1]`. laplace-torch's
    `[N, 2, 3] -> [N, 2]` flattening is `lk.reward.pairwise`'s job here."""

    def __init__(self, bias: bool = True) -> None:
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(3, 32, bias=bias), nn.ReLU(), nn.Linear(32, 1, bias=bias)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@pytest.mark.parametrize(
    ("key", "approx", "params", "bias"),
    [
        ("last_layer_full", lk.DenseGGN(joint=True), lk.subsets.last_layer, True),
        ("all_diag", lk.DiagGGN(), lk.subsets.all, True),
        ("nobias_all_kron", lk.KFAC(), lk.subsets.all, False),
    ],
    ids=["last_layer_full", "all_diag", "nobias_all_kron"],
)
def test_reward(key, approx, params, bias):
    """Bradley–Terry over pairs: laplace-torch fits classification on the two rewards and
    predicts one reward per item as regression with its variance."""

    ref = REF["reward"]
    want = ref[key]
    state = ref["state_dict"] if bias else ref["nobias_state_dict"]
    net = RewardNet(bias=bias).to(DT)
    net.load_state_dict({k: _tensor(v) for k, v in state.items()})

    pairs = TensorDataset(
        _tensor(ref["X_first"]), _tensor(ref["X_second"]), torch.tensor(ref["labels"])
    )
    loader = DataLoader(pairs, batch_size=300)

    post = lk.laplace(
        lk.reward.pairwise(net.eval()),
        lk.data.from_loader(loader, prepare=lk.reward.prepare_pairs),
        likelihood=lk.reward.bradley_terry(),
        hessian_approx=approx,
        subset=params,
        prior=lk.priors.Isotropic(_tensor(1.0)),
        hyper_tuner=None,
    )
    out = lk.predict_outputs(post, _tensor(ref["X_test"]))

    assert float(post.log_marglik()) == pytest.approx(want["log_marglik"], rel=1e-4)
    torch.testing.assert_close(out.mean, _tensor(want["f_mu"]), rtol=1e-4, atol=0)
    torch.testing.assert_close(out.cov.per_input(), _tensor(want["f_var"]), rtol=1e-4, atol=1e-12)
