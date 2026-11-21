"""Match laplace-torch on the likelihoods beyond Categorical and Gaussian (see
`make_reference.py`).

Bernoulli: the classification blobs as three one-vs-rest labels, a bias-free last layer
(so KFAC matches; see `test_parity.py`), prior precision 1e-2 and no tuning.

Bridge: the classification example's last-layer full posterior (prior 5e-4, no
tuning), plain and normalized.

Per-output sigma: a two-output regression with sigma [0.1, 0.5], prior precision 1 and no
tuning, through `lk.laplace`. KFAC fits use the bias-free network.
"""

import copy
import dataclasses
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


def _bernoulli_model() -> nn.Module:
    model = nn.Sequential(
        nn.Linear(2, 20), nn.ReLU(), nn.Linear(20, 20), nn.ReLU(), nn.Linear(20, 3, bias=False)
    ).to(DT)
    model.load_state_dict({k: _tensor(v) for k, v in REF["bernoulli"]["state_dict"].items()})

    return model.eval()


def _bernoulli(subset, approx):
    x, y = _tensor(REF["classification"]["X"]), _tensor(REF["bernoulli"]["Y"])
    model = _bernoulli_model()
    lik = lk.Bernoulli()
    batch = lk.Batch(x, y)

    fn = lk.core.ModelFn.from_module(model)
    caps = lk.core.probe(fn, lik, batch)
    fit = lk.core.fit_curvature(fn, lik, subset(fn, caps), approx, capabilities=caps)
    stat = fit.update(fit.init(), batch)
    post = fit.posterior(
        fit.finalize(stat), prior=lk.priors.Isotropic(_tensor(1e-2)), n_data=stat.count
    )

    return post, -lk.objectives.neg_log_marglik(
        dataclasses.replace(post, data_fit=(stat.data_fit,))
    )


@pytest.mark.parametrize(
    ("subset", "approx", "key"),
    [
        (lk.subsets.last_layer, lk.DenseGGN(), "last_layer_full"),
        (lk.subsets.last_layer, lk.KFAC(), "last_layer_kron"),
        (lk.subsets.all, lk.DiagGGN(), "all_diag"),
        (lk.subsets.last_layer, lk.DenseGGN(fisher="empirical"), "ef_last_layer_full"),
    ],
    ids=["last_layer_full", "last_layer_kron", "all_diag", "ef_last_layer_full"],
)
def test_bernoulli(subset, approx, key):
    ref = REF["bernoulli"][key]
    x_test = _tensor(REF["classification"]["X_test"])
    post, log_marglik = _bernoulli(subset, approx)
    out = lk.predict_outputs(post, x_test)

    torch.testing.assert_close(out.mean, _tensor(ref["f_mu"]), rtol=1e-4, atol=0)
    torch.testing.assert_close(out.cov.per_input(), _tensor(ref["f_var"]), rtol=1e-4, atol=0)
    assert float(log_marglik.detach()) == pytest.approx(ref["log_marglik"], rel=1e-4)


@pytest.mark.parametrize(
    ("subset", "approx", "key"),
    [
        (lk.subsets.last_layer, lk.DenseGGN(), "last_layer_full"),
        (lk.subsets.last_layer, lk.KFAC(), "last_layer_kron"),
        (lk.subsets.all, lk.DiagGGN(), "all_diag"),
        (lk.subsets.last_layer, lk.DenseGGN(fisher="empirical"), "ef_last_layer_full"),
    ],
    ids=["last_layer_full", "last_layer_kron", "all_diag", "ef_last_layer_full"],
)
def test_bernoulli_probit_matches_laplace_torch(subset, approx, key):
    post, _ = _bernoulli(subset, approx)
    pred = lk.predict(post, _tensor(REF["classification"]["X_test"]))

    assert isinstance(pred, torch.distributions.Bernoulli)
    torch.testing.assert_close(
        pred.probs, _tensor(REF["bernoulli"][key]["probit"]), rtol=1e-4, atol=0
    )


def _vector_sigma(subset, approx, bias: bool):
    ref = REF["vector_sigma"]
    state = ref["state_dict"] if bias else ref["nobias_state_dict"]
    model = nn.Sequential(nn.Linear(1, 50, bias=bias), nn.Tanh(), nn.Linear(50, 2, bias=bias))
    model.to(DT).load_state_dict({k: _tensor(v) for k, v in state.items()})
    x, y = _tensor(ref["X"]), _tensor(ref["Y"])

    return lk.laplace(
        model.eval(),
        [(x, y)],
        likelihood=copy.replace(lk.Gaussian(), sigma=torch.as_tensor(_tensor(ref["sigma"]))),
        hessian_approx=approx,
        subset=subset,
        prior=lk.priors.Isotropic(_tensor(1.0)),
        hyper_tuner=None,
    )


@pytest.mark.parametrize(
    ("subset", "approx", "key", "bias"),
    [
        (lk.subsets.last_layer, lk.DenseGGN(), "last_layer_full", True),
        (lk.subsets.all, lk.DiagGGN(), "all_diag", True),
        (lk.subsets.all, lk.DenseGGN(joint=True), "all_full", True),
        (lk.subsets.all, lk.KFAC(), "nobias_all_kron", False),
        (lk.subsets.last_layer, lk.KFAC(), "nobias_last_layer_kron", False),
    ],
    ids=["last_layer_full", "all_diag", "all_full", "nobias_all_kron", "nobias_last_layer_kron"],
)
def test_vector_sigma(subset, approx, key, bias):
    ref = REF["vector_sigma"][key]
    post = _vector_sigma(subset, approx, bias)
    out = lk.predict_outputs(post, _tensor(REF["vector_sigma"]["X_test"]))

    torch.testing.assert_close(out.mean, _tensor(ref["f_mu"]), rtol=1e-4, atol=0)
    torch.testing.assert_close(out.cov.per_input(), _tensor(ref["f_var"]), rtol=1e-4, atol=1e-12)
    assert float(post.log_marglik()) == pytest.approx(ref["log_marglik"], rel=1e-4)


def _classifier() -> nn.Module:
    """The classification example's network (see `test_parity.py`)."""

    model = nn.Sequential(
        nn.Linear(2, 20),
        nn.BatchNorm1d(20),
        nn.ReLU(),
        nn.Linear(20, 20),
        nn.BatchNorm1d(20),
        nn.ReLU(),
        nn.Linear(20, 4, bias=False),
    ).to(DT)
    state = REF["classification"]["state_dict"]
    model.load_state_dict(
        {
            k: torch.tensor(v) if k.endswith("num_batches_tracked") else _tensor(v)
            for k, v in state.items()
        }
    )

    return model.eval()


def _bridge(normalized: bool) -> torch.Tensor:
    cls = REF["classification"]
    x, y = _tensor(cls["X"]), torch.tensor(cls["Y"])
    post = lk.laplace(
        _classifier(),
        [(x, y)],
        likelihood=lk.Categorical(),
        hessian_approx=lk.DenseGGN(),
        subset=lk.subsets.last_layer,
        prior=lk.priors.Isotropic(_tensor(5e-4)),
        hyper_tuner=None,
    )
    link = lk.Bridge(normalized=normalized)
    pred = lk.predict(post, _tensor(cls["X_test"]), lk.Linearized(link=link))

    return pred.probs


def test_bridge_matches_laplace_torch():
    assert all(REF["bridge"]["valid"])
    torch.testing.assert_close(_bridge(False), _tensor(REF["bridge"]["probs"]), rtol=1e-4, atol=0)


def test_bridge_norm_matches_laplace_torch():
    assert all(REF["bridge"]["valid"])
    want = _tensor(REF["bridge"]["probs_norm"])

    torch.testing.assert_close(_bridge(True), want, rtol=1e-4, atol=0)
