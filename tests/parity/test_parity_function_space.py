"""Match laplace-torch's `FunctionalLaplace` with the subset of data set to every row.

The reference fits on all rows, so its GP prior is not rescaled by M / N (see
`make_reference.py`). laplace-torch keeps only the diagonal of the output Hessian. A
Gaussian's output Hessian is diagonal anyway, so its joint kernel is our `"joint"`; for a
`Categorical` it is our `"diagonal"`. Its `independent_outputs=True` is our
`"independent"`. The stored moments are latent: no observation noise.
"""

import copy
import json
from pathlib import Path

import pytest
import torch
from torch import nn

import laplacekit as lk
from laplacekit import function_space as fs

REF = json.loads(Path(__file__).with_name("reference.json").read_text())
FUNCTIONAL = REF["functional"]
DT = torch.float64


def _tensor(x: object) -> torch.Tensor:
    return torch.tensor(x, dtype=DT)


def _check(model, x, y, lik, structure, x_test, ref):
    m = x.shape[0]
    approx = fs.SubsetOfData(m) if structure is None else fs.SubsetOfData(m, structure=structure)
    gp = fs.laplace(
        model,
        [(x, y)],
        likelihood=lik,
        method=approx,
        subset=lk.subsets.all,
        prior=lk.priors.Isotropic(_tensor(1.0)),
    )
    out = lk.predict_outputs(gp, x_test)
    cov = out.cov.per_input()
    ref_mu, ref_var = _tensor(ref["f_mu"]), _tensor(ref["f_var"])

    assert gp.structure == (structure or "joint")
    assert float(gp.log_marglik()) == pytest.approx(ref["log_marglik"], rel=1e-4)
    torch.testing.assert_close(out.mean, ref_mu.reshape(out.mean.shape), rtol=1e-4, atol=0)
    torch.testing.assert_close(cov, ref_var.reshape(cov.shape), rtol=1e-4, atol=1e-12)


def test_sod_regression():
    model = nn.Sequential(nn.Linear(1, 50), nn.Tanh(), nn.Linear(50, 1)).to(DT)
    model.load_state_dict({k: _tensor(v) for k, v in REF["state_dict"].items()})
    x, y, x_test = _tensor(REF["X"]), _tensor(REF["Y"]), _tensor(REF["X_test"])
    lik = copy.replace(lk.Gaussian(), sigma=torch.as_tensor(_tensor(0.3)))

    _check(model, x, y, lik, "joint", x_test, FUNCTIONAL["regression"])


def _regression2():
    ref = FUNCTIONAL["regression2"]
    model = nn.Sequential(nn.Linear(1, 30), nn.Tanh(), nn.Linear(30, 2)).to(DT)
    model.load_state_dict({k: _tensor(v) for k, v in ref["state_dict"].items()})
    x, y, x_test = _tensor(ref["X"]), _tensor(ref["Y"]), _tensor(ref["X_test"])

    return (
        model,
        x,
        y,
        copy.replace(lk.Gaussian(), sigma=torch.as_tensor(_tensor(0.2))),
        x_test,
        ref,
    )


def test_sod_regression2_joint():
    """The default structure, "joint"."""

    model, x, y, lik, x_test, ref = _regression2()

    _check(model, x, y, lik, None, x_test, ref["joint"])


def test_sod_regression2_independent():
    model, x, y, lik, x_test, ref = _regression2()

    _check(model, x, y, lik, "independent", x_test, ref["independent"])


def _classification():
    ref = FUNCTIONAL["classification"]
    model = nn.Sequential(nn.Linear(2, 16), nn.Tanh(), nn.Linear(16, 3)).to(DT)
    model.load_state_dict({k: _tensor(v) for k, v in ref["state_dict"].items()})
    x, x_test = _tensor(ref["X"]), _tensor(ref["X_test"])
    y = torch.tensor(ref["Y"], dtype=torch.int64)

    return model, x, y, x_test, ref


def test_sod_classification_diagonal():
    """laplace-torch's joint kernel keeps the output Hessian's diagonal: our "diagonal"."""

    model, x, y, x_test, ref = _classification()

    _check(model, x, y, lk.Categorical(), "diagonal", x_test, ref["joint"])


def test_sod_classification_independent():
    model, x, y, x_test, ref = _classification()

    _check(model, x, y, lk.Categorical(), "independent", x_test, ref["independent"])
