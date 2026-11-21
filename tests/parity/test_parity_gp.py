"""Match laplace-torch's `ELLA` with its subset rows and output coordinates given.

The reference stores the subset `indices` and the coordinate drawn for each subset row, so
`lk.gp.fixed(X[indices], coords)` picks the same Jacobian rows. The basis is defined up to
signs, so the comparison uses the eigenvalues of the summed projected GGN, which don't
depend on them. Its `feature_gram` is that sum with the unit likelihood: for regression it
leaves out the 1 / sigma² factor, as our `scale * gram` does. The stored moments are latent.

laplace-torch's `VaLLA` takes its inducing inputs and output coordinates through
`lk.gp.fixed`. Its objective and gradients at L = I, and its fits, use the same leaves as
`lk.gp.fit_variational`: the lower factor, log λ, log σ² and the inducing inputs.
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
ELLA = REF["ella"]
DT = torch.float64


def _tensor(x: object) -> torch.Tensor:
    return torch.tensor(x, dtype=DT)


def _check(model, data, lik, x, x_test, ref, rank):
    idx = torch.tensor(ref["indices"])
    coords = torch.tensor(ref["coords"])
    approx = lk.gp.ELLA(lk.gp.fixed(x[idx], coords), rank)

    gp = fs.laplace(
        model,
        data,
        likelihood=lik,
        method=approx,
        subset=lk.subsets.all,
        prior=lk.priors.Isotropic(_tensor(1.0)),
    )
    out = lk.predict_outputs(gp, x_test)
    cov = out.cov.per_input()
    ref_mu, ref_var = _tensor(ref["f_mu"]), _tensor(ref["f_var"])

    eigvals = torch.linalg.eigvalsh(gp.n_data * gp.gram)
    torch.testing.assert_close(eigvals, _tensor(ref["gram_eigvals"]), rtol=1e-4, atol=1e-8)
    torch.testing.assert_close(out.mean, ref_mu.reshape(out.mean.shape), rtol=1e-4, atol=0)
    torch.testing.assert_close(cov, ref_var.reshape(cov.shape), rtol=1e-4, atol=1e-12)


def test_ella_regression():
    model = nn.Sequential(nn.Linear(1, 50), nn.Tanh(), nn.Linear(50, 1)).to(DT)
    model.load_state_dict({k: _tensor(v) for k, v in REF["state_dict"].items()})
    x, y, x_test = _tensor(REF["X"]), _tensor(REF["Y"]), _tensor(REF["X_test"])

    _check(
        model,
        [(x, y)],
        copy.replace(lk.Gaussian(), sigma=torch.as_tensor(_tensor(0.3))),
        x,
        x_test,
        ELLA["regression"],
        20,
    )


def test_ella_classification():
    model = nn.Sequential(
        nn.Linear(2, 20),
        nn.BatchNorm1d(20),
        nn.ReLU(),
        nn.Linear(20, 20),
        nn.BatchNorm1d(20),
        nn.ReLU(),
        nn.Linear(20, 4, bias=False),
    ).to(DT)
    cls = REF["classification"]
    # Batch-norm step counters are integers; everything else is float64.
    model.load_state_dict(
        {
            k: torch.tensor(v) if k.endswith("num_batches_tracked") else _tensor(v)
            for k, v in cls["state_dict"].items()
        }
    )
    model.eval()
    x, x_test = _tensor(cls["X"]), _tensor(cls["X_test"])
    y = torch.tensor(cls["Y"])

    _check(model, [(x, y)], lk.Categorical(), x, x_test, ELLA["classification"], 50)


VALLA = REF["valla"]


def test_valla_covariance_at_random_factor():
    """laplace-torch's `VaLLA` at a seeded factor: the latent moments and the KL."""

    model = nn.Sequential(nn.Linear(1, 50), nn.Tanh(), nn.Linear(50, 1)).to(DT)
    model.load_state_dict({k: _tensor(v) for k, v in REF["state_dict"].items()})
    x, y, x_test = _tensor(REF["X"]), _tensor(REF["Y"]), _tensor(REF["X_test"])
    reg = VALLA["regression"]
    ref = reg["random_factor"]
    z = _tensor(reg["inducing_locations"])
    points = lk.gp.fixed(z, torch.zeros(z.shape[0], dtype=torch.int64))

    gp = fs.laplace(
        model,
        [(x, y)],
        likelihood=copy.replace(lk.Gaussian(), sigma=torch.as_tensor(_tensor(0.3))),
        method=lk.gp.VaLLA(points),
        subset=lk.subsets.all,
        prior=lk.priors.Isotropic(_tensor(1.0)),
    )
    gp = copy.replace(gp, factor=_tensor(ref["factor"]))
    out = lk.predict_outputs(gp, x_test)
    cov = out.cov.per_input()
    ref_mu, ref_var = _tensor(ref["f_mu"]), _tensor(ref["f_var"])

    torch.testing.assert_close(out.mean, ref_mu.reshape(out.mean.shape), rtol=1e-4, atol=0)
    torch.testing.assert_close(cov, ref_var.reshape(cov.shape), rtol=1e-4, atol=1e-12)
    torch.testing.assert_close(gp.kl(), _tensor(ref["kl"]), rtol=1e-4, atol=0)


def _regression_valla(sigma: float = 0.3):
    """The regression model, data and `VariationalGP` at L = I, with the reference inputs."""

    model = nn.Sequential(nn.Linear(1, 50), nn.Tanh(), nn.Linear(50, 1)).to(DT)
    model.load_state_dict({k: _tensor(v) for k, v in REF["state_dict"].items()})
    x, y = _tensor(REF["X"]), _tensor(REF["Y"])
    z = _tensor(VALLA["regression"]["inducing_locations"])
    points = lk.gp.fixed(z, torch.zeros(z.shape[0], dtype=torch.int64))

    gp = fs.laplace(
        model,
        [(x, y)],
        likelihood=copy.replace(lk.Gaussian(), sigma=torch.as_tensor(_tensor(sigma))),
        method=lk.gp.VaLLA(points),
        subset=lk.subsets.all,
        prior=lk.priors.Isotropic(_tensor(1.0)),
    )

    return model, x, y, gp


def _classification_valla():
    model = nn.Sequential(
        nn.Linear(2, 20),
        nn.BatchNorm1d(20),
        nn.ReLU(),
        nn.Linear(20, 20),
        nn.BatchNorm1d(20),
        nn.ReLU(),
        nn.Linear(20, 4, bias=False),
    ).to(DT)
    cls = REF["classification"]
    model.load_state_dict(
        {
            k: torch.tensor(v) if k.endswith("num_batches_tracked") else _tensor(v)
            for k, v in cls["state_dict"].items()
        }
    )
    model.eval()
    x, y, x_test = _tensor(cls["X"]), torch.tensor(cls["Y"]), _tensor(cls["X_test"])
    idx = torch.tensor(VALLA["classification"]["indices"])

    gp = fs.laplace(
        model,
        [(x, y)],
        likelihood=lk.Categorical(),
        method=lk.gp.VaLLA(lk.gp.fixed(x[idx], y[idx])),
        subset=lk.subsets.all,
        prior=lk.priors.Isotropic(_tensor(1.0)),
    )

    return x, y, x_test, gp


def _objective_and_grads(gp, batch, alpha):
    """`variational_loss` and its gradients in laplace-torch's trained leaves: the lower
    factor, log λ, log σ² (for a Gaussian) and the inducing inputs."""

    factor = gp.factor.clone().requires_grad_()
    log_prec = gp.prior.prior_precision.log().clone().requires_grad_()
    z = gp.inducing.clone().requires_grad_()
    leaves = {"L": factor, "log_prior_precision": log_prec, "inducing_locations": z}
    lik = gp.likelihood

    if isinstance(lik, lk.Gaussian):
        log_var = (2 * lik.sigma.log()).clone().requires_grad_()
        leaves["log_noise_variance"] = log_var
        lik = copy.replace(lk.Gaussian(), sigma=torch.as_tensor((0.5 * log_var).exp()))

    gp = copy.replace(
        gp, factor=factor, prior=lk.priors.Isotropic(log_prec.exp()), inducing=z, likelihood=lik
    )
    loss = lk.gp.variational_loss(gp, batch, alpha=alpha)
    grads = torch.autograd.grad(loss, list(leaves.values()))

    return loss, dict(zip(leaves, grads, strict=True))


def _check_grads(grads, ref):
    for name, g in grads.items():
        want = _tensor(ref[name]).reshape(g.shape)
        torch.testing.assert_close(g, want, rtol=1e-6, atol=1e-10, msg=name)


@pytest.mark.parametrize("alpha", [1.0, 0.0], ids=["alpha1", "alpha0"])
def test_valla_objective_and_grads_at_init(alpha):
    """laplace-torch's `_objective` at L = I and its gradients, for alpha 1 and the ELBO."""

    _, x, y, gp = _regression_valla()
    ref = VALLA["regression"]["alpha"][str(alpha)]

    loss, grads = _objective_and_grads(gp, lk.Batch(x, y), alpha)

    torch.testing.assert_close(loss, _tensor(ref["objective_init"]), rtol=1e-6, atol=0)
    _check_grads(grads, ref["grads_init"])


def test_valla_regression_fit():
    """Fifty Adam steps at lr 1e-2 on one full batch: λ, σ, Z, L, the objective, f_var."""

    _, x, y, gp = _regression_valla()
    ref = VALLA["regression"]["fit"]
    x_test = _tensor(REF["X_test"])

    out = lk.gp.fit_variational(gp, [(x, y)], n_steps=50, lr=1e-2)

    def close(a, b):
        torch.testing.assert_close(a, _tensor(b).reshape(a.shape), rtol=1e-4, atol=1e-10)

    close(out.prior.prior_precision, ref["prior_precision"])
    close(out.likelihood.sigma, ref["sigma_noise"])
    close(out.inducing, ref["inducing_locations"])
    close(out.factor.tril(), ref["L"])
    close(lk.gp.variational_loss(out, lk.Batch(x, y)), ref["objective_final"])
    close(lk.predict_outputs(out, x_test).cov.per_input(), ref["f_var"])

    # The input posterior is unchanged.
    torch.testing.assert_close(gp.factor, torch.eye(20, dtype=DT))


def test_valla_classification_objective_and_fit():
    """Probit closed form, alpha 1, labels as `fixed` coordinates; then 30 Adam steps."""

    x, y, x_test, gp = _classification_valla()
    ref = VALLA["classification"]

    loss, grads = _objective_and_grads(gp, lk.Batch(x, y), 1.0)

    torch.testing.assert_close(loss, _tensor(ref["objective_init"]), rtol=1e-6, atol=0)
    _check_grads(grads, ref["grads_init"])

    out = lk.gp.fit_variational(gp, [(x, y)], n_steps=30, lr=1e-2)
    fit = ref["fit"]
    moments = lk.predict_outputs(out, x_test)

    def close(a, b):
        torch.testing.assert_close(a, _tensor(b).reshape(a.shape), rtol=1e-4, atol=1e-10)

    close(out.prior.prior_precision, fit["prior_precision"])
    close(out.inducing, fit["inducing_locations"])
    close(moments.mean, fit["f_mu"])
    close(moments.cov.per_input(), fit["f_var"])
