"""The Gaussian `sigma` is applied late: changing it never refits.

References are dense: on the last layer of a single-output regression net, the GGN is
Φᵀ Φ / sigma² with Φ = [features, 1], exactly, for Diag (its diagonal), DenseGGN and KFAC.
"""

import copy
import dataclasses
import math
import re

import pytest
import torch
from torch import nn
from torch.distributions import Normal

import laplacekit as lk
from laplacekit.core import (
    Batch,
    CapabilityError,
    LateLikelihood,
    StructureError,
    log_lik,
    tree_leaves,
)
from laplacekit.curvlinops import KFAC

F64 = torch.float64
PREC = 1.0


def _data():
    g = torch.Generator().manual_seed(3)
    x = torch.rand(60, 1, generator=g, dtype=F64) * 6.0
    y = torch.sin(x) + 0.2 * torch.randn(60, 1, generator=g, dtype=F64)

    return x, y


def _model():
    torch.manual_seed(0)
    x, y = _data()
    model = nn.Sequential(nn.Linear(1, 8), nn.Tanh(), nn.Linear(8, 1)).to(F64)
    opt = torch.optim.Adam(model.parameters(), lr=0.05)

    for _ in range(200):
        opt.zero_grad()
        nn.functional.mse_loss(model(x), y).backward()
        opt.step()

    return model.eval(), x, y


def _loader(x, y):
    return [(x[:30], y[:30]), (x[30:], y[30:])]


def _fit(model, x, y, sigma, approx, hyper_tuner=None):
    lik = copy.replace(lk.Gaussian(), sigma=torch.as_tensor(torch.tensor(sigma, dtype=F64)))

    return lk.laplace(
        model,
        _loader(x, y),
        likelihood=lik,
        hessian_approx=approx,
        subset=lk.subsets.last_layer,
        prior=lk.priors.Isotropic(torch.tensor(PREC, dtype=F64)),
        hyper_tuner=hyper_tuner,
    )


def _reference(model, x, y, x_test, sigma, prec=PREC):
    """Marginal variances, output variances and log marglik, from dense matrices."""

    with torch.no_grad():
        phi = torch.cat([model[:-1](x), torch.ones(len(x), 1, dtype=F64)], 1)
        phi_test = torch.cat([model[:-1](x_test), torch.ones(len(x_test), 1, dtype=F64)], 1)
        theta = torch.cat([model[-1].weight.reshape(-1), model[-1].bias])
        f = model(x)

    d = phi.shape[1]
    precision = phi.T @ phi / sigma**2 + prec * torch.eye(d, dtype=F64)
    cov = torch.linalg.inv(precision)
    ll = Normal(f, sigma).log_prob(y).sum()
    log_det_ratio = torch.logdet(precision) - d * math.log(prec)
    marglik = ll - 0.5 * (log_det_ratio + prec * theta @ theta)
    out_var = (phi_test @ cov * phi_test).sum(1, keepdim=True)

    return cov.diagonal(), out_var, marglik


def _flat_marginal_variance(post):
    mv = post.variance
    return torch.cat([mv["2.weight"].reshape(-1), mv["2.bias"]])


APPROXES = [lk.DiagGGN(), lk.DenseGGN(), KFAC()]
IDS = ["Diag", "DenseGGN", "KFAC"]


@pytest.mark.parametrize("approx", APPROXES, ids=IDS)
def test_sigma_tunes_without_refit(approx):
    model, x, y = _model()
    x_test = torch.linspace(-1.0, 7.0, 9, dtype=F64).reshape(-1, 1)
    post = _fit(model, x, y, 0.3, approx)

    # By hand: the same posterior with another sigma.
    moved = dataclasses.replace(
        post,
        likelihood=copy.replace(lk.Gaussian(), sigma=torch.as_tensor(torch.tensor(0.7, dtype=F64))),
    )
    refit = _fit(model, x, y, 0.7, approx)
    var, out_var, marglik = _reference(model, x, y, x_test, 0.7)
    exact = not isinstance(approx, lk.DiagGGN)

    for p in (moved, refit):
        torch.testing.assert_close(p.precision().logdet(), refit.precision().logdet())
        torch.testing.assert_close(p.log_marglik(), refit.log_marglik())

        if exact:
            torch.testing.assert_close(_flat_marginal_variance(p), var)
            torch.testing.assert_close(lk.predict_outputs(p, x_test).cov.diagonal(), out_var)
            torch.testing.assert_close(p.log_marglik(), marglik)

    torch.testing.assert_close(_flat_marginal_variance(moved), _flat_marginal_variance(refit))

    # Through `tune`: sigma moves to the marglik optimum; a refit there agrees.
    tuned = lk.tune(post, lk.Minimize(lambda p: -p.log_marglik()))
    sigma = tuned.likelihood.sigma

    assert sigma is not None and abs(float(sigma) - 0.3) > 1e-3

    # The tune moved the prior too; the prior is added after the fit, so it carries over.
    again = dataclasses.replace(_fit(model, x, y, float(sigma), approx), prior=tuned.prior)

    torch.testing.assert_close(tuned.log_marglik(), again.log_marglik())
    torch.testing.assert_close(_flat_marginal_variance(tuned), _flat_marginal_variance(again))

    if exact:
        lam = float(tuned.prior.prior_precision)
        _, _, want = _reference(model, x, y, x_test, float(sigma), lam)
        torch.testing.assert_close(tuned.log_marglik(), want)


def test_diag_is_the_diagonal_of_the_scaled_ggn():
    model, x, y = _model()
    post = _fit(model, x, y, 0.3, lk.DiagGGN())
    moved = dataclasses.replace(
        post,
        likelihood=copy.replace(lk.Gaussian(), sigma=torch.as_tensor(torch.tensor(0.7, dtype=F64))),
    )

    with torch.no_grad():
        phi = torch.cat([model[:-1](x), torch.ones(len(x), 1, dtype=F64)], 1)

    want = 1.0 / ((phi**2).sum(0) / 0.7**2 + PREC)

    torch.testing.assert_close(_flat_marginal_variance(moved), want)


def test_default_tune_moves_sigma_and_the_prior():
    model, x, y = _model()
    lik = copy.replace(lk.Gaussian(), sigma=torch.as_tensor(torch.tensor(0.3, dtype=F64)))
    post = lk.laplace(model, _loader(x, y), likelihood=lik, subset=lk.subsets.last_layer)
    sigma = post.likelihood.sigma

    assert sigma is not None and abs(float(sigma) - 0.3) > 1e-3
    assert float(post.prior.prior_precision) != 1.0
    assert "hyper_tuner    = MarginalLikelihood()" in str(post)
    assert lik.sigma is not None and float(lik.sigma) == pytest.approx(0.3)  # input unchanged

    # At the joint optimum, the marglik gradient w.r.t. both logs vanishes.
    logs = [post.prior.prior_precision.log().requires_grad_(), sigma.log().requires_grad_()]
    at = dataclasses.replace(
        post,
        prior=lk.priors.Isotropic(logs[0].exp()),
        likelihood=copy.replace(lk.Gaussian(), sigma=torch.as_tensor(logs[1].exp())),
    )
    grads = torch.autograd.grad(at.log_marglik(), logs)

    assert all(float(g.abs()) < 1e-4 for g in grads)


def test_default_tune_in_float32_matches_float64():
    """Many residual entries in float32: the joint tune still moves the prior."""

    def fit(dtype):
        torch.manual_seed(0)
        model = nn.Sequential(nn.Linear(4, 8), nn.Tanh(), nn.Linear(8, 1)).to(dtype)
        g = torch.Generator().manual_seed(1)
        x = torch.randn(200_000, 4, generator=g, dtype=dtype)
        y = 1.1 * torch.randn(200_000, 1, generator=g, dtype=dtype)
        loader = [(x[i : i + 50_000], y[i : i + 50_000]) for i in range(0, 200_000, 50_000)]
        lik = copy.replace(lk.Gaussian(), sigma=torch.as_tensor(torch.tensor(1.0, dtype=dtype)))

        return lk.laplace(model, loader, likelihood=lik, subset=lk.subsets.last_layer)

    p32, p64 = fit(torch.float32), fit(F64)

    assert p32.prior.prior_precision.dtype == torch.float32  # tuned leaves keep their dtype
    assert p32.likelihood.sigma.dtype == torch.float32
    assert float(p64.prior.prior_precision) > 10.0  # the optimum is far from the start
    assert float(p32.prior.prior_precision) == pytest.approx(
        float(p64.prior.prior_precision), rel=0.01
    )
    assert float(p32.likelihood.sigma) == pytest.approx(float(p64.likelihood.sigma), rel=0.01)


def test_fixed_sigma_stays_fixed():
    """A sigma given as a value is fixed: the default tune moves the prior only."""

    model, x, y = _model()
    lik = lk.Gaussian(sigma=1.0)
    post = lk.laplace(model, _loader(x, y), likelihood=lik, subset=lk.subsets.last_layer)

    assert float(post.likelihood.sigma) == 1.0 and not post.likelihood.learned
    assert lik.unit() is lik
    assert float(post.prior.prior_precision) != 1.0
    assert "hyper_tuner    = MarginalLikelihood()\n" in str(post)


def test_gaussian_fit_stat_and_log_lik():
    g = torch.Generator().manual_seed(0)
    f = torch.randn(5, 3, 2, generator=g, dtype=F64)
    y = torch.randn(5, 3, 2, generator=g, dtype=F64)
    w = torch.rand(5, generator=g, dtype=F64)
    sigma = torch.tensor(0.4, dtype=F64)
    lik = copy.replace(lk.Gaussian(event_dims=1), sigma=torch.as_tensor(sigma))
    per = Normal(f, sigma).log_prob(y).reshape(5, -1).sum(1)

    assert isinstance(lik, LateLikelihood)
    torch.testing.assert_close(lik.log_lik(lik.fit_stat(f, y, None)), per.sum())
    torch.testing.assert_close(lik.log_lik(lik.fit_stat(f, y, w)), (w * per).sum())
    torch.testing.assert_close(log_lik(lik, lik.fit_stat(f, y, w)), (w * per).sum())
    torch.testing.assert_close(lik.curvature_scale(), 1 / sigma**2)

    unit = lik.unit()

    assert isinstance(unit, lk.Gaussian) and not unit.learned and unit.event_dims == 1
    torch.testing.assert_close(unit.sigma, torch.tensor(1.0, dtype=F64))
    torch.testing.assert_close(
        lk.Gaussian(sigma=1.0).curvature_scale(), torch.tensor(1.0, dtype=F64)
    )

    # Statistics add: the sum over two halves is the statistic of the whole.
    halves = lik.fit_stat(f[:2], y[:2], w[:2]) + lik.fit_stat(f[2:], y[2:], w[2:])

    torch.testing.assert_close(halves, lik.fit_stat(f, y, w))


def test_log_lik_of_a_plain_likelihood_is_the_data_fit():
    stat = torch.tensor(-3.5)

    assert log_lik(lk.Categorical(), stat) is stat


def _two_outputs():
    """A two-output regression net and its data, with the noise of each output different."""

    torch.manual_seed(0)
    g = torch.Generator().manual_seed(5)
    x = torch.rand(40, 1, generator=g, dtype=F64) * 6.0
    noise = torch.randn(40, 2, generator=g, dtype=F64) * torch.tensor([0.1, 0.5], dtype=F64)
    y = torch.cat([torch.sin(x), torch.cos(x)], 1) + noise
    model = nn.Sequential(nn.Linear(1, 8), nn.Tanh(), nn.Linear(8, 2)).to(F64)
    opt = torch.optim.Adam(model.parameters(), lr=0.05)

    for _ in range(200):
        opt.zero_grad()
        nn.functional.mse_loss(model(x), y).backward()
        opt.step()

    return model.eval(), x, y


def _fit_two(model, x, y, sigma, approx, hyper_tuner=None, **kw):
    return lk.laplace(
        model,
        _loader(x, y),
        likelihood=copy.replace(lk.Gaussian(), sigma=torch.as_tensor(sigma)),
        hessian_approx=approx,
        subset=lk.subsets.last_layer,
        prior=lk.priors.Isotropic(torch.tensor(PREC, dtype=F64)),
        hyper_tuner=hyper_tuner,
        **kw,
    )


@pytest.mark.parametrize("approx", [*APPROXES, KFAC(fisher="mc")], ids=[*IDS, "KFAC-mc"])
def test_vector_sigma_equals_scalar_when_equal(approx):
    model, x, y = _two_outputs()
    x_test = torch.linspace(-1.0, 7.0, 9, dtype=F64).reshape(-1, 1)
    # The same seed, so a sampled method draws the same targets in both fits.
    vec = _fit_two(
        model,
        x,
        y,
        torch.tensor([0.3, 0.3], dtype=F64),
        approx,
        generator=torch.Generator().manual_seed(0),
    )
    one = _fit_two(
        model,
        x,
        y,
        torch.tensor(0.3, dtype=F64),
        approx,
        generator=torch.Generator().manual_seed(0),
    )

    torch.testing.assert_close(vec.log_marglik(), one.log_marglik(), rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(
        _flat_marginal_variance(vec), _flat_marginal_variance(one), rtol=1e-12, atol=1e-12
    )
    torch.testing.assert_close(
        lk.predict_outputs(vec, x_test).cov.per_input(),
        lk.predict_outputs(one, x_test).cov.per_input(),
        rtol=1e-12,
        atol=1e-12,
    )


def test_vector_sigma_is_fixed_at_fit_time():
    sigma = torch.tensor([0.1, 0.5], dtype=F64)
    lik = copy.replace(lk.Gaussian(), sigma=torch.as_tensor(sigma))
    f = torch.zeros(3, 2, dtype=F64)

    assert lik.unit() is lik
    torch.testing.assert_close(lik.curvature_scale(), torch.ones((), dtype=F64))
    torch.testing.assert_close(lik.output_hessian(f).diagonal(), (1 / sigma**2).expand(3, 2))


def test_vector_sigma_fit_stat_and_log_lik():
    g = torch.Generator().manual_seed(0)
    f = torch.randn(5, 3, 2, generator=g, dtype=F64)
    y = torch.randn(5, 3, 2, generator=g, dtype=F64)
    w = torch.rand(5, generator=g, dtype=F64)
    sigma = torch.tensor([0.4, 1.3], dtype=F64)
    lik = copy.replace(lk.Gaussian(event_dims=2), sigma=torch.as_tensor(sigma))
    per = Normal(f, sigma).log_prob(y).reshape(5, -1).sum(1)
    stat = lik.fit_stat(f, y, w)

    assert stat.shape == (2, 3, 2)
    torch.testing.assert_close(lik.log_lik(lik.fit_stat(f, y, None)), per.sum())
    torch.testing.assert_close(lik.log_lik(stat), (w * per).sum())

    halves = lik.fit_stat(f[:2], y[:2], w[:2]) + lik.fit_stat(f[2:], y[2:], w[2:])

    torch.testing.assert_close(halves, stat)


def test_vector_sigma_ef():
    """DenseEF with a per-output sigma: the sum of g gᵀ over examples, g from `jacrev` of the
    hand log-density."""

    model, x, y = _two_outputs()
    sigma = torch.tensor([0.1, 0.5], dtype=F64)
    post = _fit_two(model, x, y, sigma, lk.DenseGGN(fisher="empirical"))
    params = {k: v.detach() for k, v in model[-1].named_parameters(prefix="2")}

    def one(p, xi, yi):
        with torch.no_grad():
            h = model[:-1](xi.unsqueeze(0))

        f = h @ p["2.weight"].T + p["2.bias"]
        lp = -0.5 * (((yi - f) / sigma) ** 2 + torch.log(2 * math.pi * sigma**2))

        return lp.sum()

    grads = torch.func.vmap(torch.func.jacrev(one), in_dims=(None, 0, 0))(params, x, y)
    g = torch.cat([grads["2.weight"].reshape(40, -1), grads["2.bias"]], 1)
    precision = g.T @ g + PREC * torch.eye(g.shape[1], dtype=F64)

    torch.testing.assert_close(
        _flat_marginal_variance(post), torch.linalg.inv(precision).diagonal()
    )
    torch.testing.assert_close(post.precision().logdet(), torch.logdet(precision))


@pytest.mark.parametrize(
    ("sigma", "event"),
    [(torch.ones(3), (2,)), (torch.ones(2, 1), (2,))],
    ids=["wrong-size", "extra-dim"],
)
def test_vector_sigma_shape_mismatch_names_both(sigma, event):
    lik = copy.replace(lk.Gaussian(), sigma=torch.as_tensor(sigma.to(F64)))
    f = torch.zeros(4, *event, dtype=F64)
    want = (
        rf"Gaussian sigma of shape {re.escape(str(sigma.shape))} doesn't fit outputs with "
        rf"event shape {re.escape(str(f.shape[1:]))}"
    )

    for call in (lik.output_hessian, lik.distribution, lambda f: lik.fit_stat(f, f, None)):
        with pytest.raises(StructureError, match=want):
            call(f)


class _Counting:
    def __init__(self, items):
        self.items, self.pulled = items, 0

    def __iter__(self):
        for it in self.items:
            self.pulled += 1
            yield it


def _hooks(model):
    return sum(len(m._forward_hooks) + len(m._forward_pre_hooks) for m in model.modules())


SHAPE_MESSAGE = (
    r"Gaussian sigma of shape torch.Size\(\[3\]\) doesn't fit outputs with event shape "
    r"torch.Size\(\[2\]\)"
)


@pytest.mark.parametrize(
    "approx",
    [lk.DiagGGN(), lk.DenseGGN(), KFAC(), KFAC(fisher="mc"), None],
    ids=["Diag", "DenseGGN", "KFAC", "KFAC-mc", "default"],
)
@pytest.mark.parametrize("targets", [True, False], ids=["targets", "no-targets"])
def test_front_door_refuses_a_mismatched_sigma_before_data(approx, targets):
    """The sigma shape is checked against the probe batch's outputs: a `StructureError`
    naming both shapes, not a missing capability, and no batch past the first. A refit on
    the same model then works (no hooks left behind)."""

    model, x, y = _two_outputs()
    batches = [(x[:20], y[:20]), (x[20:], y[20:])] if targets else [Batch(x[:20]), Batch(x[20:])]
    ld = _Counting(batches)
    bad = copy.replace(lk.Gaussian(), sigma=torch.as_tensor(torch.ones(3, dtype=F64)))

    with pytest.raises(StructureError, match=SHAPE_MESSAGE):
        lk.laplace(model, ld, likelihood=bad, hessian_approx=approx, hyper_tuner=None)

    assert ld.pulled == 1
    assert _hooks(model) == 0

    good = copy.replace(lk.Gaussian(), sigma=torch.as_tensor(torch.tensor([0.1, 0.5], dtype=F64)))
    lk.laplace(model, _loader(x, y), likelihood=good, hessian_approx=approx, hyper_tuner=None)


@pytest.mark.parametrize("targets", [True, False], ids=["targets", "no-targets"])
def test_fs_laplace_refuses_a_mismatched_sigma_before_data(targets):
    model, x, y = _two_outputs()
    ld = _Counting(
        [(x[:20], y[:20]), (x[20:], y[20:])] if targets else [Batch(x[:20]), Batch(x[20:])]
    )
    bad = copy.replace(lk.Gaussian(), sigma=torch.as_tensor(torch.ones(3, dtype=F64)))

    with pytest.raises(StructureError, match=SHAPE_MESSAGE):
        lk.function_space.laplace(
            model, ld, likelihood=bad, method=lk.function_space.SubsetOfData(4)
        )

    assert ld.pulled == 1


def test_probe_lets_a_structure_error_through():
    """A likelihood's `StructureError` is the caller's mistake, not a missing transform."""

    model, x, y = _two_outputs()
    fn = lk.core.ModelFn.from_module(model)

    with pytest.raises(StructureError, match=SHAPE_MESSAGE):
        lk.core.probe(
            fn,
            copy.replace(lk.Gaussian(), sigma=torch.as_tensor(torch.ones(3, dtype=F64))),
            Batch(x, y),
        )


@pytest.mark.parametrize("targets", [True, False], ids=["targets", "no-targets"])
def test_by_hand_probe_refuses_a_mismatched_sigma_before_the_fit(targets):
    """The by-hand path meets the check on its own probe, before `fit_curvature(...).start()`."""

    model, x, y = _two_outputs()
    fn = lk.core.ModelFn.from_module(model)
    params = {k: v.detach() for k, v in model.named_parameters() if k.startswith("2.")}
    first = Batch(x[:20], y[:20]) if targets else Batch(x[:20])
    bad = copy.replace(lk.Gaussian(), sigma=torch.as_tensor(torch.ones(3, dtype=F64)))

    for chosen in (params, {}):
        with pytest.raises(StructureError, match=SHAPE_MESSAGE):
            lk.core.probe(fn, bad, first, params=chosen)

    good = copy.replace(lk.Gaussian(), sigma=torch.as_tensor(torch.tensor([0.1, 0.5], dtype=F64)))
    caps = lk.core.probe(fn, good, first, params=params)
    step = lk.core.fit_curvature(fn, good, params, lk.DiagGGN(), capabilities=caps).start()

    assert isinstance(step, lk.core.Running)


@pytest.mark.parametrize("targets", [True, False], ids=["targets", "no-targets"])
@pytest.mark.parametrize("fit", ["laplace", "function_space"])
def test_a_per_output_sigma_costs_no_extra_forward(targets, fit):
    """The sigma check reads the probe's own forward: a per-output sigma runs the model
    as often as a scalar one."""

    model, x, y = _two_outputs()
    batches = [(x[:20], y[:20]), (x[20:], y[20:])] if targets else [Batch(x[:20]), Batch(x[20:])]
    calls = []
    handle = model.register_forward_pre_hook(lambda _m, _i: calls.append(1))

    def count(sigma):
        calls.clear()
        lik = copy.replace(lk.Gaussian(), sigma=torch.as_tensor(sigma))

        if fit == "laplace":
            lk.laplace(
                model, batches, likelihood=lik, hessian_approx=lk.DiagGGN(), hyper_tuner=None
            )
        else:
            lk.function_space.laplace(
                model, batches, likelihood=lik, method=lk.function_space.SubsetOfData(4)
            )

        return len(calls)

    try:
        scalar = count(torch.tensor(0.3, dtype=F64))
        vector = count(torch.tensor([0.1, 0.5], dtype=F64))
    finally:
        handle.remove()

    assert vector == scalar > 0


@pytest.mark.parametrize("fisher", ["exact", "mc", "empirical"])
def test_kfac_removes_its_hooks_when_the_forward_fails(fisher):
    """A failure inside Curvlinops' hooked forward leaves no hooks on the module."""

    model, x, y = _two_outputs()
    fn = lk.core.ModelFn.from_module(model)
    params = {k: v.detach() for k, v in model.named_parameters()}

    class Failing(nn.Module):
        def forward(self, h):
            # Only Curvlinops' forward runs with grad: KFAC's own forward for the
            # shapes of the targets runs under no_grad.
            if torch.is_grad_enabled():
                raise RuntimeError("the forward fails")

            return h

    lik = lk.Gaussian(sigma=1.0)
    caps = lk.core.probe(fn, lik, Batch(x, y))
    model.append(Failing())  # after both Linear layers, so their hooks are installed
    ctx = lk.core.FitContext(fn, lik, lk.core.Layout.by_module(params), caps)
    method = KFAC(fisher=fisher)

    with pytest.raises(RuntimeError, match="the forward fails"):
        method.update(
            method.init(params, ctx), params, Batch(x, y), torch.Generator().manual_seed(0), ctx
        )

    assert _hooks(model) == 0


def test_log_lik_refuses_a_statistic_of_the_other_mode():
    """Swapping a fitted posterior's sigma between one element and per output raises."""

    f = torch.zeros(4, 2, dtype=F64)
    scalar = copy.replace(lk.Gaussian(), sigma=torch.as_tensor(torch.tensor(0.3, dtype=F64)))
    vector = copy.replace(lk.Gaussian(), sigma=torch.as_tensor(torch.tensor([0.1, 0.5], dtype=F64)))

    with pytest.raises(StructureError, match="statistic"):
        scalar.log_lik(vector.fit_stat(f, f, None))

    with pytest.raises(StructureError, match="statistic"):
        vector.log_lik(scalar.fit_stat(f, f, None))


def test_vector_sigma_must_be_positive():
    with pytest.raises(ValueError, match="positive"):
        copy.replace(lk.Gaussian(), sigma=torch.as_tensor(torch.tensor([0.5, 0.0])))

    with pytest.raises(ValueError, match="positive"):
        copy.replace(lk.Gaussian(), sigma=torch.as_tensor(torch.tensor(-1.0)))

    copy.replace(lk.Gaussian(), sigma=torch.as_tensor(torch.ones(1)))  # one element is fine
    copy.replace(
        lk.Gaussian(), sigma=torch.as_tensor(torch.ones(2, 3))
    )  # checked against the outputs later


def test_vector_sigma_not_tuned_by_default():
    model, x, y = _two_outputs()
    sigma = torch.tensor([0.1, 0.5], dtype=F64)
    post = _fit_two(model, x, y, sigma, lk.DenseGGN(), hyper_tuner=lk.MarginalLikelihood())

    assert "hyper_tuner    = MarginalLikelihood()" in str(post)
    assert float(post.prior.prior_precision) != PREC
    torch.testing.assert_close(post.likelihood.sigma, sigma)


def test_tuners_leave_a_vector_sigma_alone():
    """A per-output sigma shapes the curvature, so no tuner changes it, even when learned."""

    model, x, y = _two_outputs()
    sigma = torch.tensor([0.1, 0.5], dtype=F64)
    post = _fit_two(model, x, y, sigma, lk.DenseGGN())

    assert list(lk.core.hyperparameters(post)) == ["prior_precision"]

    tuned = lk.tune(post, lk.Minimize(lambda p: -p.log_marglik()))

    torch.testing.assert_close(tuned.likelihood.sigma, sigma)
    assert float(tuned.prior.prior_precision) != PREC


def test_save_load_vector_sigma(tmp_path):
    model, x, y = _two_outputs()
    sigma = torch.tensor([0.1, 0.5], dtype=F64)
    post = _fit_two(model, x, y, sigma, lk.DenseGGN())

    lk.save(post, tmp_path / "post.pt")
    loaded = lk.load(tmp_path / "post.pt", model)

    torch.testing.assert_close(loaded.likelihood.sigma, sigma)
    torch.testing.assert_close(loaded.log_marglik(), post.log_marglik())


def test_fit_context_sees_the_unit_likelihood():
    model, x, y = _model()
    fn = lk.core.ModelFn.from_module(model)
    lik = copy.replace(lk.Gaussian(), sigma=torch.as_tensor(torch.tensor(0.3, dtype=F64)))
    params = {k: v.detach() for k, v in model[-1].named_parameters(prefix="2")}
    fit = lk.core.fit_curvature(fn, lik, params, lk.DenseGGN())

    ctx_lik = fit.ctx.likelihood

    assert isinstance(ctx_lik, lk.Gaussian) and not ctx_lik.learned and float(ctx_lik.sigma) == 1.0

    stat = fit.update(fit.init(), Batch(x, y))
    post = fit.posterior(
        fit.finalize(stat), prior=lk.priors.Isotropic(torch.tensor(PREC)), n_data=stat.count
    )

    assert post.likelihood is lik
    assert stat.data_fit is not None
    torch.testing.assert_close(
        log_lik(lik, stat.data_fit), Normal(model(x), 0.3).log_prob(y).sum().detach()
    )


class _PlainCategorical:
    """A user likelihood written as a plain class, not a pytree."""

    def log_prob(self, f, y, /):
        return torch.distributions.Categorical(logits=f).log_prob(y)

    def distribution(self, f, /):
        return torch.distributions.Categorical(logits=f)


class _DiagProbit:
    """A user link for the plain likelihood: logits scaled by their marginal variances."""

    def supports(self, lik, f_mu, /):
        return isinstance(lik, _PlainCategorical)

    def __call__(self, lik, out, /, *, generator=None):
        return lik.distribution(out.mean / torch.sqrt(1 + math.pi / 8 * out.cov.diagonal()))


def test_plain_class_likelihood():
    torch.manual_seed(0)
    g = torch.Generator().manual_seed(0)
    y = torch.arange(60) % 3
    centers = torch.tensor([[-2.0, 0.0], [2.0, 0.0], [0.0, 3.0]])
    x = centers[y] + 0.5 * torch.randn(60, 2, generator=g)
    model = nn.Sequential(nn.Linear(2, 8), nn.Tanh(), nn.Linear(8, 3))
    opt = torch.optim.Adam(model.parameters(), lr=0.05)

    for _ in range(100):
        opt.zero_grad()
        nn.functional.cross_entropy(model(x), y).backward()
        opt.step()

    lik = _PlainCategorical()
    post = lk.laplace(model, [(x[:30], y[:30]), (x[30:], y[30:])], likelihood=lik)

    assert post.likelihood is lik
    assert any(leaf is lik for leaf in tree_leaves(post))
    assert float(post.prior.prior_precision) != 1.0

    probs = lk.predict(post, x, lk.predictive.Linearized(link=_DiagProbit())).probs

    assert probs.shape == (60, 3)
    torch.testing.assert_close(probs.sum(-1), torch.ones(60))


class _MaskedLate:
    """A late likelihood that also masks positions: the data-fit statistic can't see the mask."""

    def __init__(self, active: bool):
        self.inner = lk.Gaussian(sigma=1.0)
        self.active = active

    def log_prob(self, f, y, /):
        return self.inner.log_prob(f, y)

    def distribution(self, f, /):
        return self.inner.distribution(f)

    def mask(self, f, inputs, /):
        return torch.ones(f.shape, dtype=torch.bool) if self.active else None

    def unit(self):
        return self.inner

    def curvature_scale(self):
        return torch.tensor(1.0)

    def fit_stat(self, f, y, weights):
        return self.inner.fit_stat(f, y, weights)

    def log_lik(self, stat):
        return self.inner.log_lik(stat)


@pytest.mark.parametrize("active", [True, False])
def test_late_likelihood_with_an_active_mask_raises(active):
    model, x, y = _model()
    fn = lk.core.ModelFn.from_module(model)
    lik = _MaskedLate(active)
    params = {k: v.detach() for k, v in model[-1].named_parameters(prefix="2")}
    fit = lk.core.fit_curvature(fn, lik, params, lk.DenseGGN(fisher="mc"))

    if active:
        with pytest.raises(StructureError, match="mask"):
            fit.update(fit.init(), Batch(x, y))
    else:
        stat = fit.update(fit.init(), Batch(x, y))
        assert stat.data_fit is not None


class _NoBatches:
    """Data that fails the test if the fit reads it."""

    n_data = 20

    def __call__(self, generator, /):
        raise AssertionError("the data was read")

    def __iter__(self):
        raise AssertionError("the data was read")


def test_ef_refuses_learned_sigma():
    """A learned sigma is a late hyperparameter; the EF is quadratic in it, so it refuses."""

    model = nn.Linear(2, 1).to(F64)
    fn = lk.core.ModelFn.from_module(model)
    lik = copy.replace(lk.Gaussian(), sigma=torch.as_tensor(torch.tensor(0.5, dtype=F64)))
    params = {k: v.detach() for k, v in model.named_parameters()}

    with pytest.raises(CapabilityError, match="applied after the fit"):
        lk.core.fit_curvature(fn, lik, params, lk.DiagGGN(fisher="empirical"))

    with pytest.raises(CapabilityError, match=r"(?s)fixed-sigma log-prob.*DiagGGN\(fisher='mc'\)"):
        lk.laplace(
            model,
            _NoBatches(),
            likelihood=lik,
            hessian_approx=lk.DiagGGN(fisher="empirical"),
            subset=lk.subsets.all,
        )


def test_ef_with_default_gaussian_names_a_fixed_sigma():
    """`Gaussian()` learns sigma, so the EF refuses it before any data and names the
    fixed-sigma spelling."""

    model = nn.Linear(2, 1).to(F64)
    fn = lk.core.ModelFn.from_module(model)
    params = {k: v.detach() for k, v in model.named_parameters()}

    with pytest.raises(CapabilityError, match=r"lk\.Gaussian\(sigma=0\.1\)"):
        lk.core.fit_curvature(fn, lk.Gaussian(), params, lk.DiagGGN(fisher="empirical"))

    with pytest.raises(CapabilityError, match=r"lk\.Gaussian\(sigma=0\.1\)"):
        lk.laplace(
            model,
            _NoBatches(),
            likelihood=lk.Gaussian(),
            hessian_approx=lk.DiagGGN(fisher="empirical"),
            subset=lk.subsets.all,
        )


def test_ef_fits_unit_noise_gaussian():
    """`Gaussian(sigma=1.0)` has no late hyperparameter, so the EF is exact: it matches the
    `vmap(grad)` reference of `Normal(f, 1).log_prob`, and so does the data-fit term."""

    g = torch.Generator().manual_seed(0)
    model = nn.Sequential(nn.Linear(2, 4), nn.Tanh(), nn.Linear(4, 1)).to(F64)
    x = torch.randn(20, 2, generator=g, dtype=F64)
    y = torch.randn(20, 1, generator=g, dtype=F64)
    post = lk.laplace(
        model,
        [(x[:12], y[:12]), (x[12:], y[12:])],
        likelihood=lk.Gaussian(sigma=1.0),
        hessian_approx=lk.DiagGGN(fisher="empirical"),
        subset=lk.subsets.all,
        prior=lk.priors.Isotropic(torch.tensor(PREC, dtype=F64)),
        hyper_tuner=None,
    )
    params = {k: v.detach() for k, v in model.named_parameters()}

    def one(p, xi, yi):
        f = torch.func.functional_call(model, p, (xi.unsqueeze(0),))
        return Normal(f, 1.0).log_prob(yi.unsqueeze(0)).sum()

    grads = torch.func.vmap(torch.func.grad(one), in_dims=(None, 0, 0))(params, x, y)
    want = {k: (v**2).sum(0) + PREC for k, v in grads.items()}
    var = post.variance

    for k in params:
        torch.testing.assert_close(1 / var[k], want[k])

    with torch.no_grad():
        ll = Normal(model(x), 1.0).log_prob(y).sum()

    torch.testing.assert_close(log_lik(post.likelihood, post.data_fit[0]), ll)


def test_late_likelihood_without_hyperparameter_is_one_whose_unit_is_itself():
    """The rule: a `LateLikelihood` has no late hyperparameter when `unit()` returns
    itself. A plugin whose `unit()` builds a new object is refused by the EF."""

    model = nn.Linear(2, 1).to(F64)
    fn = lk.core.ModelFn.from_module(model)
    params = {k: v.detach() for k, v in model.named_parameters()}

    @dataclasses.dataclass(frozen=True)
    class Copying:
        inner: lk.Gaussian

        def log_prob(self, f, y, /):
            return self.inner.log_prob(f, y)

        def unit(self):
            return Copying(self.inner)

        def curvature_scale(self):
            return torch.ones(())

        def fit_stat(self, f, y, weights, /):
            return self.inner.fit_stat(f, y, weights)

        def log_lik(self, stat, /):
            return self.inner.log_lik(stat)

    assert isinstance(Copying(lk.Gaussian(sigma=1.0)), LateLikelihood)

    with pytest.raises(CapabilityError, match="can't fit Copying: Copying has a late") as err:
        lk.core.fit_curvature(
            fn, Copying(lk.Gaussian(sigma=1.0)), params, lk.DiagGGN(fisher="empirical")
        )

    assert "applied after the fit" in str(err.value)

    lk.core.fit_curvature(fn, lk.Gaussian(sigma=1.0), params, lk.DiagGGN(fisher="empirical"))


def test_empirical_fisher_of_a_fixed_sigma_gaussian_log_prob():
    """The alternative the refusal names: sigma baked into `from_log_prob`."""

    g = torch.Generator().manual_seed(0)
    model = nn.Linear(2, 1).to(F64)
    x = torch.randn(20, 2, generator=g, dtype=F64)
    y = torch.randn(20, 1, generator=g, dtype=F64)
    sigma = 0.5

    def lp(f, y):
        return Normal(f, sigma).log_prob(y).sum(-1)

    lik = lk.likelihoods.from_log_prob(lp)
    post = lk.laplace(
        model,
        [(x, y)],
        likelihood=lik,
        hessian_approx=lk.DiagGGN(fisher="empirical"),
        subset=lk.subsets.all,
        hyper_tuner=None,
    )
    params = {k: v.detach() for k, v in model.named_parameters()}

    def one(p, xi, yi):
        f = torch.func.functional_call(model, p, (xi.unsqueeze(0),))
        return Normal(f, sigma).log_prob(yi.unsqueeze(0)).sum()

    grads = torch.func.vmap(torch.func.grad(one), in_dims=(None, 0, 0))(params, x, y)
    want = {k: (v**2).sum(0) + 1.0 for k, v in grads.items()}
    var = post.variance

    for k in params:
        torch.testing.assert_close(1 / var[k], want[k])
