"""Continual learning: today's posterior is tomorrow's prior."""

import copy
import math

import pytest
import torch
from torch import nn

import laplacekit as lk
from laplacekit.core import BlockDiag, Dense

LAM = 2.0
SIGMA = 0.5


def regression(n: int, seed: int):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 3, generator=g, dtype=torch.float64)
    w = torch.tensor([[1.0, -2.0, 0.5]], dtype=torch.float64)
    y = x @ w.T + 0.3 + SIGMA * torch.randn(n, 1, generator=g, dtype=torch.float64)

    return x, y


def design(x):
    """Rows [x, 1]: the flat order of a Linear's weight, then its bias."""

    return torch.cat([x, torch.ones(x.shape[0], 1, dtype=x.dtype)], dim=1)


def set_params(model: nn.Linear, w: torch.Tensor) -> None:
    with torch.no_grad():
        model.weight.copy_(w[:3].reshape(1, 3))
        model.bias.copy_(w[3:])


def flat(params) -> torch.Tensor:
    return torch.cat([params["weight"].reshape(-1), params["bias"].reshape(-1)])


def dense(op) -> torch.Tensor:
    if isinstance(op, BlockDiag):
        (op,) = op.blocks.values()

    assert isinstance(op, Dense)

    return op.mat


def fit(model, x, y, prior):
    lik = copy.replace(
        lk.Gaussian(), sigma=torch.as_tensor(torch.tensor(SIGMA, dtype=torch.float64))
    )

    return lk.laplace(
        model, [(x, y)], likelihood=lik, hessian_approx=lk.DenseGGN(), prior=prior, hyper_tuner=None
    )


def continual_setup():
    """Two fits, D1 then D2 with the first posterior as prior, each at its closed-form MAP.

    Returns the two posteriors, the MAPs, and the closed-form precisions.
    """

    x1, y1 = regression(40, 0)
    x2, y2 = regression(30, 1)
    a1, a2 = design(x1), design(x2)
    eye = torch.eye(4, dtype=torch.float64)

    p1 = LAM * eye + a1.T @ a1 / SIGMA**2
    w1 = torch.linalg.solve(p1, a1.T @ y1.squeeze(1) / SIGMA**2)
    p2 = p1 + a2.T @ a2 / SIGMA**2
    w2 = torch.linalg.solve(p2, p1 @ w1 + a2.T @ y2.squeeze(1) / SIGMA**2)

    model = nn.Linear(3, 1).to(torch.float64)
    prior = lk.priors.Isotropic(torch.tensor(LAM, dtype=torch.float64))

    set_params(model, w1)
    post_1 = fit(model, x1, y1, prior)

    set_params(model, w2)  # "training" on D2 under the prior post_1
    post_2 = fit(model, x2, y2, post_1)

    return post_1, post_2, (w1, w2), (p1, p2), (x1, y1, x2, y2)


def test_continual_chain_matches_batch_dense():
    post_1, post_2, (w1, w2), _, (x1, y1, x2, y2) = continual_setup()

    # One fit on D1 and D2 together, in closed form.
    a = design(torch.cat([x1, x2]))
    y = torch.cat([y1, y2]).squeeze(1)
    p = LAM * torch.eye(4, dtype=torch.float64) + a.T @ a / SIGMA**2
    w = torch.linalg.solve(p, a.T @ y / SIGMA**2)

    torch.testing.assert_close(w2, w, rtol=0, atol=1e-10)
    torch.testing.assert_close(flat(post_2.mean), w, rtol=0, atol=1e-10)
    torch.testing.assert_close(dense(post_2.precision()), p, rtol=0, atol=1e-10)

    # The first posterior still sits at its own mean, though the module moved.
    torch.testing.assert_close(flat(post_1.location(post_2.mean)), w1, rtol=0, atol=1e-10)
    assert "hyper_tuner    = None" in repr(post_2)


def test_continual_log_marglik_uses_prior_precision():
    post_1, post_2, (w1, w2), (p1, p2), (_, _, x2, y2) = continual_setup()

    f = design(x2) @ w2
    ll = torch.distributions.Normal(f, SIGMA).log_prob(y2.squeeze(1)).sum()
    log_det_ratio = torch.logdet(p2) - torch.logdet(p1)
    scatter = (w2 - w1) @ p1 @ (w2 - w1)
    want = ll - 0.5 * (log_det_ratio + scatter)

    assert float(scatter) > 0
    torch.testing.assert_close(post_2.prior_scatter(), scatter, rtol=0, atol=1e-8)
    torch.testing.assert_close(post_2.log_marglik(), want, rtol=0, atol=1e-8)


def blobs(n: int = 90):
    g = torch.Generator().manual_seed(0)
    centers = torch.tensor([[-2.0, 0.0], [2.0, 0.0], [0.0, 3.0]])
    y = torch.arange(n) % 3
    x = centers[y] + 0.5 * torch.randn(n, 2, generator=g)

    return x, y


def classifier():
    torch.manual_seed(0)
    x, y = blobs()
    model = nn.Sequential(nn.Linear(2, 8), nn.Tanh(), nn.Linear(8, 3))
    opt = torch.optim.Adam(model.parameters(), lr=0.05)

    for _ in range(100):
        opt.zero_grad()
        nn.functional.cross_entropy(model(x), y).backward()
        opt.step()

    return model, x, y, [(x[:45], y[:45]), (x[45:], y[45:])]


def sgd_step(model, x, y):
    opt = torch.optim.SGD(model.parameters(), lr=0.5)
    opt.zero_grad()
    nn.functional.cross_entropy(model(x) + 1.0 * (y == 0)[:, None], y).backward()
    opt.step()


def test_continual_prior_survives_training():
    model, x, y, data = classifier()
    lik = lk.Categorical()
    post_1 = lk.laplace(model, data, likelihood=lik)
    before = {k: v.clone() for k, v in post_1.mean.items()}

    sgd_step(model, x, y)
    post_2 = lk.laplace(model, data, likelihood=lik, prior=post_1)

    for k, v in post_1.location(post_2.mean).items():
        torch.testing.assert_close(v, before[k], rtol=0, atol=0)
        assert not torch.equal(post_2.mean[k], before[k])

    assert float(post_2.prior_scatter()) > 0

    # post_1's curvature used the old fixed (feature) parameters.
    with pytest.raises(RuntimeError, match=r"fixed parameter '0\.weight' changed"):
        post_1.check_versions()


def test_an_all_params_posterior_predicts_after_training():
    model, x, y, data = classifier()
    post = lk.laplace(
        model, data, likelihood=lk.Categorical(), subset=lk.subsets.all, hessian_approx=lk.DiagGGN()
    )
    probs = lk.predict(post, x[:4]).probs

    sgd_step(model, x, y)

    post.check_versions()
    torch.testing.assert_close(lk.predict(post, x[:4]).probs, probs, rtol=0, atol=0)


def test_continual_kfac_kfac_refused():
    model, _, _, data = classifier()
    lik = lk.Categorical()
    post_1 = lk.laplace(model, data, likelihood=lik, hessian_approx=lk.KFAC(), hyper_tuner=None)

    with pytest.raises(lk.StructureError, match=r"KronEigenbasis \+ KronEigenbasis.*Diag"):
        lk.laplace(model, data, likelihood=lik, hessian_approx=lk.KFAC(), prior=post_1)


def test_continual_diag_diag_is_the_way_out():
    model, _, _, data = classifier()
    lik = lk.Categorical()
    post_1 = lk.laplace(model, data, likelihood=lik, hessian_approx=lk.DiagGGN())
    post_2 = lk.laplace(model, data, likelihood=lik, hessian_approx=lk.DiagGGN(), prior=post_1)

    assert math.isfinite(float(post_2.log_marglik()))


def test_front_door_mean_is_a_copy():
    model, x, y, data = classifier()
    post = lk.laplace(model, data, likelihood=lk.Categorical(), hyper_tuner=None)
    live = dict(model.named_parameters())

    assert set(post.mean) == {"2.weight", "2.bias"}

    for k, v in post.mean.items():
        assert v.untyped_storage().data_ptr() != live[k].untyped_storage().data_ptr()
        assert not v.requires_grad
        torch.testing.assert_close(v, live[k].detach(), rtol=0, atol=0)

    probs = lk.predict(post, x[:4]).probs

    with torch.no_grad():
        model[2].weight.mul_(2.0)  # a posterior parameter: the posterior keeps its own mean

    torch.testing.assert_close(lk.predict(post, x[:4]).probs, probs, rtol=0, atol=0)


def test_by_hand_posterior_keeps_the_callers_tensors():
    model, x, y, _ = classifier()
    fn = lk.core.ModelFn.from_module(model)
    params = {k: v.detach() for k, v in model.named_parameters() if k.startswith("2.")}
    caps = lk.core.probe(fn, lk.Categorical(), lk.Batch(x, y))
    fit = lk.core.fit_curvature(fn, lk.Categorical(), params, lk.DenseGGN(), capabilities=caps)
    stat = fit.update(fit.init(), lk.Batch(x, y))
    prior = lk.priors.Isotropic(torch.tensor(1.0))
    post = fit.posterior(fit.finalize(stat), prior=prior, n_data=stat.count)

    assert all(post.mean[k] is params[k] for k in params)

    with torch.no_grad():
        model[2].weight.mul_(2.0)  # the mean is a view of it, so the mean moved too

    with pytest.raises(RuntimeError, match=r"posterior parameter '2\.weight'.*shares"):
        lk.predict(post, x[:2])


def test_snapshot_owns_its_mean_and_stays_differentiable():
    model, x, y, _ = classifier()
    fn = lk.core.ModelFn.from_module(model)
    w = model[2].weight.detach().clone().requires_grad_(True)
    params = {"2.weight": w, "2.bias": model[2].bias.detach()}
    caps = lk.core.probe(fn, lk.Categorical(), lk.Batch(x, y))
    fit = lk.core.fit_curvature(fn, lk.Categorical(), params, lk.DiagGGN(), capabilities=caps)
    stat = fit.update(fit.init(), lk.Batch(x, y))
    prior = lk.priors.Isotropic(torch.tensor(1.0))
    post = fit.posterior(fit.finalize(stat), prior=prior, n_data=stat.count).snapshot()

    assert post.mean["2.weight"] is not w

    with torch.no_grad():
        w.add_(1.0)

    post.check_versions()
    (g,) = torch.autograd.grad(post.mean["2.weight"].sum(), w)
    torch.testing.assert_close(g, torch.ones_like(w))


def test_kfac_fit_with_a_diag_prior_names_the_pair():
    model, _, _, data = classifier()
    lik = lk.Categorical()
    post_1 = lk.laplace(model, data, likelihood=lik, hessian_approx=lk.DiagGGN())

    with pytest.raises(lk.StructureError, match=r"cannot add Diag to KronEigenbasis") as e:
        lk.laplace(model, data, likelihood=lik, hessian_approx=lk.KFAC(), prior=post_1)

    assert "KFAC posterior" not in str(e.value)


def test_snapshot_of_a_stale_posterior_raises():
    model, x, y, _ = classifier()
    fn = lk.core.ModelFn.from_module(model)
    params = {k: v.detach() for k, v in model.named_parameters() if k.startswith("2.")}
    caps = lk.core.probe(fn, lk.Categorical(), lk.Batch(x, y))
    fit = lk.core.fit_curvature(fn, lk.Categorical(), params, lk.DiagGGN(), capabilities=caps)
    stat = fit.update(fit.init(), lk.Batch(x, y))
    prior = lk.priors.Isotropic(torch.tensor(1.0))
    post = fit.posterior(fit.finalize(stat), prior=prior, n_data=stat.count)

    with torch.no_grad():
        model[0].weight.add_(1.0)  # a fixed parameter: the curvature is stale

    with pytest.raises(RuntimeError, match=r"fixed parameter '0\.weight' changed"):
        post.snapshot()


def test_snapshot_keeps_the_fixed_counters():
    model, x, y, _ = classifier()
    post = lk.laplace(model, [(x, y)], likelihood=lk.Categorical(), hyper_tuner=None)
    snap = post.snapshot()

    assert {k: snap.versions[k] for k in post.versions if k not in post.mean} == {
        k: v for k, v in post.versions.items() if k not in post.mean
    }
    assert all(snap.versions[k] == snap.mean[k]._version for k in snap.mean)
