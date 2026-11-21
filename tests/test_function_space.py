"""Function-space Laplace with a subset of data, against dense weight-space references."""

import copy
import dataclasses

import pytest
import torch
from torch import nn
from torch.func import functional_call, jacrev

import laplacekit as lk
from laplacekit import function_space as fs
from laplacekit.core import Dense, Done, FitContext, Layout, chain

LAM = 2.0


def blobs(n: int, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    centers = torch.tensor([[-2.0, 0.0], [2.0, 0.0], [0.0, 3.0]], dtype=torch.float64)
    y = torch.arange(n) % 3
    x = centers[y] + 0.7 * torch.randn(n, 2, generator=g, dtype=torch.float64)

    return x, y


def waves(n: int, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 2, generator=g, dtype=torch.float64)
    y = torch.stack([x[:, 0].sin(), x[:, 1].cos()], dim=1)

    return x, y + 0.1 * torch.randn(n, 2, generator=g, dtype=torch.float64)


def trained(x, y, out: int, loss) -> nn.Module:
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(2, 5), nn.Tanh(), nn.Linear(5, out)).double()
    opt = torch.optim.Adam(model.parameters(), lr=0.05)

    for _ in range(100):
        opt.zero_grad()
        loss(model(x), y).backward()
        opt.step()

    return model


def classifier(n: int = 24):
    x, y = blobs(n)

    return trained(x, y, 3, nn.functional.cross_entropy), x, y


def loader(x, y, k: int = 3):
    n = x.shape[0] // k

    return [(x[i * n : (i + 1) * n], y[i * n : (i + 1) * n]) for i in range(k)]


def isotropic():
    return lk.priors.Isotropic(torch.tensor(LAM, dtype=torch.float64))


def x_test():
    return torch.tensor([[0.0, 1.0], [3.0, -1.0], [-4.0, 4.0], [0.5, 0.5]], dtype=torch.float64)


def jacobian(model: nn.Module, x) -> torch.Tensor:
    """[N, E, P] by `jacrev`, over all parameters."""

    params = {k: v.detach() for k, v in model.named_parameters()}

    def f(p):
        return functional_call(model, p, (x,))

    jac = jacrev(f)(params)
    n, e = model(x).shape

    return torch.cat([j.reshape(n, e, -1) for j in jac.values()], dim=2)


def hessian(lik, f: torch.Tensor) -> torch.Tensor:
    """The output Hessian of -log p(y | f), [N, E, E], by hand."""

    if isinstance(lik, lk.Categorical):
        p = f.softmax(-1)

        return torch.diag_embed(p) - p.unsqueeze(2) * p.unsqueeze(1)

    eye = torch.eye(f.shape[1], dtype=f.dtype)

    return eye.expand(f.shape[0], -1, -1) / lik.sigma**2


def dense_ggn_outputs(model, lik, x, xs) -> torch.Tensor:
    """J* (λ I + Σₙ Jₙᵀ Λₙ Jₙ)⁻¹ J*ᵀ per test input: the full dense GGN posterior."""

    with torch.no_grad():
        f = model(x)

    j = jacobian(model, x)
    prec = LAM * torch.eye(j.shape[2], dtype=torch.float64)
    prec = prec + torch.einsum("nep,nef,nfq->pq", j, hessian(lik, f), j)
    js = jacobian(model, xs)

    return js @ torch.linalg.solve(prec, js.mT)


@pytest.mark.parametrize("kind", ["categorical", "gaussian"])
def test_subset_gp_equals_full_dense_ggn(kind):
    if kind == "categorical":
        model, x, y = classifier()
        lik = lk.Categorical()  # Λ is singular: p sums to one
    else:
        x, y = waves(24)
        model = trained(x, y, 2, nn.functional.mse_loss)
        lik = copy.replace(
            lk.Gaussian(), sigma=torch.as_tensor(torch.tensor(0.7, dtype=torch.float64))
        )

    gp = fs.laplace(
        model,
        loader(x, y),
        likelihood=lik,
        method=fs.SubsetOfData(x.shape[0]),
        subset=lk.subsets.all,
        prior=isotropic(),
    )
    out = gp.at(x_test())
    want = dense_ggn_outputs(model, lik, x, x_test())

    torch.testing.assert_close(out.cov.per_input(), want, rtol=0, atol=1e-10)
    torch.testing.assert_close(out.mean, model(x_test()))

    if kind == "gaussian":
        # sigma is applied late: a new one changes the covariance without a refit.
        other = copy.replace(
            lk.Gaussian(), sigma=torch.as_tensor(torch.tensor(0.3, dtype=torch.float64))
        )
        late = copy.replace(gp, likelihood=other).at(x_test())
        want = dense_ggn_outputs(model, other, x, x_test())

        torch.testing.assert_close(late.cov.per_input(), want, rtol=0, atol=1e-10)


def test_subset_gp_vector_sigma():
    """m = N with a per-output sigma: the GP posterior K** - K*n (Knn + diag sigma²)⁻¹ Kn*,
    with K = J Σ₀ Jᵀ from `jacrev` and Σ₀ = I / λ."""

    x, y = waves(24)
    model = trained(x, y, 2, nn.functional.mse_loss)
    sigma = torch.tensor([0.2, 0.6], dtype=torch.float64)
    lik = copy.replace(lk.Gaussian(), sigma=torch.as_tensor(sigma))

    gp = fs.laplace(
        model,
        loader(x, y),
        likelihood=lik,
        method=fs.SubsetOfData(x.shape[0]),
        subset=lk.subsets.all,
        prior=isotropic(),
    )
    out = gp.at(x_test())

    jn = jacobian(model, x).reshape(-1, jacobian(model, x).shape[2])
    js = jacobian(model, x_test())
    k_nn = jn @ jn.T / LAM
    k_sn = torch.einsum("sep,np->sen", js, jn) / LAM
    k_ss = js @ js.mT / LAM
    noise = torch.diag((sigma**2).repeat(x.shape[0]))
    want = k_ss - k_sn @ torch.linalg.solve(k_nn + noise, k_sn.mT)

    torch.testing.assert_close(out.cov.per_input(), want, rtol=0, atol=1e-10)
    torch.testing.assert_close(out.mean, model(x_test()))


def test_subset_gp_equals_last_layer_dense():
    model, x, y = classifier()
    lik = lk.Categorical()
    data = loader(x, y)

    # The last layer is one block, so DenseGGN there is the full dense posterior. Over all
    # parameters DenseGGN is block diagonal (one block per module) and would differ.
    post = lk.laplace(
        model,
        data,
        likelihood=lik,
        hessian_approx=lk.DenseGGN(),
        prior=isotropic(),
        hyper_tuner=None,
    )
    gp = fs.laplace(model, data, likelihood=lik, method=fs.SubsetOfData(24), prior=isotropic())

    want = lk.predict_outputs(post, x_test()).cov.per_input()
    torch.testing.assert_close(gp.at(x_test()).cov.per_input(), want, rtol=0, atol=1e-10)

    assert gp.n_subset == 24
    assert set(gp.mean) == {"2.weight", "2.bias"}
    assert "n_subset=24" in repr(gp)

    # The posterior owns its mean: training the module afterwards leaves it in place.
    assert gp.mean["2.bias"].data_ptr() != model[2].bias.data_ptr()
    before = gp.at(x_test()).cov.per_input()

    with torch.no_grad():
        model[2].bias.add_(1.0)

    torch.testing.assert_close(gp.mean["2.bias"], model[2].bias - 1.0)
    torch.testing.assert_close(gp.at(x_test()).cov.per_input(), before)


def test_weight_posterior_as_prior():
    model, x, y = classifier(36)
    lik = lk.Categorical()
    d1, d2 = loader(x[:18], y[:18]), loader(x[18:], y[18:])

    post = lk.laplace(
        model, d1, likelihood=lik, hessian_approx=lk.DenseGGN(), prior=isotropic(), hyper_tuner=None
    )
    continual = lk.laplace(
        model, d2, likelihood=lik, hessian_approx=lk.DenseGGN(), prior=post, hyper_tuner=None
    )
    gp = fs.laplace(model, d2, likelihood=lik, method=fs.SubsetOfData(18), prior=post)

    want = lk.predict_outputs(continual, x_test()).cov.per_input()
    torch.testing.assert_close(gp.at(x_test()).cov.per_input(), want, rtol=0, atol=1e-10)


@dataclasses.dataclass(frozen=True)
class _SubsetGGN:
    """The subset pass of `SubsetOfData`, read out as the GGN of the chosen examples, so
    `check_curvature` can run its laws on the statistic."""

    approx: fs.SubsetOfData

    @property
    def requires(self):
        return self.approx.requires

    def plan(self, params, ctx):
        first = self.approx.plan(params, ctx, isotropic())

        def then(picked):
            done = first.then(picked)
            assert isinstance(done, Done)
            gp = done.result
            lj = torch.einsum("mfe,mfp->mep", gp.root, gp.jac.reshape(*gp.root.shape[:2], -1))
            lj = lj.reshape(-1, lj.shape[-1])

            return Done(Dense(lj.mT @ lj))

        return chain(first.curvature, then=then)


def test_subset_statistic_monoid():
    model, x, y = classifier()
    params = {k: v.detach() for k, v in model.named_parameters()}
    batches = [lk.Batch(a, b) for a, b in loader(x, y, k=4)]

    lk.testing.check_curvature(
        _SubsetGGN(fs.SubsetOfData(7)), model, lk.Categorical(), params, batches
    )


def _subset_jac(m: int, generator: torch.Generator | None):
    model, x, y = classifier()
    gp = fs.laplace(
        model,
        loader(x, y),
        likelihood=lk.Categorical(),
        method=fs.SubsetOfData(m),
        prior=isotropic(),
        generator=generator,
    )

    return gp


def test_subset_choice():
    everything = _subset_jac(100, None)
    assert everything.n_subset == 24
    assert everything.jac.shape == (24 * 3, 18)

    some = _subset_jac(7, torch.Generator().manual_seed(3))
    again = _subset_jac(7, torch.Generator().manual_seed(3))
    other = _subset_jac(7, torch.Generator().manual_seed(4))

    assert some.n_subset == 7
    assert some.jac.shape == (7 * 3, 18)
    torch.testing.assert_close(some.jac, again.jac, rtol=0, atol=0)
    assert not torch.allclose(some.jac, other.jac)

    # A subset is wider than the full posterior: it sees less data.
    full = everything.at(x_test()).cov.diagonal().sum()
    assert some.at(x_test()).cov.diagonal().sum() > full

    with pytest.raises(ValueError, match="n_subset >= 1"):
        fs.SubsetOfData(0)


class _Counting:
    def __init__(self, items):
        self.items = items
        self.pulled = 0

    def __iter__(self):
        for it in self.items:
            self.pulled += 1
            yield it


def test_fs_laplace_refuses_before_data(monkeypatch):
    model, x, y = classifier()

    def log_prob(f, y):
        return -((f[:, 0] - y) ** 2)

    ld = _Counting(loader(x, y))

    with pytest.raises(lk.CapabilityError, match="output Hessian"):
        fs.laplace(
            model, ld, likelihood=lk.likelihoods.from_log_prob(log_prob), method=fs.SubsetOfData(4)
        )

    assert ld.pulled == 0

    # m * E: 6000 * 3 output entries.
    ld = _Counting(loader(x, y))

    with pytest.raises(lk.StructureError, match=r"m=6000.*E=3.*P=18.*MAX_SUBSET_ENTRIES"):
        fs.laplace(model, ld, likelihood=lk.Categorical(), method=fs.SubsetOfData(6000))

    assert ld.pulled == 1

    # m * E * P: 10 * 3 * 18 Jacobian entries, above a lowered budget.
    monkeypatch.setattr(fs, "MAX_SUBSET_JACOBIAN_ENTRIES", 500)
    ld = _Counting(loader(x, y))

    with pytest.raises(lk.StructureError, match=r"m=10.*E=3.*P=18.*MAX_SUBSET_JACOBIAN_ENTRIES"):
        fs.laplace(model, ld, likelihood=lk.Categorical(), method=fs.SubsetOfData(10))

    assert ld.pulled == 1

    # A prior over other parameters.
    post = lk.laplace(
        model, loader(x, y), likelihood=lk.Categorical(), subset=lk.subsets.all, hyper_tuner=None
    )
    ld = _Counting(loader(x, y))

    with pytest.raises(lk.StructureError, match="prior mean has keys"):
        fs.laplace(model, ld, likelihood=lk.Categorical(), method=fs.SubsetOfData(4), prior=post)

    assert ld.pulled == 1


def test_subset_refuses_weights():
    model, x, y = classifier()
    batches = [lk.Batch(a, b, torch.ones(a.shape[0], dtype=torch.float64)) for a, b in loader(x, y)]

    with pytest.raises(lk.CapabilityError, match="weights"):
        fs.laplace(model, batches, likelihood=lk.Categorical(), method=fs.SubsetOfData(4))


def test_at_checks_capabilities_first():
    model, x, y = classifier()
    gp = fs.laplace(model, loader(x, y), likelihood=lk.Categorical(), method=fs.SubsetOfData(4))
    blind = copy.replace(gp, caps=lk.core.Capabilities())

    with pytest.raises(lk.CapabilityError, match="vjp"):
        blind.at(x_test())


def _subset_pass(m: int, lik=None):
    model, x, y = classifier()
    lik = lik if lik is not None else lk.Categorical()
    params = {k: v.detach() for k, v in model.named_parameters()}
    fn = lk.core.ModelFn.from_module(model)
    ctx = FitContext(fn, lik, Layout.by_module(params), lk.core.probe(fn, lik, lk.Batch(x, y)))

    return fs.SubsetOfData(m).plan(params, ctx, isotropic()).curvature, params, ctx, x, y


@dataclasses.dataclass(frozen=True)
class _Spying:
    """Categorical, recording the batch size of each output Hessian it builds."""

    sizes: list

    def log_prob(self, f, y):
        return lk.Categorical().log_prob(f, y)

    def output_hessian(self, f):
        self.sizes.append(f.shape[0])

        return lk.Categorical().output_hessian(f)


def test_subset_roots_only_for_kept_rows():
    sizes: list[int] = []
    curv, params, ctx, x, y = _subset_pass(3, _Spying(sizes))
    stat = curv.init(params, ctx)

    for i, (a, b) in enumerate(loader(x, y, k=3)):  # batches of 8
        stat = curv.update(stat, params, lk.Batch(a, b), torch.Generator().manual_seed(i), ctx)

    assert sizes == [3, 3, 3]

    sizes.clear()
    curv, params, ctx, x, y = _subset_pass(20, _Spying(sizes))
    curv.update(
        curv.init(params, ctx),
        params,
        lk.Batch(x[:8], y[:8]),
        torch.Generator().manual_seed(0),
        ctx,
    )

    assert sizes == [8]


def test_subset_ranks_draw_differently(monkeypatch):
    """The fit seeds each batch's generator with the rank, so equal shards draw differently."""

    curv, params, ctx, x, y = _subset_pass(1)
    batch = lk.Batch(x[:8], y[:8])

    def picked(rank):
        monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
        monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
        monkeypatch.setattr(torch.distributed, "get_rank", lambda group=None: rank)
        fit = lk.core.fit_curvature(
            ctx.model, ctx.likelihood, params, curv, generator=torch.Generator().manual_seed(0)
        )
        stat = fit.update(fit.init(), batch).curvature

        return stat.rows[0].priority

    assert not torch.equal(picked(0), picked(1))


def test_subset_is_uniform():
    curv, params, ctx, x, y = _subset_pass(3)
    batches = [lk.Batch(a, b) for a, b in loader(x[:12], y[:12], k=3)]  # 12 examples
    hits = torch.zeros(12)
    trials = 400

    for t in range(trials):
        stat = curv.init(params, ctx)

        for i, b in enumerate(batches):
            stat = curv.update(stat, params, b, torch.Generator().manual_seed(1000 * t + i), ctx)

        chosen = stat.rows[0].inputs
        hits += (chosen.unsqueeze(1) == x[:12].unsqueeze(0)).all(-1).any(0).float()

    # Each example is kept with probability m / N = 1/4: within 4 standard errors.
    freq = hits / trials
    assert (freq - 0.25).abs().max() < 4 * (0.25 * 0.75 / trials) ** 0.5


# --- function posteriors in outputs, predict and tune --------------------------------------


def _gp_and_post():
    """A subset GP on every example and the dense weight posterior it equals (last layer)."""

    model, x, y = classifier()
    lik = lk.Categorical()
    data = loader(x, y)
    post = lk.laplace(
        model,
        data,
        likelihood=lik,
        hessian_approx=lk.DenseGGN(),
        prior=isotropic(),
        hyper_tuner=None,
    )
    gp = fs.laplace(model, data, likelihood=lik, method=fs.SubsetOfData(24), prior=isotropic())

    return gp, post, model, x, y


def test_outputs_of_a_function_posterior():
    gp, post, model, _, _ = _gp_and_post()
    out = lk.predict_outputs(gp, x_test())
    want = lk.predict_outputs(post, x_test())

    torch.testing.assert_close(out.mean, want.mean)
    torch.testing.assert_close(out.cov.per_input(), want.cov.per_input(), rtol=0, atol=1e-10)
    torch.testing.assert_close(lk.core.mean_outputs(gp, x_test()), model(x_test()))

    for kwargs in (
        {"strategy": "reverse-mode"},
        {"n_samples": "exact"},
        {"n_samples": 8},
        {"chunk_size": 4},
    ):
        with pytest.raises(ValueError, match="function-space"):
            lk.predict_outputs(gp, x_test(), **kwargs)


@pytest.mark.parametrize("link", ["probit", "mc"])
def test_predict_function_posterior_matches_outputs(link):
    gp, post, _, _, _ = _gp_and_post()
    method = lk.Linearized() if link == "probit" else lk.Linearized(lk.MC(n_samples=50))

    got = lk.predict(gp, x_test(), method, generator=torch.Generator().manual_seed(0))
    want = lk.predict(post, x_test(), method, generator=torch.Generator().manual_seed(0))

    torch.testing.assert_close(got.logits, want.logits, rtol=0, atol=1e-8)

    if link == "probit":
        # The default link of a Categorical is Probit on the exact covariance.
        torch.testing.assert_close(lk.predict(gp, x_test()).logits, want.logits, atol=1e-8, rtol=0)

    for bad in (
        lk.Linearized(strategy="reverse-mode"),
        lk.Linearized(n_samples="exact"),
        lk.Linearized(chunk_size=4),
    ):
        with pytest.raises(ValueError, match="function-space"):
            lk.predict(gp, x_test(), bad)


def test_predict_select_on_function_posterior():
    gp, post, _, _, _ = _gp_and_post()

    def first_two(f, x):
        return f[:, :2]

    got = lk.predict(gp, x_test(), select=first_two)
    want = lk.predict(post, x_test(), select=first_two)

    assert got.logits.shape == (4, 2)
    torch.testing.assert_close(got.logits, want.logits, rtol=0, atol=1e-8)


def test_weight_samples_refuses_function_posterior():
    gp, _, _, _, _ = _gp_and_post()

    with pytest.raises(TypeError, match="function-space"):
        lk.predict(gp, x_test(), lk.WeightSamples())


def test_tune_function_posterior_is_honest():
    gp, post, _, _, _ = _gp_and_post()
    xv, yv = blobs(30, seed=1)
    nll = lk.objectives.predictive_nll([lk.Batch(xv, yv)])

    tuned_gp = lk.tune(gp, lk.Minimize(nll))
    tuned_post = lk.tune(post, lk.Minimize(nll))

    # Tuning moved the precision, and both views land on the same one.
    assert not torch.isclose(tuned_gp.prior.prior_precision, gp.prior.prior_precision)
    torch.testing.assert_close(
        tuned_gp.prior.prior_precision, tuned_post.prior.prior_precision, rtol=1e-5, atol=0
    )

    # At the same precision, the posteriors agree, and the GP needs no refit.
    same = copy.replace(post, prior=tuned_gp.prior)
    want = lk.predict_outputs(same, x_test()).cov.per_input()
    torch.testing.assert_close(
        lk.predict_outputs(tuned_gp, x_test()).cov.per_input(), want, atol=1e-10, rtol=0
    )
    assert gp.prior.prior_precision == LAM


def test_scenario_6_runs():
    model, x, y = classifier()
    data = loader(x, y)
    lik = lk.Categorical()
    xv, yv = blobs(12, seed=2)
    xs = x_test()

    gp = fs.laplace(
        model,
        data,
        likelihood=lik,
        method=fs.SubsetOfData(n_subset=10),
        prior=lk.priors.Isotropic(torch.tensor(1.0, dtype=torch.float64)),
    )
    pred = lk.predict(gp, xs, lk.Linearized())
    out = lk.predict_outputs(gp, xs)
    gp = lk.tune(gp, lk.Minimize(lk.objectives.predictive_nll([lk.Batch(xv, yv)])))

    assert pred.probs.shape == (4, 3)
    assert out.cov.per_input().shape == (4, 3, 3)
    assert gp.prior.prior_precision > 0

    post = lk.laplace(model, data, likelihood=lik)
    gp2 = fs.laplace(model, data, likelihood=lik, method=fs.SubsetOfData(n_subset=5), prior=post)

    assert lk.predict(gp2, xs).probs.shape == (4, 3)


# --- staleness and ownership --------------------------------------------------------------


def test_at_raises_after_a_fixed_parameter_moves():
    gp, _, model, _, _ = _gp_and_post()
    lk.predict_outputs(gp, x_test())

    with torch.no_grad():
        model[0].weight.add_(1.0)

    with pytest.raises(RuntimeError, match=r"fixed parameter '0\.weight' changed in place"):
        gp.at(x_test())

    with pytest.raises(RuntimeError, match="fixed parameter"):
        lk.predict(gp, x_test())


def test_at_survives_a_posterior_parameter_moving():
    gp, _, model, _, _ = _gp_and_post()
    before = gp.at(x_test())

    # The mean is the posterior's own copy, so training the last layer leaves it in place.
    with torch.no_grad():
        model[2].weight.add_(1.0)

    after = gp.at(x_test())

    torch.testing.assert_close(after.mean, before.mean)
    torch.testing.assert_close(after.cov.per_input(), before.cov.per_input())


def test_staleness_guard_survives_replace_and_tune():
    gp, _, model, _, _ = _gp_and_post()
    xv, yv = blobs(30, seed=1)
    nll = lk.objectives.predictive_nll([lk.Batch(xv, yv)])

    replaced = copy.replace(gp, prior=lk.priors.Isotropic(torch.tensor(5.0, dtype=torch.float64)))
    tuned = lk.tune(gp, lk.Minimize(nll))

    with torch.no_grad():
        model[0].bias.add_(1.0)

    for posterior in (replaced, tuned):
        with pytest.raises(RuntimeError, match=r"fixed parameter '0\.bias'"):
            posterior.at(x_test())


def test_fs_laplace_keeps_the_prior_leaves():
    model, x, y = classifier()
    prec = torch.tensor(LAM, dtype=torch.float64, requires_grad=True)
    prior = lk.priors.Isotropic(prec)

    gp = fs.laplace(
        model, loader(x, y), likelihood=lk.Categorical(), method=fs.SubsetOfData(24), prior=prior
    )

    # As in `lk.laplace`, the prior comes back as given, not detached.
    assert gp.prior.prior_precision is prec
    assert gp.prior.prior_precision.requires_grad


def _picks_last_entries(model, caps):
    module = model.module
    assert module is not None
    g = torch.Generator().manual_seed(3)
    w = torch.rand(module[2].weight.shape, generator=g) < 0.4
    w[0, 0] = True

    return lk.core.Selection.from_masks(dict(module.named_parameters()), {"2.weight": w})


def test_fs_laplace_accepts_selection():
    model, x, y = classifier()
    lik = lk.Categorical()
    data = loader(x, y)

    # One block of selected entries: the subset GP on all the data equals the dense posterior.
    post = lk.laplace(
        model,
        data,
        likelihood=lik,
        subset=_picks_last_entries,
        hessian_approx=lk.DenseGGN(),
        prior=isotropic(),
        hyper_tuner=None,
    )
    gp = fs.laplace(
        model,
        data,
        likelihood=lik,
        method=fs.SubsetOfData(24),
        subset=_picks_last_entries,
        prior=isotropic(),
    )

    assert list(gp.mean) == ["2.weight[idx]"]

    want = lk.predict_outputs(post, x_test()).cov.per_input()
    torch.testing.assert_close(gp.at(x_test()).cov.per_input(), want, rtol=0, atol=1e-10)

    # Training the module afterwards makes `at` raise.
    with torch.no_grad():
        model[2].weight.add_(1.0)

    with pytest.raises(RuntimeError, match="changed in place"):
        gp.at(x_test())


# --- structures and the subset evidence ---------------------------------------------------


def _flat(params) -> torch.Tensor:
    return torch.cat([v.reshape(-1) for v in params.values()])


def _fit_all(model, x, y, lik, structure="joint", prior=None, k: int = 3):
    return fs.laplace(
        model,
        loader(x, y, k),
        likelihood=lik,
        method=fs.SubsetOfData(x.shape[0], structure=structure),
        subset=lk.subsets.all,
        prior=prior if prior is not None else isotropic(),
        generator=torch.Generator().manual_seed(0),  # the same row order in every fit
    )


def _shifted_prior(model) -> lk.priors.Isotropic:
    """Isotropic(LAM) centred away from zero, so r = J (θ̂ - μ₀) tests the location."""

    g = torch.Generator().manual_seed(5)
    mean = {
        k: 0.3 * torch.randn(v.shape, generator=g, dtype=torch.float64)
        for k, v in model.named_parameters()
    }

    return lk.priors.Isotropic(torch.tensor(LAM, dtype=torch.float64), mean)


def _dense_gp_terms(model, x, prior):
    """f, r = J (θ̂ - μ₀) and K = J Jᵀ / λ, flattened over [N E], from `jacrev`."""

    j = jacobian(model, x)
    n, e, p = j.shape
    theta = _flat(dict(model.named_parameters())).detach()
    mu0 = _flat(prior.mean) if prior.mean is not None else torch.zeros_like(theta)

    with torch.no_grad():
        f = model(x)

    r = j.reshape(n * e, p) @ (theta - mu0)
    k = j.reshape(n * e, p) @ j.reshape(n * e, p).T / prior.prior_precision

    return f, r.reshape(n, e), k.reshape(n, e, n, e)


def _output_blocks(k: torch.Tensor, c: int) -> torch.Tensor:
    return k[:, c, :, c]


def test_sod_refuses_unknown_structure():
    with pytest.raises(ValueError, match='"joint", "diagonal" or "independent", got \'foo\''):
        fs.SubsetOfData(20, structure="foo")  # ty: ignore[invalid-argument-type]


@pytest.mark.parametrize("structure", ["joint", "independent"])
@pytest.mark.parametrize("sigma", [[0.7], [0.2, 0.6]])
def test_sod_evidence_equals_dense_gaussian(structure, sigma):
    """The exact GP evidence: N(y; f - r, K + σ² I), one GP per output when independent."""

    x, y = waves(24)
    model = trained(x, y, 2, nn.functional.mse_loss)
    sig = torch.tensor(sigma, dtype=torch.float64)
    lik = copy.replace(
        lk.Gaussian(), sigma=torch.as_tensor(sig if sig.numel() > 1 else sig.reshape(()))
    )
    prior = _shifted_prior(model)

    gp = _fit_all(model, x, y, lik, structure, prior)
    want = _dense_gaussian_evidence(model, x, y, sig, prior, structure)

    torch.testing.assert_close(gp.log_marglik(), want, rtol=1e-10, atol=0)


def _dense_gaussian_evidence(model, x, y, sig, prior, structure):
    """log N(y; f - r, K + σ² I) on the rows `x`, summed over outputs when independent."""

    f, r, k = _dense_gp_terms(model, x, prior)
    n, e = f.shape
    mean = f - r
    var = (sig**2).expand(e)

    if structure != "independent":
        cov = k.reshape(n * e, n * e) + torch.diag(var.repeat(n))

        return torch.distributions.MultivariateNormal(mean.reshape(-1), cov).log_prob(y.reshape(-1))

    return sum(
        torch.distributions.MultivariateNormal(
            mean[:, c], _output_blocks(k, c) + var[c] * torch.eye(n, dtype=torch.float64)
        ).log_prob(y[:, c])
        for c in range(e)
    )


@pytest.mark.parametrize("structure", ["joint", "diagonal", "independent"])
def test_sod_evidence_equals_dense_categorical(structure):
    """R&W 3.44: Σ log p(y | f) - ½ (rᵀ (K + 1e-5 I)⁻¹ r + log|I + K H|), with H the output
    Hessian of the structure (log|I + Wᵀ K W| = log|I + K W Wᵀ|)."""

    model, x, y = classifier()
    prior = _shifted_prior(model)
    gp = _fit_all(model, x, y, lk.Categorical(), structure, prior)
    want = _dense_categorical_evidence(model, x, y, prior, structure)

    torch.testing.assert_close(gp.log_marglik(), want, rtol=1e-10, atol=0)


def _dense_categorical_evidence(model, x, y, prior, structure):
    """R&W 3.44 on the rows `x`, from `jacrev` and the softmax Hessian by hand."""

    f, r, k = _dense_gp_terms(model, x, prior)
    n, e = f.shape
    h = hessian(lk.Categorical(), f)
    eye = torch.eye(e, dtype=torch.float64)

    if structure != "joint":
        h = h * eye

    if structure == "independent":
        k = k * eye.reshape(1, e, 1, e)

    k = k.reshape(n * e, n * e)
    ident = torch.eye(n * e, dtype=k.dtype)
    log_lik = torch.distributions.Categorical(logits=f).log_prob(y).sum()
    scatter = r.reshape(-1) @ torch.linalg.solve(k + 1e-5 * ident, r.reshape(-1))
    logdet = torch.logdet(ident + k @ torch.block_diag(*h))

    return log_lik - 0.5 * (scatter + logdet)


def _observed(inputs) -> torch.Tensor:
    """The rows a masked likelihood counts: a fixed function of the inputs."""

    return inputs[:, 0] > -0.3


@lk.core.pytree
class _MaskedGaussian(lk.Gaussian):
    """A Gaussian subclass with a mask, shaped like its per-entry `log_prob`."""

    def mask(self, f, inputs):
        return _observed(inputs).unsqueeze(1).expand_as(f)


@pytest.mark.parametrize("structure", ["joint", "diagonal", "independent"])
@pytest.mark.parametrize("kind", ["gaussian", "categorical"])
def test_masked_evidence_equals_dense_on_observed_rows(kind, structure):
    """Masked rows drop out of every term: the evidence is that of the observed rows."""

    if kind == "gaussian":
        x, y = waves(24)
        model = trained(x, y, 2, nn.functional.mse_loss)
        sig = torch.tensor(0.4, dtype=torch.float64)
        lik = _MaskedGaussian(sig, learned=True)
    else:
        model, x, y = classifier()
        lik = lk.Categorical(mask_fn=lambda f, inputs: _observed(inputs))

    keep = _observed(x)
    assert 0 < int(keep.sum()) < x.shape[0]

    prior = _shifted_prior(model)

    if kind == "gaussian":
        # The fit refuses a late likelihood with a mask (its data fit takes no inputs), so
        # the masked Gaussian comes in afterwards, as a tuned sigma would.
        gp = _fit_all(
            model, x, y, copy.replace(lk.Gaussian(), sigma=torch.as_tensor(sig)), structure, prior
        )
        gp = copy.replace(gp, likelihood=lik)
    else:
        gp = _fit_all(model, x, y, lik, structure, prior)

    if kind == "gaussian":
        want = _dense_gaussian_evidence(model, x[keep], y[keep], sig, prior, structure)
    else:
        want = _dense_categorical_evidence(model, x[keep], y[keep], prior, structure)

    torch.testing.assert_close(gp.log_marglik(), want, rtol=1e-10, atol=0)


def test_at_masks_a_swapped_in_gaussian_subclass():
    """A masked `Gaussian` subclass with a late sigma: the fit runs on its `unit()`, and
    `fs.laplace` swaps the real likelihood back in. The roots keep the mask, so `at` is the
    dense GGN posterior of the observed rows only. Without targets the fit takes no data
    fit, so it accepts the late likelihood with a mask."""

    x, y = waves(24)
    model = trained(x, y, 2, nn.functional.mse_loss)
    sig = torch.tensor(0.4, dtype=torch.float64)
    keep = _observed(x)

    assert 0 < int(keep.sum()) < x.shape[0]

    gp = fs.laplace(
        model,
        [lk.Batch(x[i : i + 8]) for i in range(0, 24, 8)],
        likelihood=_MaskedGaussian(sig, learned=True),
        method=fs.SubsetOfData(x.shape[0]),
        subset=lk.subsets.all,
        prior=isotropic(),
    )

    assert type(gp.likelihood) is _MaskedGaussian

    out = gp.at(x_test())
    want = dense_ggn_outputs(
        model, copy.replace(lk.Gaussian(), sigma=torch.as_tensor(sig)), x[keep], x_test()
    )

    torch.testing.assert_close(out.cov.per_input(), want, rtol=0, atol=1e-10)


@pytest.mark.parametrize("structure", ["joint", "diagonal"])
def test_at_masks_after_a_post_fit_swap(structure):
    """M7c's scenario: a fit with a plain `Gaussian`, then `copy.replace` to a masked
    subclass. `at` drops the masked rows, as `log_marglik` does: the dense GGN posterior of
    the observed rows."""

    x, y = waves(24)
    model = trained(x, y, 2, nn.functional.mse_loss)
    sig = torch.tensor(0.4, dtype=torch.float64)
    keep = _observed(x)

    assert 0 < int(keep.sum()) < x.shape[0]

    gp = _fit_all(model, x, y, copy.replace(lk.Gaussian(), sigma=torch.as_tensor(sig)), structure)
    swapped = copy.replace(gp, likelihood=_MaskedGaussian(sig, learned=True))
    out = swapped.at(x_test())

    want = dense_ggn_outputs(
        model, copy.replace(lk.Gaussian(), sigma=torch.as_tensor(sig)), x[keep], x_test()
    )

    torch.testing.assert_close(out.cov.per_input(), want, rtol=0, atol=1e-10)


def test_sod_evidence_tunes_prior():
    """`lk.tune` on the evidence lands within a grid step of the dense argmax over λ."""

    x, y = waves(24)
    model = trained(x, y, 2, nn.functional.mse_loss)
    sigma = torch.tensor(0.3, dtype=torch.float64)
    gp = _fit_all(model, x, y, copy.replace(lk.Gaussian(), sigma=torch.as_tensor(sigma)))

    f, r, k = _dense_gp_terms(model, x, lk.priors.Isotropic(torch.tensor(1.0, dtype=torch.float64)))
    mean = (f - r).reshape(-1)
    noise = sigma**2 * torch.eye(48, dtype=torch.float64)
    grid = torch.logspace(-3, 3, 200, dtype=torch.float64)
    scores = torch.stack(
        [
            torch.distributions.MultivariateNormal(mean, k.reshape(48, 48) / lam + noise).log_prob(
                y.reshape(-1)
            )
            for lam in grid
        ]
    )
    best = int(scores.argmax())
    step = float(grid[1].log() - grid[0].log())

    tuned = lk.tune(gp, lk.Minimize(lambda g: -g.log_marglik()))

    assert 0 < best < len(grid) - 1
    assert abs(float(tuned.prior.prior_precision.log() - grid[best].log())) <= step


def test_diagonal_equals_joint_for_gaussian():
    x, y = waves(24)
    model = trained(x, y, 2, nn.functional.mse_loss)
    lik = copy.replace(
        lk.Gaussian(), sigma=torch.as_tensor(torch.tensor([0.2, 0.6], dtype=torch.float64))
    )

    joint = _fit_all(model, x, y, lik, "joint")
    diagonal = _fit_all(model, x, y, lik, "diagonal")

    assert torch.equal(joint.root, diagonal.root)
    assert torch.equal(joint.at(x_test()).cov.per_input(), diagonal.at(x_test()).cov.per_input())
    assert torch.equal(joint.log_marglik(), diagonal.log_marglik())


def test_sod_evidence_refuses_subset():
    model, x, y = classifier()
    gp = fs.laplace(model, loader(x, y), likelihood=lk.Categorical(), method=fs.SubsetOfData(7))

    assert gp.n_seen == 24

    with pytest.raises(ValueError, match=r"m < N rows is not the data's evidence.*24.*7"):
        gp.log_marglik()


def test_sod_evidence_needs_targets():
    model, x, y = classifier()
    batches = [lk.Batch(a) for a, _ in loader(x, y)]
    gp = fs.laplace(model, batches, likelihood=lk.Categorical(), method=fs.SubsetOfData(24))

    assert gp.targets is None

    with pytest.raises(ValueError, match="targets"):
        gp.log_marglik()


def test_sod_evidence_refuses_mixed_targets():
    model, x, y = classifier()
    (a, b), (c, d), (e, g) = loader(x, y)
    batches = [lk.Batch(a, b), lk.Batch(c), lk.Batch(e, g)]

    with pytest.raises(ValueError, match="some batches have targets and others don't"):
        fs.laplace(model, batches, likelihood=lk.Categorical(), method=fs.SubsetOfData(24))


def test_sod_evidence_needs_one_tensor():
    """A model whose outputs are not one tensor has no evidence here."""

    model, x, y = classifier()
    full = fs.laplace(model, loader(x, y), likelihood=lk.Categorical(), method=fs.SubsetOfData(24))
    inner = full.model
    dicted = copy.replace(
        full, model=lk.core.ModelFn(lambda p, xs: {"f": inner(p, xs)}, module=inner.module)
    )

    with pytest.raises(lk.StructureError, match="single tensor"):
        dicted.log_marglik()


def test_independent_blocks_are_diagonal():
    """One GP per output: K_c = J_c Jᵀ_c / λ and noise precision diag(H)_c, from `jacrev`."""

    model, x, y = classifier()
    gp = _fit_all(model, x, y, lk.Categorical(), "independent")
    cov = gp.at(x_test()).cov.per_input()

    with torch.no_grad():
        f = model(x)

    lam_n = hessian(lk.Categorical(), f).diagonal(dim1=1, dim2=2)  # [N, E]
    jn, js = jacobian(model, x), jacobian(model, x_test())

    for c in range(3):
        k_nn = jn[:, c] @ jn[:, c].T / LAM
        k_sn = js[:, c] @ jn[:, c].T / LAM
        k_ss = (js[:, c] * js[:, c]).sum(1) / LAM
        want = (
            k_ss
            - (k_sn @ torch.linalg.solve(k_nn + torch.diag(1 / lam_n[:, c]), k_sn.T)).diagonal()
        )

        torch.testing.assert_close(cov[:, c, c], want, rtol=1e-8, atol=1e-12)

    off = cov * (1 - torch.eye(3, dtype=cov.dtype))
    assert torch.equal(off, torch.zeros_like(off))
    assert "structure='independent'" in repr(gp)
    assert "structure='independent'" in repr(fs.SubsetOfData(4, structure="independent"))


def test_joint_default_unchanged():
    model, x, y = classifier()
    data = loader(x, y)
    lik = lk.Categorical()

    default = fs.laplace(
        model,
        data,
        likelihood=lik,
        method=fs.SubsetOfData(24),
        prior=isotropic(),
        generator=torch.Generator().manual_seed(0),
    )
    joint = fs.laplace(
        model,
        data,
        likelihood=lik,
        method=fs.SubsetOfData(24, structure="joint"),
        prior=isotropic(),
        generator=torch.Generator().manual_seed(0),
    )

    assert default.structure == "joint"
    assert torch.equal(default.root, joint.root)
    assert torch.equal(default.at(x_test()).cov.per_input(), joint.at(x_test()).cov.per_input())

    # Today's numbers: the default is the full dense GGN posterior and the joint evidence.
    gp = _fit_all(model, x, y, lik, prior=isotropic())
    want = dense_ggn_outputs(model, lik, x, x_test())

    torch.testing.assert_close(gp.at(x_test()).cov.per_input(), want, rtol=0, atol=1e-12)
    torch.testing.assert_close(
        gp.log_marglik(),
        _dense_categorical_evidence(model, x, y, isotropic(), "joint"),
        rtol=1e-12,
        atol=0,
    )
    assert repr(fs.SubsetOfData(24)) == "SubsetOfData(n_subset=24, structure='joint')"


# --- the shared helpers ------------------------------------------------------------------


def test_rows_of_picks_rows_and_keeps_shared_leaves():
    x = torch.arange(12.0).reshape(6, 2)
    scale = torch.tensor(3.0)
    inputs = {"x": x, "scale": scale, "tag": "a", "mask": torch.arange(6) % 2 == 0}
    index = torch.tensor([4, 0, 4])

    out = fs.rows_of(inputs, index)

    assert torch.equal(out["x"], torch.stack([x[4], x[0], x[4]]))
    assert torch.equal(out["mask"], torch.tensor([True, True, True]))
    assert out["scale"] is scale and out["tag"] == "a"
    assert fs.is_shared(scale) and fs.is_shared("a") and not fs.is_shared(x)


def test_jacobian_equals_jacrev_over_the_batch():
    model, x, _ = classifier(6)
    params = {k: v.detach() for k, v in model.named_parameters()}

    got = fs.jacobian(lk.core.ModelFn.from_module(model), params, x, torch.float64)

    # One jacrev over the whole batch: examples don't interact, so its rows are the same.
    want = jacobian(model, x).reshape(-1, sum(p.numel() for p in params.values()))

    assert got.shape == (6 * 3, want.shape[1])
    torch.testing.assert_close(got, want)


def test_jacobian_shares_0dim_inputs():
    lin = nn.Linear(2, 3).double()
    params = {k: v.detach() for k, v in lin.named_parameters()}

    def scaled(p, inputs):
        return functional_call(lin, p, (inputs["x"],)) * inputs["s"]

    x = torch.randn(4, 2, dtype=torch.float64)
    s = torch.tensor(2.5, dtype=torch.float64)
    got = fs.jacobian(lk.core.ModelFn(scaled), params, {"x": x, "s": s}, torch.float64)

    want = jacrev(lambda p: scaled(p, {"x": x, "s": s}))(params)
    want = torch.cat([j.reshape(12, -1) for j in want.values()], dim=1)

    torch.testing.assert_close(got, want)


def test_cov_rows_equals_the_dense_prior_covariance():
    params = {"a": torch.zeros(2, 3, dtype=torch.float64), "b": torch.zeros(4, dtype=torch.float64)}
    prec = torch.rand(10, dtype=torch.float64) + 0.5
    rows = torch.randn(5, 10, dtype=torch.float64)

    got = fs.cov_rows(lk.priors.Diagonal(prec), params, rows)

    cov = torch.diag(1 / prec)

    torch.testing.assert_close(got, rows @ cov)


@pytest.mark.parametrize(
    ("lik", "calls"),
    [
        (lk.Categorical(), 1),
        (lk.likelihoods.from_log_prob(lambda f, y: -(f - y).square().sum(-1)), 1),
        (lk.Categorical(mask_fn=lambda f, x: None), 2),
    ],
    ids=["Categorical", "from_log_prob", "with-mask_fn"],
)
def test_at_skips_the_mask_forward_without_mask_fn(lik, calls):
    """`at` runs the model once at the query inputs; a built-in likelihood without `mask_fn`
    never masks, so the second forward at the subset (for the mask) is skipped."""

    model, x, y = classifier(12)
    base = lk.core.ModelFn.from_module(model)
    seen = []

    def counted(p, inputs):
        if not lk.core.is_transformed(inputs):  # the Jacobian's calls run under vmap
            seen.append(inputs.shape[0])

        return base(p, inputs)

    gp = fs.laplace(
        lk.core.ModelFn(counted, model),
        loader(x, y),
        likelihood=lk.Categorical(),
        method=fs.SubsetOfData(6),
        prior=isotropic(),
    )
    gp = copy.replace(gp, likelihood=lik)
    seen.clear()

    gp.at(x_test())

    assert len(seen) == calls
    assert seen[0] == x_test().shape[0]
