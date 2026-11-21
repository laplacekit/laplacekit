"""Low-precision models: statistics accumulate in at least float32."""

import copy

import pytest
import torch
from torch import nn

import laplacekit as lk
from laplacekit.core import (
    Batch,
    FitContext,
    Layout,
    ModelFn,
    probe,
    stat_dtype,
    tree_leaves,
)

BF16 = torch.bfloat16


@pytest.mark.parametrize(
    ("dtype", "want"),
    [
        (torch.bfloat16, torch.float32),
        (torch.float16, torch.float32),
        (torch.float32, torch.float32),
        (torch.float64, torch.float64),
    ],
)
def test_stat_dtype(dtype, want):
    assert stat_dtype(dtype) == want


def _models():
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(3, 8), nn.Tanh(), nn.Linear(8, 3))
    x = torch.randn(64, 3)
    y = torch.randint(0, 3, (64,))

    return model, copy.deepcopy(model).to(BF16), x, y


def _ctx(model, x, y):
    params = {k: v.detach() for k, v in model.named_parameters()}
    fn = ModelFn.from_module(model)
    lik = lk.Categorical()
    caps = probe(fn, lik, Batch(x[:8], y[:8]))

    return params, FitContext(fn, lik, Layout.by_module(params), caps)


def _fold(method, params, ctx, x, y):
    stat = method.init(params, ctx)

    for i in range(4):
        b = Batch(x[16 * i : 16 * (i + 1)], y[16 * i : 16 * (i + 1)])
        stat = method.update(stat, params, b, torch.Generator().manual_seed(0 + i), ctx)

    return stat


METHODS = {
    "Diag": lk.DiagGGN(),
    "DenseGGN": lk.DenseGGN(),
    "DiagMCFisher": lk.DiagGGN(fisher="mc"),
    "KFAC": lk.KFAC(),
}


@pytest.mark.parametrize("name", list(METHODS))
def test_bf16_statistic_is_float32_and_close_to_the_float32_fit(name):
    method = METHODS[name]
    model, low, x, y = _models()
    params, ctx = _ctx(low, x.to(BF16), y)

    init = method.init(params, ctx)

    assert {t.dtype for t in tree_leaves(init)} == {torch.float32}

    stat = _fold(method, params, ctx, x.to(BF16), y)

    assert {t.dtype for t in tree_leaves(stat)} == {torch.float32}

    ref_params, ref_ctx = _ctx(model, x, y)
    ours = method.finalize(stat)
    ref = method.finalize(_fold(method, ref_params, ref_ctx, x, y))

    for b in ref.blocks:
        got, want = ours.blocks[b].diagonal(), ref.blocks[b].diagonal()

        assert got.dtype == torch.float32
        # bf16 forward passes carry about 3 significant digits.
        torch.testing.assert_close(got, want, rtol=0.05, atol=0.02 * float(want.abs().max()))


def test_bf16_model_through_the_front_door():
    _, low, x, y = _models()
    xb = x.to(BF16)
    data = [(xb[16 * i : 16 * (i + 1)], y[16 * i : 16 * (i + 1)]) for i in range(4)]

    for approx in (None, lk.KFAC()):
        post = lk.laplace(low, data, likelihood=lk.Categorical(), hessian_approx=approx)

        assert post.n_data.dtype == torch.float32
        assert {t.dtype for t in tree_leaves(post.curvature)} == {torch.float32}
        assert all(p.dtype == BF16 for p in post.mean.values())

        d = lk.predict(post, xb[:4])

        assert torch.isfinite(d.probs).all()

        draws = post.sample(3, generator=torch.Generator().manual_seed(0))

        assert all(v.dtype == BF16 and v.shape[0] == 3 for v in draws.values())
        assert all(v.dtype == BF16 for v in post.variance.values())
        assert torch.isfinite(lk.predict(post, xb[:4], lk.WeightSamples(n_samples=2)).probs).all()


def test_bf16_log_lik_accumulates_in_float32():
    _, low, x, y = _models()
    params, ctx = _ctx(low, x.to(BF16), y)
    fit = lk.core.fit_curvature(
        ctx.model, ctx.likelihood, params, lk.DiagGGN(), capabilities=ctx.caps
    )
    stat = fit.update(fit.init(), Batch(x.to(BF16), y))

    assert stat.data_fit is not None and stat.data_fit.dtype == torch.float32


def test_bf16_sampled_outputs_keep_tight_posterior_noise():
    """Sampled output covariances of a bf16 model match those of the same posterior in f32.

    With a tight posterior the weight noise is far below the bf16 spacing of the weights, so
    it must never be added to the mean in bf16.
    """

    import dataclasses

    from laplacekit.core import Dense, LaplacePosterior

    high, low, x, _ = _models()
    g = torch.Generator().manual_seed(1)
    mean = {"2.weight": high[2].weight.detach(), "2.bias": high[2].bias.detach()}
    a = torch.randn(27, 27, generator=g)
    post = LaplacePosterior(
        mean=mean,
        curvature=Dense(a @ a.T / 27),
        prior=lk.priors.Isotropic(torch.tensor(1e7)),
        n_data=torch.tensor(64.0),
        model=ModelFn.from_module(high),
        likelihood=lk.Categorical(),
        layout=Layout.by_module(mean),
        caps=lk.core.Capabilities(vjp=True, vjp_twice=True, jvp=True),
    )
    low_mean = {"2.weight": low[2].weight.detach(), "2.bias": low[2].bias.detach()}
    post_low = dataclasses.replace(post, mean=low_mean, model=ModelFn.from_module(low))
    xs = x[:4]

    for strategy in ("reverse-mode", "forward-mode"):
        # The same seed: both push the same weight noise.
        g, h = torch.Generator().manual_seed(0), torch.Generator().manual_seed(0)
        ref = lk.predict_outputs(
            post, xs, strategy=strategy, n_samples=500, generator=g
        ).cov.diagonal()
        got = lk.predict_outputs(
            post_low, xs.to(BF16), strategy=strategy, n_samples=500, generator=h
        ).cov

        assert got.draws.dtype == torch.float32
        assert float((got.diagonal() - ref).norm() / ref.norm()) < 0.05
