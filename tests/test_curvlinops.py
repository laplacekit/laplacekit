import copy

import curvlinops
import pytest
import torch
from curvlinops import KFACLinearOperator
from torch import Tensor, nn

import laplacekit as lk
import laplacekit.core as core
from laplacekit.core import Batch, FitContext, Layout, ModelFn, probe
from laplacekit.curvlinops import KFAC, KronStat

N_BATCHES = 4
BATCH = 8


def make(kind: str, bias: bool = True, dtype: torch.dtype = torch.float64):
    torch.manual_seed(0)
    out = 3 if kind == "cat" else 2
    model = nn.Sequential(
        nn.Linear(3, 6, bias=bias),
        nn.Tanh(),
        nn.Linear(6, 5, bias=bias),
        nn.Tanh(),
        nn.Linear(5, out, bias=bias),
    ).to(dtype)
    lik = (
        lk.Categorical()
        if kind == "cat"
        else copy.replace(lk.Gaussian(), sigma=torch.as_tensor(torch.tensor(0.3, dtype=dtype)))
    )
    x = torch.randn(N_BATCHES * BATCH, 3, dtype=dtype)
    y = (
        torch.randint(0, 3, (x.shape[0],))
        if kind == "cat"
        else torch.randn(x.shape[0], out).to(dtype)
    )
    params = {k: v.detach() for k, v in model.named_parameters()}
    batch = Batch(x[:BATCH], y[:BATCH])
    caps = probe(ModelFn.from_module(model), lik, batch)
    ctx = FitContext(ModelFn.from_module(model), lik, Layout.by_module(params), caps)

    return model, ctx, params, x, y


def batches(x: Tensor, y: Tensor) -> list[Batch]:
    return [
        Batch(x[i * BATCH : (i + 1) * BATCH], y[i * BATCH : (i + 1) * BATCH])
        for i in range(N_BATCHES)
    ]


def _gen(i: int) -> torch.Generator:
    """The generator of batch `i` in these hand folds."""

    return torch.Generator().manual_seed(i)


def fold(method: KFAC, params, ctx, bs) -> KronStat:
    stat = method.init(params, ctx)

    for i, b in enumerate(bs):
        stat = method.update(stat, params, b, _gen(i), ctx)

    return stat


def blockdiag_dense(op: core.BlockDiag, ctx: FitContext) -> Tensor:
    mats = []

    for b in ctx.layout.blocks:
        n = ctx.layout.size(b)
        eye = torch.eye(n, dtype=next(iter(ctx.model.module.parameters())).dtype)
        mats.append(torch.stack([op.blocks[b].mv(eye[i]) for i in range(n)], dim=1))

    return torch.block_diag(*mats)


def curvlinops_dense(model, x, y, kind: str) -> Tensor:
    loss = nn.CrossEntropyLoss(reduction="sum") if kind == "cat" else nn.MSELoss(reduction="sum")
    params = list(model.parameters())
    data = [
        (x[i * BATCH : (i + 1) * BATCH], y[i * BATCH : (i + 1) * BATCH]) for i in range(N_BATCHES)
    ]
    op = KFACLinearOperator(
        model,
        loss,
        params,
        data,
        fisher_type="type-2",
        check_deterministic=False,
        separate_weight_and_bias=False,
    )
    n = sum(p.numel() for p in params)

    return op @ torch.eye(n, dtype=params[0].dtype)


@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("kind", ["cat", "gauss"])
def test_matches_curvlinops_full_data(kind, bias):
    model, ctx, params, x, y = make(kind, bias)
    method = KFAC()
    op = method.finalize(fold(method, params, ctx, batches(x, y)))
    ours = blockdiag_dense(op, ctx) * x.shape[0]
    want = curvlinops_dense(model, x, y, kind)

    if kind == "gauss":
        want = want / (2 * 0.3**2)

    torch.testing.assert_close(ours, want, rtol=1e-5, atol=1e-8)


def test_conv2d_matches_curvlinops():
    torch.manual_seed(0)
    model = nn.Sequential(
        nn.Conv2d(2, 3, 3, padding=1), nn.Tanh(), nn.Flatten(), nn.Linear(3 * 4 * 4, 3)
    ).double()
    x = torch.randn(N_BATCHES * BATCH, 2, 4, 4, dtype=torch.double)
    y = torch.randint(0, 3, (x.shape[0],))
    params = {k: v.detach() for k, v in model.named_parameters()}
    lik = lk.Categorical()
    caps = probe(ModelFn.from_module(model), lik, Batch(x[:BATCH], y[:BATCH]))
    ctx = FitContext(ModelFn.from_module(model), lik, Layout.by_module(params), caps)
    method = KFAC()
    op = method.finalize(fold(method, params, ctx, batches(x, y)))

    # Conv2d weight is [out, in, kh, kw]; Curvlinops flattens row-major as well.
    ours = blockdiag_dense(op, ctx) * x.shape[0]
    want = curvlinops_dense(model, x, y, "cat")

    torch.testing.assert_close(ours, want, rtol=1e-5, atol=1e-8)


def test_monoid_law():
    _, ctx, params, x, y = make("cat")
    method = KFAC()
    bs = batches(x, y)
    left = fold(method, params, ctx, bs[:2])
    right = method.init(params, ctx)

    for i, b in enumerate(bs[2:]):
        right = method.update(right, params, b, _gen(i + 2), ctx)

    joint = fold(method, params, ctx, bs)
    comb = left + right

    for k in joint.a:
        torch.testing.assert_close(comb.a[k], joint.a[k], rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(comb.g[k], joint.g[k], rtol=1e-12, atol=1e-12)

    assert comb.weight == joint.weight == N_BATCHES * BATCH


def test_mc_follows_the_generator():
    _, ctx, params, x, y = make("cat")
    method = KFAC(fisher="mc")
    b = batches(x, y)[0]

    def one(seed: int) -> KronStat:
        gen = torch.Generator().manual_seed(seed)
        return method.update(method.init(params, ctx), params, b, gen, ctx)

    s1, s2, s3 = one(1), one(1), one(2)
    k = next(iter(s1.g))

    torch.testing.assert_close(s1.g[k], s2.g[k])
    assert not torch.allclose(s1.g[k], s3.g[k])


def test_weighted_batch_refused():
    _, ctx, params, x, y = make("cat")
    method = KFAC()

    assert not method.requires.weights

    with pytest.raises(lk.CapabilityError, match="weights"):
        method.update(
            method.init(params, ctx),
            params,
            Batch(x[:BATCH], y[:BATCH], torch.ones(BATCH).double()),
            torch.Generator().manual_seed(0),
            ctx,
        )


def test_unsupported_likelihood_refused_at_init():
    _, ctx, params, _, _ = make("cat")

    class Other:
        pass

    bad = FitContext(ctx.model, Other(), ctx.layout, ctx.caps)  # ty: ignore[invalid-argument-type]

    with pytest.raises(lk.CapabilityError, match="Other"):
        KFAC().init(params, bad)


def test_unhooked_block_refused_at_init():
    _, ctx, params, _, _ = make("cat")
    caps = ctx.caps
    caps = copy.replace(caps, hooked_types={k: v for k, v in caps.hooked_types.items() if k != "2"})
    bad = FitContext(ctx.model, ctx.likelihood, ctx.layout, caps)

    with pytest.raises(lk.CapabilityError, match=r"hooks.*: 2$"):
        KFAC().init(params, bad)


def test_front_door():
    model, _, _, x, y = make("cat", dtype=torch.float32)
    data = [
        (x[i * BATCH : (i + 1) * BATCH], y[i * BATCH : (i + 1) * BATCH]) for i in range(N_BATCHES)
    ]
    post = lk.laplace(
        model, data, likelihood=lk.Categorical(), subset=lk.subsets.all, hessian_approx=lk.KFAC()
    )

    assert "KFAC" in repr(post)
    assert lk.predict(post, x[:4]) is not None
    assert all(p.grad is None for p in model.parameters())


@pytest.mark.parametrize("fisher_type", ["type-2", "mc", "empirical"])
def test_guard_private_attributes(fisher_type):
    D = curvlinops.KFACLinearOperator
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(2, 2))
    op = D(
        model,
        nn.MSELoss(reduction="sum"),
        list(model.parameters()),
        [(torch.randn(2, 2), torch.randn(2, 2))],
        check_deterministic=False,
        fisher_type=fisher_type,
    )
    op.compute_kronecker_factors()
    missing = [
        n for n in ("_input_covariances", "_gradient_covariances", "_mapping") if not hasattr(op, n)
    ]

    assert not missing, f"Curvlinops private attributes gone: {missing}"
    assert set(op._mapping) == {"0"}


def test_mc_gaussian_matches_curvlinops_same_seed():
    model, ctx, params, x, y = make("gauss")
    method = KFAC(fisher="mc")
    batch = Batch(x[:BATCH], y[:BATCH])
    gen = torch.Generator().manual_seed(3)
    op = method.finalize(method.update(method.init(params, ctx), params, batch, gen, ctx))
    # The adapter seeds Curvlinops with one draw from the batch's generator.
    seed = lk.core.spawn_generator(torch.Generator().manual_seed(3)).initial_seed()
    ours = blockdiag_dense(op, ctx) * BATCH
    ref = KFACLinearOperator(
        model,
        nn.MSELoss(reduction="sum"),
        list(model.parameters()),
        [(batch.inputs, y[:BATCH])],
        fisher_type="mc",
        check_deterministic=False,
        separate_weight_and_bias=False,
        seed=seed,
    )
    n = sum(p.numel() for p in model.parameters())
    want = (ref @ torch.eye(n, dtype=torch.double)) / (2 * 0.3**2)

    torch.testing.assert_close(ours, want, rtol=1e-5, atol=1e-8)


def test_perturbed_params_refused():
    _, ctx, params, x, y = make("cat")
    method = KFAC()
    bad = {**params, "2.weight": params["2.weight"] + 1e-3}

    with pytest.raises(lk.CapabilityError, match=r"live parameters.*2\.weight"):
        method.update(
            method.init(params, ctx),
            bad,
            Batch(x[:BATCH], y[:BATCH]),
            torch.Generator().manual_seed(0),
            ctx,
        )


def test_empty_params_refused():
    _, ctx, _, _, _ = make("cat")

    with pytest.raises(lk.CapabilityError, match="at least one parameter"):
        KFAC().init({}, ctx)


def test_weight_only_subset_matches_curvlinops():
    model, _, _, x, y = make("cat")
    params = {k: v.detach() for k, v in model.named_parameters() if k.endswith("weight")}
    lik = lk.Categorical()
    caps = probe(ModelFn.from_module(model), lik, Batch(x[:BATCH], y[:BATCH]))
    ctx = FitContext(ModelFn.from_module(model), lik, Layout.by_module(params), caps)
    method = KFAC()
    op = method.finalize(fold(method, params, ctx, batches(x, y)))

    assert all(not k.has_bias for k in op.blocks.values())

    ours = blockdiag_dense(op, ctx) * x.shape[0]
    ref = KFACLinearOperator(
        model,
        nn.CrossEntropyLoss(reduction="sum"),
        [model.get_parameter(k) for k in params],
        [
            (x[i * BATCH : (i + 1) * BATCH], y[i * BATCH : (i + 1) * BATCH])
            for i in range(N_BATCHES)
        ],
        fisher_type="type-2",
        check_deterministic=False,
        separate_weight_and_bias=False,
    )
    n = sum(p.numel() for p in params.values())

    torch.testing.assert_close(ours, ref @ torch.eye(n, dtype=torch.double), rtol=1e-5, atol=1e-8)


def test_class_axis_per_position_matches_kfac_expand():
    """Logits [N, T, C], as a language model's: KFAC-expand over the positions.

    For one Linear layer with a folded bias, x~ = [x, 1]:
    A = sum_{n,t} x~ x~^T / (N T) and G = sum_{n,t} H_{nt}, and Curvlinops' full-data
    matrix is G kron A in the order (out, in + 1).
    """

    torch.manual_seed(0)
    n, t, d, c = 6, 3, 2, 4
    model = nn.Sequential(nn.Linear(d, c)).double()
    x = torch.randn(n, t, d, dtype=torch.float64)
    y = torch.randint(0, c, (n, t))
    params = {k: v.detach() for k, v in model.named_parameters()}
    lik = lk.Categorical()
    caps = probe(ModelFn.from_module(model), lik, Batch(x[:2], y[:2]))
    ctx = FitContext(ModelFn.from_module(model), lik, Layout.by_module(params), caps)
    method = KFAC()
    bs = [Batch(x[:3], y[:3]), Batch(x[3:], y[3:])]
    ours = blockdiag_dense(method.finalize(fold(method, params, ctx, bs)), ctx) * n

    xt = torch.cat([x, torch.ones(n, t, 1, dtype=x.dtype)], dim=-1).reshape(n * t, d + 1)
    a = xt.T @ xt / (n * t)
    p = torch.softmax(model(x), dim=-1).detach().reshape(n * t, c)
    g = (torch.diag_embed(p) - p.unsqueeze(-1) * p.unsqueeze(-2)).sum(0)
    full = torch.kron(g, a)  # order (out, in + 1)

    # Ours: weight row-major, then bias.
    aug = torch.arange(c * (d + 1)).reshape(c, d + 1)
    order = torch.cat([aug[:, :d].reshape(-1), aug[:, d]])
    want = full[order][:, order]

    torch.testing.assert_close(ours, want, rtol=1e-10, atol=1e-12)


def _flat(post, params) -> Tensor:
    """Params-shaped tensors, with any leading sample axes, as [..., P] in layout order."""

    parts = []

    for n in post.mean:
        lead = params[n].shape[: params[n].ndim - len(post.layout.shapes[n])]
        parts.append(params[n].reshape(*lead, -1))

    return torch.cat(parts, dim=-1)


def test_marginal_variance_and_weight_samples_on_kfac_posterior():
    model, _, _, x, y = make("cat")
    post = lk.laplace(
        model, batches(x, y), likelihood=lk.Categorical(), subset=lk.subsets.all,
        hessian_approx=lk.KFAC(), hyper_tuner=None,
    )  # fmt: skip
    cov = torch.linalg.inv(blockdiag_dense(post.precision(), post))

    torch.testing.assert_close(_flat(post, post.variance), cov.diagonal())

    draws = post.sample(20000, generator=torch.Generator().manual_seed(1))
    centred = _flat(post, draws) - _flat(post, post.mean)

    emp = centred.T @ centred / 20000

    torch.testing.assert_close(emp, cov, rtol=0, atol=0.05 * float(cov.abs().max()))

    d = lk.predict(post, x[:4], lk.WeightSamples(n_samples=8))

    assert d.probs.shape == (4, 3) and torch.isfinite(d.probs).all()


def test_live_parameter_views_skip_the_value_compare(monkeypatch):
    import laplacekit.curvlinops as adapter

    _, ctx, params, x, y = make("cat")
    method = KFAC()
    calls = []
    real = torch.equal

    def counting(a, b):
        calls.append(1)
        return real(a, b)

    monkeypatch.setattr(adapter.torch, "equal", counting)
    method.update(
        method.init(params, ctx),
        params,
        Batch(x[:BATCH], y[:BATCH]),
        torch.Generator().manual_seed(0),
        ctx,
    )

    assert calls == []  # detached views of the live parameters: same storage

    copies = {k: v.clone() for k, v in params.items()}
    method.update(
        method.init(copies, ctx),
        copies,
        Batch(x[:BATCH], y[:BATCH]),
        torch.Generator().manual_seed(0),
        ctx,
    )

    assert len(calls) == len(params)  # equal values in other storage are still accepted


def test_kfac_operator_keeps_factors():
    """`evals == count * outer(lg, la)`; the factors rebuild Curvlinops' block, and `A` is
    the per-example average of the first layer's inputs (with a ones column for the bias)."""

    model, ctx, params, x, y = make("cat")
    method = KFAC()
    op = method.finalize(fold(method, params, ctx, batches(x, y)))
    want = curvlinops_dense(model, x, y, "cat") / x.shape[0]
    start = 0

    for b in ctx.layout.blocks:
        blk = op.blocks[b]
        la, lg, count = blk.factors

        assert float(count) == 1.0
        torch.testing.assert_close(blk.evals, count * torch.outer(lg, la))

        # [W | b] in folded order, then to layout order (weight row-major, then bias).
        a = blk.evecs_a @ torch.diag(la) @ blk.evecs_a.T
        g = blk.evecs_g @ torch.diag(lg) @ blk.evecs_g.T
        out, n_in = g.shape[0], a.shape[0]
        idx = [i * n_in + j for i in range(out) for j in range(n_in - 1)]
        idx = torch.tensor(idx + [i * n_in + n_in - 1 for i in range(out)])
        n = ctx.layout.size(b)

        torch.testing.assert_close(
            torch.kron(g, a)[idx][:, idx], want[start : start + n, start : start + n]
        )

        start += n

    xb = torch.cat([x, torch.ones(x.shape[0], 1, dtype=x.dtype)], dim=1)
    la, _, _ = op.blocks["0"].factors
    qa = op.blocks["0"].evecs_a

    torch.testing.assert_close(qa @ torch.diag(la) @ qa.T, xb.T @ xb / x.shape[0])

    three = op.blocks["2"].scaled(torch.tensor(3.0, dtype=torch.float64))
    la, lg, count = three.factors

    assert float(count) == 3.0
    torch.testing.assert_close(three.evals, count * torch.outer(lg, la))
    assert op.blocks["2"].add_isotropic(torch.tensor(1.0, dtype=torch.float64)).factors == ()


# --- the empirical Fisher ------------------------------------------------------------------


def _score(lik, f: Tensor, y: Tensor, param: Tensor) -> Tensor:
    """grad of log p(y | f) with respect to `param`, flattened row-major."""

    (g,) = torch.autograd.grad(lik.log_prob(f, y).sum(), param)

    return g.reshape(-1)


@pytest.mark.parametrize("kind", ["cat", "gauss"])
def test_kfac_empirical_single_example_is_exact(kind):
    """One example: A = x x^T and G = d d^T with d = grad_f log p, so G kron A = g g^T."""

    torch.manual_seed(0)
    out = 3 if kind == "cat" else 2
    model = nn.Sequential(nn.Linear(4, out, bias=False)).double()
    lik = lk.Categorical() if kind == "cat" else lk.Gaussian(1.0)
    x = torch.randn(1, 4, dtype=torch.float64)
    y = torch.tensor([2]) if kind == "cat" else torch.randn(1, out, dtype=torch.float64)
    params = {k: v.detach() for k, v in model.named_parameters()}
    caps = probe(ModelFn.from_module(model), lik, Batch(x, y))
    ctx = FitContext(ModelFn.from_module(model), lik, Layout.by_module(params), caps)

    method = KFAC(fisher="empirical")
    op = method.finalize(fold(method, params, ctx, [Batch(x, y)]))
    ours = blockdiag_dense(op, ctx)

    g = _score(lik, model(x), y, model[0].weight)

    torch.testing.assert_close(ours, torch.outer(g, g), rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("kind", ["cat", "gauss"])
def test_kfac_empirical_matches_explicit_kron(kind):
    """Two layers with biases over a batch: per layer, A = sum_n x~ x~^T / N and
    G = sum_n d_n d_n^T with d_n the score's gradient at the layer's output."""

    torch.manual_seed(0)
    n, d_in, hidden = 7, 3, 4
    out = 3 if kind == "cat" else 2
    model = nn.Sequential(nn.Linear(d_in, hidden), nn.Tanh(), nn.Linear(hidden, out)).double()
    lik = lk.Categorical() if kind == "cat" else lk.Gaussian(1.0)
    x = torch.randn(n, d_in, dtype=torch.float64)
    y = torch.randint(0, out, (n,)) if kind == "cat" else torch.randn(n, out, dtype=torch.float64)
    params = {k: v.detach() for k, v in model.named_parameters()}
    caps = probe(ModelFn.from_module(model), lik, Batch(x, y))
    ctx = FitContext(ModelFn.from_module(model), lik, Layout.by_module(params), caps)

    method = KFAC(fisher="empirical")
    bs = [Batch(x[:3], y[:3]), Batch(x[3:], y[3:])]
    op = method.finalize(fold(method, params, ctx, bs))
    ours = blockdiag_dense(op, ctx) * n

    assert all(len(blk.factors) == 3 for blk in op.blocks.values())  # kept for KronSum

    # The layers by hand, to reach each layer's input and output.
    z1 = model[0](x)
    h = torch.tanh(z1)
    z2 = model[2](h)
    d1, d2 = torch.autograd.grad(lik.log_prob(z2, y).sum(), (z1, z2))

    blocks = []

    for inp, d in [(x, d1), (h.detach(), d2)]:
        xt = torch.cat([inp, torch.ones(n, 1, dtype=inp.dtype)], dim=1)
        a = xt.T @ xt / n
        g = d.T @ d
        full = torch.kron(g, a)  # order (out, in + 1)
        o, i = d.shape[1], inp.shape[1]
        aug = torch.arange(o * (i + 1)).reshape(o, i + 1)
        order = torch.cat([aug[:, :i].reshape(-1), aug[:, i]])  # weight row-major, then bias
        blocks.append(full[order][:, order])

    torch.testing.assert_close(ours, torch.block_diag(*blocks), rtol=1e-10, atol=1e-12)


def test_kfac_empirical_class_axis_per_position():
    """Logits [N, T, C] with targets [N, T]: KFAC-expand over the positions, with
    G = sum_{n,t} d_nt d_nt^T and d_nt the score's gradient at position t."""

    torch.manual_seed(0)
    n, t, d, c = 6, 3, 2, 4
    model = nn.Sequential(nn.Linear(d, c)).double()
    x = torch.randn(n, t, d, dtype=torch.float64)
    y = torch.randint(0, c, (n, t))
    params = {k: v.detach() for k, v in model.named_parameters()}
    lik = lk.Categorical()
    caps = probe(ModelFn.from_module(model), lik, Batch(x[:2], y[:2]))
    ctx = FitContext(ModelFn.from_module(model), lik, Layout.by_module(params), caps)

    method = KFAC(fisher="empirical")
    bs = [Batch(x[:3], y[:3]), Batch(x[3:], y[3:])]
    ours = blockdiag_dense(method.finalize(fold(method, params, ctx, bs)), ctx) * n

    f = model(x)
    (score,) = torch.autograd.grad(lik.log_prob(f, y).sum(), f)
    xt = torch.cat([x, torch.ones(n, t, 1, dtype=x.dtype)], dim=-1).reshape(n * t, d + 1)
    a = xt.T @ xt / (n * t)
    g = score.reshape(n * t, c).T @ score.reshape(n * t, c)
    full = torch.kron(g, a)  # order (out, in + 1)
    aug = torch.arange(c * (d + 1)).reshape(c, d + 1)
    order = torch.cat([aug[:, :d].reshape(-1), aug[:, d]])

    torch.testing.assert_close(ours, full[order][:, order], rtol=1e-10, atol=1e-12)


def test_kfac_empirical_requires():
    want = core.Requirements(
        hooks=(nn.Linear, nn.Conv2d), weights=False, needs_targets=True, late_scale=False
    )

    assert KFAC(fisher="empirical").requires == want
    assert KFAC().requires == core.Requirements(hooks=(nn.Linear, nn.Conv2d), weights=False)


def test_kfac_empirical_needs_targets_before_data():
    model, ctx, params, x, _ = make("cat")
    method = KFAC(fisher="empirical")

    with pytest.raises(lk.CapabilityError, match="needs targets"):
        method.update(
            method.init(params, ctx),
            params,
            Batch(x[:BATCH]),
            torch.Generator().manual_seed(0),
            ctx,
        )

    class Counting:
        def __init__(self, items):
            self.items, self.pulled = items, 0

        def __iter__(self):
            for it in self.items:
                self.pulled += 1
                yield it

    ld = Counting([Batch(x[:BATCH]), Batch(x[BATCH : 2 * BATCH])])

    with pytest.raises(lk.CapabilityError, match="needs targets"):
        lk.laplace(model, ld, likelihood=lk.Categorical(), hessian_approx=method, hyper_tuner=None)

    assert ld.pulled == 1  # the probe's batch only


def test_kfac_empirical_refuses_late_sigma():
    req = KFAC(fisher="empirical").requires

    with pytest.raises(lk.CapabilityError, match="late hyperparameter"):
        core.check_likelihood(req, lk.Gaussian(), "KFAC")

    core.check_likelihood(req, lk.Gaussian(1.0), "KFAC")
    core.check_likelihood(KFAC().requires, lk.Gaussian(), "KFAC")

    _, ctx, params, _, _ = make("gauss")  # a context built by hand, with sigma = 0.3

    with pytest.raises(lk.CapabilityError, match="late hyperparameter"):
        KFAC(fisher="empirical").init(params, ctx)


@pytest.mark.parametrize("fisher", ["exact", "empirical"])
def test_kfac_per_output_sigma_matches_explicit_kron(fisher):
    """A per-output sigma is fixed at fit time, so the empirical Fisher takes it too. Per
    layer, A = sum_n x x^T / N and G = sum_n J_n^T Λ J_n (exact) or sum_n d_n d_n^T with
    d_n = J_n^T (y_n - f_n) / sigma² (empirical), by hand on a bias-free two-layer net."""

    torch.manual_seed(0)
    n, d_in, hidden = 7, 3, 4
    sigma = torch.tensor([0.4, 1.7], dtype=torch.float64)
    model = nn.Sequential(
        nn.Linear(d_in, hidden, bias=False), nn.Tanh(), nn.Linear(hidden, 2, bias=False)
    ).double()
    lik = copy.replace(lk.Gaussian(), sigma=torch.as_tensor(sigma))
    x = torch.randn(n, d_in, dtype=torch.float64)
    y = torch.randn(n, 2, dtype=torch.float64)
    params = {k: v.detach() for k, v in model.named_parameters()}
    caps = probe(ModelFn.from_module(model), lik, Batch(x, y))
    ctx = FitContext(ModelFn.from_module(model), lik, Layout.by_module(params), caps)

    method = KFAC(fisher=fisher)
    op = method.finalize(fold(method, params, ctx, [Batch(x[:3], y[:3]), Batch(x[3:], y[3:])]))
    ours = blockdiag_dense(op, ctx) * n

    z1 = model[0](x)
    h = torch.tanh(z1)
    z2 = model[2](h)
    blocks = []

    for inp, z in [(x, z1), (h.detach(), z2)]:
        if fisher == "empirical":
            (d,) = torch.autograd.grad(z2, z, ((y - z2) / sigma**2).detach(), retain_graph=True)
            g = d.T @ d
        else:
            # J_n^T Λ J_n at the layer's output: rows of the output Jacobian, whitened.
            g = torch.zeros(z.shape[1], z.shape[1], dtype=torch.float64)

            for k in range(2):
                e = torch.zeros_like(z2)
                e[:, k] = 1 / sigma[k]
                (r,) = torch.autograd.grad(z2, z, e, retain_graph=True)
                g = g + r.T @ r

        blocks.append(torch.kron(g, inp.T @ inp / n))

    torch.testing.assert_close(ours, torch.block_diag(*blocks), rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("kind", ["cat", "gauss"])
def test_kfac_empirical_passes_check_curvature(kind):
    model, _, params, x, y = make(kind)
    lik = lk.Categorical() if kind == "cat" else lk.Gaussian(1.0)

    lk.testing.check_curvature(KFAC(fisher="empirical"), model, lik, params, batches(x, y))


# --- Bernoulli -----------------------------------------------------------------------------


def _bernoulli(event_dims: int = 0, soft: bool = False):
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(3, 6), nn.Tanh(), nn.Linear(6, 4)).double()
    lik = lk.Bernoulli(event_dims=event_dims)
    x = torch.randn(N_BATCHES * BATCH, 3, dtype=torch.float64)
    y = torch.rand(x.shape[0], 4, dtype=torch.float64)
    y = y if soft else (y > 0.5).double()
    params = {k: v.detach() for k, v in model.named_parameters()}
    caps = probe(ModelFn.from_module(model), lik, Batch(x[:BATCH], y[:BATCH]))
    ctx = FitContext(ModelFn.from_module(model), lik, Layout.by_module(params), caps)

    return model, ctx, params, x, y


def _curvlinops_bce(model, x, y, fisher_type: str) -> Tensor:
    params = list(model.parameters())
    data = [
        (x[i * BATCH : (i + 1) * BATCH], y[i * BATCH : (i + 1) * BATCH]) for i in range(N_BATCHES)
    ]
    op = KFACLinearOperator(
        model,
        nn.BCEWithLogitsLoss(reduction="sum"),
        params,
        data,
        fisher_type=fisher_type,
        check_deterministic=False,
        separate_weight_and_bias=False,
    )

    return op @ torch.eye(sum(p.numel() for p in params), dtype=torch.float64)


@pytest.mark.parametrize(
    ("fisher", "fisher_type"), [("exact", "type-2"), ("empirical", "empirical")]
)
def test_bernoulli_kfac_matches_curvlinops(fisher, fisher_type):
    # Soft labels for the empirical Fisher: its score sigmoid(f) - y needs no binary target.
    model, ctx, params, x, y = _bernoulli(soft=fisher == "empirical")
    method = KFAC(fisher=fisher)
    op = method.finalize(fold(method, params, ctx, batches(x, y)))
    ours = blockdiag_dense(op, ctx) * x.shape[0]

    torch.testing.assert_close(
        ours, _curvlinops_bce(model, x, y, fisher_type), rtol=1e-5, atol=1e-8
    )


def test_bernoulli_event_dims_kfac():
    _, ctx0, params, x, y = _bernoulli(event_dims=0)
    _, ctx1, _, _, _ = _bernoulli(event_dims=1)

    for fisher in ("exact", "mc", "empirical"):
        method = KFAC(fisher=fisher)
        s0 = fold(method, params, ctx0, batches(x, y))
        s1 = fold(method, params, ctx1, batches(x, y))

        for b in s0.a:
            torch.testing.assert_close(s1.a[b], s0.a[b], rtol=0, atol=0)
            torch.testing.assert_close(s1.g[b], s0.g[b], rtol=0, atol=0)


def test_default_picks_kfac_for_bernoulli_layers():
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(3, 8), nn.Tanh(), nn.Linear(8, 600))
    lik = lk.Bernoulli()
    x, y = torch.randn(4, 3), torch.randint(0, 2, (4, 600)).float()
    caps = probe(ModelFn.from_module(model), lik, Batch(x, y))
    params = {k: v.detach() for k, v in model.named_parameters()}
    got = lk.curvature.default(params, caps, likelihood=lik)

    assert KFAC.supports(lik) and KFAC.supports(lk.Bernoulli(event_dims=1))
    assert got.by_block(["0", "2"]) == {"0": lk.DenseGGN(), "2": KFAC()}


def test_kfac_refusal_names_diag():
    @core.pytree
    class Poisson:
        def log_prob(self, f: Tensor, y: Tensor, /) -> Tensor:
            return torch.distributions.Poisson(f.exp()).log_prob(y)

    _, ctx, params, _, _ = make("cat")
    bad = FitContext(ctx.model, Poisson(), ctx.layout, ctx.caps)
    want = (
        "KFAC supports Categorical, Gaussian and Bernoulli likelihoods, got Poisson; "
        "use DiagGGN or DenseGGN"
    )

    assert not KFAC.supports(Poisson())

    with pytest.raises(lk.CapabilityError) as e:
        KFAC().init(params, bad)

    assert str(e.value) == want


def test_bernoulli_kfac_mc_approaches_exact():
    """Many sampled targets per example: the MC gradient factor averages to the exact one
    (Curvlinops' type-2 with BCE), within 5% in Frobenius norm."""

    model, ctx, params, x, y = _bernoulli()
    repeats = 100
    bs = batches(x, y) * repeats
    method = KFAC(fisher="mc")
    stat = method.init(params, ctx)

    for i, b in enumerate(bs):
        stat = method.update(stat, params, b, _gen(i), ctx)

    exact = KFAC().finalize(fold(KFAC(), params, ctx, batches(x, y)))
    mc = method.finalize(stat)
    want = blockdiag_dense(exact, ctx)
    got = blockdiag_dense(mc, ctx)

    assert float((got - want).norm() / want.norm()) < 0.05


def test_bernoulli_kfac_empirical_refuses_targets_outside_zero_one():
    _, ctx, params, x, y = _bernoulli()
    y = y.clone()
    y[0, 0] = 2.0
    method = KFAC(fisher="empirical")

    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        method.update(
            method.init(params, ctx),
            params,
            Batch(x[:BATCH], y[:BATCH]),
            torch.Generator().manual_seed(0),
            ctx,
        )


@pytest.mark.parametrize("fisher", ["exact", "mc", "empirical"])
def test_fixed_sigma_enters_the_kfac_factors(fisher):
    """A fixed sigma is applied at fit time: on a one-output last layer KFAC is exact, so
    its predictive matches the dense GGN's with the same sigma (one example for the
    empirical Fisher, where KFAC is exact only then; many draws for the MC Fisher)."""

    torch.manual_seed(0)
    n = 1 if fisher == "empirical" else 40
    x = torch.rand(n, 1, dtype=torch.float64) * 8
    y = torch.sin(x) + 0.3 * torch.randn(n, 1, dtype=torch.float64)
    model = nn.Sequential(nn.Linear(1, 8), nn.Tanh(), nn.Linear(8, 1)).double()
    x_test = torch.linspace(-2, 10, 25, dtype=torch.float64).reshape(-1, 1)
    mc = {"mc_samples": 2000} if fisher == "mc" else {}
    common = dict(likelihood=lk.Gaussian(sigma=0.3), hyper_tuner=None, generator=_gen(0))

    kfac = lk.laplace(model, [(x, y)], hessian_approx=KFAC(fisher=fisher, **mc), **common)
    dense_fisher = "exact" if fisher == "mc" else fisher
    dense = lk.laplace(model, [(x, y)], hessian_approx=lk.DenseGGN(fisher=dense_fisher), **common)

    tol = 3e-2 if fisher == "mc" else 1e-8
    torch.testing.assert_close(
        lk.predict(kfac, x_test).stddev, lk.predict(dense, x_test).stddev, rtol=tol, atol=0
    )
