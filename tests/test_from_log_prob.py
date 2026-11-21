"""`from_log_prob` likelihoods and the empirical Fisher."""

import dataclasses
import math

import pytest
import torch
from torch import Tensor, nn
from torch.func import grad

import laplacekit as lk
import laplacekit.core as core
from laplacekit.core import Batch, FitContext, Layout, ModelFn
from laplacekit.curvature import DiagGGN, default
from laplacekit.likelihoods import Categorical, from_log_prob

N, D = 24, 4


def bradley_terry(f: Tensor, y: Tensor) -> Tensor:
    """log p(winner | scores of the two items): f [N, 2], y [N] in {0, 1}."""

    return f.gather(1, y.unsqueeze(1)).squeeze(1) - torch.logsumexp(f, dim=1)


BT = from_log_prob(bradley_terry)


class InPlace(nn.Module):
    """Per-example MLP whose output is written in place into a fresh tensor: no vmap."""

    def __init__(self) -> None:
        super().__init__()
        self.l1 = nn.Linear(D, 5)
        self.l2 = nn.Linear(5, 2)

    def forward(self, x: Tensor) -> Tensor:
        out = torch.zeros(x.shape[0], 2, dtype=x.dtype)
        out.add_(self.l2(torch.tanh(self.l1(x))))

        return out


def _mlp() -> nn.Module:
    return nn.Sequential(nn.Linear(D, 5), nn.Tanh(), nn.Linear(5, 2))


def _pairs(n: int = N, seed: int = 1) -> tuple[Tensor, Tensor]:
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, D, generator=g, dtype=torch.float64)
    y = (x[:, 0] + 0.3 * torch.randn(n, generator=g, dtype=torch.float64) > 0).long()

    return x, y


def _setup(module: nn.Module) -> tuple[FitContext, dict[str, Tensor]]:
    torch.manual_seed(0)
    module = module.double()
    fn = ModelFn.from_module(module)
    params = {k: v.detach() for k, v in module.named_parameters()}
    x, y = _pairs()
    caps = core.probe(fn, BT, Batch(x[:4], y[:4]))

    return FitContext(fn, BT, Layout.by_module(params), caps), params


def _reference(ctx: FitContext, params, x, y, w) -> dict[str, Tensor]:
    """(1/sum w) sum_i w_i g_i^2 per block, g_i from torch.func.grad one example at a time."""

    def lp(p, xi, yi):
        return bradley_terry(ctx.model(p, xi.unsqueeze(0)), yi.unsqueeze(0)).sum()

    sums = {b: torch.zeros(ctx.layout.size(b), dtype=x.dtype) for b in ctx.layout.blocks}

    for i in range(x.shape[0]):
        g = grad(lp)(params, x[i], y[i])

        for b, names in ctx.layout.blocks.items():
            gb = torch.cat([g[n].reshape(-1) for n in names])
            sums[b] = sums[b] + w[i] * gb**2

    return {b: s / w.sum() for b, s in sums.items()}


def _fit(method, ctx, params, x, y, w=None, k: int = 3):
    stat = method.init(params, ctx)
    n = x.shape[0] // k

    for i in range(k):
        s = slice(i * n, (i + 1) * n)
        batch = Batch(x[s], y[s], None if w is None else w[s])
        stat = method.update(stat, params, batch, torch.Generator().manual_seed(i), ctx)

    return method.finalize(stat)


@pytest.mark.parametrize("module", [_mlp, InPlace], ids=["vmap", "loop"])
@pytest.mark.parametrize("weighted", [False, True], ids=["unweighted", "weighted"])
def test_empirical_fisher_matches_reference(module, weighted):
    ctx, params = _setup(module())

    assert ctx.caps.vmap == (module is _mlp)

    x, y = _pairs()
    w = torch.rand(N, generator=torch.Generator().manual_seed(2), dtype=x.dtype)
    w = w if weighted else torch.ones(N, dtype=x.dtype)
    ref = _reference(ctx, params, x, y, w)
    got = _fit(DiagGGN(fisher="empirical"), ctx, params, x, y, w if weighted else None)

    assert set(got.blocks) == set(ref)

    for b in ref:
        torch.testing.assert_close(got.blocks[b].diagonal(), ref[b], rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("module", [_mlp, InPlace], ids=["vmap", "loop"])
def test_empirical_fisher_monoid_law(module):
    ctx, params = _setup(module())
    x, y = _pairs()
    method = DiagGGN(fisher="empirical")

    def fold(lo, hi):
        stat = method.init(params, ctx)

        for i in range(lo, hi):
            s = slice(i * 6, (i + 1) * 6)
            stat = method.update(
                stat, params, Batch(x[s], y[s]), torch.Generator().manual_seed(i), ctx
            )

        return stat

    joint, comb = fold(0, 4), fold(0, 2) + fold(2, 4)

    for b in joint.sum:
        torch.testing.assert_close(comb.sum[b], joint.sum[b], rtol=1e-12, atol=1e-12)

    torch.testing.assert_close(comb.weight, joint.weight)


@pytest.mark.parametrize("module", [_mlp, InPlace], ids=["vmap", "loop"])
def test_empirical_fisher_is_differentiable_in_params(module):
    """D8: the by-hand fit keeps the graph to the passed params."""

    ctx, params = _setup(module())
    x, y = _pairs()
    names = list(params)

    def logdet(p):
        op = _fit(DiagGGN(fisher="empirical"), ctx, p, x, y)
        return op.add_isotropic(torch.tensor(1.0, dtype=torch.float64)).logdet()

    leaves = {n: params[n].clone().requires_grad_() for n in names}
    g = torch.autograd.grad(logdet(leaves), [leaves[n] for n in names])
    eps = 1e-6

    for n, gn in zip(names, g, strict=True):
        flat = params[n].reshape(-1)

        for i in range(min(3, flat.numel())):

            def shifted(d, n=n, i=i):
                q = {k: v.clone() for k, v in params.items()}
                q[n].view(-1)[i] += d

                return logdet(q)

            fd = (shifted(eps) - shifted(-eps)) / (2 * eps)

            assert torch.allclose(gn.reshape(-1)[i], fd, rtol=1e-5, atol=1e-8)


@pytest.mark.parametrize("vmap", [True, False], ids=["vmap", "loop"])
def test_empirical_fisher_is_padding_invariant(vmap):
    """A masked likelihood: garbage at masked positions changes nothing."""

    t, c = 4, 3
    torch.manual_seed(0)
    module = nn.Sequential(nn.Linear(D, 6), nn.Tanh(), nn.Linear(6, c)).double()

    class PerPosition(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = module

        def forward(self, x: Tensor, mask: Tensor) -> Tensor:
            del mask
            return self.net(x)

    lik = Categorical(mask_fn=lambda f, inputs: inputs["mask"])
    model = PerPosition()
    fn = ModelFn.from_module(model)
    params = {k: v.detach() for k, v in model.named_parameters()}
    g = torch.Generator().manual_seed(3)
    x = torch.randn(6, t, D, generator=g, dtype=torch.float64)
    y = torch.randint(0, c, (6, t), generator=g)
    mask = torch.arange(t) < torch.tensor([1, 2, 4, 3, 4, 2]).unsqueeze(1)
    junk_x = torch.where(mask.unsqueeze(-1), x, 1e3 * torch.randn_like(x))
    junk_y = torch.where(mask, y, (y + 1) % c)

    caps = core.probe(fn, lik, Batch({"x": x, "mask": mask}, y))
    caps = dataclasses.replace(caps, vmap=caps.vmap and vmap)
    ctx = FitContext(fn, lik, Layout.by_module(params), caps)
    method = DiagGGN(fisher="empirical")

    def stat(xs, ys):
        return method.update(
            method.init(params, ctx),
            params,
            Batch({"x": xs, "mask": mask}, ys),
            torch.Generator().manual_seed(0),
            ctx,
        )

    clean, junk = stat(x, y), stat(junk_x, junk_y)

    for b in clean.sum:
        assert torch.equal(clean.sum[b], junk.sum[b])

    unmasked = FitContext(fn, Categorical(), ctx.layout, caps)
    plain = method.update(
        method.init(params, unmasked),
        params,
        Batch({"x": junk_x, "mask": mask}, junk_y),
        torch.Generator().manual_seed(0),
        unmasked,
    )

    assert not all(torch.equal(clean.sum[b], plain.sum[b]) for b in clean.sum)


def test_empirical_fisher_loop_runs_one_example_at_a_time():
    """Without vmap: one forward and one backward per example, each on that example alone,
    so a batch costs N single-example passes, not N backwards through the whole batch."""

    ctx, params = _setup(InPlace())
    sizes: list[int] = []
    model = ctx.model

    def counted(p, x):
        sizes.append(x.shape[0])
        return model(p, x)

    ctx = dataclasses.replace(ctx, model=ModelFn(counted, model.module))
    x, y = _pairs(8)
    method = DiagGGN(fisher="empirical")
    method.update(
        method.init(params, ctx), params, Batch(x, y), torch.Generator().manual_seed(0), ctx
    )

    assert sizes == [1] * 8


def test_from_log_prob_declares_only_log_prob():
    assert isinstance(BT, core.Likelihood)
    assert not isinstance(BT, core.HasDistribution | core.HasSample | core.HasOutputHessian)
    assert not core.can_sample(BT)
    assert "bradley_terry" in repr(BT)

    f = torch.tensor([[1.0, 0.0], [0.0, 2.0]])
    y = torch.tensor([0, 0])

    torch.testing.assert_close(BT.log_prob(f, y), bradley_terry(f, y))
    assert lk.likelihoods.from_log_prob is from_log_prob


def test_requires_targets_only():
    assert DiagGGN(fisher="empirical").requires == lk.core.Requirements(
        needs_targets=True, late_scale=False
    )


def test_default_routes_a_log_prob_only_likelihood_to_the_empirical_fisher():
    ctx, params = _setup(_mlp())
    chosen = default(params, ctx.caps, likelihood=BT)

    assert all(m == DiagGGN(fisher="empirical") for m in chosen.rules.values())

    # A likelihood that can sample keeps the GGN or the sampled Fisher.
    sampled = default(params, ctx.caps, likelihood=Categorical())

    assert not any(m == DiagGGN(fisher="empirical") for m in sampled.rules.values())


def _bt_loader():
    x, y = _pairs(n=60)
    return [(x[i : i + 20].float(), y[i : i + 20]) for i in range(0, 60, 20)]


def test_bradley_terry_through_the_front_door():
    torch.manual_seed(0)
    post = lk.laplace(_mlp(), _bt_loader(), likelihood=BT)
    text = repr(post)

    assert "DiagGGN(fisher='empirical')" in text
    assert "hyper_tuner    = MarginalLikelihood()" in text
    assert math.isfinite(float(post.log_marglik()))
    assert float(post.prior.prior_precision) != 1.0


class _Counting:
    def __init__(self, items):
        self.items, self.pulled = items, 0

    def __iter__(self):
        for it in self.items:
            self.pulled += 1
            yield it


def test_from_log_prob_without_targets_raises_before_data():
    x, _ = _pairs(n=60)
    data = _Counting([Batch(x[i : i + 20].float()) for i in range(0, 60, 20)])

    with pytest.raises(lk.CapabilityError, match="fisher=.empirical.*needs targets"):
        lk.laplace(_mlp(), data, likelihood=BT, hyper_tuner=None)

    assert data.pulled == 1  # only the batch the probe reads


def test_predict_points_to_outputs():
    post = lk.laplace(_mlp(), _bt_loader(), likelihood=BT, hyper_tuner=None)
    x = _bt_loader()[0][0]

    for call in (
        lambda: lk.predict(post, x),
        lambda: lk.predict(post, x, lk.Linearized()),
        lambda: lk.Linearized().resolve(post),
        lambda: lk.predict(post, x, lk.WeightSamples(n_samples=2)),
    ):
        with pytest.raises(lk.CapabilityError, match=r"lk\.predict_outputs"):
            call()

    out = lk.predict_outputs(post, x)

    assert out.mean.shape == (20, 2)


def test_sampled_fisher_still_refuses_a_log_prob_only_likelihood():
    with pytest.raises(lk.CapabilityError, match="HasSample or HasDistribution"):
        lk.laplace(
            _mlp(),
            _bt_loader(),
            likelihood=BT,
            hessian_approx=DiagGGN(fisher="mc"),
            hyper_tuner=None,
        )


def _per_position(f: Tensor, y: Tensor) -> Tensor:
    """Categorical log-likelihood per position: f [N, T, C], y [N, T] -> [N, T]."""

    return torch.log_softmax(f, -1).gather(-1, y.unsqueeze(-1)).squeeze(-1)


def _mask_of(f: Tensor, inputs) -> Tensor:
    del f
    return inputs["mask"]


def test_from_log_prob_mask():
    """With `mask_fn`, garbage at masked positions leaves the EF statistic and the data-fit
    term bit-identical; without it, both change."""

    t, c = 4, 3
    torch.manual_seed(0)

    class PerPosition(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Sequential(nn.Linear(D, 6), nn.Tanh(), nn.Linear(6, c))

        def forward(self, x: Tensor, mask: Tensor) -> Tensor:
            del mask
            return self.net(x)

    model = PerPosition().double()
    fn = ModelFn.from_module(model)
    params = {k: v.detach() for k, v in model.named_parameters()}
    g = torch.Generator().manual_seed(3)
    x = torch.randn(6, t, D, generator=g, dtype=torch.float64)
    y = torch.randint(0, c, (6, t), generator=g)
    mask = torch.arange(t) < torch.tensor([1, 2, 4, 3, 4, 2]).unsqueeze(1)
    junk_x = torch.where(mask.unsqueeze(-1), x, 1e3 * torch.randn_like(x))
    junk_y = torch.where(mask, y, (y + 1) % c)

    def stat(lik, xs, ys):
        fit = lk.core.fit_curvature(fn, lik, params, DiagGGN(fisher="empirical"))
        return fit.update(fit.init(), Batch({"x": xs, "mask": mask}, ys))

    masked = from_log_prob(_per_position, mask_fn=_mask_of)
    clean, junk = stat(masked, x, y), stat(masked, junk_x, junk_y)

    assert masked.mask(x, {"x": x, "mask": mask}) is mask
    assert clean.data_fit is not None and junk.data_fit is not None
    assert torch.equal(clean.data_fit, junk.data_fit)

    for b in clean.curvature.sum:
        assert torch.equal(clean.curvature.sum[b], junk.curvature.sum[b])

    plain = from_log_prob(_per_position)

    assert plain.mask(x, {"x": x, "mask": mask}) is None

    unmasked_clean, unmasked_junk = stat(plain, x, y), stat(plain, junk_x, junk_y)

    assert not torch.equal(unmasked_clean.data_fit, unmasked_junk.data_fit)
    assert not all(
        torch.equal(unmasked_clean.curvature.sum[b], unmasked_junk.curvature.sum[b])
        for b in clean.curvature.sum
    )


def test_from_log_prob_repr_shows_mask_fn():
    assert repr(from_log_prob(_per_position)) == "from_log_prob(_per_position)"
    assert repr(from_log_prob(_per_position, mask_fn=_mask_of)) == (
        "from_log_prob(_per_position, mask_fn=_mask_of)"
    )
