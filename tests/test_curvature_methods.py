import copy
import dataclasses
import glob

import pytest
import torch
from torch import Tensor, nn
from torch.func import jacrev, vmap

import laplacekit as lk
import laplacekit.core as core
import laplacekit.curvature as curvature
import laplacekit.testing as lk_testing
from laplacekit.core import Batch, Capabilities, FitContext, Layout, ModelFn, pytree
from laplacekit.curvature import (
    DENSE_MAX_PARAMS,
    DenseGGN,
    DenseHessian,
    DiagGGN,
    default,
)
from laplacekit.likelihoods import Categorical, Gaussian
from laplacekit.priors import Isotropic

DTYPES = [(torch.float32, 1e-5), (torch.float64, 1e-10)]


def make(kind: str, dtype: torch.dtype):
    torch.manual_seed(0)
    out = 3 if kind == "cat" else 1
    model = nn.Sequential(nn.Linear(3, 8), nn.Tanh(), nn.Linear(8, out)).to(dtype)
    lik = (
        Categorical()
        if kind == "cat"
        else copy.replace(Gaussian(), sigma=torch.as_tensor(torch.tensor(0.3, dtype=dtype)))
    )
    x = torch.randn(32, 3, dtype=dtype)
    params = {k: v.detach() for k, v in model.named_parameters()}
    caps = Capabilities(vjp=True, vmap=True)
    ctx = FitContext(ModelFn.from_module(model), lik, Layout.by_module(params), caps)

    return ctx, params, x


def batches(x: Tensor, k: int = 4, weights: Tensor | None = None):
    n = x.shape[0] // k
    return [
        Batch(
            x[i * n : (i + 1) * n], None, None if weights is None else weights[i * n : (i + 1) * n]
        )
        for i in range(k)
    ]


def _gen(i):
    """The generator of batch `i` in these hand folds."""

    return torch.Generator().manual_seed(i)


def fold(method, params, ctx, bs):
    stat = method.init(params, ctx)

    for i, b in enumerate(bs):
        stat = method.update(stat, params, b, _gen(i), ctx)

    return stat


def exact_ggn(ctx, params, x):
    """Full (1/N) sum_i J_i^T H_i J_i over all blocks, flat in layout order."""

    names = [n for ns in ctx.layout.blocks.values() for n in ns]
    f = ctx.model(params, x)
    h = (
        core.output_hessian(ctx.likelihood, f, generator=torch.Generator().manual_seed(0))
        .dense()
        .reshape(x.shape[0], -1, f[0].numel())
    )

    def fi(p, xi):
        return ctx.model(p, xi.unsqueeze(0)).reshape(-1)

    jac = vmap(jacrev(fi), in_dims=(None, 0))(params, x)
    j = torch.cat([jac[n].reshape(x.shape[0], f[0].numel(), -1) for n in names], dim=2)

    return torch.einsum("nep,nef,nfq->pq", j, h, j) / x.shape[0], names


def block_slices(ctx, names):
    out, off = {}, 0

    for b in ctx.layout.blocks:
        size = ctx.layout.size(b)
        out[b] = slice(off, off + size)
        off += size

    return out


@pytest.mark.parametrize("kind", ["cat", "gauss"])
@pytest.mark.parametrize(("dtype", "tol"), DTYPES)
def test_diag_and_dense_match_exact(kind, dtype, tol):
    ctx, params, x = make(kind, dtype)
    full, names = exact_ggn(ctx, params, x)
    slices = block_slices(ctx, names)
    d = DiagGGN()
    op = d.finalize(fold(d, params, ctx, batches(x)))
    g = DenseGGN()
    dop = g.finalize(fold(g, params, ctx, batches(x)))

    assert isinstance(op, core.BlockDiag) and isinstance(dop, core.BlockDiag)

    for b, s in slices.items():
        torch.testing.assert_close(
            op.blocks[b].diagonal(), full.diagonal()[s], rtol=tol, atol=tol * 1e-2
        )
        torch.testing.assert_close(dop.blocks[b].mat, full[s, s], rtol=tol, atol=tol * 1e-2)


def test_dense_and_diag_with_a_class_axis_per_position():
    """Outputs [N, T, C], as a language model's logits: the GGN sums over positions."""

    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(3, 4), nn.Tanh(), nn.Linear(4, 5)).double()
    x = torch.randn(6, 2, 3, dtype=torch.float64)  # [N, T, D]
    params = {k: v.detach() for k, v in model.named_parameters()}
    ctx = FitContext(
        ModelFn.from_module(model),
        Categorical(),
        Layout.by_module(params),
        Capabilities(vjp=True, vmap=True),
    )
    names = list(params)

    def fi(p, xi):
        return ctx.model(p, xi.unsqueeze(0))[0]  # [T, C]

    jac = vmap(jacrev(fi), in_dims=(None, 0))(params, x)
    j = torch.cat([jac[n].reshape(6, 2, 5, -1) for n in names], dim=3)  # [N, T, C, P]
    pr = torch.softmax(ctx.model(params, x), dim=-1)
    h = torch.diag_embed(pr) - pr.unsqueeze(-1) * pr.unsqueeze(-2)  # [N, T, C, C]
    full = torch.einsum("ntcp,ntcd,ntdq->pq", j, h, j) / 6

    dense = DenseGGN().finalize(fold(DenseGGN(), params, ctx, batches(x, k=2)))
    diag = DiagGGN().finalize(fold(DiagGGN(), params, ctx, batches(x, k=2)))

    for b, s in block_slices(ctx, names).items():
        torch.testing.assert_close(dense.blocks[b].mat, full[s, s])
        torch.testing.assert_close(diag.blocks[b].diagonal(), full.diagonal()[s])


@pytest.mark.parametrize("method", [DiagGGN(), DenseGGN(), DiagGGN(fisher="mc")])
def test_monoid_law(method):
    ctx, params, x = make("cat", torch.float64)
    bs = batches(x)
    left = fold(method, params, ctx, bs[:2])
    # Same keys per batch index as the joint fold uses for bs[2:].
    stat = method.init(params, ctx)

    for i, b in enumerate(bs[2:]):
        stat = method.update(stat, params, b, _gen(i + 2), ctx)

    joint = fold(method, params, ctx, bs)
    comb = left + stat

    for k in joint.sum:
        torch.testing.assert_close(comb.sum[k], joint.sum[k], rtol=1e-12, atol=1e-12)

    torch.testing.assert_close(comb.weight, joint.weight)

    assert stat.weight == 16


def test_sampled_fisher_converges_and_reproducible():
    ctx, params, x = make("cat", torch.float64)
    full, names = exact_ggn(ctx, params, x)
    slices = block_slices(ctx, names)
    method = DiagGGN(fisher="mc")
    batch = Batch(x)
    stat = method.init(params, ctx)

    for i in range(2000):
        stat = method.update(stat, params, batch, torch.Generator().manual_seed(1000 + i), ctx)

    op = method.finalize(stat)

    for b, s in slices.items():
        got = op.blocks[b].diagonal()
        want = full.diagonal()[s]
        m = want > 1e-3

        assert ((got[m] - want[m]).abs() / want[m]).max() < 0.1

    a = method.update(
        method.init(params, ctx), params, batch, torch.Generator().manual_seed(5), ctx
    )
    c = method.update(
        method.init(params, ctx), params, batch, torch.Generator().manual_seed(5), ctx
    )

    for k in a.sum:
        torch.testing.assert_close(a.sum[k], c.sum[k])


@pytest.mark.parametrize("method", [DiagGGN(), DenseGGN(), DiagGGN(fisher="mc")])
def test_weights(method):
    ctx, params, x = make("gauss", torch.float64)
    w = torch.rand(32, dtype=torch.float64) + 0.5
    s1 = fold(method, params, ctx, batches(x, weights=w))
    s2 = fold(method, params, ctx, batches(x, weights=2 * w))
    torch.testing.assert_close(s1.total_weight, w.sum())

    assert fold(method, params, ctx, batches(x)).total_weight == 32
    o1, o2 = method.finalize(s1), method.finalize(s2)

    for b in o1.blocks:
        torch.testing.assert_close(
            o1.blocks[b].diagonal(), o2.blocks[b].diagonal(), rtol=1e-9, atol=1e-12
        )


def test_weighted_diag_matches_repeated_examples():
    ctx, params, x = make("cat", torch.float64)
    w = torch.randint(1, 4, (32,)).to(torch.float64)
    method = DiagGGN()
    weighted = method.finalize(fold(method, params, ctx, [Batch(x, None, w)]))
    rep = method.finalize(fold(method, params, ctx, [Batch(x.repeat_interleave(w.long(), 0))]))

    for b in weighted.blocks:
        torch.testing.assert_close(weighted.blocks[b].diagonal(), rep.blocks[b].diagonal())


def test_weighted_fisher_matches_weighted_ggn_in_expectation():
    ctx, params, x = make("gauss", torch.float64)
    w = torch.rand(32, dtype=torch.float64) + 0.5
    method = DiagGGN(fisher="mc")
    stat = method.init(params, ctx)

    for i in range(3000):
        stat = method.update(
            stat, params, Batch(x, None, w), torch.Generator().manual_seed(2000 + i), ctx
        )

    ref = DiagGGN()
    want = ref.finalize(fold(ref, params, ctx, [Batch(x, None, w)]))
    got = method.finalize(stat)

    for b in got.blocks:
        a, c = got.blocks[b].diagonal(), want.blocks[b].diagonal()
        m = c > 1e-3

        assert ((a[m] - c[m]).abs() / c[m]).max() < 0.1


@pytest.mark.parametrize("method", [DiagGGN(), DenseGGN(), DiagGGN(fisher="mc")])
def test_scaled(method):
    ctx, params, x = make("cat", torch.float64)
    bs = batches(x)
    a, b = fold(method, params, ctx, bs[:2]), fold(method, params, ctx, bs[2:])
    s = torch.tensor(3.0, dtype=torch.float64)
    lhs, rhs = (a + b).scaled(s), a.scaled(s) + b.scaled(s)

    for k in lhs.sum:
        torch.testing.assert_close(lhs.sum[k], rhs.sum[k], rtol=1e-12, atol=1e-12)

    torch.testing.assert_close(lhs.weight, rhs.weight)
    full = a + b
    o1, o2 = method.finalize(full), method.finalize(full.scaled(s))

    for k in o1.blocks:
        torch.testing.assert_close(
            o1.blocks[k].diagonal(), o2.blocks[k].diagonal(), rtol=1e-12, atol=1e-12
        )


@pytest.mark.parametrize("method", [DiagGGN(), DenseGGN(), DiagGGN(fisher="mc")])
def test_bad_weights(method):
    ctx, params, x = make("cat", torch.float64)
    w = torch.ones(32, dtype=torch.float64)
    w[3] = -1

    with pytest.raises(ValueError, match="non-negative"):
        method.update(
            method.init(params, ctx),
            params,
            Batch(x, None, w),
            torch.Generator().manual_seed(0),
            ctx,
        )

    with pytest.raises(ValueError, match="zero"):
        method.finalize(method.init(params, ctx))


@pytest.mark.parametrize("approx", [DiagGGN(), DenseGGN()])
def test_fit_is_differentiable_in_params(approx):
    """D8: grad of the finalized curvature's logdet matches central finite differences."""

    _, _, x = make("cat", torch.float64)

    _check_d8(approx, batches(x))


# --- per_block and the probe-driven default ----------------------------------------


def _blocks_alone(method, params, ctx, bs, blocks):
    """Run `method` on only `blocks` (their params alone), as a user would by hand."""

    names = {n for b in blocks for n in ctx.layout.blocks[b]}
    sub = {n: p for n, p in params.items() if n in names}
    sub_ctx = FitContext(ctx.model, ctx.likelihood, Layout.by_module(sub), ctx.caps)

    return method.finalize(fold(method, sub, sub_ctx, bs))


def test_per_block_equals_each_method_alone():
    ctx, params, x = make("cat", torch.float64)
    bs = batches(x)
    pb = core.per_block({"0": DenseGGN(), "*": DiagGGN()})

    assert isinstance(pb, core.PerBlockCurvature)
    op = pb.finalize(fold(pb, params, ctx, bs))
    dense = _blocks_alone(DenseGGN(), params, ctx, bs, ["0"])
    diag = _blocks_alone(DiagGGN(), params, ctx, bs, ["2"])

    assert list(op.blocks) == ["0", "2"]
    assert isinstance(op.blocks["0"], core.Dense) and isinstance(op.blocks["2"], core.Diag)
    torch.testing.assert_close(op.blocks["0"].mat, dense.blocks["0"].mat, rtol=0, atol=0)
    torch.testing.assert_close(op.blocks["2"].diag, diag.blocks["2"].diag, rtol=0, atol=0)


def test_per_block_members_see_the_other_blocks_params():
    """Params that differ from the module's: each member still evaluates the model at all
    passed params, so its block matches the exact GGN at those params."""

    ctx, params, x = make("cat", torch.float64)
    moved = {k: v + 0.3 for k, v in params.items()}
    full, names = exact_ggn(ctx, moved, x)
    slices = block_slices(ctx, names)
    pb = core.per_block({"0": DenseGGN(), "2": DiagGGN()})
    op = pb.finalize(fold(pb, moved, ctx, batches(x)))

    torch.testing.assert_close(op.blocks["0"].mat, full[slices["0"], slices["0"]])
    torch.testing.assert_close(op.blocks["2"].diag, full.diagonal()[slices["2"]])


def test_per_block_first_match_wins_and_unmatched_raises():
    ctx, params, x = make("cat", torch.float64)
    pb = core.per_block({"2": DiagGGN(), "*": DenseGGN()})
    op = pb.finalize(fold(pb, params, ctx, batches(x)))

    assert isinstance(op.blocks["2"], core.Diag) and isinstance(op.blocks["0"], core.Dense)

    with pytest.raises(core.StructureError, match=r"no rule matches.*'0'"):
        core.per_block({"2": DiagGGN()}).init(params, ctx)


def test_per_block_tuple_rule_names_exact_blocks():
    ctx, params, x = make("cat", torch.float64)
    pb = core.per_block({("0", "2"): DiagGGN()})
    op = pb.finalize(fold(pb, params, ctx, batches(x)))
    ref = core.per_block({"*": DiagGGN()})

    assert pb.by_block(["0", "2"]) == {"0": DiagGGN(), "2": DiagGGN()}

    for b in ("0", "2"):
        torch.testing.assert_close(
            op.blocks[b].diag, ref.finalize(fold(ref, params, ctx, batches(x))).blocks[b].diag
        )

    with pytest.raises(core.StructureError, match=r"don't exist: \['9'\]"):
        core.per_block({("0", "9"): DiagGGN(), "*": DenseGGN()}).init(params, ctx)

    with pytest.raises(core.StructureError, match="no rule matches block '4'"):
        pb.by_block(["4"])


def test_per_block_rule_matching_no_block_raises():
    ctx, params, _ = make("cat", torch.float64)

    with pytest.raises(core.StructureError, match=r"'fc\*'"):
        core.per_block({"fc*": DiagGGN(), "*": DenseGGN()}).init(params, ctx)


def test_per_block_shadowed_rule_raises_naming_the_earlier_rule():
    ctx, params, _ = make("cat", torch.float64)

    with pytest.raises(core.StructureError, match=r"'2'.*taken by earlier rule.*'\*'"):
        core.per_block({"*": DiagGGN(), "2": DenseGGN()}).init(params, ctx)

    with pytest.raises(core.StructureError, match=r"'2'.*taken by earlier rule"):
        core.per_block({"*": DiagGGN(), "2": _TwoPass()}).plan(params, ctx)


def test_per_block_monoid_law_and_scaled():
    ctx, params, x = make("cat", torch.float64)
    bs = batches(x)
    pb = core.per_block({"0": DenseGGN(), "*": DiagGGN()})
    right = pb.init(params, ctx)

    for i, b in enumerate(bs[2:]):
        right = pb.update(right, params, b, _gen(i + 2), ctx)

    comb = fold(pb, params, ctx, bs[:2]) + right
    joint = fold(pb, params, ctx, bs)
    s = torch.tensor(3.0, dtype=torch.float64)

    for got, want in [(comb, joint), (joint.scaled(s), joint + joint + joint)]:
        for r in want.stats:
            for k in want.stats[r].sum:
                torch.testing.assert_close(got.stats[r].sum[k], want.stats[r].sum[k])

        torch.testing.assert_close(got.total_weight, want.total_weight)


def test_per_block_is_differentiable_in_params():
    """D8 holds across members: block "0" depends on block "2"'s params through the model."""

    from torch.func import grad

    ctx, params, x = make("cat", torch.float64)
    bs = batches(x)

    def logdet(p, method):
        op = method.finalize(fold(method, p, ctx, bs))
        return op.blocks["0"].add_isotropic(torch.tensor(1.0, dtype=torch.float64)).logdet()

    got = grad(logdet)(dict(params), core.per_block({"0": DenseGGN(), "*": DiagGGN()}))
    want = grad(logdet)(dict(params), DenseGGN())

    assert float(want["2.weight"].abs().max()) > 0

    for n in params:
        torch.testing.assert_close(got[n], want[n])


def test_per_block_requires_merges_members():
    from laplacekit.curvlinops import KFAC

    req = core.per_block({"0": KFAC(), "*": DiagGGN()}).requires

    assert req.vjp and req.vmap and req.hooks == (nn.Linear, nn.Conv2d) and not req.weights


class _TwoPass:
    """A minimal staged method, to check `StagedPerBlock` assigns blocks the same way."""

    @property
    def requires(self):
        return core.Requirements()

    def plan(self, params, ctx):
        return core.chain(DiagGGN(), then=core.Done)


class _Whole:
    """A member that returns a single (non-BlockDiag) operator for its one block."""

    @property
    def requires(self):
        return core.Requirements()

    def init(self, params, ctx):
        return DiagGGN().init(params, ctx)

    def update(self, stat, params, batch, generator, ctx):
        return DiagGGN().update(stat, params, batch, generator, ctx)

    def finalize(self, stat):
        return core.Diag(torch.cat(list(stat.sum.values())) / stat.weight)


def test_per_block_member_returning_one_operator():
    ctx, params, x = make("cat", torch.float64)
    bs = batches(x)
    op = core.per_block({"0": _Whole(), "*": DiagGGN()})
    out = op.finalize(fold(op, params, ctx, bs))
    want = _blocks_alone(DiagGGN(), params, ctx, bs, ["0"]).blocks["0"]

    torch.testing.assert_close(out.blocks["0"].diag, want.diag)

    with pytest.raises(core.StructureError, match="one operator for 2 blocks"):
        two = core.per_block({"*": _Whole()})
        two.finalize(fold(two, params, ctx, bs))


def test_default_choice():
    from laplacekit.curvlinops import KFAC

    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(3, 8), nn.Tanh(), nn.Linear(8, 600), nn.Tanh())
    fn = ModelFn.from_module(model)
    caps = core.probe(fn, Gaussian(sigma=1.0), Batch(torch.randn(4, 3), torch.randn(4, 600)))
    params = {k: v.detach() for k, v in model.named_parameters()}

    assert DENSE_MAX_PARAMS == 4096
    assert caps.vjp and caps.vmap and caps.hooked == {"0", "2"}
    assert Layout.by_module(params).size("2") > DENSE_MAX_PARAMS
    got = default(params, caps)

    assert isinstance(got, core.PerBlockCurvature)
    assert got.by_block(["0", "2"]) == {"0": DenseGGN(), "2": KFAC()}
    assert repr(got) == ("PerBlockCurvature({('0',): DenseGGN(), ('2',): KFAC()})")

    # Not reached by hooks: the exact diagonal when vmap works, else the sampled Fisher.
    no_hooks = Capabilities(vjp=True, vmap=True)

    assert _chosen(params, no_hooks) == {"0": DenseGGN(), "2": DiagGGN()}
    assert default(params, Capabilities()).rules == {"*": DiagGGN(fisher="mc")}


def _chosen(params, caps, **kwargs):
    """The method `default` gives each block."""

    return default(params, caps, **kwargs).by_block(Layout.by_module(params).blocks)


def _deep(n_layers: int, width: int = 3) -> tuple[nn.Module, dict[str, Tensor]]:
    torch.manual_seed(0)
    model = nn.Sequential(*[nn.Linear(width, width) for _ in range(n_layers)]).double()

    return model, {k: v.detach() for k, v in model.named_parameters()}


def _model_calls(model: nn.Module, params, caps, lik) -> int:
    """Model calls in one `update` of `default` on one batch."""

    calls = []
    fn = ModelFn.from_module(model)

    def counted(p, x):
        calls.append(1)
        return fn(p, x)

    width = next(iter(params.values())).shape[-1]
    x = torch.randn(4, width, dtype=torch.float64)
    fit = core.fit_curvature(ModelFn(counted, model), lik, params, default(params, caps))
    fit.update(fit.init(), Batch(x, torch.randn(4, width, dtype=torch.float64)))

    return len(calls)


@pytest.mark.parametrize(
    "caps", [Capabilities(), Capabilities(vjp=True, vmap=True)], ids=["sampled", "dense-ggn"]
)
def test_default_runs_each_method_once_per_batch(caps):
    """One member per chosen method: the model calls per batch don't grow with the blocks."""

    lik = Gaussian(sigma=1.0)
    few, many = _deep(2), _deep(8)

    assert len(default(many[1], caps).rules) == 1
    assert _model_calls(*few, caps, lik) == _model_calls(*many, caps, lik)


def test_default_groups_give_the_per_block_precision():
    """Two methods (DenseGGN on small blocks, DiagGGN on large ones), each run once over its
    blocks, equal the same methods run block by block."""

    torch.manual_seed(0)
    widths = [3, 70, 70, 3, 70, 70, 3]  # 70 x 70 blocks are above DENSE_MAX_PARAMS
    layers = [nn.Linear(a, b) for a, b in zip(widths, widths[1:], strict=False)]
    model = nn.Sequential(*layers).double()
    params = {k: v.detach() for k, v in model.named_parameters()}
    caps = Capabilities(vjp=True, vmap=True)
    grouped = default(params, caps)
    blocks = Layout.by_module(params).blocks
    chosen = grouped.by_block(blocks)

    assert {type(c) for c in chosen.values()} == {DenseGGN, DiagGGN}
    assert len(grouped.rules) == 2

    one_each = core.per_block({glob.escape(b): chosen[b] for b in blocks})
    lik = Gaussian(sigma=1.0)
    fn = ModelFn.from_module(model)
    x = torch.randn(6, 3, dtype=torch.float64)
    ctx = FitContext(fn, lik, Layout.by_module(params), caps)
    a = grouped.finalize(fold(grouped, params, ctx, batches(x, 2)))
    b = one_each.finalize(fold(one_each, params, ctx, batches(x, 2)))

    for name in blocks:
        v = torch.randn(Layout.by_module(params).size(name), dtype=torch.float64)

        assert type(a.blocks[name]) is type(b.blocks[name])
        torch.testing.assert_close(a.blocks[name].mv(v), b.blocks[name].mv(v))


def test_default_skips_kfac_for_a_bias_only_block():
    hooked = Capabilities(vjp=True, vmap=True, hooked_types={"fc": nn.Linear})
    no_vmap = Capabilities(hooked_types={"fc": nn.Linear})
    bias = {"fc.bias": torch.zeros(DENSE_MAX_PARAMS + 1)}

    assert _chosen(bias, hooked) == {"fc": DiagGGN()}
    assert _chosen(bias, no_vmap) == {"fc": DiagGGN(fisher="mc")}


def test_default_escapes_glob_characters():
    params = {"a[0].weight": torch.zeros(2, 2), "a0.weight": torch.zeros(2, 2)}
    caps = Capabilities(vjp=True, vmap=True)
    ctx = FitContext(ModelFn(lambda p, x: x), Gaussian(sigma=1.0), Layout.by_module(params), caps)
    big = {"a[0].weight": torch.zeros(65, 65), "a0.weight": torch.zeros(2, 2)}
    ctx = FitContext(ModelFn(lambda p, x: x), Gaussian(sigma=1.0), Layout.by_module(big), caps)
    stat = default(big, caps).init(big, ctx)

    # Names are matched exactly, never as globs: "a[0]" is not the class "[0]".
    assert sorted(blocks for _, blocks, _ in stat.members) == [("a0",), ("a[0]",)]


@pytree
class _Custom:
    """A likelihood KFAC knows nothing about."""

    def distribution(self, f):
        return torch.distributions.Normal(f, 1.0)


def _hooked_mlp():
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(3, 8), nn.Tanh(), nn.Linear(8, 600), nn.Tanh())
    fn = ModelFn.from_module(model)
    caps = core.probe(fn, Gaussian(sigma=1.0), Batch(torch.randn(4, 3), torch.randn(4, 600)))
    params = {k: v.detach() for k, v in model.named_parameters()}

    return params, caps


@pytest.mark.parametrize(
    "lik",
    [
        _Custom(),
        Categorical(class_dim=1),
        None,
    ],
    ids=["custom", "class-dim", "weighted"],
)
def test_default_skips_kfac_when_kfac_cannot_fit(lik):
    from laplacekit.curvlinops import KFAC

    params, caps = _hooked_mlp()
    weighted = lik is None
    got = default(
        params, caps, likelihood=Gaussian(sigma=1.0) if weighted else lik, weighted=weighted
    )

    fisher = "exact" if lik is None or isinstance(lik, core.HasOutputHessian) else "mc"

    assert got.by_block(["0", "2"]) == {"0": DenseGGN(fisher=fisher), "2": DiagGGN(fisher=fisher)}
    assert _chosen(params, caps, likelihood=Gaussian(sigma=1.0))["2"] == KFAC()
    assert _chosen(params, caps)["2"] == KFAC()  # no likelihood: as before


def test_kfac_supports_is_what_init_checks():
    from laplacekit.curvlinops import KFAC

    assert KFAC.supports(Categorical()) and KFAC.supports(Gaussian(sigma=1.0))
    assert KFAC.supports(copy.replace(Gaussian(), sigma=torch.as_tensor(torch.tensor(0.3))))

    params, caps = _hooked_mlp()
    fn = ModelFn.from_module(nn.Sequential(nn.Linear(3, 8), nn.Tanh(), nn.Linear(8, 600)))

    def init(lik):
        ctx = FitContext(fn, lik, Layout.by_module(params), caps)
        return KFAC().init(params, ctx)

    for lik in (
        Categorical(),
        Gaussian(sigma=1.0),
        copy.replace(Gaussian(), sigma=torch.as_tensor(torch.tensor(0.3))),
    ):
        init(lik)

    for lik in (_Custom(), Categorical(class_dim=1)):
        assert not KFAC.supports(lik)

        with pytest.raises(core.CapabilityError):
            init(lik)


class _Spy:
    """DiagGGN(fisher="mc") that records the seeds of the generators it gets."""

    def __init__(self, seen):
        self.seen = seen

    @property
    def requires(self):
        return DiagGGN(fisher="mc").requires

    def init(self, params, ctx):
        return DiagGGN(fisher="mc").init(params, ctx)

    def update(self, stat, params, batch, generator, ctx):
        self.seen.append(generator.initial_seed())
        return DiagGGN(fisher="mc").update(stat, params, batch, generator, ctx)

    def finalize(self, stat):
        return DiagGGN(fisher="mc").finalize(stat)


def test_per_block_members_get_independent_generators():
    ctx, params, x = make("cat", torch.float64)
    bs = batches(x)
    a, b = [], []
    pb = core.per_block({"0": _Spy(a), "2": _Spy(b)})
    op = pb.finalize(fold(pb, params, ctx, bs))

    assert len(a) == len(b) == len(bs) and not set(a) & set(b)

    # One sampler over both blocks shares its draws; the members' draws differ from it.
    shared = DiagGGN(fisher="mc").finalize(fold(DiagGGN(fisher="mc"), params, ctx, bs))

    assert not torch.allclose(op.blocks["2"].diag, shared.blocks["2"].diag)

    # Deterministic: the same fold twice agrees.
    again = pb.finalize(fold(pb, params, ctx, bs))
    torch.testing.assert_close(op.blocks["0"].diag, again.blocks["0"].diag, rtol=0, atol=0)


# --- large outputs ---------------------------------------------------------------------


def test_probe_records_the_event_size():
    model = nn.Sequential(nn.Linear(3, 5))
    caps = core.probe(ModelFn.from_module(model), Categorical(), Batch(torch.randn(4, 3)))

    assert caps.event_size == 5

    seq = nn.Sequential(nn.Linear(3, 7))
    caps = core.probe(
        ModelFn.from_module(seq), Categorical(), Batch(torch.randn(4, 6, 3))
    )  # [N, T, C] logits

    assert caps.event_size == 6 * 7


def _big_event(event: int) -> Capabilities:
    return Capabilities(
        vjp=True,
        vmap=True,
        hooked_types={"fc": nn.Linear},
        event_size=event,
    )


def test_default_uses_sampled_methods_above_max_dense_event():
    from laplacekit.curvlinops import KFAC

    params = {
        "fc.weight": torch.zeros(3, 2),
        "fc.bias": torch.zeros(3),
        "emb.weight": torch.zeros(2, 2),
    }
    big = core.MAX_DENSE_EVENT + 1

    # Small blocks would get the dense GGN, which builds an [N, E, E] output Hessian.
    assert _chosen(params, _big_event(big)) == {
        "fc": KFAC(fisher="mc"),
        "emb": DiagGGN(fisher="mc"),
    }
    assert _chosen(params, _big_event(core.MAX_DENSE_EVENT)) == {
        "fc": DenseGGN(),
        "emb": DenseGGN(),
    }

    # vjp without vmap: neither exact method.
    vjp_only = Capabilities(vjp=True, hooked_types={"fc": nn.Linear})

    assert _chosen(params, vjp_only) == {"fc": KFAC(), "emb": DiagGGN(fisher="mc")}

    # KFAC can't take weights: the sampled diagonal everywhere.
    assert _chosen(params, _big_event(big), weighted=True) == {
        "fc": DiagGGN(fisher="mc"),
        "emb": DiagGGN(fisher="mc"),
    }


class _Repeat(nn.Module):
    """Three logits, tiled to many classes: a large output from a tiny layer."""

    def __init__(self, times: int) -> None:
        super().__init__()
        self.fc = nn.Linear(2, 3)
        self.times = times

    def forward(self, x: Tensor) -> Tensor:
        return self.fc(x).repeat(1, self.times)


def test_front_door_picks_sampled_kfac_for_a_large_output():
    import laplacekit as lk

    torch.manual_seed(0)
    model = _Repeat(1500)
    x = torch.randn(8, 2)
    y = torch.randint(0, 4500, (8,))
    post = lk.laplace(
        model, [(x[:4], y[:4]), (x[4:], y[4:])], likelihood=Categorical(), hyper_tuner=None
    )

    assert "PerBlockCurvature({'*': KFAC(fisher='mc')})" in repr(post)


class _HeteroGaussian:
    """Only `log_prob` and `distribution`: f = [mu, log_sigma] per example."""

    def log_prob(self, f, y, /):
        return self.distribution(f).log_prob(y)

    def distribution(self, f, /):
        return torch.distributions.Normal(f[:, 0], f[:, 1].exp())


class _LogProbOnly:
    def log_prob(self, f, y, /):
        return -((f - y) ** 2).sum(-1)


def _hetero(n: int = 6):
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(3, 4), nn.Tanh(), nn.Linear(4, 2)).double()
    params = {k: v.detach() for k, v in model.named_parameters()}
    caps = Capabilities(vjp=True, vmap=True)
    ctx = FitContext(ModelFn.from_module(model), _HeteroGaussian(), Layout.by_module(params), caps)
    x = torch.randn(n, 3, dtype=torch.float64)
    y = torch.randn(n, dtype=torch.float64)

    return ctx, params, x, y


def test_output_hessian_options_declare_their_needs():
    assert DiagGGN() == DiagGGN(fisher="exact")
    exact = DiagGGN().requires

    assert exact.needs_closed_form_hessian and not exact.needs_targets
    assert DenseGGN().requires.needs_closed_form_hessian

    for method in (DiagGGN(observed=True), DenseGGN(observed=True)):
        assert method.requires.needs_targets and not method.requires.needs_closed_form_hessian

    for method in (DiagGGN(fisher="mc"), DenseGGN(fisher="mc"), lk.LowRankGGN(2, fisher="mc")):
        assert method.requires.needs_sample and not method.requires.needs_closed_form_hessian

    assert DiagGGN(fisher="empirical").requires.needs_targets
    assert "mc_samples=4" in repr(DenseGGN(fisher="mc", mc_samples=4))


@pytest.mark.parametrize(
    "make",
    [
        lambda: DiagGGN(mc_samples=2),
        lambda: DenseGGN(fisher="empirical", mc_samples=2),
        lambda: lk.LowRankGGN(2, mc_samples=2),
        lambda: lk.KFAC(mc_samples=2),
        lambda: DiagGGN(fisher="mc", observed=True),
        lambda: DiagGGN(fisher="fisher"),  # ty: ignore[invalid-argument-type]
        lambda: lk.LowRankGGN(2, fisher="empirical"),  # ty: ignore[invalid-argument-type]
    ],
)
def test_unused_or_unknown_fisher_options_are_refused(make):
    with pytest.raises(ValueError, match="fisher"):
        make()


def test_exact_without_a_closed_form_names_the_mc_fisher():
    """No silent switch from the exact output Hessian to sampling."""

    with pytest.raises(core.CapabilityError, match='fisher="mc"'):
        core.check_likelihood(DiagGGN().requires, _Custom(), "DiagGGN()")


def test_check_likelihood():
    ctx, params, _, _ = _hetero()
    method = DiagGGN(fisher="mc")

    with pytest.raises(core.CapabilityError, match=r"DiagGGN\(fisher='mc'\).*HasSample"):
        core.fit_curvature(ctx.model, _LogProbOnly(), params, method)

    with pytest.raises(core.CapabilityError, match=r"DenseGGN\(.*HasOutputHessian"):
        core.fit_curvature(
            ctx.model, _LogProbOnly(), params, core.per_block({"0": DenseGGN(), "*": DiagGGN()})
        )

    core.fit_curvature(ctx.model, _LogProbOnly(), params, DiagGGN(observed=True))
    core.fit_curvature(ctx.model, _HeteroGaussian(), params, method)


def _observed_ggn(ctx, params, x, y):
    """(1/N) sum_i J_i^T H_i J_i with H_i the Hessian of -log p(y_i | f) at f_i, by torch.func."""

    def nll(fi, yi):
        return fi[1] + 0.5 * ((yi - fi[0]) / fi[1].exp()) ** 2

    names = [n for ns in ctx.layout.blocks.values() for n in ns]
    f = ctx.model(params, x)
    h = torch.stack([torch.func.hessian(nll)(fi, yi) for fi, yi in zip(f, y, strict=True)])

    def fi(p, xi):
        return ctx.model(p, xi.unsqueeze(0)).reshape(-1)

    jac = vmap(jacrev(fi), in_dims=(None, 0))(params, x)
    j = torch.cat([jac[n].reshape(x.shape[0], 2, -1) for n in names], dim=2)

    return torch.einsum("nep,nef,nfq->pq", j, h, j) / x.shape[0], names


def test_observed_ggn_matches_reference():
    ctx, params, x, y = _hetero()
    full, names = _observed_ggn(ctx, params, x, y)
    slices = block_slices(ctx, names)
    bs = [Batch(x[:3], y[:3]), Batch(x[3:], y[3:])]

    dense = DenseGGN(observed=True)
    diag = DiagGGN(observed=True)
    got_dense = dense.finalize(fold(dense, params, ctx, bs))
    got_diag = diag.finalize(fold(diag, params, ctx, bs))

    # The observed Hessian of this likelihood is indefinite, so the GGN can be too.
    assert float(torch.linalg.eigvalsh(full)[0]) < -1e-6

    for b, s in slices.items():
        torch.testing.assert_close(got_dense.blocks[b].mat, full[s, s])
        torch.testing.assert_close(got_diag.blocks[b].diagonal(), full.diagonal()[s])


def test_observed_needs_targets_even_without_caps():
    ctx, params, x, _ = _hetero()
    method = DiagGGN(observed=True)

    with pytest.raises(core.CapabilityError, match="targets"):
        method.update(
            method.init(params, ctx), params, Batch(x), torch.Generator().manual_seed(0), ctx
        )


def test_mc_fisher_diag_and_dense_agree_and_use_the_generator():
    ctx, params, x, _ = _hetero()
    bs = batches(x, k=2)
    # The per-example MC Fisher diagonal, against the dense one with the same draws.
    diag = lk.curvature._DiagJacobian(output_hessian="mc", mc_samples=3)
    dense = DenseGGN(fisher="mc", mc_samples=3)
    a = diag.finalize(fold(diag, params, ctx, bs))
    b = dense.finalize(fold(dense, params, ctx, bs))

    for k in a.blocks:
        torch.testing.assert_close(a.blocks[k].diagonal(), b.blocks[k].diagonal())

    other = diag.update(
        diag.init(params, ctx), params, bs[0], torch.Generator().manual_seed(9), ctx
    )
    same = diag.update(diag.init(params, ctx), params, bs[0], _gen(0), ctx)
    first = fold(diag, params, ctx, bs[:1])

    assert not torch.allclose(other.sum["2"], first.sum["2"])
    torch.testing.assert_close(same.sum["2"], first.sum["2"])


@pytest.mark.parametrize("method", [DiagGGN, DenseGGN])
def test_mc_fisher_on_a_large_event_fails_at_init(method):
    """The MC Fisher builds a dense [E, E] block per example: refused before any data."""

    params = {"fc.weight": torch.zeros(3, 2)}
    big = Capabilities(vjp=True, vmap=True, event_size=core.MAX_DENSE_EVENT + 1)

    def ctx(lik, caps=big):
        return FitContext(ModelFn(lambda p, x: x), lik, Layout.by_module(params), caps)

    if method is DenseGGN:  # DiagGGN(fisher="mc") takes plain backward passes instead
        with pytest.raises(core.StructureError, match=r"_Custom an output_hessian method"):
            method(fisher="mc").init(params, ctx(_Custom()))

    with pytest.raises(core.StructureError, match="observed"):
        method(observed=True).init(params, ctx(Categorical()))

    # The closed form, or a small event, is fine.
    method().init(params, ctx(Categorical()))
    method().init(params, ctx(_Custom(), Capabilities(vjp=True, vmap=True, event_size=4)))


@pytest.mark.parametrize("method", [DiagGGN, DenseGGN])
def test_mc_samples_must_be_positive(method):
    with pytest.raises(ValueError, match="mc_samples"):
        method(mc_samples=0)


def test_default_skips_kfac_on_selection():
    from laplacekit.curvlinops import KFAC

    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(3, 8), nn.Tanh(), nn.Linear(8, 600), nn.Tanh())
    batch = Batch(torch.randn(4, 3), torch.randn(4, 600))
    lik = Gaussian(sigma=1.0)

    # The whole weight of block "2" with some of its bias: the block holds a selection.
    params = dict(model.named_parameters())
    sel = core.Selection.from_masks(params, {"2.bias": torch.arange(600) % 2 == 0})
    fn = sel.apply(ModelFn.from_module(model))
    chosen = {"2.weight": params["2.weight"].detach(), **sel.params}
    caps = core.probe(fn, lik, batch, params=chosen)

    assert caps.hooked == {"0", "2"} and caps.selected_blocks == {"2"}
    assert Layout.by_module(chosen).size("2") > DENSE_MAX_PARAMS
    assert _chosen(chosen, caps) == {"2": DiagGGN()}

    # The same block of whole tensors gets KFAC.
    whole = {"2.weight": params["2.weight"].detach(), "2.bias": params["2.bias"].detach()}
    caps = core.probe(ModelFn.from_module(model), lik, batch, params=whole)

    assert _chosen(whole, caps) == {"2": KFAC()}


def test_default_picks_no_hook_method_next_to_a_selection():
    """Whole blocks beside a selection get the non-hook choice for their size: the hook
    methods are refused for every block once anything is selected."""

    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(100, 50), nn.Tanh(), nn.Linear(50, 3))
    x, y = torch.randn(8, 100), torch.randint(0, 3, (8,))
    lik = Categorical()

    params = dict(model.named_parameters())
    mask = torch.zeros(3, 50, dtype=torch.bool)
    mask[0] = True
    sel = core.Selection.from_masks(params, {"2.weight": mask})
    fn = sel.apply(ModelFn.from_module(model))
    chosen = {"0.weight": params["0.weight"].detach(), "0.bias": params["0.bias"].detach()}
    chosen = {**chosen, **sel.params}
    caps = core.probe(fn, lik, Batch(x, y), params=chosen)

    assert caps.selected_blocks == {"2"} and Layout.by_module(chosen).size("0") > DENSE_MAX_PARAMS
    approx = default(chosen, caps, likelihood=lik)

    assert approx.requires.hooks == ()
    assert _chosen(chosen, caps, likelihood=lik) == {"0": DiagGGN(), "2": DenseGGN()}

    fit = core.fit_curvature(
        fn, lik, chosen, approx, generator=torch.Generator().manual_seed(0), capabilities=caps
    )
    step = fit.start()
    stat = step.update(step.init(), Batch(x, y))

    assert isinstance(step.finish(stat), core.Finished)


# --- the joint dense GGN and the dense empirical Fisher ----------------------------


def _targets(ctx, x: Tensor) -> Tensor:
    g = torch.Generator().manual_seed(3)

    if isinstance(ctx.likelihood, Categorical):
        return torch.randint(0, 3, (x.shape[0],), generator=g)

    return torch.randn(x.shape[0], 1, generator=g, dtype=x.dtype)


def _labelled(x: Tensor, y: Tensor, w: Tensor | None = None, k: int = 4) -> list[Batch]:
    n = x.shape[0] // k
    parts = [slice(i * n, (i + 1) * n) for i in range(k)]

    return [Batch(x[s], y[s], None if w is None else w[s]) for s in parts]


@pytest.mark.parametrize("kind", ["cat", "gauss"])
def test_dense_ggn_joint_matches_jacrev(kind):
    ctx, params, x = make(kind, torch.float64)
    full, _ = exact_ggn(ctx, params, x)
    method = DenseGGN(joint=True)
    op = method.finalize(fold(method, params, ctx, batches(x)))
    sizes = tuple((b, ctx.layout.size(b)) for b in ctx.layout.blocks)

    assert isinstance(op, core.Joint) and isinstance(op.op, core.Dense)
    assert op.sizes == sizes

    torch.testing.assert_close(op.op.mat, full, rtol=1e-10, atol=1e-12)

    # The cross-block entries are not zero.
    s = block_slices(ctx, None)

    assert float(full[s["0"], s["2"]].abs().max()) > 1e-3


def _ef_reference(ctx, params, x, y, w):
    """(1/W) sum_i w_i g_i g_i^T over all params in layout order, from a loop of
    `torch.autograd.grad`."""

    names = [n for ns in ctx.layout.blocks.values() for n in ns]
    p = {k: v.clone().requires_grad_() for k, v in params.items()}
    rows = []

    for i in range(x.shape[0]):
        f = ctx.model(p, x[i : i + 1])
        lp = core.log_prob(ctx.likelihood, f, y[i : i + 1]).sum()
        g = torch.autograd.grad(lp, [p[n] for n in names])
        rows.append(torch.cat([t.reshape(-1) for t in g]))

    g = torch.stack(rows)

    return torch.einsum("n,np,nq->pq", w, g, g) / w.sum()


@pytest.mark.parametrize("kind", ["cat", "gauss"])
@pytest.mark.parametrize("vmapped", [True, False], ids=["vmap", "loop"])
def test_dense_ef_matches_per_example_grads(kind, vmapped):
    ctx, params, x = make(kind, torch.float64)
    ctx = FitContext(ctx.model, ctx.likelihood, ctx.layout, Capabilities(vmap=vmapped))
    y = _targets(ctx, x)
    w = torch.rand(x.shape[0], generator=torch.Generator().manual_seed(4), dtype=x.dtype) + 0.5
    want = _ef_reference(ctx, params, x, y, w)
    bs = _labelled(x, y, w)

    joint = DenseGGN(fisher="empirical", joint=True)
    op = joint.finalize(fold(joint, params, ctx, bs))

    assert isinstance(op, core.Joint)

    torch.testing.assert_close(op.op.mat, want, rtol=1e-10, atol=1e-12)

    per = DenseGGN(fisher="empirical")
    blocks = per.finalize(fold(per, params, ctx, bs))

    assert isinstance(blocks, core.BlockDiag)

    for b, s in block_slices(ctx, None).items():
        torch.testing.assert_close(blocks.blocks[b].mat, want[s, s], rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("weighted", [False, True], ids=["plain", "weighted"])
@pytest.mark.parametrize(
    "method",
    [DenseGGN(joint=True), DenseGGN(fisher="empirical"), DenseGGN(fisher="empirical", joint=True)],
    ids=repr,
)
def test_joint_and_dense_ef_pass_check_curvature(method, weighted):
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(3, 5), nn.Tanh(), nn.Linear(5, 3)).double()
    params = {k: v.detach() for k, v in model.named_parameters()}
    g = torch.Generator().manual_seed(1)
    x = torch.randn(24, 3, generator=g, dtype=torch.float64)
    y = torch.randint(0, 3, (24,), generator=g)
    w = torch.rand(24, generator=g, dtype=torch.float64) + 0.5 if weighted else None

    lk_testing.check_curvature(method, model, Categorical(), params, _labelled(x, y, w))


def _check_d8(approx, bs):
    """Grad of the finalized curvature's logdet matches central finite differences."""

    from torch.func import grad

    from laplacekit.core import fit_curvature

    ctx, params, _ = make("cat", torch.float64)
    names = list(params)

    def logdet(flat):
        p = {n: flat[n] for n in names}
        # One seed for every call, so a sketch draws the same test matrix each time.
        gen = torch.Generator().manual_seed(0)
        fit = fit_curvature(ctx.model, ctx.likelihood, p, approx, generator=gen)
        stat = fit.init()

        for b in bs:
            stat = fit.update(stat, b)

        return fit.finalize(stat).add_isotropic(torch.tensor(1.0, dtype=torch.float64)).logdet()

    g = grad(logdet)(dict(params))

    assert any(float(v.abs().max()) > 0 for v in g.values())
    eps = 1e-6

    for n in names:
        flat = params[n].reshape(-1)

        for i in range(min(3, flat.numel())):

            def shifted(d, n=n, i=i):
                q = {k: v.clone() for k, v in params.items()}
                q[n].view(-1)[i] += d

                return logdet(q)

            fd = (shifted(eps) - shifted(-eps)) / (2 * eps)

            assert torch.allclose(g[n].reshape(-1)[i], fd, rtol=1e-5, atol=1e-8)


def test_joint_dense_ggn_d8():
    _, _, x = make("cat", torch.float64)

    _check_d8(DenseGGN(joint=True), batches(x))


@pytest.mark.parametrize(
    "method", [DenseGGN(fisher="empirical"), DenseGGN(fisher="empirical", joint=True)], ids=repr
)
def test_dense_ef_d8(method):
    ctx, _, x = make("cat", torch.float64)

    _check_d8(method, _labelled(x, _targets(ctx, x)))


@pytest.mark.parametrize(
    "method", [DenseGGN(joint=True), DenseGGN(fisher="empirical", joint=True)], ids=repr
)
def test_joint_method_as_a_per_block_member_raises(method):
    ctx, params, x = make("cat", torch.float64)
    pb = core.per_block({"0": method, "*": DiagGGN()})
    bs = _labelled(x, _targets(ctx, x))

    with pytest.raises(core.StructureError, match="hessian_approx= directly"):
        pb.finalize(fold(pb, params, ctx, bs))


def test_dense_ef_requires_and_exports():

    assert DenseGGN(fisher="empirical").requires == core.Requirements(
        needs_targets=True, late_scale=False
    )
    assert (
        DenseGGN(fisher="empirical", joint=True).requires == DenseGGN(fisher="empirical").requires
    )
    assert repr(DenseGGN(joint=True)) == ("DenseGGN(joint=True)")
    assert repr(DenseGGN(fisher="empirical")) == "DenseGGN(fisher='empirical')"


def test_dense_ef_needs_targets():
    ctx, params, x = make("cat", torch.float64)
    method = DenseGGN(fisher="empirical")

    with pytest.raises(
        core.CapabilityError, match=r"DenseGGN\(fisher=\"empirical\"\) needs targets"
    ):
        method.update(
            method.init(params, ctx), params, Batch(x), torch.Generator().manual_seed(0), ctx
        )


def test_joint_stats_over_different_blocks_do_not_add():
    ctx, params, x = make("cat", torch.float64)
    method = DenseGGN(joint=True)
    whole = method.init(params, ctx)
    last = {k: v for k, v in params.items() if k.startswith("2.")}
    part = method.init(
        last, FitContext(ctx.model, ctx.likelihood, Layout.by_module(last), ctx.caps)
    )

    with pytest.raises(core.StructureError, match="different blocks"):
        _ = whole + part


# --- the exact Hessian ----------------------------------------------------------------


def _hessian_reference(ctx, params, x, y, w, sigma):
    """(1/W) Hess_theta sum_i w_i (-log p(y_i | f(x_i; theta))) over all params in layout
    order, from `torch.func.hessian` of an explicit loss over one flat vector."""

    names = [n for ns in ctx.layout.blocks.values() for n in ns]
    sizes = [params[n].numel() for n in names]
    theta = torch.cat([params[n].reshape(-1) for n in names])

    def loss(t):
        p = {
            n: part.reshape(params[n].shape) for n, part in zip(names, t.split(sizes), strict=True)
        }
        f = ctx.model(p, x)

        if isinstance(ctx.likelihood, Categorical):
            nll = nn.functional.cross_entropy(f, y, reduction="none")
        else:
            nll = 0.5 * ((f - y) ** 2).sum(1) / sigma**2

        return (w * nll).sum()

    return torch.func.hessian(loss)(theta) / w.sum()


@pytest.mark.parametrize("kind", ["cat", "gauss"])
def test_dense_hessian_matches_torch_func_hessian(kind):
    ctx, params, x = make(kind, torch.float64)
    y = _targets(ctx, x)
    w = torch.rand(x.shape[0], generator=torch.Generator().manual_seed(4), dtype=x.dtype) + 0.5
    want = _hessian_reference(ctx, params, x, y, w, 0.3)
    bs = _labelled(x, y, w)

    joint = DenseHessian(joint=True)
    op = joint.finalize(fold(joint, params, ctx, bs))

    assert isinstance(op, core.Joint) and isinstance(op.op, core.Dense)
    assert op.sizes == tuple((b, ctx.layout.size(b)) for b in ctx.layout.blocks)

    torch.testing.assert_close(op.op.mat, want, rtol=1e-10, atol=1e-12)

    per = DenseHessian()
    blocks = per.finalize(fold(per, params, ctx, bs))

    assert isinstance(blocks, core.BlockDiag)

    for b, s in block_slices(ctx, None).items():
        torch.testing.assert_close(blocks.blocks[b].mat, want[s, s], rtol=1e-10, atol=1e-12)

    # Away from a minimum the Hessian is not the GGN (both unweighted here).
    ggn, _ = exact_ggn(ctx, params, x)
    unweighted = _hessian_reference(ctx, params, x, y, torch.ones_like(w), 0.3)

    assert float((unweighted - ggn).abs().max()) > 1e-3


def test_dense_hessian_late_sigma_is_linear():
    ctx, params, x = make("gauss", torch.float64)
    y = _targets(ctx, x)
    ones = torch.ones(x.shape[0], dtype=x.dtype)
    lam = torch.tensor(50.0, dtype=x.dtype)
    unit = _hessian_reference(ctx, params, x, y, ones, 1.0) * x.shape[0]
    want = unit / 0.09 + lam * torch.eye(unit.shape[0], dtype=x.dtype)

    fit = core.fit_curvature(
        ctx.model,
        copy.replace(Gaussian(), sigma=torch.as_tensor(torch.tensor(0.3, dtype=x.dtype))),
        params,
        DenseHessian(joint=True),
    )
    stat = fit.update(fit.init(), Batch(x, y))
    post = fit.posterior(fit.finalize(stat), prior=Isotropic(lam), n_data=stat.count)
    prec = post.precision()

    assert isinstance(prec, core.Joint)

    torch.testing.assert_close(prec.op.mat, want, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("weighted", [False, True], ids=["plain", "weighted"])
@pytest.mark.parametrize("method", [DenseHessian(), DenseHessian(joint=True)], ids=repr)
def test_dense_hessian_passes_check_curvature(method, weighted):
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(3, 5), nn.Tanh(), nn.Linear(5, 3)).double()
    params = {k: v.detach() for k, v in model.named_parameters()}
    g = torch.Generator().manual_seed(1)
    x = torch.randn(24, 3, generator=g, dtype=torch.float64)
    y = torch.randint(0, 3, (24,), generator=g)
    w = torch.rand(24, generator=g, dtype=torch.float64) + 0.5 if weighted else None

    lk_testing.check_curvature(method, model, Categorical(), params, _labelled(x, y, w))


@pytest.mark.parametrize("method", [DenseHessian(), DenseHessian(joint=True)], ids=repr)
def test_dense_hessian_d8(method):
    ctx, _, x = make("cat", torch.float64)

    _check_d8(method, _labelled(x, _targets(ctx, x)))


def test_dense_hessian_requires_and_exports():
    assert DenseHessian().requires == core.Requirements(
        vjp=True, vjp_twice=True, vmap=True, needs_targets=True
    )
    assert DenseHessian().requires.late_scale
    assert DenseHessian(joint=True).requires == DenseHessian().requires
    assert lk.DenseHessian is DenseHessian
    assert repr(DenseHessian()) == "DenseHessian()"


def test_dense_hessian_needs_targets():
    ctx, params, x = make("cat", torch.float64)
    method = DenseHessian()

    with pytest.raises(core.CapabilityError, match="DenseHessian needs targets"):
        method.update(
            method.init(params, ctx), params, Batch(x), torch.Generator().manual_seed(0), ctx
        )


def test_dense_hessian_indefinite_precision_raises_structure_error():
    """Far from a minimum the Hessian has negative eigenvalues; with a small prior
    precision the posterior precision is indefinite, and the error names the remedy."""

    ctx, params, x = make("gauss", torch.float64)
    y = 10 * _targets(ctx, x)
    want = _hessian_reference(ctx, params, x, y, torch.ones(x.shape[0], dtype=x.dtype), 0.3)
    lam = torch.tensor(1e-3, dtype=x.dtype)

    assert float(torch.linalg.eigvalsh(want).min()) * x.shape[0] < -1

    fit = core.fit_curvature(ctx.model, ctx.likelihood, params, DenseHessian(joint=True))
    stat = fit.update(fit.init(), Batch(x, y))
    post = fit.posterior(fit.finalize(stat), prior=Isotropic(lam), n_data=stat.count)

    with pytest.raises(core.StructureError, match="not positive definite.*GGN"):
        post.precision().logdet()


class _DetachedBackward(torch.autograd.Function):
    """sin, with a correct VJP and JVP, but a backward that is not differentiable."""

    generate_vmap_rule = True

    @staticmethod
    def forward(x):
        return torch.sin(x)

    @staticmethod
    def setup_context(ctx, inputs, output):
        ctx.save_for_backward(inputs[0])
        ctx.save_for_forward(inputs[0])

    @staticmethod
    def backward(ctx, g):
        (x,) = ctx.saved_tensors

        return g.detach() * torch.cos(x)

    @staticmethod
    def jvp(ctx, t):
        (x,) = ctx.saved_tensors

        return t * torch.cos(x)


class _ItemBackward(torch.autograd.Function):
    """sin, with a differentiable backward that calls `.item()`, so it fails under vmap."""

    generate_vmap_rule = True

    @staticmethod
    def forward(x):
        return torch.sin(x)

    @staticmethod
    def setup_context(ctx, inputs, output):
        ctx.save_for_backward(inputs[0])

    @staticmethod
    def backward(ctx, g):
        (x,) = ctx.saved_tensors
        _ = g.detach().sum().item()

        return g * torch.cos(x)


class _WithFunction(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        self.a = nn.Linear(2, 3)
        self.b = nn.Linear(3, 1)

    def forward(self, x):
        return self.b(self.fn.apply(self.a(x)))


class _ProbeBatchOnly:
    """The probe's batch, then a batch that fails if it is read."""

    def __init__(self, batch):
        self.batch = batch

    def __iter__(self):
        yield self.batch

        raise AssertionError("a batch after the probe's was read")


@pytest.mark.parametrize(
    ("fn", "missing"),
    [(_DetachedBackward, "vjp_twice"), (_ItemBackward, "vmap")],
    ids=["no-double-backward", "no-vmap"],
)
@pytest.mark.parametrize("joint", [False, True], ids=["per-block", "joint"])
def test_dense_hessian_refuses_before_data(fn, missing, joint):
    torch.manual_seed(0)
    model = _WithFunction(fn).double()
    g = torch.Generator().manual_seed(1)
    x = torch.randn(8, 2, generator=g, dtype=torch.float64)
    y = torch.randn(8, 1, generator=g, dtype=torch.float64)
    batch = Batch(x, y)
    caps = core.probe(ModelFn.from_module(model), Gaussian(), batch)

    assert not getattr(caps, missing)
    assert caps.vjp and (caps.jvp or missing == "vmap")

    with pytest.raises(core.CapabilityError, match=missing):
        lk.laplace(
            model,
            _ProbeBatchOnly(batch),
            likelihood=Gaussian(),
            hessian_approx=DenseHessian(joint=joint),
            subset=lk.subsets.all,
            hyper_tuner=None,
        )


# --- the low-rank GGN (Nystrom sketch) ----------------------------------------------------


def _regression(n: int, dtype: torch.dtype = torch.float64, hidden: int = 30):
    """A 3 -> hidden -> 1 regression net (P = 5 hidden + 1, so 151 by default) on n
    inputs: the GGN has rank at most n."""

    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(3, hidden), nn.Tanh(), nn.Linear(hidden, 1)).to(dtype)
    lik = copy.replace(Gaussian(), sigma=torch.as_tensor(torch.tensor(0.3, dtype=dtype)))
    x = torch.randn(n, 3, generator=torch.Generator().manual_seed(1), dtype=dtype)
    params = {k: v.detach() for k, v in model.named_parameters()}
    caps = Capabilities(vjp=True, jvp=True, vmap=True)
    ctx = FitContext(ModelFn.from_module(model), lik, Layout.by_module(params), caps)

    return model, ctx, params, x


def _ggn_reference(ctx, params, x, w=None):
    """(1/W) sum_i w_i J_i^T H_i J_i over all params in layout order, J_i from `jacrev`."""

    n = x.shape[0]
    w = torch.ones(n, dtype=x.dtype) if w is None else w
    names = [m for ns in ctx.layout.blocks.values() for m in ns]
    f = ctx.model(params, x)
    e = f[0].numel()
    h = (
        core.output_hessian(ctx.likelihood, f, generator=torch.Generator().manual_seed(0))
        .dense()
        .reshape(n, e, e)
    )

    def fi(p, xi):
        return ctx.model(p, xi.unsqueeze(0)).reshape(-1)

    jac = vmap(jacrev(fi), in_dims=(None, 0))(params, x)
    j = torch.cat([jac[m].reshape(n, e, -1) for m in names], dim=2)

    return torch.einsum("n,nep,nef,nfq->pq", w, j, h, j) / w.sum()


def _low_rank(method, params, ctx, bs):
    op = method.finalize(fold(method, params, ctx, bs))

    assert isinstance(op, core.Joint) and isinstance(op.op, core.LowRankPlusDiag)
    assert op.sizes == tuple((b, ctx.layout.size(b)) for b in ctx.layout.blocks)

    return op.op


def _dense_of(op: core.LowRankPlusDiag) -> Tensor:
    return op.mv(torch.eye(op.factor.shape[0], dtype=op.factor.dtype))


def test_low_rank_exact_on_rank_deficient_ggn():
    _, ctx, params, x = _regression(5)
    w = torch.rand(5, generator=torch.Generator().manual_seed(2), dtype=x.dtype) + 0.5
    want = _ggn_reference(ctx, params, x, w)

    assert ctx.layout.n_params == 151
    assert int(torch.linalg.matrix_rank(want)) == 5

    bs = batches(x, k=5, weights=w)
    op = _low_rank(lk.LowRankGGN(rank=5, oversample=5), params, ctx, bs)

    assert op.factor.shape == (151, 5)
    assert bool((op.diag == 0).all())

    torch.testing.assert_close(_dense_of(op), want, rtol=0, atol=1e-8)


def test_low_rank_top_eigenpairs_match_eigh():
    _, ctx, params, x = _regression(20)
    g = _ggn_reference(ctx, params, x)
    evals, evecs = torch.linalg.eigh(g)
    top, vecs = evals.flip(0)[:4], evecs.flip(1)[:, :4]

    assert int(torch.linalg.matrix_rank(g)) == 20

    op = _low_rank(lk.LowRankGGN(rank=4, oversample=16), params, ctx, batches(x))

    torch.testing.assert_close(op.evals, top, rtol=0, atol=1e-6)
    torch.testing.assert_close(op.factor @ op.factor.T, vecs @ vecs.T, rtol=0, atol=1e-6)


def test_low_rank_sketch_below_rank():
    _, ctx, params, x = _regression(40)
    g = _ggn_reference(ctx, params, x)
    top = torch.linalg.eigvalsh(g).flip(0)[:4]

    assert int(torch.linalg.matrix_rank(g)) == 40

    op = _low_rank(lk.LowRankGGN(rank=4, oversample=8), params, ctx, batches(x))

    # A Nystrom approximation is below G (in the Loewner order), so is each eigenvalue.
    assert bool((op.evals <= top + 1e-10).all())
    assert float(((top - op.evals) / top).abs().max()) < 0.05


def test_low_rank_same_on_every_batch_and_both_paths():
    """The test matrix depends on the seed only; the VJP of a VJP gives the JVP's result."""

    _, ctx, params, x = _regression(12)
    method = lk.LowRankGGN(rank=4, oversample=4)
    whole = _low_rank(method, params, ctx, [Batch(x)])
    split = _low_rank(method, params, ctx, batches(x, k=3))
    double = FitContext(
        ctx.model, ctx.likelihood, ctx.layout, Capabilities(vjp=True, vjp_twice=True)
    )
    reverse = _low_rank(method, params, double, batches(x, k=3))

    torch.testing.assert_close(_dense_of(split), _dense_of(whole), rtol=0, atol=1e-10)
    torch.testing.assert_close(_dense_of(reverse), _dense_of(whole), rtol=0, atol=1e-10)

    # Another fit draws another test matrix from its own seed.
    other_ctx = dataclasses.replace(ctx, seed=ctx.seed + 1)
    other = _low_rank(lk.LowRankGGN(rank=4, oversample=4), params, other_ctx, [Batch(x)])

    assert not torch.allclose(_dense_of(other), _dense_of(whole), rtol=0, atol=1e-6)


@pytest.mark.parametrize("weighted", [False, True], ids=["plain", "weighted"])
def test_low_rank_passes_check_curvature(weighted):
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(3, 5), nn.Tanh(), nn.Linear(5, 3)).double()
    params = {k: v.detach() for k, v in model.named_parameters()}
    g = torch.Generator().manual_seed(1)
    x = torch.randn(24, 3, generator=g, dtype=torch.float64)
    y = torch.randint(0, 3, (24,), generator=g)
    w = torch.rand(24, generator=g, dtype=torch.float64) + 0.5 if weighted else None

    lk_testing.check_curvature(
        lk.LowRankGGN(rank=6), model, Categorical(), params, _labelled(x, y, w)
    )


def test_low_rank_stat_is_a_sum_of_sketches_for_all_reduce(tmp_path):
    """The statistic is one [P, k] sum and a weight; it survives the pickling
    `all_gather_object` does, and adds."""

    _, ctx, params, x = _regression(12)
    method = lk.LowRankGGN(rank=4, oversample=3)
    first = fold(method, params, ctx, batches(x[:6], k=2))
    second = fold(method, params, ctx, batches(x[6:], k=2))

    assert [tuple(v.shape) for v in first.sum.values()] == [(151, 7)]

    torch.save(second, tmp_path / "stat.pt")
    gathered = torch.load(tmp_path / "stat.pt", weights_only=False)
    total = first + gathered
    want = fold(method, params, ctx, batches(x, k=4))

    for k, v in want.sum.items():
        torch.testing.assert_close(total.sum[k], v, rtol=1e-12, atol=1e-12)

    torch.testing.assert_close(total.weight, want.weight)
    assert core.all_reduce(first) is first


def test_omega_drawn_once(monkeypatch):
    """The QR of the sketch's test matrix runs once per (seed, P, k), not once per call."""

    _, ctx, params, x = _regression(12)
    method = lk.LowRankGGN(rank=4, oversample=3)
    calls = []
    draw = curvature._draw_test_matrix

    def counting(seed: int, p: int, k: int) -> Tensor:
        calls.append((seed, p, k))
        return draw(seed, p, k)

    monkeypatch.setattr(curvature, "_draw_test_matrix", counting)
    curvature._test_matrix.cache_clear()
    method.finalize(fold(method, params, ctx, batches(x, k=3)))

    assert calls == [(ctx.seed, 151, 7)]


def test_omega_cache_same_result(monkeypatch):
    _, ctx, params, x = _regression(12)
    method = lk.LowRankGGN(rank=4, oversample=3)
    curvature._test_matrix.cache_clear()
    stat = fold(method, params, ctx, batches(x, k=3))
    op = method.finalize(stat)
    cached = curvature._test_matrix

    def uncached(seed: int, p: int, k: int) -> Tensor:
        cached.cache_clear()
        return cached(seed, p, k)

    monkeypatch.setattr(curvature, "_test_matrix", uncached)
    want_stat = fold(method, params, ctx, batches(x, k=3))
    want = method.finalize(want_stat)

    torch.testing.assert_close(stat.sum, want_stat.sum, rtol=0, atol=0)
    torch.testing.assert_close(op.op.factor, want.op.factor, rtol=0, atol=0)
    torch.testing.assert_close(op.op.evals, want.op.evals, rtol=0, atol=0)


def test_omega_by_seed_and_never_shared():
    """Omega is the QR of a seeded Gaussian draw; each call returns a fresh tensor, so
    a caller writing into it leaves the cache alone."""

    method = lk.LowRankGGN(rank=2)
    cpu = torch.device("cpu")
    draw = torch.randn(20, 5, generator=torch.Generator().manual_seed(3))
    want, _ = torch.linalg.qr(draw.double(), mode="reduced")

    for dtype in (torch.float64, torch.float32):
        torch.testing.assert_close(method._omega(3, 20, 5, dtype, cpu), want.to(dtype))

    first = method._omega(3, 20, 5, torch.float64, cpu)
    first.zero_()
    again = method._omega(3, 20, 5, torch.float64, cpu)

    torch.testing.assert_close(again, want, rtol=0, atol=0)
    assert not again.requires_grad
    assert not torch.allclose(method._omega(4, 20, 5, torch.float64, cpu), want)


def test_low_rank_too_large_raises():
    _, ctx, params, _ = _regression(5)

    with pytest.raises(ValueError, match=r"P = 151"):
        lk.LowRankGGN(rank=142).init(params, ctx)

    lk.LowRankGGN(rank=141).init(params, ctx)

    with pytest.raises(ValueError, match="rank >= 1"):
        lk.LowRankGGN(rank=0)

    with pytest.raises(ValueError, match="oversample >= 0"):
        lk.LowRankGGN(rank=2, oversample=-1)


def test_low_rank_has_no_output_hessian_option():
    """The observed output Hessian can be indefinite (a heteroscedastic Gaussian), which a
    Nystrom sketch can't represent, so LowRankGGN always uses the Fisher and has no option."""

    with pytest.raises(TypeError, match="observed"):
        # The removed option is the misuse under test.
        lk.LowRankGGN(rank=4, observed=True)  # ty: ignore[unknown-argument]


def _rank_deficient_post(dtype, rank: int = 8):
    model, _, _, x = _regression(5, dtype)
    y = torch.randn(5, 1, generator=torch.Generator().manual_seed(3), dtype=dtype)
    post = lk.laplace(
        model,
        [(x, y)],
        likelihood=copy.replace(Gaussian(), sigma=torch.as_tensor(torch.tensor(0.3, dtype=dtype))),
        subset=lk.subsets.all,
        hessian_approx=lk.LowRankGGN(rank=rank),
    )

    return post, x


def _classifier_post(dtype, **kwargs):
    torch.manual_seed(0)
    g = torch.Generator().manual_seed(1)
    x = torch.randn(60, 2, generator=g).to(dtype)
    y = torch.randint(0, 3, (60,), generator=g)
    model = nn.Sequential(nn.Linear(2, 8), nn.Tanh(), nn.Linear(8, 3)).to(dtype)
    post = lk.laplace(
        model,
        [(x[:30], y[:30]), (x[30:], y[30:])],
        likelihood=Categorical(),
        subset=lk.subsets.all,
        hessian_approx=lk.LowRankGGN(rank=8),
        **kwargs,
    )

    return post, x


def _assert_finite(post, x):
    out = lk.predict_outputs(post, x[:4])
    draws = post.sample(5, generator=torch.Generator().manual_seed(0))

    assert bool(torch.isfinite(post.log_marglik()))
    assert bool(torch.isfinite(out.mean).all())
    assert bool(torch.isfinite(out.cov.diagonal()).all())
    assert all(bool(torch.isfinite(v).all()) for v in draws.values())


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32], ids=str)
def test_low_rank_posterior_finite(dtype):
    post, x = _classifier_post(dtype)

    assert isinstance(post.precision(), core.Joint)

    _assert_finite(post, x)


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32], ids=str)
def test_low_rank_rank_deficient_posterior_finite(dtype):
    """G has rank 5 and the sketch 18 columns: the Nystrom shift keeps the core matrix
    positive definite, and the null directions get (near) zero eigenvalues."""

    post, x = _rank_deficient_post(dtype)
    op = post.precision().op

    assert isinstance(op, core.LowRankPlusDiag)
    assert bool((op.evals[5:] <= 1e-3 * op.evals[0]).all())

    _assert_finite(post, x)

    v = torch.randn(151, generator=torch.Generator().manual_seed(4), dtype=dtype)

    assert bool(torch.isfinite(op.solve(v)).all())
    assert bool(torch.isfinite(op.logdet()))


def test_low_rank_continual_is_exact():
    """`prior=post` adds the second fit's low-rank term to the first posterior's
    precision: the factors are concatenated, and the sum is exact."""

    prior = Isotropic(torch.tensor(1.0, dtype=torch.float64))
    first, _ = _classifier_post(torch.float64, prior=prior, hyper_tuner=None)
    alone, _ = _classifier_post(torch.float64, prior=prior, hyper_tuner=None)
    second, _ = _classifier_post(torch.float64, prior=first, hyper_tuner=None)

    p1, p2 = first.precision().op, alone.precision().op
    both = second.precision().op

    assert isinstance(both, core.LowRankPlusDiag) and both.factor.shape[1] == 16

    want = _dense_of(p1) + _dense_of(p2) - torch.eye(p1.factor.shape[0], dtype=torch.float64)

    torch.testing.assert_close(_dense_of(both), want, rtol=1e-10, atol=1e-10)


def test_low_rank_selection():
    """On some entries of the last layer: the sketch is over the selected entries only, and
    with k = P_sel it is the dense GGN of those entries."""

    ctx, params, x = make("cat", torch.float64)
    mask = torch.rand(3, 8, generator=torch.Generator().manual_seed(5)) < 0.5
    sel = core.Selection.from_masks(params, {"2.weight": mask})
    sub_fn, sub_params = core.select(ctx.model, sel)
    n_sel = int(mask.sum())
    caps = core.probe(sub_fn, ctx.likelihood, Batch(x[:4]), params=sub_params)
    fit = core.fit_curvature(
        sub_fn,
        ctx.likelihood,
        sub_params,
        lk.LowRankGGN(rank=n_sel, oversample=0),
        capabilities=caps,
    )
    stat = fit.init()

    for b in batches(x):
        stat = fit.update(stat, b)

    op = fit.finalize(stat)

    assert isinstance(op, core.Joint) and isinstance(op.op, core.LowRankPlusDiag)
    assert op.op.factor.shape == (n_sel, n_sel)

    def f(w):
        return ctx.model({**params, "2.weight": w}, x)

    jac = jacrev(f)(params["2.weight"]).reshape(x.shape[0], 3, -1)[:, :, sel.index["2.weight"]]
    p = torch.softmax(f(params["2.weight"]), dim=-1)
    h = torch.diag_embed(p) - p.unsqueeze(-1) * p.unsqueeze(-2)
    want = torch.einsum("nkp,nkl,nlq->pq", jac, h, jac) / x.shape[0]

    torch.testing.assert_close(_dense_of(op.op), want, rtol=0, atol=1e-10)


def test_low_rank_d8():
    _, _, x = make("cat", torch.float64)

    _check_d8(lk.LowRankGGN(rank=8), batches(x))


class _NoForwardNoDouble(torch.autograd.Function):
    """sin, with a VJP that is not differentiable and no JVP."""

    @staticmethod
    def forward(x):
        return torch.sin(x)

    @staticmethod
    def setup_context(ctx, inputs, output):
        ctx.save_for_backward(inputs[0])

    @staticmethod
    def backward(ctx, g):
        (x,) = ctx.saved_tensors

        return g.detach() * torch.cos(x)


def test_low_rank_requires_and_refuses_before_data():
    assert lk.LowRankGGN(rank=3).requires == core.Requirements(
        vjp=True,
        needs_closed_form_hessian=True,
        one_of=(core.Requirements(jvp=True), core.Requirements(vjp_twice=True)),
    )
    assert repr(lk.LowRankGGN(rank=3)) == ("LowRankGGN(rank=3)")

    torch.manual_seed(0)
    model = _WithFunction(_NoForwardNoDouble).double()
    g = torch.Generator().manual_seed(1)
    batch = Batch(
        torch.randn(8, 2, generator=g, dtype=torch.float64),
        torch.randn(8, 1, generator=g, dtype=torch.float64),
    )
    caps = core.probe(ModelFn.from_module(model), Gaussian(), batch)

    assert caps.vjp and not caps.jvp and not caps.vjp_twice

    with pytest.raises(core.CapabilityError, match=r"one of \(jvp \| vjp_twice\)"):
        lk.laplace(
            model,
            _ProbeBatchOnly(batch),
            likelihood=Gaussian(),
            hessian_approx=lk.LowRankGGN(rank=2, oversample=2),
            subset=lk.subsets.all,
            hyper_tuner=None,
        )


def test_low_rank_of_a_zero_ggn_is_zero():
    """The first layer behind a zero last layer has J = 0, so G = 0 and every eigenvalue
    is zero (no Cholesky of a zero core matrix)."""

    model, _, _, x = _regression(8, hidden=6)

    with torch.no_grad():
        model[2].weight.zero_()

    first = {k: v.detach() for k, v in model.named_parameters() if k.startswith("0.")}
    caps = Capabilities(vjp=True, jvp=True)
    fit = core.fit_curvature(
        ModelFn.from_module(model), Gaussian(), first, lk.LowRankGGN(rank=3), capabilities=caps
    )
    stat = fit.init()

    for b in batches(x):
        stat = fit.update(stat, b)

    op = fit.finalize(stat)

    assert isinstance(op, core.Joint) and isinstance(op.op, core.LowRankPlusDiag)
    assert bool((op.op.evals == 0).all()) and op.op.factor.shape == (24, 3)


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32], ids=str)
def test_low_rank_many_examples_rank_deficient(dtype):
    """N = 6000 examples on 3 distinct inputs: G has rank 3 and the sketch k = 10 columns.
    Round-off in the sum Y pushes some eigenvalues of the shifted core below the shift;
    they are clamped (Tropp et al. 2017), and the result is G up to that round-off.

    The tolerance is N eps ||G||_2, the worst-case relative error of a sum of N terms."""

    _, ctx, params, _ = _regression(3, dtype, hidden=4)
    pts = torch.randn(3, 3, generator=torch.Generator().manual_seed(1), dtype=dtype)
    x = pts.repeat(2000, 1)
    want = _ggn_reference(ctx, params, pts)

    assert ctx.layout.n_params == 21
    assert int(torch.linalg.matrix_rank(want)) == 3

    method = lk.LowRankGGN(5, oversample=5)
    op = _low_rank(method, params, ctx, [Batch(xb) for xb in x.chunk(3)])
    tol = 6000 * torch.finfo(dtype).eps * float(torch.linalg.matrix_norm(want, 2))

    assert bool((op.evals[3:] <= tol).all())

    torch.testing.assert_close(_dense_of(op), want, rtol=0, atol=tol)

    # With a prior: the posterior precision N G + I, and draws L z with L L^T = its inverse.
    prec = op.scaled(torch.tensor(6000.0, dtype=dtype)).add_isotropic(
        torch.tensor(1.0, dtype=dtype)
    )
    z = torch.randn(5, 21, generator=torch.Generator().manual_seed(2), dtype=dtype)

    assert bool(torch.isfinite(prec.logdet()))
    assert bool(torch.isfinite(prec.solve(z)).all())
    assert bool(torch.isfinite(prec.cov_sqrt_mv(z)).all())


# --- templates ------------------------------------------------------------------------


def _template_toy(dtype: torch.dtype):
    """A two-layer net with a bias-less last layer, a Gaussian likelihood and targets."""

    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(3, 4), nn.Tanh(), nn.Linear(4, 2, bias=False)).to(dtype)
    lik = copy.replace(Gaussian(), sigma=torch.as_tensor(torch.tensor(0.5, dtype=dtype)))
    x = torch.randn(12, 3, dtype=dtype)
    y = torch.randn(12, 2, dtype=dtype)
    params = {k: v.detach() for k, v in model.named_parameters()}
    fn = ModelFn.from_module(model)
    caps = core.probe(fn, lik, Batch(x[:4], y[:4]))
    ctx = FitContext(fn, lik, Layout.by_module(params), caps)

    return ctx, params, [Batch(x[:6], y[:6]), Batch(x[6:], y[6:])]


def _kfac():
    from laplacekit.curvlinops import KFAC

    return KFAC()


TEMPLATED = {
    "DiagGGN": DiagGGN,
    "DiagGGN-mc": lambda: DiagGGN(fisher="mc"),
    "DiagGGN-empirical": lambda: DiagGGN(fisher="empirical"),
    "DenseGGN": DenseGGN,
    "DenseGGN-joint": lambda: DenseGGN(joint=True),
    "DenseGGN-empirical": lambda: DenseGGN(fisher="empirical"),
    "DenseGGN-empirical-joint": lambda: DenseGGN(fisher="empirical", joint=True),
    "DenseHessian": DenseHessian,
    "DenseHessian-joint": lambda: DenseHessian(joint=True),
    "LowRankGGN": lambda: lk.LowRankGGN(3, oversample=2),
    "KFAC": _kfac,
}


def _signature(op):
    """Type and static fields of every node, and the shape and dtype of every leaf."""

    def leaf(t: Tensor):
        return (tuple(t.shape), t.dtype)

    return core.tree_map(leaf, op)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.bfloat16])
@pytest.mark.parametrize("make_method", TEMPLATED.values(), ids=list(TEMPLATED))
def test_template_matches_finalize(make_method, dtype):
    """The template has finalize's structure (node types, blocks, static fields, leaf
    shapes and dtypes) and is the zero operator, without reading data."""

    if dtype == torch.bfloat16 and make_method in (TEMPLATED["KFAC"], TEMPLATED["LowRankGGN"]):
        pytest.skip("Curvlinops and the Nystrom sketch run in float32 or float64")

    ctx, params, bs = _template_toy(dtype)
    method = make_method()

    assert isinstance(method, core.HasTemplate)

    template = method.template(params, ctx)
    result = method.finalize(fold(method, params, ctx, bs))

    assert type(template) is type(result)
    # tree_map raises on a different node type, static field, block key or arity.
    core.tree_map(lambda a, b: None, template, result)
    assert core.tree_leaves(_signature(template)) == core.tree_leaves(_signature(result))

    # The zero operator: every product is zero.
    sizes = {b: ctx.layout.size(b) for b in ctx.layout.blocks}
    stat = torch.promote_types(dtype, torch.float32)  # statistics accumulate in at least f32
    v = {b: torch.randn(n, dtype=stat) for b, n in sizes.items()}
    out = template.mv(v)

    for t in core.tree_leaves(out):
        assert bool((t == 0).all())


def test_kfac_template_on_no_params_raises_like_init():
    ctx, _, _ = _template_toy(torch.float64)
    empty = copy.replace(ctx, layout=Layout.by_module({}))

    with pytest.raises(core.CapabilityError, match="KFAC needs at least one parameter"):
        _kfac().init({}, empty)

    with pytest.raises(core.CapabilityError, match="KFAC needs at least one parameter"):
        _kfac().template({}, empty)


def test_kfac_template_keeps_the_kronecker_factors():
    ctx, params, _ = _template_toy(torch.float64)
    template = _kfac().template(params, ctx)

    for b, blk in template.blocks.items():
        assert isinstance(blk, core.KronEigenbasis)
        assert len(blk.factors) == 3
        assert all(bool((f == 0).all()) for f in blk.factors), b
        assert bool((blk.evals == 0).all())
