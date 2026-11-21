"""`lk.gp` points: fixed inputs, a uniform pick from the data, and k-means."""

import laplacekit_gp.points
import pytest
import torch
from torch import nn

import laplacekit as lk
import laplacekit.core as core
from laplacekit import function_space as fs
from laplacekit.core import Done, FitContext, Layout


def blobs(n: int, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    centers = torch.tensor([[-2.0, 0.0], [2.0, 0.0], [0.0, 3.0]], dtype=torch.float64)
    y = torch.arange(n) % 3
    x = centers[y] + 0.7 * torch.randn(n, 2, generator=g, dtype=torch.float64)

    return x, y


def model() -> nn.Module:
    torch.manual_seed(0)

    return nn.Sequential(nn.Linear(2, 4), nn.Tanh(), nn.Linear(4, 3)).double()


def params_of(net: nn.Module):
    return {k: v.detach() for k, v in net.named_parameters()}


def context(net: nn.Module) -> FitContext:
    params = params_of(net)
    fn = core.ModelFn.from_module(net)

    return FitContext(fn, lk.Categorical(), Layout.by_module(params), core.Capabilities())


def split(x, y, sizes):
    out, start = [], 0

    for s in sizes:
        out.append(lk.Batch(x[start : start + s], y[start : start + s]))
        start += s

    return out


class _Collect:
    """A `Staged` whose result is the `Picked` of `points`, so the core loop runs it."""

    def __init__(self, points):
        self.points = points

    @property
    def requires(self):
        return core.Requirements()

    def plan(self, params, ctx):
        return self.points.pick(params, ctx, lambda picked: Done(picked))


def _gen(seed: int) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


def run(points, net, batches, generator):
    """The points' pass through the core fit loop, with its batch generators."""

    return fit_loop(_Collect(points), net, batches, generator)


def fit_loop(method, net, batches, generator):
    """Every pass of `method` through the core fit loop; the result of the last."""

    fn = core.ModelFn.from_module(net)
    fit = core.fit_curvature(fn, lk.Categorical(), params_of(net), method, generator=generator)
    step = fit.start()

    while isinstance(step, core.Running):
        stat = step.init()

        for b in batches:
            stat = step.update(stat, b)

        step = step.finish(stat)

    return step.result


def rows_of(x, picked):
    """The row of `x` each picked point is, in pick order."""

    return [int((x == p).all(-1).nonzero()) for p in picked]


# --- from_data ----------------------------------------------------------------------------


@pytest.mark.parametrize("sizes", [[30], [10, 10, 10], [7, 13, 4, 6]])
def test_from_data_is_dense_priority_pick(sizes):
    x, y = blobs(30)
    net, batches = model(), split(x, y, sizes)
    picked = run(lk.gp.from_data(8), net, batches, _gen(5))

    # Every row, in priority order: the m smallest priorities are its first m rows.
    everything = run(lk.gp.from_data(30), net, batches, _gen(5))
    rows = rows_of(x, picked.inputs)

    assert sorted(rows_of(x, everything.inputs)) == list(range(30))
    assert rows == rows_of(x, everything.inputs)[:8]
    torch.testing.assert_close(picked.u, everything.u[:8], rtol=0, atol=0)
    assert bool(((picked.u >= 0) & (picked.u < 1)).all())

    # The same generator picks the same rows; another one picks others.
    assert rows_of(x, run(lk.gp.from_data(8), net, batches, _gen(5)).inputs) == rows
    assert rows_of(x, run(lk.gp.from_data(8), net, batches, _gen(6)).inputs) != rows
    assert picked.coords is None
    assert picked.u.dtype == torch.float64
    assert int(picked.seen) == 30


def test_from_data_takes_mappings_and_keeps_all_when_short():
    x, y = blobs(6)
    batches = [
        lk.Batch({"x": x[:4], "t": torch.tensor(2.0)}, y[:4]),
        lk.Batch({"x": x[4:], "t": torch.tensor(2.0)}, y[4:]),
    ]

    class _Mapped(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(2, 3).double()

        def forward(self, x, t):
            return self.lin(x) * t

    picked = run(lk.gp.from_data(10), _Mapped(), batches, _gen(1))

    assert sorted(rows_of(x, picked.inputs["x"])) == list(range(6))
    assert picked.inputs["t"].ndim == 0
    assert int(picked.seen) == 6

    with pytest.raises(ValueError, match="n_points >= 1"):
        lk.gp.from_data(0)


def test_from_data_matches_subset_of_data_rows():
    x, y = blobs(30)
    net = model()
    batches = [lk.Batch(x[:12], y[:12]), lk.Batch(x[12:20], y[12:20]), lk.Batch(x[20:], y[20:])]
    prior = lk.priors.Isotropic(torch.tensor(1.0, dtype=torch.float64))

    class _Subset:
        """`fs.SubsetOfData(9)` with its prior bound, as a method the fit loop runs."""

        requires = fs.SubsetOfData(9).requires

        def plan(self, params, ctx):
            return fs.SubsetOfData(9).plan(params, ctx, prior)

    # Equally seeded generators: the GP's rows are the points' rows.
    gp = fit_loop(_Subset(), net, batches, _gen(11))
    picked = run(lk.gp.from_data(9), net, batches, _gen(11))

    torch.testing.assert_close(picked.inputs, gp.inputs, rtol=0, atol=0)


def _fold(curv, params, ctx, batches, seeds):
    stat = curv.init(params, ctx)

    for b, k in zip(batches, seeds, strict=True):
        stat = curv.update(stat, params, b, _gen(k), ctx)

    return stat


@pytest.mark.parametrize("points", [lk.gp.from_data(5), lk.gp.kmeans(4, n_iters=3)])
def test_pick_monoid_law(points):
    x, y = blobs(24)
    net = model()
    params, ctx = params_of(net), context(net)
    batches = split(x, y, [6, 9, 3, 6])
    seeds = list(range(4))

    plan = points.pick(params, ctx, lambda picked: Done(picked))
    assert isinstance(plan, core.Pass)
    curv = plan.curvature

    whole = _fold(curv, params, ctx, batches, seeds)
    a = _fold(curv, params, ctx, batches[:2], seeds[:2])
    b = _fold(curv, params, ctx, batches[2:], seeds[2:])
    empty = curv.init(params, ctx)

    for got in (a + b, b + a, empty + whole, whole + empty):
        want, have = plan.then(curv.finalize(whole)).result, plan.then(curv.finalize(got)).result
        torch.testing.assert_close(have.inputs, want.inputs, rtol=0, atol=0)
        torch.testing.assert_close(have.u, want.u, rtol=0, atol=0)
        assert int(have.seen) == int(want.seen) == 24


class _Counted(nn.Module):
    """A model that counts its forward calls."""

    def __init__(self):
        super().__init__()
        self.net = model()
        self.calls = 0

    def forward(self, x):
        self.calls += 1

        return self.net(x)


@pytest.mark.parametrize("points", [lk.gp.from_data(3), lk.gp.kmeans(3)])
def test_points_refuse_weighted_data(points):
    x, y = blobs(12)
    net = _Counted()
    batches = [lk.Batch(x[:6], y[:6], torch.ones(6, dtype=torch.float64)), lk.Batch(x[6:], y[6:])]
    picked = []

    class _Watch:
        requires = core.Requirements()

        def plan(self, params, ctx):
            def then(p):
                picked.append(p)

                return Done(p)

            return points.pick(params, ctx, then)

    fn = core.ModelFn.from_module(net)
    params = {k: v.detach() for k, v in net.named_parameters()}
    fit = core.fit_curvature(
        fn,
        lk.Categorical(),
        params,
        _Watch(),
        generator=torch.Generator().manual_seed(0),
        capabilities=core.Capabilities(),
    )
    step = fit.start()
    assert isinstance(step, core.Running)
    stat = step.init()

    with pytest.raises(core.CapabilityError, match="per-example weights"):
        step.update(stat, batches[0])

    # Refused before the pass or the data-fit term saw the batch.
    assert net.calls == 0
    assert not picked


# --- kmeans -------------------------------------------------------------------------------


def lloyd(x, centres, iters):
    """Lloyd's algorithm; an empty cluster keeps its centre. Returns the snapped rows."""

    c = centres.clone()

    for _ in range(iters):
        assign = torch.cdist(x, c).argmin(1)

        for j in range(c.shape[0]):
            members = x[assign == j]

            if members.shape[0] > 0:
                c[j] = members.mean(0)

    return torch.cdist(x, c).argmin(0)


def test_kmeans_matches_lloyd_reference():
    g = torch.Generator().manual_seed(3)
    x = torch.cat(
        [torch.randn(50, 2, generator=g, dtype=torch.float64) + c for c in (-4.0, 0.0, 4.0, 8.0)]
    )
    sizes = [64, 64, 72]
    y = torch.zeros(200, dtype=torch.int64)
    net, batches = model(), split(x, y, sizes)
    picked = run(lk.gp.kmeans(6), net, batches, _gen(2))

    # Initial centres: the 6 rows of smallest priority, which `from_data` picks with the
    # same generator; it also gives each row's u.
    everything = run(lk.gp.from_data(200), net, batches, _gen(2))
    order = torch.tensor(rows_of(x, everything.inputs))
    u = torch.empty(200, dtype=torch.float64)
    u[order] = everything.u
    init = x[order[:6]]
    rows = lloyd(x, init, 20)

    torch.testing.assert_close(picked.inputs, x[rows], rtol=0, atol=0)
    torch.testing.assert_close(picked.u, u[rows], rtol=0, atol=0)
    assert int(picked.seen) == 200

    # More iterations move the centres: the reference with fewer is a different pick.
    assert not torch.equal(lloyd(x, init, 0), rows)


def test_kmeans_keeps_all_rows_when_short():
    x, y = blobs(5)
    net, batches = model(), split(x, y, [5])
    picked = run(lk.gp.kmeans(8), net, batches, _gen(0))

    # Every row, in priority order, as `from_data` gives them.
    everything = run(lk.gp.from_data(5), net, batches, _gen(0))
    torch.testing.assert_close(picked.inputs, everything.inputs, rtol=0, atol=0)


def test_kmeans_refuses_mapping_inputs():
    net, x = model(), blobs(4)[0]
    params, ctx = params_of(net), context(net)
    curv = lk.gp.kmeans(2).pick(params, ctx, lambda picked: Done(picked)).curvature
    for inputs in ({"x": x}, x.long(), torch.tensor(1.0)):
        with pytest.raises(core.StructureError, match=r"kmeans.*tensor inputs"):
            curv.update(curv.init(params, ctx), params, lk.Batch(inputs), _gen(0), ctx)

    with pytest.raises(ValueError, match="n_points >= 1"):
        lk.gp.kmeans(0)

    with pytest.raises(ValueError, match="iters >= 0"):
        lk.gp.kmeans(2, n_iters=-1)


def test_kmeans_budget(monkeypatch):
    import laplacekit_gp.points as points

    monkeypatch.setattr(points, "MAX_KMEANS_ENTRIES", 30)
    net, x = model(), blobs(20)[0]
    params, ctx = params_of(net), context(net)
    curv = lk.gp.kmeans(2).pick(params, ctx, lambda picked: Done(picked)).curvature
    stat = curv.update(curv.init(params, ctx), params, lk.Batch(x[:15]), _gen(0), ctx)

    with pytest.raises(core.StructureError, match="MAX_KMEANS_ENTRIES=30"):
        curv.update(stat, params, lk.Batch(x[15:]), _gen(1), ctx)

    assert lk.gp.MAX_KMEANS_ENTRIES == 2**26


# --- fixed --------------------------------------------------------------------------------


def test_fixed_checks_coords():
    net = model()
    params, ctx = params_of(net), context(net)
    z = torch.randn(4, 2, dtype=torch.float64)
    coords = torch.tensor([0, 2, 1, 1])

    pts = lk.gp.fixed(z, coords, generator=_gen(7))
    assert isinstance(pts, lk.gp.Points)
    assert isinstance(pts, lk.gp.Drawn)
    assert pts.draw(generator=torch.Generator().manual_seed(1)) is z

    done = pts.pick(params, ctx, lambda picked: Done(picked))
    assert isinstance(done, Done)
    picked = done.result

    assert picked.inputs is z
    torch.testing.assert_close(picked.coords, coords, rtol=0, atol=0)
    # The documented rule: a generator from `spawn_generator` draws u.
    seed = core.spawn_generator(_gen(7)).initial_seed()
    want_u = torch.rand(4, generator=_gen(seed), dtype=torch.float64)
    torch.testing.assert_close(picked.u, want_u, rtol=0, atol=0)
    assert int(picked.seen) == 0

    # Every fit with these points gets the same u.
    again = pts.pick(params, ctx, lambda picked: Done(picked))
    assert isinstance(again, Done)
    torch.testing.assert_close(again.result.u, want_u, rtol=0, atol=0)

    # generator=None: the seed comes from torch's global RNG.
    torch.manual_seed(3)
    default = lk.gp.fixed(z).pick(params, ctx, lambda picked: Done(picked))
    assert isinstance(default, Done)
    torch.manual_seed(3)
    seed = core.spawn_generator().initial_seed()
    want_u = torch.rand(4, generator=_gen(seed), dtype=torch.float64)
    torch.testing.assert_close(default.result.u, want_u, rtol=0, atol=0)
    assert default.result.coords is None

    for bad in (coords.int(), coords[:3], coords.reshape(2, 2), coords.double()):
        with pytest.raises(core.StructureError, match="int64"):
            lk.gp.fixed(z, bad)


def test_fixed_takes_mappings():
    z = {"x": torch.randn(3, 2, dtype=torch.float64), "t": torch.tensor(1.0)}
    pts = lk.gp.fixed(z, torch.tensor([0, 1, 2]))

    net = model()
    done = pts.pick(params_of(net), context(net), lambda picked: Done(picked))
    assert isinstance(done, Done)
    assert done.result.u.shape == (3,)

    with pytest.raises(core.StructureError, match=r"\[3\]"):
        lk.gp.fixed(z, torch.tensor([0, 1]))


# --- grid and uniform ---------------------------------------------------------------------


def test_grid_points():
    low = torch.tensor([0.0, -1.0], dtype=torch.float64)
    high = torch.tensor([1.0, 1.0], dtype=torch.float64)
    pts = lk.gp.grid(low, high, (3, 2))

    assert isinstance(pts, laplacekit_gp.points.Grid)
    assert isinstance(pts, lk.gp.Points)
    assert isinstance(pts, lk.gp.Drawn)

    want = torch.tensor(
        [[0.0, -1.0], [0.0, 1.0], [0.5, -1.0], [0.5, 1.0], [1.0, -1.0], [1.0, 1.0]],
        dtype=torch.float64,
    )
    torch.testing.assert_close(
        pts.draw(generator=torch.Generator().manual_seed(3)), want, rtol=0, atol=0
    )

    # One count for every axis; `high` broadcasts against `low`.
    square = lk.gp.grid(low, 2.0, 2).draw(generator=torch.Generator().manual_seed(0))
    want = torch.tensor([[0.0, -1.0], [0.0, 2.0], [2.0, -1.0], [2.0, 2.0]], dtype=torch.float64)
    torch.testing.assert_close(square, want, rtol=0, atol=0)

    # A float `low` is one axis, in the default dtype.
    line = lk.gp.grid(-1.0, 1.0, 5).draw(generator=torch.Generator().manual_seed(0))
    want = torch.tensor([[-1.0], [-0.5], [0.0], [0.5], [1.0]], dtype=torch.get_default_dtype())
    torch.testing.assert_close(line, want, rtol=0, atol=0)


def test_grid_pick_needs_no_data():
    net = model()
    params, ctx = params_of(net), context(net)
    pts = lk.gp.grid(torch.zeros(2, dtype=torch.float64), 1.0, 3)

    done = pts.pick(params, ctx, lambda picked: Done(picked))
    assert isinstance(done, Done)
    picked = done.result

    torch.testing.assert_close(
        picked.inputs, pts.draw(generator=torch.Generator().manual_seed(0)), rtol=0, atol=0
    )
    want_u = torch.rand(9, generator=_gen(0), dtype=torch.float64)
    torch.testing.assert_close(picked.u, want_u, rtol=0, atol=0)
    assert picked.coords is None
    assert int(picked.seen) == 0


def test_grid_refuses_bad_shapes():
    with pytest.raises(core.StructureError, match="one count per axis"):
        lk.gp.grid(torch.zeros(2), 1.0, (3, 3, 3))

    with pytest.raises(core.StructureError, match="low"):
        lk.gp.grid(torch.zeros(2, 2), 1.0, 3)

    with pytest.raises(core.StructureError, match="high"):
        lk.gp.grid(torch.zeros(2), torch.ones(3), 3)

    with pytest.raises(ValueError, match="n >= 1"):
        lk.gp.grid(0.0, 1.0, 0)

    with pytest.raises(ValueError, match="high >= low"):
        lk.gp.grid(1.0, 0.0, 3)


def test_uniform_draw_follows_the_generator():
    low = torch.tensor([-1.0, 0.0, 2.0], dtype=torch.float64)
    high = torch.tensor([1.0, 0.5, 4.0], dtype=torch.float64)
    pts = lk.gp.uniform(low, high, 6)

    assert isinstance(pts, laplacekit_gp.points.Uniform)
    assert isinstance(pts, lk.gp.Points)
    assert isinstance(pts, lk.gp.Drawn)

    a = pts.draw(generator=torch.Generator().manual_seed(1))
    b = pts.draw(generator=torch.Generator().manual_seed(2))

    # The documented rule: the generator draws [m, D] uniforms.
    want = low + (high - low) * torch.rand(6, 3, generator=_gen(1), dtype=torch.float64)
    torch.testing.assert_close(a, want, rtol=0, atol=0)
    torch.testing.assert_close(
        pts.draw(generator=torch.Generator().manual_seed(1)), a, rtol=0, atol=0
    )
    assert not torch.equal(a, b)
    assert bool(((a >= low) & (a <= high)).all())

    # No generator: torch's global RNG.
    torch.manual_seed(4)
    c = pts.draw()
    torch.manual_seed(4)
    torch.testing.assert_close(pts.draw(), c, rtol=0, atol=0)


def test_uniform_pick_uses_its_seed():
    net = model()
    params, ctx = params_of(net), context(net)
    pts = lk.gp.uniform(0.0, 1.0, 5, generator=_gen(4))

    done = pts.pick(params, ctx, lambda picked: Done(picked))
    assert isinstance(done, Done)
    picked = done.result

    # One seed drawn from the generator seeds a CPU generator: the points, then u.
    gen = core.spawn_generator(_gen(4))
    want_x = torch.rand(5, 1, generator=gen, dtype=torch.get_default_dtype())
    want_u = torch.rand(5, generator=gen, dtype=torch.float64)

    torch.testing.assert_close(picked.inputs, want_x, rtol=0, atol=0)
    torch.testing.assert_close(picked.u, want_u, rtol=0, atol=0)
    assert int(picked.seen) == 0

    # Every fit with these points picks the same ones.
    again = pts.pick(params, ctx, lambda picked: Done(picked))
    assert isinstance(again, Done)
    torch.testing.assert_close(again.result.inputs, want_x, rtol=0, atol=0)

    # generator=None: the seed comes from torch's global RNG.
    torch.manual_seed(5)
    default = lk.gp.uniform(0.0, 1.0, 5).pick(params, ctx, lambda picked: Done(picked))
    assert isinstance(default, Done)
    torch.manual_seed(5)
    gen = core.spawn_generator()
    want_x = torch.rand(5, 1, generator=gen, dtype=torch.get_default_dtype())
    torch.testing.assert_close(default.result.inputs, want_x, rtol=0, atol=0)

    with pytest.raises(ValueError, match="n_points >= 1"):
        lk.gp.uniform(0.0, 1.0, 0)


def test_point_count_of_grid_and_uniform():
    from laplacekit_gp.points import point_count

    assert point_count(lk.gp.grid(torch.zeros(2), 1.0, (3, 4))) == 12
    assert point_count(lk.gp.uniform(0.0, 1.0, 7)) == 7


def test_shared_checks_are_not_public_points_api():
    """`point_count` and `check_coords` serve the methods; they are not in `__all__`."""

    import laplacekit_gp.points as points

    assert {"point_count", "check_coords"}.isdisjoint(points.__all__)
    assert callable(points.point_count) and callable(points.check_coords)
