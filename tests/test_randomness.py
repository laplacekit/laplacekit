"""Randomness follows torch: `torch.manual_seed`, or an optional `generator=`. No keys.

Each top-level call draws one seed, from its generator or from torch's global RNG, and
the protocols receive fresh `torch.Generator`s derived from it.
"""

import dataclasses
import importlib
import inspect
from collections.abc import Callable, Iterable
from typing import Any

import pytest
import torch
from torch import Tensor, nn

import laplacekit as lk
import laplacekit.core as core
from laplacekit.core import Batch, FitContext, Params, Requirements, SumStat

F64 = torch.float64
PLUGINS = (
    "laplacekit_transformers",
    "laplacekit_diffusers",
    "laplacekit_subnet",
    "laplacekit_heteroscedastic",
    "laplacekit_reward",
    "laplacekit_gp",
    "laplacekit_marglik",
)


def _model() -> nn.Module:
    # Fixed weights, without touching the global RNG the tests seed.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        return nn.Sequential(nn.Linear(3, 4), nn.Tanh(), nn.Linear(4, 3)).double()


def _batches(sizes: Iterable[int] = (5, 3, 4)) -> list[tuple[Tensor, Tensor]]:
    g = torch.Generator().manual_seed(1)
    n = sum(sizes)
    x = torch.randn(n, 3, generator=g, dtype=F64)
    y = torch.randint(0, 3, (n,), generator=g)
    out, start = [], 0

    for s in sizes:
        out.append((x[start : start + s], y[start : start + s]))
        start += s

    return out


def _fit(approx: Any, generator: torch.Generator | None = None, sizes=(5, 3, 4)) -> Tensor:
    post = lk.laplace(
        _model(),
        _batches(sizes),
        likelihood=lk.Categorical(),
        hessian_approx=approx,
        subset=lk.subsets.all,
        hyper_tuner=None,
        generator=generator,
    )

    return _diag(post)


def _diag(post: lk.LaplacePosterior) -> Tensor:
    d = post.variance

    return torch.cat([v.reshape(-1) for v in d.values()])


SAMPLED = [
    pytest.param(lambda: lk.DiagGGN(fisher="mc", mc_samples=2), id="sampled-fisher"),
    pytest.param(lambda: lk.KFAC(fisher="mc"), id="kfac-mc"),
]


def _post() -> lk.LaplacePosterior:
    return lk.laplace(
        _model(),
        _batches(),
        likelihood=lk.Categorical(),
        hessian_approx=lk.DiagGGN(),
        subset=lk.subsets.all,
        hyper_tuner=None,
    )


def _x() -> Tensor:
    return torch.randn(4, 3, generator=torch.Generator().manual_seed(9), dtype=F64)


def _draws() -> dict[str, Callable[..., Tensor]]:
    post, x = _post(), _x()
    mc = lk.Linearized(link=lk.MC(n_samples=8), n_samples=8)

    return {
        "sample_outputs": lambda **kw: lk.sample_outputs(post, x, 3, **kw),
        "network": lambda **kw: lk.sample_outputs(post, x, 3, kind="network", **kw),
        "predict_outputs": lambda **kw: lk.predict_outputs(
            post, x, n_samples=4, **kw
        ).cov.diagonal(),
        "predict-mc": lambda **kw: lk.predict(post, x, mc, **kw).probs,
        "predict-weights": lambda **kw: lk.predict(post, x, lk.WeightSamples(4), **kw).probs,
    }


# --- torch.manual_seed --------------------------------------------------------------------


@pytest.mark.parametrize("make", SAMPLED)
def test_manual_seed_repeats_a_sampled_fit(make) -> None:
    def fit(seed: int) -> Tensor:
        torch.manual_seed(seed)
        return _fit(make())

    a, b, c = fit(0), fit(0), fit(1)

    torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert not torch.allclose(a, c)


@pytest.mark.parametrize("name", list(_draws()))
def test_manual_seed_repeats_draws(name) -> None:
    draw = _draws()[name]

    def run(seed: int) -> Tensor:
        torch.manual_seed(seed)
        return draw()

    a, b, c = run(0), run(0), run(1)

    torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert not torch.allclose(a, c)


def test_unseeded_reruns_differ() -> None:
    torch.manual_seed(0)
    draw = _draws()["sample_outputs"]

    assert not torch.allclose(draw(), draw())


# --- generator= ---------------------------------------------------------------------------


@pytest.mark.parametrize("make", SAMPLED)
def test_generator_makes_a_fit_independent_of_the_global_rng(make) -> None:
    torch.manual_seed(0)
    a = _fit(make(), torch.Generator().manual_seed(3))
    torch.rand(100)  # move the global RNG
    b = _fit(make(), torch.Generator().manual_seed(3))
    c = _fit(make(), torch.Generator().manual_seed(4))

    torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert not torch.allclose(a, c)


@pytest.mark.parametrize("name", list(_draws()))
def test_generator_makes_draws_independent_of_the_global_rng(name) -> None:
    draw = _draws()[name]

    torch.manual_seed(0)
    a = draw(generator=torch.Generator().manual_seed(3))
    torch.manual_seed(1)
    b = draw(generator=torch.Generator().manual_seed(3))

    torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_a_generator_advances_like_torch() -> None:
    g = torch.Generator().manual_seed(3)
    draw = _draws()["sample_outputs"]

    assert not torch.allclose(draw(generator=g), draw(generator=g))


def test_fit_draws_do_not_touch_the_global_rng_beyond_the_seed() -> None:
    torch.manual_seed(0)
    _fit(lk.DiagGGN(fisher="mc"), torch.Generator().manual_seed(3))
    after = torch.rand(1)

    torch.manual_seed(0)
    _fit(lk.DiagGGN(), torch.Generator().manual_seed(3))

    torch.testing.assert_close(torch.rand(1), after)


# --- invariance to batching ---------------------------------------------------------------


def test_a_batch_draws_by_its_position_not_by_the_order_of_updates() -> None:
    """Another rank, or a later merge, folds a batch at the same position: same draws."""

    model = _model()
    params = {k: v.detach() for k, v in model.named_parameters()}
    fn = lk.core.ModelFn.from_module(model)
    method = lk.DiagGGN(fisher="mc")
    fit = lk.core.fit_curvature(
        fn, lk.Categorical(), params, method, generator=torch.Generator().manual_seed(3)
    )
    (x1, y1), (x2, y2) = _batches((5, 7))[:2]
    b1, b2 = Batch(x1, y1), Batch(x2, y2)

    whole = fit.update(fit.update(fit.init(), b1), b2)

    # b2 alone, starting where b1 ends, then b1: the merge equals the sequential fold.
    empty = fit.init()
    at_b2 = core.FitStat(empty.curvature, None, torch.tensor(b1.size))
    second = fit.update(at_b2, b2)
    first = fit.update(fit.init(), b1)
    merged = core.FitStat(first.curvature + second.curvature, None, whole.count)

    got, want = fit.finalize(merged), fit.finalize(whole)

    for b in want.blocks:
        torch.testing.assert_close(got.blocks[b].diag, want.blocks[b].diag, rtol=0, atol=0)

    # Not a coincidence: b2 at another position draws differently.
    moved = fit.finalize(fit.update(fit.init(), b2)).blocks
    there = fit.finalize(second).blocks

    assert not all(torch.allclose(moved[b].diag, there[b].diag) for b in moved)


# --- what the protocols receive -----------------------------------------------------------


class _Spy:
    """A curvature that records the generator each update gets, and draws from it."""

    requires = Requirements()

    def __init__(self) -> None:
        self.seen: list[Any] = []

    def init(self, params: Params, ctx: FitContext, /) -> SumStat:
        return SumStat({"x": torch.zeros((), dtype=F64)}, torch.zeros((), dtype=F64))

    def update(
        self,
        stat: SumStat,
        params: Params,
        batch: Batch,
        generator: torch.Generator,
        ctx: FitContext,
        /,
    ) -> SumStat:
        self.seen.append(generator)
        z = torch.randn((), generator=generator, dtype=F64, device=generator.device)

        return stat + SumStat({"x": z}, torch.tensor(float(batch.size), dtype=F64))

    def finalize(self, stat: SumStat, /) -> Tensor:
        return stat.sum["x"]


class _Source:
    """A data source that records the generator of each pass and shuffles with it."""

    n_data = 12

    def __init__(self) -> None:
        self.seen: list[Any] = []

    def __call__(self, generator: torch.Generator, /) -> list[Batch]:
        self.seen.append(generator)
        order = torch.randperm(4, generator=generator)
        x, y = _batches((4, 8))[0]

        return [Batch(x[order[:3]], y[order[:3]])]


def test_update_and_data_receive_fresh_generators_on_the_params_device() -> None:
    model = _model()
    params = {k: v.detach() for k, v in model.named_parameters()}
    spy, src = _Spy(), _Source()
    fit = lk.core.fit_curvature(lk.core.ModelFn.from_module(model), lk.Categorical(), params, spy)
    core.run(fit, src)

    assert spy.seen and src.seen
    assert all(isinstance(g, torch.Generator) for g in spy.seen + src.seen)
    assert all(g.device == next(iter(params.values())).device for g in spy.seen)
    assert len({id(g) for g in spy.seen}) == len(spy.seen)


def test_prepare_receives_a_generator_on_the_items_device() -> None:
    seen: list[Any] = []

    def prepare(raw: Any, generator: torch.Generator, /) -> Batch:
        seen.append(generator)
        return Batch(raw[0], raw[1])

    src = lk.data.from_loader(_batches(), prepare=prepare)
    list(src(torch.Generator().manual_seed(0)))

    assert len(seen) == 3
    assert all(isinstance(g, torch.Generator) and g.device.type == "cpu" for g in seen)


def test_predict_hands_the_generator_to_the_predictive_as_it_is() -> None:
    seen: list[Any] = []

    class Spy:
        requires = Requirements()

        def __call__(self, posterior, inputs, /, *, generator=None):  # noqa: ANN001, ANN204
            seen.append(generator)
            return lk.Linearized()(posterior, inputs, generator=generator)

    g = torch.Generator().manual_seed(0)
    lk.predict(_post(), _x(), Spy(), generator=g)
    lk.predict(_post(), _x(), Spy())

    assert seen == [g, None]


def test_weight_samples_hands_the_posterior_a_fresh_generator_on_the_means_device(
    monkeypatch,
) -> None:
    post, x = _post(), _x()
    seen: list[torch.Generator] = []
    sample = lk.LaplacePosterior.sample

    def spy(self, n, /, *, generator=None):  # noqa: ANN001, ANN202
        seen.append(generator)
        return sample(self, n, generator=generator)

    monkeypatch.setattr(lk.LaplacePosterior, "sample", spy)
    g = torch.Generator().manual_seed(0)
    lk.predict(post, x, lk.WeightSamples(4), generator=g)

    mean_device = next(iter(post.mean.values())).device

    assert len(seen) == 1 and seen[0] is not g
    assert isinstance(seen[0], torch.Generator) and seen[0].device == mean_device
    # The user's generator gave one seed to the fresh one.
    assert not torch.equal(g.get_state(), torch.Generator().manual_seed(0).get_state())


def test_noise_draws_on_the_generators_device_and_moves(monkeypatch) -> None:
    """Without a second device, check that the draws follow the generator's device."""

    post = _post()
    devices: list[torch.device] = []
    randn = torch.randn

    def spy(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        devices.append(torch.device(kwargs.get("device", "cpu")))
        return randn(*args, **kwargs)

    monkeypatch.setattr(torch, "randn", spy)
    post.sample_centered(2, generator=torch.Generator(device="cpu"))

    assert devices and all(d.type == "cpu" for d in devices)


def test_a_deterministic_predict_leaves_the_global_rng_alone() -> None:
    post, x = _post(), _x()
    torch.manual_seed(0)
    before = torch.get_rng_state()

    lk.predict(post, x, lk.Linearized(n_samples="exact"))
    lk.predict_outputs(post, x, n_samples="exact")

    assert torch.equal(torch.get_rng_state(), before)


# --- the public helpers -------------------------------------------------------------------


def test_spawn_generator_follows_torch_manual_seed_or_its_source() -> None:
    torch.manual_seed(0)
    a = core.spawn_generator().initial_seed()
    torch.manual_seed(0)
    b, c = core.spawn_generator().initial_seed(), core.spawn_generator().initial_seed()

    assert a == b != c

    g, h = torch.Generator().manual_seed(1), torch.Generator().manual_seed(1)
    before = torch.get_rng_state()

    assert core.spawn_generator(g).initial_seed() == core.spawn_generator(h).initial_seed()
    assert torch.equal(torch.get_rng_state(), before)
    assert core.spawn_generator(g).initial_seed() != core.spawn_generator(g).initial_seed()
    assert core.spawn_generator(device="cpu").device.type == "cpu"


def test_generator_for_batch_depends_on_the_state_and_the_count_only() -> None:
    g = torch.Generator().manual_seed(3)
    state = g.get_state().clone()
    a = core.generator_for_batch(g, 5).initial_seed()

    assert torch.equal(g.get_state(), state)  # the source does not advance
    assert core.generator_for_batch(g, 5).initial_seed() == a
    assert core.generator_for_batch(g, 6).initial_seed() != a
    assert core.generator_for_batch(torch.Generator().manual_seed(4), 5).initial_seed() != a

    torch.rand(4, generator=g)  # another state: other draws

    assert core.generator_for_batch(g, 5).initial_seed() != a


def test_generator_for_batch_tells_unseeded_generators_apart() -> None:
    g, h = torch.Generator(), torch.Generator()  # the same default seed
    torch.rand(1, generator=g)
    torch.rand(2, generator=h)

    seeds = [core.generator_for_batch(x, 0).initial_seed() for x in (g, h)]

    assert seeds[0] != seeds[1]


def test_from_loader_prepares_with_generator_for_batch() -> None:
    seeds: list[int] = []

    def prepare(raw: Any, generator: torch.Generator, /) -> Batch:
        seeds.append(generator.initial_seed())
        return Batch(raw[0], raw[1])

    src = lk.data.from_loader(_batches((5, 3, 4)), prepare=prepare)
    list(src(torch.Generator().manual_seed(0)))

    pass_gen = core.spawn_generator(torch.Generator().manual_seed(0))
    want = [core.generator_for_batch(pass_gen, n).initial_seed() for n in (0, 5, 8)]

    assert seeds == want


# --- keyword-only generators --------------------------------------------------------------


def test_generators_are_keyword_only() -> None:
    model = _model()
    params = {k: v.detach() for k, v in model.named_parameters()}
    fn, g = lk.core.ModelFn.from_module(model), torch.Generator()

    with pytest.raises(TypeError):
        lk.core.fit_curvature(fn, lk.Categorical(), params, lk.DiagGGN(), g)  # ty: ignore[too-many-positional-arguments]

    fit = lk.core.fit_curvature(
        fn, lk.Categorical(), params, lk.DiagGGN(), generator=g, capabilities=None
    )

    with pytest.raises(TypeError):
        core.run(fit, _Source(), g)  # ty: ignore[too-many-positional-arguments]

    core.run(fit, _Source(), generator=g)


# --- LowRankGGN's sketch ---------------------------------------------------------------------


def test_low_rank_sketch_follows_the_fit_generator() -> None:
    """The test matrix comes from the fit's seed: the same generator, the same sketch."""

    model = nn.Sequential(nn.Linear(3, 6), nn.Tanh(), nn.Linear(6, 2))
    g = torch.Generator().manual_seed(5)
    x, y = torch.randn(10, 3, generator=g), torch.randn(10, 2, generator=g)

    def fit(seed: int) -> Tensor:
        post = lk.laplace(
            model,
            [(x, y)],
            lk.Gaussian(sigma=1.0),
            hessian_approx=lk.LowRankGGN(rank=2, oversample=2),
            subset=lk.subsets.all,
            hyper_tuner=None,
            generator=torch.Generator().manual_seed(seed),
        )

        return post.curvature.op.factor

    assert torch.equal(fit(0), fit(0))
    assert not torch.allclose(fit(0), fit(1))

    # Building the method draws nothing.
    torch.manual_seed(0)
    before = torch.get_rng_state()
    lk.LowRankGGN(rank=2)

    with pytest.raises(ValueError, match="rank"):
        lk.LowRankGGN(rank=0)

    assert torch.equal(torch.get_rng_state(), before)


# --- built-ins are written like plugins ---------------------------------------------------


def test_builtins_and_plugins_import_no_private_core_names() -> None:
    import ast
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    files = [*(root / "src" / "laplacekit").glob("*.py"), *root.glob("plugins/*/src/*/*.py")]
    private = []

    for path in files:
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom) and node.module in (
                "laplacekit",
                "laplacekit.core",
            ):
                private += [f"{path.name}: {a.name}" for a in node.names if a.name.startswith("_")]

    assert private == []


# --- no keys in the public API ------------------------------------------------------------


def _public_callables(module: Any) -> list[tuple[str, Any]]:
    out = []

    for name in getattr(module, "__all__", ()):
        obj = getattr(module, name, None)

        if obj is None or isinstance(obj, type(core)):
            continue

        out.append((f"{module.__name__}.{name}", obj))

        if inspect.isclass(obj):
            for attr, member in vars(obj).items():
                if callable(member) and (not attr.startswith("_") or attr == "__call__"):
                    out.append((f"{module.__name__}.{name}.{attr}", member))

    return out


MODULES = (
    "laplacekit",
    "laplacekit.core",
    "laplacekit.data",
    "laplacekit.testing",
    "laplacekit.function_space",
    "laplacekit.predictive",
    *PLUGINS,
)


@pytest.mark.parametrize("name", MODULES)
def test_no_key_parameters_in_the_public_api(name) -> None:
    module = importlib.import_module(name)
    offenders = []

    for where, obj in _public_callables(module):
        try:
            sig = inspect.signature(obj)
        except (TypeError, ValueError):
            continue

        for p in sig.parameters.values():
            if p.name == "key" or "PRNGKey" in str(p.annotation):
                offenders.append(f"{where}({p.name})")

        if "PRNGKey" in str(sig.return_annotation):
            offenders.append(f"{where} -> PRNGKey")

    assert offenders == []


def test_no_key_fields_on_predictives() -> None:
    for cls in (lk.MC, lk.Linearized, lk.WeightSamples):
        assert "key" not in {f.name for f in dataclasses.fields(cls)}


def test_prng_key_is_not_importable() -> None:
    assert not hasattr(lk, "PRNGKey")
    assert not hasattr(core, "PRNGKey")
    assert not hasattr(core, "batch_keys")

    with pytest.raises(ImportError):
        exec("from laplacekit import PRNGKey")  # noqa: S102

    with pytest.raises(ImportError):
        exec("from laplacekit.core import PRNGKey")  # noqa: S102
