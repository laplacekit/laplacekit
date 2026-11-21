import copy
from typing import Any, cast

import pytest
import torch
from torch import Tensor
from torch.distributions import Distribution, Normal
from torch.func import grad, jvp, vmap

from laplacekit import curvature
from laplacekit.core import (
    Batch,
    Capabilities,
    CapabilityError,
    CurvatureFit,
    Done,
    Finished,
    FitContext,
    FitStat,
    HasTargetCheck,
    LaplacePosterior,
    Layout,
    ModelFn,
    Params,
    Requirements,
    Running,
    StagedCurvatureFit,
    SumStat,
    all_reduce,
    chain,
    fit_curvature,
    is_transformed,
    log_lik,
    log_prob,
    per_block,
    probe,
    tree_leaves,
)
from laplacekit.core import run as core_run
from laplacekit.core._pytree import tree_leaves_with_path
from laplacekit.core.ops import Diag
from laplacekit.likelihoods import Categorical, Gaussian
from laplacekit.priors import Isotropic


class Gauss:
    def log_prob(self, f: Tensor, y: Tensor, /) -> Tensor:
        return self.distribution(f).log_prob(y)

    def distribution(self, f: Tensor, /) -> Distribution:
        return Normal(f, 1.0)


def _model() -> ModelFn:
    return ModelFn(lambda p, x: x @ p["lin.weight"].T)


def _params() -> Params:
    return {"lin.weight": torch.tensor([[1.0, 2.0]])}


def _data(targets: bool = True) -> list[Batch]:
    g = torch.Generator().manual_seed(0)
    x = torch.randn(10, 2, generator=g)
    y = torch.randn(10, 1, generator=g)

    return [Batch(x[i : i + 4], y[i : i + 4] if targets else None) for i in range(0, 10, 4)]


class Sum:
    """Sums the inputs; records the seed of each generator it is given."""

    requires = Requirements()

    def __init__(self, needs: Requirements | None = None) -> None:
        self.seeds: list[int] = []
        self.requires = needs or Requirements()

    def init(self, params: Params, ctx: FitContext, /) -> SumStat:
        return SumStat({"x": torch.zeros(2)}, torch.zeros(()))

    def update(
        self, stat: SumStat, params: Params, batch: Batch, gen: torch.Generator, ctx: FitContext, /
    ) -> SumStat:
        self.seeds.append(gen.initial_seed())

        assert isinstance(batch.inputs, Tensor)

        return stat + SumStat({"x": batch.inputs.sum(0)}, torch.tensor(float(batch.size)))

    def finalize(self, stat: SumStat) -> Tensor:
        return stat.sum["x"]


def test_fold_and_start_agree() -> None:
    fit = fit_curvature(_model(), Gauss(), _params(), Sum())

    assert isinstance(fit, CurvatureFit)
    stat = fit.init()

    for b in _data():
        stat = fit.update(stat, b)

    assert stat.count == 10
    step = fit.start()

    assert isinstance(step, Running)
    s2 = step.init()

    for b in _data():
        s2 = step.update(s2, b)

    fin = step.finish(s2)

    assert isinstance(fin, Finished)
    assert fin.count == 10
    torch.testing.assert_close(fin.result, fit.finalize(stat))


def test_log_lik() -> None:
    fit = fit_curvature(_model(), Gauss(), _params(), Sum())
    stat = fit.init()

    for b in _data():
        stat = fit.update(stat, b)

    xs = torch.cat([b.inputs for b in _data()])
    ys = torch.cat([b.targets for b in _data()])
    expected = log_prob(Gauss(), _model()(_params(), xs), ys).sum()

    assert stat.data_fit is not None
    torch.testing.assert_close(stat.data_fit, expected)

    stat = fit.init()

    for b in _data(targets=False):
        stat = fit.update(stat, b)

    assert stat.data_fit is None


def test_generators() -> None:
    def run() -> Sum:
        c = Sum()
        fit = fit_curvature(
            _model(), Gauss(), _params(), c, generator=torch.Generator().manual_seed(3)
        )
        stat = fit.init()

        for b in _data():
            stat = fit.update(stat, b)

        return c

    a, b = run(), run()

    # The same seed, the same generators; one fresh generator per batch.
    assert a.seeds == b.seeds
    assert len(set(a.seeds)) == 3

    # A batch's generator depends on the examples before it, not on the updates before it.
    c = Sum()
    fit = fit_curvature(_model(), Gauss(), _params(), c, generator=torch.Generator().manual_seed(3))
    last = _data()[2]
    fit.update(FitStat(fit.init().curvature, None, torch.tensor(8)), last)

    assert c.seeds == a.seeds[2:]


def test_generators_follow_the_global_rng() -> None:
    def run() -> list[int]:
        c = Sum()
        fit = fit_curvature(_model(), Gauss(), _params(), c)
        fit.update(fit.init(), _data()[0])

        return c.seeds

    torch.manual_seed(0)
    a = run()
    torch.manual_seed(0)
    b = run()
    c = run()

    assert a == b != c


def test_generators_across_two_passes() -> None:
    def run() -> TwoPass:
        plan = TwoPass()
        g = torch.Generator().manual_seed(3)
        step: Running[Any] | Finished[Any] = fit_curvature(
            _model(), Gauss(), _params(), plan, generator=g
        ).start()

        while isinstance(step, Running):
            stat = step.init()

            for b in _data():
                stat = step.update(stat, b)

            step = step.finish(all_reduce(stat))

        return plan

    a, b = run(), run()

    for p, q in zip((a.first, a.second), (b.first, b.second), strict=True):
        assert p.seeds == q.seeds

    # Each pass draws afresh: no batch shares a generator with the other pass.
    assert len(a.first.seeds) == len(a.second.seeds) == 3
    assert not set(a.first.seeds) & set(a.second.seeds)


class TwoPass:
    requires = Requirements()

    def __init__(self) -> None:
        self.seeds: list[int] = []

    def plan(self, params: Params, ctx: FitContext, /) -> Any:
        first = Sum()
        second = _Centered()
        self.first, self.second = first, second

        def then(mean_sum: Tensor) -> Any:
            second.mean = mean_sum / 10
            return chain(second, then=lambda r: Done(r))

        return chain(first, then=then)


class _Centered(Sum):
    mean: Tensor

    def update(
        self, stat: SumStat, params: Params, batch: Batch, gen: torch.Generator, ctx: FitContext, /
    ) -> SumStat:
        self.seeds.append(gen.initial_seed())

        assert isinstance(batch.inputs, Tensor)

        return stat + SumStat({"x": (batch.inputs - self.mean).sum(0)}, torch.tensor(1.0))


def test_staged_two_passes() -> None:
    plan = TwoPass()
    fit = fit_curvature(_model(), Gauss(), _params(), plan)

    assert isinstance(fit, StagedCurvatureFit)
    step: Running[Any] | Finished[Any] = fit.start()
    n = 0

    while isinstance(step, Running):
        stat = step.init()

        for b in _data():
            stat = step.update(stat, b)

        step = step.finish(all_reduce(stat))
        n += 1

    assert n == 2
    assert len(step.stats) == 2
    xs = torch.cat([b.inputs for b in _data()])
    torch.testing.assert_close(step.result, (xs - xs.mean(0)).sum(0))

    assert {int(k) for k in plan.first.seeds}.isdisjoint({int(k) for k in plan.second.seeds})


def test_requirements_checked_only_with_caps() -> None:
    needy = Sum(Requirements(vmap=True))
    fit_curvature(_model(), Gauss(), _params(), needy).init()  # no caps: no check

    with pytest.raises(CapabilityError, match="vmap"):
        fit_curvature(_model(), Gauss(), _params(), needy, capabilities=Capabilities()).init()

    with pytest.raises(CapabilityError, match="vmap"):
        fit_curvature(_model(), Gauss(), _params(), needy, capabilities=Capabilities()).start()

    fit_curvature(_model(), Gauss(), _params(), needy, capabilities=Capabilities(vmap=True)).start()


def test_posterior_fills_in_context() -> None:
    caps = Capabilities(vjp=True)
    fit = fit_curvature(_model(), Gauss(), _params(), Sum(), capabilities=caps)
    op = Diag(torch.ones(2))
    prior = Isotropic(torch.tensor(1.0))
    post = fit.posterior(op, prior=prior, n_data=torch.tensor(10.0))

    assert isinstance(post, LaplacePosterior)
    assert post.caps is caps
    assert post.model is fit.model
    assert post.likelihood is fit.likelihood
    assert post.layout == Layout.by_module(_params())


def test_all_reduce_without_a_process_group() -> None:
    s = SumStat({"x": torch.ones(2)}, torch.ones(()))

    assert all_reduce(s) is s  # multi-process cases: tests/test_distributed.py


def test_all_reduce_restores_devices(monkeypatch: pytest.MonkeyPatch) -> None:
    """One process, three faked ranks: the fold in rank order, every leaf on the local device.

    The local rank (1) saw no targets, so its `data_fit` is None and the other ranks' data-fit
    terms have no local leaf to follow.
    """

    def stat(x: list[float], w: float, fit: float | None, n: int) -> FitStat[SumStat]:
        data_fit = None if fit is None else torch.tensor(fit)

        return FitStat(SumStat({"x": torch.tensor(x)}, torch.tensor(w)), data_fit, torch.tensor(n))

    parts = [
        stat([1.0, 2.0], 3.0, -4.0, 5),
        stat([6.0, 7.0], 8.0, None, 9),
        stat([1.5, 2.5], 3.5, -0.5, 2),
    ]
    local = parts[1]

    def all_gather_object(out: list[Any], obj: Any, group: Any = None) -> None:
        assert torch.equal(obj.curvature.sum["x"], local.curvature.sum["x"])  # sends its own
        out[:] = [p if r != 1 else obj for r, p in enumerate(parts)]

    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group=None: 3)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda group=None: 1)
    monkeypatch.setattr(torch.distributed, "all_gather_object", all_gather_object)

    reduced = all_reduce(local)

    assert torch.equal(reduced.curvature.sum["x"], torch.tensor([8.5, 11.5]))
    assert torch.equal(reduced.curvature.weight, torch.tensor(14.5))
    assert reduced.data_fit is not None
    assert torch.equal(reduced.data_fit, torch.tensor(-4.5))
    assert torch.equal(reduced.count, torch.tensor(16))

    devices = {path: leaf.device for path, leaf in tree_leaves_with_path(reduced)}

    assert devices == {
        ".curvature.sum['x']": local.curvature.sum["x"].device,
        ".curvature.weight": local.curvature.weight.device,
        ".data_fit": torch.device("cpu"),  # no local leaf: the CPU, as no local leaf is off it
        ".count": local.count.device,
    }


def test_weighted_log_lik_and_int_count() -> None:
    x = torch.randn(4, 2)
    y = torch.randn(4, 1)
    w = torch.tensor([0.5, 1.0, 2.0, 0.0])
    fit = fit_curvature(_model(), Gauss(), _params(), Sum())
    stat = fit.update(fit.init(), Batch(x, y, w))
    f = _model()(_params(), x)
    expected = (Normal(f, 1.0).log_prob(y).sum(1) * w).sum()

    assert stat.data_fit is not None
    torch.testing.assert_close(stat.data_fit, expected)

    assert stat.count.dtype == torch.int64
    assert stat.count == 4


def test_rerun_same_statistics() -> None:
    fit = fit_curvature(
        _model(), Gauss(), _params(), Sum(), generator=torch.Generator().manual_seed(1)
    )
    step = fit.start()

    assert isinstance(step, Running)

    def run() -> Any:
        stat = step.init()

        for b in _data():
            stat = step.update(stat, b)

        return stat

    a, b = run(), run()
    torch.testing.assert_close(a.curvature.sum["x"], b.curvature.sum["x"])
    torch.testing.assert_close(a.data_fit, b.data_fit)

    assert a.count == b.count


# --- batch requirements ----------------------------------------------------------------


def _mapping_batch() -> Batch:
    return Batch({"x": torch.randn(4, 2)}, torch.randn(4, 1))


@pytest.mark.parametrize(
    ("needs", "batch", "match"),
    [
        (Requirements(needs_targets=True), _data(targets=False)[0], "needs targets"),
        (Requirements(input_type=Tensor), _mapping_batch(), "needs Tensor inputs, got dict"),
        (
            Requirements(weights=False),
            Batch(torch.randn(4, 2), torch.randn(4, 1), torch.ones(4)),
            "per-example weights",
        ),
    ],
    ids=["targets", "input_type", "weights"],
)
def test_batch_requirements_raise_before_the_curvature_sees_data(needs, batch, match) -> None:
    method = Sum(needs)

    for step in (
        fit_curvature(_model(), Gauss(), _params(), method, capabilities=Capabilities()),
        fit_curvature(_model(), Gauss(), _params(), method, capabilities=Capabilities()).start(),
    ):
        with pytest.raises(CapabilityError, match=match):
            step.update(step.init(), batch)

    assert method.seeds == []  # update never ran


class _SquaredError:
    """Only `log_prob`: no distribution, no sampling."""

    def log_prob(self, f: Tensor, y: Tensor, /) -> Tensor:
        return -((f - y) ** 2).sum(-1)


def test_weighted_fit_any_log_prob() -> None:
    g = torch.Generator().manual_seed(3)
    x, y = torch.randn(5, 2, generator=g), torch.randn(5, 1, generator=g)
    w = torch.tensor([1.0, 3.0, 2.0, 1.0, 2.0])
    reps = w.long()
    fit = fit_curvature(_model(), _SquaredError(), _params(), Sum(), capabilities=Capabilities())

    weighted = fit.update(fit.init(), Batch(x, y, w))
    duplicated = fit.update(
        fit.init(), Batch(x.repeat_interleave(reps, 0), y.repeat_interleave(reps, 0))
    )
    f = _model()(_params(), x)

    assert weighted.data_fit is not None and duplicated.data_fit is not None
    torch.testing.assert_close(weighted.data_fit, duplicated.data_fit)
    torch.testing.assert_close(weighted.data_fit, -(w * ((f - y) ** 2).sum(-1)).sum())


def test_fit_curvature_checks_the_likelihood_without_caps() -> None:
    lik = _SquaredError()
    needy = Sum(Requirements(needs_sample=True))

    with pytest.raises(CapabilityError, match="HasSample or HasDistribution"):
        fit_curvature(_model(), lik, _params(), needy)

    with pytest.raises(CapabilityError, match="HasSample or HasDistribution"):
        fit_curvature(_model(), lik, _params(), per_block({"*": needy}))

    fit_curvature(_model(), Gauss(), _params(), needy)  # a distribution can sample

    assert needy.seeds == []


def test_each_pass_checks_the_likelihood() -> None:
    class SecondNeedsSample(TwoPass):
        def plan(self, params: Params, ctx: FitContext, /) -> Any:
            second = Sum(Requirements(needs_sample=True))

            return chain(Sum(), then=lambda _: chain(second, then=lambda r: Done(r)))

    step: Any = fit_curvature(_model(), _SquaredError(), _params(), SecondNeedsSample()).start()
    stat = step.init()

    for b in _data():
        stat = step.update(stat, b)

    step = step.finish(stat)

    with pytest.raises(CapabilityError, match="HasSample"):
        step.init()


def _mlp_fit(lik_kind: str) -> tuple[torch.nn.Module, Any, list[Batch], list[Batch]]:
    torch.manual_seed(0)
    out = 3 if lik_kind == "cat" else 2
    module = torch.nn.Sequential(torch.nn.Linear(3, 5), torch.nn.Tanh(), torch.nn.Linear(5, out))
    lik: Any = (
        Categorical()
        if lik_kind == "cat"
        else copy.replace(Gaussian(), sigma=torch.as_tensor(torch.tensor(0.5)))
    )
    x = torch.randn(12, 3)
    y = torch.randint(0, 3, (12,)) if lik_kind == "cat" else torch.randn(12, out)
    with_y = [Batch(x[i : i + 4], y[i : i + 4]) for i in range(0, 12, 4)]
    without_y = [Batch(b.inputs) for b in with_y]

    return module, lik, with_y, without_y


def _expected_log_lik(module: torch.nn.Module, lik: Any, batches: list[Batch]) -> Tensor:
    """The summed log-likelihood, straight from torch."""

    with torch.no_grad():
        total = torch.zeros(())

        for b in batches:
            assert isinstance(b.inputs, Tensor) and isinstance(b.targets, Tensor)
            f = module(b.inputs)

            if isinstance(lik, Categorical):
                total = total - torch.nn.functional.cross_entropy(f, b.targets, reduction="sum")
            else:
                total = total + Normal(f, lik.sigma).log_prob(b.targets).sum()

    return total


class _Counting:
    def __init__(self, module: torch.nn.Module) -> None:
        self.inner = ModelFn.from_module(module)
        self.calls = 0

    def __call__(self, params: Params, inputs: Any) -> Any:
        self.calls += 1
        return self.inner(params, inputs)


def _fold(fn: ModelFn, lik: Any, params: Params, approx: Any, batches: list[Batch]) -> Any:
    fit = fit_curvature(fn, lik, params, approx)
    stat = fit.init()

    for b in batches:
        stat = fit.update(stat, b)

    return stat


@pytest.mark.parametrize("lik_kind", ["cat", "gauss"])
@pytest.mark.parametrize(
    "approx",
    [
        curvature.DiagGGN(),
        curvature.DenseGGN(),
        curvature.DiagGGN(fisher="mc"),
        per_block({"0": curvature.DiagGGN(), "2": curvature.DenseGGN()}),
    ],
    ids=["Diag", "DenseGGN", "DiagMCFisher", "per_block"],
)
def test_one_forward_per_batch(approx: Any, lik_kind: str) -> None:
    module, lik, with_y, without_y = _mlp_fit(lik_kind)
    params = {k: v.detach() for k, v in module.named_parameters()}
    counting = _Counting(module)
    fn = ModelFn(counting, module)

    _fold(fn, lik, params, approx, without_y)
    calls_without = counting.calls
    counting.calls = 0
    stat = _fold(fn, lik, params, approx, with_y)

    assert counting.calls == calls_without
    assert stat.data_fit is not None
    torch.testing.assert_close(log_lik(lik, stat.data_fit), _expected_log_lik(module, lik, with_y))


class _NoModel:
    """Never calls the model."""

    requires = Requirements()

    def init(self, params: Params, ctx: FitContext, /) -> SumStat:
        return SumStat({"x": torch.zeros(())}, torch.zeros(()))

    def update(
        self, stat: SumStat, params: Params, batch: Batch, gen: torch.Generator, ctx: FitContext, /
    ) -> SumStat:
        return stat + SumStat({"x": torch.zeros(())}, torch.tensor(float(batch.size)))

    def finalize(self, stat: SumStat) -> Tensor:
        return stat.sum["x"]


def test_data_fit_without_a_model_call() -> None:
    """A curvature that never calls `ctx.model` still gets the data-fit term."""

    module, lik, with_y, _ = _mlp_fit("cat")
    params = {k: v.detach() for k, v in module.named_parameters()}
    counting = _Counting(module)
    stat = _fold(ModelFn(counting, module), lik, params, _NoModel(), with_y)

    assert counting.calls == len(with_y)  # the core's own forward, once per batch
    assert stat.data_fit is not None
    torch.testing.assert_close(stat.data_fit, _expected_log_lik(module, lik, with_y))


class _OddCalls:
    """Calls the model only in ways that must not be reused for the data-fit term."""

    requires = Requirements()

    def __init__(self, how: str) -> None:
        self.how = how

    def init(self, params: Params, ctx: FitContext, /) -> SumStat:
        return SumStat({"x": torch.zeros(())}, torch.zeros(()))

    def update(
        self, stat: SumStat, params: Params, batch: Batch, gen: torch.Generator, ctx: FitContext, /
    ) -> SumStat:
        x = batch.inputs

        def total(q: Params) -> Tensor:
            return cast(Tensor, ctx.model(q, x)).sum()

        if self.how == "grad":
            g = grad(total)(dict(params))
            s = sum(v.sum() for v in g.values())
        elif self.how == "vmap":
            stacked = {k: torch.stack([v, 2 * v]) for k, v in params.items()}
            s = vmap(total)(stacked).sum()
        elif self.how == "other_params":
            s = total({k: v + 1 for k, v in params.items()})
        else:
            assert isinstance(x, Tensor)
            s = total(dict(params)) + cast(Tensor, ctx.model(params, x.clone())).sum()

        return stat + SumStat({"x": s.detach()}, torch.tensor(float(batch.size)))

    def finalize(self, stat: SumStat) -> Tensor:
        return stat.sum["x"]


@pytest.mark.parametrize("how", ["grad", "vmap", "other_params", "full_then_copy"])
def test_only_the_full_batch_forward_at_the_fit_params_is_reused(how: str) -> None:
    module, lik, with_y, _ = _mlp_fit("cat")
    params = {k: v.detach() for k, v in module.named_parameters()}
    stat = _fold(ModelFn.from_module(module), lik, params, _OddCalls(how), with_y)

    assert stat.data_fit is not None
    torch.testing.assert_close(stat.data_fit, _expected_log_lik(module, lik, with_y))


def test_data_fit_stays_detached() -> None:
    """D8 as before the reuse: the data-fit term is a value, even for a differentiable fit."""

    module, lik, with_y, _ = _mlp_fit("cat")
    params = {k: v.detach().requires_grad_() for k, v in module.named_parameters()}

    for approx in (curvature.DiagGGN(fisher="mc"), curvature.DenseGGN(), _NoModel()):
        stat = _fold(ModelFn.from_module(module), lik, params, approx, with_y)

        assert stat.data_fit is not None
        assert not stat.data_fit.requires_grad


def test_low_precision_data_fit_is_promoted() -> None:
    module, lik, with_y, _ = _mlp_fit("cat")
    module = module.to(torch.bfloat16)
    batches = [Batch(cast(Tensor, b.inputs).bfloat16(), b.targets) for b in with_y]
    params = {k: v.detach() for k, v in module.named_parameters()}
    stat = _fold(ModelFn.from_module(module), lik, params, curvature.DiagGGN(), batches)

    assert stat.data_fit is not None
    assert stat.data_fit.dtype == torch.float32


def test_data_fit_under_a_transform() -> None:
    """Under `torch.func.grad` the recorder is skipped, but the data-fit term is kept."""

    module, lik, with_y, _ = _mlp_fit("cat")
    params = {k: v.detach() for k, v in module.named_parameters()}

    def trace(p: Params) -> tuple[Tensor, Tensor | None]:
        bound = fit_curvature(ModelFn.from_module(module), lik, p, curvature.DiagGGN())
        stat = bound.update(bound.init(), with_y[0])
        total = sum(s.sum() for s in stat.curvature.sum.values())

        return cast(Tensor, total), stat.data_fit

    _, data_fit = grad(trace, has_aux=True)(params)

    assert data_fit is not None
    torch.testing.assert_close(data_fit, _expected_log_lik(module, lik, with_y[:1]))


def test_posterior_without_caps_says_they_were_not_probed() -> None:
    """The by-hand path without caps: the error names the missing probe, not the model."""

    from laplacekit.predictive import Linearized

    module, lik, with_y, _ = _mlp_fit("cat")
    params = {k: v.detach() for k, v in module.named_parameters()}
    fit = fit_curvature(ModelFn.from_module(module), lik, params, curvature.DiagGGN())
    stat = fit.update(fit.init(), with_y[0])
    post = fit.posterior(fit.finalize(stat), prior=Isotropic(torch.tensor(1.0)), n_data=stat.count)

    with pytest.raises(CapabilityError, match=r"not probed; pass caps=probe\(") as info:
        Linearized().resolve(post)

    assert "does not support" not in str(info.value)
    assert "params=params) to fit_curvature" in str(info.value)

    probed = fit_curvature(
        ModelFn.from_module(module),
        lik,
        params,
        curvature.DiagGGN(),
        capabilities=probe(ModelFn.from_module(module), lik, with_y[0], params=params),
    )
    assert (
        Linearized()
        .resolve(
            probed.posterior(
                fit.finalize(stat), prior=Isotropic(torch.tensor(1.0)), n_data=stat.count
            )
        )
        .n_samples
        == "exact"
    )


class _CheckedGauss(Gauss):
    """A plugin likelihood with a target check; records what it is given."""

    def __init__(self, limit: float = float("inf")) -> None:
        self.limit = limit
        self.seen: list[tuple[Tensor, bool]] = []

    def check_targets(self, y: Tensor, /) -> None:
        self.seen.append((y, torch._C._functorch.is_functorch_wrapped_tensor(y)))

        if bool((y.abs() > self.limit).any()):
            raise ValueError(f"_CheckedGauss targets must lie within {self.limit}")


def test_plugin_target_check_runs_once_per_batch_outside_transforms() -> None:
    lik = _CheckedGauss()

    assert isinstance(lik, HasTargetCheck)

    batches = _data()
    _fold(_model(), lik, _params(), curvature.DiagGGN(fisher="mc"), batches)

    assert len(lik.seen) == len(batches)

    for (y, wrapped), b in zip(lik.seen, batches, strict=True):
        assert torch.equal(y, cast(Tensor, b.targets))
        assert not wrapped

    lik.seen.clear()
    _fold(_model(), lik, _params(), curvature.DiagGGN(fisher="mc"), _data(targets=False))

    assert lik.seen == []


def test_plugin_target_check_refuses_before_the_curvature_sees_the_batch() -> None:
    batches = _data()
    bad = cast(Tensor, batches[1].targets).clone()
    bad[0] = 100.0
    batches[1] = Batch(batches[1].inputs, bad)
    sums: list[int] = []

    class _Spy(curvature.DiagGGN):
        def update(self, stat, params, batch, gen, ctx, /):  # noqa: ANN001, ANN202
            sums.append(batch.size)
            return super().update(stat, params, batch, gen, ctx)

    with pytest.raises(ValueError, match="within 10"):
        _fold(_model(), _CheckedGauss(limit=10.0), _params(), _Spy(fisher="mc"), batches)

    assert sums == [batches[0].size]


def test_target_check_under_grad() -> None:
    """A by-hand fit under `torch.func.grad` over the parameters still checks its targets:
    they are not transformed, so their values can be read."""

    from laplacekit.likelihoods import Bernoulli

    x = torch.randn(4, 2)
    y = torch.tensor([[0.0], [1.0], [2.0], [1.0]])

    def total(p: Params) -> Tensor:
        fit = fit_curvature(_model(), Bernoulli(), p, curvature.DiagGGN())
        stat = fit.update(fit.init(), Batch(x, y))

        return sum((s.sum() for s in stat.curvature.sum.values()), torch.zeros(()))

    with pytest.raises(ValueError, match=r"Bernoulli targets must lie in \[0, 1\]"):
        grad(total)(_params())

    # Valid targets go through, and the check sees plain tensors.
    lik = _CheckedGauss()

    def checked(p: Params) -> Tensor:
        fit = fit_curvature(_model(), lik, p, curvature.DenseGGN(fisher="mc"))
        stat = fit.update(fit.init(), Batch(x, y))

        return sum((s.sum() for s in stat.curvature.sum.values()), torch.zeros(()))

    grad(checked)(_params())

    assert [wrapped for _, wrapped in lik.seen] == [False]


def test_target_check_skipped_under_vmap() -> None:
    """Under `vmap` over batches the targets are wrapped: their values can't be read, so
    the check is skipped."""

    lik = _CheckedGauss(limit=10.0)
    xs = torch.randn(3, 4, 2)
    ys = 100 * torch.ones(3, 4, 1)

    def fold_one(x: Tensor, y: Tensor) -> Tensor:
        fit = fit_curvature(_model(), lik, _params(), Sum())

        return fit.update(fit.init(), Batch(x, y)).curvature.sum["x"]

    got = vmap(fold_one)(xs, ys)

    torch.testing.assert_close(got, xs.sum(1))
    assert lik.seen == []


def test_transformed_reads_the_functorch_interpreter_stack() -> None:
    """Guards the private `torch._C._functorch.peek_interpreter_stack` read in `_transformed`."""

    from laplacekit.core.fit import _transformed

    seen: list[bool] = []

    def f(x: Tensor) -> Tensor:
        seen.append(_transformed())
        return x.sum()

    assert not _transformed()

    vmap(f)(torch.ones(2, 3))
    grad(f)(torch.ones(3))

    assert seen == [True, True]


def test_is_transformed_reads_functorch_wrappers() -> None:
    """Guards the private `torch._C._functorch.is_functorch_wrapped_tensor` read in
    `is_transformed`: true for vmap, grad and jvp tensors, false for plain ones."""

    seen: list[bool] = []
    plain = torch.ones(3)

    def f(x: Tensor) -> Tensor:
        seen.append(is_transformed(x))
        seen.append(is_transformed(plain))  # closed over: not wrapped
        return x.sum()

    assert not is_transformed(plain)

    vmap(f)(torch.ones(2, 3))
    grad(f)(torch.ones(3))
    jvp(f, (torch.ones(3),), (torch.ones(3),))

    assert seen == [True, False, True, False, True, False]


def _counting_all_reduce(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Wrap `all_reduce` where `core.run` looks it up; one entry per call."""

    import laplacekit.core.fit as fit_module

    calls: list[int] = []
    real = fit_module.all_reduce

    def counted(stat: Any, /, group: Any = None) -> Any:
        calls.append(1)
        return real(stat, group)

    monkeypatch.setattr(fit_module, "all_reduce", counted)

    return calls


def test_run_matches_manual_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    """A `Diag` fit through `core.run` equals the hand loop of the module docstring: same
    result, same statistics, one `all_reduce` per pass, the data called with the pass's
    generator."""

    lin = torch.nn.Linear(2, 3)
    fn = ModelFn.from_module(lin)
    params = {k: v.detach().requires_grad_(True) for k, v in lin.named_parameters()}
    batches = _data()
    batches = [Batch(b.inputs, torch.randn(b.size, 3)) for b in batches]
    seen: list[int] = []

    def data(g: torch.Generator) -> list[Batch]:
        seen.append(g.initial_seed())
        return batches

    fit = fit_curvature(fn, Gaussian(), params, curvature.DiagGGN())

    with torch.no_grad():
        stat = fit.init()

        for b in batches:
            stat = fit.update(stat, b)

        expected = fit.finalize(stat)

    calls = _counting_all_reduce(monkeypatch)
    done = core_run(fit, data, generator=torch.Generator().manual_seed(5))
    again = core_run(fit, data, generator=torch.Generator().manual_seed(5))
    core_run(fit, data, generator=torch.Generator().manual_seed(6))

    assert isinstance(done, Finished) and isinstance(again, Finished)
    assert len(calls) == 3
    assert seen[0] == seen[1] != seen[2]  # the data's generator follows `generator`
    assert len(done.stats) == 1
    assert done.count == 10
    assert done.stats[0].data_fit is not None and stat.data_fit is not None
    torch.testing.assert_close(done.stats[0].data_fit, stat.data_fit)

    for got, want in zip(tree_leaves(done.result), tree_leaves(expected), strict=True):
        assert not got.requires_grad  # the loop runs under no_grad
        torch.testing.assert_close(got, want)


def test_run_reduces_once_per_pass(monkeypatch: pytest.MonkeyPatch) -> None:
    """A two-pass method: two `all_reduce` calls, and pass i reads data with its own
    generator."""

    seen: list[int] = []

    def data(g: torch.Generator) -> list[Batch]:
        seen.append(g.initial_seed())
        return _data()

    calls = _counting_all_reduce(monkeypatch)
    fit = fit_curvature(_model(), Gauss(), _params(), TwoPass())
    done = core_run(fit, data, generator=torch.Generator().manual_seed(3))

    assert len(calls) == 2
    assert len(seen) == 2 and seen[0] != seen[1]  # each pass has its own generator
    xs = torch.cat([b.inputs for b in _data()])
    torch.testing.assert_close(done.result, (xs - xs.mean(0)).sum(0))
