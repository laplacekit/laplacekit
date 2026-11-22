import collections
import copy
import dataclasses
import math

import pytest
import torch
from torch import nn
from torch.utils._python_dispatch import TorchDispatchMode

import laplacekit as lk
from laplacekit.core import Dense, tree_leaves


def blobs(n: int = 90, dtype: torch.dtype = torch.float32):
    g = torch.Generator().manual_seed(0)
    centers = torch.tensor([[-2.0, 0.0], [2.0, 0.0], [0.0, 3.0]], dtype=dtype)
    y = torch.arange(n) % 3
    x = centers[y] + 0.5 * torch.randn(n, 2, generator=g, dtype=dtype)

    return x, y


def trained(dtype: torch.dtype = torch.float32):
    torch.manual_seed(0)
    x, y = blobs(dtype=dtype)
    model = nn.Sequential(nn.Linear(2, 8), nn.Tanh(), nn.Linear(8, 3)).to(dtype)
    opt = torch.optim.Adam(model.parameters(), lr=0.05)

    for _ in range(200):
        opt.zero_grad()
        nn.functional.cross_entropy(model(x), y).backward()
        opt.step()

    return model, x, y


def loader(x, y, k: int = 3):
    n = x.shape[0] // k
    return [(x[i * n : (i + 1) * n], y[i * n : (i + 1) * n]) for i in range(k)]


def entropy(d):
    return d.entropy()


def test_one_liner_and_print():
    model, x, y = trained()
    post = lk.laplace(model, loader(x, y), likelihood=lk.Categorical())
    text = repr(post)

    for s in (
        "last_layer",
        "DenseGGN",
        "Isotropic(",
        "n_data",
        "hyper_tuner    = MarginalLikelihood()",
    ):
        assert s in text

    assert str(post) == text
    far = 10 * x.std() * torch.ones(1, 2)
    h_far = lk.predict(post, far).entropy().mean()
    h_train = lk.predict(post, x).entropy().mean()

    assert h_far > h_train


def test_tune_none():
    model, x, y = trained()
    post = lk.laplace(model, loader(x, y), likelihood=lk.Categorical(), hyper_tuner=None)

    assert "hyper_tuner    = None" in repr(post)
    assert float(post.prior.prior_precision) == 1.0
    assert lk.predict(post, x).probs.shape == (90, 3)


def test_default_tunes_the_prior():
    model, x, y = trained()
    post = lk.laplace(model, loader(x, y), likelihood=lk.Categorical())

    assert float(post.prior.prior_precision) != 1.0


def test_user_objective_is_used():
    model, x, y = trained()
    post = lk.laplace(
        model,
        loader(x, y),
        likelihood=lk.Categorical(),
        hyper_tuner=lk.Minimize(lk.objectives.predictive_nll([lk.Batch(x, y)])),
    )

    assert float(post.prior.prior_precision) != 1.0


def test_no_targets_needs_another_tuner():
    model, x, _ = trained()
    data = [lk.Batch(x[:30]), lk.Batch(x[30:60])]

    with pytest.raises(ValueError, match="hyper_tuner=None"):
        lk.laplace(model, data, likelihood=lk.Categorical())

    post = lk.laplace(model, data, likelihood=lk.Categorical(), hyper_tuner=None)

    assert "hyper_tuner    = None" in repr(post)
    assert float(post.prior.prior_precision) == 1.0


def test_sized_data_sets_scale():
    model, x, y = trained()
    data = lk.data.from_loader(loader(x, y), n_data=1000)
    post = lk.laplace(model, data, likelihood=lk.Categorical(), hyper_tuner=None)

    assert int(post.n_data) == 1000
    post = lk.laplace(model, loader(x, y), likelihood=lk.Categorical(), hyper_tuner=None)

    assert int(post.n_data) == 90


def _sequence_post(model: lk.core.ModelFn, event_size: int) -> lk.LaplacePosterior:
    g = torch.Generator().manual_seed(0)
    w = torch.randn(3, 4, generator=g, dtype=torch.float64)
    a = torch.randn(12, 12, generator=g, dtype=torch.float64)

    return lk.LaplacePosterior(
        mean={"fc.weight": w},
        curvature=Dense(a @ a.T / 12),
        prior=lk.priors.Isotropic(torch.tensor(1.0, dtype=torch.float64)),
        n_data=torch.tensor(5.0, dtype=torch.float64),
        model=model,
        likelihood=lk.Categorical(),
        layout=lk.core.Layout.by_module({"fc.weight": w}),
        caps=lk.core.Capabilities(vjp=True, vjp_twice=True, vmap=True, event_size=event_size),
    )


def test_select_last_position():
    # Per-position logits [N, T, C]; the probe's event size is of all positions, here
    # set above MAX_DENSE_EVENT: the selected [N, C] outputs still get the exact path.
    full = _sequence_post(lk.core.ModelFn(lambda p, x: x @ p["fc.weight"].T), event_size=10_000)
    last = _sequence_post(lk.core.ModelFn(lambda p, x: x[:, -1] @ p["fc.weight"].T), event_size=3)
    x = torch.randn(5, 6, 4, generator=torch.Generator().manual_seed(1), dtype=torch.float64)

    got = lk.predict(
        full, x, select=lambda f, x: f[:, -1], generator=torch.Generator().manual_seed(0)
    )
    ref = lk.predict(last, x, generator=torch.Generator().manual_seed(0))

    assert got.probs.shape == (5, 3)
    torch.testing.assert_close(got.probs, ref.probs)

    method = lk.WeightSamples(n_samples=4)
    torch.testing.assert_close(
        lk.predict(
            full,
            x,
            method,
            select=lambda f, x: f[:, -1],
            generator=torch.Generator().manual_seed(0),
        ).probs,
        lk.predict(last, x, method, generator=torch.Generator().manual_seed(0)).probs,
    )


def test_by_hand_equals_one_liner():
    model, x, y = trained(torch.float64)
    data = loader(x, y)
    post = lk.laplace(model, data, likelihood=lk.Categorical(), hyper_tuner=None)

    fn = lk.core.ModelFn.from_module(model)
    lik = lk.Categorical()
    caps = lk.core.probe(fn, lik, lk.Batch.from_pair(data[0]))
    params = lk.subsets.last_layer(fn, caps)
    fit = lk.core.fit_curvature(
        fn, lik, params, lk.curvature.default(params, caps), generator=None, capabilities=caps
    )
    stat = fit.init()

    for pair in data:
        stat = fit.update(stat, lk.Batch.from_pair(pair))

    post2 = fit.posterior(
        fit.finalize(stat), prior=lk.priors.Isotropic(torch.tensor(1.0)), n_data=stat.count
    )
    a = tree_leaves(post.precision())
    b = tree_leaves(post2.precision())

    assert len(a) == len(b)

    for u, v in zip(a, b, strict=True):
        assert torch.equal(u, v)


class _Counting:
    def __init__(self, items):
        self.items = items
        self.pulled = 0

    def __iter__(self):
        for it in self.items:
            self.pulled += 1
            yield it


def test_capability_error_before_fitting():
    class InPlace(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(2, 3)

        def forward(self, x):
            h = self.lin.bias * 1.0
            h.add_(self.lin(x).sum(0))

            return h.expand(x.shape[0], 3)

    model = InPlace()
    x, y = blobs()
    ld = _Counting(loader(x, y))

    try:
        lk.laplace(model, ld, likelihood=lk.Categorical(), hessian_approx=lk.DiagGGN())
    except lk.CapabilityError:
        pass
    else:
        pytest.fail("expected CapabilityError")

    assert ld.pulled == 1


def test_one_shot_iterable_is_rejected():
    model, x, y = trained()

    with pytest.raises(TypeError, match="re-iterable.*generator"):
        lk.laplace(model, (b for b in loader(x, y)), likelihood=lk.Categorical())


def test_laplace_leaves_model_grads_alone():
    model, x, y = trained()
    model.zero_grad(set_to_none=True)  # training left grads behind
    lk.laplace(model, loader(x, y), likelihood=lk.Categorical())

    assert all(p.grad is None for p in model.parameters())


def test_by_hand_tune_without_freezing_the_model():
    model, x, y = trained()
    model.zero_grad(set_to_none=True)
    data = loader(x, y)
    fn = lk.core.ModelFn.from_module(model)
    lik = lk.Categorical()
    caps = lk.core.probe(fn, lik, lk.Batch.from_pair(data[0]))
    params = lk.subsets.last_layer(fn, caps)
    fit = lk.core.fit_curvature(
        fn, lik, params, lk.curvature.default(params, caps), generator=None, capabilities=caps
    )
    stat = fit.init()

    for pair in data:
        stat = fit.update(stat, lk.Batch.from_pair(pair))

    post = fit.posterior(
        fit.finalize(stat), prior=lk.priors.Isotropic(torch.tensor(1.0)), n_data=stat.count
    )
    post = dataclasses.replace(post, data_fit=(stat.data_fit,))
    tuned = lk.tune(post)

    assert float(tuned.prior.prior_precision) != 1.0
    assert all(p.grad is None for p in model.parameters())


def test_per_block_prior_through_the_front_door():
    model, x, y = trained()
    prior = lk.priors.Layerwise({"0": torch.tensor(1.0), "2": torch.tensor(1.0)})
    post = lk.laplace(
        model, loader(x, y), likelihood=lk.Categorical(), subset=lk.subsets.all, prior=prior
    )

    assert "hyper_tuner    = MarginalLikelihood()" in repr(post)
    assert all(float(v) != 1.0 for v in post.prior.prior_precision.values())


def _vector_priors(dtype):
    return {
        "diagonal": (lk.subsets.last_layer, lk.priors.Diagonal(torch.ones(27, dtype=dtype))),
        "per_block": (
            lk.subsets.all,
            lk.priors.Layerwise(
                {"0": torch.ones(24, dtype=dtype), "2": torch.ones(27, dtype=dtype)}
            ),
        ),
    }


@pytest.mark.parametrize("kind", ["diagonal", "per_block"])
def test_vector_prior_tunes_a_float32_model(kind):
    """A vector precision on a float32 model tunes as in float64 and stays float32."""

    model, x, y = trained()

    def fit(dtype):
        params, prior = _vector_priors(dtype)[kind]
        data = loader(x.to(dtype), y)

        return lk.laplace(
            copy.deepcopy(model).to(dtype),
            data,
            likelihood=lk.Categorical(),
            subset=params,
            prior=prior,
        )

    p32, p64 = fit(torch.float32), fit(torch.float64)
    got, want = tree_leaves(p32.prior.prior_precision), tree_leaves(p64.prior.prior_precision)

    assert all(t.dtype == torch.float32 for t in got)
    assert not torch.allclose(want[0], torch.ones_like(want[0]))  # the tune moved

    # Some entries are weakly identified, so compare the float64 marglik at both optima.
    prec = p32.prior.prior_precision
    cast = {k: v.double() for k, v in prec.items()} if isinstance(prec, dict) else prec.double()
    at32 = copy.replace(p64, prior=copy.replace(p64.prior, prior_precision=cast))

    assert float(at32.log_marglik()) == pytest.approx(float(p64.log_marglik()), abs=1e-3)


def test_other_prior_types_are_not_tuned():
    model, x, y = trained()
    post = lk.laplace(
        model,
        loader(x, y),
        likelihood=lk.Categorical(),
        prior=_FixedPrior(lk.priors.Isotropic(torch.tensor(1.0))),
    )

    # Nothing to tune: the prior is not a TunablePrior and Categorical learns nothing.
    assert float(post.prior.prior_precision) == 1.0


class _FixedPrior:
    def __init__(self, inner):
        self.inner = inner
        self.prior_precision = inner.prior_precision

    def log_prob(self, params):
        return self.inner.log_prob(params)

    def location(self, like):
        return self.inner.location(like)

    def add_precision_to(self, op):
        return self.inner.add_precision_to(op)


def test_front_door_fit_runs_without_grad(monkeypatch):
    seen = []
    real = lk.front.fit_curvature

    def spying(*args, **kwargs):
        fit = real(*args, **kwargs)
        start = fit.start

        def start_():
            step = start()
            update = step.update

            def upd(stat, batch):
                seen.append(torch.is_grad_enabled())
                out = update(stat, batch)

                assert not any(t.requires_grad for t in tree_leaves(out))

                return out

            object.__setattr__(step, "update", upd)

            return step

        object.__setattr__(fit, "start", start_)

        return fit

    monkeypatch.setattr(lk.front, "fit_curvature", spying)
    model, x, y = trained()
    lk.laplace(model, loader(x, y), likelihood=lk.Categorical(), hyper_tuner=None)

    assert seen and not any(seen)


def test_scale_is_float():
    model, x, y = trained()
    sized = lk.laplace(
        model,
        lk.data.from_loader(loader(x, y), n_data=1000),
        likelihood=lk.Categorical(),
        hyper_tuner=None,
    )
    counted = lk.laplace(model, loader(x, y), likelihood=lk.Categorical(), hyper_tuner=None)

    assert sized.n_data.dtype == counted.n_data.dtype == torch.float32


def test_predictions_are_detached_from_the_model():
    model, x, y = trained()

    model.zero_grad(set_to_none=True)
    post = lk.laplace(model, loader(x, y), likelihood=lk.Categorical())

    assert lk.predict(post, x).probs.requires_grad is False
    assert lk.predict_outputs(post, x).mean.requires_grad is False
    assert all(p.grad is None for p in model.parameters())


def test_tuning_through_the_front_door_reaches_the_leaf():
    model, x, y = trained()
    batches = [lk.Batch(xb, yb) for xb, yb in loader(x, y)]
    post = lk.laplace(
        model,
        loader(x, y),
        likelihood=lk.Categorical(),
        hyper_tuner=lk.Minimize(lk.objectives.predictive_nll(batches)),
    )

    assert float(post.prior.prior_precision) != 1.0


def test_repr_is_multiline_and_aligned():
    model, x, y = trained()
    post = lk.laplace(model, loader(x, y), likelihood=lk.Categorical())
    lines = repr(post).splitlines()

    assert lines[0] == "LaplacePosterior("
    assert lines[-1] == ")"
    assert lines[1].startswith('  subset         = last_layer {"2.weight": (3, 8), "2.bias": (3,)}')
    assert lines[2] == "  hessian_approx = PerBlockCurvature({'*': DenseGGN()})"
    assert lines[3].startswith("  curvature      = ")
    assert lines[4].startswith("  prior          = Isotropic(prior_precision=")
    assert lines[5] == "  likelihood     = Categorical()"
    assert lines[6] == "  hyper_tuner    = MarginalLikelihood()"
    assert lines[7].startswith("  n_data         = ")


def test_params_row_summarizes_many_tensors():
    torch.manual_seed(0)
    layers = [m for _ in range(5) for m in (nn.Linear(2, 2), nn.Tanh())]
    model = nn.Sequential(*layers[:-1]).double()
    x = torch.randn(8, 2, dtype=torch.float64)
    y = torch.randint(0, 2, (8,))
    post = lk.laplace(
        model, [(x, y)], likelihood=lk.Categorical(), subset=lk.subsets.all, hyper_tuner=None
    )
    row = next(line for line in str(post).splitlines() if line.startswith("  subset"))

    assert row == (
        '  subset         = all {"0.weight": (2, 2), "0.bias": (2,), "2.weight": (2, 2), ...} '
        "(10 tensors, 30 parameters)"
    )
    assert len(post.mean) == 10


# --- per-block curvature through the front door -------------------------------------


def wide():
    torch.manual_seed(0)
    x, y = blobs()
    model = nn.Sequential(
        nn.Linear(2, 80), nn.Tanh(), nn.Linear(80, 60), nn.Tanh(), nn.Linear(60, 3)
    )

    return model, x, y


def test_print_shows_the_per_block_map():
    model, x, y = wide()
    post = lk.laplace(
        model, loader(x, y), likelihood=lk.Categorical(), subset=lk.subsets.all, hyper_tuner=None
    )
    # Blocks with the same method share a rule, so each method runs once per batch.
    want = "PerBlockCurvature({('0', '4'): DenseGGN(), ('2',): KFAC()})"

    assert repr(post).splitlines()[2] == f"  hessian_approx = {want}"
    assert list(post.curvature.blocks) == ["0", "2", "4"]
    assert isinstance(post.curvature.blocks["2"], lk.core.KronEigenbasis)


@lk.core.pytree
class _CountingDiag:
    """Diag, counting its updates."""

    calls: list = lk.core.static()

    @property
    def requires(self):
        return lk.DiagGGN().requires

    def init(self, params, ctx):
        return lk.DiagGGN().init(params, ctx)

    def update(self, stat, params, batch, generator, ctx):
        self.calls.append(batch.size)
        return lk.DiagGGN().update(stat, params, batch, generator, ctx)

    def finalize(self, stat):
        return lk.DiagGGN().finalize(stat)


@lk.core.pytree
class _TwoPassDiag:
    """A staged test double: Diag twice, summed. So the result is twice Diag."""

    @property
    def requires(self):
        return lk.DiagGGN().requires

    def plan(self, params, ctx):
        def second(first):
            def add(again):
                return lk.core.Done(
                    lk.core.BlockDiag(
                        {
                            b: lk.core.Diag(op.diag + again.blocks[b].diag)
                            for b, op in first.blocks.items()
                        }
                    )
                )

            return lk.core.chain(lk.DiagGGN(), then=add)

        return lk.core.chain(lk.DiagGGN(), then=second)


def _alone(model, data, approx, block):
    """`approx` fitted on one block's params alone, through the front door."""

    params = {k: v for k, v in model.named_parameters() if k.startswith(f"{block}.")}
    post = lk.laplace(
        model,
        data,
        likelihood=lk.Categorical(),
        subset=params,
        hessian_approx=approx,
        hyper_tuner=None,
    )

    return post.curvature.blocks[block]


def test_staged_per_block_runs_through_laplace():
    model, x, y = trained(torch.float64)
    data = loader(x, y)
    calls = []
    approx = lk.per_block({"0": _TwoPassDiag(), "*": _CountingDiag(calls)})

    assert isinstance(approx, lk.core.StagedPerBlock)
    post = lk.laplace(
        model,
        data,
        likelihood=lk.Categorical(),
        subset=lk.subsets.all,
        hessian_approx=approx,
        hyper_tuner=None,
    )

    assert calls == [30, 30, 30]  # the single-pass member sits out the second pass
    d0 = _alone(model, data, lk.DiagGGN(), "0").diag
    d2 = _alone(model, data, lk.DiagGGN(), "2").diag

    torch.testing.assert_close(post.curvature.blocks["0"].diag, 2 * d0)
    torch.testing.assert_close(post.curvature.blocks["2"].diag, d2)
    assert "StagedPerBlock({'0': _TwoPassDiag(), '*': _CountingDiag(" in repr(post)


def test_unmatched_block_raises_before_data():
    model, x, y = trained()
    ld = _Counting(loader(x, y))

    with pytest.raises(lk.StructureError, match="'0'"):
        lk.laplace(
            model,
            ld,
            likelihood=lk.Categorical(),
            subset=lk.subsets.all,
            hessian_approx=lk.per_block({"2": lk.DiagGGN()}),
        )

    assert ld.pulled == 1


def test_default_with_weighted_batches_avoids_kfac():
    model, x, y = wide()
    w = torch.linspace(0.5, 1.5, x.shape[0])
    data = [lk.Batch(x[i : i + 30], y[i : i + 30], w[i : i + 30]) for i in range(0, 90, 30)]
    post = lk.laplace(
        model, data, likelihood=lk.Categorical(), subset=lk.subsets.all, hyper_tuner=None
    )

    assert "('2',): DiagGGN()" in repr(post)
    assert lk.predict(post, x).probs.shape == (90, 3)


# --- the fixed-parameter guard ---------------------------------------------------------


@pytest.mark.parametrize(
    "use",
    [
        lambda post, x: lk.predict(post, x),
        lambda post, x: lk.predict(post, x, lk.WeightSamples(n_samples=2)),
        lambda post, x: lk.predict_outputs(post, x),
        lambda post, x: lk.core.mean_outputs(post, x),
    ],
    ids=["predict", "WeightSamples", "predict_outputs", "mean_outputs"],
)
def test_editing_a_fixed_parameter_raises(use):
    model, x, y = trained()
    post = lk.laplace(model, loader(x, y), likelihood=lk.Categorical(), hyper_tuner=None)

    assert set(post.versions) == {"0.weight", "0.bias", "2.weight", "2.bias"}
    use(post, x[:2])  # fine before the edit

    with torch.no_grad():
        model[0].weight.add_(1.0)

    with pytest.raises(RuntimeError, match=r"'0\.weight' changed"):
        use(post, x[:2])


def test_by_hand_posterior_records_versions():
    model, x, y = trained()
    fn = lk.core.ModelFn.from_module(model)
    params = {k: v.detach() for k, v in model.named_parameters() if k.startswith("2.")}
    caps = lk.core.probe(fn, lk.Categorical(), lk.Batch(x, y))
    fit = lk.core.fit_curvature(fn, lk.Categorical(), params, lk.DenseGGN(), capabilities=caps)
    stat = fit.update(fit.init(), lk.Batch(x, y))
    post = fit.posterior(
        fit.finalize(stat), prior=lk.priors.Isotropic(torch.tensor(1.0)), n_data=stat.count
    )

    assert set(post.versions) == {"0.weight", "0.bias", "2.weight", "2.bias"}

    with torch.no_grad():
        model[0].bias.zero_()

    with pytest.raises(RuntimeError, match=r"'0\.bias' changed"):
        lk.predict(post, x[:2])


@lk.core.pytree
class _NeedsTargets:
    """Declares that it needs targets; counts updates."""

    calls: list = lk.core.static(default_factory=list)

    @property
    def requires(self):
        return lk.core.Requirements(needs_targets=True)

    def init(self, params, ctx, /):
        return lk.curvature.DiagGGN().init(params, ctx)

    def update(self, stat, params, batch, generator, ctx, /):
        self.calls.append(1)
        return stat

    def finalize(self, stat, /):
        return lk.curvature.DiagGGN().finalize(stat)


def test_batch_requirements_checked_before_data():
    model, x, _ = trained()
    method = _NeedsTargets()
    ld = _Counting([lk.Batch(x[:30]), lk.Batch(x[30:])])

    with pytest.raises(lk.CapabilityError, match="needs targets"):
        lk.laplace(model, ld, likelihood=lk.Categorical(), hessian_approx=method)

    assert ld.pulled == 1 and method.calls == []


def test_print_shows_the_options_of_a_bare_method():
    model, x, y = trained()
    post = lk.laplace(
        model,
        loader(x, y),
        likelihood=lk.Categorical(),
        hessian_approx=lk.KFAC(fisher="mc"),
        hyper_tuner=None,
    )

    assert "hessian_approx = KFAC(fisher='mc')" in repr(post)


def test_log_marglik_on_the_posterior_matches_the_objective():
    model, x, y = trained()
    data = loader(x, y)
    post = lk.laplace(model, data, likelihood=lk.Categorical())
    fn = lk.core.ModelFn.from_module(model)
    caps = lk.core.probe(fn, lk.Categorical(), lk.Batch.from_pair(data[0]))
    fit = lk.core.fit_curvature(
        fn, lk.Categorical(), dict(post.mean), lk.DenseGGN(), capabilities=caps
    )
    stat = fit.init()

    for pair in data:
        stat = fit.update(stat, lk.Batch.from_pair(pair))

    want = -lk.objectives.neg_log_marglik(dataclasses.replace(post, data_fit=(stat.data_fit,)))

    torch.testing.assert_close(post.log_marglik(), want)


def test_log_marglik_needs_targets():
    model, x, _ = trained()
    post = lk.laplace(
        model, [lk.Batch(x[:45]), lk.Batch(x[45:])], likelihood=lk.Categorical(), hyper_tuner=None
    )

    with pytest.raises(ValueError, match="targets"):
        post.log_marglik()


class _Source:
    """A Data source written by hand: a callable with a size."""

    def __init__(self, batches):
        self.batches = batches
        self.n_data = 1000

    def __call__(self, generator, /):
        return iter(self.batches)


def test_a_data_source_is_used_as_is():
    model, x, y = trained()
    post = lk.laplace(
        model, _Source([lk.Batch(x, y)]), likelihood=lk.Categorical(), hyper_tuner=None
    )

    assert float(post.n_data) == 1000.0
    assert isinstance(_Source([]), lk.data.Data)


def test_editing_the_front_door_mean_raises():
    model, x, y = trained()
    post = lk.laplace(model, loader(x, y), likelihood=lk.Categorical(), hyper_tuner=None)
    lk.predict(post, x[:2])

    with torch.no_grad():
        model[2].weight.mul_(2.0)  # the mean is a copy: the posterior does not move

    lk.predict(post, x[:2])

    with torch.no_grad():
        post.mean["2.weight"].mul_(2.0)  # the copy itself

    with pytest.raises(RuntimeError, match=r"'2\.weight' changed.*mean"):
        lk.predict(post, x[:2])


def test_a_copied_mean_does_not_follow_the_module():
    model, x, y = trained()
    fn = lk.core.ModelFn.from_module(model)
    caps = lk.core.probe(fn, lk.Categorical(), lk.Batch(x, y))
    params = {k: v.detach().clone() for k, v in model.named_parameters() if k.startswith("2.")}
    fit = lk.core.fit_curvature(fn, lk.Categorical(), params, lk.DenseGGN(), capabilities=caps)
    stat = fit.update(fit.init(), lk.Batch(x, y))
    prior = lk.priors.Isotropic(torch.tensor(1.0))
    post = fit.posterior(fit.finalize(stat), prior=prior, n_data=stat.count)
    before = lk.predict(post, x[:2]).probs

    with torch.no_grad():
        model[2].weight.mul_(2.0)  # not used: the forward reads the copied mean

    torch.testing.assert_close(lk.predict(post, x[:2]).probs, before)

    with torch.no_grad():
        post.mean["2.weight"].add_(1.0)  # the mean itself

    with pytest.raises(RuntimeError, match=r"posterior parameter '2\.weight'"):
        lk.predict(post, x[:2])


def test_laplace_bare_linear():
    x, y = blobs()
    post = lk.laplace(nn.Linear(2, 3), loader(x, y), likelihood=lk.Categorical())

    assert dict(post.layout.blocks) == {"<root>": ("weight", "bias")}
    assert torch.isfinite(post.log_marglik())


def test_laplace_probes_subset(monkeypatch):
    model, x, y = trained()
    calls = []
    real = lk.core.probe

    def spy(model, lik, batch, /, params=None):
        calls.append(None if params is None else dict(params))

        return real(model, lik, batch, params=params)

    monkeypatch.setattr(lk.core, "probe", spy)
    post = lk.laplace(model, loader(x, y), likelihood=lk.Categorical())

    assert len(calls) == 2
    assert calls[0] == {}
    assert set(calls[1]) == {"2.weight", "2.bias"}
    assert post.caps.probed == frozenset({"2.weight", "2.bias"})


class _CountedLoader:
    """A loader that counts the batches it hands out."""

    def __init__(self, batches):
        self.batches, self.served = batches, 0

    def __iter__(self):
        for b in self.batches:
            self.served += 1
            yield b


class _LogProbOnly:
    """Squared error to one-hot targets, with no distribution."""

    def log_prob(self, f, y, /):
        return -((f - y) ** 2).sum(-1)


def _one_hot_loader(x, y):
    return loader(x, nn.functional.one_hot(y, 3).to(x.dtype))


def test_front_door_checks_the_likelihood_before_the_fit():
    model, x, y = trained()
    data = _CountedLoader(_one_hot_loader(x, y))

    with pytest.raises(lk.CapabilityError, match="HasSample or HasDistribution"):
        lk.laplace(
            model,
            data,
            likelihood=_LogProbOnly(),
            hessian_approx=lk.DiagGGN(fisher="mc"),
            hyper_tuner=None,
        )

    assert data.served == 0  # an explicit method is checked before the probe reads a batch


def test_front_door_default_fits_a_log_prob_only_likelihood_with_the_empirical_fisher():
    model, x, y = trained()
    post = lk.laplace(model, _one_hot_loader(x, y), likelihood=_LogProbOnly(), hyper_tuner=None)

    assert "DiagGGN(fisher='empirical')" in repr(post)


def test_front_door_fits_a_log_prob_only_likelihood_with_the_observed_hessian():
    model, x, y = trained()
    post = lk.laplace(
        model,
        _one_hot_loader(x, y),
        likelihood=_LogProbOnly(),
        hessian_approx=lk.DenseGGN(observed=True),
        hyper_tuner=None,
    )

    assert "DenseGGN(observed=True)" in repr(post)


def test_posterior_prior_over_other_params_raises_before_the_pass():
    model, x, y = trained()
    lik = lk.Categorical()
    post = lk.laplace(model, loader(x, y), likelihood=lik, subset=lk.subsets.all, hyper_tuner=None)

    with pytest.raises(lk.StructureError, match="prior mean has keys"):
        post.location({"2.weight": model[2].weight})

    with pytest.raises(lk.StructureError, match="prior mean shapes differ"):
        post.location({**post.mean, "2.bias": torch.zeros(5)})

    with pytest.raises(lk.StructureError, match="float64"):
        post.location({k: v.double() for k, v in post.mean.items()})

    ld = _Counting(loader(x, y))

    with pytest.raises(lk.StructureError, match="prior mean has keys"):
        lk.laplace(model, ld, likelihood=lik, prior=post)

    assert ld.pulled == 1  # the probe's batch only


def _regression(dtype: torch.dtype = torch.float32):
    """A sine with noise 0.3 and an MLP trained to it: its train RMSE is the noise."""

    g = torch.Generator().manual_seed(1)  # a draw whose noise std is 0.300
    x = torch.rand(400, 1, generator=g, dtype=dtype) * 6.0
    y = torch.sin(x) + 0.3 * torch.randn(400, 1, generator=g, dtype=dtype)

    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(1, 32), nn.Tanh(), nn.Linear(32, 1)).to(dtype)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)

    for _ in range(2000):
        opt.zero_grad()
        nn.functional.mse_loss(model(x), y).backward()
        opt.step()

    return model.eval(), x, y


def test_default_gaussian_sigma_is_tuned():
    """`Gaussian()` carries a tensor sigma, and the default tune fits it with the prior."""

    model, x, y = _regression()

    with torch.no_grad():
        rmse = float((model(x) - y).pow(2).mean().sqrt())

    assert rmse == pytest.approx(0.3, rel=0.05)  # the fit is good, so sigma can find the noise

    post = lk.laplace(model, loader(x, y, k=4), likelihood=lk.Gaussian())

    assert "hyper_tuner    = MarginalLikelihood()\n" in str(post)
    assert post.likelihood.learned
    assert float(post.likelihood.sigma) == pytest.approx(0.3, rel=0.1)
    assert float(post.prior.prior_precision) != 1.0


def test_default_sigma_on_a_float64_model():
    """The default sigma and prior are float32, and a float64 model tunes them in float64."""

    model, x, y = _regression(torch.float64)
    post = lk.laplace(model, loader(x, y, k=4), likelihood=lk.Gaussian())

    assert post.likelihood.sigma.shape == () and post.likelihood.sigma.dtype == torch.float64
    assert (
        post.prior.prior_precision.shape == () and post.prior.prior_precision.dtype == torch.float64
    )
    assert float(post.likelihood.sigma) == pytest.approx(0.3, rel=0.1)

    pred = lk.predict(post, x)

    assert pred.mean.dtype == torch.float64 and pred.variance.dtype == torch.float64


# --- TunablePrior: a plugin prior that names its tunable leaves gets the default tune ------


@lk.core.pytree
class _PluginIsotropic:
    """N(0, prec^-1 I) from the public API only, without `tunable`."""

    prior_precision: torch.Tensor

    def log_prob(self, params):
        terms = [
            (
                0.5 * (self.prior_precision.log() - math.log(2 * math.pi))
                - 0.5 * self.prior_precision * p**2
            ).sum()
            for p in params.values()
        ]

        return torch.stack(terms).sum()

    def location(self, like):
        return {k: torch.zeros_like(v) for k, v in like.items()}

    def add_precision_to(self, op):
        return op.add_isotropic(self.prior_precision)


@lk.core.pytree
class _TunablePluginIsotropic(_PluginIsotropic):
    def tunable(self):
        return self.prior_precision


def test_builtin_priors_are_tunable_priors():
    priors = [
        lk.priors.Isotropic(torch.tensor(1.0)),
        lk.priors.Diagonal(torch.ones(3)),
        lk.priors.Layerwise({"0": torch.tensor(1.0)}),
    ]

    assert all(isinstance(p, lk.core.TunablePrior) for p in priors)
    assert isinstance(_TunablePluginIsotropic(torch.tensor(1.0)), lk.core.TunablePrior)
    assert not isinstance(_PluginIsotropic(torch.tensor(1.0)), lk.core.TunablePrior)
    assert not isinstance(_FixedPrior(lk.priors.Isotropic(torch.tensor(1.0))), lk.core.TunablePrior)


def test_plugin_tunable_prior_gets_the_default_tune():
    """It tunes to the same precision as the built-in `Isotropic`, its independent twin."""

    model, x, y = trained()

    def fit(prior):
        return lk.laplace(model, loader(x, y), likelihood=lk.Categorical(), prior=prior)

    post = fit(_TunablePluginIsotropic(torch.tensor(1.0)))
    want = fit(lk.priors.Isotropic(torch.tensor(1.0)))

    assert "hyper_tuner    = MarginalLikelihood()\n" in repr(post)
    assert float(post.prior.prior_precision) != 1.0
    assert float(post.prior.prior_precision) == pytest.approx(
        float(want.prior.prior_precision), rel=1e-4
    )


def test_plugin_prior_without_tunable_is_not_tuned():
    model, x, y = trained()
    post = lk.laplace(
        model, loader(x, y), likelihood=lk.Categorical(), prior=_PluginIsotropic(torch.tensor(1.0))
    )

    assert float(post.prior.prior_precision) == 1.0


def test_per_block_prior_tunes_jointly_with_gaussian_sigma():
    """The joint tune takes each block's precision and sigma; a mapping of precisions used
    to be passed whole, which `where` refused."""

    model, x, y = _regression()
    prior = lk.priors.Layerwise({"0": torch.tensor(1.0), "2": torch.tensor(1.0)})
    post = lk.laplace(
        model, loader(x, y, k=4), likelihood=lk.Gaussian(), subset=lk.subsets.all, prior=prior
    )

    assert "hyper_tuner    = MarginalLikelihood()\n" in str(post)
    assert post.likelihood.learned
    assert all(float(v) != 1.0 for v in post.prior.prior_precision.values())
    assert float(post.likelihood.sigma) == pytest.approx(0.3, rel=0.1)


def test_explicit_tune_of_a_prior_without_tunable_raises_before_the_fit(monkeypatch):
    model, x, y = trained()
    monkeypatch.setattr(lk.front, "fit_curvature", None)  # the fit must not start

    with pytest.raises(lk.StructureError, match="_PluginIsotropic is not a TunablePrior"):
        lk.laplace(
            model,
            loader(x, y),
            likelihood=lk.Categorical(),
            prior=_PluginIsotropic(torch.tensor(1.0)),
            hyper_tuner=lk.Minimize(lk.objectives.predictive_nll([lk.Batch(x, y)])),
        )


# --- the TunablePrior contract is checked before any data is read --------------------------


class _Unreadable:
    """Data that fails the test when a batch past the first (the probe's) is read."""

    def __iter__(self):
        x, y = blobs()
        yield x[:30], y[:30]
        raise AssertionError("the data was read")


@lk.core.pytree
class _CopiesTunable(_PluginIsotropic):
    def tunable(self):
        return self.prior_precision.clone()


@lk.core.pytree
class _FloatTunable(_PluginIsotropic):
    def tunable(self):
        return float(self.prior_precision)


@pytest.mark.parametrize(
    ("prior", "match"),
    [
        (_CopiesTunable(torch.tensor(1.0)), r"_CopiesTunable\.tunable\(\) .* not a leaf"),
        (_FloatTunable(torch.tensor(1.0)), r"_FloatTunable\.tunable\(\) .* got float"),
        (_TunablePluginIsotropic(torch.tensor(-1.0)), r"_TunablePluginIsotropic.* positive"),
    ],
    ids=["copy", "float", "negative"],
)
def test_broken_tunable_prior_raises_before_the_data(prior, match):
    model, _, _ = trained()

    with pytest.raises(lk.StructureError, match=match):
        lk.laplace(model, _Unreadable(), likelihood=lk.Categorical(), prior=prior)


def test_broken_tunable_prior_with_an_explicit_tune_raises_before_the_data():
    model, x, y = trained()

    with pytest.raises(lk.StructureError, match=r"_CopiesTunable\.tunable\(\)"):
        lk.laplace(
            model,
            _Unreadable(),
            likelihood=lk.Categorical(),
            prior=_CopiesTunable(torch.tensor(1.0)),
            hyper_tuner=lk.Minimize(lk.objectives.predictive_nll([lk.Batch(x, y)])),
        )


def test_non_tunable_prior_with_an_explicit_tune_raises_before_the_data():
    model, x, y = trained()

    with pytest.raises(lk.StructureError, match="_PluginIsotropic is not a TunablePrior"):
        lk.laplace(
            model,
            _Unreadable(),
            likelihood=lk.Categorical(),
            prior=_PluginIsotropic(torch.tensor(1.0)),
            hyper_tuner=lk.Minimize(lk.objectives.predictive_nll([lk.Batch(x, y)])),
        )


def test_broken_tunable_prior_is_not_checked_without_a_tune():
    """`hyper_tuner=None` never calls `tunable`, so the fit goes ahead."""

    model, x, y = trained()
    post = lk.laplace(
        model,
        loader(x, y),
        likelihood=lk.Categorical(),
        prior=_CopiesTunable(torch.tensor(1.0)),
        hyper_tuner=None,
    )

    assert "hyper_tuner    = None" in repr(post)


class _Counted:
    """A loader that counts the batches it yields."""

    def __init__(self, items):
        self.items = items
        self.pulled = 0

    def __iter__(self):
        for it in self.items:
            self.pulled += 1
            yield it


@lk.core.pytree
class _PlainKFAC:
    """KFAC as a plugin would wrap it, without a template."""

    @property
    def requires(self):
        return lk.KFAC().requires

    def init(self, params, ctx):
        return lk.KFAC().init(params, ctx)

    def update(self, stat, params, batch, generator, ctx):
        return lk.KFAC().update(stat, params, batch, generator, ctx)

    def finalize(self, stat):
        return lk.KFAC().finalize(stat)


@pytest.mark.parametrize(
    "approx",
    [_PlainKFAC(), lk.core.PerBlockCurvature({"*": lk.KFAC()})],
    ids=["plugin", "direct-PerBlockCurvature"],
)
def test_diagonal_prior_without_template_refused_after_the_fit(monkeypatch, approx):
    """Without a template the structure exists only after `finalize`: the refusal follows
    the fit's batches and comes before the tune, which is never entered."""

    assert not isinstance(approx, lk.core.HasTemplate)

    model, x, y = trained()
    data = _Counted(loader(x, y))
    calls = []

    def objective(post):
        calls.append(post)
        return post.log_marglik()

    with pytest.raises(lk.StructureError, match="no Kronecker form for a diagonal prior") as e:
        lk.laplace(
            model,
            data,
            likelihood=lk.Categorical(),
            hessian_approx=approx,
            prior=lk.priors.Diagonal(torch.ones(27)),
            hyper_tuner=lk.Minimize(objective),
        )

    assert not str(e.value).startswith("before the data pass")
    assert data.pulled == len(data.items) + 1  # the probe's batch, then the fit's batches
    assert calls == []


def test_kfac_diagonal_prior_before_data():
    model, x, y = trained()
    data = _Counted(loader(x, y))

    with pytest.raises(
        lk.StructureError,
        match=r"^before the data pass: .*no Kronecker form for a diagonal prior",
    ):
        lk.laplace(
            model,
            data,
            likelihood=lk.Categorical(),
            hessian_approx=lk.KFAC(),
            prior=lk.priors.Diagonal(torch.ones(27)),
            hyper_tuner=None,
        )

    assert data.pulled == 1  # the probe's batch only


def test_per_block_kfac_diagonal_prior_before_data():
    model, x, y = trained()
    data = _Counted(loader(x, y))

    with pytest.raises(
        lk.StructureError,
        match=r"^before the data pass: .*no Kronecker form for a diagonal prior",
    ):
        lk.laplace(
            model,
            data,
            likelihood=lk.Categorical(),
            hessian_approx=lk.per_block({"*": lk.KFAC()}),
            prior=lk.priors.Diagonal(torch.ones(27)),
            hyper_tuner=None,
        )

    assert data.pulled == 1  # the probe's batch only


@lk.core.pytree
class _SeesLikelihood:
    """`Diag` with a template; records the likelihood its template and its `init` see."""

    seen: list = lk.core.static()

    @property
    def requires(self):
        return lk.DiagGGN().requires

    def template(self, params, ctx):
        self.seen.append(("template", ctx.likelihood))
        return lk.DiagGGN().template(params, ctx)

    def init(self, params, ctx):
        self.seen.append(("init", ctx.likelihood))
        return lk.DiagGGN().init(params, ctx)

    def update(self, stat, params, batch, generator, ctx):
        return lk.DiagGGN().update(stat, params, batch, generator, ctx)

    def finalize(self, stat):
        return lk.DiagGGN().finalize(stat)


def test_prior_check_sees_the_unit_likelihood():
    """The template check gets the likelihood the fit uses: the unit one, sigma fixed at 1."""

    torch.manual_seed(0)
    model = nn.Linear(2, 1)
    x = torch.randn(12, 2)
    y = torch.randn(12, 1)
    approx = _SeesLikelihood([])

    lk.laplace(
        model,
        loader(x, y),
        likelihood=copy.replace(lk.Gaussian(), sigma=torch.as_tensor(torch.tensor(0.5))),
        hessian_approx=approx,
        hyper_tuner=None,
    )

    assert [what for what, _ in approx.seen] == ["template", "init"]

    for _, lik in approx.seen:
        assert isinstance(lik, lk.Gaussian) and not lik.learned and float(lik.sigma) == 1.0


def test_kfac_posterior_prior_before_data():
    model, x, y = trained()
    lik = lk.Categorical()
    post = lk.laplace(
        model, loader(x, y), likelihood=lik, hessian_approx=lk.KFAC(), hyper_tuner=None
    )
    data = _Counted(loader(x, y))

    with pytest.raises(lk.StructureError, match=r"^before the data pass: .*KronSum"):
        lk.laplace(model, data, likelihood=lik, hessian_approx=lk.KFAC(), prior=post)

    assert data.pulled == 1


@pytest.mark.parametrize(
    "approx",
    [
        lk.per_block({"*": lk.LowRankGGN(2, oversample=2)}),
        lk.core.PerBlockCurvature({"*": lk.LowRankGGN(2, oversample=2)}),
        lk.per_block({"0": _TwoPassDiag(), "*": lk.LowRankGGN(2, oversample=2)}),
    ],
    ids=["templated", "direct", "staged"],
)
def test_per_block_joint_member_before_data(approx):
    """A joint member is refused before the data pass: by the prior check when the
    `per_block` has a template, else when its first pass starts."""

    model, x, y = trained()
    data = _Counted(loader(x, y))

    with pytest.raises(lk.StructureError, match="per_block member .* joint operator"):
        lk.laplace(
            model,
            data,
            likelihood=lk.Categorical(),
            subset=lk.subsets.all,
            hessian_approx=approx,
            hyper_tuner=None,
        )

    assert data.pulled == 1


def _kfac_post(model, x, y):
    return lk.laplace(
        model, loader(x, y), likelihood=lk.Categorical(), hessian_approx=lk.KFAC(), hyper_tuner=None
    )


def _same_structure_post(model, x, y, approx):
    return lk.laplace(
        model, loader(x, y), likelihood=lk.Categorical(), hessian_approx=approx, hyper_tuner=None
    )


_VALID = {
    "Diag+Diagonal": (
        lambda m, x, y: lk.DiagGGN(),
        lambda m, x, y: lk.priors.Diagonal(torch.ones(27)),
    ),
    "KFAC+Isotropic": (
        lambda m, x, y: lk.KFAC(),
        lambda m, x, y: lk.priors.Isotropic(torch.tensor(2.0)),
    ),
    "KFAC+PerBlockCurvature": (
        lambda m, x, y: lk.KFAC(),
        lambda m, x, y: lk.priors.Layerwise({"2": torch.tensor(3.0)}),
    ),
    "KFAC+KronSum": (
        lambda m, x, y: lk.KFAC(),
        lambda m, x, y: lk.priors.KronSum.of(_kfac_post(m, x, y)),
    ),
    "per_block-KFAC+Isotropic": (
        lambda m, x, y: lk.per_block({"*": lk.KFAC()}),
        lambda m, x, y: lk.priors.Isotropic(torch.tensor(2.0)),
    ),
    "LowRankGGN+Isotropic": (
        lambda m, x, y: lk.LowRankGGN(4, oversample=2),
        lambda m, x, y: lk.priors.Isotropic(torch.tensor(2.0)),
    ),
    "LowRankGGN+Diagonal": (
        lambda m, x, y: lk.LowRankGGN(4, oversample=2),
        lambda m, x, y: lk.priors.Diagonal(torch.linspace(1.0, 2.0, 27)),
    ),
    "LowRankGGN+LowRankSum": (
        lambda m, x, y: lk.LowRankGGN(4, oversample=2),
        lambda m, x, y: lk.priors.LowRankSum.of(
            _same_structure_post(m, x, y, lk.LowRankGGN(4, oversample=2)), rank=3
        ),
    ),
    "LowRankGGN+PerBlockCurvature": (
        lambda m, x, y: lk.LowRankGGN(4, oversample=2),
        lambda m, x, y: lk.priors.Layerwise({"2": torch.tensor(3.0)}),
    ),
    "Diag+LaplacePosterior": (
        lambda m, x, y: lk.DiagGGN(),
        lambda m, x, y: _same_structure_post(m, x, y, lk.DiagGGN()),
    ),
    "DenseGGN+LaplacePosterior": (
        lambda m, x, y: lk.DenseGGN(),
        lambda m, x, y: _same_structure_post(m, x, y, lk.DenseGGN()),
    ),
}


@pytest.mark.parametrize("case", _VALID.values(), ids=list(_VALID))
def test_valid_combinations_unchanged(monkeypatch, case):
    """Valid priors pass the check: every batch is read and the posterior equals a fit
    without the check."""

    make_approx, make_prior = case
    model, x, y = trained()
    prior = make_prior(model, x, y)

    def fit():
        data = _Counted(loader(x, y))
        post = lk.laplace(
            model,
            data,
            likelihood=lk.Categorical(),
            hessian_approx=make_approx(model, x, y),
            prior=prior,
            hyper_tuner=None,
            generator=torch.Generator().manual_seed(0),  # the same sketch both times
        )

        return post, data.pulled

    checked, pulled = fit()

    assert pulled == 1 + len(loader(x, y))

    monkeypatch.setattr(lk.front, "_check_prior_fits", lambda *a: None)
    unchecked, _ = fit()

    a = tree_leaves(checked.precision())
    b = tree_leaves(unchecked.precision())

    assert len(a) == len(b)

    for u, v in zip(a, b, strict=True):
        assert torch.equal(u, v)

    for k in checked.mean:
        assert torch.equal(checked.mean[k], unchecked.mean[k])


# --- index-set subnetworks (`Selection`) ---------------------------------------------------


def _last_layer_masks(model):
    """Some entries of the last layer: one block, "2"."""

    g = torch.Generator().manual_seed(3)
    w = torch.rand(model[2].weight.shape, generator=g) < 0.3
    b = torch.tensor([True, False, True])
    w[0, 0] = True

    return {"2.weight": w, "2.bias": b}


def _picks_last_entries(model, caps):
    module = model.module
    assert module is not None

    return lk.core.Selection.from_masks(dict(module.named_parameters()), _last_layer_masks(module))


def test_selection_kfac_refused_before_fit_batches():
    model, x, y = trained()
    data = _Counted(loader(x, y))

    with pytest.raises(lk.CapabilityError, match=r"KFAC.*blocks \['2'\] hold index selections"):
        lk.laplace(
            model,
            data,
            likelihood=lk.Categorical(),
            subset=_picks_last_entries,
            hessian_approx=lk.KFAC(),
            hyper_tuner=None,
        )

    assert data.pulled == 1  # the probe's batch only

    # By hand: the probe of the selected parameters refuses hook methods, without blocks.
    fn = lk.core.ModelFn.from_module(model)
    sel = _picks_last_entries(fn, None)
    sub_fn, sub_params = lk.core.select(fn, sel)
    caps = lk.core.probe(sub_fn, lk.Categorical(), lk.Batch(x[:4], y[:4]), params=sub_params)

    with pytest.raises(lk.CapabilityError, match="index selections"):
        caps.check(lk.KFAC().requires, "KFAC")

    fit = lk.core.fit_curvature(sub_fn, lk.Categorical(), sub_params, lk.KFAC(), capabilities=caps)

    with pytest.raises(lk.CapabilityError, match="index selections"):
        fit.start()


def _dense_subnetwork_precision(model, x, sel, prior_prec):
    """J^T H J + prior_prec I over the selected entries, from `jacrev` on the full tensors."""

    params = {k: v.detach() for k, v in model.named_parameters()}
    names = ["2.weight", "2.bias"]

    def f(w, b):
        return torch.func.functional_call(model, {**params, "2.weight": w, "2.bias": b}, (x,))

    jw, jb = torch.func.jacrev(f, argnums=(0, 1))(params["2.weight"], params["2.bias"])
    n, k = x.shape[0], jw.shape[1]
    flat = {"2.weight": jw.reshape(n, k, -1), "2.bias": jb.reshape(n, k, -1)}
    j = torch.cat([flat[m][:, :, sel.index[m]] for m in names], dim=-1)  # [N, K, P_sel]

    p = torch.softmax(f(params["2.weight"], params["2.bias"]), dim=-1)
    h = torch.diag_embed(p) - p.unsqueeze(-1) * p.unsqueeze(-2)
    ggn = torch.einsum("nkp,nkl,nlq->pq", j, h, j)

    return ggn + prior_prec * torch.eye(ggn.shape[0], dtype=ggn.dtype)


def test_laplace_on_selection_matches_dense_subnetwork():
    model, x, y = trained(torch.float64)
    prior = lk.priors.Isotropic(torch.tensor(0.7, dtype=torch.float64))
    post = lk.laplace(
        model,
        loader(x, y),
        likelihood=lk.Categorical(),
        subset=_picks_last_entries,
        hessian_approx=lk.DenseGGN(),
        prior=prior,
        hyper_tuner=None,
    )
    sel = _picks_last_entries(lk.core.ModelFn.from_module(model), None)

    assert list(post.mean) == ["2.weight[idx]", "2.bias[idx]"]
    assert '"2.weight[idx]"' in repr(post) and "_picks_last_entries" in repr(post)

    got = post.precision().blocks["2"].mat
    want = _dense_subnetwork_precision(model, x, sel, 0.7)

    torch.testing.assert_close(got, want, rtol=1e-8, atol=1e-10)


def test_posterior_carries_selection_index():
    model, x, y = trained()
    post = lk.laplace(
        model,
        loader(x, y),
        likelihood=lk.Categorical(),
        subset=_picks_last_entries,
        hyper_tuner=None,
    )
    masks = _last_layer_masks(model)

    assert list(post.index) == ["2.weight[idx]", "2.bias[idx]"]
    torch.testing.assert_close(
        post.index["2.weight[idx]"], masks["2.weight"].flatten().nonzero()[:, 0]
    )
    torch.testing.assert_close(post.index["2.bias[idx]"], torch.tensor([0, 2]))

    # A bare `Selection` works too, named "given"; whole tensors leave the index empty.
    sel = _picks_last_entries(lk.core.ModelFn.from_module(model), None)
    given = lk.laplace(
        model, loader(x, y), likelihood=lk.Categorical(), subset=sel, hyper_tuner=None
    )
    whole = lk.laplace(model, loader(x, y), likelihood=lk.Categorical(), hyper_tuner=None)

    assert "given" in repr(given) and set(given.index) == set(post.index)
    assert dict(whole.index) == {}

    # It survives the tune and `snapshot`.
    tuned = lk.laplace(model, loader(x, y), likelihood=lk.Categorical(), subset=sel)

    assert torch.equal(tuned.index["2.bias[idx]"], torch.tensor([0, 2]))


def test_selection_posterior_detects_training():
    model, x, y = trained()
    post = lk.laplace(
        model,
        loader(x, y),
        likelihood=lk.Categorical(),
        subset=_picks_last_entries,
        hyper_tuner=None,
    )
    lk.predict_outputs(post, x[:3])

    # Train only the selected source tensor: its left-out entries are what moved.
    opt = torch.optim.SGD([model[2].weight], lr=0.1)
    opt.zero_grad()
    nn.functional.cross_entropy(model(x), y).backward()
    opt.step()

    for use in (
        lambda: lk.predict_outputs(post, x[:3]),
        lambda: lk.predict(post, x[:3]),
        lambda: lk.predict(post, x[:3], lk.WeightSamples(2)),
    ):
        with pytest.raises(RuntimeError, match="fixed parameter '2.weight' changed in place"):
            use()


# --- dense tuning from one eigendecomposition --------------------------------------------


class _CountLinalg(TorchDispatchMode):
    """Counts the aten ops whose name contains "eigh" or "cholesky"."""

    def __init__(self) -> None:
        super().__init__()
        self.calls: collections.Counter[str] = collections.Counter()

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        name = func.overloadpacket.__name__

        for kind in ("eigh", "cholesky"):
            if kind in name:
                self.calls[kind] += 1

        return func(*args, **(kwargs or {}))


def _small_classifier(n: int, dtype: torch.dtype):
    """An MLP with 99 weights on `n` blob points, three classes."""

    torch.manual_seed(0)
    x, y = blobs(n=n, dtype=dtype)
    model = nn.Sequential(nn.Linear(2, 16), nn.Tanh(), nn.Linear(16, 3)).to(dtype)
    opt = torch.optim.Adam(model.parameters(), lr=0.05)

    for _ in range(100):
        opt.zero_grad()
        nn.functional.cross_entropy(model(x), y).backward()
        opt.step()

    return model, x, y


def _cholesky_tuned(model, data, likelihood, where):
    """The reference: the same fit tuned on the Cholesky path (no cached eigenvectors)."""

    post = lk.laplace(
        model,
        data,
        likelihood=likelihood,
        hessian_approx=lk.DenseGGN(joint=True),
        subset=lk.subsets.all,
        hyper_tuner=None,
    )
    assert post.curvature.op.eig == ()

    return lk.tune(post, lk.Minimize(lambda p: -p.log_marglik()))


def test_tune_one_eigh():
    model, x, y = trained()
    counter = _CountLinalg()

    with counter:
        post = lk.laplace(
            model,
            loader(x, y),
            likelihood=lk.Categorical(),
            hessian_approx=lk.DenseGGN(joint=True),
            subset=lk.subsets.all,
        )

    assert counter.calls["eigh"] == 1
    assert counter.calls["cholesky"] == 0
    assert len(post.curvature.op.eig) == 2
    assert "MarginalLikelihood()" in str(post)


def test_tune_same_optimum():
    """lambda and sigma match the Cholesky path's in float64.

    The log marglik (about -131) is flat at its optimum: moving lambda by 1e-6 relative
    changes it by about 1e-11, which is round-off. Both paths must reach the same
    objective value to 1e-8. lambda and sigma are compared at 1e-4, the precision the
    objective resolves. CI runners differ by 1.2e-6 in lambda, depending on their BLAS.
    """

    model, x, y = _regression(torch.float64)
    data = loader(x, y, k=4)
    lik = copy.replace(lk.Gaussian(), sigma=torch.as_tensor(torch.tensor(1.0, dtype=torch.float64)))

    post = lk.laplace(
        model, data, likelihood=lik, hessian_approx=lk.DenseGGN(joint=True), subset=lk.subsets.all
    )
    want = _cholesky_tuned(
        model, data, lik, lambda p: (p.prior.prior_precision, p.likelihood.sigma)
    )

    assert len(post.curvature.op.eig) == 2
    torch.testing.assert_close(post.log_marglik(), want.log_marglik(), rtol=0, atol=1e-8)

    torch.testing.assert_close(
        post.prior.prior_precision, want.prior.prior_precision, rtol=1e-4, atol=0
    )
    torch.testing.assert_close(post.likelihood.sigma, want.likelihood.sigma, rtol=1e-4, atol=0)


def test_eigen_rank_deficient_float32():
    """N C < P in float32: the GGN has round-off eigenvalues of either sign, and the tune
    still finds the Cholesky path's lambda.

    The float32 objective is noisy at about 1e-5 near its flat optimum, so either path
    stops within about 3e-3 of the float64 optimum (measured over five data sizes: 0 to
    2.6e-3 apart). The tolerance is 5e-3.
    """

    model, x, y = _small_classifier(n=12, dtype=torch.float32)
    data = loader(x, y, k=2)
    p = sum(w.numel() for w in model.parameters())

    assert 12 * 3 < p

    post = lk.laplace(
        model,
        data,
        likelihood=lk.Categorical(),
        hessian_approx=lk.DenseGGN(joint=True),
        subset=lk.subsets.all,
    )
    want = _cholesky_tuned(model, data, lk.Categorical(), lambda p: p.prior.prior_precision)

    evals = post.curvature.op.eig[0]
    assert evals.dtype == torch.float32 and (evals[: p - 12 * 3].abs() < 1e-4).all()
    torch.testing.assert_close(
        post.prior.prior_precision, want.prior.prior_precision, rtol=5e-3, atol=0
    )
    assert torch.isfinite(post.log_marglik())
    assert torch.isfinite(lk.predict(post, x).probs).all()


def test_eigen_indefinite_raises():
    """A joint DenseHessian away from the optimum is indefinite: with a small prior, the
    tune raises StructureError instead of returning NaN."""

    x, y = blobs(n=12, dtype=torch.float64)
    torch.manual_seed(1)
    model = nn.Sequential(nn.Linear(2, 16), nn.Tanh(), nn.Linear(16, 3)).to(torch.float64)
    small = lk.priors.Isotropic(torch.tensor(1e-3, dtype=torch.float64))
    kwargs = {
        "likelihood": lk.Categorical(),
        "hessian_approx": lk.DenseHessian(joint=True),
        "subset": lk.subsets.all,
        "prior": small,
    }

    untuned = lk.laplace(model, loader(x, y, k=2), hyper_tuner=None, **kwargs)
    prec = untuned.n_data * untuned.curvature.op.mat + 1e-3 * torch.eye(99, dtype=torch.float64)

    assert torch.linalg.eigvalsh(prec).min() < 0

    with pytest.raises(lk.StructureError, match="not positive definite"):
        lk.laplace(model, loader(x, y, k=2), **kwargs)
