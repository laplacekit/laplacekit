"""`lk.save` and `lk.load(f, model)`: round trips give the same predictions."""

import copy
import dataclasses
import io

import pytest
import torch
from torch import nn

import laplacekit as lk
from laplacekit import function_space as fs
from laplacekit import subsets
from laplacekit.core import (
    Capabilities,
    Dense,
    Joint,
    LowRankPlusDiag,
    ModelFn,
    Selection,
    StructureError,
)

F64 = torch.float64
X_TEST = torch.tensor([[0.0, 1.0], [3.0, -1.0], [-2.0, 0.5]], dtype=F64)


def _mlp(seed: int = 0, dtype: torch.dtype = F64) -> nn.Module:
    torch.manual_seed(seed)

    return nn.Sequential(nn.Linear(2, 16), nn.Tanh(), nn.Linear(16, 3)).to(dtype)


def _data(n: int, seed: int, dtype: torch.dtype = F64) -> list[lk.Batch]:
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 2, generator=g, dtype=dtype)

    return [lk.Batch(x, torch.randn(n, 3, generator=g, dtype=dtype))]


def _fit(model, data, approx, **kwargs) -> lk.LaplacePosterior:
    kwargs.setdefault(
        "likelihood",
        copy.replace(lk.Gaussian(), sigma=torch.as_tensor(torch.tensor(0.5, dtype=F64))),
    )
    kwargs.setdefault("hyper_tuner", None)

    return lk.laplace(model, data, hessian_approx=approx, **kwargs)


def _round_trip(tree, model):
    f = io.BytesIO()
    lk.save(tree, f)
    f.seek(0)

    return lk.load(f, model)


def _select(subset, model: nn.Module) -> Selection:
    out = subset(ModelFn.from_module(model), Capabilities())
    assert isinstance(out, Selection)

    return out


def _assert_same_predictions(a: lk.LaplacePosterior, b: lk.LaplacePosterior) -> None:
    assert torch.equal(a.log_marglik(), b.log_marglik())

    out_a, out_b = lk.predict_outputs(a, X_TEST), lk.predict_outputs(b, X_TEST)

    assert torch.equal(out_a.mean, out_b.mean)
    assert torch.equal(out_a.cov.per_input(), out_b.cov.per_input())

    draws_a, draws_b = (
        a.sample(4, generator=torch.Generator().manual_seed(3)),
        b.sample(4, generator=torch.Generator().manual_seed(3)),
    )

    assert draws_a.keys() == draws_b.keys()
    assert all(torch.equal(draws_a[k], draws_b[k]) for k in draws_a)


@pytest.mark.parametrize(
    "approx", [lk.DiagGGN(), lk.DenseGGN(), lk.KFAC(), lk.LowRankGGN(rank=3)], ids=repr
)
def test_round_trip_laplace_posterior(approx) -> None:
    model = _mlp()
    post = _fit(model, _data(12, 0), approx)

    loaded = _round_trip(post, model)

    assert type(loaded) is type(post)
    assert type(loaded.curvature) is type(post.curvature)
    assert repr(loaded) == repr(post)
    _assert_same_predictions(loaded, post)


def test_round_trip_kfac_keeps_factors_and_low_rank_is_joint() -> None:
    model = _mlp()
    kfac = _fit(model, _data(12, 0), lk.KFAC())
    low = _fit(model, _data(12, 0), lk.LowRankGGN(rank=3))

    blocks = _round_trip(kfac, model).curvature.blocks
    joint = _round_trip(low, model).curvature

    for name, block in blocks.items():
        assert len(block.factors) == 3
        assert all(
            torch.equal(a, b)
            for a, b in zip(block.factors, kfac.curvature.blocks[name].factors, strict=True)
        )

    assert isinstance(joint, Joint) and isinstance(joint.op, LowRankPlusDiag)


def test_round_trip_selection_posterior() -> None:
    """A `Selection` posterior loads with the selected model, `sel.apply(fn)`."""

    model = _mlp()
    sel = _select(lk.subnet.random(30), model)
    post = _fit(model, _data(12, 0), lk.DiagGGN(), subset=sel)

    loaded = _round_trip(post, sel.apply(ModelFn.from_module(model)))

    assert loaded.index.keys() == post.index.keys()
    _assert_same_predictions(loaded, post)


def test_round_trip_continual_prior() -> None:
    """A posterior whose prior is an earlier posterior (or a `KronSum` of one) loads whole."""

    model = _mlp()
    post_1 = _fit(model, _data(12, 0), lk.KFAC())
    post_2 = _fit(model, _data(9, 2), lk.KFAC(), prior=lk.priors.KronSum.of(post_1))
    diag_1 = _fit(model, _data(12, 0), lk.DiagGGN())
    diag_2 = _fit(model, _data(9, 2), lk.DiagGGN(), prior=diag_1)

    loaded = _round_trip(post_2, model)
    chained = _round_trip(diag_2, model)

    assert isinstance(loaded.prior, lk.priors.KronSum)
    assert isinstance(chained.prior, lk.LaplacePosterior)
    _assert_same_predictions(loaded, post_2)
    _assert_same_predictions(chained, diag_2)


def test_round_trip_fixed_and_learned_sigma() -> None:
    model = _mlp()
    fixed = _fit(model, _data(12, 0), lk.DiagGGN(), likelihood=lk.Gaussian(sigma=1.0))
    tuned = _fit(
        model,
        _data(12, 0),
        lk.DiagGGN(),
        likelihood=lk.Gaussian(),
        hyper_tuner=lk.MarginalLikelihood(),
    )

    assert float(tuned.prior.prior_precision) != 1.0

    for post in (fixed, tuned):
        loaded = _round_trip(post, model)

        assert loaded.prior.mean is None
        assert loaded.likelihood.learned == post.likelihood.learned
        assert torch.equal(loaded.likelihood.sigma, post.likelihood.sigma)
        assert torch.equal(loaded.prior.prior_precision, post.prior.prior_precision)
        _assert_same_predictions(loaded, post)


def test_round_trip_subset_gp() -> None:
    model = _mlp()
    prior = lk.priors.Isotropic(torch.tensor(2.0, dtype=F64))
    lik = copy.replace(lk.Gaussian(), sigma=torch.as_tensor(torch.tensor(0.5, dtype=F64)))
    gp = fs.laplace(model, _data(12, 0), lik, method=fs.SubsetOfData(6), prior=prior)

    loaded = _round_trip(gp, model)
    out, want = (lk.predict_outputs(b, X_TEST) for b in (loaded, gp))

    assert torch.equal(out.mean, want.mean)
    assert torch.equal(out.cov.per_input(), want.cov.per_input())


def test_loaded_posterior_passes_version_guard() -> None:
    model = _mlp()
    post = _fit(model, _data(12, 0), lk.KFAC())

    f = io.BytesIO()
    lk.save(post, f)
    state = {k: v.clone() for k, v in model.state_dict().items()}

    # a fresh process: another model whose weights are then restored
    restored = _mlp(seed=1)
    restored.load_state_dict(state)
    f.seek(0)
    loaded = lk.load(f, restored)

    loaded.check_versions()
    assert torch.equal(
        lk.predict_outputs(loaded, X_TEST).mean, lk.predict_outputs(post, X_TEST).mean
    )

    with torch.no_grad():
        restored[0].weight.add_(1.0)

    with pytest.raises(RuntimeError, match="changed in place"):
        lk.predict_outputs(loaded, X_TEST)


def test_file_is_read_weights_only() -> None:
    f = io.BytesIO()
    lk.save(_fit(_mlp(), _data(12, 0), lk.DiagGGN()), f)
    f.seek(0)

    blob = torch.load(f, weights_only=True)

    assert blob["format"] == "laplacekit/4"
    assert blob["spec"]["class"] == "laplacekit.core.posteriors:LaplacePosterior"
    assert all(isinstance(v, torch.Tensor) for v in blob["tensors"].values())


def test_load_refuses_other_files() -> None:
    f = io.BytesIO()
    torch.save({"weights": torch.ones(2)}, f)
    f.seek(0)

    with pytest.raises(ValueError, match="not a laplacekit/4 file"):
        lk.load(f, _mlp())


def test_save_refuses_what_it_cannot_name() -> None:
    with pytest.raises(TypeError, match=r"\['call'\].*lk\.core\.serializable"):
        lk.save({"call": print}, io.BytesIO())

    with pytest.raises(TypeError, match="not a registered class"):
        lk.save({"thing": _Plain(2)}, io.BytesIO())


def test_registered_dataclass_round_trips() -> None:
    f = io.BytesIO()
    lk.save({"thing": _Registered(2)}, f)
    f.seek(0)

    assert lk.load(f, _mlp())["thing"] == _Registered(2)


@dataclasses.dataclass(frozen=True)
class _Plain:
    n: int


@lk.core.serializable
@dataclasses.dataclass(frozen=True)
class _Registered:
    n: int


def test_load_moves_leaves_to_the_model() -> None:
    model = _mlp()
    post = _fit(model, _data(12, 0), lk.DiagGGN())

    loaded = _round_trip(post, model)
    meta = lk.load(_saved(post), model, map_location="meta")

    assert all(v.device.type == "cpu" for v in loaded.mean.values())
    assert all(v.device.type == "meta" for v in meta.mean.values())


def _saved(obj) -> io.BytesIO:
    f = io.BytesIO()
    lk.save(obj, f)
    f.seek(0)

    return f


def test_round_trip_dense_block() -> None:
    model = _mlp()
    post = _fit(model, _data(12, 0), lk.DenseGGN())

    loaded = _round_trip(post, model)

    assert all(isinstance(b, Dense) for b in loaded.curvature.blocks.values())


def test_save_load_roundtrip_with_eig() -> None:
    """A tuned joint DenseGGN keeps its eigendecomposition through a round trip."""

    model = _mlp()
    kwargs = {"subset": subsets.all, "hyper_tuner": lk.MarginalLikelihood()}
    post = _fit(model, _data(12, 0), lk.DenseGGN(joint=True), **kwargs)

    loaded = _round_trip(post, model)

    assert len(loaded.curvature.op.eig) == 2
    assert all(
        torch.equal(a, b)
        for a, b in zip(loaded.curvature.op.eig, post.curvature.op.eig, strict=True)
    )
    _assert_same_predictions(loaded, post)


def test_save_writes_views_compactly() -> None:
    """A view leaf is written as its own entries, not its whole storage."""

    big = torch.zeros(1000, 10)
    f, g = io.BytesIO(), io.BytesIO()
    lk.save({"a": big[:, :1]}, f)
    lk.save({"a": big[:, :1].clone()}, g)

    assert len(f.getvalue()) < 2 * len(g.getvalue())


# --- safety and model checks -------------------------------------------------------------


def _forged(spec: dict) -> io.BytesIO:
    f = io.BytesIO()
    torch.save({"format": "laplacekit/4", "spec": spec, "tensors": {}}, f)
    f.seek(0)

    return f


@pytest.mark.parametrize(
    "key",
    [
        "laplacekit.core.serialize:pickle.FunctionType",
        "importlib._bootstrap_external:_code_type",
        "laplacekit_evil.payload:Run",
        "os:system",
    ],
)
def test_load_refuses_classes_outside_the_registry(key) -> None:
    with pytest.raises(StructureError, match="registered|outside laplacekit"):
        lk.load(_forged({"class": key, "fields": {}}), _mlp())


def test_load_calls_no_function_from_the_file(tmp_path) -> None:
    """A public laplacekit function is not callable from a file: functions are data."""

    target = tmp_path / "victim.pt"
    target.write_bytes(b"keep")
    spec = {"function": "laplacekit.core.serialize:save"}

    with pytest.raises(StructureError, match="not registered"):
        lk.load(_forged(spec), _mlp())

    assert target.read_bytes() == b"keep"


def test_load_refuses_a_foreign_package_with_the_prefix(tmp_path, monkeypatch) -> None:
    pkg = tmp_path / "laplacekit_evil"
    pkg.mkdir()
    marker = tmp_path / "ran"
    (pkg / "__init__.py").write_text(f"open({str(marker)!r}, 'w').write('x')\n")
    monkeypatch.syspath_prepend(str(tmp_path))

    with pytest.raises(StructureError, match="outside laplacekit"):
        lk.load(_forged({"class": "laplacekit_evil:Thing", "fields": {}}), _mlp())

    assert not marker.exists()


def test_selection_posterior_loads_with_the_plain_module() -> None:
    model = _mlp()
    sel = _select(lk.subnet.random(30), model)
    post = _fit(model, _data(12, 0), lk.DiagGGN(), subset=sel)

    loaded = _round_trip(post, model)

    _assert_same_predictions(loaded, post)
    assert float(lk.predict_outputs(loaded, X_TEST).variance.sum()) > 0


def test_load_refuses_other_weights_and_architectures() -> None:
    model = _mlp()
    post = _fit(model, _data(12, 0), lk.DiagGGN())
    f = io.BytesIO()
    lk.save(post, f)

    for other, match in [
        (_mlp(seed=1), "other values than when saved"),
        (_wide(), "other shape"),
    ]:
        f.seek(0)

        with pytest.raises(StructureError, match=match):
            lk.load(f, other)


def _wide() -> nn.Module:
    return nn.Sequential(nn.Linear(2, 16), nn.Tanh(), nn.Linear(16, 4)).to(F64)


def test_lambda_mask_needs_the_likelihood_back() -> None:
    x = torch.randn(12, 2, dtype=F64)
    y = torch.randint(0, 3, (12,))
    lik = lk.Categorical(mask_fn=lambda f, inputs: None)
    model = _mlp()
    post = lk.laplace(model, [(x, y)], lik, hessian_approx=lk.DiagGGN(), hyper_tuner=None)
    f = io.BytesIO()
    lk.save(post, f)
    f.seek(0)

    with pytest.raises(StructureError, match=r"\(likelihood\.mask_fn\).*likelihood=\.\.\."):
        lk.load(f, model)

    f.seek(0)
    loaded = lk.load(f, model, likelihood=lik)

    assert loaded.likelihood is lik


# --- the whole module, malformed files ---------------------------------------------------


def _blob(obj) -> dict:
    f = io.BytesIO()
    lk.save(obj, f)
    f.seek(0)

    return torch.load(f, weights_only=True)


def _file(blob: dict) -> io.BytesIO:
    f = io.BytesIO()
    torch.save(blob, f)
    f.seek(0)

    return f


def _tensor_keys(spec) -> list[str]:
    """The tensor keys in `spec`, depth first."""

    if isinstance(spec, dict):
        if "tensor" in spec:
            return [spec["tensor"]]

        return [k for v in spec.values() for k in _tensor_keys(v)]

    if isinstance(spec, list):
        return [k for v in spec for k in _tensor_keys(v)]

    return []


def test_load_refuses_a_change_outside_the_fitted_entries() -> None:
    """As `load_state_dict(strict=True)`: every parameter and buffer, not only the fitted."""

    model = _mlp()
    post = _fit(model, _data(12, 0), lk.DiagGGN())  # the last layer
    f = _saved(post)

    with torch.no_grad():
        model[0].weight.add_(0.1)

    with pytest.raises(StructureError, match=r"'0\.weight'"):
        lk.load(f, model)


def test_load_refuses_a_change_of_an_unselected_entry() -> None:
    model = _mlp()
    sel = _select(lk.subnet.random(30), model)
    post = _fit(model, _data(12, 0), lk.DiagGGN(), subset=sel)
    f = _saved(post)
    picked = post.index["2.weight[idx]"]
    other = next(i for i in range(model[2].weight.numel()) if i not in set(picked.tolist()))

    with torch.no_grad():
        model[2].weight.view(-1)[other] += 0.1

    with pytest.raises(StructureError, match=r"'2\.weight'"):
        lk.load(f, model)


def test_load_refuses_a_changed_buffer() -> None:
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(2, 8), nn.BatchNorm1d(8), nn.Linear(8, 3)).to(F64).eval()
    post = _fit(model, _data(12, 0), lk.DiagGGN())
    f = _saved(post)
    model[1].running_mean.add_(1.0)

    with pytest.raises(StructureError, match=r"'1\.running_mean'"):
        lk.load(f, model)


def test_load_refuses_an_index_beyond_the_model() -> None:
    model = _mlp()
    sel = _select(lk.subnet.random(30), model)
    blob = _blob(_fit(model, _data(12, 0), lk.DiagGGN(), subset=sel))
    index = blob["spec"]["fields"]["index"]
    key = _tensor_keys(index)[0]
    blob["tensors"][key] = blob["tensors"][key] + 10_000

    with pytest.raises(StructureError, match="index"):
        lk.load(_file(blob), model)


def test_load_refuses_a_wrong_shaped_curvature() -> None:
    model = _mlp()
    blob = _blob(_fit(model, _data(12, 0), lk.DiagGGN()))
    key = _tensor_keys(blob["spec"]["fields"]["curvature"])[0]
    blob["tensors"][key] = torch.ones(5, dtype=F64)

    with pytest.raises(StructureError, match="curvature"):
        lk.load(_file(blob), model)


def _deep(n: int) -> dict:
    spec: dict = {"tuple": []}

    for _ in range(n):
        spec = {"tuple": [spec]}

    return spec


@pytest.mark.parametrize(
    "blob",
    [
        {"format": "laplacekit/4", "spec": None},
        {"format": "laplacekit/4", "spec": None, "tensors": [1]},
        {"format": "laplacekit/4", "spec": {"tensor": "7"}, "tensors": {}},
        {"format": "laplacekit/4", "spec": {"tensor": 7}, "tensors": {}},
        {"format": "laplacekit/4", "spec": {"tensor": "0"}, "tensors": {"0": 1}},
        {"format": "laplacekit/4", "spec": {"tuple": 5}, "tensors": {}},
        {"format": "laplacekit/4", "spec": {"dict": [[1]]}, "tensors": {}},
        {"format": "laplacekit/4", "spec": {"dict": [[[1], 2]]}, "tensors": {}},
        {"format": "laplacekit/4", "spec": {"dtype": 5}, "tensors": {}},
        {"format": "laplacekit/4", "spec": {"function": 5}, "tensors": {}},
        {"format": "laplacekit/4", "spec": {"model": 5}, "tensors": {}},
        {"format": "laplacekit/4", "spec": {"module_type": "torch.nn.nope:X"}, "tensors": {}},
        {
            "format": "laplacekit/4",
            "spec": {"class": "laplacekit.nope:X", "fields": {}},
            "tensors": {},
        },
        {
            "format": "laplacekit/4",
            "spec": {"class": "laplacekit.likelihoods:Gaussian", "fields": 5},
            "tensors": {},
        },
        {
            "format": "laplacekit/4",
            "spec": {"class": "laplacekit.likelihoods:Gaussian", "fields": {"sigma": 1, "x": 2}},
            "tensors": {},
        },
        {"format": "laplacekit/4", "spec": _deep(500), "tensors": {}},
    ],
    ids=lambda b: str(b.get("spec"))[:40],
)
def test_malformed_files_raise_load_errors(blob) -> None:
    with pytest.raises((ValueError, StructureError)):
        lk.load(_file(blob), _mlp())


def test_likelihood_override_must_match_the_saved_one() -> None:
    model = _mlp()
    tuned = _fit(
        model,
        _data(12, 0),
        lk.DiagGGN(),
        likelihood=lk.Gaussian(),
        hyper_tuner=lk.MarginalLikelihood(),
    )
    f = _saved(tuned)

    with pytest.raises(StructureError, match="Categorical.*Gaussian"):
        lk.load(f, model, likelihood=lk.Categorical())

    f.seek(0)

    with pytest.raises(StructureError, match="sigma"):
        lk.load(f, model, likelihood=lk.Gaussian())

    f.seek(0)

    with pytest.raises(StructureError, match="Likelihood"):
        lk.load(f, model, likelihood="gaussian")

    f.seek(0)
    same = copy.replace(lk.Gaussian(), sigma=tuned.likelihood.sigma.clone())

    assert lk.load(f, model, likelihood=same).likelihood is same
