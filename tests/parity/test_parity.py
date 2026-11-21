"""Match laplace-torch: log marginal likelihood and GLM predictive moments.

Regression is laplace-torch's regression example; classification is the docs
classification example's setup (see `make_reference.py`).

KFAC: laplace-torch keeps a layer's bias as its own Kronecker block (Curvlinops'
`separate_weight_and_bias=True`), while laplacekit folds it in as an extra input column
of ones, `[W | b]`. The two agree on the weight-weight and bias-bias blocks; laplace-torch
drops the weight-bias cross terms. So bias-free layers match exactly, and with biases
laplace-torch's precision is ours with those cross terms set to zero.
"""

import copy
import dataclasses
import json
from pathlib import Path

import pytest
import torch
from torch import nn

import laplacekit as lk

REF = json.loads(Path(__file__).with_name("reference.json").read_text())
DT = torch.float64


def _tensor(x: object) -> torch.Tensor:
    return torch.tensor(x, dtype=DT)


def _regression(bias: bool = True) -> nn.Module:
    model = nn.Sequential(nn.Linear(1, 50, bias=bias), nn.Tanh(), nn.Linear(50, 1, bias=bias))
    state = REF["state_dict"] if bias else REF["nobias_state_dict"]
    model.to(DT).load_state_dict({k: _tensor(v) for k, v in state.items()})

    return model


def _classifier() -> nn.Module:
    model = nn.Sequential(
        nn.Linear(2, 20),
        nn.BatchNorm1d(20),
        nn.ReLU(),
        nn.Linear(20, 20),
        nn.BatchNorm1d(20),
        nn.ReLU(),
        nn.Linear(20, 4, bias=False),
    ).to(DT)
    state = REF["classification"]["state_dict"]
    # Batch-norm step counters are integers; everything else is float64.
    model.load_state_dict(
        {
            k: torch.tensor(v) if k.endswith("num_batches_tracked") else _tensor(v)
            for k, v in state.items()
        }
    )

    return model.eval()


def _posterior(model, lik, batch, subset, approx, prec):
    fn = lk.core.ModelFn.from_module(model)
    caps = lk.core.probe(fn, lik, batch)
    picked = subset(fn, caps)

    # A `Selection` fits its entries through a model that scatters them back.
    if isinstance(picked, lk.core.Selection):
        fn, picked = lk.core.select(fn, picked)
        caps = lk.core.probe(fn, lik, batch, params=picked)

    fit = lk.core.fit_curvature(fn, lik, picked, approx, capabilities=caps)
    stat = fit.update(fit.init(), batch)
    post = fit.posterior(
        fit.finalize(stat), prior=lk.priors.Isotropic(_tensor(prec)), n_data=stat.count
    )

    return post, -lk.objectives.neg_log_marglik(
        dataclasses.replace(post, data_fit=(stat.data_fit,))
    )


def _fit(subset, approx, ref, bias: bool = True):
    model = _regression(bias)
    x, y, x_test = _tensor(REF["X"]), _tensor(REF["Y"]), _tensor(REF["X_test"])
    lik = copy.replace(lk.Gaussian(), sigma=torch.as_tensor(_tensor(0.3)))
    prec = ref.get("prior_prec", 1.0)
    post, log_marglik = _posterior(model, lik, lk.Batch(x, y), subset, approx, prec)
    out = lk.predict_outputs(post, x_test)
    f_mu = out.mean
    f_var = out.cov.diagonal()
    ref_mu, ref_var = _tensor(ref["f_mu"]), _tensor(ref["f_var"])

    # The reference stores flat lists; ours are [N, 1].
    assert f_mu.shape == f_var.shape == (ref_mu.numel(), 1) and ref_mu.numel() == ref_var.numel()
    assert torch.allclose(f_mu, ref_mu.reshape(f_mu.shape), rtol=1e-4)
    assert torch.allclose(f_var, ref_var.reshape(f_var.shape), rtol=1e-4)
    assert float(log_marglik.detach()) == pytest.approx(ref["log_marglik"], rel=1e-4)

    return post


def test_last_layer_full():
    _fit(lk.subsets.last_layer, lk.DenseGGN(), REF["last_layer_full"])


def test_hessian_last_layer_full():
    _fit(lk.subsets.last_layer, lk.DenseHessian(), REF["hessian_last_layer_full"])


def test_hessian_all_full():
    """At prior precision 1 the all-weights Hessian precision is indefinite, so the
    reference uses the `prior_prec` it stores."""

    ref = REF["hessian_all_full"]
    post = _fit(lk.subsets.all, lk.DenseHessian(joint=True), ref)

    assert ref["prior_prec"] != 1.0
    assert isinstance(post.precision(), lk.core.Joint)


def test_all_diag():
    _fit(lk.subsets.all, lk.DiagGGN(), REF["all_diag"])


def test_all_kfac_without_biases():
    _fit(lk.subsets.all, lk.KFAC(), REF["nobias_all_kron"], bias=False)


def test_last_layer_kfac_without_biases():
    _fit(lk.subsets.last_layer, lk.KFAC(), REF["nobias_last_layer_kron"], bias=False)


def _continual_kfac(splits: list[int]) -> lk.LaplacePosterior:
    """KFAC on the bias-free model, fit on the parts of the data in turn, each later fit
    with `KronSum` of the one before as its prior."""

    model = _regression(bias=False)
    x, y = _tensor(REF["X"]), _tensor(REF["Y"])
    bounds = [0, *splits, x.shape[0]]
    prior: lk.core.Prior = lk.priors.Isotropic(_tensor(1.0))
    post = None

    for lo, hi in zip(bounds[:-1], bounds[1:], strict=True):
        post = lk.laplace(
            model,
            [lk.Batch(x[lo:hi], y[lo:hi])],
            likelihood=copy.replace(lk.Gaussian(), sigma=torch.as_tensor(_tensor(0.3))),
            hessian_approx=lk.KFAC(),
            subset=lk.subsets.all,
            prior=prior,
            hyper_tuner=None,
        )
        prior = lk.priors.KronSum.of(post)

    assert post is not None

    return post


def _check_continual(post: lk.LaplacePosterior, ref: dict) -> None:
    prec = post.precision()
    dense = torch.block_diag(
        *(blk.mv(torch.eye(blk.diagonal().numel(), dtype=DT)) for blk in prec.blocks.values())
    )
    out = lk.predict_outputs(post, _tensor(REF["X_test"]))
    ref_mu, ref_var = _tensor(ref["f_mu"]), _tensor(ref["f_var"])

    assert torch.allclose(dense, _tensor(ref["precision"]), rtol=1e-4)
    assert torch.allclose(out.mean, ref_mu.reshape(out.mean.shape), rtol=1e-4)
    assert torch.allclose(out.cov.diagonal(), ref_var.reshape(out.mean.shape), rtol=1e-4)


def test_kronsum_matches_laplace_torch_continual():
    """laplace-torch's `fit(D2, override=False)`: the output factors add and the input
    factors are averaged, weighted by the example counts."""

    _check_continual(_continual_kfac([60]), REF["nobias_online_kron"])


def test_kronsum_matches_laplace_torch_continual_three_fits():
    _check_continual(_continual_kfac([50, 100]), REF["nobias_online_kron_3"])


def _masks(model: nn.Module, indices: list[int]) -> dict[str, torch.Tensor]:
    """Bool masks from indices into `parameters_to_vector(model.parameters())`."""

    flat = torch.zeros(sum(p.numel() for p in model.parameters()), dtype=torch.bool)
    flat[indices] = True
    masks, start = {}, 0

    for name, p in model.named_parameters():
        masks[name] = flat[start : start + p.numel()].reshape(p.shape)
        start += p.numel()

    return masks


def _flat(model: nn.Module, sel: lk.core.Selection) -> list[int]:
    """The selected entries as indices into `parameters_to_vector(model.parameters())`."""

    offsets, start = {}, 0

    for name, p in model.named_parameters():
        offsets[name] = start
        start += p.numel()

    return sorted(int(offsets[name] + i) for name, idx in sel.index.items() for i in idx)


def _picked(subset, model: nn.Module) -> lk.core.Selection:
    out = subset(lk.core.ModelFn.from_module(model), lk.core.Capabilities())
    assert isinstance(out, lk.core.Selection)

    return out


def _diag_all() -> lk.LaplacePosterior:
    """laplace-torch's `LargestVarianceDiagLaplaceSubnetMask` fit: diagonal GGN over all
    weights, prior precision 1, on the whole regression data."""

    x, y = _tensor(REF["X"]), _tensor(REF["Y"])

    return lk.laplace(
        _regression(),
        [(x, y)],
        likelihood=copy.replace(lk.Gaussian(), sigma=torch.as_tensor(_tensor(0.3))),
        subset=lk.subsets.all,
        hessian_approx=lk.DiagGGN(),
        prior=lk.priors.Isotropic(_tensor(1.0)),
        hyper_tuner=None,
    )


def _strategy(key: str) -> lk.core.Subset:
    """The subset that picks the entries of `REF[key]`, through `lk.subnet` where
    laplace-torch picked them with a strategy."""

    if key == "subnet_magnitude":
        return lk.subnet.largest_magnitude(30)

    if key == "subnet_variance":
        return lk.subnet.largest_variance(_diag_all(), 30)

    return lk.subsets.from_mask(_masks(_regression(), REF[key]["indices"]))


def test_magnitude_indices_match_laplace_torch():
    model = _regression()
    top = (
        torch.nn.utils.parameters_to_vector(model.parameters()).detach().abs().sort(descending=True)
    )
    gap = REF["subnet_magnitude"]["gap"]

    # The 30th and 31st largest scores sit further apart than float64 noise on their scale.
    assert gap > 1e-6
    assert gap > 1e-4 * float(top.values[29])

    sel = _picked(lk.subnet.largest_magnitude(30), model)

    assert _flat(model, sel) == REF["subnet_magnitude"]["indices"]


def test_largest_variance_indices_match_laplace_torch():
    model = _regression()
    post = _diag_all()

    assert post.n_data == 150

    var = post.variance
    top = torch.cat([var[n].reshape(-1) for n in post.mean]).sort(descending=True)
    gap = REF["subnet_variance"]["gap"]

    assert gap > 1e-6
    assert gap > 1e-4 * float(top.values[29])

    sel = _picked(lk.subnet.largest_variance(post, 30), model)

    assert _flat(model, sel) == REF["subnet_variance"]["indices"]


def test_subnet_magnitude_diag():
    ref = REF["subnet_magnitude"]
    post = _fit(_strategy("subnet_magnitude"), lk.DiagGGN(), ref["diag"])

    assert sum(m.numel() for m in post.mean.values()) == 30


def test_all_full_regression():
    post = _fit(lk.subsets.all, lk.DenseGGN(joint=True), REF["all_full"])

    assert isinstance(post.precision(), lk.core.Joint)


@pytest.mark.parametrize("key", ["subnet_magnitude", "subnet_variance", "subnet_names"])
def test_subnet_full(key):
    """laplace-torch's subnetwork Laplace keeps the full matrix across the chosen entries,
    so the selected entries of two layers (`subnet_names`) are coupled too."""

    ref = REF[key]
    post = _fit(_strategy(key), lk.DenseGGN(joint=True), ref["full"])

    assert sum(m.numel() for m in post.mean.values()) == len(ref["indices"])


def _dense(op, layout) -> torch.Tensor:
    mats = []

    for b in layout.blocks:
        eye = torch.eye(layout.size(b), dtype=DT)
        mats.append(torch.stack([op.blocks[b].mv(e) for e in eye], dim=1))

    return torch.block_diag(*mats)


def test_all_kfac_with_biases_is_ours_without_weight_bias_terms():
    model = _regression()
    x, y = _tensor(REF["X"]), _tensor(REF["Y"])
    lik = copy.replace(lk.Gaussian(), sigma=torch.as_tensor(_tensor(0.3)))
    post, _ = _posterior(model, lik, lk.Batch(x, y), lk.subsets.all, lk.KFAC(), 1.0)
    ours = _dense(post.precision(), post.layout)

    # Zero the weight-bias cross terms within each layer.
    start = 0

    for b, names in post.layout.blocks.items():
        n_w = post.layout.size(b) - post.layout.shapes[f"{b}.bias"][0]
        end = start + post.layout.size(b)
        ours[start : start + n_w, start + n_w : end] = 0
        ours[start + n_w : end, start : start + n_w] = 0
        start = end

        assert names == (f"{b}.weight", f"{b}.bias")

    want = _tensor(REF["all_kron"]["precision"])

    torch.testing.assert_close(ours, want, rtol=1e-6, atol=1e-8)


def _classify(subset, approx, ref):
    cls = REF["classification"]
    x, y, x_test = _tensor(cls["X"]), torch.tensor(cls["Y"]), _tensor(cls["X_test"])
    post, log_marglik = _posterior(
        _classifier(), lk.Categorical(), lk.Batch(x, y), subset, approx, 5e-4
    )
    out = lk.predict_outputs(post, x_test)

    torch.testing.assert_close(out.mean, _tensor(ref["f_mu"]), rtol=1e-6, atol=1e-8)
    torch.testing.assert_close(out.cov.per_input(), _tensor(ref["f_var"]), rtol=1e-6, atol=1e-10)
    assert float(log_marglik.detach()) == pytest.approx(ref["log_marglik"], rel=1e-6)

    return post


@pytest.mark.parametrize(
    ("approx", "key"),
    [(lk.DenseGGN(), "last_layer_full"), (lk.KFAC(), "last_layer_kron")],
    ids=["full", "kfac"],
)
def test_classification_last_layer(approx, key):
    post = _classify(lk.subsets.last_layer, approx, REF["classification"][key])

    assert set(post.mean) == {"6.weight"}


def test_ef_last_layer_full():
    post = _classify(
        lk.subsets.last_layer,
        lk.DenseGGN(fisher="empirical"),
        REF["classification"]["ef_last_layer_full"],
    )

    assert set(post.mean) == {"6.weight"}


def test_ef_last_layer_kron():
    post = _classify(
        lk.subsets.last_layer,
        lk.KFAC(fisher="empirical"),
        REF["classification"]["ef_last_layer_kron"],
    )

    assert set(post.mean) == {"6.weight"}


def test_ef_all_full():
    post = _classify(
        lk.subsets.all,
        lk.DenseGGN(fisher="empirical", joint=True),
        REF["classification"]["ef_all_full"],
    )

    assert isinstance(post.precision(), lk.core.Joint)


def test_tuning_the_classifier_prior_from_a_far_start_reaches_the_optimum():
    """The docs page tunes from the weight decay, 5e-4; the optimum is near 1."""

    cls = REF["classification"]
    x, y = _tensor(cls["X"]), torch.tensor(cls["Y"])
    lik = lk.Categorical()
    fn = lk.core.ModelFn.from_module(_classifier())
    caps = lk.core.probe(fn, lik, lk.Batch(x, y))
    params = {k: v.detach() for k, v in lk.subsets.last_layer(fn, caps).items()}
    fit = lk.core.fit_curvature(fn, lik, params, lk.DenseGGN(), capabilities=caps)
    stat = fit.update(fit.init(), lk.Batch(x, y))
    post = fit.posterior(
        fit.finalize(stat), prior=lk.priors.Isotropic(_tensor(5e-4)), n_data=stat.count
    )
    post = dataclasses.replace(post, data_fit=(stat.data_fit,))
    objective = lk.objectives.neg_log_marglik
    best = lk.tune(post, lk.GridSearch(objective, values=torch.logspace(-4, 3, 2001, dtype=DT)))
    tuned = lk.tune(post, lk.Minimize(objective))

    assert float(tuned.prior.prior_precision) == pytest.approx(
        float(best.prior.prior_precision), rel=0.01
    )
    assert float(objective(tuned)) <= float(objective(best)) + 1e-6


@pytest.mark.parametrize(
    ("subset", "approx", "key", "bias"),
    [
        (lk.subsets.last_layer, lk.DenseGGN(), "last_layer_full", True),
        (lk.subsets.all, lk.DiagGGN(), "all_diag", True),
        (lk.subsets.all, lk.KFAC(), "nobias_all_kron", False),
    ],
    ids=["last-layer-full", "all-diag", "all-kfac-without-biases"],
)
def test_joint_tune_matches_laplace_torch(subset, approx, key, bias):
    """The front door tunes `prior.prior_precision` and `sigma` together, from one fit at sigma
    0.3."""

    ref = REF["joint_tune"][key]
    x, y = _tensor(REF["X"]), _tensor(REF["Y"])
    post = lk.laplace(
        _regression(bias),
        [lk.Batch(x, y)],
        likelihood=copy.replace(lk.Gaussian(), sigma=torch.as_tensor(_tensor(0.3))),
        hessian_approx=approx,
        subset=subset,
    )
    sigma = post.likelihood.sigma

    assert sigma is not None
    assert float(post.prior.prior_precision) == pytest.approx(ref["prior_prec"], rel=1e-3)
    assert float(sigma) == pytest.approx(ref["sigma"], rel=1e-3)
    assert float(post.log_marglik()) == pytest.approx(ref["log_marglik"], rel=1e-3)
