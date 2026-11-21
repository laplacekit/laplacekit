"""The user-facing API after the UX overhaul: argument names, defaults, errors, tuners.

Numbers are checked against independent references (closed forms, dense matrices), not
against the library's own helpers.
"""

import copy
import inspect
import io
import math

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import laplacekit as lk
from laplacekit import core


def _data(n: int = 60):
    g = torch.Generator().manual_seed(0)
    x = torch.randn(n, 2, generator=g)
    y = (x[:, 0] > 0).long() + (x[:, 1] > 0).long()

    return x, y


def _model():
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(2, 6), nn.Tanh(), nn.Linear(6, 3))


def _batches(x, y, k: int = 3):
    n = x.shape[0] // k
    return [(x[i * n : (i + 1) * n], y[i * n : (i + 1) * n]) for i in range(k)]


# --- front door ---------------------------------------------------------------------


def test_laplace_signature():
    sig = inspect.signature(lk.laplace)
    names = list(sig.parameters)

    assert names == [
        "model",
        "data",
        "likelihood",
        "hessian_approx",
        "subset",
        "prior",
        "hyper_tuner",
        "generator",
    ]
    assert sig.parameters["likelihood"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert sig.parameters["hessian_approx"].default is None
    assert sig.parameters["subset"].default is lk.subsets.last_layer
    assert isinstance(sig.parameters["hyper_tuner"].default, lk.MarginalLikelihood)


def test_likelihood_is_positional():
    x, y = _data()
    post = lk.laplace(_model(), _batches(x, y), lk.Categorical(), hyper_tuner=None)

    assert isinstance(post, lk.LaplacePosterior)


def test_print_rows():
    x, y = _data()
    post = lk.laplace(_model(), _batches(x, y), lk.Categorical())
    text = repr(post)

    for row in ("subset", "hessian_approx", "prior", "hyper_tuner", "n_data"):
        assert f"\n  {row} " in text

    assert "approx   " not in text.replace("hessian_approx", "")
    assert "scale" not in text


def test_predict_method_by_keyword():
    x, y = _data()
    post = lk.laplace(_model(), _batches(x, y), lk.Categorical(), hyper_tuner=None)
    d = lk.predict(post, x, method=lk.Linearized(link=lk.Probit()))

    assert d.probs.shape == (60, 3)
    assert list(inspect.signature(lk.predict).parameters)[:3] == ["posterior", "inputs", "method"]


def test_function_space_laplace_takes_method():
    sig = inspect.signature(lk.function_space.laplace)

    assert "method" in sig.parameters
    assert "approx" not in sig.parameters
    assert "subset" in sig.parameters


# --- floats and defaults -----------------------------------------------------------


def test_priors_take_floats_and_default_to_one():
    assert float(lk.priors.Isotropic().prior_precision) == 1.0
    assert float(lk.priors.Isotropic(2.0).prior_precision) == 2.0
    assert isinstance(lk.priors.Isotropic(2.0).prior_precision, torch.Tensor)
    assert float(lk.priors.Diagonal().prior_precision) == 1.0
    assert float(lk.priors.Layerwise().prior_precision) == 1.0


def test_scalar_layerwise_and_diagonal_expand_and_tune_per_entry():
    x, y = _data()

    lw = lk.laplace(_model(), _batches(x, y), lk.Categorical(), prior=lk.priors.Layerwise())
    assert set(lw.prior.prior_precision) == {"2"}

    diag = lk.laplace(
        _model(),
        _batches(x, y),
        lk.Categorical(),
        hessian_approx=lk.DiagGGN(),
        prior=lk.priors.Diagonal(),
    )
    assert diag.prior.prior_precision.shape == (21,)
    assert diag.prior.prior_precision.unique().numel() > 1


def test_gaussian_sigma_none_is_learned_and_a_value_is_fixed():
    learned = lk.Gaussian()
    fixed = lk.Gaussian(sigma=0.3)

    assert learned.learned and not fixed.learned
    assert isinstance(fixed.sigma, torch.Tensor)
    assert float(fixed.sigma) == pytest.approx(0.3)

    g = torch.Generator().manual_seed(1)
    x = torch.randn(50, 1, generator=g)
    y = 2 * x + 0.1 * torch.randn(50, 1, generator=g)
    model = nn.Linear(1, 1)

    with torch.no_grad():
        model.weight.fill_(2.0)
        model.bias.zero_()

    tuned = lk.laplace(model, [(x, y)], lk.Gaussian(), subset=lk.subsets.all)
    kept = lk.laplace(model, [(x, y)], lk.Gaussian(sigma=0.3), subset=lk.subsets.all)

    # The marginal likelihood's sigma is near the residual scale, 0.1.
    assert 0.05 < float(tuned.likelihood.sigma) < 0.2
    assert float(kept.likelihood.sigma) == pytest.approx(0.3)
    assert tuned.likelihood.learned


def test_learned_per_output_sigma_is_left_to_marglik():
    lik = copy.replace(lk.Gaussian(), sigma=torch.as_tensor(torch.ones(3)))

    assert lik.learned and lik.tunable() == {}
    assert lik.unit() is lik


# --- errors --------------------------------------------------------------------------


def test_errors_subclass_builtins():
    assert issubclass(lk.StructureError, ValueError)
    assert issubclass(lk.CapabilityError, RuntimeError)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"likelihood": "classification"}, r"likelihood.*lk\.Categorical\(\)"),
        ({"likelihood": lk.Categorical}, r"instance.*lk\.Categorical\(\)"),
        ({"hessian_approx": "kron"}, r"hessian_approx.*lk\.KFAC\(\)"),
        ({"hessian_approx": lk.KFAC}, r"instance.*lk\.KFAC\(\)"),
        ({"prior": 1.0}, r"prior.*lk\.priors\.Isotropic\(1\.0\)"),
        ({"subset": "all"}, r"subset.*lk\.subsets\.all"),
        ({"hyper_tuner": "marglik"}, r"hyper_tuner.*lk\.MarginalLikelihood\(\)"),
    ],
)
def test_front_door_type_errors(kwargs, match):
    x, y = _data()
    args = {"likelihood": lk.Categorical(), **kwargs}

    with pytest.raises(TypeError, match=match):
        lk.laplace(_model(), _batches(x, y), **args)


def test_single_pair_as_data_is_a_type_error():
    x, y = _data()

    with pytest.raises(TypeError, match=r"\[\(inputs, targets\)\]"):
        lk.laplace(_model(), (x, y), lk.Categorical())


def test_dataset_as_data_is_a_type_error():
    x, y = _data()

    with pytest.raises(TypeError, match="DataLoader"):
        lk.laplace(_model(), TensorDataset(x, y), lk.Categorical())


def test_predict_link_as_method_is_a_type_error():
    x, y = _data()
    post = lk.laplace(_model(), _batches(x, y), lk.Categorical(), hyper_tuner=None)

    with pytest.raises(TypeError, match=r"lk\.Linearized\(link=lk\.Probit\(\)\)"):
        lk.predict(post, x, lk.Probit())


def test_categorical_refuses_float_targets():
    x, y = _data()

    with pytest.raises(ValueError, match="class indices"):
        lk.laplace(_model(), _batches(x, y.float()), lk.Categorical())


def test_objectives_take_the_data_forms_of_laplace():
    x, y = _data()
    post = lk.laplace(_model(), _batches(x, y), lk.Categorical(), hyper_tuner=None)
    loader = DataLoader(TensorDataset(x, y), batch_size=20)

    a = lk.objectives.predictive_nll([(x, y)])(post)
    b = lk.objectives.predictive_nll(loader)(post)
    c = lk.objectives.predictive_nll([lk.Batch(x, y)])(post)

    assert torch.allclose(a, b) and torch.allclose(a, c)

    with pytest.raises(TypeError, match=r"\[\(inputs, targets\)\]"):
        lk.objectives.predictive_nll((x, y))


# --- tuning --------------------------------------------------------------------------


def _linear_regression():
    """Bayesian linear regression: the evidence has a closed form."""

    g = torch.Generator().manual_seed(3)
    x = torch.randn(40, 3, generator=g, dtype=torch.float64)
    y = x @ torch.tensor([[1.0], [-2.0], [0.5]], dtype=torch.float64)
    y = y + 0.3 * torch.randn(40, 1, generator=g, dtype=torch.float64)
    model = nn.Linear(3, 1, bias=False).double()

    with torch.no_grad():
        model.weight.copy_(torch.linalg.lstsq(x, y).solution.T)

    return model, x, y


def _closed_form_log_evidence(x, y, lam, sigma, mean):
    """log N(y; 0, sigma^2 I + X X^T / lam) after moving the MAP mean (exact for linear)."""

    n = x.shape[0]
    cov = sigma**2 * torch.eye(n, dtype=x.dtype) + x @ x.T / lam

    return torch.distributions.MultivariateNormal(torch.zeros(n, dtype=x.dtype), cov).log_prob(
        y[:, 0]
    )


def test_marginal_likelihood_tuner_matches_closed_form_optimum():
    model, x, y = _linear_regression()
    post = lk.laplace(
        model,
        [(x, y)],
        lk.Gaussian(sigma=0.3),
        hessian_approx=lk.DenseGGN(),
        subset=lk.subsets.all,
        hyper_tuner=None,
    )
    tuned = lk.tune(post, lk.MarginalLikelihood())
    lam = float(tuned.prior.prior_precision)

    # The tuned precision maximizes the evidence over a fine grid of the closed form.
    grid = torch.logspace(-3, 2, 400, dtype=torch.float64)
    ev = torch.stack([_closed_form_log_evidence(x, y, g, 0.3, None) for g in grid])
    best = float(grid[ev.argmax()])

    assert math.log(lam) == pytest.approx(math.log(best), abs=0.05)


def test_tuners_are_callable_and_leave_the_input_alone():
    model, x, y = _linear_regression()
    post = lk.laplace(model, [(x, y)], lk.Gaussian(), subset=lk.subsets.all, hyper_tuner=None)
    tuned = lk.MarginalLikelihood()(post)

    assert float(post.prior.prior_precision) == 1.0
    assert float(post.likelihood.sigma) == 1.0
    assert float(tuned.prior.prior_precision) != 1.0
    assert float(tuned.likelihood.sigma) != 1.0


def test_grid_search_tuner_picks_the_best_value():
    x, y = _data()
    post = lk.laplace(_model(), _batches(x, y), lk.Categorical(), hyper_tuner=None)
    nll = lk.objectives.predictive_nll([(x, y)])
    values = [0.01, 1.0, 100.0]
    tuned = lk.tune(post, lk.GridSearch(nll, values=values))

    scores = [float(nll(lk.tune(post, lk.GridSearch(nll, values=[v])))) for v in values]
    want = values[min(range(3), key=scores.__getitem__)]

    assert float(tuned.prior.prior_precision) == pytest.approx(want)


def test_minimize_tuner_on_a_validation_objective():
    x, y = _data()
    post = lk.laplace(
        _model(), _batches(x, y), lk.Categorical(), hessian_approx=lk.DiagGGN(), hyper_tuner=None
    )
    nll = lk.objectives.predictive_nll([(x, y)])
    tuned = lk.tune(post, lk.Minimize(nll))

    assert float(nll(tuned)) <= float(nll(post)) + 1e-6


def test_laplace_with_grid_search_tuner():
    x, y = _data()
    nll = lk.objectives.predictive_nll([(x, y)])
    post = lk.laplace(
        _model(), _batches(x, y), lk.Categorical(), hyper_tuner=lk.GridSearch(nll, [0.1, 10.0])
    )

    assert float(post.prior.prior_precision) in (pytest.approx(0.1), pytest.approx(10.0))


def test_marginal_likelihood_without_targets_raises_before_the_fit():
    x, _ = _data()

    with pytest.raises(ValueError, match="hyper_tuner=None"):
        lk.laplace(
            _model(),
            [lk.Batch(x)],
            lk.Categorical(),
            hessian_approx=lk.DiagGGN(fisher="mc"),
        )


# --- posteriors and outputs --------------------------------------------------------


def test_posterior_names():
    x, y = _data()
    post = lk.laplace(_model(), _batches(x, y), lk.Categorical(), hyper_tuner=None)

    assert isinstance(post, lk.LaplacePosterior)
    assert isinstance(post, core.WeightPosterior)
    assert isinstance(post, core.Prior)
    assert float(post.n_data) == 60
    assert float(post.prior_precision) == 1.0


def test_predict_outputs_is_distribution_like():
    x, y = _data()
    post = lk.laplace(
        _model(),
        _batches(x, y),
        lk.Categorical(),
        hessian_approx=lk.DenseGGN(joint=True),
        hyper_tuner=None,
    )
    out = lk.predict_outputs(post, x[:5])

    assert out.mean.shape == (5, 3)
    assert torch.allclose(out.variance, out.covariance_matrix.diagonal(dim1=-2, dim2=-1))
    assert torch.allclose(out.stddev, out.variance.sqrt())
    assert out.covariance_matrix.shape == (5, 3, 3)

    # The covariance is J Sigma J^T: check against a dense Jacobian.
    params = dict(post.mean)
    cov = torch.linalg.inv(post.precision().op.mat)

    def f(flat):
        w, b = flat[:18].reshape(3, 6), flat[18:]
        h = torch.tanh(x[:5] @ post.model.module[0].weight.T + post.model.module[0].bias)
        return h @ w.T + b

    flat = torch.cat([params["2.weight"].reshape(-1), params["2.bias"]])
    jac = torch.autograd.functional.jacobian(f, flat).reshape(15, 21)
    want = (jac @ cov.to(jac.dtype) @ jac.T).reshape(5, 3, 5, 3)
    want = torch.stack([want[i, :, i, :] for i in range(5)])

    assert torch.allclose(out.covariance_matrix, want, atol=1e-5)


def test_save_and_load_without_a_template():
    x, y = _data()
    model = _model()
    post = lk.laplace(model, _batches(x, y), lk.Categorical(), hessian_approx=lk.KFAC())
    buf = io.BytesIO()
    lk.save(post, buf)
    buf.seek(0)
    back = lk.load(buf, model)

    assert type(back) is type(post)
    assert torch.allclose(back.prior_precision, post.prior_precision)
    assert torch.allclose(lk.predict(back, x).probs, lk.predict(post, x).probs)


# --- names ---------------------------------------------------------------------------


def test_top_level_is_user_names():
    for name in ("Pass", "Done", "Running", "Finished", "Stat", "FitStat", "static", "pytree"):
        assert not hasattr(lk, name), name

    for name in ("fs", "dist", "outputs", "grid_search", "DEFAULT", "Belief"):
        assert not hasattr(lk, name), name

    assert lk.distributed.__name__ == "laplacekit.distributed"


def test_builtin_modules_have_all():
    for mod in (
        lk.curvature,
        lk.likelihoods,
        lk.priors,
        lk.predictive,
        lk.subsets,
        lk.objectives,
        lk.data,
        lk.tuning,
    ):
        assert "Any" not in mod.__all__
        assert all(hasattr(mod, n) for n in mod.__all__)


def test_curvature_names():
    for name in (
        "DiagGGN",
        "LowRankGGN",
        "DenseGGN",
        "DenseHessian",
        "KFAC",
    ):
        assert hasattr(lk, name), name

    for old in (
        "Diag",
        "EmpiricalFisher" + "Diag",
        "SampledFisher" + "Diag",
        "Low" + "Rank",
        "DiagEF",
        "DiagMCFisher",
        "DenseEF",
    ):
        assert not hasattr(lk, old), old

    assert "seed" not in inspect.signature(lk.LowRankGGN).parameters


def test_learned_sigma_matches_the_closed_form_evidence():
    """`lk.Gaussian()` tunes sigma with the prior; the optimum is that of the exact evidence."""

    model, x, y = _linear_regression()
    post = lk.laplace(
        model, [(x, y)], lk.Gaussian(), hessian_approx=lk.DenseGGN(), subset=lk.subsets.all
    )
    lams = torch.logspace(-2, 2, 41, dtype=torch.float64)
    sigmas = torch.linspace(0.2, 0.45, 51, dtype=torch.float64)
    ev = torch.stack(
        [torch.stack([_closed_form_log_evidence(x, y, a, s, None) for s in sigmas]) for a in lams]
    )
    best = sigmas[ev.max(0).values.argmax()]

    assert float(post.likelihood.sigma) == pytest.approx(float(best), abs=0.01)


def test_sigma_init_starts_the_learned_sigma():
    model, x, y = _linear_regression()
    post = lk.laplace(model, [(x, y)], lk.Gaussian(), subset=lk.subsets.all, hyper_tuner=None)
    one_step = lk.MarginalLikelihood(n_steps=1, sigma_init=5.0)

    started = float(lk.tune(post, one_step).likelihood.sigma)
    from_one = float(lk.tune(post, lk.MarginalLikelihood(n_steps=1)).likelihood.sigma)

    # One step from 5 and one step from 1 land in different places.
    assert started != pytest.approx(from_one, rel=1e-3)

    fixed = lk.laplace(
        model, [(x, y)], lk.Gaussian(sigma=0.3), subset=lk.subsets.all, hyper_tuner=None
    )

    with pytest.raises(ValueError, match=r"sigma_init.*lk\.Gaussian\(\)"):
        lk.tune(fixed, lk.MarginalLikelihood(sigma_init=2.0))


def test_tune_takes_no_none():
    x, y = _data()
    post = lk.laplace(_model(), _batches(x, y), lk.Categorical(), hyper_tuner=None)

    with pytest.raises(TypeError, match="HyperTuner"):
        lk.tune(post, None)  # ty: ignore[invalid-argument-type]

    assert "/" not in str(inspect.signature(lk.tune))


def test_print_says_when_nothing_was_tuned_and_indents_a_posterior_prior():
    x, y = _data()
    first = lk.laplace(_model(), _batches(x, y), lk.Categorical(), hessian_approx=lk.DiagGGN())
    second = lk.laplace(
        _model(), _batches(x, y), lk.Categorical(), hessian_approx=lk.DiagGGN(), prior=first
    )
    lines = repr(second).splitlines()

    assert "  hyper_tuner    = None (nothing to tune)" in lines
    prior = lines.index(next(line for line in lines if line.startswith("  prior ")))

    assert lines[prior + 1].startswith(" " * 19)


def test_float64_model_tunes_in_float64_under_a_float32_default():
    model, x, y = _linear_regression()
    post = lk.laplace(model, [(x, y)], lk.Gaussian(), subset=lk.subsets.all)

    assert torch.get_default_dtype() == torch.float32
    assert post.likelihood.sigma.dtype == torch.float64
    assert post.prior.prior_precision.dtype == torch.float64


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"strategy": "jvp"}, '"forward-mode", "reverse-mode" or None'),
        ({"n_samples": "auto"}, r'int \(at least 2\), "exact" or None'),
        ({"n_samples": 1}, "n_samples >= 2"),
        ({"chunk_size": 0}, "chunk_size"),
    ],
)
def test_output_options_are_validated(kwargs, match):
    x, y = _data()
    post = lk.laplace(_model(), _batches(x, y), lk.Categorical(), hyper_tuner=None)

    with pytest.raises(ValueError, match=match):
        lk.predict_outputs(post, x, **kwargs)

    with pytest.raises(ValueError, match=match):
        lk.predict(post, x, method=lk.Linearized(**kwargs))


def test_tune_rewrites_the_printed_hyper_tuner():
    model, x, y = _linear_regression()
    post = lk.laplace(model, [(x, y)], lk.Gaussian(), subset=lk.subsets.all, hyper_tuner=None)
    tuner = lk.MarginalLikelihood(n_steps=3)
    grid = lk.GridSearch(lk.objectives.predictive_nll([(x, y)]), values=[0.1, 1.0])

    assert dict(lk.tune(post, tuner).provenance)["hyper_tuner"] == repr(tuner)
    assert dict(grid(post).provenance)["hyper_tuner"] == repr(grid)
    assert repr(tuner) in str(lk.tune(post, tuner))


@pytest.mark.parametrize("value", [0.0, -1.0, math.inf, math.nan])
def test_sigma_init_is_positive_and_finite(value):
    with pytest.raises(ValueError, match="sigma_init"):
        lk.MarginalLikelihood(sigma_init=value)

    with pytest.raises(ValueError, match="sigma_init"):
        lk.Minimize(lk.objectives.predictive_nll([]), sigma_init=value)


def test_a_float_sigma_stays_exact_in_a_float64_fit():
    model, x, y = _linear_regression()

    assert lk.Gaussian(0.3).sigma.item() == 0.3

    post = lk.laplace(model, [(x, y)], lk.Gaussian(0.3), subset=lk.subsets.all, hyper_tuner=None)
    want = lk.laplace(
        model,
        [(x, y)],
        lk.Gaussian(torch.tensor(0.3, dtype=torch.float64)),
        subset=lk.subsets.all,
        hyper_tuner=None,
    )

    assert post.likelihood.sigma.dtype == torch.float64
    assert post.likelihood.sigma.item() == 0.3
    torch.testing.assert_close(
        post.curvature.blocks["<root>"].mat, want.curvature.blocks["<root>"].mat, rtol=0, atol=0
    )


def test_a_float_sigma_takes_the_fit_dtype_in_float32():
    torch.manual_seed(0)
    model = nn.Linear(2, 1)
    x, y = torch.randn(8, 2), torch.randn(8, 1)
    post = lk.laplace(model, [(x, y)], lk.Gaussian(0.3), subset=lk.subsets.all, hyper_tuner=None)

    assert post.likelihood.sigma.dtype == torch.float32
    assert post.curvature.blocks["<root>"].mat.dtype == torch.float32


def test_kfac_refuses_an_unknown_fisher():
    with pytest.raises(ValueError, match="fisher"):
        lk.KFAC(fisher="bogus")  # ty: ignore[invalid-argument-type]


def test_exact_hint_follows_what_the_likelihood_can_do():
    log_only = lk.likelihoods.from_log_prob(lambda f, y: -((f - y) ** 2).sum(-1))

    with pytest.raises(
        lk.CapabilityError, match=r'has none; .*fisher="empirical" or observed=True$'
    ):
        core.check_likelihood(lk.DiagGGN().requires, log_only, "DiagGGN()")

    class Sampled:
        def log_prob(self, f, y, /):
            return -((f - y) ** 2).sum(-1)

        def sample(self, f, /, *, generator=None):
            return f

    with pytest.raises(
        lk.CapabilityError, match=r'has none; pass fisher="mc" for the Monte Carlo Fisher$'
    ):
        core.check_likelihood(lk.DiagGGN().requires, Sampled(), "DiagGGN()")
