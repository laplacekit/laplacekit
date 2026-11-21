"""Hyper tuners: `MarginalLikelihood`, `Minimize`, `GridSearch` and `lk.tune`."""

import copy
import dataclasses
import json
from pathlib import Path

import pytest
import torch
from torch import Tensor, nn
from torch.distributions import Independent, Normal

import laplacekit as lk
from laplacekit.core import (
    Batch,
    BlockDiag,
    Capabilities,
    Dense,
    LaplacePosterior,
    Layout,
    ModelFn,
    StructureError,
    detach_leaves,
    hyperparameters,
    pytree,
    replace_leaves,
    tree_leaves,
)
from laplacekit.likelihoods import Gaussian
from laplacekit.objectives import neg_log_marglik, predictive_nll
from laplacekit.predictive import Linearized
from laplacekit.priors import Diagonal, Isotropic, Layerwise

F64 = torch.float64
SIGMA = 0.5


def _blr(prec: float = 1.0, mu_shift: float = 0.0, data_fit: bool = True):
    """Bayesian linear regression with a fixed sigma, applied at fit time."""

    g = torch.Generator().manual_seed(1)
    x = torch.randn(40, 3, generator=g, dtype=F64)
    y = x @ torch.tensor([1.0, -2.0, 0.5], dtype=F64) + SIGMA * torch.randn(
        40, generator=g, dtype=F64
    )
    h = x.T @ x / SIGMA**2
    w = torch.linalg.solve(h + prec * torch.eye(3, dtype=F64), x.T @ y / SIGMA**2) + mu_shift
    params = {"fc.weight": w}
    model = ModelFn(lambda p, xb: xb @ p["fc.weight"])
    lik = Gaussian(sigma=torch.tensor(SIGMA, dtype=F64))
    fit = lik.fit_stat(model(params, x), y, None)
    post = LaplacePosterior(
        mean=params,
        curvature=Dense(h),
        prior=Isotropic(torch.tensor(prec, dtype=F64)),
        n_data=torch.tensor(1.0, dtype=F64),
        model=model,
        likelihood=lik,
        layout=Layout.by_module(params),
        caps=Capabilities(vjp=True, vmap=True),
        data_fit=(fit,) if data_fit else (),
    )

    return post, x, y


def _tensors(tree) -> list[torch.Tensor]:
    """The tensor leaves (a prior without a mean has a `None` leaf)."""

    return [t for t in tree_leaves(tree) if isinstance(t, torch.Tensor)]


def _large_blr(dtype: torch.dtype, n: int = 200_000):
    """A last-layer-like posterior whose marglik is of order n, built in float64 and cast.

    Its sigma is learned, so it is applied after the fit and tunes with the prior.
    """

    g = torch.Generator().manual_seed(1)
    x = torch.randn(n, 3, generator=g, dtype=F64)
    y = x @ torch.tensor([0.3, -0.2, 0.1], dtype=F64) + 1.1 * torch.randn(n, generator=g, dtype=F64)
    w = torch.linalg.solve(x.T @ x + torch.eye(3, dtype=F64), x.T @ y)
    data_fit = torch.stack([((y - x @ w) ** 2).sum(), torch.tensor(float(n), dtype=F64)])

    params = {"fc.weight": w.to(dtype)}

    return LaplacePosterior(
        mean=params,
        curvature=Dense((x.T @ x / n).to(dtype)),
        prior=Isotropic(torch.tensor(1.0, dtype=dtype)),
        n_data=torch.tensor(float(n), dtype=dtype),
        model=ModelFn(lambda p, xb: xb @ p["fc.weight"]),
        likelihood=copy.replace(Gaussian(), sigma=torch.as_tensor(torch.tensor(1.0, dtype=dtype))),
        layout=Layout.by_module(params),
        caps=Capabilities(vjp=True, vmap=True),
        data_fit=(data_fit.to(dtype),),
    )


def _grid_optimum(post, objective) -> tuple[float, float]:
    """The best prior precision on a fine log grid, and its objective."""

    values = torch.logspace(-5, 5, 4001, dtype=F64)
    scores = [
        float(objective(dataclasses.replace(post, prior=Isotropic(v))).detach()) for v in values
    ]
    best = min(range(len(values)), key=scores.__getitem__)

    return float(values[best]), scores[best]


def test_neg_log_marglik_formula() -> None:
    post, x, y = _blr(2.0, mu_shift=0.3)

    w = post.mean["fc.weight"]
    ll = torch.distributions.Normal(x @ w, SIGMA).log_prob(y).sum()
    scatter = 2.0 * (w @ w)
    prec = x.T @ x / SIGMA**2 + 2.0 * torch.eye(3, dtype=F64)
    ldr = torch.logdet(prec) - torch.logdet(2.0 * torch.eye(3, dtype=F64))
    expected = -(ll - 0.5 * (ldr + scatter))

    torch.testing.assert_close(post.precision().mat, prec)

    assert torch.allclose(neg_log_marglik(post), expected)


def test_marginal_likelihood_needs_targets() -> None:
    post, _, _ = _blr(data_fit=False)

    with pytest.raises(ValueError, match="data_fit"):
        lk.MarginalLikelihood()(post)


def test_marginal_likelihood_changes_only_the_hyperparameters() -> None:
    post, _, _ = _blr()
    before = [t.clone() for t in _tensors(post)]
    out = lk.tune(post, lk.MarginalLikelihood(n_steps=5))

    assert out is not post

    for a, b in zip(before, _tensors(post), strict=True):
        assert torch.equal(a, b)

    changed = [
        i
        for i, (a, b) in enumerate(zip(_tensors(post), _tensors(out), strict=True))
        if not torch.equal(a, b)
    ]

    # The sigma is fixed, so only the prior precision moves.
    assert [_tensors(out)[i] for i in changed] == [out.prior.prior_precision]
    assert out.prior.prior_precision > 0


def test_minimize_matches_grid_search() -> None:
    post, _, _ = _blr(mu_shift=0.2)
    values = list(torch.logspace(-3, 3, 400, dtype=F64))
    best = lk.tune(post, lk.GridSearch(neg_log_marglik, values=values))
    tuned = lk.tune(post, lk.Minimize(neg_log_marglik, n_steps=300, lr=0.1))
    ratio = (tuned.prior.prior_precision / best.prior.prior_precision).item()

    assert abs(ratio - 1) < 0.05
    assert best.prior.prior_precision is not post.prior.prior_precision


def test_grid_search_needs_a_prior_precision() -> None:
    post, _, _ = _blr()
    prior_post = dataclasses.replace(post, prior=post)

    with pytest.raises(StructureError, match="no hyperparameters"):
        lk.tune(prior_post, lk.GridSearch(neg_log_marglik, values=[1.0]))

    with pytest.raises(StructureError, match="not a TunablePrior"):
        lk.GridSearch(neg_log_marglik, values=[1.0]).check(post, post.likelihood, Batch(None))


def test_grid_search_needs_values() -> None:
    with pytest.raises(ValueError, match="at least one value"):
        lk.GridSearch(neg_log_marglik, values=[])


def test_tune_refuses_what_is_not_a_tuner() -> None:
    post, _, _ = _blr()

    with pytest.raises(TypeError, match=r"lk\.MarginalLikelihood\(\)"):
        lk.tune(post, neg_log_marglik)  # ty: ignore[invalid-argument-type]

    with pytest.raises(TypeError, match="instance"):
        lk.tune(post, lk.MarginalLikelihood)  # ty: ignore[invalid-argument-type]


def test_predictive_nll_value() -> None:
    post, x, y = _blr()
    nll = predictive_nll([Batch(x, y)])(post)
    d = Linearized()(post, x)

    assert torch.allclose(nll, -d.log_prob(y).mean())
    assert nll.requires_grad is False


def test_tune_does_not_touch_grads_of_live_parameters() -> None:
    post, _, _ = _blr()
    live = {"fc.weight": torch.nn.Parameter(post.mean["fc.weight"].clone())}
    live_post = dataclasses.replace(post, mean=live)
    lk.tune(live_post, lk.MarginalLikelihood(n_steps=3))
    lk.tune(live_post, lk.MarginalLikelihood(n_steps=3))  # twice: no graph reuse

    assert live["fc.weight"].grad is None


def test_priors_refuse_non_positive_precision() -> None:
    with pytest.raises(ValueError, match="positive"):
        Isotropic(-1.0)

    with pytest.raises(ValueError, match="positive"):
        Layerwise({"fc": 0.0})


def test_layerwise_mapping_tunes() -> None:
    post, _, _ = _blr()
    prior = Layerwise({"fc": torch.tensor(1.0, dtype=F64)})
    blocked = dataclasses.replace(
        post, prior=prior, curvature=BlockDiag({"fc": Dense(post.curvature.mat)})
    )
    out = lk.tune(blocked, lk.MarginalLikelihood(n_steps=20))

    assert float(out.prior.prior_precision["fc"]) != 1.0


def test_grid_search_returns_a_copy() -> None:
    post, _, _ = _blr()
    out = lk.tune(post, lk.GridSearch(neg_log_marglik, values=[2.0]))

    assert out.prior.prior_precision is not post.prior.prior_precision
    assert float(out.prior.prior_precision) == 2.0
    assert float(post.prior.prior_precision) == 1.0


def _dense_scatter(post: LaplacePosterior, p0: torch.Tensor) -> torch.Tensor:
    mu0 = post.prior.location(post.mean)
    delta = post.mean["fc.weight"] - mu0["fc.weight"]

    return delta @ p0 @ delta


def test_prior_scatter_diagonal_prior() -> None:
    post, _, _ = _blr(mu_shift=0.3)
    d = torch.tensor([1.0, 2.0, 3.0], dtype=F64)
    post = dataclasses.replace(post, prior=Diagonal(d))

    assert torch.allclose(post.prior_scatter(), _dense_scatter(post, torch.diag(d)))


def test_prior_scatter_with_nonzero_prior_mean() -> None:
    post, _, _ = _blr(mu_shift=0.3)
    mu0 = {"fc.weight": torch.tensor([0.5, -1.0, 2.0], dtype=F64)}
    prior = dataclasses.replace(
        post, mean=mu0, curvature=Dense(torch.diag(torch.tensor([2.0, 4.0, 5.0], dtype=F64)))
    )
    informed = dataclasses.replace(post, prior=prior)
    p0 = informed.prior.precision().mat
    ref = _dense_scatter(informed, p0)

    assert torch.allclose(informed.prior_scatter(), ref)
    assert float(ref) != float(post.prior_scatter())


def test_predictive_nll_averages_over_batches() -> None:
    post, x, y = _blr()
    batches = [Batch(x[:30], y[:30]), Batch(x[30:], y[30:])]  # sizes 30 and 10
    nll = predictive_nll(batches)(post)
    direct = -sum(Linearized()(post, b.inputs).log_prob(b.targets).sum() for b in batches) / 40

    assert torch.allclose(nll, direct)


@pytest.mark.parametrize("start", [5e-4, 1.0, 1e4])
def test_tune_converges_from_a_far_start(start) -> None:
    post, _, _ = _blr(start)
    want_prec, want_score = _grid_optimum(post, neg_log_marglik)
    out = lk.tune(post, lk.MarginalLikelihood())
    got = float(out.prior.prior_precision)

    assert got == pytest.approx(want_prec, rel=0.01)
    assert float(neg_log_marglik(out).detach()) <= want_score + 1e-6


def test_tune_keeps_location() -> None:
    post, _, _ = _blr(2.0)
    mu0 = {"fc.weight": torch.tensor([0.5, -1.0, 0.2], dtype=F64)}
    located = dataclasses.replace(post, prior=Isotropic(torch.tensor(1.0, dtype=F64), mean=mu0))
    out = lk.tune(located, lk.MarginalLikelihood(n_steps=20))

    assert float(out.prior.prior_precision) != 1.0
    assert out.prior.mean.keys() == mu0.keys()
    assert torch.equal(out.prior.mean["fc.weight"], mu0["fc.weight"])

    # The tuned precision is the marglik optimum for the shifted prior, not the zero-mean one.
    zero = lk.tune(post, lk.MarginalLikelihood(n_steps=20))

    assert not torch.allclose(out.prior.prior_precision, zero.prior.prior_precision)


def test_tune_float32_objective_matches_float64() -> None:
    """A float32 marglik of order 1e5 still moves the prior when tuned with sigma."""

    post32, post64 = _large_blr(torch.float32), _large_blr(F64)
    out32 = lk.tune(post32, lk.MarginalLikelihood())
    out64 = lk.tune(post64, lk.MarginalLikelihood())

    assert float(out64.prior.prior_precision) == pytest.approx(21.34, rel=0.01)
    assert float(out32.prior.prior_precision) == pytest.approx(
        float(out64.prior.prior_precision), rel=0.01
    )
    assert float(out32.likelihood.sigma) == pytest.approx(float(out64.likelihood.sigma), rel=1e-4)

    # At the tuned sigma, a fine float64 grid over the prior agrees.
    lik = copy.replace(Gaussian(), sigma=torch.as_tensor(out64.likelihood.sigma))
    want, _ = _grid_optimum(dataclasses.replace(post64, likelihood=lik), neg_log_marglik)

    assert float(out32.prior.prior_precision) == pytest.approx(want, rel=0.01)


@pytest.mark.parametrize("kind", ["diagonal", "layerwise"])
def test_tune_vector_precision_on_a_float32_posterior(kind) -> None:
    """A dimensioned precision leaf keeps the float32 operator usable during the tune."""

    def tuned(dtype):
        post = _large_blr(dtype, n=1000)
        ones = torch.ones(3, dtype=dtype)

        if kind == "diagonal":
            post = dataclasses.replace(post, prior=Diagonal(ones))
        else:
            post = dataclasses.replace(
                post, prior=Layerwise({"fc": ones}), curvature=BlockDiag({"fc": post.curvature})
            )

        return lk.tune(post, lk.MarginalLikelihood())

    got = tree_leaves(tuned(torch.float32).prior.prior_precision)
    want = tree_leaves(tuned(F64).prior.prior_precision)

    assert all(t.dtype == torch.float32 for t in got)
    assert not torch.allclose(want[0], torch.ones_like(want[0]))

    for a, b in zip(got, want, strict=True):
        torch.testing.assert_close(a.double(), b, rtol=0.01, atol=0.0)


def test_tune_keeps_leaf_dtypes() -> None:
    post = _large_blr(torch.float32, n=1000)
    out = lk.tune(post, lk.MarginalLikelihood(n_steps=5))
    best = lk.tune(post, lk.GridSearch(neg_log_marglik, values=torch.logspace(-2, 2, 9)))

    for tuned in (out, best):
        before, after = _tensors(post), _tensors(tuned)

        assert [(t.dtype, t.device) for t in after] == [(t.dtype, t.device) for t in before]

    assert out.prior.prior_precision.dtype == out.likelihood.sigma.dtype == torch.float32
    assert best.prior.prior_precision.dtype == torch.float32

    # Leaves that were not tuned keep their exact values.
    assert torch.equal(out.curvature.mat, post.curvature.mat)
    assert torch.equal(best.mean["fc.weight"], post.mean["fc.weight"])
    assert torch.equal(best.likelihood.sigma, post.likelihood.sigma)


def test_tune_predictive_nll_on_a_float32_posterior() -> None:
    """The model still sees float32 parameters and inputs while the prior is tuned."""

    post = _large_blr(torch.float32, n=1000)
    post = dataclasses.replace(
        post, n_data=torch.tensor(2.0), likelihood=Gaussian(sigma=torch.tensor(1.0))
    )
    g = torch.Generator().manual_seed(2)
    x = torch.randn(50, 3, generator=g)
    y = x @ torch.tensor([0.3, -0.2, 0.1]) + 1.1 * torch.randn(50, generator=g)
    out = lk.tune(post, lk.Minimize(predictive_nll([Batch(x, y)]), n_steps=5))

    assert out.prior.prior_precision.dtype == torch.float32
    assert float(out.prior.prior_precision) != 1.0


def test_hyperparameters_are_named_leaves() -> None:
    post = _large_blr(torch.float32, n=10)
    found = hyperparameters(post)

    assert list(found) == ["prior_precision", "sigma"]
    assert found["prior_precision"] is post.prior.prior_precision
    assert found["sigma"] is post.likelihood.sigma

    blocked = dataclasses.replace(post, prior=Layerwise({"fc": 1.0}))

    assert list(hyperparameters(blocked)) == ["prior_precision['fc']", "sigma"]


def test_detach_and_replace_leaves() -> None:
    post, _, _ = _blr()
    live = dataclasses.replace(post, prior=Isotropic(torch.tensor(2.0, dtype=F64).requires_grad_()))

    frozen = detach_leaves(live)

    assert not any(isinstance(t, torch.Tensor) and t.requires_grad for t in tree_leaves(frozen))
    assert live.prior.prior_precision.requires_grad

    new = torch.tensor(5.0, dtype=F64)
    out = replace_leaves(frozen, (frozen.prior.prior_precision,), (new,))

    assert out.prior.prior_precision is new
    assert out.mean["fc.weight"] is frozen.mean["fc.weight"]
    assert frozen.prior.prior_precision == 2.0


@pytree
class _Noisy:
    """A Gaussian whose noise `t` is a hyperparameter applied at fit time: no `unit()`."""

    t: Tensor

    def distribution(self, f: Tensor) -> Independent:
        return Independent(Normal(f, self.t), 1)

    def log_prob(self, f: Tensor, y: Tensor) -> Tensor:
        return self.distribution(f).log_prob(y)

    def tunable(self) -> dict[str, Tensor]:
        return {"t": self.t}


def _noisy_post():
    g = torch.Generator().manual_seed(0)
    x = torch.randn(20, 2, generator=g, dtype=F64)
    y = x.sum(1, keepdim=True) + 0.1 * torch.randn(20, 1, generator=g, dtype=F64)
    model = nn.Linear(2, 1).to(F64)
    lik = _Noisy(torch.tensor(0.5, dtype=F64))

    return lk.laplace(
        model, [Batch(x, y)], lik, hessian_approx=lk.DiagGGN(fisher="mc"), hyper_tuner=None
    )


class _Counter:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, p) -> Tensor:
        self.calls += 1

        return torch.tensor(0.0, dtype=F64)


def test_tuners_refuse_a_fit_time_likelihood_hyperparameter() -> None:
    post = _noisy_post()
    objective = _Counter()

    with pytest.raises(
        StructureError, match=r"^Minimize can\'t change t \(likelihood\.t\).*_Noisy applies it"
    ):
        lk.tune(post, lk.Minimize(objective))

    with pytest.raises(StructureError, match=r"^MarginalLikelihood can't change t"):
        lk.tune(post, lk.MarginalLikelihood())

    assert objective.calls == 0

    # The grid tunes the prior precision only, so the fit-time noise stays as it is.
    out = lk.tune(post, lk.GridSearch(objective, values=[0.1, 1.0]))

    assert objective.calls == 2
    assert out.likelihood.t is not post.likelihood.t
    assert torch.equal(out.likelihood.t, post.likelihood.t)


def test_marginal_likelihood_tunes_a_learned_sigma_to_laplace_torch() -> None:
    """A learned sigma tunes jointly with the prior, to laplace-torch's optimum."""

    ref_path = Path(__file__).parent / "parity" / "reference.json"
    ref = json.loads(ref_path.read_text())
    want = ref["joint_tune"]["all_diag"]

    model = nn.Sequential(nn.Linear(1, 50), nn.Tanh(), nn.Linear(50, 1)).to(F64)
    model.load_state_dict({k: torch.tensor(v, dtype=F64) for k, v in ref["state_dict"].items()})
    x, y = torch.tensor(ref["X"], dtype=F64), torch.tensor(ref["Y"], dtype=F64)
    post = lk.laplace(
        model,
        [Batch(x, y)],
        copy.replace(Gaussian(), sigma=torch.as_tensor(torch.tensor(0.3, dtype=F64))),
        hessian_approx=lk.DiagGGN(),
        subset=lk.subsets.all,
        prior=Isotropic(torch.tensor(1.0, dtype=F64)),
        hyper_tuner=None,
    )

    out = lk.tune(post)

    assert float(out.prior.prior_precision) == pytest.approx(want["prior_prec"], rel=1e-4)
    assert float(out.likelihood.sigma) == pytest.approx(want["sigma"], rel=1e-4)


def test_minimize_names_grid_search_for_detached_objective() -> None:
    post, _, _ = _blr()

    def detached(p) -> Tensor:
        return neg_log_marglik(p).detach()

    with pytest.raises(ValueError, match=r"no gradient with respect to.*goes with lk\.GridSearch"):
        lk.tune(post, lk.Minimize(detached))


def test_dict_likelihood_checked_per_leaf() -> None:
    post = _noisy_post()
    lik = {
        "late": copy.replace(Gaussian(), sigma=torch.as_tensor(torch.tensor(0.5, dtype=F64))),
        "fit": _Noisy(torch.tensor(0.2, dtype=F64)),
    }
    multi = dataclasses.replace(post, likelihood=lik)

    assert list(hyperparameters(multi)) == ["prior_precision", "late.sigma", "fit.t"]

    objective = _Counter()

    # The fit-time leaf of one entry is refused, named by its key.
    with pytest.raises(StructureError, match=r"likelihood\['fit'\]\.t"):
        lk.tune(multi, lk.Minimize(objective))

    assert objective.calls == 0


@pytree
class _Shifted:
    """A late `sigma` next to a `shift` that `fit_stat` applies at fit time."""

    sigma: Tensor | None
    shift: Tensor
    tune_shift: bool = False

    def unit(self) -> "_Shifted":
        return self if self.sigma is None else _Shifted(None, self.shift)

    def curvature_scale(self) -> Tensor:
        return torch.ones((), dtype=F64) if self.sigma is None else self.sigma**-2

    def fit_stat(self, f: Tensor, y: Tensor, weights: Tensor | None) -> Tensor:
        sq = ((y - f - self.shift) ** 2).sum()

        return torch.stack([sq, torch.tensor(float(f.numel()), dtype=F64)])

    def log_lik(self, stat: Tensor) -> Tensor:
        var = torch.ones((), dtype=F64) if self.sigma is None else self.sigma**2

        return -0.5 * (stat[1] * torch.log(2 * torch.pi * var) + stat[0] / var)

    def distribution(self, f: Tensor) -> Independent:
        s = torch.ones_like(f) if self.sigma is None else self.sigma.expand_as(f)

        return Independent(Normal(f + self.shift, s), 1)

    def log_prob(self, f: Tensor, y: Tensor) -> Tensor:
        return self.distribution(f).log_prob(y)

    def tunable(self) -> dict[str, Tensor]:
        found = {} if self.sigma is None else {"sigma": self.sigma}

        return {**found, "shift": self.shift} if self.tune_shift else found


def _shifted_post(tune_shift: bool):
    g = torch.Generator().manual_seed(0)
    x = torch.randn(30, 2, generator=g, dtype=F64)
    y = x.sum(1, keepdim=True) + 3.0 + 0.1 * torch.randn(30, 1, generator=g, dtype=F64)
    model = nn.Linear(2, 1).to(F64)
    lik = _Shifted(torch.tensor(0.5, dtype=F64), torch.tensor(1.0, dtype=F64), tune_shift)

    return lk.laplace(
        model, [Batch(x, y)], lik, hessian_approx=lk.DiagGGN(fisher="mc"), hyper_tuner=None
    )


def test_fit_time_leaf_next_to_late_sigma_is_refused() -> None:
    post = _shifted_post(tune_shift=True)
    objective = _Counter()

    with pytest.raises(StructureError, match=r"^Minimize can't change shift .*likelihood\.shift"):
        lk.tune(post, lk.Minimize(objective))

    assert objective.calls == 0


def test_late_sigma_next_to_fit_time_leaf_still_tunes() -> None:
    """Only a fit-time leaf is refused; the late sigma of the same likelihood tunes."""

    post = _shifted_post(tune_shift=False)
    out = lk.tune(post, lk.MarginalLikelihood())

    assert float(out.likelihood.sigma) != 0.5
    assert torch.equal(out.likelihood.shift, post.likelihood.shift)
