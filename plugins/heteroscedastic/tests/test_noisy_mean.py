"""`MeanVarianceGaussian`, `moments` and the `NoisyMean` predictive.

The parity with laplace-torch is in `tests/parity/test_parity_plugins.py`.
"""

import pytest
import torch
from laplacekit_heteroscedastic import (
    MeanVarianceGaussian,
    Moments,
    NaturalGaussian,
    NaturalHead,
    NoisyMean,
    moments,
)
from torch import nn
from torch.distributions import Normal
from torch.func import functional_call, jacrev

import laplacekit as lk
from laplacekit.core import (
    Batch,
    CapabilityError,
    HasOutputCheck,
    StructureError,
    observed_output_hessian,
)
from laplacekit.testing import check_link

DT = torch.float64
PREC = 1.0


def _data(n: int = 30, seed: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)

    return torch.randn(n, 2, generator=g, dtype=DT), torch.randn(n, 1, generator=g, dtype=DT)


def _natural_model(seed: int = 0) -> nn.Sequential:
    torch.manual_seed(seed)

    return nn.Sequential(nn.Linear(2, 5), nn.Tanh(), nn.Linear(5, 2), NaturalHead()).to(DT)


class _MeanVarianceNet(nn.Module):
    """A shared body, a mean head and a softplus variance head; returns `[mean, var]`."""

    def __init__(self) -> None:
        super().__init__()

        self.body = nn.Sequential(nn.Linear(2, 5), nn.Tanh())
        self.mean = nn.Linear(5, 1)
        self.var = nn.Sequential(nn.Linear(5, 1), nn.Softplus())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.body(x)

        return torch.cat([self.mean(h), self.var(h) + 1e-3], -1)


def _mean_variance_model(seed: int = 0) -> _MeanVarianceNet:
    torch.manual_seed(seed)

    return _MeanVarianceNet().to(DT)


def _fit(model, lik, hessian_approx=None, subset=lk.subsets.all) -> lk.LaplacePosterior:
    x, y = _data()

    return lk.laplace(
        model,
        [(x, y)],
        likelihood=lik,
        hessian_approx=lk.DenseGGN(joint=True) if hessian_approx is None else hessian_approx,
        subset=subset,
        prior=lk.priors.Isotropic(torch.tensor(PREC, dtype=DT)),
        hyper_tuner=None,
    )


def _flat(model: nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.detach() for k, v in model.named_parameters()}


def _jacobian(fn, params: dict[str, torch.Tensor]) -> torch.Tensor:
    """d fn / d θ as [N, P], with θ flattened in `params` order."""

    jac = jacrev(fn)(params)
    n = next(iter(jac.values())).shape[0]

    return torch.cat([jac[k].reshape(n, -1) for k in params], 1)


def _natural_moments(f: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    eta1, eta2 = f.chunk(2, -1)

    return -eta1 / (2 * eta2), -1 / (2 * eta2)


# --- MeanVarianceGaussian -------------------------------------------------------------------------


def test_mean_variance_log_prob_and_distribution():
    g = torch.Generator().manual_seed(0)
    mean = torch.randn(6, 3, generator=g, dtype=DT)
    var = torch.rand(6, 3, generator=g, dtype=DT) + 0.1
    var[0, 0] = 1e-9  # below eps
    f = torch.cat([mean, var], -1)
    y = torch.randn(6, 3, generator=g, dtype=DT)
    lik = MeanVarianceGaussian()
    clamped = var.clamp_min(1e-6)

    want = Normal(mean, clamped.sqrt()).log_prob(y).sum(-1)
    torch.testing.assert_close(lik.log_prob(f, y), want)

    d = lik.distribution(f)
    torch.testing.assert_close(d.mean, mean)
    torch.testing.assert_close(d.variance, clamped)


def test_mean_variance_log_prob_holds_the_variance_fixed():
    """The Laplace is over the mean: log_prob has no gradient in the variance."""

    f = torch.tensor([[0.3, 0.7]], dtype=DT, requires_grad=True)
    y = torch.tensor([[1.1]], dtype=DT)
    (grad,) = torch.autograd.grad(MeanVarianceGaussian().log_prob(f, y).sum(), f)

    torch.testing.assert_close(grad, torch.tensor([[(1.1 - 0.3) / 0.7, 0.0]], dtype=DT))


def test_mean_variance_output_hessian_matches_autograd():
    g = torch.Generator().manual_seed(2)
    f = torch.cat(
        [torch.randn(4, 2, generator=g, dtype=DT), torch.rand(4, 2, generator=g, dtype=DT) + 0.2],
        -1,
    )
    y = torch.randn(4, 2, generator=g, dtype=DT)
    lik = MeanVarianceGaussian()
    hess = lik.output_hessian(f)
    want = observed_output_hessian(lik, f, y).dense()

    torch.testing.assert_close(hess.dense(), want)
    torch.testing.assert_close(hess.diagonal(), want.diagonal(dim1=-2, dim2=-1))

    v = torch.randn(4, 4, generator=g, dtype=DT)
    torch.testing.assert_close(hess.mv(v), torch.einsum("nij,nj->ni", want, v))
    torch.testing.assert_close(hess.sqrt_mv(hess.sqrt_mv(v)), hess.mv(v))


def test_mean_variance_refuses_odd_outputs_and_mismatched_targets():
    lik = MeanVarianceGaussian()

    with pytest.raises(StructureError, match="even"):
        lik.log_prob(torch.ones(4, 3, dtype=DT), torch.zeros(4, 1, dtype=DT))

    with pytest.raises(StructureError, match="targets"):
        lik.log_prob(torch.ones(4, 2, dtype=DT), torch.zeros(4, 2, dtype=DT))


def test_mean_variance_checks_its_outputs():
    lik = MeanVarianceGaussian()

    assert isinstance(lik, HasOutputCheck)
    lik.check_outputs(torch.ones(4, 2, dtype=DT))

    for bad in (torch.ones(4, 3, dtype=DT), torch.ones(4, dtype=DT)):
        with pytest.raises(StructureError, match="MeanVarianceGaussian needs .* even"):
            lik.check_outputs(bad)


class _Pulled:
    def __init__(self, batches: list) -> None:
        self.batches, self.pulled = batches, 0

    def __iter__(self):
        for b in self.batches:
            self.pulled += 1
            yield b


@pytest.mark.parametrize(
    "approx",
    [lk.DiagGGN(), lk.DenseGGN(), None],
    ids=["Diag", "DenseGGN", "default"],
)
@pytest.mark.parametrize("targets", [True, False], ids=["targets", "no-targets"])
def test_mean_variance_odd_outputs_raise_before_a_second_batch(approx, targets):
    x, y = _data()
    model = nn.Linear(x.shape[1], 3).to(DT)
    batches = [(x[:15], y[:15]), (x[15:], y[15:])] if targets else [Batch(x[:15]), Batch(x[15:])]
    ld = _Pulled(batches)

    with pytest.raises(StructureError, match="MeanVarianceGaussian needs .* even"):
        lk.laplace(
            model,
            ld,
            likelihood=MeanVarianceGaussian(),
            hessian_approx=approx,
            subset=lk.subsets.all,
            hyper_tuner=None,
        )

    assert ld.pulled == 1


def test_mean_variance_variance_head_only_equals_prior():
    """A fit over the variance head alone gets no curvature from the data."""

    model = _mean_variance_model()
    params = {k: v for k, v in model.named_parameters() if k.startswith("var.")}

    for approx in (lk.DenseGGN(joint=True), lk.DenseGGN(fisher="empirical", joint=True)):
        post = _fit(model, MeanVarianceGaussian(), approx, params)
        prec = post.precision().op.mat

        torch.testing.assert_close(prec, PREC * torch.eye(prec.shape[0], dtype=DT))


def test_mean_variance_ef_matches_mean_only():
    """DenseEF is Σ g gᵀ with g the gradient of log N(y | mean(x; θ), var(x; θ̂)) in θ."""

    model = _mean_variance_model()
    x, y = _data()
    post = _fit(model, MeanVarianceGaussian(), lk.DenseGGN(fisher="empirical", joint=True))
    params = _flat(model)

    with torch.no_grad():
        var = model(x)[:, 1:]

    def one(p, xi, yi, vi):
        mean = functional_call(model, p, (xi.unsqueeze(0),))[0, :1]

        return Normal(mean, vi.sqrt()).log_prob(yi).sum()

    grads = torch.func.vmap(jacrev(one), in_dims=(None, 0, 0, 0))(params, x, y, var)
    g = torch.cat([grads[k].reshape(x.shape[0], -1) for k in params], 1)
    want = g.T @ g + PREC * torch.eye(g.shape[1], dtype=DT)

    torch.testing.assert_close(post.precision().op.mat, want)


# --- moments ------------------------------------------------------------------------------


def test_natural_epistemic_equals_linearized_mean():
    """gᵀ Σₖ g on the output covariance equals linearizing μ(x; θ) in θ under the posterior.

    The reference builds the posterior by hand: the GGN Σ Jᵀ ∇²A(η) J plus the prior, from
    `jacrev` and the closed-form Hessian of a Gaussian in its natural parameters.
    """

    model = _natural_model()
    x, _ = _data()
    x_test = torch.randn(7, 2, generator=torch.Generator().manual_seed(5), dtype=DT)
    post = _fit(model, NaturalGaussian())
    params = _flat(model)

    def f(p, inputs):
        return functional_call(model, p, (inputs,))

    jac = jacrev(f)(params, x)  # [N, 2, ...]
    j = torch.cat([jac[k].reshape(x.shape[0], 2, -1) for k in params], -1)
    mu, var = _natural_moments(f(params, x))
    a, b, c = var, 2 * mu * var, 4 * mu**2 * var + 2 * var**2
    h = torch.stack([torch.cat([a, b], -1), torch.cat([b, c], -1)], -2)  # [N, 2, 2]
    ggn = torch.einsum("nip,nij,njq->pq", j, h, j)
    cov = torch.linalg.inv(ggn + PREC * torch.eye(ggn.shape[0], dtype=DT))

    def mean(p):
        return _natural_moments(f(p, x_test))[0][:, 0]

    jm = _jacobian(mean, params)  # [M, P]
    want = torch.einsum("mp,pq,mq->m", jm, cov, jm)[:, None]

    m = moments(post.likelihood, lk.predict_outputs(post, x_test))
    mu_t, var_t = _natural_moments(f(params, x_test))

    assert isinstance(m, Moments)
    torch.testing.assert_close(m.epistemic, want, rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(m.mean, mu_t)
    torch.testing.assert_close(m.aleatoric, var_t)


def test_natural_moments_per_position_and_output():
    """Outputs [N, T, 2K]: each (position, k) reads its own 2×2 block on (k, K + k)."""

    g = torch.Generator().manual_seed(3)
    n, t, k = 3, 2, 2
    eta1 = torch.randn(n, t, k, generator=g, dtype=DT)
    eta2 = -0.5 * torch.randn(n, t, k, generator=g, dtype=DT).exp()
    f = torch.cat([eta1, eta2], -1)
    e = t * 2 * k
    root = torch.randn(n, e, e, generator=g, dtype=DT)
    blocks = root @ root.mT
    out = lk.core.OutputGaussian(f, lk.core.PerInputCov(blocks, (t, 2 * k)))

    mu, var = _natural_moments(f)
    grad = torch.zeros(n, t, k, t, 2 * k, dtype=DT)  # dμ[t, k] / df[t', j], by hand

    for ti in range(t):
        for ki in range(k):
            grad[:, ti, ki, ti, ki] = var[:, ti, ki]
            grad[:, ti, ki, ti, k + ki] = 2 * mu[:, ti, ki] * var[:, ti, ki]

    grad = grad.reshape(n, t * k, e)
    want = torch.einsum("nae,nef,naf->na", grad, blocks, grad).reshape(n, t, k)

    torch.testing.assert_close(moments(NaturalGaussian(), out).epistemic, want)


def test_mean_variance_moments():
    g = torch.Generator().manual_seed(4)
    mean = torch.randn(5, 2, generator=g, dtype=DT)
    var = torch.rand(5, 2, generator=g, dtype=DT)
    var[1, 1] = 0.0
    root = torch.randn(5, 4, 4, generator=g, dtype=DT)
    blocks = root @ root.mT
    out = lk.core.OutputGaussian(torch.cat([mean, var], -1), lk.core.PerInputCov(blocks, (4,)))

    m = moments(MeanVarianceGaussian(), out)

    torch.testing.assert_close(m.mean, mean)
    torch.testing.assert_close(m.aleatoric, var.clamp_min(1e-6))
    torch.testing.assert_close(m.epistemic, blocks.diagonal(dim1=-2, dim2=-1)[:, :2])


def test_moments_refuses_other_likelihoods():
    out = lk.core.OutputGaussian(
        torch.zeros(2, 2, dtype=DT), lk.core.PerInputCov(torch.zeros(2, 2, 2, dtype=DT), (2,))
    )

    with pytest.raises(CapabilityError, match="Gaussian"):
        moments(lk.Gaussian(), out)


# --- the predictive -----------------------------------------------------------------------


@pytest.mark.parametrize("which", ["natural", "mean_variance"])
def test_predict_default_is_noisy_mean(which):
    if which == "natural":
        model, lik = _natural_model(), NaturalGaussian()
    else:
        model, lik = _mean_variance_model(), MeanVarianceGaussian()

    post = _fit(model, lik)
    x_test = torch.randn(7, 2, generator=torch.Generator().manual_seed(5), dtype=DT)
    m = moments(lik, lk.predict_outputs(post, x_test))

    assert lik.default_link(None) == NoisyMean()
    assert lik.default_link(64) == NoisyMean()

    d = lk.predict(post, x_test)

    assert isinstance(d, Normal)
    torch.testing.assert_close(d.mean, m.mean)
    torch.testing.assert_close(d.variance, m.epistemic + m.aleatoric)


@pytest.mark.parametrize("which", ["natural", "mean_variance"])
def test_weight_samples_average(which):
    """The moments of explicit `post.sample` draws, pushed through the model by hand: the
    spread of μ_s over S − 1, plus the mean of σ²_s."""

    if which == "natural":
        model, lik = _natural_model(), NaturalGaussian()
    else:
        model, lik = _mean_variance_model(), MeanVarianceGaussian()

    post = _fit(model, lik)
    x_test = torch.randn(7, 2, generator=torch.Generator().manual_seed(5), dtype=DT)
    # WeightSamples draws from a generator spawned from the one it is given.
    draws = post.sample(20, generator=lk.core.spawn_generator(torch.Generator().manual_seed(3)))

    mus, variances = [], []

    for i in range(20):
        f = functional_call(model, {k: v[i] for k, v in draws.items()}, (x_test,))

        if which == "natural":
            mu, var = _natural_moments(f)
        else:
            mu, var = f[:, :1], f[:, 1:].clamp_min(1e-6)

        mus.append(mu)
        variances.append(var)

    mu_s, var_s = torch.stack(mus), torch.stack(variances)
    mean = mu_s.mean(0)
    var = ((mu_s - mean) ** 2).sum(0) / 19 + var_s.mean(0)

    # The method spawns its own generator from this one, as `lk.predict` hands it on.
    d = lk.WeightSamples(n_samples=20)(post, x_test, generator=torch.Generator().manual_seed(3))

    assert isinstance(d, Normal)
    torch.testing.assert_close(d.mean, mean)
    torch.testing.assert_close(d.variance, var)


def test_check_link():
    x_test = torch.randn(7, 2, generator=torch.Generator().manual_seed(5), dtype=DT)

    for model, lik in (
        (_natural_model(), NaturalGaussian()),
        (_mean_variance_model(), MeanVarianceGaussian()),
    ):
        post = _fit(model, lik)
        out = lk.predict_outputs(post, x_test)

        assert NoisyMean().supports(lik, out.mean)

        check_link(NoisyMean(), lik, out)

        # A likelihood it doesn't support is refused.
        assert not NoisyMean().supports(lk.Gaussian(), out.mean)

        check_link(NoisyMean(), lk.Gaussian(), out)


def test_mean_variance_kfac_refused():
    assert not lk.KFAC.supports(MeanVarianceGaussian())
