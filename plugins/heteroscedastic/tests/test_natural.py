"""`NaturalGaussian` and `NaturalHead`: the Gaussian in natural parameters (Immer et al. 2023).

The parity with laplace-torch is in `tests/parity/test_parity_plugins.py`.
"""

import math

import pytest
import torch
from laplacekit_heteroscedastic import NaturalGaussian, NaturalHead
from torch import nn
from torch.distributions import Normal
from torch.func import hessian

import laplacekit as lk
from laplacekit.core import (
    Batch,
    CapabilityError,
    HasOutputCheck,
    StructureError,
    observed_output_hessian,
)
from laplacekit.testing import check_curvature

DT = torch.float64


def _eta(n: int, k: int, seed: int = 0) -> torch.Tensor:
    """Natural parameters [n, 2K] with η₂ < 0."""

    g = torch.Generator().manual_seed(seed)
    eta1 = torch.randn(n, k, generator=g, dtype=DT)
    eta2 = -0.5 * torch.randn(n, k, generator=g, dtype=DT).exp()

    return torch.cat([eta1, eta2], -1)


def _moments(f: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    eta1, eta2 = f.chunk(2, -1)

    return -eta1 / (2 * eta2), -1 / (2 * eta2)


def _model(k: int = 1, seed: int = 0) -> nn.Sequential:
    torch.manual_seed(seed)

    return nn.Sequential(nn.Linear(2, 8), nn.Tanh(), nn.Linear(8, 2 * k), NaturalHead()).to(DT)


def _data(n: int = 40, k: int = 1, seed: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)

    return torch.randn(n, 2, generator=g, dtype=DT), torch.randn(n, k, generator=g, dtype=DT)


class _OneBatch:
    """Yields the first batch; touching any later batch fails the test."""

    def __init__(self, batches: list) -> None:
        self.batches = batches
        self.pulled = 0

    def __iter__(self):
        for b in self.batches:
            self.pulled += 1

            if self.pulled > 1:
                raise AssertionError("data read past the probe's batch")

            yield b


@pytest.mark.parametrize("k", [1, 3])
def test_natural_output_hessian_matches_autograd(k):
    f = _eta(5, k)
    y = torch.randn(5, k, dtype=DT)
    lik = NaturalGaussian()
    hess = lik.output_hessian(f)
    dense = hess.dense()

    # An exponential family in its natural parameters: the observed Hessian at any y.
    torch.testing.assert_close(dense, observed_output_hessian(lik, f, y).dense())

    # An independent reference: the Hessian of -log N(y | μ, σ) through torch.func.
    def nll(fi: torch.Tensor, yi: torch.Tensor) -> torch.Tensor:
        mu, var = _moments(fi)

        return -Normal(mu, var.sqrt()).log_prob(yi).sum()

    want = torch.stack([hessian(nll)(f[i], y[i]) for i in range(5)])
    torch.testing.assert_close(dense, want)

    # sqrt_mv gives L with L Lᵀ = H; mv and diagonal agree with dense.
    basis = torch.eye(2 * k, dtype=DT)
    cols = torch.stack([hess.sqrt_mv(basis[j].expand(5, -1)) for j in range(2 * k)], -1)
    torch.testing.assert_close(cols @ cols.transpose(1, 2), want)

    v = torch.randn(5, 2 * k, dtype=DT)
    torch.testing.assert_close(hess.mv(v), torch.einsum("nij,nj->ni", want, v))
    torch.testing.assert_close(hess.diagonal(), torch.diagonal(want, dim1=1, dim2=2))


def test_natural_hessian_keeps_position_axes():
    """Outputs [N, T, 2K] give per-position blocks; dense is over the flattened event."""

    f = _eta(6, 2).reshape(3, 2, 4)
    hess = NaturalGaussian().output_hessian(f)
    dense = hess.dense()

    assert dense.shape == (3, 8, 8)

    v = torch.randn(3, 2, 4, dtype=DT)
    torch.testing.assert_close(
        hess.mv(v).reshape(3, 8), torch.einsum("nij,nj->ni", dense, v.reshape(3, 8))
    )
    torch.testing.assert_close(dense[:, :4, 4:], torch.zeros(3, 4, 4, dtype=DT))


def test_natural_dense_refuses_large_events():
    f = _eta(1, lk.core.MAX_DENSE_EVENT // 2 + 1)

    with pytest.raises(StructureError, match="too large"):
        NaturalGaussian().output_hessian(f).dense()


@pytest.mark.parametrize("k", [1, 3])
def test_natural_log_prob_matches_normal(k):
    f = _eta(7, k)
    y = torch.randn(7, k, dtype=DT)
    mu, var = _moments(f)
    want = Normal(mu, var.sqrt()).log_prob(y).sum(-1)
    lik = NaturalGaussian()

    torch.testing.assert_close(lik.log_prob(f, y), want)

    d = lik.distribution(f)
    torch.testing.assert_close(d.mean, mu)
    torch.testing.assert_close(d.stddev, var.sqrt())


def test_natural_log_prob_is_per_position():
    f = _eta(6, 2).reshape(3, 2, 4)
    y = torch.randn(3, 2, 2, dtype=DT)
    mu, var = _moments(f)

    got = NaturalGaussian().log_prob(f, y)

    assert got.shape == (3, 2)
    torch.testing.assert_close(got, Normal(mu, var.sqrt()).log_prob(y).sum(-1))


def test_odd_outputs_and_mismatched_targets_raise():
    lik = NaturalGaussian()
    odd = torch.zeros(4, 3, dtype=DT)

    with pytest.raises(StructureError, match="even"):
        lik.log_prob(odd, torch.zeros(4, 1, dtype=DT))

    with pytest.raises(StructureError, match="even"):
        lik.output_hessian(odd)

    with pytest.raises(StructureError, match="even"):
        lik.distribution(odd)

    with pytest.raises(StructureError, match="targets"):
        lik.log_prob(_eta(4, 1), torch.zeros(4, 2, dtype=DT))


def test_odd_outputs_raise_before_data():
    """The probe runs `log_prob` on the first batch, so the shape error comes before the fit."""

    x, y = _data()
    model = nn.Sequential(nn.Linear(2, 3)).to(DT)
    ld = _OneBatch([(x[:20], y[:20]), (x[20:], y[20:])])

    with pytest.raises(StructureError, match="even"):
        lk.laplace(
            model, ld, likelihood=NaturalGaussian(), hessian_approx=lk.DiagGGN(), hyper_tuner=None
        )

    assert ld.pulled == 1


def test_natural_checks_its_outputs():
    lik = NaturalGaussian()

    assert isinstance(lik, HasOutputCheck)
    lik.check_outputs(_eta(4, 2))

    for bad in (torch.zeros(4, 3, dtype=DT), torch.zeros(4, dtype=DT)):
        with pytest.raises(StructureError, match="even"):
            lik.check_outputs(bad)


@pytest.mark.parametrize(
    "approx",
    [lk.DiagGGN(), lk.DenseGGN(), None],
    ids=["Diag", "DenseGGN", "default"],
)
@pytest.mark.parametrize("targets", [True, False], ids=["targets", "no-targets"])
def test_odd_outputs_raise_before_a_second_batch(approx, targets):
    """The probe's forward meets the odd width, with or without targets. (KFAC refuses the
    likelihood itself before any data; see below.)"""

    x, y = _data()
    model = nn.Sequential(nn.Linear(2, 3)).to(DT)
    batches = [(x[:20], y[:20]), (x[20:], y[20:])] if targets else [Batch(x[:20]), Batch(x[20:])]
    ld = _OneBatch(batches)

    with pytest.raises(StructureError, match="even"):
        lk.laplace(model, ld, likelihood=NaturalGaussian(), hessian_approx=approx, hyper_tuner=None)

    assert ld.pulled == 1


@pytest.mark.parametrize("targets", [True, False], ids=["targets", "no-targets"])
def test_odd_outputs_raise_on_the_by_hand_probe(targets):
    x, y = _data()
    fn = lk.core.ModelFn.from_module(nn.Sequential(nn.Linear(2, 3)).to(DT))
    batch = Batch(x, y) if targets else Batch(x)

    for params in (None, {}):
        with pytest.raises(StructureError, match="even"):
            lk.core.probe(fn, NaturalGaussian(), batch, params=params)


def test_natural_kfac_refused_before_data():
    x, y = _data()
    ld = _OneBatch([(x[:20], y[:20]), (x[20:], y[20:])])
    lik = NaturalGaussian()

    with pytest.raises(CapabilityError, match=r"got NaturalGaussian; use DiagGGN or DenseGGN"):
        lk.laplace(_model(), ld, likelihood=lik, hessian_approx=lk.KFAC(), hyper_tuner=None)

    assert ld.pulled == 1
    assert not lk.KFAC.supports(lik)


def test_default_never_kfac():
    """A layer too large for the dense GGN gets KFAC under a Gaussian, never here."""

    model = nn.Sequential(nn.Linear(2, 100), nn.Tanh(), nn.Linear(100, 100), nn.Tanh())
    model.append(nn.Linear(100, 2)).append(NaturalHead()).to(DT)
    x, y = _data()
    fn = lk.core.ModelFn.from_module(model)
    caps = lk.core.probe(fn, NaturalGaussian(), Batch(x, y))
    params = {k: v for k, v in model.named_parameters() if k.startswith("2.")}

    gaussian = lk.curvature.default(params, caps, likelihood=lk.Gaussian())
    natural = lk.curvature.default(params, caps, likelihood=NaturalGaussian())

    assert gaussian.by_block(["2"]) == {"2": lk.KFAC()}
    assert natural.by_block(["2"]) == {"2": lk.DiagGGN()}

    post = lk.laplace(
        model, [(x, y)], likelihood=NaturalGaussian(), subset=params, hyper_tuner=None
    )

    assert "KFAC" not in dict(post.provenance)["hessian_approx"]


def test_positive_eta2_names_natural_head():
    f = torch.tensor([[0.3, -1.0], [0.2, 0.5], [0.1, 0.0]], dtype=DT)
    lik = NaturalGaussian()
    msg = r"NaturalGaussian needs η₂ < 0; 2 entries are not; end the model with NaturalHead\(\)"

    with pytest.raises(ValueError, match=msg):
        lik.output_hessian(f)

    with pytest.raises(ValueError, match=msg):
        lik.distribution(f)

    nan = torch.tensor([[0.3, math.nan]], dtype=DT)

    with pytest.raises(ValueError, match="1 entries are not"):
        lik.output_hessian(nan)


def _plain_head() -> nn.Sequential:
    """A model ending in a plain `Linear`: η₂ = 1 > 0 for every input."""

    lin = nn.Linear(2, 2).to(DT)

    with torch.no_grad():
        lin.weight.zero_()
        lin.bias.copy_(torch.tensor([0.0, 1.0]))

    return nn.Sequential(lin)


def test_natural_checks_eta2_in_its_outputs():
    lik = NaturalGaussian()
    msg = r"η₂ < 0; 2 entries are not; end the model with NaturalHead\(\)"

    with pytest.raises(ValueError, match=msg):
        lik.check_outputs(torch.tensor([[0.3, 1.0], [0.2, 0.0]], dtype=DT))


@pytest.mark.parametrize(
    "approx",
    [lk.DiagGGN(fisher="empirical"), lk.DenseGGN(fisher="empirical"), lk.DiagGGN()],
    ids=["EFDiag", "DenseEF", "Diag"],
)
def test_positive_eta2_raises_before_data(approx):
    """The empirical Fisher differentiates `log_prob`, which does not check η₂, so without the
    probe's check the posterior would be NaN."""

    x, y = _data()
    ld = _OneBatch([(x[:20], y[:20]), (x[20:], y[20:])])

    with pytest.raises(ValueError, match=r"η₂ < 0.*NaturalHead"):
        lk.laplace(
            _plain_head(), ld, likelihood=NaturalGaussian(), hessian_approx=approx, hyper_tuner=None
        )

    assert ld.pulled == 1


def test_natural_head_exp():
    h = torch.randn(4, 6, dtype=DT)
    h1, h2 = h.chunk(2, -1)

    torch.testing.assert_close(NaturalHead()(h), torch.cat([h1, -0.5 * h2.exp()], -1))


def test_softplus_head_strictly_negative():
    h = torch.tensor([[0.3, -100.0], [-0.2, 2.0]], dtype=DT)
    eta = NaturalHead(link="softplus")(h)
    sp = nn.functional.softplus(h[:, 1:])

    assert bool((eta[:, 1] < 0).all())
    torch.testing.assert_close(eta, torch.cat([h[:, :1], -0.5 * (sp + 1e-8)], -1))


def test_natural_head_refuses_an_unknown_link():
    with pytest.raises(ValueError, match="relu"):
        NaturalHead(link="relu")  # ty: ignore[invalid-argument-type]  # the runtime check


def test_check_curvature_diag():
    model = _model(k=2)
    x, y = _data(k=2)
    params = {k: v.detach() for k, v in model.named_parameters()}
    batches = [Batch(x[:20], y[:20]), Batch(x[20:], y[20:])]

    check_curvature(lk.DiagGGN(), model, NaturalGaussian(), params, batches)
