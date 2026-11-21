import copy
import dataclasses
from collections.abc import Mapping

import pytest
import torch
from torch import nn
from torch.distributions import MultivariateNormal, Normal

import laplacekit as lk
from laplacekit.core import (
    BlockDiag,
    Dense,
    Diag,
    Joint,
    KronEigenbasis,
    LowRankPlusDiag,
    StructureError,
)
from laplacekit.priors import Diagonal, Isotropic, Layerwise


def test_isotropic_adds_to_diag() -> None:
    d = torch.tensor([1.0, 2.0, 3.0])
    out = Isotropic(torch.tensor(2.0)).add_precision_to(Diag(d))

    assert torch.equal(out.diag, d + 2)


def test_per_block_adds_per_block() -> None:
    op = BlockDiag({"a": Diag(torch.ones(2)), "b": Diag(torch.ones(3))})
    out = Layerwise({"a": torch.tensor(1.0), "b": torch.tensor(3.0)}).add_precision_to(op)

    assert torch.equal(out.blocks["a"].diag, torch.full((2,), 2.0))
    assert torch.equal(out.blocks["b"].diag, torch.full((3,), 4.0))


def test_per_block_on_bare_diag_raises() -> None:
    with pytest.raises(StructureError, match="BlockDiag"):
        Layerwise({"a": torch.tensor(1.0)}).add_precision_to(Diag(torch.ones(2)))


def test_per_block_misspelled_key_raises() -> None:
    op = BlockDiag({"a": Diag(torch.ones(2))})

    with pytest.raises(StructureError, match="missing"):
        Layerwise({"aa": torch.tensor(1.0)}).add_precision_to(op)


def test_diagonal_on_blockdiag_splits_by_block() -> None:
    a, h = torch.tensor([1.0, 2.0]), torch.eye(3) * 2
    op = BlockDiag({"a": Diag(a), "b": Dense(h)})
    d = torch.tensor([10.0, 20.0, 30.0, 40.0, 50.0])
    out = Diagonal(d).add_precision_to(op)

    assert torch.equal(out.blocks["a"].diag, a + d[:2])
    assert torch.equal(out.blocks["b"].mat, h + torch.diag(d[2:]))

    with pytest.raises(StructureError, match="entries"):
        Diagonal(d[:3]).add_precision_to(op)


def test_isotropic_log_prob() -> None:
    p = {"a.weight": torch.randn(3, 2), "a.bias": torch.randn(3)}
    got = Isotropic(torch.tensor(2.0)).log_prob(p)
    ref = sum(Normal(0.0, 1 / 2**0.5).log_prob(x).sum() for x in p.values())

    assert torch.allclose(got, ref)


def test_location_and_cov_mv() -> None:
    p = {"a.weight": torch.randn(3, 2), "a.bias": torch.randn(3)}
    prec = torch.tensor(4.0)

    assert all((z == 0).all() for z in Isotropic(prec).location(p).values())
    out = Isotropic(prec).cov_mv(p)

    assert torch.allclose(out["a.bias"], p["a.bias"] / 4)


def test_diagonal_and_per_block_log_prob_and_cov() -> None:
    p = {"a.weight": torch.randn(2), "a.bias": torch.randn(1), "b.weight": torch.randn(2)}
    d = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    flat = torch.cat(list(p.values()))

    assert torch.allclose(Diagonal(d).log_prob(p), Normal(0.0, d.rsqrt()).log_prob(flat).sum())
    cov = Diagonal(d).cov_mv(p)

    assert torch.allclose(torch.cat(list(cov.values())), flat / d)
    pb = Layerwise({"a": torch.tensor(2.0), "b": torch.tensor(5.0)})
    ref = (
        Normal(0.0, 2.0**-0.5).log_prob(torch.cat([p["a.weight"], p["a.bias"]])).sum()
        + Normal(0.0, 5.0**-0.5).log_prob(p["b.weight"]).sum()
    )

    assert torch.allclose(pb.log_prob(p), ref)
    assert torch.allclose(pb.cov_mv(p)["b.weight"], p["b.weight"] / 5)


@pytest.mark.parametrize("prior", [Isotropic(torch.tensor(2.0)), Layerwise({})])
def test_log_prob_of_no_params_is_zero(prior) -> None:
    assert float(prior.log_prob({})) == 0.0


F64 = torch.float64
SIGMA = 0.5


class _TwoBlocks(nn.Module):
    """y = A x[:, :2] + B x[:, 2:]: a linear model over two modules."""

    def __init__(self) -> None:
        super().__init__()
        self.a = nn.Linear(2, 1, bias=False)
        self.b = nn.Linear(2, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.a(x[:, :2]) + self.b(x[:, 2:])


def _linear_gaussian():
    """Features whose two blocks are orthogonal, so the per-module GGN is exact."""

    g = torch.Generator().manual_seed(0)
    q, _ = torch.linalg.qr(torch.randn(30, 4, generator=g, dtype=F64))
    x = q * torch.tensor([3.0, 2.0, 4.0, 1.5], dtype=F64)
    x = torch.cat([x[:, :2] @ torch.tensor([[1.0, 0.5], [0.0, 1.0]], dtype=F64), x[:, 2:]], 1)
    y = x @ torch.tensor([1.0, -2.0, 0.5, 0.3], dtype=F64) + SIGMA * torch.randn(
        30, generator=g, dtype=F64
    )

    return x, y[:, None]


def _map_model(x, y, p0, mu0):
    """The model at the MAP of N(mu0, p0^-1) and the Gaussian likelihood."""

    prec = x.T @ x / SIGMA**2 + p0
    w = torch.linalg.solve(prec, x.T @ y[:, 0] / SIGMA**2 + p0 @ mu0)
    model = _TwoBlocks().to(F64)

    with torch.no_grad():
        model.a.weight.copy_(w[None, :2])
        model.b.weight.copy_(w[None, 2:])

    return model


def _log_evidence(x, y, p0, mu0):
    """log N(y; X mu0, sigma^2 I + X p0^-1 X^T)."""

    cov = SIGMA**2 * torch.eye(len(y), dtype=F64) + x @ torch.linalg.solve(p0, x.T)

    return MultivariateNormal(x @ mu0, cov).log_prob(y[:, 0])


def _mean(mu0):
    return {"a.weight": mu0[None, :2], "b.weight": mu0[None, 2:]}


MU0 = torch.tensor([0.7, -1.2, 0.4, 2.0], dtype=F64)


@pytest.mark.parametrize(
    ("make", "p0"),
    [
        (lambda m: Isotropic(torch.tensor(2.0, dtype=F64), mean=m), 2.0 * torch.eye(4, dtype=F64)),
        (
            lambda m: Diagonal(torch.tensor([1.0, 2.0, 3.0, 0.5], dtype=F64), mean=m),
            torch.diag(torch.tensor([1.0, 2.0, 3.0, 0.5], dtype=F64)),
        ),
        (
            lambda m: Layerwise(
                {"a": torch.tensor(1.5, dtype=F64), "b": torch.tensor(4.0, dtype=F64)}, mean=m
            ),
            torch.diag(torch.tensor([1.5, 1.5, 4.0, 4.0], dtype=F64)),
        ),
    ],
    ids=["isotropic", "diagonal", "per_block"],
)
def test_prior_mean_marglik(make, p0) -> None:
    x, y = _linear_gaussian()
    model = _map_model(x, y, p0, MU0)
    post = lk.laplace(
        model,
        [(x[:15], y[:15]), (x[15:], y[15:])],
        likelihood=copy.replace(
            lk.Gaussian(), sigma=torch.as_tensor(torch.tensor(SIGMA, dtype=F64))
        ),
        hessian_approx=lk.DenseGGN(),
        subset=dict(model.named_parameters()),
        prior=make(_mean(MU0)),
        hyper_tuner=None,
    )

    torch.testing.assert_close(post.log_marglik(), _log_evidence(x, y, p0, MU0))


def test_location_log_prob_is_shifted() -> None:
    p = {"a.weight": torch.randn(3, 2), "a.bias": torch.randn(3)}
    mu = {"a.weight": torch.randn(3, 2), "a.bias": torch.randn(3)}
    prec = torch.tensor(2.0)
    ref = sum(Normal(mu[k], 1 / 2**0.5).log_prob(p[k]).sum() for k in p)

    assert torch.allclose(Isotropic(prec, mean=mu).log_prob(p), ref)
    assert Isotropic(prec, mean=mu).location(p) is mu

    d = torch.arange(1.0, 10.0)
    flat_p = torch.cat([x.reshape(-1) for x in p.values()])
    flat_mu = torch.cat([x.reshape(-1) for x in mu.values()])

    assert torch.allclose(
        Diagonal(d, mean=mu).log_prob(p), Normal(flat_mu, d.rsqrt()).log_prob(flat_p).sum()
    )
    assert torch.allclose(Layerwise({"a": prec}, mean=mu).log_prob(p), ref)


@pytest.mark.parametrize(
    "mean",
    [
        {"a.weight": torch.zeros(1, 2, dtype=F64)},
        {"a.weight": torch.zeros(1, 2, dtype=F64), "b.weight": torch.zeros(2, 1, dtype=F64)},
        {"a.weight": torch.zeros(1, 2, dtype=F64), "c.weight": torch.zeros(1, 2, dtype=F64)},
        {"a.weight": torch.zeros(1, 2), "b.weight": torch.zeros(1, 2)},
        {
            "a.weight": torch.zeros(1, 2, dtype=F64, device="meta"),
            "b.weight": torch.zeros(1, 2, dtype=F64, device="meta"),
        },
    ],
    ids=["missing", "shape", "name", "dtype", "device"],
)
def test_location_mismatch_raises(mean) -> None:
    x, y = _linear_gaussian()
    model = _TwoBlocks().to(F64)
    ld = _Counting([(x[:15], y[:15]), (x[15:], y[15:])])

    with pytest.raises(StructureError, match="prior mean"):
        lk.laplace(
            model,
            ld,
            likelihood=copy.replace(
                lk.Gaussian(), sigma=torch.as_tensor(torch.tensor(SIGMA, dtype=F64))
            ),
            subset=dict(model.named_parameters()),
            prior=Isotropic(torch.tensor(1.0, dtype=F64), mean=mean),
        )

    assert ld.pulled == 1  # the probe's batch only

    fit = lk.core.fit_curvature(
        lk.core.ModelFn.from_module(model),
        copy.replace(lk.Gaussian(), sigma=torch.as_tensor(torch.tensor(SIGMA, dtype=F64))),
        dict(model.named_parameters()),
        lk.DenseGGN(),
    )
    curvature = BlockDiag(
        {"a": Dense(torch.eye(2, dtype=F64)), "b": Dense(torch.eye(2, dtype=F64))}
    )

    with pytest.raises(StructureError, match="prior mean"):
        fit.posterior(
            curvature,
            prior=Diagonal(torch.ones(4, dtype=F64), mean=mean),
            n_data=torch.tensor(30.0, dtype=F64),
        )


class _Counting:
    def __init__(self, items):
        self.items = items
        self.pulled = 0

    def __iter__(self):
        for it in self.items:
            self.pulled += 1
            yield it


def test_diagonal_and_perblock_priors_on_joint() -> None:
    g = torch.Generator().manual_seed(0)
    a = torch.randn(5, 5, generator=g, dtype=torch.float64)
    h = a @ a.T
    op = Joint(Dense(h), (("a", 2), ("b", 3)))

    d = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float64)
    out = Diagonal(d).add_precision_to(op)

    assert isinstance(out, Joint)
    assert torch.allclose(out.op.mat, h + torch.diag(d))

    pb = {"a": torch.tensor(0.5, dtype=torch.float64), "b": torch.tensor(2.0, dtype=torch.float64)}
    out = Layerwise(pb).add_precision_to(op)
    want = torch.tensor([0.5, 0.5, 2.0, 2.0, 2.0], dtype=torch.float64)

    assert torch.allclose(out.op.mat, h + torch.diag(want))

    with pytest.raises(StructureError, match="Diagonal prior has 4 entries.*BlockDiag or Joint.*5"):
        Diagonal(d[:4]).add_precision_to(op)


def test_per_block_message_names_both_per_block_operators() -> None:
    with pytest.raises(
        StructureError, match=r"per-block operator \(BlockDiag or Joint\), got Dense"
    ):
        Layerwise({"a": torch.tensor(1.0)}).add_precision_to(Dense(torch.eye(2)))


def test_diagonal_prior_on_kfac_refused_even_constant() -> None:
    f64 = torch.float64
    kron = KronEigenbasis(
        torch.eye(3, dtype=f64), torch.eye(2, dtype=f64), torch.ones(2, 3, dtype=f64), True, (2, 2)
    )
    op = BlockDiag({"a": kron})

    diagonal = Diagonal(torch.ones(6, dtype=f64))
    vector_per_block = Layerwise({"a": torch.ones(6, dtype=f64)})

    for prior in (diagonal, vector_per_block):
        with pytest.raises(StructureError) as e:
            prior.add_precision_to(op)

        msg = str(e.value)

        assert "Kronecker-factored curvature (KFAC or EKFAC)" in msg
        assert "lk.priors.Isotropic, or lk.priors.Layerwise with a scalar per block" in msg
        assert "hessian_approx=lk.DiagGGN() or lk.DenseGGN()" in msg

    iso = Isotropic(torch.tensor(2.0, dtype=torch.float64)).add_precision_to(op)
    per = Layerwise({"a": torch.tensor(2.0, dtype=torch.float64)}).add_precision_to(op)

    for out in (iso, per):
        assert torch.equal(out.blocks["a"].evals, torch.full((2, 3), 3.0, dtype=torch.float64))


# --- KronSum: continual learning with KFAC ---------------------------------------------

F64 = torch.float64


def _mlp(seed: int = 0) -> nn.Module:
    torch.manual_seed(seed)

    return nn.Sequential(nn.Linear(2, 4), nn.Tanh(), nn.Linear(4, 3)).to(F64)


def _rows(n: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 2, generator=g, dtype=F64)

    return x, torch.randn(n, 3, generator=g, dtype=F64)


def _kfac(model, x, y, sigma: float, prior) -> lk.LaplacePosterior:
    data = x if isinstance(x, _Unread) else [lk.Batch(x, y)]

    return lk.laplace(
        model,
        data,
        likelihood=copy.replace(
            lk.Gaussian(), sigma=torch.as_tensor(torch.tensor(sigma, dtype=F64))
        ),
        hessian_approx=lk.KFAC(),
        subset=lk.subsets.all,
        prior=prior,
        hyper_tuner=None,
    )


def _iso(prec: float = 1.0) -> Isotropic:
    return Isotropic(torch.tensor(prec, dtype=F64))


def _dense(op) -> torch.Tensor:
    """A `BlockDiag` operator as one matrix, through `mv` on unit vectors."""

    mats = []

    for blk in op.blocks.values():
        n = blk.diagonal().numel()
        mats.append(blk.mv(torch.eye(n, dtype=F64)))

    return torch.block_diag(*mats)


def _layer_factors(model, x) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Per layer, by hand: `A` the per-example average of `[input, 1] [input, 1]^T` and `G`
    the sum over examples of `J^T J`, with `J` the Jacobian of the outputs with respect to
    that layer's pre-activation (the unit Gaussian GGN)."""

    n = x.shape[0]
    w2 = model[2].weight.detach()

    with torch.no_grad():
        h = torch.tanh(model[0](x))

    # d f / d z1 = W2 diag(1 - h^2), one [3, 4] per example; d f / d z2 = I.
    j1 = w2.unsqueeze(0) * (1 - h.square()).unsqueeze(1)
    g1 = torch.einsum("nci,ncj->ij", j1, j1)
    g2 = n * torch.eye(3, dtype=F64)

    out = []

    for inp, g in [(x, g1), (h, g2)]:
        xt = torch.cat([inp, torch.ones(n, 1, dtype=F64)], dim=1)
        out.append((xt.T @ xt / n, g))

    return out


def _layout_order(o: int, i: int) -> torch.Tensor:
    """From the `(out, in + 1)` order of `kron(G, A)` to layout order (weight, then bias)."""

    aug = torch.arange(o * (i + 1)).reshape(o, i + 1)

    return torch.cat([aug[:, :i].reshape(-1), aug[:, i]])


def test_kronsum_matches_explicit_factor_sum() -> None:
    """Two fits with different sigmas: c = N / sigma^2 per fit,
    `A = (c1 A1 + c2 A2) / (c1 + c2)`, `G = G1 / sigma1^2 + G2 / sigma2^2`, plus the first
    fit's prior, built with `torch.kron` from the layers by hand."""

    model = _mlp()
    x1, y1 = _rows(7, 1)
    x2, y2 = _rows(5, 2)
    s1, s2 = 0.5, 0.8

    post_1 = _kfac(model, x1, y1, s1, _iso(2.0))
    post_2 = _kfac(model, x2, y2, s2, lk.priors.KronSum.of(post_1))

    c1, c2 = 7 / s1**2, 5 / s2**2
    blocks = []

    for (a1, g1), (a2, g2) in zip(
        _layer_factors(model, x1), _layer_factors(model, x2), strict=True
    ):
        a = (c1 * a1 + c2 * a2) / (c1 + c2)
        g = g1 / s1**2 + g2 / s2**2
        order = _layout_order(g.shape[0], a.shape[0] - 1)
        blocks.append(torch.kron(g, a)[order][:, order])

    want = torch.block_diag(*blocks) + 2.0 * torch.eye(sum(b.shape[0] for b in blocks), dtype=F64)

    torch.testing.assert_close(_dense(post_2.precision()), want, rtol=1e-10, atol=1e-10)


def test_kronsum_carries_the_first_fit_over() -> None:
    """With one sigma the summed factors are those of one fit on all the rows, and clearly
    not those of a fit on the second part alone."""

    model = _mlp()
    x, y = _rows(12, 3)

    post_1 = _kfac(model, x[:7], y[:7], 0.5, _iso())
    continual = _dense(_kfac(model, x[7:], y[7:], 0.5, lk.priors.KronSum.of(post_1)).precision())
    alone = _dense(_kfac(model, x[7:], y[7:], 0.5, _iso()).precision())
    batch = _dense(_kfac(model, x, y, 0.5, _iso()).precision())

    assert float((continual - alone).norm() / alone.norm()) > 0.3
    torch.testing.assert_close(continual, batch, rtol=1e-10, atol=1e-10)


def test_kronsum_chains_three_fits() -> None:
    model = _mlp()
    x, y = _rows(12, 4)

    post = _kfac(model, x[:4], y[:4], 0.5, _iso())

    for lo, hi in [(4, 9), (9, 12)]:
        post = _kfac(model, x[lo:hi], y[lo:hi], 0.5, lk.priors.KronSum.of(post))

    batch = _kfac(model, x, y, 0.5, _iso())

    torch.testing.assert_close(
        _dense(post.precision()), _dense(batch.precision()), rtol=1e-10, atol=1e-10
    )


def test_kron_sum_of_chain_matches_dense() -> None:
    """Three fits with different sizes and sigmas. Per layer, by Ritter et al.'s rule with
    `c_t = N_t / sigma_t^2`: `G = sum_t G_t / sigma_t^2` (`G_t` summed over examples) and
    `A = sum_t c_t A_t / sum_t c_t`, so the precision is `G kron A` plus the first prior."""

    model = _mlp()
    x, y = _rows(12, 9)
    parts = [(0, 4, 0.5), (4, 9, 0.8), (9, 12, 0.6)]

    post = _kfac(model, x[:4], y[:4], 0.5, _iso(2.0))

    for lo, hi, sigma in parts[1:]:
        post = _kfac(model, x[lo:hi], y[lo:hi], sigma, lk.priors.KronSum.of(post))

    by_fit = [(hi - lo, s, _layer_factors(model, x[lo:hi])) for lo, hi, s in parts]
    blocks = []

    for layer in range(2):
        c = [n / s**2 for n, s, _ in by_fit]
        a = sum(ct * f[layer][0] for ct, (_, _, f) in zip(c, by_fit, strict=True)) / sum(c)
        g = sum(f[layer][1] / s**2 for _, s, f in by_fit)
        order = _layout_order(g.shape[0], a.shape[0] - 1)
        blocks.append(torch.kron(g, a)[order][:, order])

    want = torch.block_diag(*blocks) + 2.0 * torch.eye(sum(b.shape[0] for b in blocks), dtype=F64)

    torch.testing.assert_close(_dense(post.precision()), want, rtol=1e-10, atol=1e-10)


def test_kron_sum_of_eigh_count(monkeypatch: pytest.MonkeyPatch) -> None:
    """The chain is summed as it is built: one `precision()` after five fits decomposes the
    two factors of each layer once, not once per earlier fit."""

    model = _mlp()
    x, y = _rows(15, 10)

    post = _kfac(model, x[:3], y[:3], 0.5, _iso())

    for lo in range(3, 15, 3):
        post = _kfac(model, x[lo : lo + 3], y[lo : lo + 3], 0.5, lk.priors.KronSum.of(post))

    calls = []
    eigh = torch.linalg.eigh

    def counting(*args, **kwargs):
        calls.append(1)

        return eigh(*args, **kwargs)

    monkeypatch.setattr(torch.linalg, "eigh", counting)
    post.precision()

    assert len(calls) == 2 * 2  # two factors, two layers


def _nodes(obj: object) -> list[object]:
    """Every object reachable through dataclass fields, mappings and sequences."""

    out = [obj]

    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        children = [getattr(obj, f.name) for f in dataclasses.fields(obj)]
    elif isinstance(obj, Mapping):
        children = list(obj.values())
    elif isinstance(obj, (tuple, list)):
        children = list(obj)
    else:
        children = []

    for child in children:
        out.extend(_nodes(child))

    return out


def test_kron_sum_holds_no_posterior() -> None:
    model = _mlp()
    x, y = _rows(9, 11)

    post = _kfac(model, x[:3], y[:3], 0.5, _iso())

    for lo in (3, 6):
        post = _kfac(model, x[lo : lo + 3], y[lo : lo + 3], 0.5, lk.priors.KronSum.of(post))

    prior = lk.priors.KronSum.of(post)
    nodes = _nodes(prior)

    assert isinstance(prior.base, Isotropic)
    assert any(isinstance(n, KronEigenbasis) for n in nodes)
    assert not any(isinstance(n, lk.LaplacePosterior) for n in nodes)


def test_kronsum_zero_scale_is_old_precision() -> None:
    """At zero scale the new fit adds nothing: what `log_det_ratio` takes as the prior."""

    model = _mlp()
    x, y = _rows(12, 5)
    post_1 = _kfac(model, x[:7], y[:7], 0.5, _iso(3.0))
    post_2 = _kfac(model, x[7:], y[7:], 0.5, lk.priors.KronSum.of(post_1))

    zero = post_2.curvature.scaled(torch.zeros((), dtype=F64))
    old = _dense(post_2.prior.add_precision_to(zero))

    torch.testing.assert_close(old, _dense(post_1.precision()), rtol=1e-10, atol=1e-10)
    assert torch.isfinite(post_2.log_det_ratio())


def test_kronsum_location_log_prob_and_repr() -> None:
    model = _mlp()
    x, y = _rows(6, 6)
    post_1 = _kfac(model, x, y, 0.5, _iso())
    prior = lk.priors.KronSum.of(post_1)
    shifted = {k: v + 0.1 for k, v in post_1.mean.items()}

    assert prior.location(post_1.mean) is post_1.mean
    torch.testing.assert_close(prior.log_prob(shifted), post_1.log_prob(shifted))
    assert (
        repr(prior)
        == "KronSum(summed=<2 KFAC blocks>, base=Isotropic(prior_precision=1), mean=<4 tensors>)"
    )
    assert not isinstance(prior, lk.core.TunablePrior)
    assert not isinstance(prior, lk.core.PriorWithCovariance)


def _dense_log_prob(prec: torch.Tensor, mean, params) -> torch.Tensor:
    """log N(params; mean, prec^-1) by `torch.distributions`, flat in mapping order."""

    flat_mean = torch.cat([v.reshape(-1) for v in mean.values()])
    flat = torch.cat([params[k].reshape(-1) for k in mean])

    return MultivariateNormal(flat_mean, precision_matrix=prec).log_prob(flat)


def test_kronsum_log_prob_after_two_fits_is_dense() -> None:
    """A prior summing two fits against the dense Gaussian of the summed factors (built as
    in `test_kronsum_matches_explicit_factor_sum`) plus the base prior."""

    model = _mlp()
    x1, y1 = _rows(7, 1)
    x2, y2 = _rows(5, 2)
    s1, s2 = 0.5, 0.8

    post_1 = _kfac(model, x1, y1, s1, _iso(2.0))
    post_2 = _kfac(model, x2, y2, s2, lk.priors.KronSum.of(post_1))
    prior = lk.priors.KronSum.of(post_2)

    c1, c2 = 7 / s1**2, 5 / s2**2
    blocks = []

    for (a1, g1), (a2, g2) in zip(
        _layer_factors(model, x1), _layer_factors(model, x2), strict=True
    ):
        a = (c1 * a1 + c2 * a2) / (c1 + c2)
        g = g1 / s1**2 + g2 / s2**2
        order = _layout_order(g.shape[0], a.shape[0] - 1)
        blocks.append(torch.kron(g, a)[order][:, order])

    prec = torch.block_diag(*blocks) + 2.0 * torch.eye(27, dtype=F64)
    shifted = {k: v + 0.05 * (i + 1) for i, (k, v) in enumerate(post_2.mean.items())}

    torch.testing.assert_close(
        prior.log_prob(shifted), _dense_log_prob(prec, post_2.mean, shifted), rtol=1e-10, atol=1e-10
    )


def test_low_rank_sum_log_prob_after_two_fits_is_dense() -> None:
    model = _mlp()
    x, y = _rows(12, 24)
    post_1 = _low_rank(model, x[:6], y[:6], 0.5, _iso(2.0))
    post_2 = _low_rank(model, x[6:], y[6:], 0.7, lk.priors.LowRankSum.of(post_1))
    prior = lk.priors.LowRankSum.of(post_2)

    summed = _scaled_low_rank(post_1) + _scaled_low_rank(post_2)
    prec = _top_r(summed, 3) + 2.0 * torch.eye(27, dtype=F64)
    shifted = {k: v - 0.03 * (i + 1) for i, (k, v) in enumerate(post_2.mean.items())}

    torch.testing.assert_close(
        prior.log_prob(shifted), _dense_log_prob(prec, post_2.mean, shifted), rtol=1e-9, atol=1e-9
    )


class _Unread:
    """Data that fails the test if the fit reads it."""

    size = 6

    def __call__(self, generator, /):
        raise AssertionError("the data was read")

    def __iter__(self):
        raise AssertionError("the data was read")


def test_kronsum_refuses_a_posterior_without_factors_when_built() -> None:
    """The earlier posterior is checked by `KronSum.of(post)`, before the next fit reads data."""

    model = _mlp()
    x, y = _rows(6, 7)
    lik = copy.replace(lk.Gaussian(), sigma=torch.as_tensor(torch.tensor(0.5, dtype=F64)))
    kfac = _kfac(model, x, y, 0.5, _iso())
    no_factors = BlockDiag(
        {b: copy.replace(blk, factors=()) for b, blk in kfac.curvature.blocks.items()}
    )
    diag = lk.laplace(
        model,
        [lk.Batch(x, y)],
        likelihood=lik,
        hessian_approx=lk.DiagGGN(),
        subset=lk.subsets.all,
        hyper_tuner=None,
    )

    cases = [
        (
            copy.replace(kfac, curvature=no_factors),
            "EKFAC's corrected eigenvalues have no Kronecker",
        ),
        (
            diag,
            r"KronSum needs the posterior's curvature to be a BlockDiag of KronEigenbasis.*Diag",
        ),
        (
            copy.replace(kfac, curvature=no_factors.blocks["0"]),
            r"KronEigenbasis.*got KronEigenbasis",
        ),
    ]

    for post, match in cases:
        with pytest.raises(StructureError, match=match):
            _kfac(model, _Unread(), None, 0.5, lk.priors.KronSum.of(post))


def test_kronsum_refuses_a_new_curvature_without_factors() -> None:
    model = _mlp()
    x, y = _rows(6, 7)
    lik = copy.replace(lk.Gaussian(), sigma=torch.as_tensor(torch.tensor(0.5, dtype=F64)))
    kfac = _kfac(model, x, y, 0.5, _iso())
    no_factors = BlockDiag(
        {b: copy.replace(blk, factors=()) for b, blk in kfac.curvature.blocks.items()}
    )

    with pytest.raises(StructureError, match="EKFAC's corrected eigenvalues have no Kronecker"):
        lk.priors.KronSum.of(kfac).add_precision_to(no_factors)

    # A Diag fit with a KFAC posterior: refused from Diag's template, before the data.
    with pytest.raises(StructureError, match=r"KronSum.*the new curvature.*KronEigenbasis.*Diag"):
        lk.laplace(
            model,
            [lk.Batch(x, y)],
            likelihood=lik,
            hessian_approx=lk.DiagGGN(),
            subset=lk.subsets.all,
            prior=lk.priors.KronSum.of(kfac),
            hyper_tuner=None,
        )

    # Different blocks: a last-layer fit against an all-weights one.
    last = lk.laplace(
        model,
        [lk.Batch(x, y)],
        likelihood=lik,
        hessian_approx=lk.KFAC(),
        prior=_iso(),
        hyper_tuner=None,
    )

    with pytest.raises(StructureError, match="block names differ"):
        lk.priors.KronSum.of(last).add_precision_to(kfac.curvature)

    # Same block names, but the layers have no bias.
    torch.manual_seed(0)
    nobias = nn.Sequential(nn.Linear(2, 4, bias=False), nn.Tanh(), nn.Linear(4, 3, bias=False))
    other = _kfac(nobias.to(F64), x, y, 0.5, _iso())

    with pytest.raises(StructureError, match=r"block '0' differs.*with bias.*without bias"):
        lk.priors.KronSum.of(kfac).add_precision_to(other.curvature)

    # Nothing to average when both fits have zero scale.
    zero = torch.zeros((), dtype=F64)

    with pytest.raises(ValueError, match="both fits have zero scale"):
        lk.priors.KronSum.of(copy.replace(kfac, n_data=zero)).add_precision_to(
            kfac.curvature.scaled(zero)
        )


def test_plain_kfac_prior_still_raises_and_points_to_kronsum() -> None:
    model = _mlp()
    x, y = _rows(6, 8)
    post_1 = _kfac(model, x, y, 0.5, _iso())

    with pytest.raises(StructureError, match=r"KronEigenbasis \+ KronEigenbasis") as e:
        _kfac(model, x, y, 0.5, post_1)

    assert "lk.priors.KronSum.of(post)" in str(e.value)


# --- LowRankSum: LowRankGGN for continual learning, truncated to a fixed rank
# --------------------------------


def _low_rank(model, x, y, sigma: float, prior, rank: int = 3) -> lk.LaplacePosterior:
    data = x if isinstance(x, _Unread) else [lk.Batch(x, y)]

    return lk.laplace(
        model,
        data,
        likelihood=copy.replace(
            lk.Gaussian(), sigma=torch.as_tensor(torch.tensor(sigma, dtype=F64))
        ),
        hessian_approx=lk.LowRankGGN(rank, oversample=3),
        subset=lk.subsets.all,
        prior=prior,
        hyper_tuner=None,
        generator=torch.Generator().manual_seed(0),  # the same sketch in every fit
    )


def _joint_dense(op) -> torch.Tensor:
    """A `Joint` operator as one matrix, through its inner operator on unit vectors."""

    n = op.op.diagonal().numel()

    return op.op.mv(torch.eye(n, dtype=F64))


def _scaled_low_rank(post) -> torch.Tensor:
    """`N / sigma^2 U diag(evals) U^T`, the post's scaled curvature, built densely."""

    sigma = post.likelihood.sigma
    u, e = post.curvature.op.factor, post.curvature.op.evals

    return post.n_data / sigma.square() * (u * e) @ u.T


def _top_r(m: torch.Tensor, r: int) -> torch.Tensor:
    e, v = torch.linalg.eigh(m)

    return (v[:, -r:] * e[-r:]) @ v[:, -r:].T


def test_low_rank_sum_full_rank_is_exact() -> None:
    """With a rank at or above the summed width, the result equals plain `prior=post`."""

    model = _mlp()
    x, y = _rows(12, 21)
    post_1 = _low_rank(model, x[:6], y[:6], 0.5, _iso(2.0))

    plain = _low_rank(model, x[6:], y[6:], 0.7, post_1)
    summed = _low_rank(model, x[6:], y[6:], 0.7, lk.priors.LowRankSum.of(post_1, rank=6))

    want = _joint_dense(plain.precision())

    torch.testing.assert_close(_joint_dense(summed.precision()), want, rtol=1e-10, atol=1e-10)

    # Plain `prior=post` concatenates, so its width grows; it equals the dense sum.
    reference = _scaled_low_rank(post_1) + _scaled_low_rank(plain) + 2.0 * torch.eye(27, dtype=F64)

    assert plain.precision().op.factor.shape[1] == 6
    torch.testing.assert_close(want, reference, rtol=1e-10, atol=1e-10)


def test_low_rank_sum_truncates() -> None:
    """The precision is the dense top-r of the summed low-rank part plus the base diagonal."""

    model = _mlp()
    x, y = _rows(12, 22)
    post_1 = _low_rank(model, x[:6], y[:6], 0.5, _iso(2.0))
    post_2 = _low_rank(model, x[6:], y[6:], 0.7, lk.priors.LowRankSum.of(post_1))

    summed = _scaled_low_rank(post_1) + _scaled_low_rank(post_2)
    want = _top_r(summed, 3) + 2.0 * torch.eye(27, dtype=F64)
    prec = post_2.precision()

    assert prec.op.factor.shape[1] == 3
    torch.testing.assert_close(_joint_dense(prec), want, rtol=1e-9, atol=1e-9)

    # A smaller rank than the curvature's width truncates the first fit too.
    prior = lk.priors.LowRankSum.of(post_1, rank=2)
    torch.testing.assert_close(
        _joint_dense(prior.summed), _top_r(_scaled_low_rank(post_1), 2), rtol=1e-9, atol=1e-9
    )


def test_low_rank_sum_keeps_its_rank_along_a_chain() -> None:
    """`rank=None` on a post whose prior is a `LowRankSum` keeps that prior's rank, not the
    post's (narrower) curvature width; an explicit rank still wins."""

    model = _mlp()
    x, y = _rows(12, 25)
    post_1 = _low_rank(model, x[:6], y[:6], 0.5, _iso(2.0))
    post_2 = _low_rank(model, x[6:], y[6:], 0.7, lk.priors.LowRankSum.of(post_1, rank=5))
    summed = _scaled_low_rank(post_1) + _scaled_low_rank(post_2)

    prior = lk.priors.LowRankSum.of(post_2)

    assert post_2.curvature.op.factor.shape[1] == 3
    assert prior.rank == 5 and prior.summed.op.factor.shape[1] == 5
    torch.testing.assert_close(_joint_dense(prior.summed), _top_r(summed, 5), rtol=1e-9, atol=1e-9)

    cut = lk.priors.LowRankSum.of(post_2, rank=2)

    assert cut.rank == 2
    torch.testing.assert_close(_joint_dense(cut.summed), _top_r(summed, 2), rtol=1e-9, atol=1e-9)


def test_width_bounded() -> None:
    """Five continual steps keep the width at the rank and follow the dense recursion
    `S_k = top_r(S_{k-1} + c_k U_k Lambda_k U_k^T)`, with the first prior added once."""

    model = _mlp()
    x, y = _rows(20, 23)
    post = _low_rank(model, x[:4], y[:4], 0.5, _iso(1.5))
    reference = _top_r(_scaled_low_rank(post), 3)

    for lo in range(4, 20, 4):
        prior = lk.priors.LowRankSum.of(post)

        assert prior.summed.op.factor.shape[1] <= 3
        assert prior.summed.op.evals.shape[0] <= 3
        assert isinstance(prior.base, Isotropic)
        torch.testing.assert_close(_joint_dense(prior.summed), reference, rtol=1e-9, atol=1e-9)

        post = _low_rank(model, x[lo : lo + 4], y[lo : lo + 4], 0.5, prior)
        reference = _top_r(reference + _scaled_low_rank(post), 3)

        assert post.precision().op.factor.shape[1] <= 3
        torch.testing.assert_close(
            _joint_dense(post.precision()),
            reference + 1.5 * torch.eye(27, dtype=F64),
            rtol=1e-9,
            atol=1e-9,
        )


def test_refuses_kfac_post() -> None:
    model = _mlp()
    x, y = _rows(6, 24)
    kfac = _kfac(model, x, y, 0.5, _iso())

    with pytest.raises(
        StructureError,
        match=r"LowRankSum needs the posterior's curvature to be a Joint of LowRankPlusDiag"
        r".*BlockDiag.*lk\.priors\.KronSum\.of\(post\)",
    ):
        lk.priors.LowRankSum.of(kfac)

    post = _low_rank(model, x, y, 0.5, _iso())

    with pytest.raises(ValueError, match=r"rank >= 1, got 0"):
        lk.priors.LowRankSum.of(post, rank=0)

    # The new curvature is checked too, before the data pass in `lk.laplace`.
    with pytest.raises(StructureError, match=r"before the data pass: LowRankSum.*new curvature"):
        _kfac(model, x, y, 0.5, lk.priors.LowRankSum.of(post))

    other = _low_rank(nn.Sequential(nn.Linear(2, 3)).to(F64), x, y, 0.5, _iso())

    with pytest.raises(StructureError, match=r"Joint blocks differ"):
        lk.priors.LowRankSum.of(post).add_precision_to(other.curvature)


def test_low_rank_sum_location_log_prob_and_repr() -> None:
    model = _mlp()
    x, y = _rows(6, 25)
    post = _low_rank(model, x, y, 0.5, _iso(2.0))
    prior = lk.priors.LowRankSum.of(post, rank=2)

    assert prior.location(post.mean) is post.mean
    assert repr(prior) == (
        "LowRankSum(summed=<rank 2, 27 parameters>, base=Isotropic(prior_precision=2), rank=2, "
        "mean=<4 tensors>)"
    )

    names = list(post.mean)
    flat_mean = torch.cat([post.mean[k].reshape(-1) for k in names])
    g = torch.Generator().manual_seed(3)
    params = {
        k: v + 0.1 * torch.randn(v.shape, generator=g, dtype=F64) for k, v in post.mean.items()
    }
    flat = torch.cat([params[k].reshape(-1) for k in names])
    prec = _top_r(_scaled_low_rank(post), 2) + 2.0 * torch.eye(27, dtype=F64)
    want = MultivariateNormal(flat_mean, precision_matrix=prec).log_prob(flat)

    torch.testing.assert_close(prior.log_prob(params), want, rtol=1e-10, atol=1e-10)


def test_plain_low_rank_prior_docstrings_point_to_low_rank_sum() -> None:
    assert "LowRankSum" in (LowRankPlusDiag.add_same.__doc__ or "")
