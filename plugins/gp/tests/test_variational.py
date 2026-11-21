"""VaLLA: `lk.gp.VaLLA` and `VariationalGP`, against dense references built here.

The references take Jacobians from `torch.func.jacrev`, kernels as J Σ₀ Jᵀ with an
explicit Σ₀, solves from `torch.linalg`, and the KL from `torch.distributions`.
"""

import copy

import laplacekit_gp._jacobian as jacobian_mod
import laplacekit_gp.nystrom as nystrom
import laplacekit_gp.variational as variational
import pytest
import torch
from laplacekit_gp import Picked
from torch import Tensor, nn
from torch.distributions import MultivariateNormal, kl_divergence
from torch.func import functional_call, jacrev

import laplacekit as lk
from laplacekit import function_space as fs
from laplacekit.core import CapabilityError, Done, StructureError, pytree

DT = torch.float64
LAM = 2.0


# --- data and dense references ------------------------------------------------------------


def blobs(n: int, seed: int = 0) -> tuple[Tensor, Tensor]:
    g = torch.Generator().manual_seed(seed)
    centers = torch.tensor([[-2.0, 0.0], [2.0, 0.0], [0.0, 3.0]], dtype=DT)
    y = torch.arange(n) % 3
    x = centers[y] + 0.7 * torch.randn(n, 2, generator=g, dtype=DT)

    return x, y


def net(out: int = 3, hidden: int = 5, inp: int = 2) -> nn.Module:
    torch.manual_seed(0)

    return nn.Sequential(nn.Linear(inp, hidden), nn.Tanh(), nn.Linear(hidden, out)).double()


def loader(x: Tensor, y: Tensor, k: int = 3) -> list[tuple[Tensor, Tensor]]:
    n = x.shape[0] // k

    return [(x[i * n : (i + 1) * n], y[i * n : (i + 1) * n]) for i in range(k)]


def isotropic(lam: float = LAM) -> lk.priors.Isotropic:
    return lk.priors.Isotropic(torch.tensor(lam, dtype=DT))


def x_test() -> Tensor:
    return torch.tensor([[0.0, 1.0], [3.0, -1.0], [-4.0, 4.0], [0.5, 0.5]], dtype=DT)


def jacobian(model: nn.Module, x: Tensor) -> Tensor:
    """[N, E, P] by `jacrev`, over all parameters in module order; differentiable in x."""

    params = {k: v.detach() for k, v in model.named_parameters()}
    jac = jacrev(lambda p: functional_call(model, p, (x,)))(params)
    n = x.shape[0]

    return torch.cat([j.reshape(n, -1, j[0, 0].numel()) for j in jac.values()], dim=2)


def random_factor(m: int, seed: int = 3) -> Tensor:
    g = torch.Generator().manual_seed(seed)

    return torch.randn(m, m, generator=g, dtype=DT).tril(-1) + torch.diag(
        1.0 + torch.rand(m, generator=g, dtype=DT)
    )


def dense_moments(
    model: nn.Module, z: Tensor, coords: Tensor, factor: Tensor, cov0: Tensor, x: Tensor
) -> tuple[Tensor, Tensor]:
    """Per test input, Kₓₓ − Kₓ_z (K_zz + (L Lᵀ)⁻¹)⁻¹ K_zx, and KL(q(f_z) ‖ p(f_z)).

    K = J Σ₀ Jᵀ with Σ₀ = `cov0` dense; q(f_z) = N(0, (K_zz⁻¹ + L Lᵀ)⁻¹).
    """

    jz = jacobian(model, z)[torch.arange(z.shape[0]), coords]
    jx = jacobian(model, x)
    n, e, _ = jx.shape

    k_zz = jz @ cov0 @ jz.T
    k_zx = jz @ cov0 @ jx.reshape(n * e, -1).T
    k_xx = torch.einsum("nep,pq,nfq->nef", jx, cov0, jx)

    llt = factor @ factor.T
    middle = torch.linalg.solve(k_zz + torch.linalg.inv(llt), k_zx).reshape(-1, n, e)
    cov = k_xx - torch.einsum("mne,mnf->nef", k_zx.reshape(-1, n, e), middle)

    s = torch.linalg.inv(torch.linalg.inv(k_zz) + llt)
    zero = torch.zeros(z.shape[0], dtype=DT)
    kl = kl_divergence(MultivariateNormal(zero, s), MultivariateNormal(zero, k_zz))

    return cov, kl


def n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


class Passes:
    """A re-iterable list that counts its iterations and the items pulled."""

    def __init__(self, items):
        self.items = list(items)
        self.iters = 0
        self.pulled = 0

    def __iter__(self):
        self.iters += 1

        for it in self.items:
            self.pulled += 1
            yield it


def fit(model, data, lik, approx, prior=None, subset=None):
    return fs.laplace(
        model,
        data,
        likelihood=lik,
        method=approx,
        subset=lk.subsets.all if subset is None else subset,
        prior=isotropic() if prior is None else prior,
    )


Z_COORDS = torch.tensor([0, 1, 2, 2, 1, 0])


def fitted(model: nn.Module | None = None) -> tuple[nn.Module, Tensor, lk.gp.VariationalGP]:
    model = net() if model is None else model
    x, y = blobs(24)
    z = x[:6]
    gp = fit(model, loader(x, y), lk.Categorical(), lk.gp.VaLLA(lk.gp.fixed(z, Z_COORDS)))

    return model, z, gp


# --- the posterior ---------------------------------------------------------------------------


def test_covariance_equals_pseudo_observation_gp():
    """At an invertible L, `at` is the GP conditioned on pseudo-observations at the inducing
    outputs with noise covariance (L Lᵀ)⁻¹, and its gradients match the dense ones."""

    model, z, gp = fitted()
    factor = random_factor(6)
    cov0 = torch.eye(n_params(model), dtype=DT) / LAM
    want, _ = dense_moments(model, z, Z_COORDS, factor, cov0, x_test())

    out = copy.replace(gp, factor=factor).at(x_test())

    assert out.cov.per_input().shape == (4, 3, 3)
    torch.testing.assert_close(out.mean, model(x_test()).detach())
    torch.testing.assert_close(out.cov.per_input(), want, rtol=1e-8, atol=1e-12)

    # Gradients in the factor, the prior precision and the inducing inputs.
    weight = torch.randn(4, 3, 3, generator=torch.Generator().manual_seed(0), dtype=DT)

    def grads(fn):
        f = factor.clone().requires_grad_()
        lam = torch.tensor(LAM, dtype=DT, requires_grad=True)
        zz = z.clone().requires_grad_()
        loss = (fn(f, lam, zz) * weight).sum()

        return torch.autograd.grad(loss, (f, lam, zz))

    def ours(f, lam, zz):
        gp2 = copy.replace(gp, factor=f, prior=lk.priors.Isotropic(lam), inducing=zz)

        return gp2.at(x_test()).cov.per_input()

    def dense(f, lam, zz):
        eye = torch.eye(n_params(model), dtype=DT)

        return dense_moments(model, zz, Z_COORDS, f.tril(), eye / lam, x_test())[0]

    for g_ours, g_dense in zip(grads(ours), grads(dense), strict=True):
        torch.testing.assert_close(g_ours, g_dense, rtol=1e-7, atol=1e-10)


def test_kl_equals_torch_kl():
    model, z, gp = fitted()
    factor = random_factor(6)
    cov0 = torch.eye(n_params(model), dtype=DT) / LAM
    _, want = dense_moments(model, z, Z_COORDS, factor, cov0, x_test())

    kl = copy.replace(gp, factor=factor).kl()

    assert kl.shape == ()
    torch.testing.assert_close(kl, want, rtol=1e-8, atol=1e-12)

    # Gradients flow to the factor, the prior and the inducing inputs.
    f = factor.clone().requires_grad_()
    lam = torch.tensor(LAM, dtype=DT, requires_grad=True)
    zz = z.clone().requires_grad_()
    gp2 = copy.replace(gp, factor=f, prior=lk.priors.Isotropic(lam), inducing=zz)
    grads = torch.autograd.grad(gp2.kl(), (f, lam, zz))

    f_ref = factor.clone().requires_grad_()
    lam_ref = torch.tensor(LAM, dtype=DT, requires_grad=True)
    z_ref = z.clone().requires_grad_()
    eye = torch.eye(n_params(model), dtype=DT)
    ref = dense_moments(model, z_ref, Z_COORDS, f_ref.tril(), eye / lam_ref, x_test())[1]
    want_grads = torch.autograd.grad(ref, (f_ref, lam_ref, z_ref))

    for g_ours, g_dense in zip(grads, want_grads, strict=True):
        torch.testing.assert_close(g_ours, g_dense, rtol=1e-7, atol=1e-10)


def test_identity_factor_dense():
    """With L = I the KL is ½(log|I + K_zz| − tr((I + K_zz)⁻¹ K_zz)), the dense value."""

    model, z, gp = fitted()
    cov0 = torch.eye(n_params(model), dtype=DT) / LAM
    want_cov, want_kl = dense_moments(model, z, Z_COORDS, torch.eye(6, dtype=DT), cov0, x_test())

    torch.testing.assert_close(gp.kl(), want_kl, rtol=1e-8, atol=1e-12)
    torch.testing.assert_close(gp.at(x_test()).cov.per_input(), want_cov, rtol=1e-8, atol=1e-12)


@pytest.mark.parametrize("points", ["fixed", "from_data"])
def test_identity_factor_init_and_count(points):
    model = net()
    x, y = blobs(30)
    data = Passes(loader(x, y))
    pick = lk.gp.fixed(x[:5]) if points == "fixed" else lk.gp.from_data(5)

    gp = fit(model, data, lk.Categorical(), lk.gp.VaLLA(pick))

    torch.testing.assert_close(gp.factor, torch.eye(5, dtype=DT))
    torch.testing.assert_close(gp.n_data, torch.tensor(30.0, dtype=DT))
    assert gp.coords.dtype == torch.int64 and gp.coords.shape == (5,)
    assert gp.inducing.shape == (5, 2)

    # The probe's batch, then a count pass, after a pick pass with from_data.
    assert data.iters == (2 if points == "fixed" else 3)


def test_count_honours_weights_with_fixed_points():
    model = net()
    x, y = blobs(12)
    w = torch.linspace(0.5, 2.0, 12, dtype=DT)
    data = [lk.Batch(x[:6], y[:6], weights=w[:6]), lk.Batch(x[6:], y[6:], weights=w[6:])]

    gp = fit(model, data, lk.Categorical(), lk.gp.VaLLA(lk.gp.fixed(x[:4])))

    torch.testing.assert_close(gp.n_data, w.sum())


def test_predicted_coords_are_argmax():
    model = net()
    x, y = blobs(30)

    # Steeper weights, so the points predict different classes.
    with torch.no_grad():
        model[0].weight.mul_(3.0)
        model[2].weight.mul_(3.0)

    gp = fit(model, loader(x, y), lk.Categorical(), lk.gp.VaLLA(lk.gp.from_data(10)))

    with torch.no_grad():
        want = model(gp.inducing).argmax(-1)

    assert len(set(want.tolist())) > 1
    torch.testing.assert_close(gp.coords, want)


def test_random_coords_are_floor_u_e():
    model = net()
    x, y = blobs(24)
    points = lk.gp.fixed(x[:9], generator=torch.Generator().manual_seed(5))
    picked = points.pick({}, None, Done).result
    want = (picked.u * 3).floor().long()

    gp = fit(model, loader(x, y), lk.Categorical(), lk.gp.VaLLA(points, coords="random"))

    assert len(set(want.tolist())) > 1
    torch.testing.assert_close(gp.coords, want)


def test_given_coords_win():
    model, _, gp = fitted()

    torch.testing.assert_close(gp.coords, Z_COORDS)


def test_any_pushable_prior():
    model = net()
    x, y = blobs(24)
    z = x[:6]
    prec = torch.linspace(0.5, 3.0, n_params(model), dtype=DT)
    prior = lk.priors.Diagonal(prec)
    gp = fit(
        model,
        loader(x, y),
        lk.Categorical(),
        lk.gp.VaLLA(lk.gp.fixed(z, Z_COORDS)),
        prior=prior,
    )
    factor = random_factor(6)
    want_cov, want_kl = dense_moments(model, z, Z_COORDS, factor, torch.diag(1 / prec), x_test())

    gp = copy.replace(gp, factor=factor)

    torch.testing.assert_close(gp.at(x_test()).cov.per_input(), want_cov, rtol=1e-8, atol=1e-12)
    torch.testing.assert_close(gp.kl(), want_kl, rtol=1e-8, atol=1e-12)


def test_upper_triangle_ignored():
    _, _, gp = fitted()
    factor = random_factor(6)
    junk = factor + torch.randn(6, 6, dtype=DT).triu(1)

    a = copy.replace(gp, factor=factor)
    b = copy.replace(gp, factor=junk)

    torch.testing.assert_close(b.at(x_test()).cov.per_input(), a.at(x_test()).cov.per_input())
    torch.testing.assert_close(b.kl(), a.kl())


def test_float32_model_returns_float32():
    model = net().float()
    x, y = blobs(24)
    x = x.float()
    gp = fit(
        model,
        loader(x, y),
        lk.Categorical(),
        lk.gp.VaLLA(lk.gp.fixed(x[:6], Z_COORDS)),
        prior=lk.priors.Isotropic(LAM),
    )
    out = gp.at(x_test().float())

    assert gp.factor.dtype == torch.float32
    assert out.cov.per_input().dtype == torch.float32
    assert gp.kl().dtype == torch.float32


def test_mapping_inputs():
    """Inducing inputs may be mappings, as every `Points` allows."""

    class Keyed(nn.Module):
        def __init__(self):
            super().__init__()
            self.inner = net()

        def forward(self, x):
            return self.inner(x)

    model = Keyed()
    x, y = blobs(24)
    data = [lk.Batch({"x": x[i : i + 8]}, y[i : i + 8]) for i in range(0, 24, 8)]
    gp = fit(
        model,
        data,
        lk.Categorical(),
        lk.gp.VaLLA(lk.gp.fixed({"x": x[:6]}, Z_COORDS)),
    )
    factor = random_factor(6)
    cov0 = torch.eye(n_params(model), dtype=DT) / LAM
    want, _ = dense_moments(model.inner, x[:6], Z_COORDS, factor, cov0, x_test())

    out = copy.replace(gp, factor=factor).at({"x": x_test()})

    torch.testing.assert_close(out.cov.per_input(), want, rtol=1e-8, atol=1e-12)


def test_duplicate_points_match_the_dense_gp():
    """Repeated inducing points give a singular K_zz; the I + Lᵀ K_zz L form needs no
    inverse of it, so the covariance still matches the pseudo-observation GP."""

    model = net()
    x, y = blobs(24)
    z = torch.cat([x[:3], x[:3]])
    gp = fit(model, loader(x, y), lk.Categorical(), lk.gp.VaLLA(lk.gp.fixed(z, Z_COORDS)))
    factor = random_factor(6)
    cov0 = torch.eye(n_params(model), dtype=DT) / LAM
    jz = jacobian(model, z)[torch.arange(6), Z_COORDS]

    assert int(torch.linalg.matrix_rank(jz @ jz.T)) < 6

    jx = jacobian(model, x_test())
    n, e, _ = jx.shape
    k_zz = jz @ cov0 @ jz.T
    k_zx = jz @ cov0 @ jx.reshape(n * e, -1).T
    k_xx = torch.einsum("nep,pq,nfq->nef", jx, cov0, jx)
    middle = torch.linalg.solve(k_zz + torch.linalg.inv(factor @ factor.T), k_zx)
    want = k_xx - torch.einsum("mne,mnf->nef", k_zx.reshape(-1, n, e), middle.reshape(-1, n, e))

    gp = copy.replace(gp, factor=factor)

    torch.testing.assert_close(gp.at(x_test()).cov.per_input(), want, rtol=1e-8, atol=1e-12)
    assert bool(torch.isfinite(gp.kl()))


def test_chunked_at_matches_one_block(monkeypatch):
    """`at` takes the query Jacobian a few inputs at a time; the blocks change nothing."""

    model, _, gp = fitted()
    gp = copy.replace(gp, factor=random_factor(6))
    whole = gp.at(x_test()).cov.per_input()

    # One input's E P = 3 * 33 rows per block: four query blocks, plus the inducing rows.
    calls = []

    def counted(*args):
        calls.append(args[3] is None)

        return jacobian_mod.rows(*args)

    monkeypatch.setattr(jacobian_mod, "VJP_CHUNK_ENTRIES", 3 * n_params(model))
    monkeypatch.setattr(variational, "rows", counted)
    blocks = gp.at(x_test())

    assert calls.count(True) == 4
    torch.testing.assert_close(blocks.cov.per_input(), whole, rtol=1e-12, atol=1e-14)
    torch.testing.assert_close(blocks.mean, model(x_test()).detach())


# --- the count pass -----------------------------------------------------------------------


def test_count_pass_is_a_monoid():
    from laplacekit_gp.variational import Count

    model = net()
    x, y = blobs(12)
    params = {k: v.detach() for k, v in model.named_parameters()}
    w = torch.linspace(0.5, 2.0, 12, dtype=DT)
    batches = [lk.Batch(x[:5], y[:5], weights=w[:5]), lk.Batch(x[5:], y[5:])]
    count = Count()

    def fold(bs):
        stat = count.init(params, None)

        for b in bs:
            stat = count.update(stat, params, b, torch.Generator().manual_seed(0), None)

        return stat

    whole = fold(batches)
    parts = fold(batches[:1]) + fold(batches[1:])

    torch.testing.assert_close(whole.weight, parts.weight)
    torch.testing.assert_close(count.finalize(whole).weight, w[:5].sum() + 7)
    torch.testing.assert_close(whole.scaled(torch.tensor(2.0)).weight, 2 * whole.weight)


# --- refusals -----------------------------------------------------------------------------


def test_coords_option_checked():
    with pytest.raises(ValueError, match="coords"):
        lk.gp.VaLLA(lk.gp.from_data(5), coords="argmax")  # ty: ignore[invalid-argument-type]


def test_requirements_refused_before_data():
    """The method declares what every pass needs, so a mismatch fails at the probe."""

    x, y = blobs(24)

    class NoVmap(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = net()

        def forward(self, x):
            out = self.lin(x)

            return out if bool(out.sum() > -1e9) else -out  # data-dependent: no vmap

    data = Passes(loader(x, y))

    with pytest.raises(CapabilityError, match="vmap"):
        fit(NoVmap(), data, lk.Categorical(), lk.gp.VaLLA(lk.gp.from_data(6)))

    assert data.pulled == 1

    # The pick pass ignores weights, so weighted data is refused at the first batch.
    w = torch.ones(8, dtype=DT)
    data = Passes([lk.Batch(a, b, w) for a, b in loader(x, y)])

    with pytest.raises(CapabilityError, match="weights"):
        fit(net(), data, lk.Categorical(), lk.gp.VaLLA(lk.gp.from_data(6)))

    assert data.pulled == 1


class GivenPicked:
    """Points that hand over given inputs and coordinates, unknown to the method."""

    def __init__(self, inputs, coords):
        self.inputs, self.coords = inputs, coords

    def pick(self, params, ctx, then, /):
        m = self.inputs.shape[0]
        u = torch.zeros(m, dtype=DT)

        return then(Picked(self.inputs, u, self.coords, torch.zeros((), dtype=torch.int64)))


@pytest.mark.parametrize("bad", [3, -1])
def test_out_of_range_coords_refused(bad):
    """A coordinate outside [0, E) would give a zero Jacobian row: refused, naming E."""

    x, y = blobs(24)
    coords = torch.tensor([0, 1, bad, 2])
    match = rf"coordinate {bad} .*point 2.*E=3"

    # Given points: against the probe's E, before any data.
    for approx in (
        lk.gp.VaLLA(lk.gp.fixed(x[:4], coords)),
        lk.gp.ELLA(lk.gp.fixed(x[:4], coords), 2),
    ):
        data = Passes(loader(x, y))

        with pytest.raises(StructureError, match=match):
            fit(net(), data, lk.Categorical(), approx)

        assert data.pulled == 1

    # Coordinates from other points: when they are handed over, still before data.
    for approx in (
        lk.gp.VaLLA(GivenPicked(x[:4], coords)),
        lk.gp.ELLA(GivenPicked(x[:4], coords), 2),
    ):
        data = Passes(loader(x, y))

        with pytest.raises(StructureError, match=match):
            fit(net(), data, lk.Categorical(), approx)

        assert data.pulled == 1


def test_budget_before_data(monkeypatch):
    model = net()
    x, y = blobs(24)
    p = n_params(model)  # 33

    # The inducing rows: M P = 6 * 33, known before data for given and picked points.
    monkeypatch.setattr(nystrom, "MAX_BASIS_ENTRIES", 6 * p - 1)

    for points in (lk.gp.fixed(x[:6]), lk.gp.from_data(6), lk.gp.kmeans(6)):
        data = Passes(loader(x, y))

        with pytest.raises(StructureError, match=r"M=6.*P=33.*MAX_BASIS_ENTRIES"):
            fit(model, data, lk.Categorical(), lk.gp.VaLLA(points))

        assert data.pulled == 1

    # Points the method can't count: checked when they are handed over.
    data = Passes(loader(x, y))

    with pytest.raises(StructureError, match=r"M=6.*P=33.*MAX_BASIS_ENTRIES"):
        fit(model, data, lk.Categorical(), lk.gp.VaLLA(GivenPicked(x[:6], None)))

    assert data.pulled == 1

    # One input's Jacobian, E P = 3 * 33, is the smallest block `at` can take.
    monkeypatch.setattr(nystrom, "MAX_BASIS_ENTRIES", 3 * p - 1)
    data = Passes(loader(x, y))

    with pytest.raises(StructureError, match=r"E=3.*P=33.*MAX_BASIS_ENTRIES"):
        fit(model, data, lk.Categorical(), lk.gp.VaLLA(lk.gp.fixed(x[:2])))

    assert data.pulled == 1


class Named(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = net()

    def forward(self, x):
        return {"logits": self.lin(x)}


class DictCategorical:
    """One likelihood that reads a dict of outputs: it gets past the probe."""

    def __init__(self):
        self.lik = lk.Categorical()

    def log_prob(self, f, y):
        return self.lik.log_prob(f["logits"], y)

    def output_hessian(self, f):
        return self.lik.output_hessian(f["logits"])


def test_dict_outputs_refused_before_data():
    x, y = blobs(24)
    data = Passes([(a, {"logits": b}) for a, b in loader(x, y)])
    lik = {"logits": lk.Categorical()}

    with pytest.raises(StructureError, match="one tensor"):
        fit(Named(), data, lik, lk.gp.VaLLA(lk.gp.from_data(6)))

    assert data.pulled == 1

    # One likelihood over a dict: the coordinates need f at given points, in `plan`.
    data = Passes(loader(x, y))

    with pytest.raises(StructureError, match="one tensor, got dict"):
        fit(Named(), data, DictCategorical(), lk.gp.VaLLA(lk.gp.fixed(x[:6])))

    assert data.pulled == 1


# --- display and staleness ----------------------------------------------------------------


def test_repr():
    approx = lk.gp.VaLLA(lk.gp.from_data(6))

    assert repr(approx) == "VaLLA(points=from_data(6), coords='predicted')"

    _, _, gp = fitted()

    assert repr(gp) == "VariationalGP(m=6, parameters=33, prior=Isotropic(prior_precision=2))"


def test_staleness_guard():
    model = net()
    x, y = blobs(24)
    gp = fit(
        model,
        loader(x, y),
        lk.Categorical(),
        lk.gp.VaLLA(lk.gp.from_data(6)),
        subset=lk.subsets.last_layer,
    )
    before = gp.at(x_test())

    # The mean is the posterior's own copy: moving the fitted layer leaves the GP alone.
    with torch.no_grad():
        model[2].weight.add_(1.0)

    torch.testing.assert_close(gp.at(x_test()).cov.per_input(), before.cov.per_input())

    with torch.no_grad():
        model[0].weight.add_(1.0)

    stale = r"fixed parameter '0\.weight' changed in place.*the GP linearizes the model at its"

    with pytest.raises(RuntimeError, match=stale):
        gp.at(x_test())

    with pytest.raises(RuntimeError, match=stale):
        gp.kl()


# --- the objective --------------------------------------------------------------------------


def wave(n: int, out: int = 1, seed: int = 1) -> tuple[Tensor, Tensor]:
    g = torch.Generator().manual_seed(seed)
    x = torch.linspace(-2.0, 2.0, n, dtype=DT)[:, None]
    y = torch.cat([torch.sin(k * x) for k in range(1, out + 1)], 1)

    return x, y + 0.2 * torch.randn(n, out, generator=g, dtype=DT)


def regression_gp(sigma: float | Tensor = 0.3, out: int = 1):
    model = net(out=out, hidden=6, inp=1)
    x, y = wave(12, out)
    z = x[::3]
    sig = sigma if isinstance(sigma, Tensor) else torch.tensor(sigma, dtype=DT)
    points = lk.gp.fixed(z, torch.zeros(z.shape[0], dtype=torch.int64))
    gp = fit(
        model,
        loader(x, y),
        copy.replace(lk.Gaussian(), sigma=torch.as_tensor(sig)),
        lk.gp.VaLLA(points),
    )

    return model, x, y, gp


@pytest.mark.parametrize("alpha", [0.3, 1.0])
def test_gaussian_alpha_energy_equals_quadrature(alpha):
    """Per point, D = (1/α) log ∫ N(y; f, σ²)^α N(f; μ, v) df, on a 4001-point grid."""

    model, x, y, gp = regression_gp()
    gp = copy.replace(gp, factor=random_factor(4))
    cov0 = torch.eye(n_params(model), dtype=DT) / LAM
    var, _ = dense_moments(model, x[::3], torch.zeros(4, dtype=torch.int64), gp.factor, cov0, x)
    mu = model(x).detach()

    for i in range(x.shape[0]):
        m, v, t = mu[i, 0], var[i, 0, 0], y[i, 0]
        f = torch.linspace(float(m - 12 * v.sqrt()), float(m + 12 * v.sqrt()), 4001, dtype=DT)
        lik = torch.distributions.Normal(f, 0.3).log_prob(t)
        prior = torch.distributions.Normal(m, v.sqrt()).log_prob(f)
        want = torch.logsumexp(alpha * lik + prior, 0) + (f[1] - f[0]).log()
        want = want / alpha

        loss = lk.gp.variational_loss(gp, lk.Batch(x[i : i + 1], y[i : i + 1]), alpha=alpha)
        got = (gp.kl() - loss) / gp.n_data

        torch.testing.assert_close(got, want, rtol=1e-7, atol=1e-9)


def test_gaussian_alpha_zero_is_the_elbo():
    """α = 0: D = Σ E_q[log N(y; f, σ²)] = −½ Σ [((y − μ)² + v)/σ² + log 2πσ²]."""

    model, x, y, gp = regression_gp()
    cov0 = torch.eye(n_params(model), dtype=DT) / LAM
    var, _ = dense_moments(model, x[::3], torch.zeros(4, dtype=torch.int64), gp.factor, cov0, x)
    mu = model(x).detach()
    want = (
        -0.5
        * (
            ((y - mu) ** 2 + var[:, :, 0]) / 0.09
            + torch.log(2 * torch.pi * torch.tensor(0.09, dtype=DT))
        ).sum()
    )

    loss = lk.gp.variational_loss(gp, lk.Batch(x, y), alpha=0.0)

    torch.testing.assert_close(loss, -(gp.n_data / 12) * want + gp.kl(), rtol=1e-10, atol=1e-10)


def test_weighted_batch():
    """With weights, D = Σ wₙ dₙ with the closed-form dₙ, and B = Σ wₙ."""

    model, x, y, gp = regression_gp()
    w = torch.linspace(0.5, 2.0, 12, dtype=DT)
    cov0 = torch.eye(n_params(model), dtype=DT) / LAM
    var, _ = dense_moments(model, x[::3], torch.zeros(4, dtype=torch.int64), gp.factor, cov0, x)
    v, mu, s2 = var[:, :, 0], model(x).detach(), torch.tensor(0.09, dtype=DT)
    d = -0.5 * ((y - mu) ** 2 / (s2 + v) + torch.log1p(v / s2) + torch.log(2 * torch.pi * s2))
    want = (w * d[:, 0]).sum()

    loss = lk.gp.variational_loss(gp, lk.Batch(x, y, weights=w))

    torch.testing.assert_close(
        loss, -(gp.n_data / w.sum()) * want + gp.kl(), rtol=1e-10, atol=1e-10
    )


@pytest.mark.parametrize("alpha", [0.0, 0.5, 1.0])
def test_mc_matches_closed_form_gaussian(alpha):
    """S = 4096 draws: within 3 standard errors of the closed form. The standard error comes
    from independent draws here, through the delta method for the log of the mean."""

    model, x, y, gp = regression_gp()
    s = 4096
    batch = lk.Batch(x, y)
    exact = lk.gp.variational_loss(gp, batch, alpha=alpha)
    mc = lk.gp.variational_loss(
        gp, batch, alpha=alpha, n_samples=s, generator=torch.Generator().manual_seed(5)
    )

    cov0 = torch.eye(n_params(model), dtype=DT) / LAM
    var, _ = dense_moments(model, x[::3], torch.zeros(4, dtype=torch.int64), gp.factor, cov0, x)
    mu = model(x).detach()[:, 0]
    g = torch.Generator().manual_seed(11)
    f = mu + var[:, 0, 0].sqrt() * torch.randn(s, 12, generator=g, dtype=DT)
    ll = torch.distributions.Normal(f, 0.3).log_prob(y[:, 0])

    if alpha == 0.0:
        se_n = ll.std(0) / s**0.5
    else:
        p = (alpha * ll).exp()
        se_n = p.std(0) / (s**0.5 * p.mean(0)) / alpha

    se = gp.n_data / 12 * se_n.square().sum().sqrt()

    assert abs(float(mc - exact)) < 3 * float(se), (float(mc), float(exact), float(se))


def test_mc_follows_the_generator():
    _, x, y, gp = regression_gp()
    batch = lk.Batch(x, y)

    a = lk.gp.variational_loss(gp, batch, n_samples=8, generator=torch.Generator().manual_seed(1))
    b = lk.gp.variational_loss(gp, batch, n_samples=8, generator=torch.Generator().manual_seed(1))
    c = lk.gp.variational_loss(gp, batch, n_samples=8, generator=torch.Generator().manual_seed(2))

    assert bool(a == b) and bool(a != c)


def test_categorical_probit_closed_form():
    """laplace-torch's heuristic: log p(y | μ / √(1 + π/8 · diag Σ)), the same for any α."""

    model, z, gp = fitted()
    x, y = blobs(24)
    cov0 = torch.eye(n_params(model), dtype=DT) / LAM
    var, _ = dense_moments(model, z, Z_COORDS, gp.factor, cov0, x)
    scaled = model(x).detach() / (1 + torch.pi / 8 * var.diagonal(dim1=-2, dim2=-1)).sqrt()
    want = torch.distributions.Categorical(logits=scaled).log_prob(y).sum()

    loss = lk.gp.variational_loss(gp, lk.Batch(x, y))

    torch.testing.assert_close(loss, -(gp.n_data / 24) * want + gp.kl(), rtol=1e-10, atol=1e-10)


def test_from_log_prob_needs_samples():
    model, z, gp = fitted()
    x, y = blobs(24)
    lp = lk.likelihoods.from_log_prob(
        lambda f, t: torch.distributions.Categorical(logits=f).log_prob(t)
    )
    gp = copy.replace(gp, likelihood=lp)

    with pytest.raises(CapabilityError, match="n_samples="):
        lk.gp.variational_loss(gp, lk.Batch(x, y))

    loss = lk.gp.variational_loss(gp, lk.Batch(x, y), n_samples=64)

    assert bool(torch.isfinite(loss))


def test_categorical_alpha_needs_samples():
    _, _, gp = fitted()
    x, y = blobs(24)

    with pytest.raises(ValueError, match="n_samples="):
        lk.gp.variational_loss(gp, lk.Batch(x, y), alpha=0.5)

    loss = lk.gp.variational_loss(gp, lk.Batch(x, y), alpha=0.5, n_samples=64)

    assert bool(torch.isfinite(loss))


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"alpha": -0.1}, "alpha"),
        ({"alpha": 1.5}, "alpha"),
        ({"n_samples": 1}, "n_samples"),
        ({"n_samples": 0}, "n_samples"),
    ],
)
def test_options_checked(kwargs, match):
    _, _, gp = fitted()
    x, y = blobs(24)

    with pytest.raises(ValueError, match=match):
        lk.gp.variational_loss(gp, lk.Batch(x, y), **kwargs)


def test_needs_targets():
    _, _, gp = fitted()
    x, _ = blobs(24)

    with pytest.raises(ValueError, match="targets"):
        lk.gp.variational_loss(gp, lk.Batch(x))


# --- fit_variational -------------------------------------------------------------------------


def test_fit_trains_the_leaves_and_lowers_the_objective():
    _, x, y, gp = regression_gp()
    batch = lk.Batch(x, y)

    out = lk.gp.fit_variational(gp, loader(x, y), n_steps=30, lr=1e-2)

    assert isinstance(out, lk.gp.VariationalGP)
    assert not out.factor.requires_grad and not out.inducing.requires_grad
    assert float(lk.gp.variational_loss(out, batch)) < float(lk.gp.variational_loss(gp, batch))

    for new, old in [
        (out.factor, gp.factor),
        (out.prior.prior_precision, gp.prior.prior_precision),
        (out.likelihood.sigma, gp.likelihood.sigma),
        (out.inducing, gp.inducing),
    ]:
        assert not torch.equal(new, old)

    # The input posterior is unchanged.
    torch.testing.assert_close(gp.factor, torch.eye(4, dtype=DT))
    assert float(gp.prior.prior_precision) == LAM and float(gp.likelihood.sigma) == 0.3


def test_fit_train_inputs_false():
    _, x, y, gp = regression_gp()

    out = lk.gp.fit_variational(gp, loader(x, y), n_steps=5, lr=1e-2, learn_inducing=False)

    assert torch.equal(out.inducing, gp.inducing)
    assert not torch.equal(out.factor, gp.factor)


@pytest.mark.parametrize(("prior", "noise"), [(False, False), (False, True), (True, False)])
def test_fit_holds_prior_and_noise(prior, noise):
    """Held leaves stay bit-identical; the factor, Z and the other leaf still train."""

    _, x, y, gp = regression_gp()

    out = lk.gp.fit_variational(
        gp, loader(x, y), n_steps=5, lr=1e-2, learn_prior=prior, learn_noise=noise
    )

    assert torch.equal(out.prior.prior_precision, gp.prior.prior_precision) != prior
    assert torch.equal(out.likelihood.sigma, gp.likelihood.sigma) != noise
    assert not torch.equal(out.factor, gp.factor)
    assert not torch.equal(out.inducing, gp.inducing)


def test_fit_held_hyper_matches_a_hand_written_adam():
    """With λ and σ held, Adam over (L, Z) only."""

    _, x, y, gp = regression_gp()
    batches = loader(x, y)

    out = lk.gp.fit_variational(
        gp, batches, n_steps=7, lr=1e-2, learn_prior=False, learn_noise=False
    )

    leaves = [gp.factor.clone().requires_grad_(), gp.inducing.clone().requires_grad_()]
    opt = torch.optim.Adam(leaves, lr=1e-2)

    for k in range(7):
        bx, by = batches[k % 3]
        cur = copy.replace(gp, factor=leaves[0], inducing=leaves[1])
        opt.zero_grad()
        lk.gp.variational_loss(cur, lk.Batch(bx, by)).backward()
        opt.step()

    torch.testing.assert_close(out.factor, leaves[0].detach(), rtol=1e-10, atol=1e-12)
    torch.testing.assert_close(out.inducing, leaves[1].detach(), rtol=1e-10, atol=1e-12)
    assert torch.equal(out.prior.prior_precision, gp.prior.prior_precision)
    assert torch.equal(out.likelihood.sigma, gp.likelihood.sigma)


def test_fit_matches_a_hand_written_adam():
    """Adam over (L, log λ, log σ², Z), one batch per step, cycling through the loader."""

    _, x, y, gp = regression_gp()
    batches = loader(x, y)

    out = lk.gp.fit_variational(gp, batches, n_steps=7, lr=1e-2)

    leaves = [
        gp.factor.clone().requires_grad_(),
        gp.prior.prior_precision.log().clone().requires_grad_(),
        (2 * gp.likelihood.sigma.log()).clone().requires_grad_(),
        gp.inducing.clone().requires_grad_(),
    ]
    opt = torch.optim.Adam(leaves, lr=1e-2)

    for k in range(7):
        bx, by = batches[k % 3]
        f, lp, lv, z = leaves
        cur = copy.replace(
            gp,
            factor=f,
            prior=lk.priors.Isotropic(lp.exp()),
            likelihood=copy.replace(lk.Gaussian(), sigma=torch.as_tensor((0.5 * lv).exp())),
            inducing=z,
        )
        opt.zero_grad()
        lk.gp.variational_loss(cur, lk.Batch(bx, by)).backward()
        opt.step()

    torch.testing.assert_close(out.factor, leaves[0].detach(), rtol=1e-10, atol=1e-12)
    torch.testing.assert_close(
        out.prior.prior_precision, leaves[1].detach().exp(), rtol=1e-10, atol=0
    )
    torch.testing.assert_close(
        out.likelihood.sigma, (0.5 * leaves[2].detach()).exp(), rtol=1e-10, atol=0
    )
    torch.testing.assert_close(out.inducing, leaves[3].detach(), rtol=1e-10, atol=1e-12)


def test_fit_leaves_model_untouched():
    model = net()
    x, y = blobs(24)
    flags = {n: p.requires_grad for n, p in model.named_parameters()}
    gp = fit(
        model,
        loader(x, y),
        lk.Categorical(),
        lk.gp.VaLLA(lk.gp.fixed(x[:6], Z_COORDS)),
        subset=lk.subsets.last_layer,
    )
    snapshot = {n: (p.detach().clone(), p._version) for n, p in model.named_parameters()}

    out = lk.gp.fit_variational(gp, loader(x, y), n_steps=4, lr=1e-2)

    for n, p in model.named_parameters():
        assert torch.equal(p, snapshot[n][0]) and p._version == snapshot[n][1], n
        assert p.requires_grad == flags[n] and p.grad is None, n

    for k, v in out.mean.items():
        assert torch.equal(v, gp.mean[k]) and not v.requires_grad


def test_fit_under_no_grad():
    """Grad is enabled inside the loop, whatever the caller's mode."""

    _, x, y, gp = regression_gp()

    with torch.no_grad():
        out = lk.gp.fit_variational(gp, loader(x, y), n_steps=3, lr=1e-2)

    assert not torch.equal(out.factor, gp.factor)


def test_mapping_inducing_not_trained():
    class Keyed(nn.Module):
        def __init__(self):
            super().__init__()
            self.inner = net()

        def forward(self, x):
            return self.inner(x)

    model = Keyed()
    x, y = blobs(24)
    data = [lk.Batch({"x": x[i : i + 8]}, y[i : i + 8]) for i in range(0, 24, 8)]
    gp = fit(model, data, lk.Categorical(), lk.gp.VaLLA(lk.gp.fixed({"x": x[:6]}, Z_COORDS)))

    out = lk.gp.fit_variational(gp, data, n_steps=3, lr=1e-2)

    assert torch.equal(out.inducing["x"], x[:6])
    assert not torch.equal(out.factor, gp.factor)
    assert not torch.equal(out.prior.prior_precision, gp.prior.prior_precision)


def test_vector_sigma_fixed():
    """M7b's per-output σ is fixed at fit time, so it is not trained."""

    sigma = torch.tensor([0.3, 0.5], dtype=DT)
    _, x, y, gp = regression_gp(sigma, out=2)

    out = lk.gp.fit_variational(gp, loader(x, y), n_steps=3, lr=1e-2)

    assert torch.equal(out.likelihood.sigma, sigma)
    assert not torch.equal(out.prior.prior_precision, gp.prior.prior_precision)


@pytree
class PlainPrior:
    """A pushable prior without `tunable`."""

    prior_precision: Tensor

    def location(self, like):
        return {k: torch.zeros_like(v) for k, v in like.items()}

    def cov_mv(self, v):
        return {k: x / self.prior_precision for k, x in v.items()}


def test_non_tunable_prior_not_trained():
    model, _, gp = fitted()
    x, y = blobs(24)
    prior = PlainPrior(torch.tensor(LAM, dtype=DT))
    gp = copy.replace(gp, prior=prior)

    out = lk.gp.fit_variational(gp, loader(x, y), n_steps=2, lr=1e-2)

    assert torch.equal(out.prior.prior_precision, prior.prior_precision)
    assert not torch.equal(out.factor, gp.factor)


def test_vector_precision_trained():
    """A tunable prior's leaves are trained entrywise, through their logs."""

    model, _, gp = fitted()
    x, y = blobs(24)
    prior = lk.priors.Diagonal(torch.full((n_params(model),), LAM, dtype=DT))
    gp = copy.replace(gp, prior=prior)

    out = lk.gp.fit_variational(gp, loader(x, y), n_steps=2, lr=1e-2)

    assert out.prior.prior_precision.shape == prior.prior_precision.shape
    assert bool((out.prior.prior_precision != prior.prior_precision).all())


def test_fit_follows_the_generator_and_cycles_data(monkeypatch):
    """The data passes and each step's Monte Carlo draws take turns on one generator,
    seeded from `generator`; the batches cycle, one pass per epoch."""

    _, x, y, gp = regression_gp()
    seen = []
    drawn = []
    loss = variational.variational_loss

    class Recorded:
        n_data = None

        def __call__(self, generator):
            seen.append(generator)

            return [lk.Batch(a, b) for a, b in loader(x, y)]

    def recorded_loss(*args, **kwargs):
        drawn.append(kwargs["generator"])

        return loss(*args, **kwargs)

    monkeypatch.setattr(variational, "variational_loss", recorded_loss)
    g = torch.Generator().manual_seed(3)
    a = lk.gp.fit_variational(gp, Recorded(), n_steps=7, lr=1e-2, n_samples=4, generator=g)

    assert len(seen) == 3 and len(drawn) == 7  # 3 batches per epoch: 3 passes for 7 steps
    assert len({id(t) for t in seen + drawn}) == 1
    assert isinstance(seen[0], torch.Generator) and seen[0] is not g

    def fit(generator):
        return lk.gp.fit_variational(
            gp, loader(x, y), n_steps=7, lr=1e-2, n_samples=4, generator=generator
        )

    b = fit(torch.Generator().manual_seed(3))
    c = fit(torch.Generator().manual_seed(4))

    assert torch.equal(a.factor, b.factor) and not torch.equal(a.factor, c.factor)

    # Without a generator: torch's global RNG, so `torch.manual_seed` repeats the fit.
    torch.manual_seed(1)
    d = fit(None)
    torch.manual_seed(1)

    assert torch.equal(d.factor, fit(None).factor)


def test_data_and_mc_draws_never_share_a_state(monkeypatch):
    """One batch per epoch, drawn by a shuffling source: no data pass and no Monte Carlo
    step starts from the state of another, so they never reuse random numbers."""

    _, x, y, gp = regression_gp()
    states = []
    loss = variational.variational_loss

    class OneBatch:
        n_data = None

        def __call__(self, generator):
            states.append(generator.get_state().clone())
            torch.rand((), generator=generator)  # a shuffle

            return [lk.Batch(x, y)]

    def recorded_loss(*args, **kwargs):
        states.append(kwargs["generator"].get_state().clone())

        return loss(*args, **kwargs)

    monkeypatch.setattr(variational, "variational_loss", recorded_loss)
    lk.gp.fit_variational(
        gp, OneBatch(), n_steps=5, lr=1e-2, n_samples=4, generator=torch.Generator().manual_seed(9)
    )

    assert len(states) == 10
    assert all(not torch.equal(a, b) for i, a in enumerate(states) for b in states[i + 1 :])


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [({"n_steps": 0}, "n_steps"), ({"lr": 0.0}, "lr"), ({"alpha": 2.0}, "alpha")],
)
def test_fit_options_checked_before_data(kwargs, match):
    _, x, y, gp = regression_gp()
    data = Passes(loader(x, y))
    options = {"n_steps": 3} | kwargs

    with pytest.raises(ValueError, match=match):
        lk.gp.fit_variational(gp, data, **options)

    assert data.pulled == 0


def test_fit_likelihood_refused_before_data():
    _, _, gp = fitted()
    x, y = blobs(24)
    data = Passes(loader(x, y))

    with pytest.raises(ValueError, match="n_samples="):
        lk.gp.fit_variational(gp, data, n_steps=3, alpha=0.5)

    assert data.pulled == 0


# --- review fixes: sampling at repeated eigenvalues, Bernoulli, masks, targets ---------------


def last_layer_valla(classes: int = 5):
    """Last-layer VaLLA with C = 5 and inducing coordinates (0, 1, 2): at L = I, classes 3
    and 4 keep the prior variance, so every block has exactly repeated eigenvalues."""

    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(2, 8), nn.Tanh(), nn.Linear(8, classes)).double()
    x = torch.randn(30, 2, dtype=DT)
    y = torch.randint(0, classes, (30,))
    gp = fit(
        model,
        [(x, y)],
        lk.Categorical(),
        lk.gp.VaLLA(lk.gp.fixed(x[:3], torch.tensor([0, 1, 2]))),
        subset=lk.subsets.last_layer,
    )

    return x, y, gp


def test_last_layer_blocks_have_repeated_eigenvalues():
    """The guard for the tests below: the setup really has exactly repeated eigenvalues."""

    x, _, gp = last_layer_valla()
    ev = torch.linalg.eigvalsh(gp.at(x[:4]).cov.per_input())

    assert float((ev[:, 1:] - ev[:, :-1]).abs().min()) == 0.0


def test_mc_gradients_finite_at_repeated_eigenvalues():
    x, y, gp = last_layer_valla()
    factor = gp.factor.clone().requires_grad_()
    lam = gp.prior.prior_precision.clone().requires_grad_()
    z = gp.inducing.clone().requires_grad_()
    cur = copy.replace(gp, factor=factor, prior=lk.priors.Isotropic(lam), inducing=z)

    loss = lk.gp.variational_loss(
        cur, lk.Batch(x, y), alpha=0.5, n_samples=16, generator=torch.Generator().manual_seed(1)
    )
    grads = torch.autograd.grad(loss, (factor, lam, z))

    assert bool(torch.isfinite(loss))

    for g in grads:
        assert bool(torch.isfinite(g).all())


@pytest.mark.parametrize("steps", [1, 3])
def test_mc_fit_finite_at_repeated_eigenvalues(steps):
    x, y, gp = last_layer_valla()

    out = lk.gp.fit_variational(gp, [(x, y)], n_steps=steps, lr=1e-2, alpha=0.5, n_samples=16)

    for t in (out.factor, out.prior.prior_precision, out.inducing):
        assert bool(torch.isfinite(t).all())

    assert not torch.equal(out.factor, gp.factor)


def test_mc_draws_match_the_block_moments():
    """The draws' empirical moments match the per-input mean and covariance."""

    x, y, gp = last_layer_valla()
    out = gp.at(x[:3])
    s = 20000
    f = variational._draws(out, torch.Generator().manual_seed(4), s)
    blocks = out.cov.per_input()

    centred = f - f.mean(0)
    emp = torch.einsum("sne,snf->nef", centred, centred) / (s - 1)

    assert f.shape == (s, 3, 5)
    torch.testing.assert_close(f.mean(0), out.mean, rtol=0, atol=0.05)
    torch.testing.assert_close(emp, blocks, rtol=0, atol=0.05 * float(blocks.abs().max()))


def test_bernoulli_probit_closed_form():
    """log p(y | μ / √(1 + π/8 · diag Σ)) for a Bernoulli likelihood, against a dense
    reference and torch's BCE."""

    model = net(out=3)
    x, _ = blobs(24)
    g = torch.Generator().manual_seed(2)
    y = torch.randint(0, 2, (24, 3), generator=g).to(DT)
    z = x[:6]
    gp = fit(model, loader(x, y), lk.Bernoulli(), lk.gp.VaLLA(lk.gp.fixed(z, Z_COORDS)))
    cov0 = torch.eye(n_params(model), dtype=DT) / LAM
    var, _ = dense_moments(model, z, Z_COORDS, gp.factor, cov0, x)
    scaled = model(x).detach() / (1 + torch.pi / 8 * var.diagonal(dim1=-2, dim2=-1)).sqrt()
    want = -torch.nn.functional.binary_cross_entropy_with_logits(scaled, y, reduction="sum")

    loss = lk.gp.variational_loss(gp, lk.Batch(x, y))

    torch.testing.assert_close(loss, -(gp.n_data / 24) * want + gp.kl(), rtol=1e-10, atol=1e-10)


def test_bad_targets_refused():
    model = net(out=3)
    x, _ = blobs(24)
    y = torch.zeros(24, 3, dtype=DT)
    gp = fit(model, loader(x, y), lk.Bernoulli(), lk.gp.VaLLA(lk.gp.fixed(x[:6], Z_COORDS)))
    bad = torch.full((24, 3), 2.0, dtype=DT)

    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        lk.gp.variational_loss(gp, lk.Batch(x, bad))

    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        lk.gp.fit_variational(gp, [(x, bad)], n_steps=2)


def test_mc_mask_follows_the_inputs():
    """A mask from the inputs drops its rows from every draw: changing a masked target
    changes nothing, and masking every row leaves only the KL."""

    model, z, gp = fitted()
    x, y = blobs(24)
    keep = x[:, 0] > 0
    lik = lk.Categorical(mask_fn=lambda f, inputs: inputs[:, 0] > 0)
    gp = copy.replace(gp, likelihood=lik)
    moved = torch.where(keep, y, (y + 1) % 3)

    def mc(gp, batch):
        gen = torch.Generator().manual_seed(6)
        return lk.gp.variational_loss(gp, batch, alpha=0.5, n_samples=32, generator=gen)

    a = mc(gp, lk.Batch(x, y))
    b = mc(gp, lk.Batch(x, moved))
    c = mc(gp, lk.Batch(x, (y + 1) % 3))

    torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert not torch.equal(a, c)

    none = copy.replace(gp, likelihood=lk.Categorical(mask_fn=lambda f, inputs: inputs[:, 0] > 99))
    loss = mc(none, lk.Batch(x, y))

    torch.testing.assert_close(loss, gp.kl(), rtol=1e-12, atol=1e-12)


def test_fit_checks_versions_before_data():
    model = net()
    x, y = blobs(24)
    gp = fit(
        model,
        loader(x, y),
        lk.Categorical(),
        lk.gp.VaLLA(lk.gp.fixed(x[:6], Z_COORDS)),
        subset=lk.subsets.last_layer,
    )
    data = Passes(loader(x, y))

    with torch.no_grad():
        model[0].weight.add_(1.0)

    with pytest.raises(RuntimeError, match="changed in place"):
        lk.gp.fit_variational(gp, data, n_steps=2)

    assert data.pulled == 0
