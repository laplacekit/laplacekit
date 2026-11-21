"""ELLA: `lk.gp.ELLA` and `NystromGP`, against dense references built here.

The references take Jacobians from `torch.func.jacrev`, the output Hessians by hand, and
the Nyström basis from `torch.linalg.eigh` of the kernel of the picked Jacobian rows.
"""

import copy
import math

import laplacekit_gp.nystrom as nystrom
import pytest
import torch
from torch import Tensor, nn
from torch.func import functional_call, jacrev

import laplacekit as lk
from laplacekit import function_space as fs
from laplacekit.core import CapabilityError, Done, StructureError

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
    """[N, E, P] by `jacrev`, over all parameters in module order."""

    params = {k: v.detach() for k, v in model.named_parameters()}
    jac = jacrev(lambda p: functional_call(model, p, (x,)))(params)
    n = x.shape[0]

    return torch.cat([j.reshape(n, -1, j[0, 0].numel()) for j in jac.values()], dim=2)


def categorical_hessian(f: Tensor) -> Tensor:
    p = f.softmax(-1)

    return torch.diag_embed(p) - p.unsqueeze(2) * p.unsqueeze(1)


def ella_basis(rows: Tensor, rank: int) -> Tensor:
    """V = Rᵀ U Λ^{-1/2} from the top `rank` eigenpairs of R Rᵀ."""

    evals, evecs = torch.linalg.eigh(rows @ rows.T)
    top = torch.argsort(evals, descending=True)[:rank]

    return rows.T @ (evecs[:, top] / evals[top].sqrt())


def projected_cov(phi: Tensor, h: Tensor, phi_test: Tensor, lam: float, c: float = 1.0) -> Tensor:
    """Per test input, Φ* (c Σ Φᵀ H Φ + λ I)⁻¹ Φ*ᵀ."""

    r = phi.shape[-1]
    prec = c * torch.einsum("ner,nef,nfs->rs", phi, h, phi) + lam * torch.eye(r, dtype=DT)

    return phi_test @ torch.linalg.solve(prec, phi_test.mT)


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
        generator=torch.Generator().manual_seed(0),  # the same picks in every fit
    )


# --- the method -----------------------------------------------------------------------------


def test_full_rank_nystrom_equals_dense_ggn():
    """At the training inputs, a full-rank basis gives the dense GGN posterior.

    V has orthonormal columns; let Π = V Vᵀ. When V spans J's row space, J = J Π and
    Jᵀ H J = Π Jᵀ H J Π, so λ I + Jᵀ H J = V (λ I + Φᵀ H Φ) Vᵀ + λ (I - Π) with Φ = J V. Its
    inverse is V (λ I + Φᵀ H Φ)⁻¹ Vᵀ + (I - Π) / λ, and J Σ Jᵀ = Φ (λ I + Φᵀ H Φ)⁻¹ Φᵀ: the
    Nyström GP. This holds for any V spanning J's row space; ELLA's basis does when the
    rank is that of J.
    """

    model = net(out=1, hidden=30, inp=1)

    # Steep units with kinks spread over the inputs keep J J^T well conditioned.
    with torch.no_grad():
        model[0].weight.fill_(8.0)
        model[0].bias.copy_(-8.0 * torch.linspace(-3.0, 3.0, 30, dtype=DT))

    g = torch.Generator().manual_seed(1)
    x = torch.linspace(-3.0, 3.0, 40, dtype=DT).reshape(-1, 1)
    y = x.sin() + 0.1 * torch.randn(40, 1, generator=g, dtype=DT)
    lik = copy.replace(lk.Gaussian(), sigma=torch.as_tensor(torch.tensor(0.3, dtype=DT)))

    # ELLA's rule: every one of the N E = 40 directions is above the tolerance.
    j = jacobian(model, x).reshape(40, -1)
    evals = torch.linalg.eigvalsh(j @ j.T)
    tol = torch.finfo(DT).eps * max(j.shape) * evals.max()

    assert j.shape == (40, 91)
    assert int((evals > tol).sum()) == 40

    approx = lk.gp.ELLA(lk.gp.fixed(x), rank=40, outputs="all")
    gp = fit(model, [(x, y)], lik, approx)
    post = lk.laplace(
        model,
        [(x, y)],
        likelihood=lik,
        hessian_approx=lk.DenseGGN(joint=True),  # the full GGN, not one block per module
        subset=lk.subsets.all,
        prior=isotropic(),
        hyper_tuner=None,
    )

    ours = lk.predict_outputs(gp, x)
    want = lk.predict_outputs(post, x)

    torch.testing.assert_close(ours.mean, want.mean)
    torch.testing.assert_close(ours.cov.per_input(), want.cov.per_input(), rtol=1e-6, atol=0)


@pytest.mark.parametrize("strategy", ["forward-mode", "reverse-mode"])
def test_bernoulli_dense_reference(strategy):
    model = net(out=1, hidden=6)
    x, labels = blobs(30)
    y = (labels == 0).to(DT).unsqueeze(1)
    pts = torch.arange(12)

    approx = lk.gp.ELLA(lk.gp.fixed(x[pts]), rank=8, strategy=strategy)
    gp = fit(model, loader(x, y), lk.Bernoulli(), approx)

    # One output, so every row is coordinate 0.
    v = ella_basis(jacobian(model, x[pts])[:, 0], 8)

    with torch.no_grad():
        p = model(x).sigmoid()

    h = (p * (1 - p)).unsqueeze(2)
    want = projected_cov(jacobian(model, x) @ v, h, jacobian(model, x_test()) @ v, LAM)
    out = gp.at(x_test())

    assert gp.strategy == strategy
    torch.testing.assert_close(out.mean, model(x_test()).detach())
    torch.testing.assert_close(out.cov.per_input(), want, rtol=1e-8, atol=1e-12)


def test_categorical_given_coordinates():
    """Three outputs, one given coordinate per point, and the Gram over three batches."""

    model = net()
    x, y = blobs(24)
    coords = torch.tensor([0, 1, 2, 2, 1, 0, 1, 2, 0, 1])

    gp = fit(model, loader(x, y), lk.Categorical(), lk.gp.ELLA(lk.gp.fixed(x[:10], coords), 7))

    rows = jacobian(model, x[:10])[torch.arange(10), coords]
    v = ella_basis(rows, 7)

    with torch.no_grad():
        h = categorical_hessian(model(x))

    want = projected_cov(jacobian(model, x) @ v, h, jacobian(model, x_test()) @ v, LAM)

    torch.testing.assert_close(gp.at(x_test()).cov.per_input(), want, rtol=1e-8, atol=1e-12)
    torch.testing.assert_close(gp.n_data, torch.tensor(24.0, dtype=DT))


def test_random_coordinates_are_floor_u_e():
    """Without coordinates, point i takes output ⌊uᵢ E⌋ of its `Picked.u`."""

    model = net()
    x, y = blobs(24)
    points = lk.gp.fixed(x[:9], generator=torch.Generator().manual_seed(5))

    picked = points.pick({}, None, Done).result
    coords = (picked.u * 3).floor().long()

    drawn = fit(model, loader(x, y), lk.Categorical(), lk.gp.ELLA(points, 6))
    given = fit(
        model,
        loader(x, y),
        lk.Categorical(),
        lk.gp.ELLA(lk.gp.fixed(x[:9], coords), 6),
    )

    assert len(set(coords.tolist())) > 1
    torch.testing.assert_close(
        drawn.at(x_test()).cov.per_input(), given.at(x_test()).cov.per_input()
    )


def test_from_data_fits_and_scales_by_the_data():
    model = net()
    x, y = blobs(30)

    gp = fit(model, loader(x, y), lk.Categorical(), lk.gp.ELLA(lk.gp.from_data(12), 6))
    cov = gp.at(x_test()).cov.per_input()

    assert gp.basis.shape == (sum(p.numel() for p in model.parameters()), 6)
    assert gp.gram.shape == (6, 6)
    torch.testing.assert_close(gp.n_data, torch.tensor(30.0, dtype=DT))
    assert bool(torch.isfinite(cov).all())
    assert bool((torch.linalg.eigvalsh(cov) > 0).all())


def test_reward_pairs_compose():
    """Bradley–Terry pairs fit the GP; single items give each reward's variance."""

    model = net(out=1, hidden=6)
    fn = lk.reward.pairwise(model)
    g = torch.Generator().manual_seed(4)
    xa = torch.randn(20, 2, generator=g, dtype=DT)
    xb = torch.randn(20, 2, generator=g, dtype=DT)
    labels = torch.randint(0, 2, (20,), generator=g)
    coords = torch.tensor([0, 1, 1, 0, 1, 0, 0, 1])
    data = [lk.Batch(lk.reward.pair(xa, xb), labels)]

    points = lk.gp.fixed(lk.reward.pair(xa[:8], xb[:8]), coords)
    gp = fit(fn, data, lk.reward.bradley_terry(), lk.gp.ELLA(points, 6))

    # Pair n has the Jacobian rows [∇r(aₙ), ∇r(bₙ)]: [N, 2, P].
    ja, jb = jacobian(model, xa)[:, 0], jacobian(model, xb)[:, 0]
    j = torch.stack([ja, jb], dim=1)
    v = ella_basis(j[torch.arange(8), coords], 6)

    with torch.no_grad():
        h = categorical_hessian(torch.cat([model(xa), model(xb)], dim=1))

    items = x_test()
    want = projected_cov(j @ v, h, jacobian(model, items) @ v, LAM)
    out = lk.predict_outputs(gp, items)

    assert out.mean.shape == (4, 1)
    torch.testing.assert_close(out.mean, model(items).detach())
    torch.testing.assert_close(out.cov.per_input(), want, rtol=1e-8, atol=1e-12)


def test_late_prior_and_sigma():
    """Tuning the prior precision and sigma reads no data, and matches a fresh fit there."""

    model = net(out=2, hidden=6)
    g = torch.Generator().manual_seed(2)
    x = torch.randn(30, 2, generator=g, dtype=DT)
    y = torch.stack([x[:, 0].sin(), x[:, 1].cos()], 1) + 0.1 * torch.randn(30, 2, generator=g)
    xv = torch.randn(12, 2, generator=g, dtype=DT)
    yv = torch.stack([xv[:, 0].sin(), xv[:, 1].cos()], 1)
    data = Passes(loader(x, y))
    approx = lk.gp.ELLA(lk.gp.from_data(15), 8)
    lik = copy.replace(lk.Gaussian(), sigma=torch.as_tensor(torch.tensor(1.0, dtype=DT)))

    gp = fit(model, data, lik, approx, prior=isotropic(1.0))
    iters, pulled = data.iters, data.pulled

    nll = lk.objectives.predictive_nll([(xv, yv)])
    tuned = lk.tune(gp, lk.Minimize(nll))

    assert (data.iters, data.pulled) == (iters, pulled)
    assert not torch.isclose(tuned.prior.prior_precision, gp.prior.prior_precision)
    assert not torch.isclose(tuned.likelihood.sigma, gp.likelihood.sigma)

    fresh = fit(
        model,
        loader(x, y),
        copy.replace(lk.Gaussian(), sigma=torch.as_tensor(tuned.likelihood.sigma.detach())),
        approx,
        prior=lk.priors.Isotropic(tuned.prior.prior_precision.detach()),
    )

    torch.testing.assert_close(
        tuned.at(x_test()).cov.per_input(), fresh.at(x_test()).cov.per_input()
    )


# --- refusals -----------------------------------------------------------------------------


def test_rank_above_usable_raises_before_gram_pass():
    """A Jacobian of lower rank shows only in the picked rows: after the pick pass."""

    torch.manual_seed(0)
    model = nn.Linear(2, 1).double()  # P = 3, so at most three directions
    x, _ = blobs(24)
    y = torch.zeros(24, 1, dtype=DT)
    data = Passes(loader(x, y))
    lik = copy.replace(lk.Gaussian(), sigma=torch.as_tensor(torch.tensor(1.0, dtype=DT)))

    with pytest.raises(ValueError, match=r"rank=4.*usable rank is 3") as err:
        fit(model, data, lik, lk.gp.ELLA(lk.gp.from_data(6), 4))

    assert "lower rank" in str(err.value) and "more points" in str(err.value)
    assert "another seed" in str(err.value)

    # The probe's batch, then the pick pass; never the Gram pass.
    assert data.iters == 2
    assert data.pulled == 1 + 3


@pytest.mark.parametrize(
    "points", [lk.gp.from_data(5), lk.gp.kmeans(5), lk.gp.fixed(blobs(5)[0])], ids=repr
)
def test_rank_above_the_point_count_raises_before_data(points):
    """Five points give at most five rows with one output each, 15 with all three."""

    x, y = blobs(24)
    data = Passes(loader(x, y))

    with pytest.raises(ValueError, match=r"rank=6.*at most 5 Jacobian rows"):
        fit(net(), data, lk.Categorical(), lk.gp.ELLA(points, 6))

    assert data.pulled == 1

    with pytest.raises(ValueError, match=r"rank=16.*at most 15 Jacobian rows"):
        fit(net(), data, lk.Categorical(), lk.gp.ELLA(points, 16, outputs="all"))


def test_duplicate_points_lose_rank_not_finiteness():
    model = net()
    x, y = blobs(24)
    coords = torch.tensor([0, 1, 2, 0, 1])
    dup = torch.cat([x[:5], x[:2]])
    dup_coords = torch.cat([coords, coords[:2]])

    # Two duplicated rows: the kernel has rank 5, so rank 6 is refused before data.
    data = Passes(loader(x, y))

    with pytest.raises(ValueError, match="usable rank is 5"):
        fit(model, data, lk.Categorical(), lk.gp.ELLA(lk.gp.fixed(dup, dup_coords), 6))

    assert data.iters == 1

    # At the usable rank both span the same rows, so they give the same GP.
    a = fit(model, loader(x, y), lk.Categorical(), lk.gp.ELLA(lk.gp.fixed(dup, dup_coords), 5))
    b = fit(model, loader(x, y), lk.Categorical(), lk.gp.ELLA(lk.gp.fixed(x[:5], coords), 5))
    cov = a.at(x_test()).cov.per_input()

    assert bool(torch.isfinite(cov).all())
    torch.testing.assert_close(cov, b.at(x_test()).cov.per_input(), rtol=1e-6, atol=1e-10)


def test_kmeans_duplicates_are_refused():
    """k-means snaps centres to rows; with one distinct row it picks duplicates."""

    model = net(out=1)
    x = torch.zeros(20, 2, dtype=DT)
    x[5] = 1.0
    y = torch.zeros(20, 1, dtype=DT)
    lik = copy.replace(lk.Gaussian(), sigma=torch.as_tensor(torch.tensor(1.0, dtype=DT)))

    with pytest.raises(ValueError, match="usable rank is [12]"):
        fit(model, loader(x, y, k=2), lik, lk.gp.ELLA(lk.gp.kmeans(3), 3))


@pytest.mark.parametrize("kind", ["diagonal", "per_block", "vector_isotropic", "posterior"])
def test_non_isotropic_prior_refused_before_data(kind):
    model = net()
    x, y = blobs(24)
    n_params = sum(p.numel() for p in model.parameters())

    if kind == "diagonal":
        prior = lk.priors.Diagonal(torch.ones(n_params, dtype=DT))
    elif kind == "per_block":
        prior = lk.priors.Layerwise({"0": torch.tensor(1.0, dtype=DT), "2": torch.tensor(1.0)})
    elif kind == "vector_isotropic":
        prior = lk.priors.Isotropic(torch.ones(n_params, dtype=DT))
    else:
        prior = lk.laplace(
            model,
            loader(x, y),
            likelihood=lk.Categorical(),
            subset=lk.subsets.all,
            hyper_tuner=None,
        )

    data = Passes(loader(x, y))

    with pytest.raises(StructureError, match=r"Isotropic.*SubsetOfData.*VaLLA"):
        fit(model, data, lk.Categorical(), lk.gp.ELLA(lk.gp.from_data(6), 4), prior=prior)

    assert data.pulled == 1


def test_basis_budget_before_data(monkeypatch):
    model = net()
    x, y = blobs(24)
    n_params = sum(p.numel() for p in model.parameters())  # 33

    # P rank: 33 * 4 entries.
    monkeypatch.setattr(nystrom, "MAX_BASIS_ENTRIES", 33 * 4 - 1)
    data = Passes(loader(x, y))

    with pytest.raises(StructureError, match=r"P=33.*rank=4.*MAX_BASIS_ENTRIES"):
        fit(model, data, lk.Categorical(), lk.gp.ELLA(lk.gp.from_data(6), 4))

    assert data.pulled == 1

    # The rows of given points: m k P = 6 * 3 * 33, known before data.
    monkeypatch.setattr(nystrom, "MAX_BASIS_ENTRIES", 6 * 3 * n_params - 1)
    data = Passes(loader(x, y))
    approx = lk.gp.ELLA(lk.gp.fixed(x[:6]), 4, outputs="all")

    with pytest.raises(StructureError, match=r"m=6.*k=3.*P=33.*MAX_BASIS_ENTRIES"):
        fit(model, data, lk.Categorical(), approx)

    assert data.pulled == 1

    # Points from the data: checked after the pick pass, before the Gram pass.
    data = Passes(loader(x, y))
    approx = lk.gp.ELLA(lk.gp.from_data(6), 4, outputs="all")

    with pytest.raises(StructureError, match=r"m=6.*k=3.*P=33.*MAX_BASIS_ENTRIES"):
        fit(model, data, lk.Categorical(), approx)

    assert data.iters == 2

    # Reverse mode holds one input's E P rows at a time: 3 * 33 entries.
    monkeypatch.setattr(nystrom, "MAX_BASIS_ENTRIES", 3 * n_params - 1)
    data = Passes(loader(x, y))
    approx = lk.gp.ELLA(lk.gp.fixed(x[:2]), 2, strategy="reverse-mode")

    with pytest.raises(StructureError, match=r"E=3.*P=33.*forward-mode"):
        fit(model, data, lk.Categorical(), approx)

    assert data.pulled == 1


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
        fit(NoVmap(), data, lk.Categorical(), lk.gp.ELLA(lk.gp.from_data(6), 4))

    assert data.pulled == 1

    # The pick passes ignore weights, so weighted data is refused at the first batch.
    w = torch.ones(8, dtype=DT)
    data = Passes([lk.Batch(a, b, w) for a, b in loader(x, y)])

    with pytest.raises(CapabilityError, match="weights"):
        fit(net(), data, lk.Categorical(), lk.gp.ELLA(lk.gp.from_data(6), 4))

    assert data.pulled == 1

    # Given points need no pick pass, and the Gram honours weights.
    batches = [lk.Batch(a, b, 2 * w) for a, b in loader(x, y)]
    weighted = fit(net(), batches, lk.Categorical(), lk.gp.ELLA(lk.gp.fixed(x[:6]), 4))
    torch.testing.assert_close(weighted.n_data, torch.tensor(48.0, dtype=DT))

    # No output Hessian and no way to draw targets: refused before the probe.
    def log_prob(f, y):
        return -((f[:, 0] - y) ** 2)

    data = Passes(loader(x, y))

    with pytest.raises(CapabilityError, match="output Hessian"):
        fit(
            net(),
            data,
            lk.likelihoods.from_log_prob(log_prob),
            lk.gp.ELLA(lk.gp.from_data(6), 4),
        )

    assert data.pulled == 0


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


def test_dict_outputs_with_one_likelihood_refused_at_the_basis():
    x, y = blobs(24)
    data = Passes(loader(x, y))

    # Given points: the basis is built in `plan`, before any data.
    with pytest.raises(StructureError, match="one tensor, got dict"):
        fit(Named(), data, DictCategorical(), lk.gp.ELLA(lk.gp.fixed(x[:6]), 4))

    assert data.pulled == 1


def test_dict_outputs_refused_before_data():
    x, y = blobs(24)
    data = Passes([(a, {"logits": b}) for a, b in loader(x, y)])
    lik = {"logits": lk.Categorical()}

    with pytest.raises(StructureError, match="one tensor"):
        fit(Named(), data, lik, lk.gp.ELLA(lk.gp.from_data(6), 4))

    assert data.pulled == 1


def test_nystrom_arguments():
    with pytest.raises(ValueError, match="rank >= 1"):
        lk.gp.ELLA(lk.gp.from_data(5), 0)

    with pytest.raises(ValueError, match="outputs"):
        lk.gp.ELLA(lk.gp.from_data(5), 2, outputs="every")


# --- display and staleness ----------------------------------------------------------------


class _Square(torch.autograd.Function):
    """x², with a backward and a vmap rule but no forward-mode rule."""

    generate_vmap_rule = True

    @staticmethod
    def forward(x):
        return x * x

    @staticmethod
    def setup_context(ctx, inputs, output):
        ctx.save_for_backward(inputs[0])

    @staticmethod
    def backward(ctx, g):
        (x,) = ctx.saved_tensors

        return 2 * x * g


class NoJvp(nn.Module):
    def __init__(self):
        super().__init__()
        self.inner = net()

    def forward(self, x):
        return self.inner[2](_Square.apply(self.inner[0](x)))


def test_repr_shows_resolved_via():
    model = net()
    x, y = blobs(24)
    approx = lk.gp.ELLA(lk.gp.from_data(6), 4)

    assert repr(approx) == "ELLA(rank=4, points=from_data(6), outputs='random', strategy=None)"

    gp = fit(model, loader(x, y), lk.Categorical(), approx)

    assert (
        repr(gp) == "NystromGP(rank=4, parameters=33, strategy='forward-mode', "
        "prior=Isotropic(prior_precision=2))"
    )


@pytest.mark.parametrize("rank", [2, 4, 8])
def test_default_via_is_forward_mode_when_the_probe_found_it(rank):
    """Whatever the rank against E = 3: jvp with forward mode, vjp without it."""

    x, y = blobs(24)
    approx = lk.gp.ELLA(lk.gp.from_data(9), rank)

    forward = fit(net(), loader(x, y), lk.Categorical(), approx)
    reverse = fit(NoJvp(), loader(x, y), lk.Categorical(), approx)

    assert forward.strategy == "forward-mode" and forward.requires.jvp and not forward.requires.vjp
    assert reverse.strategy == "reverse-mode" and reverse.requires.vjp and not reverse.requires.jvp

    # Both modes give the same GP.
    torch.testing.assert_close(
        forward.at(x_test()).cov.per_input(),
        copy.replace(forward, strategy="reverse-mode").at(x_test()).cov.per_input(),
    )


def test_explicit_via_is_honoured_and_a_missing_mode_refused_before_data():
    x, y = blobs(24)

    gp = fit(
        net(),
        loader(x, y),
        lk.Categorical(),
        lk.gp.ELLA(lk.gp.from_data(6), 2, strategy="reverse-mode"),
    )

    assert gp.strategy == "reverse-mode"
    assert lk.gp.ELLA(lk.gp.from_data(6), 2, strategy="forward-mode").requires.jvp

    data = Passes(loader(x, y))

    with pytest.raises(CapabilityError, match="jvp"):
        fit(
            NoJvp(),
            data,
            lk.Categorical(),
            lk.gp.ELLA(lk.gp.from_data(6), 2, strategy="forward-mode"),
        )

    assert data.pulled == 1


def test_nystrom_refuses_unknown_via():
    with pytest.raises(ValueError, match=r'"forward-mode", "reverse-mode" or None, got \'foo\''):
        lk.gp.ELLA(lk.gp.from_data(6), 2, strategy="foo")  # ty: ignore[invalid-argument-type]


def test_staleness_guard():
    model = net()
    x, y = blobs(24)
    gp = fit(
        model,
        loader(x, y),
        lk.Categorical(),
        lk.gp.ELLA(lk.gp.from_data(6), 3),
        subset=lk.subsets.last_layer,
    )
    before = gp.at(x_test())

    # The mean is the posterior's own copy: moving the fitted layer leaves the GP alone.
    with torch.no_grad():
        model[2].weight.add_(1.0)

    torch.testing.assert_close(gp.at(x_test()).cov.per_input(), before.cov.per_input())

    with torch.no_grad():
        model[0].weight.add_(1.0)

    with pytest.raises(
        RuntimeError,
        match=r"fixed parameter '0\.weight' changed in place.*the basis and the Gram used its",
    ):
        gp.at(x_test())

    tuned = copy.replace(gp, prior=isotropic(5.0))

    with pytest.raises(RuntimeError, match="fixed parameter"):
        lk.predict_outputs(tuned, x_test())


def test_posterior_needs_a_one_element_curvature_scale():
    model = net(out=2, hidden=6)
    x = torch.randn(12, 2, dtype=DT)
    y = torch.randn(12, 2, dtype=DT)
    gp = fit(
        model,
        loader(x, y),
        copy.replace(lk.Gaussian(), sigma=torch.as_tensor(torch.tensor(1.0, dtype=DT))),
        lk.gp.ELLA(lk.gp.from_data(4), 3),
    )

    @lk.core.pytree
    class TwoScales(lk.Gaussian):
        def curvature_scale(self):
            return torch.ones(2, dtype=DT)

    bad = copy.replace(gp, likelihood=TwoScales(torch.tensor(1.0, dtype=DT), learned=True))

    with pytest.raises(StructureError, match="one-element curvature_scale"):
        bad.at(x_test())

    assert math.isfinite(float(gp.at(x_test()).cov.per_input().sum()))
