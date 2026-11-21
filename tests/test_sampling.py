"""`sample_outputs`: joint draws of a posterior's outputs, against `jacrev` and explicit Σ."""

import pytest
import torch
from torch import Tensor, nn
from torch.func import functional_call, jacrev

import laplacekit as lk
from laplacekit import function_space as fs
from laplacekit.core import (
    BlockDiag,
    Capabilities,
    Dense,
    Joint,
    KronEigenbasis,
    LaplacePosterior,
    Layout,
    LowRankPlusDiag,
    ModelFn,
)
from laplacekit.priors import Isotropic

D, H, C, N = 3, 4, 3, 4
F64 = torch.float64
LAM, SCALE = 1.0, 5.0
CAPS = Capabilities(jvp=True, vjp=True, vjp_twice=True, vmap=True)
NAMES = ("l1.weight", "l2.weight")


def _gen(seed: int) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


def _derived(seed: int) -> torch.Generator:
    """The generator `sample_outputs(..., generator=_gen(seed))` draws its noise from."""

    return lk.core.spawn_generator(_gen(seed))


def forward(p, x):
    return torch.tanh(x @ p["l1.weight"].T) @ p["l2.weight"].T


def _psd(n: int, g: torch.Generator) -> Tensor:
    a = torch.randn(n, n, generator=g, dtype=F64)

    return a @ a.T / n


def _orthogonal(n: int, g: torch.Generator) -> Tensor:
    return torch.linalg.qr(torch.randn(n, n, generator=g, dtype=F64))[0]


def curvature(kind: str, g: torch.Generator):
    """A curvature operator and its dense [P, P] matrix, built from the same tensors."""

    p1, p2 = H * D, C * H

    if kind == "block_dense":
        a1, a2 = _psd(p1, g), _psd(p2, g)

        return BlockDiag({"l1": Dense(a1), "l2": Dense(a2)}), torch.block_diag(a1, a2)

    if kind == "joint_dense":
        a = _psd(p1 + p2, g)

        return Joint(Dense(a), (("l1", p1), ("l2", p2))), a

    if kind == "joint_low_rank":
        u = torch.randn(p1 + p2, 5, generator=g, dtype=F64)
        e = torch.rand(5, generator=g, dtype=F64) + 0.5
        op = LowRankPlusDiag(u, e, torch.zeros(p1 + p2, dtype=F64))

        return Joint(op, (("l1", p1), ("l2", p2))), u @ torch.diag(e) @ u.T

    # A Kronecker eigenbasis for the first layer (row-major W), a dense second one.
    qa, qg = _orthogonal(D, g), _orthogonal(H, g)
    ev = torch.rand(H, D, generator=g, dtype=F64)
    kron = KronEigenbasis(qa, qg, ev, has_bias=False, weight_shape=(H, D))
    q = torch.kron(qg, qa)
    a2 = _psd(p2, g)

    return BlockDiag({"l1": kron, "l2": Dense(a2)}), torch.block_diag(
        q @ torch.diag(ev.reshape(-1)) @ q.T, a2
    )


def post(kind: str = "joint_dense", caps: Capabilities = CAPS):
    """A hand-built posterior of the MLP, its explicit Σ, and test inputs."""

    g = torch.Generator().manual_seed(0)
    mean = {
        "l1.weight": torch.randn(H, D, generator=g, dtype=F64),
        "l2.weight": torch.randn(C, H, generator=g, dtype=F64),
    }
    op, dense = curvature(kind, g)
    p = LaplacePosterior(
        mean=mean,
        curvature=op,
        prior=Isotropic(torch.tensor(LAM, dtype=F64)),
        n_data=torch.tensor(SCALE, dtype=F64),
        model=ModelFn(forward),
        likelihood=lk.Categorical(),
        layout=Layout.by_module(mean),
        caps=caps,
    )
    sigma = torch.linalg.inv(LAM * torch.eye(dense.shape[0], dtype=F64) + SCALE * dense)
    x = torch.randn(N, D, generator=g, dtype=F64)

    return p, sigma, x


def jacobian(mean, x) -> Tensor:
    """[N, C, P] by `jacrev`, parameters in `NAMES` order."""

    jac = jacrev(lambda m: forward(m, x))(dict(mean))

    return torch.cat([jac[k].reshape(N, C, -1) for k in NAMES], dim=2)


def flat(params, names=NAMES) -> Tensor:
    """Draws [n, ...] per name to [n, P]."""

    return torch.cat([params[k].reshape(params[k].shape[0], -1) for k in names], dim=1)


def assert_moments(draws: Tensor, mean: Tensor, cov: Tensor) -> None:
    """Draws [n, K] against N(mean, cov), entry by entry within 5 standard errors.

    With the mean known, the empirical covariance entry (i, j) has standard error
    sqrt((K_ij^2 + K_ii K_jj) / n) and the empirical mean entry sqrt(K_ii / n).
    """

    n = draws.shape[0]
    d = draws - mean
    var = cov.diagonal()

    se_mean = (var / n).sqrt()
    se_cov = ((cov**2 + var[:, None] * var[None, :]) / n).sqrt()

    assert ((d.mean(0)).abs() <= 5 * se_mean).all()
    assert ((d.T @ d / n - cov).abs() <= 5 * se_cov).all()


@pytest.mark.parametrize("strategy", ["forward-mode", "reverse-mode"])
def test_linearized_samples_equal_mean_plus_jacobian_noise(strategy) -> None:
    p, _, x = post()
    got = lk.sample_outputs(p, x, 6, generator=_gen(7), strategy=strategy)
    eps = flat(p.sample_centered(6, generator=_derived(7)))
    want = forward(p.mean, x) + torch.einsum("ncp,sp->snc", jacobian(p.mean, x), eps)

    assert got.shape == (6, N, C)
    torch.testing.assert_close(got, want, rtol=0, atol=1e-10)


@pytest.mark.parametrize("kind", ["block_dense", "joint_dense", "joint_low_rank", "kron"])
def test_linearized_sample_moments_match_jacobian_covariance(kind) -> None:
    p, sigma, x = post(kind)
    n = 4000

    got = lk.sample_outputs(p, x, n, generator=torch.Generator().manual_seed(1))
    j = jacobian(p.mean, x).reshape(N * C, -1)

    # Joint over the batch: the full [N C, N C] covariance, cross-input blocks too.
    assert_moments(got.reshape(n, -1), forward(p.mean, x).reshape(-1), j @ sigma @ j.T)


def test_network_samples_equal_forward_at_weight_draws() -> None:
    p, _, x = post()
    got = lk.sample_outputs(p, x, 5, generator=_gen(3), kind="network")
    eps = p.sample_centered(5, generator=_derived(3))
    want = torch.stack([forward({k: p.mean[k] + eps[k][s] for k in NAMES}, x) for s in range(5)])

    torch.testing.assert_close(got, want, rtol=0, atol=1e-12)

    # Not the linearization: the network is nonlinear in its first layer.
    assert not torch.allclose(got, lk.sample_outputs(p, x, 5, generator=_gen(3)), atol=1e-6)


@pytest.mark.parametrize("kind", ["linearized", "network"])
def test_samples_are_joint_across_inputs(kind) -> None:
    p, _, x = post()
    twice = torch.cat([x, x])

    got = lk.sample_outputs(p, twice, 8, generator=torch.Generator().manual_seed(2), kind=kind)

    assert torch.equal(got[:, :N], got[:, N:])
    assert not torch.allclose(got[0], got[1])


def _gp(n: int = 24):
    g = torch.Generator().manual_seed(0)
    centers = torch.tensor([[-2.0, 0.0], [2.0, 0.0], [0.0, 3.0]], dtype=F64)
    y = torch.arange(n) % 3
    x = centers[y] + 0.7 * torch.randn(n, 2, generator=g, dtype=F64)

    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(2, 5), nn.Tanh(), nn.Linear(5, 3)).double()
    data = [(x[i * 8 : (i + 1) * 8], y[i * 8 : (i + 1) * 8]) for i in range(3)]
    prior = Isotropic(torch.tensor(2.0, dtype=F64))
    gp = fs.laplace(
        model, data, likelihood=lk.Categorical(), method=fs.SubsetOfData(n), prior=prior
    )

    return gp, torch.tensor([[0.0, 1.0], [3.0, -1.0], [-4.0, 4.0]], dtype=F64)


def test_function_posterior_linearized_uses_cov_sample() -> None:
    gp, x = _gp()
    got = lk.sample_outputs(gp, x, 7, generator=_gen(4))
    out = gp.at(x)
    want = out.mean + out.cov.sample(7, generator=_derived(4))

    assert got.shape == (7, 3, 3)
    torch.testing.assert_close(got, want, rtol=0, atol=0)

    with pytest.raises(ValueError, match="function-space.*strategy='forward-mode'"):
        lk.sample_outputs(gp, x, 7, strategy="forward-mode")


def test_function_posterior_network_is_type_error() -> None:
    gp, x = _gp()

    with pytest.raises(TypeError, match="function-space posteriors have no weights to sample"):
        lk.sample_outputs(gp, x, 7, kind="network")


def test_sample_outputs_seeded() -> None:
    p, _, x = post()

    for kind in ("linearized", "network"):
        a = lk.sample_outputs(p, x, 4, generator=torch.Generator().manual_seed(5), kind=kind)
        b = lk.sample_outputs(p, x, 4, generator=torch.Generator().manual_seed(5), kind=kind)
        c = lk.sample_outputs(p, x, 4, generator=torch.Generator().manual_seed(6), kind=kind)
        torch.manual_seed(1)
        default = lk.sample_outputs(p, x, 4, kind=kind)
        torch.manual_seed(1)
        again = lk.sample_outputs(p, x, 4, kind=kind)

        assert torch.equal(a, b)
        assert not torch.allclose(a, c)
        assert torch.equal(default, again)  # `torch.manual_seed` repeats the draws


def test_refusals_come_before_the_forward() -> None:
    calls = []

    def counted(p, x):
        calls.append(1)
        return forward(p, x)

    p, _, x = post(caps=Capabilities(vjp=True, vmap=True))  # no jvp, no vjp_twice
    p = LaplacePosterior(
        mean=p.mean,
        curvature=p.curvature,
        prior=p.prior,
        n_data=p.n_data,
        model=ModelFn(counted),
        likelihood=p.likelihood,
        layout=p.layout,
        caps=p.caps,
    )

    with pytest.raises(ValueError, match="n_samples >= 1"):
        lk.sample_outputs(p, x, 0)

    with pytest.raises(ValueError, match="kind"):
        lk.sample_outputs(p, x, 2, kind="glm")  # ty: ignore[invalid-argument-type] -- the runtime check

    with pytest.raises(lk.CapabilityError, match="vjp_twice"):
        lk.sample_outputs(p, x, 2)

    with pytest.raises(lk.CapabilityError, match="jvp"):
        lk.sample_outputs(p, x, 2, strategy="forward-mode")

    with pytest.raises(ValueError, match="strategy"):
        lk.sample_outputs(p, x, 2, kind="network", strategy="reverse-mode")

    assert calls == []

    # Network samples need forward passes only.
    assert lk.sample_outputs(p, x, 2, kind="network").shape == (2, N, C)


# --- through the front door: a `Selection` with a low-rank curvature -----------------------


def _selection_case():
    g = torch.Generator().manual_seed(0)
    centers = torch.tensor([[-2.0, 0.0], [2.0, 0.0], [0.0, 3.0]], dtype=F64)
    y = torch.arange(24) % 3
    x = centers[y] + 0.7 * torch.randn(24, 2, generator=g, dtype=F64)

    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(2, 6), nn.Tanh(), nn.Linear(6, 3)).double()
    params = dict(model.named_parameters())
    w = torch.rand(params["2.weight"].shape, generator=g) < 0.5
    w[0, 0] = True
    sel = lk.core.Selection.from_masks(params, {"2.weight": w, "0.bias": torch.ones(6, dtype=bool)})
    data = [(x[i * 8 : (i + 1) * 8], y[i * 8 : (i + 1) * 8]) for i in range(3)]
    p = lk.laplace(
        model,
        data,
        likelihood=lk.Categorical(),
        subset=sel,
        hessian_approx=lk.LowRankGGN(rank=4, oversample=4),
        prior=Isotropic(torch.tensor(0.5, dtype=F64)),
        hyper_tuner=None,
    )

    return model, sel, p, x[:5]


def _at_selected(model, sel, x, values):
    """The model's outputs with the selected entries set to `values` (name -> [k])."""

    base = {k: v.detach() for k, v in model.named_parameters()}
    params = dict(base)

    for name, idx in sel.index.items():
        params[name] = base[name].flatten().index_put((idx,), values[name]).reshape_as(base[name])

    return functional_call(model, params, (x,))


def test_selection_and_low_rank_posteriors() -> None:
    model, sel, p, x = _selection_case()
    names = list(sel.index)
    picked = [f"{k}[idx]" for k in names]
    m = {k: p.mean[f"{k}[idx]"] for k in names}

    assert isinstance(p.curvature, Joint) and isinstance(p.curvature.op, LowRankPlusDiag)

    # Explicit Σ from the precision's own tensors (the selected entries in layout order).
    op = p.precision().op
    assert isinstance(op, LowRankPlusDiag)
    sigma = torch.linalg.inv(op.factor @ torch.diag(op.evals) @ op.factor.T + torch.diag(op.diag))

    jac = jacrev(lambda v: _at_selected(model, sel, x, v))(m)
    j = torch.cat([jac[k].reshape(5 * 3, -1) for k in names], dim=1)
    f = _at_selected(model, sel, x, m)

    # Linearized: equal to f + J ε for the posterior's own noise, and with the moments of
    # N(f, J Σ Jᵀ).
    eps = flat(p.sample_centered(3, generator=_derived(8)), picked)
    got = lk.sample_outputs(p, x, 3, generator=_gen(8))

    torch.testing.assert_close(got.reshape(3, -1), f.reshape(-1) + eps @ j.T, rtol=0, atol=1e-10)

    n = 4000
    many = lk.sample_outputs(p, x, n, generator=torch.Generator().manual_seed(9))
    assert_moments(many.reshape(n, -1), f.reshape(-1), j @ sigma @ j.T)

    # Network: the forward with the selected entries moved, the rest fixed.
    net = lk.sample_outputs(p, x, 3, generator=_gen(8), kind="network")
    noise = p.sample_centered(3, generator=_derived(8))
    want = torch.stack(
        [
            _at_selected(model, sel, x, {k: m[k] + noise[f"{k}[idx]"][s] for k in names})
            for s in range(3)
        ]
    )

    torch.testing.assert_close(net, want, rtol=0, atol=1e-12)


@pytest.mark.parametrize("kind", ["linearized", "network"])
def test_non_tensor_outputs_refused_on_the_first_forward(kind) -> None:
    calls = []

    def as_dict(p, x):
        calls.append(1)
        return {"logits": forward(p, x)}

    p, _, x = post()
    p = LaplacePosterior(
        mean=p.mean,
        curvature=p.curvature,
        prior=p.prior,
        n_data=p.n_data,
        model=ModelFn(as_dict),
        likelihood=p.likelihood,
        layout=p.layout,
        caps=p.caps,
    )

    with pytest.raises(lk.StructureError, match="one tensor, got dict"):
        lk.sample_outputs(p, x, 50, kind=kind)

    assert len(calls) == 1
