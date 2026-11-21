"""`lk.gp` FSP-Laplace: the function-space prior, its RKHS norm and the posterior, against
dense references."""

import copy
import math

import laplacekit_gp.points
import pytest
import torch
from laplacekit_gp._gram import ProjectedGram
from laplacekit_gp._lanczos import lanczos, pinv_root
from laplacekit_gp.fsp import FSPPosterior, RangePrecision, fsp_basis
from torch import nn
from torch.func import functional_call, jacrev

import laplacekit as lk
import laplacekit.core as core
from laplacekit.core import Capabilities, FitContext, Layout, StructureError


def points(n: int, d: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)

    return torch.randn(n, d, generator=g, dtype=torch.float64)


def rbf_gram(x: torch.Tensor, ell: float, var: float) -> torch.Tensor:
    """The RBF Gram matrix by explicit loops."""

    n = x.shape[0]
    k = torch.empty(n, n, dtype=torch.float64)

    for i in range(n):
        for j in range(n):
            k[i, j] = var * math.exp(-float(((x[i] - x[j]) ** 2).sum()) / (2 * ell**2))

    return k


def prior(ell=0.8, var=1.5, **kw) -> lk.gp.FunctionPrior:
    kernel = lk.gp.RBF(
        torch.tensor(ell, dtype=torch.float64), torch.tensor(var, dtype=torch.float64)
    )

    return lk.gp.FunctionPrior(kernel, **kw)


def test_cov_adds_relative_jitter():
    x = points(6, 2, seed=0)
    k = rbf_gram(x, 0.8, 1.5)

    for jitter in (1e-6, 1e-2):
        want = k + jitter * k.diagonal().mean() * torch.eye(6, dtype=torch.float64)
        torch.testing.assert_close(prior(jitter=jitter).cov(x), want, rtol=1e-12, atol=0)

    assert prior().jitter == 1e-6
    assert prior().mean is None


def test_rkhs_norm_equals_dense_solve():
    x = points(9, 2, seed=1)
    g = torch.Generator().manual_seed(2)
    f = torch.randn(9, 3, generator=g, dtype=torch.float64)

    def mean(z: torch.Tensor) -> torch.Tensor:
        return torch.stack([z[:, 0], z[:, 1] ** 2, torch.sin(z.sum(1))], dim=1)

    k = rbf_gram(x, 0.8, 1.5)
    k = k + 1e-6 * k.diagonal().mean() * torch.eye(9, dtype=torch.float64)

    # Zero mean, then a given mean; one solve per output.
    for p, mu in ((prior(), torch.zeros(9, 3, dtype=torch.float64)), (prior(mean=mean), mean(x))):
        d = f - mu
        want = sum(d[:, e] @ torch.linalg.solve(k, d[:, e]) for e in range(3))

        torch.testing.assert_close(lk.gp.rkhs_norm(p, x, f), want, rtol=1e-8, atol=0)

    # One output as a vector.
    d = f[:, 0]
    want = d @ torch.linalg.solve(k, d)
    torch.testing.assert_close(lk.gp.rkhs_norm(prior(), x, d), want, rtol=1e-8, atol=0)


def test_rkhs_norm_is_interpolant_norm():
    x = points(8, 1, seed=3)
    kernel = lk.gp.Matern("5/2", torch.tensor(0.5, dtype=torch.float64))
    k = kernel(x, x)
    alpha = torch.randn(8, generator=torch.Generator().manual_seed(4), dtype=torch.float64)
    f = k @ alpha

    got = lk.gp.rkhs_norm(lk.gp.FunctionPrior(kernel, jitter=1e-10), x, f)

    torch.testing.assert_close(got, alpha @ k @ alpha, rtol=1e-6, atol=0)


def test_rkhs_norm_is_differentiable():
    x = points(7, 2, seed=5)
    ell = torch.tensor(0.9, dtype=torch.float64, requires_grad=True)
    f = torch.randn(7, 2, generator=torch.Generator().manual_seed(6), dtype=torch.float64)
    f.requires_grad_(True)

    p = lk.gp.FunctionPrior(lk.gp.RBF(ell))
    df, dell = torch.autograd.grad(lk.gp.rkhs_norm(p, x, f), (f, ell))

    # d/df fᵀK⁻¹f = 2 K⁻¹ f; d/dl by a central difference on the dense solve.
    def dense(length: float) -> torch.Tensor:
        k = rbf_gram(x, length, 1.0)
        k = k + 1e-6 * k.diagonal().mean() * torch.eye(7, dtype=torch.float64)
        fd = f.detach()

        return (fd * torch.linalg.solve(k, fd)).sum()

    k = rbf_gram(x, 0.9, 1.0)
    k = k + 1e-6 * k.diagonal().mean() * torch.eye(7, dtype=torch.float64)
    h = 1e-6

    torch.testing.assert_close(df, 2 * torch.linalg.solve(k, f.detach()), rtol=1e-7, atol=0)
    torch.testing.assert_close(dell, (dense(0.9 + h) - dense(0.9 - h)) / (2 * h), rtol=1e-5, atol=0)


def test_rkhs_norm_refuses_bad_inputs():
    x = points(5, 2, seed=7)
    f = torch.zeros(5, 2, dtype=torch.float64)

    with pytest.raises(StructureError, match="tensor"):
        lk.gp.rkhs_norm(prior(), {"x": x}, f)  # ty: ignore[invalid-argument-type]  # the refusal

    with pytest.raises(StructureError, match="5"):
        lk.gp.rkhs_norm(prior(), x, torch.zeros(4, 2, dtype=torch.float64))

    with pytest.raises(StructureError, match="mean"):
        lk.gp.rkhs_norm(prior(mean=lambda z: z[:, :1]), x, torch.zeros(5, 3, dtype=torch.float64))


def test_function_prior_is_a_pytree_with_kernel_leaves():
    p = prior(mean=torch.sin, jitter=1e-4)
    leaves = lk.core.tree_leaves(p)

    assert len(leaves) == 2
    assert all(isinstance(t, torch.Tensor) for t in leaves)

    doubled = lk.core.tree_map(lambda t: 2 * t, p)
    assert isinstance(doubled, lk.gp.FunctionPrior)
    assert doubled.mean is torch.sin
    assert doubled.jitter == 1e-4


# --- the posterior: Lanczos, the FSP basis and FSPPosterior ----------------------------------

DT = torch.float64
SIGMA = 0.1


def spd(n: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    a = torch.randn(n, n, generator=g, dtype=DT)

    return a @ a.T + n * torch.eye(n, dtype=DT)


def test_lanczos_full_steps_inverts():
    k = spd(12, seed=10)
    v0 = torch.randn(12, generator=torch.Generator().manual_seed(11), dtype=DT)

    q, t = lanczos(lambda v: k @ v, v0, 12)
    inv = torch.linalg.inv(k)

    assert q.shape == (12, 12) and t.shape == (12, 12)
    torch.testing.assert_close(q @ torch.linalg.solve(t, q.T), inv, rtol=1e-8, atol=1e-12)

    root = pinv_root(q, t)
    torch.testing.assert_close(root @ root.T, inv, rtol=1e-8, atol=1e-12)


def test_lanczos_orthonormal_tridiagonal():
    k = spd(15, seed=12)
    v0 = torch.randn(15, generator=torch.Generator().manual_seed(13), dtype=DT)

    q, t = lanczos(lambda v: k @ v, v0, 6)

    assert q.shape == (15, 6)
    torch.testing.assert_close(q.T @ q, torch.eye(6, dtype=DT), rtol=0, atol=1e-12)
    torch.testing.assert_close(q.T @ k @ q, t, rtol=0, atol=1e-10)
    assert torch.equal(t, t.T)
    assert torch.equal(t.triu(2), torch.zeros_like(t))
    torch.testing.assert_close(q[:, 0], v0 / v0.norm(), rtol=0, atol=1e-14)

    # Three distinct eigenvalues: the Krylov space has dimension 3, so it stops there.
    w, _ = torch.linalg.qr(torch.randn(9, 9, generator=torch.Generator().manual_seed(14), dtype=DT))
    k3 = w @ torch.diag(torch.tensor([1.0, 2.0, 5.0], dtype=DT).repeat(3)) @ w.T
    q3, t3 = lanczos(lambda v: k3 @ v, torch.ones(9, dtype=DT), 9)

    assert q3.shape == (9, 3) and t3.shape == (3, 3)
    torch.testing.assert_close(torch.linalg.eigvalsh(t3), torch.tensor([1.0, 2.0, 5.0], dtype=DT))


def toy_net(hidden: int = 6, out: int = 1, seed: int = 0) -> nn.Module:
    torch.manual_seed(seed)

    return nn.Sequential(
        nn.Linear(1, hidden),
        nn.Tanh(),
        nn.Linear(hidden, hidden),
        nn.Tanh(),
        nn.Linear(hidden, out),
    ).double()


def params_of(net: nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.detach().clone() for k, v in net.named_parameters()}


def flat_jacobian(net: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """[N E, P] by `jacrev`, rows input-major (row n E + e)."""

    params = params_of(net)
    jac = jacrev(lambda p: functional_call(net, p, (x,)).reshape(-1))(params)

    return torch.cat([jac[k].reshape(jac[k].shape[0], -1) for k in params], dim=1)


def toy_data(n: int = 40, seed: int = 20) -> tuple[torch.Tensor, torch.Tensor]:
    """Appendix B.1's toy: sin 2πx plus noise on [-1, -0.5] and [0.5, 1]."""

    g = torch.Generator().manual_seed(seed)
    u = torch.rand(n, generator=g, dtype=DT) * 0.5
    x = torch.cat([u[: n // 2] - 1.0, u[n // 2 :] + 0.5]).unsqueeze(-1)
    y = torch.sin(2 * math.pi * x) + SIGMA * torch.randn(x.shape, generator=g, dtype=DT)

    return x, y


def fold_gram(net: nn.Module, x: torch.Tensor, y: torch.Tensor, basis: torch.Tensor):
    """`ProjectedGram` folded over two batches with the unit Gaussian, then finalized."""

    params = params_of(net)
    ctx = FitContext(
        core.ModelFn.from_module(net),
        copy.replace(lk.Gaussian(), sigma=torch.as_tensor(torch.tensor(1.0))),
        Layout.by_module(params),
        Capabilities(),
    )
    gram = ProjectedGram(basis, "reverse-mode")
    stat = gram.init(params, ctx)
    half = x.shape[0] // 2

    for i, s in enumerate((slice(0, half), slice(half, None))):
        stat = gram.update(
            stat, params, lk.Batch(x[s], y[s]), torch.Generator().manual_seed(i), ctx
        )

    return gram.finalize(stat)


def build(net, basis, prior_evals, gram, scale, context_jac, prior_var, **kw) -> FSPPosterior:
    mean = kw.pop("mean", params_of(net))
    fn = core.ModelFn.from_module(net)
    lik = kw.pop(
        "likelihood",
        copy.replace(lk.Gaussian(), sigma=torch.as_tensor(torch.tensor(SIGMA, dtype=DT))),
    )
    caps = kw.pop("caps", Capabilities())

    return FSPPosterior(
        model=fn,
        likelihood=lik,
        mean=mean,
        caps=caps,
        layout=Layout.by_module(mean),
        basis=basis,
        prior_evals=prior_evals,
        gram=gram,
        n_data=scale,
        context_jac=context_jac,
        prior_var=prior_var,
        kernel="RBF",
        strategy="reverse-mode",
        **kw,
    )


def toy(n_c: int = 8, ell: float = 1.0):
    """The dense-reference toy: an untrained 1-6-6-1 tanh net (P = 61) and 8 context points."""

    net = toy_net()
    x, y = toy_data()
    c = torch.linspace(-2, 2, n_c, dtype=DT).unsqueeze(-1)
    p = prior(ell=ell, var=1.0)
    jac_c = flat_jacobian(net, c)
    fb = fsp_basis(jac_c, p, c, 1, n_c)
    stat = fold_gram(net, x, y, fb.basis)
    caps = core.probe(core.ModelFn.from_module(net), lk.Gaussian(), lk.Batch(x, y))
    post = build(
        net,
        fb.basis,
        fb.prior_evals,
        stat.gram,
        stat.weight,
        fb.context_jac,
        fb.prior_var,
        caps=caps,
    )
    k = rbf_gram(c, ell, 1.0)
    k = k + 1e-6 * k.diagonal().mean() * torch.eye(n_c, dtype=DT)

    return net, x, c, jac_c, k, fb, post


def truncated_root(u: torch.Tensor, a: torch.Tensor, jac_c: torch.Tensor, var: torch.Tensor):
    """S = U U_A D_A^{-1/2} over A's eigenvalues above eps r max, ascending, then the first
    k columns dropped, with k the smallest such that diag J_C S Sᵀ J_Cᵀ <= var (1 + 100 eps).
    Returns S, k."""

    d, w = torch.linalg.eigh((a + a.T) / 2)
    keep = d > torch.finfo(DT).eps * a.shape[0] * d.max()
    s = u @ w[:, keep] / d[keep].sqrt()
    bound = var * (1 + 100 * torch.finfo(DT).eps)
    k = 0

    while k < s.shape[1] and not bool((((jac_c @ s[:, k:]) ** 2).sum(1) <= bound).all()):
        k += 1

    return s[:, k:], k


def as_params(net: nn.Module, v: torch.Tensor) -> dict[str, torch.Tensor]:
    out, i = {}, 0

    for name, p in net.named_parameters():
        out[name] = v[i : i + p.numel()].reshape(p.shape)
        i += p.numel()

    return out


def flat(params: dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.cat([v.reshape(-1) for v in params.values()])


def test_fsp_equals_dense_reference():
    net, x, c, jac_c, k, fb, post = toy()
    n_params = jac_c.shape[1]

    # Conditioning guard: the reference and FSP agree only on a well-conditioned D_M².
    assert float(fb.prior_evals.min() / fb.prior_evals.max()) >= 1e-7

    # Reference: the range basis of P_prior = J_Cᵀ K⁻¹ J_C, the posterior on it.
    p_prior = jac_c.T @ torch.linalg.solve(k, jac_c)
    evals, evecs = torch.linalg.eigh(p_prior)
    u = evecs[:, -c.shape[0] :]
    assert fb.basis.shape[1] == u.shape[1] == 8

    jac_x = flat_jacobian(net, x)
    ggn = jac_x.T @ jac_x / SIGMA**2
    a = u.T @ p_prior @ u + u.T @ ggn @ u
    s, _ = truncated_root(u, a, jac_c, k.diagonal())
    sigma = s @ s.T

    g = torch.Generator().manual_seed(21)

    for _ in range(3):
        v = torch.randn(n_params, generator=g, dtype=DT)
        got = flat(post.cov_mv(as_params(net, v)))
        torch.testing.assert_close(
            got, sigma @ v, rtol=1e-6, atol=1e-6 * float((sigma @ v).abs().max())
        )

    # J(x) Σ J(x)ᵀ at test inputs, by cov_mv on the rows and by the exact pushforward.
    xs = torch.linspace(-2.5, 2.5, 9, dtype=DT).unsqueeze(-1)
    jac_s = flat_jacobian(net, xs)
    want = jac_s @ sigma @ jac_s.T
    got = torch.stack([jac_s @ flat(post.cov_mv(as_params(net, row))) for row in jac_s])

    torch.testing.assert_close(got, want, rtol=1e-6, atol=1e-6 * float(want.abs().max()))

    out = lk.predict_outputs(post, xs, strategy="reverse-mode", n_samples="exact")
    torch.testing.assert_close(out.cov.diagonal().reshape(-1), want.diagonal(), rtol=1e-6, atol=0)

    root = lk.predict_outputs(post, xs, strategy="forward-mode", n_samples="exact")
    torch.testing.assert_close(root.cov.diagonal().reshape(-1), want.diagonal(), rtol=1e-6, atol=0)


def test_multi_output_full_rank():
    torch.manual_seed(1)
    net = nn.Sequential(nn.Linear(1, 6), nn.Tanh(), nn.Linear(6, 3)).double()
    c = torch.linspace(-1.5, 1.5, 5, dtype=DT).unsqueeze(-1)
    jac_c = flat_jacobian(net, c)  # [15, 33]
    assert jac_c.shape == (15, 33)

    fb = fsp_basis(jac_c, prior(ell=1.0, var=1.0), c, 3, 5)

    k = rbf_gram(c, 1.0, 1.0)
    k = k + 1e-6 * k.diagonal().mean() * torch.eye(5, dtype=DT)
    want = jac_c.T @ torch.kron(torch.linalg.inv(k).contiguous(), torch.eye(3, dtype=DT)) @ jac_c
    m_mt = fb.basis @ torch.diag(fb.prior_evals) @ fb.basis.T

    assert fb.basis.shape == (33, 15)
    torch.testing.assert_close(m_mt, want, rtol=1e-8, atol=1e-8 * float(want.abs().max()))
    torch.testing.assert_close(fb.context_jac, jac_c @ fb.basis)
    torch.testing.assert_close(fb.prior_var, k.diagonal().repeat_interleave(3))


def test_fsp_basis_refuses_bad_inputs():
    c = torch.linspace(-1, 1, 4, dtype=DT).unsqueeze(-1)

    with pytest.raises(ValueError, match="zero"):
        fsp_basis(torch.zeros(4, 7, dtype=DT), prior(), c, 1, 4)

    with pytest.raises(StructureError, match="rows"):
        fsp_basis(torch.ones(6, 7, dtype=DT), prior(), c, 1, 4)

    with pytest.raises(ValueError, match="rank"):
        fsp_basis(torch.ones(4, 7, dtype=DT), prior(), c, 1, 0)


def hand_built(net: nn.Module, r: int, d: torch.Tensor, a: torch.Tensor, seed: int = 30, **kw):
    """An FSPPosterior with a random orthonormal basis and the gram for which
    diag(d) + scale c gram = a (scale 4, σ = 1 so c = 1). The prior variance at C is
    infinite, so the truncation never fires."""

    n_params = sum(p.numel() for p in kw.get("mean", dict(net.named_parameters())).values())
    g = torch.Generator().manual_seed(seed)
    u, _ = torch.linalg.qr(torch.randn(n_params, r, generator=g, dtype=DT))
    scale = torch.tensor(4.0, dtype=DT)
    gram = (a - torch.diag(d)) / scale
    jac = torch.randn(3, r, generator=g, dtype=DT)
    lik = copy.replace(lk.Gaussian(), sigma=torch.as_tensor(torch.tensor(1.0, dtype=DT)))
    var = torch.full((3,), math.inf, dtype=DT)

    return u, build(net, u, d, gram, scale, jac, var, likelihood=lik, **kw)


def test_nonpositive_A_eigenvalues_dropped():
    torch.manual_seed(2)
    net = nn.Linear(3, 2).double()  # P = 8

    # A = diag(0, 0.5, 2, 3) exactly (dyadic entries): its first direction, U e₁, is free.
    lam = torch.tensor([0.0, 0.5, 2.0, 3.0], dtype=DT)
    u, post = hand_built(net, 4, torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=DT), torch.diag(lam))

    # The test's pseudo-inverse: the zero direction dropped.
    sigma = u[:, 1:] @ torch.diag(1 / lam[1:]) @ u[:, 1:].T
    eye = torch.eye(8, dtype=DT)
    got = torch.stack([flat(post.cov_mv(as_params(net, e))) for e in eye])

    assert torch.isfinite(got).all()
    torch.testing.assert_close(got, sigma, rtol=1e-8, atol=1e-10)
    zero = torch.zeros(8, dtype=DT)
    torch.testing.assert_close(flat(post.cov_mv(as_params(net, u[:, 0]))), zero, rtol=0, atol=1e-10)
    assert "rank=3 of 4" in repr(post)

    # The square root, the marginal variance and the noise agree with Σ.
    roots = torch.stack([flat(post.cov_sqrt_mv(as_params(net, e))) for e in eye], dim=1)
    torch.testing.assert_close(roots @ roots.T, sigma, rtol=1e-8, atol=1e-10)
    torch.testing.assert_close(flat(post.variance), sigma.diagonal(), rtol=1e-8, atol=0)

    noise = post.sample_centered(20000, generator=torch.Generator().manual_seed(3))
    z = torch.cat([v.reshape(20000, -1) for v in noise.values()], dim=1)
    assert z.dtype == DT
    torch.testing.assert_close(z.T @ z / 20000, sigma, rtol=0, atol=0.05 * float(sigma.abs().max()))

    draws = post.sample(20000, generator=torch.Generator().manual_seed(3))
    for name, p in net.named_parameters():
        torch.testing.assert_close(draws[name], p.detach() + noise[name])


def test_truncation_bounds_prior_variance():
    net, x, c, jac_c, k, fb, post = toy()

    # Full rank and well conditioned: nothing is dropped, and the bound holds.
    var = (
        lk.predict_outputs(post, c, strategy="reverse-mode", n_samples="exact")
        .cov.diagonal()
        .reshape(-1)
    )
    assert bool((var <= k.diagonal() + 1e-10).all())
    assert "rank=8 of 8" in repr(post)

    # Weak data (σ = 100) and one prior eigenvalue shrunk by 1e-8: the untruncated
    # variance at C exceeds the prior.
    d = fb.prior_evals.clone()
    d[0] = d[0] * 1e-8
    weak = copy.replace(lk.Gaussian(), sigma=torch.as_tensor(torch.tensor(100.0, dtype=DT)))
    broken = copy.replace(post, prior_evals=d, likelihood=weak)
    a = torch.diag(d) + post.n_data / 100.0**2 * post.gram

    dd, w = torch.linalg.eigh(a)
    s_full = fb.basis @ w / dd.sqrt()
    assert not bool((((jac_c @ s_full) ** 2).sum(1) <= k.diagonal()).all())

    s, dropped = truncated_root(fb.basis, a, jac_c, k.diagonal())
    assert dropped == 3
    assert f"rank={8 - dropped} of 8" in repr(broken)

    v = torch.randn(jac_c.shape[1], generator=torch.Generator().manual_seed(40), dtype=DT)
    want = s @ (s.T @ v)
    got = flat(broken.cov_mv(as_params(net, v)))
    torch.testing.assert_close(got, want, rtol=1e-6, atol=1e-6 * float(want.abs().max()))

    var = (
        lk.predict_outputs(broken, c, strategy="reverse-mode", n_samples="exact")
        .cov.diagonal()
        .reshape(-1)
    )
    assert bool((var <= k.diagonal() + 1e-10).all())


def test_truncation_ignores_rounding_at_the_prior_variance():
    """Where the data barely reaches, the variance at C equals the prior's up to rounding.
    No column may be dropped for that."""

    net, x, c, jac_c, k, fb, post = toy(ell=0.3)
    weak = copy.replace(
        post,
        likelihood=copy.replace(lk.Gaussian(), sigma=torch.as_tensor(torch.tensor(1e6, dtype=DT))),
    )

    # Dense reference, untruncated: J_C U A⁻¹ Uᵀ J_Cᵀ on P_prior's range.
    p_prior = jac_c.T @ torch.linalg.solve(k, jac_c)
    u = torch.linalg.eigh(p_prior)[1][:, -8:]
    jac_x = flat_jacobian(net, x)
    a = u.T @ p_prior @ u + u.T @ jac_x.T @ jac_x @ u / 1e12
    want = (jac_c @ u @ torch.linalg.solve(a, u.T @ jac_c.T)).diagonal()

    assert fb.basis.shape[1] == 8
    assert "rank=8 of 8" in repr(weak)

    got = (
        lk.predict_outputs(weak, c, strategy="reverse-mode", n_samples="exact")
        .cov.diagonal()
        .reshape(-1)
    )
    torch.testing.assert_close(got, want, rtol=1e-6, atol=0)
    torch.testing.assert_close(got, k.diagonal(), rtol=1e-3, atol=0)


def test_range_precision_refuses_inverse_ops():
    net, x, c, jac_c, k, fb, post = toy()
    n_params = jac_c.shape[1]
    prec = post.precision()

    assert isinstance(prec, RangePrecision)
    assert isinstance(prec, core.Operator)

    v = torch.randn(n_params, generator=torch.Generator().manual_seed(50), dtype=DT)

    for call in (
        lambda: prec.solve(v),
        prec.inverse_diagonal,
        prec.logdet,
        lambda: prec.cov_sqrt_mv(v),
        lambda: prec.add_isotropic(torch.tensor(1.0, dtype=DT)),
        lambda: prec.add_diagonal(torch.ones(n_params, dtype=DT)),
        lambda: prec.add_same(prec),
    ):
        with pytest.raises(StructureError, match="degenerate precision of rank 8 < 61"):
            call()

    # On the range, mv is the pseudo-inverse of Σ (built densely from cov_mv's columns).
    sigma = torch.stack(
        [flat(post.cov_mv(as_params(net, e))) for e in torch.eye(n_params, dtype=DT)]
    )
    evals, evecs = torch.linalg.eigh((sigma + sigma.T) / 2)
    top = evecs[:, -8:]
    pinv = top @ torch.diag(1 / evals[-8:]) @ top.T
    in_range = top @ (top.T @ v)

    torch.testing.assert_close(
        prec.mv(in_range),
        pinv @ in_range,
        rtol=1e-6,
        atol=1e-6 * float((pinv @ in_range).abs().max()),
    )
    torch.testing.assert_close(
        prec.diagonal(), pinv.diagonal(), rtol=1e-6, atol=1e-6 * float(pinv.diagonal().max())
    )

    half = prec.scaled(torch.tensor(0.5, dtype=DT))
    assert isinstance(half, RangePrecision)
    torch.testing.assert_close(half.mv(in_range), 0.5 * prec.mv(in_range))


def test_degenerate_refusals():
    net, x, c, jac_c, k, fb, post = toy()
    params = params_of(net)

    assert isinstance(post, core.WeightPosterior)
    assert not isinstance(post, core.FunctionPosterior)
    assert post.location(params) is post.mean

    with pytest.raises(StructureError, match="degenerate"):
        post.log_prob(params)

    with pytest.raises(StructureError, match="degenerate"):
        post.add_precision_to(core.Diag(torch.ones(61, dtype=DT)))

    with pytest.raises(StructureError):
        post.location({k: v.float() for k, v in params.items()})


def test_staleness_through_cov_mv():
    torch.manual_seed(3)
    net = nn.Sequential(nn.Linear(2, 3), nn.Tanh(), nn.Linear(3, 2)).double()
    mean = {k: v.detach().clone() for k, v in net.named_parameters() if k.startswith("2.")}
    versions = {k: v._version for k, v in net.named_parameters() if k.startswith("0.")}
    lam = torch.tensor([1.0, 2.0, 3.0], dtype=DT)
    _, post = hand_built(net, 3, lam, torch.diag(2 * lam), mean=mean, versions=versions)
    v = {k: torch.ones_like(t) for k, t in mean.items()}

    post.cov_mv(v)

    with torch.no_grad():
        net[0].weight.add_(0.0)

    for call in (
        lambda: post.cov_mv(v),
        lambda: post.cov_sqrt_mv(v),
        lambda: post.sample_centered(2, generator=torch.Generator().manual_seed(0)),
        lambda: post.sample(2, generator=torch.Generator().manual_seed(0)),
    ):
        with pytest.raises(
            RuntimeError,
            match=r"'0\.weight' changed in place.*the Gram matrix used its old value",
        ):
            call()


# --- the front door: fsp_laplace ------------------------------------------------------------


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


def halves(x: torch.Tensor, y: torch.Tensor) -> list[tuple[torch.Tensor, torch.Tensor]]:
    half = x.shape[0] // 2

    return [(x[:half], y[:half]), (x[half:], y[half:])]


def context8() -> lk.gp.Points:
    """linspace(-2, 2, 8), the toy's context points."""

    return lk.gp.grid(torch.tensor(-2.0, dtype=DT), torch.tensor(2.0, dtype=DT), 8)


def front(net, data, sigma: float = SIGMA, **kw) -> FSPPosterior:
    kw.setdefault("prior", prior(ell=1.0, var=1.0))
    kw.setdefault("context", context8())
    kw.setdefault("rank", 8)
    kw.setdefault("subset", lk.subsets.all)

    return lk.gp.fsp_laplace(
        net,
        data,
        likelihood=copy.replace(
            lk.Gaussian(), sigma=torch.as_tensor(torch.tensor(sigma, dtype=DT))
        ),
        **kw,
    )


def test_mean_outputs_guards_plugin_posterior():
    """`lk.core.mean_outputs` and `lk.predict_outputs` check a plugin posterior's versions
    before the forward: an in-place edit of a fixed parameter raises."""

    net = toy_net()
    x, y = toy_data()
    post = lk.gp.fsp_laplace(
        net,
        halves(x, y),
        likelihood=copy.replace(
            lk.Gaussian(), sigma=torch.as_tensor(torch.tensor(SIGMA, dtype=DT))
        ),
        prior=prior(),
        context=context8(),
        rank=8,
    )

    assert isinstance(post, core.HasVersions)
    lk.core.mean_outputs(post, x)

    with torch.no_grad():
        net[0].weight.add_(0.0)

    stale = r"fixed parameter '0\.weight' changed in place.*the Gram matrix used its old value"

    with pytest.raises(RuntimeError, match=stale):
        lk.core.mean_outputs(post, x)

    with pytest.raises(RuntimeError, match=stale):
        lk.predict_outputs(post, x)


def dense_output_cov(net, x, c, xs, sigma: float = SIGMA, ell: float = 1.0) -> torch.Tensor:
    """J(xs) Σ J(xs)ᵀ from `jacrev`, an explicit RBF Gram matrix and `torch.linalg`: the range
    of P_prior = J_Cᵀ K⁻¹ J_C, A = Uᵀ (P_prior + G) U with the dense GGN, then the truncation."""

    k = rbf_gram(c, ell, 1.0)
    k = k + 1e-6 * k.diagonal().mean() * torch.eye(c.shape[0], dtype=DT)
    jac_c = flat_jacobian(net, c)
    p_prior = jac_c.T @ torch.linalg.solve(k, jac_c)
    evals, evecs = torch.linalg.eigh(p_prior)
    u = evecs[:, -c.shape[0] :]

    # Conditioning guard: the reference and FSP agree only on a well-conditioned D_M².
    top = evals[-c.shape[0] :]
    assert float(top.min() / top.max()) >= 1e-7

    jac_x = flat_jacobian(net, x)
    a = u.T @ p_prior @ u + u.T @ (jac_x.T @ jac_x / sigma**2) @ u
    s, _ = truncated_root(u, a, jac_c, k.diagonal())
    jac_s = flat_jacobian(net, xs)

    return jac_s @ s @ s.T @ jac_s.T


def test_front_door_equals_dense_reference():
    net = toy_net()
    x, y = toy_data()
    c = torch.linspace(-2, 2, 8, dtype=DT).unsqueeze(-1)
    xs = torch.linspace(-2.5, 2.5, 9, dtype=DT).unsqueeze(-1)

    post = front(net, halves(x, y))
    want = dense_output_cov(net, x, c, xs)

    assert isinstance(post, FSPPosterior)
    assert post.basis.shape == (61, 8)
    assert post.kernel == "RBF"
    assert post.strategy == "forward-mode"
    torch.testing.assert_close(post.n_data, torch.tensor(40.0, dtype=DT))

    for strategy in ("forward-mode", "reverse-mode"):
        out = lk.predict_outputs(post, xs, strategy=strategy, n_samples="exact")
        got = out.cov.diagonal().reshape(-1)

        torch.testing.assert_close(out.mean, net(xs).detach())
        torch.testing.assert_close(got, want.diagonal(), rtol=1e-6, atol=0)

    # The whole covariance across inputs, from the exact square root.
    root = lk.predict_outputs(post, xs, strategy="forward-mode", n_samples="exact").cov
    assert isinstance(root, lk.core.FactorCov)
    f = root.factor.reshape(9, -1)
    torch.testing.assert_close(f @ f.T, want, rtol=1e-6, atol=1e-6 * float(want.abs().max()))


def test_result_owns_its_parameters():
    net = toy_net()
    x, y = toy_data()
    post = front(net, halves(x, y))
    before = {k: v.clone() for k, v in post.mean.items()}

    for name, p in net.named_parameters():
        assert post.mean[name] is not p
        assert post.mean[name].data_ptr() != p.data_ptr()
        assert not post.mean[name].requires_grad

    with torch.no_grad():
        for p in net.parameters():
            p.add_(1.0)

    for name in before:
        torch.testing.assert_close(post.mean[name], before[name])


def test_consumers_accept_fsp_posterior(tmp_path):
    net = toy_net()
    x, y = toy_data()
    xs = torch.linspace(-2.5, 2.5, 9, dtype=DT).unsqueeze(-1)
    post = front(net, halves(x, y))
    exact = (
        lk.predict_outputs(post, xs, strategy="reverse-mode", n_samples="exact")
        .cov.diagonal()
        .reshape(-1)
    )

    # outputs: jvp and vjp exact (above), and Monte Carlo.
    mc = lk.predict_outputs(
        post,
        xs,
        strategy="reverse-mode",
        n_samples=4000,
        generator=torch.Generator().manual_seed(1),
    )
    torch.testing.assert_close(mc.cov.diagonal().reshape(-1), exact, rtol=0.1, atol=0)

    # predict: Linearized adds σ² to the linearized variance.
    lin = lk.predict(post, xs)
    torch.testing.assert_close(lin.variance.reshape(-1), exact + SIGMA**2, rtol=1e-6, atol=0)
    assert isinstance(lk.predict(post, xs, lk.Linearized()), torch.distributions.Distribution)

    # WeightSamples runs (draws of the network; see the caveat in the docstring).
    ws = lk.predict(post, xs, lk.WeightSamples(n_samples=8))
    assert torch.isfinite(ws.mean).all()

    # sample_outputs, linearized: the draws' variance matches the exact one.
    draws = lk.sample_outputs(post, xs, 4000, generator=torch.Generator().manual_seed(2))
    assert draws.shape == (4000, 9, 1)
    torch.testing.assert_close(draws.var(0).reshape(-1), exact, rtol=0.1, atol=0)
    network = lk.sample_outputs(
        post, xs, 3, generator=torch.Generator().manual_seed(2), kind="network"
    )
    assert network.shape == (3, 9, 1)

    # save, then load with the model.
    lk.save(post, tmp_path / "post.pt")
    loaded = lk.load(tmp_path / "post.pt", net)

    torch.testing.assert_close(loaded.gram, post.gram)
    torch.testing.assert_close(
        lk.predict_outputs(loaded, xs, strategy="reverse-mode", n_samples="exact")
        .cov.diagonal()
        .reshape(-1),
        exact,
        rtol=1e-12,
        atol=0,
    )


def test_fsp_posterior_as_prior_before_data():
    """The refusal comes from the method's template, before the data pass."""

    net = toy_net()
    x, y = toy_data()
    post = front(net, halves(x, y))
    data = Passes(halves(x, y))

    with pytest.raises(StructureError, match="^before the data pass: .*degenerate"):
        lk.laplace(
            net,
            data,
            likelihood=copy.replace(
                lk.Gaussian(), sigma=torch.as_tensor(torch.tensor(SIGMA, dtype=DT))
            ),
            subset=lk.subsets.all,
            prior=post,
            hessian_approx=lk.DenseGGN(),
            hyper_tuner=None,
        )

    assert data.pulled == 1  # the probe's batch only


def test_sigma_late():
    net = toy_net()
    x, y = toy_data()
    g = torch.Generator().manual_seed(60)
    xv = torch.rand(15, 1, generator=g, dtype=DT) * 2 - 1
    yv = torch.sin(2 * math.pi * xv) + 0.3 * torch.randn(xv.shape, generator=g, dtype=DT)
    xs = torch.linspace(-2.5, 2.5, 9, dtype=DT).unsqueeze(-1)

    data = Passes(halves(x, y))
    post = front(net, data, sigma=1.0)
    iters, pulled = data.iters, data.pulled

    nll = lk.objectives.predictive_nll([(xv, yv)])
    tuned = lk.tune(post, lk.Minimize(nll))
    sigma = float(tuned.likelihood.sigma)

    assert (data.iters, data.pulled) == (iters, pulled)
    assert not math.isclose(sigma, 1.0, rel_tol=1e-3)

    refit = front(net, halves(x, y), sigma=sigma)

    torch.testing.assert_close(
        lk.predict_outputs(tuned, xs, strategy="reverse-mode", n_samples="exact").cov.diagonal(),
        lk.predict_outputs(refit, xs, strategy="reverse-mode", n_samples="exact").cov.diagonal(),
        rtol=1e-8,
        atol=0,
    )

    # And the refit matches the dense reference at that σ.
    c = torch.linspace(-2, 2, 8, dtype=DT).unsqueeze(-1)
    want = dense_output_cov(net, x, c, xs, sigma=sigma)
    got = (
        lk.predict_outputs(refit, xs, strategy="reverse-mode", n_samples="exact")
        .cov.diagonal()
        .reshape(-1)
    )
    torch.testing.assert_close(got, want.diagonal(), rtol=1e-6, atol=0)


class NoVmap(nn.Module):
    def __init__(self):
        super().__init__()
        self.inner = toy_net()

    def forward(self, x):
        out = self.inner(x)

        return out if bool(out.sum() > -1e9) else -out  # data-dependent: no vmap


def test_requirements_before_data():
    x, y = toy_data()
    data = Passes(halves(x, y))

    with pytest.raises(core.CapabilityError, match="vmap"):
        front(NoVmap(), data)

    assert data.pulled == 1

    # No output Hessian and no way to draw targets: refused before the probe.
    def log_prob(f, y):
        return -((f[:, 0] - y[:, 0]) ** 2)

    data = Passes(halves(x, y))

    with pytest.raises(core.CapabilityError, match="output Hessian"):
        lk.gp.fsp_laplace(
            toy_net(),
            data,
            likelihood=lk.likelihoods.from_log_prob(log_prob),
            prior=prior(),
            context=context8(),
            rank=8,
        )

    assert data.pulled == 0


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
        self.inner = toy_net()

    def forward(self, x):
        h = self.inner[1](self.inner[0](x))

        return self.inner[4](_Square.apply(self.inner[2](h)))


def test_jvp_free_model_fits_through_vjp():
    net = NoJvp()
    x, y = toy_data()
    c = torch.linspace(-2, 2, 8, dtype=DT).unsqueeze(-1)
    xs = torch.linspace(-2.5, 2.5, 9, dtype=DT).unsqueeze(-1)

    post = front(net, halves(x, y))

    assert post.strategy == "reverse-mode"
    assert not post.caps.jvp

    want = dense_output_cov(net, x, c, xs)
    got = (
        lk.predict_outputs(post, xs, strategy="reverse-mode", n_samples="exact")
        .cov.diagonal()
        .reshape(-1)
    )
    torch.testing.assert_close(got, want.diagonal(), rtol=1e-6, atol=0)

    # Forward mode asked for, and missing: refused before data.
    data = Passes(halves(x, y))

    with pytest.raises(core.CapabilityError, match="jvp"):
        front(net, data, strategy="forward-mode")

    assert data.pulled == 1

    # Given, and found: honoured.
    assert front(toy_net(), halves(x, y), strategy="reverse-mode").strategy == "reverse-mode"


def test_context_budget_before_data(monkeypatch):
    import laplacekit_gp.fsp as fsp

    x, y = toy_data()

    # n_C E P = 8 * 1 * 61 entries.
    monkeypatch.setattr(fsp, "MAX_CONTEXT_JACOBIAN_ENTRIES", 8 * 61 - 1)
    data = Passes(halves(x, y))

    with pytest.raises(StructureError, match=r"n_C=8.*E=1.*P=61.*MAX_CONTEXT_JACOBIAN_ENTRIES"):
        front(toy_net(), data)

    assert data.pulled == 1

    # Points from the data: the count is known, so it is checked before the pick pass.
    data = Passes(halves(x, y))

    with pytest.raises(StructureError, match=r"n_C=8.*MAX_CONTEXT_JACOBIAN_ENTRIES"):
        front(toy_net(), data, context=lk.gp.from_data(8))

    assert data.pulled == 1

    monkeypatch.setattr(fsp, "MAX_CONTEXT_JACOBIAN_ENTRIES", 8 * 61)
    assert front(toy_net(), halves(x, y)).basis.shape == (61, 8)


def test_bad_arguments_refused_before_data():
    x, y = toy_data()
    c = torch.linspace(-2, 2, 8, dtype=DT).unsqueeze(-1)

    data = Passes(halves(x, y))

    with pytest.raises(ValueError, match="rank"):
        front(toy_net(), data, rank=0)

    assert data.pulled == 0

    # Context points that are not one tensor: the kernel takes [n_C, D] tensors.
    data = Passes(halves(x, y))

    with pytest.raises(StructureError, match="context"):
        front(toy_net(), data, context=lk.gp.fixed({"x": c}))

    assert data.pulled == 1

    # A 0-dim tensor (built by hand, past `fixed`'s own check) is named as such.
    with pytest.raises(StructureError, match="got a 0-dim tensor"):
        front(
            toy_net(), halves(x, y), context=laplacekit_gp.points.Fixed(torch.tensor(1.0, dtype=DT))
        )


@pytest.mark.parametrize(
    ("out", "likelihood", "targets", "error", "match", "pulled"),
    [
        (1, lk.Bernoulli(), lambda y: torch.full_like(y, 3.0), ValueError, r"in \[0, 1\]", 2),
        (
            2,
            copy.replace(lk.Gaussian(), sigma=torch.as_tensor(torch.tensor(SIGMA, dtype=DT))),
            lambda y: y[:, 0],
            StructureError,
            "targets shaped",
            1,
        ),
        (
            2,
            copy.replace(
                lk.Gaussian(), sigma=torch.as_tensor(torch.tensor([0.1, 0.2, 0.3], dtype=DT))
            ),
            lambda y: y.repeat(1, 2),
            StructureError,
            "sigma of shape",
            1,
        ),
    ],
    ids=["bernoulli-targets", "gaussian-target-shape", "sigma-length"],
)
def test_likelihood_checks(out, likelihood, targets, error, match, pulled):
    """The likelihood's checks: shapes on the probe batch, before any pass; target values
    on the first batch of the pass."""

    x, y = toy_data()
    data = Passes(halves(x, targets(y)))

    with pytest.raises(error, match=match):
        lk.gp.fsp_laplace(
            toy_net(out=out), data, likelihood=likelihood, prior=prior(), context=context8(), rank=8
        )

    assert data.pulled == pulled


def test_context_from_the_data():
    """Context points picked from the data: a pick pass, then the Gram pass."""

    net = toy_net()
    x, y = toy_data()
    data = Passes(halves(x, y))
    post = front(net, data, context=lk.gp.from_data(6), rank=6)

    assert data.iters == 3
    assert post.basis.shape == (61, 6)
    torch.testing.assert_close(post.n_data, torch.tensor(40.0, dtype=DT))


def test_last_layer_by_default():
    net = toy_net()
    x, y = toy_data()
    post = lk.gp.fsp_laplace(
        net,
        halves(x, y),
        likelihood=copy.replace(
            lk.Gaussian(), sigma=torch.as_tensor(torch.tensor(SIGMA, dtype=DT))
        ),
        prior=prior(),
        context=context8(),
        rank=8,
    )

    assert set(post.mean) == {"4.weight", "4.bias"}
    assert post.basis.shape[0] == 7
    assert set(post.versions) == {"0.weight", "0.bias", "2.weight", "2.bias"}


def test_kernel_not_a_leaf():
    """The kernel is fixed at fit time: `tune` can't reach it."""

    net = toy_net()
    x, y = toy_data()
    ell = torch.tensor(1.0, dtype=DT)
    p = lk.gp.FunctionPrior(lk.gp.RBF(ell, torch.tensor(1.0, dtype=DT)))
    post = front(net, halves(x, y), prior=p)
    leaves = lk.core.tree_leaves(post)

    assert isinstance(post.kernel, str)
    assert not any(t is ell for t in leaves)
    assert "kernel=RBF" in repr(post)

    # The tuners see the prior precision and the likelihood's sigma only.
    assert set(lk.core.hyperparameters(post)) <= {"prior_precision", "sigma"}
