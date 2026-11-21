import copy
import dataclasses
import time
from collections.abc import Mapping

import pytest
import torch
from torch import Tensor, nn
from torch.func import functional_call, jacrev

import laplacekit as lk
from laplacekit.core import (
    MAX_DENSE_EVENT,
    BlockDiag,
    Capabilities,
    CapabilityError,
    Dense,
    Diag,
    FactorCov,
    HasCovMany,
    HasVersions,
    Joint,
    KronEigenbasis,
    LaplacePosterior,
    Layout,
    ModelFn,
    PerInputCov,
    SampleCov,
    StructureError,
    WeightPosterior,
    mean_outputs,
    predict_outputs,
    resolve_strategy,
)
from laplacekit.core.posteriors import _MAX_FACTOR_ENTRIES
from laplacekit.priors import Isotropic

D = 5


def _h(seed: int = 0) -> Tensor:
    g = torch.Generator().manual_seed(seed)
    a = torch.randn(D, D, generator=g, dtype=torch.float64)

    return a @ a.T / D


def _params() -> dict[str, Tensor]:
    return {
        "fc.weight": torch.randn(2, 2, dtype=torch.float64),
        "fc.bias": torch.randn(1, dtype=torch.float64),
    }


def _model() -> ModelFn:
    return ModelFn(lambda p, x: x)


def _dense_post(h: Tensor | None = None) -> LaplacePosterior[Dense, Isotropic]:
    params = {
        "fc.weight": torch.randn(D - 1, dtype=torch.float64),
        "fc.bias": torch.randn(1, dtype=torch.float64),
    }
    return LaplacePosterior(
        mean=params,
        curvature=Dense(_h() if h is None else h),
        prior=Isotropic(torch.tensor(1.0, dtype=torch.float64)),
        n_data=torch.tensor(10.0, dtype=torch.float64),
        model=_model(),
        likelihood=None,
        layout=Layout.by_module(params),
    )


def _prec() -> Tensor:
    return 10 * _h() + torch.eye(D, dtype=torch.float64)


def test_precision_and_log_det_ratio() -> None:
    post = _dense_post()

    assert torch.allclose(post.precision().mat, _prec())
    ref = torch.logdet(_prec()) - torch.logdet(torch.eye(D, dtype=torch.float64))

    assert torch.allclose(post.log_det_ratio(), ref)


def test_cov_mv() -> None:
    post = _dense_post()
    v = {
        "fc.weight": torch.randn(D - 1, dtype=torch.float64),
        "fc.bias": torch.randn(1, dtype=torch.float64),
    }
    got = post.layout.flatten(post.cov_mv(v), "fc")

    assert torch.allclose(got, torch.linalg.solve(_prec(), post.layout.flatten(v, "fc")))


def test_sample_covariance_and_marginal_variance() -> None:
    post = _dense_post()
    s = post.sample(20000, generator=torch.Generator().manual_seed(0))

    assert s["fc.weight"].shape == (20000, D - 1)
    flat = torch.cat([s["fc.weight"], s["fc.bias"]], dim=-1)
    flat = flat - flat.mean(0)
    emp = flat.T @ flat / flat.shape[0]
    cov = torch.linalg.inv(_prec())

    assert (emp - cov).abs().max() < 0.05
    mv = post.variance

    assert torch.allclose(post.layout.flatten(mv, "fc"), cov.diagonal())


def test_sample_is_centered_on_mean() -> None:
    post = _dense_post()
    s = post.sample(20000, generator=torch.Generator().manual_seed(1))

    assert torch.allclose(s["fc.bias"].mean(0), post.mean["fc.bias"], atol=0.05)


def test_noise_is_sample_minus_mean() -> None:
    post = _dense_post()
    noise = post.sample_centered(5, generator=torch.Generator().manual_seed(2))
    draws = post.sample(5, generator=torch.Generator().manual_seed(2))

    for k in post.mean:
        assert torch.equal(draws[k], post.mean[k] + noise[k])


def test_noise_keeps_float32_for_bf16_means() -> None:
    post = _dense_post()
    low = {k: v.to(torch.bfloat16) for k, v in post.mean.items()}
    post = LaplacePosterior(
        mean=low,
        curvature=Dense(_h().float()),
        prior=Isotropic(torch.tensor(1.0)),
        n_data=torch.tensor(10.0),
        model=_model(),
        likelihood=None,
        layout=Layout.by_module(low),
    )

    assert {
        v.dtype
        for v in post.sample_centered(3, generator=torch.Generator().manual_seed(0)).values()
    } == {torch.float32}
    assert {
        v.dtype for v in post.sample(3, generator=torch.Generator().manual_seed(0)).values()
    } == {torch.bfloat16}


def test_continual_update_adds_precision() -> None:
    post = _dense_post()
    h2 = _h(1)
    out = post.add_precision_to(Dense(h2))

    assert torch.allclose(out.mat, _prec() + h2)


def test_log_prob_matches_mvn() -> None:
    post = _dense_post()
    p = {k: x + 0.1 for k, x in post.mean.items()}
    mvn = torch.distributions.MultivariateNormal(
        post.layout.flatten(post.mean, "fc"), precision_matrix=_prec()
    )

    assert torch.allclose(post.log_prob(p), mvn.log_prob(post.layout.flatten(p, "fc")))


def test_blockdiag_curvature() -> None:
    params = {
        "a.weight": torch.randn(2, dtype=torch.float64),
        "b.weight": torch.randn(3, dtype=torch.float64),
    }
    post = LaplacePosterior(
        mean=params,
        curvature=BlockDiag(
            {
                "a": Diag(torch.ones(2, dtype=torch.float64)),
                "b": Diag(2 * torch.ones(3, dtype=torch.float64)),
            }
        ),
        prior=Isotropic(torch.tensor(1.0, dtype=torch.float64)),
        n_data=torch.tensor(1.0, dtype=torch.float64),
        model=_model(),
        likelihood=None,
        layout=Layout.by_module(params),
    )
    mv = post.variance

    assert torch.allclose(mv["a.weight"], torch.full((2,), 0.5, dtype=torch.float64))
    assert torch.allclose(mv["b.weight"], torch.full((3,), 1 / 3, dtype=torch.float64))
    s = post.sample(4, generator=torch.Generator().manual_seed(0))

    assert s["b.weight"].shape == (4, 3)


def test_non_blockdiag_needs_single_block_layout() -> None:
    params = {
        "a.weight": torch.randn(2, dtype=torch.float64),
        "b.weight": torch.randn(2, dtype=torch.float64),
    }
    post = LaplacePosterior(
        mean=params,
        curvature=Dense(torch.eye(4, dtype=torch.float64)),
        prior=Isotropic(torch.tensor(1.0, dtype=torch.float64)),
        n_data=torch.tensor(1.0, dtype=torch.float64),
        model=_model(),
        likelihood=None,
        layout=Layout.by_module(params),
    )

    with pytest.raises(StructureError, match="one block"):
        post.cov_mv(params)


def _bd_post(blocks: dict[str, Diag | Dense], sizes: dict[str, int]) -> LaplacePosterior:
    params = {f"{b}.weight": torch.randn(n, dtype=torch.float64) for b, n in sizes.items()}
    return LaplacePosterior(
        mean=params,
        curvature=BlockDiag(blocks),
        prior=Isotropic(torch.tensor(1.0, dtype=torch.float64)),
        n_data=torch.tensor(3.0, dtype=torch.float64),
        model=_model(),
        likelihood=None,
        layout=Layout.by_module(params),
    )


def test_marginal_variance_blockdiag_with_dense_blocks() -> None:
    h1, h2 = _h(2), _h(3)[:3, :3]
    post = _bd_post(
        {"a": Dense(h1), "b": Dense(h2), "c": Diag(torch.tensor([1.0, 2.0], dtype=torch.float64))},
        {"a": D, "b": 3, "c": 2},
    )
    mv = post.variance
    eye = lambda n: torch.eye(n, dtype=torch.float64)  # noqa: E731

    assert torch.allclose(mv["a.weight"], torch.linalg.inv(3 * h1 + eye(D)).diagonal())
    assert torch.allclose(mv["b.weight"], torch.linalg.inv(3 * h2 + eye(3)).diagonal())
    assert torch.allclose(
        mv["c.weight"], 1 / (3 * torch.tensor([1.0, 2.0], dtype=torch.float64) + 1)
    )


def test_custom_layout_rejected() -> None:
    params = {
        "a.weight": torch.randn(2, dtype=torch.float64),
        "b.weight": torch.randn(2, dtype=torch.float64),
    }
    one_block = Layout(
        blocks={"all": ("a.weight", "b.weight")},
        shapes={k: tuple(v.shape) for k, v in params.items()},
    )

    with pytest.raises(StructureError, match="custom layouts"):
        LaplacePosterior(
            mean=params,
            curvature=Dense(torch.eye(4, dtype=torch.float64)),
            prior=Isotropic(torch.tensor(1.0, dtype=torch.float64)),
            n_data=torch.tensor(1.0, dtype=torch.float64),
            model=_model(),
            likelihood=None,
            layout=one_block,
        )


# --- chunked outputs -----------------------------------------------------------------


F64 = torch.float64


def _mlp(e: int) -> nn.Module:
    torch.manual_seed(0)

    return nn.Sequential(nn.Linear(3, 4), nn.Tanh(), nn.Linear(4, e)).to(F64)


def _spd(n: int, g: torch.Generator) -> Tensor:
    a = torch.randn(n, n, generator=g, dtype=F64)

    return a @ a.T / n + 0.1 * torch.eye(n, dtype=F64)


def _orth(n: int, g: torch.Generator) -> Tensor:
    return torch.linalg.qr(torch.randn(n, n, generator=g, dtype=F64))[0]


def _kron_dense(op: KronEigenbasis) -> Tensor:
    """The dense matrix of `op` in layout order (weight row-major, then bias)."""

    out, n_in = op.evals.shape
    q = torch.kron(op.evecs_g, op.evecs_a)
    folded = q @ torch.diag(op.evals.reshape(-1)) @ q.T  # row-major vec of [W | b]
    weight = [o * n_in + j for o in range(out) for j in range(n_in - 1)]
    perm = torch.tensor(weight + [o * n_in + n_in - 1 for o in range(out)])

    return folded[perm][:, perm]


VJP_VMAP = Capabilities(vjp=True, vmap=True)


def _post(
    module: nn.Module, kind: str, caps: Capabilities = VJP_VMAP
) -> tuple[LaplacePosterior, Tensor]:
    """A posterior with one block per module of `module`, and its dense covariance."""

    mean = {k: v.detach() for k, v in module.named_parameters()}
    layout = Layout.by_module(mean)
    sizes = {b: sum(mean[n].numel() for n in ns) for b, ns in layout.blocks.items()}
    g = torch.Generator().manual_seed(1)
    curv: dict[str, Diag | Dense | KronEigenbasis] = {}

    if kind == "diag":
        diags = {b: torch.rand(n, generator=g, dtype=F64) + 0.1 for b, n in sizes.items()}
        curv = {b: Diag(d) for b, d in diags.items()}
        dense = {b: torch.diag(d) for b, d in diags.items()}
    elif kind == "dense":
        mats = {b: _spd(n, g) for b, n in sizes.items()}
        curv = {b: Dense(m) for b, m in mats.items()}
        dense = mats
    else:
        dense = {}

        for b in layout.blocks:
            w = mean[f"{b}.weight"]
            out, n_in = w.shape[0], w.shape[1] + 1
            evals = torch.rand(out, n_in, generator=g, dtype=F64) + 0.1
            op = KronEigenbasis(_orth(n_in, g), _orth(out, g), evals, True, tuple(w.shape))
            curv[b] = op
            dense[b] = _kron_dense(op)

    post = LaplacePosterior(
        mean=mean,
        curvature=BlockDiag(curv),
        prior=Isotropic(torch.tensor(0.5, dtype=F64)),
        n_data=torch.tensor(3.0, dtype=F64),
        model=ModelFn.from_module(module),
        likelihood=None,
        layout=layout,
        caps=caps,
    )
    prec = torch.block_diag(*(3.0 * dense[b] + 0.5 * torch.eye(sizes[b], dtype=F64) for b in sizes))

    return post, torch.linalg.inv(prec)


def _chunk_post(e: int, kind: str) -> tuple[LaplacePosterior, Tensor]:
    """A two-block posterior on an MLP with `e` outputs, and its dense covariance."""

    return _post(_mlp(e), kind)


@pytest.mark.parametrize("kind", ["dense", "kfac"])
@pytest.mark.parametrize(("e", "chunk_size"), [(1, 1), (1, 256), (7, 1), (7, 3), (7, 256)])
def test_outputs_chunked_matches_dense(kind: str, e: int, chunk_size: int) -> None:
    post, sigma = _chunk_post(e, kind)
    module = post.model.module
    assert module is not None
    x = torch.randn(6, 3, generator=torch.Generator().manual_seed(2), dtype=F64)
    names = list(post.mean)
    shapes = [post.mean[n].shape for n in names]
    sizes = [post.mean[n].numel() for n in names]

    def f(flat: Tensor, xi: Tensor) -> Tensor:
        parts = torch.split(flat, sizes)
        p = {n: t.reshape(s) for n, t, s in zip(names, parts, shapes, strict=True)}
        return functional_call(module, p, (xi,))

    w = torch.cat([post.mean[n].reshape(-1) for n in names])
    jac = torch.stack([jacrev(f)(w, x[i]) for i in range(x.shape[0])])  # [N, E, P]
    ref = jac @ sigma @ jac.mT
    out = predict_outputs(post, x, chunk_size=chunk_size)

    assert out.cov.per_input().shape == (6, e, e)
    assert torch.allclose(out.cov.per_input(), ref, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("chunk_size", [0, -2])
def test_outputs_chunk_must_be_positive(chunk_size: int) -> None:
    post, _ = _chunk_post(2, "dense")

    with pytest.raises(ValueError, match="chunk"):
        predict_outputs(post, torch.zeros(1, 3, dtype=F64), chunk_size=chunk_size)


# --- sampled output covariances ------------------------------------------------------


def _draws(s: int = 4, n: int = 3, event: tuple[int, ...] = (2, 3)) -> Tensor:
    return torch.randn(s, n, *event, generator=torch.Generator().manual_seed(5), dtype=F64)


def test_sample_cov_moments() -> None:
    d = _draws()
    cov = SampleCov(d)
    flat = d.reshape(4, 3, 6)
    ref = torch.stack([flat[:, i].T @ flat[:, i] / 4 for i in range(3)])

    assert torch.allclose(cov.diagonal(), (d**2).mean(0))
    assert cov.diagonal().shape == (3, 2, 3)
    assert torch.allclose(cov.per_input(), ref)
    assert torch.allclose(cov.scaled(torch.tensor(4.0, dtype=F64)).per_input(), 4 * ref)
    assert torch.allclose(cov.scaled(torch.tensor(4.0, dtype=F64)).diagonal(), 4 * (d**2).mean(0))


def test_sample_cov_sample_picks_stored_draws() -> None:
    d = _draws()
    cov = SampleCov(d)
    got = cov.sample(3, generator=torch.Generator().manual_seed(1))

    assert got.shape == (3, 3, 2, 3)

    picked = [int(((d - g) == 0).all(dim=(1, 2, 3)).nonzero()) for g in got]

    assert len(set(picked)) == 3
    assert torch.equal(cov.sample(3, generator=torch.Generator().manual_seed(1)), got)
    assert torch.equal(
        cov.sample(4, generator=torch.Generator().manual_seed(0)).sort(0).values, d.sort(0).values
    )

    with pytest.raises(ValueError, match="4"):
        cov.sample(5, generator=torch.Generator().manual_seed(0))


def test_sample_cov_per_input_refuses_large_events() -> None:
    cov = SampleCov(torch.zeros(2, 1, 4097))

    with pytest.raises(StructureError, match="4096"):
        cov.per_input()


def test_per_input_cov_sample_matches_blocks() -> None:
    a = torch.randn(3, 4, 4, generator=torch.Generator().manual_seed(6), dtype=F64)
    blocks = a @ a.mT / 4
    s = PerInputCov(blocks, (2, 2)).sample(40000, generator=torch.Generator().manual_seed(0))

    assert s.shape == (40000, 3, 2, 2)

    flat = s.reshape(40000, 3, 4)
    emp = torch.einsum("sne,snf->nef", flat, flat) / 40000

    assert (emp - blocks).abs().max() < 0.05


def test_factor_cov_matches_dense() -> None:
    g = torch.Generator().manual_seed(7)
    f = torch.randn(3, 2, 4, 5, generator=g, dtype=F64) / 5**0.5
    f[1] = f[0]  # inputs 0 and 1 share their factor rows
    cov = FactorCov(f)
    flat = f.reshape(3, 8, 5)
    ref = flat @ flat.mT

    assert torch.allclose(cov.diagonal(), torch.stack([b.diagonal() for b in ref]).reshape(3, 2, 4))
    assert torch.allclose(cov.per_input(), ref)

    s = torch.tensor(2.5, dtype=F64)

    assert torch.allclose(cov.scaled(s).per_input(), s * ref)
    assert torch.allclose(cov.scaled(s).diagonal(), s * cov.diagonal())

    draws = cov.sample(20000, generator=torch.Generator().manual_seed(0))

    assert draws.shape == (20000, 3, 2, 4)
    assert draws.dtype == F64

    d = draws.reshape(20000, 3, 8)
    emp = torch.einsum("sne,snf->nef", d, d) / 20000

    assert (emp - ref).abs().max() < 0.05

    # One z per draw for all inputs: equal factor rows give equal draws.
    assert torch.equal(draws[:, 0], draws[:, 1])
    assert torch.equal(cov.sample(20000, generator=torch.Generator().manual_seed(0)), draws)


def test_factor_cov_per_input_refuses_large_event() -> None:
    cov = FactorCov(torch.zeros(1, 4097, 2))

    with pytest.raises(StructureError, match="MAX_DENSE_EVENT=4096"):
        cov.per_input()

    assert cov.diagonal().shape == (1, 4097)


def _diag_post() -> LaplacePosterior:
    module = nn.Linear(3, 2).to(F64)
    mean = {k: v.detach() for k, v in module.named_parameters()}
    size = sum(v.numel() for v in mean.values())

    return LaplacePosterior(
        mean=mean,
        curvature=Diag(torch.rand(size, generator=torch.Generator().manual_seed(1), dtype=F64)),
        prior=Isotropic(torch.tensor(0.5, dtype=F64)),
        n_data=torch.tensor(3.0, dtype=F64),
        model=ModelFn.from_module(module),
        likelihood=None,
        layout=Layout.by_module(mean),
    )


@pytest.mark.parametrize("kind", ["diag", "dense", "single_dense", "kfac"])
def test_cov_sqrt_mv_squares_to_cov(kind: str) -> None:
    if kind == "diag":
        post = _diag_post()
    elif kind == "single_dense":
        post = _dense_post()
    else:
        post = _chunk_post(3, kind)[0]

    names = list(post.mean)
    shapes = [post.mean[n].shape for n in names]
    sizes = [post.mean[n].numel() for n in names]
    p = sum(sizes)

    def unflat(v: Tensor) -> dict[str, Tensor]:
        parts = torch.split(v, sizes)
        return {n: t.reshape(sh) for n, t, sh in zip(names, parts, shapes, strict=True)}

    def flat(v: dict[str, Tensor]) -> Tensor:
        return torch.cat([v[n].reshape(-1) for n in names])

    eye = torch.eye(p, dtype=F64)
    root = torch.stack([flat(post.cov_sqrt_mv(unflat(e))) for e in eye], dim=1)
    sigma = torch.stack([flat(post.cov_mv(unflat(e))) for e in eye], dim=1)

    assert torch.allclose(root @ root.T, sigma, rtol=0, atol=1e-10)


def _with_caps(post: LaplacePosterior, caps: Capabilities) -> LaplacePosterior:
    import dataclasses

    return dataclasses.replace(post, caps=caps)


ALL = Capabilities(vjp=True, vjp_twice=True, jvp=True, vmap=True)


def test_sample_cov_converges() -> None:
    post, _ = _chunk_post(7, "dense")
    post = _with_caps(post, ALL)
    x = torch.randn(6, 3, generator=torch.Generator().manual_seed(2), dtype=F64)
    exact = predict_outputs(post, x, n_samples="exact").cov.diagonal()
    got = {}

    for strategy in ("forward-mode", "reverse-mode"):
        out = predict_outputs(
            post, x, strategy=strategy, n_samples=20000, generator=torch.Generator().manual_seed(3)
        )

        assert isinstance(out.cov, SampleCov)
        assert out.cov.draws.shape == (20000, 6, 7)
        assert torch.allclose(out.cov.draws.mean(0), torch.zeros(6, 7, dtype=F64), atol=1e-12)
        assert (out.cov.diagonal() - exact).norm() / exact.norm() < 2e-2
        got[strategy] = out.cov.draws

    torch.testing.assert_close(got["forward-mode"], got["reverse-mode"], rtol=1e-5, atol=1e-8)


def test_outputs_default_samples_without_vmap() -> None:
    post, _ = _chunk_post(3, "dense")
    post = _with_caps(post, Capabilities(vjp=True, vjp_twice=True))
    x = torch.randn(4, 3, generator=torch.Generator().manual_seed(2), dtype=F64)
    out = predict_outputs(post, x)

    assert isinstance(out.cov, SampleCov)
    assert out.cov.draws.shape == (64, 4, 3)
    torch.manual_seed(0)
    unseeded = predict_outputs(post, x).cov.draws
    torch.manual_seed(0)

    assert torch.equal(unseeded, predict_outputs(post, x).cov.draws)

    with pytest.raises(CapabilityError, match="vmap"):
        predict_outputs(post, x, n_samples="exact")

    with pytest.raises(CapabilityError, match="jvp"):
        predict_outputs(post, x, strategy="forward-mode", n_samples=8)


@pytest.mark.parametrize("n_samples", [0, 1])
def test_outputs_needs_two_samples(n_samples: int) -> None:
    post, _ = _chunk_post(2, "dense")

    with pytest.raises(ValueError, match="samples"):
        predict_outputs(post, torch.zeros(1, 3, dtype=F64), n_samples=n_samples)


def test_large_event_defaults_to_samples() -> None:
    e = 5000
    torch.manual_seed(0)
    module = nn.Linear(2, e).to(F64)
    mean = {k: v.detach() for k, v in module.named_parameters()}
    size = sum(v.numel() for v in mean.values())
    post = LaplacePosterior(
        mean=mean,
        curvature=Diag(torch.rand(size, generator=torch.Generator().manual_seed(1), dtype=F64)),
        prior=Isotropic(torch.tensor(1.0, dtype=F64)),
        n_data=torch.tensor(3.0, dtype=F64),
        model=ModelFn.from_module(module),
        likelihood=None,
        layout=Layout.by_module(mean),
        caps=ALL,
    )
    x = torch.randn(2, 2, generator=torch.Generator().manual_seed(2), dtype=F64)
    out = predict_outputs(post, x)

    assert isinstance(out.cov, SampleCov)
    assert out.cov.draws.shape == (64, 2, e)

    # Var f_ie = sum_j x_ij^2 / prec_w[e, j] + 1 / prec_b[e]; checked loosely at 64 draws.
    prec = 3.0 * post.curvature.diag + 1.0
    w, b = prec[: 2 * e].reshape(e, 2), prec[2 * e :]
    ref = (x**2) @ (1 / w).T + 1 / b

    assert (out.cov.diagonal().mean(1) / ref.mean(1) - 1).abs().max() < 0.1


def test_exact_outputs_build_no_identity(monkeypatch) -> None:
    """Each chunk's basis rows are made on their own: no [E, E] identity."""

    post, _ = _chunk_post(7, "dense")
    x = torch.randn(6, 3, generator=torch.Generator().manual_seed(2), dtype=F64)
    want = predict_outputs(post, x, n_samples="exact", chunk_size=3).cov.per_input()

    eye = torch.eye

    def no_eye(*args, **kwargs):
        assert args[0] != 7, "an [E, E] identity"  # E = 7; the parameter count is 16

        return eye(*args, **kwargs)

    monkeypatch.setattr(torch, "eye", no_eye)
    got = predict_outputs(post, x, n_samples="exact", chunk_size=3).cov.per_input()

    torch.testing.assert_close(got, want, rtol=0, atol=0)


@pytest.mark.parametrize(("chunk_size", "passes"), [(3, 6), (256, 1)])
def test_outputs_streams_in_chunks(chunk_size: int, passes: int) -> None:
    """With k = ceil(E / chunk_size) chunks, tile (a, b <= a) recomputes chunk b: k(k+1)/2
    passes."""

    post, _ = _chunk_post(7, "dense")
    x = torch.randn(6, 3, generator=torch.Generator().manual_seed(2), dtype=F64)
    model = post.model
    batches: list[int] = []

    def counted(params, inputs):
        batches.append(inputs.shape[0])
        return model(params, inputs)

    post = dataclasses.replace(post, model=ModelFn(counted, model.module))
    predict_outputs(post, x, n_samples="exact", chunk_size=chunk_size)

    # One forward on the whole batch, then one per-input (vmapped) pass per chunk.
    assert batches == [6] + [1] * passes


# --- forward-mode (jvp) exact pushforward --------------------------------------------


def _conv() -> nn.Module:
    torch.manual_seed(0)

    return nn.Sequential(nn.Conv2d(1, 2, 3, padding=1), nn.Tanh(), nn.Conv2d(2, 1, 3)).to(F64)


def _ref_jac(post: LaplacePosterior, x: Tensor) -> Tensor:
    """Dense per-input Jacobians [N, E, P] by `jacrev`, in the mean's flat order."""

    module = post.model.module
    assert module is not None
    names = list(post.mean)
    shapes = [post.mean[n].shape for n in names]
    sizes = [post.mean[n].numel() for n in names]

    def f(flat: Tensor, xi: Tensor) -> Tensor:
        parts = torch.split(flat, sizes)
        p = {n: t.reshape(s) for n, t, s in zip(names, parts, shapes, strict=True)}
        return functional_call(module, p, (xi.unsqueeze(0),))[0].reshape(-1)

    w = torch.cat([post.mean[n].reshape(-1) for n in names])

    return torch.stack([jacrev(f)(w, xi) for xi in x])


_CASES = [
    ("mlp", "diag"),
    ("mlp", "dense"),
    ("mlp", "kfac"),
    ("conv", "diag"),
    ("conv", "dense"),
]


def _case(net: str, kind: str) -> tuple[LaplacePosterior, Tensor, Tensor]:
    g = torch.Generator().manual_seed(2)

    if net == "mlp":
        post, sigma = _post(_mlp(7), kind, ALL)
        x = torch.randn(5, 3, generator=g, dtype=F64)
    else:
        post, sigma = _post(_conv(), kind, ALL)
        x = torch.randn(3, 1, 4, 4, generator=g, dtype=F64)

    return post, sigma, x


@pytest.mark.parametrize(("net", "kind"), _CASES)
@pytest.mark.parametrize("chunk_size", [3, None])
def test_jvp_outputs_equal_dense_reference(net: str, kind: str, chunk_size) -> None:
    post, sigma, x = _case(net, kind)
    jac = _ref_jac(post, x)
    out = predict_outputs(
        post, x, strategy="forward-mode", n_samples="exact", chunk_size=chunk_size
    )
    f = post.model(post.mean, x)

    assert isinstance(out.cov, FactorCov)
    assert out.cov.factor.shape == (*f.shape, sigma.shape[0])
    assert out.cov.factor.dtype == F64
    assert torch.equal(out.mean, f)
    torch.testing.assert_close(out.cov.per_input(), jac @ sigma @ jac.mT, rtol=0, atol=1e-10)


@pytest.mark.parametrize(("net", "kind"), _CASES)
def test_jvp_and_vjp_agree(net: str, kind: str) -> None:
    post, _, x = _case(net, kind)
    a = predict_outputs(post, x, strategy="forward-mode", n_samples="exact")
    b = predict_outputs(post, x, strategy="reverse-mode", n_samples="exact")

    assert isinstance(a.cov, FactorCov) and isinstance(b.cov, PerInputCov)
    torch.testing.assert_close(a.cov.per_input(), b.cov.per_input(), rtol=0, atol=1e-10)
    torch.testing.assert_close(a.cov.diagonal(), b.cov.diagonal(), rtol=0, atol=1e-10)


def test_resolve_via_table() -> None:
    fwd = Capabilities(jvp=True, vmap=True)

    # Both exact: forward mode when it has fewer columns than the dense blocks.
    assert resolve_strategy(ALL, 10, 5) == "forward-mode"
    assert resolve_strategy(ALL, 5, 10) == "reverse-mode"
    assert resolve_strategy(ALL, 5, 5) == "reverse-mode"
    assert resolve_strategy(ALL, None, 5) == "reverse-mode"

    # Only one exact path fits.
    assert resolve_strategy(ALL, MAX_DENSE_EVENT + 1, 10) == "forward-mode"
    assert resolve_strategy(fwd, 10, 100) == "forward-mode"
    assert resolve_strategy(ALL, 10, _MAX_FACTOR_ENTRIES) == "reverse-mode"  # E * P over the budget
    assert resolve_strategy(Capabilities(vjp=True, vmap=True), 10, 5) == "reverse-mode"

    # Neither: sampled, by a VJP of a VJP (one recorded forward) when the probe found
    # vjp_twice, else by a JVP per draw.
    assert (
        resolve_strategy(Capabilities(vjp=True, vjp_twice=True, jvp=True), 10, 5) == "reverse-mode"
    )
    assert resolve_strategy(Capabilities(vjp=True, vjp_twice=True), 10, 5) == "reverse-mode"
    assert resolve_strategy(Capabilities(vjp=True, jvp=True), 10, 5) == "forward-mode"
    assert resolve_strategy(ALL, MAX_DENSE_EVENT + 1, _MAX_FACTOR_ENTRIES) == "reverse-mode"

    # Explicit exact outputs ignore the budgets: the cheaper path the capabilities allow.
    huge = MAX_DENSE_EVENT + 1

    assert (
        resolve_strategy(ALL, huge, _MAX_FACTOR_ENTRIES, n_samples="exact") == "reverse-mode"
    )  # P > E
    assert resolve_strategy(ALL, huge, 10, n_samples="exact") == "forward-mode"  # P < E
    assert resolve_strategy(ALL, 10, _MAX_FACTOR_ENTRIES, n_samples="exact") == "reverse-mode"
    assert resolve_strategy(ALL, None, 10, n_samples="exact") == "reverse-mode"
    assert resolve_strategy(fwd, huge, _MAX_FACTOR_ENTRIES, n_samples="exact") == "forward-mode"
    assert (
        resolve_strategy(Capabilities(vjp=True, vmap=True), huge, 10, n_samples="exact")
        == "reverse-mode"
    )
    twice_fwd = Capabilities(vjp=True, vjp_twice=True, jvp=True)

    assert resolve_strategy(twice_fwd, 10, 5, n_samples="exact") == "forward-mode"
    assert (
        resolve_strategy(Capabilities(vjp=True, vjp_twice=True), 10, 5, n_samples="exact")
        == "reverse-mode"
    )

    # Explicit sample counts: as sampled outputs.
    assert resolve_strategy(ALL, 5, 10, n_samples=8) == "reverse-mode"
    assert resolve_strategy(ALL, 10, 5, n_samples=8) == "reverse-mode"
    assert (
        resolve_strategy(Capabilities(vjp=True, jvp=True, vmap=True), 5, 10, n_samples=8)
        == "forward-mode"
    )
    assert (
        resolve_strategy(Capabilities(vjp=True, vjp_twice=True, vmap=True), 5, 10, n_samples=8)
        == "reverse-mode"
    )


def test_explicit_samples_use_jvp_without_vjp_twice() -> None:
    """P = 51 > E = 7 would pick vjp for exact outputs; sampled ones need only one JVP."""

    post, _, x = _case("mlp", "dense")
    post = dataclasses.replace(post, caps=Capabilities(vjp=True, jvp=True, vmap=True))
    out = predict_outputs(post, x, n_samples=8)

    assert isinstance(out.cov, SampleCov)
    assert out.cov.draws.shape[0] == 8


def _few_params(caps: Capabilities = ALL) -> LaplacePosterior:
    """f(x) = tanh(x w^T) flattened: E = 3 * 4 = 12 outputs per input, P = 4 parameters."""

    w = torch.randn(4, generator=torch.Generator().manual_seed(3), dtype=F64)
    mean = {"w": w}

    return LaplacePosterior(
        mean=mean,
        curvature=Dense(_spd(4, torch.Generator().manual_seed(4))),
        prior=Isotropic(torch.tensor(0.5, dtype=F64)),
        n_data=torch.tensor(3.0, dtype=F64),
        model=ModelFn(lambda p, x: torch.tanh(x.unsqueeze(-1) * p["w"]).flatten(1)),
        likelihood=None,
        layout=Layout.by_module(mean),
        caps=caps,
    )


def test_default_via_gives_factor_when_fewer_params_than_outputs() -> None:
    x = torch.randn(5, 3, generator=torch.Generator().manual_seed(2), dtype=F64)

    # Resolved after the forward (the probe's event size is unknown) and from the probe's.
    for caps in (ALL, dataclasses.replace(ALL, event_size=12)):
        post = _few_params(caps)
        out = predict_outputs(post, x)

        assert isinstance(out.cov, FactorCov)
        torch.testing.assert_close(
            out.cov.per_input(),
            predict_outputs(post, x, strategy="reverse-mode").cov.per_input(),
            rtol=0,
            atol=1e-10,
        )


def test_default_via_gives_per_input_when_more_params_than_outputs() -> None:
    post, _, x = _case("mlp", "dense")

    assert isinstance(predict_outputs(post, x).cov, PerInputCov)


def test_default_samples_follow_factor_budget(monkeypatch) -> None:
    post = _few_params()
    x = torch.randn(5, 3, generator=torch.Generator().manual_seed(2), dtype=F64)

    monkeypatch.setattr("laplacekit.core.posteriors._MAX_FACTOR_ENTRIES", 4)
    monkeypatch.setattr("laplacekit.core.posteriors.MAX_DENSE_EVENT", 4)
    out = predict_outputs(post, x)

    assert isinstance(out.cov, SampleCov)
    assert out.cov.draws.shape == (64, 5, 12)

    # The budget steers the default only: explicit n_samples="exact" stays exact.
    assert isinstance(
        predict_outputs(post, x, strategy="forward-mode", n_samples="exact").cov, FactorCov
    )
    assert isinstance(
        predict_outputs(post, x, strategy="reverse-mode", n_samples="exact").cov, PerInputCov
    )


def test_explicit_exact_picks_the_cheaper_path_over_budget(monkeypatch) -> None:
    """n_samples="exact" with neither budget fitting: factor if P < E, else dense blocks."""

    monkeypatch.setattr("laplacekit.core.posteriors._MAX_FACTOR_ENTRIES", 1)
    monkeypatch.setattr("laplacekit.core.posteriors.MAX_DENSE_EVENT", 1)
    x = torch.randn(5, 3, generator=torch.Generator().manual_seed(2), dtype=F64)

    assert isinstance(
        predict_outputs(_few_params(), x, n_samples="exact").cov, FactorCov
    )  # P 4 < E 12

    post, _, x = _case("mlp", "dense")  # P = 51 > E = 7

    assert isinstance(predict_outputs(post, x, n_samples="exact").cov, PerInputCov)


def _counted(post: LaplacePosterior) -> tuple[LaplacePosterior, list[int]]:
    calls: list[int] = []
    model = post.model

    def counted(params, inputs):
        calls.append(1)
        return model(params, inputs)

    return dataclasses.replace(post, model=ModelFn(counted, model.module)), calls


def test_jvp_refuses_without_jvp_before_forward() -> None:
    no_jvp = Capabilities(vjp=True, vjp_twice=True, vmap=True)
    post, calls = _counted(_case("mlp", "dense")[0])
    post = dataclasses.replace(post, caps=no_jvp)
    x = torch.zeros(2, 3, dtype=F64)

    for n_samples in ("exact", None, 8):
        with pytest.raises(CapabilityError, match="jvp"):
            predict_outputs(post, x, strategy="forward-mode", n_samples=n_samples)

    no_vmap = dataclasses.replace(post, caps=Capabilities(jvp=True))

    with pytest.raises(CapabilityError, match="vmap"):
        predict_outputs(no_vmap, x, strategy="forward-mode", n_samples="exact")

    assert calls == []


def test_outputs_resolves_with_the_calls_own_event_size() -> None:
    """vjp and vmap only, probe E = 4098 > MAX_DENSE_EVENT, call E = 12: `n_samples=None`
    is exact on the call's outputs, not refused for the probe's event size."""

    post = _few_params(Capabilities(vjp=True, vmap=True, event_size=MAX_DENSE_EVENT + 2))
    x = torch.randn(2, 3, generator=torch.Generator().manual_seed(2), dtype=F64)
    want = predict_outputs(_few_params(), x, strategy="reverse-mode", n_samples="exact")
    got = predict_outputs(post, x)

    assert isinstance(got.cov, PerInputCov)
    torch.testing.assert_close(got.cov.per_input(), want.cov.per_input(), rtol=0, atol=1e-12)


def test_scalar_input_leaf_is_shared() -> None:
    """A 0-dim leaf reaches the model unchanged and acts like the same value per input."""

    w = torch.randn(3, 4, generator=torch.Generator().manual_seed(3), dtype=F64)
    mean = {"fc.weight": w}
    seen: list[int] = []

    def f(p, inp):
        seen.append(inp["t"].ndim)
        return torch.tanh(inp["x"] @ p["fc.weight"].T * inp["t"].reshape(-1, 1))

    post = LaplacePosterior(
        mean=mean,
        curvature=Dense(_spd(12, torch.Generator().manual_seed(4))),
        prior=Isotropic(torch.tensor(0.5, dtype=F64)),
        n_data=torch.tensor(3.0, dtype=F64),
        model=ModelFn(f),
        likelihood=None,
        layout=Layout.by_module(mean),
        caps=ALL,
    )
    x = torch.randn(5, 4, generator=torch.Generator().manual_seed(2), dtype=F64)
    t = torch.tensor(0.7, dtype=F64)
    expanded = predict_outputs(
        post, {"x": x, "t": t.expand(5)}, strategy="reverse-mode", n_samples="exact"
    )
    seen.clear()

    for strategy in ("forward-mode", "reverse-mode"):
        out = predict_outputs(post, {"x": x, "t": t}, strategy=strategy, n_samples="exact")

        torch.testing.assert_close(out.mean, expanded.mean, rtol=0, atol=1e-12)
        torch.testing.assert_close(
            out.cov.per_input(), expanded.cov.per_input(), rtol=0, atol=1e-10
        )

    assert seen and set(seen) == {0}


def test_large_event_with_few_params_gives_factor() -> None:
    """E > MAX_DENSE_EVENT and a small subset: exact by default, through the factor."""

    w = torch.randn(3, generator=torch.Generator().manual_seed(3), dtype=F64)
    mean = {"w": w}
    post = LaplacePosterior(
        mean=mean,
        curvature=Diag(torch.tensor([1.0, 2.0, 3.0], dtype=F64)),
        prior=Isotropic(torch.tensor(0.5, dtype=F64)),
        n_data=torch.tensor(1.0, dtype=F64),
        model=ModelFn(lambda p, x: x * p["w"].sum() + x**2 * p["w"][0]),
        likelihood=None,
        layout=Layout.by_module(mean),
        caps=dataclasses.replace(ALL, event_size=5000),
    )
    x = torch.randn(2, 5000, generator=torch.Generator().manual_seed(2), dtype=F64)
    out = predict_outputs(post, x)

    # J = (x + x^2, x, x) and Sigma = diag(1 / (d + 0.5)).
    var = 1 / torch.tensor([1.5, 2.5, 3.5], dtype=F64)
    ref = x**2 * var.sum() + 2 * x**3 * var[0] + x**4 * var[0]

    assert isinstance(out.cov, FactorCov)
    torch.testing.assert_close(out.cov.diagonal(), ref, rtol=1e-10, atol=1e-12)

    with pytest.raises(StructureError, match="MAX_DENSE_EVENT"):
        out.cov.per_input()


def _joint_post() -> tuple[LaplacePosterior, Tensor]:
    """A posterior over three blocks with one dense curvature across all of them."""

    g = torch.Generator().manual_seed(5)
    params = {
        "a.weight": torch.randn(2, generator=g, dtype=torch.float64),
        "b.weight": torch.randn(3, generator=g, dtype=torch.float64),
        "c.weight": torch.randn(2, 2, generator=g, dtype=torch.float64),
    }
    a = torch.randn(9, 9, generator=g, dtype=torch.float64)
    h = a @ a.T / 9
    post = LaplacePosterior(
        mean=params,
        curvature=Joint(Dense(h), (("a", 2), ("b", 3), ("c", 4))),
        prior=Isotropic(torch.tensor(1.0, dtype=torch.float64)),
        n_data=torch.tensor(3.0, dtype=torch.float64),
        model=_model(),
        likelihood=None,
        layout=Layout.by_module(params),
    )

    return post, 3 * h + torch.eye(9, dtype=torch.float64)


def test_laplace_gaussian_over_joint() -> None:
    post, prec = _joint_post()
    cov = torch.linalg.inv(prec)
    g = torch.Generator().manual_seed(6)
    v = {k: torch.randn(x.shape, generator=g, dtype=torch.float64) for k, x in post.mean.items()}
    flat_v = torch.cat([x.reshape(-1) for x in v.values()])
    got = post.cov_mv(v)

    assert got["c.weight"].shape == (2, 2)
    assert torch.allclose(torch.cat([x.reshape(-1) for x in got.values()]), cov @ flat_v)

    mv = post.variance
    assert torch.allclose(torch.cat([x.reshape(-1) for x in mv.values()]), cov.diagonal())

    noise = post.sample_centered(20000, generator=torch.Generator().manual_seed(0))
    assert noise["c.weight"].shape == (20000, 2, 2)
    flat = torch.cat([x.reshape(20000, -1) for x in noise.values()], dim=-1)
    emp = flat.T @ flat / flat.shape[0]
    assert (emp - cov).abs().max() < 0.05

    p = {k: x + 0.1 for k, x in post.mean.items()}
    mvn = torch.distributions.MultivariateNormal(
        torch.cat([x.reshape(-1) for x in post.mean.values()]), precision_matrix=prec
    )
    assert torch.allclose(
        post.log_prob(p), mvn.log_prob(torch.cat([x.reshape(-1) for x in p.values()]))
    )

    assert "curvature  = Joint[Dense] (3 blocks, 9 parameters)" in repr(post)


def test_joint_posterior_as_prior_adds_precision() -> None:
    post, prec = _joint_post()
    h2 = torch.eye(9, dtype=torch.float64)
    out = post.add_precision_to(Joint(Dense(h2), (("a", 2), ("b", 3), ("c", 4))))

    assert isinstance(out, Joint)
    assert torch.allclose(out.op.mat, prec + h2)


# --- batched covariance products (HasCovMany) -------------------------------------------


def _joint_dense_fit() -> tuple[LaplacePosterior, Tensor, Tensor, Tensor]:
    """A `DenseGGN(joint=True)` posterior, a test batch, its dense Jacobians [N, E, P] and
    the dense covariance, built from `jacrev` (precision = J^T J / sigma^2 + lambda I)."""

    torch.manual_seed(0)
    module = nn.Sequential(nn.Linear(2, 6), nn.Tanh(), nn.Linear(6, 3)).to(F64)
    g = torch.Generator().manual_seed(3)
    x = torch.randn(10, 2, generator=g, dtype=F64)
    y = torch.randn(10, 3, generator=g, dtype=F64)
    sigma, lam = 0.5, 0.7
    post = lk.laplace(
        module,
        [(x[:5], y[:5]), (x[5:], y[5:])],
        likelihood=copy.replace(
            lk.Gaussian(), sigma=torch.as_tensor(torch.tensor(sigma, dtype=F64))
        ),
        hessian_approx=lk.DenseGGN(joint=True),
        prior=Isotropic(torch.tensor(lam, dtype=F64)),
        subset=lk.subsets.all,
        hyper_tuner=None,
    )
    jac = _ref_jac(post, x).flatten(0, 1)  # [N E, P]
    eye = torch.eye(jac.shape[-1], dtype=F64)
    sigma_w = torch.linalg.inv(jac.T @ jac / sigma**2 + lam * eye)
    x_test = torch.randn(4, 2, generator=g, dtype=F64)

    return post, x_test, _ref_jac(post, x_test), sigma_w


def _refuse(*_: object) -> object:
    raise AssertionError("the single-vector product was called")


def test_outputs_uses_cov_many(monkeypatch) -> None:
    post, x, jac, sigma = _joint_dense_fit()
    ref = jac @ sigma @ jac.mT

    assert isinstance(post, HasCovMany)

    monkeypatch.setattr(LaplacePosterior, "cov_mv", _refuse)
    monkeypatch.setattr(LaplacePosterior, "cov_sqrt_mv", _refuse)

    vjp_out = predict_outputs(post, x, strategy="reverse-mode", n_samples="exact")
    jvp_out = predict_outputs(post, x, strategy="forward-mode", n_samples="exact")

    assert isinstance(vjp_out.cov, PerInputCov) and isinstance(jvp_out.cov, FactorCov)
    torch.testing.assert_close(vjp_out.cov.per_input(), ref, rtol=0, atol=1e-10)
    torch.testing.assert_close(jvp_out.cov.per_input(), ref, rtol=0, atol=1e-10)


def test_cov_many_matches_dense_covariance() -> None:
    post, _, _, sigma = _joint_dense_fit()
    p = sigma.shape[0]
    names = list(post.mean)
    eye = torch.eye(p, dtype=F64)
    rows = torch.randn(5, p, generator=torch.Generator().manual_seed(8), dtype=F64)

    def unflat(m: Tensor) -> dict[str, Tensor]:
        parts = torch.split(m, [post.mean[n].numel() for n in names], dim=-1)
        return {
            n: t.reshape(m.shape[0], *post.mean[n].shape) for n, t in zip(names, parts, strict=True)
        }

    def flat(v: Mapping[str, Tensor]) -> Tensor:
        return torch.cat([v[n].reshape(v[n].shape[0], -1) for n in names], dim=-1)

    got = post.cov_many(unflat(rows))
    root = flat(post.cov_sqrt_many(unflat(eye))).T  # columns L e_i

    assert all(got[n].shape == (5, *post.mean[n].shape) for n in names)
    assert all(got[n].dtype == post.mean[n].dtype for n in names)
    torch.testing.assert_close(flat(got), rows @ sigma, rtol=0, atol=1e-10)
    torch.testing.assert_close(root @ root.T, sigma, rtol=0, atol=1e-10)


class _PluginGaussian:
    """A plugin `WeightPosterior` that only delegates: it has no `cov_many`."""

    def __init__(self, inner: LaplacePosterior) -> None:
        self.inner = inner

    @property
    def mean(self):
        return self.inner.mean

    @property
    def model(self):
        return self.inner.model

    @property
    def likelihood(self):
        return self.inner.likelihood

    @property
    def layout(self):
        return self.inner.layout

    @property
    def caps(self):
        return self.inner.caps

    def precision(self):
        return self.inner.precision()

    def cov_mv(self, v):
        return self.inner.cov_mv(v)

    def cov_sqrt_mv(self, v):
        return self.inner.cov_sqrt_mv(v)

    def sample(self, n, *, generator=None):
        return self.inner.sample(n, generator=generator)

    def sample_centered(self, n, *, generator=None):
        return self.inner.sample_centered(n, generator=generator)

    @property
    def variance(self):
        return self.inner.variance

    def log_prob(self, params):
        return self.inner.log_prob(params)

    def location(self, like):
        return self.inner.location(like)

    def add_precision_to(self, op):
        return self.inner.add_precision_to(op)


def test_plugin_posterior_without_cov_many() -> None:
    post, x, jac, sigma = _joint_dense_fit()
    plugin = _PluginGaussian(post)
    ref = jac @ sigma @ jac.mT

    assert isinstance(plugin, WeightPosterior) and not isinstance(plugin, HasCovMany)

    vjp_out = predict_outputs(plugin, x, strategy="reverse-mode", n_samples="exact")
    jvp_out = predict_outputs(plugin, x, strategy="forward-mode", n_samples="exact")

    assert isinstance(vjp_out.cov, PerInputCov) and isinstance(jvp_out.cov, FactorCov)
    torch.testing.assert_close(vjp_out.cov.per_input(), ref, rtol=0, atol=1e-10)
    torch.testing.assert_close(jvp_out.cov.per_input(), ref, rtol=0, atol=1e-10)


class _Versioned(_PluginGaussian):
    """A plugin `WeightPosterior` with its own staleness guard, which can be made stale."""

    stale = False

    def check_versions(self) -> None:
        if self.stale:
            raise RuntimeError("plugin posterior is stale")


def test_has_versions_guards_plugin_posteriors() -> None:
    post, x, _, _ = _joint_dense_fit()
    calls = []
    model = post.model
    counted = dataclasses.replace(
        post, model=ModelFn(lambda p, xb: calls.append(1) or model(p, xb), model.module)
    )
    plugin = _Versioned(counted)

    assert isinstance(post, HasVersions) and isinstance(plugin, HasVersions)
    assert not isinstance(_PluginGaussian(post), HasVersions)

    plugin.stale = True

    for call in (
        lambda: mean_outputs(plugin, x),
        lambda: predict_outputs(plugin, x),
        lambda: lk.sample_outputs(plugin, x, 2),
    ):
        with pytest.raises(RuntimeError, match="plugin posterior is stale"):
            call()

    assert calls == []

    plugin.stale = False
    torch.testing.assert_close(mean_outputs(plugin, x), model.module(x))


def test_dense_predict_time() -> None:
    """`lk.predict` on a 2,701-weight dense posterior at 400 inputs: 26 s with one
    vmapped solve per vector, 0.15 s batched. The bound of 1 s leaves room for a loaded
    machine and still catches the per-vector path."""

    torch.manual_seed(0)
    module = nn.Sequential(
        nn.Linear(1, 50), nn.Tanh(), nn.Linear(50, 50), nn.Tanh(), nn.Linear(50, 1)
    ).to(F64)
    x = torch.linspace(-2.0, 2.0, 64, dtype=F64).reshape(-1, 1)
    post = lk.laplace(
        module,
        [(x, torch.sin(3 * x))],
        likelihood=copy.replace(lk.Gaussian(), sigma=torch.as_tensor(torch.tensor(0.1, dtype=F64))),
        hessian_approx=lk.DenseGGN(joint=True),
        prior=Isotropic(torch.tensor(1.0, dtype=F64)),
        subset=lk.subsets.all,
        hyper_tuner=None,
    )
    x_test = torch.linspace(-2.0, 2.0, 400, dtype=F64).reshape(-1, 1)

    assert post.layout.n_params == 2701

    start = time.perf_counter()
    pred = lk.predict(post, x_test)
    elapsed = time.perf_counter() - start

    assert pred.stddev.shape == (400, 1)
    assert elapsed < 1.0, f"dense predict took {elapsed:.2f} s"
