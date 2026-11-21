"""The Jacobian helpers and the projected GGN, against dense `jacrev` references."""

import copy
from collections.abc import Mapping

import pytest
import torch
from laplacekit_gp._gram import GramStat, ProjectedGram
from laplacekit_gp._jacobian import choose_via, features, rows
from torch import Tensor, nn
from torch.func import functional_call, jacrev

import laplacekit as lk
import laplacekit.core as core
from laplacekit.core import Capabilities, CapabilityError, FitContext, Layout, StructureError

E = 3


def model(out: int = E) -> nn.Module:
    torch.manual_seed(0)

    return nn.Sequential(nn.Linear(2, 4), nn.Tanh(), nn.Linear(4, out)).double()


def params_of(net: nn.Module) -> dict[str, Tensor]:
    return {k: v.detach() for k, v in net.named_parameters()}


def data(n: int, seed: int = 0) -> tuple[Tensor, Tensor]:
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 2, generator=g, dtype=torch.float64)
    y = torch.randint(0, E, (n,), generator=g)

    return x, y


def dense_jacobian(net: nn.Module, x: Tensor) -> Tensor:
    """[N, E', P] by `jacrev` of the whole batch."""

    params = params_of(net)

    def fwd(p: Mapping[str, Tensor]) -> Tensor:
        return functional_call(net, dict(p), (x,)).reshape(x.shape[0], -1)

    jac = jacrev(fwd)(params)

    return torch.cat([jac[k].reshape(*jac[k].shape[:2], -1) for k in params], dim=2)


def basis_for(net: nn.Module, r: int, seed: int = 1) -> Tensor:
    p = sum(v.numel() for v in net.parameters())
    g = torch.Generator().manual_seed(seed)

    return torch.randn(p, r, generator=g, dtype=torch.float64)


def context(net: nn.Module, lik=None) -> FitContext:
    params = params_of(net)
    lik = lk.Categorical() if lik is None else lik

    return FitContext(
        core.ModelFn.from_module(net), lik, Layout.by_module(params), core.Capabilities()
    )


def fold(gram: ProjectedGram, net: nn.Module, batches: list[lk.Batch], lik=None) -> GramStat:
    ctx = context(net, lik)
    params = params_of(net)
    stat = gram.init(params, ctx)

    for i, b in enumerate(batches):
        stat = gram.update(stat, params, b, torch.Generator().manual_seed(i), ctx)

    return stat


def dense_ggn(net: nn.Module, x: Tensor, y: Tensor, w: Tensor | None = None, lik=None) -> Tensor:
    """Σ wₙ Jₙᵀ Hₙ Jₙ with H the observed Hessian (the Fisher for these likelihoods)."""

    j = dense_jacobian(net, x)
    f = net(x).detach()
    lik = lk.Categorical() if lik is None else lik
    h = core.observed_output_hessian(lik, f, y).dense()
    w = torch.ones(x.shape[0], dtype=torch.float64) if w is None else w

    return torch.einsum("n,nep,nef,nfq->pq", w, j, h, j)


# --- the Jacobian helpers -------------------------------------------------------------------


def test_rows_equal_jacrev():
    net = model()
    x, _ = data(6)
    params = params_of(net)
    fn = core.ModelFn.from_module(net)
    ref = dense_jacobian(net, x)  # [N, E, P]

    full = rows(fn, params, x, None, torch.float64)
    torch.testing.assert_close(full, ref.reshape(-1, ref.shape[-1]))

    coords = torch.tensor([0, 2, 1, 1, 0, 2])
    picked = rows(fn, params, x, coords, torch.float64)
    torch.testing.assert_close(picked, ref[torch.arange(6), coords])


def test_rows_cast_and_share_scalar_inputs():
    class Scaled(nn.Module):
        def __init__(self):
            super().__init__()
            torch.manual_seed(0)
            self.lin = nn.Linear(2, E).double()

        def forward(self, x, t):
            return self.lin(x) * t

    net = Scaled()
    x, _ = data(4)
    params = params_of(net)
    fn = core.ModelFn.from_module(net)
    inputs = {"x": x, "t": torch.tensor(2.0, dtype=torch.float64)}

    def fwd(p):
        return functional_call(net, dict(p), (), {"x": x, "t": 2.0}).reshape(4, -1)

    jac = jacrev(fwd)(params)
    ref = torch.cat([jac[k].reshape(*jac[k].shape[:2], -1) for k in params], dim=2)

    out = rows(fn, params, inputs, None, torch.float32)
    assert out.dtype == torch.float32
    torch.testing.assert_close(out, ref.reshape(-1, ref.shape[-1]).float())


@pytest.mark.parametrize("r", [5, 70])
def test_jvp_and_vjp_features_agree(r):
    net = model()
    x, _ = data(5)
    params = params_of(net)
    fn = core.ModelFn.from_module(net)
    basis = basis_for(net, r)
    ref = dense_jacobian(net, x) @ basis  # [N, E, r]

    for strategy in ("forward-mode", "reverse-mode"):
        f, phi = features(fn, params, x, basis, strategy, torch.float64)
        torch.testing.assert_close(f, net(x).detach())
        assert phi.shape == (5, E, r)
        torch.testing.assert_close(phi, ref)


def test_choose_via_by_capability():
    """Forward mode whenever the probe found it (with vmap), else reverse mode."""

    assert choose_via(Capabilities(vjp=True, jvp=True, vmap=True)) == "forward-mode"
    assert choose_via(Capabilities(jvp=True, vmap=True)) == "forward-mode"
    assert choose_via(Capabilities(vjp=True, vmap=True)) == "reverse-mode"

    neither = Capabilities(vjp=True, jvp=True)
    strategy = choose_via(neither)

    assert strategy == "reverse-mode"

    with pytest.raises(CapabilityError, match="vmap"):
        neither.check(ProjectedGram(torch.zeros(3, 2), strategy).requires, "ProjectedGram")


# --- the projected GGN ----------------------------------------------------------------------


def targets(kind: str, x: Tensor, y: Tensor) -> tuple[object, Tensor]:
    """A likelihood and matching targets: categorical, Gaussian with a per-output sigma,
    and Bernoulli on every output."""

    if kind == "categorical":
        return lk.Categorical(), y

    g = torch.Generator().manual_seed(2)

    if kind == "gaussian":
        sigma = torch.tensor([0.5, 1.0, 2.0], dtype=torch.float64)
        return copy.replace(lk.Gaussian(), sigma=torch.as_tensor(sigma)), torch.randn(
            x.shape[0], E, generator=g
        ).double()

    return lk.Bernoulli(), torch.randint(0, 2, (x.shape[0], E), generator=g).double()


@pytest.mark.parametrize("kind", ["categorical", "gaussian", "bernoulli"])
@pytest.mark.parametrize("strategy", ["forward-mode", "reverse-mode"])
def test_gram_equals_dense_projected_ggn(strategy, kind):
    net = model()
    x, labels = data(12)
    lik, y = targets(kind, x, labels)
    basis = basis_for(net, 4)
    batches = [lk.Batch(x[:5], y[:5]), lk.Batch(x[5:], y[5:])]

    stat = fold(ProjectedGram(basis, strategy), net, batches, lik)
    out = ProjectedGram(basis, strategy).finalize(stat)
    ref = basis.T @ dense_ggn(net, x, y, lik=lik) @ basis

    torch.testing.assert_close(stat.gram, ref)
    torch.testing.assert_close(stat.gram, stat.gram.T, rtol=0, atol=0)
    torch.testing.assert_close(out.gram, ref / 12)
    torch.testing.assert_close(out.weight, torch.tensor(12.0, dtype=torch.float64))


class _Counted:
    """An output Hessian that counts its `mv` and `sqrt_mv` calls."""

    def __init__(self, inner, calls):
        self.inner, self.calls = inner, calls

    def mv(self, v):
        self.calls["mv"] += 1
        return self.inner.mv(v)

    def sqrt_mv(self, v):
        self.calls["sqrt_mv"] += 1
        return self.inner.sqrt_mv(v)

    def diagonal(self):
        return self.inner.diagonal()

    def dense(self):
        return self.inner.dense()


class _Counting:
    """A categorical likelihood whose output Hessians count their calls."""

    def __init__(self):
        self.lik = lk.Categorical()
        self.calls = {"mv": 0, "sqrt_mv": 0}

    def log_prob(self, f, y):
        return self.lik.log_prob(f, y)

    def output_hessian(self, f):
        return _Counted(self.lik.output_hessian(f), self.calls)


@pytest.mark.parametrize("out", [3, 40])
def test_gram_calls_the_hessian_once_per_batch_whatever_the_event_size(out):
    net = model(out)
    x, _ = data(6)
    y = torch.randint(0, out, (6,))
    lik = _Counting()
    batches = [lk.Batch(x[:3], y[:3]), lk.Batch(x[3:], y[3:])]

    stat = fold(ProjectedGram(basis_for(net, 5), "reverse-mode"), net, batches, lik)
    ref = basis_for(net, 5).T @ dense_ggn(net, x, y) @ basis_for(net, 5)

    torch.testing.assert_close(stat.gram, ref)
    assert lik.calls == {"mv": 2, "sqrt_mv": 0}


def test_finalize_refuses_zero_weight():
    net = model()
    x, y = data(4)
    gram = ProjectedGram(basis_for(net, 2), "reverse-mode")
    stat = fold(gram, net, [lk.Batch(x, y, weights=torch.zeros(4, dtype=torch.float64))])

    with pytest.raises(ValueError, match="positive weight"):
        gram.finalize(stat)


def test_gram_monoid_law():
    net = model()
    x, y = data(10)
    gram = ProjectedGram(basis_for(net, 3), "reverse-mode")
    a = [lk.Batch(x[:4], y[:4])]
    b = [lk.Batch(x[4:7], y[4:7]), lk.Batch(x[7:], y[7:])]
    ctx = context(net)
    params = params_of(net)

    def fold_from(stat, batches, start):
        for i, batch in enumerate(batches):
            stat = gram.update(stat, params, batch, torch.Generator().manual_seed(start + i), ctx)

        return stat

    whole = fold_from(gram.init(params, ctx), a + b, 0)
    parts = fold_from(gram.init(params, ctx), a, 0) + fold_from(gram.init(params, ctx), b, 1)
    zero = gram.init(params, ctx)

    torch.testing.assert_close(whole.gram, parts.gram)
    torch.testing.assert_close(whole.weight, parts.weight)
    torch.testing.assert_close((zero + whole).gram, whole.gram)
    torch.testing.assert_close(whole.scaled(torch.tensor(2.0)).gram, 2 * whole.gram)


def test_weights_scale_gram():
    net = model()
    x, y = data(8)
    w = torch.linspace(0.5, 2.0, 8, dtype=torch.float64)
    basis = basis_for(net, 4)

    stat = fold(ProjectedGram(basis, "forward-mode"), net, [lk.Batch(x, y, weights=w)])
    ref = basis.T @ dense_ggn(net, x, y, w) @ basis

    torch.testing.assert_close(stat.gram, ref)
    torch.testing.assert_close(stat.weight, w.sum())


def test_gram_requirements_follow_via():
    basis = torch.zeros(3, 2)
    caps = Capabilities(vjp=True, vmap=True)

    vjp = ProjectedGram(basis, "reverse-mode").requires
    jvp = ProjectedGram(basis, "forward-mode").requires

    assert vjp.needs_output_hessian and vjp.vjp and vjp.vmap and not vjp.jvp
    assert jvp.needs_output_hessian and jvp.jvp and jvp.vmap and not jvp.vjp

    caps.check(vjp, "ProjectedGram")

    with pytest.raises(CapabilityError, match="jvp"):
        caps.check(jvp, "ProjectedGram")


def test_gram_refuses_unknown_via():
    with pytest.raises(ValueError, match=r'"forward-mode" or "reverse-mode", got \'foo\''):
        ProjectedGram(torch.zeros(3, 2), "foo")  # ty: ignore[invalid-argument-type]


def test_gram_refuses_a_basis_of_the_wrong_size():
    net = model()
    ctx = context(net)

    with pytest.raises(StructureError, match="basis"):
        ProjectedGram(basis_for(net, 2)[:-1], "reverse-mode").init(params_of(net), ctx)


# --- the MC Fisher, masks, parameter order and output structure --------------------------


class _SampledCategorical:
    """A categorical likelihood without an output Hessian: the fit takes the MC Fisher."""

    def log_prob(self, f, y):
        return torch.distributions.Categorical(logits=f).log_prob(y)

    def distribution(self, f):
        return torch.distributions.Categorical(logits=f)


def test_gram_with_the_mc_fisher_equals_its_dense_form():
    net = model()
    x, y = data(9)
    lik = _SampledCategorical()
    basis = basis_for(net, 4)
    batches = [lk.Batch(x[:4], y[:4]), lk.Batch(x[4:], y[4:])]

    assert not isinstance(lik, core.HasOutputHessian)

    stat = fold(ProjectedGram(basis, "reverse-mode"), net, batches, lik)

    # The same draws, by batch generator, as dense [E, E] blocks.
    ref = torch.zeros(4, 4, dtype=torch.float64)

    for i, b in enumerate(batches):
        j = dense_jacobian(net, b.inputs)
        f = net(b.inputs).detach()
        h = core.output_hessian(lik, f, generator=torch.Generator().manual_seed(i)).dense()
        ref += basis.T @ torch.einsum("nep,nef,nfq->pq", j, h, j) @ basis

    torch.testing.assert_close(stat.gram, ref)


def test_mc_fisher_refused_for_a_large_event_before_data(monkeypatch):
    import laplacekit_gp._gram as gram_module

    net = model()
    monkeypatch.setattr(gram_module, "MAX_DENSE_EVENT", E - 1)
    ctx = context(net, _SampledCategorical())
    ctx = FitContext(ctx.model, ctx.likelihood, ctx.layout, Capabilities(event_size=E))

    with pytest.raises(StructureError, match="MC Fisher"):
        ProjectedGram(basis_for(net, 2), "reverse-mode").init(params_of(net), ctx)


def test_gram_drops_masked_entries():
    net = model()
    x, y = data(10)
    keep = x[:, 0] > 0
    lik = lk.Categorical(mask_fn=lambda f, inputs: inputs[:, 0] > 0)
    basis = basis_for(net, 4)

    assert 0 < int(keep.sum()) < 10

    stat = fold(ProjectedGram(basis, "forward-mode"), net, [lk.Batch(x, y)], lik)
    ref = basis.T @ dense_ggn(net, x[keep], y[keep]) @ basis

    torch.testing.assert_close(stat.gram, ref)


def test_basis_rows_follow_the_params_order():
    """A subset of the parameters, in reverse module order: P follows the mapping."""

    net = model()
    x, y = data(7)
    full = params_of(net)
    names = ["2.bias", "2.weight", "0.bias"]
    params = {k: full[k] for k in names}
    sizes = [params[k].numel() for k in names]
    g = torch.Generator().manual_seed(3)
    basis = torch.randn(sum(sizes), 3, generator=g, dtype=torch.float64)

    def fwd(p):
        return functional_call(net, {**full, **p}, (x,))

    jac = jacrev(fwd)(params)
    j = torch.cat([jac[k].reshape(7, E, -1) for k in names], dim=2)
    h = core.observed_output_hessian(lk.Categorical(), net(x).detach(), y).dense()
    ref = basis.T @ torch.einsum("nep,nef,nfq->pq", j, h, j) @ basis

    ctx = FitContext(
        core.ModelFn.from_module(net), lk.Categorical(), Layout.by_module(params), Capabilities()
    )

    for strategy in ("forward-mode", "reverse-mode"):
        gram = ProjectedGram(basis, strategy)
        stat = gram.update(
            gram.init(params, ctx), params, lk.Batch(x, y), torch.Generator().manual_seed(0), ctx
        )

        torch.testing.assert_close(stat.gram, ref)


def test_features_refuse_dict_outputs():
    class Two(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(2, E).double()

        def forward(self, x):
            return {"a": self.lin(x)}

    net = Two()
    x, _ = data(3)
    fn = core.ModelFn.from_module(net)
    params = params_of(net)

    for strategy in ("forward-mode", "reverse-mode"):
        with pytest.raises(StructureError, match="one tensor"):
            features(fn, params, x, basis_for(net, 2), strategy, torch.float64)


def test_vjp_features_in_chunks_of_inputs(monkeypatch):
    """With a small chunk budget the vjp path takes the inputs a few at a time."""

    import laplacekit_gp._jacobian as jacobian_module

    net = model()
    x, _ = data(7)
    params = params_of(net)
    p = sum(v.numel() for v in params.values())
    basis = basis_for(net, 4)
    ref = dense_jacobian(net, x) @ basis

    calls = []
    real = jacobian_module.rows

    def counting(*args, **kwargs):
        calls.append(args[2].shape[0])
        return real(*args, **kwargs)

    monkeypatch.setattr(jacobian_module, "VJP_CHUNK_ENTRIES", 2 * E * p)
    monkeypatch.setattr(jacobian_module, "rows", counting)

    _, phi = features(
        core.ModelFn.from_module(net), params, x, basis, "reverse-mode", torch.float64
    )

    assert calls == [2, 2, 2, 1]
    torch.testing.assert_close(phi, ref)
