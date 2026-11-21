"""Masked likelihoods: padded positions change neither the curvature nor the data fit."""

import pytest
import torch
from torch import Tensor, nn
from torch.func import hessian, jacrev, vmap

import laplacekit.core as core
from laplacekit.core import (
    Batch,
    Capabilities,
    CapabilityError,
    FitContext,
    Layout,
    MaskedOutputHessian,
    ModelFn,
    StructureError,
    fit_curvature,
)
from laplacekit.curvature import DenseGGN, DiagGGN, default
from laplacekit.likelihoods import Categorical

N, C, D = 6, 3, 3


class PerPosition(nn.Module):
    """An MLP applied to each position: x [N, T, D] -> logits [N, T, C]. The mask is data."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(D, 8), nn.Tanh(), nn.Linear(8, C))

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        del mask
        return self.net(x)


def _mask_fn(f, inputs):
    return inputs["mask"]


MASKED = Categorical(mask_fn=_mask_fn)


def _model() -> tuple[ModelFn, dict[str, Tensor]]:
    torch.manual_seed(0)
    module = PerPosition().double()
    params = {k: v.detach() for k, v in module.named_parameters()}

    return ModelFn.from_module(module), params


def _data(t: int, seed: int = 1) -> tuple[Tensor, Tensor, Tensor]:
    """Inputs, targets and a right-padding mask with lengths between 1 and t."""

    g = torch.Generator().manual_seed(seed)
    x = torch.randn(N, t, D, generator=g, dtype=torch.float64)
    y = torch.randint(0, C, (N, t), generator=g)
    lengths = torch.randint(1, t + 1, (N,), generator=g)
    mask = torch.arange(t).unsqueeze(0) < lengths.unsqueeze(1)

    return x, y, mask


def _batch(x: Tensor, y: Tensor, mask: Tensor) -> Batch:
    return Batch({"x": x, "mask": mask}, y)


def _run(method, batch: Batch, lik=MASKED):
    fn, params = _model()
    fit = fit_curvature(fn, lik, params, method, generator=torch.Generator().manual_seed(3))
    stat = fit.update(fit.init(), batch)

    return stat.curvature.sum, stat.data_fit


def _garbage_at_padding(x: Tensor, y: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
    g = torch.Generator().manual_seed(7)
    x2 = torch.where(mask.unsqueeze(-1), x, 5 * torch.randn(x.shape, generator=g, dtype=x.dtype))
    y2 = torch.where(mask, y, torch.randint(0, C, y.shape, generator=g))

    return x2, y2


METHODS = [
    DiagGGN(),
    DenseGGN(),
    DiagGGN(fisher="mc"),
    DiagGGN(observed=True),
    DenseGGN(observed=True),
]


@pytest.mark.parametrize("method", METHODS, ids=repr)
def test_padding_changes_nothing(method):
    x, y, mask = _data(5)
    x2, y2 = _garbage_at_padding(x, y, mask)

    assert not torch.equal(x, x2) and not torch.equal(y, y2)

    s1, ll1 = _run(method, _batch(x, y, mask))
    s2, ll2 = _run(method, _batch(x2, y2, mask))

    for b in s1:
        assert torch.equal(s1[b], s2[b]), b

    assert torch.equal(ll1, ll2)

    # Without the mask, the garbage shows.
    u1, _ = _run(method, _batch(x, y, mask), Categorical())
    u2, _ = _run(method, _batch(x2, y2, mask), Categorical())

    assert not all(torch.equal(u1[b], u2[b]) for b in u1)


@pytest.mark.parametrize("method", [DiagGGN(), DenseGGN()], ids=repr)
def test_padding_from_3_to_5_matches_unpadded(method):
    x, y, mask = _data(3)
    g = torch.Generator().manual_seed(9)
    x5 = torch.cat([x, torch.randn(N, 2, D, generator=g, dtype=x.dtype)], dim=1)
    y5 = torch.cat([y, torch.randint(0, C, (N, 2), generator=g)], dim=1)
    mask5 = torch.cat([mask, torch.zeros(N, 2, dtype=torch.bool)], dim=1)

    s3, ll3 = _run(method, _batch(x, y, mask))
    s5, ll5 = _run(method, _batch(x5, y5, mask5))

    for b in s3:
        torch.testing.assert_close(s3[b], s5[b], atol=1e-6, rtol=1e-6)

    torch.testing.assert_close(ll3, ll5, atol=1e-6, rtol=1e-6)


def _reference_ggn(fn: ModelFn, params, x: Tensor, mask: Tensor) -> tuple[Tensor, list[str]]:
    """sum over unmasked (n, t) of J_ntᵀ (diag p - p pᵀ) J_nt, flat over all params."""

    names = list(params)
    total = torch.zeros(())

    for n in range(x.shape[0]):
        for t in range(x.shape[1]):
            if not bool(mask[n, t]):
                continue

            xi = {"x": x[n, t].reshape(1, 1, D), "mask": mask[n, t].reshape(1, 1)}

            def f(p, xi=xi):
                return fn(p, xi).reshape(-1)

            jac = jacrev(f)(params)
            j = torch.cat([jac[k].reshape(C, -1) for k in names], dim=1)
            p = torch.softmax(f(params), -1)
            h = torch.diag(p) - torch.outer(p, p)
            total = total + j.T @ h @ j

    return total, names


def test_masked_ggn_matches_reference():
    fn, params = _model()
    x, y, mask = _data(4)
    ref, names = _reference_ggn(fn, params, x, mask)
    dense, _ = _run(DenseGGN(), _batch(x, y, mask))
    diag, _ = _run(DiagGGN(), _batch(x, y, mask))
    layout = Layout.by_module(params)
    off = 0

    for b in layout.blocks:
        size = layout.size(b)
        block = ref[off : off + size, off : off + size]
        torch.testing.assert_close(dense[b], block)
        torch.testing.assert_close(diag[b], block.diagonal())
        off += size


def test_masked_hessian_sqrt():
    torch.manual_seed(0)
    t = 4
    f = torch.randn(N, t, C, dtype=torch.float64)
    mask = torch.rand(N, t) < 0.6
    hess = core.output_hessian(
        MASKED, f, generator=torch.Generator().manual_seed(0), inputs={"mask": mask}
    )

    assert isinstance(hess, MaskedOutputHessian)

    # Independent reference: autograd Hessian of -log softmax, masked by hand.
    y = torch.randint(0, C, (N, t))

    def nll(fi, yi):
        return -torch.log_softmax(fi, -1).gather(-1, yi.unsqueeze(-1)).sum()

    e = t * C
    h = vmap(lambda fi, yi: hessian(nll)(fi, yi).reshape(e, e))(f, y)
    m = mask.unsqueeze(-1).expand(N, t, C).reshape(N, e).to(f.dtype)
    mhm = m.unsqueeze(2) * h * m.unsqueeze(1)

    basis = torch.eye(e, dtype=f.dtype).reshape(e, 1, t, C).expand(e, N, t, C)
    a = torch.stack([hess.sqrt_mv(basis[i]).reshape(N, e) for i in range(e)], dim=2)
    hv = torch.stack([hess.mv(basis[i]).reshape(N, e) for i in range(e)], dim=2)

    torch.testing.assert_close(a @ a.mT, mhm)
    torch.testing.assert_close(hv, mhm)
    torch.testing.assert_close(hess.dense(), mhm)
    torch.testing.assert_close(hess.diagonal().reshape(N, e), mhm.diagonal(dim1=1, dim2=2))


def test_masked_mc_and_observed_hessians():
    torch.manual_seed(0)
    f = torch.randn(N, 2, C, dtype=torch.float64)
    mask = torch.tensor([[True, False]] * N)
    inputs = {"mask": mask}
    y = torch.randint(0, C, (N, 2))
    e = 2 * C
    m = mask.unsqueeze(-1).expand(N, 2, C).reshape(N, e).to(f.dtype)

    class Plain:
        """Categorical without a closed form, so the MC Fisher is used."""

        def log_prob(self, f, y):
            return MASKED.log_prob(f, y)

        def distribution(self, f):
            return MASKED.distribution(f)

        def mask(self, f, inputs):
            return inputs["mask"]

    mc = core.output_hessian(
        Plain(), f, generator=torch.Generator().manual_seed(0), inputs=inputs, mc_samples=4
    )
    inner = core.output_hessian(
        Plain(), f, generator=torch.Generator().manual_seed(0), mc_samples=4
    ).dense()
    torch.testing.assert_close(mc.dense(), m.unsqueeze(2) * inner * m.unsqueeze(1))

    obs = core.observed_output_hessian(MASKED, f, y, inputs=inputs)
    inner = core.observed_output_hessian(MASKED, f, y).dense()
    torch.testing.assert_close(obs.dense(), m.unsqueeze(2) * inner * m.unsqueeze(1))


def test_log_prob_zeroes_masked_positions():
    torch.manual_seed(0)
    f = torch.randn(N, 4, C)
    y = torch.randint(0, C, (N, 4))
    mask = torch.rand(N, 4) < 0.5
    ref = (torch.log_softmax(f, -1).gather(-1, y.unsqueeze(-1)).squeeze(-1) * mask).sum(1)

    torch.testing.assert_close(core.log_prob(MASKED, f, y, inputs={"mask": mask}), ref)
    # Without inputs there is no mask.
    torch.testing.assert_close(core.log_prob(MASKED, f, y), core.log_prob(Categorical(), f, y))


def test_mask_none_means_every_position():
    lik = Categorical(mask_fn=lambda f, inputs: None)
    f = torch.randn(N, 4, C)
    y = torch.randint(0, C, (N, 4))
    hess = core.output_hessian(lik, f, generator=torch.Generator().manual_seed(0), inputs={})

    assert not isinstance(hess, MaskedOutputHessian)
    torch.testing.assert_close(
        core.log_prob(lik, f, y, inputs={}), core.log_prob(Categorical(), f, y)
    )


def test_mask_shape_mismatch_raises():
    f = torch.randn(N, 4, C)
    y = torch.randint(0, C, (N, 4))

    with pytest.raises(StructureError, match="mask"):
        core.log_prob(MASKED, f, y, inputs={"mask": torch.ones(N, 3, dtype=torch.bool)})

    with pytest.raises(StructureError, match="mask"):
        core.output_hessian(
            MASKED,
            f,
            generator=torch.Generator().manual_seed(0),
            inputs={"mask": torch.ones(N, 3).bool()},
        )

    # Classes on axis 1: the positions [N, T] are not a leading part of f's shape [N, C, T].
    lik = Categorical(class_dim=1, mask_fn=_mask_fn)
    ft = torch.randn(N, C, 4)

    with pytest.raises(StructureError, match="trailing"):
        core.output_hessian(
            lik,
            ft,
            generator=torch.Generator().manual_seed(0),
            inputs={"mask": torch.ones(N, 4).bool()},
        )

    with pytest.raises(StructureError, match="bool"):
        core.log_prob(MASKED, f, y, inputs={"mask": torch.ones(N, 4)})


def _kfac_setup():
    torch.manual_seed(0)
    module = nn.Sequential(nn.Linear(3, 8), nn.Tanh(), nn.Linear(8, C))
    fn = ModelFn.from_module(module)
    caps = core.probe(fn, Categorical(), Batch(torch.randn(4, 3), torch.randint(0, C, (4,))))
    params = {k: v.detach() for k, v in module.named_parameters()}

    return fn, params, caps


def test_kfac_refuses_mask():
    from laplacekit.curvlinops import KFAC

    fn, params, caps = _kfac_setup()

    assert KFAC.supports(Categorical())
    assert not KFAC.supports(MASKED)

    ctx = FitContext(fn, MASKED, Layout.by_module(params), caps)

    # The refusal says why: Curvlinops' input factor sums over every position.
    with pytest.raises(CapabilityError, match="mask_fn.*every position.*DenseGGN"):
        KFAC().init(params, ctx)


def test_default_skips_kfac_for_a_masked_likelihood():
    from laplacekit.curvlinops import KFAC

    params = {"fc.weight": torch.zeros(3, 2), "fc.bias": torch.zeros(3)}
    vjp_only = Capabilities(vjp=True, hooked_types={"fc": nn.Linear})
    big = Capabilities(
        vjp=True, vmap=True, hooked_types={"fc": nn.Linear}, event_size=core.MAX_DENSE_EVENT + 1
    )

    assert default(params, vjp_only, likelihood=Categorical()).by_block(["fc"])["fc"] == KFAC()
    assert default(params, vjp_only, likelihood=MASKED).by_block(["fc"])["fc"] == DiagGGN(
        fisher="mc"
    )
    assert default(params, big, likelihood=Categorical()).by_block(["fc"])["fc"] == KFAC(
        fisher="mc"
    )
    assert default(params, big, likelihood=MASKED).by_block(["fc"])["fc"] == DiagGGN(fisher="mc")
