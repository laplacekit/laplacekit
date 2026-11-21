"""`lk.gp` kernels against their closed forms, written out here."""

import math

import pytest
import torch

import laplacekit as lk
from laplacekit.core import StructureError


def inputs(n: int, shape: tuple[int, ...], seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)

    return torch.randn(n, *shape, generator=g, dtype=torch.float64)


def pairwise(x: torch.Tensor, y: torch.Tensor, scale: torch.Tensor | float = 1.0):
    """Euclidean distances by explicit loops over the flattened, scaled rows."""

    xf = x.reshape(x.shape[0], -1) / scale
    yf = y.reshape(y.shape[0], -1) / scale
    out = torch.empty(xf.shape[0], yf.shape[0], dtype=torch.float64)

    for i in range(xf.shape[0]):
        for j in range(yf.shape[0]):
            out[i, j] = math.sqrt(float(((xf[i] - yf[j]) ** 2).sum()))

    return out


def check_gram(k, x):
    g = k(x, x)

    torch.testing.assert_close(g, g.T, rtol=0, atol=1e-12)
    assert float(torch.linalg.eigvalsh(g).min()) > -1e-10


def test_kernels_match_closed_forms():
    x = inputs(7, (2, 3), seed=0)
    y = inputs(5, (2, 3), seed=1)
    ell = torch.tensor(1.3, dtype=torch.float64)
    var = torch.tensor(0.7, dtype=torch.float64)
    period = torch.tensor(2.1, dtype=torch.float64)
    r = pairwise(x, y)
    s = r / 1.3

    expected = {
        "rbf": 0.7 * torch.exp(-(r**2) / (2 * 1.3**2)),
        "m12": 0.7 * torch.exp(-s),
        "m32": 0.7 * (1 + math.sqrt(3) * s) * torch.exp(-math.sqrt(3) * s),
        "m52": 0.7 * (1 + math.sqrt(5) * s + 5 * s**2 / 3) * torch.exp(-math.sqrt(5) * s),
        "per": 0.7 * torch.exp(-2 * torch.sin(math.pi * r / 2.1) ** 2 / 1.3**2),
    }
    kernels = {
        "rbf": lk.gp.RBF(ell, var),
        "m12": lk.gp.Matern("1/2", ell, var),
        "m32": lk.gp.Matern("3/2", ell, var),
        "m52": lk.gp.Matern("5/2", ell, var),
        "per": lk.gp.Periodic(period, ell, var),
    }

    for name, k in kernels.items():
        assert isinstance(k, lk.gp.Kernel)

        if name == "per":
            continue

        torch.testing.assert_close(k(x, y), expected[name], rtol=1e-12, atol=1e-14)
        check_gram(k, x)

    # The periodic kernel on one input entry, also as a 1-D input.
    x1, y1 = inputs(9, (1,), seed=11), inputs(4, (1,), seed=12)
    r = pairwise(x1, y1)
    want = 0.7 * torch.exp(-2 * torch.sin(math.pi * r / 2.1) ** 2 / 1.3**2)

    torch.testing.assert_close(kernels["per"](x1, y1), want, rtol=1e-12, atol=1e-14)
    torch.testing.assert_close(kernels["per"](x1.squeeze(1), y1.squeeze(1)), want)
    check_gram(kernels["per"], x1)


def test_periodic_nd_is_product():
    x = inputs(7, (2, 3), seed=13)
    y = inputs(5, (2, 3), seed=14)
    xf, yf = x.reshape(7, -1), y.reshape(5, -1)
    k = lk.gp.Periodic(
        torch.tensor(2.1, dtype=torch.float64),
        torch.tensor(1.3, dtype=torch.float64),
        torch.tensor(0.7, dtype=torch.float64),
    )

    # One unit-variance 1-D kernel per flattened entry, multiplied; the variance once.
    want = torch.ones(7, 5, dtype=torch.float64)

    for d in range(xf.shape[1]):
        r = (xf[:, d, None] - yf[None, :, d]).abs()
        want = want * torch.exp(-2 * torch.sin(math.pi * r / 2.1) ** 2 / 1.3**2)

    want = 0.7 * want

    torch.testing.assert_close(k(x, y), want, rtol=1e-12, atol=1e-14)


def test_periodic_nd_psd():
    x = inputs(50, (3,), seed=15)
    k = lk.gp.Periodic(
        torch.tensor(1.7, dtype=torch.float64),
        torch.tensor(0.9, dtype=torch.float64),
        torch.tensor(1.4, dtype=torch.float64),
    )
    g = k(x, x)

    torch.testing.assert_close(g, g.T, rtol=0, atol=1e-12)
    assert float(torch.linalg.eigvalsh(g).min()) >= -1e-10


def test_lengthscale_per_input_entry():
    x = inputs(6, (3,), seed=2)
    y = inputs(4, (3,), seed=3)
    ell = torch.tensor([0.5, 1.0, 2.0], dtype=torch.float64)
    r = pairwise(x, y, ell)

    torch.testing.assert_close(lk.gp.RBF(ell)(x, y), torch.exp(-(r**2) / 2), rtol=1e-12, atol=0)
    torch.testing.assert_close(
        lk.gp.Matern("3/2", ell)(x, y),
        (1 + math.sqrt(3) * r) * torch.exp(-math.sqrt(3) * r),
        rtol=1e-12,
        atol=0,
    )

    with pytest.raises(StructureError, match="lengthscale"):
        lk.gp.RBF(torch.ones(4, dtype=torch.float64))(x, y)


def test_variance_defaults_to_one():
    x = inputs(4, (2,), seed=4)
    k = lk.gp.RBF(torch.tensor(1.0, dtype=torch.float64))

    torch.testing.assert_close(k.variance, torch.tensor(1.0))
    torch.testing.assert_close(k(x, x).diagonal(), torch.ones(4, dtype=torch.float64))


def test_matern_half_is_exponential():
    x = inputs(6, (2,), seed=5)
    ell = torch.tensor(0.8, dtype=torch.float64)

    torch.testing.assert_close(
        lk.gp.Matern("1/2", ell)(x, x), torch.exp(-pairwise(x, x) / 0.8), rtol=1e-12, atol=0
    )


def test_matern_limits():
    x = torch.linspace(-2, 2, 15, dtype=torch.float64).unsqueeze(1)
    ell = torch.tensor(0.7, dtype=torch.float64)
    rbf = torch.exp(-(pairwise(x, x) ** 2) / (2 * 0.7**2))

    gaps = {
        nu: float((lk.gp.Matern(nu, ell)(x, x) - rbf).abs().max()) for nu in ("1/2", "3/2", "5/2")
    }

    assert gaps["5/2"] < gaps["3/2"] < gaps["1/2"]


def test_matern_refuses_other_nu():
    with pytest.raises(ValueError, match="nu"):
        lk.gp.Matern("2", torch.tensor(1.0))  # ty: ignore[invalid-argument-type]  # the refusal


def test_sum_product_operators():
    x = inputs(6, (1,), seed=6)
    y = inputs(3, (1,), seed=7)
    ell = torch.tensor(0.9, dtype=torch.float64)
    a = lk.gp.RBF(ell, torch.tensor(2.0, dtype=torch.float64))
    b = lk.gp.Periodic(torch.tensor(1.5, dtype=torch.float64), ell)
    c = lk.gp.Matern("5/2", ell)
    r = pairwise(x, y)
    ka = 2.0 * torch.exp(-(r**2) / (2 * 0.81))
    kb = torch.exp(-2 * torch.sin(math.pi * r / 1.5) ** 2 / 0.81)
    s = r / 0.9
    kc = (1 + math.sqrt(5) * s + 5 * s**2 / 3) * torch.exp(-math.sqrt(5) * s)

    assert isinstance(a + b, lk.gp.Sum)
    assert isinstance(a * b, lk.gp.Product)
    assert (a + b) == lk.gp.Sum(a, b)

    torch.testing.assert_close((a + b)(x, y), ka + kb, rtol=1e-12, atol=0)
    torch.testing.assert_close((a * b)(x, y), ka * kb, rtol=1e-12, atol=0)
    torch.testing.assert_close((a * b + c)(x, y), ka * kb + kc, rtol=1e-12, atol=0)
    torch.testing.assert_close((c * (a + b))(x, y), kc * (ka + kb), rtol=1e-12, atol=0)
    check_gram(a * b + c, x)


def test_kernel_leaves_are_tensors_and_differentiable():
    k = lk.gp.Matern("3/2", torch.tensor(0.9, dtype=torch.float64)) + lk.gp.RBF(
        torch.tensor(1.2, dtype=torch.float64)
    )
    leaves = lk.core.tree_leaves(k)

    assert len(leaves) == 4
    assert all(isinstance(t, torch.Tensor) for t in leaves)

    ell = torch.tensor(0.9, dtype=torch.float64, requires_grad=True)
    x = inputs(5, (2,), seed=8)
    (grad,) = torch.autograd.grad(lk.gp.Matern("1/2", ell)(x, x).sum(), ell)

    # d/dl sum exp(-r/l) = sum exp(-r/l) r / l^2, with zero at r = 0
    r = pairwise(x, x)
    expected = (torch.exp(-r / 0.9) * r / 0.81).sum()

    torch.testing.assert_close(grad, expected, rtol=1e-12, atol=0)


def test_kernel_refuses_mapping_inputs():
    k = lk.gp.RBF(torch.tensor(1.0, dtype=torch.float64))
    x = inputs(3, (2,), seed=9)

    with pytest.raises(StructureError, match="tensor"):
        k({"x": x}, x)  # ty: ignore[invalid-argument-type]  # the refusal

    with pytest.raises(StructureError, match="tensor"):
        k(x, torch.tensor(1.0))

    with pytest.raises(StructureError, match="entries"):
        k(x, inputs(3, (3,), seed=10))


def test_kernel_refuses_bad_hyperparameters():
    with pytest.raises(ValueError, match="positive"):
        lk.gp.RBF(torch.tensor(-1.0))

    with pytest.raises(ValueError, match="positive"):
        lk.gp.Periodic(torch.tensor(0.0), torch.tensor(1.0))

    with pytest.raises(StructureError, match="variance"):
        lk.gp.RBF(torch.tensor(1.0), torch.ones(2))
