from collections.abc import Callable, Mapping
from typing import Any, cast

import pytest
import torch
from torch import Tensor

from laplacekit.core import (
    BlockDiag,
    Dense,
    Diag,
    HasBlocks,
    Joint,
    KronEigenbasis,
    LowRankPlusDiag,
    Operator,
    StructureError,
    with_eigen,
)

DTYPES = [(torch.float32, 1e-5), (torch.float64, 1e-10)]
N, SPLIT = 5, 2


def spd(gen: torch.Generator, dtype: torch.dtype) -> Tensor:
    a = torch.randn(N, N, generator=gen, dtype=dtype)
    return a @ a.T + 0.1 * torch.eye(N, dtype=dtype)


class Case:
    """An operator kind, with the dense matrix it should equal."""

    def __init__(self, kind: str, dtype: torch.dtype, seed: int):
        self.kind, self.dtype = kind, dtype
        self.gen = torch.Generator().manual_seed(seed)

    def dense(self) -> Tensor:
        p = spd(self.gen, self.dtype)

        if self.kind == "Diag":
            return torch.diag(p.diagonal())

        if self.kind == "BlockDiag":
            m = torch.zeros(N, N, dtype=self.dtype)
            m[:SPLIT, :SPLIT] = torch.diag(p.diagonal()[:SPLIT])
            q = p[SPLIT:, SPLIT:]
            m[SPLIT:, SPLIT:] = q @ q.T + torch.eye(N - SPLIT, dtype=self.dtype)

            return m

        return p

    def build(self, m: Tensor):
        if self.kind == "Diag":
            return Diag(m.diagonal())

        if self.kind == "Dense":
            return Dense(m)

        return BlockDiag({"a": Diag(m[:SPLIT, :SPLIT].diagonal()), "b": Dense(m[SPLIT:, SPLIT:])})

    def vec(self, v: Tensor):
        if self.kind == "BlockDiag":
            return {"a": v[:SPLIT], "b": v[SPLIT:]}

        return v

    def flat(self, v: Tensor | Mapping[str, Tensor]) -> Tensor:
        if isinstance(v, Mapping):
            return torch.cat([v["a"], v["b"]])

        return v


@pytest.fixture(params=["Diag", "Dense", "BlockDiag"])
def kind(request: pytest.FixtureRequest) -> str:
    return request.param


@pytest.fixture(params=DTYPES, ids=["f32", "f64"])
def case(request: pytest.FixtureRequest, kind: str) -> tuple[Case, float]:
    dtype, tol = request.param
    return Case(kind, dtype, 0), tol


def close(a: Tensor, b: Tensor, tol: float) -> None:
    torch.testing.assert_close(a, b, rtol=tol, atol=tol)


def test_mv_diagonal_logdet_solve(case: tuple[Case, float]) -> None:
    c, tol = case
    p = c.dense()
    op = c.build(p)
    v = torch.randn(N, generator=c.gen, dtype=c.dtype)
    close(c.flat(op.mv(c.vec(v))), p @ v, tol)
    close(c.flat(op.diagonal()), p.diagonal(), tol)
    close(op.logdet(), torch.logdet(p), tol)
    close(c.flat(op.solve(c.vec(v))), torch.linalg.solve(p, v), tol)


def test_cov_sqrt_mv(case: tuple[Case, float]) -> None:
    c, tol = case
    p = c.dense()
    op = c.build(p)
    eye = torch.eye(N, dtype=c.dtype)
    cols = [c.flat(op.cov_sqrt_mv(c.vec(eye[:, i]))) for i in range(N)]
    lmat = torch.stack(cols, dim=1)
    close(lmat @ lmat.T, torch.linalg.inv(p), tol * 10)


def test_scaled_isotropic_diagonal(case: tuple[Case, float]) -> None:
    c, tol = case
    p = c.dense()
    op = c.build(p)
    eye = torch.eye(N, dtype=c.dtype)
    s = torch.tensor(2.5, dtype=c.dtype)
    d = torch.rand(N, generator=c.gen, dtype=c.dtype) + 0.5

    def dense_of(o) -> Tensor:
        return torch.stack([c.flat(o.mv(c.vec(eye[:, i]))) for i in range(N)], dim=1)

    close(dense_of(op.scaled(s)), s * p, tol)
    close(dense_of(op.add_isotropic(s)), p + s * eye, tol)
    close(dense_of(op.add_diagonal(c.vec(d))), p + torch.diag(d), tol)


def test_add_same(case: tuple[Case, float]) -> None:
    c, tol = case
    p1, p2 = c.dense(), c.dense()
    out = c.build(p1).add_same(c.build(p2))
    eye = torch.eye(N, dtype=c.dtype)
    got = torch.stack([c.flat(out.mv(c.vec(eye[:, i]))) for i in range(N)], dim=1)
    close(got, p1 + p2, tol)


def test_add_same_mismatch_names_both_types() -> None:
    with pytest.raises(StructureError, match="Dense") as e:
        Dense(torch.eye(2)).add_same(Diag(torch.ones(2)))

    assert "Diag" in str(e.value)

    with pytest.raises(StructureError, match="Diag") as e:
        Diag(torch.ones(2)).add_same(Dense(torch.eye(2)))

    assert "Dense" in str(e.value)


def test_blockdiag_add_per_block_and_key_mismatch() -> None:
    bd = BlockDiag({"a": Diag(torch.ones(2)), "b": Diag(torch.ones(3))})
    out = bd.add_per_block({"a": torch.full((2,), 2.0), "b": torch.full((3,), 3.0)})

    assert out.blocks["b"].diag.tolist() == [4.0, 4.0, 4.0]
    other = BlockDiag({"a": Diag(torch.ones(2)), "c": Diag(torch.ones(3))})

    with pytest.raises(StructureError, match="block names"):
        bd.add_same(other)


def _bd() -> BlockDiag[Diag]:
    return BlockDiag({"a": Diag(torch.ones(2)), "b": Diag(torch.ones(3))})


@pytest.mark.parametrize("method", ["mv", "solve", "cov_sqrt_mv", "add_diagonal", "add_per_block"])
@pytest.mark.parametrize(
    ("keys", "missing", "extra"),
    [(("a", "bb"), "'b'", "'bb'"), (("a", "b", "c"), "[]", "'c'"), (("a",), "'b'", "[]")],
)
def test_blockdiag_key_mismatch(
    method: str, keys: tuple[str, ...], missing: str, extra: str
) -> None:
    v = {k: torch.ones(2) for k in keys}

    with pytest.raises(StructureError) as e:
        getattr(_bd(), method)(v)

    msg = str(e.value)

    assert f"missing {'[' + missing + ']' if missing != '[]' else missing}" in msg
    assert f"extra {'[' + extra + ']' if extra != '[]' else extra}" in msg


# KronEigenbasis, against a dense matrix in layout order built independently.

KRON_SHAPES = [
    pytest.param((3, 4), False, id="linear"),
    pytest.param((3, 4), True, id="linear-bias"),
    pytest.param((3, 2, 2, 2), True, id="conv-bias"),
    pytest.param((3, 2, 2, 2), False, id="conv"),
]


def _orthogonal(gen: torch.Generator, n: int, dtype: torch.dtype) -> Tensor:
    return torch.linalg.qr(torch.randn(n, n, generator=gen, dtype=dtype))[0]


def _kron_case(
    shape: tuple[int, ...], has_bias: bool, dtype: torch.dtype
) -> tuple[KronEigenbasis, Tensor]:
    """The operator and its dense matrix in layout order (weight row-major, then bias)."""

    gen = torch.Generator().manual_seed(0)
    out = shape[0]
    n_in = int(torch.tensor(shape[1:]).prod()) + int(has_bias)
    q_a, q_g = _orthogonal(gen, n_in, dtype), _orthogonal(gen, out, dtype)
    evals = torch.rand(out, n_in, generator=gen, dtype=dtype) + 0.5
    op = KronEigenbasis(q_a, q_g, evals, has_bias=has_bias, weight_shape=shape)

    # Dense in the folded order: index (i, j) of [W | b] -> i * n_in + j.
    q = torch.kron(q_g, q_a)
    folded = q @ torch.diag(evals.reshape(-1)) @ q.T

    # Permutation from layout order to folded order.
    n_w = n_in - int(has_bias)
    layout_to_folded = [i * n_in + j for i in range(out) for j in range(n_w)]
    layout_to_folded += [i * n_in + n_w for i in range(out)] if has_bias else []
    idx = torch.tensor(layout_to_folded)

    return op, folded[idx][:, idx]


@pytest.fixture(params=DTYPES, ids=["f32", "f64"])
def kdtype(request: pytest.FixtureRequest) -> tuple[torch.dtype, float]:
    return request.param


@pytest.mark.parametrize(("shape", "has_bias"), KRON_SHAPES)
def test_kron_eigenbasis_matches_dense(
    shape: tuple[int, ...], has_bias: bool, kdtype: tuple[torch.dtype, float]
) -> None:
    dtype, tol = kdtype
    op, dense = _kron_case(shape, has_bias, dtype)
    v = torch.randn(dense.shape[0], dtype=dtype, generator=torch.Generator().manual_seed(1))
    tol = tol * 10

    close(op.mv(v), dense @ v, tol)
    close(op.diagonal(), dense.diagonal(), tol)
    close(op.logdet(), torch.logdet(dense), tol)
    close(op.solve(v), torch.linalg.solve(dense, v), tol)
    close(op.scaled(torch.tensor(2.5, dtype=dtype)).mv(v), 2.5 * dense @ v, tol)

    eye = torch.eye(dense.shape[0], dtype=dtype)
    close(op.add_isotropic(torch.tensor(0.7, dtype=dtype)).mv(v), (dense + 0.7 * eye) @ v, tol)

    with pytest.raises(StructureError, match="no Kronecker form for a diagonal prior"):
        op.add_diagonal(torch.full((dense.shape[0],), 0.3, dtype=dtype))


@pytest.mark.parametrize(("shape", "has_bias"), KRON_SHAPES)
def test_kron_eigenbasis_cov_sqrt(
    shape: tuple[int, ...], has_bias: bool, kdtype: tuple[torch.dtype, float]
) -> None:
    dtype, tol = kdtype
    op, dense = _kron_case(shape, has_bias, dtype)
    n = dense.shape[0]
    cols = torch.stack([op.cov_sqrt_mv(e) for e in torch.eye(n, dtype=dtype)], dim=1)

    close(cols @ cols.T, torch.linalg.inv(dense), tol * 10)


def test_kron_eigenbasis_refuses_any_diagonal() -> None:
    """KFAC has no Kronecker form for a diagonal prior, even a constant one."""

    op, _ = _kron_case((3, 4), True, torch.float64)
    msg = r"Kronecker-factored curvature \(KFAC or EKFAC\), has no Kronecker form for a diagonal"

    for d in (torch.arange(15, dtype=torch.float64), torch.full((15,), 0.3, dtype=torch.float64)):
        with pytest.raises(StructureError, match=msg):
            op.add_diagonal(d)


def test_kfac_plus_kfac_points_to_kronsum() -> None:
    op, _ = _kron_case((3, 4), False, torch.float64)

    with pytest.raises(StructureError, match="KronEigenbasis \\+ KronEigenbasis.*Diag") as e:
        op.add_same(op)

    assert str(e.value).endswith(
        "; for continual learning with KFAC, opt into lk.priors.KronSum.of(post), which sums "
        "the Kronecker factors (approximate)"
    )


def test_kron_eigenbasis_add_same_other_type_names_the_pair() -> None:
    op, _ = _kron_case((3, 4), False, torch.float64)

    with pytest.raises(StructureError, match="cannot add Diag to KronEigenbasis"):
        op.add_same(Diag(torch.ones(12, dtype=torch.float64)))


# --- batched vectors and the inverse diagonal ----------------------------------------
# Vectors carry the operator's axis last; leading axes are a batch.


def _rows(c: Case, m: Tensor):
    """[n, P] rows as the operator's vector type."""

    if c.kind == "BlockDiag":
        return {"a": m[:, :SPLIT], "b": m[:, SPLIT:]}

    return m


def _flat_rows(v: Tensor | Mapping[str, Tensor]) -> Tensor:
    if isinstance(v, Mapping):
        return torch.cat([v["a"], v["b"]], dim=-1)

    return v


def test_batched_vectors(case: tuple[Case, float]) -> None:
    c, tol = case
    p = c.dense()
    op = c.build(p)
    m = torch.randn(3, N, generator=c.gen, dtype=c.dtype)
    inv = torch.linalg.inv(p)

    close(_flat_rows(op.mv(_rows(c, m))), m @ p.T, tol)
    close(_flat_rows(op.solve(_rows(c, m))), m @ inv.T, tol * 10)

    roots = torch.stack([c.flat(op.cov_sqrt_mv(c.vec(row))) for row in m])
    close(_flat_rows(op.cov_sqrt_mv(_rows(c, m))), roots, tol)


def test_inverse_diagonal(case: tuple[Case, float]) -> None:
    c, tol = case
    p = c.dense()
    op = c.build(p)

    close(c.flat(op.inverse_diagonal()), torch.linalg.inv(p).diagonal(), tol * 10)


@pytest.mark.parametrize(("shape", "has_bias"), KRON_SHAPES)
def test_kron_eigenbasis_batched_vectors(
    shape: tuple[int, ...], has_bias: bool, kdtype: tuple[torch.dtype, float]
) -> None:
    dtype, tol = kdtype
    op, dense = _kron_case(shape, has_bias, dtype)
    m = torch.randn(4, dense.shape[0], dtype=dtype, generator=torch.Generator().manual_seed(2))
    tol = tol * 10

    close(op.mv(m), m @ dense.T, tol)
    close(op.solve(m), m @ torch.linalg.inv(dense).T, tol)

    roots = op.cov_sqrt_mv(m)

    assert roots.shape == m.shape
    close(roots, torch.stack([op.cov_sqrt_mv(row) for row in m]), tol)

    # Two batch axes.
    close(op.mv(m.reshape(2, 2, -1)), (m @ dense.T).reshape(2, 2, -1), tol)


@pytest.mark.parametrize(("shape", "has_bias"), KRON_SHAPES)
def test_kron_eigenbasis_inverse_diagonal(
    shape: tuple[int, ...], has_bias: bool, kdtype: tuple[torch.dtype, float]
) -> None:
    dtype, tol = kdtype
    op, dense = _kron_case(shape, has_bias, dtype)

    close(op.inverse_diagonal(), torch.linalg.inv(dense).diagonal(), tol * 10)


@pytest.mark.parametrize(("shape", "has_bias"), KRON_SHAPES)
def test_kron_eigenbasis_scaled_and_shifted_solve_logdet(
    shape: tuple[int, ...], has_bias: bool, kdtype: tuple[torch.dtype, float]
) -> None:
    dtype, tol = kdtype
    op, dense = _kron_case(shape, has_bias, dtype)
    s, prec = torch.tensor(2.5, dtype=dtype), torch.tensor(0.7, dtype=dtype)
    moved = op.scaled(s).add_isotropic(prec)
    want = s * dense + prec * torch.eye(dense.shape[0], dtype=dtype)
    v = torch.randn(dense.shape[0], dtype=dtype, generator=torch.Generator().manual_seed(3))
    tol = tol * 10

    close(moved.solve(v), torch.linalg.solve(want, v), tol)
    close(moved.logdet(), torch.logdet(want), tol)


def test_kron_eigenbasis_checks_shapes() -> None:
    op, _ = _kron_case((3, 4), True, torch.float64)

    with pytest.raises(StructureError, match="evals"):
        KronEigenbasis(op.evecs_a, op.evecs_g, op.evals[:, :-1], True, (3, 4))

    with pytest.raises(StructureError, match="evecs_a"):
        KronEigenbasis(op.evecs_a, op.evecs_g, op.evals, False, (3, 4))

    with pytest.raises(StructureError, match="evecs_g"):
        KronEigenbasis(op.evecs_a, op.evecs_g[:2, :2], op.evals, True, (3, 4))


def _kron_with_factors(count: float) -> KronEigenbasis:
    gen = torch.Generator().manual_seed(2)
    q_a, q_g = _orthogonal(gen, 5, torch.float64), _orthogonal(gen, 3, torch.float64)
    la = torch.rand(5, generator=gen, dtype=torch.float64) + 0.5
    lg = torch.rand(3, generator=gen, dtype=torch.float64) + 0.5
    c = torch.tensor(count, dtype=torch.float64)

    return KronEigenbasis(q_a, q_g, c * torch.outer(lg, la), True, (3, 4), (la, lg, c))


def test_kron_eigenbasis_factors_follow_the_algebra() -> None:
    op = _kron_with_factors(2.0)
    la, lg, count = op.factors

    assert KronEigenbasis(op.evecs_a, op.evecs_g, op.evals, True, (3, 4)).factors == ()

    s = op.scaled(torch.tensor(3.0, dtype=torch.float64))
    sa, sg, sc = s.factors

    assert torch.equal(sa, la) and torch.equal(sg, lg)
    assert float(sc) == 6.0
    torch.testing.assert_close(s.evals, sc * torch.outer(sg, sa))

    assert op.add_isotropic(torch.tensor(0.5, dtype=torch.float64)).factors == ()


def test_kron_eigenbasis_checks_factor_shapes() -> None:
    op = _kron_with_factors(1.0)
    la, lg, count = op.factors
    args = (op.evecs_a, op.evecs_g, op.evals, True, (3, 4))

    with pytest.raises(StructureError, match="evals_a"):
        KronEigenbasis(*args, (la[:-1], lg, count))

    with pytest.raises(StructureError, match="evals_g"):
        KronEigenbasis(*args, (la, lg[:-1], count))

    with pytest.raises(StructureError, match="count"):
        KronEigenbasis(*args, (la, lg, count.reshape(1)))

    with pytest.raises(StructureError, match="factors"):
        KronEigenbasis(*args, (la, lg))


# --- Joint: one operator over all blocks ------------------------------------------------

JOINT_SIZES = (("a", 2), ("b", 3), ("c", 4))


def _joint_case() -> tuple[Tensor, Joint[Dense]]:
    gen = torch.Generator().manual_seed(7)
    n = sum(s for _, s in JOINT_SIZES)
    a = torch.randn(n, n, generator=gen, dtype=torch.float64)
    m = a @ a.T + 0.5 * torch.eye(n, dtype=torch.float64)

    return m, Joint(Dense(m), JOINT_SIZES)


def _split(v: Tensor) -> dict[str, Tensor]:
    parts = torch.split(v, [s for _, s in JOINT_SIZES], dim=-1)

    return {k: p for (k, _), p in zip(JOINT_SIZES, parts, strict=True)}


def _cat(v: Mapping[str, Tensor]) -> Tensor:
    return torch.cat([v[k] for k, _ in JOINT_SIZES], dim=-1)


def test_joint_matches_dense() -> None:
    m, op = _joint_case()
    n = m.shape[0]
    v = torch.randn(n, generator=torch.Generator().manual_seed(1), dtype=torch.float64)
    inv = torch.linalg.inv(m)
    tol = 1e-10

    close(_cat(op.mv(_split(v))), m @ v, tol)
    close(_cat(op.solve(_split(v))), inv @ v, tol)
    close(_cat(op.diagonal()), m.diagonal(), tol)
    close(_cat(op.inverse_diagonal()), inv.diagonal(), tol)
    close(op.logdet(), torch.linalg.slogdet(m)[1], tol)

    # cov_sqrt_mv is L v with L L^T = inverse: build L column by column.
    eye = torch.eye(n, dtype=torch.float64)
    root = torch.stack([_cat(op.cov_sqrt_mv(_split(eye[i]))) for i in range(n)], dim=1)
    close(root @ root.T, inv, tol)

    prec = torch.tensor(0.7, dtype=torch.float64)
    close(_cat(op.add_isotropic(prec).mv(_split(v))), (m + prec * eye) @ v, tol)
    close(_cat(op.scaled(prec).mv(_split(v))), prec * m @ v, tol)

    d = torch.rand(n, generator=torch.Generator().manual_seed(2), dtype=torch.float64)
    close(_cat(op.add_diagonal(_split(d)).mv(_split(v))), (m + torch.diag(d)) @ v, tol)

    per_block = {
        "a": torch.tensor(0.3, dtype=torch.float64),
        "b": torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64),
        "c": torch.tensor(5.0, dtype=torch.float64),
    }
    f64 = torch.float64
    want = torch.cat(
        [torch.full((2,), 0.3, dtype=f64), per_block["b"], torch.full((4,), 5.0, dtype=f64)]
    )
    added = op.add_per_block(per_block)

    assert isinstance(added, Joint) and isinstance(added.op, Dense)
    close(_cat(added.mv(_split(v))), (m + torch.diag(want)) @ v, tol)
    assert op.block_sizes() == {"a": 2, "b": 3, "c": 4}


def test_joint_is_blocked_and_blockdiag_block_sizes() -> None:
    _, op = _joint_case()
    bd = BlockDiag({"a": Diag(torch.ones(2)), "b": Dense(torch.eye(3))})

    assert isinstance(op, HasBlocks) and isinstance(bd, HasBlocks)
    assert not isinstance(Dense(torch.eye(3)), HasBlocks)
    assert bd.block_sizes() == {"a": 2, "b": 3}


def test_joint_add_same_requires_same_sizes() -> None:
    m, op = _joint_case()
    v = torch.randn(m.shape[0], generator=torch.Generator().manual_seed(3), dtype=torch.float64)

    close(_cat(op.add_same(op).mv(_split(v))), 2 * m @ v, 1e-10)

    other = Joint(Dense(m), (("a", 2), ("b", 4), ("c", 3)))

    with pytest.raises(StructureError, match=r"\('b', 3\).*\('b', 4\)"):
        op.add_same(other)

    with pytest.raises(StructureError, match="cannot add Dense to Joint"):
        op.add_same(Dense(m))  # ty: ignore[invalid-argument-type]  # deliberate misuse


def test_joint_batched_vectors() -> None:
    m, op = _joint_case()
    rows = torch.randn(
        4, m.shape[0], generator=torch.Generator().manual_seed(4), dtype=torch.float64
    )
    inv = torch.linalg.inv(m)

    out = op.mv(_split(rows))
    assert out["b"].shape == (4, 3)
    close(_cat(out), rows @ m.T, 1e-10)
    close(_cat(op.solve(_split(rows))), rows @ inv.T, 1e-10)

    root = _cat(op.cov_sqrt_mv(_split(rows)))
    assert root.shape == (4, m.shape[0])


def test_joint_checks_keys_and_sizes() -> None:
    m, op = _joint_case()
    v = _split(torch.ones(m.shape[0], dtype=torch.float64))

    with pytest.raises(StructureError, match=r"missing \['c'\]"):
        op.mv({"a": v["a"], "b": v["b"]})

    with pytest.raises(StructureError, match=r"'b' has shape \(2,\), expected 3 entries"):
        op.mv({**v, "b": v["b"][:2]})

    with pytest.raises(StructureError, match="mapping keyed by block"):
        op.add_diagonal(torch.ones(m.shape[0]))  # ty: ignore[invalid-argument-type]  # misuse


# --- LowRankPlusDiag: diag(d) + U diag(s) U^T ---------------------------------------------

LR_N, LR_R = 12, 3
F64 = torch.float64


def _low_rank_parts(seed: int = 11) -> tuple[Tensor, Tensor, Tensor]:
    gen = torch.Generator().manual_seed(seed)
    u = torch.randn(LR_N, LR_R, generator=gen, dtype=F64)
    s = torch.rand(LR_R, generator=gen, dtype=F64) + 0.5
    d = torch.rand(LR_N, generator=gen, dtype=F64) + 0.2

    return u, s, d


def _low_rank_dense(u: Tensor, s: Tensor, d: Tensor) -> Tensor:
    """P built explicitly, as an independent reference."""

    return torch.diag(d) + u @ torch.diag(s) @ u.T


def _root(apply: Callable[[Tensor], Tensor], n: int, dtype: torch.dtype = F64) -> Tensor:
    """The matrix of a linear map, column by column on the identity."""

    eye = torch.eye(n, dtype=dtype)

    return torch.stack([apply(eye[i]) for i in range(n)], dim=1)


def _check_low_rank_against(op: LowRankPlusDiag, p: Tensor) -> None:
    n = p.shape[0]
    gen = torch.Generator().manual_seed(5)
    v = torch.randn(n, generator=gen, dtype=F64)
    rows = torch.randn(4, n, generator=gen, dtype=F64)
    inv = torch.linalg.inv(p)
    tol = 1e-10

    close(op.mv(v), p @ v, tol)
    close(op.mv(rows), rows @ p.T, tol)
    close(op.diagonal(), p.diagonal(), tol)
    close(op.inverse_diagonal(), inv.diagonal(), tol)
    close(op.logdet(), torch.linalg.slogdet(p)[1], tol)
    close(op.solve(v), inv @ v, tol)
    close(op.solve(rows), rows @ inv.T, tol)

    root = _root(op.cov_sqrt_mv, n)
    close(root @ root.T, inv, tol)
    close(op.cov_sqrt_mv(rows), rows @ root.T, tol)


def test_low_rank_matches_dense() -> None:
    u, s, d = _low_rank_parts()
    p = _low_rank_dense(u, s, d)
    op = LowRankPlusDiag(u, s, d)
    eye = torch.eye(LR_N, dtype=F64)

    _check_low_rank_against(op, p)

    k = torch.tensor(0.7, dtype=F64)
    _check_low_rank_against(op.scaled(k), k * p)
    _check_low_rank_against(op.add_isotropic(k), p + k * eye)

    d2 = torch.rand(LR_N, generator=torch.Generator().manual_seed(3), dtype=F64)
    _check_low_rank_against(op.add_diagonal(d2), p + torch.diag(d2))

    # A bare curvature (zero diagonal) still has products and a diagonal.
    bare = LowRankPlusDiag(u, s, torch.zeros(LR_N, dtype=F64))
    v = torch.randn(LR_N, generator=torch.Generator().manual_seed(4), dtype=F64)
    close(bare.mv(v), u @ torch.diag(s) @ u.T @ v, 1e-10)
    close(bare.diagonal(), (u @ torch.diag(s) @ u.T).diagonal(), 1e-10)


def test_low_rank_cov_sqrt_squares_to_inverse() -> None:
    u, s, d = _low_rank_parts(seed=21)
    op = LowRankPlusDiag(u, s, d)
    root = _root(op.cov_sqrt_mv, LR_N)

    torch.testing.assert_close(
        root @ root.T, torch.linalg.inv(_low_rank_dense(u, s, d)), rtol=0, atol=1e-10
    )


def test_low_rank_zero_evals_and_rank_deficient_factor() -> None:
    u, s, d = _low_rank_parts(seed=31)
    u = torch.cat([u[:, :2], u[:, 1:2]], dim=1)  # two equal columns
    s = torch.tensor([0.0, 1.5, 0.8], dtype=F64)
    op = LowRankPlusDiag(u, s, d)
    p = _low_rank_dense(u, s, d)

    assert all(bool(torch.isfinite(t).all()) for t in [op.logdet(), op.inverse_diagonal()])
    assert bool(torch.isfinite(_root(op.cov_sqrt_mv, LR_N)).all())
    _check_low_rank_against(op, p)


def test_low_rank_scaled_zero_keeps_prior_logdet() -> None:
    u, s, d = _low_rank_parts()
    lam = torch.tensor(2.5, dtype=F64)
    prior_only = LowRankPlusDiag(u, s, d).scaled(torch.tensor(0.0, dtype=F64)).add_isotropic(lam)

    close(prior_only.logdet(), LR_N * lam.log(), 1e-12)


def test_low_rank_add_same_is_exact() -> None:
    u1, s1, d1 = _low_rank_parts(seed=41)
    u2, s2, d2 = _low_rank_parts(seed=42)
    total = LowRankPlusDiag(u1, s1, d1).add_same(LowRankPlusDiag(u2[:, :2], s2[:2], d2))

    assert total.factor.shape == (LR_N, LR_R + 2) and total.evals.shape == (LR_R + 2,)
    _check_low_rank_against(
        total, _low_rank_dense(u1, s1, d1) + _low_rank_dense(u2[:, :2], s2[:2], d2)
    )

    with pytest.raises(StructureError, match="cannot add Diag to LowRankPlusDiag"):
        LowRankPlusDiag(u1, s1, d1).add_same(Diag(d1))


def _top_r(m: Tensor, r: int) -> Tensor:
    """The best rank-`r` approximation of a symmetric PSD `m`, from its dense `eigh`."""

    e, v = torch.linalg.eigh(m)

    return (v[:, -r:] * e[-r:]) @ v[:, -r:].T


def test_truncated_matches_dense_top_r() -> None:
    gen = torch.Generator().manual_seed(61)
    u = torch.randn(LR_N, 5, generator=gen, dtype=F64)
    s = torch.rand(5, generator=gen, dtype=F64) + 0.5
    d = torch.rand(LR_N, generator=gen, dtype=F64) + 0.2
    low = u @ torch.diag(s) @ u.T
    op = LowRankPlusDiag(u, s, d)

    for r in (1, 2, 4):
        cut = op.truncated(r)

        assert cut.factor.shape == (LR_N, r) and cut.evals.shape == (r,)
        assert torch.equal(cut.diag, d)
        close(cut.factor.T @ cut.factor, torch.eye(r, dtype=F64), 1e-10)
        _check_low_rank_against(cut, torch.diag(d) + _top_r(low, r))

    # At or above the width it is exact.
    for r in (5, 9):
        _check_low_rank_against(op.truncated(r), torch.diag(d) + low)

    # Zero eigenvalues add nothing; the top `r` of the rest are kept.
    zeroed = LowRankPlusDiag(u, torch.cat([s[:3], torch.zeros(2, dtype=F64)]), d)
    low_3 = u[:, :3] @ torch.diag(s[:3]) @ u[:, :3].T
    _check_low_rank_against(zeroed.truncated(2), torch.diag(d) + _top_r(low_3, 2))
    _check_low_rank_against(zeroed.truncated(4), torch.diag(d) + low_3)

    # All zero: nothing is left but the diagonal.
    empty = LowRankPlusDiag(u, torch.zeros(5, dtype=F64), d).truncated(2)
    assert empty.factor.shape == (LR_N, 0)
    _check_low_rank_against(empty, torch.diag(d))

    with pytest.raises(ValueError, match=r"rank >= 1.*got 0"):
        op.truncated(0)


def test_truncated_more_columns_than_rows() -> None:
    """A factor wider than it is tall has at most `n` nonzero eigenvalues."""

    gen = torch.Generator().manual_seed(62)
    u = torch.randn(4, 7, generator=gen, dtype=F64)
    s = torch.rand(7, generator=gen, dtype=F64) + 0.5
    d = torch.rand(4, generator=gen, dtype=F64) + 0.2
    low = u @ torch.diag(s) @ u.T

    _check_low_rank_against(LowRankPlusDiag(u, s, d).truncated(2), torch.diag(d) + _top_r(low, 2))
    _check_low_rank_against(LowRankPlusDiag(u, s, d).truncated(6), torch.diag(d) + low)


def test_low_rank_shape_mismatch_raises() -> None:
    u, s, d = _low_rank_parts()

    with pytest.raises(StructureError, match=r"factor .*2-D"):
        LowRankPlusDiag(u[:, 0], s, d)

    with pytest.raises(StructureError, match=r"evals .*\(2,\).*expected \(3,\)"):
        LowRankPlusDiag(u, s[:2], d)

    with pytest.raises(StructureError, match=r"diag .*\(11,\).*expected \(12,\)"):
        LowRankPlusDiag(u, s, d[:11])

    with pytest.raises(StructureError, match=r"12 .* 11"):
        LowRankPlusDiag(u, s, d).add_same(LowRankPlusDiag(u[:11], s, d[:11]))


def test_low_rank_in_joint() -> None:
    u, s, _ = _low_rank_parts(seed=51)
    sizes = (("a", 2), ("b", 4), ("c", 6))
    op = Joint(LowRankPlusDiag(u, s, torch.zeros(LR_N, dtype=F64)), sizes)
    per_block = {
        "a": torch.tensor(0.3, dtype=F64),
        "b": torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=F64),
        "c": torch.tensor(5.0, dtype=F64),
    }
    full = torch.cat(
        [torch.full((2,), 0.3, dtype=F64), per_block["b"], torch.full((6,), 5.0, dtype=F64)]
    )
    added = op.add_per_block(per_block)
    p = _low_rank_dense(u, s, full)
    inv = torch.linalg.inv(p)

    def split(v: Tensor) -> dict[str, Tensor]:
        parts = torch.split(v, [n for _, n in sizes], dim=-1)
        return {k: x for (k, _), x in zip(sizes, parts, strict=True)}

    def cat(v: Mapping[str, Tensor]) -> Tensor:
        return torch.cat([v[k] for k, _ in sizes], dim=-1)

    v = torch.randn(LR_N, generator=torch.Generator().manual_seed(6), dtype=F64)
    tol = 1e-10

    assert isinstance(added, Joint) and isinstance(added.op, LowRankPlusDiag)
    close(cat(added.mv(split(v))), p @ v, tol)
    close(cat(added.solve(split(v))), inv @ v, tol)
    close(cat(added.inverse_diagonal()), inv.diagonal(), tol)
    close(added.logdet(), torch.linalg.slogdet(p)[1], tol)

    root = _root(lambda x: cat(added.cov_sqrt_mv(split(x))), LR_N)
    close(root @ root.T, inv, tol)


def _as_f64(op: LowRankPlusDiag) -> Tensor:
    """The dense P of a (float32) operator, in float64 from the same stored entries."""

    u, s, d = op.factor.double(), op.evals.double(), op.diag.double()

    return _low_rank_dense(u, s, d)


def test_low_rank_float32_full_rank_matches_dense() -> None:
    u, s, d = _low_rank_parts()
    op = LowRankPlusDiag(u.float(), s.float(), d.float())
    p = _as_f64(op)
    inv = torch.linalg.inv(p)
    v = torch.randn(LR_N, generator=torch.Generator().manual_seed(8), dtype=F64)
    root = _root(op.cov_sqrt_mv, LR_N, torch.float32).double()
    tol = 1e-4  # a well-conditioned P (cond ~ 1e2) at float32 eps ~ 1e-7

    close(op.mv(v.float()).double(), p @ v, tol)
    close(op.solve(v.float()).double(), inv @ v, tol)
    close(op.inverse_diagonal().double(), inv.diagonal(), tol)
    close(op.logdet().double(), torch.linalg.slogdet(p)[1], tol)
    close(root @ root.T, inv, tol)


@pytest.mark.parametrize("jitter", [0.0, 1e-4], ids=["duplicate", "jitter-1e-4"])
def test_low_rank_float32_nearly_dependent_factor(jitter: float) -> None:
    """Continual learning: `add_same` of two posteriors that share directions, with a large
    curvature over a weak prior (s / d up to 1e9)."""

    n, r = 60, 4
    gen = torch.Generator().manual_seed(9)
    u, _ = torch.linalg.qr(torch.randn(n, r, generator=gen, dtype=F64))
    u2 = u + jitter * torch.randn(n, r, generator=gen, dtype=F64)
    d = torch.full((n,), 1e-3)
    first = LowRankPlusDiag(u.float(), torch.logspace(6, 3, r), d)
    op = first.add_same(LowRankPlusDiag(u2.float(), torch.logspace(5, 3, r), torch.zeros(n)))

    p = _as_f64(op)
    inv = torch.linalg.inv(p)
    v = torch.randn(n, generator=gen, dtype=F64)

    # B = D^-1/2 U S^1/2 has norm ~ 3e4. A backward-stable float32 factorization of B
    # (no Gram matrix B^T B, which squares the condition) moves each eigenvalue of
    # I + B B^T by about eps * |B| * sqrt(lam) relative, ~ 1e-4 here. A Gram matrix
    # moves them by eps * |B|^2 ~ 1e2, which is no accuracy at all (or a failed
    # Cholesky). The tolerances leave a factor ~ 10 over the stable floor.
    root = _root(op.cov_sqrt_mv, n, torch.float32).double()
    cov = (root @ root.T).diagonal()
    torch.testing.assert_close(cov, inv.diagonal(), rtol=1e-3, atol=0)
    torch.testing.assert_close(op.inverse_diagonal().double(), inv.diagonal(), rtol=1e-3, atol=0)

    x = op.solve(v.float()).double()
    assert float((x - inv @ v).norm() / (inv @ v).norm()) < 1e-3

    assert abs(float(op.logdet().double() - torch.linalg.slogdet(p)[1])) < 1e-3  # nats


def test_low_rank_diag_gradient_with_zero_evals() -> None:
    """Tuning a prior differentiates through `diag`; a zero in `evals` must not make
    that gradient NaN."""

    u, _, d = _low_rank_parts(seed=61)
    s = torch.tensor([0.0, 1.5, 0.8], dtype=F64)
    v = torch.randn(LR_N, generator=torch.Generator().manual_seed(7), dtype=F64)

    def loss(op_diag: Tensor, dense: bool) -> Tensor:
        if dense:
            p = _low_rank_dense(u, s, op_diag)
            return torch.linalg.slogdet(p)[1] + v @ torch.linalg.solve(p, v)

        op = LowRankPlusDiag(u, s, op_diag)
        return op.logdet() + v @ op.solve(v)

    got = torch.func.grad(lambda x: loss(x, False))(d)
    want = torch.func.grad(lambda x: loss(x, True))(d)

    close(got, want, 1e-10)


def _low_rank_losses(
    u: Tensor, s: Tensor, v: Tensor, w: Tensor
) -> dict[str, Callable[[Tensor, bool], Tensor]]:
    """Scalar functions of `diag`: through the operator, or through a dense inverse.

    The `cov_sqrt_mv` one depends on the root only through `L L^T`, so any root of
    `P^-1` gives the same value and gradient.
    """

    n = u.shape[0]

    def inverse(op_diag: Tensor) -> Tensor:
        return torch.linalg.inv(_low_rank_dense(u, s, op_diag))

    def logdet(op_diag: Tensor, dense: bool) -> Tensor:
        if dense:
            return torch.linalg.slogdet(_low_rank_dense(u, s, op_diag))[1]
        return LowRankPlusDiag(u, s, op_diag).logdet()

    def solve(op_diag: Tensor, dense: bool) -> Tensor:
        if dense:
            return v @ inverse(op_diag) @ w
        return v @ LowRankPlusDiag(u, s, op_diag).solve(w)

    def inverse_diagonal(op_diag: Tensor, dense: bool) -> Tensor:
        if dense:
            return w @ inverse(op_diag).diagonal()
        return w @ LowRankPlusDiag(u, s, op_diag).inverse_diagonal()

    def cov_sqrt_mv(op_diag: Tensor, dense: bool) -> Tensor:
        if dense:
            return v @ inverse(op_diag) @ v
        root = LowRankPlusDiag(u, s, op_diag).cov_sqrt_mv(torch.eye(n, dtype=u.dtype)).T
        return (v @ root).square().sum()

    return {
        "logdet": logdet,
        "solve": solve,
        "inverse_diagonal": inverse_diagonal,
        "cov_sqrt_mv": cov_sqrt_mv,
    }


def test_grad_at_tied_singular_values() -> None:
    """An orthonormal factor, equal evals and an isotropic diag: every singular value of
    the scaled factor is the same. The gradients must be finite and equal the dense ones."""

    n, r = 10, 3
    gen = torch.Generator().manual_seed(71)
    u, _ = torch.linalg.qr(torch.randn(n, r, generator=gen, dtype=F64))
    s = torch.full((r,), 2.0, dtype=F64)
    d = torch.full((n,), 0.5, dtype=F64)
    v = torch.randn(n, generator=gen, dtype=F64)
    w = torch.randn(n, generator=gen, dtype=F64)

    for name, loss in _low_rank_losses(u, s, v, w).items():
        got = torch.func.grad(lambda x, f=loss: f(x, False))(d)
        want = torch.func.grad(lambda x, f=loss: f(x, True))(d)

        assert bool(torch.isfinite(got).all()), name
        close(got, want, 1e-10)


def test_low_rank_grad_matches_dense_at_random_factors() -> None:
    """The gradients with respect to `diag`, in float64, at a generic factor."""

    for seed in (72, 73):
        gen = torch.Generator().manual_seed(seed)
        u = torch.randn(LR_N, 4, generator=gen, dtype=F64)
        s = torch.rand(4, generator=gen, dtype=F64) + 0.5
        d = torch.rand(LR_N, generator=gen, dtype=F64) + 0.2
        v = torch.randn(LR_N, generator=gen, dtype=F64)
        w = torch.randn(LR_N, generator=gen, dtype=F64)

        for loss in _low_rank_losses(u, s, v, w).values():
            got = torch.func.grad(lambda x, f=loss: f(x, False))(d)
            want = torch.func.grad(lambda x, f=loss: f(x, True))(d)

            close(got, want, 1e-9)


@pytest.mark.parametrize("jitter", [1e-5, 1e-3], ids=["jitter-1e-5", "jitter-1e-3"])
def test_float32_nearly_dependent_columns_tie(jitter: float) -> None:
    """`add_same` of a posterior with a near copy of itself, in float32, with equal evals:
    the large singular values of the scaled factor tie, and the small ones sit near 1.
    Values and gradients match the float64 dense inverse. The jitter is never zero: with
    exactly dependent columns finite gradients are not guaranteed (see `LowRankPlusDiag`);
    1e-5 is still far above float32 rounding of the unit-norm columns."""

    n, r = 60, 4
    gen = torch.Generator().manual_seed(74)
    u, _ = torch.linalg.qr(torch.randn(n, r, generator=gen, dtype=F64))
    u2 = u + jitter * torch.randn(n, r, generator=gen, dtype=F64)
    s = torch.full((r,), 1e3)
    d = torch.full((n,), 1e-3)
    op = LowRankPlusDiag(u.float(), s, d).add_same(LowRankPlusDiag(u2.float(), s, torch.zeros(n)))

    p = _as_f64(op)
    inv = torch.linalg.inv(p)
    v = torch.randn(n, generator=gen, dtype=F64)

    # M7a's float32 tolerances for nearly dependent columns.
    root = _root(op.cov_sqrt_mv, n, torch.float32).double()
    torch.testing.assert_close((root @ root.T).diagonal(), inv.diagonal(), rtol=1e-3, atol=0)
    torch.testing.assert_close(op.inverse_diagonal().double(), inv.diagonal(), rtol=1e-3, atol=0)

    x = op.solve(v.float()).double()
    assert float((x - inv @ v).norm() / (inv @ v).norm()) < 1e-3
    assert abs(float(op.logdet().double() - torch.linalg.slogdet(p)[1])) < 1e-3  # nats

    # Gradients with respect to `diag`, against the dense float64 ones.
    w = torch.randn(n, generator=gen, dtype=F64)
    losses = _low_rank_losses(op.factor, op.evals, v.float(), w.float())
    dense = _low_rank_losses(op.factor.double(), op.evals.double(), v, w)

    for name, loss in losses.items():
        got = torch.func.grad(lambda x, f=loss: f(x, False))(op.diag).double()
        want = torch.func.grad(lambda x, f=dense[name]: f(x, True))(op.diag.double())

        assert bool(torch.isfinite(got).all()), name
        assert float((got - want).norm() / want.norm()) < 1e-3, name


_ONES = torch.ones(3, dtype=torch.float64)


@pytest.mark.parametrize(
    "call",
    [
        lambda op: op.logdet(),
        lambda op: op.solve(_ONES),
        lambda op: op.inverse_diagonal(),
        lambda op: op.cov_sqrt_mv(_ONES),
    ],
    ids=["logdet", "solve", "inverse_diagonal", "cov_sqrt_mv"],
)
def test_dense_indefinite_raises_structure_error(call: Callable[[Dense], Tensor]) -> None:
    op = Dense(torch.diag(torch.tensor([1.0, -2.0, 3.0], dtype=torch.float64)))

    with pytest.raises(StructureError, match="not positive definite.*prior precision"):
        call(op)


# --- the batch contract: [n, P] in one call equals n single-vector calls --------------


def _contract_ops() -> dict[str, tuple[object, Tensor, Callable, Callable]]:
    """Each built-in operator, its dense matrix, and its vector <-> flat maps."""

    gen = torch.Generator().manual_seed(21)
    a = torch.randn(N, N, generator=gen, dtype=F64)
    dense = a @ a.T + 0.1 * torch.eye(N, dtype=F64)
    d = torch.rand(N, generator=gen, dtype=F64) + 0.5
    kron, kron_dense = _kron_case((3, 2, 2, 2), True, F64)
    u, s, lr_d = _low_rank_parts()
    joint_dense, joint = _joint_case()
    bd_dense = torch.block_diag(torch.diag(d[:SPLIT]), dense[SPLIT:, SPLIT:])
    bd = BlockDiag({"a": Diag(d[:SPLIT]), "b": Dense(dense[SPLIT:, SPLIT:])})

    def same(v: Tensor) -> Tensor:
        return v

    def bd_vec(v: Tensor) -> dict[str, Tensor]:
        return {"a": v[..., :SPLIT], "b": v[..., SPLIT:]}

    def bd_flat(v: Mapping[str, Tensor]) -> Tensor:
        return torch.cat([v["a"], v["b"]], dim=-1)

    return {
        "Diag": (Diag(d), torch.diag(d), same, same),
        "Dense": (Dense(dense), dense, same, same),
        "KronEigenbasis": (kron, kron_dense, same, same),
        "LowRankPlusDiag": (LowRankPlusDiag(u, s, lr_d), _low_rank_dense(u, s, lr_d), same, same),
        "BlockDiag": (bd, bd_dense, bd_vec, bd_flat),
        "Joint": (joint, joint_dense, _split, _cat),
    }


@pytest.mark.parametrize(
    "name", ["Diag", "Dense", "KronEigenbasis", "LowRankPlusDiag", "BlockDiag", "Joint"]
)
def test_operator_contract_batches(name: str) -> None:
    """`solve` and `cov_sqrt_mv` on [3, P] equal the stacked single-vector results."""

    op, dense, vec, flat = _contract_ops()[name]
    op = cast(Operator[Any], op)
    p = dense.shape[0]
    m = torch.randn(3, p, generator=torch.Generator().manual_seed(3), dtype=F64)
    inv = torch.linalg.inv(dense)

    solved = flat(op.solve(vec(m)))
    close(solved, torch.stack([flat(op.solve(vec(row))) for row in m]), 1e-10)
    close(solved, m @ inv.T, 1e-9)

    roots = flat(op.cov_sqrt_mv(vec(m)))
    close(roots, torch.stack([flat(op.cov_sqrt_mv(vec(row))) for row in m]), 1e-10)

    # The root squares to the inverse: L from its columns, L L^T = P^-1.
    root = torch.stack([flat(op.cov_sqrt_mv(vec(e))) for e in torch.eye(p, dtype=F64)], dim=1)
    close(root @ root.T, inv, 1e-9)
    close(roots, m @ root.T, 1e-10)


def _dense_big(p: int = 40) -> Tensor:
    gen = torch.Generator().manual_seed(9)
    a = torch.randn(p, p, generator=gen, dtype=F64)

    return a @ a.T / p + 0.1 * torch.eye(p, dtype=F64)


def test_dense_batched_solve_matches_inverse() -> None:
    mat = _dense_big()
    v = torch.randn(7, mat.shape[0], generator=torch.Generator().manual_seed(1), dtype=F64)
    want = (torch.linalg.inv(mat) @ v.T).T

    close(Dense(mat).solve(v), want, 1e-10)
    close(Dense(mat).solve(v.reshape(7, 1, -1)), want.reshape(7, 1, -1), 1e-10)
    close(Dense(mat).solve(v[0]), want[0], 1e-10)


def test_dense_batched_cov_sqrt_root() -> None:
    mat = _dense_big()
    eye = torch.eye(mat.shape[0], dtype=F64)

    # Rows e_i give the rows of (L e_i), i.e. the columns of L.
    root = Dense(mat).cov_sqrt_mv(eye).T

    close(root @ root.T, torch.linalg.inv(mat), 1e-10)


# --- Dense with a cached eigendecomposition ---------------------------------------------


def _eigen_op() -> tuple[Dense, Tensor]:
    """An eigen-path precision after `scaled` and `add_isotropic`, and its dense matrix."""

    mat = _dense_big()
    s, lam = torch.tensor(3.0, dtype=F64), torch.tensor(0.7, dtype=F64)
    eye = torch.eye(mat.shape[0], dtype=F64)

    return Dense(mat).with_eigen().scaled(s).add_isotropic(lam), s * mat + lam * eye


def test_dense_eigen_matches_dense() -> None:
    op, want = _eigen_op()
    p = want.shape[0]
    inv = torch.linalg.inv(want)
    gen = torch.Generator().manual_seed(4)
    v = torch.randn(p, generator=gen, dtype=F64)
    m = torch.randn(7, 1, p, generator=gen, dtype=F64)

    evals, evecs = op.eig
    close(evals, torch.linalg.eigvalsh(want), 1e-10)
    close(evecs @ torch.diag(evals) @ evecs.T, want, 1e-10)

    close(op.mv(v), want @ v, 1e-10)
    close(op.diagonal(), want.diagonal(), 1e-10)
    close(op.logdet(), torch.linalg.slogdet(want)[1], 1e-10)
    close(op.solve(v), inv @ v, 1e-10)
    close(op.solve(m), m @ inv.T, 1e-10)
    close(op.inverse_diagonal(), inv.diagonal(), 1e-10)

    # Rows e_i give the columns of L; L L^T = P^-1.
    root = op.cov_sqrt_mv(torch.eye(p, dtype=F64)).T
    close(root @ root.T, inv, 1e-10)
    close(op.cov_sqrt_mv(m), m @ root.T, 1e-10)
    close(op.cov_sqrt_mv(v), root @ v, 1e-10)


def test_eig_dropped_by_add_diagonal() -> None:
    mat = _dense_big()
    d = torch.linspace(0.5, 2.0, mat.shape[0], dtype=F64)
    op = Dense(mat).with_eigen()

    shifted = op.add_diagonal(d)
    summed = op.add_same(Dense(mat))

    assert len(op.eig) == 2
    assert shifted.eig == () and summed.eig == ()
    close(shifted.logdet(), torch.linalg.slogdet(mat + torch.diag(d))[1], 1e-10)
    close(summed.logdet(), torch.linalg.slogdet(2 * mat)[1], 1e-10)


@pytest.mark.parametrize(
    "call",
    [
        lambda op: op.logdet(),
        lambda op: op.solve(_ONES),
        lambda op: op.inverse_diagonal(),
        lambda op: op.cov_sqrt_mv(_ONES),
    ],
    ids=["logdet", "solve", "inverse_diagonal", "cov_sqrt_mv"],
)
def test_dense_eigen_indefinite_raises(call: Callable[[Dense], Tensor]) -> None:
    """A curvature with a negative eigenvalue: positive after a large prior, refused with
    a small one."""

    curv = Dense(torch.diag(torch.tensor([1.0, -2.0, 3.0], dtype=F64))).with_eigen()

    call(curv.add_isotropic(torch.tensor(5.0, dtype=F64)))

    with pytest.raises(StructureError, match="not positive definite.*prior precision"):
        call(curv.add_isotropic(torch.tensor(1.0, dtype=F64)))


def test_dense_eigen_under_vmap() -> None:
    """Under `vmap` the positivity check, which reads data, is skipped."""

    mat = _dense_big()
    eye = torch.eye(mat.shape[0], dtype=F64)
    lams = torch.tensor([0.1, 1.0, 10.0], dtype=F64)
    op = Dense(mat).with_eigen()

    got = torch.vmap(lambda lam: op.add_isotropic(lam).logdet())(lams)
    want = torch.stack([torch.linalg.slogdet(mat + lam * eye)[1] for lam in lams])

    close(got, want, 1e-10)


def test_with_eigen_maps_dense_operators() -> None:
    joint_dense, joint = _joint_case()
    mat = _dense_big(4)
    bd = BlockDiag({"a": Dense(mat), "b": Diag(torch.ones(2, dtype=F64))})
    diag = Diag(torch.ones(3, dtype=F64))

    assert len(cast(Dense, with_eigen(Dense(mat))).eig) == 2

    j = cast(Joint[Dense], with_eigen(joint))
    close(j.op.eig[1] @ torch.diag(j.op.eig[0]) @ j.op.eig[1].T, joint_dense, 1e-10)

    b = cast(BlockDiag[Any], with_eigen(bd))
    assert len(b.blocks["a"].eig) == 2 and b.blocks["b"] is bd.blocks["b"]

    assert with_eigen(diag) is diag
