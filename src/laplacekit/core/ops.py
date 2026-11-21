"""Precision operators: a curvature plus a prior.

Each operator acts on one block's flat vector, except the per-block operators
(`HasBlocks`), which take mappings keyed by block name. `BlockDiag` holds one operator per
block, `Joint` one operator across all blocks.

`Operator[V]` is generic in the vector type `V`: `Tensor` for single-block operators,
`Mapping[str, Tensor]` for `BlockDiag` and `Joint`. A vector's last axis is the
operator's; any leading axes are a batch, so `[n, P]` holds `n` vectors. Adding a
structure an operator can't hold raises `StructureError` naming both types.
"""

import copy
import math
from abc import abstractmethod
from collections.abc import Callable, Mapping
from typing import Protocol, Self, cast, runtime_checkable

import torch
from torch import Tensor

from laplacekit.core._functorch import is_transformed
from laplacekit.core._pytree import pytree, static
from laplacekit.core.types import StructureError


@runtime_checkable
class Operator[V](Protocol):
    """A precision operator on vectors `V`: products, solves, its diagonal and log-determinant.

    A vector's leading axes are a batch: `mv`, `solve` and `cov_sqrt_mv` take `[..., P]`
    (or a mapping of such tensors) and act on each vector. A plugin operator held in a
    `LaplacePosterior` must accept this, since batched covariance products pass `[n, P]`.
    """

    @abstractmethod
    def mv(self, v: V, /) -> V:
        """The product $P v$ with the precision $P$.

        Args:
            v: A vector `[..., P]`, or a mapping of such vectors keyed by block name.

        Returns:
            $P v$, shaped like `v`.
        """
        ...

    @abstractmethod
    def diagonal(self) -> V:
        r"""The diagonal of the precision, $\operatorname{diag}(P)$.

        Returns:
            A vector `[P]`, or a mapping of them keyed by block name.
        """
        ...

    @abstractmethod
    def inverse_diagonal(self) -> V:
        r"""The diagonal of the covariance, $\operatorname{diag}(P^{-1})$.

        Operators compute it without forming $P^{-1}$ where their structure allows; `Dense`
        without a cached eigendecomposition forms it.

        Returns:
            A vector `[P]`, or a mapping of them keyed by block name.

        Raises:
            StructureError: Implementations may raise it when $P$ is not positive definite.
        """
        ...

    @abstractmethod
    def logdet(self) -> Tensor:
        r"""The log-determinant $\log \det P$.

        Returns:
            A 0-dim tensor.

        Raises:
            StructureError: Implementations may raise it when $P$ is not positive definite.
        """
        ...

    @abstractmethod
    def solve(self, v: V, /) -> V:
        """The solve $P^{-1} v$.

        Args:
            v: A vector `[..., P]`, or a mapping of such vectors keyed by block name.

        Returns:
            $P^{-1} v$, shaped like `v`.

        Raises:
            StructureError: Implementations may raise it when $P$ is not positive definite.
        """
        ...

    @abstractmethod
    def cov_sqrt_mv(self, v: V, /) -> V:
        r"""The product $L v$ with a square root $L$ of the covariance, $L L^\top = P^{-1}$.

        Args:
            v: A vector `[..., P]`, or a mapping of such vectors keyed by block name.

        Returns:
            $L v$, shaped like `v`.

        Raises:
            StructureError: Implementations may raise it when $P$ is not positive definite.
        """
        ...

    @abstractmethod
    def scaled(self, s: Tensor, /) -> Self:
        """The operator for $s P$.

        Args:
            s: A 0-dim factor.

        Returns:
            An operator of the same type.
        """
        ...

    @abstractmethod
    def add_isotropic(self, precision: Tensor, /) -> Self:
        r"""The operator for $P + \lambda I$.

        Args:
            precision: The precision $\lambda$, a 0-dim tensor.

        Returns:
            An operator of the same type.
        """
        ...

    @abstractmethod
    def add_diagonal(self, d: V, /) -> Self:
        r"""The operator for $P + \operatorname{diag}(d)$.

        Args:
            d: The diagonal $d$, shaped like a vector of the operator.

        Returns:
            An operator of the same type.

        Raises:
            StructureError: Implementations raise it when their structure can't hold a
                diagonal.
        """
        ...

    @abstractmethod
    def add_same(self, other: "Operator[V]", /) -> Self:
        """The operator for $P + P'$, with $P'$ an operator of the same type.

        Args:
            other: The operator $P'$.

        Returns:
            An operator of the same type.

        Raises:
            StructureError: `other` has another type, or implementations can't hold the sum.
        """
        ...


@runtime_checkable
class HasBlocks(Protocol):
    """An operator on mappings keyed by block name, such as `BlockDiag` or `Joint`.

    Per-block priors dispatch on this protocol.
    """

    @abstractmethod
    def block_sizes(self) -> Mapping[str, int]:
        """Each block's flat size.

        Returns:
            A mapping from block name to flat size, in block order.
        """
        ...

    @abstractmethod
    def add_per_block(self, precision: Mapping[str, Tensor], /) -> Self:
        r"""The operator with a precision added to each block.

        A 0-dim `precision[k]` adds $\lambda_k I$ to block $k$; a vector adds
        $\operatorname{diag}(\lambda_k)$.

        Args:
            precision: A mapping from block name to a 0-dim tensor or a vector `[P_k]`, with the
                operator's block names.

        Returns:
            An operator of the same type.

        Raises:
            StructureError: `precision` is not a mapping, or its keys differ from the blocks.
                Implementations also raise it when a block can't hold a diagonal.
        """
        ...


# torch.linalg is untyped in the stubs; give the functions we use real types.
_cholesky = cast(Callable[..., Tensor], torch.linalg.cholesky)
_solve_tri = cast(Callable[..., Tensor], torch.linalg.solve_triangular)
_qr = cast(Callable[..., tuple[Tensor, Tensor]], torch.linalg.qr)
_eigh = cast(Callable[..., tuple[Tensor, Tensor]], torch.linalg.eigh)


def _rows_solve(a: Tensor, y: Tensor, *, upper: bool) -> Tensor:
    """`y a^-1` for a triangular `a`, on the last axis of `y` (any leading axes)."""

    if y.ndim == 1:
        return _solve_tri(a, y.unsqueeze(0), upper=upper, left=False).squeeze(0)

    return _solve_tri(a, y, upper=upper, left=False)


def _mismatch(a: object, b: object) -> StructureError:
    return StructureError(
        f"cannot add {type(b).__name__} to {type(a).__name__}: add_same needs the same type"
    )


def _check_keys(blocks: Mapping[str, object], v: Mapping[str, object], what: str) -> None:
    """Raise if `v` and `blocks` have different keys, naming the missing and extra ones."""

    if not isinstance(v, Mapping):
        raise StructureError(f"{what} must be a mapping keyed by block, got {type(v).__name__}")

    if v.keys() != blocks.keys():
        missing = sorted(blocks.keys() - v.keys())
        extra = sorted(v.keys() - blocks.keys())

        raise StructureError(
            f"{what} keys differ from the blocks: missing {missing}, extra {extra}"
        )


@pytree
class Diag:
    r"""A diagonal precision $P = \operatorname{diag}(d)$, stored as its diagonal.

    Args:
        diag: The diagonal $d$, `[P]`.
    """

    diag: Tensor

    def mv(self, v: Tensor, /) -> Tensor:
        r"""$d \odot v$.

        Args:
            v: A vector `[..., P]`.

        Returns:
            $P v$, shaped like `v`.
        """

        return self.diag * v

    def diagonal(self) -> Tensor:
        """$d$.

        Returns:
            A vector `[P]`.
        """

        return self.diag

    def inverse_diagonal(self) -> Tensor:
        """$1 / d$, elementwise.

        Returns:
            A vector `[P]`.
        """

        return 1.0 / self.diag

    def logdet(self) -> Tensor:
        r"""$\sum_i \log d_i$.

        Returns:
            A 0-dim tensor.
        """

        return self.diag.log().sum()

    def solve(self, v: Tensor, /) -> Tensor:
        """$v / d$, elementwise.

        Args:
            v: A vector `[..., P]`.

        Returns:
            $P^{-1} v$, shaped like `v`.
        """

        return v / self.diag

    def cov_sqrt_mv(self, v: Tensor, /) -> Tensor:
        r"""$v / \sqrt{d}$, elementwise: $L = \operatorname{diag}(d)^{-1/2}$.

        Args:
            v: A vector `[..., P]`.

        Returns:
            $L v$, shaped like `v`.
        """

        return v / self.diag.sqrt()

    def scaled(self, s: Tensor, /) -> Self:
        r"""$\operatorname{diag}(s d)$.

        Args:
            s: A 0-dim factor.

        Returns:
            An operator of the same type.
        """

        return type(self)(s * self.diag)

    def add_isotropic(self, precision: Tensor, /) -> Self:
        r"""$\operatorname{diag}(d + \lambda)$.

        Args:
            precision: The precision $\lambda$, a 0-dim tensor.

        Returns:
            An operator of the same type.
        """

        return type(self)(self.diag + precision)

    def add_diagonal(self, d: Tensor, /) -> Self:
        r"""$\operatorname{diag}(d + d')$, with $d'$ the argument `d`.

        Args:
            d: The diagonal $d$, shaped like a vector of the operator.

        Returns:
            An operator of the same type.
        """

        return type(self)(self.diag + d)

    def add_same(self, other: Operator[Tensor], /) -> Self:
        """The `Diag` of the summed diagonals.

        Args:
            other: The operator $P'$.

        Returns:
            An operator of the same type.

        Raises:
            StructureError: `other` is not a `Diag`.
        """

        if not isinstance(other, Diag):
            raise _mismatch(self, other)

        return type(self)(self.diag + other.diag)


@pytree
class Dense:
    r"""A full precision matrix $P$, for small blocks such as a last layer.

    With `eig` set, `logdet`, `solve`, `cov_sqrt_mv` and `inverse_diagonal` read the
    eigendecomposition instead of factoring `mat`, so a prior-precision tune costs one
    `eigh` in all. `scaled` and `add_isotropic` keep it; `add_diagonal` and `add_same`
    drop it. Without it, they use the Cholesky factor $P = U^\top U$.

    Args:
        mat: The matrix $P$, `[P, P]`.
        eig: `()` or `(evals [P], evecs [P, P])`, the eigendecomposition
            $P = Q \operatorname{diag}(e) Q^\top$ with eigenvalues $e$ and eigenvectors
            $Q$. `with_eigen` sets it. It holds a second `[P, P]` array (384 MB at 9,802
            weights in float32). It is computed once from `mat`; replacing `mat`
            afterwards leaves it stale, and gradients through it go through `eigh`'s
            backward. Only detached posteriors should carry it.
    """

    mat: Tensor
    eig: tuple[()] | tuple[Tensor, Tensor] = ()

    def with_eigen(self) -> Self:
        r"""The same matrix with its eigendecomposition cached in `eig`.

        The eigenvalues are kept as `eigh` returns them, unclamped: round-off on a
        rank-deficient GGN gives tiny values of either sign, and a clamp would hide a
        genuinely indefinite `DenseHessian`. The positivity check runs on the posterior
        precision, after the prior is added.

        Returns:
            A `Dense` with the same `mat` and `eig` from `eigh` of
            $\tfrac{1}{2}(P + P^\top)$.
        """

        evals, evecs = _eigh(0.5 * (self.mat + self.mat.mT))

        return copy.replace(self, eig=(evals, evecs))

    def _not_pd(self) -> StructureError:
        n = self.mat.shape[-1]

        return StructureError(
            f"Dense: the {n}x{n} matrix is not positive definite, so a posterior precision "
            "has no Cholesky factor. Raise the prior precision, or use a PSD curvature "
            "(DenseGGN, a Fisher) instead of DenseHessian"
        )

    def _upper(self) -> Tensor:
        try:
            return _cholesky(self.mat, upper=True)

        except torch.linalg.LinAlgError as e:
            raise self._not_pd() from e

    def _positive_eig(self, eig: tuple[Tensor, Tensor]) -> tuple[Tensor, Tensor]:
        """`eig`, after checking the eigenvalues are positive.

        The check is skipped under `torch.func`, where the values can't be read.
        """

        evals, evecs = eig

        if not is_transformed(evals) and bool((evals <= 0).any()):
            raise self._not_pd()

        return evals, evecs

    def mv(self, v: Tensor, /) -> Tensor:
        """$P v$.

        Args:
            v: A vector `[..., P]`.

        Returns:
            $P v$, shaped like `v`.
        """

        return (self.mat @ v.unsqueeze(-1)).squeeze(-1)

    def diagonal(self) -> Tensor:
        r"""$\operatorname{diag}(P)$.

        Returns:
            A vector `[P]`.
        """

        return self.mat.diagonal()

    def inverse_diagonal(self) -> Tensor:
        r"""$\operatorname{diag}(P^{-1})$: $\sum_j Q_{ij}^2 / e_j$ from `eig`.

        Without `eig`, it forms $P^{-1} = U^{-1} U^{-\top}$ and takes its diagonal.

        Returns:
            A vector `[P]`.

        Raises:
            StructureError: $P$ is not positive definite: its Cholesky factorization
                fails or, with `eig`, an eigenvalue is $\le 0$ (not checked under
                `torch.func`).
        """

        if self.eig:
            evals, evecs = self._positive_eig(self.eig)

            return evecs.square() @ (1.0 / evals)

        return torch.cholesky_inverse(self._upper(), upper=True).diagonal()

    def logdet(self) -> Tensor:
        r"""$\sum_j \log e_j$ from `eig`, else $2 \sum_i \log U_{ii}$.

        Returns:
            A 0-dim tensor.

        Raises:
            StructureError: $P$ is not positive definite: its Cholesky factorization
                fails or, with `eig`, an eigenvalue is $\le 0$ (not checked under
                `torch.func`).
        """

        if self.eig:
            return self._positive_eig(self.eig)[0].log().sum()

        return 2 * self._upper().diagonal().log().sum()

    def _columns(self, v: Tensor, apply: Callable[[Tensor], Tensor]) -> Tensor:
        """`apply` once to the vectors of `v` (`[..., P]`), as the columns of a `[P, m]` matrix.

        So `U` is not copied for each vector.
        """

        cols = v.reshape(-1, self.mat.shape[-1]).mT

        return apply(cols).mT.reshape(v.shape)

    def solve(self, v: Tensor, /) -> Tensor:
        r"""$Q \operatorname{diag}(e)^{-1} Q^\top v$ from `eig`.

        Without `eig`, two triangular solves with the Cholesky factor $U$.

        Args:
            v: A vector `[..., P]`.

        Returns:
            $P^{-1} v$, shaped like `v`.

        Raises:
            StructureError: $P$ is not positive definite: its Cholesky factorization
                fails or, with `eig`, an eigenvalue is $\le 0$ (not checked under
                `torch.func`).
        """

        if self.eig:
            # P^-1 = Q diag(1/e) Q^T.
            evals, evecs = self._positive_eig(self.eig)
            inv = (1.0 / evals).unsqueeze(-1)

            return self._columns(v, lambda c: evecs @ (inv * (evecs.mT @ c)))

        upper = self._upper()

        return self._columns(v, lambda c: torch.cholesky_solve(c, upper, upper=True))

    def cov_sqrt_mv(self, v: Tensor, /) -> Tensor:
        r"""$L v$ with $L = Q \operatorname{diag}(e)^{-1/2}$ from `eig`, else $L = U^{-1}$.

        Args:
            v: A vector `[..., P]`.

        Returns:
            $L v$, shaped like `v`.

        Raises:
            StructureError: $P$ is not positive definite: its Cholesky factorization
                fails or, with `eig`, an eigenvalue is $\le 0$ (not checked under
                `torch.func`).
        """

        if self.eig:
            # L = Q diag(e^-1/2) gives L L^T = P^-1.
            evals, evecs = self._positive_eig(self.eig)
            inv_sqrt = evals.rsqrt().unsqueeze(-1)

            return self._columns(v, lambda c: evecs @ (inv_sqrt * c))

        # P = U^T U, so L = U^-1 gives L L^T = P^-1.
        upper = self._upper()

        return self._columns(v, lambda c: _solve_tri(upper, c, upper=True))

    def scaled(self, s: Tensor, /) -> Self:
        """$s P$, keeping `eig` with eigenvalues $s e$.

        Args:
            s: A 0-dim factor.

        Returns:
            An operator of the same type.
        """

        eig = (s * self.eig[0], self.eig[1]) if self.eig else ()

        return type(self)(s * self.mat, eig)

    def add_isotropic(self, precision: Tensor, /) -> Self:
        r"""$P + \lambda I$, keeping `eig` with eigenvalues $e + \lambda$.

        Args:
            precision: The precision $\lambda$, a 0-dim tensor.

        Returns:
            An operator of the same type.
        """

        eye = torch.eye(self.mat.shape[-1], dtype=self.mat.dtype, device=self.mat.device)
        eig = (self.eig[0] + precision, self.eig[1]) if self.eig else ()

        return type(self)(self.mat + precision * eye, eig)

    def add_diagonal(self, d: Tensor, /) -> Self:
        r"""$P + \operatorname{diag}(d)$, dropping `eig`.

        Args:
            d: The diagonal $d$, shaped like a vector of the operator.

        Returns:
            An operator of the same type.
        """

        return type(self)(self.mat + torch.diag(d))

    def add_same(self, other: Operator[Tensor], /) -> Self:
        """$P + P'$, dropping `eig`.

        Args:
            other: The operator $P'$.

        Returns:
            An operator of the same type.

        Raises:
            StructureError: `other` is not a `Dense`.
        """

        if not isinstance(other, Dense):
            raise _mismatch(self, other)

        return type(self)(self.mat + other.mat)


@pytree
class KronEigenbasis:
    r"""A Kronecker-factored matrix held in its eigenbasis, for one layer's block.

    The matrix acts on the folded matrix $M = [W \mid b]$ of shape `[out, in (+1)]`:

    $$P \operatorname{vec}(M) = (Q_G \otimes Q_A) \operatorname{diag}(\operatorname{vec} S)
    (Q_G \otimes Q_A)^\top \operatorname{vec}(M),$$

    with $\operatorname{vec}$ row-major. That is
    $M \mapsto Q_G \big(S \odot (Q_G^\top M Q_A)\big) Q_A^\top$. The flat vector is in
    layout order: the weight row-major (Conv2d weights as `[out, in*kh*kw]`), then the
    bias. The dense matrix is never formed.

    Args:
        evecs_a: $Q_A$, `[in (+1), in (+1)]`.
        evecs_g: $Q_G$, `[out, out]`.
        evals: The eigenvalues $S$, `[out, in (+1)]`.
        has_bias: Whether the block has a bias, the last column of $M$.
        weight_shape: The weight's shape, `(out, in, ...)`.
        factors: `()` or `(evals_a [in (+1)], evals_g [out], count [])`, the eigenvalues
            $\lambda_A$ and $\lambda_G$ of the two Kronecker factors and a count $c$, with
            $S = c \, \lambda_G \lambda_A^\top$. KFAC sets it; EKFAC (corrected
            eigenvalues) leaves it empty. `scaled(s)` keeps it and multiplies `count` by
            `s`, so in a posterior `count` is the fit's scale, not always an example count.
            Adding a prior drops it.

    Raises:
        StructureError: `evecs_a`, `evecs_g` or `evals` has the wrong shape for
            `weight_shape` and `has_bias`, or `factors` is neither `()` nor three tensors
            of the shapes above.
    """

    evecs_a: Tensor
    evecs_g: Tensor
    evals: Tensor
    has_bias: bool = static()
    weight_shape: tuple[int, ...] = static()
    factors: tuple[Tensor, ...] = ()

    def __post_init__(self) -> None:
        out = self.weight_shape[0]
        n_in = math.prod(self.weight_shape[1:]) + int(self.has_bias)
        want = {"evecs_a": (n_in, n_in), "evecs_g": (out, out), "evals": (out, n_in)}
        got = {"evecs_a": self.evecs_a, "evecs_g": self.evecs_g, "evals": self.evals}

        for name, shape in want.items():
            if tuple(got[name].shape) != shape:
                raise StructureError(
                    f"KronEigenbasis {name} is {tuple(got[name].shape)}, expected {shape} for "
                    f"weight {self.weight_shape} {'with' if self.has_bias else 'without'} bias"
                )

        if not self.factors:
            return

        if len(self.factors) != 3:
            raise StructureError(
                f"KronEigenbasis factors has {len(self.factors)} tensors, expected () or "
                "(evals_a, evals_g, count)"
            )

        want_f = {"evals_a": (n_in,), "evals_g": (out,), "count": ()}

        for (name, shape), t in zip(want_f.items(), self.factors, strict=True):
            if tuple(t.shape) != shape:
                raise StructureError(
                    f"KronEigenbasis factors {name} is {tuple(t.shape)}, expected {shape}"
                )

    def _fold(self, v: Tensor) -> Tensor:
        """`[..., P]` in layout order to `[..., out, in (+1)]`."""

        out = self.weight_shape[0]
        lead = v.shape[:-1]
        n_weight = v.shape[-1] - (out if self.has_bias else 0)
        w = v[..., :n_weight].reshape(*lead, out, -1)

        if not self.has_bias:
            return w

        return torch.cat([w, v[..., n_weight:].unsqueeze(-1)], dim=-1)

    def _unfold(self, m: Tensor) -> Tensor:
        """`[..., out, in (+1)]` to `[..., P]` in layout order."""

        lead = m.shape[:-2]

        if not self.has_bias:
            return m.reshape(*lead, -1)

        return torch.cat([m[..., :-1].reshape(*lead, -1), m[..., -1]], dim=-1)

    def _to_eigen(self, m: Tensor) -> Tensor:
        return self.evecs_g.T @ m @ self.evecs_a

    def _from_eigen(self, m: Tensor) -> Tensor:
        return self.evecs_g @ m @ self.evecs_a.T

    def mv(self, v: Tensor, /) -> Tensor:
        r"""$Q_G \big(S \odot (Q_G^\top M Q_A)\big) Q_A^\top$ on the folded `v`.

        Args:
            v: A vector `[..., P]`.

        Returns:
            $P v$, shaped like `v`.
        """

        return self._unfold(self._from_eigen(self.evals * self._to_eigen(self._fold(v))))

    def diagonal(self) -> Tensor:
        r"""$(Q_G \odot Q_G) \, S \, (Q_A \odot Q_A)^\top$, unfolded.

        Returns:
            A vector `[P]`.
        """

        d = self.evecs_g.square() @ self.evals @ self.evecs_a.square().T
        return self._unfold(d)

    def inverse_diagonal(self) -> Tensor:
        r"""$(Q_G \odot Q_G) \, (1 / S) \, (Q_A \odot Q_A)^\top$, unfolded.

        Returns:
            A vector `[P]`.
        """

        d = self.evecs_g.square() @ (1.0 / self.evals) @ self.evecs_a.square().T
        return self._unfold(d)

    def logdet(self) -> Tensor:
        r"""$\sum_{ij} \log S_{ij}$.

        Returns:
            A 0-dim tensor.
        """

        return self.evals.log().sum()

    def solve(self, v: Tensor, /) -> Tensor:
        r"""$Q_G \big((Q_G^\top M Q_A) / S\big) Q_A^\top$ on the folded `v`.

        Args:
            v: A vector `[..., P]`.

        Returns:
            $P^{-1} v$, shaped like `v`.
        """

        return self._unfold(self._from_eigen(self._to_eigen(self._fold(v)) / self.evals))

    def cov_sqrt_mv(self, v: Tensor, /) -> Tensor:
        r"""$L v$ with $L = (Q_G \otimes Q_A) \operatorname{diag}(\operatorname{vec} S)^{-1/2}$.

        On the folded `v`, that is $Q_G (M / \sqrt{S}) Q_A^\top$.

        Args:
            v: A vector `[..., P]`.

        Returns:
            $L v$, shaped like `v`.
        """

        # L = (Q_G kron Q_A) diag(evals)^-1/2, so L L^T = P^-1.
        return self._unfold(self._from_eigen(self._fold(v) / self.evals.sqrt()))

    def scaled(self, s: Tensor, /) -> Self:
        """$s P$: $s S$, and `count` times $s$ in `factors`.

        Args:
            s: A 0-dim factor.

        Returns:
            An operator of the same type.
        """

        if not self.factors:
            return copy.replace(self, evals=s * self.evals)

        la, lg, count = self.factors

        return copy.replace(self, evals=s * self.evals, factors=(la, lg, count * s))

    def add_isotropic(self, precision: Tensor, /) -> Self:
        r"""$P + \lambda I$: $\lambda$ added to every eigenvalue, $S + \lambda$; drops `factors`.

        Args:
            precision: The precision $\lambda$, a 0-dim tensor.

        Returns:
            An operator of the same type.
        """

        return copy.replace(self, evals=self.evals + precision, factors=())

    def add_diagonal(self, d: Tensor, /) -> Self:
        """Always raises: a diagonal prior has no Kronecker form, even a constant one.

        Args:
            d: The diagonal $d$, shaped like a vector of the operator.

        Returns:
            An operator of the same type.

        Raises:
            StructureError: Always. The message points to a per-block scalar prior or a
                curvature without Kronecker structure.
        """

        raise StructureError(
            "KronEigenbasis, a Kronecker-factored curvature (KFAC or EKFAC), has no Kronecker "
            "form for a diagonal prior: give one scalar per block (lk.priors.Isotropic, or "
            "lk.priors.Layerwise with a scalar per block), or, for a per-weight diagonal prior, "
            "fit a curvature without Kronecker structure, e.g. hessian_approx=lk.DiagGGN() or "
            "lk.DenseGGN()"
        )

    def add_same(self, other: Operator[Tensor], /) -> Self:
        """Always raises: the sum of two Kronecker eigenbases has no cheap Kronecker form.

        Args:
            other: The operator $P'$.

        Returns:
            An operator of the same type.

        Raises:
            StructureError: Always. For another type the message names both types. For a
                `KronEigenbasis` it points to a `Diag` curvature (or `Dense` on small
                blocks), or, for continual learning with KFAC, to
                `lk.priors.KronSum.of(post)`, which sums the Kronecker factors (approximate).
        """

        if not isinstance(other, KronEigenbasis):
            raise _mismatch(self, other)

        raise StructureError(
            "cannot add KronEigenbasis + KronEigenbasis (e.g. a KFAC posterior as the prior "
            "of a KFAC fit): the sum has no cheap Kronecker form; fit both with a Diag "
            "curvature (or Dense on small blocks) instead; for continual learning with KFAC, "
            "opt into lk.priors.KronSum.of(post), which sums the Kronecker factors (approximate)"
        )


@pytree
class LowRankPlusDiag:
    r"""A diagonal plus a low-rank term, $P = D + U \operatorname{diag}(s) U^\top$.

    Here $D = \operatorname{diag}(d)$. Products and the diagonal work for any $d$;
    `solve`, `inverse_diagonal`, `logdet` and `cov_sqrt_mv` need $d > 0$, which a prior
    gives. They use the Woodbury identity on a thin QR of the scaled factor and a second
    QR of a $2r \times r$ matrix, so each costs $O(n r^2)$; the dense matrix is never
    formed. With $B = D^{-1/2} U \operatorname{diag}(s)^{1/2}$, the precision is
    $P = D^{1/2} (I + B B^\top) D^{1/2}$. The thin QR $B = Q R$ and the thin QR
    $[R^\top; I] = Q_2 T$ give $T^\top T = I + R R^\top$, so

    $$P^{-1} = D^{-1/2} \big(I - Q (I - (T^\top T)^{-1}) Q^\top\big) D^{-1/2}, \qquad
    \log \det P = \sum_i \log d_i + 2 \sum_j \log |T_{jj}|.$$

    Through those four, an `evals` entry that is exactly zero gets no gradient (its column
    is dropped), and gradients may be NaN when the scaled factor has exactly dependent
    columns (the derivative of QR). Repeated singular values are fine.

    Args:
        factor: $U$, `[n, r]`.
        evals: $s$, `[r]`, non-negative.
        diag: $d$, `[n]`; zeros for a bare curvature.

    Raises:
        StructureError: `factor` is not 2-D, or `evals` or `diag` has the wrong shape for
            it.
    """

    factor: Tensor
    evals: Tensor
    diag: Tensor

    def __post_init__(self) -> None:
        if self.factor.ndim != 2:
            raise StructureError(
                f"LowRankPlusDiag factor must be 2-D [n, r], got {tuple(self.factor.shape)}"
            )

        n, r = self.factor.shape
        want = {"evals": (r,), "diag": (n,)}
        got = {"evals": self.evals, "diag": self.diag}

        for name, shape in want.items():
            if tuple(got[name].shape) != shape:
                raise StructureError(
                    f"LowRankPlusDiag {name} is {tuple(got[name].shape)}, expected {shape} "
                    f"for a factor of shape {(n, r)}"
                )

    def _woodbury(self) -> tuple[Tensor, Tensor, Tensor]:
        """`D^-1/2` (as a vector), and `Q`, `T` with `I + B B^T = I + Q (T^T T - I) Q^T`.

        `B = D^-1/2 U S^1/2`, so `P = D^1/2 (I + B B^T) D^1/2`. With `B = Q R` (thin QR) and
        the thin QR `[R^T; I] = Q_2 T`, `T^T T = I + R R^T`: `T^T` is the Cholesky factor
        of `I + R R^T`, and `T` is upper triangular with `|T_ii| >= 1`. The Gram matrices
        `B^T B` and `R R^T` are never formed: they square the condition number, which loses
        all accuracy in float32 when the factor has nearly dependent columns (as `add_same`
        gives). `[R^T; I]` always has full column rank, and `B` does unless its columns are
        exactly dependent, so the derivatives are finite at repeated singular values too.
        """

        inv_sqrt_d = self.diag.rsqrt()

        # A zero eigenvalue adds nothing, so its column is dropped (exact). Kept, its zero
        # column of B would make the QR's derivative (and that of sqrt at 0) NaN.
        keep = self.evals != 0
        b = inv_sqrt_d.unsqueeze(-1) * self.factor[:, keep] * self.evals[keep].sqrt()
        q, r = _qr(b, mode="reduced")

        eye = torch.eye(r.shape[0], dtype=r.dtype, device=r.device)
        _, t = _qr(torch.cat([r.T, eye]), mode="reduced")

        return inv_sqrt_d, q, t

    def mv(self, v: Tensor, /) -> Tensor:
        r"""$D v + U \operatorname{diag}(s) U^\top v$.

        Args:
            v: A vector `[..., P]`.

        Returns:
            $P v$, shaped like `v`.
        """

        return self.diag * v + ((v @ self.factor) * self.evals) @ self.factor.T

    def diagonal(self) -> Tensor:
        r"""$d + (U \odot U) s$.

        Returns:
            A vector `[P]`.
        """

        return self.diag + self.factor.square() @ self.evals

    def inverse_diagonal(self) -> Tensor:
        r"""The diagonal of $P^{-1}$ from the Woodbury factors, without forming $P^{-1}$.

        Entry $i$ is $\big(1 - \lVert Q_{i,:} \rVert^2 + \lVert M_{i,:} \rVert^2\big) / d_i$,
        with $M = Q T^{-1}$.

        Returns:
            A vector `[P]`.
        """

        # P^-1 = D^-1/2 (I - Q Q^T + M M^T) D^-1/2 with M = Q T^-1.
        _, q, t = self._woodbury()
        m = _solve_tri(t, q, upper=True, left=False)

        return (1.0 - q.square().sum(-1) + m.square().sum(-1)) / self.diag

    def logdet(self) -> Tensor:
        r"""$\sum_i \log d_i + 2 \sum_j \log |T_{jj}|$.

        Returns:
            A 0-dim tensor.
        """

        # log det P = sum log d + log det(T^T T) = sum log d + 2 sum log |T_ii|.
        _, _, t = self._woodbury()

        return self.diag.log().sum() + 2.0 * t.diagonal().abs().log().sum()

    def solve(self, v: Tensor, /) -> Tensor:
        r"""$P^{-1} v$ by the Woodbury identity, with two triangular solves in $T$.

        Args:
            v: A vector `[..., P]`.

        Returns:
            $P^{-1} v$, shaped like `v`.
        """

        # Woodbury: (I + B B^T)^-1 = I - Q (I - (T^T T)^-1) Q^T. On row vectors y = x Q,
        # y (T^T T)^-1 = y T^-1 T^-T.
        inv_sqrt_d, q, t = self._woodbury()
        x = inv_sqrt_d * v
        y = x @ q
        z = _rows_solve(t.T, _rows_solve(t, y, upper=True), upper=False)

        return inv_sqrt_d * (x - (y - z) @ q.T)

    def cov_sqrt_mv(self, v: Tensor, /) -> Tensor:
        r"""$L v$ with $L = D^{-1/2} (I - Q Q^\top + Q T^{-1} Q^\top)$.

        Args:
            v: A vector `[..., P]`.

        Returns:
            $L v$, shaped like `v`.
        """

        # L = D^-1/2 (I - Q Q^T + Q T^-1 Q^T), so L L^T = D^-1/2 (I - Q Q^T
        # + Q (T^T T)^-1 Q^T) D^-1/2 = P^-1. On a row vector, Q T^-1 Q^T v is (v Q) T^-T Q^T.
        inv_sqrt_d, q, t = self._woodbury()
        y = v @ q
        z = _rows_solve(t.T, y, upper=False)

        return inv_sqrt_d * (v - (y - z) @ q.T)

    def scaled(self, s: Tensor, /) -> Self:
        r"""$s P$: $s d$ and $s$ times `evals`.

        Args:
            s: A 0-dim factor.

        Returns:
            An operator of the same type.
        """

        return copy.replace(self, evals=s * self.evals, diag=s * self.diag)

    def add_isotropic(self, precision: Tensor, /) -> Self:
        r"""$P + \lambda I$: $\lambda$ added to $d$.

        Args:
            precision: The precision $\lambda$, a 0-dim tensor.

        Returns:
            An operator of the same type.
        """

        return copy.replace(self, diag=self.diag + precision)

    def add_diagonal(self, d: Tensor, /) -> Self:
        r"""$P + \operatorname{diag}(d')$: the argument $d'$ added to $d$.

        Args:
            d: The diagonal $d$, shaped like a vector of the operator.

        Returns:
            An operator of the same type.
        """

        return copy.replace(self, diag=self.diag + d)

    def truncated(self, rank: int, /) -> Self:
        r"""The same `diag`, with the low-rank part cut to its top `rank` eigenpairs.

        The eigenpairs of $U \operatorname{diag}(s) U^\top$ below the top `rank` are
        dropped, so their precision is lost. Below the width, the columns whose `evals`
        entry is zero are dropped first, and the top `rank` are taken from the rest. It
        costs a thin QR $U \operatorname{diag}(s)^{1/2} = Q R$ and an $r \times r$ `eigh`
        of $R R^\top$.

        Args:
            rank: The number of eigenpairs to keep, at least 1.

        Returns:
            The operator unchanged (exact) when `rank` is at or above the width. Otherwise
            one with `min(rank, nonzero, n)` columns, with `nonzero` the count of nonzero
            `evals` and `n` the number of rows, so it can be narrower than `rank`. Its
            eigenvalues are clamped at zero.

        Raises:
            ValueError: `rank` is below 1.
        """

        if rank < 1:
            raise ValueError(f"LowRankPlusDiag.truncated needs rank >= 1, got {rank}")

        if rank >= self.factor.shape[1]:
            return self

        # B = U S^1/2 = Q R, so B B^T = Q (R R^T) Q^T and R R^T = V diag(lam) V^T.
        keep = self.evals != 0
        b = self.factor[:, keep] * self.evals[keep].sqrt()
        q, r = _qr(b, mode="reduced")
        lam, v = _eigh(r @ r.T)

        # `eigh` sorts ascending; the top `rank` are the last ones.
        top = lam.flip(0)[:rank].clamp(min=0)
        w = q @ v.flip(1)[:, :rank]

        return copy.replace(self, factor=w, evals=top)

    def add_same(self, other: Operator[Tensor], /) -> Self:
        """The exact sum: the factors and evals concatenated, the diagonals summed.

        The width grows with each sum, as for a `LowRankGGN` posterior used as the prior of a
        `LowRankGGN` fit; `lk.priors.LowRankSum` truncates it back to a fixed rank instead.

        Args:
            other: The operator $P'$.

        Returns:
            An operator of the same type.

        Raises:
            StructureError: `other` is not a `LowRankPlusDiag`, or its size $n$ differs.
        """

        if not isinstance(other, LowRankPlusDiag):
            raise _mismatch(self, other)

        n, n_other = self.factor.shape[0], other.factor.shape[0]

        if n != n_other:
            raise StructureError(
                f"cannot add LowRankPlusDiag of size {n} and LowRankPlusDiag of size {n_other}"
            )

        return type(self)(
            torch.cat([self.factor, other.factor], dim=-1),
            torch.cat([self.evals, other.evals]),
            self.diag + other.diag,
        )


@pytree
class BlockDiag[Op: Operator[Tensor]]:
    r"""One operator per block, keyed by block name (see `layout`).

    The precision is block diagonal, $P = \operatorname{blockdiag}(P_1, \dots, P_K)$.
    Vectors here are mappings from block name to that block's flat vector.

    Args:
        blocks: The operator $P_k$ of each block, keyed by block name.
    """

    blocks: Mapping[str, Op]

    def mv(self, v: Mapping[str, Tensor], /) -> Mapping[str, Tensor]:
        """$P_k v_k$ for each block $k$.

        Args:
            v: A mapping of vectors `[..., P_k]`, keyed by block name.

        Returns:
            $P v$, shaped like `v`.

        Raises:
            StructureError: `v` is not a mapping, or its keys differ from the blocks.
        """

        _check_keys(self.blocks, v, "vector")
        return {k: op.mv(v[k]) for k, op in self.blocks.items()}

    def block_sizes(self) -> Mapping[str, int]:
        """Each block's flat size, the length of its diagonal.

        Returns:
            A mapping from block name to flat size, in block order.
        """

        return {k: op.diagonal().numel() for k, op in self.blocks.items()}

    def diagonal(self) -> Mapping[str, Tensor]:
        """Each block's diagonal.

        Returns:
            A mapping of vectors `[P_k]`, keyed by block name.
        """

        return {k: op.diagonal() for k, op in self.blocks.items()}

    def inverse_diagonal(self) -> Mapping[str, Tensor]:
        """Each block's `inverse_diagonal`.

        Returns:
            A mapping of vectors `[P_k]`, keyed by block name.

        Raises:
            StructureError: A block raises it (such as a `Dense` that is not positive definite).
        """

        return {k: op.inverse_diagonal() for k, op in self.blocks.items()}

    def logdet(self) -> Tensor:
        r"""$\sum_k \log \det P_k$.

        Returns:
            A 0-dim tensor.

        Raises:
            StructureError: A block raises it (such as a `Dense` that is not positive definite).
        """

        return torch.stack([op.logdet() for op in self.blocks.values()]).sum()

    def solve(self, v: Mapping[str, Tensor], /) -> Mapping[str, Tensor]:
        """$P_k^{-1} v_k$ for each block $k$.

        Args:
            v: A mapping of vectors `[..., P_k]`, keyed by block name.

        Returns:
            $P^{-1} v$, shaped like `v`.

        Raises:
            StructureError: `v` is not a mapping, its keys differ from the blocks, or a
                block raises it (such as a `Dense` that is not positive definite).
        """

        _check_keys(self.blocks, v, "vector")
        return {k: op.solve(v[k]) for k, op in self.blocks.items()}

    def cov_sqrt_mv(self, v: Mapping[str, Tensor], /) -> Mapping[str, Tensor]:
        """$L_k v_k$ for each block $k$, with $L_k$ the block's covariance square root.

        Args:
            v: A mapping of vectors `[..., P_k]`, keyed by block name.

        Returns:
            $L v$, shaped like `v`.

        Raises:
            StructureError: `v` is not a mapping, its keys differ from the blocks, or a
                block raises it (such as a `Dense` that is not positive definite).
        """

        _check_keys(self.blocks, v, "vector")
        return {k: op.cov_sqrt_mv(v[k]) for k, op in self.blocks.items()}

    def scaled(self, s: Tensor, /) -> Self:
        """Each block scaled by $s$.

        Args:
            s: A 0-dim factor.

        Returns:
            An operator of the same type.
        """

        return type(self)({k: op.scaled(s) for k, op in self.blocks.items()})

    def add_isotropic(self, precision: Tensor, /) -> Self:
        r"""$P_k + \lambda I$ for each block $k$.

        Args:
            precision: The precision $\lambda$, a 0-dim tensor.

        Returns:
            An operator of the same type.
        """

        return type(self)({k: op.add_isotropic(precision) for k, op in self.blocks.items()})

    def add_diagonal(self, d: Mapping[str, Tensor], /) -> Self:
        """The same as `add_per_block(d)`: a 0-dim entry is added as an isotropic term.

        Args:
            d: The diagonal $d$, shaped like a vector of the operator.

        Returns:
            An operator of the same type.

        Raises:
            StructureError: `d` is not a mapping, its keys differ from the blocks, or a
                block can't hold a diagonal (a `KronEigenbasis` given a vector).
        """

        return self.add_per_block(d)

    def add_per_block(self, precision: Mapping[str, Tensor], /) -> Self:
        """Add `precision[k]` to block `k`: a scalar to every diagonal entry, or a vector.

        A 0-dim `precision[k]` goes through the block's `add_isotropic`, a vector through its
        `add_diagonal`.

        Args:
            precision: A mapping from block name to a 0-dim tensor or a vector `[P_k]`, with the
                operator's block names.

        Returns:
            An operator of the same type.

        Raises:
            StructureError: `precision` is not a mapping, its keys differ from the blocks, or a
                block can't hold a diagonal (a `KronEigenbasis` given a vector).
        """

        _check_keys(self.blocks, precision, "precision")

        return type(self)(
            {
                k: op.add_isotropic(precision[k])
                if precision[k].ndim == 0
                else op.add_diagonal(precision[k])
                for k, op in self.blocks.items()
            }
        )

    def add_same(self, other: Operator[Mapping[str, Tensor]], /) -> Self:
        """$P_k + P'_k$ for each block $k$.

        Args:
            other: The operator $P'$.

        Returns:
            An operator of the same type.

        Raises:
            StructureError: `other` is not a `BlockDiag`, its block names differ, or a
                block's `add_same` raises it.
        """

        if not isinstance(other, BlockDiag):
            raise _mismatch(self, other)

        theirs: Mapping[str, Op] = other.blocks

        if theirs.keys() != self.blocks.keys():
            raise StructureError(
                f"BlockDiag block names differ: {sorted(self.blocks)} vs {sorted(theirs)}"
            )

        return type(self)({k: op.add_same(theirs[k]) for k, op in self.blocks.items()})


@pytree
class Joint[Op: Operator[Tensor]]:
    """One operator across all blocks, on vectors keyed by block name.

    A vector's blocks are concatenated in `sizes` order (leading batch axes kept), passed
    to `op`, and split back.

    Args:
        op: The operator on the concatenated vector.
        sizes: Each block's name and flat size, in concatenation order.
    """

    op: Op
    sizes: tuple[tuple[str, int], ...] = static()

    def block_sizes(self) -> Mapping[str, int]:
        """`sizes` as a mapping.

        Returns:
            A mapping from block name to flat size, in block order.
        """

        return dict(self.sizes)

    def _cat(self, v: Mapping[str, Tensor], what: str) -> Tensor:
        sizes = self.block_sizes()
        _check_keys(sizes, v, what)

        for k, n in self.sizes:
            if v[k].shape[-1:] != (n,):
                raise StructureError(
                    f"{what} block {k!r} has shape {tuple(v[k].shape)}, expected {n} entries "
                    "on the last axis"
                )

        return torch.cat([v[k] for k, _ in self.sizes], dim=-1)

    def _split(self, x: Tensor) -> Mapping[str, Tensor]:
        parts = torch.split(x, [n for _, n in self.sizes], dim=-1)

        return {k: p for (k, _), p in zip(self.sizes, parts, strict=True)}

    def mv(self, v: Mapping[str, Tensor], /) -> Mapping[str, Tensor]:
        """`op.mv` on the concatenated `v`, split back into blocks.

        Args:
            v: A mapping of vectors `[..., P_k]`, keyed by block name.

        Returns:
            $P v$, shaped like `v`.

        Raises:
            StructureError: `v` is not a mapping, its keys differ from the blocks, or a
                block's last axis is not its size.
        """

        return self._split(self.op.mv(self._cat(v, "vector")))

    def diagonal(self) -> Mapping[str, Tensor]:
        """`op.diagonal()`, split into blocks.

        Returns:
            A mapping of vectors `[P_k]`, keyed by block name.
        """

        return self._split(self.op.diagonal())

    def inverse_diagonal(self) -> Mapping[str, Tensor]:
        """`op.inverse_diagonal()`, split into blocks.

        Returns:
            A mapping of vectors `[P_k]`, keyed by block name.

        Raises:
            StructureError: `op` raises it (such as a `Dense` that is not positive definite).
        """

        return self._split(self.op.inverse_diagonal())

    def logdet(self) -> Tensor:
        """`op.logdet()`.

        Returns:
            A 0-dim tensor.

        Raises:
            StructureError: `op` raises it (such as a `Dense` that is not positive definite).
        """

        return self.op.logdet()

    def solve(self, v: Mapping[str, Tensor], /) -> Mapping[str, Tensor]:
        """`op.solve` on the concatenated `v`, split back into blocks.

        Args:
            v: A mapping of vectors `[..., P_k]`, keyed by block name.

        Returns:
            $P^{-1} v$, shaped like `v`.

        Raises:
            StructureError: `v` is not a mapping, its keys differ from the blocks, or a
                block's last axis is not its size. `op` may raise it too (such as a `Dense`
                that is not positive definite).
        """

        return self._split(self.op.solve(self._cat(v, "vector")))

    def cov_sqrt_mv(self, v: Mapping[str, Tensor], /) -> Mapping[str, Tensor]:
        """`op.cov_sqrt_mv` on the concatenated `v`, split back into blocks.

        Args:
            v: A mapping of vectors `[..., P_k]`, keyed by block name.

        Returns:
            $L v$, shaped like `v`.

        Raises:
            StructureError: `v` is not a mapping, its keys differ from the blocks, or a
                block's last axis is not its size. `op` may raise it too (such as a `Dense`
                that is not positive definite).
        """

        return self._split(self.op.cov_sqrt_mv(self._cat(v, "vector")))

    def scaled(self, s: Tensor, /) -> Self:
        """`op.scaled(s)`.

        Args:
            s: A 0-dim factor.

        Returns:
            An operator of the same type.
        """

        return copy.replace(self, op=self.op.scaled(s))

    def add_isotropic(self, precision: Tensor, /) -> Self:
        r"""`op.add_isotropic(precision)`.

        Args:
            precision: The precision $\lambda$, a 0-dim tensor.

        Returns:
            An operator of the same type.
        """

        return copy.replace(self, op=self.op.add_isotropic(precision))

    def add_diagonal(self, d: Mapping[str, Tensor], /) -> Self:
        """`op.add_diagonal` with the concatenated `d`.

        Args:
            d: The diagonal $d$, shaped like a vector of the operator.

        Returns:
            An operator of the same type.

        Raises:
            StructureError: `d` is not a mapping, its keys differ from the blocks, a
                block's last axis is not its size, or `op` can't hold a diagonal.
        """

        return copy.replace(self, op=self.op.add_diagonal(self._cat(d, "diagonal")))

    def add_per_block(self, precision: Mapping[str, Tensor], /) -> Self:
        """Add `precision[k]` to block `k`: a scalar to every diagonal entry, or a vector.

        A 0-dim `precision[k]` is expanded to the block's size, and the whole goes through
        `add_diagonal`.

        Args:
            precision: A mapping from block name to a 0-dim tensor or a vector `[P_k]`, with the
                operator's block names.

        Returns:
            An operator of the same type.

        Raises:
            StructureError: `precision` is not a mapping, its keys differ from the blocks, a
                vector's last axis is not its block's size, or `op` can't hold a diagonal.
        """

        _check_keys(self.block_sizes(), precision, "precision")

        full = {
            k: precision[k].expand(n) if precision[k].ndim == 0 else precision[k]
            for k, n in self.sizes
        }

        return self.add_diagonal(full)

    def add_same(self, other: Operator[Mapping[str, Tensor]], /) -> Self:
        """`op.add_same(other.op)`.

        Args:
            other: The operator $P'$.

        Returns:
            An operator of the same type.

        Raises:
            StructureError: `other` is not a `Joint`, its `sizes` differ, or `op.add_same`
                raises it.
        """

        if not isinstance(other, Joint):
            raise _mismatch(self, other)

        theirs: tuple[tuple[str, int], ...] = other.sizes

        if theirs != self.sizes:
            raise StructureError(f"Joint blocks differ: {list(self.sizes)} vs {list(theirs)}")

        return copy.replace(self, op=self.op.add_same(cast(Op, other.op)))


def with_eigen[V](op: Operator[V], /) -> Operator[V]:
    """`op` with each `Dense` in it carrying its eigendecomposition (`Dense.with_eigen`).

    Tuning the prior of the result costs one `eigh` per `Dense` instead of a Cholesky
    factorization per step.

    Args:
        op: The operator. It maps a `Dense`, the `Dense` inside a `Joint`, and the `Dense`
            blocks of a `BlockDiag`; any other operator or block comes back unchanged.

    Returns:
        An operator of the same type.
    """

    if isinstance(op, Dense):
        return cast(Operator[V], op.with_eigen())

    if isinstance(op, Joint) and isinstance(op.op, Dense):
        return cast(Operator[V], copy.replace(op, op=op.op.with_eigen()))

    if isinstance(op, BlockDiag):
        blocks: Mapping[str, Operator[Tensor]] = op.blocks
        mapped = {k: b.with_eigen() if isinstance(b, Dense) else b for k, b in blocks.items()}

        return cast(Operator[V], copy.replace(op, blocks=mapped))

    return op
