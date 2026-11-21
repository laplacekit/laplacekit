"""Built-in priors. A prior is any weight-space posterior (see `laplacekit.core.posteriors`).

`add_precision_to` is the prior's half of the double dispatch: it calls the operator's
`add_isotropic` / `add_diagonal` / `add_per_block`, so the sum keeps the curvature's
structure. `Isotropic`, `Diagonal` and `Layerwise` also have `cov_mv`, so they can be
pushed to function space, and `tunable`, so `lk.laplace` tunes their precision by default
(`TunablePrior`).

Each prior takes an optional `mean` (a mapping like the fitted parameters); `None` means
zero. A `mean` whose names, shapes, dtypes or devices differ from the parameters raises
`StructureError`.

`KronSum.of(post)` turns a KFAC posterior into the prior of the next KFAC fit (continual learning
with KFAC). It sums the Kronecker factors, an approximation; a plain KFAC posterior as that
prior raises.
`KronSum` has neither `cov_mv` nor `tunable`.

`LowRankSum.of(post)` does the same for a `LowRankGGN` posterior: it sums the low-rank parts
and keeps the top `rank` eigenpairs, an approximation. A plain `LowRankGGN` posterior as that
prior is exact, but the width grows with every fit. `LowRankSum` has neither `cov_mv` nor
`tunable` either.

`Diagonal` is flat over the parameters in mapping order, which for a per-block operator
(`BlockDiag` or `Joint`) must be the block order of `Layout.by_module`. `Layerwise` groups
parameters into blocks by module prefix (`Layout.by_module`).
"""

import copy
import math
from collections.abc import Iterable, Mapping
from typing import Any, Self, cast

import torch
from torch import Tensor

from laplacekit.core import (
    BlockDiag,
    HasBlocks,
    Joint,
    KronEigenbasis,
    LaplacePosterior,
    Layout,
    LowRankPlusDiag,
    Operator,
    Params,
    Prior,
    StructureError,
    check_location,
    curvature_scale,
    pytree,
    static,
)

__all__ = ["Diagonal", "Isotropic", "KronSum", "Layerwise", "LowRankSum"]


def _summary(t: Tensor) -> str:
    """A number for a scalar, the shape otherwise."""

    return f"{t.item():.3g}" if t.numel() == 1 else str(tuple(t.shape))


def _normal_log_prob(x: Tensor, prec: Tensor) -> Tensor:
    return 0.5 * (prec.log() - math.log(2 * math.pi)) - 0.5 * prec * x * x


def _as_precision(value: float | Tensor, what: str, /) -> Tensor:
    """`value` as a floating tensor, refusing non-positive entries."""

    if isinstance(value, bool) or not isinstance(value, float | int | Tensor):
        raise TypeError(
            f"{what} takes a float or a tensor as prior_precision, got {type(value).__name__}"
        )

    t = torch.as_tensor(value)

    if not t.is_floating_point():
        t = t.to(torch.get_default_dtype())

    if not bool((t > 0).all()):
        raise ValueError(f"{what} prior_precision must be positive, got {t}")

    return t


def _mean_repr(mean: Params | None) -> str:
    return "" if mean is None else f", mean=<{len(mean)} tensors>"


def _location(mean: Params | None, like: Params) -> Params:
    """`mean`, checked against the names, shapes, dtypes and devices of `like`.

    Zeros when `None`.
    """

    if mean is None:
        return {k: torch.zeros_like(x) for k, x in like.items()}

    check_location(mean, like)

    return mean


def _centred(mean: Params | None, params: Params) -> Params:
    """`params - mean`, in the order of `params`."""

    if mean is None:
        return params

    mu = _location(mean, params)

    return {k: x - mu[k] for k, x in params.items()}


@pytree
class Isotropic:
    r"""Isotropic Gaussian prior $\mathcal{N}(\mu, \lambda^{-1} I)$ over all parameters.

    `Isotropic()` has precision 1. `Isotropic(10.0)` sets it.

    Args:
        prior_precision: Prior precision $\lambda > 0$, a float or a 0-dim tensor. The
            marginal-likelihood tuner of `lk.laplace` tunes it by default.
        mean: Prior mean $\mu$ as a parameter mapping. `None` means zero.
    """

    prior_precision: Tensor
    mean: Params | None = None

    def __init__(self, prior_precision: float | Tensor = 1.0, mean: Params | None = None) -> None:
        object.__setattr__(self, "prior_precision", _as_precision(prior_precision, "Isotropic"))
        object.__setattr__(self, "mean", mean)

    def __repr__(self) -> str:
        return f"Isotropic(prior_precision={_summary(self.prior_precision)}{_mean_repr(self.mean)})"

    def log_prob(self, params: Params, /) -> Tensor:
        r"""Log density at `params`.

        Args:
            params: Parameter mapping with the posterior's keys and shapes.

        Returns:
            A 0-dim tensor, $\log q(\theta)$.
        """

        delta = _centred(self.mean, params)
        terms = (_normal_log_prob(x, self.prior_precision).sum() for x in delta.values())

        return sum(
            terms,
            torch.zeros((), dtype=self.prior_precision.dtype, device=self.prior_precision.device),
        )

    def location(self, like: Params, /) -> Params:
        """The mean, shaped like `like`.

        Args:
            like: Parameter mapping whose names, shapes, dtypes and devices the mean must have.

        Returns:
            The mean, a parameter mapping.
        """

        return _location(self.mean, like)

    def tunable(self) -> Tensor:
        """The precision.

        Returns:
            The prior's precision, which is a leaf of the prior and positive, since the
            tune works on its log.
        """

        return self.prior_precision

    def add_precision_to[Op: Operator[Any]](self, op: Op, /) -> Op:
        r"""`op` plus the precision, as `op.add_isotropic(prior_precision)`.

        Args:
            op: A precision operator.

        Returns:
            An operator of the same type, $\text{op} + \Lambda_0$.
        """

        return op.add_isotropic(self.prior_precision)

    def cov_mv(self, v: Params, /) -> Params:
        r"""Covariance times `v`: `v / prior_precision`.

        Args:
            v: Parameter mapping with the posterior's keys and shapes.

        Returns:
            $\Sigma v$, a parameter mapping.
        """

        return {k: x / self.prior_precision for k, x in v.items()}


@pytree
class Diagonal:
    r"""Diagonal Gaussian prior $\mathcal{N}(\mu, \operatorname{diag}(\lambda)^{-1})$.

    On a per-block operator (`BlockDiag` or `Joint`), $\lambda$ is split in the operator's
    block order, which must be the block order of `Layout.by_module` (as for every
    operator a fit returns). A hand-built one in another order gets the wrong entries.

    `Diagonal()` starts every precision at 1, and `lk.laplace` expands it to one precision
    per parameter (`expand`) before the tune.

    Args:
        prior_precision: Prior precisions $\lambda > 0$: a float or 0-dim tensor (the same
            for every parameter), or a vector `[P]` flat over all parameters, in mapping
            order. The marginal-likelihood tuner of `lk.laplace` tunes it by default.
        mean: Prior mean $\mu$ as a parameter mapping. `None` means zero.
    """

    prior_precision: Tensor
    mean: Params | None = None

    def __init__(self, prior_precision: float | Tensor = 1.0, mean: Params | None = None) -> None:
        object.__setattr__(self, "prior_precision", _as_precision(prior_precision, "Diagonal"))
        object.__setattr__(self, "mean", mean)

    def __repr__(self) -> str:
        return f"Diagonal(prior_precision={_summary(self.prior_precision)}{_mean_repr(self.mean)})"

    def expand(self, params: Params, /) -> Self:
        """The prior with one precision per parameter entry.

        Args:
            params: The parameters the prior is over.

        Returns:
            `self` when the precision is already a vector, else a copy whose precision is
            the scalar repeated over all entries of `params`.
        """

        if self.prior_precision.ndim > 0:
            return self

        size = sum(x.numel() for x in params.values())

        return copy.replace(self, prior_precision=self.prior_precision.expand(size).clone())

    def _vector(self, size: int) -> Tensor:
        lam = self.prior_precision
        return lam.expand(size) if lam.ndim == 0 else lam

    def _flat(self, params: Params) -> Tensor:
        return torch.cat([x.reshape(-1) for x in params.values()])

    def _like(self, flat: Tensor, like: Params) -> Params:
        sizes = [x.numel() for x in like.values()]
        parts = torch.split(flat, sizes)

        return {k: p.reshape(x.shape) for (k, x), p in zip(like.items(), parts, strict=True)}

    def log_prob(self, params: Params, /) -> Tensor:
        r"""Log density at `params`.

        Args:
            params: Parameter mapping with the posterior's keys and shapes.

        Returns:
            A 0-dim tensor, $\log q(\theta)$.
        """

        flat = self._flat(_centred(self.mean, params))

        return _normal_log_prob(flat, self._vector(flat.numel())).sum()

    def location(self, like: Params, /) -> Params:
        """The mean, shaped like `like`.

        Args:
            like: Parameter mapping whose names, shapes, dtypes and devices the mean must have.

        Returns:
            The mean, a parameter mapping.
        """

        return _location(self.mean, like)

    def tunable(self) -> Tensor:
        """The precisions.

        Returns:
            The prior's precision, which is a leaf of the prior and positive, since the
            tune works on its log.
        """

        return self.prior_precision

    def add_precision_to[Op: Operator[Any]](self, op: Op, /) -> Op:
        r"""`op` plus $\operatorname{diag}(\lambda)$.

        A per-block operator gets $\lambda$ split by block through `op.add_per_block`; any
        other operator gets `op.add_diagonal(prior_precision)`.

        Args:
            op: A precision operator.

        Returns:
            An operator of the same type, $\text{op} + \Lambda_0$.

        Raises:
            StructureError: `prior_precision` has another size than the per-block operator, or the
                operator can't hold a diagonal (a `KronEigenbasis`, say).
        """

        if isinstance(op, HasBlocks):
            # Split in `Layout.by_module` block order, which is the layout operators use.
            by_block = op.block_sizes()
            sizes = list(by_block.values())

            lam = self._vector(sum(sizes))

            if lam.numel() != sum(sizes):
                raise StructureError(
                    f"Diagonal prior has {lam.numel()} entries, the per-block operator "
                    f"(BlockDiag or Joint) needs {sum(sizes)}"
                )

            parts = dict(zip(by_block, torch.split(lam, sizes), strict=True))

            return cast(Op, op.add_per_block(parts))

        if self.prior_precision.ndim == 0:
            return op.add_isotropic(self.prior_precision)

        return op.add_diagonal(self.prior_precision)

    def cov_mv(self, v: Params, /) -> Params:
        r"""Covariance times `v`: $v / \lambda$, elementwise.

        Args:
            v: Parameter mapping with the posterior's keys and shapes.

        Returns:
            $\Sigma v$, a parameter mapping.
        """

        flat = self._flat(v)

        return self._like(flat / self._vector(flat.numel()), v)


@pytree
class Layerwise:
    r"""Gaussian prior with one precision per block, $\prod_b \mathcal{N}(\mu_b, \lambda_b^{-1} I)$.

    Blocks group parameters by module prefix (`Layout.by_module`). Needs a per-block
    operator (`BlockDiag` or `Joint`).

    `Layerwise()` starts every block at precision 1, and `lk.laplace` expands it to one
    precision per block (`expand`) before the tune.

    Args:
        prior_precision: The precisions $\lambda_b > 0$: a float or 0-dim tensor (the same
            for every block), or a mapping from block name to a float or 0-dim tensor,
            whose keys must be the blocks of the parameters. The marginal-likelihood tuner
            of `lk.laplace` tunes them by default.
        mean: Prior mean $\mu$ as a parameter mapping. `None` means zero.
    """

    prior_precision: Tensor | Mapping[str, Tensor]
    mean: Params | None = None

    def __init__(
        self,
        prior_precision: float | Tensor | Mapping[str, float | Tensor] = 1.0,
        mean: Params | None = None,
    ) -> None:
        lam: Tensor | Mapping[str, Tensor] = (
            {b: _as_precision(v, "Layerwise") for b, v in prior_precision.items()}
            if isinstance(prior_precision, Mapping)
            else _as_precision(prior_precision, "Layerwise")
        )
        object.__setattr__(self, "prior_precision", lam)
        object.__setattr__(self, "mean", mean)

    def __repr__(self) -> str:
        lam = self.prior_precision

        if isinstance(lam, Mapping):
            items = ", ".join(f'"{b}": {_summary(p)}' for b, p in lam.items())
            shown = f"{{{items}}}"
        else:
            shown = _summary(lam)

        return f"Layerwise(prior_precision={shown}{_mean_repr(self.mean)})"

    def expand(self, params: Params, /) -> Self:
        """The prior with one precision per block of `params`.

        Args:
            params: The parameters the prior is over.

        Returns:
            `self` when the precision is already a mapping, else a copy whose precision
            maps each block (`Layout.by_module`) to its own copy of the scalar.
        """

        lam = self.prior_precision

        if isinstance(lam, Mapping):
            return self

        blocks = Layout.by_module(params).blocks

        return copy.replace(self, prior_precision={b: lam.clone() for b in blocks})

    def _per_block(self, blocks: Iterable[str]) -> Mapping[str, Tensor]:
        lam = self.prior_precision

        if isinstance(lam, Mapping):
            return lam

        return dict.fromkeys(blocks, lam)

    def _by_block(self, params: Params) -> dict[str, list[tuple[str, Tensor]]]:
        layout = Layout.by_module(params)
        lam = self.prior_precision

        if isinstance(lam, Mapping) and layout.blocks.keys() != lam.keys():
            raise StructureError(
                f"Layerwise keys {sorted(lam)} differ from the blocks {sorted(layout.blocks)}"
            )

        return {b: [(n, params[n]) for n in names] for b, names in layout.blocks.items()}

    def log_prob(self, params: Params, /) -> Tensor:
        r"""Log density at `params`.

        Args:
            params: Parameter mapping with the posterior's keys and shapes.

        Returns:
            A 0-dim tensor, $\log q(\theta)$.

        Raises:
            StructureError: The keys of `prior_precision` differ from the blocks of the parameters.
        """

        blocks = self._by_block(_centred(self.mean, params))
        lam = self._per_block(blocks)
        terms = (_normal_log_prob(x, lam[b]).sum() for b, items in blocks.items() for _, x in items)

        return sum(terms, torch.zeros(()))

    def location(self, like: Params, /) -> Params:
        """The mean, shaped like `like`.

        Args:
            like: Parameter mapping whose names, shapes, dtypes and devices the mean must have.

        Returns:
            The mean, a parameter mapping.
        """

        return _location(self.mean, like)

    def tunable(self) -> Tensor | Mapping[str, Tensor]:
        """The precisions, by block, or the one shared by all blocks.

        Returns:
            The prior's precision, which is a leaf of the prior and positive, since the
            tune works on its log.
        """

        return self.prior_precision

    def add_precision_to[Op: Operator[Any]](self, op: Op, /) -> Op:
        r"""`op` plus $\lambda_b I$ on each block $b$, through `op.add_per_block`.

        Args:
            op: A precision operator.

        Returns:
            An operator of the same type, $\text{op} + \Lambda_0$.

        Raises:
            StructureError: `op` is not a per-block operator (`BlockDiag` or `Joint`), or
                its block names differ from the keys of `prior_precision`.
        """

        if not isinstance(op, HasBlocks):
            raise StructureError(
                "Layerwise prior needs a per-block operator (BlockDiag or Joint), "
                f"got {type(op).__name__}"
            )

        return cast(Op, op.add_per_block(self._per_block(op.block_sizes())))

    def cov_mv(self, v: Params, /) -> Params:
        r"""Covariance times `v`: $v_b / \lambda_b$ on each block $b$.

        Args:
            v: Parameter mapping with the posterior's keys and shapes.

        Returns:
            $\Sigma v$, a parameter mapping.

        Raises:
            StructureError: The keys of `prior_precision` differ from the blocks of the parameters.
        """

        blocks = self._by_block(v)
        lam = self._per_block(blocks)

        return {n: x / lam[b] for b, items in blocks.items() for n, x in items}


def _fit_scale(post: LaplacePosterior[Any, Any]) -> Tensor:
    r"""What a fit's curvature is multiplied by in its precision.

    That is the scale times the likelihood's curvature factor ($1 / \sigma^2$ for a
    `Gaussian`).
    """

    factor = curvature_scale(post.likelihood)

    return post.n_data if factor is None else post.n_data * factor


def _chain_log_prob(
    base: Prior, summed: Operator[Any], mean: Params, params: Params, dtype: torch.dtype
) -> Tensor:
    r"""$\log \mathcal{N}(\theta; \mu, P^{-1})$, with $P$ the base precision added to `summed`.

    `params - mean` is cast to `dtype`, the summed curvature's: the dtype of the fits'
    statistics, at least float32 (`stat_dtype`), as `LaplacePosterior.log_prob` does. The
    operator works in that dtype, so lower-precision parameters are cast up first.
    """

    layout = Layout.by_module(mean)
    prec = base.add_precision_to(summed)
    delta = layout.flatten_all({k: (params[k] - mean[k]).to(dtype) for k in mean})
    prec_delta = prec.mv(delta)

    quad = sum((delta[b] * prec_delta[b]).sum() for b in delta)
    dim = sum(d.numel() for d in delta.values())

    return 0.5 * prec.logdet() - 0.5 * dim * math.log(2 * math.pi) - 0.5 * quad


def _kron_blocks(op: object, what: str) -> Mapping[str, KronEigenbasis]:
    """The blocks of a `BlockDiag` of `KronEigenbasis` that keeps its Kronecker factors."""

    want = "a BlockDiag of KronEigenbasis (KFAC)"

    if not isinstance(op, BlockDiag):
        raise StructureError(f"KronSum needs {what} to be {want}, got {type(op).__name__}")

    blocks = cast(Mapping[str, object], op.blocks)

    for b, blk in blocks.items():
        if not isinstance(blk, KronEigenbasis):
            raise StructureError(
                f"KronSum needs {what} to be {want}, but block {b!r} is {type(blk).__name__}"
            )

        if not blk.factors:
            raise StructureError(
                f"KronSum needs the Kronecker factors of {what}, but block {b!r} has none: "
                "EKFAC's corrected eigenvalues have no Kronecker factors (and adding a prior "
                "drops them)"
            )

    return cast(Mapping[str, KronEigenbasis], blocks)


def _matrix(evecs: Tensor, evals: Tensor) -> Tensor:
    return (evecs * evals) @ evecs.T


def _eigh(m: Tensor) -> tuple[Tensor, Tensor]:
    """Eigenvalues (clamped at zero, as the sum of two PSD matrices) and eigenvectors."""

    evals, evecs = torch.linalg.eigh(0.5 * (m + m.T))

    return evals.clamp(min=0), evecs


def _kron_sum_block(b: str, old: KronEigenbasis, new: KronEigenbasis) -> KronEigenbasis:
    if (old.has_bias, old.weight_shape) != (new.has_bias, new.weight_shape):
        raise StructureError(
            f"KronSum block {b!r} differs: weight {old.weight_shape} "
            f"{'with' if old.has_bias else 'without'} bias in the prior, "
            f"{new.weight_shape} {'with' if new.has_bias else 'without'} bias in the new fit"
        )

    la_o, lg_o, c_o = old.factors
    la_n, lg_n, c_n = new.factors
    total = c_o + c_n

    if not bool(total > 0):
        raise ValueError(f"KronSum block {b!r}: both fits have zero scale, nothing to average")

    # The input factor is averaged, weighted by each fit's count; the output factor adds.
    a = (c_o * _matrix(old.evecs_a, la_o) + c_n * _matrix(new.evecs_a, la_n)) / total
    g = c_o * _matrix(old.evecs_g, lg_o) + c_n * _matrix(new.evecs_g, lg_n)

    la, qa = _eigh(a)
    lg, qg = _eigh(g)
    lg = lg / total

    return KronEigenbasis(
        qa, qg, total * torch.outer(lg, la), new.has_bias, new.weight_shape, (la, lg, total)
    )


def _kron_sum(
    old: Mapping[str, KronEigenbasis], new: Mapping[str, KronEigenbasis]
) -> BlockDiag[KronEigenbasis]:
    """Ritter et al.'s sum, block by block."""

    if old.keys() != new.keys():
        raise StructureError(
            f"KronSum block names differ: {sorted(old)} in the prior, {sorted(new)} in the new fit"
        )

    return BlockDiag({b: _kron_sum_block(b, old[b], new[b]) for b in new})


@pytree
class KronSum:
    r"""A chain of KFAC posteriors as the prior of a KFAC fit, by summing Kronecker factors.

    The prior is $\mathcal{N}(\mu, (\Lambda_0 + S)^{-1})$, with $\Lambda_0$ the precision
    of `base` and $S$ the summed curvature.

    Build it with `KronSum.of(post)`. Approximate (Ritter et al. 2018): the sum of two
    Kronecker products has no Kronecker form, so per layer the sum of an older and a newer
    curvature is $G \otimes A$ with

    $$
    G = c_o G_o + c_n G_n, \qquad A = \frac{c_o A_o + c_n A_n}{c_o + c_n}.
    $$

    Here $c$ is each fit's scale times the likelihood's curvature factor ($1 / \sigma^2$
    for a `Gaussian`); with one $\sigma$ for every fit the weights on $A$ are the example
    counts. $A$ and $G$ are the per-example factors. This is laplace-torch's rule for
    sequential fits.

    The chain is held flat. So `KronSum.of` of a posterior fitted with a `KronSum` prior
    extends the sum, and forming a precision decomposes each factor once, however long the
    chain.

    Both curvatures must be `BlockDiag` of `KronEigenbasis` with Kronecker factors (KFAC,
    not EKFAC) and the same blocks; others raise `StructureError`, for `post` in `of`, for
    the new fit when its precision is formed (in `lk.laplace`, before the data pass when the
    method has a template). Not tuned by default and not pushable to function space.

    Args:
        summed: $S$, the sum of every earlier fit's scaled curvature.
        base: The first prior in the chain that is not a `KronSum`, added once, after the
            sum.
        mean: The last posterior's mean $\mu$.

    Raises:
        StructureError: `summed` is not a `BlockDiag` of `KronEigenbasis` with Kronecker
            factors.
    """

    summed: BlockDiag[KronEigenbasis]
    base: Prior
    mean: Params

    def __post_init__(self) -> None:
        _kron_blocks(self.summed, "the summed curvature")

    @classmethod
    def of(cls, post: LaplacePosterior[Any, Any], /) -> Self:
        r"""The prior for the fit after `post`: its scaled curvature added to the chain.

        With $C$ the curvature of `post` and $c$ its scale times the likelihood's curvature
        factor, the new $S$ is $cC$ when the prior of `post` is not a `KronSum`, and the
        Kronecker sum of the prior's $S$ and $cC$ (as in the class description) when it is.

        Args:
            post: A KFAC posterior.

        Returns:
            The `KronSum`, with `post.mean` as its mean. Its `base` is the prior of `post`,
            or that prior's `base` when it is a `KronSum`.

        Raises:
            StructureError: The curvature of `post` is not a `BlockDiag` of
                `KronEigenbasis` with Kronecker factors, or a block differs from the
                chain's (its name, weight shape or bias).
            ValueError: A block has zero scale in both the chain and `post`.
        """

        blocks = _kron_blocks(post.curvature, "the posterior's curvature")
        scale = _fit_scale(post)
        scaled = {b: blk.scaled(scale) for b, blk in blocks.items()}

        if isinstance(post.prior, KronSum):
            summed = _kron_sum(post.prior.summed.blocks, scaled)
            base = post.prior.base
        else:
            summed = BlockDiag(scaled)
            base = post.prior

        return cls(summed, base, post.mean)

    def __repr__(self) -> str:
        n = len(self.summed.blocks)

        return (
            f"KronSum(summed=<{n} KFAC block{'s' * (n != 1)}>, base={self.base!r}"
            f"{_mean_repr(self.mean)})"
        )

    def log_prob(self, params: Params, /) -> Tensor:
        r"""$\log \mathcal{N}(\theta; \mu, (\Lambda_0 + S)^{-1})$, in the dtype of $S$.

        Args:
            params: Parameter mapping with the posterior's keys and shapes.

        Returns:
            A 0-dim tensor, $\log q(\theta)$.
        """

        dtype = next(iter(self.summed.blocks.values())).evals.dtype

        return _chain_log_prob(self.base, self.summed, self.mean, params, dtype)

    def location(self, like: Params, /) -> Params:
        """The last posterior's mean.

        Args:
            like: Parameter mapping whose names, shapes, dtypes and devices the mean must have.

        Returns:
            The mean, a parameter mapping.
        """

        check_location(self.mean, like)

        return self.mean

    def add_precision_to[Op: Operator[Any]](self, op: Op, /) -> Op:
        r"""The Kronecker sum of $S$ and `op`, plus the precision $\Lambda_0$ of `base`.

        Args:
            op: A precision operator.

        Returns:
            An operator of the same type, $\text{op} + \Lambda_0$.

        Raises:
            StructureError: `op` is not a `BlockDiag` of `KronEigenbasis` with Kronecker
                factors, a block differs from those of $S$ (its name, weight shape or
                bias), or `base` can't add its precision.
            ValueError: A block has zero scale in both $S$ and `op`.
        """

        new = _kron_blocks(op, "the new curvature")

        return cast(Op, self.base.add_precision_to(_kron_sum(self.summed.blocks, new)))


def _low_rank(op: object, what: str, hint: str = "") -> Joint[LowRankPlusDiag]:
    """`op` as a `Joint` of `LowRankPlusDiag` (a `LowRankGGN` curvature)."""

    if isinstance(op, Joint) and isinstance(op.op, LowRankPlusDiag):
        return cast(Joint[LowRankPlusDiag], op)

    got = f"Joint[{type(op.op).__name__}]" if isinstance(op, Joint) else type(op).__name__

    raise StructureError(
        f"LowRankSum needs {what} to be a Joint of LowRankPlusDiag (LowRankGGN), got {got}{hint}"
    )


def _truncated(op: Joint[LowRankPlusDiag], rank: int) -> Joint[LowRankPlusDiag]:
    return copy.replace(op, op=op.op.truncated(rank))


@pytree
class LowRankSum:
    r"""A chain of `LowRankGGN` posteriors as the prior of a `LowRankGGN` fit, cut to a fixed rank.

    The prior is $\mathcal{N}(\mu, (\Lambda_0 + S)^{-1})$, with $\Lambda_0$ the precision
    of `base` and $S$ the summed curvature. Build it with `LowRankSum.of(post)`. Each step
    sums the scaled low-rank curvatures and keeps the top $r$ eigenpairs of the sum:

    $$
    S_t = \operatorname{trunc}_r(S_{t-1} + c_t C_t),
    $$

    with $C_t$ the curvature of fit $t$ and $c_t$ its scale times the likelihood's curvature
    factor. Approximate: the eigenpairs below the top $r$ are dropped, so the prior has less
    precision (more variance) in those directions. A plain `LowRankGGN` posterior as the prior
    is exact, but its width grows by the curvature's rank with every fit.

    The chain is held flat, as in `KronSum`.

    Both curvatures must be a `Joint` of `LowRankPlusDiag` over the same blocks; others
    raise `StructureError`, for `post` in `of`, for the new fit when its precision is formed
    (in `lk.laplace`, before the data pass). Not tuned by default and not pushable to
    function space.

    Args:
        summed: $S$, the truncated sum of every earlier fit's scaled curvature (zero
            diagonal).
        base: The first prior in the chain that is not a `LowRankSum`, added once, after
            the sum.
        mean: The last posterior's mean $\mu$.
        rank: The rank $r$ kept, at least 1.

    Raises:
        StructureError: `summed` is not a `Joint` of `LowRankPlusDiag`.
        ValueError: `rank < 1`.
    """

    summed: Joint[LowRankPlusDiag]
    base: Prior
    mean: Params
    rank: int = static()

    def __post_init__(self) -> None:
        _low_rank(self.summed, "the summed curvature")

        if self.rank < 1:
            raise ValueError(f"LowRankSum needs rank >= 1, got {self.rank}")

    @classmethod
    def of(cls, post: LaplacePosterior[Any, Any], /, rank: int | None = None) -> Self:
        r"""The prior for the fit after `post`: its scaled curvature added to the chain.

        With $C$ the curvature of `post` and $c$ its scale times the likelihood's curvature
        factor, the new $S$ is $\operatorname{trunc}_r(S + cC)$ when the prior of `post` is
        a `LowRankSum` with summed curvature $S$, and $\operatorname{trunc}_r(cC)$
        otherwise.

        Args:
            post: A `LowRankGGN` posterior.
            rank: The rank $r$ kept, at least 1. `None` keeps the rank of the prior of
                `post` when that is a `LowRankSum`, and the width of the curvature of
                `post` otherwise.

        Returns:
            The `LowRankSum`, with `post.mean` as its mean. Its `base` is the prior of
            `post`, or that prior's `base` when it is a `LowRankSum`.

        Raises:
            StructureError: The curvature of `post` is not a `Joint` of `LowRankPlusDiag`
                (the message points a KFAC posterior to `KronSum.of`), or its blocks or
                size differ from the chain's.
            ValueError: `rank < 1`.
        """

        hint = "; for a KFAC posterior, use lk.priors.KronSum.of(post)"
        curvature = _low_rank(post.curvature, "the posterior's curvature", hint)

        if rank is None and isinstance(post.prior, LowRankSum):
            rank = post.prior.rank
        elif rank is None:
            rank = curvature.op.factor.shape[1]

        if rank < 1:
            raise ValueError(f"LowRankSum needs rank >= 1, got {rank}")

        scaled = curvature.scaled(_fit_scale(post))

        if isinstance(post.prior, LowRankSum):
            summed = post.prior.summed.add_same(scaled)
            base = post.prior.base
        else:
            summed = scaled
            base = post.prior

        return cls(_truncated(summed, rank), base, post.mean, rank)

    def __repr__(self) -> str:
        width, n = self.summed.op.factor.shape[1], self.summed.op.factor.shape[0]

        return (
            f"LowRankSum(summed=<rank {width}, {n} parameters>, base={self.base!r}, "
            f"rank={self.rank}{_mean_repr(self.mean)})"
        )

    def log_prob(self, params: Params, /) -> Tensor:
        r"""$\log \mathcal{N}(\theta; \mu, (\Lambda_0 + S)^{-1})$, in the dtype of $S$.

        Args:
            params: Parameter mapping with the posterior's keys and shapes.

        Returns:
            A 0-dim tensor, $\log q(\theta)$.
        """

        return _chain_log_prob(
            self.base, self.summed, self.mean, params, self.summed.op.factor.dtype
        )

    def location(self, like: Params, /) -> Params:
        """The last posterior's mean.

        Args:
            like: Parameter mapping whose names, shapes, dtypes and devices the mean must have.

        Returns:
            The mean, a parameter mapping.
        """

        check_location(self.mean, like)

        return self.mean

    def add_precision_to[Op: Operator[Any]](self, op: Op, /) -> Op:
        r"""$\operatorname{trunc}_r(S + \text{op})$ plus the precision $\Lambda_0$ of `base`.

        Args:
            op: A precision operator.

        Returns:
            An operator of the same type, $\text{op} + \Lambda_0$.

        Raises:
            StructureError: `op` is not a `Joint` of `LowRankPlusDiag`, its blocks or size
                differ from those of $S$, or `base` can't add its precision.
        """

        new = _low_rank(op, "the new curvature")
        total = _truncated(self.summed.add_same(new), self.rank)

        return cast(Op, self.base.add_precision_to(total))
