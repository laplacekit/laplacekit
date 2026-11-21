"""The inputs where a GP method evaluates its kernel.

`fixed` uses given inputs. `from_data` draws them uniformly from the training data, and
`kmeans` takes k-means centres of the training inputs, each moved to its nearest data
row. `grid` and `uniform` place points in a box and need no data.
"""

import copy
import math
from abc import abstractmethod
from collections.abc import Callable, Sequence
from typing import Any, Protocol, Self, cast, runtime_checkable

import torch
from torch import Tensor

from laplacekit.core import (
    Batch,
    Done,
    FitContext,
    Inputs,
    Params,
    Pass,
    Requirements,
    StructureError,
    chain,
    pytree,
    spawn_generator,
    static,
    tree_leaves,
    tree_map,
)
from laplacekit.function_space import is_shared

__all__ = [
    "MAX_KMEANS_ENTRIES",
    "Drawn",
    "Picked",
    "Points",
    "fixed",
    "from_data",
    "grid",
    "kmeans",
    "uniform",
]

MAX_KMEANS_ENTRIES = 2**26
"""The most input entries (rows times entries per row) that the `kmeans` pass may hold."""


@pytree
class Picked:
    """The points a method uses, with what it needs to place outputs on them.

    Args:
        inputs: The $m$ points, along the leading axis.
        u: Uniform draws in $[0, 1)$, `[m]` float64. Methods that draw an output
            coordinate per point use them.
        coords: Output coordinates, `[m]` int64, used in place of `u`. Only `fixed` sets
            them.
        seen: The number of examples the pass read, or 0 when there was no pass.
    """

    inputs: Inputs
    u: Tensor
    coords: Tensor | None
    seen: Tensor


@runtime_checkable
class Points(Protocol):
    """Where a method evaluates its kernel."""

    @abstractmethod
    def pick[R](
        self,
        params: Params,
        ctx: FitContext,
        then: Callable[[Picked], Pass[R] | Done[R]],
        /,
    ) -> Pass[R] | Done[R]:
        """Pick the points and hand them to `then`.

        Points that need no data call `then` directly. The others `chain` a pass over the
        data first.

        Args:
            params: The fitted params. Points move to their device.
            ctx: The fit's context.
            then: The method's next step, which takes the `Picked` points.

        Returns:
            What `then` returns, or a pass over the data chained before `then`.
        """
        ...


@runtime_checkable
class Drawn(Protocol):
    """Points that need no data."""

    @abstractmethod
    def draw(self, *, generator: torch.Generator | None = None) -> Inputs:
        """Draw the points.

        Args:
            generator: The generator to draw with. `None` means torch's global RNG.
                Points that are not random ignore it.

        Returns:
            The points, with the examples along the leading axis.
        """
        ...


# --- helpers --------------------------------------------------------------------------


def _size(inputs: Inputs) -> int | None:
    """The number of examples, read from the leading axis of the first per-example leaf."""

    for leaf in tree_leaves(inputs):
        if not is_shared(leaf):
            return leaf.shape[0]

    return None


def _device(params: Params) -> torch.device:
    p = next(iter(params.values()), None)

    return p.device if p is not None else torch.device("cpu")


def _draw_seed(generator: torch.Generator | None) -> int:
    """One seed drawn from `generator`, or from torch's global RNG when `None`."""

    return spawn_generator(generator).initial_seed()


def _draws(generator: torch.Generator, n: int, device: torch.device) -> tuple[Tensor, Tensor]:
    """Draw priorities and uniforms for a batch of `n` examples from its generator.

    The generator draws the priorities first, then `u`. The priorities are the ones
    `fs.SubsetOfData` draws. The fit seeds each batch's generator with the rank under
    `torch.distributed`, so ranks with equal shards draw differently.
    """

    g = generator
    priority = torch.rand(n, generator=g, dtype=torch.float64, device=g.device)
    u = torch.rand(n, generator=g, dtype=torch.float64, device=g.device)

    return priority.to(device), u.to(device)


@pytree
class _Rows:
    inputs: Inputs  # leading dim k
    priority: Tensor  # [k]
    u: Tensor  # [k]


def _merge(parts: tuple[_Rows, ...], keep: int | None) -> _Rows:
    """Merge rows into one block sorted by priority, keeping the `keep` smallest or all of them."""

    priority = torch.cat([r.priority for r in parts])
    order = torch.argsort(priority, stable=True)

    if keep is not None:
        order = order[:keep]

    def take(*xs: Any) -> Any:
        if is_shared(xs[0]):
            return xs[0]

        return torch.cat(xs)[order.to(xs[0].device)]

    inputs = tree_map(take, *(r.inputs for r in parts))
    u = torch.cat([r.u for r in parts])[order]

    return _Rows(inputs, priority[order], u)


@pytree
class _Ranked:
    """Rows with their priorities, keeping the `keep` smallest, or all rows when `keep` is None.

    It is a monoid whose identity holds no rows. `rows()` sorts by priority, so its result
    does not depend on the order of the batches. It is not a sum, so `scaled` returns it
    unchanged. With `keep`, `+` keeps one sorted block of the `keep` smallest. With
    `keep=None`, as k-means uses it, `+` only collects blocks so that the pass sorts once,
    and it raises `StructureError` above `MAX_KMEANS_ENTRIES`.
    """

    blocks: tuple[_Rows, ...]
    seen: Tensor  # examples seen
    keep: int | None = static()

    @property
    def total_weight(self) -> Tensor:
        return self.seen

    def __add__(self, other: Self, /) -> Self:
        blocks = self.blocks + other.blocks
        seen = self.seen + other.seen

        if self.keep is None:
            _check_budget(sum(_entries(r.inputs) for r in blocks))

            return copy.replace(self, blocks=blocks, seen=seen)

        if not blocks:
            return copy.replace(self, seen=seen)

        return copy.replace(self, blocks=(_merge(blocks, self.keep),), seen=seen)

    def rows(self, what: str, /) -> _Rows:
        """All kept rows in one block, sorted by priority. Raises `ValueError` when it is empty."""

        if not self.blocks:
            raise ValueError(f"{what}: the data yielded no examples")

        return _merge(self.blocks, self.keep)

    def scaled(self, s: Tensor, /) -> Self:
        return self


def _entries(inputs: Inputs) -> int:
    return sum(t.numel() for t in tree_leaves(inputs) if isinstance(t, Tensor))


def _check_budget(entries: int) -> None:
    if entries > MAX_KMEANS_ENTRIES:
        raise StructureError(
            f"kmeans holds {entries} input entries, above "
            f"MAX_KMEANS_ENTRIES={MAX_KMEANS_ENTRIES}; use from_data, or fewer examples"
        )


def _requirements() -> Requirements:
    # Weights would change which rows matter, and the pick ignores them: refuse them.
    return Requirements(additive_stat=False, weights=False)


def _init(params: Params, keep: int | None) -> _Ranked:
    return _Ranked((), torch.zeros((), dtype=torch.int64, device=_device(params)), keep)


def _batch_rows(stat: _Ranked, inputs: Inputs, n: int, generator: torch.Generator) -> _Ranked:
    """Add the batch's rows to `stat`: its `keep` smallest, or all of them."""

    priority, u = _draws(generator, n, stat.seen.device)
    rows = _Rows(inputs, priority, u)

    if stat.keep is not None:
        rows = _merge((rows,), stat.keep)
    count = torch.tensor(n, dtype=torch.int64, device=stat.seen.device)

    return stat + _Ranked((rows,), count, stat.keep)


# --- fixed ----------------------------------------------------------------------------


@pytree
class Fixed:
    """Given inputs, with optional output coordinates. `fixed` builds it and checks them.

    Args:
        inputs: The points, as for `fixed`.
        coords: The output coordinates, as for `fixed`.
    """

    inputs: Inputs
    coords: Tensor | None = None
    _seed: int = static(default_factory=lambda: _draw_seed(None), kw_only=True)

    def __repr__(self) -> str:
        coords = "given" if self.coords is not None else None

        return f"fixed(m={_size(self.inputs)}, coords={coords})"

    def pick[R](
        self,
        params: Params,
        ctx: FitContext,
        then: Callable[[Picked], Pass[R] | Done[R]],
        /,
    ) -> Pass[R] | Done[R]:
        """Hand the inputs to `then` without a pass, with `Picked.u` drawn as `fixed` says.

        Args:
            params: The fitted parameters. The points move to their device.
            ctx: The fit's context.
            then: The method's next step, which takes the `Picked` points.

        Returns:
            What `then` returns, or a pass over the data chained before `then`.
        """

        m = _size(self.inputs) or 0
        device = _device(params)
        gen = torch.Generator().manual_seed(self._seed)
        u = torch.rand(m, generator=gen, dtype=torch.float64).to(device)
        seen = torch.zeros((), dtype=torch.int64, device=device)

        return then(Picked(self.inputs, u, self.coords, seen))

    def draw(self, *, generator: torch.Generator | None = None) -> Inputs:
        """Return the given inputs.

        Args:
            generator: Ignored, since the inputs are given.

        Returns:
            The given inputs.
        """

        return self.inputs


def fixed(
    inputs: Inputs, /, coords: Tensor | None = None, *, generator: torch.Generator | None = None
) -> Fixed:
    """Use the given inputs as the points. The fit reads no data for them.

    Args:
        inputs: The $m$ points along the leading axis. A mapping of tensors also works.
        coords: One output coordinate per point, `[m]` int64. Methods that would draw a
            coordinate use these instead.
        generator: Seeds `Picked.u`. `fixed` draws one seed from it when it is called,
            or from torch's global RNG when `None`, so `torch.manual_seed` makes the draw
            repeatable. Every fit with these points gets the same `u`, and the `generator=`
            of the fit does not change it.

    Returns:
        The points.

    Raises:
        StructureError: `inputs` has no per-example leaf, or `coords` is not int64 of
            shape `[m]`.
    """

    m = _size(inputs)

    if m is None:
        raise StructureError(
            "fixed needs inputs with a leading example axis (a tensor of at least 1 dim)"
        )

    if coords is not None and (coords.dtype != torch.int64 or tuple(coords.shape) != (m,)):
        raise StructureError(
            f"fixed needs coords as an int64 tensor of shape [{m}], got {coords.dtype} of "
            f"shape {list(coords.shape)}"
        )

    return Fixed(inputs, coords, _seed=_draw_seed(generator))


# --- from data ------------------------------------------------------------------------


@pytree
class _FromDataPass:
    n_points: int = static()

    def __repr__(self) -> str:
        return f"from_data({self.n_points})"

    @property
    def requires(self) -> Requirements:
        return _requirements()

    def init(self, params: Params, ctx: FitContext, /) -> _Ranked:
        return _init(params, self.n_points)

    def update(
        self,
        stat: _Ranked,
        params: Params,
        batch: Batch,
        generator: torch.Generator,
        ctx: FitContext,
        /,
    ) -> _Ranked:
        return _batch_rows(stat, batch.inputs, batch.size, generator)

    def finalize(self, stat: _Ranked, /) -> Picked:
        rows = stat.rows("from_data")

        return Picked(rows.inputs, rows.u, None, stat.seen)


@pytree
class FromData:
    """$m$ training inputs drawn uniformly without replacement. `from_data` builds it.

    Args:
        n_points: The number of points $m$.

    Raises:
        ValueError: `n_points` is below 1.
    """

    n_points: int = static()

    def __post_init__(self) -> None:
        if self.n_points < 1:
            raise ValueError(f"from_data needs n_points >= 1, got {self.n_points}")

    def __repr__(self) -> str:
        return f"from_data({self.n_points})"

    def pick[R](
        self,
        params: Params,
        ctx: FitContext,
        then: Callable[[Picked], Pass[R] | Done[R]],
        /,
    ) -> Pass[R] | Done[R]:
        """A pass over the data that picks the points, followed by `then`.

        Args:
            params: The fitted parameters. The points move to their device.
            ctx: The fit's context.
            then: The method's next step, which takes the `Picked` points.

        Returns:
            What `then` returns, or a pass over the data chained before `then`.
        """

        return chain(_FromDataPass(self.n_points), then)


def from_data(n_points: int, /) -> FromData:
    """Pick $m$ training inputs uniformly without replacement, in one pass over the data.

    Every example gets a uniform priority, and the $m$ smallest are kept. Each batch's
    generator draws the priorities and then `Picked.u`. The fit seeds that generator from
    its own seed, the pass's position in the method's chain, the number of examples before
    the batch and, under `torch.distributed`, the rank. The pick is therefore fixed by the
    fit's seed, the data order and batching, the pass position and the rank. When both
    run as the first pass over the same batches with the same seed, it is the subset that
    `fs.SubsetOfData(m)` picks.

    The pass runs no forward pass, and mapping inputs work. Weighted batches raise
    `CapabilityError` during the fit.

    Args:
        n_points: The number of points $m$. When there are fewer training inputs, all of them
            become points.

    Returns:
        The points.

    Raises:
        ValueError: `n_points` is below 1.
    """

    return FromData(n_points)


# --- k-means --------------------------------------------------------------------------


def _check_inputs(inputs: Inputs) -> Tensor:
    if not isinstance(inputs, Tensor) or inputs.ndim == 0 or not inputs.is_floating_point():
        got = (
            f"{inputs.dtype} tensor of {inputs.ndim} dims"
            if isinstance(inputs, Tensor)
            else type(inputs).__name__
        )

        raise StructureError(
            f"kmeans needs tensor inputs: one floating tensor with a leading example axis, "
            f"got {got}; use from_data or fixed for other inputs"
        )

    return inputs


def _dist(x: Tensor, c: Tensor) -> Tensor:
    # Exact differences: the matmul form loses digits for inputs far from the origin.
    return torch.cdist(x, c, compute_mode="donot_use_mm_for_euclid_dist")


def _lloyd(x: Tensor, m: int, iters: int) -> Tensor:
    """Run Lloyd's algorithm on x, `[N, D]`, starting from its first m rows.

    An empty cluster keeps its centre. Returns the index of the row nearest to each final
    centre.
    """

    c = x[:m].clone()

    for _ in range(iters):
        assign = _dist(x, c).argmin(1)
        sums = torch.zeros_like(c).index_add_(0, assign, x)
        counts = torch.bincount(assign, minlength=m).to(c.dtype).unsqueeze(1)
        c = torch.where(counts > 0, sums / counts.clamp(min=1), c)

    return _dist(x, c).argmin(0)


@pytree
class _KMeansPass:
    n_points: int = static()
    n_iters: int = static()

    def __repr__(self) -> str:
        return f"kmeans({self.n_points}, n_iters={self.n_iters})"

    @property
    def requires(self) -> Requirements:
        return _requirements()

    def init(self, params: Params, ctx: FitContext, /) -> _Ranked:
        return _init(params, None)

    def update(
        self,
        stat: _Ranked,
        params: Params,
        batch: Batch,
        generator: torch.Generator,
        ctx: FitContext,
        /,
    ) -> _Ranked:
        inputs = _check_inputs(batch.inputs)

        return _batch_rows(stat, inputs, batch.size, generator)

    def finalize(self, stat: _Ranked, /) -> Picked:
        rows = stat.rows("kmeans")
        x = cast(Tensor, rows.inputs)  # `update` checked it
        n = x.shape[0]

        if n <= self.n_points:
            return Picked(x, rows.u, None, stat.seen)

        snap = _lloyd(x.reshape(n, -1), self.n_points, self.n_iters)

        return Picked(x[snap], rows.u[snap], None, stat.seen)


@pytree
class KMeans:
    """k-means centres of the training inputs, each moved to a data row. `kmeans` builds it.

    Args:
        n_points: The number of points $m$.
        n_iters: The number of Lloyd iterations.

    Raises:
        ValueError: `n_points` is below 1 or `n_iters` is negative.
    """

    n_points: int = static()
    n_iters: int = static(default=20)

    def __post_init__(self) -> None:
        if self.n_points < 1:
            raise ValueError(f"kmeans needs n_points >= 1, got {self.n_points}")

        if self.n_iters < 0:
            raise ValueError(f"kmeans needs n_iters >= 0, got {self.n_iters}")

    def __repr__(self) -> str:
        return f"kmeans({self.n_points}, n_iters={self.n_iters})"

    def pick[R](
        self,
        params: Params,
        ctx: FitContext,
        then: Callable[[Picked], Pass[R] | Done[R]],
        /,
    ) -> Pass[R] | Done[R]:
        """A pass that collects the rows and runs k-means, followed by `then`.

        Args:
            params: The fitted parameters. The points move to their device.
            ctx: The fit's context.
            then: The method's next step, which takes the `Picked` points.

        Returns:
            What `then` returns, or a pass over the data chained before `then`.
        """

        return chain(_KMeansPass(self.n_points, self.n_iters), then)


def kmeans(n_points: int, /, n_iters: int = 20) -> KMeans:
    """Pick points by k-means on the training inputs.

    One pass collects every row, each with a priority drawn as in `from_data`. Lloyd's
    algorithm starts from the $m$ rows of smallest priority and runs `n_iters` iterations,
    and an empty cluster keeps its centre. Each centre then moves to its nearest row, as
    in laplace-torch, which runs `scipy`'s `kmeans2` with other random draws. As with
    `from_data`, the fit's seed, the data order and batching, the pass position and the rank
    fix the result. On CUDA the centre sums use atomic adds, so two runs can differ
    slightly.

    The inputs must be one floating-point tensor. Distances use the rows flattened after
    the first axis. Other inputs raise `StructureError` during the fit, and so does a pass
    that holds more than `MAX_KMEANS_ENTRIES` input entries. Weighted batches raise
    `CapabilityError`.

    Args:
        n_points: The number of points $m$. When there are $m$ rows or fewer, every row becomes
            a point.
        n_iters: The number of Lloyd iterations.

    Returns:
        The points.

    Raises:
        ValueError: `n_points` is below 1 or `n_iters` is negative.
    """

    return KMeans(n_points, n_iters)


# --- grid and uniform ------------------------------------------------------------------


def _box(low: Tensor | float, high: Tensor | float, what: str) -> tuple[Tensor, Tensor]:
    """`low` and `high` as `[D]` tensors of one floating-point dtype, with D taken from `low`."""

    tensors = [t for t in (low, high) if isinstance(t, Tensor)]
    dtype = torch.get_default_dtype()

    for i, t in enumerate(t for t in tensors if t.is_floating_point()):
        dtype = t.dtype if i == 0 else torch.promote_types(dtype, t.dtype)

    device = tensors[0].device if tensors else None
    lo = torch.as_tensor(low, dtype=dtype, device=device)

    if lo.ndim > 1:
        raise StructureError(
            f"{what} needs low as a float, a 0-dim or a 1-dim tensor, got shape {list(lo.shape)}"
        )

    lo = lo.reshape(-1)
    hi = torch.as_tensor(high, dtype=dtype, device=device)

    if hi.ndim > 1 or hi.numel() not in (1, lo.numel()):
        raise StructureError(
            f"{what} needs high as a float or a tensor of {lo.numel()} entries (one per "
            f"axis of low), got shape {list(hi.shape)}"
        )

    hi = hi.reshape(-1).expand_as(lo)

    if not bool((hi >= lo).all()):
        raise ValueError(f"{what} needs high >= low on every axis, got low {lo}, high {hi}")

    return lo, hi


def _no_pass[R](
    inputs: Tensor,
    generator: torch.Generator,
    params: Params,
    then: Callable[[Picked], Pass[R] | Done[R]],
) -> Pass[R] | Done[R]:
    """Call `then` on `inputs`, moved to the parameters' device, with `u` from `generator`."""

    device = _device(params)
    m = inputs.shape[0]
    u = torch.rand(m, generator=generator, dtype=torch.float64).to(device)
    seen = torch.zeros((), dtype=torch.int64, device=device)

    return then(Picked(inputs.to(device), u, None, seen))


@pytree
class Grid:
    """The product grid over a box. `grid` builds it and checks the box.

    Args:
        low: The lower corner, `[D]`.
        high: The upper corner, `[D]`.
        n: The number of points on each of the $D$ axes.
    """

    low: Tensor  # [D]
    high: Tensor  # [D]
    n: tuple[int, ...] = static()

    def __repr__(self) -> str:
        return f"grid(D={len(self.n)}, n={self.n})"

    def pick[R](
        self,
        params: Params,
        ctx: FitContext,
        then: Callable[[Picked], Pass[R] | Done[R]],
        /,
    ) -> Pass[R] | Done[R]:
        """Hand the grid to `then` without a pass.

        `Picked.u` is drawn from a CPU generator seeded with 0, so it is the same for
        every fit.

        Args:
            params: The fitted parameters. The points move to their device.
            ctx: The fit's context.
            then: The method's next step, which takes the `Picked` points.

        Returns:
            What `then` returns, or a pass over the data chained before `then`.
        """

        return _no_pass(self.draw(), torch.Generator().manual_seed(0), params, then)

    def draw(self, *, generator: torch.Generator | None = None) -> Tensor:
        """Return the grid, `[prod(n), D]`.

        Args:
            generator: Ignored, since the grid is fixed.

        Returns:
            The grid points, `[prod(n), D]`.
        """

        axes = [
            torch.linspace(float(a), float(b), k, dtype=self.low.dtype, device=self.low.device)
            for a, b, k in zip(self.low, self.high, self.n, strict=True)
        ]

        return torch.cartesian_prod(*axes).reshape(-1, len(axes))


def grid(low: Tensor | float, high: Tensor | float, n: int | Sequence[int]) -> Grid:
    """Place points on the product grid of `torch.linspace(low[d], high[d], n[d])`.

    The points are `[prod(n), D]`, and the last axis varies fastest. Their dtype is the
    promoted dtype of the floating-point tensors given, or the default dtype if there are
    none. `draw` ignores its generator. `pick` moves the points to the parameters' device
    and draws `Picked.u` from a CPU generator seeded with 0. The fit reads no data for a
    grid, so data weights play no part.

    Args:
        low: The lower corner. $D$ is 1 for a float or a 0-dim tensor, else its length.
        high: The upper corner, as a float, a one-entry tensor, or one value per axis.
        n: The number of points per axis. An int applies to every axis.

    Returns:
        The grid.

    Raises:
        StructureError: `low` has more than 1 dim, `high` has more than 1 dim or neither 1
            nor $D$ entries, or `n` does not give one count per axis.
        ValueError: A count is below 1, or `high` is not at least `low` on every axis.
    """

    lo, hi = _box(low, high, "grid")
    counts = (n,) * lo.numel() if isinstance(n, int) else tuple(n)

    if len(counts) != lo.numel():
        raise StructureError(
            f"grid needs one count per axis: low has {lo.numel()} axes, n has {len(counts)}"
        )

    if any(k < 1 for k in counts):
        raise ValueError(f"grid needs n >= 1 on every axis, got {counts}")

    return Grid(lo, hi, counts)


@pytree
class Uniform:
    """$m$ uniform points in a box, new ones on each `draw`. `uniform` builds it.

    Args:
        low: The lower corner, `[D]`.
        high: The upper corner, `[D]`.
        n_points: The number of points $m$.
    """

    low: Tensor  # [D]
    high: Tensor  # [D]
    n_points: int = static()
    _seed: int = static(default_factory=lambda: _draw_seed(None), kw_only=True)

    def __repr__(self) -> str:
        return f"uniform(D={self.low.numel()}, n_points={self.n_points})"

    def pick[R](
        self,
        params: Params,
        ctx: FitContext,
        then: Callable[[Picked], Pass[R] | Done[R]],
        /,
    ) -> Pass[R] | Done[R]:
        """Hand points drawn as `uniform` says to `then` without a pass.

        Args:
            params: The fitted parameters. The points move to their device.
            ctx: The fit's context.
            then: The method's next step, which takes the `Picked` points.

        Returns:
            What `then` returns, or a pass over the data chained before `then`.
        """

        gen = torch.Generator().manual_seed(self._seed)

        return _no_pass(self.draw(generator=gen), gen, params, then)

    def draw(self, *, generator: torch.Generator | None = None) -> Tensor:
        """Draw new points, `[m, D]`.

        Args:
            generator: The generator to draw with, on any device. `None` means torch's
                global RNG.

        Returns:
            The points, `[m, D]`, on the box's device.
        """

        shape = (self.n_points, self.low.numel())
        device = generator.device if generator is not None else self.low.device
        r = torch.rand(shape, generator=generator, dtype=self.low.dtype, device=device)

        return self.low + (self.high - self.low) * r.to(self.low.device)


def uniform(
    low: Tensor | float,
    high: Tensor | float,
    n_points: int,
    *,
    generator: torch.Generator | None = None,
) -> Uniform:
    r"""Draw $m$ points uniformly from the box $[\text{low}, \text{high}]$, as `[m, D]`.

    `draw(generator=g)` draws uniforms $r$, `[m, D]`, and returns
    $\text{low} + (\text{high} - \text{low}) \, r$. Each call gives new points, the way
    FSP-Laplace draws new context points at every training step. `pick` draws the points
    and then `Picked.u` from a CPU generator seeded once, when `uniform` is called, so
    every fit with these points gets the same ones. It moves the points to the
    parameters' device. The fit reads no data for these
    points, so data weights play no part. The dtype follows the rule of `grid`.

    Args:
        low: The lower corner, as for `grid`.
        high: The upper corner, as for `grid`.
        n_points: The number of points $m$.
        generator: Seeds the points `pick` uses. `uniform` draws one seed from it when
            it is called, or from torch's global RNG when `None`, so `torch.manual_seed`
            makes them repeatable. The `generator=` of the fit does not change them.

    Returns:
        The points.

    Raises:
        ValueError: `n_points` is below 1, or `high` is not at least `low` on every axis.
        StructureError: `low` or `high` has a shape `grid` refuses.
    """

    if n_points < 1:
        raise ValueError(f"uniform needs n_points >= 1, got {n_points}")

    lo, hi = _box(low, high, "uniform")

    return Uniform(lo, hi, n_points, _seed=_draw_seed(generator))


# --- shared checks ----------------------------------------------------------------------


def point_count(points: Points, /) -> int | None:
    """The most points a built-in `Points` gives, or None when that is unknown before the data."""

    if isinstance(points, Fixed):
        return _size(points.inputs)

    if isinstance(points, FromData | KMeans | Uniform):
        return points.n_points

    if isinstance(points, Grid):
        return math.prod(points.n)

    return None


def check_coords(coords: Tensor, event: int, what: str, /) -> None:
    """Raise `StructureError` when an output coordinate lies outside [0, E).

    A coordinate out of range would select no output, and its point would silently drop out.
    """

    bad = ((coords < 0) | (coords >= event)).nonzero()

    if bad.numel():
        i = int(bad[0, 0])

        raise StructureError(
            f"{what} got coordinate {int(coords[i])} for point {i}, outside [0, E) with "
            f"E={event} output entries per example; coordinates index the flattened outputs"
        )
