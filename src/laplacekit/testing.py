"""Conformance checks for plugin authors.

They cover curvatures, priors, likelihoods, links, subsets, objectives and predictives.
Call them from your own tests:

    lk.testing.check_curvature(MyCurvature(), model, likelihood, params, batches)
    lk.testing.check_prior(MyPrior(prior_precision), params)
    lk.testing.check_likelihood(MyLikelihood(), f, y)
    lk.testing.check_link(MyLink(), likelihood, outputs)
    lk.testing.check_subset(my_subset, model, caps)
    lk.testing.check_objective(my_objective, posterior, where)
    lk.testing.check_predictive(MyPredictive(), posterior, inputs)

Each raises `AssertionError` naming the broken law and the object, e.g.
"monoid law: fold(a ++ b) != fold(a) + fold(b) for MyCurvature (max rel err 3e-02)".
The curvature and prior laws run in float64: pass float64 params and a float64 model
(`model.double()`). Checks that build dense matrices run only for at most
`DENSE_MAX_PARAMS` parameters.
"""

import inspect
import types
from collections.abc import Callable, Mapping, Sequence
from typing import Any, cast

import torch
from torch import Tensor, nn
from torch.distributions import Categorical, Distribution

import laplacekit.core as core
from laplacekit.core import (
    MAX_DENSE_EVENT,
    Batch,
    BlockDiag,
    Capabilities,
    CapabilityError,
    Curvature,
    Dense,
    Diag,
    Done,
    FitContext,
    HasBlocks,
    HasDistribution,
    HasOutputHessian,
    HasTemplate,
    HessianApprox,
    Inputs,
    Joint,
    KronEigenbasis,
    LateLikelihood,
    Layout,
    Likelihood,
    LinkApprox,
    LowRankPlusDiag,
    ModelFn,
    Objective,
    Operator,
    OutputGaussian,
    Outputs,
    Params,
    Pass,
    Predictive,
    PriorWithCovariance,
    Requirements,
    Selection,
    Staged,
    Stat,
    StructureError,
    Subset,
    Targets,
    TunablePrior,
    chain,
    generator_for_batch,
    spawn_generator,
    tree_leaves,
    tree_map,
)
from laplacekit.core import check_likelihood as check_requirements

__all__ = [
    "DENSE_MAX_PARAMS",
    "RTOL",
    "check_curvature",
    "check_likelihood",
    "check_link",
    "check_objective",
    "check_predictive",
    "check_prior",
    "check_subset",
]

RTOL = 1e-6
r"""Relative tolerance of every law, measured against the largest entry of the reference.

A result $a$ matches its reference $b$ when
$\max_i |a_i - b_i| \le \mathrm{RTOL} \cdot \max_i |b_i|$, or $\max_i |a_i| \le \mathrm{RTOL}$
when $b$ is zero. Different shapes never match.
"""

DENSE_MAX_PARAMS = 512
"""Dense reconstructions run only up to this many parameters."""

_WEIGHT_FACTOR = 3.0
_REFUSED = (CapabilityError, StructureError)


# --- shared helpers ---------------------------------------------------------------------


def _label(obj: object) -> str:
    """A function's qualified name without `<locals>` (`last_layers.subset`), else the type's."""

    if isinstance(obj, types.FunctionType):
        return obj.__qualname__.replace("<locals>.", "")

    return type(obj).__name__


def _fail(law: str, detail: str, obj: object, err: float | None = None) -> AssertionError:
    tail = "" if err is None else f" (max rel err {err:.1e})"

    return AssertionError(f"{law}: {detail} for {_label(obj)}{tail}")


def _rel_err(got: Tensor, want: Tensor) -> float:
    """Largest absolute difference over the largest entry of `want`. Inf on a shape mismatch."""

    if got.shape != want.shape:
        return float("inf")

    if want.numel() == 0:
        return 0.0

    diff = float((got - want).abs().max())
    scale = float(want.abs().max())

    return diff / scale if scale > 0 else diff


def _tree_err(got: Any, want: Any) -> float:
    """`_rel_err` over matching pytree leaves. Non-tensor leaves must be equal."""

    a, b = tree_leaves(got), tree_leaves(want)

    if len(a) != len(b):
        return float("inf")

    errs = [
        _rel_err(x, y)
        if isinstance(x, Tensor) and isinstance(y, Tensor)
        else (0.0 if x == y else float("inf"))
        for x, y in zip(a, b, strict=True)
    ]

    return max(errs, default=0.0)


def _same_structure(a: Any, b: Any) -> bool:
    """Same number of leaves, and tensor leaves of the same shapes."""

    xs, ys = tree_leaves(a), tree_leaves(b)

    if len(xs) != len(ys):
        return False

    return all(
        x.shape == y.shape if isinstance(x, Tensor) and isinstance(y, Tensor) else True
        for x, y in zip(xs, ys, strict=True)
    )


def _check_float64(params: Params, what: str) -> None:
    wrong = [f"{n} is {p.dtype}" for n, p in params.items() if p.dtype != torch.float64]

    if wrong:
        raise ValueError(
            f"{what} runs its laws in float64; pass float64 params and model "
            f"(model.double()): {', '.join(wrong)}"
        )


def _n_params(params: Params) -> int:
    return sum(p.numel() for p in params.values())


def _inner(op: HasBlocks) -> tuple[tuple[str, type], ...] | type | None:
    """The operator types inside a per-block operator: per block, or the joint one."""

    if isinstance(op, BlockDiag):
        return tuple((k, type(b)) for k, b in cast(BlockDiag[Any], op).blocks.items())

    if isinstance(op, Joint):
        return type(cast(Joint[Any], op).op)

    return None


def _name(op: object) -> str:
    """`Diag`, `BlockDiag[Diag]` for a block operator with one block type, `Joint[Dense]`."""

    if isinstance(op, HasBlocks):
        inner = _inner(op)

        if isinstance(inner, tuple):
            names = sorted({t.__name__ for _, t in inner})
            return f"{type(op).__name__}[{' | '.join(names)}]"

        if inner is not None:
            return f"{type(op).__name__}[{inner.__name__}]"

    return type(op).__name__


def _structure(op: object) -> object:
    if isinstance(op, HasBlocks):
        return (type(op), tuple(op.block_sizes().items()), _inner(op))

    return type(op)


class _Flat:
    """Moves an operator's vectors to and from one flat tensor.

    A per-block operator (`HasBlocks`) takes mappings keyed by block, flattened in layout
    order. Any other operator takes a tensor over all parameters.
    """

    def __init__(self, op: object, layout: Layout, sizes: Mapping[str, int]) -> None:
        self.blocks = isinstance(op, HasBlocks)
        self.layout = layout
        self.sizes = sizes
        self.n = sum(sizes.values())

    def flat(self, v: Any) -> Tensor:
        if self.blocks:
            m = cast(Mapping[str, Tensor], v)

            return torch.cat([m[b].reshape(-1) for b in self.sizes])

        return cast(Tensor, v).reshape(-1)

    def unflat(self, x: Tensor) -> Any:
        if not self.blocks:
            return x

        parts = torch.split(x, list(self.sizes.values()))

        return dict(zip(self.sizes, parts, strict=True))

    def params(self, v: Params) -> Tensor:
        """Params-shaped `v` as this operator's flat vector."""

        if self.blocks:
            return torch.cat([self.layout.flatten(v, b) for b in self.sizes])

        return torch.cat([x.reshape(-1) for x in v.values()])

    def dense(self, op: Operator[Any], like: Tensor) -> Tensor:
        """The matrix of `op`, column by column from `mv`."""

        eye = torch.eye(self.n, dtype=like.dtype, device=like.device)

        return torch.stack([self.flat(op.mv(self.unflat(eye[i]))) for i in range(self.n)], dim=1)


def _flat_for(op: object, layout: Layout) -> _Flat:
    if isinstance(op, HasBlocks):
        return _Flat(op, layout, {b: layout.size(b) for b in layout.blocks})

    # A single-block operator is flat over the params in mapping order (as `priors.Diagonal`
    # assumes); `params()` uses that order, so it must agree with the layout's.
    names = [n for ns in layout.blocks.values() for n in ns]

    if names != list(layout.shapes):
        raise ValueError(
            f"{type(op).__name__} acts on all params at once, but the params are not grouped "
            f"by module ({list(layout.shapes)}), so its flat order is ambiguous; pass them "
            "grouped by module"
        )

    return _Flat(op, layout, {"": sum(layout.size(b) for b in layout.blocks)})


# --- curvatures ---------------------------------------------------------------------------


def _counts(batches: Sequence[Batch]) -> list[int]:
    """The examples before each batch: its place in the pass."""

    counts: list[int] = []
    count = 0

    for b in batches:
        counts.append(count)
        count += b.size

    return counts


def _fold(
    curv: Curvature[Any, Any],
    params: Params,
    ctx: FitContext,
    batches: Sequence[Batch],
    pass_gen: torch.Generator,
    counts: Sequence[int],
) -> Any:
    stat = curv.init(params, ctx)
    p = next(iter(params.values()), None)
    device = p.device if p is not None else None

    # A fresh generator per update: each fold sees the same draws for the same batch.
    for batch, n in zip(batches, counts, strict=True):
        gen = generator_for_batch(pass_gen, n, device=device)
        stat = curv.update(stat, params, batch, gen, ctx)

    return stat


def _check_pass(
    curv: Curvature[Any, Any],
    params: Params,
    ctx: FitContext,
    batches: Sequence[Batch],
    pass_gen: torch.Generator,
    counts: Sequence[int],
    additive: bool,
) -> Any:
    """The statistic laws for one pass. Returns the fold over all batches."""

    identity = curv.init(params, ctx)

    if not isinstance(identity, Stat):
        raise _fail("stat protocol", "init must return a Stat (total_weight, +, scaled)", curv)

    if not isinstance(identity.total_weight, Tensor):
        raise _fail("stat protocol", "total_weight must be a Tensor", curv)

    err = _tree_err(identity + identity, identity)

    if err > RTOL:
        raise _fail("monoid identity", "init + init != init", curv, err)

    split = len(batches) // 2
    whole = _fold(curv, params, ctx, batches, pass_gen, counts)
    left = _fold(curv, params, ctx, batches[:split], pass_gen, counts[:split])
    right = _fold(curv, params, ctx, batches[split:], pass_gen, counts[split:])
    err = _tree_err(left + right, whole)

    if err > RTOL:
        raise _fail("monoid law", "fold(a ++ b) != fold(a) + fold(b)", curv, err)

    err = _tree_err(identity + whole, whole)

    if err > RTOL:
        raise _fail("monoid identity", "init + fold(a) != fold(a)", curv, err)

    two = torch.tensor(2.0, dtype=torch.float64)
    doubled = whole.scaled(two)

    if not _same_structure(doubled, whole):
        raise _fail("scaled", "scaled(2) changed the statistic's structure", curv)

    if additive:
        err = _tree_err(doubled, whole + whole)

        if err > RTOL:
            raise _fail("scaled", "scaled(2) != stat + stat for an additive_stat", curv, err)

    return whole


def _run(
    approx: HessianApprox[Any],
    params: Params,
    ctx: FitContext,
    batches: Sequence[Batch],
    base: torch.Generator,
    laws: bool,
) -> Any:
    """Run every pass of `approx`, checking the statistic laws of each when `laws` is set."""

    step: Pass[Any] | Done[Any] = (
        approx.plan(params, ctx) if isinstance(approx, Staged) else chain(approx, then=Done)
    )
    index = 0

    while isinstance(step, Pass):
        curv = step.curvature

        if not isinstance(curv, Curvature):
            raise _fail("protocol", f"pass {index} is not a Curvature", approx)

        check_requirements(curv.requires, ctx.likelihood, f"pass {index} of {approx!r}")
        # Pure in `base`: every run of the passes sees the same draws.
        pass_gen = generator_for_batch(base, index)
        counts = _counts(batches)
        stat = (
            _check_pass(curv, params, ctx, batches, pass_gen, counts, curv.requires.additive_stat)
            if laws
            else _fold(curv, params, ctx, batches, pass_gen, counts)
        )
        step = step.then(curv.finalize(stat))
        index += 1

    return step.result


def _check_template(curv: HasTemplate, op: Operator[Any], params: Params, ctx: FitContext) -> None:
    template = curv.template(params, ctx)

    if type(template) is not type(op):
        raise _fail("template", f"the template is {_name(template)}, the result {_name(op)}", curv)

    # Raises on a different node type, static field, block key or number of children.
    try:
        tree_map(lambda a, b: None, template, op)
    except ValueError as e:
        raise _fail("template", f"its structure differs from the result's ({e})", curv) from e

    def describe(x: object) -> object:
        return (tuple(x.shape), x.dtype) if isinstance(x, Tensor) else x

    pairs = zip(tree_leaves(template), tree_leaves(op), strict=True)

    for i, (a, b) in enumerate(pairs):
        if describe(a) != describe(b):
            raise _fail(
                "template",
                f"leaf {i} of {_name(op)} is {describe(a)} in the template, "
                f"{describe(b)} in the result",
                curv,
            )


def _check_layout(op: Operator[Any], approx: object, layout: Layout) -> None:
    if isinstance(op, HasBlocks):
        sizes = op.block_sizes()

        if list(sizes) != list(layout.blocks):
            raise _fail(
                "layout",
                f"{_name(op)} keys {list(sizes)} differ from layout blocks {list(layout.blocks)}",
                approx,
            )

        for b, n in sizes.items():
            if n != layout.size(b):
                raise _fail(
                    "layout", f"block {b!r} has {n} entries, the layout {layout.size(b)}", approx
                )

        return

    n = cast(Tensor, op.diagonal()).numel()
    total = sum(layout.size(b) for b in layout.blocks)

    if n != total:
        raise _fail("layout", f"operator has {n} entries, the params {total}", approx)


def _probe_err(a: Operator[Any], b: Operator[Any], flat: _Flat, like: Tensor) -> float:
    """How far apart two operators are on their diagonals and on two random vectors."""

    g = torch.Generator().manual_seed(0)
    vs = [torch.randn(flat.n, generator=g, dtype=like.dtype).to(like.device) for _ in range(2)]
    errs = [_rel_err(flat.flat(a.diagonal()), flat.flat(b.diagonal()))]
    errs += [_rel_err(flat.flat(a.mv(flat.unflat(v))), flat.flat(b.mv(flat.unflat(v)))) for v in vs]

    return max(errs)


def _reweighted(batches: Sequence[Batch], factor: float) -> list[Batch]:
    out: list[Batch] = []

    for b in batches:
        w = (
            torch.ones(b.size, dtype=torch.float64)
            if b.weights is None
            else b.weights.to(torch.float64)
        )
        out.append(Batch(b.inputs, b.targets, factor * w))

    return out


def _check_dense(op: Operator[Any], approx: object, flat: _Flat, like: Tensor) -> None:
    dense = flat.dense(op, like)
    err = _rel_err(flat.flat(op.diagonal()), dense.diagonal())

    if err > RTOL:
        raise _fail("diagonal", "diagonal() differs from the diagonal of mv", approx, err)

    g = torch.Generator().manual_seed(1)
    v = torch.randn(flat.n, generator=g, dtype=like.dtype).to(like.device)
    err = _rel_err(flat.flat(op.mv(flat.unflat(v))), dense @ v)

    if err > RTOL:
        raise _fail("mv", "mv is not linear", approx, err)

    one = torch.ones((), dtype=like.dtype, device=like.device)
    eye = torch.eye(flat.n, dtype=like.dtype, device=like.device)
    want = cast(Tensor, torch.linalg.slogdet(dense + eye)[1])
    err = _rel_err(op.add_isotropic(one).logdet(), want)

    if err > RTOL:
        raise _fail(
            "logdet", "add_isotropic(1).logdet() differs from the dense logdet", approx, err
        )

    want_inv = torch.linalg.inv(dense + eye).diagonal()
    err = _rel_err(flat.flat(op.add_isotropic(one).inverse_diagonal()), want_inv)

    if err > RTOL:
        raise _fail(
            "inverse_diagonal",
            "add_isotropic(1).inverse_diagonal() differs from the dense inverse",
            approx,
            err,
        )


def check_curvature(
    curv: HessianApprox[Any],
    model: nn.Module | ModelFn,
    likelihood: Likelihood,
    params: Params,
    batches: Sequence[Batch],
    *,
    generator: torch.Generator | None = None,
) -> None:
    r"""Check a curvature (`Curvature` or `Staged`) against the core contracts.

    `curv` must satisfy `Curvature` or `Staged`, and `requires` must be a `Requirements`.
    Write $\mathrm{fold}(a)$ for the statistic after `init` and one `update` per batch of
    $a$, and $a \mathbin{+\!\!+} b$ for the batches of $a$ followed by those of $b$. On
    every pass (one for a `Curvature`):

    - `init` returns a `Stat` with a tensor `total_weight`, and it is the identity:
      $\mathrm{init} + \mathrm{init} = \mathrm{init}$ and
      $\mathrm{init} + \mathrm{fold}(a) = \mathrm{fold}(a)$.
    - Monoid law: $\mathrm{fold}(a \mathbin{+\!\!+} b) = \mathrm{fold}(a) + \mathrm{fold}(b)$,
      with $a$ the first half of `batches` (rounded down) and $b$ the rest. Each batch
      gets a generator with the same seed in both folds.
    - `scaled(2)` keeps the statistic's structure and, when `requires.additive_stat` is
      True (the statistic is a sum), equals $s + s$ for the statistic $s$.

    On the result of the last pass:

    - It is an `Operator`. A per-block operator (`HasBlocks`, such as `BlockDiag` or
      `Joint`) is keyed by the layout's blocks in layout order, each with the block's size;
      any other operator covers all params.
    - Template: when `curv` has one (`HasTemplate`), it has the result's structure: the
      same operator types, blocks and static fields, and leaves of the same shapes and
      dtypes.
    - Weight scaling: multiplying every example weight by 3 leaves the result unchanged
      (`finalize` returns averages), compared on the diagonal and on two random vectors.
      Skipped when `requires.weights` is False.
    - Dense checks, only for at most `DENSE_MAX_PARAMS` parameters, with $A$ the matrix
      `mv` builds column by column: `diagonal()` is $\operatorname{diag}(A)$, `mv` is
      linear ($\mathrm{mv}(v) = A v$ for a random $v$), and `add_isotropic(1)` has the
      log-determinant $\log\det(A + I)$ and the inverse diagonal
      $\operatorname{diag}\big((A + I)^{-1}\big)$. The identity is added because
      curvatures are often singular.

    Every equality holds up to `RTOL`. Errors that `curv` itself raises propagate.

    Args:
        curv: The approximation under test.
        model: The model, as a module or a `ModelFn`. It must be float64.
        likelihood: The likelihood the model is fitted with.
        params: Float64 parameter mapping. The layout is `Layout.by_module(params)`. An
            operator that is not `HasBlocks` covers all params in mapping order, so such a
            result needs params grouped by module.
        batches: At least two batches. The capabilities come from `probe` on the first.
        generator: Seeds the passes. One seed is drawn from it, or from torch's global
            RNG when `None` (`lk.core.spawn_generator`). Each batch's update generator is
            `lk.core.generator_for_batch` of its pass's generator and the examples before
            it, so both folds of the monoid law see the same draws for a batch.

    Raises:
        AssertionError: A law above fails, or `curv`, a pass, `requires` or the result is
            not of the type it must be. The message names the law and `curv`.
        ValueError: `params` are not float64, there are fewer than two batches, or the
            result is not `HasBlocks` and the params are not grouped by module. Also raised
            by `probe` when the likelihood's `log_prob` refuses the first batch's targets.
        CapabilityError: The model or the likelihood lacks what `requires` asks for, as
            in a fit. The capabilities are checked once, on `curv.requires`; each pass's
            likelihood needs are checked the same way.
        StructureError: `probe` refuses the model or the likelihood on the first batch
            (see `lk.core.probe`).
    """

    base = spawn_generator(generator)
    _check_float64(params, "check_curvature")

    if len(batches) < 2:
        raise ValueError("check_curvature needs at least two batches to split for the monoid law")

    if not isinstance(curv, Staged | Curvature):
        raise _fail("protocol", "satisfies neither Curvature nor Staged", curv)

    requires: object = curv.requires

    if not isinstance(requires, Requirements):
        raise _fail("requires", f"requires is {type(requires).__name__}, not Requirements", curv)

    fn = ModelFn.from_module(model) if isinstance(model, nn.Module) else model
    caps = core.probe(fn, likelihood, batches[0])
    caps.check(requires, repr(curv))
    check_requirements(requires, likelihood, repr(curv))
    layout = Layout.by_module(params)
    ctx = FitContext(fn, likelihood, layout, caps)
    like = next(iter(params.values()))

    op = _run(curv, params, ctx, batches, base, laws=True)

    if not isinstance(op, Operator):
        raise _fail(
            "operator protocol", f"the result is {type(op).__name__}, not an Operator", curv
        )

    op = cast(Operator[Any], op)
    _check_layout(op, curv, layout)

    if isinstance(curv, HasTemplate):
        _check_template(curv, op, params, ctx)

    flat = _flat_for(op, layout)

    if requires.weights:
        unit = _run(curv, params, ctx, _reweighted(batches, 1.0), base, laws=False)
        scaled = _run(curv, params, ctx, _reweighted(batches, _WEIGHT_FACTOR), base, laws=False)
        err = _probe_err(scaled, unit, flat, like)

        if err > RTOL:
            raise _fail(
                "weight scaling",
                f"scaling every weight by {_WEIGHT_FACTOR:g} changed the result "
                "(finalize must return averages)",
                curv,
                err,
            )

    if flat.n <= DENSE_MAX_PARAMS:
        _check_dense(op, curv, flat, like)


# --- priors -------------------------------------------------------------------------------


def _test_operators(layout: Layout, like: Tensor, dense: bool) -> list[Operator[Any]]:
    """Diagonal and rank-one-plus-diagonal test operators, single and per block.

    When small enough, also Kronecker and dense ones. The Kronecker one refuses a diagonal
    prior.
    """

    def ones(n: int) -> Tensor:
        return torch.ones(n, dtype=like.dtype, device=like.device)

    def two_eye(n: int) -> Tensor:
        return 2 * torch.eye(n, dtype=like.dtype, device=like.device)

    def kron(n: int) -> KronEigenbasis:
        # A [1, n] weight without bias: the identity in the eigenbasis.
        eye_n = torch.eye(n, dtype=like.dtype, device=like.device)
        eye_1 = torch.eye(1, dtype=like.dtype, device=like.device)

        return KronEigenbasis(eye_n, eye_1, ones(n).unsqueeze(0), False, (1, n))

    total = sum(layout.size(b) for b in layout.blocks)
    sizes = tuple((b, layout.size(b)) for b in layout.blocks)
    ops: list[Operator[Any]] = [
        BlockDiag({b: Diag(ones(layout.size(b))) for b in layout.blocks}),
        Diag(ones(total)),
        LowRankPlusDiag(ones(total).unsqueeze(-1), ones(1), ones(total)),
    ]

    # The Kronecker one holds an n x n eigenbasis per block, so it is gated like Dense.
    if dense:
        ops += [
            BlockDiag({b: kron(layout.size(b)) for b in layout.blocks}),
            BlockDiag({b: Dense(two_eye(layout.size(b))) for b in layout.blocks}),
            Dense(two_eye(total)),
            Joint(Dense(two_eye(total)), sizes),
        ]

    return ops


def _zero_operators(layout: Layout, like: Tensor, dense: bool) -> list[Operator[Any]]:
    """Zero operators to read a prior's precision off.

    Diagonal ones, and dense ones when small enough.
    """

    def zeros(*shape: int) -> Tensor:
        return torch.zeros(*shape, dtype=like.dtype, device=like.device)

    total = sum(layout.size(b) for b in layout.blocks)
    sizes = tuple((b, layout.size(b)) for b in layout.blocks)
    ops: list[Operator[Any]] = [
        BlockDiag({b: Diag(zeros(layout.size(b))) for b in layout.blocks}),
        Diag(zeros(total)),
    ]

    if dense:
        ops += [
            BlockDiag({b: Dense(zeros(layout.size(b), layout.size(b))) for b in layout.blocks}),
            Dense(zeros(total, total)),
            Joint(Dense(zeros(total, total)), sizes),
        ]

    return ops


def _is_diagonal(op: Operator[Any]) -> bool:
    if isinstance(op, HasBlocks):
        inner = _inner(op)

        if isinstance(inner, tuple):
            return all(issubclass(t, Diag) for _, t in inner)

        return isinstance(inner, type) and issubclass(inner, Diag)

    return isinstance(op, Diag)


def _try_add(prior: PriorWithCovariance, op: Operator[Any]) -> Operator[Any] | None:
    """`prior.add_precision_to(op)`, or None when the prior refuses the structure."""

    try:
        return prior.add_precision_to(op)
    except _REFUSED:
        return None


def _check_tunable(prior: TunablePrior) -> None:
    sel: object = prior.tunable()
    picked = tuple(sel.values()) if isinstance(sel, Mapping) else (sel,)
    leaves = tree_leaves(prior)

    for t in picked:
        if not isinstance(t, Tensor):
            raise _fail(
                "tunable",
                f"tunable() must return a tensor or a mapping of tensors, got {type(t).__name__}",
                prior,
            )

        if not any(t is leaf for leaf in leaves):
            raise _fail(
                "tunable",
                "tunable() returned a tensor that is not a leaf of the prior (a copy?)",
                prior,
            )

        if not bool((t > 0).all()):
            raise _fail("tunable", "tunable() must return positive tensors", prior)


def check_prior(prior: PriorWithCovariance, params: Params) -> None:
    r"""Check a prior against the core contracts.

    - It satisfies `PriorWithCovariance` (`log_prob`, `location`, `add_precision_to`,
      `cov_mv`), and `log_prob(params)` is a tensor with one finite element.
    - For a `TunablePrior`, `tunable()` returns positive tensors that are leaves of the
      prior itself (not copies), alone or in a mapping.
    - `add_precision_to` keeps the operator type, block by block for a `BlockDiag`. It is
      tried on `Diag`, `BlockDiag[Diag]`, `LowRankPlusDiag`, and for at most
      `DENSE_MAX_PARAMS` parameters on `KronEigenbasis` per block, `Dense`,
      `BlockDiag[Dense]` and `Joint[Dense]`. Refusing one with `StructureError` or
      `CapabilityError` is allowed; refusing all is not. Types compare exactly, so
      returning a subclass of the operator's type fails.
    - `cov_mv` is the inverse of the prior's precision: $\mathrm{cov\_mv}(v) = P^{-1} v$
      for a random $v$, where $P$ is read off by adding the prior to each zero operator
      it accepts: `Diag`, `BlockDiag[Diag]`, and for at most `DENSE_MAX_PARAMS`
      parameters `Dense`, `BlockDiag[Dense]` and `Joint[Dense]`. Refusing all fails.

    Every equality holds up to `RTOL`. Errors that the prior raises, other than a refusal
    by `add_precision_to`, propagate.

    Args:
        prior: The prior.
        params: Float64 parameter mapping. Blocks follow `Layout.by_module(params)`.

    Raises:
        AssertionError: A law above fails. The message names the law and `prior`.
        ValueError: `params` are not float64, or the prior accepts a single-block zero
            operator (`Diag`, `Dense`) and the params are not grouped by module.
    """

    _check_float64(params, "check_prior")

    if not isinstance(prior, PriorWithCovariance):
        raise _fail("protocol", "not a PriorWithCovariance", prior)

    if isinstance(prior, TunablePrior):
        _check_tunable(prior)

    lp: object = prior.log_prob(params)

    if not isinstance(lp, Tensor) or lp.numel() != 1 or not bool(torch.isfinite(lp).all()):
        raise _fail("log_prob", f"log_prob(params) must be a finite scalar, got {lp!r}", prior)

    layout = Layout.by_module(params)
    like = next(iter(params.values()))
    dense = _n_params(params) <= DENSE_MAX_PARAMS
    accepted = 0

    for op in _test_operators(layout, like, dense):
        out = _try_add(prior, op)

        if out is None:
            continue

        if _structure(out) != _structure(op):
            raise _fail(
                "add_precision_to keeps operator type",
                f"turned {_name(op)} into {_name(out)}",
                prior,
            )

        accepted += 1

    if accepted == 0:
        raise _fail("add_precision_to", "refused every test operator", prior)

    g = torch.Generator().manual_seed(0)
    v = {
        n: torch.randn(p.shape, generator=g, dtype=p.dtype).to(p.device) for n, p in params.items()
    }
    checked = 0

    for zero in _zero_operators(layout, like, dense):
        prec = _try_add(prior, zero)

        if prec is None:
            continue

        flat = _flat_for(zero, layout)
        x = flat.params(v)
        want = (
            x / flat.flat(prec.diagonal())
            if _is_diagonal(zero)
            else torch.linalg.solve(flat.dense(prec, like), x)
        )
        err = _rel_err(flat.params(prior.cov_mv(v)), want)

        if err > RTOL:
            raise _fail("cov_mv", "cov_mv is not the inverse of the precision", prior, err)

        checked += 1

    if checked == 0:
        raise _fail("cov_mv", "cannot check: the prior refused every zero operator", prior)


# --- likelihoods --------------------------------------------------------------------------


def _check_unit(lik: LateLikelihood) -> None:
    unit = lik.unit()

    if unit is not lik:
        kept = tree_leaves(unit)

        if all(any(a is b for b in kept) for a in tree_leaves(lik)):
            raise _fail(
                "unit identity",
                "unit() is a new object that drops no hyperparameter; without a late "
                "hyperparameter it must return self",
                lik,
            )

    if isinstance(unit, LateLikelihood) and unit.unit() is not unit:
        raise _fail("unit idempotence", "unit().unit() is not unit()", lik)


def _check_output_hessian(lik: HasOutputHessian, f: Tensor) -> None:
    n, event = f.shape[0], f[0].numel()

    if event > MAX_DENSE_EVENT:
        return

    hess = lik.output_hessian(f)
    dense = hess.dense()

    if tuple(dense.shape) != (n, event, event):
        raise _fail(
            "output Hessian shape",
            f"dense() is shaped {tuple(dense.shape)}, not {(n, event, event)}",
            lik,
        )

    g = torch.Generator().manual_seed(0)
    v = torch.randn(f.shape, generator=g, dtype=f.dtype).to(f.device)
    got = hess.mv(v).reshape(n, event)
    want = torch.einsum("nef,nf->ne", dense, v.reshape(n, event))
    err = _rel_err(got, want)

    if err > RTOL:
        raise _fail("output Hessian mv", "mv(v) != dense() @ v", lik, err)

    err = _rel_err(dense.mT, dense)

    if err > RTOL:
        raise _fail("output Hessian symmetry", "dense() is not symmetric", lik, err)

    scale = float(dense.abs().max()) if dense.numel() else 0.0
    low = float(torch.linalg.eigvalsh(dense).min()) if dense.numel() else 0.0

    if low < -RTOL * scale:
        raise _fail("output Hessian PSD", f"dense() has a negative eigenvalue {low:.1e}", lik)


def check_likelihood(likelihood: Likelihood, f: Outputs, y: Targets, /) -> None:
    r"""Check a likelihood's laws at outputs `f` and targets `y` (one batch).

    Not `core.check_likelihood`, which checks that a likelihood has the hooks a method's
    `Requirements` ask for.

    - It satisfies `Likelihood`, and `core.log_prob` is shaped `[N]`, one value per
      example.
    - `HasDistribution`: `distribution(f).log_prob(y)`, summed over each example's
      positions, equals `core.log_prob`.
    - `LateLikelihood`: `unit()` returns self unless it drops a hyperparameter, that is,
      unless some leaf of the likelihood (by identity) is not a leaf of `unit()`; and,
      when `unit()` is itself a `LateLikelihood`, `unit().unit()` is `unit()`.
    - `HasOutputHessian`: with $H$ = `dense()`, $H$ is shaped `[N, E, E]`,
      $\mathrm{mv}(v) = H v$ for a random $v$, $H = H^\top$, and $H$ is positive
      semi-definite: its smallest eigenvalue is at least $-\mathrm{RTOL} \cdot \max |H|$.
      These run only for events up to `MAX_DENSE_EVENT` entries and a tensor `f`.

    Masks are not applied: `log_prob` is compared over every position. Equalities hold
    up to `RTOL`. Errors that the likelihood raises propagate, except a `StructureError`
    from `core.log_prob`, which fails the shape law.

    Args:
        likelihood: The likelihood.
        f: Model outputs for one batch. Pass float64 so that `RTOL` holds.
        y: Targets for the same batch.

    Raises:
        AssertionError: A law above fails. The message names the law and `likelihood`.
    """

    if not isinstance(likelihood, Likelihood):
        raise _fail("protocol", "not a Likelihood", likelihood)

    n = tree_leaves(f)[0].shape[0]

    try:
        lp = core.log_prob(likelihood, f, y)
    except StructureError as e:
        raise _fail("log_prob shape", str(e), likelihood) from e

    if tuple(lp.shape) != (n,):
        raise _fail(
            "log_prob shape",
            f"log_prob summed over positions is shaped {tuple(lp.shape)}, not ({n},)",
            likelihood,
        )

    if isinstance(likelihood, HasDistribution):
        dist_lp = likelihood.distribution(f).log_prob(cast(Tensor, y))
        dist_lp = dist_lp.reshape(dist_lp.shape[0], -1).sum(1) if dist_lp.ndim else dist_lp
        err = _rel_err(dist_lp, lp)

        if err > RTOL:
            raise _fail(
                "distribution",
                "distribution(f).log_prob(y), summed over positions, != log_prob",
                likelihood,
                err,
            )

    if isinstance(likelihood, LateLikelihood):
        _check_unit(likelihood)

    if isinstance(likelihood, HasOutputHessian) and isinstance(f, Tensor):
        _check_output_hessian(likelihood, f)


# --- links --------------------------------------------------------------------------------


def _finite(t: Tensor) -> bool:
    return bool(torch.isfinite(t).all())


def _check_distribution(dist: Distribution, link: object) -> None:
    if isinstance(dist, Categorical):
        p = cast(Tensor, dist.probs)  # a lazy_property in the stubs

        if not _finite(p) or bool((p < 0).any()):
            raise _fail("probabilities", "probabilities must be finite and non-negative", link)

        err = _rel_err(p.sum(-1), torch.ones_like(p.sum(-1)))

        if err > RTOL:
            raise _fail("probabilities", "probabilities must sum to 1 over the classes", link, err)

        return

    try:
        mean, var = dist.mean, dist.variance
    except NotImplementedError as e:
        raise _fail("moments", f"{type(dist).__name__} has no mean or variance", link) from e

    if not _finite(mean) or not _finite(var) or bool((var < 0).any()):
        raise _fail("moments", "mean and variance must be finite, variance non-negative", link)


def _check_takes_generator(obj: object, what: str) -> None:
    """Fail unless `obj(...)` takes a `generator` keyword, as its caller passes one."""

    try:
        params = inspect.signature(cast(Callable[..., Any], obj)).parameters
    except (TypeError, ValueError):
        return

    kinds = {p.kind for p in params.values()}
    named = params.get("generator")

    if inspect.Parameter.VAR_KEYWORD in kinds:
        return

    if named is None or named.kind is inspect.Parameter.POSITIONAL_ONLY:
        raise _fail(
            "generator",
            f"__call__ must take a generator keyword (generator=None), as the {what}'s "
            "caller passes one",
            obj,
        )


def check_link(link: LinkApprox, likelihood: Likelihood, outputs: OutputGaussian) -> None:
    """Check a link approximation against the core contracts.

    - It satisfies `LinkApprox`, and `supports` returns a bool.
    - It is called with a `generator` keyword, as `Linearized` calls it.
    - `supports` agrees with applying it: when False, the call raises `CapabilityError` or
      `StructureError`; when True, it raises neither.
    - It returns a `Distribution`. A torch `Categorical` has finite, non-negative
      probabilities summing to 1 over the classes (up to `RTOL`); any other distribution
      has a finite mean and a finite, non-negative variance.

    When `supports` returns True, errors other than `CapabilityError` and `StructureError`
    from applying the link propagate.

    Args:
        link: The link approximation.
        likelihood: The likelihood to apply it with.
        outputs: The Gaussian over the outputs. `supports` sees its mean.

    Raises:
        AssertionError: A law above fails, including a call that raises another error
            after `supports` returned False. The message names the law and `link`.
    """

    if not isinstance(link, LinkApprox):
        raise _fail("protocol", "not a LinkApprox", link)

    lik = type(likelihood).__name__
    ok: object = link.supports(likelihood, cast(Tensor, outputs.mean))

    if not isinstance(ok, bool):
        raise _fail("supports", f"supports must return a bool, got {type(ok).__name__}", link)

    _check_takes_generator(link, "link")
    mean = cast(Tensor, outputs.mean)

    def call() -> Distribution:
        gen = torch.Generator(device=mean.device).manual_seed(0)
        return link(likelihood, outputs, generator=gen)

    if not ok:
        try:
            call()
        except _REFUSED:
            return
        except Exception as e:
            raise _fail(
                "supports",
                f"supports returned False for {lik} and applying it raised "
                f"{type(e).__name__}, not CapabilityError or StructureError",
                link,
            ) from e

        raise _fail(
            "supports", f"supports returned False for {lik} but applying it did not raise", link
        )

    try:
        dist: object = call()
    except _REFUSED as e:
        raise _fail(
            "supports",
            f"supports returned True for {lik} but applying it raised {type(e).__name__}",
            link,
        ) from e

    if not isinstance(dist, Distribution):
        raise _fail(
            "distribution", f"the result is {type(dist).__name__}, not a Distribution", link
        )

    _check_distribution(dist, link)


# --- subsets ------------------------------------------------------------------------------


def _same_tensor(t: Tensor, p: Tensor) -> bool:
    """`t` is `p`, or a view of all of `p` that shares its storage, such as `p.detach()`."""

    return (
        t.data_ptr() == p.data_ptr()
        and t.shape == p.shape
        and t.stride() == p.stride()
        and t.dtype == p.dtype
    )


def _check_chosen(chosen: object, module: nn.Module, subset: object) -> Mapping[Any, Any]:
    if not isinstance(chosen, Mapping):
        raise _fail("mapping", f"the result is {type(chosen).__name__}, not a Mapping", subset)

    chosen = cast(Mapping[Any, Any], chosen)

    if not chosen:
        raise _fail(
            "non-empty",
            "selected no parameters; with nothing to select, raise StructureError naming "
            "what is missing (as subsets.named does)",
            subset,
        )

    own = dict(module.named_parameters(remove_duplicate=False))

    for name, t in chosen.items():
        if not isinstance(name, str) or name not in own:
            raise _fail("names", f"{name!r} is not a parameter name of the module", subset)

        if not isinstance(t, Tensor):
            raise _fail("own parameters", f"{name!r} is {type(t).__name__}, not a Tensor", subset)

        if not _same_tensor(t, own[name]):
            raise _fail("own parameters", f"{name!r} is a copy, not the module's parameter", subset)

    return chosen


def check_subset(subset: Subset, model: nn.Module | ModelFn, caps: Capabilities) -> None:
    """Check a subset against the core contracts.

    - It returns a non-empty `Mapping` from parameter names of the module (tied aliases
      included) to the module's own parameters under those names. A view of the whole
      parameter, such as `p.detach()`, counts; a copy does not.
    - Or it returns a `Selection` whose `base` is such a mapping.
    - It is deterministic: a second call selects the same names, in the same order, and
      the same tensors (and, for a `Selection`, the same entries).

    Args:
        subset: The subset.
        model: The model, as a module or a `ModelFn` with a module.
        caps: What the subset reads, usually the forward order from
            `lk.core.probe(fn, likelihood, batch, params={})`.

    Raises:
        AssertionError: A law above fails. The message names the law and `subset`.
        ValueError: `model` is a `ModelFn` without a module.
        StructureError: Passed through from the subset, which raises it (or
            `CapabilityError`) when it has nothing to select, naming what is missing, as
            `subsets.named` does.
        CapabilityError: Passed through from the subset, as above.
    """

    fn = ModelFn.from_module(model) if isinstance(model, nn.Module) else model
    module = fn.module

    if module is None:
        raise ValueError("check_subset needs a module to compare names, but ModelFn.module is None")

    one, two = subset(fn, caps), subset(fn, caps)

    if isinstance(one, Selection) != isinstance(two, Selection):
        raise _fail("deterministic", "one call returned a Selection, the other did not", subset)

    first = _check_chosen(one.base if isinstance(one, Selection) else one, module, subset)
    second = _check_chosen(two.base if isinstance(two, Selection) else two, module, subset)
    same = list(first) == list(second) and all(_same_tensor(first[n], second[n]) for n in first)

    if not same:
        raise _fail(
            "deterministic",
            f"a second call selected {list(second)}, the first {list(first)} "
            "(or other tensors under the same names)",
            subset,
        )

    if isinstance(one, Selection) and isinstance(two, Selection):
        if not all(torch.equal(one.index[n], two.index[n]) for n in one.index):
            raise _fail("deterministic", "a second call selected other entries", subset)


# --- objectives ---------------------------------------------------------------------------


def check_objective[P](objective: Objective[P], posterior: P) -> None:
    r"""Check a tuning objective against the core contracts.

    The objective is called on a copy of `posterior` whose leaves are detached and whose
    hyperparameters $\phi_i$ (`lk.core.hyperparameters`, what `lk.Minimize` tunes)
    require grad.

    - It returns a 0-dim tensor $L$, and $L$ is finite.
    - It is differentiable with respect to every hyperparameter: each
      $\partial L / \partial \phi_i$ exists and is finite.

    `posterior` is not changed. Errors that the objective raises propagate.

    Args:
        objective: The tuning objective.
        posterior: The posterior to evaluate it at.

    Raises:
        AssertionError: A law above fails. The message names the law and `objective`.
        ValueError: The posterior has no hyperparameters.
    """

    frozen = core.detach_leaves(posterior)
    found = core.hyperparameters(frozen)

    if not found:
        raise ValueError(f"{type(posterior).__name__} has no hyperparameters to check against")

    names, old = tuple(found), tuple(found.values())
    new = tuple(t.clone().requires_grad_() for t in old)

    with torch.enable_grad():
        loss: object = objective(core.replace_leaves(frozen, old, new))

    if not isinstance(loss, Tensor):
        raise _fail("0-dim", f"the result is {type(loss).__name__}, not a Tensor", objective)

    if loss.ndim != 0:
        raise _fail(
            "0-dim", f"the result must be a 0-dim tensor, got shape {tuple(loss.shape)}", objective
        )

    if not _finite(loss):
        raise _fail("finite", f"the result is {float(loss.detach())}", objective)

    if not loss.requires_grad:
        raise _fail(
            "differentiable",
            "the result does not depend on the selected leaves (is it detached?)",
            objective,
        )

    grads = torch.autograd.grad(loss, new, allow_unused=True)

    for name, g in zip(names, grads, strict=True):
        if g is None:
            raise _fail("differentiable", f"no gradient reaches {name}", objective)

        if not _finite(g):
            raise _fail("differentiable", f"the gradient for {name} is not finite", objective)


# --- predictives --------------------------------------------------------------------------


def check_predictive[B](predictive: Predictive[B], posterior: B, inputs: Inputs) -> None:
    """Check a predictive against the core contracts.

    - `requires` is a `Requirements`.
    - It takes a `generator` keyword, as `lk.predict` calls it.
    - It returns a `Distribution` whose batch shape starts with the number of examples in
      `inputs`: the leading dimension of the first input leaf, as `Batch.size` counts it.
      Trailing batch dimensions (outputs, sequence positions) are allowed.

    Errors that the predictive raises propagate.

    Args:
        predictive: The predictive.
        posterior: The posterior to predict with.
        inputs: The inputs to predict at.

    Raises:
        AssertionError: A law above fails. The message names the law and `predictive`.
        ValueError: `inputs` has no tensor leaves, or its first leaf is 0-dim.
    """

    requires: object = predictive.requires

    if not isinstance(requires, Requirements):
        raise _fail(
            "requires", f"requires is {type(requires).__name__}, not Requirements", predictive
        )

    _check_takes_generator(predictive, "predictive")
    batch = Batch(inputs)
    n = batch.size
    device = tree_leaves(batch.inputs)[0].device
    dist: object = predictive(
        posterior, inputs, generator=torch.Generator(device=device).manual_seed(0)
    )

    if not isinstance(dist, Distribution):
        raise _fail(
            "distribution", f"the result is {type(dist).__name__}, not a Distribution", predictive
        )

    shape = tuple(dist.batch_shape)

    if shape[:1] != (n,):
        raise _fail(
            "batch shape",
            f"batch_shape {shape} does not start with the {n} examples of the inputs",
            predictive,
        )
