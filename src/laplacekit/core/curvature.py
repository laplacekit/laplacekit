r"""Area 1, weight space: curvature.

A single-pass curvature is a fold over batches into a statistic, then `finalize`.
Statistics form a monoid: `init` gives the identity $e$ and `+` combines,

$$e + s = s + e = s, \qquad (s_1 + s_2) + s_3 = s_1 + (s_2 + s_3).$$

That makes streaming, resuming, all_reduce, and late data mixing the same operation.
The law to test for every curvature is

$$\operatorname{fold}(a \mathbin{+\!\!+} b) = \operatorname{fold}(a) + \operatorname{fold}(b),$$

for batch sequences $a$ and $b$ and their concatenation $a \mathbin{+\!\!+} b$.

Conventions:

- Every method gets a `FitContext` (model, likelihood, layout, capabilities) up
  front, so it can raise `CapabilityError` in `init` or `plan`, before any data.
- `finalize` returns the per-example average: the weighted sum over examples divided by
  the total weight,

  $$\bar{C} = \frac{\sum_n w_n C_n}{\sum_n w_n}.$$

  The posterior multiplies it by `scale`, the effective data size $N$. Its precision is
  $c N \bar{C} + \Lambda_0$, with $\Lambda_0$ the prior precision and $c$ the
  `curvature_scale` of a `LateLikelihood` (no factor for other likelihoods).
- Methods needing several passes (EKFAC) are `Staged`: `plan` returns a chain of
  passes built with `chain(...)`, ending in `Done`. Users never see passes;
  `lk.laplace` runs them all, iterating the data once per pass.
- Randomness: each batch's `update` gets a fresh `torch.Generator` on the params'
  device, seeded from the fit's seed, the pass number and the examples before the
  batch. A sampled method draws from it, so a batch draws the same numbers on every rerun
  and in every merge order.
"""

import copy
import dataclasses
import fnmatch
import itertools
from abc import abstractmethod
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any, Generic, Protocol, Self, TypeVar, cast, overload, runtime_checkable

import torch
from torch import Tensor

from laplacekit.core._pytree import pytree, static
from laplacekit.core._seed import _Seed
from laplacekit.core.layout import Layout
from laplacekit.core.likelihoods import Likelihood
from laplacekit.core.model import ModelFn
from laplacekit.core.ops import BlockDiag, Joint, Operator
from laplacekit.core.types import (
    Batch,
    Capabilities,
    Inputs,
    Outputs,
    Params,
    Requirements,
    StructureError,
)


@dataclasses.dataclass(frozen=True)
class FitContext:
    """What every pass knows about the fit, fixed for its duration.

    Args:
        model: The model as a function of the params.
        likelihood: The likelihood the curvature sees: the unit one (see
            `LateLikelihood`).
        layout: How the params group into blocks.
        caps: What the probe recorded about the model.
        seed: A seed fixed for the whole fit, drawn from the fit's `generator`, for draws
            that must be the same on every batch (a sketch's test matrix). Per-batch draws
            use the `generator` that `update` gets instead.
    """

    model: ModelFn
    likelihood: Likelihood
    layout: Layout
    caps: Capabilities
    seed: int = 0


@runtime_checkable
class Stat(Protocol):
    """A statistic that merges under `+`, a monoid with the curvature's `init` as identity.

    Merging is associative: folding batches one by one and adding partial folds give the
    same statistic.
    """

    @property
    @abstractmethod
    def total_weight(self) -> Tensor:
        """The summed weight of the examples folded in so far.

        It is the denominator of `finalize`'s average.
        """
        ...

    @abstractmethod
    def __add__(self, other: Self, /) -> Self:
        """The statistic of both data sets together.

        `+` is how batches, partial folds and the shards of other processes combine, so
        it must be associative, with the curvature's `init` as its identity.

        Args:
            other: The statistic of other data, from the same curvature and parameters.

        Returns:
            A statistic of the same type, as if one fold had seen both data sets.
        """
        ...

    @abstractmethod
    def scaled(self, s: Tensor, /) -> Self:
        """The statistic with every summand multiplied by `s`.

        Meaningful only when the statistic is a sum (`Requirements.additive_stat`).

        Args:
            s: A 0-dim factor.

        Returns:
            A statistic of the same type.
        """
        ...


@runtime_checkable
class Curvature[S: Stat, R](Protocol):
    r"""One pass over the data, a fold of batches into a statistic `S`.

    `R` is an Operator for an ordinary curvature, or an intermediate result (an
    eigenbasis, a subset) for one pass of a staged method.
    """

    @property
    @abstractmethod
    def requires(self) -> Requirements:
        """What the method needs from the model and the likelihood."""
        ...

    @abstractmethod
    def init(self, params: Params, ctx: FitContext, /) -> S:
        """The empty statistic, the identity of `+`.

        This is the place to refuse a fit before any data.

        Args:
            params: The params the curvature is taken at.
            ctx: The fit's context.

        Returns:
            A statistic $e$ with $e + s = s$ for every statistic $s$ of this fit.

        Raises:
            CapabilityError: Implementations may raise it when the model or likelihood
                lacks something the method needs.
            StructureError: Implementations may raise it when the params or layout don't
                fit the method.
            ValueError: Implementations may raise it when the method's settings don't
                fit the params (e.g. `LowRankGGN` with `rank + oversample` above the
                parameter count).
        """
        ...

    @abstractmethod
    def update(
        self, stat: S, params: Params, batch: Batch, generator: torch.Generator, ctx: FitContext, /
    ) -> S:
        """Fold one batch into the statistic.

        Pure: it returns a new statistic.

        Args:
            stat: The statistic so far.
            params: The params the curvature is taken at.
            batch: The batch to add. Honour `batch.weights` when `requires.weights` is
                True.
            generator: The batch's own generator, on the params' device. Sampled methods
                draw from it, e.g. `torch.randn(..., generator=generator)`.
            ctx: The fit's context.

        Returns:
            `stat` with the batch folded in, so that updating with batches $a$, then $b$
            equals adding the statistics of $a$ and of $b$.

        Raises:
            CapabilityError: Implementations may raise it when the method needs targets
                and the batch has none (the empirical Fisher, the Hessian, an observed
                output Hessian).
        """
        ...

    @abstractmethod
    def finalize(self, stat: S, /) -> R:
        r"""The result of the pass.

        Args:
            stat: The statistic over all the data, after `all_reduce`.

        Returns:
            For an ordinary curvature, an operator holding the per-example average
            $\bar{C} = \sum_n w_n C_n / \sum_n w_n$. The posterior multiplies it by
            `scale`. For a pass of a staged method, the intermediate result.

        Raises:
            ValueError: Implementations may raise it when the total weight is zero.
        """
        ...


# --- multi-pass methods, as continuations -----------------------------------------

_R_co = TypeVar("_R_co", covariant=True)


@pytree
class Pass(Generic[_R_co]):  # noqa: UP046
    """Run `curvature` over the data, then call `then` with its result.

    Build it with `chain`, which checks the types.

    Args:
        curvature: The pass's curvature.
        then: Takes `curvature.finalize`'s result and returns the next `Pass`, or `Done`.
    """

    curvature: Curvature[Any, Any] = static()
    then: "Callable[[Any], Pass[_R_co] | Done[_R_co]]" = static()


@pytree
class Done(Generic[_R_co]):  # noqa: UP046
    """The last step of a `Staged` plan.

    Args:
        result: The fit's result.
    """

    result: _R_co


def chain[M, R](curvature: Curvature[Any, M], /, then: Callable[[M], Pass[R] | Done[R]]) -> Pass[R]:
    """A type-safe pass.

    Args:
        curvature: The pass's curvature.
        then: The continuation. It must accept exactly what `curvature.finalize` returns.

    Returns:
        A `Pass` running `curvature`, then `then`.
    """

    return Pass(curvature, then)


@runtime_checkable
class Staged(Protocol[_R_co]):  # explicit variance: inference fails through Pass's recursion
    """A weight-space method needing several passes."""

    @property
    @abstractmethod
    def requires(self) -> Requirements:
        """What the method needs from the model and the likelihood."""
        ...

    @abstractmethod
    def plan(self, params: Params, ctx: FitContext, /) -> Pass[_R_co]:
        """The method's passes, before any data.

        Args:
            params: The params the curvature is taken at.
            ctx: The fit's context.

        Returns:
            The first `Pass` of a chain built with `chain(...)`, ending in `Done`.

        Raises:
            CapabilityError: Implementations may raise it when the model or likelihood
                lacks something the method needs.
            StructureError: Implementations may raise it when the params or layout don't
                fit the method.
        """
        ...


type HessianApprox[Op: Operator[Any]] = Curvature[Any, Op] | Staged[Op]
"""What `lk.laplace(hessian_approx=...)` takes."""


@runtime_checkable
class HasTemplate(Protocol):
    """A `Curvature` or `Staged` method that knows its result's structure before any data.

    It reads no data, so checks that need only the structure (does a prior fit the
    curvature?) can run on it before the data pass.

    A template is a real operator, so it costs what the result costs: a `Dense` template
    holds $P^2$ zeros, and adding a prior to it touches every one (about 0.1 s at 9,802
    weights in float32).
    """

    @abstractmethod
    def template(self, params: Params, ctx: FitContext, /) -> Operator[Any]:
        """A zero operator shaped like the result.

        Args:
            params: The params the curvature is taken at.
            ctx: The fit's context.

        Returns:
            A zero operator with exactly the structure `finalize` (or the last pass)
            returns: the same operator types, blocks, static fields, leaf shapes and
            dtypes, with `KronEigenbasis.factors` present as zeros when `finalize` sets
            them.
        """
        ...


# --- statistics for the native curvature methods ----------------------------------


def stat_dtype(dtype: torch.dtype, /) -> torch.dtype:
    """The dtype statistics accumulate in.

    A bf16 running sum stops changing after a few hundred terms, so a statistic is
    allocated in this dtype and each batch's contribution is cast to it before adding.

    Args:
        dtype: The model's dtype.

    Returns:
        `dtype`, but at least float32.
    """

    return torch.promote_types(dtype, torch.float32)


@pytree
class SumStat:
    """Per-block weighted sums and the total weight.

    Args:
        sum: Weighted sum per block, vectors or matrices (see `layout`).
        weight: The total weight, a 0-dim tensor.
    """

    sum: Mapping[str, Tensor]
    weight: Tensor

    @property
    def total_weight(self) -> Tensor:
        """`weight`."""

        return self.weight

    def __add__(self, other: Self, /) -> Self:
        """Blockwise sums and the weights added.

        Args:
            other: The statistic of other data, from the same curvature and parameters.

        Returns:
            A statistic of the same type, as if one fold had seen both data sets.
        """

        return type(self)(
            {k: v + other.sum[k] for k, v in self.sum.items()}, self.weight + other.weight
        )

    def scaled(self, s: Tensor, /) -> Self:
        """Every sum and the weight times `s`.

        Args:
            s: A 0-dim factor.

        Returns:
            A statistic of the same type.
        """

        return type(self)({k: s * v for k, v in self.sum.items()}, s * self.weight)


# --- a method per block -------------------------------------------------------------

type BlockRule = str | tuple[str, ...]
"""A `per_block` rule: a glob over block names, or a tuple of exact block names."""

type _Members = tuple[tuple[BlockRule, tuple[str, ...], int], ...]
"""Per member: its rule, the blocks it covers (in layout order), and the rule's position,
which seeds the member's own update generator."""


def _matches(rule: BlockRule, block: str) -> bool:
    if isinstance(rule, tuple):
        return block in rule

    return fnmatch.fnmatchcase(block, rule)


def _assign(rules: Mapping[BlockRule, object], layout: Layout) -> _Members:
    """Each block goes to the first rule that matches its name."""

    unknown = [b for r in rules if isinstance(r, tuple) for b in r if b not in layout.blocks]

    if unknown:
        raise StructureError(
            f"per_block: rules name blocks that don't exist: {unknown} "
            f"(blocks: {list(layout.blocks)})"
        )

    groups: dict[BlockRule, list[str]] = {}
    unmatched: list[str] = []

    for b in layout.blocks:
        rule = next((r for r in rules if _matches(r, b)), None)

        if rule is None:
            unmatched.append(b)
        else:
            groups.setdefault(rule, []).append(b)

    if unmatched:
        raise StructureError(
            f"per_block: no rule matches blocks {unmatched} (rules: {list(rules)}); "
            "add a rule for them, e.g. '*'"
        )

    unused = [_why_unused(r, rules, groups, layout) for r in rules if r not in groups]

    if unused:
        raise StructureError(f"per_block: every rule needs a block; {'; '.join(unused)}")

    if not groups:
        raise StructureError("per_block needs at least one block")

    position = {r: i for i, r in enumerate(rules)}

    return tuple((r, tuple(bs), position[r]) for r, bs in groups.items())


def _why_unused(
    rule: BlockRule,
    rules: Mapping[BlockRule, object],
    groups: Mapping[BlockRule, list[str]],
    layout: Layout,
) -> str:
    """Why `rule` got no block: it matches none, or earlier rules took its matches."""

    matches = [b for b in layout.blocks if _matches(rule, b)]

    if not matches:
        return f"rule {rule!r} matches no block (blocks: {list(layout.blocks)})"

    earlier = [r for r in rules if r in groups and set(groups[r]) & set(matches)]

    return f"rule {rule!r} matches {matches}, all taken by earlier rule(s) {earlier}"


def _restrict(
    params: Params, ctx: FitContext, blocks: tuple[str, ...]
) -> tuple[Params, FitContext]:
    """A member's view: its blocks' params and layout.

    The model still sees every passed param, so members agree on the point the model is
    evaluated at.
    """

    layout = ctx.layout.restrict(blocks)
    names = {n for ns in layout.blocks.values() for n in ns}
    own = {n: p for n, p in params.items() if n in names}
    rest = {n: p for n, p in params.items() if n not in names}

    if not rest:
        return own, copy.replace(ctx, layout=layout)

    model = ctx.model

    def forward(q: Params, x: Inputs) -> Outputs:
        return model({**rest, **q}, x)

    return own, copy.replace(ctx, model=ModelFn(forward, model.module), layout=layout)


def _merge_requirements(reqs: Sequence[Requirements]) -> Requirements:
    """What all of `reqs` need at once."""

    types = {r.input_type for r in reqs if r.input_type is not None}

    if len(types) > 1:
        raise StructureError(
            f"per_block members need different input types: {sorted(t.__name__ for t in types)}"
        )

    hooks = tuple(dict.fromkeys(h for r in reqs for h in r.hooks))
    # Each member needs one of its alternatives: all members, any combination of theirs.
    alternatives = [r.one_of for r in reqs if r.one_of]
    one_of = (
        tuple(_merge_requirements(c) for c in itertools.product(*alternatives))
        if alternatives
        else ()
    )

    return Requirements(
        vjp=any(r.vjp for r in reqs),
        vjp_twice=any(r.vjp_twice for r in reqs),
        jvp=any(r.jvp for r in reqs),
        vmap=any(r.vmap for r in reqs),
        hooks=hooks,
        needs_targets=any(r.needs_targets for r in reqs),
        needs_sample=any(r.needs_sample for r in reqs),
        needs_distribution=any(r.needs_distribution for r in reqs),
        needs_output_hessian=any(r.needs_output_hessian for r in reqs),
        needs_closed_form_hessian=any(r.needs_closed_form_hessian for r in reqs),
        late_scale=all(r.late_scale for r in reqs),
        additive_stat=all(r.additive_stat for r in reqs),
        weights=all(r.weights for r in reqs),
        input_type=next(iter(types), None),
        one_of=one_of,
    )


def _rules_repr(name: str, rules: Mapping[BlockRule, object]) -> str:
    return f"{name}({dict(rules)!r})"


@pytree
class BlockStat:
    """One statistic per member of a per-block curvature, keyed by its rule.

    `PerBlockCurvature.init` builds it.

    Args:
        stats: Each member's statistic, keyed by its rule.
        members: Per member: its rule, the blocks it covers (in layout order), and the
            rule's position, which seeds the member's own update generator.
        order: All blocks, in layout order.
    """

    stats: Mapping[BlockRule, Stat]
    members: _Members = static()
    order: tuple[str, ...] = static()

    @property
    def total_weight(self) -> Tensor:
        """Every member sees the same batches, so the first one speaks for all."""

        return next(iter(self.stats.values())).total_weight

    def __add__(self, other: Self, /) -> Self:
        """Each member's statistics added.

        Args:
            other: The statistic of other data, from the same curvature and parameters.

        Returns:
            A statistic of the same type, as if one fold had seen both data sets.
        """

        stats = {r: s + other.stats[r] for r, s in self.stats.items()}

        return type(self)(stats, self.members, self.order)

    def scaled(self, s: Tensor, /) -> Self:
        """Each member's statistic scaled by `s`.

        Args:
            s: A 0-dim factor.

        Returns:
            A statistic of the same type.
        """

        return type(self)(
            {r: st.scaled(s) for r, st in self.stats.items()}, self.members, self.order
        )


def _init_members(
    curvatures: Mapping[BlockRule, Curvature[Any, Any]],
    members: _Members,
    order: tuple[str, ...],
    params: Params,
    ctx: FitContext,
) -> BlockStat:
    stats: dict[BlockRule, Stat] = {}

    for rule, blocks, _ in members:
        p, c = _restrict(params, ctx, blocks)
        stats[rule] = curvatures[rule].init(p, c)

    return BlockStat(stats, members, order)


def _update_members(
    curvatures: Mapping[BlockRule, Curvature[Any, Any]],
    stat: BlockStat,
    params: Params,
    batch: Batch,
    generator: torch.Generator,
    ctx: FitContext,
) -> BlockStat:
    """Members run one after another on the same batch, each with its own generator.

    One seed is drawn from `generator`; member $i$ gets a generator from that seed and
    its rule's position, so its draws don't depend on the other members.
    """

    seed = _Seed.draw(generator)
    stats: dict[BlockRule, Stat] = {}

    for rule, blocks, index in stat.members:
        p, c = _restrict(params, ctx, blocks)
        own = seed.fold_in(index).generator(generator.device)
        stats[rule] = curvatures[rule].update(stat.stats[rule], p, batch, own, c)

    return BlockStat(stats, stat.members, stat.order)


def _joint_member(rule: BlockRule) -> str:
    return (
        f"per_block member {rule!r} returned a joint operator over all its blocks; "
        "pass a joint curvature as hessian_approx= directly"
    )


def _refuse_joint_templates(
    rules: Mapping[BlockRule, HessianApprox[Any]],
    members: _Members,
    params: Params,
    ctx: FitContext,
) -> None:
    """Refuse a member whose template is a `Joint` before the data pass.

    The merge would refuse it only after. Members without a template are checked at the
    merge.
    """

    for rule, blocks, _ in members:
        method = rules[rule]

        if not isinstance(method, HasTemplate):
            continue

        p, c = _restrict(params, ctx, blocks)

        if isinstance(method.template(p, c), Joint):
            raise StructureError(_joint_member(rule))


def _merge_results(
    results: Mapping[BlockRule, Any], members: _Members, order: tuple[str, ...]
) -> BlockDiag[Any]:
    """One `BlockDiag` in layout order.

    A member's `BlockDiag` gives its blocks; any other operator is the member's single
    block. A `Joint` member raises: it couples its blocks, which a `BlockDiag` can't hold.
    """

    merged: dict[str, Any] = {}

    for rule, blocks, _ in members:
        out = results[rule]

        if isinstance(out, Joint):
            raise StructureError(_joint_member(rule))

        if isinstance(out, BlockDiag):
            got = cast(BlockDiag[Any], out).blocks

            if set(got) != set(blocks):
                raise StructureError(
                    f"per_block rule {rule!r} returned blocks {sorted(got)}, "
                    f"expected {sorted(blocks)}"
                )

            merged.update(got)
        elif len(blocks) == 1:
            merged[blocks[0]] = out
        else:
            raise StructureError(
                f"per_block rule {rule!r} returned one operator for {len(blocks)} blocks "
                f"{list(blocks)}; it must return a BlockDiag keyed by block"
            )

    return BlockDiag({b: merged[b] for b in order})


@pytree
class PerBlockCurvature:
    """A single-pass curvature per block, chosen by rule over block names (first match wins).

    Each member sees only its blocks' params and layout. The results are merged into one
    `BlockDiag` in layout order. Build with `per_block`, which adds a template
    (`HasTemplate`) when every member has one.

    Args:
        rules: The method per rule, `{rule: curvature}`.
    """

    rules: Mapping[BlockRule, Curvature[Any, Any]] = static()

    def __repr__(self) -> str:
        return _rules_repr("PerBlockCurvature", self.rules)

    def by_block(self, blocks: Iterable[str], /) -> dict[str, Curvature[Any, Any]]:
        """The method each block gets.

        Args:
            blocks: Block names.

        Returns:
            For each block, the method of its first matching rule.

        Raises:
            StructureError: No rule matches a block.
        """

        out: dict[str, Curvature[Any, Any]] = {}

        for b in blocks:
            rule = next((r for r in self.rules if _matches(r, b)), None)

            if rule is None:
                raise StructureError(f"per_block: no rule matches block {b!r}")

            out[b] = self.rules[rule]

        return out

    @property
    def requires(self) -> Requirements:
        """What all members need at once.

        Raises:
            StructureError: Members need different input types.
        """

        return _merge_requirements([c.requires for c in self.rules.values()])

    def init(self, params: Params, ctx: FitContext, /) -> BlockStat:
        """Each member's empty statistic, on its own blocks.

        Args:
            params: The params the curvature is taken at.
            ctx: The fit's context.

        Returns:
            A statistic $e$ with $e + s = s$ for every statistic $s$ of this fit.

        Raises:
            StructureError: A rule names a block that doesn't exist, a block matches no
                rule, a rule gets no block, there are no blocks, or a member's template is
                a `Joint`.
        """

        members = _assign(self.rules, ctx.layout)
        _refuse_joint_templates(self.rules, members, params, ctx)

        return _init_members(self.rules, members, tuple(ctx.layout.blocks), params, ctx)

    def update(
        self,
        stat: BlockStat,
        params: Params,
        batch: Batch,
        generator: torch.Generator,
        ctx: FitContext,
        /,
    ) -> BlockStat:
        """Each member's update in turn on the same batch, each with its own generator.

        Args:
            stat: The statistic so far.
            params: The params the curvature is taken at.
            batch: The batch to add.
            generator: The batch's generator. Each member gets a fresh one derived from
                it and the member's position.
            ctx: The fit's context.

        Returns:
            `stat` with the batch added. Updating with batches $a$, then $b$, equals adding
            the statistics of $a$ and of $b$.
        """

        return _update_members(self.rules, stat, params, batch, generator, ctx)

    def finalize(self, stat: BlockStat, /) -> BlockDiag[Any]:
        """The members' results merged into one `BlockDiag` in layout order.

        Args:
            stat: The statistic over all the data, after `all_reduce`.

        Returns:
            One `BlockDiag` that holds each member's operator for each of its blocks.

        Raises:
            StructureError: A member returns a `Joint`, a `BlockDiag` over other blocks,
                or a single operator for several blocks.
        """

        results = {r: self.rules[r].finalize(s) for r, s in stat.stats.items()}

        return _merge_results(results, stat.members, stat.order)


def _templates(
    rules: Mapping[BlockRule, HessianApprox[Any]], params: Params, ctx: FitContext
) -> BlockDiag[Any]:
    """Each member's template on its own blocks, merged like the results."""

    members = _assign(rules, ctx.layout)
    results: dict[BlockRule, Any] = {}

    for rule, blocks, _ in members:
        p, c = _restrict(params, ctx, blocks)
        results[rule] = cast(HasTemplate, rules[rule]).template(p, c)

    return _merge_results(results, members, tuple(ctx.layout.blocks))


def _all_templated(rules: Mapping[BlockRule, object]) -> bool:
    return all(isinstance(m, HasTemplate) for m in rules.values())


@pytree
class _TemplatedPerBlock(PerBlockCurvature):
    """A `PerBlockCurvature` whose members all have templates; `per_block` builds it."""

    __repr__ = PerBlockCurvature.__repr__

    def __replace__(self, /, **changes: Any) -> PerBlockCurvature:
        """New rules keep the template only if every member has one."""

        rules = dataclasses.replace(PerBlockCurvature(self.rules), **changes).rules

        return _TemplatedPerBlock(rules) if _all_templated(rules) else PerBlockCurvature(rules)

    def template(self, params: Params, ctx: FitContext, /) -> BlockDiag[Any]:
        """The members' templates, each under its blocks."""

        return _templates(self.rules, params, ctx)


@pytree
class _Joint:
    """One pass of a `StagedPerBlock`: the current pass of every member still running."""

    curvatures: Mapping[BlockRule, Curvature[Any, Any]] = static()
    members: _Members = static()
    order: tuple[str, ...] = static()

    @property
    def requires(self) -> Requirements:
        return _merge_requirements([c.requires for c in self.curvatures.values()])

    def init(self, params: Params, ctx: FitContext, /) -> BlockStat:
        return _init_members(self.curvatures, self.members, self.order, params, ctx)

    def update(
        self,
        stat: BlockStat,
        params: Params,
        batch: Batch,
        generator: torch.Generator,
        ctx: FitContext,
        /,
    ) -> BlockStat:
        return _update_members(self.curvatures, stat, params, batch, generator, ctx)

    def finalize(self, stat: BlockStat, /) -> Mapping[BlockRule, Any]:
        return {r: self.curvatures[r].finalize(s) for r, s in stat.stats.items()}


def _joint_pass(
    passes: Mapping[BlockRule, Pass[Any]],
    done: Mapping[BlockRule, Any],
    members: _Members,
    order: tuple[str, ...],
) -> Pass[BlockDiag[Any]]:
    """Run the members' current passes together. Members that are done sit out the rest."""

    running = tuple(m for m in members if m[0] in passes)
    joint = _Joint({r: p.curvature for r, p in passes.items()}, running, order)

    def then(results: Mapping[BlockRule, Any]) -> Pass[BlockDiag[Any]] | Done[BlockDiag[Any]]:
        finished = dict(done)
        later: dict[BlockRule, Pass[Any]] = {}

        for rule, out in results.items():
            step = passes[rule].then(out)

            if isinstance(step, Done):
                finished[rule] = step.result
            else:
                later[rule] = step

        if later:
            return _joint_pass(later, finished, members, order)

        return Done(_merge_results(finished, members, order))

    return chain(joint, then=then)


def _finish[R](result: R) -> Done[R]:
    return Done(result)


@pytree
class StagedPerBlock:
    """Per-block rules where some members need several passes.

    All members share the first pass; single-pass members then sit out the later ones.
    Build with `per_block`, which adds a template (`HasTemplate`) when every member has
    one.

    Args:
        rules: The method per rule, `{rule: method}`.
    """

    rules: Mapping[BlockRule, HessianApprox[Any]] = static()

    def __repr__(self) -> str:
        return _rules_repr("StagedPerBlock", self.rules)

    @property
    def requires(self) -> Requirements:
        """What all members need at once.

        Raises:
            StructureError: Members need different input types.
        """

        return _merge_requirements([c.requires for c in self.rules.values()])

    def plan(self, params: Params, ctx: FitContext, /) -> Pass[BlockDiag[Any]]:
        """Joint passes of all members, merged into one `BlockDiag` at the end.

        Returns:
            The first pass. When run, the last pass raises `StructureError` when a member
            returns a `Joint`, a `BlockDiag` over other blocks, or a single operator for
            several blocks.

        Args:
            params: The params the curvature is taken at.
            ctx: The fit's context.

        Raises:
            StructureError: A rule names a block that doesn't exist, a block matches no
                rule, a rule gets no block, there are no blocks, or a member's template is
                a `Joint`.
        """

        members = _assign(self.rules, ctx.layout)
        _refuse_joint_templates(self.rules, members, params, ctx)
        passes: dict[BlockRule, Pass[Any]] = {}

        for rule, blocks, _ in members:
            method = self.rules[rule]

            if isinstance(method, Staged):
                p, c = _restrict(params, ctx, blocks)
                passes[rule] = method.plan(p, c)
            else:
                passes[rule] = chain(method, then=_finish)

        return _joint_pass(passes, {}, members, tuple(ctx.layout.blocks))


@pytree
class _TemplatedStagedPerBlock(StagedPerBlock):
    """A `StagedPerBlock` whose members all have templates; `per_block` builds it."""

    __repr__ = StagedPerBlock.__repr__

    def __replace__(self, /, **changes: Any) -> StagedPerBlock:
        """New rules keep the template only if every member has one."""

        rules = dataclasses.replace(StagedPerBlock(self.rules), **changes).rules

        return _TemplatedStagedPerBlock(rules) if _all_templated(rules) else StagedPerBlock(rules)

    def template(self, params: Params, ctx: FitContext, /) -> BlockDiag[Any]:
        """The members' templates, each under its blocks."""

        return _templates(self.rules, params, ctx)


@overload
def per_block[K: BlockRule](rules: Mapping[K, Curvature[Any, Any]], /) -> PerBlockCurvature: ...


@overload
def per_block[K: BlockRule](
    rules: Mapping[K, HessianApprox[Any]], /
) -> PerBlockCurvature | StagedPerBlock: ...


def per_block[K: BlockRule](
    rules: Mapping[K, HessianApprox[Any]], /
) -> PerBlockCurvature | StagedPerBlock:
    """A method per block; the first rule matching a block wins.

    Every block must match a rule and every rule some block, else `init` / `plan` raises
    `StructureError`. Each rule is one member: it runs once per batch over all its blocks,
    with its own update generator.

    Args:
        rules: `{rule: method}`. A rule is a glob over block names, or a tuple of exact
            block names (each must be a block).

    Returns:
        A `PerBlockCurvature` when all members are single-pass; a `StagedPerBlock` when any member
        is multi-pass. When every member has a template (`HasTemplate`), so does the
        result: the members' templates, each under its blocks.
    """

    own: dict[BlockRule, HessianApprox[Any]] = dict(rules.items())
    templated = _all_templated(own)

    if any(isinstance(m, Staged) for m in own.values()):
        return _TemplatedStagedPerBlock(own) if templated else StagedPerBlock(own)

    curvatures = cast(Mapping[BlockRule, Curvature[Any, Any]], own)

    return _TemplatedPerBlock(curvatures) if templated else PerBlockCurvature(curvatures)
