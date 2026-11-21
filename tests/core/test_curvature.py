"""Curvature templates: a data-free zero operator with `finalize`'s structure."""

import copy
from typing import Any

import pytest
import torch
from torch import Tensor

from laplacekit.core import (
    BlockDiag,
    Capabilities,
    Dense,
    Diag,
    Done,
    FitContext,
    HasTemplate,
    Joint,
    Layout,
    ModelFn,
    Params,
    Pass,
    PerBlockCurvature,
    Requirements,
    StagedPerBlock,
    StructureError,
    chain,
    per_block,
    pytree,
    static,
)
from laplacekit.likelihoods import Gaussian


def _setup() -> tuple[Params, FitContext]:
    """Three blocks of sizes 2, 3 and 4; the model is never called."""

    params = {
        "a.weight": torch.zeros(2),
        "b.weight": torch.zeros(3),
        "c.weight": torch.zeros(2, 2),
    }
    model = ModelFn(lambda p, x: x)
    ctx = FitContext(model, Gaussian(), Layout.by_module(params), Capabilities())

    return params, ctx


@pytree
class _NoTemplate:
    """A per-block diagonal curvature without a template (its passes are never run)."""

    @property
    def requires(self) -> Requirements:
        return Requirements()

    def init(self, params: Params, ctx: FitContext, /) -> Any:
        raise NotImplementedError

    def update(self, stat: Any, params: Params, batch: Any, gen: Any, ctx: Any, /) -> Any:
        raise NotImplementedError

    def finalize(self, stat: Any, /) -> Any:
        raise NotImplementedError


@pytree
class _DiagTemplate(_NoTemplate):
    """Templates a `BlockDiag` of `Diag` over the blocks it sees, filled with `fill` (to tell
    members apart)."""

    fill: float = static(default=0.0)

    def template(self, params: Params, ctx: FitContext, /) -> BlockDiag[Diag]:
        sizes = {b: sum(params[n].numel() for n in ns) for b, ns in ctx.layout.blocks.items()}

        return BlockDiag({b: Diag(torch.full((n,), self.fill)) for b, n in sizes.items()})


@pytree
class _DenseTemplate(_NoTemplate):
    """Templates one `Dense` over everything it sees."""

    def template(self, params: Params, ctx: FitContext, /) -> Dense:
        n = sum(p.numel() for p in params.values())

        return Dense(torch.zeros(n, n))


@pytree
class _StagedTemplate:
    """A two-pass method with a template."""

    @property
    def requires(self) -> Requirements:
        return Requirements()

    def plan(self, params: Params, ctx: FitContext, /) -> Pass[Any]:
        return chain(_NoTemplate(), then=Done)

    def template(self, params: Params, ctx: FitContext, /) -> BlockDiag[Diag]:
        return _DiagTemplate(2.0).template(params, ctx)


@pytree
class _StagedNoTemplate:
    @property
    def requires(self) -> Requirements:
        return Requirements()

    def plan(self, params: Params, ctx: FitContext, /) -> Pass[Any]:
        return chain(_NoTemplate(), then=Done)


def _blocks(op: Any) -> dict[str, tuple[type, tuple[int, ...], Tensor]]:
    out = {}

    for b, blk in op.blocks.items():
        leaf = blk.diag if isinstance(blk, Diag) else blk.mat
        out[b] = (type(blk), tuple(leaf.shape), leaf)

    return out


def test_per_block_template() -> None:
    """Each member's template sits under its own blocks, built from its blocks alone."""

    params, ctx = _setup()
    pb = per_block({"b": _DenseTemplate(), "*": _DiagTemplate(1.0)})

    assert isinstance(pb, PerBlockCurvature)
    assert isinstance(pb, HasTemplate)

    got = pb.template(params, ctx)

    assert isinstance(got, BlockDiag)
    assert list(got.blocks) == ["a", "b", "c"]

    blocks = _blocks(got)

    assert blocks["a"][:2] == (Diag, (2,))
    assert blocks["b"][:2] == (Dense, (3, 3))
    assert blocks["c"][:2] == (Diag, (4,))
    assert bool((blocks["a"][2] == 1).all())
    assert bool((blocks["c"][2] == 1).all())


def test_per_block_has_no_template_when_a_member_lacks_one() -> None:
    pb = per_block({"b": _DenseTemplate(), "*": _NoTemplate()})

    assert isinstance(pb, PerBlockCurvature)
    assert not isinstance(pb, HasTemplate)
    assert repr(pb) == "PerBlockCurvature({'b': _DenseTemplate(), '*': _NoTemplate()})"


def test_per_block_with_template_keeps_its_repr_and_equality() -> None:
    a = per_block({"b": _DenseTemplate(), "*": _DiagTemplate()})
    b = per_block({"b": _DenseTemplate(), "*": _DiagTemplate()})

    assert a == b
    assert repr(a) == "PerBlockCurvature({'b': _DenseTemplate(), '*': _DiagTemplate()})"


def test_staged_per_block_template() -> None:
    params, ctx = _setup()
    spb = per_block({"b": _StagedTemplate(), "a": _DenseTemplate(), "*": _DiagTemplate(3.0)})

    assert isinstance(spb, StagedPerBlock)
    assert isinstance(spb, HasTemplate)

    got = spb.template(params, ctx)

    assert isinstance(got, BlockDiag)

    blocks = _blocks(got)

    assert list(blocks) == ["a", "b", "c"]
    assert blocks["a"][:2] == (Dense, (2, 2))
    assert blocks["b"][:2] == (Diag, (3,))
    assert blocks["c"][:2] == (Diag, (4,))
    assert bool((blocks["b"][2] == 2).all())
    assert bool((blocks["c"][2] == 3).all())


def test_staged_per_block_has_no_template_when_a_member_lacks_one() -> None:
    for rules in (
        {"b": _StagedNoTemplate(), "*": _DenseTemplate()},
        {"b": _StagedTemplate(), "*": _NoTemplate()},
    ):
        spb = per_block(rules)

        assert isinstance(spb, StagedPerBlock)
        assert not isinstance(spb, HasTemplate)


@pytree
class _JointTemplate(_NoTemplate):
    """Templates one `Joint` `Dense` over all its blocks."""

    def template(self, params: Params, ctx: FitContext, /) -> Joint[Dense]:
        sizes = tuple((b, ctx.layout.size(b)) for b in ctx.layout.blocks)
        n = sum(s for _, s in sizes)

        return Joint(Dense(torch.zeros(n, n)), sizes)


def test_per_block_joint_member_at_init() -> None:
    """A member whose template is a `Joint` is refused before any member starts (the
    members here raise `NotImplementedError` when they do)."""

    params, ctx = _setup()

    for pb in (
        PerBlockCurvature({"b": _JointTemplate(), "*": _NoTemplate()}),
        PerBlockCurvature({"*": _JointTemplate()}),
    ):
        with pytest.raises(StructureError, match=r"per_block member '(b|\*)' .*joint operator"):
            pb.init(params, ctx)

    staged = StagedPerBlock({"b": _StagedNoTemplate(), "*": _JointTemplate()})

    with pytest.raises(StructureError, match=r"per_block member '\*' .*joint operator"):
        staged.plan(params, ctx)


def test_replace_on_a_per_block_follows_the_new_rules() -> None:
    """`copy.replace` with new rules gives what `per_block` would build from them."""

    params, ctx = _setup()
    pb = per_block({"b": _DenseTemplate(), "*": _DiagTemplate()})
    plain = copy.replace(pb, rules={"b": _DenseTemplate(), "*": _NoTemplate()})

    assert type(plain) is PerBlockCurvature
    assert not isinstance(plain, HasTemplate)

    again = copy.replace(plain, rules={"*": _DiagTemplate(5.0)})

    assert (
        type(again) is PerBlockCurvature
    )  # a directly built PerBlockCurvature stays without a template

    templated = copy.replace(pb, rules={"*": _DiagTemplate(5.0)})

    assert isinstance(templated, HasTemplate)
    assert bool((templated.template(params, ctx).blocks["b"].diag == 5).all())

    spb = per_block({"b": _StagedTemplate(), "*": _DiagTemplate()})
    staged_plain = copy.replace(spb, rules={"b": _StagedTemplate(), "*": _NoTemplate()})

    assert type(staged_plain) is StagedPerBlock
    assert not isinstance(staged_plain, HasTemplate)

    staged = copy.replace(spb, rules={"b": _StagedTemplate(), "*": _DiagTemplate(4.0)})

    assert isinstance(staged, HasTemplate)
    assert bool((staged.template(params, ctx).blocks["c"].diag == 4).all())

    with pytest.raises(TypeError):
        copy.replace(pb, other=1)
