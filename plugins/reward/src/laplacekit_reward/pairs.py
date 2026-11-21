"""Preference pairs, the `pairwise` model wrapper, and the Bradley–Terry likelihood.

A `Pair` holds the two sides of a batch of pairs, and `pair` builds one. `pairwise` wraps
a reward model so that it scores a `Pair` as `[N, 2]`, with one call on both sides, and
single items as `[N, 1]`. `bradley_terry` is the likelihood over the two rewards, and
`prepare_pairs` turns `(first, second, label)` rows into a `Batch`.
"""

from collections.abc import Mapping, Sequence
from typing import Any, NamedTuple

import torch
from torch import Tensor, nn

import laplacekit as lk
from laplacekit.core import Inputs, Outputs, Params, StructureError

__all__ = ["Pair", "bradley_terry", "pair", "pairwise", "prepare_pairs"]


_PADDED = "tokenize both sides in one padded call"


class Pair(NamedTuple):
    """A batch of preference pairs, where `first[i]` and `second[i]` are the items of pair `i`.

    Each side is an input of the reward model: a tensor `[N, ...]`, or a mapping of
    tensors such as a tokenizer's output. A named tuple is a torch pytree, so `vmap` and
    the probe see the tensors of both sides. `pairwise` recognizes a `Pair` by its class,
    so a plain tuple does not count as one.

    Args:
        first: The first item of each pair.
        second: The second item of each pair.
    """

    first: Inputs
    second: Inputs


def pair(first: Inputs, second: Inputs, /) -> Pair:
    """Build `Pair(first, second)`.

    Args:
        first: The first item of each pair.
        second: The second item of each pair.

    Returns:
        The `Pair`.
    """

    return Pair(first, second)


def _side_shapes(what: str, a: Tensor, b: Tensor) -> None:
    if a.shape != b.shape:
        raise StructureError(
            f"Pair sides differ{what}: {tuple(a.shape)} and {tuple(b.shape)}; {_PADDED}"
        )


def _concat(first: Inputs, second: Inputs) -> Inputs:
    """Concatenate both sides along dim 0 into `[2N, ...]`, key by key for mappings."""

    if isinstance(first, Tensor) and isinstance(second, Tensor):
        _side_shapes("", first, second)

        return torch.cat([first, second])

    if isinstance(first, Mapping) and isinstance(second, Mapping):
        for k in first.keys() ^ second.keys():
            side = "first" if k in first else "second"

            raise StructureError(f"Pair sides differ at {k!r}: only {side} has it; {_PADDED}")

        both: dict[str, Tensor] = {}

        for k, a in first.items():
            b = second[k]

            if not isinstance(a, Tensor) or not isinstance(b, Tensor):
                raise StructureError(
                    f"Pair sides must hold tensors, got {k!r}: {type(a).__name__} and "
                    f"{type(b).__name__}"
                )

            _side_shapes(f" at {k!r}", a, b)
            both[k] = torch.cat([a, b])

        return both

    raise StructureError(
        f"Pair sides must both be tensors or both mappings of tensors, got "
        f"{type(first).__name__} and {type(second).__name__}"
    )


def _rewards(out: Outputs) -> Tensor:
    """The inner model's outputs as one reward per item, `[M]`."""

    if not isinstance(out, Tensor) or out.ndim == 0 or out[0:1].numel() > 1:
        shape = tuple(out.shape) if isinstance(out, Tensor) else type(out).__name__

        raise StructureError(f"a reward model returns one value per item, got outputs {shape}")

    return out.reshape(-1)


def pairwise(model: nn.Module | lk.core.ModelFn, /) -> lk.core.ModelFn:
    """Wrap a reward model as a `ModelFn` that scores pairs and single items.

    On a `Pair`, the two sides are concatenated along dim 0 and the inner model runs once
    on `[2N, ...]`, so module hooks see one call per layer. The result is `[N, 2]`, the
    rewards of the first and the second items.

    Any other input is one item per row and gets its reward, `[N, 1]`. That includes a
    plain dict with the keys `"first"` and `"second"`.

    Args:
        model: The reward model, as a module or a `ModelFn`, with one output per item.

    Returns:
        The `ModelFn`. Its module is the inner one, so the parameter names stay the same.
        A call raises `StructureError` when the inner output has more than one value per
        item, or when the two sides of a `Pair` are not both tensors or both mappings of
        tensors, or differ in shape. Mapping sides need the same keys and the same shape
        for each key, and the error names the key that differs.
    """

    inner = lk.core.ModelFn.from_module(model) if isinstance(model, nn.Module) else model

    def forward(params: Params, inputs: Inputs) -> Outputs:
        if isinstance(inputs, Pair):
            r = _rewards(inner(params, _concat(inputs.first, inputs.second)))
            n = r.shape[0] // 2

            return torch.stack([r[:n], r[n:]], -1)

        return _rewards(inner(params, inputs)).unsqueeze(-1)

    return lk.core.ModelFn(forward, inner.module)


def bradley_terry() -> lk.Categorical:
    r"""The Bradley–Terry likelihood over a pair's rewards `[N, 2]`, which is `lk.Categorical()`.

    Let $r = (r_1, r_2)$, where $r_1$ is the first item's reward at index 0 of the class
    axis. Then

    $$p(\text{first} \succ \text{second}) = \sigma(r_1 - r_2) = \operatorname{softmax}(r)_1,
    \qquad H = \sigma(r_1 - r_2) \, \sigma(r_2 - r_1)
    \begin{pmatrix} 1 & -1 \\ -1 & 1 \end{pmatrix},$$

    where $H$ is the output Hessian. It is the $\operatorname{diag}(q) - q q^\top$ of
    `Categorical` for two classes, with $q = \operatorname{softmax}(r)$. As in
    laplace-torch, the target is the index of the preferred item, 0 for the first.

    `lk.predict_outputs(post, x)` on single items gives each reward with its variance.
    `lk.predict(post, pair)` on a `Pair` uses the binary probit on $r_1 - r_2$ instead of
    the bridge. `lk.predict` on single items raises `CapabilityError`, since a single
    reward makes only one class.

    Returns:
        `lk.Categorical()`.
    """

    return lk.Categorical()


def prepare_pairs(raw: Sequence[Any], generator: torch.Generator, /) -> lk.Batch:
    """Turn a `(first, second, label)` row, as a `TensorDataset` of pairs gives it, into a `Batch`.

    Args:
        raw: The row `(first, second, label)`. `label` is a tensor, the index of the
            preferred item.
        generator: Unused: nothing is drawn.

    Returns:
        A `Batch` with inputs `pair(first, second)` and targets `label.long()`.

    Raises:
        StructureError: `raw` does not have 3 entries, or `label` is not a tensor.
    """

    del generator

    if len(raw) != 3:
        raise StructureError(f"prepare_pairs needs (first, second, label), got {len(raw)} entries")

    if not isinstance(raw[2], Tensor):
        raise StructureError(f"prepare_pairs needs a label tensor, got {type(raw[2]).__name__}")

    first, second, label = raw

    return lk.Batch(pair(first, second), label.long())
