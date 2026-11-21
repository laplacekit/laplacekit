"""Tokenized preferences from a language-model dataset, as a `Batch` of pairs."""

from collections.abc import Mapping

import torch
from torch import Tensor

import laplacekit as lk
from laplacekit.core import StructureError
from laplacekit_reward.pairs import pair

__all__ = ["prepare_preferences"]


_SIDES = ("chosen", "rejected")


def prepare_preferences(
    raw: Mapping[str, Mapping[str, Tensor]], generator: torch.Generator, /
) -> lk.Batch:
    """Turn tokenized preferences `{"chosen": ..., "rejected": ...}` into a `Batch` of pairs.

    Args:
        raw: The batch. `raw["chosen"]` and `raw["rejected"]` are tokenizer outputs,
            `[N, T]`. Tokenize both sides in one padded call so that they have the same
            length. Other keys are dropped.
        generator: Unused: nothing is drawn.

    Returns:
        A `Batch` with inputs `pair(chosen, rejected)` and targets `zeros(N)`, because
        the chosen side comes first and is the preferred one.

    Raises:
        StructureError: A side is missing, or is not a non-empty mapping of tensors.
    """

    del generator

    missing = [k for k in _SIDES if k not in raw]

    if missing:
        raise StructureError(
            f"prepare_preferences needs {' and '.join(map(repr, missing))} in the batch"
        )

    sides = [_tokenized(k, raw[k]) for k in _SIDES]
    first = next(iter(sides[0].values()))
    targets = torch.zeros(first.shape[0], dtype=torch.long, device=first.device)

    return lk.Batch(pair(*sides), targets)


def _tokenized(name: str, side: object) -> dict[str, Tensor]:
    """A tokenizer output as a plain dict of tensors."""

    want = f"prepare_preferences needs {name!r} to be a tokenizer's output, a mapping of tensors"

    if not isinstance(side, Mapping):
        raise StructureError(f"{want}, got {type(side).__name__}")

    if not side:
        raise StructureError(f"{want}, got an empty mapping")

    for k, v in side.items():
        if not isinstance(v, Tensor):
            raise StructureError(f"{want}, got {type(v).__name__} at {k!r}")

    return dict(side)
