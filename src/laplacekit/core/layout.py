"""How parameters map to blocks and to the flat vectors operators work on.

- A block is a group of parameter names. By default, one block per module:
  "fc.weight" and "fc.bias" form block "fc". Parameters of the root module (names
  without a dot) form one block, `ROOT_BLOCK`.
- Within a block, the flat vector is the block's parameters in the order the
  layout lists them, each flattened row-major. For a linear layer that order is
  weight then bias.
- Operators act on one block's flat vector. `BlockDiag` keys are block names.
"""

import dataclasses
import math
from collections.abc import Iterable, Mapping

import torch
from torch import Tensor

from laplacekit.core.types import ROOT_BLOCK, Params


@dataclasses.dataclass(frozen=True)
class Layout:
    """Parameter names grouped into blocks, and each parameter's shape.

    Args:
        blocks: Block name mapped to its parameter names, in flat order.
        shapes: Parameter name mapped to its shape.
    """

    blocks: Mapping[str, tuple[str, ...]]
    shapes: Mapping[str, tuple[int, ...]]

    @staticmethod
    def by_module(params: Params, /) -> "Layout":
        """One block per module prefix (everything before the last dot).

        Names without a dot go to `ROOT_BLOCK`.

        Args:
            params: Parameter mapping. Its order sets the block order and the order within
                each block.

        Returns:
            The layout, with each parameter's shape.
        """

        blocks: dict[str, list[str]] = {}

        for name in params:
            blocks.setdefault(name.rpartition(".")[0] or ROOT_BLOCK, []).append(name)

        return Layout(
            blocks={b: tuple(ns) for b, ns in blocks.items()},
            shapes={n: tuple(p.shape) for n, p in params.items()},
        )

    def restrict(self, blocks: Iterable[str], /) -> "Layout":
        """The layout of some blocks only.

        Args:
            blocks: Names of the blocks to keep, in the order the new layout lists them.

        Returns:
            A layout with those blocks and their parameters' shapes.

        Raises:
            KeyError: A name in `blocks` is not a block of this layout.
        """

        kept = {b: self.blocks[b] for b in blocks}
        names = [n for ns in kept.values() for n in ns]

        return Layout(blocks=kept, shapes={n: self.shapes[n] for n in names})

    def flatten(self, params: Params, block: str, /, *, lead: int = 0) -> Tensor:
        """The block's flat vector.

        Args:
            params: Parameter mapping holding the block's parameters.
            block: Block name.
            lead: Number of leading axes of each parameter that stay in front (a batch of
                parameter sets). The vectors run along the last axis.

        Returns:
            A tensor of shape `(*lead_shape, size(block))`.

        Raises:
            KeyError: `block` is not a block of this layout.
        """

        return torch.cat(
            [params[n].reshape(*params[n].shape[:lead], -1) for n in self.blocks[block]], dim=-1
        )

    def unflatten(self, vec: Tensor, block: str, /) -> Params:
        """Back to the block's parameters, the inverse of `flatten`.

        Args:
            vec: The block's flat vector along the last axis. Leading axes (a batch) stay
                in front.
            block: Block name.

        Returns:
            Parameter mapping of the block's names, each shaped `(*lead_shape, *shape)`.

        Raises:
            KeyError: `block` is not a block of this layout.
        """

        names = self.blocks[block]
        sizes = [math.prod(self.shapes[n]) for n in names]
        parts: tuple[Tensor, ...] = torch.split(vec, sizes, dim=-1)
        lead = vec.shape[:-1]

        return {n: p.reshape(*lead, *self.shapes[n]) for n, p in zip(names, parts, strict=True)}

    def flatten_all(self, params: Params, /, *, lead: int = 0) -> Mapping[str, Tensor]:
        """Every block's flat vector.

        Args:
            params: Parameter mapping holding every block's parameters.
            lead: Leading axes that stay in front, as in `flatten`.

        Returns:
            Block name mapped to its flat vector.
        """

        return {b: self.flatten(params, b, lead=lead) for b in self.blocks}

    def size(self, block: str, /) -> int:
        """Length of a block's flat vector.

        Args:
            block: Block name.

        Returns:
            The number of entries in the block's parameters.

        Raises:
            KeyError: `block` is not a block of this layout.
        """

        return sum(math.prod(self.shapes[n]) for n in self.blocks[block])

    @property
    def n_params(self) -> int:
        """Entries of every parameter in the layout ($P$)."""

        return sum(math.prod(s) for s in self.shapes.values())
