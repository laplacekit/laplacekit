"""The subset protocol, and index-set subnetworks (`Selection`)."""

import dataclasses
import math
from abc import abstractmethod
from collections.abc import Mapping
from typing import Final, Protocol

import torch
from torch import Tensor

from laplacekit.core.model import ModelFn
from laplacekit.core.types import Capabilities, Inputs, Outputs, Params, StructureError

_SELECTED_SUFFIX: Final = "[idx]"
"""Appended to a parameter's name to name its selected entries: `"fc.weight[idx]"`."""


@dataclasses.dataclass(frozen=True, eq=False)
class Selection:
    """Some entries of some parameters, each turned into its own 1-D parameter.

    `params` holds the selected values under `"<name>[idx]"`, and `apply(model)` scatters
    them back into the full tensors.

    `base` holds live views of the parameters, so the entries left out follow the module
    when it changes. Training the module after fitting is caught by the posterior's
    `check_versions` on `outputs`, `predict` and `sample`, since every module parameter is
    a fixed parameter of a posterior over a `Selection`.

    A `Selection` reads only the tensors it was given.

    Args:
        base: Each source name mapped to the full tensor, a detached view of the given
            parameter.
        index: Each source name mapped to its selected flat (row-major) indices: sorted,
            unique, non-empty int64.

    Raises:
        StructureError: `base` and `index` have different names or order, nothing is
            selected, or an index is not 1-D int64, is empty, is not sorted and unique, or
            is out of range for its parameter.
    """

    base: Params
    index: Mapping[str, Tensor]

    def __post_init__(self) -> None:
        if list(self.base) != list(self.index):
            raise StructureError(
                f"Selection: base and index need the same names in the same order, got "
                f"{list(self.base)} and {list(self.index)}"
            )

        if not self.index:
            raise StructureError("Selection: no entries selected")

        for name, idx in self.index.items():
            size = math.prod(self.base[name].shape)

            if idx.dtype != torch.int64 or idx.dim() != 1:
                raise StructureError(
                    f"Selection: the index of {name!r} must be a 1-D int64 tensor, got "
                    f"{idx.dtype} with shape {tuple(idx.shape)}"
                )

            if idx.numel() == 0:
                raise StructureError(f"Selection: the index of {name!r} must be non-empty")

            if bool((idx[1:] <= idx[:-1]).any()):
                raise StructureError(f"Selection: the index of {name!r} must be sorted and unique")

            if int(idx[0]) < 0 or int(idx[-1]) >= size:
                raise StructureError(
                    f"Selection: the index of {name!r} is out of range for a parameter with "
                    f"{size} entries"
                )

    @staticmethod
    def from_masks(params: Params, masks: Mapping[str, Tensor], /) -> "Selection":
        """The entries where each bool mask is `True`.

        Masks with no `True` are dropped.

        Args:
            params: The parameters. The result follows their order, so the selected
                parameters stay grouped by module.
            masks: Parameter name mapped to a bool mask shaped like its parameter.

        Returns:
            The selection.

        Raises:
            StructureError: A mask names a parameter not in `params`, is not bool, has
                another shape than its parameter, or no mask selects an entry.
        """

        unknown = [n for n in masks if n not in params]

        if unknown:
            raise StructureError(f"Selection.from_masks: {unknown} are not in params")

        base: dict[str, Tensor] = {}
        index: dict[str, Tensor] = {}

        for name, p in params.items():
            if name not in masks:
                continue

            mask = masks[name]

            if mask.dtype != torch.bool:
                raise StructureError(
                    f"Selection.from_masks: the mask of {name!r} must be bool, got {mask.dtype}"
                )

            if mask.shape != p.shape:
                raise StructureError(
                    f"Selection.from_masks: the mask of {name!r} has shape "
                    f"{tuple(mask.shape)}, the parameter {tuple(p.shape)}"
                )

            idx = mask.reshape(-1).nonzero().reshape(-1).to(p.device)

            if idx.numel() > 0:
                base[name] = p.detach()
                index[name] = idx

        if not index:
            raise StructureError(
                f"Selection.from_masks: the masks of {list(masks)} select no entry"
            )

        return Selection(base=base, index=index)

    @property
    def params(self) -> Params:
        """Selected name mapped to a copy of the selected entries, in `base` order."""

        return {
            f"{src}{_SELECTED_SUFFIX}": full.reshape(-1)[self.index[src]]
            for src, full in self.base.items()
        }

    @property
    def selected_index(self) -> Mapping[str, Tensor]:
        """Selected name mapped to its flat indices, keyed like `params`."""

        return {f"{src}{_SELECTED_SUFFIX}": idx for src, idx in self.index.items()}

    def masks(self) -> dict[str, Tensor]:
        """The inverse of `from_masks`, for saving a selection.

        Returns:
            Source name mapped to a bool mask shaped like the full tensor.
        """

        out: dict[str, Tensor] = {}

        for src, full in self.base.items():
            flat = torch.zeros(full.numel(), dtype=torch.bool, device=full.device)
            out[src] = flat.index_fill(0, self.index[src], True).reshape(full.shape)

        return out

    def apply(self, model: ModelFn, /) -> ModelFn:
        """`model` called with the selected parameters scattered back into the full tensors.

        Names without the suffix pass through unchanged. The scatter is out of place, so the
        result differentiates in the selected values under `torch.func` transforms.

        Args:
            model: The model over the full parameters.

        Returns:
            A model over the selected parameters (plus any unsuffixed ones), with
            `model.module`. When called, it raises `StructureError` when a suffixed name is not
            selected, a value has the wrong shape or dtype, or a parameter comes both whole
            and selected.
        """

        base, index = self.base, self.index

        def forward(params: Params, inputs: Inputs) -> Outputs:
            rest: dict[str, Tensor] = {}
            full: dict[str, Tensor] = {}

            for name, values in params.items():
                src = name.removesuffix(_SELECTED_SUFFIX)

                if src == name:
                    rest[name] = values
                    continue

                if src not in index:
                    raise StructureError(
                        f"Selection.apply: {name!r} is not selected; selected are "
                        f"{[f'{s}{_SELECTED_SUFFIX}' for s in index]}"
                    )

                # Under vmap, `values.shape` is the per-example shape.
                shape, dtype = tuple(index[src].shape), base[src].dtype

                if tuple(values.shape) != shape or values.dtype != dtype:
                    raise StructureError(
                        f"Selection.apply: {name!r} needs shape {shape} and dtype {dtype}, "
                        f"got {tuple(values.shape)} and {values.dtype}"
                    )

                fixed = base[src].detach().reshape(-1)
                full[src] = fixed.scatter(0, index[src], values).reshape(base[src].shape)

            both = [n for n in full if n in rest]

            if both:
                raise StructureError(
                    f"Selection.apply: got both {both[0]!r} and {f'{both[0]}{_SELECTED_SUFFIX}'!r}"
                )

            return model({**rest, **full}, inputs)

        return ModelFn(forward, model.module)


class Subset(Protocol):
    """Chooses which parameters get a posterior.

    The choice is whole parameters (`Params`), or some of their entries (`Selection`).
    """

    @abstractmethod
    def __call__(self, model: ModelFn, caps: Capabilities, /) -> Params | Selection:
        """Choose the parameters.

        Args:
            model: The model.
            caps: What the probe found.

        Returns:
            Whole parameters, or a `Selection` of some of their entries.
        """
        ...


def select(model: ModelFn, chosen: Params | Selection, /) -> tuple[ModelFn, Params]:
    """The model and parameters to fit.

    Args:
        model: The model.
        chosen: What a `Subset` returned.

    Returns:
        `(model, chosen)` for `Params`, and `(chosen.apply(model), chosen.params)` for a
        `Selection`.
    """

    if isinstance(chosen, Selection):
        return chosen.apply(model), chosen.params

    return model, chosen
