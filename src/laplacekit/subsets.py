"""Built-in subsets: which parameters get a posterior.

Each subset returns the module's own parameters by name, or a `Selection` of their entries
(a subnetwork, as in Daxberger et al. 2021).

Random, largest-magnitude and largest-variance subnetworks are in the `laplacekit-subnet`
plugin, as `lk.subnet`.
"""

import fnmatch
from collections.abc import Mapping

from torch import Tensor, nn

from laplacekit.core import (
    ROOT_BLOCK,
    Capabilities,
    CapabilityError,
    ModelFn,
    Params,
    Selection,
    StructureError,
    Subset,
)

__all__ = ["all", "from_mask", "last_layer", "named", "trainable"]


def _module(model: ModelFn, subset: str) -> nn.Module:
    if model.module is None:
        raise StructureError(f"subsets.{subset} needs a module, but ModelFn.module is None")

    return model.module


def all(model: ModelFn, caps: Capabilities, /) -> Params:  # noqa: A001
    """Every parameter of the module.

    A tied tensor is listed once, under its first name, and all its uses are differentiated
    (see `ModelFn.from_module`).

    Args:
        model: The model. It needs a module.
        caps: The probe's capabilities (unused).

    Returns:
        The parameters by name.

    Raises:
        StructureError: `model.module` is `None`.
    """

    return dict(_module(model, "all").named_parameters(remove_duplicate=True))


def trainable(model: ModelFn, caps: Capabilities, /) -> Params:
    """Parameters with `requires_grad=True`, e.g. LoRA adapters.

    Args:
        model: The model. It needs a module.
        caps: The probe's capabilities (unused).

    Returns:
        The parameters by name. A tied tensor is listed once, under its first name.

    Raises:
        StructureError: `model.module` is `None`.
    """

    module = _module(model, "trainable")

    return {n: p for n, p in module.named_parameters(remove_duplicate=True) if p.requires_grad}


def last_layer(model: ModelFn, caps: Capabilities, /) -> Params:
    """Own parameters of the last parameterized module that ran in the probe.

    Args:
        model: The model. It needs a module.
        caps: The probe's capabilities. `caps.module_order` gives the last module;
            `ROOT_BLOCK` stands for the root module.

    Returns:
        The module's own parameters, by their full names.

    Raises:
        StructureError: `model.module` is `None`, or it has no submodule by the name the
            probe recorded.
        CapabilityError: The probe found no parameterized module.
    """

    module = _module(model, "last_layer")

    if not caps.module_order:
        raise CapabilityError("subsets.last_layer: the probe found no parameterized module")

    name = caps.module_order[-1]

    if name == ROOT_BLOCK:
        return dict(module.named_parameters(recurse=False))

    try:
        last = module.get_submodule(name)
    except AttributeError as e:
        raise StructureError(f"subsets.last_layer: no module named {name!r}") from e

    return {f"{name}.{n}": p for n, p in last.named_parameters(recurse=False)}


def named(*patterns: str) -> Subset:
    """Parameters whose name matches any of the glob patterns.

    Whole tensors, as laplace-torch's `ParamNameSubnetMask` (exact names) and
    `ModuleNameSubnetMask` (a module's parameters, `"name.*"`) select them.

    Args:
        *patterns: Glob patterns over parameter names, e.g. `"0.*"`, matched with
            `fnmatch.fnmatchcase`.

    Returns:
        The `Subset`. A tied tensor is listed once, under its first name. When called, it
        raises `StructureError` when the model has no module or no parameter matches.
    """

    def subset(model: ModelFn, caps: Capabilities, /) -> Params:
        module = _module(model, "named")
        out = {
            n: p
            for n, p in module.named_parameters(remove_duplicate=True)
            if any(fnmatch.fnmatchcase(n, pat) for pat in patterns)
        }

        if not out:
            raise StructureError(f"subsets.named{patterns}: no parameter matches")

        return out

    return subset


def from_mask(masks: Mapping[str, Tensor], /) -> Subset:
    """The entries where each bool mask is `True`.

    This is laplace-torch's `subnetwork_indices`, given as masks.

    Args:
        masks: Module parameter name mapped to a bool mask shaped like its parameter. A
            mask under a tied alias goes to the name `subsets.all` lists.

    Returns:
        The `Subset`, which gives a `Selection`. When called, it raises `StructureError`
        when the model has no module, a name is not a parameter of the module, two masks fall
        on one tied tensor, or `Selection.from_masks` refuses the masks (not bool, another
        shape, no entry selected).
    """

    def subset(model: ModelFn, caps: Capabilities, /) -> Selection:
        module = _module(model, "from_mask")
        own = dict(module.named_parameters(remove_duplicate=False))
        unknown = [name for name in masks if name not in own]

        if unknown:
            raise StructureError(f"subsets.from_mask: {unknown} are not parameters of the module")

        listed = all(model, caps)
        first = {id(p): name for name, p in listed.items()}
        moved: dict[str, Tensor] = {}

        for name, mask in masks.items():
            target = first[id(own[name])]

            if target in moved:
                raise StructureError(
                    f"subsets.from_mask: two masks fall on the tied tensor {target!r}; give one"
                )

            moved[target] = mask

        return Selection.from_masks(listed, moved)

    return subset
