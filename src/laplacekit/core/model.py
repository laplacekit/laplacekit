"""The model boundary. The library never looks inside `forward`."""

import dataclasses
from collections.abc import Callable, Mapping

from torch import Tensor, nn
from torch.func import functional_call

from laplacekit.core.types import Inputs, Outputs, Params


def tied_names(module: nn.Module, /) -> dict[str, tuple[str, ...]]:
    """Names of tied parameters.

    Args:
        module: The module.

    Returns:
        Each parameter name of a tensor with several names, mapped to all its names.
        Tensors with one name are left out.
    """

    names: dict[int, list[str]] = {}

    for k, p in module.named_parameters(remove_duplicate=False):
        names.setdefault(id(p), []).append(k)

    return {k: tuple(ns) for ns in names.values() if len(ns) > 1 for k in ns}


@dataclasses.dataclass(frozen=True)
class ModelFn:
    """`forward(params, inputs) -> outputs`, plus the module when there is one.

    `params` holds the parameters that get a posterior; every other parameter is
    read from `module` at call time. The module is needed only by hook-based
    backends and for the parameter-version guard.

    A tied tensor is one parameter. Passing it under any of its names (`head.weight`
    from `last_layer`, or `emb.weight` from `subsets.all`) swaps every name, so all its
    uses are differentiated. Pass one name per tensor.

    Args:
        forward: `forward(params, inputs) -> outputs`.
        module: The module, needed only by hook-based backends and for the
            parameter-version guard. `None` when there is none.
    """

    forward: Callable[[Params, Inputs], Outputs]
    module: nn.Module | None = None

    def __call__(self, params: Params, inputs: Inputs) -> Outputs:
        """Run `forward`.

        Args:
            params: The parameters that get a posterior.
            inputs: Inputs.

        Returns:
            The outputs.
        """

        return self.forward(params, inputs)

    @staticmethod
    def from_module(module: nn.Module) -> "ModelFn":
        """Wrap a module with `torch.func.functional_call`.

        Only the passed `params` are variables. The module's other parameters are passed
        detached, so the outputs never depend on them through autograd.

        Args:
            module: The module. It is called as `module(x)` for tensor inputs and
                `module(**x)` for mapping inputs.

        Returns:
            A `ModelFn` holding `module`. When called, it raises `TypeError` for inputs that
            are neither a tensor nor a mapping.
        """

        def forward(params: Params, inputs: Inputs) -> Outputs:
            # Only the passed params are variables. The rest of the module is fixed, so it is
            # passed detached and outputs never depend on it through autograd. A tensor
            # passed under one name reaches its other names through functional_call's
            # tying, so none of its names goes into `fixed`.
            passed = {
                id(p) for k, p in module.named_parameters(remove_duplicate=False) if k in params
            }
            fixed = {k: v.detach() for k, v in module.named_parameters() if id(v) not in passed}
            merged = {**fixed, **params}

            if isinstance(inputs, Tensor):
                return functional_call(module, merged, (inputs,))

            if isinstance(inputs, Mapping):
                return functional_call(module, merged, (), dict(inputs))

            raise TypeError(f"from_module takes a tensor or a mapping, got {type(inputs).__name__}")

        return ModelFn(forward, module)
