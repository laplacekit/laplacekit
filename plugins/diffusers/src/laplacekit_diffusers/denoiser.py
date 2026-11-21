"""Wrap a `diffusers` denoiser as a `ModelFn`."""

from collections.abc import Mapping

from diffusers.utils import BaseOutput
from torch import nn
from torch.func import functional_call

import laplacekit as lk
from laplacekit.core import (
    Inputs,
    Outputs,
    Params,
)

__all__ = ["denoiser"]


_POSITIONAL = ("sample", "timestep")


def denoiser(unet: nn.Module, /) -> lk.core.ModelFn:
    """Wrap a `diffusers` denoiser as a `ModelFn` that returns the output's `.sample`.

    Args:
        unet: The denoiser.

    Returns:
        A `ModelFn` whose `forward(params, inputs)` calls
        `unet(inputs["sample"], inputs["timestep"], **rest)`, where `rest` holds the other
        entries of `inputs` (`class_labels`, `encoder_hidden_states`, ...). The sample
        and the timestep go in by position, because a DiT calls its first argument
        `hidden_states`. Parameters missing from `params` stay fixed and are passed
        detached. A call raises `StructureError` when the inputs are not a mapping, lack
        `sample` or `timestep`, or the output is not a `BaseOutput` with a `sample`.
    """

    def forward(params: Params, inputs: Inputs) -> Outputs:
        if not isinstance(inputs, Mapping):
            raise lk.StructureError(
                f"lk.diffusers.model needs inputs {{'sample', 'timestep', ...}}, got "
                f"{type(inputs).__name__}"
            )

        missing = [k for k in _POSITIONAL if k not in inputs]

        if missing:
            raise lk.StructureError(f"lk.diffusers.model needs {', '.join(missing)} in the inputs")

        # As in `ModelFn.from_module`: a tensor passed under one name reaches its other names
        # through functional_call's tying, so none of its names goes into `fixed`.
        passed = {id(p) for k, p in unet.named_parameters(remove_duplicate=False) if k in params}
        fixed = {k: v.detach() for k, v in unet.named_parameters() if id(v) not in passed}
        rest = {k: v for k, v in inputs.items() if k not in _POSITIONAL}

        out = functional_call(
            unet, {**fixed, **params}, (inputs["sample"], inputs["timestep"]), rest
        )

        if not isinstance(out, BaseOutput) or "sample" not in out:
            raise lk.StructureError(
                f"lk.diffusers.model needs a model that returns a diffusers BaseOutput with a "
                f"`sample`, got {type(out).__name__}"
            )

        return out["sample"]

    return lk.core.ModelFn(forward, unet)
