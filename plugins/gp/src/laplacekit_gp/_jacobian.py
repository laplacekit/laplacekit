"""Jacobian rows, and products of the Jacobian with a basis, for a model's flattened outputs.

Parameters are flattened in the order of `params`, and the P columns follow that order.
Outputs are flattened per example, so E is the number of output entries of one example.
"""

import math
from collections.abc import Mapping
from typing import Any, cast

import torch
from torch import Tensor
from torch.func import jvp, vjp, vmap

from laplacekit.core import (
    Capabilities,
    Inputs,
    ModelFn,
    Params,
    Strategy,
    StructureError,
    tree_map,
)
from laplacekit.function_space import is_shared, jacobian

__all__ = ["JVP_CHUNK", "VJP_CHUNK_ENTRIES", "choose_via", "features", "rows"]

JVP_CHUNK = 64
"""The number of basis columns that `features` pushes through one vmapped `jvp`."""

VJP_CHUNK_ENTRIES = 2**24
"""The most Jacobian entries (inputs times E times P) in one block of `features` with
`strategy="reverse-mode"`. The E P rows of a single input always go in one block."""


def choose_via(caps: Capabilities, /) -> Strategy:
    """Choose the mode for `features`: jvp when the probe found jvp and vmap, else vjp.

    Forward mode pushes the r basis columns through the model. Reverse mode builds all
    E P Jacobian rows of each input and multiplies them by the `[P, r]` basis, which costs
    E P memory and 2 E P r flops per input. Forward mode was faster on the models
    measured, even with r above E, so it is the default, as in laplace-torch's `ELLA`.
    Without jvp or without vmap, the choice is vjp. Without vmap, the method's check of
    its requirements then refuses the model.
    """

    return "forward-mode" if caps.jvp and caps.vmap else "reverse-mode"


def _in_dims(inputs: Inputs) -> tuple[Any, Any]:
    """Which leaves are shared by every input, and the matching `vmap` in_dims.

    Shared leaves are 0-dim tensors and non-tensors.
    """

    shared = tree_map(is_shared, inputs)
    in_dims = tree_map(lambda sh: None if sh else 0, shared)

    return shared, in_dims


def _flat(grads: Mapping[str, Tensor], names: list[str], lead: int) -> Tensor:
    """Concatenate per-parameter tensors, keeping `lead` leading axes."""

    return torch.cat([grads[k].reshape(*grads[k].shape[:lead], -1) for k in names], dim=lead)


def rows(
    model: ModelFn, params: Params, inputs: Inputs, coords: Tensor | None, dtype: torch.dtype
) -> Tensor:
    """Jacobian rows of the flattened outputs, `[m k, P]` for m inputs.

    With `coords=None`, it returns the rows of every output entry (k = E, row n E + e),
    like `fs.jacobian`. Otherwise `coords`, `[m]` int64, selects one flattened entry per
    input (k = 1), computed by a VJP with a one-hot cotangent per input, vmapped over the
    inputs.
    """

    if coords is None:
        return jacobian(model, params, inputs, dtype)

    names = list(params)
    p = dict(params)
    shared, in_dims = _in_dims(inputs)

    def lead(x: Inputs) -> Inputs:
        return tree_map(lambda t, sh: t if sh else t.unsqueeze(0), x, shared)

    def one(x: Inputs, c: Tensor) -> Tensor:
        xb = lead(x)

        def fwd(q: Params) -> Tensor:
            return cast(Tensor, model(q, xb)).reshape(-1)

        out, pull = cast(Any, vjp(fwd, p))
        hot = (torch.arange(out.shape[0], device=out.device) == c).to(out.dtype)

        return _flat(cast(Mapping[str, Tensor], pull(hot)[0]), names, 0)

    return vmap(one, in_dims=(in_dims, 0))(inputs, coords).to(dtype)


def features(
    model: ModelFn,
    params: Params,
    inputs: Inputs,
    basis: Tensor,
    strategy: Strategy,
    dtype: torch.dtype,
) -> tuple[Tensor, Tensor]:
    """The outputs f and Φ = J(x) basis, `[N, E, r]`, for a `[P, r]` basis.

    `strategy="forward-mode"` pushes the basis columns forward, `JVP_CHUNK` at a time under `vmap`,
    after casting each column to its parameter's dtype. `strategy="reverse-mode"` takes the Jacobian
    rows (`rows`) of a few inputs at a time, up to `VJP_CHUNK_ENTRIES` entries or one
    input, and multiplies each block by the basis. Φ comes back in `dtype`, and f as the
    model returns it. A model whose outputs are not one tensor raises `StructureError`.
    """

    p = dict(params)
    out = model(p, inputs)

    if not isinstance(out, Tensor):
        raise StructureError(
            f"features need a model returning one tensor, got {type(out).__name__}"
        )

    f = out
    n, e, r = f.shape[0], math.prod(f.shape[1:]), basis.shape[1]

    if strategy == "reverse-mode":
        n_params = sum(v.numel() for v in p.values())
        step = max(1, VJP_CHUNK_ENTRIES // max(1, e * n_params))
        shared, _ = _in_dims(inputs)
        b = basis.to(dtype)
        blocks = []

        # The rows of `step` inputs at a time, each block projected before the next.
        for s in range(0, n, step):
            part = tree_map(lambda t, sh, s=s: t if sh else t[s : s + step], inputs, shared)
            blocks.append(rows(model, params, part, None, dtype) @ b)

        return f, torch.cat(blocks).reshape(n, e, r)

    names = list(p)
    sizes = [p[k].numel() for k in names]

    def fwd(q: Params) -> Tensor:
        return cast(Tensor, model(q, inputs))

    def push(column: Tensor) -> Tensor:
        parts = torch.split(column, sizes)
        tangent = {
            k: v.reshape(p[k].shape).to(p[k].dtype) for k, v in zip(names, parts, strict=True)
        }

        return cast(Tensor, jvp(fwd, (p,), (tangent,))[1]).reshape(n, e)

    cols = [vmap(push)(basis[:, s : s + JVP_CHUNK].T).to(dtype) for s in range(0, r, JVP_CHUNK)]

    return f, torch.cat(cols).permute(1, 2, 0)
