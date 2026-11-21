"""Reads of torch's private functorch state.

torch has no public API for these, so they live here, each guarded by a test in
`tests/core/test_fit.py` or `tests/core/test_seed.py`.
"""

from contextlib import AbstractContextManager

import torch
from torch import Tensor


def transformed() -> bool:
    """Whether a `torch.func` transform (`vmap`, `grad`, ...) is running.

    Tensors under one are wrapped and have no storage. This reads the private functorch
    interpreter stack (guarded by `test_transformed_reads_the_functorch_interpreter_stack`).

    Returns:
        True inside a transform.
    """

    return torch._C._functorch.peek_interpreter_stack() is not None


def is_transformed(t: Tensor, /) -> bool:
    """Whether `t` is wrapped by a `torch.func` transform (a `vmap`, `grad` or `jvp` level).

    A wrapped tensor's values can't be read, so a check on its data (`bool(t.any())`) must be
    skipped. This reads the private `torch._C._functorch.is_functorch_wrapped_tensor`
    (guarded by `test_is_transformed_reads_functorch_wrappers`).

    Args:
        t: The tensor. One created outside the transform and only closed over is not
            wrapped.

    Returns:
        True when `t` is wrapped.
    """

    return torch._C._functorch.is_functorch_wrapped_tensor(t)


def outside_transforms() -> AbstractContextManager[None]:
    """A block that `torch.func` transforms don't see, for drawing a seed under `vmap`.

    `vmap` refuses random operations by default. A seed is drawn once per call, not per
    vmapped example, so it is drawn with functorch disabled. This uses the private
    `torch._C._DisableFuncTorch` (guarded by `test_draw_works_under_transforms`).

    Returns:
        A context manager.
    """

    return torch._C._DisableFuncTorch()
