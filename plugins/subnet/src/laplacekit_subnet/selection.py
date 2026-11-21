r"""The subnetwork strategies of Daxberger et al. 2021, which choose the `n` weight entries to keep.

`random` draws them uniformly, `largest_magnitude` takes the largest $|\theta|$, and
`largest_variance` takes the largest marginal variances, from a posterior or from given
variances such as those of `swag_variance`. Each returns a `Subset` that gives a
`Selection` of the picked entries. Ties at the cut go to the lower index.
"""

from collections.abc import Mapping

import torch
from torch import Tensor, nn

import laplacekit as lk
from laplacekit.core import (
    Capabilities,
    ModelFn,
    Params,
    Selection,
    Subset,
    WeightPosterior,
)

__all__ = ["largest_magnitude", "largest_variance", "random"]


def random(
    n: int, /, *, generator: torch.Generator | None = None, subset: Subset = lk.subsets.all
) -> Subset:
    """Pick `n` entries of the parameters of `of`, uniformly without replacement.

    This is laplace-torch's `RandomSubnetMask` (Daxberger et al. 2021), with other random
    draws.

    Args:
        n: The number of entries to pick, from 1 to the number of entries of the
            parameters of `of`.
        generator: Seeds the draw. `random` draws one seed from it when it is called,
            or from torch's global RNG when `None`, so `torch.manual_seed` makes the pick
            repeatable. The returned `Subset` picks the same entries every time it is
            called, and the `generator=` of the fit does not change them.
        subset: The subset to pick from. It must give whole parameters, not a `Selection`.

    Returns:
        The `Subset`, which gives a `Selection` of the picked entries. A call raises
        `StructureError` when `of` gives a `Selection` or refuses the model (as
        `lk.subsets.all` does without a module), and `ValueError` when `n` is not in
        $[1, D]$ for the $D$ entries of the parameters of `of`.
    """

    # Drawn now, so every call of the subset picks the same entries.
    seed = lk.core.spawn_generator(generator).initial_seed()

    def pick(model: ModelFn, caps: Capabilities, /) -> Selection:
        params = _whole(subset, model, caps, "random")
        size = sum(p.numel() for p in params.values())
        _check_n(n, size, "random")
        scores = torch.rand(size, generator=torch.Generator().manual_seed(seed))

        return _top(params, scores, n)

    return pick


def largest_magnitude(n: int, /, *, subset: Subset = lk.subsets.all) -> Subset:
    r"""Pick the `n` entries of the parameters of `of` with the largest $|\theta|$.

    This is laplace-torch's `LargestMagnitudeSubnetMask` (Daxberger et al. 2021), and it
    picks the same entries: the first `n` of a descending `argsort`. Ties go to the lower
    index.

    Args:
        n: The number of entries to pick, from 1 to the number of entries of the
            parameters of `of`.
        subset: The subset to pick from. It must give whole parameters, not a `Selection`.

    Returns:
        The `Subset`, which gives a `Selection` of the picked entries. A call raises
        `StructureError` when `of` gives a `Selection` or refuses the model (as
        `lk.subsets.all` does without a module), and `ValueError` when `n` is not in
        $[1, D]$ for the $D$ entries of the parameters of `of`.
    """

    def pick(model: ModelFn, caps: Capabilities, /) -> Selection:
        params = _whole(subset, model, caps, "largest_magnitude")
        _check_n(n, sum(p.numel() for p in params.values()), "largest_magnitude")
        scores = torch.cat([p.detach().reshape(-1).abs() for p in params.values()])

        return _top(params, scores, n)

    return pick


def largest_variance(source: WeightPosterior | Params, n: int, /) -> Subset:
    """Pick the `n` entries with the largest marginal variance.

    This covers laplace-torch's `LargestVarianceDiagLaplaceSubnetMask` and
    `LargestVarianceSWAGSubnetMask` (Daxberger et al. 2021). Ties at the cut go to the
    lower index.

    Args:
        source: A posterior, usually a `Diag` fit over `lk.subsets.all`, or a mapping from
            parameter names to variances, such as `swag_variance` returns. Its names and
            shapes must be those of the module's parameters, so a posterior over a
            `Selection` raises `StructureError` when the subset is called.
        n: The number of entries to pick, from 1 to the number of entries of `source`.

    Returns:
        The `Subset`, which gives a `Selection` of the picked entries. A call raises
        `StructureError` when the model has no module, a name in `source` is not a
        parameter of the module, or a variance has another shape than its parameter.

    Raises:
        ValueError: `n` is not in $[1, D]$ for the $D$ entries of `source`.
        StructureError: `source` is a posterior whose `variance` raises it, for
            example because its precision is not positive definite.
    """

    if isinstance(source, Mapping):
        var = dict(source)
        what = "variances"
    else:
        var = dict(source.variance)
        var = {name: var[name] for name in source.mean}
        what = "posterior's"

    names = list(var)
    size = sum(var[name].numel() for name in names)
    _check_n(n, size, "largest_variance")
    scores = torch.cat([var[name].detach().reshape(-1) for name in names])

    def pick(model: ModelFn, caps: Capabilities, /) -> Selection:
        own = dict(_module(model, "largest_variance").named_parameters(remove_duplicate=False))
        missing = [name for name in names if name not in own]

        if missing:
            raise lk.StructureError(
                f"subnet.largest_variance: the {what} {missing} are not parameters of the "
                "module; give whole parameters, not a Selection"
            )

        params = {name: own[name] for name in names}

        for name in names:
            if var[name].shape != params[name].shape:
                raise lk.StructureError(
                    f"subnet.largest_variance: the variance of {name!r} has shape "
                    f"{tuple(var[name].shape)}, the parameter {tuple(params[name].shape)}"
                )

        return _top(params, scores, n)

    return pick


def _module(model: ModelFn, subset: str) -> nn.Module:
    if model.module is None:
        raise lk.StructureError(f"subnet.{subset} needs a module, but ModelFn.module is None")

    return model.module


def _whole(source: Subset, model: ModelFn, caps: Capabilities, what: str) -> Params:
    params = source(model, caps)

    if isinstance(params, Selection):
        raise lk.StructureError(
            f"subnet.{what}: subset= must give whole parameters (Params), got a Selection"
        )

    return params


def _check_n(n: int, size: int, subset: str) -> None:
    if not 1 <= n <= size:
        raise ValueError(f"subnet.{subset}: n must be in [1, {size}], the entry count, got {n}")


def _top(params: Params, scores: Tensor, n: int) -> Selection:
    """The `n` largest scores, flattened over `params` in mapping order, as a `Selection`."""

    # As laplace-torch picks them, with ties at the cut going to the lower index.
    picked = torch.argsort(scores, descending=True, stable=True)[:n].sort().values
    flat = torch.zeros(scores.numel(), dtype=torch.bool)
    flat[picked.cpu()] = True

    masks: dict[str, Tensor] = {}
    start = 0

    for name, p in params.items():
        masks[name] = flat[start : start + p.numel()].reshape(p.shape).to(p.device)
        start += p.numel()

    return Selection.from_masks(params, masks)
