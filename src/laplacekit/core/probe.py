"""Capability probe: run the model once and record what works."""

import warnings
from collections.abc import Callable, Mapping
from functools import partial
from typing import cast

import torch
from torch import Tensor, nn
from torch.func import grad, jvp, vjp, vmap
from torch.utils.hooks import RemovableHandle

from laplacekit.core._pytree import tree_leaves, tree_map
from laplacekit.core.layout import ROOT_BLOCK
from laplacekit.core.likelihoods import HasOutputCheck, LikelihoodTree, log_prob
from laplacekit.core.model import ModelFn, tied_names
from laplacekit.core.subsets import _SELECTED_SUFFIX
from laplacekit.core.types import (
    NOT_PROBED,
    TRANSFORMS,
    Batch,
    Capabilities,
    Params,
    PyTree,
    StructureError,
)

_N_EXAMPLES = 2


def _total_log_prob(lik: LikelihoodTree, f: PyTree, y: PyTree) -> Tensor:
    if isinstance(lik, Mapping):
        fm, ym = cast(Mapping[str, PyTree], f), cast(Mapping[str, PyTree], y)
        return sum((_total_log_prob(lik[k], fm[k], ym[k]) for k in lik), torch.zeros(()))

    return log_prob(lik, f, y).sum()


def _check_outputs(lik: LikelihoodTree, f: PyTree) -> None:
    """Run each `HasOutputCheck` leaf of `lik` on its outputs.

    A likelihood mapping needs a mapping of outputs holding each of its keys, or
    `StructureError`.
    """

    if isinstance(lik, Mapping):
        keys = sorted(lik)

        if not isinstance(f, Mapping):
            raise StructureError(
                f"the likelihood is a mapping with keys {keys}, so the model must return a "
                f"mapping of outputs with those keys; got {type(f).__name__}"
            )

        fm = cast(Mapping[str, PyTree], f)
        missing = [k for k in keys if k not in fm]

        if missing:
            raise StructureError(
                f"the likelihood has keys {missing} that the model's outputs lack; the "
                f"outputs have keys {sorted(fm)}"
            )

        for k, sub in lik.items():
            _check_outputs(sub, fm[k])

        return

    if isinstance(lik, HasOutputCheck):
        try:
            lik.check_outputs(f)
        except (AttributeError, TypeError) as e:
            if isinstance(f, Tensor):
                raise

            raise _wrong_outputs(lik, f, "check_outputs", e) from e


def _check_log_prob(lik: LikelihoodTree, f: PyTree, y: PyTree) -> None:
    """Run each leaf's `log_prob` on its outputs and targets.

    A likelihood mapping needs a mapping of targets holding each of its keys. A leaf's
    failure on outputs that are not a tensor (a dict reaching a tensor likelihood) with an
    `AttributeError` or `TypeError` raises `StructureError` naming both. A `StructureError`
    or `ValueError` propagates, the latter with a note naming the likelihood and the output
    type; any other failure is left to the transforms, which record it.
    """

    if isinstance(lik, Mapping):
        keys = sorted(lik)

        if not isinstance(y, Mapping):
            raise StructureError(
                f"the likelihood is a mapping with keys {keys}, so the targets must be a "
                f"mapping with those keys; got {type(y).__name__}"
            )

        ym = cast(Mapping[str, PyTree], y)
        missing = [k for k in keys if k not in ym]

        if missing:
            raise StructureError(
                f"the likelihood has keys {missing} that the targets lack; the targets have "
                f"keys {sorted(ym)}"
            )

        fm = cast(Mapping[str, PyTree], f)

        for k, sub in lik.items():
            _check_log_prob(sub, fm[k], ym[k])

        return

    try:
        log_prob(lik, f, y)
    except StructureError:
        raise
    except ValueError as e:
        # A value mistake of the caller (targets out of the support), or a torch
        # distribution refusing a non-tensor argument: kept as is, with what was probed.
        e.add_note(
            f"raised by {type(lik).__name__}.log_prob in the probe, on the model's outputs "
            f"of type {type(f).__name__}"
        )
        raise
    except (AttributeError, TypeError) as e:
        if not isinstance(f, Tensor):
            raise _wrong_outputs(lik, f, "log_prob", e) from e
    except Exception:  # noqa: BLE001
        pass  # not about the outputs' type: the transforms record it


def _wrong_outputs(lik: object, f: PyTree, where: str, e: Exception) -> StructureError:
    name = type(lik).__name__
    hint = (
        " A model returning a mapping needs a likelihood mapping with the same keys."
        if isinstance(f, Mapping)
        else ""
    )

    return StructureError(
        f"{name}.{where} failed on the model's outputs of type {type(f).__name__} "
        f"({_failure(e)}); does {name} take this output type?{hint}"
    )


def _total_output(f: PyTree) -> Tensor:
    """Sum of every output entry: a scalar to differentiate when there are no targets."""

    return sum((t.sum() for t in tree_leaves(f)), torch.zeros(()))


def _dot(a: PyTree, b: PyTree) -> Tensor:
    """Sum of the elementwise products of two matching trees, in float64."""

    return sum(
        (
            (x.double() * y.double()).sum()
            for x, y in zip(tree_leaves(a), tree_leaves(b), strict=True)
        ),
        torch.zeros((), dtype=torch.float64),
    )


def _first(t: Tensor, n: int) -> Tensor:
    return t[:n].detach()


def _unsqueeze(t: Tensor) -> Tensor:
    return t.unsqueeze(0)


def _failure(e: Exception) -> str:
    return f"{type(e).__name__}: {e}".splitlines()[0]


def _event_size(out: PyTree, n: int) -> int:
    """Output entries per example, over all leaves."""

    return sum(t.numel() // n for t in tree_leaves(out))


def _module_order(
    module: nn.Module, run: Callable[[], object]
) -> tuple[tuple[str, ...], dict[str, type[nn.Module]]]:
    """Forward order of parameterized modules, and the types of Linear/Conv2d that fired.

    The root module, when it owns parameters, comes first as `ROOT_BLOCK`: its own
    parameters may be used anywhere in its forward, so it is the last layer only when no
    other module has parameters. It is never in the fired types, since hook backends need a
    module path.
    """

    order: list[str] = []
    fired: dict[str, type[nn.Module]] = {}
    handles: list[RemovableHandle] = []
    named = cast(list[tuple[str, nn.Module]], list(module.named_modules()))

    if len(module._parameters) > 0:
        order.append(ROOT_BLOCK)

    for name, mod in named:
        if name and len(mod._parameters) > 0:

            def hook(m: nn.Module, _i: object, _o: object, name: str = name) -> None:
                if name not in order:
                    order.append(name)

                if isinstance(m, nn.Linear | nn.Conv2d):
                    fired[name] = type(m)

            handles.append(mod.register_forward_hook(hook))

    try:
        run()
    finally:
        for h in handles:
            h.remove()

    return tuple(order), fired


def probe(
    model: ModelFn, lik: LikelihoodTree, batch: Batch, /, params: Params | None = None
) -> Capabilities:
    """Run the model on the first two examples and record what works.

    The probe records the forward order and the output size per example, then tries vjp,
    the VJP of a VJP, jvp and vmap(grad) on the first two examples. vmap(grad) goes
    through the likelihood's `log_prob` when the batch has targets, else through the sum
    of the outputs. The double VJP must also agree with the plain one, which catches a
    backward that silently returns zeros when differentiated. The order pass runs once
    under `no_grad`.

    A `HasOutputCheck` likelihood (each such leaf of a tree) checks the order pass's
    outputs, with or without targets, before any transform is tried; what it raises
    propagates, except an `AttributeError` or `TypeError` on outputs that are not a
    tensor, which becomes a `StructureError` (see Raises). With targets, the likelihood's
    `log_prob` runs on those outputs too. Any other failure is recorded as a missing
    capability, except the errors under Raises, which propagate.

    Args:
        model: The model as a function of the params.
        lik: The likelihood, or a mapping of them. A likelihood mapping needs outputs and
            targets that are mappings with its keys.
        batch: The batch whose first two examples are probed.
        params: The params the transforms are taken with respect to; every other
            parameter stays fixed and is never cloned or differentiated. `None` means every
            module parameter. An empty mapping skips the transforms (all False, "not
            probed"). The probed names are literal: a tied tensor probed under one name
            and fitted under another is refused. Names that are not the module's own
            parameters (a `Selection`'s `"fc.weight[idx]"`) are recorded as `selected`.

    Returns:
        The `Capabilities`: which transforms work, with the failure messages of those that
        don't, the module order, the hooked Linear and Conv2d types, the event size, the
        tied names, and the probed and selected names.

    Raises:
        StructureError: The likelihood is a mapping and the outputs or targets are not
            mappings with its keys; a leaf's check or `log_prob` fails with an
            `AttributeError` or `TypeError` on outputs that are not a tensor (a dict
            reaching a tensor likelihood), and the error names the likelihood and the
            output type; or the model, the likelihood or a transform raises a
            `StructureError` (a likelihood whose shapes don't fit the outputs, say).
        ValueError: The likelihood's `log_prob` raises it on the order pass's outputs
            (targets outside the support, say). It propagates with a note naming the
            likelihood and the output type.
    """

    n = min(_N_EXAMPLES, batch.size)
    inputs = tree_map(partial(_first, n=n), batch.inputs)
    targets = None if batch.targets is None else tree_map(partial(_first, n=n), batch.targets)
    module = model.module
    failures: dict[str, str] = {}
    caps: dict[str, bool] = {}

    if params is None:
        chosen: Params = dict(module.named_parameters()) if module is not None else {}
    else:
        chosen = params

    order, fired, event_size = _order_pass(model, lik, chosen, inputs, targets, n, failures)

    if params is not None and not params:
        for name in TRANSFORMS:
            caps[name] = False
            failures[name] = NOT_PROBED
    else:
        _try_transforms(model, lik, chosen, inputs, targets, caps, failures)

    return Capabilities(
        vjp=caps["vjp"],
        vjp_twice=caps["vjp_twice"],
        jvp=caps["jvp"],
        vmap=caps["vmap"],
        hooked_types=fired,
        module_order=order,
        event_size=event_size,
        tied=tied_names(module) if module is not None else {},
        failures=failures,
        probed=frozenset(params) if params is not None else None,
        selected=_selected(params),
    )


def _selected(params: Params | None) -> frozenset[str]:
    """Probed names of a `Selection`'s parameters, those ending in `_SELECTED_SUFFIX`."""

    if params is None:
        return frozenset()

    return frozenset(n for n in params if n.endswith(_SELECTED_SUFFIX))


def _order_pass(
    model: ModelFn,
    lik: LikelihoodTree,
    params: Params,
    inputs: PyTree,
    targets: PyTree | None,
    n: int,
    failures: dict[str, str],
) -> tuple[tuple[str, ...], dict[str, type[nn.Module]], int | None]:
    """One forward under `no_grad`: module order, hooked types, and the event size.

    The likelihood's output check runs on that forward's output, and with targets its
    `log_prob` too (`_check_log_prob`); a `StructureError` from either propagates.
    """

    module = model.module
    order: tuple[str, ...] = ()
    fired: dict[str, type[nn.Module]] = {}
    out: list[PyTree] = []

    def run() -> None:
        out.append(model(params, inputs))

    try:
        with torch.no_grad():
            if module is not None:
                order, fired = _module_order(module, run)
            else:
                run()
    except StructureError:
        raise
    except Exception as e:  # noqa: BLE001
        failures["event_size"] = _failure(e)

        if module is not None:
            failures["module_order"] = _failure(e)

        return order, fired, None

    with torch.no_grad():
        _check_outputs(lik, out[0])

        if targets is not None:
            _check_log_prob(lik, out[0], targets)

    try:
        event_size = _event_size(out[0], n)
    except StructureError:
        raise
    except Exception as e:  # noqa: BLE001
        failures["event_size"] = _failure(e)

        return order, fired, None

    return order, fired, event_size


def _try_transforms(
    model: ModelFn,
    lik: LikelihoodTree,
    params: Params,
    inputs: PyTree,
    targets: PyTree | None,
    caps: dict[str, bool],
    failures: dict[str, str],
) -> None:
    """Try vjp, the double VJP, jvp and vmap(grad) with respect to copies of `params`."""

    params = {k: v.detach().clone() for k, v in params.items()}

    def fwd(p: Params) -> PyTree:
        return model(p, inputs)

    def attempt(name: str, fn: Callable[[], object]) -> None:
        try:
            # torch lazily imports a decomposition module on first use of its transforms, and
            # that module calls `torch.jit.script`, which warns on Python 3.14+. Not our code.
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore", message=".*torch.jit.script. is not supported", category=FutureWarning
                )
                fn()

            caps[name] = True
        except StructureError:
            raise  # a shape or structure mistake of the caller, not a missing transform
        except Exception as e:  # noqa: BLE001
            caps[name] = False
            failures[name] = _failure(e)

    def try_vjp() -> None:
        out, pull = vjp(fwd, params)[:2]
        pull(tree_map(torch.ones_like, out))

    def try_vjp_twice() -> None:
        # J t as the VJP of u -> J^T u, as `outputs` computes it. With t = J^T 1, the dot
        # 1 . J t must equal |J^T 1|^2; a backward that is not differentiable gives 0.
        out, pull = vjp(fwd, params)[:2]
        ones = tree_map(torch.ones_like, out)
        t = pull(ones)[0]
        pull_t = vjp(lambda u: pull(u)[0], tree_map(torch.zeros_like, out))[1]
        jt = pull_t(t)[0]

        got = _dot(ones, jt)
        want = _dot(t, t)
        tol = torch.finfo(next(iter(params.values())).dtype).eps ** 0.5

        if not bool(torch.isclose(got, want, rtol=tol, atol=0.0)):
            raise ValueError(
                f"the double VJP disagrees with the VJP (1.Jt = {got.item():.4g}, "
                f"|J^T 1|^2 = {want.item():.4g}); the backward is not differentiable"
            )

    def try_jvp() -> None:
        tangent = {k: torch.ones_like(v) for k, v in params.items()}

        jvp(fwd, (params,), (tangent,))

    def try_vmap() -> None:
        # With targets, through the likelihood; without, through the outputs alone, so a
        # likelihood that can't sample still gets the model probed.
        if targets is None:

            def per_output(p: Params, x: PyTree) -> Tensor:
                return _total_output(model(p, tree_map(_unsqueeze, x)))

            vmap(grad(per_output), in_dims=(None, 0))(params, inputs)

            return

        def per_example(p: Params, x: PyTree, y: PyTree) -> Tensor:
            x1 = tree_map(_unsqueeze, x)
            y1 = tree_map(_unsqueeze, y)

            return _total_log_prob(lik, model(p, x1), y1)

        vmap(grad(per_example), in_dims=(None, 0, 0))(params, inputs, targets)

    attempt("vjp", try_vjp)
    attempt("vjp_twice", try_vjp_twice)
    attempt("jvp", try_jvp)
    attempt("vmap", try_vmap)
