"""The one place that touches torch's (private) pytree registry.

`@pytree` turns a class into a frozen dataclass whose fields are pytree children,
except fields marked `static()`, which go into the treedef context. That makes
`torch.func.grad` and `vmap` see the tensors inside `@pytree` objects.
"""

import dataclasses
from collections.abc import Callable, Mapping
from typing import Any, cast, dataclass_transform, overload

import torch.utils._pytree as _pt

_STATIC = "laplacekit_static"

PYTREE_CLASSES: dict[str, type] = {}
"""Every `@pytree` class, by `"module:qualname"`: the classes `lk.load` may rebuild."""


@overload
def static[T](*, default: T, kw_only: bool = False) -> T: ...


@overload
def static[T](*, default_factory: Callable[[], T], kw_only: bool = False) -> T: ...


@overload
def static(*, kw_only: bool = False) -> Any: ...


def static(**kwargs: Any) -> Any:
    """Mark a field as static: part of the structure, not a pytree leaf.

    Args:
        **kwargs: Passed to `dataclasses.field`: `default` or `default_factory`, and
            `kw_only=True` for a field that only a keyword sets.

    Returns:
        A dataclass field.
    """

    return dataclasses.field(metadata={_STATIC: True}, **kwargs)


@dataclass_transform(frozen_default=True, field_specifiers=(dataclasses.field, static))
def pytree[C: type](cls: C) -> C:
    """Make `cls` a frozen dataclass registered as a torch pytree node.

    Args:
        cls: The class to decorate. Its fields marked `static()` go into the treedef
            context; the others are children.

    Returns:
        The frozen dataclass, registered.
    """

    own_repr = "__repr__" in cls.__dict__
    dc: type[Any] = dataclasses.dataclass(frozen=True)(cls)
    fs = dataclasses.fields(dc)

    if not own_repr:
        dc.__repr__ = _compact_repr
    children = tuple(f.name for f in fs if not f.metadata.get(_STATIC, False))
    statics = tuple(f.name for f in fs if f.metadata.get(_STATIC, False))

    def flatten(obj: Any) -> tuple[list[Any], tuple[Any, ...]]:
        return [getattr(obj, n) for n in children], tuple(getattr(obj, n) for n in statics)

    def unflatten(values: Any, context: tuple[Any, ...]) -> Any:
        obj = object.__new__(dc)

        for n, v in zip(children, values, strict=True):
            object.__setattr__(obj, n, v)

        for n, v in zip(statics, context, strict=True):
            object.__setattr__(obj, n, v)

        return obj

    def flatten_with_keys(obj: Any) -> tuple[list[tuple[Any, Any]], tuple[Any, ...]]:
        vals, ctx = flatten(obj)
        return [(_pt.GetAttrKey(n), v) for n, v in zip(children, vals, strict=True)], ctx

    PYTREE_CLASSES[f"{dc.__module__}:{dc.__qualname__}"] = dc

    _pt.register_pytree_node(
        dc,
        flatten,
        unflatten,
        serialized_type_name=f"{dc.__module__}.{dc.__qualname__}",
        flatten_with_keys_fn=flatten_with_keys,
    )

    return cast(C, dc)


def _is_default(f: dataclasses.Field[Any], value: Any) -> bool:
    if f.default is not dataclasses.MISSING:
        default = f.default
    elif f.default_factory is not dataclasses.MISSING:
        default = f.default_factory()
    else:
        return False

    try:
        same = value == default
    except Exception:  # noqa: BLE001  # a tensor against a default: not comparable
        return False

    return same is True


def _compact_repr(self: Any) -> str:
    """`Name(field=value, ...)` with the fields at their defaults left out, as `nn.Module`."""

    shown = ", ".join(
        f"{f.name}={getattr(self, f.name)!r}"
        for f in dataclasses.fields(self)
        if f.repr and not _is_default(f, getattr(self, f.name))
    )

    return f"{type(self).__qualname__.rsplit('.', 1)[-1]}({shown})"


def tree_map(fn: Callable[..., Any], tree: Any, *rest: Any) -> Any:
    """Apply `fn` to the leaves of `tree`.

    Args:
        fn: Called on each leaf of `tree`, followed by the matching leaves of `rest`.
        tree: The tree.
        *rest: Trees with the same structure as `tree`.

    Returns:
        A tree like `tree` holding the results.
    """

    return _pt.tree_map(fn, tree, *rest)


def tree_leaves(tree: Any) -> list[Any]:
    """Flat list of the leaves of `tree`.

    Args:
        tree: The tree.

    Returns:
        The leaves. Static fields are excluded.
    """

    return _pt.tree_leaves(tree)


def tree_leaves_with_path(tree: Any) -> list[tuple[str, Any]]:
    """The leaves of `tree`, each with its path.

    Args:
        tree: The tree.

    Returns:
        `(path, leaf)` pairs, the path as a string (e.g. `.curvature.sum['0.weight']`).
    """

    return [(_pt.keystr(path), leaf) for path, leaf in _pt.tree_flatten_with_path(tree)[0]]


def tree_map_with_path(fn: Callable[[str, Any], Any], tree: Any) -> Any:
    """Apply `fn(path, leaf)` to the leaves of `tree`.

    Args:
        fn: Called with each leaf's path (a string, as in `tree_leaves_with_path`) and the
            leaf.
        tree: The tree.

    Returns:
        A tree like `tree` holding the results.
    """

    return _pt.tree_map_with_path(lambda path, leaf: fn(_pt.keystr(path), leaf), tree)


def _opaque(x: Any) -> bool:
    """Whether `x` is a mapping torch's pytree treats as a leaf.

    That is any `Mapping` whose type is not a registered node (a `UserDict`, a transformers
    `BatchEncoding`, a dict subclass).
    """

    return isinstance(x, Mapping) and type(x) not in _pt.SUPPORTED_NODES


def as_pytree(tree: Any) -> Any:
    """Turn each mapping that torch's pytree would treat as a leaf into a plain `dict`.

    This works at any depth, so `vmap` and `tree_map` see the mapping's tensors.

    Args:
        tree: The tree.

    Returns:
        The converted tree. A tree without such a mapping comes back as it is.
    """

    if not any(_opaque(x) for x in _pt.tree_leaves(tree, is_leaf=_opaque)):
        return tree

    def plain(x: Any) -> Any:
        if not _opaque(x):
            return x

        return {k: as_pytree(v) for k, v in cast(Mapping[Any, Any], x).items()}

    return _pt.tree_map(plain, tree, is_leaf=_opaque)
