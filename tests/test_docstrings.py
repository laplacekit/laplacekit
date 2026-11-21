"""Docstrings of the public API: Google sections that match the signatures."""

import dataclasses
import importlib
import inspect
import typing
from collections.abc import Callable, Iterator
from types import ModuleType

import griffe
import pytest

import laplacekit
import laplacekit.core

SUBMODULES = (
    "priors",
    "likelihoods",
    "curvature",
    "predictive",
    "subsets",
    "objectives",
    "data",
    "function_space",
    "testing",
    "distributed",
    "tuning",
)

PLUGINS = (
    "laplacekit_transformers",
    "laplacekit_diffusers",
    "laplacekit_subnet",
    "laplacekit_heteroscedastic",
    "laplacekit_reward",
    "laplacekit_gp",
    "laplacekit_marglik",
)

NO_RETURN = (None, type(None), "None", typing.NoReturn, typing.Never, "NoReturn", "Never")


def _is_ours(obj: object) -> bool:
    module = getattr(obj, "__module__", None) or ""

    return (inspect.isclass(obj) or inspect.isfunction(obj)) and module.startswith("laplacekit")


def _exported(module: ModuleType) -> Iterator[object]:
    for name in module.__all__:
        yield getattr(module, name)


def _defined(module: ModuleType) -> Iterator[object]:
    for name, obj in vars(module).items():
        if not name.startswith("_") and getattr(obj, "__module__", None) == module.__name__:
            yield obj


def _top_level() -> list[object]:
    """Public functions and classes, each once."""

    candidates = [*_exported(laplacekit), *_exported(laplacekit.core)]

    for name in SUBMODULES:
        candidates.extend(_defined(importlib.import_module(f"laplacekit.{name}")))

    for name in PLUGINS:
        candidates.extend(_exported(importlib.import_module(name)))

    seen: dict[int, object] = {}

    for obj in candidates:
        if _is_ours(obj):
            seen.setdefault(id(obj), obj)

    return list(seen.values())


# Dunders that are part of a public contract: calling an object, and adding statistics.
DUNDERS = {"__call__", "__add__"}


def _methods(cls: type) -> Iterator[tuple[str, Callable[..., object]]]:
    """Public methods, classmethods and staticmethods that `cls` defines itself.

    A dataclass field whose default is a function (`Plugin.health`) is a field, not a method.
    """

    fields = {f.name for f in dataclasses.fields(cls)} if dataclasses.is_dataclass(cls) else set()

    for name, attr in vars(cls).items():
        if (name.startswith("_") and name not in DUNDERS) or name in fields:
            continue

        if isinstance(attr, (classmethod, staticmethod)):
            yield name, attr.__func__
        elif inspect.isfunction(attr):
            yield name, attr


class Item(typing.NamedTuple):
    """One documented object: a function, a class, or a method of a class."""

    qualname: str
    module: str
    obj: Callable[..., object]
    owner: type | None


def scope() -> list[Item]:
    """Every public object whose docstring the tests check."""

    items = []

    for obj in _top_level():
        assert callable(obj)
        module = obj.__module__
        items.append(Item(f"{module}.{obj.__qualname__}", module, obj, None))

        if inspect.isclass(obj):
            for name, method in _methods(obj):
                items.append(Item(f"{module}.{obj.__qualname__}.{name}", module, method, obj))

    return sorted(items, key=lambda item: item.qualname)


SCOPE = scope()
MODULES = sorted({item.module for item in SCOPE})


def _sections(item: Item) -> list[griffe.DocstringSection]:
    text = inspect.getdoc(item.obj) if item.owner is None else item.obj.__doc__
    docstring = griffe.Docstring(inspect.cleandoc(text or ""), parser="google")

    return docstring.parse(warnings=False)


def _documented(sections: list[griffe.DocstringSection], kind: griffe.DocstringSectionKind) -> list:
    return [entry for s in sections if s.kind is kind for entry in s.value]


def _exempt(item: Item) -> bool:
    """Protocols and exceptions, whose fields are not constructor arguments.

    Methods are never exempt: an implementation of a protocol method documents its own
    arguments and result, so its reference entry stands on its own.
    """

    obj = item.obj

    return inspect.isclass(obj) and (typing.is_protocol(obj) or issubclass(obj, BaseException))


def _is_classmethod(item: Item) -> bool:
    return item.owner is not None and isinstance(
        vars(item.owner).get(item.obj.__name__), classmethod
    )


def _items(module: str) -> list[Item]:
    return [item for item in SCOPE if item.module == module]


def _missing_docstrings(module: str) -> list[str]:
    return [
        f"{item.qualname}: no docstring"
        for item in _items(module)
        if not (item.obj.__doc__ or "").strip()
    ]


def _mismatched_args(module: str) -> list[str]:
    """Parameters missing from `Args:`, and `Args:` names that are not parameters."""

    problems = []

    for item in _items(module):
        if inspect.isclass(item.obj) and _exempt(item):
            continue

        params = inspect.signature(item.obj).parameters
        implicit = {"cls"} if _is_classmethod(item) else {"self"}
        wanted = set(params) - implicit if item.owner is not None else set(params)
        # The seed a fit or a set of points keeps is private: not documented.
        wanted = wanted - {"_seed"}
        documented = {
            p.name.lstrip("*")
            for p in _documented(_sections(item), griffe.DocstringSectionKind.parameters)
        }

        if wanted - documented and not _exempt(item):
            problems.append(f"{item.qualname}: Args misses {sorted(wanted - documented)}")

        if documented - wanted:
            problems.append(f"{item.qualname}: Args has no such {sorted(documented - wanted)}")

    return problems


def _undocumented_returns(module: str) -> list[str]:
    problems = []

    for item in _items(module):
        if inspect.isclass(item.obj):
            continue

        if _exempt(item):
            continue

        sections = _sections(item)

        if inspect.signature(item.obj).return_annotation in NO_RETURN:
            continue

        if inspect.isgeneratorfunction(item.obj):
            kind = griffe.DocstringSectionKind.yields
        else:
            kind = griffe.DocstringSectionKind.returns

        if not any(s.kind is kind for s in sections):
            problems.append(f"{item.qualname}: no {kind.value.capitalize()} section")

    return problems


@pytest.mark.parametrize("module", MODULES)
def test_has_docstring(module: str) -> None:
    problems = _missing_docstrings(module)

    assert not problems, "\n".join(problems)


@pytest.mark.parametrize("module", MODULES)
def test_args_cover_parameters(module: str) -> None:
    problems = _mismatched_args(module)

    assert not problems, "\n".join(problems)


@pytest.mark.parametrize("module", MODULES)
def test_returns_documented(module: str) -> None:
    problems = _undocumented_returns(module)

    assert not problems, "\n".join(problems)


@pytest.mark.parametrize("module", MODULES)
def test_no_control_chars(module: str) -> None:
    bad = [
        item.qualname
        for item in _items(module)
        if any(ord(c) < 0x20 and c != "\n" for c in item.obj.__doc__ or "")
    ]

    assert not bad, f"control characters (a missing r prefix?): {bad}"


def test_field_defaults_are_not_methods() -> None:
    """A dataclass field with a function default is documented under `Args:`."""

    assert "health" not in dict(_methods(laplacekit.Plugin))
    assert "log_prob" in dict(_methods(laplacekit.priors.Isotropic))
