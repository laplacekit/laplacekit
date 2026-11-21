"""The API reference: two halves, Everyday and Building blocks, covering the public names."""

import importlib
import inspect
import re
import typing
from pathlib import Path
from types import ModuleType
from typing import TypeAliasType

import griffe
import pytest
import yaml

import laplacekit as lk
import laplacekit.core as core

CONFIG = yaml.safe_load((Path(__file__).parents[1] / "docs" / "_quarto.yml").read_text())
QUARTODOC = CONFIG["quartodoc"]
HALVES = ("Everyday", "Building blocks")

# `lk.fs` is another name for `function_space`.
ALIASES = {"core"}  # the core API sits under `core.` entries, by name

# Plugin packages reached as `lk.<alias>`, each documented on one module page.
MODULE_PAGES = {
    "laplacekit_transformers": "transformers",
    "laplacekit_diffusers": "diffusers",
    "laplacekit_subnet": "subnet",
    "laplacekit_heteroscedastic": "heteroscedastic",
    "laplacekit_reward": "reward",
    "laplacekit_gp": "gp",
    "laplacekit_marglik": "marglik",
}


def entries() -> list[tuple[str, str, str | None]]:
    """`(half, name, package)` for every entry, in order."""

    found = []
    half = None

    for section in QUARTODOC["sections"]:
        if "title" in section:
            half = section["title"]

        for item in section.get("contents", []):
            name = item if isinstance(item, str) else item["name"]
            default = section.get("package", QUARTODOC["package"])
            package = default if isinstance(item, str) else item.get("package", default)
            found.append((half, name, package))

    return found


def type_aliases() -> list[tuple[str, str]]:
    """`(half, name)` for each type alias in `lk.__all__` that a section's `desc` names.

    quartodoc can't render `type` statements, so a section's description states them.
    """

    aliases = [name for name in core.__all__ if isinstance(getattr(core, name), TypeAliasType)]
    found = []
    half = None

    for section in QUARTODOC["sections"]:
        half = section.get("title", half)
        found.extend((half, name) for name in aliases if f"`{name}" in section.get("desc", ""))

    return found


def resolve(name: str, package: str | None) -> object:
    """The object an entry documents."""

    path = name if package is None else f"{package}.{name}"
    module, _, attr = path.partition(".")
    obj: object = importlib.import_module(module)

    for part in attr.split(".") if attr else []:
        obj = getattr(obj, part)

    return obj


def covers(name: str, package: str | None) -> str | None:
    """The name in `lk.__all__` (or alias module) an entry covers, if any."""

    if package is None:
        return MODULE_PAGES.get(name)

    if package != "laplacekit":
        return None

    head, _, rest = name.partition(".")

    if head in ALIASES:
        return None

    if head == "core":
        top = getattr(lk, rest, None)
        return rest if top is not None and top is resolve(name, package) else None

    if rest and isinstance(getattr(lk, head), ModuleType):
        return head  # a module counts as covered when its members appear

    return name


def test_sections_fall_under_two_captions():
    sections = QUARTODOC["sections"]
    titles = [section["title"] for section in sections if "title" in section]

    assert titles == list(HALVES)
    assert "title" in sections[0]

    for section in sections:
        assert ("title" in section) != ("subtitle" in section), section

        if "title" in section:
            assert "contents" not in section, section  # the API sits in the subsections
        else:
            assert section["contents"], section


def test_every_entry_resolves():
    for _, name, package in entries():
        resolve(name, package)


def test_type_aliases_are_stated():
    assert {name for _, name in type_aliases()} >= {"Posterior", "Params", "HessianApprox"}


def test_no_entry_appears_twice():
    names = [(package, name) for _, name, package in entries()]

    assert len(names) == len(set(names))


def test_every_public_name_sits_in_exactly_one_half():
    expected = (set(lk.__all__) - ALIASES) | set(MODULE_PAGES.values())
    halves: dict[str, set[str]] = {}

    for half, name, package in entries():
        covered = covers(name, package)

        if covered is not None:
            halves.setdefault(covered, set()).add(half)

    assert set(halves) == expected
    assert {name: found for name, found in halves.items() if len(found) != 1} == {}


def test_documented_core_names_are_public():
    for _, name, package in entries():
        head, _, rest = name.partition(".")

        if package == "laplacekit" and head == "core":
            assert rest in core.__all__, name


def test_everyday_holds_the_front_door():
    everyday = {covers(name, package) for half, name, package in entries() if half == HALVES[0]}

    assert {
        "laplace",
        "predict",
        "tune",
        "transformers",
        "diffusers",
        "subnet",
        "heteroscedastic",
        "reward",
        "gp",
        "marglik",
        "plugins",
    } <= everyday
    assert {"fit_curvature", "Batch", "testing", "Plugin", "Curvature"}.isdisjoint(everyday)


@pytest.mark.parametrize("package", sorted(MODULE_PAGES))
def test_module_page_signatures_name_no_private_types(package):
    """Signatures of a module page name public types only (`help` and IDEs show them)."""

    plugin = importlib.import_module(package)

    for name in plugin.__all__:
        obj = getattr(plugin, name)

        if not callable(obj) or isinstance(obj, type):
            continue

        for parameter in inspect.signature(obj).parameters.values():
            # `torch.Generator` is `torch._C.Generator` at run time: torch's, not ours.
            annotation = str(parameter.annotation).replace("torch._C.", "torch.")

            assert not re.search(r"\b_[A-Za-z]", annotation), (name, parameter)


def test_every_entry_has_a_docstring():
    """What quartodoc reads (with griffe), so a page never shows only a signature."""

    loader = griffe.GriffeLoader()
    packages = {}
    missing = []

    for _, name, package in entries():
        root, _, path = (name if package is None else f"{package}.{name}").partition(".")

        if root not in packages:
            packages[root] = loader.load(root)

        loaded = packages[root]
        obj = loaded[path] if path else loaded

        if obj.docstring is None:
            missing.append(name)

    assert missing == []


def _protocols() -> dict[str, type]:
    """Every public protocol, by the path users import it from."""

    found: dict[str, type] = {}
    modules = [lk, core, lk.data, *map(importlib.import_module, MODULE_PAGES)]

    for module in modules:
        names = getattr(module, "__all__", None) or [n for n in vars(module) if n[0] != "_"]

        for name in names:
            obj = getattr(module, name)

            if inspect.isclass(obj) and typing.is_protocol(obj):
                found.setdefault(f"{obj.__module__}.{obj.__qualname__}", obj)

    return found


def _entry_items() -> list[dict]:
    """Every entry as a dict with `name`, `package` and its options."""

    items = []

    for section in QUARTODOC["sections"]:
        default = section.get("package", QUARTODOC["package"])

        for item in section.get("contents", []):
            item = {"name": item} if isinstance(item, str) else dict(item)
            item.setdefault("package", default)
            items.append(item)

    return items


def _own_entry(proto: type) -> dict | None:
    """The entry that documents `proto` on its own (not through a module page)."""

    for item in _entry_items():
        try:
            obj = resolve(item["name"], item["package"])
        except (AttributeError, ImportError):
            continue

        if obj is proto:
            return item

    return None


@pytest.mark.parametrize("path", sorted(_protocols()))
def test_every_protocol_has_its_own_entry(path):
    """Core protocols get their own entry. A plugin's sit on its module page, unless dunder."""

    proto = _protocols()[path]
    dunder = any(name.startswith("_") for name in typing.get_protocol_members(proto))

    if path.startswith("laplacekit.") or dunder:
        assert _own_entry(proto) is not None, path


@pytest.mark.parametrize("path", sorted(_protocols()))
def test_protocol_entries_render_every_member(path):
    """quartodoc hides `_` names, so dunder members (`__call__`, `__add__`) need `members:`."""

    proto = _protocols()[path]
    entry = _own_entry(proto)
    wanted = typing.get_protocol_members(proto)

    if entry is None:
        return  # a plugin protocol on its module page, see test_every_protocol_has_its_own_entry

    if not any(name.startswith("_") for name in wanted):
        assert "members" not in entry, path
        return

    assert set(entry.get("members", [])) == wanted, path


def test_protocols_are_found():
    assert {"Stat", "Subset", "Data", "OutputHessian", "Predictive", "Kernel"} <= {
        p.rsplit(".", 1)[1] for p in _protocols()
    }
