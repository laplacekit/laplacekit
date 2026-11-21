import importlib
import pkgutil
import sys
import types
from importlib import metadata
from pathlib import Path

import pytest

import laplacekit as lk
from laplacekit import HealthCheck

GROUP = "laplacekit.plugins"
SUPPORTS = ">=0.1a0,<0.2"


class FakeDist:
    def __init__(self, name, version="1.0", summary=""):
        self.name = name
        self.version = version
        self.metadata = {"Summary": summary}


class FakeEntryPoint:
    """Stands in for an installed `laplacekit.plugins` entry point: imports `module:attr`."""

    group = GROUP

    def __init__(self, name, value, dist):
        self.name = name
        self.value = value
        self.module, _, self.attr = value.partition(":")
        self.dist = dist

    def load(self):
        return getattr(importlib.import_module(self.module), self.attr)


def fake_entry_points(monkeypatch, *eps):
    def entry_points(*, group):
        assert group == GROUP

        return metadata.EntryPoints(eps)

    monkeypatch.setattr(metadata, "entry_points", entry_points)


def fake_module(monkeypatch, name, **attrs):
    """A plugin module in `sys.modules`; `__all__` lists the attributes given."""

    module = types.ModuleType(name)
    module.__dict__.update(attrs)
    module.__all__ = [n for n in attrs if n != "plugin"]
    monkeypatch.setitem(sys.modules, name, module)

    return module


def installed(monkeypatch, version):
    """Pretend the installed laplacekit has `version`."""

    real = metadata.version

    def fake(dist):
        return version if dist == "laplacekit" else real(dist)

    monkeypatch.setattr(metadata, "version", fake)


class Thing:
    """A thing the plugin provides.

    More detail on a second line.
    """


def make_thing():
    """Make a thing."""

    return Thing()


def undocumented():
    pass


def wrapped():
    """Make a thing from a long description that wraps
    onto a second line. Then more detail.

    A second paragraph.
    """


class Instance:
    """The type of an instance."""


def good_health():
    return (HealthCheck("dep", "ok", "1.2"),)


def test_plugins_lists_name_version_summary_and_provides(monkeypatch):
    installed(monkeypatch, "0.1a0")
    plugin = lk.Plugin(supports=SUPPORTS, health=good_health)
    fake_module(
        monkeypatch,
        "fake_b",
        Thing=Thing,
        make_thing=make_thing,
        undocumented=undocumented,
        LIMIT=3,
        health=good_health,
        plugin=plugin,
    )
    fake_module(monkeypatch, "fake_a", plugin=lk.Plugin(supports=SUPPORTS))
    fake_entry_points(
        monkeypatch,
        FakeEntryPoint("fake_b", "fake_b:plugin", FakeDist("laplacekit-b", "0.3", "Bee things")),
        FakeEntryPoint("fake_a", "fake_a:plugin", FakeDist("laplacekit-a", "0.2", "Aye")),
    )

    found = lk.plugins()

    assert isinstance(found, lk.Plugins)
    assert len(found) == 2
    assert [p.name for p in found] == ["laplacekit-a", "laplacekit-b"]  # by distribution

    b = found[1]

    assert (b.name, b.version, b.summary) == ("laplacekit-b", "0.3", "Bee things")
    assert b.supported and b.supports == SUPPORTS and b.error is None
    assert b.provides == (
        ("Thing", "A thing the plugin provides."),
        ("make_thing", "Make a thing."),
        ("undocumented", ""),
        ("LIMIT", ""),
    )
    assert found[0].provides == ()

    lines = str(b).splitlines()

    assert lines[0] == "laplacekit-b 0.3  Bee things"
    assert lines[1].startswith("  ") and "Thing" in lines[1]
    assert "A thing the plugin provides." in lines[1]
    assert len(lines) == 5

    text = str(found)

    assert text.index("laplacekit-a") < text.index("laplacekit-b")


def test_plugin_outside_supports_is_listed_and_health_errors(monkeypatch):
    installed(monkeypatch, "0.1a0")
    fake_module(monkeypatch, "fake_new", plugin=lk.Plugin(supports=">=9", health=good_health))
    fake_entry_points(
        monkeypatch, FakeEntryPoint("fake_new", "fake_new:plugin", FakeDist("laplacekit-new"))
    )

    (info,) = lk.plugins()

    assert not info.supported
    assert info.supports == ">=9"
    assert info.error is None

    report = lk.health()
    checks = report.packages["fake_new"]

    assert checks[0] == HealthCheck(
        "supports", "error", "laplacekit-new supports >=9, installed laplacekit 0.1a0"
    )
    assert checks[1:] == (HealthCheck("dep", "ok", "1.2"),)  # its own checks still run
    assert not report.ok


def test_plugin_inside_supports_has_an_ok_check(monkeypatch):
    installed(monkeypatch, "0.1a0")
    fake_module(monkeypatch, "fake_ok", plugin=lk.Plugin(supports=SUPPORTS, health=good_health))
    fake_entry_points(
        monkeypatch, FakeEntryPoint("fake_ok", "fake_ok:plugin", FakeDist("laplacekit-ok"))
    )

    checks = lk.health().packages["fake_ok"]

    assert [c.name for c in checks] == ["supports", "dep"]
    assert checks[0].status == "ok"
    assert SUPPORTS in checks[0].detail and "0.1a0" in checks[0].detail


@pytest.mark.parametrize("spec", [">=0.1a0,<0.2", "<0.2", "==0.1a0"])
def test_prerelease_laplacekit_satisfies_lower_bound(monkeypatch, spec):
    installed(monkeypatch, "0.1a0")
    fake_module(monkeypatch, "fake_pre", plugin=lk.Plugin(supports=spec))
    fake_entry_points(
        monkeypatch, FakeEntryPoint("fake_pre", "fake_pre:plugin", FakeDist("laplacekit-pre"))
    )

    (info,) = lk.plugins()

    assert info.supported
    assert lk.health().packages["fake_pre"][0].status == "ok"


def test_entry_point_not_a_plugin_errors(monkeypatch):
    fake_module(monkeypatch, "fake_old", health=good_health)
    fake_entry_points(
        monkeypatch, FakeEntryPoint("fake_old", "fake_old:health", FakeDist("laplacekit-old"))
    )

    (info,) = lk.plugins()

    assert info.name == "laplacekit-old"
    assert info.error is not None
    assert "fake_old" in info.error and "function" in info.error
    assert info.provides == () and not info.supported

    report = lk.health()
    (check,) = report.packages["fake_old"]

    assert check.status == "error"
    assert "fake_old" in check.detail and "function" in check.detail
    assert not report.ok


def test_plugin_import_failure_is_listed_with_error(monkeypatch):
    fake_entry_points(
        monkeypatch,
        FakeEntryPoint("fake_missing", "fake_missing_xyz:plugin", FakeDist("laplacekit-gone")),
    )

    (info,) = lk.plugins()

    assert info.name == "laplacekit-gone"
    assert info.error is not None and "fake_missing_xyz" in info.error
    assert info.provides == ()
    assert "fake_missing_xyz" in str(info)

    (check,) = lk.health().packages["fake_missing"]

    assert check.status == "error"
    assert "ModuleNotFoundError" in check.detail


def test_plugin_invalid_supports_raises_valueerror():
    with pytest.raises(ValueError, match="not a version"):
        lk.Plugin(supports="not a version")


def test_plugin_health_defaults_to_no_checks():
    assert lk.Plugin(supports=SUPPORTS).health() == ()


def test_installed_plugins_are_listed():
    found = {p.name: p for p in lk.plugins()}

    assert {
        "laplacekit-transformers",
        "laplacekit-diffusers",
        "laplacekit-subnet",
        "laplacekit-heteroscedastic",
        "laplacekit-reward",
        "laplacekit-gp",
        "laplacekit-marglik",
    } <= set(found)

    plugin = found["laplacekit-transformers"]

    assert plugin.error is None and plugin.supported
    assert plugin.summary.endswith("on causal language models and sequence classifiers")
    assert "health" not in dict(plugin.provides) and "laplace_lora" in dict(plugin.provides)

    for _, line in plugin.provides:
        assert line == "" or line.endswith("."), line

    diffusion = found["laplacekit-diffusers"]

    assert diffusion.error is None and diffusion.supported
    assert [n for n, _ in diffusion.provides] == ["denoiser", "laplace"]


SPLIT = ("subnet", "heteroscedastic", "reward", "gp", "marglik", "transformers", "diffusers")


@pytest.mark.parametrize("name", SPLIT)
def test_first_party_plugins_define_their_names_in_modules(name):
    pkg = importlib.import_module(f"laplacekit_{name}")
    own = [n for n in pkg.__all__ if getattr(getattr(pkg, n), "__module__", None) == pkg.__name__]

    # `__init__` holds the package docstring, the re-exports, `health` and the plugin record.
    assert own in ([], ["health"]), own

    for info in pkgutil.iter_modules(pkg.__path__):
        if info.name.startswith("_"):
            continue

        module = importlib.import_module(f"{pkg.__name__}.{info.name}")

        assert set(module.__all__) <= set(pkg.__all__), info.name


def test_provides_shows_the_whole_first_sentence(monkeypatch):
    installed(monkeypatch, "0.1a0")
    fake_module(
        monkeypatch,
        "fake_c",
        wrapped=wrapped,
        instance=Instance(),
        plugin=lk.Plugin(supports=SUPPORTS),
    )
    fake_entry_points(
        monkeypatch, FakeEntryPoint("fake_c", "fake_c:plugin", FakeDist("laplacekit-c"))
    )

    (c,) = lk.plugins()

    assert c.provides == (
        ("wrapped", "Make a thing from a long description that wraps onto a second line."),
        ("instance", ""),
    )


def test_every_workspace_plugin_ships_py_typed():
    """PEP 561: without the marker, users' type checkers treat the plugin as untyped."""

    root = Path(__file__).parents[1] / "plugins"
    packages = [p for p in root.glob("*/src/*") if (p / "__init__.py").is_file()]

    assert {p.name for p in packages} >= {
        "laplacekit_transformers",
        "laplacekit_subnet",
        "laplacekit_gp",
        "laplacekit_marglik",
    }
    assert [p.name for p in packages if not (p / "py.typed").is_file()] == []
