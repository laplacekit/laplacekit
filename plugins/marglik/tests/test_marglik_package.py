"""`lk.marglik`: the package and its plugin record."""

import sys
from importlib import metadata

import pytest

import laplacekit as lk


def test_entry_point_loads_the_plugin_record():
    (ep,) = [
        ep
        for ep in metadata.entry_points(group="laplacekit.plugins")
        if ep.name == "laplacekit_marglik"
    ]

    assert ep.load() is lk.marglik.plugin
    assert ep.load() == lk.Plugin(supports=">=0.1a0,<0.2")


def test_alias_is_the_module():
    import laplacekit_marglik

    assert lk.marglik is laplacekit_marglik
    assert lk.marglik.__all__ == ["Trained", "train"]


def test_plugins_lists_it():
    (info,) = [p for p in lk.plugins() if p.name == "laplacekit-marglik"]

    assert info.error is None and info.supported
    assert "Immer et al. 2021" in lk.marglik.__doc__


def test_missing_extra_names_the_pip_command(monkeypatch):
    monkeypatch.setitem(sys.modules, "laplacekit_marglik", None)

    with pytest.raises(ImportError, match=r'pip install "laplacekit\[marglik\]"'):
        lk.marglik  # noqa: B018
