"""`lk.heteroscedastic`: the package and its plugin record."""

from importlib import metadata

import laplacekit as lk


def test_entry_point_loads_the_plugin_record():
    (ep,) = [
        ep
        for ep in metadata.entry_points(group="laplacekit.plugins")
        if ep.name == "laplacekit_heteroscedastic"
    ]

    assert ep.load() is lk.heteroscedastic.plugin
    assert ep.load() == lk.Plugin(supports=">=0.1a0,<0.2")
