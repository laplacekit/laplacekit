"""`lk.<alias>` imports a plugin package on first use."""

import importlib
import subprocess
import sys
from importlib import metadata

import pytest
from packaging.requirements import Requirement

import laplacekit as lk

# alias -> (plugin package, the modules the plugin imports from its library)
PLUGINS = {
    "transformers": ("laplacekit_transformers", ("transformers", "transformers.utils")),
    "diffusers": ("laplacekit_diffusers", ("diffusers", "diffusers.utils")),
    "subnet": ("laplacekit_subnet", ()),  # built on laplacekit alone
    "heteroscedastic": ("laplacekit_heteroscedastic", ()),
    "reward": ("laplacekit_reward", ()),
    "gp": ("laplacekit_gp", ()),
    "marglik": ("laplacekit_marglik", ()),
}
ALIASES = pytest.mark.parametrize("alias", list(PLUGINS))
WITH_LIBRARY = pytest.mark.parametrize("alias", [a for a, (_, libs) in PLUGINS.items() if libs])


def _import_fails(monkeypatch, alias, missing):
    real = importlib.import_module

    def import_module(name, package=None):
        if name == PLUGINS[alias][0]:
            raise ModuleNotFoundError(f"No module named {missing!r}", name=missing)

        return real(name, package)

    monkeypatch.setattr(importlib, "import_module", import_module)


@ALIASES
def test_import_laplacekit_does_not_import_the_plugin(alias):
    package, libraries = PLUGINS[alias]
    code = (
        "import sys\n"
        "import laplacekit as lk\n"
        "from laplacekit import *\n"
        + "".join(f"assert {lib!r} not in sys.modules, '{lib} imported'\n" for lib in libraries[:1])
        + f"assert {package!r} not in sys.modules, '{package} imported'\n"
        f"assert {alias!r} not in lk.__all__\n"
        f"import {package}\n"
        f"assert lk.{alias} is {package}\n"
        f"assert lk.{alias}.plugin is {package}.plugin\n"
    )
    run = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)

    assert run.returncode == 0, run.stderr


@ALIASES
def test_alias_is_the_plugin(alias):
    assert getattr(lk, alias) is importlib.import_module(PLUGINS[alias][0])


@ALIASES
def test_alias_missing_names_pip_extra(monkeypatch, alias):
    _import_fails(monkeypatch, alias, PLUGINS[alias][0])

    with pytest.raises(ImportError) as e:
        getattr(lk, alias)

    assert str(e.value) == (
        f'lk.{alias} needs the laplacekit-{alias} package: pip install "laplacekit[{alias}]"'
    )
    assert e.value.__cause__ is None and e.value.__suppress_context__


@ALIASES
def test_alias_missing_package_seen_by_the_import_system(monkeypatch, alias):
    monkeypatch.setitem(sys.modules, PLUGINS[alias][0], None)  # an absent package

    with pytest.raises(ImportError, match=rf'pip install "laplacekit\[{alias}\]"'):
        getattr(lk, alias)


@ALIASES
def test_alias_broken_plugin_reraises(monkeypatch, alias):
    _import_fails(monkeypatch, alias, alias)

    with pytest.raises(ModuleNotFoundError) as e:
        getattr(lk, alias)

    assert e.value.name == alias


@WITH_LIBRARY
def test_alias_reports_a_broken_install_as_it_is(monkeypatch, alias):
    package, libraries = PLUGINS[alias]
    importlib.import_module(package)  # so that delitem has something to remove

    # The package and its modules, so that the import runs again and reaches the library.
    for name in [m for m in sys.modules if m == package or m.startswith(f"{package}.")]:
        monkeypatch.delitem(sys.modules, name)

    for lib in libraries:
        monkeypatch.setitem(sys.modules, lib, None)

    with pytest.raises(ModuleNotFoundError) as e:
        getattr(lk, alias)

    assert e.value.name is not None and e.value.name.startswith(alias)


def test_sl_hf_is_gone():
    with pytest.raises(AttributeError, match="'hf'"):
        lk.hf  # noqa: B018


def test_unknown_attribute_is_an_attribute_error():
    with pytest.raises(AttributeError, match="nope"):
        lk.nope  # noqa: B018


@ALIASES
def test_extra_installs_the_plugin(alias):
    meta = metadata.metadata("laplacekit")
    requires = [Requirement(r) for r in meta.get_all("Requires-Dist", [])]
    plugin = [r for r in requires if r.name == f"laplacekit-{alias}"]
    own = [r for r in plugin if r.marker is not None and r.marker.evaluate({"extra": alias})]

    assert alias in meta.get_all("Provides-Extra", [])
    assert len(own) == 1 and not own[0].specifier  # unpinned: `supports` is the guard
    assert all(r.marker is not None for r in plugin)
    assert not any(r.marker.evaluate({"extra": ""}) for r in plugin)
    assert any(r.marker.evaluate({"extra": "all"}) for r in plugin)


def test_subnet_alias_and_plugin_record(monkeypatch):
    """`lk.subnet` is the first-party `laplacekit-subnet`: an extra of the core, listed by
    `lk.plugins()` with its public names, and named in the error when it is absent."""

    import laplacekit_subnet

    assert lk.subnet is laplacekit_subnet
    assert lk.subnet.plugin == lk.Plugin(supports=">=0.1a0,<0.2")
    assert "subnet" in metadata.metadata("laplacekit").get_all("Provides-Extra", [])

    (info,) = [p for p in lk.plugins() if p.name == "laplacekit-subnet"]

    assert info.error is None and info.supported
    assert [name for name, _ in info.provides] == [
        "largest_magnitude",
        "largest_variance",
        "random",
        "rescaled_prior",
        "swag_variance",
    ]

    monkeypatch.setitem(sys.modules, "laplacekit_subnet", None)

    with pytest.raises(ImportError, match=r'pip install "laplacekit\[subnet\]"'):
        lk.subnet  # noqa: B018


def test_heteroscedastic_alias_and_plugin_record(monkeypatch):
    """`lk.heteroscedastic` is the first-party `laplacekit-heteroscedastic`: an extra of the
    core, listed by `lk.plugins()`, and named in the error when it is absent."""

    import laplacekit_heteroscedastic

    assert lk.heteroscedastic is laplacekit_heteroscedastic
    assert lk.heteroscedastic.plugin == lk.Plugin(supports=">=0.1a0,<0.2")
    assert "heteroscedastic" in metadata.metadata("laplacekit").get_all("Provides-Extra", [])

    (info,) = [p for p in lk.plugins() if p.name == "laplacekit-heteroscedastic"]

    assert info.error is None and info.supported
    assert [name for name, _ in info.provides] == [
        "MeanVarianceGaussian",
        "Moments",
        "NaturalGaussian",
        "NaturalHead",
        "NoisyMean",
        "moments",
    ]

    monkeypatch.setitem(sys.modules, "laplacekit_heteroscedastic", None)

    with pytest.raises(ImportError, match=r'pip install "laplacekit\[heteroscedastic\]"'):
        lk.heteroscedastic  # noqa: B018


def test_reward_alias_and_plugin_record(monkeypatch):
    """`lk.reward` is the first-party `laplacekit-reward`: an extra of the core, listed by
    `lk.plugins()`, and named in the error when it is absent."""

    import laplacekit_reward

    assert lk.reward is laplacekit_reward
    assert lk.reward.plugin == lk.Plugin(supports=">=0.1a0,<0.2")
    assert "reward" in metadata.metadata("laplacekit").get_all("Provides-Extra", [])

    (info,) = [p for p in lk.plugins() if p.name == "laplacekit-reward"]

    assert info.error is None and info.supported
    assert [name for name, _ in info.provides] == [
        "Pair",
        "bradley_terry",
        "pair",
        "pairwise",
        "prepare_pairs",
        "prepare_preferences",
    ]

    monkeypatch.setitem(sys.modules, "laplacekit_reward", None)

    with pytest.raises(ImportError, match=r'pip install "laplacekit\[reward\]"'):
        lk.reward  # noqa: B018


def test_gp_alias_and_plugin_record(monkeypatch):
    """`lk.gp` is the first-party `laplacekit-gp`: an extra of the core, listed by
    `lk.plugins()`, and named in the error when it is absent."""

    import laplacekit_gp

    assert lk.gp is laplacekit_gp
    assert lk.gp.plugin == lk.Plugin(supports=">=0.1a0,<0.2")
    assert "gp" in metadata.metadata("laplacekit").get_all("Provides-Extra", [])

    (info,) = [p for p in lk.plugins() if p.name == "laplacekit-gp"]

    assert info.error is None and info.supported
    names = [name for name, _ in info.provides]
    assert names == [
        "MAX_BASIS_ENTRIES",
        "MAX_CONTEXT_JACOBIAN_ENTRIES",
        "MAX_KMEANS_ENTRIES",
        "RBF",
        "Drawn",
        "FSPPosterior",
        "FunctionPrior",
        "Kernel",
        "Matern",
        "ELLA",
        "NystromGP",
        "Periodic",
        "Picked",
        "Points",
        "Product",
        "RangePrecision",
        "Sum",
        "VaLLA",
        "VariationalGP",
        "fit_variational",
        "fixed",
        "from_data",
        "fsp_laplace",
        "grid",
        "kmeans",
        "rkhs_norm",
        "uniform",
        "variational_loss",
    ]

    monkeypatch.setitem(sys.modules, "laplacekit_gp", None)

    with pytest.raises(ImportError, match=r'pip install "laplacekit\[gp\]"'):
        lk.gp  # noqa: B018


def test_marglik_alias_and_plugin_record(monkeypatch):
    """`lk.marglik` is the first-party `laplacekit-marglik`: an extra of the core, listed by
    `lk.plugins()`, and named in the error when it is absent."""

    import laplacekit_marglik

    assert lk.marglik is laplacekit_marglik
    assert lk.marglik.plugin == lk.Plugin(supports=">=0.1a0,<0.2")
    assert "marglik" in metadata.metadata("laplacekit").get_all("Provides-Extra", [])

    (info,) = [p for p in lk.plugins() if p.name == "laplacekit-marglik"]

    assert info.error is None and info.supported
    assert [name for name, _ in info.provides] == ["Trained", "train"]

    monkeypatch.setitem(sys.modules, "laplacekit_marglik", None)

    with pytest.raises(ImportError, match=r'pip install "laplacekit\[marglik\]"'):
        lk.marglik  # noqa: B018
