import subprocess
import sys
from importlib import metadata

import pytest
import torch

import laplacekit
import laplacekit as lk
from laplacekit import HealthCheck, HealthReport

CORE = "laplacekit"


class FakeEntryPoint:
    """Stands in for an installed `laplacekit.plugins` entry point; `fn` is its `health`."""

    group = "laplacekit.plugins"

    def __init__(self, name, fn, value="pkg:plugin", dist=None):
        self.name = name
        self.value = value
        self.module = value.partition(":")[0]
        self.dist = dist
        self._fn = fn

    def load(self):
        return lk.Plugin(supports=">=0", health=self._fn)


class FakeDist:
    def __init__(self, name):
        self.name = name


def fake_entry_points(monkeypatch, *eps):
    def entry_points(*, group):
        assert group == "laplacekit.plugins"

        return metadata.EntryPoints(eps)

    monkeypatch.setattr(metadata, "entry_points", entry_points)


def test_health_core_ok(monkeypatch):
    fake_entry_points(monkeypatch)  # the core only

    report = lk.health()
    core = report.packages[CORE]

    assert list(report.packages) == [CORE]
    names = [c.name for c in core]

    assert names == ["torch", "curvlinops-for-pytorch", "autograd", "float64"]
    assert all(c.status == "ok" for c in core), core
    assert torch.__version__.split("+")[0] in core[0].detail
    assert "3.0.1" in core[1].detail

    for t in ("vjp", "jvp", "vmap", "vjp_twice"):
        assert t in core[2].detail


def test_health_core_comes_first(monkeypatch):
    fake_entry_points(monkeypatch)  # the core only

    assert next(iter(lk.health().packages)) == CORE


def test_health_curvlinops_wrong_pin_is_error(monkeypatch):
    fake_entry_points(monkeypatch)  # the core only

    real = metadata.version

    def version(dist):
        return "3.0.0" if dist == "curvlinops-for-pytorch" else real(dist)

    monkeypatch.setattr(metadata, "version", version)
    report = lk.health()
    check = report.packages[CORE][1]

    assert check.status == "error"
    assert "3.0.0" in check.detail and "3.0.1" in check.detail
    assert not report.ok


def test_health_curvlinops_missing_is_error(monkeypatch):
    fake_entry_points(monkeypatch)  # the core only

    real = metadata.version

    def version(dist):
        if dist == "curvlinops-for-pytorch":
            raise metadata.PackageNotFoundError(dist)

        return real(dist)

    monkeypatch.setattr(metadata, "version", version)
    check = lk.health().packages[CORE][1]

    assert check.status == "error"
    assert "not installed" in check.detail


def test_health_old_torch_is_error(monkeypatch):
    fake_entry_points(monkeypatch)  # the core only

    monkeypatch.setattr(torch, "__version__", torch.torch_version.TorchVersion("2.14.1"))
    check = lk.health().packages[CORE][0]

    assert check.status == "error"
    assert "2.14.1" in check.detail and "2.15" in check.detail


def test_health_collects_installed_plugins():
    report = lk.health()

    with_checks = ["laplacekit_transformers", "laplacekit_diffusers"]

    assert set(with_checks) <= set(report.packages)

    for pkg in with_checks:
        checks = report.packages[pkg]

        assert checks[0].name == "supports"
        assert len(checks) > 1
        assert all(isinstance(c, HealthCheck) for c in checks)
        assert all(c.status == "ok" for c in checks), checks

    # The subnet, heteroscedastic, reward, gp and marglik plugins have no checks of their own,
    # only `supports`.
    for pkg in (
        "laplacekit_subnet",
        "laplacekit_heteroscedastic",
        "laplacekit_reward",
        "laplacekit_gp",
        "laplacekit_marglik",
    ):
        (supports,) = report.packages[pkg]

        assert supports.name == "supports" and supports.status == "ok"

    assert report.ok


def test_health_plugin_raising_is_an_error_check(monkeypatch):
    def broken():
        raise RuntimeError("no GPU driver")

    def fine():
        return (HealthCheck("thing", "ok", "works"),)

    fake_entry_points(monkeypatch, FakeEntryPoint("broken", broken), FakeEntryPoint("fine", fine))
    report = lk.health()

    assert list(report.packages) == [CORE, "broken", "fine"]

    supports, check = report.packages["broken"]

    assert supports.name == "supports" and supports.status == "ok"
    assert check.status == "error"
    assert "RuntimeError" in check.detail and "no GPU driver" in check.detail
    assert report.packages["fine"][1:] == (HealthCheck("thing", "ok", "works"),)
    assert not report.ok


def test_health_plugin_failing_to_import_is_an_error_check(monkeypatch):
    class Unloadable(FakeEntryPoint):
        def load(self):
            raise ModuleNotFoundError("No module named 'transformers'", name="transformers")

    fake_entry_points(monkeypatch, Unloadable("laplacekit_transformers", None))
    (check,) = lk.health().packages["laplacekit_transformers"]

    assert check.status == "error"
    assert "ModuleNotFoundError" in check.detail and "transformers" in check.detail


def test_health_plugin_returning_non_checks_is_an_error_check(monkeypatch):
    fake_entry_points(monkeypatch, FakeEntryPoint("odd", lambda: ("all good",)))
    _, check = lk.health().packages["odd"]

    assert check.status == "error"
    assert "str" in check.detail


def test_health_without_plugins_lists_only_the_core(monkeypatch):
    fake_entry_points(monkeypatch)

    assert list(lk.health().packages) == [CORE]


def test_health_report_prints():
    report = HealthReport(
        {
            "laplacekit": (HealthCheck("torch", "ok", "2.15.0"),),
            "laplacekit_transformers": (
                HealthCheck("transformers", "ok", "5.23.0"),
                HealthCheck("peft", "warning", "not installed"),
            ),
            "broken": (HealthCheck("health", "error", "RuntimeError: boom"),),
        }
    )
    text = str(report)
    lines = text.splitlines()

    assert lines[0] == "laplacekit"
    assert "- OK torch: 2.15.0" in lines
    assert "- WARNING peft: not installed" in lines
    assert "- ERROR health: RuntimeError: boom" in lines
    assert text.index("laplacekit_transformers") < text.index("broken")
    assert not report.ok
    assert HealthReport({"x": (HealthCheck("a", "warning", "b"),)}).ok


def test_check_rejects_unknown_status():
    with pytest.raises(ValueError, match="status"):
        HealthCheck("a", "fine", "b")  # type: ignore[arg-type]


def test_importing_laplacekit_imports_no_plugin():
    code = (
        "import sys\n"
        "import laplacekit as lk\n"
        "assert 'laplacekit_transformers' not in sys.modules\n"
        "assert 'laplacekit_subnet' not in sys.modules\n"
        "assert 'laplacekit_heteroscedastic' not in sys.modules\n"
        "assert 'laplacekit_reward' not in sys.modules\n"
        "assert 'laplacekit_gp' not in sys.modules\n"
        "assert 'laplacekit_marglik' not in sys.modules\n"
    )
    run = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)

    assert run.returncode == 0, run.stderr


def test_health_probe_failure_is_error(monkeypatch):
    fake_entry_points(monkeypatch)  # the core only

    caps = lk.core.Capabilities(
        vjp=True, vmap=True, failures={"jvp": "aten::foo", "vjp_twice": "bar"}
    )
    monkeypatch.setattr(lk.core, "probe", lambda *args, **kwargs: caps)
    check = lk.health().packages[CORE][2]

    assert check.status == "error"
    assert "jvp: aten::foo" in check.detail and "vjp_twice: bar" in check.detail
    assert "vmap" not in check.detail


def test_health_leaves_the_rng_alone():
    torch.manual_seed(123)
    want = torch.rand(3)
    torch.manual_seed(123)
    lk.health()

    torch.testing.assert_close(torch.rand(3), want)


def test_health_is_the_callable():
    assert callable(lk.health)
    assert lk.health is laplacekit.health
    assert isinstance(lk.health(), HealthReport)


def test_health_raising_core_check_is_an_error_check(monkeypatch):
    fake_entry_points(monkeypatch)

    def cholesky(*args, **kwargs):
        raise ValueError("no LAPACK")

    monkeypatch.setattr(torch.linalg, "cholesky", cholesky)
    report = lk.health()
    names = [c.name for c in report.packages[CORE]]
    check = report.packages[CORE][3]

    assert names == ["torch", "curvlinops-for-pytorch", "autograd", "float64"]
    assert check.status == "error"
    assert "ValueError" in check.detail and "no LAPACK" in check.detail
    assert not report.ok


def test_health_duplicate_entry_name_is_an_error_check(monkeypatch):
    calls = []

    def fn():
        calls.append(1)

        return (HealthCheck("x", "ok", ""),)

    fake_entry_points(
        monkeypatch,
        FakeEntryPoint("dup", fn, "dup_a:plugin", FakeDist("dist-a")),
        FakeEntryPoint("dup", fn, "dup_b:plugin", FakeDist("dist-b")),
    )
    (check,) = lk.health().packages["dup"]

    assert check.status == "error"
    assert "2 times" in check.detail
    assert "dup_a:plugin (from dist-a)" in check.detail
    assert "dup_b:plugin (from dist-b)" in check.detail
    assert calls == []


def test_health_plugin_cannot_replace_the_core(monkeypatch):
    def fn():
        return (HealthCheck("fake", "ok", "all fine"),)

    fake_entry_points(monkeypatch, FakeEntryPoint(CORE, fn, "evil:plugin", FakeDist("evil")))
    checks = lk.health().packages[CORE]

    assert [c.name for c in checks] == [
        "torch",
        "curvlinops-for-pytorch",
        "autograd",
        "float64",
        "health",
    ]
    assert checks[-1].status == "error"
    assert "reserved" in checks[-1].detail and "evil:plugin" in checks[-1].detail
