"""`lk.health()` and `lk.plugins()`: the install, the core and each registered plugin.

`health`, `HealthCheck`, `HealthReport`, `Plugin`, `PluginInfo`, `Plugins` and `plugins` are public
through `laplacekit`; this module is private.

A plugin registers one `Plugin` record under the entry-point group `laplacekit.plugins`,
named after its import package:

```toml
[project.entry-points."laplacekit.plugins"]
laplacekit_subnet = "laplacekit_subnet:plugin"
```

Discovery reads the installed metadata. A plugin is imported only when `lk.health()` or
`lk.plugins()` runs. A plugin without the entry point is not listed.
"""

import importlib
import inspect
import re
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from importlib import metadata
from types import MappingProxyType
from typing import Literal, get_args

import torch
from packaging.specifiers import InvalidSpecifier, SpecifierSet
from torch import nn
from torch.torch_version import TorchVersion

import laplacekit.core as core
from laplacekit.core import Batch, ModelFn
from laplacekit.likelihoods import Gaussian

__all__ = ["HealthCheck", "Plugin", "PluginInfo", "Plugins", "HealthReport", "health", "plugins"]

type Status = Literal["ok", "warning", "error"]

_GROUP = "laplacekit.plugins"

TORCH_FLOOR = "2.15"
CURVLINOPS = "curvlinops-for-pytorch"
CURVLINOPS_PIN = "3.0.1"

_CORE = "laplacekit"


@dataclass(frozen=True)
class HealthCheck:
    """One result of `lk.health()`.

    Args:
        name: What was checked, such as `"torch"`.
        status: `"ok"`, `"warning"` (works, with something missing) or `"error"`
            (broken).
        detail: What was found, in one line.

    Raises:
        ValueError: `status` is not `"ok"`, `"warning"` or `"error"`.
    """

    name: str
    status: Status
    detail: str

    def __post_init__(self) -> None:
        if self.status not in get_args(Status.__value__):
            raise ValueError(
                f"HealthCheck status must be 'ok', 'warning' or 'error', got {self.status!r}"
            )


@dataclass(frozen=True)
class HealthReport:
    """What `lk.health()` found, per package. `print` lists the checks.

    Args:
        packages: The checks of each package, the core (`"laplacekit"`) first. It is
            stored as a read-only mapping.
    """

    packages: Mapping[str, tuple[HealthCheck, ...]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "packages", MappingProxyType(dict(self.packages)))

    @property
    def ok(self) -> bool:
        """True when no check is an error. Warnings still count as ok."""

        return all(c.status != "error" for checks in self.packages.values() for c in checks)

    def __str__(self) -> str:
        lines: list[str] = []

        for pkg, checks in self.packages.items():
            if lines:
                lines.append("")

            lines.append(pkg)
            lines.extend(f"- {c.status.upper()} {c.name}: {c.detail}" for c in checks)

        return "\n".join(lines)


@dataclass(frozen=True)
class Plugin:
    """A plugin's record, the target of its `laplacekit.plugins` entry point.

    The entry point is named after the plugin's import package.

    Args:
        supports: PEP 440 specifier set of the laplacekit versions the plugin works with,
            such as `">=0.1a0,<0.2"`. Pre-releases count.
        health: Returns the plugin's own checks for `lk.health()`. The default returns
            none. If it raises, or returns anything but `HealthCheck` objects, `lk.health()`
            reports one `error` check in their place.

    Raises:
        ValueError: `supports` is not a valid PEP 440 specifier set.
    """

    supports: str
    health: Callable[[], tuple[HealthCheck, ...]] = lambda: ()

    def __post_init__(self) -> None:
        try:
            SpecifierSet(self.supports)
        except InvalidSpecifier:
            raise ValueError(
                "Plugin supports must be a PEP 440 specifier set such as '>=0.1a0,<0.2', "
                f"got {self.supports!r}"
            ) from None


@dataclass(frozen=True)
class PluginInfo:
    """One registered plugin: its package metadata, its `supports` range, what it provides.

    `lk.plugins()` builds these. `print` shows one.

    Args:
        name: The distribution's name, or the entry name when the entry point has no
            distribution.
        version: The distribution's version (`""` without a distribution).
        summary: The distribution's summary (`""` without one).
        supported: Whether the installed laplacekit is in `supports`. False when the
            plugin could not be loaded.
        supports: The record's `supports` range (`""` when the plugin could not be loaded).
        provides: Each name in the plugin module's `__all__`, other than its `Plugin`
            record and `health` function, paired with the first sentence of its docstring
            (`""` without one).
        error: Why the plugin could not be loaded; `provides` is then empty. `None` when
            it loaded.
    """

    name: str
    version: str
    summary: str
    supported: bool
    supports: str
    provides: tuple[tuple[str, str], ...] = ()
    error: str | None = None

    def __str__(self) -> str:
        lines = [f"{self.name} {self.version}  {self.summary}".rstrip()]

        if self.error is not None:
            lines.append(f"  error: {self.error}")
        elif not self.supported:
            lines.append(f"  not supported: needs laplacekit {self.supports}")

        lines.extend(f"  {name}: {doc}" if doc else f"  {name}" for name, doc in self.provides)

        return "\n".join(lines)


@dataclass(frozen=True)
class Plugins:
    """The registered plugins. `print` lists them.

    It iterates, indexes and has a length like a tuple of `PluginInfo`.

    Args:
        plugins: One `PluginInfo` per plugin, sorted by distribution name.
    """

    plugins: tuple[PluginInfo, ...] = ()

    def __iter__(self) -> Iterator[PluginInfo]:
        return iter(self.plugins)

    def __len__(self) -> int:
        return len(self.plugins)

    def __getitem__(self, i: int, /) -> PluginInfo:
        return self.plugins[i]

    def __str__(self) -> str:
        return "\n\n".join(str(p) for p in self.plugins)


def _torch() -> HealthCheck:
    version = TorchVersion(torch.__version__)

    if version < TORCH_FLOOR:
        return HealthCheck("torch", "error", f"{version} is older than the required {TORCH_FLOOR}")

    return HealthCheck("torch", "ok", f"{version} (>= {TORCH_FLOOR})")


def _curvlinops() -> HealthCheck:
    try:
        version = metadata.version(CURVLINOPS)
    except metadata.PackageNotFoundError:
        return HealthCheck(CURVLINOPS, "error", f"not installed; KFAC needs =={CURVLINOPS_PIN}")

    if version != CURVLINOPS_PIN:
        return HealthCheck(
            CURVLINOPS,
            "error",
            f"{version} installed, but the KFAC adapter is pinned to =={CURVLINOPS_PIN}",
        )

    return HealthCheck(CURVLINOPS, "ok", version)


def _autograd() -> HealthCheck:
    """Probe a 2-layer MLP on the CPU."""

    with torch.random.fork_rng(devices=[]):  # leaves the caller's RNG state alone
        torch.manual_seed(0)
        module = nn.Sequential(nn.Linear(3, 4), nn.Tanh(), nn.Linear(4, 2))
        x = torch.randn(2, 3)
        y = torch.randn(2, 2)

    caps = core.probe(ModelFn.from_module(module), Gaussian(sigma=1.0), Batch(x, y))

    found = {"vjp": caps.vjp, "jvp": caps.jvp, "vmap": caps.vmap, "vjp_twice": caps.vjp_twice}
    missing = [t for t, ok in found.items() if not ok]

    if missing:
        why = "; ".join(f"{t}: {caps.failures.get(t, 'failed')}" for t in missing)
        return HealthCheck("autograd", "error", f"the probe of a 2-layer MLP failed: {why}")

    return HealthCheck("autograd", "ok", f"{', '.join(found)} work on a 2-layer CPU MLP")


def _float64() -> HealthCheck:
    x = torch.ones(2, 2, dtype=torch.float64)
    torch.linalg.cholesky(x + torch.eye(2, dtype=torch.float64))

    return HealthCheck("float64", "ok", "tensors and a Cholesky factor in float64 work")


_CORE_CHECKS: tuple[tuple[str, Callable[[], HealthCheck]], ...] = (
    ("torch", _torch),
    (CURVLINOPS, _curvlinops),
    ("autograd", _autograd),
    ("float64", _float64),
)


def _core() -> tuple[HealthCheck, ...]:
    """Run each core check. One that raises becomes an `error` check under its name."""

    checks: list[HealthCheck] = []

    for name, check in _CORE_CHECKS:
        try:
            checks.append(check())
        except Exception as e:
            checks.append(HealthCheck(name, "error", f"{type(e).__name__}: {e}"))

    return tuple(checks)


def _run(load: Callable[[], Callable[[], Iterable[object]]]) -> tuple[HealthCheck, ...]:
    """Load and call one plugin's `health`. Any exception becomes one `error` check."""

    try:
        checks = tuple(load()())
    except Exception as e:
        return (HealthCheck("health", "error", f"{type(e).__name__}: {e}"),)

    good = tuple(c for c in checks if isinstance(c, HealthCheck))

    if len(good) < len(checks):
        wrong = ", ".join(
            sorted({type(c).__name__ for c in checks if not isinstance(c, HealthCheck)})
        )
        return (
            HealthCheck(
                "health", "error", f"health() must return HealthCheck objects, got {wrong}"
            ),
        )

    return good


def _dist_name(ep: metadata.EntryPoint) -> str:
    return ep.dist.name if ep.dist is not None else ep.name


def _load(ep: metadata.EntryPoint) -> Plugin:
    """Import the entry point's target. Anything but a `Plugin` raises `TypeError`."""

    target = ep.load()

    if not isinstance(target, Plugin):
        raise TypeError(
            f"entry point {ep.name!r} ({ep.value}) must name an lk.Plugin, "
            f"got {type(target).__name__}"
        )

    return target


def _supported(plugin: Plugin, version: str) -> bool:
    return SpecifierSet(plugin.supports).contains(version, prereleases=True)


def _supports(ep: metadata.EntryPoint, plugin: Plugin) -> HealthCheck:
    version = metadata.version(_CORE)

    if not _supported(plugin, version):
        detail = f"{_dist_name(ep)} supports {plugin.supports}, installed laplacekit {version}"
        return HealthCheck("supports", "error", detail)

    return HealthCheck("supports", "ok", f"laplacekit {version} is in {plugin.supports}")


def _plugin(ep: metadata.EntryPoint) -> tuple[HealthCheck, ...]:
    """The `supports` check, then the plugin's own. A record that fails to load is one error."""

    try:
        plugin = _load(ep)
    except Exception as e:
        return (HealthCheck("plugin", "error", f"{type(e).__name__}: {e}"),)

    return (_supports(ep, plugin), *_run(lambda: plugin.health))


def _first_line(obj: object) -> str:
    """The first sentence of the docstring, joined across wrapped lines."""

    # Any other value would show the docstring of its type.
    if not (inspect.isclass(obj) or inspect.isroutine(obj) or inspect.ismodule(obj)):
        return ""

    doc = inspect.getdoc(obj)

    if not doc:
        return ""

    paragraph = " ".join(doc.split("\n\n")[0].split())

    return re.split(r"(?<=[.!?])\s", paragraph, maxsplit=1)[0]


def _provides(ep: metadata.EntryPoint, plugin: Plugin) -> tuple[tuple[str, str], ...]:
    module = importlib.import_module(ep.module)
    names: Iterable[str] = vars(module).get("__all__", ())
    found: list[tuple[str, str]] = []

    for name in names:
        obj = getattr(module, name)

        if obj is plugin or obj is plugin.health:
            continue

        found.append((name, _first_line(obj)))

    return tuple(found)


def _info(ep: metadata.EntryPoint) -> PluginInfo:
    dist = ep.dist
    version = dist.version if dist is not None else ""
    summary = (dist.metadata.get("Summary") or "") if dist is not None else ""

    try:
        plugin = _load(ep)
        provides = _provides(ep, plugin)
    except Exception as e:
        error = f"{type(e).__name__}: {e}"
        return PluginInfo(_dist_name(ep), version, summary, False, "", error=error)

    supported = _supported(plugin, metadata.version(_CORE))

    return PluginInfo(_dist_name(ep), version, summary, supported, plugin.supports, provides)


def plugins() -> Plugins:
    """List the plugins registered under `laplacekit.plugins`, by distribution name.

    Each plugin is imported. One that fails to import, or whose entry point names something
    other than a `Plugin`, is listed with its `error`.

    Returns:
        A `Plugins`, one `PluginInfo` per entry point, sorted by distribution name.
    """

    found = (_info(ep) for ep in metadata.entry_points(group=_GROUP))

    return Plugins(tuple(sorted(found, key=lambda p: p.name)))


def health() -> HealthReport:
    """HealthCheck the install: the core, then each plugin registered under `laplacekit.plugins`.

    Core checks: the torch version floor, the exact Curvlinops pin, a probe of a 2-layer
    CPU MLP (vjp, jvp, vmap, vjp_twice) and float64 support. Each plugin gets a `supports`
    check (is the installed laplacekit in its range?), then its own checks. A check or a
    plugin `health` that raises becomes one `error` check naming the exception, and so does
    a plugin that fails to load. Plugins are listed by entry name. An entry name registered
    more than once is one `error` check naming each registration, and none of them runs.
    The entry name `"laplacekit"` is reserved: registering it adds one `error` check to
    the core's checks, and the plugin does not run.

    Returns:
        A `HealthReport` with the core's checks under `"laplacekit"` first, then each plugin's
        checks under its entry name.
    """

    packages: dict[str, tuple[HealthCheck, ...]] = {_CORE: _core()}
    by_name: dict[str, list[metadata.EntryPoint]] = {}

    for ep in metadata.entry_points(group=_GROUP):
        by_name.setdefault(ep.name, []).append(ep)

    for name in sorted(by_name):
        eps = by_name[name]

        if name == _CORE:  # a plugin can't replace the core's checks
            packages[_CORE] += (_duplicate(name, eps),)
        elif len(eps) > 1:
            packages[name] = (_duplicate(name, eps),)
        else:
            packages[name] = _plugin(eps[0])

    return HealthReport(packages)


def _duplicate(name: str, eps: list[metadata.EntryPoint]) -> HealthCheck:
    def where(ep: metadata.EntryPoint) -> str:
        dist = ep.dist.name if ep.dist is not None else "an unknown distribution"

        return f"{ep.value} (from {dist})"

    regs = ", ".join(where(ep) for ep in eps)

    if name == _CORE:
        return HealthCheck(
            "health", "error", f"entry name {name!r} is reserved for the core: {regs}"
        )

    return HealthCheck(
        "health", "error", f"entry name {name!r} is registered {len(eps)} times: {regs}"
    )
