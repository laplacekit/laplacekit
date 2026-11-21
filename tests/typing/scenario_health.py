"""Scenario: check the install, core and plugins, and list the plugins."""

import laplacekit as lk

report: lk.HealthReport = lk.health()
ok: bool = report.ok
checks: tuple[lk.HealthCheck, ...] = report.packages["laplacekit"]
text: str = str(report)


def health() -> tuple[lk.HealthCheck, ...]:
    return (lk.HealthCheck("dep", "ok", "1.0"),)


plugin: lk.Plugin = lk.Plugin(supports=">=0.1a0,<0.2", health=health)
bare: lk.Plugin = lk.Plugin(supports=">=0.1a0")
found: lk.Plugins = lk.plugins()
count: int = len(found)

for info in found:
    name: str = info.name
    supported: bool = info.supported
    provides: tuple[tuple[str, str], ...] = info.provides
    error: str | None = info.error

first: lk.PluginInfo = found[0]
listing: str = str(found)
