"""The API reference build: quartodoc with the parser options of `docs/quartodoc_build.py`."""

import importlib.util
from pathlib import Path

import pytest
import quartodoc
import quartodoc.parsers

ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "docs" / "quartodoc_build.py"
SPEC = importlib.util.spec_from_file_location("quartodoc_build", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
BUILD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BUILD)


def returns_rows(name: str) -> list[str]:
    """Body rows of the Returns table on `name`'s reference page."""

    page = quartodoc.MdRenderer().render(
        quartodoc.blueprint(quartodoc.Auto(name=name), package="laplacekit", parser="google")
    )
    table = page.split("## Returns", 1)[1].split("\n\n## ", 1)[0]

    return [line for line in table.splitlines() if line.startswith("|")][2:]


@pytest.fixture
def configured(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(quartodoc.parsers, "DEFAULT_OPTIONS", {})
    BUILD.configure()


@pytest.mark.usefixtures("configured")
def test_wrapped_returns_is_one_row() -> None:
    """`laplace` wraps its `Returns:` text over three unindented lines."""

    rows = returns_rows("laplace")

    assert len(rows) == 1
    assert "post.log_marglik()" in rows[0]


def test_makefile_builds_through_the_script() -> None:
    makefile = (ROOT / "makefile").read_text()

    assert "docs/quartodoc_build.py build --config docs/_quarto.yml" in makefile
