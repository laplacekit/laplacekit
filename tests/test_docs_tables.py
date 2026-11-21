"""Tables scroll inside their own box, so a wide one doesn't widen the page."""

import re
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).parents[1]
DOCS = ROOT / "docs"
CONFIG = yaml.safe_load((DOCS / "_quarto.yml").read_text())
THEME = (DOCS / "assets" / "theme.scss").read_text()


def test_filter_is_registered():
    assert "tables.lua" in CONFIG["filters"]


@pytest.mark.skipif(shutil.which("quarto") is None, reason="needs quarto (`make sync`)")
def test_every_table_is_wrapped(tmp_path):
    page = tmp_path / "page.md"
    page.write_text("| a | b |\n|-|-|\n| 1 | 2 |\n\ntext\n\n| c |\n|-|\n| 3 |\n")
    command = ["quarto", "pandoc", str(page), "--lua-filter", str(DOCS / "tables.lua")]
    command += ["--to", "html", "--wrap=none"]
    html = subprocess.run(command, check=True, capture_output=True, text=True).stdout

    wrapped = re.findall(r'<div class="table-scroll">\s*<table', html)

    assert len(wrapped) == 2
    assert html.count("<table") == 2


def test_the_wrapper_scrolls():
    match = re.search(r"^main\.content \.table-scroll \{([^}]*)\}", THEME, re.M)

    assert match
    assert "overflow-x: auto" in match.group(1)
    assert "max-width: 100%" in match.group(1)
