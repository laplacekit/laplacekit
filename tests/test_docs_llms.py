"""The Markdown docs for coding agents: llms.txt and the ways to find it."""

import shutil
import subprocess
import tomllib
from pathlib import Path

import pytest
import yaml

import laplacekit as lk

ROOT = Path(__file__).parents[1]
DOCS = ROOT / "docs"
CONFIG = yaml.safe_load((DOCS / "_quarto.yml").read_text())
LLMS_TXT = "https://laplacekit.org/llms.txt"


def test_site_publishes_llms_txt():
    assert CONFIG["website"]["llms-txt"] is True


def test_package_docstring_points_to_llms_txt():
    """Coding agents read the installed package, not the website."""

    assert LLMS_TXT in (lk.__doc__ or "")


def test_pypi_links_to_llms_txt():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]

    assert LLMS_TXT in project["urls"].values()


def test_every_page_footer_links_to_llms_txt():
    assert LLMS_TXT in str(CONFIG["website"]["page-footer"])


def test_filter_is_registered():
    assert "llms-link.lua" in CONFIG["filters"]


@pytest.mark.skipif(shutil.which("uv") is None, reason="needs uv for Quarto's pandoc")
def test_each_page_links_its_markdown_copy(tmp_path):
    """The filter adds `<link rel="alternate">` to the page's own `.llms.md`."""

    source = tmp_path / "page.md"
    source.write_text("# A page\n\nSome text.\n")

    out = tmp_path / "basics.html"
    subprocess.run(
        [
            "uv",
            "run",
            "quarto",
            "pandoc",
            "--standalone",
            "--metadata",
            "title=A page",
            "--lua-filter",
            str(DOCS / "llms-link.lua"),
            "--output",
            str(out),
            str(source),
        ],
        check=True,
        cwd=ROOT,
    )

    head = out.read_text().split("</head>")[0]
    assert '<link rel="alternate" type="text/markdown" href="basics.llms.md">' in head


def load_llms_txt():
    import importlib.util

    spec = importlib.util.spec_from_file_location("llms_txt", DOCS / "llms_txt.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_llms_txt_follows_the_sidebars(tmp_path):
    """Sections in sidebar order, nav titles, and Quarto's titles elsewhere."""

    for page in ["index", "quickstart/basics", "reference/laplace", "reference/index"]:
        (tmp_path / f"{page}.llms.md").parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / f"{page}.llms.md").write_text("x")

    (tmp_path / "llms.txt").write_text(
        "# laplacekit\n\n## Pages\n\n"
        "- [laplace](https://laplacekit.org/reference/laplace.llms.md)\n"
        "- [Reference](https://laplacekit.org/reference/index.llms.md)\n"
    )
    config = {
        "website": {
            "site-url": "https://laplacekit.org",
            "navbar": {"left": [{"text": "Home", "href": "index.qmd"}]},
            "sidebar": [
                {
                    "title": "Quickstart",
                    "contents": [
                        {
                            "section": "Start",
                            "contents": [{"text": "Basics", "href": "quickstart/basics.py"}],
                        }
                    ],
                }
            ],
        }
    }
    reference = {
        "title": "Reference",
        "contents": [
            "reference/index.qmd",
            {"contents": [{"section": "Fit", "contents": ["reference/laplace.qmd"]}]},
        ],
    }

    text = load_llms_txt().build(config, reference, tmp_path, "A library.")

    assert text == (
        "# laplacekit\n\n"
        "> A library.\n\n"
        "- [Home](https://laplacekit.org/index.llms.md)\n\n"
        "## Quickstart\n\n"
        "### Start\n\n"
        "- [Basics](https://laplacekit.org/quickstart/basics.llms.md)\n\n"
        "## Reference\n\n"
        "- [Reference](https://laplacekit.org/reference/index.llms.md)\n\n"
        "### Fit\n\n"
        "- [laplace](https://laplacekit.org/reference/laplace.llms.md)\n"
    )


def test_llms_txt_refuses_a_page_without_its_markdown_copy(tmp_path):
    (tmp_path / "llms.txt").write_text("# laplacekit\n")
    config = {
        "website": {
            "site-url": "https://laplacekit.org",
            "navbar": {"left": []},
            "sidebar": [{"title": "Q", "contents": [{"text": "B", "href": "q/b.py"}]}],
        }
    }

    with pytest.raises(FileNotFoundError, match="q/b.llms.md"):
        load_llms_txt().build(config, {"title": "R", "contents": []}, tmp_path, "A.")


def test_make_docs_writes_llms_txt_after_the_render():
    makefile = (ROOT / "makefile").read_text()
    recipe = makefile.split("\ndocs:", 1)[1].split("\n\n", 1)[0]

    assert recipe.index("render docs") < recipe.index("docs/llms_txt.py")
