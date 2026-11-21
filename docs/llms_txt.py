"""Rewrite the rendered `llms.txt` in the order of the site's sidebars.

Quarto's `website: llms-txt` writes a Markdown copy of every page and an
`llms.txt` that lists them in no particular order. This script keeps the
copies and rewrites the index: the navbar pages first, then one section per
sidebar (Quickstart, In depth, Plugins, Reference), with the sidebar's own
sections and titles. `make docs` runs it after `quarto render`.

Usage: `python docs/llms_txt.py` from the repository root.
"""

import re
import tomllib
from pathlib import Path

import yaml

DOCS = Path(__file__).parent
ROOT = DOCS.parent


def quarto_titles(site: Path) -> dict[str, str]:
    """Page titles from Quarto's own `llms.txt`, keyed by URL."""

    text = (site / "llms.txt").read_text()
    return {
        url: title.replace("**", "")
        for title, url in re.findall(r"^- \[(.*)\]\((.*)\)$", text, re.M)
    }


def build(config: dict, reference: dict, site: Path, description: str) -> str:
    """The new `llms.txt`, from the site config and the rendered `site`."""

    base = config["website"]["site-url"].rstrip("/")
    titles = quarto_titles(site)
    lines = ["# laplacekit", "", f"> {description}", ""]

    def link(href: str, text: str | None, sidebar_title: str) -> str:
        page = Path(href).with_suffix(".llms.md").as_posix()

        if not (site / page).is_file():
            raise FileNotFoundError(
                f"no Markdown copy of {href}: {site / page}"
            )

        url = f"{base}/{page}"
        if text is None and Path(href).stem == "index":
            text = sidebar_title

        return f"- [{text or titles.get(url) or Path(href).stem}]({url})"

    def walk(contents: list, sidebar_title: str) -> None:
        for item in contents:
            if isinstance(item, str):
                lines.append(link(item, None, sidebar_title))

            elif "href" in item:
                lines.append(
                    link(item["href"], item.get("text"), sidebar_title)
                )

            else:
                if "section" in item:
                    lines.extend(["", f"### {item['section']}", ""])

                walk(item["contents"], sidebar_title)

    sidebars = [*config["website"]["sidebar"], reference]
    in_sidebars = {
        href
        for sidebar in sidebars
        for href in re.findall(
            r"[\w./-]+\.(?:qmd|py)", str(sidebar["contents"])
        )
    }

    for item in config["website"]["navbar"]["left"]:
        if item["href"] not in in_sidebars:
            lines.append(link(item["href"], item.get("text"), ""))

    for sidebar in sidebars:
        lines.extend(["", f"## {sidebar['title']}", ""])
        walk(sidebar["contents"], sidebar["title"])

    text = "\n".join(lines) + "\n"
    return re.sub(r"\n{3,}", "\n\n", text)


def main() -> None:
    """Rewrite `docs/_site/llms.txt`."""

    config = yaml.safe_load((DOCS / "_quarto.yml").read_text())
    reference = yaml.safe_load(
        (DOCS / "reference" / "_sidebar.yml").read_text()
    )
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    site = DOCS / config["project"]["output-dir"]

    text = build(
        config, reference["website"]["sidebar"][0], site, project["description"]
    )
    (site / "llms.txt").write_text(text)


if __name__ == "__main__":
    main()
