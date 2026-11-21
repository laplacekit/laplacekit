"""The name and the logo: prose writes **laplacekit**, a filter sets it like the
wordmark, and the logo files come in matching light and dark versions."""

import colorsys
import re
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).parents[1]
DOCS = ROOT / "docs"
ASSETS = DOCS / "assets"
CONFIG = yaml.safe_load((DOCS / "_quarto.yml").read_text())
WEBSITE = CONFIG["website"]
BRAND = '<span class="brand">laplace<strong>kit</strong></span>'
INK = {"light": "#1F2937", "dark": "#E6EDF3"}


def contrast(a: str, b: str) -> float:
    """WCAG 2 contrast ratio of two `#RRGGBB` colours."""

    def luminance(hex_colour: str) -> float:
        channels = [int(hex_colour[i : i + 2], 16) / 255 for i in (1, 3, 5)]
        linear = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in channels]

        return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]

    hi, lo = sorted([luminance(a), luminance(b)], reverse=True)

    return (hi + 0.05) / (lo + 0.05)


def scss_value(path: Path, name: str) -> str:
    match = re.search(rf"^\${name}:\s*(#[0-9A-Fa-f]{{6}});", path.read_text(), re.M)

    assert match, f"${name} not in {path.name}"

    return match.group(1)


def prose_files() -> list[Path]:
    """Hand-written docs pages and the README (the reference is generated)."""

    pages = [
        path
        for pattern in ("*.qmd", "*.py")
        for path in DOCS.rglob(pattern)
        if "reference" not in path.relative_to(DOCS).parts
        and "_site" not in path.relative_to(DOCS).parts
        and "node_modules" not in path.relative_to(DOCS).parts
    ]

    return [*pages, ROOT / "README.md"]


# ---------- the name in prose ----------


def test_prose_names_laplacekit():
    for path in prose_files():
        assert "scala" + "place" not in path.read_text(), path


def test_titles_name_laplacekit():
    assert WEBSITE["title"] == "laplacekit"
    assert "Why laplacekit" in [item["text"] for item in WEBSITE["navbar"]["left"]]

    for page in ("index.qmd", "why.qmd"):
        title = yaml.safe_load((DOCS / page).read_text().split("---")[1])["title"]

        assert "**laplacekit**" in title, page


def test_readme_sets_the_name_bold_in_monospace():
    readme = (ROOT / "README.md").read_text()

    assert "<code><b>laplacekit</b></code>" in readme
    assert "<code>laplace<b>kit</b></code>" not in readme


def test_readme_opens_with_the_wordmark_for_both_modes():
    head = (ROOT / "README.md").read_text().split("\n## ")[0]

    assert 'srcset="docs/assets/wordmark-dark.svg"' in head
    assert 'src="docs/assets/wordmark-light.svg"' in head
    assert not head.startswith("# ")


def test_filter_is_registered():
    assert "brand.lua" in CONFIG["filters"]
    assert (DOCS / "brand.lua").is_file()


@pytest.mark.skipif(shutil.which("quarto") is None, reason="needs quarto (`make sync`)")
def test_filter_sets_bold_laplacekit_like_the_wordmark(tmp_path):
    page = tmp_path / "page.md"
    page.write_text(
        "---\ntitle: Why **laplacekit**\n---\n\n"
        "**laplacekit** fits, and **bold** stays bold.\n\n"
        "| | **laplacekit** |\n|-|-|\n| a | b |\n\n"
        "`laplacekit` and laplacekit stay as they are.\n"
    )

    command = ["quarto", "pandoc", str(page), "--lua-filter", str(DOCS / "brand.lua")]
    command += ["--standalone", "--to", "html", "--wrap=none"]
    html = subprocess.run(command, check=True, capture_output=True, text=True).stdout

    assert html.count(BRAND) == 3  # the title, the paragraph, the table header
    assert "<strong>bold</strong>" in html
    assert "<code>laplacekit</code> and laplacekit stay" in html
    assert "<title>Why laplacekit</title>" in html


# ---------- the wordmark font and colour ----------


def test_wordmark_font_ships_with_its_licence():
    brand = (ASSETS / "brand.scss").read_text()
    urls = re.findall(r'url\("([^"]+\.woff2)"\)', brand)

    assert len(urls) == 1  # the name is set bold throughout

    # Quarto resolves theme URLs from the project root and copies the files.
    for url in urls:
        font = DOCS / url

        assert font.is_file(), url
        assert (font.parent / "OFL.txt").is_file()


def test_wordmark_font_does_not_use_the_reserved_name():
    """The OFL reserves "Monaspace" for unmodified fonts, and the subset is modified."""

    assert "Monaspace" not in re.sub(r"//.*", "", (ASSETS / "brand.scss").read_text())


def test_the_whole_name_is_bold():
    rule = re.search(r"\.brand \{([^}]*)\}", (ASSETS / "brand.scss").read_text()).group(1)

    assert "font-weight: 700" in rule


def hue(hex_colour: str) -> float:
    r, g, b = (int(hex_colour[i : i + 2], 16) / 255 for i in (1, 3, 5))

    return colorsys.rgb_to_hls(r, g, b)[0] * 360


@pytest.mark.parametrize("mode", ["light", "dark"])
def test_links_sit_beside_the_brand_colour_without_using_it(mode):
    """Links take the cool end of the magma ramp (indigo); magenta is only for the name."""

    theme = ASSETS / f"theme-{mode}.scss"
    link, brand = scss_value(theme, "lk-link"), scss_value(theme, "lk-brand")
    gap = abs(hue(link) - hue(brand))

    assert 230 <= hue(link) <= 270
    assert min(gap, 360 - gap) >= 45
    assert scss_value(theme, "lk-note") == link

    for name in ("lk-link", "lk-link-hover"):
        assert contrast(scss_value(theme, name), scss_value(theme, "lk-bg")) >= 4.5


@pytest.mark.parametrize("mode", ["light", "dark"])
def test_brand_colour_passes_aa_as_text(mode):
    theme = ASSETS / f"theme-{mode}.scss"

    assert contrast(scss_value(theme, "lk-brand"), scss_value(theme, "lk-bg")) >= 4.5


@pytest.mark.parametrize("mode", ["light", "dark"])
def test_both_theme_modes_load_the_brand_layer(mode):
    assert "assets/brand.scss" in CONFIG["format"]["html"]["theme"][mode]


# ---------- logo files ----------


def test_navbar_shows_the_wordmark_without_a_title():
    navbar = WEBSITE["navbar"]

    assert navbar["logo"] == {
        "light": {"path": "assets/wordmark-light.svg", "alt": "laplacekit"},
        "dark": {"path": "assets/wordmark-dark.svg", "alt": "laplacekit"},
    }
    assert navbar["title"] is False


@pytest.mark.parametrize("name", ["logo", "wordmark"])
def test_dark_version_differs_from_light_only_in_colour(name):
    light = (ASSETS / f"{name}{'-light' if name == 'wordmark' else ''}.svg").read_text()
    dark = (ASSETS / f"{name}-dark.svg").read_text()
    colourless = lambda svg: re.sub(r"#[0-9A-Fa-f]{6}", "#", svg)  # noqa: E731

    assert colourless(light) == colourless(dark)
    assert INK["light"] in light and INK["dark"] not in light
    assert INK["dark"] in dark and INK["light"] not in dark


def test_favicon_follows_the_browser_theme():
    favicon = (ASSETS / "favicon.svg").read_text()

    assert "@media (prefers-color-scheme:dark)" in favicon
    assert INK["light"] in favicon and INK["dark"] in favicon


def test_wordmarks_are_labelled_images():
    for mode in ("light", "dark"):
        svg = (ASSETS / f"wordmark-{mode}.svg").read_text()

        assert 'role="img"' in svg
        assert 'aria-label="laplacekit"' in svg


def test_landing_page_title_carries_the_logo_in_both_modes():
    title = yaml.safe_load((DOCS / "index.qmd").read_text().split("---")[1])["title"]

    assert "![](assets/logo.svg){.brand-mark .light-content}" in title
    assert "![](assets/logo-dark.svg){.brand-mark .dark-content}" in title
    assert title.endswith("**laplacekit**")
    assert ".brand-mark" in (ASSETS / "brand.scss").read_text()


def test_landing_mark_is_spaced_like_the_navbar_wordmark():
    """The title's mark keeps the wordmark's proportions: 28 px mark, 10 px gap,
    20 px text on a 22.2 px baseline, in ems of the text size."""

    rule = re.search(r"\.brand-mark \{([^}]*)\}", (ASSETS / "brand.scss").read_text()).group(1)
    em = {k: float(v) for k, v in re.findall(r"([\w-]+): (-?[\d.]+)em;", rule)}

    assert em == {"height": 28 / 20, "margin-right": 10 / 20, "vertical-align": -round(7.8 / 20, 2)}
