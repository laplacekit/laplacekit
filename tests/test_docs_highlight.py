"""Build-time Shiki highlighting of the docs: the wiring, the palettes, and the
post-render script on a small page."""

import html
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).parents[1]
DOCS = ROOT / "docs"
SHIKI = DOCS / "shiki"
CONFIG = yaml.safe_load((DOCS / "_quarto.yml").read_text())


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


def test_post_render_runs_the_script():
    assert CONFIG["project"]["post-render"] == "node shiki/highlight.mjs"
    assert (SHIKI / "highlight.mjs").is_file()


def test_shiki_loads_every_language_the_pages_fence():
    """Blocks in other languages keep Quarto's colours, which do not match the rest."""

    langs = re.search(r"const LANGS = \[(.*?)\];", (SHIKI / "highlight.mjs").read_text())
    loaded = set(re.findall(r'"([\w-]+)"', langs.group(1)))
    plain = {"mermaid", "text"}  # a diagram and plain text: nothing to highlight
    fenced = set()

    for page in DOCS.rglob("*"):
        parts = page.relative_to(DOCS).parts

        if page.suffix in {".qmd", ".py", ".ejs"} and not {"_site", "node_modules"} & set(parts):
            fenced |= set(re.findall(r"^(?:# )?```\{?\.?([\w-]+)", page.read_text(), re.M))

    assert fenced - plain <= loaded, fenced - plain - loaded


def test_shiki_is_pinned_with_a_lockfile():
    wanted = json.loads((DOCS / "package.json").read_text())["dependencies"]["shiki"]
    lock = json.loads((DOCS / "package-lock.json").read_text())

    assert re.fullmatch(r"\d+\.\d+\.\d+", wanted)
    assert lock["packages"]["node_modules/shiki"]["version"] == wanted


def test_make_docs_installs_the_lockfile():
    makefile = (ROOT / "makefile").read_text()

    assert "npm ci --prefix docs" in makefile
    assert re.search(r"^docs: .*docs/node_modules/\.package-lock\.json", makefile, re.M)


@pytest.mark.parametrize("mode", ["light", "dark"])
def test_token_colours_pass_aa_on_the_code_background(mode):
    theme = json.loads((SHIKI / f"laplacekit-{mode}.json").read_text())
    background = theme["colors"]["editor.background"]
    colours = [theme["colors"]["editor.foreground"]]
    colours += [rule["settings"]["foreground"] for rule in theme["tokenColors"]]

    surface = scss_value(DOCS / "assets" / f"theme-{mode}.scss", "lk-surface")

    assert background.upper() == surface.upper()

    for colour in colours:
        assert contrast(colour, background) >= 4.5, f"{mode}: {colour} on {background}"


def test_dark_code_blocks_are_lighter_than_the_page():
    dark = DOCS / "assets" / "theme-dark.scss"
    page, code = scss_value(dark, "lk-bg"), scss_value(dark, "lk-surface")

    assert contrast(code, "#000000") > contrast(page, "#000000")


def line(block: int, n: int, body: str) -> str:
    """One code line as Quarto writes it."""

    anchor = f"cb{block}-{n}"
    link = f'<a href="#{anchor}" aria-hidden="true" tabindex="-1"></a>'

    return f'<span id="{anchor}">{link}{body}</span>'


def tok(kind: str, text: str) -> str:
    """A token as Quarto's own highlighter marks it."""

    return f'<span class="{kind}">{text}</span>'


PYTHON = "\n".join(
    [
        line(1, 1, f"{tok('im', 'import')} laplacekit {tok('im', 'as')} lk"),
        line(1, 2, ""),
        line(
            1, 3, f"post {tok('op', '=')} lk.laplace(model, loader, likelihood{tok('op', '=')}lik)"
        ),
        line(
            1,
            4,
            f"{tok('bu', 'print')}(post.mean, {tok('st', '&quot;a &lt; b &amp; c&quot;')})"
            f"  {tok('co', '# note')}",
        ),
    ]
)
OUTPUT = "<pre><code>tensor([0.1, 0.9])  # print(x)</code></pre>"
MERMAID = f'<code class="sourceCode mermaid">{line(2, 1, "graph LR")}</code>'
PAGE = f"""<html><body>
<div class="sourceCode cell-code" id="cb1"><pre class="sourceCode python code-with-copy">\
<code class="sourceCode python">{PYTHON}</code></pre></div>
<div class="cell-output cell-output-stdout">
{OUTPUT}
</div>
<div class="sourceCode" id="cb2"><pre class="sourceCode mermaid">{MERMAID}</pre></div>
</body></html>
"""

SOURCE = "\n".join(
    [
        "import laplacekit as lk",
        "",
        "post = lk.laplace(model, loader, likelihood=lik)",
        'print(post.mean, "a < b & c")  # note',
    ]
)


def colour_of(block: str, text: str) -> str:
    """The light colour of the token spelling `text`."""

    token = rf'<span style="--shiki-light:(#[0-9A-Fa-f]{{6}})[^"]*">{re.escape(text)}</span>'
    match = re.search(token, block)

    assert match, f"no token {text!r}"

    return match.group(1).upper()


@pytest.mark.skipif(
    shutil.which("node") is None or not (DOCS / "node_modules" / "shiki").is_dir(),
    reason="needs node and `npm ci --prefix docs`",
)
def test_post_render_highlights_source_and_leaves_outputs(tmp_path):
    page = tmp_path / "page.html"
    page.write_text(PAGE)
    env = {k: v for k, v in os.environ.items() if not k.startswith("QUARTO_")}

    command = ["node", str(SHIKI / "highlight.mjs"), str(page)]
    subprocess.run(command, check=True, env=env, capture_output=True)
    out = page.read_text()

    python = re.search(r'<code class="sourceCode python"[^>]*>(.*?)</code>', out, re.S).group(0)
    anchor = r'<span id="(cb1-\d)"><a href="#\1" aria-hidden="true" tabindex="-1"></a>'
    lines = re.findall(anchor, python)
    text = html.unescape(re.sub(r"<[^>]+>", "", python))

    assert 'data-shiki=""' in python
    assert "--shiki-dark:" in python
    assert lines == ["cb1-1", "cb1-2", "cb1-3", "cb1-4"]
    assert text == SOURCE

    function, parameter = colour_of(python, "laplace"), colour_of(python, "likelihood")
    attribute = colour_of(python, ".mean") if ">.mean<" in python else colour_of(python, "mean")

    assert len({function, parameter, attribute}) == 3

    assert OUTPUT in out
    assert MERMAID in out
