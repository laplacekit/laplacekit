"""The docs config: theme files, light and dark modes, navbar, and section sidebars."""

import re
import tomllib
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).parents[1]
DOCS = ROOT / "docs"
CONFIG = yaml.safe_load((DOCS / "_quarto.yml").read_text())
WEBSITE = CONFIG["website"]
HTML = CONFIG["format"]["html"]
SECTIONS = ("quickstart", "examples", "plugins")


def hrefs(contents: list) -> list[str]:
    """Every page a sidebar lists, sections included, in order."""

    found = []

    for item in contents:
        if isinstance(item, str):
            found.append(item)
        elif "href" in item:
            found.append(item["href"])
        else:
            found.extend(hrefs(item.get("contents", [])))

    return found


def sidebars() -> dict[str, list[str]]:
    """The pages of each sidebar in `_quarto.yml`, by id."""

    return {bar["id"]: hrefs(bar.get("contents", [])) for bar in WEBSITE["sidebar"]}


def test_theme_has_light_and_dark():
    assert set(HTML["theme"]) == {"light", "dark"}
    assert set(HTML["highlight-style"]) == {"light", "dark"}


@pytest.mark.parametrize("mode", ["light", "dark"])
def test_theme_files_exist(mode):
    layers = HTML["theme"][mode]
    files = [layer for layer in layers if layer.endswith(".scss")]

    assert files, f"no SCSS layer in the {mode} theme"

    for name in files:
        assert (DOCS / name).is_file(), name


def test_logos_exist_for_both_modes():
    logo = WEBSITE["navbar"]["logo"]

    assert set(logo) == {"light", "dark"}

    for name in [*(mode["path"] for mode in logo.values()), WEBSITE["favicon"]]:
        assert (DOCS / name).is_file(), name


def test_navbar_has_one_plain_link_per_section():
    left = WEBSITE["navbar"]["left"]

    assert [item["text"] for item in left] == [
        "Home",
        "Quickstart",
        "In depth",
        "Plugins",
        "Reference",
        "Why laplacekit",
    ]
    assert all("menu" not in item for item in left)


def test_navbar_links_open_the_first_page_of_their_sidebar():
    links = {item["text"]: item["href"] for item in WEBSITE["navbar"]["left"]}
    bars = sidebars()

    assert links["Home"] == "index.qmd"
    assert links["Why laplacekit"] == "why.qmd"
    assert links["Quickstart"] == bars["quickstart"][0]
    assert links["In depth"] == bars["in-depth"][0]
    assert links["Plugins"] == bars["plugins"][0]
    assert links["Reference"] == f"{CONFIG['quartodoc']['dir']}/index.qmd"


def test_sidebar_pages_exist():
    for bar, pages in sidebars().items():
        for page in pages:
            assert (DOCS / page).is_file(), f"{bar}: {page}"


def test_every_section_page_sits_in_exactly_one_sidebar():
    listed = [page for pages in sidebars().values() for page in pages]
    on_disk = sorted(
        str(path.relative_to(DOCS))
        for section in SECTIONS
        for pattern in ("*.py", "*.qmd")
        for path in (DOCS / section).glob(pattern)
    )

    assert on_disk
    assert sorted(listed) == on_disk


def test_home_and_why_have_no_sidebar():
    listed = {page for pages in sidebars().values() for page in pages}

    assert {"index.qmd", "why.qmd"}.isdisjoint(listed)


def test_reference_sidebar_comes_from_quartodoc():
    sidebar = CONFIG["quartodoc"]["sidebar"]
    file = sidebar if isinstance(sidebar, str) else sidebar["file"]

    assert file.startswith(CONFIG["quartodoc"]["dir"] + "/")
    assert file in CONFIG["metadata-files"]


@pytest.mark.parametrize("bar", ["quickstart", "in-depth", "plugins"])
def test_section_sidebars_group_pages_under_captions(bar):
    (sidebar,) = [item for item in WEBSITE["sidebar"] if item["id"] == bar]

    for group in sidebar["contents"]:
        assert set(group) == {"section", "contents"}, group
        assert group["contents"]


def test_plugins_sidebar_captions():
    (sidebar,) = [item for item in WEBSITE["sidebar"] if item["id"] == "plugins"]

    captions = [group["section"] for group in sidebar["contents"]]

    assert captions == ["Using plugins", "Writing plugins", "Guides", "Concepts", "Shipping"]


def test_plugins_open_on_the_plugin_list():
    """Most readers come to use a plugin, so the tab opens on the list, not the guides."""

    links = {item["text"]: item["href"] for item in WEBSITE["navbar"]["left"]}
    (sidebar,) = [item for item in WEBSITE["sidebar"] if item["id"] == "plugins"]

    assert links["Plugins"] == "plugins/index.qmd"
    assert sidebar["contents"][0]["contents"] == [
        {"text": "Plugin list", "href": "plugins/index.qmd"}
    ]


def test_input_addons_follow_the_mode():
    """Bootstrap's addon boxes (the plugin list's search icon) default to light grey."""

    theme = (DOCS / "assets" / "theme.scss").read_text()

    for name in ("input-group-addon-bg", "input-group-addon-color", "input-bg", "input-color"):
        assert re.search(rf"^\${name}: \$lk-[a-z-]+;", theme, re.M), name


GUIDES = [
    "A subset",
    "A prior",
    "A tuning objective",
    "A likelihood",
    "A predictive",
    "A curvature",
    "A function-space approximation",
    "A library integration",
]


def test_plugin_guides_go_from_the_smallest_contract():
    (sidebar,) = [item for item in WEBSITE["sidebar"] if item["id"] == "plugins"]
    (guides,) = [group for group in sidebar["contents"] if group["section"] == "Guides"]

    assert [page["text"] for page in guides["contents"]] == GUIDES

    for page in guides["contents"]:
        assert page["href"].endswith(".py"), page  # every guide runs


CONCEPTS = [
    "Requirements and capabilities",
    "Statistics",
    "Immutability and pytrees",
    "Differentiability",
    "Several processes",
]


def test_plugin_concepts_are_short_static_pages():
    (sidebar,) = [item for item in WEBSITE["sidebar"] if item["id"] == "plugins"]
    (concepts,) = [group for group in sidebar["contents"] if group["section"] == "Concepts"]

    assert [page["text"] for page in concepts["contents"]] == CONCEPTS

    for page in concepts["contents"]:
        assert page["href"].endswith(".qmd"), page  # code is shown, not run
        assert len((DOCS / page["href"]).read_text().splitlines()) <= 90, page


def test_in_depth_has_no_extending_pages():
    """The likelihood and curvature tutorials moved to the plugin guides."""

    pages = sidebars()["in-depth"]

    assert "examples/likelihoods.py" not in pages
    assert "examples/plugins.py" not in pages


def test_mode_follows_the_system_setting():
    """Like the author's website: the system light/dark setting decides, on every
    load and when it changes."""

    assert HTML["respect-user-color-scheme"] is True

    script = (DOCS / "assets" / "color-scheme.html").read_text()

    assert "assets/color-scheme.html" in HTML["include-in-header"]

    # The stored toggle is cleared before Quarto's own script reads it.
    assert 'localStorage.removeItem("quarto-color-scheme")' in script
    assert 'matchMedia("(prefers-color-scheme: dark)")' in script
    assert 'addEventListener("change"' in script
    assert "quartoToggleColorScheme" in script


THEME_RULES = (DOCS / "assets" / "theme.scss").read_text().split("/*-- scss:rules --*/")[1]


def rule(selector: str) -> str:
    """The body of the first rule with exactly this selector."""

    match = re.search(rf"^{re.escape(selector)} \{{([^}}]*)\}}", THEME_RULES, re.M)

    assert match, selector

    return match.group(1)


def test_page_title_sits_clear_of_the_first_paragraph():
    assert "margin-bottom: 2rem" in rule("#title-block-header")


def test_callouts_look_like_material_admonitions():
    """A thin border on all sides, rounded corners, a soft shadow, and a title bar
    tinted with the callout's colour, for every kind of callout."""

    box = rule("div.callout.callout")

    assert "border: 0.075rem solid" in box
    assert "border-radius: 0.2rem" in box
    assert "box-shadow:" in box
    assert "border-top: none" not in THEME_RULES

    colours = dict(re.findall(r'^\s+"(\w+)": \$(lk-\w+),$', THEME_RULES, re.M))

    assert colours == {
        "note": "lk-note",
        "tip": "lk-tip",
        "warning": "lk-warning",
        "caution": "lk-caution",
        "important": "lk-important",
    }

    loop = THEME_RULES.split("@each $name, $colour in $lk-callouts")[1]

    assert "border-color: $colour" in loop
    assert "background-color: rgba($colour, 0.1)" in loop


def test_there_is_no_light_dark_switch():
    """The mode follows the system only, so Quarto's toggle is hidden everywhere."""

    assert "display: none" in rule(".quarto-color-scheme-toggle")
    assert "toggle still works" not in (DOCS / "assets" / "color-scheme.html").read_text()


def test_type_reads_like_a_github_readme():
    """GitHub sets 16px system text, code at 85%, and headings at 32/24/20/16px.
    Inter is taller per pixel, so its body is 15px to read the same. Lines are 1.6,
    a little looser than GitHub's 1.5, as on the author's blog."""

    defaults = (DOCS / "assets" / "theme.scss").read_text().split("/*-- scss:rules --*/")[0]
    value = dict(re.findall(r"^\$([\w-]+):\s*([^;]+);", defaults, re.M))

    assert value["font-size-root"] == "16px"
    assert value["font-size-base"] == "0.9375rem"
    assert value["line-height-base"] == "1.6"
    assert value["code-font-size"] == "0.85em"
    assert [value[f"h{i}-font-size"] for i in (1, 2, 3, 4)] == ["2rem", "1.5rem", "1.25rem", "1rem"]


def test_text_has_room():
    """Headings sit well below the text before them and close to their own; paragraphs,
    code, outputs and lists get room between them. The values follow the author's blog
    (Tailwind Typography)."""

    # Quarto's `#quarto-content section:first-of-type h2:first-of-type` would squeeze
    # the first heading after an intro; only a heading right under the title sits flush.
    assert "margin-top: 2.75rem" in rule("#quarto-content main.content h2")
    assert "margin-top: 0" in rule("#quarto-content #title-block-header + section > h2:first-child")
    assert "margin-top: 2rem" in rule("main.content h3")
    assert "margin-top: 1.5rem" in rule("main.content h4")
    assert "margin-bottom: 1.25rem" in rule("main.content p")
    assert "margin: 1.5rem 0" in rule("main.content div.sourceCode")
    assert "margin: 1.5rem 0" in rule("main.content .table")
    assert "line-height: 1.5" in rule("main.content pre.sourceCode")
    assert "padding-left" in rule("main.content .cell-output pre")
    assert "margin-top: 0.4rem" in rule("main.content li + li")


def test_output_sits_close_to_its_code():
    assert "margin-bottom: 0.75rem" in rule("main.content .cell div.sourceCode")


def test_code_column_fits_80_characters_in_a_1200px_window():
    """Narrow side columns leave the body room for 80-character code lines."""

    grid = HTML["grid"]

    assert "body-width" not in grid  # Quarto's default; narrower makes code scroll
    assert int(grid["sidebar-width"].removesuffix("px")) <= 220
    assert int(grid["margin-width"].removesuffix("px")) <= 200


def test_docs_code_is_formatted_at_80_columns():
    ruff = tomllib.loads((DOCS / "ruff.toml").read_text())

    assert ruff["line-length"] == 80


def test_no_docs_code_line_is_longer_than_80_characters():
    """E501 is off for the scripts (prose cells hold LaTeX), so check code here."""

    long = [
        f"{path.relative_to(DOCS)}:{n}"
        for folder in ("quickstart", "examples", "plugins")
        for path in sorted((DOCS / folder).glob("*.py"))
        for n, line in enumerate(path.read_text().splitlines(), 1)
        if len(line) > 80 and not line.startswith("#")
    ]

    assert long == []


def test_page_is_centred_on_wide_screens():
    """A docked sidebar pins the page to the left edge; a floating one centres it."""

    styles = {bar.get("style") for bar in WEBSITE["sidebar"]} - {None}

    assert styles == {"floating"}


def test_in_depth_basics_have_heteroscedastic_after_regression():
    (sidebar,) = [item for item in WEBSITE["sidebar"] if item["id"] == "in-depth"]
    (basics,) = [group for group in sidebar["contents"] if group["section"] == "Basics"]

    assert basics["contents"] == [
        {"text": "Classification", "href": "examples/classification.py"},
        {"text": "Regression", "href": "examples/regression.py"},
        {"text": "Heteroscedastic regression", "href": "examples/heteroscedastic.py"},
        {"text": "Choosing a curvature", "href": "examples/curvature.py"},
    ]


def test_in_depth_models_have_reward_after_language_models():
    (sidebar,) = [item for item in WEBSITE["sidebar"] if item["id"] == "in-depth"]
    (models,) = [group for group in sidebar["contents"] if group["section"] == "Models"]

    assert models["contents"] == [
        {"text": "Language models", "href": "examples/llm.py"},
        {"text": "Reward models", "href": "examples/reward.py"},
        {"text": "Diffusion models", "href": "examples/diffusion.py"},
    ]


def test_pages_are_frozen_during_development():
    assert CONFIG["execute"]["freeze"] == "auto"


def test_frozen_outputs_are_tracked():
    ignored = (ROOT / ".gitignore").read_text().splitlines()

    assert not any(line.strip().rstrip("/").endswith("_freeze") for line in ignored)


def test_docs_full_clears_the_freeze_then_renders():
    makefile = (ROOT / "makefile").read_text()
    target = re.search(r"^docs-full:(.*)\n((?:\t.*\n)*)", makefile, re.MULTILINE)

    assert target, "no docs-full target"

    recipe = target.group(1) + target.group(2)

    clear = recipe.find("rm -rf docs/_freeze")
    render = recipe.find("$(MAKE) docs")

    assert 0 <= clear < render, recipe


def test_docs_workflow_renders_the_freeze_and_deploys_it():
    """CI never executes pages: it renders the committed freeze and uploads the site."""

    workflow = yaml.safe_load((ROOT / ".github" / "workflows" / "docs.yml").read_text())
    (job,) = workflow["jobs"].values()
    runs = [step.get("run") for step in job["steps"]]
    deploy = [s for s in job["steps"] if str(s.get("uses", "")).startswith("cloudflare/wrangler")]

    assert workflow[True]["push"]["branches"] == ["main"]  # YAML reads `on` as True
    assert "make docs" in runs and "make docs-full" not in runs
    assert len(deploy) == 1 and "pages deploy docs/_site" in deploy[0]["with"]["command"]


def test_ci_runs_each_gate_as_its_own_job():
    """The gates of `make check`, one job each, so a failure names its gate."""

    jobs = yaml.safe_load((ROOT / ".github" / "workflows" / "ci.yml").read_text())["jobs"]
    want = {
        "lint": "ruff check",
        "format": "ruff format --check",
        "typecheck": "make typecheck",
        "imports": "make imports",
        "test": "make test",
    }

    for job, command in want.items():
        runs = [step.get("run") or "" for step in jobs[job]["steps"]]

        assert any(run.endswith(command) for run in runs), job


def test_navbar_links_the_repository_next_to_search():
    """A GitHub icon on the right of the navbar, where the search button sits."""

    (link,) = WEBSITE["navbar"]["right"]

    assert link == {
        "icon": "github",
        "href": "https://github.com/laplacekit/laplacekit",
        "aria-label": "GitHub",
    }


def test_figures_shrink_to_the_column():
    """Quarto writes a plot's pixel size on its <img>; on a phone it must shrink. The
    rule has the weight of a bare `img`, so a sized image such as the landing page's
    `.brand-mark` keeps its own height."""

    assert not re.search(r"^main\.content img \{", THEME_RULES, re.M)

    body = rule(":where(main.content) img")

    assert "max-width: 100%" in body
    assert "height: auto" in body


def test_wide_display_math_scrolls_by_itself():
    """A display equation wider than the screen scrolls inside its own box instead of
    widening the page."""

    body = rule('main.content mjx-container[display="true"]')

    assert "overflow-x: auto" in body
    assert "max-width: 100%" in body


def test_breadcrumbs_share_one_baseline():
    """Bootstrap floats the `>` before every item but the first, which drops those
    items below the first one."""

    assert "align-items: baseline" in rule(".quarto-title-breadcrumbs .breadcrumb")
    assert "float: none" in rule(
        ".quarto-title-breadcrumbs .breadcrumb-item + .breadcrumb-item::before"
    )


def test_reference_titles_sit_under_their_breadcrumbs():
    """quartodoc puts the h1 in the first section, not the title block, so neither
    the title block's bottom margin nor the h1's top margin applies there."""

    assert "margin-bottom: 0" in rule("#title-block-header:not(:has(h1))")
    assert "margin-top: 0" in rule("#title-block-header + section > h1:first-child")


def test_github_icon_matches_the_search_icon():
    """Quarto draws the search icon as a 26px SVG; the GitHub glyph would otherwise
    take the menu's 15px text size."""

    assert "font-size: 1.375rem" in rule(".navbar .nav-link .bi-github")


def test_math_is_set_in_computer_modern():
    """MathJax 4 defaults to New Computer Modern; the original TeX fonts are the
    standard Computer Modern. The config must load before MathJax does."""

    assert "assets/mathjax.html" in HTML["include-in-header"]

    config = (DOCS / "assets" / "mathjax.html").read_text()

    assert "window.MathJax" in config
    assert re.search(r"font:\s*\"mathjax-tex\"", config)


def test_linked_names_in_a_first_column_do_not_break():
    """On a phone the reference summary squeezed `laplace` into `lapla` / `ce`."""

    assert "white-space: nowrap" in rule("main.content td:first-child > a:only-child")
    assert "white-space: nowrap" in rule(".doc-section table td:first-child")


def test_phone_breadcrumbs_stay_on_one_line():
    """The bar under the navbar keeps one line: earlier items shrink with an
    ellipsis, the current page keeps its full name."""

    assert "flex-wrap: nowrap" in rule(".quarto-secondary-nav .breadcrumb")

    item = rule(".quarto-secondary-nav .breadcrumb-item")

    assert "text-overflow: ellipsis" in item
    assert "white-space: nowrap" in item
    assert "flex-shrink: 0" in rule(".quarto-secondary-nav .breadcrumb-item:last-child")
