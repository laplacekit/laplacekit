"""The Plugins tab: the plugin list shows the workspace plugins, and the Overview names the same."""

import ast
import re
import tomllib
from pathlib import Path

import pytest
import yaml
from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet

import laplacekit as lk

ROOT = Path(__file__).parents[1]
DOCS = ROOT / "docs"
PLUGINS = DOCS / "plugins"
EXAMPLE = "laplacekit-example"
EXAMPLE_REPO = "https://github.com/laplacekit/laplacekit-plugin-example"
REPO = "https://github.com/laplacekit/laplacekit"
KINDS = {"first-party", "third-party"}
FIELDS = {"title", "description", "categories", "kind", "use", "guide", "source"}


def current(text: str) -> str:
    """`text` with today's names: the plugin list already uses the `laplacekit` ones."""

    return text.replace("laplacekit", "laplacekit")


def named(page: Path) -> set[str]:
    """The `laplacekit-*` packages a page names in code spans."""

    return set(re.findall(r"`(laplacekit-[a-z]+)`", page.read_text()))


def workspace_plugins() -> dict[str, Path]:
    """Each workspace plugin's distribution name, with its folder."""

    found = {}

    for path in (ROOT / "plugins").glob("*/pyproject.toml"):
        found[tomllib.loads(path.read_text())["project"]["name"]] = path.parent

    return found


def split_page(page: Path) -> tuple[dict, str]:
    """A page's front matter and its body."""

    _, front, body = page.read_text().split("---", 2)

    return yaml.safe_load(front), body


def kinds_of_contribution() -> set[str]:
    """The kinds in the Overview's "What a plugin can add" table, e.g. "Curvature"."""

    rows = re.findall(r"^\| \[An? ([a-z -]+)\]", (PLUGINS / "overview.qmd").read_text(), re.M)

    return {row[0].upper() + row[1:] for row in rows}


ENTRIES = yaml.safe_load((PLUGINS / "list.yml").read_text())
IDS = [entry["title"] for entry in ENTRIES]


def test_list_shows_every_workspace_plugin():
    assert sorted(map(current, IDS)) == sorted(workspace_plugins())


def test_example_comes_from_its_own_repository():
    """The docs install the example, a template for new plugins, from its repository."""

    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())
    groups = pyproject["dependency-groups"]

    assert EXAMPLE not in workspace_plugins()
    assert EXAMPLE in groups["docs"] and EXAMPLE not in groups["dev"]
    assert pyproject["tool"]["uv"]["sources"][EXAMPLE] == {"git": EXAMPLE_REPO}
    assert EXAMPLE_REPO in (PLUGINS / "first.py").read_text()


def test_list_uses_the_new_names():
    for entry in ENTRIES:
        assert entry["title"].startswith("laplacekit-"), entry["title"]
        assert "scala" + "place" not in entry["use"], entry["title"]
        assert entry["use"].startswith("import laplacekit as lk\n"), entry["title"]


def test_overview_names_the_listed_plugins():
    assert named(PLUGINS / "overview.qmd") == set(map(current, IDS))


def test_overview_table_lists_every_kind_of_contribution():
    assert kinds_of_contribution() == {
        "Subset",
        "Prior",
        "Tuning objective",
        "Likelihood",
        "Predictive",
        "Curvature",
        "Function-space approximation",
        "Library integration",
    }


def test_tab_opens_on_the_list_rendered_from_the_data():
    front, _ = split_page(PLUGINS / "index.qmd")
    (listing,) = front["listing"]

    assert front["title"] == "Plugin list"
    assert listing["contents"] == "list.yml"
    assert listing["template"] == "list.ejs"
    assert (PLUGINS / listing["template"]).is_file()
    assert listing["categories"] is True


def test_list_page_is_only_the_list():
    """The sidebar already points to the guides, so the page has no prose."""

    front, body = split_page(PLUGINS / "index.qmd")
    (listing,) = front["listing"]

    assert body.split() == [":::", f"{{#{listing['id']}}}", ":::"]


def test_pages_that_print_the_plugins_name_the_example():
    printing = [page for page in DOCS.glob("*/*.py") if "print(lk.plugins())" in page.read_text()]

    assert PLUGINS / "first.py" in printing

    for page in printing:
        assert f"`{EXAMPLE}`" in page.read_text(), page.relative_to(DOCS)


def test_workspace_plugins_depend_on_the_core_they_support():
    """The `laplacekit` dependency matches the record's `supports`, as Publishing says."""

    supports = {p.name: p.supports for p in lk.plugins()}

    for path in (ROOT / "plugins").glob("*/pyproject.toml"):
        project = tomllib.loads(path.read_text())["project"]
        core = [r for r in map(Requirement, project["dependencies"]) if r.name == "laplacekit"]

        assert [r.specifier for r in core] == [SpecifierSet(supports[project["name"]])], path


@pytest.mark.parametrize("entry", ENTRIES, ids=IDS)
def test_entry_has_the_fields(entry):
    assert FIELDS <= set(entry), FIELDS - set(entry)
    assert set(entry) <= FIELDS | {"extra", "paper"}, set(entry) - FIELDS
    assert entry["kind"] in KINDS
    assert entry["description"].endswith(".")


@pytest.mark.parametrize("entry", ENTRIES, ids=IDS)
def test_category_is_a_kind_of_contribution(entry):
    assert len(entry["categories"]) == 1
    assert set(entry["categories"]) <= kinds_of_contribution()


@pytest.mark.parametrize("entry", ENTRIES, ids=IDS)
def test_first_party_entries_and_only_those_name_their_extra(entry):
    extras = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"][
        "optional-dependencies"
    ]
    packages = {name: [Requirement(r).name for r in reqs] for name, reqs in extras.items()}

    if entry["kind"] == "first-party":
        assert packages[entry["extra"]] == [current(entry["title"])]
    else:
        assert "extra" not in entry
        assert current(entry["title"]) not in [n for names in packages.values() for n in names]


@pytest.mark.parametrize("entry", ENTRIES, ids=IDS)
def test_entries_install_from_pypi_and_workspace_entries_link_their_folder(entry):
    assert "git" not in entry  # the install tab shows the PyPI name only

    if current(entry["title"]) not in workspace_plugins():
        return

    folder = workspace_plugins()[current(entry["title"])].relative_to(ROOT).as_posix()

    assert entry["source"] == f"{REPO}/tree/main/{folder}"


@pytest.mark.parametrize("entry", ENTRIES, ids=IDS)
def test_guide_is_a_docs_page(entry):
    assert (PLUGINS / entry["guide"]).resolve().is_file(), entry["guide"]


@pytest.mark.parametrize("entry", ENTRIES, ids=IDS)
def test_use_snippet_names_what_the_plugin_provides(entry):
    """The snippet's imports run, every `lk.a.b` it names exists, and what it calls is bound."""

    tree = ast.parse(current(entry["use"]))
    imports = [node for node in tree.body if isinstance(node, ast.Import | ast.ImportFrom)]
    namespace: dict = {}

    assert imports, "the snippet imports nothing"

    exec(compile(ast.Module(imports, []), entry["title"], "exec"), namespace)

    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and ast.unparse(node).startswith("lk."):
            obj = lk

            for part in ast.unparse(node).split(".")[1:]:
                obj = getattr(obj, part)

    called = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }

    assert called <= set(namespace), called - set(namespace)


def test_all_extra_installs_every_first_party_plugin():
    extras = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"][
        "optional-dependencies"
    ]
    plugins = {Requirement(r).name for name, reqs in extras.items() if name != "all" for r in reqs}
    first_party = {current(e["title"]) for e in ENTRIES if e["kind"] == "first-party"}

    assert {Requirement(r).name for r in extras["all"]} == plugins == first_party
