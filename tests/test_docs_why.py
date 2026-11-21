"""The Why page's table "Coming from laplace-torch" names only things that exist."""

import re
from pathlib import Path

import laplacekit as lk

WHY = Path(__file__).parents[1] / "docs" / "why.qmd"
HEADING = "## Coming from `laplace-torch`"

# A dotted name at the start of a code span: `lk.priors.KronSum.of(post)` gives
# `lk.priors.KronSum.of`. `fs` is `laplacekit.function_space`, as on the docs pages.
NAME = re.compile(r"`((?:lk|fs)(?:\.[A-Za-z_]\w*)+)")


def section() -> str:
    """The text of the section, up to the next second-level heading."""

    text = WHY.read_text()
    assert HEADING in text, f"{WHY.name} has no section {HEADING!r}"
    body = text.split(HEADING, 1)[1]

    return re.split(r"^## ", body, maxsplit=1, flags=re.MULTILINE)[0]


def table_rows() -> list[str]:
    return [line for line in section().splitlines() if line.startswith("|")]


def resolve(dotted: str) -> object:
    head, *rest = dotted.split(".")
    obj: object = lk if head == "lk" else lk.function_space

    for part in rest:
        obj = getattr(obj, part)

    return obj


def test_why_table_names_resolve() -> None:
    names = sorted({name for row in table_rows() for name in NAME.findall(row)})
    assert len(names) > 40, f"the table names only {len(names)} lk.* names"

    missing = []

    for name in names:
        try:
            resolve(name)
        except AttributeError:
            missing.append(name)

    assert not missing, f"names in the table that don't resolve: {missing}"


def test_why_table_has_its_groups() -> None:
    groups = (
        "Fitting",
        "Curvature",
        "Subsets",
        "Priors and tuning",
        "Likelihoods",
        "Predictives",
        "Function space",
        "Training",
        "Utilities",
    )
    text = "\n".join(table_rows())

    for group in groups:
        assert f"**{group}**" in text, f"the table has no group {group!r}"


def test_why_table_maps_names_without_explanations() -> None:
    """Two columns, `laplace-torch` and **laplacekit**, with no difference column and
    no callout explaining the differences."""

    for row in table_rows():
        assert row.count("|") == 3, row

    assert "Known differences" not in section()
    assert "known differences" not in (WHY.parents[1] / "README.md").read_text()
