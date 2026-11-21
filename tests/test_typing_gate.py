"""ty rejects exactly the marked mistakes and nothing else under tests/typing."""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TYPING = ROOT / "tests" / "typing"


def _ty() -> str:
    ty = shutil.which("ty")

    if ty is None:
        pytest.fail("the typing gate needs `ty` on PATH; run the tests through `uv run`")

    return ty


def _errors() -> dict[str, set[int]]:
    run = subprocess.run(
        [
            _ty(),
            "check",
            "--config",
            'src.include=["tests/typing"]',
            "--output-format",
            "gitlab",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    report = json.loads(run.stdout)
    found: dict[str, set[int]] = {}

    for d in report:
        if d["severity"] in ("major", "critical", "blocker"):
            found.setdefault(Path(d["location"]["path"]).name, set()).add(
                d["location"]["positions"]["begin"]["line"]
            )

    return found


def test_mistakes_fail_exactly():
    marked = {
        i
        for i, line in enumerate((TYPING / "test_mistakes.py").read_text().splitlines(), 1)
        if "# EXPECTED ERROR" in line
    }

    assert marked
    found = _errors()

    assert found.pop("test_mistakes.py", set()) == marked
    assert found == {}, f"unexpected errors outside test_mistakes.py: {found}"


def test_missing_ty_fails_with_a_clear_message(monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda name: None)

    with pytest.raises(pytest.fail.Exception, match="needs `ty` on PATH"):
        _ty()
