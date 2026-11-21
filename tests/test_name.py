"""The package is `laplacekit`: no tracked file keeps the old name."""

import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
OLD = "scala" + "place"  # split so this file doesn't match itself


def tracked_files() -> list[Path]:
    try:
        out = subprocess.run(
            ["git", "ls-files", "-z"], cwd=ROOT, capture_output=True, text=True, check=True
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("not a git checkout")

    return [ROOT / name for name in out.split("\0") if name]


def test_no_tracked_file_uses_the_old_name():
    stale = []

    for path in tracked_files():
        if not path.exists():  # deleted in the working tree, not yet committed
            continue

        if OLD in str(path.relative_to(ROOT)).lower():
            stale.append(str(path.relative_to(ROOT)))
            continue

        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, FileNotFoundError):
            continue

        if OLD in text.lower():
            stale.append(str(path.relative_to(ROOT)))

    assert not stale, stale
