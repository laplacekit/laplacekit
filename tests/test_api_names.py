"""Renamed public names: the new ones exist, the old ones are gone."""

import dataclasses
import importlib.util
import inspect
import re
import subprocess
from pathlib import Path

import pytest

import laplacekit as lk
from laplacekit import core

ROOT = Path(__file__).resolve().parents[1]

# Old names are split so this file doesn't match its own patterns.
KEY = "Ke" + "y"
PRNG_KEY = "PRNG" + "Key"  # renamed from KEY, then dropped for torch generators
BATCH_KEYS = "batch_" + "keys"
BLOCKED = "Block" + "ed"
COV_MANY = "Cov" + "Many"
GRID = "gr" + "id"
SELECTED_SUFFIX = "SELECTED" + "_SUFFIX"
MAX_FACTOR_ENTRIES = "MAX_FACTOR" + "_ENTRIES"
ESTIMATORS = "estim" + "ators"
ESTIMATOR = "estim" + "ator"
STAGED_ESTIMATOR = "Staged" + "Estim" + "ator"

RENAMED = {
    # old name: new name, in `lk.core`
    BLOCKED: "HasBlocks",
    COV_MANY: "HasCovMany",
    ESTIMATOR: "fit_curvature",
    ESTIMATOR.capitalize(): "CurvatureFit",
    STAGED_ESTIMATOR: "StagedCurvatureFit",
    "Laplace" + "Gaussian": "LaplacePosterior",
    "Weight" + "Gaussian": "WeightPosterior",
    "Function" + "Belief": "FunctionPosterior",
    "Weight" + "Belief": "Prior",
    "Pushable" + "WeightBelief": "PriorWithCovariance",
    "Weight" + "Approx": "HessianApprox",
    "Bel" + "ief": "Posterior",
    "Vi" + "a": "Strategy",
    "out" + "puts": "predict_outputs",
}
FIT_NAMES = [ESTIMATOR, ESTIMATOR.capitalize(), STAGED_ESTIMATOR]
# Renamed and still re-exported by `lk`.
RENAMED_IN_LK = {"Laplace" + "Gaussian": "LaplacePosterior", "out" + "puts": "predict_outputs"}
# Names that must not be attributes of `lk`.
GONE_FROM_LK = [
    KEY,
    PRNG_KEY,
    BLOCKED,
    COV_MANY,
    GRID,
    GRID + "_search",
    *FIT_NAMES,
    "DEF" + "AULT",
    "f" + "s",
    "di" + "st",
    "Di" + "ag",
    "Low" + "Rank",
    "EmpiricalFisher" + "Diag",
    "SampledFisher" + "Diag",
    "Pass",
    "Done",
    "Running",
    "Finished",
    "Stat",
    "FitStat",
    "static",
    "pytree",
    "probe",
    "Che" + "ck",
    "Rep" + "ort",
]
# Neither exported nor attributes of `core`.
PRIVATE = [
    SELECTED_SUFFIX,
    MAX_FACTOR_ENTRIES,
    KEY,
    PRNG_KEY,
    BATCH_KEYS,
    "DEF" + "AULT",
    "Def" + "ault",
    "where" + "_leaves",
    "Wh" + "ere",
    GRID + "_search",
    "tu" + "ne",
]

# Old usages no tracked text file may contain.
OLD_USAGES = [
    rf"\blk\.{KEY}\b",
    rf"\bcore\.{KEY}\b",
    rf"\b{KEY}\.from_int\b",
    rf":\s*{KEY}\b",
    rf"->\s*{KEY}\b",
    rf"\[{KEY}\b",
    rf"\b{KEY}\]",
    rf"\b{KEY}\(",
    rf"\b{KEY}\)",
    rf",\s*{KEY}\s*,",
    rf"`{KEY}`",
    rf"^\s*{KEY},\s*$",
    rf"import .*\b{KEY}\b",
    rf"^\s*- {KEY}\s*$",
    rf"\bclass {KEY}\b",
    rf"\b{PRNG_KEY}\b",
    rf"\b{BATCH_KEYS}\b",
    # Prose that still calls randomness a key.
    r"(?i)\b(fit|batch|pass|update|prepare|random|split|seed|its own|explicit random)"
    r"'?s? ke(y|ys)\b",
    r"(?i)\bke(y|ys) (of its own|for (prepare|update|the draws)|seeds?|splits?|come from)\b",
    r"(?i)\b(drawn|fresh) (again |anew )?for each ke(y|ys)\b",
    r"(?i)\b(probe|passes|statistics),? (and|,) ke(y|ys)\b",
    r"(?i)\band keys as `lk\.laplace`",
    rf"\b{BLOCKED}\b",
    rf"\b{COV_MANY}\b",
    # `lk.gp.{GRID}` (the gp context points) keeps its name.
    rf"\blk\.{GRID}\b",
    rf"\bcore\.{GRID}\b",
    rf"^\s*- {GRID}$",
    rf"from laplacekit(\.core)? import [^(\n]*\b{GRID}\b",
    rf"from laplacekit(\.core)? import \([^)]*\b{GRID}\b[^)]*\)",
    rf"\b{SELECTED_SUFFIX}\b",
    rf"\b{MAX_FACTOR_ENTRIES}\b",
    rf"\blk\.{ESTIMATORS}\b",
    rf"\blaplacekit\.{ESTIMATORS}\b",
    rf"\bimport {ESTIMATORS}\b",
    rf"from laplacekit import [^(\n]*\b{ESTIMATORS}\b",
    rf"from laplacekit import \([^)]*\b{ESTIMATORS}\b[^)]*\)",
    rf"\b{ESTIMATOR}\(",
    rf"\blk\.{ESTIMATOR}\b",
    rf"\bcore\.{ESTIMATOR}\b",
    rf"\b{ESTIMATOR.capitalize()}\b",
    rf"\b{STAGED_ESTIMATOR}\b",
    # The UX overhaul: argument and module names that are gone.
    r"\blk\.grid" + r"_search\b",
    r"\bwhere" + r"=lambda\b",
    r"\blk\.out" + r"puts\(",
    r"\blk\.f" + r"s\.",
    r"\blk\.di" + r"st\b",
    r"\blk\.DEF" + r"AULT\b",
    r"\bLaplace" + r"Gaussian\b",
    r"\bWeight" + r"Belief\b",
    r"\bWeight" + r"Approx\b",
    r"(?i)\bonl" + r"ine laplace\b",
    r"(?<![\w])appr" + r"ox=",
    r"(?<![\w])tu" + r"ne=",
    r"\bparams=lk\.sub" + r"sets\.",
    r"\.pr" + r"ec\b",
    r"\bsigma=No" + r"ne\b",
]

SUFFIXES = {".py", ".qmd", ".md", ".yml"}
SKIP = {"tests/test_api_names.py", "tests/test_name.py", "tests/test_randomness.py"}


def tracked_files() -> list[Path]:
    try:
        out = subprocess.run(
            ["git", "ls-files", "-z"], cwd=ROOT, capture_output=True, text=True, check=True
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("not a git checkout")

    return [ROOT / name for name in out.split("\0") if name]


@pytest.mark.parametrize(("old", "new"), list(RENAMED.items()))
def test_core_exports_the_new_name_only(old, new):
    assert new in core.__all__ and hasattr(core, new)
    assert old not in core.__all__ and not hasattr(core, old)


@pytest.mark.parametrize(("old", "new"), list(RENAMED_IN_LK.items()))
def test_lk_exports_the_new_name_only(old, new):
    assert new in lk.__all__ and hasattr(lk, new)
    assert old not in lk.__all__


@pytest.mark.parametrize("old", GONE_FROM_LK)
def test_lk_lacks_the_old_name(old):
    assert not hasattr(lk, old)


@pytest.mark.parametrize("name", PRIVATE)
def test_core_keeps_the_name_private(name):
    assert name not in core.__all__ and not hasattr(core, name)


def test_curvature_methods_live_in_lk_curvature():
    assert lk.curvature.__name__ == "laplacekit.curvature"
    assert lk.curvature.DiagGGN is lk.DiagGGN
    assert not hasattr(lk, ESTIMATORS)
    assert importlib.util.find_spec(f"laplacekit.{ESTIMATORS}") is None


def test_no_tracked_file_uses_an_old_name():
    patterns = [re.compile(p, re.MULTILINE) for p in OLD_USAGES]
    stale = []

    for path in tracked_files():
        rel = path.relative_to(ROOT).as_posix()

        if rel in SKIP or path.suffix not in SUFFIXES or not path.exists():
            continue

        text = path.read_text(encoding="utf-8")

        for pattern in patterns:
            for match in pattern.finditer(text):
                line = text.count("\n", 0, match.start()) + 1
                stale.append(f"{rel}:{line}: {pattern.pattern}")

    assert not stale, "\n".join(stale)


# Parameter and field names that the UX overhaul removed from the public API.
REMOVED_PARAMS = {
    "app" + "rox": "hessian_approx",
    "pr" + "ec": "prior_precision",
    "whe" + "re": "a hyper tuner",
    "vi" + "a": "strategy",
    "ke" + "y": "generator",
    "sca" + "le": "n_data",
    "si" + "ze": "n_data",
    "sam" + "ples": "n_samples",
    "chu" + "nk": "chunk_size",
    "tu" + "ne": "hyper_tuner",
    "hyper" + "steps": "n_hypersteps",
    "hyper" + "_lr": "lr_hyper",
    "prior" + "_init": "the prior's own precision",
    "train" + "_noise": "a learned sigma",
    "o" + "f": "subset",
}
# `params` is the parameter mapping (the torch sense) everywhere but the front doors.
FRONT_DOORS = ["laplace", "function_space.laplace", "gp.fsp_laplace", "diffusers.laplace"]
FRONT_DOORS += ["transformers.laplace_lora", "marglik.train"]
# `every` stays where it means a step interval of its own (SWAG snapshots).
ALLOWED = {("swag_variance", "every")}


def _public_modules():
    mods = [lk, core, lk.curvature, lk.likelihoods, lk.priors, lk.predictive, lk.subsets]
    mods += [lk.objectives, lk.data, lk.tuning, lk.function_space, lk.distributed, lk.testing]
    mods += [getattr(lk, a) for a in ("transformers", "diffusers", "subnet")]
    mods += [getattr(lk, a) for a in ("heteroscedastic", "reward", "gp", "marglik")]

    return mods


def _public_callables():
    seen: set[int] = set()

    for mod in _public_modules():
        for name in mod.__all__:
            obj = getattr(mod, name)
            found = [(name, obj)]

            if isinstance(obj, type):
                members = vars(obj).items()
                found += [(f"{name}.{k}", v) for k, v in members if not k.startswith("_")]

            for label, item in found:
                if id(item) in seen or not callable(item):
                    continue

                seen.add(id(item))

                try:
                    yield label, inspect.signature(item), item
                except (TypeError, ValueError):
                    continue


def test_public_signatures_lack_the_removed_names():
    stale = []

    for label, sig, item in _public_callables():
        for p in sig.parameters.values():
            if p.name in REMOVED_PARAMS and (label, p.name) not in ALLOWED:
                stale.append(f"{label}({p.name}=...): use {REMOVED_PARAMS[p.name]}")

            if p.default is not inspect.Parameter.empty and type(p.default).__name__ == "_Default":
                stale.append(f"{label}({p.name}=DEFAULT)")

        if dataclasses.is_dataclass(item):
            for f in dataclasses.fields(item):
                if f.name in REMOVED_PARAMS:
                    stale.append(f"{label}.{f.name}: use {REMOVED_PARAMS[f.name]}")

    assert not stale, "\n".join(stale)


@pytest.mark.parametrize("path", FRONT_DOORS)
def test_front_doors_take_subset_not_params(path):
    obj: object = lk

    for part in path.split("."):
        obj = getattr(obj, part)

    params = inspect.signature(obj).parameters  # ty: ignore[invalid-argument-type]

    assert "params" not in params
    assert "subset" in params or path in ("marglik.train", "transformers.laplace_lora")
