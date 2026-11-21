# Everything runs through uv, so the tools come from the project's venv.
# Quarto too: the `quarto-cli` package in the docs group ships the binary.
UV ?= uv run
QUARTO ?= $(UV) --group docs quarto
export QUARTO_PYTHON ?= $(CURDIR)/.venv/bin/python

TESTS ?= tests plugins/transformers/tests plugins/diffusers/tests plugins/subnet/tests plugins/heteroscedastic/tests plugins/reward/tests plugins/gp/tests plugins/marglik/tests
ARGS ?=
# Port for `make docs-serve`; override with `make docs-serve PORT=...`.
PORT ?= 4200

.PHONY: sync test lint fix typecheck imports check docs-reference docs docs-full docs-serve clean

sync:
	uv sync --group docs

# Examples:
#   make test
#   make test TESTS=tests/test_front.py ARGS='-k tune'
test:
	$(UV) pytest $(TESTS) -q $(ARGS)

lint:
	$(UV) ruff check
	$(UV) ruff format --check

# Fix lint warnings, then format (order matters).
fix:
	-$(UV) ruff check --fix
	$(UV) ruff format

typecheck:
	$(UV) ty check

imports:
	$(UV) lint-imports

# The same gate CI runs.
check: lint typecheck imports test

# torch warns that torch.jit.script is unsupported on Python 3.14+ when its own
# code calls it. Not ours, so the docs kernels ignore that one message (pytest
# does the same in pyproject.toml).
docs-reference docs docs-serve: export PYTHONWARNINGS = ignore:`torch.jit.script` is not supported:FutureWarning:torch.jit._script

docs-reference:
	$(UV) --group docs python docs/quartodoc_build.py build --config docs/_quarto.yml

# Shiki, for the post-render highlighting step, pinned by docs/package-lock.json.
docs/node_modules/.package-lock.json: docs/package-lock.json
	npm ci --prefix docs --no-audit --no-fund

# Pages are frozen (`freeze: auto` in docs/_quarto.yml): a page re-executes only
# when its source changed; the others reuse their outputs in docs/_freeze/.
docs: docs-reference docs/node_modules/.package-lock.json
	$(QUARTO) render docs

# Drops the frozen outputs, so every page re-executes (milestone ends).
docs-full:
	rm -rf docs/_freeze
	$(MAKE) docs

# Live preview with reload, at http://localhost:$(PORT).
docs-serve: docs-reference docs/node_modules/.package-lock.json
	$(QUARTO) preview docs --port $(PORT)

clean:
	rm -rf docs/_site docs/reference docs/objects.json docs/.quarto docs/node_modules .pytest_cache .ruff_cache
