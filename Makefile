# Every target is a thin wrapper over `python -m ...`, so the repo is equally
# usable without make — Windows without a make binary, for instance.

PY ?= python

.PHONY: help test lint demo verify-sources clean

help:
	@echo "test            unit tests — no API key, no network"
	@echo "lint            ruff"
	@echo "demo            audit the worked example from docs/DESIGN.md §1 (needs a key)"
	@echo "verify-sources  hash-check the archived Gazette notifications"
	@echo "clean           remove __pycache__"

# Runs on a fresh clone with no key and no network. That is the point: a
# reliability harness whose own tests need a working provider is measuring the
# provider.
test:
	$(PY) -m pytest tests -q

lint:
	$(PY) -m ruff check agent chaos suite tests

demo:
	$(PY) -m agent --demo

verify-sources:
	$(PY) -c "from agent.gazette import verify_sources; import json; print(json.dumps(verify_sources(), indent=2))"

clean:
	$(PY) -c "import pathlib,shutil; [shutil.rmtree(p, ignore_errors=True) for p in pathlib.Path('.').rglob('__pycache__')]"
