PYTHON ?= python3

.PHONY: install lint format-check typecheck test check

install:
	$(PYTHON) -m pip install -e '.[geometry,dev]'

lint:
	$(PYTHON) -m ruff check src tests

format-check:
	$(PYTHON) -m ruff format --check src tests

typecheck:
	$(PYTHON) -m mypy src/vector_map_former

test:
	$(PYTHON) -m pytest

check: lint format-check typecheck test
