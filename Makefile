.PHONY: install test typecheck lint clean build

install:
	pip install -e ".[dev]"
	pip install scipy-stubs

test:
	pytest tests/ -v

test-fast:
	pytest tests/ -m "not slow" -v

typecheck:
	mypy src/vbacktest/ --strict

lint: typecheck

clean:
	rm -rf dist/ build/ *.egg-info src/*.egg-info __pycache__ .mypy_cache .pytest_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete

build:
	pip install build
	python -m build
