# xarray development setup

## Setup

```bash
uv sync
```

## Run tests

```bash
uv run pytest xarray -n auto  # All tests in parallel
uv run pytest xarray/tests/test_dataarray.py  # Specific file
```

## Linting & type checking

```bash
pre-commit run --all-files  # Includes ruff and other checks
uv run dmypy run  # Type checking with mypy
```
