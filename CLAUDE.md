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

## GitHub Interaction Guidelines

- **NEVER impersonate the user on GitHub** - Do not post comments, create issues, or interact with the xarray GitHub repository unless explicitly instructed
- Never create GitHub issues or PRs unless explicitly requested by the user
- Never post "update" messages, progress reports, or explanatory comments on GitHub issues/PRs unless specifically asked
- Always require explicit user direction before creating pull requests or pushing to the xarray GitHub repository
