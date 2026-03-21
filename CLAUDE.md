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

## Code Style Guidelines

### Import Organization

- **Always place imports at the top of the file** in the standard import section
- Never add imports inside functions or nested scopes unless there's a specific
  reason (e.g., circular import avoidance, optional dependencies in TYPE_CHECKING)
- Group imports following PEP 8 conventions:
  1. Standard library imports
  2. Related third-party imports
  3. Local application/library specific imports

## GitHub Interaction Guidelines

- **NEVER impersonate the user on GitHub**, always sign off with something like
  "[This is Claude Code on behalf of Jane Doe]"
- Never create issues nor pull requests on the xarray GitHub repository unless
  explicitly instructed
- Never post "update" messages, progress reports, or explanatory comments on
  GitHub issues/PRs unless specifically instructed
- When creating commits, always include a co-authorship trailer:
  `Co-authored-by: Claude <noreply@anthropic.com>`
