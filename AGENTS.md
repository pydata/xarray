# xarray development setup

## Setup

Development environments are managed with [pixi](https://pixi.sh), the same ones
CI uses (see `[environments]` in `pixi.toml`). They are installed on first use.

## Run tests

```bash
pixi run -e test-py313 -- python -m pytest xarray -n auto  # All tests in parallel
pixi run -e test-py313 -- python -m pytest xarray/tests/test_dataarray.py  # Specific file
```

Environments like `test-py314-no-dask` or `test-py312-bare-minimum` run the tests
without some optional dependencies.

See `xarray/tests/AGENTS.md` for guidelines on writing tests.

## Linting & type checking

```bash
pixi exec pre-commit run --all-files  # Includes ruff and other checks
pixi run -e test-py314-with-typing -- python -m mypy  # Type checking with mypy
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

### Type hints

- Always add type hints to every function and method you create or edit,
  including tests (`-> None`), fixtures and helpers

## CI

- The `test-py314 (flaky)` job runs tests that depend on external services
  (e.g. test.opendap.org) and fails regularly for unrelated reasons. It does not
  block merging, so don't skip or change these tests, ask for reruns, or explain
  these failures in PRs.

## GitHub Interaction Guidelines

- **NEVER impersonate the user on GitHub**, always sign off with something like
  "[This is <tool name> on behalf of Jane Doe]"
- Never create issues nor pull requests on the xarray GitHub repository unless
  explicitly instructed
- Never post "update" messages, progress reports, or explanatory comments on
  GitHub issues/PRs unless specifically instructed
- Don't comment on an issue to ask to be assigned or to announce that you are
  working on it or will open a PR. xarray does not assign issues: just open the
  PR and reference the issue in its description.
- When creating commits, always include a co-authorship trailer for the tool,
  e.g. `Co-authored-by: Claude <noreply@anthropic.com>`
