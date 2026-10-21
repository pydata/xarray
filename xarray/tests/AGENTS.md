# Testing Guidelines for xarray

## Handling Optional Dependencies

xarray has many optional dependencies that may not be available in all testing environments. Always use the standard decorators and patterns when writing tests that require specific dependencies.

### Standard Decorators

**ALWAYS use decorators** like `@requires_dask`, `@requires_cftime`, etc. instead of conditional `if` statements.

All available decorators are defined in `xarray/tests/__init__.py` (look for `requires_*` decorators).

Use the dask helpers from `xarray.tests` instead of importing `dask.array`
directly. In dask-array expression test mode those helpers point at the
registered chunk manager, so tests can run against either implementation.

### Skip at collection time, not at runtime

❌ **WRONG - Do not check for a dependency inside the test:**

```python
def test_mean() -> None:
    if has_dask:  # WRONG!
        ds = ds.chunk({})
    result = ds.mean()
```

❌ **ALSO WRONG - Avoid pytest.mark.skipif in parametrize:**

```python
@pytest.mark.parametrize(
    "chunk",
    [
        pytest.param(
            True, marks=pytest.mark.skipif(not has_dask, reason="requires dask")
        ),
        False,
    ],
)
def test_something(chunk: bool) -> None: ...
```

✅ **CORRECT - Parametrize, e.g. with the `use_dask` fixture:** it runs the test
once with `False` (id `numpy`) and once with `True` (id `dask`), and the dask case
is skipped at collection time if dask is not installed. Sharing the test body
avoids duplicate code and checks that numpy and dask give the same result:

```python
def test_mean(use_dask: bool) -> None:
    ds = create_test_data()
    if use_dask:
        ds = ds.chunk()
    result = ds.mean()
```

For other optional dependencies, mark the parameters that need them:

```python
@pytest.mark.parametrize(
    "calendar", ["standard", pytest.param("noleap", marks=requires_cftime)]
)
def test_something(calendar: str) -> None: ...
```

Only write a separate test, decorated with e.g. `@requires_dask`, if it checks
something specific to that dependency, e.g. that the result stays lazy.

### Fixtures

Shared fixtures are defined in `xarray/tests/conftest.py`, e.g. `use_dask`,
`compute_backend`, `use_cftime` and `time_unit`, and test data like `ds`, `da` and
`simple_datatree`. Check there before writing your own fixture or parametrization.

### Multiple dependencies

When a test requires multiple optional dependencies:

```python
@requires_dask
@requires_scipy
def test_interpolation_with_dask() -> None: ...
```

### Importing optional dependencies in tests

Skip tests at collection time with the `requires_*` decorators (or
`pytest.param(..., marks=requires_*)` in a parametrization), not at runtime with
`pytest.importorskip`. Import the optional dependency inside the decorated test:

```python
@requires_cftime
def test_cftime_functionality() -> None:
    import cftime

    ...
```

### Key Points

- CI environments intentionally exclude certain dependencies (e.g.,
  `test-py314-no-dask`, `test-py312-bare-minimum`)
- A test failing in `test-py314-no-dask` because it uses dask is a test bug, not a
  CI issue
- Look at similar existing tests for patterns to follow
