# Testing Guidelines for xarray

## Handling Optional Dependencies

xarray has many optional dependencies that may not be available in all testing environments. Always use the standard decorators and patterns when writing tests that require specific dependencies.

### Standard Decorators

**ALWAYS use these decorators** instead of conditional `if` statements:

```python
# For dask-specific tests
@requires_dask
def test_something_with_dask(): ...


# For tests requiring cftime
@requires_cftime
def test_datetime_with_cftime(): ...


# For tests requiring scipy
@requires_scipy
def test_interpolation(): ...


# For tests requiring matplotlib
@requires_matplotlib
def test_plotting(): ...


# For tests requiring numba
@requires_numba
def test_numba_acceleration(): ...


# For tests requiring pint
@requires_pint
def test_units(): ...


# For tests requiring sparse
@requires_sparse
def test_sparse_arrays(): ...


# For tests requiring bottleneck
@requires_bottleneck
def test_bottleneck_functions(): ...


# For tests requiring flox
@requires_flox
def test_flox_groupby(): ...
```

### DO NOT use conditional imports

❌ **WRONG - Do not do this:**

```python
def test_mean_with_cftime():
    if has_dask:  # WRONG!
        ds = ds.chunk({})
        result = ds.mean()
```

✅ **CORRECT - Do this instead:**

```python
def test_mean_with_cftime():
    # Test without dask
    result = ds.mean()


@requires_dask
def test_mean_with_cftime_dask():
    # Separate test for dask functionality
    ds = ds.chunk({})
    result = ds.mean()
```

### Multiple dependencies

When a test requires multiple optional dependencies:

```python
@requires_dask
@requires_scipy
def test_interpolation_with_dask(): ...
```

### Importing optional dependencies in tests

For imports within test functions, use `pytest.importorskip`:

```python
def test_cftime_functionality():
    cftime = pytest.importorskip("cftime")
    # Now use cftime
```

### Testing in CI environments

xarray CI runs tests in various environments:

- `all-but-dask`: Has all dependencies except dask
- `bare-minimum`: Minimal dependencies only
- `all-but-numba`: Has all dependencies except numba

Your tests must handle these environments correctly by using the decorators above.

### Common patterns

1. **Split tests by dependency** - Don't mix optional dependency code with base functionality:

   ```python
   def test_base_functionality():
       # Core test without optional deps
       result = ds.mean()
       assert result is not None


   @requires_dask
   def test_dask_functionality():
       # Dask-specific test
       ds_chunked = ds.chunk({})
       result = ds_chunked.mean()
       assert result is not None
   ```

2. **Use fixtures for dependency-specific setup**:

   ```python
   @pytest.fixture
   def dask_array():
       pytest.importorskip("dask.array")
       import dask.array as da

       return da.from_array([1, 2, 3], chunks=2)
   ```

3. **Check available implementations**:

   ```python
   from xarray.core.duck_array_ops import available_implementations


   @pytest.mark.parametrize("implementation", available_implementations())
   def test_with_available_backends(implementation): ...
   ```

### Finding the decorators

All test requirement decorators are defined in:

- `xarray/tests/__init__.py` (look for `requires_*` decorators)
- Import them as: `from . import requires_dask, requires_scipy, ...`

### Remember

- CI environments intentionally exclude certain dependencies to test compatibility
- A test failing in "all-but-dask" because it uses dask is a test bug, not a CI issue
- Always check which decorators already exist before creating new ones
- When in doubt, look at similar existing tests for patterns to follow
