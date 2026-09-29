# Upgrade Minimum Dependency Versions

This skill upgrades xarray's minimum dependency versions to match the policy defined in `ci/policy.yaml`.

## When to Use

Run this skill when:
- Preparing a new release that bumps minimum versions
- The policy file has been updated and manifests need to catch up
- `pixi run policy-min-versions` shows `<` for any packages

## Steps

### 1. Check Current Policy Status

```bash
pixi run policy-min-versions
```

This shows a table with:
- `=` means the version matches policy
- `<` means the version needs to be upgraded
- `>` means the version exceeds policy (usually fine)

### 2. Update Version Pins in pixi.toml

Update versions in these sections of `pixi.toml`:

- `[feature.minimal.dependencies]` - for numpy, pandas
- `[feature.minimum-scipy.dependencies]` - for scipy
- `[feature.min-versions.dependencies]` - for all other pinned dependencies
- `[package.run-dependencies]` - for packaging

Example changes:
```toml
# [feature.minimal.dependencies]
numpy = "2.0.*"  # was "1.26.*"

# [feature.minimum-scipy.dependencies]
scipy = "1.15.*"  # was "1.13.*"

# [feature.min-versions.dependencies]
zarr = "3.0.*"  # was "2.18.*"
dask-core = "2025.1.*"  # was "2024.6.*"
```

### 3. Update pyproject.toml

Update the corresponding versions in `pyproject.toml` optional dependencies:
- `dependencies` - numpy, packaging, pandas
- `[project.optional-dependencies]` - accel, io, viz sections

### 4. Verify Lock File

```bash
pixi lock
```

This must pass. If it fails, there may be dependency conflicts to resolve.

### 5. Verify Policy Compliance

```bash
pixi run policy-min-versions
```

All packages should now show `=`.

### 6. Clean Up Obsolete Test Decorators

In `xarray/tests/__init__.py`, remove any `has_*` / `requires_*` decorators for versions that are now guaranteed by the new minimums. For example:
- If numpy >= 2.0 is now required, remove `has_numpy_2` / `requires_numpy_2`
- If dask >= 2025.1 is now required, remove `has_dask_ge_2024_*` / `has_dask_ge_2025_1_0` / `has_dask_expr` / `requires_dask_expr`
- If zarr >= 3.0 is now required, remove `has_zarr_v3` / `requires_zarr_v3`

Then search for all usages of the deleted decorators and fix those files:

```bash
# Search for usages of deleted decorators
rg "has_scipy_ge_|has_dask_ge_|has_zarr_v3[^_]|has_pandas_ge_|has_numpy_2|has_dask_expr|requires_dask_expr" xarray/tests/
```

For each file found:
- Remove imports of deleted decorators
- Remove `@requires_*` decorators that are always true
- Remove `if has_*:` conditionals (keep only the true branch)
- Remove `if not has_*:` conditionals (delete the code block)
- Delete tests marked with `@pytest.mark.skipif(has_*, ...)` (always skipped)

### 7. Update whats-new.rst

Add a table to `doc/whats-new.rst` under "Breaking Changes" documenting the version changes (in alphabetical order):

```rst
Breaking Changes
~~~~~~~~~~~~~~~~

- The minimum versions of some dependencies were changed:

  ===================== =========  =======
   Package                    Old      New
  ===================== =========  =======
    boto3                  1.34     1.36
    cartopy                0.23     0.24
    dask                 2024.6   2025.1
    distributed          2024.6   2025.1
    h5netcdf                1.4      1.5
    iris                    3.9     3.11
    lxml                    5.1      5.3
    matplotlib              3.8     3.10
    numpy                   1.26      2.0
    packaging              24.1     24.2
    rasterio                1.3      1.4
    scipy                   1.13     1.15
    toolz                  0.12      1.0
    zarr                   2.18      3.0
  ===================== =========  =======
```

## Common Issues

### Lock file conflicts
If `pixi lock` fails, check for incompatible version combinations. Some packages (like h5py/hdf5) are noted in comments as prone to conflicts.

### Test failures after cleanup
After removing version guards, some tests may fail if they relied on version-specific behavior. Update the test logic to use only the new minimum version's behavior.
