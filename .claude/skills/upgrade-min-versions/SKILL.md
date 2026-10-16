---
name: upgrade-min-versions
description: Upgrade the minimum versions of xarray's dependencies (numpy, pandas, dask, zarr, scipy, ...) to match ci/policy.yaml, including pixi.toml/pyproject.toml pins, test decorator cleanup and the whats-new table. Use when bumping dependency minimums or when `pixi run policy-min-versions` shows `<`. For dropping a Python version, use drop-python-version instead.
---

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

All packages should now show `=`, and the command must exit with 0. Only a `>`
(pin newer than the policy) fails the check; it is allowed for packages listed
under `ignored_violations` in `ci/policy.yaml`.

Remove `overrides` in `ci/policy.yaml` that are no longer needed.

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
- Replace `@requires_*` version decorators with the plain `@requires_<package>`
  decorator, unless the test or its class already has one. They also skip the
  test when the package is not installed at all, e.g. `@requires_dask_ge_2024_11_0`
  in the `bare-minimum` and `no-dask` environments.
- Remove `if has_*:` conditionals (keep only the true branch), but use
  `if has_<package>:` instead if the test doesn't require the package already.
- Remove `if not has_*:` conditionals (delete the code block)
- Delete tests marked with `@pytest.mark.skipif(has_*, ...)` (always skipped)

Also grep for decorators matching the new minimum exactly (e.g. `has_zarr_v3_dtypes`
for zarr 3.1) and for version guards defined elsewhere (e.g. `HAS_STRING_DTYPE`
in `xarray/compat/npcompat.py`).

### 7. Clean Up Version Guards in the Source

The source code has version guards too, which become dead code with the new
minimums. Search for them, not only for `module_available(..., minversion=...)`:

```bash
rg -n -i "minversion|__version__|mod_version|NumpyVersion|oldest supported|remove (once|when)|delete when|requires (numpy|dask|pandas|zarr)|numpy ?[<>]=? ?[0-9]" xarray --glob '!xarray/tests/**'
```

Also go through TODO and compat comments mentioning a dependency version, in
the source and the tests (e.g. `# use np.cumulative_sum(...) when we support
numpy>=2`, `# TODO: delete when min dask>=2023.12.1`, or `try: ... except
ImportError:` fallbacks for old versions):

```bash
rg -n -i "(TODO|FIXME|NOTE|compat|workaround|legacy|once|when|until|older|drop)[^\n]{0,80}\b(numpy|pandas|scipy|dask|zarr|netcdf4|h5netcdf|flox|numbagg|bottleneck|sparse|pint|matplotlib|cftime|numba)\b[^\n]{0,40}([<>]=?|version|v?[0-9]+\.[0-9]+)" xarray --glob '!xarray/core/_aggregations.py'
```

- Remove branches for versions older than the new minimum and keep the code for
  the new versions, e.g. `try: from numpy.exceptions import RankWarning` /
  `except ImportError:` becomes a plain import at the top of the module.
- `module_available("flox", minversion=...)` also checks that the package is
  installed, so replace it with `module_available("flox")`, not `True`.
- Guards for versions _above_ the new minimum stay (e.g. `flox>=0.10.5`).
- Leave TODOs that depend on more than the version (e.g. a behavior change or
  refactor) for a separate PR, and mention them in the PR.
- Delete compat modules that only re-export something available in all
  supported versions, and their entries in `pyproject.toml` (e.g. coverage
  `omit`).
- The flox checks in `xarray/core/_aggregations.py` are generated: change
  `min_flox_version` in `xarray/util/generate_aggregations.py` and run
  `pixi run generate-aggregations`. The generator writes to
  `../xarray/xarray/...` relative to the current directory, so in a worktree
  that is not named `xarray` it overwrites the main checkout instead. Run it
  from a checkout named `xarray`, or check `git status` in both afterwards.

### 8. Update whats-new.rst

Add a table to `doc/whats-new.rst` under "Breaking Changes" documenting the version changes (in alphabetical order):

```rst
Breaking Changes
~~~~~~~~~~~~~~~~

- The minimum versions of some dependencies were changed (:pull:`<PR number>`):

  ===========  ======  ======
  Package         Old     New
  ===========  ======  ======
  boto3          1.34    1.36
  cartopy        0.23    0.24
  dask         2024.6  2025.1
  distributed  2024.6  2025.1
  h5netcdf        1.4     1.5
  iris            3.9    3.11
  lxml            5.1     5.3
  matplotlib      3.8    3.10
  numpy          1.26     2.0
  packaging      24.1    24.2
  rasterio        1.3     1.4
  scipy          1.13    1.15
  toolz          0.12     1.0
  zarr           2.18     3.0
  ===========  ======  ======
```

Only edit this entry, older releases contain the same sentence.

Every value must lie within the `=` signs of its column, otherwise the docs
build fails with "Malformed table". Size the columns to the longest value, e.g.
`2025.9` for dask.

### 9. Open the PR

Always ask the user before opening the PR, and show them the description first.

Follow `.github/PULL_REQUEST_TEMPLATE.md`: a short description (an intro line
and a few one-line bullets, e.g. the notable bumps and any policy override),
the checklist with the relevant items ticked (whats-new) and the AI disclosure
section filled in. Then add the PR number to the whats-new entry.

## Common Issues

### Lock file conflicts

If `pixi lock` fails, check for incompatible version combinations. Some packages (like h5py/hdf5) are noted in comments as prone to conflicts.

A package that is not bumped by the policy can still conflict with the new
minimums, e.g. netcdf4 1.6 has no builds for numpy 2. Bump it to the oldest
version that solves, and make sure the policy check still passes.

### Test failures after cleanup

After removing version guards, some tests may fail if they relied on version-specific behavior. Update the test logic to use only the new minimum version's behavior.

Run the full test suite in an environment with the latest versions (e.g.
`test-py314`) as well, not only in `test-py312-min-versions`: a workaround whose
TODO names an old version may still be needed for newer versions.

Run the touched tests not only in `test-py312-min-versions` but also in
`test-py312-bare-minimum` and `test-py314-no-dask`, which catch removed
decorators that also skipped tests for missing packages. Run mypy in the CI
environment, `pixi run -e test-py314-with-typing -- python -m mypy`; the
`typing` environment lacks optional dependencies and reports false positives.
