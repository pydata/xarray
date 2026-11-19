---
name: drop-python-version
description: Drop support for the oldest Python version in xarray (e.g. 3.12 -> minimum 3.13) - pyproject.toml, pixi environments, the CI matrix, docs and config files, dead version-gated code, ruff fallout, whats-new, and the GitHub required status checks. Use when the minimum dependency policy (ci/policy.yaml) allows raising the minimum Python version.
---

# Drop a Python Version

Raises xarray's minimum supported Python version. Below, `OLD` is the version
being dropped and `NEW` the new minimum, e.g. `OLD=3.12`, `NEW=3.13`
(`pyOLD` = `py312`, `pyNEW` = `py313` in pixi and CI names).

For reference, Python 3.11 was dropped in pydata/xarray#11649.

## When to Use

- `ci/policy.yaml` (`python: 30` months) allows dropping `OLD`
- Usually done separately from bumping other dependency minimums
  (see the `upgrade-min-versions` skill)

## Steps

### 1. Find all references to the old version

```bash
git grep -nE "3\.OLD_MINOR\b|pyOLD" -- ':!doc/whats-new.rst' ':!pixi.lock'
git grep -n "sys.version_info" -- 'xarray/*.py'
```

e.g. `git grep -nE "3\.12\b|py312"`. Not every hit must change (see step 5).

### 2. pyproject.toml

- `requires-python = ">=NEW"`
- Classifiers: remove `Programming Language :: Python :: OLD`, and make sure
  every supported version up to the newest tested one is listed.

### 3. pixi.toml

- **Features:** if a `[feature.pyNEW.dependencies]` already exists, delete
  `[feature.pyOLD.dependencies]`. Otherwise rename it to `pyNEW` and set
  `python = "NEW.*"`.
- **`[feature.minimal.dependencies]`:** `python = "NEW.*"`.
- **Environments:** rename all `test-pyOLD*` environments to `test-pyNEW*`
  (`test-pyOLD`, `-with-typing`, `-bare-minimum`, `-bare-min-and-scipy`,
  `-min-versions`) and switch their `"pyOLD"` feature to `"pyNEW"`. If a
  `test-pyNEW` environment already exists, merge rather than duplicate.
- **Policy tasks:** update the `pixi:test-pyOLD-*` arguments in
  `policy-bare-minimum`, `policy-bare-min-and-scipy`, `policy-min-versions`
  and the combined policy task.

Then check the lock file resolves (`pixi.lock` is gitignored, but must work):

```bash
pixi lock
pixi run policy-min-versions
```

### 4. CI workflows

- `.github/workflows/ci.yaml`: the "bookend" `pixi-env` matrix and the
  `include` entries for the minimum Python version (`bare-minimum`,
  `bare-min-and-scipy`, `min-versions`, `with-typing` + mypy).
- `.github/workflows/ci-additional.yaml`: the `mypy-min` job (`name: Mypy OLD`
  and `PIXI_ENV`) and the `pixi-env` matrix of the typing jobs.
- `nightly-wheels.yml`, `pypi-release.yaml` pin a `python-version` for
  building. Only bump these if they are the dropped version.

### 5. Other files

- `.binder/environment.yml`: `python=NEW`
- `.github/ISSUE_TEMPLATE/bugreport.yml`: `# requires-python = ">=NEW"` in the
  MVCE script header
- `asv_bench/asv.conf.json`: `"pythons": ["NEW"]`
- `doc/getting-started-guide/installing.rst`: "Python (NEW or later)"
- `pixi.toml` `[feature.doc.dependencies]` and `.readthedocs.yaml` pin docs
  Python versions independently. Leave them unless they are `OLD`.

### 6. Remove dead version-gated code

For every `sys.version_info >= (OLD_MAJOR, NEW_MINOR)` (or `< ...`) check,
keep only the branch that is now always taken, and drop now-unused
`import sys`. Also look for:

- `typing_extensions` imports of things that are in `typing` as of `NEW`
- `# type: ignore` comments that only existed for the old version
- tests skipped or xfailed only on `OLD`

### 7. Ruff and pre-commit

Ruff's target version is derived from `requires-python`, so new rules apply:

```bash
pre-commit run --all-files
```

- Small fixes (e.g. f-string quote style after PEP 701, pyupgrade rewrites):
  apply them in this PR.
- Large syntax migrations (e.g. PEP 695 type parameters when moving to 3.12:
  `UP040`, `UP046`, `UP047`): temporarily add the rules to `ignore` in
  `pyproject.toml` with a comment, and do the migration in a follow-up PR.

### 8. Test

```bash
uv run pytest xarray -n auto
uv run dmypy run
pixi run -e test-pyNEW-bare-minimum pytest xarray -n auto
```

### 9. whats-new.rst

Under "Breaking Changes":

```rst
- Support for Python OLD has been dropped. The minimum required Python version
  is now NEW, in line with xarray's
  :ref:`minimum dependency policy <mindeps_policy>` (:pull:`XXXXX`).
  By `Name <https://github.com/handle>`_.
```

### 10. Required status checks (GitHub ruleset)

The `main` branch ruleset requires CI checks by **name**, including
`os | test-pyOLD*` jobs. After renaming the environments, those checks never
report, so they show as "Expected — Waiting for status to be reported" and the
PR is blocked even with green CI. This is expected, not a CI failure.

A repo admin must replace them:

1. Find the ruleset:
   ```bash
   gh api repos/pydata/xarray/rules/branches/main \
     --jq '.[] | select(.type=="required_status_checks") | .ruleset_id, .parameters.required_status_checks[].context'
   ```
2. Open `https://github.com/pydata/xarray/settings/rules/<ruleset_id>`, under
   "Require status checks to pass" delete each `test-pyOLD*` entry and use
   "+ Add checks" to add the matching `test-pyNEW*` job. Names must match
   exactly (e.g. `ubuntu-latest | test-pyNEW-with-typing (mypy)`). GitHub only
   suggests checks that ran recently, so the PR's CI must have run first.
3. Do this right before or right after merging: other open PRs not yet
   rebased on the new matrix will show the `pyNEW` checks as pending.

## Follow-ups

- New syntax/stdlib features available with `NEW` (e.g. PEP 695 with 3.12)
- Bump other dependency minimums (`upgrade-min-versions` skill)
