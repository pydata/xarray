#!/usr/bin/env python
"""Script to fix remaining import errors in variable.py and dataset.py."""

files_to_fix = [
    "/Users/maximilian/workspace/xarray/xarray/core/variable.py",
    "/Users/maximilian/workspace/xarray/xarray/core/dataset.py",
]

for filepath in files_to_fix:
    print(f"Fixing imports in {filepath}")
    with open(filepath) as f:
        content = f.read()

    # Replace import statement
    content = content.replace(
        "from xarray.computation.computation import apply_ufunc",
        "from xarray.computation.apply_ufunc import apply_ufunc",
    )

    with open(filepath, "w") as f:
        f.write(content)

print("All imports fixed!")
