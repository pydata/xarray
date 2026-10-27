from __future__ import annotations

import ast
from pathlib import Path

import pytest

import xarray as xr
from xarray import accessors


def _type_checking_imports() -> list[tuple[str, str]]:
    """Return the (module, name) pairs imported in accessors' TYPE_CHECKING block."""
    tree = ast.parse(Path(accessors.__file__).read_text())
    (block,) = (
        node
        for node in tree.body
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Name)
        and node.test.id == "TYPE_CHECKING"
    )
    return [
        (node.module, alias.name)
        for node in block.body
        if isinstance(node, ast.ImportFrom)
        and node.module is not None
        and not node.module.startswith("xarray.")
        for alias in node.names
    ]


@pytest.mark.parametrize(("module", "name"), _type_checking_imports())
def test_external_accessor_import(module: str, name: str) -> None:
    # the packages aren't installed in CI, so this only runs where they are
    mod = pytest.importorskip(module)
    assert hasattr(mod, name), f"{module} has no attribute {name!r}"


@pytest.mark.parametrize(
    "mixin",
    [accessors.DataArrayExternalAccessorMixin, accessors.DatasetExternalAccessorMixin],
)
def test_external_accessors_not_defined_at_runtime(mixin: type) -> None:
    assert mixin.__annotations__
    for name in mixin.__annotations__:
        assert not hasattr(mixin, name)


def test_external_accessor_missing() -> None:
    ds = xr.Dataset({"a": ("x", [1])})
    if "rio" in dir(xr.Dataset):
        pytest.skip("rioxarray is imported")
    assert not hasattr(ds, "rio")
    assert not hasattr(ds["a"], "rio")
