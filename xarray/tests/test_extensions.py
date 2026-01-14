from __future__ import annotations

import pickle

import pytest

import xarray as xr
from xarray.accessors import (
    DATAARRAY_ACCESSORS,
    DATASET_ACCESSORS,
    DATATREE_ACCESSORS,
    _is_package_available,
)
from xarray.core.extensions import register_datatree_accessor
from xarray.tests import assert_identical


@register_datatree_accessor("example_accessor")
@xr.register_dataset_accessor("example_accessor")
@xr.register_dataarray_accessor("example_accessor")
class ExampleAccessor:
    """For the pickling tests below."""

    def __init__(self, xarray_obj):
        self.obj = xarray_obj


class TestAccessor:
    def test_register(self) -> None:
        @register_datatree_accessor("demo")
        @xr.register_dataset_accessor("demo")
        @xr.register_dataarray_accessor("demo")
        class DemoAccessor:
            """Demo accessor."""

            def __init__(self, xarray_obj):
                self._obj = xarray_obj

            @property
            def foo(self):
                return "bar"

        dt: xr.DataTree = xr.DataTree()
        assert dt.demo.foo == "bar"

        ds = xr.Dataset()
        assert ds.demo.foo == "bar"

        da = xr.DataArray(0)
        assert da.demo.foo == "bar"
        # accessor is cached
        assert ds.demo is ds.demo

        # check descriptor
        assert ds.demo.__doc__ == "Demo accessor."
        # TODO: typing doesn't seem to work with accessors
        assert xr.Dataset.demo.__doc__ == "Demo accessor."  # type: ignore[attr-defined]
        assert isinstance(ds.demo, DemoAccessor)
        assert xr.Dataset.demo is DemoAccessor  # type: ignore[attr-defined]

        # ensure we can remove it
        del xr.Dataset.demo  # type: ignore[attr-defined]
        assert not hasattr(xr.Dataset, "demo")

        with pytest.warns(Warning, match="overriding a preexisting attribute"):

            @xr.register_dataarray_accessor("demo")
            class Foo:
                pass

        # it didn't get registered again
        assert not hasattr(xr.Dataset, "demo")

    def test_pickle_dataset(self) -> None:
        ds = xr.Dataset()
        ds_restored = pickle.loads(pickle.dumps(ds))
        assert_identical(ds, ds_restored)

        # state save on the accessor is restored
        assert ds.example_accessor is ds.example_accessor
        ds.example_accessor.value = "foo"
        ds_restored = pickle.loads(pickle.dumps(ds))
        assert_identical(ds, ds_restored)
        assert ds_restored.example_accessor.value == "foo"

    def test_pickle_dataarray(self) -> None:
        array = xr.Dataset()
        assert array.example_accessor is array.example_accessor
        array_restored = pickle.loads(pickle.dumps(array))
        assert_identical(array, array_restored)

    def test_broken_accessor(self) -> None:
        # regression test for GH933

        @xr.register_dataset_accessor("stupid_accessor")
        class BrokenAccessor:
            def __init__(self, xarray_obj):
                raise AttributeError("broken")

        with pytest.raises(RuntimeError, match=r"error initializing"):
            _ = xr.Dataset().stupid_accessor


class TestExternalAccessors:
    """Tests for typed external accessor properties."""

    def test_hasattr_false_for_uninstalled(self) -> None:
        """hasattr returns False for accessors whose packages are not installed."""
        da = xr.DataArray([1, 2, 3])
        ds = xr.Dataset({"a": da})
        dt = xr.DataTree(ds)

        for name, (_, _, _, top_pkg) in DATAARRAY_ACCESSORS.items():
            if not _is_package_available(top_pkg):
                assert not hasattr(da, name), f"hasattr should be False for {name}"

        for name, (_, _, _, top_pkg) in DATASET_ACCESSORS.items():
            if not _is_package_available(top_pkg):
                assert not hasattr(ds, name), f"hasattr should be False for {name}"

        for name, (_, _, _, top_pkg) in DATATREE_ACCESSORS.items():
            if not _is_package_available(top_pkg):
                assert not hasattr(dt, name), f"hasattr should be False for {name}"

    def test_hasattr_true_for_installed(self) -> None:
        """hasattr returns True for accessors whose packages are installed."""
        da = xr.DataArray([1, 2, 3])
        ds = xr.Dataset({"a": da})

        for name, (_, _, _, top_pkg) in DATAARRAY_ACCESSORS.items():
            if _is_package_available(top_pkg):
                assert hasattr(da, name), f"hasattr should be True for {name}"

        for name, (_, _, _, top_pkg) in DATASET_ACCESSORS.items():
            if _is_package_available(top_pkg):
                assert hasattr(ds, name), f"hasattr should be True for {name}"

    def test_attribute_error_for_uninstalled(self) -> None:
        """Accessing uninstalled accessor raises AttributeError."""
        da = xr.DataArray([1, 2, 3])

        for name, (_, _, _, top_pkg) in DATAARRAY_ACCESSORS.items():
            if not _is_package_available(top_pkg):
                with pytest.raises(AttributeError):
                    getattr(da, name)
                break  # Only need to test one

    def test_external_accessor_no_overwrite(self) -> None:
        """Known external accessors don't overwrite typed properties."""
        # The property should remain a property, not get replaced by _CachedAccessor
        for name in DATAARRAY_ACCESSORS:
            attr = getattr(xr.DataArray, name)
            assert isinstance(attr, property), f"{name} should remain a property"

        for name in DATASET_ACCESSORS:
            attr = getattr(xr.Dataset, name)
            assert isinstance(attr, property), f"{name} should remain a property"

        for name in DATATREE_ACCESSORS:
            attr = getattr(xr.DataTree, name)
            assert isinstance(attr, property), f"{name} should remain a property"

    def test_accessor_caching(self) -> None:
        """Accessor instances are cached on the object."""
        da = xr.DataArray([1, 2, 3])

        for name, (_, _, _, top_pkg) in DATAARRAY_ACCESSORS.items():
            if _is_package_available(top_pkg):
                accessor1 = getattr(da, name)
                accessor2 = getattr(da, name)
                assert accessor1 is accessor2, f"{name} accessor should be cached"
                break  # Only need to test one installed accessor
