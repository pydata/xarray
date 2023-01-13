from __future__ import annotations

import pandas as pd
import pytest

from xarray.core.alignment import align
from xarray.core.coordinates import Coordinates
from xarray.core.dataarray import DataArray
from xarray.core.dataset import Dataset
from xarray.core.indexes import PandasIndex, PandasMultiIndex
from xarray.tests import assert_identical, source_ndarray


class TestCoordinates:
    def test_init_noindex(self) -> None:
        coords = Coordinates(coords={"foo": ("x", [0, 1, 2])})
        expected = Dataset(coords={"foo": ("x", [0, 1, 2])})
        assert_identical(coords.to_dataset(), expected)

    def test_init_from_coords(self) -> None:
        expected = Dataset(coords={"foo": ("x", [0, 1, 2])})
        coords = Coordinates(coords=expected.coords)
        assert_identical(coords.to_dataset(), expected)

        # test variables copied
        assert coords.variables["foo"] is not expected.variables["foo"]

        # default index
        expected = Dataset(coords={"x": ("x", [0, 1, 2])})
        coords = Coordinates(coords=expected.coords, indexes=expected.xindexes)
        assert_identical(coords.to_dataset(), expected)

    def test_init_empty(self) -> None:
        coords = Coordinates()
        assert len(coords) == 0

    def test_init_index_error(self) -> None:
        idx = PandasIndex([1, 2, 3], "x")
        with pytest.raises(ValueError, match="no coordinate variables found"):
            Coordinates(indexes={"x": idx})

        with pytest.raises(TypeError, match=".* is not an `xarray.indexes.Index`"):
            Coordinates(coords={"x": ("x", [1, 2, 3])}, indexes={"x": "not_an_xarray_index"})  # type: ignore

    def test_init_dim_sizes_conflict(self) -> None:
        with pytest.raises(ValueError):
            Coordinates(coords={"foo": ("x", [1, 2]), "bar": ("x", [1, 2, 3, 4])})

    def test_from_pandas_multiindex(self) -> None:
        midx = pd.MultiIndex.from_product([["a", "b"], [1, 2]], names=("one", "two"))
        coords = Coordinates.from_pandas_multiindex(midx, "x")

        assert isinstance(coords.xindexes["x"], PandasMultiIndex)
        assert coords.xindexes["x"].index.equals(midx)
        assert coords.xindexes["x"].dim == "x"

        expected = PandasMultiIndex(midx, "x").create_variables()
        assert list(coords.variables) == list(expected)
        for name in ("x", "one", "two"):
            assert_identical(expected[name], coords.variables[name])

    def test_dims(self) -> None:
        _ds = Dataset(coords={"x": [0, 1, 2]})
        coords = Coordinates(coords=_ds.coords, indexes=_ds.xindexes)
        assert coords.dims == {"x": 3}

    def test_sizes(self) -> None:
        _ds = Dataset(coords={"x": [0, 1, 2]})
        coords = Coordinates(coords=_ds.coords, indexes=_ds.xindexes)
        assert coords.sizes == {"x": 3}

    def test_dtypes(self) -> None:
        _ds = Dataset(coords={"x": [0, 1, 2]})
        coords = Coordinates(coords=_ds.coords, indexes=_ds.xindexes)
        assert coords.dtypes == {"x": int}

    def test_getitem(self) -> None:
        _ds = Dataset(coords={"x": [0, 1, 2]})
        coords = Coordinates(coords=_ds.coords, indexes=_ds.xindexes)
        assert_identical(
            coords["x"],
            DataArray([0, 1, 2], coords={"x": [0, 1, 2]}, name="x"),
        )

    def test_delitem(self) -> None:
        _ds = Dataset(coords={"x": [0, 1, 2]})
        coords = Coordinates(coords=_ds.coords, indexes=_ds.xindexes)
        del coords["x"]
        assert "x" not in coords

    def test_update(self) -> None:
        _ds = Dataset(coords={"x": [0, 1, 2]})
        coords = Coordinates(coords=_ds.coords, indexes=_ds.xindexes)

        coords.update({"y": ("y", [4, 5, 6])})
        assert "y" in coords
        assert "y" in coords.xindexes
        expected = DataArray([4, 5, 6], coords={"y": [4, 5, 6]}, name="y")
        assert_identical(coords["y"], expected)

    def test_equals(self):
        _ds = Dataset(coords={"x": [0, 1, 2]})
        coords = Coordinates(coords=_ds.coords, indexes=_ds.xindexes)

        assert coords.equals(coords)
        assert not coords.equals("no_a_coords")

    def test_identical(self):
        _ds = Dataset(coords={"x": [0, 1, 2]})
        coords = Coordinates(coords=_ds.coords, indexes=_ds.xindexes)

        assert coords.identical(coords)
        assert not coords.identical("no_a_coords")

    def test_copy(self) -> None:
        no_index_coords = Coordinates({"foo": ("x", [1, 2, 3])})
        copied = no_index_coords.copy()
        assert_identical(no_index_coords, copied)
        v0 = no_index_coords.variables["foo"]
        v1 = copied.variables["foo"]
        assert v0 is not v1
        assert source_ndarray(v0.data) is source_ndarray(v1.data)

        deep_copied = no_index_coords.copy(deep=True)
        assert_identical(no_index_coords.to_dataset(), deep_copied.to_dataset())
        v0 = no_index_coords.variables["foo"]
        v1 = deep_copied.variables["foo"]
        assert v0 is not v1
        assert source_ndarray(v0.data) is not source_ndarray(v1.data)

    def test_align(self) -> None:
        _ds = Dataset(coords={"x": [0, 1, 2]})
        coords = Coordinates(coords=_ds.coords, indexes=_ds.xindexes)

        left = coords

        # test Coordinates._reindex_callback
        right = coords.to_dataset().isel(x=[0, 1]).coords
        left2, right2 = align(left, right, join="inner")
        assert_identical(left2, right2)

        # test Coordinates._overwrite_indexes
        right.update({"x": ("x", [4, 5, 6])})
        left2, right2 = align(left, right, join="override")
        assert_identical(left2, left)
        assert_identical(left2, right2)
