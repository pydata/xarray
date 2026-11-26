import numpy as np
import pytest

import xarray as xr
import xarray.core.missing
from xarray.tests import (
    requires_dask,
    requires_polars,
    requires_pyarrow,
)


class TestDatasetToArrow:
    @requires_pyarrow
    def test_to_arrow_1d(self) -> None:
        import pyarrow as pa

        ds = xr.Dataset(
            {"temperature": ("x", [1.0, 2.0, 3.0])},
            coords={"x": [10, 20, 30]},
        )
        table = ds.to_arrow()

        assert isinstance(table, pa.Table)
        assert set(table.column_names) == {"x", "temperature"}
        assert table.num_rows == 3
        assert table.schema.field("x").type == pa.int64()
        assert table.schema.field("temperature").type == pa.float64()
        np.testing.assert_array_equal(table["x"].to_pylist(), [10, 20, 30])
        np.testing.assert_array_equal(table["temperature"].to_pylist(), [1.0, 2.0, 3.0])

    @requires_pyarrow
    def test_to_arrow_2d_multiple_data_vars(self) -> None:
        import pyarrow as pa

        ds = xr.Dataset(
            {
                "temperature": (
                    ["x", "y"],
                    np.arange(6, dtype=float).reshape(2, 3),
                ),
                "precipitation": (
                    ["x", "y"],
                    np.arange(6, dtype=float).reshape(2, 3) * 2,
                ),
            },
            coords={"x": [0, 1], "y": [10, 20, 30]},
        )
        table = ds.to_arrow()

        assert isinstance(table, pa.Table)
        assert set(table.column_names) == {"x", "y", "temperature", "precipitation"}
        assert table.num_rows == 6
        assert table.schema.field("x").type == pa.int64()
        assert table.schema.field("y").type == pa.int64()
        assert table.schema.field("temperature").type == pa.float64()
        assert table.schema.field("precipitation").type == pa.float64()
        np.testing.assert_array_equal(
            table["temperature"].to_pylist(), list(np.arange(6, dtype=float))
        )
        np.testing.assert_array_equal(
            table["precipitation"].to_pylist(), list(np.arange(6, dtype=float) * 2)
        )
        # x repeats for each y: [0,0,0,1,1,1]
        np.testing.assert_array_equal(table["x"].to_pylist(), [0, 0, 0, 1, 1, 1])
        # y cycles for each x: [10,20,30,10,20,30]
        np.testing.assert_array_equal(table["y"].to_pylist(), [10, 20, 30, 10, 20, 30])

    @requires_pyarrow
    def test_to_arrow_no_coords(self) -> None:
        import pyarrow as pa

        ds = xr.Dataset({"data": ("x", [1, 2, 3])})
        table = ds.to_arrow()

        assert isinstance(table, pa.Table)
        assert table.column_names == ["data"]
        np.testing.assert_array_equal(table["data"].to_pylist(), [1, 2, 3])

    @requires_pyarrow
    def test_to_arrow_curvilinear_coords(self) -> None:
        import pyarrow as pa

        # non-dimension coordinates spanning multiple dims (e.g. a curvilinear
        # grid with 2D lat/lon) should be supported
        lat = np.array([[10.0, 11.0, 12.0], [13.0, 14.0, 15.0]])
        lon = np.array([[20.0, 21.0, 22.0], [23.0, 24.0, 25.0]])
        ds = xr.Dataset(
            {"data": (["x", "y"], np.arange(6, dtype=float).reshape(2, 3))},
            coords={"lat": (["x", "y"], lat), "lon": (["x", "y"], lon)},
        )
        table = ds.to_arrow()

        assert isinstance(table, pa.Table)
        assert set(table.column_names) == {"lat", "lon", "data"}
        assert table.num_rows == 6
        np.testing.assert_array_equal(table["lat"].to_pylist(), lat.ravel())
        np.testing.assert_array_equal(table["lon"].to_pylist(), lon.ravel())
        np.testing.assert_array_equal(
            table["data"].to_pylist(), np.arange(6, dtype=float)
        )

    @requires_pyarrow
    def test_to_arrow_schema_metadata(self) -> None:
        import json

        ds = xr.Dataset(
            {"temperature": ("x", [1.0, 2.0, 3.0], {"units": "K"})},
            coords={"x": ("x", [10, 20, 30], {"units": "m"})},
            attrs={"title": "test"},
        )
        table = ds.to_arrow()
        schema = table.schema

        assert schema.metadata[b"xarray:arrow_schema_version"] == b"v1"

        xarray_meta = json.loads(schema.metadata[b"xarray"])
        assert xarray_meta == {
            "dims": ["x"],
            "shape": [3],
            "attrs": {"title": "test"},
            "coords": {
                "x": {
                    "dims": ["x"],
                    "attrs": {"units": "m"},
                    "dtype": "int64",
                    "shape": [3],
                }
            },
            "variables": {
                "temperature": {
                    "dims": ["x"],
                    "attrs": {"units": "K"},
                    "dtype": "float64",
                    "shape": [3],
                }
            },
        }

    @requires_pyarrow
    def test_to_arrow_matches_pa_table(self) -> None:
        import pyarrow as pa

        ds = xr.Dataset(
            {"temperature": (["x", "y"], np.arange(6, dtype=float).reshape(2, 3))},
            coords={"x": [0, 1], "y": [10, 20, 30]},
        )

        assert ds.to_arrow().equals(pa.table(ds))

    @requires_dask
    @requires_pyarrow
    def test_to_arrow_chunked_raises(self) -> None:
        ds = xr.Dataset(
            {"temperature": ("x", [1.0, 2.0, 3.0])},
            coords={"x": [10, 20, 30]},
        ).chunk()

        with pytest.raises(ValueError, match="must be computed first"):
            ds.to_arrow()

    @requires_polars
    def test_to_arrow_polars_dataframe(self) -> None:
        import polars as pl

        ds = xr.Dataset(
            {"temperature": ("x", [1.0, 2.0, 3.0])},
            coords={"x": [10, 20, 30]},
        )
        df = pl.from_arrow(ds.to_arrow())

        assert isinstance(df, pl.DataFrame)
        assert set(df.columns) == {"x", "temperature"}
        assert len(df) == 3
        np.testing.assert_array_equal(df["x"].to_list(), [10, 20, 30])
        np.testing.assert_array_equal(df["temperature"].to_list(), [1.0, 2.0, 3.0])


class TestDataArrayArrowPyCapsule:
    @requires_pyarrow
    def test_pyarrow_table_1d(self):
        import pyarrow as pa

        da = xr.DataArray(
            [1.0, 2.0, 3.0],
            dims=["x"],
            coords={"x": [10, 20, 30]},
            name="temperature",
        )
        table = pa.table(da)

        assert isinstance(table, pa.Table)
        assert set(table.column_names) == {"x", "temperature"}
        assert table.num_rows == 3
        assert table.schema.field("x").type == pa.int64()
        assert table.schema.field("temperature").type == pa.float64()
        np.testing.assert_array_equal(table["x"].to_pylist(), [10, 20, 30])
        np.testing.assert_array_equal(table["temperature"].to_pylist(), [1.0, 2.0, 3.0])

    @requires_pyarrow
    def test_pyarrow_table_2d(self):
        import pyarrow as pa

        da = xr.DataArray(
            np.arange(6, dtype=float).reshape(2, 3),
            dims=["x", "y"],
            coords={"x": [0, 1], "y": [10, 20, 30]},
            name="data",
        )
        table = pa.table(da)

        assert isinstance(table, pa.Table)
        assert set(table.column_names) == {"x", "y", "data"}
        assert table.num_rows == 6
        assert table.schema.field("x").type == pa.int64()
        assert table.schema.field("y").type == pa.int64()
        assert table.schema.field("data").type == pa.float64()
        np.testing.assert_array_equal(
            table["data"].to_pylist(), list(np.arange(6, dtype=float))
        )

    @requires_pyarrow
    def test_data_array_unnamed_raises(self) -> None:
        import pyarrow as pa

        da = xr.DataArray([1, 2, 3], dims=["x"], coords={"x": [0, 1, 2]})

        with pytest.raises(ValueError, match="unnamed DataArray"):
            pa.table(da)
        with pytest.raises(ValueError, match="unnamed DataArray"):
            pa.schema(da)

    @requires_polars
    def test_polars_dataframe_1d(self):
        import polars as pl

        da = xr.DataArray(
            [1.0, 2.0, 3.0],
            dims=["x"],
            coords={"x": [10, 20, 30]},
            name="temperature",
        )
        df = pl.from_arrow(da)

        assert isinstance(df, pl.DataFrame)
        assert set(df.columns) == {"x", "temperature"}
        assert len(df) == 3
        np.testing.assert_array_equal(df["x"].to_list(), [10, 20, 30])
        np.testing.assert_array_equal(df["temperature"].to_list(), [1.0, 2.0, 3.0])

    @requires_polars
    def test_polars_dataframe_2d(self):
        import polars as pl

        da = xr.DataArray(
            np.arange(6, dtype=float).reshape(2, 3),
            dims=["x", "y"],
            coords={"x": [0, 1], "y": [10, 20, 30]},
            name="data",
        )
        df = pl.from_arrow(da)

        assert isinstance(df, pl.DataFrame)
        assert set(df.columns) == {"x", "y", "data"}
        assert len(df) == 6
        np.testing.assert_array_equal(
            df["data"].to_list(), list(np.arange(6, dtype=float))
        )
        # x repeats for each y: [0,0,0,1,1,1]
        np.testing.assert_array_equal(df["x"].to_list(), [0, 0, 0, 1, 1, 1])
        # y cycles for each x: [10,20,30,10,20,30]
        np.testing.assert_array_equal(df["y"].to_list(), [10, 20, 30, 10, 20, 30])

    @requires_dask
    @requires_pyarrow
    def test_dask_dataarray(self):
        import dask.array as da
        import pyarrow as pa

        dask_da = xr.DataArray(
            da.from_array(np.arange(6, dtype=float).reshape(2, 3)),
            dims=["x", "y"],
            coords={"x": [0, 1], "y": [10, 20, 30]},
            name="data",
        )
        with pytest.raises(ValueError):
            pa.table(dask_da)

    @requires_dask
    @requires_pyarrow
    def test_dask_coord_raises(self) -> None:
        import pyarrow as pa

        da = xr.DataArray([1, 2], dims="x", name="v").assign_coords(
            c=xr.DataArray([10, 20], dims="x").chunk()
        )
        with pytest.raises(ValueError, match="must be computed first"):
            pa.table(da)

    @requires_polars
    @requires_pyarrow
    def test_polars_pyarrow_consistent(self):
        import polars as pl
        import pyarrow as pa

        da = xr.DataArray(
            np.arange(6, dtype=float).reshape(2, 3),
            dims=["x", "y"],
            coords={"x": [0, 1], "y": [10, 20, 30]},
            name="data",
        )
        pa_table = pa.table(da)
        pl_df = pl.from_arrow(da)

        for col in pa_table.column_names:
            np.testing.assert_array_equal(
                pa_table[col].to_pylist(), pl_df[col].to_list()
            )

    @requires_pyarrow
    def test_arrow_schema_fields(self):
        import pyarrow as pa

        da = xr.DataArray(
            np.arange(6, dtype=float).reshape(2, 3),
            dims=["x", "y"],
            coords={"x": [0, 1], "y": [10, 20, 30]},
            name="data",
        )
        schema = pa.schema(da)

        assert isinstance(schema, pa.Schema)
        assert schema.names == ["x", "y", "data"]
        assert schema.field("x").type == pa.int64()
        assert schema.field("y").type == pa.int64()
        assert schema.field("data").type == pa.float64()

    @requires_pyarrow
    def test_arrow_schema_metadata(self):
        import json

        import pyarrow as pa

        da = xr.DataArray(
            [1.0, 2.0, 3.0],
            dims=["x"],
            coords={"x": [10, 20, 30]},
            name="temperature",
            attrs={"units": "K", "long_name": "temperature"},
        )
        schema = pa.schema(da)

        assert schema.metadata[b"xarray:arrow_schema_version"] == b"v1"

        xarray_meta = json.loads(schema.metadata[b"xarray"])
        assert xarray_meta == {
            "name": "temperature",
            "dims": ["x"],
            "shape": [3],
            "attrs": {"units": "K", "long_name": "temperature"},
            "coords": {
                "x": {
                    "dims": ["x"],
                    "attrs": {},
                    "dtype": "int64",
                    "shape": [3],
                }
            },
        }

    @requires_pyarrow
    def test_pyarrow_table_curvilinear_coords(self):
        import pyarrow as pa

        # non-dimension coordinates spanning multiple dims (e.g. a curvilinear
        # grid with 2D lat/lon) should be supported
        lat = np.array([[10.0, 11.0, 12.0], [13.0, 14.0, 15.0]])
        lon = np.array([[20.0, 21.0, 22.0], [23.0, 24.0, 25.0]])
        da = xr.DataArray(
            np.arange(6, dtype=float).reshape(2, 3),
            dims=["x", "y"],
            coords={"lat": (["x", "y"], lat), "lon": (["x", "y"], lon)},
            name="data",
        )
        table = pa.table(da)

        assert isinstance(table, pa.Table)
        assert set(table.column_names) == {"lat", "lon", "data"}
        assert table.num_rows == 6
        assert table.schema.field("lat").type == pa.float64()
        assert table.schema.field("lon").type == pa.float64()
        assert table.schema.field("data").type == pa.float64()
        np.testing.assert_array_equal(table["lat"].to_pylist(), lat.ravel())
        np.testing.assert_array_equal(table["lon"].to_pylist(), lon.ravel())
        np.testing.assert_array_equal(
            table["data"].to_pylist(), np.arange(6, dtype=float)
        )

    @requires_pyarrow
    def test_pyarrow_table_transposed_coords(self):
        import pyarrow as pa

        lat = np.array([[10.0, 11.0], [12.0, 13.0], [14.0, 15.0]])

        # Array with swapped dims order
        da = xr.DataArray(
            np.arange(6, dtype=float).reshape(2, 3),
            dims=["x", "y"],
            coords={"lat": (["y", "x"], lat)},
            name="data",
        )
        table = pa.table(da)

        assert isinstance(table, pa.Table)
        assert set(table.column_names) == {"lat", "data"}
        assert table.num_rows == 6
        assert table.schema.field("lat").type == pa.float64()
        np.testing.assert_array_equal(table["lat"].to_pylist(), lat.T.ravel())
        np.testing.assert_array_equal(
            table["data"].to_pylist(), np.arange(6, dtype=float)
        )
