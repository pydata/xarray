import numpy as np
import pytest

try:
    import tiledb
except ImportError:
    pass

from xarray import DataArray, Dataset


@pytest.fixture
def create_tiledb_example(tmpdir):
    # Define data
    float_data = np.linspace(
        -1.0, 1.0, num=32, endpoint=True, dtype=np.float64
    ).reshape(8, 4)
    int_data = np.arange(0, 32, dtype=np.int32).reshape(8, 4)
    # Create expected dataset
    expected = Dataset(
        data_vars={
            "pressure": DataArray(
                data=float_data,
                dims=["time", "x"],
                attrs={"long_name": "example float data"},
            ),
            "count": DataArray(
                data=int_data,
                dims=["time", "x"],
                attrs={"long_name": "example int data"},
            ),
        },
        coords={"time": np.arange(1, 9), "x": np.arange(1, 5)},
        attrs={"global_1": "value1", "global_2": "value2"},
    )
    array_uri = str(tmpdir.join("tiledb_example_1"))
    schema = tiledb.ArraySchema(
        domain=tiledb.Domain(
            tiledb.Dim(name="time", domain=(1, 8), tile=4, dtype=np.int32),
            tiledb.Dim(name="x", domain=(1, 4), tile=4, dtype=np.int32),
        ),
        sparse=False,
        attrs=[
            tiledb.Attr(name="count", dtype=np.int32),
            tiledb.Attr(name="pressure", dtype=np.float64),
        ],
    )
    tiledb.DenseArray.create(array_uri, schema)
    with tiledb.DenseArray(array_uri, mode="w") as array:
        array[:, :] = {
            "pressure": float_data,
            "count": int_data,
        }
        array.meta["global_1"] = "value1"
        array.meta["global_2"] = "value2"
        array.meta["__tiledb_attr.float_data.long_name"] = "example float data"
        array.meta["__tiledb_attr.int_data.long_name"] = "example int data"
    return array_uri, expected


@pytest.fixture
def create_tiledb_datetime_example(tmpdir):
    _data = np.linspace(-1.0, 20.0, num=16, endpoint=True, dtype=np.float64)
    _date = np.arange(np.datetime64("2000-01-01"), np.datetime64("2000-01-17"))
    # Create expected dataset
    expected = Dataset(
        data_vars={"temperature": DataArray(data=_data, dims="date")},
        coords={"date": _date},
    )
    # Create TileDB array
    array_uri = str(tmpdir.join("tiledb_example_2"))
    schema = tiledb.ArraySchema(
        domain=tiledb.Domain(
            tiledb.Dim(
                name="date",
                domain=(np.datetime64("2000-01-01"), np.datetime64("2000-01-16")),
                tile=np.timedelta64(4, "D"),
                dtype=np.datetime64("", "D"),
            ),
        ),
        attrs=[tiledb.Attr(name="temperature", dtype=np.float64)],
    )
    tiledb.DenseArray.create(array_uri, schema)
    with tiledb.DenseArray(array_uri, mode="w") as array:
        array[:] = {"temperature": _data}
    return array_uri, expected
