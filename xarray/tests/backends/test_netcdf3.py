from __future__ import annotations

import numpy as np
import pytest

import xarray as xr
from xarray.backends.netcdf3 import coerce_nc3_dtype, encode_nc3_variable
from xarray.tests import (
    assert_array_equal,
    dask_array_api,
    raise_if_dask_computes,
    requires_dask,
    requires_netCDF4,
)


@requires_dask
@pytest.mark.parametrize(
    "dtype", ["int64", "uint64", "uint32", "uint16", "uint8", "bool"]
)
def test_encode_nc3_variable_lazy(dtype) -> None:
    values = np.array([0, 1, 2], dtype=dtype)
    array = dask_array_api.from_array(values, chunks=1)
    with raise_if_dask_computes():
        actual = encode_nc3_variable(xr.Variable("x", array))
    assert actual.chunks == array.chunks
    assert actual.dtype == coerce_nc3_dtype(values).dtype
    assert_array_equal(actual.data.compute(), coerce_nc3_dtype(values))


@requires_dask
@pytest.mark.parametrize("dtype", ["int64", "uint64", "uint32", "uint16", "uint8"])
def test_encode_nc3_variable_lazy_unsafe(dtype) -> None:
    values = np.array([0, np.iinfo(dtype).max], dtype=dtype)
    with raise_if_dask_computes():
        actual = encode_nc3_variable(
            xr.Variable("x", dask_array_api.from_array(values, chunks=1))
        )
    with pytest.raises(ValueError, match="could not safely cast"):
        actual.data.compute()


@pytest.mark.parametrize(
    "dtype", ["int64", "uint64", "uint32", "uint16", "uint8", "bool", "float64"]
)
def test_encode_nc3_variable_numpy_lazy(dtype, monkeypatch) -> None:
    values = np.array([0, 1, 2], dtype=dtype)
    expected = coerce_nc3_dtype(values)
    calls = []

    def track_coercion(array):
        calls.append(array)
        return coerce_nc3_dtype(array)

    monkeypatch.setattr("xarray.backends.netcdf3.coerce_nc3_dtype", track_coercion)
    actual = encode_nc3_variable(xr.Variable("x", values))
    assert actual.dtype == expected.dtype
    assert not calls
    assert_array_equal(actual.data, expected)
    assert len(calls) == 1


@pytest.mark.parametrize("dtype", ["int64", "uint64", "uint32", "uint16", "uint8"])
def test_encode_nc3_variable_numpy_lazy_unsafe(dtype) -> None:
    values = np.array([0, np.iinfo(dtype).max], dtype=dtype)
    actual = encode_nc3_variable(xr.Variable("x", values))
    with pytest.raises(ValueError, match="could not safely cast"):
        _ = actual.data


@requires_dask
def test_nc3_integer_days_lazy() -> None:
    values = np.array([1, 2, 3], dtype="int32")
    var = xr.Variable(
        "x", dask_array_api.from_array(values, chunks=1), {"units": "days"}
    )
    with raise_if_dask_computes():
        encoded = encode_nc3_variable(var)
    assert encoded.dtype == values.dtype
    assert_array_equal(encoded.data.compute(), values)


@pytest.mark.parametrize("dtype", ["int64", ">i8"])
@pytest.mark.parametrize("fill_value", [-1, np.nan])
def test_nc3_time_sentinel(dtype, fill_value) -> None:
    values = np.array([1, np.iinfo("int64").min, 3], dtype=dtype)
    var = xr.Variable("x", values, {"units": "days", "_FillValue": fill_value})
    encoded = encode_nc3_variable(var)
    assert_array_equal(encoded.data, [1, fill_value, 3])


@requires_dask
@requires_netCDF4
@pytest.mark.parametrize("format", ["NETCDF3_CLASSIC", "NETCDF4_CLASSIC"])
@pytest.mark.parametrize("dtype,units", [("int64", None), ("int32", "days")])
def test_nc3_write_compute_false(tmp_path, format, dtype, units) -> None:
    values = np.array([1, 2, 3], dtype=dtype)
    attrs = {"units": units} if units is not None else {}
    dataset = xr.Dataset(
        {"counts": ("x", dask_array_api.from_array(values, chunks=1), attrs)}
    )
    path = tmp_path / "lazy.nc"
    with raise_if_dask_computes():
        write = dataset.to_netcdf(path, engine="netcdf4", format=format, compute=False)
    write.compute()
    with xr.open_dataset(path, decode_times=False, decode_timedelta=False) as actual:
        assert_array_equal(actual["counts"].values, values)
        assert actual["counts"].attrs == attrs


@requires_dask
@requires_netCDF4
@pytest.mark.parametrize("format", ["NETCDF3_CLASSIC", "NETCDF4_CLASSIC"])
@pytest.mark.parametrize("chunked", [False, True])
def test_nc3_write_unsafe(tmp_path, format, chunked) -> None:
    values = np.array([0, np.iinfo("int64").max], dtype="int64")
    data = dask_array_api.from_array(values, chunks=1) if chunked else values
    dataset = xr.Dataset({"counts": ("x", data)})
    with pytest.raises(ValueError, match="could not safely cast"):
        dataset.to_netcdf(tmp_path / "unsafe.nc", engine="netcdf4", format=format)
