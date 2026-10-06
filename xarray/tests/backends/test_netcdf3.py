from __future__ import annotations

import numpy as np
import pytest

import xarray as xr
from xarray.backends import writers
from xarray.backends.netcdf3 import coerce_nc3_dtype, encode_nc3_variable
from xarray.tests import (
    assert_array_equal,
    dask_array_api,
    raise_if_dask_computes,
    requires_dask,
    requires_netCDF4,
    requires_scipy,
)

_NC3_ENGINE_FORMATS = [
    pytest.param("scipy", "NETCDF3_CLASSIC", marks=requires_scipy),
    pytest.param("netcdf4", "NETCDF3_CLASSIC", marks=requires_netCDF4),
    pytest.param("netcdf4", "NETCDF4_CLASSIC", marks=requires_netCDF4),
]


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
    var = xr.Variable("x", dask_array_api.from_array(values, chunks=1))
    with raise_if_dask_computes():
        actual = encode_nc3_variable(var, name="counts")
    with pytest.raises(ValueError, match="could not safely cast") as excinfo:
        actual.data.compute()
    assert excinfo.value.__notes__ == [
        f"Raised while encoding variable 'counts' with value {var!r}"
    ]


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
    var = xr.Variable("x", values)
    actual = encode_nc3_variable(var, name="counts")
    with pytest.raises(ValueError, match="could not safely cast") as excinfo:
        _ = actual.data
    assert excinfo.value.__notes__ == [
        f"Raised while encoding variable 'counts' with value {var!r}"
    ]


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


@pytest.mark.parametrize("engine,format", _NC3_ENGINE_FORMATS)
@pytest.mark.parametrize("chunked", [False, pytest.param(True, marks=requires_dask)])
def test_nc3_write_unsafe(tmp_path, engine, format, chunked) -> None:
    values = np.array([0, np.iinfo("int64").max], dtype="int64")
    data = dask_array_api.from_array(values, chunks=1) if chunked else values
    dataset = xr.Dataset({"counts": ("x", data)})
    expected_note = (
        "Raised while encoding variable 'counts' with value "
        f"{dataset['counts'].variable!r}"
    )
    with pytest.raises(ValueError, match="could not safely cast") as excinfo:
        dataset.to_netcdf(tmp_path / "unsafe.nc", engine=engine, format=format)
    assert excinfo.value.__notes__ == [expected_note]


@requires_dask
@pytest.mark.parametrize("engine,format", _NC3_ENGINE_FORMATS)
def test_nc3_write_compute_false_unsafe_note(
    tmp_path, engine, format, monkeypatch
) -> None:
    values = np.array([0, np.iinfo("int64").max], dtype="int64")
    dataset = xr.Dataset({"counts": ("x", dask_array_api.from_array(values, chunks=1))})
    expected_note = (
        "Raised while encoding variable 'counts' with value "
        f"{dataset['counts'].variable!r}"
    )
    stores = []
    delayed_close = writers.delayed_close_after_writes

    def capture_store(writes, store):
        stores.append(store)
        return delayed_close(writes, store)

    monkeypatch.setattr(writers, "delayed_close_after_writes", capture_store)
    try:
        with raise_if_dask_computes():
            write = dataset.to_netcdf(
                tmp_path / "unsafe-delayed.nc",
                engine=engine,
                format=format,
                compute=False,
            )
        with pytest.raises(ValueError, match="could not safely cast") as excinfo:
            write.compute(scheduler="synchronous")
        assert excinfo.value.__notes__ == [expected_note]
    finally:
        # A failing write never reaches its delayed close task.
        for store in stores:
            store.close()
