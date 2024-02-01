from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import xarray as xr
from xarray import DataArray, Dataset
from xarray.tests import create_test_data, requires_dask


@pytest.fixture(params=["numpy", pytest.param("dask", marks=requires_dask)])
def backend(request):
    return request.param


@pytest.fixture(params=["numbagg", "bottleneck", None])
def compute_backend(request):
    if request.param is None:
        options = dict(use_bottleneck=False, use_numbagg=False)
    elif request.param == "bottleneck":
        options = dict(use_bottleneck=True, use_numbagg=False)
    elif request.param == "numbagg":
        options = dict(use_bottleneck=False, use_numbagg=True)
    else:
        raise ValueError

    with xr.set_options(**options):
        yield request.param


@pytest.fixture(params=[1])
def ds(request, backend):
    if request.param == 1:
        ds = Dataset(
            dict(
                z1=(["y", "x"], np.random.randn(2, 8)),
                z2=(["time", "y"], np.random.randn(10, 2)),
            ),
            dict(
                x=("x", np.linspace(0, 1.0, 8)),
                time=("time", np.linspace(0, 1.0, 10)),
                c=("y", ["a", "b"]),
                y=range(2),
            ),
        )
    elif request.param == 2:
        ds = Dataset(
            dict(
                z1=(["time", "y"], np.random.randn(10, 2)),
                z2=(["time"], np.random.randn(10)),
                z3=(["x", "time"], np.random.randn(8, 10)),
            ),
            dict(
                x=("x", np.linspace(0, 1.0, 8)),
                time=("time", np.linspace(0, 1.0, 10)),
                c=("y", ["a", "b"]),
                y=range(2),
            ),
        )
    elif request.param == 3:
        ds = create_test_data()
    else:
        raise ValueError

    if backend == "dask":
        return ds.chunk()

    return ds


@pytest.fixture(params=[1])
def da(request, backend):
    if request.param == 1:
        times = pd.date_range("2000-01-01", freq="1D", periods=21)
        da = DataArray(
            np.random.random((3, 21, 4)),
            dims=("a", "time", "x"),
            coords=dict(time=times),
        )

    if request.param == 2:
        da = DataArray([0, np.nan, 1, 2, np.nan, 3, 4, 5, np.nan, 6, 7], dims="time")

    if request.param == "repeating_ints":
        da = DataArray(
            np.tile(np.arange(12), 5).reshape(5, 4, 3),
            coords={"x": list("abc"), "y": list("defg")},
            dims=list("zyx"),
        )

    if backend == "dask":
        return da.chunk()
    elif backend == "numpy":
        return da
    else:
        raise ValueError


@pytest.fixture(params=[Dataset, DataArray])
def type(request):
    return request.param


@pytest.fixture(params=[1])
def d(request, backend, type) -> DataArray | Dataset:
    """
    For tests which can test either a DataArray or a Dataset.
    """
    result: DataArray | Dataset
    if request.param == 1:
        ds = Dataset(
            dict(
                a=(["x", "z"], np.arange(24).reshape(2, 12)),
                b=(["y", "z"], np.arange(100, 136).reshape(3, 12).astype(np.float64)),
            ),
            dict(
                x=("x", np.linspace(0, 1.0, 2)),
                y=range(3),
                z=("z", pd.date_range("2000-01-01", periods=12)),
                w=("x", ["a", "b"]),
            ),
        )
        if type == DataArray:
            result = ds["a"].assign_coords(w=ds.coords["w"])
        elif type == Dataset:
            result = ds
        else:
            raise ValueError
    else:
        raise ValueError

    if backend == "dask":
        return result.chunk()
    elif backend == "numpy":
        return result
    else:
        raise ValueError
