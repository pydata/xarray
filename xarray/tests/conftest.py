import numpy as np
import pandas as pd
import pytest

from xarray import DataArray, Dataset

from . import create_test_data, requires_dask


@pytest.fixture(params=["numpy", pytest.param("dask", marks=requires_dask)])
def backend(request):
    return request.param


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "apply_marks(marks): function to attach marks to tests and test variants",
    )


def always_sequence(obj):
    if not isinstance(obj, (list, tuple)):
        obj = [obj]

    return obj


def pytest_collection_modifyitems(session, config, items):
    for item in items:
        mark = item.get_closest_marker("apply_marks")
        if mark is None:
            continue

        marks = mark.args[0]
        if not isinstance(marks, dict):
            continue

        possible_marks = marks.get(item.originalname)
        if possible_marks is None:
            continue

        if not isinstance(possible_marks, dict):
            for mark in always_sequence(possible_marks):
                item.add_marker(mark)
            continue

        variant = item.name[len(item.originalname) :]
        to_attach = possible_marks.get(variant)
        if to_attach is None:
            continue

        for mark in always_sequence(to_attach):
            item.add_marker(mark)


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
