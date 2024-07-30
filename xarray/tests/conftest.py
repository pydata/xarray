from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import xarray as xr
from xarray import DataArray, Dataset
from xarray.core.datatree import DataTree
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


@pytest.fixture(scope="module")
def create_test_datatree():
    """
    Create a test datatree with this structure:

    <datatree.DataTree>
    |-- set1
    |   |-- <xarray.Dataset>
    |   |   Dimensions:  ()
    |   |   Data variables:
    |   |       a        int64 0
    |   |       b        int64 1
    |   |-- set1
    |   |-- set2
    |-- set2
    |   |-- <xarray.Dataset>
    |   |   Dimensions:  (x: 2)
    |   |   Data variables:
    |   |       a        (x) int64 2, 3
    |   |       b        (x) int64 0.1, 0.2
    |   |-- set1
    |-- set3
    |-- <xarray.Dataset>
    |   Dimensions:  (x: 2, y: 3)
    |   Data variables:
    |       a        (y) int64 6, 7, 8
    |       set0     (x) int64 9, 10

    The structure has deliberately repeated names of tags, variables, and
    dimensions in order to better check for bugs caused by name conflicts.
    """

    def _create_test_datatree(modify=lambda ds: ds):
        set1_data = modify(xr.Dataset({"a": 0, "b": 1}))
        set2_data = modify(xr.Dataset({"a": ("x", [2, 3]), "b": ("x", [0.1, 0.2])}))
        root_data = modify(xr.Dataset({"a": ("y", [6, 7, 8]), "set0": ("x", [9, 10])}))

        # Avoid using __init__ so we can independently test it
        root: DataTree = DataTree(data=root_data)
        set1: DataTree = DataTree(name="set1", parent=root, data=set1_data)
        DataTree(name="set1", parent=set1)
        DataTree(name="set2", parent=set1)
        set2: DataTree = DataTree(name="set2", parent=root, data=set2_data)
        DataTree(name="set1", parent=set2)
        DataTree(name="set3", parent=root)

        return root

    return _create_test_datatree


@pytest.fixture(scope="module")
def simple_datatree(create_test_datatree):
    """
    Invoke create_test_datatree fixture (callback).

    Returns a DataTree.
    """
    return create_test_datatree()


@pytest.fixture(scope="module")
def create_test_multidataset_withoutroot_datatree():
    """
    Create a test datatree with this structure:

    <xarray.DataTree>
    Group: /
    ├── Group: /Main
    │       Dimensions:  (yi: 50, xi: 60, t: 30)
    │       Coordinates:
    │         * t        (t) int64 240B 0 1 2 3 4 5 6 7 8 9 ... 21 22 23 24 25 26 27 28 29
    │       Dimensions without coordinates: yi, xi
    │       Data variables:
    │           x        (yi, xi) int64 24kB 0 0 0 0 0 0 0 0 0 ... 49 49 49 49 49 49 49 49
    │           y        (yi, xi) int64 24kB 0 1 2 3 4 5 6 7 8 ... 52 53 54 55 56 57 58 59
    │           u        (t, yi, xi) float64 720kB 1.0 1.0 1.0 1.0 1.0 ... 1.0 1.0 1.0 1.0
    │           v        (t, yi, xi) float64 720kB 1.0 1.0 1.0 1.0 1.0 ... 1.0 1.0 1.0 1.0
    │           p        (t, yi, xi) float64 720kB 1.0 1.0 1.0 1.0 1.0 ... 1.0 1.0 1.0 1.0
    ├── Group: /ovr1
    │       Dimensions:  (yi: 51, xi: 51, t: 10)
    │       Coordinates:
    │         * t        (t) int64 80B 10 11 12 13 14 15 16 17 18 19
    │       Dimensions without coordinates: yi, xi
    │       Data variables:
    │           x        (yi, xi) float64 21kB 150.0 150.0 150.0 150.0 ... 200.0 200.0 200.0
    │           y        (yi, xi) int64 21kB 0 1 2 3 4 5 6 7 8 ... 43 44 45 46 47 48 49 50
        │           u        (t, yi, xi) float64 208kB 1.0 1.0 1.0 1.0 1.0 ... 1.0 1.0 1.0 1.0
        │           v        (t, yi, xi) float64 208kB 1.0 1.0 1.0 1.0 1.0 ... 1.0 1.0 1.0 1.0
        │           p        (t, yi, xi) float64 208kB 1.0 1.0 1.0 1.0 1.0 ... 1.0 1.0 1.0 1.0
        │           cx       (t) int64 80B 10 11 12 13 14 15 16 17 18 19
        │           cy       (t) int64 80B 10 11 12 13 14 15 16 17 18 19
        └── Group: /ovr2
                Dimensions:  (yi: 51, xi: 51, t: 20)
                Coordinates:
                  * t        (t) int64 160B 10 11 12 13 14 15 16 17 ... 22 23 24 25 26 27 28 29
                Dimensions without coordinates: yi, xi
                Data variables:
                    x        (yi, xi) int64 21kB 0 0 0 0 0 0 0 0 0 ... 50 50 50 50 50 50 50 50
                    y        (yi, xi) int64 21kB 0 1 2 3 4 5 6 7 8 ... 43 44 45 46 47 48 49 50
                    u        (t, yi, xi) float64 416kB 1.0 1.0 1.0 1.0 1.0 ... 1.0 1.0 1.0 1.0
                    v        (t, yi, xi) float64 416kB 1.0 1.0 1.0 1.0 1.0 ... 1.0 1.0 1.0 1.0
                    p        (t, yi, xi) float64 416kB 1.0 1.0 1.0 1.0 1.0 ... 1.0 1.0 1.0 1.0
                    cx       (t) int64 160B 10 11 12 13 14 15 16 17 ... 22 23 24 25 26 27 28 29
                    cy       (t) int64 160B 10 11 12 13 14 15 16 17 ... 22 23 24 25 26 27 28 29

    The structure reflects a set of groups that share identical dimension,
    coordinate and variable names which do not exist in the root group, so
    that there should be no problem with the cascading inheritance.

    The 'u','v' and 'p' variables in each group depend on their group-specific
    coordinates 't', 'x' and 'y', and dimensions 't', 'xi' and 'yi'. The 'cx'
    and 'cy' variables depend on only the 't' coordinate/dimension for the
    respective group. The 2D 'x' and 'y' variables could be treated like coordinates
    for the arrays that share their 'xi' and 'yi' dimensions.
    """

    def _create_test_multidataset_withoutroot_datatree(
        modify=lambda ds: ds,
    ):
        main_data = modify(
            xr.Dataset(
                {
                    "x": (["yi", "xi"], np.meshgrid(np.arange(60), np.arange(50))[1]),
                    "y": (["yi", "xi"], np.meshgrid(np.arange(60), np.arange(50))[0]),
                    "t": ("t", np.arange(30)),
                    "u": (["t", "yi", "xi"], np.ones((30, 50, 60))),
                    "v": (["t", "yi", "xi"], np.ones((30, 50, 60))),
                    "p": (["t", "yi", "xi"], np.ones((30, 50, 60))),
                }
            )
        )
        ovr1_data = modify(
            xr.Dataset(
                {
                    "x": (
                        ["yi", "xi"],
                        np.meshgrid(np.arange(51), np.arange(51) + 150.0)[1],
                    ),
                    "y": (["yi", "xi"], np.meshgrid(np.arange(51), np.arange(51))[0]),
                    "t": ("t", np.arange(10, 20)),
                    "u": (["t", "yi", "xi"], np.ones((10, 51, 51))),
                    "v": (["t", "yi", "xi"], np.ones((10, 51, 51))),
                    "p": (["t", "yi", "xi"], np.ones((10, 51, 51))),
                    "cx": ("t", np.arange(10, 20)),
                    "cy": ("t", np.arange(10, 20)),
                }
            )
        )
        ovr2_data = modify(
            xr.Dataset(
                {
                    "x": (["yi", "xi"], np.meshgrid(np.arange(51), np.arange(51))[1]),
                    "y": (["yi", "xi"], np.meshgrid(np.arange(51), np.arange(51))[0]),
                    "t": ("t", np.arange(10, 30)),
                    "u": (["t", "yi", "xi"], np.ones((20, 51, 51))),
                    "v": (["t", "yi", "xi"], np.ones((20, 51, 51))),
                    "p": (["t", "yi", "xi"], np.ones((20, 51, 51))),
                    "cx": ("t", np.arange(10, 30)),
                    "cy": ("t", np.arange(10, 30)),
                }
            )
        )
        root_data = modify(xr.Dataset())

        # Avoid using __init__ so we can independently test it
        root: DataTree = DataTree(data=root_data)
        DataTree(name="Main", parent=root, data=main_data)
        DataTree(name="ovr1", parent=root, data=ovr1_data)
        DataTree(name="ovr2", parent=root, data=ovr2_data)

        return root

    return _create_test_multidataset_withoutroot_datatree
