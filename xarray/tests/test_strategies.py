import numpy as np
import numpy.testing as npt
import pytest

pytest.importorskip("hypothesis")
# isort: split

import hypothesis.extra.numpy as npst
import hypothesis.strategies as st
from hypothesis import given
from hypothesis.errors import Unsatisfiable

from xarray import DataArray, Dataset
from xarray.core.variable import Variable
from xarray.testing.strategies import (
    coordinate_variables,
    data_variables,
    dataarrays,
    datasets,
    dimension_names,
    dimension_sizes,
    np_arrays,
    valid_dtypes,
    variables,
)


class TestNumpyArraysStrategy:
    @given(np_arrays())
    def test_given_nothing(self, arr):
        assert isinstance(arr, np.ndarray)

    @given(np_arrays(dtype=np.dtype("int32")))
    def test_fixed_dtype(self, arr):
        assert arr.dtype == np.dtype("int32")

    @given(st.data())
    def test_arbitrary_valid_dtype(self, data):
        valid_dtype = data.draw(valid_dtypes)
        arr = data.draw(np_arrays(dtype=valid_dtype))
        assert arr.dtype == valid_dtype

    @given(np_arrays(shape=(2, 3)))
    def test_fixed_shape(self, arr):
        assert arr.shape == (2, 3)

    @given(st.data())
    def test_arbitrary_shape(self, data):
        shape = data.draw(npst.array_shapes())
        arr = data.draw(np_arrays(shape=shape))
        assert arr.shape == shape


class TestDimensionNamesStrategy:
    @given(dimension_names())
    def test_types(self, dims):
        assert isinstance(dims, list)
        for d in dims:
            assert isinstance(d, str)

    @given(dimension_names())
    def test_unique(self, dims):
        assert len(set(dims)) == len(dims)

    @given(dimension_names(min_dims=3, max_dims=3))
    def test_fixed_number_of_dims(self, dims):
        assert isinstance(dims, list)
        assert len(dims) == 3


class TestDimensionSizesStrategy:
    @given(dimension_sizes())
    def test_types(self, dims):
        assert isinstance(dims, dict)
        for d, n in dims.items():
            assert isinstance(d, str)
            assert isinstance(n, int)

    @given(dimension_sizes(min_dims=3, max_dims=3))
    def test_fixed_number_of_dims(self, dims):
        assert isinstance(dims, dict)
        assert len(dims) == 3


class TestVariablesStrategy:
    @given(variables())
    def test_given_nothing(self, var):
        assert isinstance(var, Variable)

    @given(st.data())
    def test_given_fixed_dims_list_and_fixed_data(self, data):
        dims = ["x", "y"]
        arr = np.asarray([[1, 2], [3, 4]])
        var = data.draw(variables(dims=st.just(dims), data=st.just(arr)))

        assert list(var.dims) == dims
        npt.assert_equal(var.data, arr)

    @given(st.data())
    def test_given_arbitrary_dims_list_and_arbitrary_data(self, data):
        arrs = np_arrays(shape=(2, 3))
        dims = dimension_names(min_dims=2, max_dims=2)
        var = data.draw(variables(data=arrs, dims=dims))
        assert var.shape == (2, 3)

        dims = dimension_names(min_dims=3)
        with pytest.raises(Unsatisfiable):
            data.draw(variables(data=arrs, dims=dims))

    @given(st.data())
    def test_given_fixed_data(self, data):
        arr = np.asarray([[1, 2], [3, 4]])
        var = data.draw(variables(data=st.just(arr)))

        npt.assert_equal(var.data, arr)

    @given(st.data())
    def test_given_arbitrary_data(self, data):
        shape = (2, 3)
        arrs = np_arrays(shape=shape)
        var = data.draw(variables(data=arrs))

        assert var.data.shape == shape

    @given(st.data())
    def test_given_fixed_dims_list(self, data):
        dims = ["x", "y"]
        var = data.draw(variables(dims=st.just(dims)))

        assert list(var.dims) == dims

    @given(st.data())
    def test_given_arbitrary_dims_list(self, data):
        dims = dimension_names(min_dims=1, max_dims=1)
        var = data.draw(variables(dims=dims))

        assert len(list(var.dims)) == 1

    @given(st.data())
    def test_given_fixed_sizes(self, data):
        dims = {"x": 3, "y": 4}
        var = data.draw(variables(dims=st.just(dims)))

        assert var.dims == ("x", "y")
        assert var.shape == (3, 4)

    @given(st.data())
    def test_given_fixed_sizes_and_arbitrary_data(self, data):
        arrs = np_arrays(shape=(2, 3))
        var = data.draw(variables(data=arrs, dims=st.just({"x": 2, "y": 3})))

        assert var.shape == (2, 3)


class TestCoordinateVariablesStrategy:
    @given(coordinate_variables(dim_sizes={"x": 2, "y": 3}))
    def test_alignable(self, coord_vars):

        # TODO there must be a better way of checking align-ability than this
        for v in coord_vars.values():
            if "x" in v.dims:
                assert v.sizes["x"] == 2
            if "y" in v.dims:
                assert v.sizes["y"] == 3
            if not set(v.dims).issubset({"x", "y"}):
                assert False, v

    @given(st.data())
    def test_valid_set_of_coords(self, data):
        coord_vars = data.draw(coordinate_variables(dim_sizes={"x": 2, "y": 3}))

        arr = data.draw(np_arrays(shape=(2, 3)))
        da = DataArray(data=arr, coords=coord_vars, dims=["x", "y"])
        assert isinstance(da, DataArray)

    def test_generates_1d_dim_coords(self):
        # TODO having a `hypothesis.find(strat, predicate)` function would be very useful here
        # see https://github.com/HypothesisWorks/hypothesis/issues/3436#issuecomment-1212369645
        ...

    def test_generates_non_dim_coords(self):
        ...


class TestDataArraysStrategy:
    @given(dataarrays())
    def test_given_nothing(self, da):
        assert isinstance(da, DataArray)

    @given(st.data())
    def test_given_dims(self, data):
        da = data.draw(dataarrays(dims=st.just(["x", "y"])))
        assert da.dims == ("x", "y")

        da = data.draw(dataarrays(dims=st.just({"x": 2, "y": 3})))
        assert da.sizes == {"x": 2, "y": 3}

    @given(st.data())
    def test_given_data(self, data):
        shape = (2, 3)
        arrs = np_arrays(shape=shape)
        da = data.draw(dataarrays(data=arrs))

        assert da.shape == shape

    @given(st.data())
    def test_given_data_and_dims(self, data):
        arrs = np_arrays(shape=(2, 3))
        dims = dimension_names(min_dims=2, max_dims=2)
        da = data.draw(dataarrays(data=arrs, dims=dims))
        assert da.shape == (2, 3)

        dims = dimension_names(min_dims=3, max_dims=3)
        with pytest.raises(Unsatisfiable):
            data.draw(dataarrays(data=arrs, dims=dims))

        arrs = np_arrays(shape=(3, 4))
        dims = st.just({"x": 3, "y": 4})
        da = data.draw(dataarrays(data=arrs, dims=dims))
        assert da.sizes == {"x": 3, "y": 4}


class TestDatasetsStrategy:
    @given(datasets())
    def test_given_nothing(self, ds):
        assert isinstance(ds, Dataset)
