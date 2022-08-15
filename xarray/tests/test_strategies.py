import hypothesis.extra.numpy as npst
import hypothesis.strategies as st
import numpy as np
import numpy.testing as npt
import pytest
from hypothesis import given

from xarray import DataArray, Dataset
from xarray.core.variable import Variable
from xarray.testing.strategies import (
    dataarrays,
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

    @given(dimension_names(min_ndims=3, max_ndims=3))
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

    @given(dimension_sizes(min_ndims=3, max_ndims=3))
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
        dims = dimension_names(min_ndims=2)
        var = data.draw(variables(data=arrs, dims=dims))

        assert var.shape == (2, 3)

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
        dims = dimension_names(min_ndims=1, max_ndims=1)
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

    @given(st.data())
    def test_convert(self, data):
        arr = st.just(np.asarray([1, 2, 3]))
        var = data.draw(variables(data=arr, convert=lambda x: x * 2))

        npt.assert_equal(var.data, np.asarray([2, 4, 6]))


@pytest.mark.xfail
class TestDataArraysStrategy:
    @given(dataarrays())
    def test_given_nothing(self, da):
        print(da)
        assert isinstance(da, DataArray)
        assert False


@pytest.mark.xfail
@given(st.data())
def test_chained_chunking_example(data):
    import dask.array.strategies as dast

    def chunk(da):
        return da.chunk(dast.chunks(da.shape))

    chunked_dataarrays = xrst.dataarrays().flatmap(chunk)

    chunked_da = data.draw(chunked_dataarrays())

    assert ...
