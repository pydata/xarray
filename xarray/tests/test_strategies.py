import hypothesis.extra.numpy as npst
import hypothesis.strategies as st
import numpy as np
import numpy.testing as npt
import pytest
from hypothesis import given

from xarray import DataArray, Dataset, merge
from xarray.core.variable import Variable
from xarray.testing.strategies import (
    coordinate_variables,
    data_variables,
    dataarrays,
    datasets,
    dimension_names,
    dimension_sizes,
    np_arrays,
    subsequences_of,
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


# All from the unfinished PR https://github.com/HypothesisWorks/hypothesis/pull/1533
class TestSubsequencesOfStrategy:
    @pytest.mark.xfail(
        reason="Can't work out how to import assert_no_examples from hypothesis.tests.common.debug"
    )
    def test_subsequence_of_empty(self):
        sub_seq_strat = st.lists(st.none(), max_size=0)
        assert_no_examples(sub_seq_strat)

    @given(st.data(), st.lists(st.integers()))
    def test_subsequence_sizing(self, data, seq):
        sub_seq_strat = subsequences_of(seq)
        sub_seq = data.draw(sub_seq_strat)

        assert isinstance(sub_seq, list)
        assert len(sub_seq) <= len(seq)

    @given(st.data(), st.lists(st.integers()))
    def test_subsequence_only_original_elements(self, data, seq):
        sub_seq_strat = subsequences_of(seq)
        sub_seq = data.draw(sub_seq_strat)

        assert isinstance(sub_seq, list)
        assert len(sub_seq) <= len(seq)

    @given(st.data(), st.lists(st.integers()))
    def test_subsequence_elements_not_over_drawn(self, data, seq):
        sub_seq_strat = subsequences_of(seq)
        sub_seq = data.draw(sub_seq_strat)

        assert not (set(sub_seq) - set(seq))

    @given(st.data(), st.lists(st.integers()))
    def test_subsequence_original_elements_not_over_produced(self, data, seq):
        sub_seq_strat = subsequences_of(seq)
        sub_seq = data.draw(sub_seq_strat)

        # Per unique item, check that they don't occur in the subsequence
        # more times that they appear in the source.
        for item in set(sub_seq):
            assert sub_seq.count(item) <= seq.count(item)

    @given(st.data(), st.lists(st.integers()))
    def test_subsequence_max_size_constraint(self, data, seq):
        max_size_strat = st.integers(min_value=0, max_value=len(seq))
        max_size = data.draw(max_size_strat)

        sub_seq_strat = subsequences_of(seq, max_size=max_size)
        sub_seq = data.draw(sub_seq_strat)

        assert len(sub_seq) <= max_size

    @given(st.data(), st.lists(st.integers()))
    def test_subsequence_min_size_constraint(self, data, seq):
        min_size_strat = st.integers(min_value=0, max_value=len(seq))
        min_size = data.draw(min_size_strat)

        sub_seq_strat = subsequences_of(seq, min_size=min_size)
        sub_seq = data.draw(sub_seq_strat)

        assert len(sub_seq) >= min_size

    @given(st.data(), st.lists(st.integers()))
    def test_subsequence_min_max_size_constraint(self, data, seq):
        min_size_strat = st.integers(min_value=0, max_value=len(seq))
        min_size = data.draw(min_size_strat)

        max_size_strat = st.integers(min_value=min_size, max_value=len(seq))
        max_size = data.draw(max_size_strat)

        sub_seq_strat = subsequences_of(seq, min_size=min_size, max_size=max_size)
        sub_seq = data.draw(sub_seq_strat)

        assert min_size <= len(sub_seq) <= max_size

    # this is a new test, important for keeping dimension names in order
    @given(st.data(), st.lists(st.integers()))
    def test_ordering_preserved(self, data, seq):
        subsequence_of_dims = data.draw(subsequences_of(seq))
        assert sorted(subsequence_of_dims) == subsequence_of_dims


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


@pytest.mark.xfail
@given(st.data())
def test_chained_chunking_example(data):
    import dask.array.strategies as dast

    def chunk(da):
        return da.chunk(dast.chunks(da.shape))

    chunked_dataarrays = xrst.dataarrays().flatmap(chunk)

    chunked_da = data.draw(chunked_dataarrays())

    assert ...
