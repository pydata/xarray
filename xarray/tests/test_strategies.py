import contextlib

import numpy as np
import numpy.testing as npt
import pytest

pytest.importorskip("hypothesis")
# isort: split

import hypothesis.extra.numpy as npst
import hypothesis.strategies as st
from hypothesis import Phase, given, settings
from hypothesis.errors import InvalidArgument
from hypothesis.extra.array_api import make_strategies_namespace

from xarray import DataArray, Dataset
from xarray.core.variable import Variable
from xarray.testing.strategies import (
    attrs,
    coordinate_variables,
    data_variables,
    dataarrays,
    datasets,
    dimension_names,
    dimension_sizes,
    supported_dtypes,
    unique_subset_of,
    variables,
)
from xarray.tests import requires_numpy_array_api

ALLOWED_ATTRS_VALUES_TYPES = (int, bool, str, np.ndarray)


class TestDimensionNamesStrategy:
    @given(dimension_names())
    def test_types(self, dims):
        assert isinstance(dims, list)
        for d in dims:
            assert isinstance(d, str)

    @given(dimension_names())
    def test_unique(self, dims):
        assert len(set(dims)) == len(dims)

    @given(st.data(), st.tuples(st.integers(0, 10), st.integers(0, 10)).map(sorted))
    def test_number_of_dims(self, data, ndims):
        min_dims, max_dims = ndims
        dim_names = data.draw(dimension_names(min_dims=min_dims, max_dims=max_dims))
        assert isinstance(dim_names, list)
        assert min_dims <= len(dim_names) <= max_dims


class TestDimensionSizesStrategy:
    @given(dimension_sizes())
    def test_types(self, dims):
        assert isinstance(dims, dict)
        for d, n in dims.items():
            assert isinstance(d, str)
            assert len(d) >= 1
            assert isinstance(n, int)
            assert n >= 0

    @given(st.data(), st.tuples(st.integers(0, 10), st.integers(0, 10)).map(sorted))
    def test_number_of_dims(self, data, ndims):
        min_dims, max_dims = ndims
        dim_sizes = data.draw(dimension_sizes(min_dims=min_dims, max_dims=max_dims))
        assert isinstance(dim_sizes, dict)
        assert min_dims <= len(dim_sizes) <= max_dims

    @given(st.data())
    def test_restrict_names(self, data):
        capitalized_names = st.text(st.characters(), min_size=1).map(str.upper)
        dim_sizes = data.draw(dimension_sizes(dim_names=capitalized_names))
        for dim in dim_sizes.keys():
            assert dim.upper() == dim


def check_dict_values(dictionary: dict, allowed_attrs_values_types) -> bool:
    """Helper function to assert that all values in recursive dict match one of a set of types."""
    for key, value in dictionary.items():
        if isinstance(value, allowed_attrs_values_types) or value is None:
            continue
        elif isinstance(value, dict):
            # If the value is a dictionary, recursively check it
            if not check_dict_values(value, allowed_attrs_values_types):
                return False
        else:
            # If the value is not an integer or a dictionary, it's not valid
            return False
    return True


class TestAttrsStrategy:
    @given(attrs())
    def test_type(self, attrs):
        assert isinstance(attrs, dict)
        check_dict_values(attrs, ALLOWED_ATTRS_VALUES_TYPES)


class TestVariablesStrategy:
    @given(variables())
    def test_given_nothing(self, var):
        assert isinstance(var, Variable)

    @given(st.data())
    def test_given_incorrect_types(self, data):
        with pytest.raises(TypeError, match="dims must be provided as a"):
            data.draw(variables(dims=["x", "y"]))  # type: ignore[arg-type]

        with pytest.raises(TypeError, match="dtype must be provided as a"):
            data.draw(variables(dtype=np.dtype("int32")))  # type: ignore[arg-type]

        with pytest.raises(TypeError, match="attrs must be provided as a"):
            data.draw(variables(attrs=dict()))  # type: ignore[arg-type]

        with pytest.raises(TypeError, match="Callable"):
            data.draw(variables(array_strategy_fn=np.array([0])))  # type: ignore[arg-type]

    @given(st.data(), dimension_names())
    def test_given_fixed_dim_names(self, data, fixed_dim_names):
        var = data.draw(variables(dims=st.just(fixed_dim_names)))

        assert list(var.dims) == fixed_dim_names

    @given(st.data(), dimension_sizes())
    def test_given_fixed_dim_sizes(self, data, dim_sizes):
        var = data.draw(variables(dims=st.just(dim_sizes)))

        assert var.dims == tuple(dim_sizes.keys())
        assert var.shape == tuple(dim_sizes.values())

    @given(st.data(), supported_dtypes())
    def test_given_fixed_dtype(self, data, dtype):
        var = data.draw(variables(dtype=st.just(dtype)))

        assert var.dtype == dtype

    @given(st.data(), npst.arrays(shape=npst.array_shapes(), dtype=supported_dtypes()))
    def test_given_fixed_data_dims_and_dtype(self, data, arr):
        def fixed_array_strategy_fn(*, shape=None, dtype=None):
            """The fact this ignores shape and dtype is only okay because compatible shape & dtype will be passed separately."""
            return st.just(arr)

        dim_names = data.draw(dimension_names(min_dims=arr.ndim, max_dims=arr.ndim))
        dim_sizes = {name: size for name, size in zip(dim_names, arr.shape)}

        var = data.draw(
            variables(
                array_strategy_fn=fixed_array_strategy_fn,
                dims=st.just(dim_sizes),
                dtype=st.just(arr.dtype),
            )
        )

        npt.assert_equal(var.data, arr)
        assert var.dtype == arr.dtype

    @given(st.data(), st.integers(0, 3))
    def test_given_array_strat_arbitrary_size_and_arbitrary_data(self, data, ndims):
        dim_names = data.draw(dimension_names(min_dims=ndims, max_dims=ndims))

        def array_strategy_fn(*, shape=None, dtype=None):
            return npst.arrays(shape=shape, dtype=dtype)

        var = data.draw(
            variables(
                array_strategy_fn=array_strategy_fn,
                dims=st.just(dim_names),
                dtype=supported_dtypes(),
            )
        )

        assert var.ndim == ndims

    @given(st.data())
    def test_catch_unruly_dtype_from_custom_array_strategy_fn(self, data):
        def dodgy_array_strategy_fn(*, shape=None, dtype=None):
            """Dodgy function which ignores the dtype it was passed"""
            return npst.arrays(shape=shape, dtype=npst.floating_dtypes())

        with pytest.raises(
            ValueError, match="returned an array object with a different dtype"
        ):
            data.draw(
                variables(
                    array_strategy_fn=dodgy_array_strategy_fn,
                    dtype=st.just(np.dtype("int32")),
                )
            )

    @given(st.data())
    def test_catch_unruly_shape_from_custom_array_strategy_fn(self, data):
        def dodgy_array_strategy_fn(*, shape=None, dtype=None):
            """Dodgy function which ignores the shape it was passed"""
            return npst.arrays(shape=(3, 2), dtype=dtype)

        with pytest.raises(
            ValueError, match="returned an array object with a different shape"
        ):
            data.draw(
                variables(
                    array_strategy_fn=dodgy_array_strategy_fn,
                    dims=st.just({"a": 2, "b": 1}),
                    dtype=supported_dtypes(),
                )
            )

    @requires_numpy_array_api
    @given(st.data())
    def test_make_strategies_namespace(self, data):
        """
        Test not causing a hypothesis.InvalidArgument by generating a dtype that's not in the array API.

        We still want to generate dtypes not in the array API by default, but this checks we don't accidentally override
        the user's choice of dtypes with non-API-compliant ones.
        """
        from numpy import (
            array_api as np_array_api,  # requires numpy>=1.26.0, and we expect a UserWarning to be raised
        )

        np_array_api_st = make_strategies_namespace(np_array_api)

        data.draw(
            variables(
                array_strategy_fn=np_array_api_st.arrays,
                dtype=np_array_api_st.scalar_dtypes(),
            )
        )


class TestUniqueSubsetOf:
    @given(st.data())
    def test_invalid(self, data):
        with pytest.raises(TypeError, match="must be an Iterable or a Mapping"):
            data.draw(unique_subset_of(0))  # type: ignore[call-overload]

        with pytest.raises(ValueError, match="length-zero object"):
            data.draw(unique_subset_of({}))

    @given(st.data(), dimension_sizes(min_dims=1))
    def test_mapping(self, data, dim_sizes):
        subset_of_dim_sizes = data.draw(unique_subset_of(dim_sizes))

        for dim, length in subset_of_dim_sizes.items():
            assert dim in dim_sizes
            assert dim_sizes[dim] == length

    @given(st.data(), dimension_names(min_dims=1))
    def test_iterable(self, data, dim_names):
        subset_of_dim_names = data.draw(unique_subset_of(dim_names))

        for dim in subset_of_dim_names:
            assert dim in dim_names


class TestReduction:
    """
    These tests are for checking that the examples given in the docs page on testing actually work.
    """

    @given(st.data(), variables(dims=dimension_names(min_dims=1)))
    def test_mean(self, data, var):
        """
        Test that given a Variable of at least one dimension,
        the mean of the Variable is always equal to the mean of the underlying array.
        """

        # specify arbitrary reduction along at least one dimension
        reduction_dims = data.draw(unique_subset_of(var.dims, min_size=1))

        # create expected result (using nanmean because arrays with Nans will be generated)
        reduction_axes = tuple(var.get_axis_num(dim) for dim in reduction_dims)
        expected = np.nanmean(var.data, axis=reduction_axes)

        # assert property is always satisfied
        result = var.mean(dim=reduction_dims).data
        npt.assert_equal(expected, result)


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

    def test_sometimes_generates_1d_dim_coords(self):
        found_one = False

        @given(st.data())
        @settings(phases=[Phase.generate])
        def inner(data):
            coord_vars = data.draw(coordinate_variables(dim_sizes={"x": 2, "y": 3}))
            for name, var in coord_vars.items():
                if var.ndim == 1 and name == var.dims[0]:
                    nonlocal found_one
                    found_one = True
                    raise AssertionError  # early stopping - test is correct but slower without this

        with contextlib.suppress(AssertionError):
            inner()

        assert found_one

    def test_sometimes_generates_non_dim_coords(self):
        found_one = False

        @given(st.data())
        @settings(phases=[Phase.generate])
        def inner(data):
            coord_vars = data.draw(coordinate_variables(dim_sizes={"x": 2, "y": 3}))
            for name, var in coord_vars.items():
                if var.ndim != 1 or (var.ndim == 1 and name != var.dims[0]):
                    nonlocal found_one
                    found_one = True
                    raise AssertionError  # early stopping - test is correct but slower without this

        with contextlib.suppress(AssertionError):
            inner()

        assert found_one

    @given(st.data())
    def test_restrict_names(self, data):
        capitalized_names = st.text(st.characters(), min_size=1).map(str.upper)
        coord_vars = data.draw(
            coordinate_variables(
                dim_sizes={"x": 2, "y": 3}, coord_names=capitalized_names
            )
        )
        for name in coord_vars.keys():
            if name not in ["x", "y"]:
                assert name.upper() == name


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
        with pytest.raises(InvalidArgument):
            data.draw(dataarrays(data=arrs, dims=dims))

        arrs = np_arrays(shape=(3, 4))
        dims = st.just({"x": 3, "y": 4})
        da = data.draw(dataarrays(data=arrs, dims=dims))
        assert da.sizes == {"x": 3, "y": 4}


class TestDataVariablesStrategy:
    @given(st.data())
    def test_given_only_sizes(self, data):
        dim_sizes = {"x": 2, "y": 3}
        data_vars = data.draw(data_variables(dim_sizes=dim_sizes))
        for k, v in data_vars.items():
            assert isinstance(v, Variable)
            assert set(v.sizes.items()).issubset(set(dim_sizes.items()))

    @given(st.data())
    def test_restrict_names(self, data):
        capitalized_names = st.text(st.characters(), min_size=1).map(str.upper)
        data_vars = data.draw(
            data_variables(dim_sizes={"x": 2, "y": 3}, var_names=capitalized_names)
        )
        for name in data_vars.keys():
            assert name.upper() == name


class TestDatasetsStrategy:
    @given(datasets())
    def test_given_nothing(self, ds):
        assert isinstance(ds, Dataset)

    @given(st.data())
    def test_given_data(self, data):
        dim_sizes = {"x": 3, "y": 4}
        data_vars = data.draw(data_variables(dim_sizes=dim_sizes))
        ds = data.draw(datasets(data_vars=st.just(data_vars)))
        assert set(ds.sizes.items()).issubset(set(dim_sizes.items()))

    @given(st.data())
    def test_given_dims(self, data):
        dims = ["x", "y"]
        ds = data.draw(datasets(dims=st.just(dims)))
        assert set(ds.dims).issubset(set(dims))

        dim_sizes = {"x": 3, "y": 4}
        ds = data.draw(datasets(dims=st.just(dim_sizes)))
        assert set(ds.sizes.items()).issubset(set(dim_sizes.items()))

    @given(st.data())
    def test_given_data_and_dims(self, data):
        # pass dims as mapping
        dim_sizes = {"x": 3, "y": 4}
        data_vars = data.draw(data_variables(dim_sizes=dim_sizes))
        ds = data.draw(datasets(data_vars=st.just(data_vars), dims=st.just(dim_sizes)))
        assert set(ds.sizes.items()).issubset(set(dim_sizes.items()))

        incompatible_dim_sizes = {"x": 1, "y": 4}
        data_vars = {"foo": Variable(data=[0, 1, 2], dims="x")}
        with pytest.raises(InvalidArgument, match="drawn variable"):
            data.draw(
                datasets(
                    data_vars=st.just(data_vars), dims=st.just(incompatible_dim_sizes)
                )
            )

    @pytest.mark.xfail(reason="not implemented")
    @given(st.data())
    def test_given_data_and_dims_as_sequence(self, data):
        # pass dims as sequence
        dim_sizes = {"x": 3, "y": 4}
        dims = list(dim_sizes.keys())
        data_vars = data.draw(data_variables(dim_sizes=dim_sizes))
        ds = data.draw(datasets(data_vars=st.just(data_vars), dims=st.just(dims)))
        assert set(ds.sizes.items()).issubset(set(dim_sizes.items()))
