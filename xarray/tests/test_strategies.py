import numpy as np
import numpy.testing as npt
import pytest

pytest.importorskip("hypothesis")
# isort: split

import hypothesis.extra.numpy as npst
import hypothesis.strategies as st
from hypothesis import given
from hypothesis.extra.array_api import make_strategies_namespace

from xarray.core.variable import Variable
from xarray.testing.strategies import (
    attrs,
    dimension_names,
    dimension_sizes,
    supported_dtypes,
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

    @given(st.data())
    def test_restrict_names(self, data):
        capitalized_names = st.text(st.characters(), min_size=1).map(str.upper)
        dim_sizes = data.draw(dimension_sizes(dim_names=capitalized_names))
        for dim in dim_sizes.keys():
            assert dim.upper() == dim


def check_dict_values(dictionary: dict) -> bool:
    for key, value in dictionary.items():
        if isinstance(value, ALLOWED_ATTRS_VALUES_TYPES) or value is None:
            continue
        elif isinstance(value, dict):
            # If the value is a dictionary, recursively check it
            if not check_dict_values(value):
                return False
        else:
            # If the value is not an integer or a dictionary, it's not valid
            return False
    return True


class TestAttrsStrategy:
    @given(attrs())
    def test_type(self, attrs):
        assert isinstance(attrs, dict)
        check_dict_values(attrs)


class TestVariablesStrategy:
    @given(variables())
    def test_given_nothing(self, var):
        assert isinstance(var, Variable)

    @given(st.data())
    def test_given_incorrect_types(self, data):
        with pytest.raises(TypeError, match="SearchStrategy object"):
            data.draw(variables(dims=["x", "y"]))  # type: ignore[arg-type]

        with pytest.raises(TypeError, match="SearchStrategy object"):
            data.draw(variables(dtype=np.dtype("int32")))  # type: ignore[arg-type]

        with pytest.raises(TypeError, match="SearchStrategy object"):
            data.draw(variables(attrs=dict()))  # type: ignore[arg-type]

        with pytest.raises(TypeError, match="Callable"):
            data.draw(variables(array_strategy_fn=np.array([0])))  # type: ignore[arg-type]

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
        var = data.draw(variables(dims=st.just(dims)))  # type: ignore[arg-type]

        assert var.dims == ("x", "y")
        assert var.shape == (3, 4)

    @given(st.data())
    def test_given_fixed_dtype(self, data):
        var = data.draw(variables(dtype=st.just(np.dtype("int32"))))

        assert var.dtype == np.dtype("int32")

    @given(st.data())
    def test_given_fixed_data(self, data):
        arr = np.asarray([[1, 2], [3, 4]])

        def fixed_array_strategy_fn(*, shape=None, dtype=None):
            return st.just(arr)

        var = data.draw(
            variables(
                array_strategy_fn=fixed_array_strategy_fn, dims=st.just({"x": 2, "y": 2}), dtype=st.just(arr.dtype)  # type: ignore[arg-type]
            )
        )

        npt.assert_equal(var.data, arr)
        assert var.dtype == arr.dtype

    @given(st.data())
    def test_given_fixed_dims_and_fixed_data(self, data):
        dims = {"x": 2, "y": 2}
        arr = np.asarray([[1, 2], [3, 4]])

        def fixed_array_strategy_fn(*, shape=None, dtype=None):
            return st.just(arr)

        var = data.draw(
            variables(
                array_strategy_fn=fixed_array_strategy_fn,
                dims=st.just(dims),  # type: ignore[arg-type]
                dtype=st.just(arr.dtype),
            )
        )

        assert var.sizes == dims
        npt.assert_equal(var.data, arr)

    @given(st.data())
    def test_given_fixed_shape_arbitrary_dims_and_arbitrary_data(self, data):
        dims = dimension_names(min_dims=2, max_dims=2)

        def array_strategy_fn(*, shape=None, dtype=None):
            return npst.arrays(shape=shape, dtype=dtype)

        var = data.draw(
            variables(
                array_strategy_fn=array_strategy_fn,
                dims=dims,
                dtype=supported_dtypes(),
            )
        )

        assert var.ndim == 2

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
