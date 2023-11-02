import numpy as np
import numpy.testing as npt
import pytest

pytest.importorskip("hypothesis")
# isort: split

import hypothesis.extra.numpy as npst
import hypothesis.strategies as st
from hypothesis import given

from xarray.core.variable import Variable
from xarray.testing.strategies import (
    attrs,
    dimension_names,
    dimension_sizes,
    np_arrays,
    numeric_dtypes,
    variables,
)

ALLOWED_ATTRS_VALUES_TYPES = (int, bool, str, np.ndarray)


class TestNumpyArraysStrategy:
    @given(np_arrays())
    def test_given_nothing(self, arr):
        assert isinstance(arr, np.ndarray)

    @given(np_arrays(dtype=np.dtype("int32")))
    def test_fixed_dtype(self, arr):
        assert arr.dtype == np.dtype("int32")

    @given(st.data())
    def test_arbitrary_valid_dtype(self, data):
        valid_dtype = data.draw(numeric_dtypes())
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
                array_strategy_fn=fixed_array_strategy_fn, dtype=st.just(arr.dtype)  # type: ignore[arg-type]
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

        def fixed_shape_array_strategy_fn(*, shape=None, dtype=None):
            return np_arrays(shape=shape, dtype=dtype)

        var = data.draw(
            variables(
                array_strategy_fn=fixed_shape_array_strategy_fn,
                dims=dims,
            )
        )

        assert var.ndim == 2
