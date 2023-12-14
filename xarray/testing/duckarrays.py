from abc import abstractmethod
from typing import TYPE_CHECKING

import hypothesis.extra.numpy as npst
import hypothesis.strategies as st
import numpy as np
import numpy.testing as npt
import pytest
from hypothesis import given, note

import xarray as xr
import xarray.testing.strategies as xrst
from xarray.testing.assertions import assert_identical
from xarray.core.types import T_DuckArray

if TYPE_CHECKING:
    from xarray.core.types import _DTypeLikeNested, _ShapeLike


__all__ = [
    "VariableConstructorTests",
    "VariableReduceTests",
]


class ArrayConstructorChecksMixin:
    """Mixin for checking results of Variable/DataArray constructors."""

    def check(self, var, arr):
        self.check_types(var, arr)
        self.check_values(var, arr)
        self.check_attributes(var, arr)

    def check_types(self, var, arr):
        # test type of wrapped array
        assert isinstance(
            var.data, type(arr)
        ), f"found {type(var.data)}, expected {type(arr)}"

    def check_attributes(self, var, arr):
        # test ndarray attributes are exposed correctly
        assert var.ndim == arr.ndim
        assert var.shape == arr.shape
        assert var.dtype == arr.dtype
        assert var.size == arr.size

    def check_values(self, var, arr):
        # test coercion to numpy
        npt.assert_equal(var.to_numpy(), np.asarray(arr))


class VariableConstructorTests(ArrayConstructorChecksMixin):
    shapes = npst.array_shapes()
    dtypes = xrst.supported_dtypes()

    @staticmethod
    @abstractmethod
    def array_strategy_fn(
        *, shape: "_ShapeLike", dtype: "_DTypeLikeNested"
    ) -> st.SearchStrategy[T_DuckArray]:
        # TODO can we just make this an attribute?
        ...

    @given(st.data())
    def test_construct(self, data) -> None:
        shape = data.draw(self.shapes)
        dtype = data.draw(self.dtypes)
        arr = data.draw(self.array_strategy_fn(shape=shape, dtype=dtype))

        dim_names = data.draw(
            xrst.dimension_names(min_dims=len(shape), max_dims=len(shape))
        )
        var = xr.Variable(data=arr, dims=dim_names)

        self.check(var, arr)


class VariableReduceTests:
    dtypes = xrst.supported_dtypes()

    @staticmethod
    @abstractmethod
    def array_strategy_fn(
        *, shape: "_ShapeLike", dtype: "_DTypeLikeNested"
    ) -> st.SearchStrategy[T_DuckArray]:
        # TODO can we just make this an attribute?
        ...

    def check_reduce(self, var, op, dim, *args, **kwargs):
        actual = getattr(var, op)(dim=dim, *args, **kwargs)

        data = np.asarray(var.data)
        expected = getattr(var.copy(data=data), op)(*args, **kwargs)

        # create expected result (using nanmean because arrays with Nans will be generated)
        reduce_axes = tuple(var.get_axis_num(d) for d in dim)
        data = np.asarray(var.data)
        expected = getattr(var.copy(data=data), op)(*args, axis=reduce_axes, **kwargs)

        note(f"actual:\n{actual}")
        note(f"expected:\n{expected}")

        assert_identical(actual, expected)

    @pytest.mark.parametrize(
        "method",
        (
            "all",
            "any",
            # "cumprod",  # not in array API
            # "cumsum",  # not in array API
            # "max",  # only in array API for real numeric dtypes
            # "max",  # only in array API for real floating point dtypes
            # "median",  # not in array API
            # "min",  # only in array API for real numeric dtypes
            # "prod",  # only in array API for numeric dtypes
            # "std",  # TypeError: std() got an unexpected keyword argument 'ddof'
            # "sum",  # only in array API for numeric dtypes
            # "var",  # TypeError: std() got an unexpected keyword argument 'ddof'
        ),
    )
    @given(st.data())
    def test_reduce(self, method, data):
        """
        Test that the reduction applied to an xarray Variable is always equal
        to the same reduction applied to the underlying array.
        """

        var = data.draw(
            xrst.variables(
                array_strategy_fn=self.array_strategy_fn,
                dims=xrst.dimension_names(min_dims=1),
                dtype=self.dtypes,
            )
        )

        # specify arbitrary reduction along at least one dimension
        reduce_dims = data.draw(xrst.unique_subset_of(var.dims, min_size=1))

        self.check_reduce(var, method, dim=reduce_dims)
