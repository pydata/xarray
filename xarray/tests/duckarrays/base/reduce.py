import hypothesis.strategies as st
import numpy as np
import pytest
from hypothesis import given, note

import xarray as xr

from ... import assert_identical
from . import strategies
from .utils import valid_dims_from_axes


class VariableReduceTests:
    def check_reduce(self, obj, op, *args, **kwargs):
        actual = getattr(obj, op)(*args, **kwargs)

        data = np.asarray(obj.data)
        expected = getattr(obj.copy(data=data), op)(*args, **kwargs)

        note(f"actual:\n{actual}")
        note(f"expected:\n{expected}")

        assert_identical(actual, expected)

    @staticmethod
    def create(op, shape):
        return strategies.numpy_array(shape)

    @pytest.mark.parametrize(
        "method",
        (
            "all",
            "any",
            "cumprod",
            "cumsum",
            "max",
            "mean",
            "median",
            "min",
            "prod",
            "std",
            "sum",
            "var",
        ),
    )
    @given(st.data())
    def test_reduce(self, method, data):
        var = data.draw(strategies.variable(lambda shape: self.create(method, shape)))

        reduce_axes = data.draw(strategies.valid_axis(var.ndim))
        reduce_dims = valid_dims_from_axes(var.dims, reduce_axes)

        self.check_reduce(var, method, dim=reduce_dims)


class DataArrayReduceTests:
    def check_reduce(self, obj, op, *args, **kwargs):
        actual = getattr(obj, op)(*args, **kwargs)

        data = np.asarray(obj.data)
        expected = getattr(obj.copy(data=data), op)(*args, **kwargs)

        note(f"actual:\n{actual}")
        note(f"expected:\n{expected}")

        assert_identical(actual, expected)

    @staticmethod
    def create(op, shape):
        return strategies.numpy_array(shape)

    @pytest.mark.parametrize(
        "method",
        (
            "all",
            "any",
            "cumprod",
            "cumsum",
            "max",
            "mean",
            "median",
            "min",
            "prod",
            "std",
            "sum",
            "var",
        ),
    )
    @given(st.data())
    def test_reduce(self, method, data):
        raw = data.draw(self.create(method))
        dims = strategies.create_dimension_names(raw.ndim)
        arr = xr.DataArray(dims=dims, data=raw)

        reduce_axes = data.draw(strategies.valid_axis(raw.ndim))
        reduce_dims = valid_dims_from_axes(dims, reduce_axes)

        self.check_reduce(arr, method, dim=reduce_dims)
