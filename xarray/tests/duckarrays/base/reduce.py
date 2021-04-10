import hypothesis.strategies as st
import pytest
from hypothesis import given

import xarray as xr

from .utils import create_dimension_names, numpy_array, valid_axes, valid_dims_from_axes


class ReduceMethodTests:
    @staticmethod
    def create(op):
        return numpy_array

    @pytest.mark.parametrize(
        "method",
        (
            "all",
            "any",
            "argmax",
            "argmin",
            "argsort",
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
    def test_variable_reduce(self, method, data):
        raw = data.draw(self.create(method))
        dims = create_dimension_names(raw.ndim)
        var = xr.Variable(dims, raw)

        reduce_axes = data.draw(valid_axes(raw.ndim))
        reduce_dims = valid_dims_from_axes(dims, reduce_axes)

        actual = getattr(var, method)(dim=reduce_dims)
        print(actual)

        assert False
