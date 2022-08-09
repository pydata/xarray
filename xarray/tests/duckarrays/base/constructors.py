import hypothesis.strategies as st
import numpy as np
import numpy.testing as npt
from hypothesis import given, settings

import xarray as xr

from . import strategies
from .utils import create_dimension_names


class ArrayConstructorChecks:
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
        assert var.nbytes == arr.nbytes

    def check_values(self, var, arr):
        # test coercion to numpy
        npt.assert_equal(var.to_numpy(), np.asarray(arr))


class VariableConstructorTests(ArrayConstructorChecks):
    @staticmethod
    def create(shape, dtypes):
        return strategies.numpy_array(shape, dtypes)

    @given(st.data())
    @settings(deadline=None)
    def test_construct(self, data):
        arr = data.draw(
            strategies.duckarray(lambda shape, dtypes: self.create(shape, dtypes))
        )

        var = xr.Variable(dims=create_dimension_names(arr.ndim), data=arr)

        self.check(var, arr)


class DataArrayConstructorTests(ArrayConstructorChecks):
    @staticmethod
    def create(shape, dtypes):
        return strategies.numpy_array(shape, dtypes)

    @given(st.data())
    @settings(deadline=None)
    def test_construct(self, data):
        arr = data.draw(
            strategies.duckarray(lambda shape, dtypes: self.create(shape, dtypes))
        )

        da = xr.DataArray(
            name=data.draw(st.none() | st.text(min_size=1)),
            dims=create_dimension_names(arr.ndim),
            data=arr,
        )

        self.check(da, arr)


class DatasetConstructorTests:
    # TODO can we re-use the strategy for building datasets?
    ...
