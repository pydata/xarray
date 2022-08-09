import hypothesis.strategies as st
import numpy as np
import numpy.testing as npt
from hypothesis import given, note, settings


from . import strategies

import xarray as xr


class VariableConstructorTests:
    def check(self, var, arr):
        self.check_types(var, arr)
        self.check_values(var, arr)
        self.check_attributes(var, arr)

    def check_types(self, var, arr):
        # test type of wrapped array
        assert isinstance(var.data, type(arr)), f"found {type(var.data)}, expected {type(arr)}"

    def check_attributes(self, var, arr):
        # test ndarry attributes are exposed correctly
        assert var.ndim == arr.ndim
        assert var.shape == arr.shape
        assert var.dtype == arr.dtype
        assert var.size == arr.size
        assert var.nbytes == arr.nbytes

    def check_values(self, var, arr):
        # test coersion to numpy
        npt.assert_equal(var.to_numpy(), np.asarray(arr))

    @staticmethod
    def create(shape, dtypes):
        return strategies.numpy_array(shape, dtypes)

    @given(st.data())
    @settings(deadline=None)
    def test_construct(self, data):
        arr = data.draw(
            strategies.duckarray(lambda shape, dtypes: self.create(shape, dtypes))
        )

        # TODO generalize to N dimensions
        dims = ["dim_0", "dim_1", "dim_2"]

        var = xr.Variable(dims=dims[0:arr.ndim], data=arr)

        self.check(var, arr)


class DataArrayConstructorTests:
    ...


class DatasetConstructorTests:
    ...
