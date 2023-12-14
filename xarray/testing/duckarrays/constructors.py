from abc import abstractmethod
from typing import TYPE_CHECKING

import hypothesis.extra.numpy as npst
import hypothesis.strategies as st
import numpy as np
import numpy.testing as npt
from hypothesis import given

import xarray as xr
import xarray.testing.strategies as xrst
from xarray.core.types import T_DuckArray

if TYPE_CHECKING:
    from xarray.core.types import _DTypeLikeNested, _ShapeLike


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
