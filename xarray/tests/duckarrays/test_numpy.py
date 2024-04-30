from typing import TYPE_CHECKING

import hypothesis.extra.numpy as npst
import hypothesis.strategies as st

from xarray.core.types import T_DuckArray
from xarray.testing import duckarrays
from xarray.testing.strategies import supported_dtypes

if TYPE_CHECKING:
    from xarray.core.types import _DTypeLikeNested, _ShapeLike


class TestConstructors(duckarrays.ConstructorTests):
    dtypes = supported_dtypes()

    @staticmethod
    def array_strategy_fn(
        shape: "_ShapeLike",
        dtype: "_DTypeLikeNested",
    ) -> st.SearchStrategy[T_DuckArray]:
        return npst.arrays(shape=shape, dtype=dtype)


class TestReductions(duckarrays.ReduceTests):
    dtypes = supported_dtypes()

    @staticmethod
    def array_strategy_fn(
        shape: "_ShapeLike",
        dtype: "_DTypeLikeNested",
    ) -> st.SearchStrategy[T_DuckArray]:
        return npst.arrays(shape=shape, dtype=dtype)
