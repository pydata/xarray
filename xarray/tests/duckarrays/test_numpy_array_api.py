import warnings
from typing import TYPE_CHECKING

import hypothesis.strategies as st
from hypothesis.extra.array_api import make_strategies_namespace

from xarray.core.types import T_DuckArray
from xarray.testing import duckarrays
from xarray.tests import _importorskip

if TYPE_CHECKING:
    from xarray.core.types import _DTypeLikeNested, _ShapeLike


with warnings.catch_warnings():
    # ignore the warning that the array_api is experimental raised by numpy
    warnings.simplefilter("ignore")
    _importorskip("numpy", "1.26.0")
    import numpy.array_api as nxp

nxps = make_strategies_namespace(nxp)


class TestVariableConstructors(duckarrays.VariableConstructorTests):
    dtypes = nxps.scalar_dtypes()

    @staticmethod
    def array_strategy_fn(
        shape: "_ShapeLike",
        dtype: "_DTypeLikeNested",
    ) -> st.SearchStrategy[T_DuckArray]:
        return nxps.arrays(shape=shape, dtype=dtype)


class TestVariableReductions(duckarrays.VariableReduceTests):
    dtypes = nxps.scalar_dtypes()

    @staticmethod
    def array_strategy_fn(
        shape: "_ShapeLike",
        dtype: "_DTypeLikeNested",
    ) -> st.SearchStrategy[T_DuckArray]:
        return nxps.arrays(shape=shape, dtype=dtype)
