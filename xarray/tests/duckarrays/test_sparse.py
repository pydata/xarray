from typing import TYPE_CHECKING

import hypothesis.extra.numpy as npst
import hypothesis.strategies as st
import numpy as np
import numpy.testing as npt
import pytest

import xarray.testing.strategies as xrst
from xarray.testing import duckarrays
from xarray.tests import _importorskip

if TYPE_CHECKING:
    from xarray.core.types import _DTypeLikeNested, _ShapeLike


_importorskip("sparse")
import sparse


@pytest.fixture(autouse=True)
def disable_bottleneck():
    from xarray import set_options

    with set_options(use_bottleneck=False):
        yield


# sparse does not support float16
sparse_dtypes = xrst.supported_dtypes().filter(
    lambda dtype: (not np.issubdtype(dtype, np.float16))
)


@st.composite
def sparse_arrays_fn(
    draw: st.DrawFn,
    *,
    shape: "_ShapeLike",
    dtype: "_DTypeLikeNested",
) -> sparse.COO:
    """When called generates an arbitrary sparse.COO array of the given shape and dtype."""
    np_arr = draw(npst.arrays(dtype, shape))

    def to_sparse(arr: np.ndarray) -> sparse.COO:
        if arr.ndim == 0:
            return arr

        return sparse.COO.from_numpy(arr)

    return to_sparse(np_arr)


class TestConstructors(duckarrays.ConstructorTests):
    dtypes = sparse_dtypes()

    @staticmethod
    def array_strategy_fn(
        shape: "_ShapeLike",
        dtype: "_DTypeLikeNested",
    ) -> st.SearchStrategy[sparse.COO]:
        return sparse_arrays_fn

    def check_values(self, var, arr):
        npt.assert_equal(var.to_numpy(), arr.todense())


class TestReductions(duckarrays.ReduceTests):
    dtypes = nxps.scalar_dtypes()

    @staticmethod
    def array_strategy_fn(
        shape: "_ShapeLike",
        dtype: "_DTypeLikeNested",
    ) -> st.SearchStrategy[T_DuckArray]:
        return nxps.arrays(shape=shape, dtype=dtype)
