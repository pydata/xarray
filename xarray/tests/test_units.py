import numpy as np
import pytest

import xarray as xr
from xarray.core.npcompat import IS_NEP18_ACTIVE

pint = pytest.importorskip("pint")
pytestmark = [
    pytest.mark.skipif(
        not IS_NEP18_ACTIVE, reason="NUMPY_EXPERIMENTAL_ARRAY_FUNCTION is not enabled"
    ),
    pytest.mark.filterwarnings("error::pint.errors.UnitStrippedWarning"),
]


unit_registry = pint.UnitRegistry()


def assert_equal_with_units(a, b):
    a_ = a if not isinstance(a, (xr.Dataset, xr.DataArray, xr.Variable)) else a.data
    b_ = b if not isinstance(b, (xr.Dataset, xr.DataArray, xr.Variable)) else b.data

    assert np.allclose(a_, b_)
    if hasattr(a_, "units") or hasattr(b_, "units"):
        assert (hasattr(a_, "units") and hasattr(b_, "units")) and a_.units == b_.units


@pytest.fixture(params=[float, int])
def dtype(request):
    return request.param


class TestDataArray:
    @pytest.mark.filterwarnings("error::pint.errors.UnitStrippedWarning")
    def test_init(self):
        array = np.arange(10) * unit_registry.m
        data_array = xr.DataArray(data=array)

        assert_equal_with_units(array, data_array)

    @pytest.mark.parametrize(
        "func",
        (
            np.all,
            np.any,
            np.argmax,
            np.argmin,
            np.max,
            np.mean,
            np.median,
            np.min,
            np.prod,
            np.sum,
            np.std,
            np.var,
        ),
    )
    def test_aggregation(self, func, dtype):
        array = np.arange(10).astype(dtype) * unit_registry.m
        data_array = xr.DataArray(data=array)

        result_array = func(array)
        result_data_array = func(data_array)

        assert_equal_with_units(result_array, result_data_array)
