import operator
from distutils.version import LooseVersion

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

# pint version supporting __array_function__
pint_version = "0.10"


def use_pint_dev_or_xfail(reason):
    return pytest.mark.xfail(LooseVersion(pint.__version__) < pint_version, reason=reason)


unit_registry = pint.UnitRegistry()


def assert_equal_with_units(a, b):
    from pint.quantity import BaseQuantity

    a_ = a if not isinstance(a, (xr.Dataset, xr.DataArray, xr.Variable)) else a.data
    b_ = b if not isinstance(b, (xr.Dataset, xr.DataArray, xr.Variable)) else b.data

    # workaround until pint implements allclose in __array_function__
    if isinstance(a_, BaseQuantity) or isinstance(b_, BaseQuantity):
        assert (hasattr(a_, "magnitude") and hasattr(b_, "magnitude")) and np.allclose(
            a_.magnitude, b_.magnitude, equal_nan=True
        )
    else:
        assert np.allclose(a_, b_, equal_nan=True)

    if hasattr(a_, "units") or hasattr(b_, "units"):
        assert (hasattr(a_, "units") and hasattr(b_, "units")) and a_.units == b_.units


@pytest.fixture(params=[float, int])
def dtype(request):
    return request.param


class TestDataArray:
    @use_pint_dev_or_xfail(reason="pint does not implement __array_function__ yet")
    def test_init(self):
        array = np.arange(10) * unit_registry.m
        data_array = xr.DataArray(data=array)

        assert_equal_with_units(array, data_array)

    @pytest.mark.xfail(
        reason="pint does not implement __array_function__ for aggregation functions yet"
    )
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

    @pytest.mark.parametrize(
        "func",
        (
            pytest.param(
                operator.neg,
                id="negate",
                marks=use_pint_dev_or_xfail(
                    reason="pint does not implement __array_function__ yet"
                ),
            ),
            pytest.param(
                abs,
                id="absolute",
                marks=use_pint_dev_or_xfail(
                    reason="pint does not implement __array_function__ yet"
                ),
            ),
            pytest.param(
                np.round,
                id="round",
                marks=pytest.mark.xfail(reason="pint does not implement round"),
            ),
        ),
    )
    def test_unary_operations(self, func, dtype):
        array = np.arange(10).astype(dtype) * unit_registry.m
        data_array = xr.DataArray(data=array)

        assert_equal_with_units(func(array), func(data_array))

    @pytest.mark.parametrize(
        "func",
        (
            pytest.param(
                lambda x: 2 * x,
                id="multiply",
                marks=use_pint_dev_or_xfail(
                    reason="pint does not implement __array_function__ yet"
                ),
            ),
            pytest.param(
                lambda x: x + x,
                id="add",
                marks=use_pint_dev_or_xfail(
                    reason="pint does not implement __array_function__ yet"
                ),
            ),
            pytest.param(
                lambda x: x[0] + x,
                id="add scalar",
                marks=use_pint_dev_or_xfail(
                    reason="pint does not implement __array_function__ yet"
                ),
            ),
            pytest.param(
                lambda x: x.T @ x,
                id="matrix multiply",
                marks=pytest.mark.xfail(
                    reason="pint does not support matrix multiplication yet"
                ),
            ),
        ),
    )
    def test_binary_operations(self, func, dtype):
        array = np.arange(10).astype(dtype) * unit_registry.m
        data_array = xr.DataArray(data=array)

        assert_equal_with_units(func(array), func(data_array))

    @pytest.mark.parametrize(
        "indices",
        (
            pytest.param(
                4,
                id="single index",
                marks=pytest.mark.xfail(
                    reason="single index isel() tries to coerce to int"
                ),
            ),
            pytest.param(
                [5, 2, 9, 1],
                id="multiple indices",
                marks=use_pint_dev_or_xfail(
                    reason="pint does not implement __array_function__ yet"
                ),
            ),
        ),
    )
    def test_isel(self, indices, dtype):
        array = np.arange(10).astype(dtype) * unit_registry.s
        x = np.arange(len(array)) * unit_registry.m
        data_array = xr.DataArray(data=array, coords={"x": x}, dims=["x"])

        assert_equal_with_units(array[indices], data_array.isel(x=indices))

    @pytest.mark.parametrize(
        "values,error",
        (
            pytest.param(
                12,
                KeyError,
                id="single value without unit",
                marks=use_pint_dev_or_xfail(
                    reason="pint does not implement __array_function__ yet"
                ),
            ),
            pytest.param(
                12 * unit_registry.degree,
                KeyError,
                id="single value with incorrect unit",
                marks=use_pint_dev_or_xfail(
                    reason="pint does not implement __array_function__ yet"
                ),
            ),
            pytest.param(
                12 * unit_registry.s,
                None,
                id="single value with correct unit",
                marks=pytest.mark.xfail(reason="single value tries to coerce to int"),
            ),
            pytest.param(
                [10, 5, 13],
                KeyError,
                id="multiple values without unit",
                marks=use_pint_dev_or_xfail(
                    reason="pint does not implement __array_function__ yet"
                ),
            ),
            pytest.param(
                [10, 5, 13] * unit_registry.degree,
                KeyError,
                id="multiple values with incorrect unit",
                marks=use_pint_dev_or_xfail(
                    reason="pint does not implement __array_function__ yet"
                ),
            ),
            pytest.param(
                [10, 5, 13] * unit_registry.s,
                None,
                id="multiple values with correct unit",
                marks=use_pint_dev_or_xfail(
                    reason="pint does not implement __array_function__ yet"
                ),
            ),
        ),
    )
    def test_sel(self, values, error, dtype):
        array = np.linspace(5, 10, 20).astype(dtype) * unit_registry.m
        x = np.arange(len(array)) * unit_registry.s
        data_array = xr.DataArray(data=array, coords={"x": x}, dims=["x"])

        if error is not None:
            with pytest.raises(error):
                data_array.sel(x=values)
        else:
            assert_equal_with_units(array[values.magnitude], data_array.sel(x=values))

    @pytest.mark.parametrize(
        "values,error",
        (
            pytest.param(
                12,
                KeyError,
                id="single value without unit",
                marks=use_pint_dev_or_xfail(
                    reason="pint does not implement __array_function__ yet"
                ),
            ),
            pytest.param(
                12 * unit_registry.degree,
                KeyError,
                id="single value with incorrect unit",
                marks=use_pint_dev_or_xfail(
                    reason="pint does not implement __array_function__ yet"
                ),
            ),
            pytest.param(
                12 * unit_registry.s,
                None,
                id="single value with correct unit",
                marks=pytest.mark.xfail(reason="single value tries to coerce to int"),
            ),
            pytest.param(
                [10, 5, 13],
                KeyError,
                id="multiple values without unit",
                marks=use_pint_dev_or_xfail(
                    reason="pint does not implement __array_function__ yet"
                ),
            ),
            pytest.param(
                [10, 5, 13] * unit_registry.degree,
                KeyError,
                id="multiple values with incorrect unit",
                marks=use_pint_dev_or_xfail(
                    reason="pint does not implement __array_function__ yet"
                ),
            ),
            pytest.param(
                [10, 5, 13] * unit_registry.s,
                None,
                id="multiple values with correct unit",
                marks=use_pint_dev_or_xfail(
                    reason="pint does not implement __array_function__ yet"
                ),
            ),
        ),
    )
    def test_loc(self, values, error, dtype):
        array = np.linspace(5, 10, 20).astype(dtype) * unit_registry.m
        x = np.arange(len(array)) * unit_registry.s
        data_array = xr.DataArray(data=array, coords={"x": x}, dims=["x"])

        if error is not None:
            with pytest.raises(error):
                data_array.loc[values]
        else:
            assert_equal_with_units(array[values.magnitude], data_array.loc[values])

    @pytest.mark.xfail(reason="indexing calls np.asarray")
    @pytest.mark.parametrize("shape", (
        (10, 20),
        (10, 20, 1),
        (10, 1, 20),
        (1, 10, 20),
        (1, 10, 1, 20),
    ))
    def test_squeeze(self, shape, dtype):
        names = "xyzt"
        coords = {
            name: np.arange(length).astype(dtype) * (unit_registry.m if name != "t" else unit_registry.s)
            for name, length in zip(names, shape)
        }
        array = np.arange(10 * 20).astype(dtype).reshape(shape) * unit_registry.J
        data_array = xr.DataArray(data=array, coords=coords, dims=tuple(names[:len(shape)]))

        assert_equal_with_units(np.squeeze(array), data_array.squeeze())
        # try squeezing the dimensions separately
        names = tuple(dim for dim, coord in coords.items() if len(coord) == 1)
        for index, name in enumerate(names):
            assert_equal_with_units(np.squeeze(array, axis=index), data_array.squeeze(dim=name))
