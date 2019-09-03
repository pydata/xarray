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
    # pytest.mark.filterwarnings("ignore:::pint[.*]"),
]


def use_pint_version_or_xfail(*, version, reason):
    return pytest.mark.xfail(LooseVersion(pint.__version__) < version, reason=reason)


require_pint_array_function = use_pint_version_or_xfail(
    version="0.10", reason="pint does not implement __array_function__ yet"
)

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


def strip_units(data_array):
    def magnitude(da):
        if isinstance(da, xr.Variable):
            data = da.data
        else:
            data = da

        try:
            return data.magnitude
        except AttributeError:
            return data

    array = magnitude(data_array)
    coords = {name: magnitude(values) for name, values in data_array.coords.items()}

    return xr.DataArray(data=array, coords=coords, dims=tuple(coords.keys()))


@pytest.fixture(params=[float, int])
def dtype(request):
    return request.param


class TestDataArray:
    @require_pint_array_function
    @pytest.mark.filterwarnings("error:::pint[.*]")
    def test_init(self):
        array = np.arange(10) * unit_registry.m
        data_array = xr.DataArray(data=array)

        assert_equal_with_units(array, data_array)

    @require_pint_array_function
    @pytest.mark.filterwarnings("error:::pint[.*]")
    def test_repr(self):
        array = np.linspace(1, 2, 10) * unit_registry.m
        x = np.arange(len(array)) * unit_registry.s
        data_array = xr.DataArray(data=array, coords={"x": x}, dims=["x"])

        # FIXME: this just checks that the repr does not raise
        # warnings or errors, but does not check the result
        repr(data_array)

    @require_pint_array_function
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

    @require_pint_array_function
    @pytest.mark.parametrize(
        "func",
        (
            pytest.param(operator.neg, id="negate"),
            pytest.param(abs, id="absolute"),
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

    @require_pint_array_function
    @pytest.mark.parametrize(
        "func",
        (
            pytest.param(lambda x: 2 * x, id="multiply"),
            pytest.param(lambda x: x + x, id="add"),
            pytest.param(lambda x: x[0] + x, id="add scalar"),
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

    @require_pint_array_function
    @pytest.mark.parametrize(
        "units,error",
        (
            pytest.param(unit_registry.dimensionless, None, id="dimensionless"),
            pytest.param(
                unit_registry.m, pint.errors.DimensionalityError, id="incorrect unit"
            ),
            pytest.param(unit_registry.degree, None, id="correct unit"),
        ),
    )
    def test_univariate_ufunc(self, units, error, dtype):
        array = np.arange(10).astype(dtype) * units
        data_array = xr.DataArray(data=array)

        if error is not None:
            with pytest.raises(error):
                np.sin(data_array)
        else:
            assert_equal_with_units(np.sin(array), np.sin(data_array))

    @require_pint_array_function
    def test_bivariate_ufunc(self, dtype):
        unit = unit_registry.m
        array = np.arange(10).astype(dtype) * unit
        data_array = xr.DataArray(data=array)

        result_array = np.maximum(array, 0 * unit)
        assert_equal_with_units(result_array, np.maximum(data_array, 0 * unit))
        assert_equal_with_units(result_array, np.maximum(0 * unit, data_array))

    @require_pint_array_function
    @pytest.mark.parametrize(
        "indices",
        (
            pytest.param(4, id="single index"),
            pytest.param([5, 2, 9, 1], id="multiple indices"),
        ),
    )
    def test_isel(self, indices, dtype):
        array = np.arange(10).astype(dtype) * unit_registry.s
        x = np.arange(len(array)) * unit_registry.m
        data_array = xr.DataArray(data=array, coords={"x": x}, dims=["x"])

        assert_equal_with_units(array[indices], data_array.isel(x=indices))

    @require_pint_array_function
    @pytest.mark.parametrize(
        "values",
        (
            pytest.param(12, id="single value"),
            pytest.param([10, 5, 13], id="list of multiple values"),
            pytest.param(np.array([9, 3, 7, 12]), id="array of multiple values"),
        ),
    )
    @pytest.mark.parametrize(
        "units,error",
        (
            pytest.param(1, KeyError, id="no units"),
            pytest.param(unit_registry.dimensionless, KeyError, id="dimensionless"),
            pytest.param(unit_registry.degree, KeyError, id="incorrect unit"),
            pytest.param(unit_registry.s, None, id="correct unit"),
        ),
    )
    def test_sel(self, values, units, error, dtype):
        array = np.linspace(5, 10, 20).astype(dtype) * unit_registry.m
        x = np.arange(len(array)) * unit_registry.s
        data_array = xr.DataArray(data=array, coords={"x": x}, dims=["x"])

        values_with_units = values * units

        if error is not None:
            with pytest.raises(error):
                data_array.sel(x=values_with_units)
        else:
            result_array = array[values]
            result_data_array = data_array.sel(x=values_with_units)
            assert_equal_with_units(result_array, result_data_array)

    @require_pint_array_function
    @pytest.mark.parametrize(
        "values",
        (
            pytest.param(12, id="single value"),
            pytest.param([10, 5, 13], id="list of multiple values"),
            pytest.param(np.array([9, 3, 7, 12]), id="array of multiple values"),
        ),
    )
    @pytest.mark.parametrize(
        "units,error",
        (
            pytest.param(1, KeyError, id="no units"),
            pytest.param(unit_registry.dimensionless, KeyError, id="dimensionless"),
            pytest.param(unit_registry.degree, KeyError, id="incorrect unit"),
            pytest.param(unit_registry.s, None, id="correct unit"),
        ),
    )
    def test_loc(self, values, units, error, dtype):
        array = np.linspace(5, 10, 20).astype(dtype) * unit_registry.m
        x = np.arange(len(array)) * unit_registry.s
        data_array = xr.DataArray(data=array, coords={"x": x}, dims=["x"])

        values_with_units = values * units

        if error is not None:
            with pytest.raises(error):
                data_array.loc[values_with_units]
        else:
            result_array = array[values]
            result_data_array = data_array.loc[values_with_units]
            assert_equal_with_units(result_array, result_data_array)

    @require_pint_array_function
    @pytest.mark.xfail(reason="tries to coerce using asarray")
    @pytest.mark.parametrize(
        "shape",
        (
            pytest.param((10, 20), id="nothing squeezable"),
            pytest.param((10, 20, 1), id="last dimension squeezable"),
            pytest.param((10, 1, 20), id="middle dimension squeezable"),
            pytest.param((1, 10, 20), id="first dimension squeezable"),
            pytest.param((1, 10, 1, 20), id="first and last dimension squeezable"),
        ),
    )
    def test_squeeze(self, shape, dtype):
        names = "xyzt"
        coords = {
            name: np.arange(length).astype(dtype)
            * (unit_registry.m if name != "t" else unit_registry.s)
            for name, length in zip(names, shape)
        }
        array = np.arange(10 * 20).astype(dtype).reshape(shape) * unit_registry.J
        data_array = xr.DataArray(
            data=array, coords=coords, dims=tuple(names[: len(shape)])
        )

        result_array = array.squeeze()
        result_data_array = data_array.squeeze()
        assert_equal_with_units(result_array, result_data_array)

        # try squeezing the dimensions separately
        names = tuple(dim for dim, coord in coords.items() if len(coord) == 1)
        for index, name in enumerate(names):
            assert_equal_with_units(
                np.squeeze(array, axis=index), data_array.squeeze(dim=name)
            )

    @require_pint_array_function
    @pytest.mark.xfail(
        reason="interp() mistakes quantities as objects instead of numeric type arrays"
    )
    @pytest.mark.parametrize(
        "unit,error",
        (
            pytest.param(1, None, id="without unit"),
            pytest.param(unit_registry.dimensionless, None, id="dimensionless"),
            pytest.param(unit_registry.s, None, id="with incorrect unit"),
            pytest.param(unit_registry.m, None, id="with correct unit"),
        ),
    )
    def test_interp(self, unit, error):
        array = np.linspace(1, 2, 10 * 5).reshape(10, 5) * unit_registry.degK
        new_coords = (np.arange(10) + 0.5) * unit
        coords = {
            "x": np.arange(10) * unit_registry.m,
            "y": np.arange(5) * unit_registry.m,
        }

        data_array = xr.DataArray(array, coords=coords, dims=("x", "y"))

        if error is not None:
            with pytest.raises(error):
                data_array.interp(x=new_coords)
        else:
            new_coords_ = (
                new_coords.magnitude if hasattr(new_coords, "magnitude") else new_coords
            )
            result_array = strip_units(data_array).interp(
                x=new_coords_ * unit_registry.degK
            )
            result_data_array = data_array.interp(x=new_coords)

            assert_equal_with_units(result_array, result_data_array)

    @require_pint_array_function
    @pytest.mark.xfail(reason="tries to coerce using asarray")
    @pytest.mark.parametrize(
        "unit,error",
        (
            pytest.param(1, None, id="without unit"),
            pytest.param(unit_registry.dimensionless, None, id="dimensionless"),
            pytest.param(unit_registry.s, None, id="with incorrect unit"),
            pytest.param(unit_registry.m, None, id="with correct unit"),
        ),
    )
    def test_interp_like(self, unit, error):
        array = np.linspace(1, 2, 10 * 5).reshape(10, 5) * unit_registry.degK
        coords = {
            "x": (np.arange(10) + 0.3) * unit_registry.m,
            "y": (np.arange(5) + 0.3) * unit_registry.m,
        }

        data_array = xr.DataArray(array, coords=coords, dims=("x", "y"))
        new_data_array = xr.DataArray(
            data=np.empty((20, 10)),
            coords={"x": np.arange(20) * unit, "y": np.arange(10) * unit},
            dims=("x", "y"),
        )

        if error is not None:
            with pytest.raises(error):
                data_array.interp_like(new_data_array)
        else:
            result_array = (
                xr.DataArray(
                    data=array.magnitude,
                    coords={name: value.magnitude for name, value in coords.items()},
                    dims=("x", "y"),
                ).interp_like(strip_units(new_data_array))
                * unit_registry.degK
            )
            result_data_array = data_array.interp_like(new_data_array)

            assert_equal_with_units(result_array, result_data_array)

    @require_pint_array_function
    @pytest.mark.xfail(
        reason="pint does not implement np.result_type in __array_function__ yet"
    )
    @pytest.mark.parametrize(
        "unit,error",
        (
            pytest.param(1, None, id="without unit"),
            pytest.param(unit_registry.dimensionless, None, id="dimensionless"),
            pytest.param(unit_registry.s, None, id="with incorrect unit"),
            pytest.param(unit_registry.m, None, id="with correct unit"),
        ),
    )
    def test_reindex(self, unit, error):
        array = np.linspace(1, 2, 10 * 5).reshape(10, 5) * unit_registry.degK
        new_coords = (np.arange(10) + 0.5) * unit
        coords = {
            "x": np.arange(10) * unit_registry.m,
            "y": np.arange(5) * unit_registry.m,
        }

        data_array = xr.DataArray(array, coords=coords, dims=("x", "y"))

        if error is not None:
            with pytest.raises(error):
                data_array.interp(x=new_coords)
        else:
            result_array = strip_units(data_array).reindex(
                x=(
                    new_coords.magnitude
                    if hasattr(new_coords, "magnitude")
                    else new_coords
                )
                * unit_registry.degK
            )
            result_data_array = data_array.reindex(x=new_coords)

            assert_equal_with_units(result_array, result_data_array)

    @require_pint_array_function
    @pytest.mark.xfail(
        reason="pint does not implement np.result_type in __array_function__ yet"
    )
    @pytest.mark.parametrize(
        "unit,error",
        (
            pytest.param(1, None, id="without unit"),
            pytest.param(unit_registry.dimensionless, None, id="dimensionless"),
            pytest.param(unit_registry.s, None, id="with incorrect unit"),
            pytest.param(unit_registry.m, None, id="with correct unit"),
        ),
    )
    def test_reindex_like(self, unit, error):
        array = np.linspace(1, 2, 10 * 5).reshape(10, 5) * unit_registry.degK
        coords = {
            "x": (np.arange(10) + 0.3) * unit_registry.m,
            "y": (np.arange(5) + 0.3) * unit_registry.m,
        }

        data_array = xr.DataArray(array, coords=coords, dims=("x", "y"))
        new_data_array = xr.DataArray(
            data=np.empty((20, 10)),
            coords={"x": np.arange(20) * unit, "y": np.arange(10) * unit},
            dims=("x", "y"),
        )

        if error is not None:
            with pytest.raises(error):
                data_array.reindex_like(new_data_array)
        else:
            result_array = (
                xr.DataArray(
                    data=array.magnitude,
                    coords={name: value.magnitude for name, value in coords.items()},
                    dims=("x", "y"),
                ).reindex_like(strip_units(new_data_array))
                * unit_registry.degK
            )
            result_data_array = data_array.reindex_like(new_data_array)

            assert_equal_with_units(result_array, result_data_array)
