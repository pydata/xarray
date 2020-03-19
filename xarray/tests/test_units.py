import functools
import operator
from distutils.version import LooseVersion

import numpy as np
import pandas as pd
import pytest

import xarray as xr
from xarray.core import formatting
from xarray.core.npcompat import IS_NEP18_ACTIVE
from xarray.testing import assert_allclose, assert_identical

from .test_variable import _PAD_XR_NP_ARGS, VariableSubclassobjects

pint = pytest.importorskip("pint")
DimensionalityError = pint.errors.DimensionalityError


# make sure scalars are converted to 0d arrays so quantities can
# always be treated like ndarrays
unit_registry = pint.UnitRegistry(force_ndarray=True)
Quantity = unit_registry.Quantity


pytestmark = [
    pytest.mark.skipif(
        not IS_NEP18_ACTIVE, reason="NUMPY_EXPERIMENTAL_ARRAY_FUNCTION is not enabled"
    ),
    # TODO: remove this once pint has a released version with __array_function__
    pytest.mark.skipif(
        not hasattr(unit_registry.Quantity, "__array_function__"),
        reason="pint does not implement __array_function__ yet",
    ),
    # pytest.mark.filterwarnings("ignore:::pint[.*]"),
]


def is_compatible(unit1, unit2):
    def dimensionality(obj):
        if isinstance(obj, (unit_registry.Quantity, unit_registry.Unit)):
            unit_like = obj
        else:
            unit_like = unit_registry.dimensionless

        return unit_like.dimensionality

    return dimensionality(unit1) == dimensionality(unit2)


def compatible_mappings(first, second):
    return {
        key: is_compatible(unit1, unit2)
        for key, (unit1, unit2) in merge_mappings(first, second)
    }


def array_extract_units(obj):
    if isinstance(obj, (xr.Variable, xr.DataArray, xr.Dataset)):
        obj = obj.data

    try:
        return obj.units
    except AttributeError:
        return None


def array_strip_units(array):
    try:
        return array.magnitude
    except AttributeError:
        return array


def array_attach_units(data, unit):
    if isinstance(data, Quantity):
        raise ValueError(f"cannot attach unit {unit} to quantity {data}")

    try:
        quantity = data * unit
    except np.core._exceptions.UFuncTypeError:
        if isinstance(unit, unit_registry.Unit):
            raise

        quantity = data

    return quantity


def extract_units(obj):
    if isinstance(obj, xr.Dataset):
        vars_units = {
            name: array_extract_units(value) for name, value in obj.data_vars.items()
        }
        coords_units = {
            name: array_extract_units(value) for name, value in obj.coords.items()
        }

        units = {**vars_units, **coords_units}
    elif isinstance(obj, xr.DataArray):
        vars_units = {obj.name: array_extract_units(obj)}
        coords_units = {
            name: array_extract_units(value) for name, value in obj.coords.items()
        }

        units = {**vars_units, **coords_units}
    elif isinstance(obj, xr.Variable):
        vars_units = {None: array_extract_units(obj.data)}

        units = {**vars_units}
    elif isinstance(obj, Quantity):
        vars_units = {None: array_extract_units(obj)}

        units = {**vars_units}
    else:
        units = {}

    return units


def strip_units(obj):
    if isinstance(obj, xr.Dataset):
        data_vars = {
            strip_units(name): strip_units(value)
            for name, value in obj.data_vars.items()
        }
        coords = {
            strip_units(name): strip_units(value) for name, value in obj.coords.items()
        }

        new_obj = xr.Dataset(data_vars=data_vars, coords=coords)
    elif isinstance(obj, xr.DataArray):
        data = array_strip_units(obj.data)
        coords = {
            strip_units(name): (
                (value.dims, array_strip_units(value.data))
                if isinstance(value.data, Quantity)
                else value  # to preserve multiindexes
            )
            for name, value in obj.coords.items()
        }

        new_obj = xr.DataArray(
            name=strip_units(obj.name), data=data, coords=coords, dims=obj.dims
        )
    elif isinstance(obj, xr.Variable):
        data = array_strip_units(obj.data)
        new_obj = obj.copy(data=data)
    elif isinstance(obj, unit_registry.Quantity):
        new_obj = obj.magnitude
    elif isinstance(obj, (list, tuple)):
        return type(obj)(strip_units(elem) for elem in obj)
    else:
        new_obj = obj

    return new_obj


def attach_units(obj, units):
    if not isinstance(obj, (xr.DataArray, xr.Dataset, xr.Variable)):
        units = units.get("data", None) or units.get(None, None) or 1
        return array_attach_units(obj, units)

    if isinstance(obj, xr.Dataset):
        data_vars = {
            name: attach_units(value, units) for name, value in obj.data_vars.items()
        }

        coords = {
            name: attach_units(value, units) for name, value in obj.coords.items()
        }

        new_obj = xr.Dataset(data_vars=data_vars, coords=coords, attrs=obj.attrs)
    elif isinstance(obj, xr.DataArray):
        # try the array name, "data" and None, then fall back to dimensionless
        data_units = (
            units.get(obj.name, None)
            or units.get("data", None)
            or units.get(None, None)
            or 1
        )

        data = array_attach_units(obj.data, data_units)

        coords = {
            name: (
                (value.dims, array_attach_units(value.data, units.get(name) or 1))
                if name in units
                # to preserve multiindexes
                else value
            )
            for name, value in obj.coords.items()
        }
        dims = obj.dims
        attrs = obj.attrs

        new_obj = xr.DataArray(
            name=obj.name, data=data, coords=coords, attrs=attrs, dims=dims
        )
    else:
        data_units = units.get("data", None) or units.get(None, None) or 1

        data = array_attach_units(obj.data, data_units)
        new_obj = obj.copy(data=data)

    return new_obj


def convert_units(obj, to):
    # preprocess
    to = {
        key: None if not isinstance(value, unit_registry.Unit) else value
        for key, value in to.items()
    }
    if isinstance(obj, xr.Dataset):
        data_vars = {
            name: convert_units(array.variable, {None: to.get(name)})
            for name, array in obj.data_vars.items()
        }
        coords = {
            name: convert_units(array.variable, {None: to.get(name)})
            for name, array in obj.coords.items()
        }

        new_obj = xr.Dataset(data_vars=data_vars, coords=coords, attrs=obj.attrs)
    elif isinstance(obj, xr.DataArray):
        name = obj.name

        new_units = (
            to.get(name, None) or to.get("data", None) or to.get(None, None) or None
        )
        data = convert_units(obj.variable, {None: new_units})

        coords = {
            name: (array.dims, convert_units(array.variable, {None: to.get(name)}))
            for name, array in obj.coords.items()
            if name != obj.name
        }

        new_obj = xr.DataArray(
            name=name, data=data, coords=coords, attrs=obj.attrs, dims=obj.dims
        )
    elif isinstance(obj, xr.Variable):
        new_data = convert_units(obj.data, to)
        new_obj = obj.copy(data=new_data)
    elif isinstance(obj, unit_registry.Quantity):
        units = to.get(None)
        new_obj = obj.to(units) if units is not None else obj
    else:
        new_obj = obj

    return new_obj


def assert_units_equal(a, b):
    __tracebackhide__ = True
    assert extract_units(a) == extract_units(b)


def assert_equal_with_units(a, b):
    # works like xr.testing.assert_equal, but also explicitly checks units
    # so, it is more like assert_identical
    __tracebackhide__ = True

    if isinstance(a, xr.Dataset) or isinstance(b, xr.Dataset):
        a_units = extract_units(a)
        b_units = extract_units(b)

        a_without_units = strip_units(a)
        b_without_units = strip_units(b)

        assert a_without_units.equals(b_without_units), formatting.diff_dataset_repr(
            a, b, "equals"
        )
        assert a_units == b_units
    else:
        a = a if not isinstance(a, (xr.DataArray, xr.Variable)) else a.data
        b = b if not isinstance(b, (xr.DataArray, xr.Variable)) else b.data

        assert type(a) == type(b) or (
            isinstance(a, Quantity) and isinstance(b, Quantity)
        )

        # workaround until pint implements allclose in __array_function__
        if isinstance(a, Quantity) or isinstance(b, Quantity):
            assert (
                hasattr(a, "magnitude") and hasattr(b, "magnitude")
            ) and np.allclose(a.magnitude, b.magnitude, equal_nan=True)
            assert (hasattr(a, "units") and hasattr(b, "units")) and a.units == b.units
        else:
            assert np.allclose(a, b, equal_nan=True)


@pytest.fixture(params=[float, int])
def dtype(request):
    return request.param


def merge_mappings(*mappings):
    for key in set(mappings[0]).intersection(*mappings[1:]):
        yield key, tuple(m[key] for m in mappings)


def merge_args(default_args, new_args):
    from itertools import zip_longest

    fill_value = object()
    return [
        second if second is not fill_value else first
        for first, second in zip_longest(default_args, new_args, fillvalue=fill_value)
    ]


class method:
    """ wrapper class to help with passing methods via parametrize

    This is works a bit similar to using `partial(Class.method, arg, kwarg)`
    """

    def __init__(self, name, *args, **kwargs):
        self.name = name
        self.args = args
        self.kwargs = kwargs

    def __call__(self, obj, *args, **kwargs):
        from collections.abc import Callable
        from functools import partial

        all_args = merge_args(self.args, args)
        all_kwargs = {**self.kwargs, **kwargs}

        func = getattr(obj, self.name, None)
        if func is None or not isinstance(func, Callable):
            # fall back to module level numpy functions if not a xarray object
            if not isinstance(obj, (xr.Variable, xr.DataArray, xr.Dataset)):
                numpy_func = getattr(np, self.name)
                func = partial(numpy_func, obj)
                # remove typical xarray args like "dim"
                exclude_kwargs = ("dim", "dims")
                all_kwargs = {
                    key: value
                    for key, value in all_kwargs.items()
                    if key not in exclude_kwargs
                }
            else:
                raise AttributeError(f"{obj} has no method named '{self.name}'")

        return func(*all_args, **all_kwargs)

    def __repr__(self):
        return f"method_{self.name}"


class function:
    """ wrapper class for numpy functions

    Same as method, but the name is used for referencing numpy functions
    """

    def __init__(self, name_or_function, *args, function_label=None, **kwargs):
        if callable(name_or_function):
            self.name = (
                function_label
                if function_label is not None
                else name_or_function.__name__
            )
            self.func = name_or_function
        else:
            self.name = name_or_function if function_label is None else function_label
            self.func = getattr(np, name_or_function)
            if self.func is None:
                raise AttributeError(
                    f"module 'numpy' has no attribute named '{self.name}'"
                )

        self.args = args
        self.kwargs = kwargs

    def __call__(self, *args, **kwargs):
        all_args = merge_args(self.args, args)
        all_kwargs = {**self.kwargs, **kwargs}

        return self.func(*all_args, **all_kwargs)

    def __repr__(self):
        return f"function_{self.name}"


def test_apply_ufunc_dataarray(dtype):
    func = functools.partial(
        xr.apply_ufunc, np.mean, input_core_dims=[["x"]], kwargs={"axis": -1}
    )

    array = np.linspace(0, 10, 20).astype(dtype) * unit_registry.m
    x = np.arange(20) * unit_registry.s
    data_array = xr.DataArray(data=array, dims="x", coords={"x": x})

    expected = attach_units(func(strip_units(data_array)), extract_units(data_array))
    actual = func(data_array)

    assert_units_equal(expected, actual)
    assert_identical(expected, actual)


def test_apply_ufunc_dataset(dtype):
    func = functools.partial(
        xr.apply_ufunc, np.mean, input_core_dims=[["x"]], kwargs={"axis": -1}
    )

    array1 = np.linspace(0, 10, 5 * 10).reshape(5, 10).astype(dtype) * unit_registry.m
    array2 = np.linspace(0, 10, 5).astype(dtype) * unit_registry.m

    x = np.arange(5) * unit_registry.s
    y = np.arange(10) * unit_registry.m

    ds = xr.Dataset(
        data_vars={"a": (("x", "y"), array1), "b": ("x", array2)},
        coords={"x": x, "y": y},
    )

    expected = attach_units(func(strip_units(ds)), extract_units(ds))
    actual = func(ds)

    assert_units_equal(expected, actual)
    assert_identical(expected, actual)


@pytest.mark.parametrize(
    "unit,error",
    (
        pytest.param(1, DimensionalityError, id="no_unit"),
        pytest.param(
            unit_registry.dimensionless, DimensionalityError, id="dimensionless"
        ),
        pytest.param(unit_registry.s, DimensionalityError, id="incompatible_unit"),
        pytest.param(unit_registry.mm, None, id="compatible_unit"),
        pytest.param(unit_registry.m, None, id="identical_unit"),
    ),
    ids=repr,
)
@pytest.mark.parametrize(
    "variant",
    (
        "data",
        pytest.param("dims", marks=pytest.mark.xfail(reason="indexes strip units")),
        "coords",
    ),
)
@pytest.mark.parametrize("fill_value", (10, np.nan))
def test_align_dataarray(fill_value, variant, unit, error, dtype):
    original_unit = unit_registry.m

    variants = {
        "data": (unit, original_unit, original_unit),
        "dims": (original_unit, unit, original_unit),
        "coords": (original_unit, original_unit, unit),
    }
    data_unit, dim_unit, coord_unit = variants.get(variant)

    array1 = np.linspace(0, 10, 2 * 5).reshape(2, 5).astype(dtype) * original_unit
    array2 = np.linspace(0, 8, 2 * 5).reshape(2, 5).astype(dtype) * data_unit
    x = np.arange(2) * original_unit

    y1 = np.arange(5) * original_unit
    y2 = np.arange(2, 7) * dim_unit
    y_a1 = np.array([3, 5, 7, 8, 9]) * original_unit
    y_a2 = np.array([7, 8, 9, 11, 13]) * coord_unit

    coords1 = {"x": x, "y": y1}
    coords2 = {"x": x, "y": y2}
    if variant == "coords":
        coords1["y_a"] = ("y", y_a1)
        coords2["y_a"] = ("y", y_a2)

    data_array1 = xr.DataArray(data=array1, coords=coords1, dims=("x", "y"))
    data_array2 = xr.DataArray(data=array2, coords=coords2, dims=("x", "y"))

    fill_value = fill_value * data_unit
    func = function(xr.align, join="outer", fill_value=fill_value)
    if error is not None and not (
        np.isnan(fill_value) and not isinstance(fill_value, Quantity)
    ):
        with pytest.raises(error):
            func(data_array1, data_array2)

        return

    stripped_kwargs = {
        key: strip_units(
            convert_units(value, {None: original_unit if data_unit != 1 else None})
        )
        for key, value in func.kwargs.items()
    }
    units_a = extract_units(data_array1)
    units_b = extract_units(data_array2)
    expected_a, expected_b = func(
        strip_units(data_array1),
        strip_units(convert_units(data_array2, units_a)),
        **stripped_kwargs,
    )
    expected_a = attach_units(expected_a, units_a)
    if isinstance(array2, Quantity):
        expected_b = convert_units(attach_units(expected_b, units_a), units_b)
    else:
        expected_b = attach_units(expected_b, units_b)

    actual_a, actual_b = func(data_array1, data_array2)

    assert_units_equal(expected_a, actual_a)
    assert_allclose(expected_a, actual_a)
    assert_units_equal(expected_b, actual_b)
    assert_allclose(expected_b, actual_b)


@pytest.mark.parametrize(
    "unit,error",
    (
        pytest.param(1, DimensionalityError, id="no_unit"),
        pytest.param(
            unit_registry.dimensionless, DimensionalityError, id="dimensionless"
        ),
        pytest.param(unit_registry.s, DimensionalityError, id="incompatible_unit"),
        pytest.param(unit_registry.mm, None, id="compatible_unit"),
        pytest.param(unit_registry.m, None, id="identical_unit"),
    ),
    ids=repr,
)
@pytest.mark.parametrize(
    "variant",
    (
        "data",
        pytest.param("dims", marks=pytest.mark.xfail(reason="indexes strip units")),
        "coords",
    ),
)
@pytest.mark.parametrize("fill_value", (np.float64(10), np.float64(np.nan)))
def test_align_dataset(fill_value, unit, variant, error, dtype):
    original_unit = unit_registry.m

    variants = {
        "data": (unit, original_unit, original_unit),
        "dims": (original_unit, unit, original_unit),
        "coords": (original_unit, original_unit, unit),
    }
    data_unit, dim_unit, coord_unit = variants.get(variant)

    array1 = np.linspace(0, 10, 2 * 5).reshape(2, 5).astype(dtype) * original_unit
    array2 = np.linspace(0, 10, 2 * 5).reshape(2, 5).astype(dtype) * data_unit

    x = np.arange(2) * original_unit

    y1 = np.arange(5) * original_unit
    y2 = np.arange(2, 7) * dim_unit
    y_a1 = np.array([3, 5, 7, 8, 9]) * original_unit
    y_a2 = np.array([7, 8, 9, 11, 13]) * coord_unit

    coords1 = {"x": x, "y": y1}
    coords2 = {"x": x, "y": y2}
    if variant == "coords":
        coords1["y_a"] = ("y", y_a1)
        coords2["y_a"] = ("y", y_a2)

    ds1 = xr.Dataset(data_vars={"a": (("x", "y"), array1)}, coords=coords1)
    ds2 = xr.Dataset(data_vars={"a": (("x", "y"), array2)}, coords=coords2)

    fill_value = fill_value * data_unit
    func = function(xr.align, join="outer", fill_value=fill_value)
    if error is not None and not (
        np.isnan(fill_value) and not isinstance(fill_value, Quantity)
    ):
        with pytest.raises(error):
            func(ds1, ds2)

        return

    stripped_kwargs = {
        key: strip_units(
            convert_units(value, {None: original_unit if data_unit != 1 else None})
        )
        for key, value in func.kwargs.items()
    }
    units_a = extract_units(ds1)
    units_b = extract_units(ds2)
    expected_a, expected_b = func(
        strip_units(ds1), strip_units(convert_units(ds2, units_a)), **stripped_kwargs
    )
    expected_a = attach_units(expected_a, units_a)
    if isinstance(array2, Quantity):
        expected_b = convert_units(attach_units(expected_b, units_a), units_b)
    else:
        expected_b = attach_units(expected_b, units_b)

    actual_a, actual_b = func(ds1, ds2)

    assert_units_equal(expected_a, actual_a)
    assert_allclose(expected_a, actual_a)
    assert_units_equal(expected_b, actual_b)
    assert_allclose(expected_b, actual_b)


def test_broadcast_dataarray(dtype):
    array1 = np.linspace(0, 10, 2) * unit_registry.Pa
    array2 = np.linspace(0, 10, 3) * unit_registry.Pa

    a = xr.DataArray(data=array1, dims="x")
    b = xr.DataArray(data=array2, dims="y")

    units_a = extract_units(a)
    units_b = extract_units(b)
    expected_a, expected_b = xr.broadcast(strip_units(a), strip_units(b))
    expected_a = attach_units(expected_a, units_a)
    expected_b = convert_units(attach_units(expected_b, units_a), units_b)

    actual_a, actual_b = xr.broadcast(a, b)

    assert_units_equal(expected_a, actual_a)
    assert_identical(expected_a, actual_a)
    assert_units_equal(expected_b, actual_b)
    assert_identical(expected_b, actual_b)


def test_broadcast_dataset(dtype):
    array1 = np.linspace(0, 10, 2) * unit_registry.Pa
    array2 = np.linspace(0, 10, 3) * unit_registry.Pa

    x1 = np.arange(2)
    y1 = np.arange(3)

    x2 = np.arange(2, 4)
    y2 = np.arange(3, 6)

    ds = xr.Dataset(
        data_vars={"a": ("x", array1), "b": ("y", array2)}, coords={"x": x1, "y": y1}
    )
    other = xr.Dataset(
        data_vars={
            "a": ("x", array1.to(unit_registry.hPa)),
            "b": ("y", array2.to(unit_registry.hPa)),
        },
        coords={"x": x2, "y": y2},
    )

    units_a = extract_units(ds)
    units_b = extract_units(other)
    expected_a, expected_b = xr.broadcast(strip_units(ds), strip_units(other))
    expected_a = attach_units(expected_a, units_a)
    expected_b = attach_units(expected_b, units_b)

    actual_a, actual_b = xr.broadcast(ds, other)

    assert_units_equal(expected_a, actual_a)
    assert_identical(expected_a, actual_a)
    assert_units_equal(expected_b, actual_b)
    assert_identical(expected_b, actual_b)


@pytest.mark.parametrize(
    "unit,error",
    (
        pytest.param(1, DimensionalityError, id="no_unit"),
        pytest.param(
            unit_registry.dimensionless, DimensionalityError, id="dimensionless"
        ),
        pytest.param(unit_registry.s, DimensionalityError, id="incompatible_unit"),
        pytest.param(unit_registry.mm, None, id="compatible_unit"),
        pytest.param(unit_registry.m, None, id="identical_unit"),
    ),
    ids=repr,
)
@pytest.mark.parametrize(
    "variant",
    (
        "data",
        pytest.param("dims", marks=pytest.mark.xfail(reason="indexes strip units")),
        "coords",
    ),
)
def test_combine_by_coords(variant, unit, error, dtype):
    original_unit = unit_registry.m

    variants = {
        "data": (unit, original_unit, original_unit),
        "dims": (original_unit, unit, original_unit),
        "coords": (original_unit, original_unit, unit),
    }
    data_unit, dim_unit, coord_unit = variants.get(variant)

    array1 = np.zeros(shape=(2, 3), dtype=dtype) * original_unit
    array2 = np.zeros(shape=(2, 3), dtype=dtype) * original_unit
    x = np.arange(1, 4) * 10 * original_unit
    y = np.arange(2) * original_unit
    z = np.arange(3) * original_unit

    other_array1 = np.ones_like(array1) * data_unit
    other_array2 = np.ones_like(array2) * data_unit
    other_x = np.arange(1, 4) * 10 * dim_unit
    other_y = np.arange(2, 4) * dim_unit
    other_z = np.arange(3, 6) * coord_unit

    ds = xr.Dataset(
        data_vars={"a": (("y", "x"), array1), "b": (("y", "x"), array2)},
        coords={"x": x, "y": y, "z": ("x", z)},
    )
    other = xr.Dataset(
        data_vars={"a": (("y", "x"), other_array1), "b": (("y", "x"), other_array2)},
        coords={"x": other_x, "y": other_y, "z": ("x", other_z)},
    )

    if error is not None:
        with pytest.raises(error):
            xr.combine_by_coords([ds, other])

        return

    units = extract_units(ds)
    expected = attach_units(
        xr.combine_by_coords(
            [strip_units(ds), strip_units(convert_units(other, units))]
        ),
        units,
    )
    actual = xr.combine_by_coords([ds, other])

    assert_units_equal(expected, actual)
    assert_identical(expected, actual)


@pytest.mark.parametrize(
    "unit,error",
    (
        pytest.param(1, DimensionalityError, id="no_unit"),
        pytest.param(
            unit_registry.dimensionless, DimensionalityError, id="dimensionless"
        ),
        pytest.param(unit_registry.s, DimensionalityError, id="incompatible_unit"),
        pytest.param(unit_registry.mm, None, id="compatible_unit"),
        pytest.param(unit_registry.m, None, id="identical_unit"),
    ),
    ids=repr,
)
@pytest.mark.parametrize(
    "variant",
    (
        "data",
        pytest.param("dims", marks=pytest.mark.xfail(reason="indexes strip units")),
        "coords",
    ),
)
def test_combine_nested(variant, unit, error, dtype):
    original_unit = unit_registry.m

    variants = {
        "data": (unit, original_unit, original_unit),
        "dims": (original_unit, unit, original_unit),
        "coords": (original_unit, original_unit, unit),
    }
    data_unit, dim_unit, coord_unit = variants.get(variant)

    array1 = np.zeros(shape=(2, 3), dtype=dtype) * original_unit
    array2 = np.zeros(shape=(2, 3), dtype=dtype) * original_unit

    x = np.arange(1, 4) * 10 * original_unit
    y = np.arange(2) * original_unit
    z = np.arange(3) * original_unit

    ds1 = xr.Dataset(
        data_vars={"a": (("y", "x"), array1), "b": (("y", "x"), array2)},
        coords={"x": x, "y": y, "z": ("x", z)},
    )
    ds2 = xr.Dataset(
        data_vars={
            "a": (("y", "x"), np.ones_like(array1) * data_unit),
            "b": (("y", "x"), np.ones_like(array2) * data_unit),
        },
        coords={
            "x": np.arange(3) * dim_unit,
            "y": np.arange(2, 4) * dim_unit,
            "z": ("x", np.arange(-3, 0) * coord_unit),
        },
    )
    ds3 = xr.Dataset(
        data_vars={
            "a": (("y", "x"), np.zeros_like(array1) * np.nan * data_unit),
            "b": (("y", "x"), np.zeros_like(array2) * np.nan * data_unit),
        },
        coords={
            "x": np.arange(3, 6) * dim_unit,
            "y": np.arange(4, 6) * dim_unit,
            "z": ("x", np.arange(3, 6) * coord_unit),
        },
    )
    ds4 = xr.Dataset(
        data_vars={
            "a": (("y", "x"), -1 * np.ones_like(array1) * data_unit),
            "b": (("y", "x"), -1 * np.ones_like(array2) * data_unit),
        },
        coords={
            "x": np.arange(6, 9) * dim_unit,
            "y": np.arange(6, 8) * dim_unit,
            "z": ("x", np.arange(6, 9) * coord_unit),
        },
    )

    func = function(xr.combine_nested, concat_dim=["x", "y"])
    if error is not None:
        with pytest.raises(error):
            func([[ds1, ds2], [ds3, ds4]])

        return

    units = extract_units(ds1)
    convert_and_strip = lambda ds: strip_units(convert_units(ds, units))
    expected = attach_units(
        func(
            [
                [strip_units(ds1), convert_and_strip(ds2)],
                [convert_and_strip(ds3), convert_and_strip(ds4)],
            ]
        ),
        units,
    )
    actual = func([[ds1, ds2], [ds3, ds4]])

    assert_units_equal(expected, actual)
    assert_identical(expected, actual)


@pytest.mark.parametrize(
    "unit,error",
    (
        pytest.param(1, DimensionalityError, id="no_unit"),
        pytest.param(
            unit_registry.dimensionless, DimensionalityError, id="dimensionless"
        ),
        pytest.param(unit_registry.s, DimensionalityError, id="incompatible_unit"),
        pytest.param(unit_registry.mm, None, id="compatible_unit"),
        pytest.param(unit_registry.m, None, id="identical_unit"),
    ),
    ids=repr,
)
@pytest.mark.parametrize(
    "variant",
    (
        "data",
        pytest.param("dims", marks=pytest.mark.xfail(reason="indexes strip units")),
    ),
)
def test_concat_dataarray(variant, unit, error, dtype):
    original_unit = unit_registry.m

    variants = {"data": (unit, original_unit), "dims": (original_unit, unit)}
    data_unit, dims_unit = variants.get(variant)

    array1 = np.linspace(0, 5, 10).astype(dtype) * unit_registry.m
    array2 = np.linspace(-5, 0, 5).astype(dtype) * data_unit
    x1 = np.arange(5, 15) * original_unit
    x2 = np.arange(5) * dims_unit

    arr1 = xr.DataArray(data=array1, coords={"x": x1}, dims="x")
    arr2 = xr.DataArray(data=array2, coords={"x": x2}, dims="x")

    if error is not None:
        with pytest.raises(error):
            xr.concat([arr1, arr2], dim="x")

        return

    units = extract_units(arr1)
    expected = attach_units(
        xr.concat(
            [strip_units(arr1), strip_units(convert_units(arr2, units))], dim="x"
        ),
        units,
    )
    actual = xr.concat([arr1, arr2], dim="x")

    assert_units_equal(expected, actual)
    assert_identical(expected, actual)


@pytest.mark.parametrize(
    "unit,error",
    (
        pytest.param(1, DimensionalityError, id="no_unit"),
        pytest.param(
            unit_registry.dimensionless, DimensionalityError, id="dimensionless"
        ),
        pytest.param(unit_registry.s, DimensionalityError, id="incompatible_unit"),
        pytest.param(unit_registry.mm, None, id="compatible_unit"),
        pytest.param(unit_registry.m, None, id="identical_unit"),
    ),
    ids=repr,
)
@pytest.mark.parametrize(
    "variant",
    (
        "data",
        pytest.param("dims", marks=pytest.mark.xfail(reason="indexes strip units")),
    ),
)
def test_concat_dataset(variant, unit, error, dtype):
    original_unit = unit_registry.m

    variants = {"data": (unit, original_unit), "dims": (original_unit, unit)}
    data_unit, dims_unit = variants.get(variant)

    array1 = np.linspace(0, 5, 10).astype(dtype) * unit_registry.m
    array2 = np.linspace(-5, 0, 5).astype(dtype) * data_unit
    x1 = np.arange(5, 15) * original_unit
    x2 = np.arange(5) * dims_unit

    ds1 = xr.Dataset(data_vars={"a": ("x", array1)}, coords={"x": x1})
    ds2 = xr.Dataset(data_vars={"a": ("x", array2)}, coords={"x": x2})

    if error is not None:
        with pytest.raises(error):
            xr.concat([ds1, ds2], dim="x")

        return

    units = extract_units(ds1)
    expected = attach_units(
        xr.concat([strip_units(ds1), strip_units(convert_units(ds2, units))], dim="x"),
        units,
    )
    actual = xr.concat([ds1, ds2], dim="x")

    assert_units_equal(expected, actual)
    assert_identical(expected, actual)


@pytest.mark.parametrize(
    "unit,error",
    (
        pytest.param(1, DimensionalityError, id="no_unit"),
        pytest.param(
            unit_registry.dimensionless, DimensionalityError, id="dimensionless"
        ),
        pytest.param(unit_registry.s, DimensionalityError, id="incompatible_unit"),
        pytest.param(unit_registry.mm, None, id="compatible_unit"),
        pytest.param(unit_registry.m, None, id="identical_unit"),
    ),
    ids=repr,
)
@pytest.mark.parametrize(
    "variant",
    (
        "data",
        pytest.param("dims", marks=pytest.mark.xfail(reason="indexes strip units")),
        "coords",
    ),
)
def test_merge_dataarray(variant, unit, error, dtype):
    original_unit = unit_registry.m

    variants = {
        "data": (unit, original_unit, original_unit),
        "dims": (original_unit, unit, original_unit),
        "coords": (original_unit, original_unit, unit),
    }
    data_unit, dim_unit, coord_unit = variants.get(variant)

    array1 = np.linspace(0, 1, 2 * 3).reshape(2, 3).astype(dtype) * original_unit
    x1 = np.arange(2) * original_unit
    y1 = np.arange(3) * original_unit
    u1 = np.linspace(10, 20, 2) * original_unit
    v1 = np.linspace(10, 20, 3) * original_unit

    array2 = np.linspace(1, 2, 2 * 4).reshape(2, 4).astype(dtype) * data_unit
    x2 = np.arange(2, 4) * dim_unit
    z2 = np.arange(4) * original_unit
    u2 = np.linspace(20, 30, 2) * coord_unit
    w2 = np.linspace(10, 20, 4) * original_unit

    array3 = np.linspace(0, 2, 3 * 4).reshape(3, 4).astype(dtype) * data_unit
    y3 = np.arange(3, 6) * dim_unit
    z3 = np.arange(4, 8) * dim_unit
    v3 = np.linspace(10, 20, 3) * coord_unit
    w3 = np.linspace(10, 20, 4) * coord_unit

    arr1 = xr.DataArray(
        name="a",
        data=array1,
        coords={"x": x1, "y": y1, "u": ("x", u1), "v": ("y", v1)},
        dims=("x", "y"),
    )
    arr2 = xr.DataArray(
        name="a",
        data=array2,
        coords={"x": x2, "z": z2, "u": ("x", u2), "w": ("z", w2)},
        dims=("x", "z"),
    )
    arr3 = xr.DataArray(
        name="a",
        data=array3,
        coords={"y": y3, "z": z3, "v": ("y", v3), "w": ("z", w3)},
        dims=("y", "z"),
    )

    if error is not None:
        with pytest.raises(error):
            xr.merge([arr1, arr2, arr3])

        return

    units = {name: original_unit for name in list("axyzuvw")}

    convert_and_strip = lambda arr: strip_units(convert_units(arr, units))
    expected_units = {
        "a": original_unit,
        "u": original_unit,
        "v": original_unit,
        "w": original_unit,
        "x": original_unit,
        "y": original_unit,
        "z": original_unit,
    }

    expected = convert_units(
        attach_units(
            xr.merge(
                [
                    convert_and_strip(arr1),
                    convert_and_strip(arr2),
                    convert_and_strip(arr3),
                ]
            ),
            units,
        ),
        expected_units,
    )

    actual = xr.merge([arr1, arr2, arr3])

    assert_units_equal(expected, actual)
    assert_allclose(expected, actual)


@pytest.mark.parametrize(
    "unit,error",
    (
        pytest.param(1, DimensionalityError, id="no_unit"),
        pytest.param(
            unit_registry.dimensionless, DimensionalityError, id="dimensionless"
        ),
        pytest.param(unit_registry.s, DimensionalityError, id="incompatible_unit"),
        pytest.param(unit_registry.mm, None, id="compatible_unit"),
        pytest.param(unit_registry.m, None, id="identical_unit"),
    ),
    ids=repr,
)
@pytest.mark.parametrize(
    "variant",
    (
        "data",
        pytest.param("dims", marks=pytest.mark.xfail(reason="indexes strip units")),
        "coords",
    ),
)
def test_merge_dataset(variant, unit, error, dtype):
    original_unit = unit_registry.m

    variants = {
        "data": (unit, original_unit, original_unit),
        "dims": (original_unit, unit, original_unit),
        "coords": (original_unit, original_unit, unit),
    }
    data_unit, dim_unit, coord_unit = variants.get(variant)

    array1 = np.zeros(shape=(2, 3), dtype=dtype) * original_unit
    array2 = np.zeros(shape=(2, 3), dtype=dtype) * original_unit

    x = np.arange(11, 14) * original_unit
    y = np.arange(2) * original_unit
    z = np.arange(3) * original_unit

    ds1 = xr.Dataset(
        data_vars={"a": (("y", "x"), array1), "b": (("y", "x"), array2)},
        coords={"x": x, "y": y, "u": ("x", z)},
    )
    ds2 = xr.Dataset(
        data_vars={
            "a": (("y", "x"), np.ones_like(array1) * data_unit),
            "b": (("y", "x"), np.ones_like(array2) * data_unit),
        },
        coords={
            "x": np.arange(3) * dim_unit,
            "y": np.arange(2, 4) * dim_unit,
            "u": ("x", np.arange(-3, 0) * coord_unit),
        },
    )
    ds3 = xr.Dataset(
        data_vars={
            "a": (("y", "x"), np.full_like(array1, np.nan) * data_unit),
            "b": (("y", "x"), np.full_like(array2, np.nan) * data_unit),
        },
        coords={
            "x": np.arange(3, 6) * dim_unit,
            "y": np.arange(4, 6) * dim_unit,
            "u": ("x", np.arange(3, 6) * coord_unit),
        },
    )

    func = function(xr.merge)
    if error is not None:
        with pytest.raises(error):
            func([ds1, ds2, ds3])

        return

    units = extract_units(ds1)
    convert_and_strip = lambda ds: strip_units(convert_units(ds, units))
    expected_units = {name: original_unit for name in list("abxyzu")}
    expected = convert_units(
        attach_units(
            func(
                [convert_and_strip(ds1), convert_and_strip(ds2), convert_and_strip(ds3)]
            ),
            units,
        ),
        expected_units,
    )
    actual = func([ds1, ds2, ds3])

    assert_units_equal(expected, actual)
    assert_allclose(expected, actual)


@pytest.mark.parametrize("func", (xr.zeros_like, xr.ones_like))
def test_replication_dataarray(func, dtype):
    array = np.linspace(0, 10, 20).astype(dtype) * unit_registry.s
    data_array = xr.DataArray(data=array, dims="x")

    numpy_func = getattr(np, func.__name__)
    units = extract_units(numpy_func(data_array))
    expected = attach_units(func(data_array), units)
    actual = func(data_array)

    assert_units_equal(expected, actual)
    assert_identical(expected, actual)


@pytest.mark.parametrize("func", (xr.zeros_like, xr.ones_like))
def test_replication_dataset(func, dtype):
    array1 = np.linspace(0, 10, 20).astype(dtype) * unit_registry.s
    array2 = np.linspace(5, 10, 10).astype(dtype) * unit_registry.Pa
    x = np.arange(20).astype(dtype) * unit_registry.m
    y = np.arange(10).astype(dtype) * unit_registry.m
    z = y.to(unit_registry.mm)

    ds = xr.Dataset(
        data_vars={"a": ("x", array1), "b": ("y", array2)},
        coords={"x": x, "y": y, "z": ("y", z)},
    )

    numpy_func = getattr(np, func.__name__)
    units = extract_units(ds.map(numpy_func))
    expected = attach_units(func(strip_units(ds)), units)

    actual = func(ds)

    assert_units_equal(expected, actual)
    assert_identical(expected, actual)


@pytest.mark.xfail(
    reason=(
        "pint is undecided on how `full_like` should work, so incorrect errors "
        "may be expected: hgrecco/pint#882"
    )
)
@pytest.mark.parametrize(
    "unit,error",
    (
        pytest.param(1, DimensionalityError, id="no_unit"),
        pytest.param(
            unit_registry.dimensionless, DimensionalityError, id="dimensionless"
        ),
        pytest.param(unit_registry.m, DimensionalityError, id="incompatible_unit"),
        pytest.param(unit_registry.ms, None, id="compatible_unit"),
        pytest.param(unit_registry.s, None, id="identical_unit"),
    ),
    ids=repr,
)
def test_replication_full_like_dataarray(unit, error, dtype):
    array = np.linspace(0, 5, 10) * unit_registry.s
    data_array = xr.DataArray(data=array, dims="x")

    fill_value = -1 * unit
    if error is not None:
        with pytest.raises(error):
            xr.full_like(data_array, fill_value=fill_value)

        return

    units = {**extract_units(data_array), **{None: unit if unit != 1 else None}}
    expected = attach_units(
        xr.full_like(strip_units(data_array), fill_value=strip_units(fill_value)), units
    )
    actual = xr.full_like(data_array, fill_value=fill_value)

    assert_units_equal(expected, actual)
    assert_identical(expected, actual)


@pytest.mark.xfail(
    reason=(
        "pint is undecided on how `full_like` should work, so incorrect errors "
        "may be expected: hgrecco/pint#882"
    )
)
@pytest.mark.parametrize(
    "unit,error",
    (
        pytest.param(1, DimensionalityError, id="no_unit"),
        pytest.param(
            unit_registry.dimensionless, DimensionalityError, id="dimensionless"
        ),
        pytest.param(unit_registry.m, DimensionalityError, id="incompatible_unit"),
        pytest.param(unit_registry.ms, None, id="compatible_unit"),
        pytest.param(unit_registry.s, None, id="identical_unit"),
    ),
    ids=repr,
)
def test_replication_full_like_dataset(unit, error, dtype):
    array1 = np.linspace(0, 10, 20).astype(dtype) * unit_registry.s
    array2 = np.linspace(5, 10, 10).astype(dtype) * unit_registry.Pa
    x = np.arange(20).astype(dtype) * unit_registry.m
    y = np.arange(10).astype(dtype) * unit_registry.m
    z = y.to(unit_registry.mm)

    ds = xr.Dataset(
        data_vars={"a": ("x", array1), "b": ("y", array2)},
        coords={"x": x, "y": y, "z": ("y", z)},
    )

    fill_value = -1 * unit
    if error is not None:
        with pytest.raises(error):
            xr.full_like(ds, fill_value=fill_value)

        return

    units = {
        **extract_units(ds),
        **{name: unit if unit != 1 else None for name in ds.data_vars},
    }
    expected = attach_units(
        xr.full_like(strip_units(ds), fill_value=strip_units(fill_value)), units
    )
    actual = xr.full_like(ds, fill_value=fill_value)

    assert_units_equal(expected, actual)
    assert_identical(expected, actual)


@pytest.mark.parametrize(
    "unit,error",
    (
        pytest.param(1, DimensionalityError, id="no_unit"),
        pytest.param(
            unit_registry.dimensionless, DimensionalityError, id="dimensionless"
        ),
        pytest.param(unit_registry.s, DimensionalityError, id="incompatible_unit"),
        pytest.param(unit_registry.mm, None, id="compatible_unit"),
        pytest.param(unit_registry.m, None, id="identical_unit"),
    ),
    ids=repr,
)
@pytest.mark.parametrize("fill_value", (np.nan, 10.2))
def test_where_dataarray(fill_value, unit, error, dtype):
    array = np.linspace(0, 5, 10).astype(dtype) * unit_registry.m

    x = xr.DataArray(data=array, dims="x")
    cond = x < 5 * unit_registry.m
    fill_value = fill_value * unit

    if error is not None and not (
        np.isnan(fill_value) and not isinstance(fill_value, Quantity)
    ):
        with pytest.raises(error):
            xr.where(cond, x, fill_value)

        return

    expected = attach_units(
        xr.where(
            cond,
            strip_units(x),
            strip_units(convert_units(fill_value, {None: unit_registry.m})),
        ),
        extract_units(x),
    )
    actual = xr.where(cond, x, fill_value)

    assert_units_equal(expected, actual)
    assert_identical(expected, actual)


@pytest.mark.parametrize(
    "unit,error",
    (
        pytest.param(1, DimensionalityError, id="no_unit"),
        pytest.param(
            unit_registry.dimensionless, DimensionalityError, id="dimensionless"
        ),
        pytest.param(unit_registry.s, DimensionalityError, id="incompatible_unit"),
        pytest.param(unit_registry.mm, None, id="compatible_unit"),
        pytest.param(unit_registry.m, None, id="identical_unit"),
    ),
    ids=repr,
)
@pytest.mark.parametrize("fill_value", (np.nan, 10.2))
def test_where_dataset(fill_value, unit, error, dtype):
    array1 = np.linspace(0, 5, 10).astype(dtype) * unit_registry.m
    array2 = np.linspace(-5, 0, 10).astype(dtype) * unit_registry.m
    x = np.arange(10) * unit_registry.s

    ds = xr.Dataset(data_vars={"a": ("x", array1), "b": ("x", array2)}, coords={"x": x})
    cond = x < 5 * unit_registry.s
    fill_value = fill_value * unit

    if error is not None and not (
        np.isnan(fill_value) and not isinstance(fill_value, Quantity)
    ):
        with pytest.raises(error):
            xr.where(cond, ds, fill_value)

        return

    expected = attach_units(
        xr.where(
            cond,
            strip_units(ds),
            strip_units(convert_units(fill_value, {None: unit_registry.m})),
        ),
        extract_units(ds),
    )
    actual = xr.where(cond, ds, fill_value)

    assert_units_equal(expected, actual)
    assert_identical(expected, actual)


def test_dot_dataarray(dtype):
    array1 = (
        np.linspace(0, 10, 5 * 10).reshape(5, 10).astype(dtype)
        * unit_registry.m
        / unit_registry.s
    )
    array2 = (
        np.linspace(10, 20, 10 * 20).reshape(10, 20).astype(dtype) * unit_registry.s
    )

    data_array = xr.DataArray(data=array1, dims=("x", "y"))
    other = xr.DataArray(data=array2, dims=("y", "z"))

    expected = attach_units(
        xr.dot(strip_units(data_array), strip_units(other)), {None: unit_registry.m}
    )
    actual = xr.dot(data_array, other)

    assert_units_equal(expected, actual)
    assert_identical(expected, actual)


def delete_attrs(*to_delete):
    def wrapper(cls):
        for item in to_delete:
            setattr(cls, item, None)

        return cls

    return wrapper


@delete_attrs(
    "test_getitem_with_mask",
    "test_getitem_with_mask_nd_indexer",
    "test_index_0d_string",
    "test_index_0d_datetime",
    "test_index_0d_timedelta64",
    "test_0d_time_data",
    "test_index_0d_not_a_time",
    "test_datetime64_conversion",
    "test_timedelta64_conversion",
    "test_pandas_period_index",
    "test_1d_math",
    "test_1d_reduce",
    "test_array_interface",
    "test___array__",
    "test_copy_index",
    "test_concat_number_strings",
    "test_concat_fixed_len_str",
    "test_concat_mixed_dtypes",
    "test_pandas_datetime64_with_tz",
    "test_pandas_data",
    "test_multiindex",
)
class TestVariable(VariableSubclassobjects):
    @staticmethod
    def cls(dims, data, *args, **kwargs):
        return xr.Variable(
            dims, unit_registry.Quantity(data, unit_registry.m), *args, **kwargs
        )

    def example_1d_objects(self):
        for data in [
            range(3),
            0.5 * np.arange(3),
            0.5 * np.arange(3, dtype=np.float32),
            np.array(["a", "b", "c"], dtype=object),
        ]:
            yield (self.cls("x", data), data)

    @pytest.mark.parametrize(
        "func",
        (
            method("all"),
            method("any"),
            method("argmax"),
            method("argmin"),
            method("argsort"),
            method("cumprod"),
            method("cumsum"),
            method("max"),
            method("mean"),
            method("median"),
            method("min"),
            pytest.param(
                method("prod"),
                marks=pytest.mark.xfail(reason="not implemented by pint"),
            ),
            method("std"),
            method("sum"),
            method("var"),
        ),
        ids=repr,
    )
    def test_aggregation(self, func, dtype):
        array = np.linspace(0, 1, 10).astype(dtype) * (
            unit_registry.m if func.name != "cumprod" else unit_registry.dimensionless
        )
        variable = xr.Variable("x", array)

        units = extract_units(func(array))
        expected = attach_units(func(strip_units(variable)), units)
        actual = func(variable)

        assert_units_equal(expected, actual)
        xr.testing.assert_identical(expected, actual)

    @pytest.mark.parametrize(
        "func",
        (
            method("astype", np.float32),
            method("conj"),
            method("conjugate"),
            method("clip", min=2, max=7),
        ),
        ids=repr,
    )
    @pytest.mark.parametrize(
        "unit,error",
        (
            pytest.param(1, DimensionalityError, id="no_unit"),
            pytest.param(
                unit_registry.dimensionless, DimensionalityError, id="dimensionless"
            ),
            pytest.param(unit_registry.s, DimensionalityError, id="incompatible_unit"),
            pytest.param(unit_registry.cm, None, id="compatible_unit"),
            pytest.param(unit_registry.m, None, id="identical_unit"),
        ),
    )
    def test_numpy_methods(self, func, unit, error, dtype):
        array = np.linspace(0, 1, 10).astype(dtype) * unit_registry.m
        variable = xr.Variable("x", array)

        args = [
            item * unit if isinstance(item, (int, float, list)) else item
            for item in func.args
        ]
        kwargs = {
            key: value * unit if isinstance(value, (int, float, list)) else value
            for key, value in func.kwargs.items()
        }

        if error is not None and func.name in ("searchsorted", "clip"):
            with pytest.raises(error):
                func(variable, *args, **kwargs)

            return

        converted_args = [
            strip_units(convert_units(item, {None: unit_registry.m})) for item in args
        ]
        converted_kwargs = {
            key: strip_units(convert_units(value, {None: unit_registry.m}))
            for key, value in kwargs.items()
        }

        units = extract_units(func(array, *args, **kwargs))
        expected = attach_units(
            func(strip_units(variable), *converted_args, **converted_kwargs), units
        )
        actual = func(variable, *args, **kwargs)

        assert_units_equal(expected, actual)
        xr.testing.assert_allclose(expected, actual)

    @pytest.mark.parametrize(
        "func", (method("item", 5), method("searchsorted", 5)), ids=repr
    )
    @pytest.mark.parametrize(
        "unit,error",
        (
            pytest.param(1, DimensionalityError, id="no_unit"),
            pytest.param(
                unit_registry.dimensionless, DimensionalityError, id="dimensionless"
            ),
            pytest.param(unit_registry.s, DimensionalityError, id="incompatible_unit"),
            pytest.param(unit_registry.cm, None, id="compatible_unit"),
            pytest.param(unit_registry.m, None, id="identical_unit"),
        ),
    )
    def test_raw_numpy_methods(self, func, unit, error, dtype):
        array = np.linspace(0, 1, 10).astype(dtype) * unit_registry.m
        variable = xr.Variable("x", array)

        args = [
            item * unit
            if isinstance(item, (int, float, list)) and func.name != "item"
            else item
            for item in func.args
        ]
        kwargs = {
            key: value * unit
            if isinstance(value, (int, float, list)) and func.name != "item"
            else value
            for key, value in func.kwargs.items()
        }

        if error is not None and func.name != "item":
            with pytest.raises(error):
                func(variable, *args, **kwargs)

            return

        converted_args = [
            strip_units(convert_units(item, {None: unit_registry.m}))
            if func.name != "item"
            else item
            for item in args
        ]
        converted_kwargs = {
            key: strip_units(convert_units(value, {None: unit_registry.m}))
            if func.name != "item"
            else value
            for key, value in kwargs.items()
        }

        units = extract_units(func(array, *args, **kwargs))
        expected = attach_units(
            func(strip_units(variable), *converted_args, **converted_kwargs), units
        )
        actual = func(variable, *args, **kwargs)

        assert_units_equal(expected, actual)
        np.testing.assert_allclose(expected, actual)

    @pytest.mark.parametrize(
        "func", (method("isnull"), method("notnull"), method("count")), ids=repr
    )
    def test_missing_value_detection(self, func):
        array = (
            np.array(
                [
                    [1.4, 2.3, np.nan, 7.2],
                    [np.nan, 9.7, np.nan, np.nan],
                    [2.1, np.nan, np.nan, 4.6],
                    [9.9, np.nan, 7.2, 9.1],
                ]
            )
            * unit_registry.degK
        )
        variable = xr.Variable(("x", "y"), array)

        expected = func(strip_units(variable))
        actual = func(variable)

        assert_units_equal(expected, actual)
        xr.testing.assert_identical(expected, actual)

    @pytest.mark.parametrize(
        "unit,error",
        (
            pytest.param(1, DimensionalityError, id="no_unit"),
            pytest.param(
                unit_registry.dimensionless, DimensionalityError, id="dimensionless"
            ),
            pytest.param(unit_registry.s, DimensionalityError, id="incompatible_unit"),
            pytest.param(unit_registry.cm, None, id="compatible_unit"),
            pytest.param(unit_registry.m, None, id="identical_unit"),
        ),
    )
    def test_missing_value_fillna(self, unit, error):
        value = 10
        array = (
            np.array(
                [
                    [1.4, 2.3, np.nan, 7.2],
                    [np.nan, 9.7, np.nan, np.nan],
                    [2.1, np.nan, np.nan, 4.6],
                    [9.9, np.nan, 7.2, 9.1],
                ]
            )
            * unit_registry.m
        )
        variable = xr.Variable(("x", "y"), array)

        fill_value = value * unit

        if error is not None:
            with pytest.raises(error):
                variable.fillna(value=fill_value)

            return

        expected = attach_units(
            strip_units(variable).fillna(
                value=fill_value.to(unit_registry.m).magnitude
            ),
            extract_units(variable),
        )
        actual = variable.fillna(value=fill_value)

        assert_units_equal(expected, actual)
        xr.testing.assert_identical(expected, actual)

    @pytest.mark.parametrize(
        "unit",
        (
            pytest.param(1, id="no_unit"),
            pytest.param(unit_registry.dimensionless, id="dimensionless"),
            pytest.param(unit_registry.s, id="incompatible_unit"),
            pytest.param(unit_registry.cm, id="compatible_unit",),
            pytest.param(unit_registry.m, id="identical_unit"),
        ),
    )
    @pytest.mark.parametrize(
        "convert_data",
        (
            pytest.param(False, id="no_conversion"),
            pytest.param(True, id="with_conversion"),
        ),
    )
    @pytest.mark.parametrize(
        "func",
        (
            method("equals"),
            pytest.param(
                method("identical"),
                marks=pytest.mark.skip(reason="behaviour of identical is unclear"),
            ),
        ),
        ids=repr,
    )
    def test_comparisons(self, func, unit, convert_data, dtype):
        array = np.linspace(0, 1, 9).astype(dtype)
        quantity1 = array * unit_registry.m
        variable = xr.Variable("x", quantity1)

        if convert_data and is_compatible(unit_registry.m, unit):
            quantity2 = convert_units(array * unit_registry.m, {None: unit})
        else:
            quantity2 = array * unit
        other = xr.Variable("x", quantity2)

        expected = func(
            strip_units(variable),
            strip_units(
                convert_units(other, extract_units(variable))
                if is_compatible(unit_registry.m, unit)
                else other
            ),
        )
        if func.name == "identical":
            expected &= extract_units(variable) == extract_units(other)
        else:
            expected &= all(
                compatible_mappings(
                    extract_units(variable), extract_units(other)
                ).values()
            )

        actual = func(variable, other)

        assert expected == actual

    @pytest.mark.parametrize(
        "unit",
        (
            pytest.param(1, id="no_unit"),
            pytest.param(unit_registry.dimensionless, id="dimensionless"),
            pytest.param(unit_registry.s, id="incompatible_unit"),
            pytest.param(unit_registry.cm, id="compatible_unit"),
            pytest.param(unit_registry.m, id="identical_unit"),
        ),
    )
    def test_broadcast_equals(self, unit, dtype):
        base_unit = unit_registry.m
        left_array = np.ones(shape=(2, 2), dtype=dtype) * base_unit
        value = (
            (1 * base_unit).to(unit).magnitude if is_compatible(unit, base_unit) else 1
        )
        right_array = np.full(shape=(2,), fill_value=value, dtype=dtype) * unit

        left = xr.Variable(("x", "y"), left_array)
        right = xr.Variable("x", right_array)

        units = {
            **extract_units(left),
            **({} if is_compatible(unit, base_unit) else {None: None}),
        }
        expected = strip_units(left).broadcast_equals(
            strip_units(convert_units(right, units))
        ) & is_compatible(unit, base_unit)
        actual = left.broadcast_equals(right)

        assert expected == actual

    @pytest.mark.parametrize(
        "indices",
        (
            pytest.param(4, id="single index"),
            pytest.param([5, 2, 9, 1], id="multiple indices"),
        ),
    )
    def test_isel(self, indices, dtype):
        array = np.linspace(0, 5, 10).astype(dtype) * unit_registry.s
        variable = xr.Variable("x", array)

        expected = attach_units(
            strip_units(variable).isel(x=indices), extract_units(variable)
        )
        actual = variable.isel(x=indices)

        assert_units_equal(expected, actual)
        xr.testing.assert_identical(expected, actual)

    @pytest.mark.parametrize(
        "unit,error",
        (
            pytest.param(1, DimensionalityError, id="no_unit"),
            pytest.param(
                unit_registry.dimensionless, DimensionalityError, id="dimensionless"
            ),
            pytest.param(unit_registry.s, DimensionalityError, id="incompatible_unit"),
            pytest.param(unit_registry.cm, None, id="compatible_unit"),
            pytest.param(unit_registry.m, None, id="identical_unit"),
        ),
    )
    @pytest.mark.parametrize(
        "func",
        (
            function(lambda x, *_: +x, function_label="unary_plus"),
            function(lambda x, *_: -x, function_label="unary_minus"),
            function(lambda x, *_: abs(x), function_label="absolute"),
            function(lambda x, y: x + y, function_label="sum"),
            function(lambda x, y: y + x, function_label="commutative_sum"),
            function(lambda x, y: x * y, function_label="product"),
            function(lambda x, y: y * x, function_label="commutative_product"),
        ),
        ids=repr,
    )
    def test_1d_math(self, func, unit, error, dtype):
        base_unit = unit_registry.m
        array = np.arange(5).astype(dtype) * base_unit
        variable = xr.Variable("x", array)

        values = np.ones(5)
        y = values * unit

        if error is not None and func.name in ("sum", "commutative_sum"):
            with pytest.raises(error):
                func(variable, y)

            return

        units = extract_units(func(array, y))
        if all(compatible_mappings(units, extract_units(y)).values()):
            converted_y = convert_units(y, units)
        else:
            converted_y = y

        if all(compatible_mappings(units, extract_units(variable)).values()):
            converted_variable = convert_units(variable, units)
        else:
            converted_variable = variable

        expected = attach_units(
            func(strip_units(converted_variable), strip_units(converted_y)), units
        )
        actual = func(variable, y)

        assert_units_equal(expected, actual)
        xr.testing.assert_allclose(expected, actual)

    @pytest.mark.parametrize(
        "unit,error",
        (
            pytest.param(1, DimensionalityError, id="no_unit"),
            pytest.param(
                unit_registry.dimensionless, DimensionalityError, id="dimensionless"
            ),
            pytest.param(unit_registry.s, DimensionalityError, id="incompatible_unit"),
            pytest.param(unit_registry.cm, None, id="compatible_unit"),
            pytest.param(unit_registry.m, None, id="identical_unit"),
        ),
    )
    @pytest.mark.parametrize(
        "func", (method("where"), method("_getitem_with_mask")), ids=repr
    )
    def test_masking(self, func, unit, error, dtype):
        base_unit = unit_registry.m
        array = np.linspace(0, 5, 10).astype(dtype) * base_unit
        variable = xr.Variable("x", array)
        cond = np.array([True, False] * 5)

        other = -1 * unit

        if error is not None:
            with pytest.raises(error):
                func(variable, cond, other)

            return

        expected = attach_units(
            func(
                strip_units(variable),
                cond,
                strip_units(
                    convert_units(
                        other,
                        {None: base_unit}
                        if is_compatible(base_unit, unit)
                        else {None: None},
                    )
                ),
            ),
            extract_units(variable),
        )
        actual = func(variable, cond, other)

        assert_units_equal(expected, actual)
        xr.testing.assert_identical(expected, actual)

    def test_squeeze(self, dtype):
        shape = (2, 1, 3, 1, 1, 2)
        names = list("abcdef")
        array = np.ones(shape=shape) * unit_registry.m
        variable = xr.Variable(names, array)

        expected = attach_units(
            strip_units(variable).squeeze(), extract_units(variable)
        )
        actual = variable.squeeze()

        assert_units_equal(expected, actual)
        xr.testing.assert_identical(expected, actual)

        names = tuple(name for name, size in zip(names, shape) if shape == 1)
        for name in names:
            expected = attach_units(
                strip_units(variable).squeeze(dim=name), extract_units(variable)
            )
            actual = variable.squeeze(dim=name)

            assert_units_equal(expected, actual)
            xr.testing.assert_identical(expected, actual)

    @pytest.mark.parametrize(
        "func",
        (
            method("coarsen", windows={"y": 2}, func=np.mean),
            pytest.param(
                method("quantile", q=[0.25, 0.75]),
                marks=pytest.mark.xfail(reason="nanquantile not implemented"),
            ),
            pytest.param(
                method("rank", dim="x"),
                marks=pytest.mark.xfail(reason="rank not implemented for non-ndarray"),
            ),
            method("roll", {"x": 2}),
            pytest.param(
                method("rolling_window", "x", 3, "window"),
                marks=pytest.mark.xfail(reason="converts to ndarray"),
            ),
            method("reduce", np.std, "x"),
            method("round", 2),
            method("shift", {"x": -2}),
            method("transpose", "y", "x"),
        ),
        ids=repr,
    )
    def test_computation(self, func, dtype):
        base_unit = unit_registry.m
        array = np.linspace(0, 5, 5 * 10).reshape(5, 10).astype(dtype) * base_unit
        variable = xr.Variable(("x", "y"), array)

        expected = attach_units(func(strip_units(variable)), extract_units(variable))

        actual = func(variable)

        assert_units_equal(expected, actual)
        xr.testing.assert_identical(expected, actual)

    @pytest.mark.parametrize(
        "unit,error",
        (
            pytest.param(1, DimensionalityError, id="no_unit"),
            pytest.param(
                unit_registry.dimensionless, DimensionalityError, id="dimensionless"
            ),
            pytest.param(unit_registry.s, DimensionalityError, id="incompatible_unit"),
            pytest.param(unit_registry.cm, None, id="compatible_unit"),
            pytest.param(unit_registry.m, None, id="identical_unit"),
        ),
    )
    def test_searchsorted(self, unit, error, dtype):
        base_unit = unit_registry.m
        array = np.linspace(0, 5, 10).astype(dtype) * base_unit
        variable = xr.Variable("x", array)

        value = 0 * unit

        if error is not None:
            with pytest.raises(error):
                variable.searchsorted(value)

            return

        expected = strip_units(variable).searchsorted(
            strip_units(convert_units(value, {None: base_unit}))
        )

        actual = variable.searchsorted(value)

        assert_units_equal(expected, actual)
        np.testing.assert_allclose(expected, actual)

    def test_stack(self, dtype):
        array = np.linspace(0, 5, 3 * 10).reshape(3, 10).astype(dtype) * unit_registry.m
        variable = xr.Variable(("x", "y"), array)

        expected = attach_units(
            strip_units(variable).stack(z=("x", "y")), extract_units(variable)
        )
        actual = variable.stack(z=("x", "y"))

        assert_units_equal(expected, actual)
        xr.testing.assert_identical(expected, actual)

    def test_unstack(self, dtype):
        array = np.linspace(0, 5, 3 * 10).astype(dtype) * unit_registry.m
        variable = xr.Variable("z", array)

        expected = attach_units(
            strip_units(variable).unstack(z={"x": 3, "y": 10}), extract_units(variable)
        )
        actual = variable.unstack(z={"x": 3, "y": 10})

        assert_units_equal(expected, actual)
        xr.testing.assert_identical(expected, actual)

    @pytest.mark.parametrize(
        "unit,error",
        (
            pytest.param(1, DimensionalityError, id="no_unit"),
            pytest.param(
                unit_registry.dimensionless, DimensionalityError, id="dimensionless"
            ),
            pytest.param(unit_registry.s, DimensionalityError, id="incompatible_unit"),
            pytest.param(unit_registry.cm, None, id="compatible_unit"),
            pytest.param(unit_registry.m, None, id="identical_unit"),
        ),
    )
    def test_concat(self, unit, error, dtype):
        array1 = (
            np.linspace(0, 5, 9 * 10).reshape(3, 6, 5).astype(dtype) * unit_registry.m
        )
        array2 = np.linspace(5, 10, 10 * 3).reshape(3, 2, 5).astype(dtype) * unit

        variable = xr.Variable(("x", "y", "z"), array1)
        other = xr.Variable(("x", "y", "z"), array2)

        if error is not None:
            with pytest.raises(error):
                xr.Variable.concat([variable, other], dim="y")

            return

        units = extract_units(variable)
        expected = attach_units(
            xr.Variable.concat(
                [strip_units(variable), strip_units(convert_units(other, units))],
                dim="y",
            ),
            units,
        )
        actual = xr.Variable.concat([variable, other], dim="y")

        assert_units_equal(expected, actual)
        xr.testing.assert_identical(expected, actual)

    def test_set_dims(self, dtype):
        array = np.linspace(0, 5, 3 * 10).reshape(3, 10).astype(dtype) * unit_registry.m
        variable = xr.Variable(("x", "y"), array)

        dims = {"z": 6, "x": 3, "a": 1, "b": 4, "y": 10}
        expected = attach_units(
            strip_units(variable).set_dims(dims), extract_units(variable)
        )
        actual = variable.set_dims(dims)

        assert_units_equal(expected, actual)
        xr.testing.assert_identical(expected, actual)

    def test_copy(self, dtype):
        array = np.linspace(0, 5, 10).astype(dtype) * unit_registry.m
        other = np.arange(10).astype(dtype) * unit_registry.s

        variable = xr.Variable("x", array)
        expected = attach_units(
            strip_units(variable).copy(data=strip_units(other)), extract_units(other)
        )
        actual = variable.copy(data=other)

        assert_units_equal(expected, actual)
        xr.testing.assert_identical(expected, actual)

    @pytest.mark.parametrize(
        "unit",
        (
            pytest.param(1, id="no_unit"),
            pytest.param(unit_registry.dimensionless, id="dimensionless"),
            pytest.param(unit_registry.s, id="incompatible_unit"),
            pytest.param(unit_registry.cm, id="compatible_unit"),
            pytest.param(unit_registry.m, id="identical_unit"),
        ),
    )
    def test_no_conflicts(self, unit, dtype):
        base_unit = unit_registry.m
        array1 = (
            np.array(
                [
                    [6.3, 0.3, 0.45],
                    [np.nan, 0.3, 0.3],
                    [3.7, np.nan, 0.2],
                    [9.43, 0.3, 0.7],
                ]
            )
            * base_unit
        )
        array2 = np.array([np.nan, 0.3, np.nan]) * unit

        variable = xr.Variable(("x", "y"), array1)
        other = xr.Variable("y", array2)

        expected = strip_units(variable).no_conflicts(
            strip_units(
                convert_units(
                    other, {None: base_unit if is_compatible(base_unit, unit) else None}
                )
            )
        ) & is_compatible(base_unit, unit)
        actual = variable.no_conflicts(other)

        assert expected == actual

    @pytest.mark.parametrize("xr_arg, np_arg", _PAD_XR_NP_ARGS)
    def test_pad_constant_values(self, dtype, xr_arg, np_arg):
        data = np.arange(4 * 3 * 2).reshape(4, 3, 2).astype(dtype) * unit_registry.m
        v = xr.Variable(["x", "y", "z"], data)

        actual = v.pad(**xr_arg, mode="constant")
        expected = xr.Variable(
            v.dims,
            np.pad(
                v.data.astype(float), np_arg, mode="constant", constant_values=np.nan,
            ),
        )
        xr.testing.assert_identical(expected, actual)
        assert_units_equal(expected, actual)
        assert isinstance(actual._data, type(v._data))

        # for the boolean array, we pad False
        data = np.full_like(data, False, dtype=bool).reshape(4, 3, 2)
        v = xr.Variable(["x", "y", "z"], data)
        actual = v.pad(**xr_arg, mode="constant", constant_values=data.flat[0])
        expected = xr.Variable(
            v.dims,
            np.pad(v.data, np_arg, mode="constant", constant_values=v.data.flat[0]),
        )
        xr.testing.assert_identical(actual, expected)
        assert_units_equal(expected, actual)

    @pytest.mark.parametrize(
        "unit,error",
        (
            pytest.param(
                1,
                DimensionalityError,
                id="no_unit",
                marks=pytest.mark.xfail(
                    LooseVersion(pint.__version__) < LooseVersion("0.10.2"),
                    reason="bug in pint's implementation of np.pad",
                ),
            ),
            pytest.param(
                unit_registry.dimensionless, DimensionalityError, id="dimensionless"
            ),
            pytest.param(unit_registry.s, DimensionalityError, id="incompatible_unit"),
            pytest.param(unit_registry.cm, None, id="compatible_unit"),
            pytest.param(unit_registry.m, None, id="identical_unit"),
        ),
    )
    def test_pad_unit_constant_value(self, unit, error, dtype):
        array = np.linspace(0, 5, 3 * 10).reshape(3, 10).astype(dtype) * unit_registry.m
        variable = xr.Variable(("x", "y"), array)

        fill_value = -100 * unit

        func = method("pad", mode="constant", x=(2, 3), y=(1, 4))
        if error is not None:
            with pytest.raises(error):
                func(variable, constant_values=fill_value)

            return

        units = extract_units(variable)
        expected = attach_units(
            func(
                strip_units(variable),
                constant_values=strip_units(convert_units(fill_value, units)),
            ),
            units,
        )
        actual = func(variable, constant_values=fill_value)

        assert_units_equal(expected, actual)
        xr.testing.assert_identical(expected, actual)


class TestDataArray:
    @pytest.mark.filterwarnings("error:::pint[.*]")
    @pytest.mark.parametrize(
        "variant",
        (
            pytest.param(
                "with_dims",
                marks=pytest.mark.xfail(reason="units in indexes are not supported"),
            ),
            pytest.param("with_coords"),
            pytest.param("without_coords"),
        ),
    )
    def test_init(self, variant, dtype):
        array = np.linspace(1, 2, 10, dtype=dtype) * unit_registry.m

        x = np.arange(len(array)) * unit_registry.s
        y = x.to(unit_registry.ms)

        variants = {
            "with_dims": {"x": x},
            "with_coords": {"y": ("x", y)},
            "without_coords": {},
        }

        kwargs = {"data": array, "dims": "x", "coords": variants.get(variant)}
        data_array = xr.DataArray(**kwargs)

        assert isinstance(data_array.data, Quantity)
        assert all(
            {
                name: isinstance(coord.data, Quantity)
                for name, coord in data_array.coords.items()
            }.values()
        )

    @pytest.mark.filterwarnings("error:::pint[.*]")
    @pytest.mark.parametrize(
        "func", (pytest.param(str, id="str"), pytest.param(repr, id="repr"))
    )
    @pytest.mark.parametrize(
        "variant",
        (
            pytest.param(
                "with_dims",
                marks=pytest.mark.xfail(reason="units in indexes are not supported"),
            ),
            pytest.param("with_coords"),
            pytest.param("without_coords"),
        ),
    )
    def test_repr(self, func, variant, dtype):
        array = np.linspace(1, 2, 10, dtype=dtype) * unit_registry.m
        x = np.arange(len(array)) * unit_registry.s
        y = x.to(unit_registry.ms)

        variants = {
            "with_dims": {"x": x},
            "with_coords": {"y": ("x", y)},
            "without_coords": {},
        }

        kwargs = {"data": array, "dims": "x", "coords": variants.get(variant)}
        data_array = xr.DataArray(**kwargs)

        # FIXME: this just checks that the repr does not raise
        # warnings or errors, but does not check the result
        func(data_array)

    @pytest.mark.parametrize(
        "func",
        (
            pytest.param(
                function("all"),
                marks=pytest.mark.xfail(reason="not implemented by pint yet"),
            ),
            pytest.param(
                function("any"),
                marks=pytest.mark.xfail(reason="not implemented by pint yet"),
            ),
            function("argmax"),
            function("argmin"),
            function("max"),
            function("mean"),
            pytest.param(
                function("median"),
                marks=pytest.mark.xfail(reason="not implemented by xarray"),
            ),
            function("min"),
            pytest.param(
                function("prod"),
                marks=pytest.mark.xfail(reason="not implemented by pint yet"),
            ),
            function("sum"),
            function("std"),
            function("var"),
            function("cumsum"),
            pytest.param(
                function("cumprod"),
                marks=pytest.mark.xfail(reason="not implemented by pint yet"),
            ),
            pytest.param(
                method("all"),
                marks=pytest.mark.xfail(reason="not implemented by pint yet"),
            ),
            pytest.param(
                method("any"),
                marks=pytest.mark.xfail(reason="not implemented by pint yet"),
            ),
            method("argmax"),
            method("argmin"),
            method("max"),
            method("mean"),
            method("median"),
            method("min"),
            pytest.param(
                method("prod"),
                marks=pytest.mark.xfail(
                    reason="comparison of quantity with ndarrays in nanops not implemented"
                ),
            ),
            method("sum"),
            method("std"),
            method("var"),
            method("cumsum"),
            pytest.param(
                method("cumprod"),
                marks=pytest.mark.xfail(reason="pint does not implement cumprod yet"),
            ),
        ),
        ids=repr,
    )
    def test_aggregation(self, func, dtype):
        array = np.arange(10).astype(dtype) * (
            unit_registry.m if func.name != "cumprod" else unit_registry.dimensionless
        )
        data_array = xr.DataArray(data=array, dims="x")

        # units differ based on the applied function, so we need to
        # first compute the units
        units = extract_units(func(array))
        expected = attach_units(func(strip_units(data_array)), units)
        actual = func(data_array)

        assert_equal_with_units(expected, actual)

    @pytest.mark.parametrize(
        "func",
        (
            pytest.param(operator.neg, id="negate"),
            pytest.param(abs, id="absolute"),
            pytest.param(np.round, id="round"),
        ),
    )
    def test_unary_operations(self, func, dtype):
        array = np.arange(10).astype(dtype) * unit_registry.m
        data_array = xr.DataArray(data=array)

        units = extract_units(func(array))
        expected = attach_units(func(strip_units(data_array)), units)
        actual = func(data_array)

        assert_equal_with_units(expected, actual)

    @pytest.mark.parametrize(
        "func",
        (
            pytest.param(lambda x: 2 * x, id="multiply"),
            pytest.param(lambda x: x + x, id="add"),
            pytest.param(lambda x: x[0] + x, id="add scalar"),
            pytest.param(lambda x: x.T @ x, id="matrix multiply"),
        ),
    )
    def test_binary_operations(self, func, dtype):
        array = np.arange(10).astype(dtype) * unit_registry.m
        data_array = xr.DataArray(data=array)

        units = extract_units(func(array))
        expected = attach_units(func(strip_units(data_array)), units)
        actual = func(data_array)

        assert_equal_with_units(expected, actual)

    @pytest.mark.parametrize(
        "comparison",
        (
            pytest.param(operator.lt, id="less_than"),
            pytest.param(operator.ge, id="greater_equal"),
            pytest.param(operator.eq, id="equal"),
        ),
    )
    @pytest.mark.parametrize(
        "unit,error",
        (
            pytest.param(1, ValueError, id="without_unit"),
            pytest.param(
                unit_registry.dimensionless, DimensionalityError, id="dimensionless"
            ),
            pytest.param(unit_registry.s, DimensionalityError, id="incompatible_unit"),
            pytest.param(unit_registry.mm, None, id="compatible_unit"),
            pytest.param(unit_registry.m, None, id="identical_unit"),
        ),
    )
    def test_comparison_operations(self, comparison, unit, error, dtype):
        array = (
            np.array([10.1, 5.2, 6.5, 8.0, 21.3, 7.1, 1.3]).astype(dtype)
            * unit_registry.m
        )
        data_array = xr.DataArray(data=array)

        value = 8
        to_compare_with = value * unit

        # incompatible units are all not equal
        if error is not None and comparison is not operator.eq:
            with pytest.raises(error):
                comparison(array, to_compare_with)

            with pytest.raises(error):
                comparison(data_array, to_compare_with)

            return

        actual = comparison(data_array, to_compare_with)

        expected_units = {None: unit_registry.m if array.check(unit) else None}
        expected = array.check(unit) & comparison(
            strip_units(data_array),
            strip_units(convert_units(to_compare_with, expected_units)),
        )

        assert_equal_with_units(expected, actual)

    @pytest.mark.parametrize(
        "units,error",
        (
            pytest.param(unit_registry.dimensionless, None, id="dimensionless"),
            pytest.param(unit_registry.m, DimensionalityError, id="incompatible_unit"),
            pytest.param(unit_registry.degree, None, id="compatible_unit"),
        ),
    )
    def test_univariate_ufunc(self, units, error, dtype):
        array = np.arange(10).astype(dtype) * units
        data_array = xr.DataArray(data=array)

        func = function("sin")

        if error is not None:
            with pytest.raises(error):
                np.sin(data_array)

            return

        expected = attach_units(
            func(strip_units(convert_units(data_array, {None: unit_registry.radians}))),
            {None: unit_registry.dimensionless},
        )
        actual = func(data_array)

        assert_equal_with_units(expected, actual)

    @pytest.mark.xfail(reason="xarray's `np.maximum` strips units")
    @pytest.mark.parametrize(
        "unit,error",
        (
            pytest.param(1, DimensionalityError, id="without_unit"),
            pytest.param(
                unit_registry.dimensionless, DimensionalityError, id="dimensionless"
            ),
            pytest.param(unit_registry.s, DimensionalityError, id="incompatible_unit"),
            pytest.param(unit_registry.mm, None, id="compatible_unit"),
            pytest.param(unit_registry.m, None, id="identical_unit"),
        ),
    )
    def test_bivariate_ufunc(self, unit, error, dtype):
        original_unit = unit_registry.m
        array = np.arange(10).astype(dtype) * original_unit
        data_array = xr.DataArray(data=array)

        if error is not None:
            with pytest.raises(error):
                np.maximum(data_array, 0 * unit)

            return

        expected_units = {None: original_unit}
        expected = attach_units(
            np.maximum(
                strip_units(data_array),
                strip_units(convert_units(0 * unit, expected_units)),
            ),
            expected_units,
        )

        actual = np.maximum(data_array, 0 * unit)
        assert_equal_with_units(expected, actual)

        actual = np.maximum(0 * unit, data_array)
        assert_equal_with_units(expected, actual)

    @pytest.mark.parametrize("property", ("T", "imag", "real"))
    def test_numpy_properties(self, property, dtype):
        array = (
            np.arange(5 * 10).astype(dtype)
            + 1j * np.linspace(-1, 0, 5 * 10).astype(dtype)
        ).reshape(5, 10) * unit_registry.s

        data_array = xr.DataArray(data=array, dims=("x", "y"))

        expected = attach_units(
            getattr(strip_units(data_array), property), extract_units(data_array)
        )
        actual = getattr(data_array, property)

        assert_equal_with_units(expected, actual)

    @pytest.mark.parametrize(
        "func",
        (method("conj"), method("argsort"), method("conjugate"), method("round")),
        ids=repr,
    )
    def test_numpy_methods(self, func, dtype):
        array = np.arange(10).astype(dtype) * unit_registry.m
        data_array = xr.DataArray(data=array, dims="x")

        units = extract_units(func(array))
        expected = attach_units(strip_units(data_array), units)
        actual = func(data_array)

        assert_equal_with_units(expected, actual)

    @pytest.mark.parametrize(
        "func",
        (
            method("clip", min=3, max=8),
            pytest.param(
                method("searchsorted", v=5),
                marks=pytest.mark.xfail(
                    reason="searchsorted somehow requires a undocumented `keys` argument"
                ),
            ),
        ),
        ids=repr,
    )
    @pytest.mark.parametrize(
        "unit,error",
        (
            pytest.param(1, DimensionalityError, id="no_unit"),
            pytest.param(
                unit_registry.dimensionless, DimensionalityError, id="dimensionless"
            ),
            pytest.param(unit_registry.s, DimensionalityError, id="incompatible_unit"),
            pytest.param(unit_registry.cm, None, id="compatible_unit"),
            pytest.param(unit_registry.m, None, id="identical_unit"),
        ),
    )
    def test_numpy_methods_with_args(self, func, unit, error, dtype):
        array = np.arange(10).astype(dtype) * unit_registry.m
        data_array = xr.DataArray(data=array)

        scalar_types = (int, float)
        kwargs = {
            key: (value * unit if isinstance(value, scalar_types) else value)
            for key, value in func.kwargs.items()
        }
        if error is not None:
            with pytest.raises(error):
                func(data_array, **kwargs)

            return

        units = extract_units(data_array)
        expected_units = extract_units(func(array, **kwargs))
        stripped_kwargs = {
            key: strip_units(convert_units(value, units))
            for key, value in kwargs.items()
        }
        expected = attach_units(
            func(strip_units(data_array), **stripped_kwargs), expected_units
        )
        actual = func(data_array, **kwargs)

        assert_equal_with_units(expected, actual)

    @pytest.mark.parametrize(
        "func", (method("isnull"), method("notnull"), method("count")), ids=repr
    )
    def test_missing_value_detection(self, func, dtype):
        array = (
            np.array(
                [
                    [1.4, 2.3, np.nan, 7.2],
                    [np.nan, 9.7, np.nan, np.nan],
                    [2.1, np.nan, np.nan, 4.6],
                    [9.9, np.nan, 7.2, 9.1],
                ]
            )
            * unit_registry.degK
        )
        x = np.arange(array.shape[0]) * unit_registry.m
        y = np.arange(array.shape[1]) * unit_registry.m

        data_array = xr.DataArray(data=array, coords={"x": x, "y": y}, dims=("x", "y"))

        expected = func(strip_units(data_array))
        actual = func(data_array)

        assert_equal_with_units(expected, actual)

    @pytest.mark.xfail(reason="ffill and bfill lose units in data")
    @pytest.mark.parametrize("func", (method("ffill"), method("bfill")), ids=repr)
    def test_missing_value_filling(self, func, dtype):
        array = (
            np.array([1.4, np.nan, 2.3, np.nan, np.nan, 9.1]).astype(dtype)
            * unit_registry.degK
        )
        x = np.arange(len(array))
        data_array = xr.DataArray(data=array, coords={"x": x}, dims="x")

        expected = attach_units(
            func(strip_units(data_array), dim="x"), extract_units(data_array)
        )
        actual = func(data_array, dim="x")

        assert_equal_with_units(expected, actual)

    @pytest.mark.parametrize(
        "unit,error",
        (
            pytest.param(1, DimensionalityError, id="no_unit"),
            pytest.param(
                unit_registry.dimensionless, DimensionalityError, id="dimensionless"
            ),
            pytest.param(unit_registry.s, DimensionalityError, id="incompatible_unit"),
            pytest.param(
                unit_registry.cm,
                None,
                id="compatible_unit",
                marks=pytest.mark.xfail(reason="fillna converts to value's unit"),
            ),
            pytest.param(unit_registry.m, None, id="identical_unit"),
        ),
    )
    @pytest.mark.parametrize(
        "fill_value",
        (
            pytest.param(-1, id="python_scalar"),
            pytest.param(np.array(-1), id="numpy_scalar"),
            pytest.param(np.array([-1]), id="numpy_array"),
        ),
    )
    def test_fillna(self, fill_value, unit, error, dtype):
        original_unit = unit_registry.m
        array = (
            np.array([1.4, np.nan, 2.3, np.nan, np.nan, 9.1]).astype(dtype)
            * original_unit
        )
        data_array = xr.DataArray(data=array)

        func = method("fillna")

        value = fill_value * unit
        if error is not None:
            with pytest.raises(error):
                func(data_array, value=value)

            return

        units = extract_units(data_array)
        expected = attach_units(
            func(
                strip_units(data_array), value=strip_units(convert_units(value, units))
            ),
            units,
        )
        actual = func(data_array, value=value)

        assert_equal_with_units(expected, actual)

    def test_dropna(self, dtype):
        array = (
            np.array([1.4, np.nan, 2.3, np.nan, np.nan, 9.1]).astype(dtype)
            * unit_registry.m
        )
        x = np.arange(len(array))
        data_array = xr.DataArray(data=array, coords={"x": x}, dims=["x"])

        units = extract_units(data_array)
        expected = attach_units(strip_units(data_array).dropna(dim="x"), units)
        actual = data_array.dropna(dim="x")

        assert_equal_with_units(expected, actual)

    @pytest.mark.parametrize(
        "unit",
        (
            pytest.param(
                1,
                id="no_unit",
                marks=pytest.mark.xfail(
                    reason="pint's isin implementation does not work well with mixed args"
                ),
            ),
            pytest.param(unit_registry.dimensionless, id="dimensionless"),
            pytest.param(unit_registry.s, id="incompatible_unit"),
            pytest.param(unit_registry.cm, id="compatible_unit"),
            pytest.param(unit_registry.m, id="identical_unit"),
        ),
    )
    def test_isin(self, unit, dtype):
        array = (
            np.array([1.4, np.nan, 2.3, np.nan, np.nan, 9.1]).astype(dtype)
            * unit_registry.m
        )
        data_array = xr.DataArray(data=array, dims="x")

        raw_values = np.array([1.4, np.nan, 2.3]).astype(dtype)
        values = raw_values * unit

        units = {None: unit_registry.m if array.check(unit) else None}
        expected = strip_units(data_array).isin(
            strip_units(convert_units(values, units))
        ) & array.check(unit)
        actual = data_array.isin(values)

        assert_equal_with_units(expected, actual)

    @pytest.mark.parametrize(
        "variant",
        (
            pytest.param(
                "masking",
                marks=pytest.mark.xfail(reason="array(nan) is not a quantity"),
            ),
            "replacing_scalar",
            "replacing_array",
            pytest.param(
                "dropping",
                marks=pytest.mark.xfail(reason="array(nan) is not a quantity"),
            ),
        ),
    )
    @pytest.mark.parametrize(
        "unit,error",
        (
            pytest.param(1, DimensionalityError, id="no_unit"),
            pytest.param(
                unit_registry.dimensionless, DimensionalityError, id="dimensionless"
            ),
            pytest.param(unit_registry.s, DimensionalityError, id="incompatible_unit"),
            pytest.param(unit_registry.cm, None, id="compatible_unit"),
            pytest.param(unit_registry.m, None, id="identical_unit"),
        ),
    )
    def test_where(self, variant, unit, error, dtype):
        original_unit = unit_registry.m
        array = np.linspace(0, 1, 10).astype(dtype) * original_unit

        data_array = xr.DataArray(data=array)

        condition = data_array < 0.5 * original_unit
        other = np.linspace(-2, -1, 10).astype(dtype) * unit
        variant_kwargs = {
            "masking": {"cond": condition},
            "replacing_scalar": {"cond": condition, "other": -1 * unit},
            "replacing_array": {"cond": condition, "other": other},
            "dropping": {"cond": condition, "drop": True},
        }
        kwargs = variant_kwargs.get(variant)
        kwargs_without_units = {
            key: strip_units(
                convert_units(
                    value, {None: original_unit if array.check(unit) else None}
                )
            )
            for key, value in kwargs.items()
        }

        if variant not in ("masking", "dropping") and error is not None:
            with pytest.raises(error):
                data_array.where(**kwargs)

            return

        expected = attach_units(
            strip_units(data_array).where(**kwargs_without_units),
            extract_units(data_array),
        )
        actual = data_array.where(**kwargs)

        assert_equal_with_units(expected, actual)

    @pytest.mark.xfail(reason="interpolate strips units")
    def test_interpolate_na(self, dtype):
        array = (
            np.array([-1.03, 0.1, 1.4, np.nan, 2.3, np.nan, np.nan, 9.1])
            * unit_registry.m
        )
        x = np.arange(len(array))
        data_array = xr.DataArray(data=array, coords={"x": x}, dims="x").astype(dtype)

        units = extract_units(data_array)
        expected = attach_units(strip_units(data_array).interpolate_na(dim="x"), units)
        actual = data_array.interpolate_na(dim="x")

        assert_equal_with_units(expected, actual)

    @pytest.mark.parametrize(
        "unit,error",
        (
            pytest.param(1, DimensionalityError, id="no_unit"),
            pytest.param(
                unit_registry.dimensionless, DimensionalityError, id="dimensionless"
            ),
            pytest.param(unit_registry.s, DimensionalityError, id="incompatible_unit"),
            pytest.param(
                unit_registry.cm,
                None,
                id="compatible_unit",
                marks=pytest.mark.xfail(reason="depends on reindex"),
            ),
            pytest.param(
                unit_registry.m,
                None,
                id="identical_unit",
                marks=pytest.mark.xfail(reason="depends on reindex"),
            ),
        ),
    )
    def test_combine_first(self, unit, error, dtype):
        array = np.zeros(shape=(2, 2), dtype=dtype) * unit_registry.m
        other_array = np.ones_like(array) * unit

        data_array = xr.DataArray(
            data=array, coords={"x": ["a", "b"], "y": [-1, 0]}, dims=["x", "y"]
        )
        other = xr.DataArray(
            data=other_array, coords={"x": ["b", "c"], "y": [0, 1]}, dims=["x", "y"]
        )

        if error is not None:
            with pytest.raises(error):
                data_array.combine_first(other)

            return

        units = extract_units(data_array)
        expected = attach_units(
            strip_units(data_array).combine_first(
                strip_units(convert_units(other, units))
            ),
            units,
        )
        actual = data_array.combine_first(other)

        assert_equal_with_units(expected, actual)

    @pytest.mark.parametrize(
        "unit",
        (
            pytest.param(1, id="no_unit"),
            pytest.param(unit_registry.dimensionless, id="dimensionless"),
            pytest.param(unit_registry.s, id="incompatible_unit"),
            pytest.param(unit_registry.cm, id="compatible_unit"),
            pytest.param(unit_registry.m, id="identical_unit"),
        ),
    )
    @pytest.mark.parametrize(
        "variation",
        (
            "data",
            pytest.param(
                "dims", marks=pytest.mark.xfail(reason="units in indexes not supported")
            ),
            "coords",
        ),
    )
    @pytest.mark.parametrize("func", (method("equals"), method("identical")), ids=repr)
    def test_comparisons(self, func, variation, unit, dtype):
        def is_compatible(a, b):
            a = a if a is not None else 1
            b = b if b is not None else 1
            quantity = np.arange(5) * a

            return a == b or quantity.check(b)

        data = np.linspace(0, 5, 10).astype(dtype)
        coord = np.arange(len(data)).astype(dtype)

        base_unit = unit_registry.m
        array = data * (base_unit if variation == "data" else 1)
        x = coord * (base_unit if variation == "dims" else 1)
        y = coord * (base_unit if variation == "coords" else 1)

        variations = {
            "data": (unit, 1, 1),
            "dims": (1, unit, 1),
            "coords": (1, 1, unit),
        }
        data_unit, dim_unit, coord_unit = variations.get(variation)

        data_array = xr.DataArray(data=array, coords={"x": x, "y": ("x", y)}, dims="x")

        other = attach_units(
            strip_units(data_array), {None: data_unit, "x": dim_unit, "y": coord_unit}
        )

        units = extract_units(data_array)
        other_units = extract_units(other)

        equal_arrays = all(
            is_compatible(units[name], other_units[name]) for name in units.keys()
        ) and (
            strip_units(data_array).equals(
                strip_units(convert_units(other, extract_units(data_array)))
            )
        )
        equal_units = units == other_units
        expected = equal_arrays and (func.name != "identical" or equal_units)

        actual = func(data_array, other)

        assert expected == actual

    @pytest.mark.parametrize(
        "unit",
        (
            pytest.param(1, id="no_unit"),
            pytest.param(unit_registry.dimensionless, id="dimensionless"),
            pytest.param(unit_registry.s, id="incompatible_unit"),
            pytest.param(unit_registry.cm, id="compatible_unit"),
            pytest.param(unit_registry.m, id="identical_unit"),
        ),
    )
    def test_broadcast_like(self, unit, dtype):
        array1 = np.linspace(1, 2, 2 * 1).reshape(2, 1).astype(dtype) * unit_registry.Pa
        array2 = np.linspace(0, 1, 2 * 3).reshape(2, 3).astype(dtype) * unit_registry.Pa

        x1 = np.arange(2) * unit_registry.m
        x2 = np.arange(2) * unit
        y1 = np.array([0]) * unit_registry.m
        y2 = np.arange(3) * unit

        arr1 = xr.DataArray(data=array1, coords={"x": x1, "y": y1}, dims=("x", "y"))
        arr2 = xr.DataArray(data=array2, coords={"x": x2, "y": y2}, dims=("x", "y"))

        expected = attach_units(
            strip_units(arr1).broadcast_like(strip_units(arr2)), extract_units(arr1)
        )
        actual = arr1.broadcast_like(arr2)

        assert_equal_with_units(expected, actual)

    @pytest.mark.parametrize(
        "unit",
        (
            pytest.param(1, id="no_unit"),
            pytest.param(unit_registry.dimensionless, id="dimensionless"),
            pytest.param(unit_registry.s, id="incompatible_unit"),
            pytest.param(unit_registry.cm, id="compatible_unit"),
            pytest.param(unit_registry.m, id="identical_unit"),
        ),
    )
    def test_broadcast_equals(self, unit, dtype):
        left_array = np.ones(shape=(2, 2), dtype=dtype) * unit_registry.m
        right_array = np.ones(shape=(2,), dtype=dtype) * unit

        left = xr.DataArray(data=left_array, dims=("x", "y"))
        right = xr.DataArray(data=right_array, dims="x")

        units = {
            **extract_units(left),
            **({} if left_array.check(unit) else {None: None}),
        }
        expected = strip_units(left).broadcast_equals(
            strip_units(convert_units(right, units))
        ) & left_array.check(unit)
        actual = left.broadcast_equals(right)

        assert expected == actual

    @pytest.mark.parametrize(
        "func",
        (
            method("pipe", lambda da: da * 10),
            method("assign_coords", y2=("y", np.arange(10) * unit_registry.mm)),
            method("assign_attrs", attr1="value"),
            method("rename", x2="x_mm"),
            method("swap_dims", {"x": "x2"}),
            method(
                "expand_dims",
                dim={"z": np.linspace(10, 20, 12) * unit_registry.s},
                axis=1,
            ),
            method("drop_vars", "x"),
            method("reset_coords", names="x2"),
            method("copy"),
            method("astype", np.float32),
            method("item", 1),
        ),
        ids=repr,
    )
    def test_content_manipulation(self, func, dtype):
        quantity = (
            np.linspace(0, 10, 5 * 10).reshape(5, 10).astype(dtype)
            * unit_registry.pascal
        )
        x = np.arange(quantity.shape[0]) * unit_registry.m
        y = np.arange(quantity.shape[1]) * unit_registry.m
        x2 = x.to(unit_registry.mm)

        data_array = xr.DataArray(
            name="data",
            data=quantity,
            coords={"x": x, "x2": ("x", x2), "y": y},
            dims=("x", "y"),
        )

        stripped_kwargs = {
            key: array_strip_units(value) for key, value in func.kwargs.items()
        }
        units = {**{"x_mm": x2.units, "x2": x2.units}, **extract_units(data_array)}

        expected = attach_units(func(strip_units(data_array), **stripped_kwargs), units)
        actual = func(data_array)

        assert_equal_with_units(expected, actual)

    @pytest.mark.parametrize(
        "func", (pytest.param(method("copy", data=np.arange(20))),), ids=repr
    )
    @pytest.mark.parametrize(
        "unit",
        (
            pytest.param(1, id="no_unit"),
            pytest.param(unit_registry.dimensionless, id="dimensionless"),
            pytest.param(unit_registry.degK, id="with_unit"),
        ),
    )
    def test_content_manipulation_with_units(self, func, unit, dtype):
        quantity = np.linspace(0, 10, 20, dtype=dtype) * unit_registry.pascal
        x = np.arange(len(quantity)) * unit_registry.m

        data_array = xr.DataArray(data=quantity, coords={"x": x}, dims="x")

        kwargs = {key: value * unit for key, value in func.kwargs.items()}

        expected = attach_units(
            func(strip_units(data_array)), {None: unit, "x": x.units}
        )

        actual = func(data_array, **kwargs)
        assert_equal_with_units(expected, actual)

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

        data_array = xr.DataArray(data=array, coords={"x": x}, dims="x")

        expected = attach_units(
            strip_units(data_array).isel(x=indices), extract_units(data_array)
        )
        actual = data_array.isel(x=indices)

        assert_equal_with_units(expected, actual)

    @pytest.mark.xfail(reason="indexes don't support units")
    @pytest.mark.parametrize(
        "raw_values",
        (
            pytest.param(10, id="single_value"),
            pytest.param([10, 5, 13], id="list_of_values"),
            pytest.param(np.array([9, 3, 7, 12]), id="array_of_values"),
        ),
    )
    @pytest.mark.parametrize(
        "unit,error",
        (
            pytest.param(1, KeyError, id="no_units"),
            pytest.param(unit_registry.dimensionless, KeyError, id="dimensionless"),
            pytest.param(unit_registry.degree, KeyError, id="incompatible_unit"),
            pytest.param(unit_registry.dm, KeyError, id="compatible_unit"),
            pytest.param(unit_registry.m, None, id="identical_unit"),
        ),
    )
    def test_sel(self, raw_values, unit, error, dtype):
        array = np.linspace(5, 10, 20).astype(dtype) * unit_registry.m
        x = np.arange(len(array)) * unit_registry.m
        data_array = xr.DataArray(data=array, coords={"x": x}, dims="x")

        values = raw_values * unit

        if error is not None and not (
            isinstance(raw_values, (int, float)) and x.check(unit)
        ):
            with pytest.raises(error):
                data_array.sel(x=values)

            return

        expected = attach_units(
            strip_units(data_array).sel(
                x=strip_units(convert_units(values, {None: array.units}))
            ),
            extract_units(data_array),
        )
        actual = data_array.sel(x=values)
        assert_equal_with_units(expected, actual)

    @pytest.mark.xfail(reason="indexes don't support units")
    @pytest.mark.parametrize(
        "raw_values",
        (
            pytest.param(10, id="single_value"),
            pytest.param([10, 5, 13], id="list_of_values"),
            pytest.param(np.array([9, 3, 7, 12]), id="array_of_values"),
        ),
    )
    @pytest.mark.parametrize(
        "unit,error",
        (
            pytest.param(1, KeyError, id="no_units"),
            pytest.param(unit_registry.dimensionless, KeyError, id="dimensionless"),
            pytest.param(unit_registry.degree, KeyError, id="incompatible_unit"),
            pytest.param(unit_registry.dm, KeyError, id="compatible_unit"),
            pytest.param(unit_registry.m, None, id="identical_unit"),
        ),
    )
    def test_loc(self, raw_values, unit, error, dtype):
        array = np.linspace(5, 10, 20).astype(dtype) * unit_registry.m
        x = np.arange(len(array)) * unit_registry.m
        data_array = xr.DataArray(data=array, coords={"x": x}, dims="x")

        values = raw_values * unit

        if error is not None and not (
            isinstance(raw_values, (int, float)) and x.check(unit)
        ):
            with pytest.raises(error):
                data_array.loc[{"x": values}]

            return

        expected = attach_units(
            strip_units(data_array).loc[
                {"x": strip_units(convert_units(values, {None: array.units}))}
            ],
            extract_units(data_array),
        )
        actual = data_array.loc[{"x": values}]
        assert_equal_with_units(expected, actual)

    @pytest.mark.xfail(reason="indexes don't support units")
    @pytest.mark.parametrize(
        "raw_values",
        (
            pytest.param(10, id="single_value"),
            pytest.param([10, 5, 13], id="list_of_values"),
            pytest.param(np.array([9, 3, 7, 12]), id="array_of_values"),
        ),
    )
    @pytest.mark.parametrize(
        "unit,error",
        (
            pytest.param(1, KeyError, id="no_units"),
            pytest.param(unit_registry.dimensionless, KeyError, id="dimensionless"),
            pytest.param(unit_registry.degree, KeyError, id="incompatible_unit"),
            pytest.param(unit_registry.dm, KeyError, id="compatible_unit"),
            pytest.param(unit_registry.m, None, id="identical_unit"),
        ),
    )
    def test_drop_sel(self, raw_values, unit, error, dtype):
        array = np.linspace(5, 10, 20).astype(dtype) * unit_registry.m
        x = np.arange(len(array)) * unit_registry.m
        data_array = xr.DataArray(data=array, coords={"x": x}, dims="x")

        values = raw_values * unit

        if error is not None and not (
            isinstance(raw_values, (int, float)) and x.check(unit)
        ):
            with pytest.raises(error):
                data_array.drop_sel(x=values)

            return

        expected = attach_units(
            strip_units(data_array).drop_sel(
                x=strip_units(convert_units(values, {None: x.units}))
            ),
            extract_units(data_array),
        )
        actual = data_array.drop_sel(x=values)
        assert_equal_with_units(expected, actual)

    @pytest.mark.parametrize(
        "shape",
        (
            pytest.param((10, 20), id="nothing_squeezable"),
            pytest.param((10, 20, 1), id="last_dimension_squeezable"),
            pytest.param((10, 1, 20), id="middle_dimension_squeezable"),
            pytest.param((1, 10, 20), id="first_dimension_squeezable"),
            pytest.param((1, 10, 1, 20), id="first_and_last_dimension_squeezable"),
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

        expected = attach_units(
            strip_units(data_array).squeeze(), extract_units(data_array)
        )
        actual = data_array.squeeze()
        assert_equal_with_units(expected, actual)

        # try squeezing the dimensions separately
        names = tuple(dim for dim, coord in coords.items() if len(coord) == 1)
        for index, name in enumerate(names):
            expected = attach_units(
                strip_units(data_array).squeeze(dim=name), extract_units(data_array)
            )
            actual = data_array.squeeze(dim=name)
            assert_equal_with_units(expected, actual)

    @pytest.mark.parametrize(
        "func",
        (method("head", x=7, y=3), method("tail", x=7, y=3), method("thin", x=7, y=3)),
        ids=repr,
    )
    def test_head_tail_thin(self, func, dtype):
        array = np.linspace(1, 2, 10 * 5).reshape(10, 5) * unit_registry.degK

        coords = {
            "x": np.arange(10) * unit_registry.m,
            "y": np.arange(5) * unit_registry.m,
        }

        data_array = xr.DataArray(data=array, coords=coords, dims=("x", "y"))

        expected = attach_units(
            func(strip_units(data_array)), extract_units(data_array)
        )
        actual = func(data_array)

        assert_equal_with_units(expected, actual)

    @pytest.mark.xfail(reason="indexes don't support units")
    @pytest.mark.parametrize(
        "unit,error",
        (
            pytest.param(1, DimensionalityError, id="no_unit"),
            pytest.param(
                unit_registry.dimensionless, DimensionalityError, id="dimensionless"
            ),
            pytest.param(unit_registry.s, DimensionalityError, id="incompatible_unit"),
            pytest.param(unit_registry.cm, None, id="compatible_unit"),
            pytest.param(unit_registry.m, None, id="identical_unit"),
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

            return

        units = extract_units(data_array)
        expected = attach_units(
            strip_units(data_array).interp(
                x=strip_units(convert_units(new_coords, {None: unit_registry.m}))
            ),
            units,
        )
        actual = data_array.interp(x=new_coords)

        assert_equal_with_units(expected, actual)

    @pytest.mark.xfail(reason="indexes strip units")
    @pytest.mark.parametrize(
        "unit,error",
        (
            pytest.param(1, DimensionalityError, id="no_unit"),
            pytest.param(
                unit_registry.dimensionless, DimensionalityError, id="dimensionless"
            ),
            pytest.param(unit_registry.s, DimensionalityError, id="incompatible_unit"),
            pytest.param(unit_registry.cm, None, id="compatible_unit"),
            pytest.param(unit_registry.m, None, id="identical_unit"),
        ),
    )
    def test_interp_like(self, unit, error):
        array = np.linspace(1, 2, 10 * 5).reshape(10, 5) * unit_registry.degK
        coords = {
            "x": (np.arange(10) + 0.3) * unit_registry.m,
            "y": (np.arange(5) + 0.3) * unit_registry.m,
        }

        data_array = xr.DataArray(array, coords=coords, dims=("x", "y"))
        other = xr.DataArray(
            data=np.empty((20, 10)) * unit_registry.degK,
            coords={"x": np.arange(20) * unit, "y": np.arange(10) * unit},
            dims=("x", "y"),
        )

        if error is not None:
            with pytest.raises(error):
                data_array.interp_like(other)

            return

        units = extract_units(data_array)
        expected = attach_units(
            strip_units(data_array).interp_like(
                strip_units(convert_units(other, units))
            ),
            units,
        )
        actual = data_array.interp_like(other)

        assert_equal_with_units(expected, actual)

    @pytest.mark.xfail(reason="indexes don't support units")
    @pytest.mark.parametrize(
        "unit,error",
        (
            pytest.param(1, DimensionalityError, id="no_unit"),
            pytest.param(
                unit_registry.dimensionless, DimensionalityError, id="dimensionless"
            ),
            pytest.param(unit_registry.s, DimensionalityError, id="incompatible_unit"),
            pytest.param(unit_registry.cm, None, id="compatible_unit"),
            pytest.param(unit_registry.m, None, id="identical_unit"),
        ),
    )
    def test_reindex(self, unit, error, dtype):
        array = (
            np.linspace(1, 2, 10 * 5).reshape(10, 5).astype(dtype) * unit_registry.degK
        )
        new_coords = (np.arange(10) + 0.5) * unit
        coords = {
            "x": np.arange(10) * unit_registry.m,
            "y": np.arange(5) * unit_registry.m,
        }

        data_array = xr.DataArray(array, coords=coords, dims=("x", "y"))
        func = method("reindex")

        if error is not None:
            with pytest.raises(error):
                func(data_array, x=new_coords)

            return

        expected = attach_units(
            func(
                strip_units(data_array),
                x=strip_units(convert_units(new_coords, {None: unit_registry.m})),
            ),
            {None: unit_registry.degK},
        )
        actual = func(data_array, x=new_coords)

        assert_equal_with_units(expected, actual)

    @pytest.mark.xfail(reason="indexes don't support units")
    @pytest.mark.parametrize(
        "unit,error",
        (
            pytest.param(1, DimensionalityError, id="no_unit"),
            pytest.param(
                unit_registry.dimensionless, DimensionalityError, id="dimensionless"
            ),
            pytest.param(unit_registry.s, DimensionalityError, id="incompatible_unit"),
            pytest.param(unit_registry.cm, None, id="compatible_unit"),
            pytest.param(unit_registry.m, None, id="identical_unit"),
        ),
    )
    def test_reindex_like(self, unit, error, dtype):
        array = (
            np.linspace(1, 2, 10 * 5).reshape(10, 5).astype(dtype) * unit_registry.degK
        )
        coords = {
            "x": (np.arange(10) + 0.3) * unit_registry.m,
            "y": (np.arange(5) + 0.3) * unit_registry.m,
        }

        data_array = xr.DataArray(array, coords=coords, dims=("x", "y"))
        other = xr.DataArray(
            data=np.empty((20, 10)) * unit_registry.degK,
            coords={"x": np.arange(20) * unit, "y": np.arange(10) * unit},
            dims=("x", "y"),
        )

        if error is not None:
            with pytest.raises(error):
                data_array.reindex_like(other)

            return

        units = extract_units(data_array)
        expected = attach_units(
            strip_units(data_array).reindex_like(
                strip_units(convert_units(other, units))
            ),
            units,
        )
        actual = data_array.reindex_like(other)

        assert_equal_with_units(expected, actual)

    @pytest.mark.parametrize(
        "func",
        (method("unstack"), method("reset_index", "z"), method("reorder_levels")),
        ids=repr,
    )
    def test_stacking_stacked(self, func, dtype):
        array = (
            np.linspace(0, 10, 5 * 10).reshape(5, 10).astype(dtype) * unit_registry.m
        )
        x = np.arange(array.shape[0])
        y = np.arange(array.shape[1])

        data_array = xr.DataArray(
            name="data", data=array, coords={"x": x, "y": y}, dims=("x", "y")
        )
        stacked = data_array.stack(z=("x", "y"))

        expected = attach_units(func(strip_units(stacked)), {"data": unit_registry.m})
        actual = func(stacked)

        assert_equal_with_units(expected, actual)

    @pytest.mark.xfail(reason="indexes don't support units")
    def test_to_unstacked_dataset(self, dtype):
        array = (
            np.linspace(0, 10, 5 * 10).reshape(5, 10).astype(dtype)
            * unit_registry.pascal
        )
        x = np.arange(array.shape[0]) * unit_registry.m
        y = np.arange(array.shape[1]) * unit_registry.s

        data_array = xr.DataArray(
            data=array, coords={"x": x, "y": y}, dims=("x", "y")
        ).stack(z=("x", "y"))

        func = method("to_unstacked_dataset", dim="z")

        expected = attach_units(
            func(strip_units(data_array)),
            {"y": y.units, **dict(zip(x.magnitude, [array.units] * len(y)))},
        ).rename({elem.magnitude: elem for elem in x})
        actual = func(data_array)

        assert_equal_with_units(expected, actual)

    @pytest.mark.parametrize(
        "func",
        (
            method("transpose", "y", "x", "z"),
            method("stack", a=("x", "y")),
            method("set_index", x="x2"),
            pytest.param(
                method("shift", x=2), marks=pytest.mark.xfail(reason="strips units")
            ),
            method("roll", x=2, roll_coords=False),
            method("sortby", "x2"),
        ),
        ids=repr,
    )
    def test_stacking_reordering(self, func, dtype):
        array = (
            np.linspace(0, 10, 2 * 5 * 10).reshape(2, 5, 10).astype(dtype)
            * unit_registry.m
        )
        x = np.arange(array.shape[0])
        y = np.arange(array.shape[1])
        z = np.arange(array.shape[2])
        x2 = np.linspace(0, 1, array.shape[0])[::-1]

        data_array = xr.DataArray(
            name="data",
            data=array,
            coords={"x": x, "y": y, "z": z, "x2": ("x", x2)},
            dims=("x", "y", "z"),
        )

        expected = attach_units(func(strip_units(data_array)), {None: unit_registry.m})
        actual = func(data_array)

        assert_equal_with_units(expected, actual)

    @pytest.mark.parametrize(
        "func",
        (
            method("diff", dim="x"),
            method("differentiate", coord="x"),
            method("integrate", dim="x"),
            pytest.param(
                method("quantile", q=[0.25, 0.75]),
                marks=pytest.mark.xfail(reason="nanquantile not implemented"),
            ),
            method("reduce", func=np.sum, dim="x"),
            pytest.param(
                lambda x: x.dot(x),
                id="method_dot",
                marks=pytest.mark.xfail(
                    reason="pint does not implement the dot method"
                ),
            ),
        ),
        ids=repr,
    )
    def test_computation(self, func, dtype):
        array = (
            np.linspace(0, 10, 5 * 10).reshape(5, 10).astype(dtype) * unit_registry.m
        )

        x = np.arange(array.shape[0]) * unit_registry.m
        y = np.arange(array.shape[1]) * unit_registry.s

        data_array = xr.DataArray(data=array, coords={"x": x, "y": y}, dims=("x", "y"))

        # we want to make sure the output unit is correct
        units = {
            **extract_units(data_array),
            **(
                {}
                if isinstance(func, (function, method))
                else extract_units(func(array.reshape(-1)))
            ),
        }

        expected = attach_units(func(strip_units(data_array)), units)
        actual = func(data_array)

        assert_equal_with_units(expected, actual)

    @pytest.mark.parametrize(
        "func",
        (
            method("groupby", "x"),
            method("groupby_bins", "y", bins=4),
            method("coarsen", y=2),
            pytest.param(
                method("rolling", y=3),
                marks=pytest.mark.xfail(reason="rolling strips units"),
            ),
            pytest.param(
                method("rolling_exp", y=3),
                marks=pytest.mark.xfail(reason="units not supported by numbagg"),
            ),
        ),
        ids=repr,
    )
    def test_computation_objects(self, func, dtype):
        array = (
            np.linspace(0, 10, 5 * 10).reshape(5, 10).astype(dtype) * unit_registry.m
        )

        x = np.array([0, 0, 1, 2, 2]) * unit_registry.m
        y = np.arange(array.shape[1]) * 3 * unit_registry.s

        data_array = xr.DataArray(data=array, coords={"x": x, "y": y}, dims=("x", "y"))
        units = extract_units(data_array)

        expected = attach_units(func(strip_units(data_array)).mean(), units)
        actual = func(data_array).mean()

        assert_equal_with_units(expected, actual)

    def test_resample(self, dtype):
        array = np.linspace(0, 5, 10).astype(dtype) * unit_registry.m

        time = pd.date_range("10-09-2010", periods=len(array), freq="1y")
        data_array = xr.DataArray(data=array, coords={"time": time}, dims="time")
        units = extract_units(data_array)

        func = method("resample", time="6m")

        expected = attach_units(func(strip_units(data_array)).mean(), units)
        actual = func(data_array).mean()

        assert_equal_with_units(expected, actual)

    @pytest.mark.parametrize(
        "func",
        (
            method("assign_coords", z=(["x"], np.arange(5) * unit_registry.s)),
            method("first"),
            method("last"),
            pytest.param(
                method("quantile", q=[0.25, 0.5, 0.75], dim="x"),
                marks=pytest.mark.xfail(reason="nanquantile not implemented"),
            ),
        ),
        ids=repr,
    )
    def test_grouped_operations(self, func, dtype):
        array = (
            np.linspace(0, 10, 5 * 10).reshape(5, 10).astype(dtype) * unit_registry.m
        )

        x = np.arange(array.shape[0]) * unit_registry.m
        y = np.arange(array.shape[1]) * 3 * unit_registry.s

        data_array = xr.DataArray(data=array, coords={"x": x, "y": y}, dims=("x", "y"))
        units = {**extract_units(data_array), **{"z": unit_registry.s, "q": None}}

        stripped_kwargs = {
            key: (
                strip_units(value)
                if not isinstance(value, tuple)
                else tuple(strip_units(elem) for elem in value)
            )
            for key, value in func.kwargs.items()
        }
        expected = attach_units(
            func(strip_units(data_array).groupby("y"), **stripped_kwargs), units
        )
        actual = func(data_array.groupby("y"))

        assert_equal_with_units(expected, actual)


class TestDataset:
    @pytest.mark.parametrize(
        "unit,error",
        (
            pytest.param(1, DimensionalityError, id="no_unit"),
            pytest.param(
                unit_registry.dimensionless, DimensionalityError, id="dimensionless"
            ),
            pytest.param(unit_registry.s, DimensionalityError, id="incompatible_unit"),
            pytest.param(unit_registry.mm, None, id="compatible_unit"),
            pytest.param(unit_registry.m, None, id="same_unit"),
        ),
    )
    @pytest.mark.parametrize(
        "shared",
        (
            "nothing",
            pytest.param("dims", marks=pytest.mark.xfail(reason="indexes strip units")),
            pytest.param(
                "coords",
                marks=pytest.mark.xfail(reason="reindex does not work with pint yet"),
            ),
        ),
    )
    def test_init(self, shared, unit, error, dtype):
        original_unit = unit_registry.m
        scaled_unit = unit_registry.mm

        a = np.linspace(0, 1, 10).astype(dtype) * unit_registry.Pa
        b = np.linspace(-1, 0, 12).astype(dtype) * unit_registry.Pa

        raw_x = np.arange(a.shape[0])
        x = raw_x * original_unit
        x2 = x.to(scaled_unit)

        raw_y = np.arange(b.shape[0])
        y = raw_y * unit
        y_units = unit if isinstance(y, unit_registry.Quantity) else None
        if isinstance(y, unit_registry.Quantity):
            if y.check(scaled_unit):
                y2 = y.to(scaled_unit)
            else:
                y2 = y * 1000
            y2_units = y2.units
        else:
            y2 = y * 1000
            y2_units = None

        variants = {
            "nothing": ({"x": x, "x2": ("x", x2)}, {"y": y, "y2": ("y", y2)}),
            "dims": (
                {"x": x, "x2": ("x", strip_units(x2))},
                {"x": y, "y2": ("x", strip_units(y2))},
            ),
            "coords": ({"x": raw_x, "y": ("x", x2)}, {"x": raw_y, "y": ("x", y2)}),
        }
        coords_a, coords_b = variants.get(shared)

        dims_a, dims_b = ("x", "y") if shared == "nothing" else ("x", "x")

        arr1 = xr.DataArray(data=a, coords=coords_a, dims=dims_a)
        arr2 = xr.DataArray(data=b, coords=coords_b, dims=dims_b)
        if error is not None and shared != "nothing":
            with pytest.raises(error):
                xr.Dataset(data_vars={"a": arr1, "b": arr2})

            return

        actual = xr.Dataset(data_vars={"a": arr1, "b": arr2})

        expected_units = {
            "a": a.units,
            "b": b.units,
            "x": x.units,
            "x2": x2.units,
            "y": y_units,
            "y2": y2_units,
        }
        expected = attach_units(
            xr.Dataset(data_vars={"a": strip_units(arr1), "b": strip_units(arr2)}),
            expected_units,
        )
        assert_equal_with_units(actual, expected)

    @pytest.mark.parametrize(
        "func", (pytest.param(str, id="str"), pytest.param(repr, id="repr"))
    )
    @pytest.mark.parametrize(
        "variant",
        (
            pytest.param(
                "with_dims",
                marks=pytest.mark.xfail(reason="units in indexes are not supported"),
            ),
            pytest.param("with_coords"),
            pytest.param("without_coords"),
        ),
    )
    @pytest.mark.filterwarnings("error:::pint[.*]")
    def test_repr(self, func, variant, dtype):
        array1 = np.linspace(1, 2, 10, dtype=dtype) * unit_registry.Pa
        array2 = np.linspace(0, 1, 10, dtype=dtype) * unit_registry.degK

        x = np.arange(len(array1)) * unit_registry.s
        y = x.to(unit_registry.ms)

        variants = {
            "with_dims": {"x": x},
            "with_coords": {"y": ("x", y)},
            "without_coords": {},
        }

        data_array = xr.Dataset(
            data_vars={"a": ("x", array1), "b": ("x", array2)},
            coords=variants.get(variant),
        )

        # FIXME: this just checks that the repr does not raise
        # warnings or errors, but does not check the result
        func(data_array)

    @pytest.mark.parametrize(
        "func",
        (
            pytest.param(
                function("all"),
                marks=pytest.mark.xfail(reason="not implemented by pint"),
            ),
            pytest.param(
                function("any"),
                marks=pytest.mark.xfail(reason="not implemented by pint"),
            ),
            function("argmax"),
            function("argmin"),
            function("max"),
            function("min"),
            function("mean"),
            pytest.param(
                function("median"),
                marks=pytest.mark.xfail(
                    reason="np.median does not work with dataset yet"
                ),
            ),
            function("sum"),
            pytest.param(
                function("prod"),
                marks=pytest.mark.xfail(reason="not implemented by pint"),
            ),
            function("std"),
            function("var"),
            function("cumsum"),
            pytest.param(
                function("cumprod"),
                marks=pytest.mark.xfail(reason="fails within xarray"),
            ),
            pytest.param(
                method("all"), marks=pytest.mark.xfail(reason="not implemented by pint")
            ),
            pytest.param(
                method("any"), marks=pytest.mark.xfail(reason="not implemented by pint")
            ),
            method("argmax"),
            method("argmin"),
            method("max"),
            method("min"),
            method("mean"),
            method("median"),
            method("sum"),
            pytest.param(
                method("prod"),
                marks=pytest.mark.xfail(reason="not implemented by pint"),
            ),
            method("std"),
            method("var"),
            method("cumsum"),
            pytest.param(
                method("cumprod"), marks=pytest.mark.xfail(reason="fails within xarray")
            ),
        ),
        ids=repr,
    )
    def test_aggregation(self, func, dtype):
        unit_a = (
            unit_registry.Pa if func.name != "cumprod" else unit_registry.dimensionless
        )
        unit_b = (
            unit_registry.kg / unit_registry.m ** 3
            if func.name != "cumprod"
            else unit_registry.dimensionless
        )
        a = xr.DataArray(data=np.linspace(0, 1, 10).astype(dtype) * unit_a, dims="x")
        b = xr.DataArray(data=np.linspace(-1, 0, 10).astype(dtype) * unit_b, dims="x")
        x = xr.DataArray(data=np.arange(10).astype(dtype) * unit_registry.m, dims="x")
        y = xr.DataArray(
            data=np.arange(10, 20).astype(dtype) * unit_registry.s, dims="x"
        )

        ds = xr.Dataset(data_vars={"a": a, "b": b}, coords={"x": x, "y": y})

        actual = func(ds)
        expected = attach_units(
            func(strip_units(ds)),
            {
                "a": extract_units(func(a)).get(None),
                "b": extract_units(func(b)).get(None),
            },
        )

        assert_equal_with_units(actual, expected)

    @pytest.mark.parametrize("property", ("imag", "real"))
    def test_numpy_properties(self, property, dtype):
        ds = xr.Dataset(
            data_vars={
                "a": xr.DataArray(
                    data=np.linspace(0, 1, 10) * unit_registry.Pa, dims="x"
                ),
                "b": xr.DataArray(
                    data=np.linspace(-1, 0, 15) * unit_registry.Pa, dims="y"
                ),
            },
            coords={
                "x": np.arange(10) * unit_registry.m,
                "y": np.arange(15) * unit_registry.s,
            },
        )
        units = extract_units(ds)

        actual = getattr(ds, property)
        expected = attach_units(getattr(strip_units(ds), property), units)

        assert_equal_with_units(actual, expected)

    @pytest.mark.parametrize(
        "func",
        (
            method("astype", float),
            method("conj"),
            method("argsort"),
            method("conjugate"),
            method("round"),
        ),
        ids=repr,
    )
    def test_numpy_methods(self, func, dtype):
        ds = xr.Dataset(
            data_vars={
                "a": xr.DataArray(
                    data=np.linspace(1, -1, 10) * unit_registry.Pa, dims="x"
                ),
                "b": xr.DataArray(
                    data=np.linspace(-1, 1, 15) * unit_registry.Pa, dims="y"
                ),
            },
            coords={
                "x": np.arange(10) * unit_registry.m,
                "y": np.arange(15) * unit_registry.s,
            },
        )
        units = {
            "a": array_extract_units(func(ds.a)),
            "b": array_extract_units(func(ds.b)),
            "x": unit_registry.m,
            "y": unit_registry.s,
        }

        actual = func(ds)
        expected = attach_units(func(strip_units(ds)), units)

        assert_equal_with_units(actual, expected)

    @pytest.mark.parametrize("func", (method("clip", min=3, max=8),), ids=repr)
    @pytest.mark.parametrize(
        "unit,error",
        (
            pytest.param(1, DimensionalityError, id="no_unit"),
            pytest.param(
                unit_registry.dimensionless, DimensionalityError, id="dimensionless"
            ),
            pytest.param(unit_registry.s, DimensionalityError, id="incompatible_unit"),
            pytest.param(unit_registry.cm, None, id="compatible_unit"),
            pytest.param(unit_registry.m, None, id="identical_unit"),
        ),
    )
    def test_numpy_methods_with_args(self, func, unit, error, dtype):
        data_unit = unit_registry.m
        ds = xr.Dataset(
            data_vars={
                "a": xr.DataArray(data=np.arange(10) * data_unit, dims="x"),
                "b": xr.DataArray(data=np.arange(15) * data_unit, dims="y"),
            },
            coords={
                "x": np.arange(10) * unit_registry.m,
                "y": np.arange(15) * unit_registry.s,
            },
        )
        units = extract_units(ds)

        kwargs = {
            key: (value * unit if isinstance(value, (int, float)) else value)
            for key, value in func.kwargs.items()
        }

        if error is not None:
            with pytest.raises(error):
                func(ds, **kwargs)

            return

        stripped_kwargs = {
            key: strip_units(convert_units(value, {None: data_unit}))
            for key, value in kwargs.items()
        }

        actual = func(ds, **kwargs)
        expected = attach_units(func(strip_units(ds), **stripped_kwargs), units)

        assert_equal_with_units(actual, expected)

    @pytest.mark.parametrize(
        "func", (method("isnull"), method("notnull"), method("count")), ids=repr
    )
    def test_missing_value_detection(self, func, dtype):
        array1 = (
            np.array(
                [
                    [1.4, 2.3, np.nan, 7.2],
                    [np.nan, 9.7, np.nan, np.nan],
                    [2.1, np.nan, np.nan, 4.6],
                    [9.9, np.nan, 7.2, 9.1],
                ]
            )
            * unit_registry.degK
        )
        array2 = (
            np.array(
                [
                    [np.nan, 5.7, 12.0, 7.2],
                    [np.nan, 12.4, np.nan, 4.2],
                    [9.8, np.nan, 4.6, 1.4],
                    [7.2, np.nan, 6.3, np.nan],
                    [8.4, 3.9, np.nan, np.nan],
                ]
            )
            * unit_registry.Pa
        )

        x = np.arange(array1.shape[0]) * unit_registry.m
        y = np.arange(array1.shape[1]) * unit_registry.m
        z = np.arange(array2.shape[0]) * unit_registry.m

        ds = xr.Dataset(
            data_vars={
                "a": xr.DataArray(data=array1, dims=("x", "y")),
                "b": xr.DataArray(data=array2, dims=("z", "x")),
            },
            coords={"x": x, "y": y, "z": z},
        )

        expected = func(strip_units(ds))
        actual = func(ds)

        assert_equal_with_units(expected, actual)

    @pytest.mark.xfail(reason="ffill and bfill lose the unit")
    @pytest.mark.parametrize("func", (method("ffill"), method("bfill")), ids=repr)
    def test_missing_value_filling(self, func, dtype):
        array1 = (
            np.array([1.4, np.nan, 2.3, np.nan, np.nan, 9.1]).astype(dtype)
            * unit_registry.degK
        )
        array2 = (
            np.array([4.3, 9.8, 7.5, np.nan, 8.2, np.nan]).astype(dtype)
            * unit_registry.Pa
        )

        x = np.arange(len(array1))

        ds = xr.Dataset(
            data_vars={
                "a": xr.DataArray(data=array1, dims="x"),
                "b": xr.DataArray(data=array2, dims="x"),
            },
            coords={"x": x},
        )

        expected = attach_units(
            func(strip_units(ds), dim="x"),
            {"a": unit_registry.degK, "b": unit_registry.Pa},
        )
        actual = func(ds, dim="x")

        assert_equal_with_units(expected, actual)

    @pytest.mark.parametrize(
        "unit,error",
        (
            pytest.param(1, DimensionalityError, id="no_unit"),
            pytest.param(
                unit_registry.dimensionless, DimensionalityError, id="dimensionless"
            ),
            pytest.param(unit_registry.s, DimensionalityError, id="incompatible_unit"),
            pytest.param(
                unit_registry.cm,
                None,
                id="compatible_unit",
                marks=pytest.mark.xfail(
                    reason="where converts the array, not the fill value"
                ),
            ),
            pytest.param(unit_registry.m, None, id="identical_unit"),
        ),
    )
    @pytest.mark.parametrize(
        "fill_value",
        (
            pytest.param(-1, id="python_scalar"),
            pytest.param(np.array(-1), id="numpy_scalar"),
            pytest.param(np.array([-1]), id="numpy_array"),
        ),
    )
    def test_fillna(self, fill_value, unit, error, dtype):
        array1 = (
            np.array([1.4, np.nan, 2.3, np.nan, np.nan, 9.1]).astype(dtype)
            * unit_registry.m
        )
        array2 = (
            np.array([4.3, 9.8, 7.5, np.nan, 8.2, np.nan]).astype(dtype)
            * unit_registry.m
        )
        ds = xr.Dataset(
            data_vars={
                "a": xr.DataArray(data=array1, dims="x"),
                "b": xr.DataArray(data=array2, dims="x"),
            }
        )

        if error is not None:
            with pytest.raises(error):
                ds.fillna(value=fill_value * unit)

            return

        actual = ds.fillna(value=fill_value * unit)
        expected = attach_units(
            strip_units(ds).fillna(
                value=strip_units(
                    convert_units(fill_value * unit, {None: unit_registry.m})
                )
            ),
            {"a": unit_registry.m, "b": unit_registry.m},
        )

        assert_equal_with_units(expected, actual)

    def test_dropna(self, dtype):
        array1 = (
            np.array([1.4, np.nan, 2.3, np.nan, np.nan, 9.1]).astype(dtype)
            * unit_registry.degK
        )
        array2 = (
            np.array([4.3, 9.8, 7.5, np.nan, 8.2, np.nan]).astype(dtype)
            * unit_registry.Pa
        )
        x = np.arange(len(array1))
        ds = xr.Dataset(
            data_vars={
                "a": xr.DataArray(data=array1, dims="x"),
                "b": xr.DataArray(data=array2, dims="x"),
            },
            coords={"x": x},
        )

        expected = attach_units(
            strip_units(ds).dropna(dim="x"),
            {"a": unit_registry.degK, "b": unit_registry.Pa},
        )
        actual = ds.dropna(dim="x")

        assert_equal_with_units(expected, actual)

    @pytest.mark.parametrize(
        "unit",
        (
            pytest.param(1, id="no_unit"),
            pytest.param(unit_registry.dimensionless, id="dimensionless"),
            pytest.param(unit_registry.s, id="incompatible_unit"),
            pytest.param(unit_registry.cm, id="compatible_unit"),
            pytest.param(unit_registry.m, id="same_unit"),
        ),
    )
    def test_isin(self, unit, dtype):
        array1 = (
            np.array([1.4, np.nan, 2.3, np.nan, np.nan, 9.1]).astype(dtype)
            * unit_registry.m
        )
        array2 = (
            np.array([4.3, 9.8, 7.5, np.nan, 8.2, np.nan]).astype(dtype)
            * unit_registry.m
        )
        x = np.arange(len(array1))
        ds = xr.Dataset(
            data_vars={
                "a": xr.DataArray(data=array1, dims="x"),
                "b": xr.DataArray(data=array2, dims="x"),
            },
            coords={"x": x},
        )

        raw_values = np.array([1.4, np.nan, 2.3]).astype(dtype)
        values = raw_values * unit

        if (
            isinstance(values, unit_registry.Quantity)
            and values.check(unit_registry.m)
            and unit != unit_registry.m
        ):
            raw_values = values.to(unit_registry.m).magnitude

        expected = strip_units(ds).isin(raw_values)
        if not isinstance(values, unit_registry.Quantity) or not values.check(
            unit_registry.m
        ):
            expected.a[:] = False
            expected.b[:] = False
        actual = ds.isin(values)

        assert_equal_with_units(actual, expected)

    @pytest.mark.parametrize(
        "variant", ("masking", "replacing_scalar", "replacing_array", "dropping")
    )
    @pytest.mark.parametrize(
        "unit,error",
        (
            pytest.param(1, DimensionalityError, id="no_unit"),
            pytest.param(
                unit_registry.dimensionless, DimensionalityError, id="dimensionless"
            ),
            pytest.param(unit_registry.s, DimensionalityError, id="incompatible_unit"),
            pytest.param(unit_registry.cm, None, id="compatible_unit"),
            pytest.param(unit_registry.m, None, id="same_unit"),
        ),
    )
    def test_where(self, variant, unit, error, dtype):
        original_unit = unit_registry.m
        array1 = np.linspace(0, 1, 10).astype(dtype) * original_unit
        array2 = np.linspace(-1, 0, 10).astype(dtype) * original_unit

        ds = xr.Dataset(
            data_vars={
                "a": xr.DataArray(data=array1, dims="x"),
                "b": xr.DataArray(data=array2, dims="x"),
            },
            coords={"x": np.arange(len(array1))},
        )

        condition = ds < 0.5 * original_unit
        other = np.linspace(-2, -1, 10).astype(dtype) * unit
        variant_kwargs = {
            "masking": {"cond": condition},
            "replacing_scalar": {"cond": condition, "other": -1 * unit},
            "replacing_array": {"cond": condition, "other": other},
            "dropping": {"cond": condition, "drop": True},
        }
        kwargs = variant_kwargs.get(variant)
        if variant not in ("masking", "dropping") and error is not None:
            with pytest.raises(error):
                ds.where(**kwargs)

            return

        kwargs_without_units = {
            key: strip_units(convert_units(value, {None: original_unit}))
            for key, value in kwargs.items()
        }

        expected = attach_units(
            strip_units(ds).where(**kwargs_without_units),
            {"a": original_unit, "b": original_unit},
        )
        actual = ds.where(**kwargs)

        assert_equal_with_units(expected, actual)

    @pytest.mark.xfail(reason="interpolate strips units")
    def test_interpolate_na(self, dtype):
        array1 = (
            np.array([1.4, np.nan, 2.3, np.nan, np.nan, 9.1]).astype(dtype)
            * unit_registry.degK
        )
        array2 = (
            np.array([4.3, 9.8, 7.5, np.nan, 8.2, np.nan]).astype(dtype)
            * unit_registry.Pa
        )
        x = np.arange(len(array1))
        ds = xr.Dataset(
            data_vars={
                "a": xr.DataArray(data=array1, dims="x"),
                "b": xr.DataArray(data=array2, dims="x"),
            },
            coords={"x": x},
        )

        expected = attach_units(
            strip_units(ds).interpolate_na(dim="x"),
            {"a": unit_registry.degK, "b": unit_registry.Pa},
        )
        actual = ds.interpolate_na(dim="x")

        assert_equal_with_units(expected, actual)

    @pytest.mark.xfail(reason="wrong argument order for `where`")
    @pytest.mark.parametrize(
        "unit,error",
        (
            pytest.param(1, DimensionalityError, id="no_unit"),
            pytest.param(
                unit_registry.dimensionless, DimensionalityError, id="dimensionless"
            ),
            pytest.param(unit_registry.s, DimensionalityError, id="incompatible_unit"),
            pytest.param(unit_registry.cm, None, id="compatible_unit"),
            pytest.param(unit_registry.m, None, id="same_unit"),
        ),
    )
    def test_combine_first(self, unit, error, dtype):
        array1 = (
            np.array([1.4, np.nan, 2.3, np.nan, np.nan, 9.1]).astype(dtype)
            * unit_registry.m
        )
        array2 = (
            np.array([4.3, 9.8, 7.5, np.nan, 8.2, np.nan]).astype(dtype)
            * unit_registry.m
        )
        x = np.arange(len(array1))
        ds = xr.Dataset(
            data_vars={
                "a": xr.DataArray(data=array1, dims="x"),
                "b": xr.DataArray(data=array2, dims="x"),
            },
            coords={"x": x},
        )
        other_array1 = np.ones_like(array1) * unit
        other_array2 = -1 * np.ones_like(array2) * unit
        other = xr.Dataset(
            data_vars={
                "a": xr.DataArray(data=other_array1, dims="x"),
                "b": xr.DataArray(data=other_array2, dims="x"),
            },
            coords={"x": np.arange(array1.shape[0])},
        )

        if error is not None:
            with pytest.raises(error):
                ds.combine_first(other)

            return

        expected = attach_units(
            strip_units(ds).combine_first(
                strip_units(
                    convert_units(other, {"a": unit_registry.m, "b": unit_registry.m})
                )
            ),
            {"a": unit_registry.m, "b": unit_registry.m},
        )
        actual = ds.combine_first(other)

        assert_equal_with_units(expected, actual)

    @pytest.mark.parametrize(
        "unit",
        (
            pytest.param(1, id="no_unit"),
            pytest.param(unit_registry.dimensionless, id="dimensionless"),
            pytest.param(unit_registry.s, id="incompatible_unit"),
            pytest.param(unit_registry.cm, id="compatible_unit"),
            pytest.param(unit_registry.m, id="identical_unit"),
        ),
    )
    @pytest.mark.parametrize(
        "variation",
        (
            "data",
            pytest.param(
                "dims", marks=pytest.mark.xfail(reason="units in indexes not supported")
            ),
            "coords",
        ),
    )
    @pytest.mark.parametrize("func", (method("equals"), method("identical")), ids=repr)
    def test_comparisons(self, func, variation, unit, dtype):
        def is_compatible(a, b):
            a = a if a is not None else 1
            b = b if b is not None else 1
            quantity = np.arange(5) * a

            return a == b or quantity.check(b)

        array1 = np.linspace(0, 5, 10).astype(dtype)
        array2 = np.linspace(-5, 0, 10).astype(dtype)

        coord = np.arange(len(array1)).astype(dtype)

        original_unit = unit_registry.m
        quantity1 = array1 * original_unit
        quantity2 = array2 * original_unit
        x = coord * original_unit
        y = coord * original_unit

        units = {"data": (unit, 1, 1), "dims": (1, unit, 1), "coords": (1, 1, unit)}
        data_unit, dim_unit, coord_unit = units.get(variation)

        ds = xr.Dataset(
            data_vars={
                "a": xr.DataArray(data=quantity1, dims="x"),
                "b": xr.DataArray(data=quantity2, dims="x"),
            },
            coords={"x": x, "y": ("x", y)},
        )

        other_units = {
            "a": data_unit if quantity1.check(data_unit) else None,
            "b": data_unit if quantity2.check(data_unit) else None,
            "x": dim_unit if x.check(dim_unit) else None,
            "y": coord_unit if y.check(coord_unit) else None,
        }
        other = attach_units(strip_units(convert_units(ds, other_units)), other_units)

        units = extract_units(ds)
        other_units = extract_units(other)

        equal_ds = all(
            is_compatible(units[name], other_units[name]) for name in units.keys()
        ) and (strip_units(ds).equals(strip_units(convert_units(other, units))))
        equal_units = units == other_units
        expected = equal_ds and (func.name != "identical" or equal_units)

        actual = func(ds, other)

        assert expected == actual

    @pytest.mark.parametrize(
        "unit",
        (
            pytest.param(1, id="no_unit"),
            pytest.param(unit_registry.dimensionless, id="dimensionless"),
            pytest.param(unit_registry.s, id="incompatible_unit"),
            pytest.param(unit_registry.cm, id="compatible_unit"),
            pytest.param(unit_registry.m, id="identical_unit"),
        ),
    )
    def test_broadcast_like(self, unit, dtype):
        array1 = np.linspace(1, 2, 2 * 1).reshape(2, 1).astype(dtype) * unit_registry.Pa
        array2 = np.linspace(0, 1, 2 * 3).reshape(2, 3).astype(dtype) * unit_registry.Pa

        x1 = np.arange(2) * unit_registry.m
        x2 = np.arange(2) * unit
        y1 = np.array([0]) * unit_registry.m
        y2 = np.arange(3) * unit

        ds1 = xr.Dataset(
            data_vars={"a": (("x", "y"), array1)}, coords={"x": x1, "y": y1}
        )
        ds2 = xr.Dataset(
            data_vars={"a": (("x", "y"), array2)}, coords={"x": x2, "y": y2}
        )

        expected = attach_units(
            strip_units(ds1).broadcast_like(strip_units(ds2)), extract_units(ds1)
        )
        actual = ds1.broadcast_like(ds2)

        assert_equal_with_units(expected, actual)

    @pytest.mark.parametrize(
        "unit",
        (
            pytest.param(1, id="no_unit"),
            pytest.param(unit_registry.dimensionless, id="dimensionless"),
            pytest.param(unit_registry.s, id="incompatible_unit"),
            pytest.param(unit_registry.cm, id="compatible_unit"),
            pytest.param(unit_registry.m, id="identical_unit"),
        ),
    )
    def test_broadcast_equals(self, unit, dtype):
        left_array1 = np.ones(shape=(2, 3), dtype=dtype) * unit_registry.m
        left_array2 = np.zeros(shape=(3, 6), dtype=dtype) * unit_registry.m

        right_array1 = np.ones(shape=(2,)) * unit
        right_array2 = np.ones(shape=(3,)) * unit

        left = xr.Dataset(
            data_vars={
                "a": xr.DataArray(data=left_array1, dims=("x", "y")),
                "b": xr.DataArray(data=left_array2, dims=("y", "z")),
            }
        )
        right = xr.Dataset(
            data_vars={
                "a": xr.DataArray(data=right_array1, dims="x"),
                "b": xr.DataArray(data=right_array2, dims="y"),
            }
        )

        units = {
            **extract_units(left),
            **({} if left_array1.check(unit) else {"a": None, "b": None}),
        }
        expected = strip_units(left).broadcast_equals(
            strip_units(convert_units(right, units))
        ) & left_array1.check(unit)
        actual = left.broadcast_equals(right)

        assert expected == actual

    @pytest.mark.parametrize(
        "func",
        (method("unstack"), method("reset_index", "v"), method("reorder_levels")),
        ids=repr,
    )
    def test_stacking_stacked(self, func, dtype):
        array1 = (
            np.linspace(0, 10, 5 * 10).reshape(5, 10).astype(dtype) * unit_registry.m
        )
        array2 = (
            np.linspace(-10, 0, 5 * 10 * 15).reshape(5, 10, 15).astype(dtype)
            * unit_registry.m
        )

        x = np.arange(array1.shape[0])
        y = np.arange(array1.shape[1])
        z = np.arange(array2.shape[2])

        ds = xr.Dataset(
            data_vars={
                "a": xr.DataArray(data=array1, dims=("x", "y")),
                "b": xr.DataArray(data=array2, dims=("x", "y", "z")),
            },
            coords={"x": x, "y": y, "z": z},
        )

        stacked = ds.stack(v=("x", "y"))

        expected = attach_units(
            func(strip_units(stacked)), {"a": unit_registry.m, "b": unit_registry.m}
        )
        actual = func(stacked)

        assert_equal_with_units(expected, actual)

    @pytest.mark.xfail(reason="does not work with quantities yet")
    def test_to_stacked_array(self, dtype):
        labels = np.arange(5).astype(dtype) * unit_registry.s
        arrays = {name: np.linspace(0, 1, 10) * unit_registry.m for name in labels}

        ds = xr.Dataset(
            data_vars={
                name: xr.DataArray(data=array, dims="x")
                for name, array in arrays.items()
            }
        )

        func = method("to_stacked_array", "z", variable_dim="y", sample_dims=["x"])

        actual = func(ds).rename(None)
        expected = attach_units(
            func(strip_units(ds)).rename(None),
            {None: unit_registry.m, "y": unit_registry.s},
        )

        assert_equal_with_units(expected, actual)

    @pytest.mark.parametrize(
        "func",
        (
            method("transpose", "y", "x", "z1", "z2"),
            method("stack", a=("x", "y")),
            method("set_index", x="x2"),
            pytest.param(
                method("shift", x=2),
                marks=pytest.mark.xfail(reason="tries to concatenate nan arrays"),
            ),
            method("roll", x=2, roll_coords=False),
            method("sortby", "x2"),
        ),
        ids=repr,
    )
    def test_stacking_reordering(self, func, dtype):
        array1 = (
            np.linspace(0, 10, 2 * 5 * 10).reshape(2, 5, 10).astype(dtype)
            * unit_registry.Pa
        )
        array2 = (
            np.linspace(0, 10, 2 * 5 * 15).reshape(2, 5, 15).astype(dtype)
            * unit_registry.degK
        )

        x = np.arange(array1.shape[0])
        y = np.arange(array1.shape[1])
        z1 = np.arange(array1.shape[2])
        z2 = np.arange(array2.shape[2])

        x2 = np.linspace(0, 1, array1.shape[0])[::-1]

        ds = xr.Dataset(
            data_vars={
                "a": xr.DataArray(data=array1, dims=("x", "y", "z1")),
                "b": xr.DataArray(data=array2, dims=("x", "y", "z2")),
            },
            coords={"x": x, "y": y, "z1": z1, "z2": z2, "x2": ("x", x2)},
        )

        expected = attach_units(
            func(strip_units(ds)), {"a": unit_registry.Pa, "b": unit_registry.degK}
        )
        actual = func(ds)

        assert_equal_with_units(expected, actual)

    @pytest.mark.xfail(reason="indexes strip units")
    @pytest.mark.parametrize(
        "indices",
        (
            pytest.param(4, id="single index"),
            pytest.param([5, 2, 9, 1], id="multiple indices"),
        ),
    )
    def test_isel(self, indices, dtype):
        array1 = np.arange(10).astype(dtype) * unit_registry.s
        array2 = np.linspace(0, 1, 10).astype(dtype) * unit_registry.Pa

        x = np.arange(len(array1)) * unit_registry.m
        ds = xr.Dataset(
            data_vars={
                "a": xr.DataArray(data=array1, dims="x"),
                "b": xr.DataArray(data=array2, dims="x"),
            },
            coords={"x": x},
        )

        expected = attach_units(
            strip_units(ds).isel(x=indices),
            {"a": unit_registry.s, "b": unit_registry.Pa, "x": unit_registry.m},
        )
        actual = ds.isel(x=indices)

        assert_equal_with_units(expected, actual)

    @pytest.mark.xfail(reason="indexes don't support units")
    @pytest.mark.parametrize(
        "raw_values",
        (
            pytest.param(10, id="single_value"),
            pytest.param([10, 5, 13], id="list_of_values"),
            pytest.param(np.array([9, 3, 7, 12]), id="array_of_values"),
        ),
    )
    @pytest.mark.parametrize(
        "unit,error",
        (
            pytest.param(1, KeyError, id="no_units"),
            pytest.param(unit_registry.dimensionless, KeyError, id="dimensionless"),
            pytest.param(unit_registry.degree, KeyError, id="incompatible_unit"),
            pytest.param(unit_registry.dm, KeyError, id="compatible_unit"),
            pytest.param(unit_registry.m, None, id="identical_unit"),
        ),
    )
    def test_sel(self, raw_values, unit, error, dtype):
        array1 = np.linspace(5, 10, 20).astype(dtype) * unit_registry.degK
        array2 = np.linspace(0, 5, 20).astype(dtype) * unit_registry.Pa
        x = np.arange(len(array1)) * unit_registry.m

        ds = xr.Dataset(
            data_vars={
                "a": xr.DataArray(data=array1, dims="x"),
                "b": xr.DataArray(data=array2, dims="x"),
            },
            coords={"x": x},
        )

        values = raw_values * unit

        if error is not None and not (
            isinstance(raw_values, (int, float)) and x.check(unit)
        ):
            with pytest.raises(error):
                ds.sel(x=values)

            return

        expected = attach_units(
            strip_units(ds).sel(x=strip_units(convert_units(values, {None: x.units}))),
            {"a": array1.units, "b": array2.units, "x": x.units},
        )
        actual = ds.sel(x=values)
        assert_equal_with_units(expected, actual)

    @pytest.mark.xfail(reason="indexes don't support units")
    @pytest.mark.parametrize(
        "raw_values",
        (
            pytest.param(10, id="single_value"),
            pytest.param([10, 5, 13], id="list_of_values"),
            pytest.param(np.array([9, 3, 7, 12]), id="array_of_values"),
        ),
    )
    @pytest.mark.parametrize(
        "unit,error",
        (
            pytest.param(1, KeyError, id="no_units"),
            pytest.param(unit_registry.dimensionless, KeyError, id="dimensionless"),
            pytest.param(unit_registry.degree, KeyError, id="incompatible_unit"),
            pytest.param(unit_registry.dm, KeyError, id="compatible_unit"),
            pytest.param(unit_registry.m, None, id="identical_unit"),
        ),
    )
    def test_drop_sel(self, raw_values, unit, error, dtype):
        array1 = np.linspace(5, 10, 20).astype(dtype) * unit_registry.degK
        array2 = np.linspace(0, 5, 20).astype(dtype) * unit_registry.Pa
        x = np.arange(len(array1)) * unit_registry.m

        ds = xr.Dataset(
            data_vars={
                "a": xr.DataArray(data=array1, dims="x"),
                "b": xr.DataArray(data=array2, dims="x"),
            },
            coords={"x": x},
        )

        values = raw_values * unit

        if error is not None and not (
            isinstance(raw_values, (int, float)) and x.check(unit)
        ):
            with pytest.raises(error):
                ds.drop_sel(x=values)

            return

        expected = attach_units(
            strip_units(ds).drop_sel(
                x=strip_units(convert_units(values, {None: x.units}))
            ),
            extract_units(ds),
        )
        actual = ds.drop_sel(x=values)
        assert_equal_with_units(expected, actual)

    @pytest.mark.xfail(reason="indexes don't support units")
    @pytest.mark.parametrize(
        "raw_values",
        (
            pytest.param(10, id="single_value"),
            pytest.param([10, 5, 13], id="list_of_values"),
            pytest.param(np.array([9, 3, 7, 12]), id="array_of_values"),
        ),
    )
    @pytest.mark.parametrize(
        "unit,error",
        (
            pytest.param(1, KeyError, id="no_units"),
            pytest.param(unit_registry.dimensionless, KeyError, id="dimensionless"),
            pytest.param(unit_registry.degree, KeyError, id="incompatible_unit"),
            pytest.param(unit_registry.dm, KeyError, id="compatible_unit"),
            pytest.param(unit_registry.m, None, id="identical_unit"),
        ),
    )
    def test_loc(self, raw_values, unit, error, dtype):
        array1 = np.linspace(5, 10, 20).astype(dtype) * unit_registry.degK
        array2 = np.linspace(0, 5, 20).astype(dtype) * unit_registry.Pa
        x = np.arange(len(array1)) * unit_registry.m

        ds = xr.Dataset(
            data_vars={
                "a": xr.DataArray(data=array1, dims="x"),
                "b": xr.DataArray(data=array2, dims="x"),
            },
            coords={"x": x},
        )

        values = raw_values * unit

        if error is not None and not (
            isinstance(raw_values, (int, float)) and x.check(unit)
        ):
            with pytest.raises(error):
                ds.loc[{"x": values}]

            return

        expected = attach_units(
            strip_units(ds).loc[
                {"x": strip_units(convert_units(values, {None: x.units}))}
            ],
            {"a": array1.units, "b": array2.units, "x": x.units},
        )
        actual = ds.loc[{"x": values}]
        assert_equal_with_units(expected, actual)

    @pytest.mark.parametrize(
        "func",
        (
            method("head", x=7, y=3, z=6),
            method("tail", x=7, y=3, z=6),
            method("thin", x=7, y=3, z=6),
        ),
        ids=repr,
    )
    def test_head_tail_thin(self, func, dtype):
        array1 = np.linspace(1, 2, 10 * 5).reshape(10, 5) * unit_registry.degK
        array2 = np.linspace(1, 2, 10 * 8).reshape(10, 8) * unit_registry.Pa

        coords = {
            "x": np.arange(10) * unit_registry.m,
            "y": np.arange(5) * unit_registry.m,
            "z": np.arange(8) * unit_registry.m,
        }

        ds = xr.Dataset(
            data_vars={
                "a": xr.DataArray(data=array1, dims=("x", "y")),
                "b": xr.DataArray(data=array2, dims=("x", "z")),
            },
            coords=coords,
        )

        expected = attach_units(func(strip_units(ds)), extract_units(ds))
        actual = func(ds)

        assert_equal_with_units(expected, actual)

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
        array1 = (
            np.linspace(0, 1, 10 * 20).astype(dtype).reshape(shape) * unit_registry.degK
        )
        array2 = (
            np.linspace(1, 2, 10 * 20).astype(dtype).reshape(shape) * unit_registry.Pa
        )

        ds = xr.Dataset(
            data_vars={
                "a": xr.DataArray(data=array1, dims=tuple(names[: len(shape)])),
                "b": xr.DataArray(data=array2, dims=tuple(names[: len(shape)])),
            },
            coords=coords,
        )
        units = extract_units(ds)

        expected = attach_units(strip_units(ds).squeeze(), units)

        actual = ds.squeeze()
        assert_equal_with_units(actual, expected)

        # try squeezing the dimensions separately
        names = tuple(dim for dim, coord in coords.items() if len(coord) == 1)
        for name in names:
            expected = attach_units(strip_units(ds).squeeze(dim=name), units)
            actual = ds.squeeze(dim=name)
            assert_equal_with_units(actual, expected)

    @pytest.mark.xfail(reason="ignores units")
    @pytest.mark.parametrize(
        "unit,error",
        (
            pytest.param(1, DimensionalityError, id="no_unit"),
            pytest.param(
                unit_registry.dimensionless, DimensionalityError, id="dimensionless"
            ),
            pytest.param(unit_registry.s, DimensionalityError, id="incompatible_unit"),
            pytest.param(unit_registry.cm, None, id="compatible_unit"),
            pytest.param(unit_registry.m, None, id="identical_unit"),
        ),
    )
    def test_interp(self, unit, error):
        array1 = np.linspace(1, 2, 10 * 5).reshape(10, 5) * unit_registry.degK
        array2 = np.linspace(1, 2, 10 * 8).reshape(10, 8) * unit_registry.Pa

        coords = {
            "x": np.arange(10) * unit_registry.m,
            "y": np.arange(5) * unit_registry.m,
            "z": np.arange(8) * unit_registry.s,
        }

        ds = xr.Dataset(
            data_vars={
                "a": xr.DataArray(data=array1, dims=("x", "y")),
                "b": xr.DataArray(data=array2, dims=("x", "z")),
            },
            coords=coords,
        )

        new_coords = (np.arange(10) + 0.5) * unit

        if error is not None:
            with pytest.raises(error):
                ds.interp(x=new_coords)

            return

        units = extract_units(ds)
        expected = attach_units(
            strip_units(ds).interp(x=strip_units(convert_units(new_coords, units))),
            units,
        )
        actual = ds.interp(x=new_coords)

        assert_equal_with_units(actual, expected)

    @pytest.mark.xfail(reason="ignores units")
    @pytest.mark.parametrize(
        "unit,error",
        (
            pytest.param(1, DimensionalityError, id="no_unit"),
            pytest.param(
                unit_registry.dimensionless, DimensionalityError, id="dimensionless"
            ),
            pytest.param(unit_registry.s, DimensionalityError, id="incompatible_unit"),
            pytest.param(unit_registry.cm, None, id="compatible_unit"),
            pytest.param(unit_registry.m, None, id="identical_unit"),
        ),
    )
    def test_interp_like(self, unit, error, dtype):
        array1 = (
            np.linspace(0, 10, 10 * 5).reshape(10, 5).astype(dtype) * unit_registry.degK
        )
        array2 = (
            np.linspace(10, 20, 10 * 8).reshape(10, 8).astype(dtype) * unit_registry.Pa
        )

        coords = {
            "x": np.arange(10) * unit_registry.m,
            "y": np.arange(5) * unit_registry.m,
            "z": np.arange(8) * unit_registry.m,
        }

        ds = xr.Dataset(
            data_vars={
                "a": xr.DataArray(data=array1, dims=("x", "y")),
                "b": xr.DataArray(data=array2, dims=("x", "z")),
            },
            coords=coords,
        )

        other = xr.Dataset(
            data_vars={
                "c": xr.DataArray(data=np.empty((20, 10)), dims=("x", "y")),
                "d": xr.DataArray(data=np.empty((20, 15)), dims=("x", "z")),
            },
            coords={
                "x": (np.arange(20) + 0.3) * unit,
                "y": (np.arange(10) - 0.2) * unit,
                "z": (np.arange(15) + 0.4) * unit,
            },
        )

        if error is not None:
            with pytest.raises(error):
                ds.interp_like(other)

            return

        units = extract_units(ds)
        expected = attach_units(
            strip_units(ds).interp_like(strip_units(convert_units(other, units))), units
        )
        actual = ds.interp_like(other)

        assert_equal_with_units(actual, expected)

    @pytest.mark.xfail(reason="indexes don't support units")
    @pytest.mark.parametrize(
        "unit,error",
        (
            pytest.param(1, DimensionalityError, id="no_unit"),
            pytest.param(
                unit_registry.dimensionless, DimensionalityError, id="dimensionless"
            ),
            pytest.param(unit_registry.s, DimensionalityError, id="incompatible_unit"),
            pytest.param(unit_registry.cm, None, id="compatible_unit"),
            pytest.param(unit_registry.m, None, id="identical_unit"),
        ),
    )
    def test_reindex(self, unit, error, dtype):
        array1 = (
            np.linspace(1, 2, 10 * 5).reshape(10, 5).astype(dtype) * unit_registry.degK
        )
        array2 = (
            np.linspace(1, 2, 10 * 8).reshape(10, 8).astype(dtype) * unit_registry.Pa
        )

        coords = {
            "x": np.arange(10) * unit_registry.m,
            "y": np.arange(5) * unit_registry.m,
            "z": np.arange(8) * unit_registry.s,
        }

        ds = xr.Dataset(
            data_vars={
                "a": xr.DataArray(data=array1, dims=("x", "y")),
                "b": xr.DataArray(data=array2, dims=("x", "z")),
            },
            coords=coords,
        )

        new_coords = (np.arange(10) + 0.5) * unit

        if error is not None:
            with pytest.raises(error):
                ds.reindex(x=new_coords)

            return

        expected = attach_units(
            strip_units(ds).reindex(
                x=strip_units(convert_units(new_coords, {None: coords["x"].units}))
            ),
            extract_units(ds),
        )
        actual = ds.reindex(x=new_coords)

        assert_equal_with_units(actual, expected)

    @pytest.mark.xfail(reason="indexes don't support units")
    @pytest.mark.parametrize(
        "unit,error",
        (
            pytest.param(1, DimensionalityError, id="no_unit"),
            pytest.param(
                unit_registry.dimensionless, DimensionalityError, id="dimensionless"
            ),
            pytest.param(unit_registry.s, DimensionalityError, id="incompatible_unit"),
            pytest.param(unit_registry.cm, None, id="compatible_unit"),
            pytest.param(unit_registry.m, None, id="identical_unit"),
        ),
    )
    def test_reindex_like(self, unit, error, dtype):
        array1 = (
            np.linspace(0, 10, 10 * 5).reshape(10, 5).astype(dtype) * unit_registry.degK
        )
        array2 = (
            np.linspace(10, 20, 10 * 8).reshape(10, 8).astype(dtype) * unit_registry.Pa
        )

        coords = {
            "x": np.arange(10) * unit_registry.m,
            "y": np.arange(5) * unit_registry.m,
            "z": np.arange(8) * unit_registry.m,
        }

        ds = xr.Dataset(
            data_vars={
                "a": xr.DataArray(data=array1, dims=("x", "y")),
                "b": xr.DataArray(data=array2, dims=("x", "z")),
            },
            coords=coords,
        )

        other = xr.Dataset(
            data_vars={
                "c": xr.DataArray(data=np.empty((20, 10)), dims=("x", "y")),
                "d": xr.DataArray(data=np.empty((20, 15)), dims=("x", "z")),
            },
            coords={
                "x": (np.arange(20) + 0.3) * unit,
                "y": (np.arange(10) - 0.2) * unit,
                "z": (np.arange(15) + 0.4) * unit,
            },
        )

        if error is not None:
            with pytest.raises(error):
                ds.reindex_like(other)

            return

        units = extract_units(ds)
        expected = attach_units(
            strip_units(ds).reindex_like(strip_units(convert_units(other, units))),
            units,
        )
        actual = ds.reindex_like(other)

        assert_equal_with_units(expected, actual)

    @pytest.mark.parametrize(
        "func",
        (
            method("diff", dim="x"),
            method("differentiate", coord="x"),
            method("integrate", coord="x"),
            pytest.param(
                method("quantile", q=[0.25, 0.75]),
                marks=pytest.mark.xfail(reason="nanquantile not implemented"),
            ),
            method("reduce", func=np.sum, dim="x"),
            method("map", np.fabs),
        ),
        ids=repr,
    )
    def test_computation(self, func, dtype):
        array1 = (
            np.linspace(-5, 5, 10 * 5).reshape(10, 5).astype(dtype) * unit_registry.degK
        )
        array2 = (
            np.linspace(10, 20, 10 * 8).reshape(10, 8).astype(dtype) * unit_registry.Pa
        )
        x = np.arange(10) * unit_registry.m
        y = np.arange(5) * unit_registry.m
        z = np.arange(8) * unit_registry.m

        ds = xr.Dataset(
            data_vars={
                "a": xr.DataArray(data=array1, dims=("x", "y")),
                "b": xr.DataArray(data=array2, dims=("x", "z")),
            },
            coords={"x": x, "y": y, "z": z},
        )

        units = extract_units(ds)

        expected = attach_units(func(strip_units(ds)), units)
        actual = func(ds)

        assert_equal_with_units(expected, actual)

    @pytest.mark.parametrize(
        "func",
        (
            method("groupby", "x"),
            method("groupby_bins", "x", bins=4),
            method("coarsen", x=2),
            pytest.param(
                method("rolling", x=3), marks=pytest.mark.xfail(reason="strips units")
            ),
            pytest.param(
                method("rolling_exp", x=3),
                marks=pytest.mark.xfail(reason="uses numbagg which strips units"),
            ),
        ),
        ids=repr,
    )
    def test_computation_objects(self, func, dtype):
        array1 = (
            np.linspace(-5, 5, 10 * 5).reshape(10, 5).astype(dtype) * unit_registry.degK
        )
        array2 = (
            np.linspace(10, 20, 10 * 5 * 8).reshape(10, 5, 8).astype(dtype)
            * unit_registry.Pa
        )
        x = np.arange(10) * unit_registry.m
        y = np.arange(5) * unit_registry.m
        z = np.arange(8) * unit_registry.m

        ds = xr.Dataset(
            data_vars={
                "a": xr.DataArray(data=array1, dims=("x", "y")),
                "b": xr.DataArray(data=array2, dims=("x", "y", "z")),
            },
            coords={"x": x, "y": y, "z": z},
        )
        units = extract_units(ds)

        args = [] if func.name != "groupby" else ["y"]
        reduce_func = method("mean", *args)
        expected = attach_units(reduce_func(func(strip_units(ds))), units)
        actual = reduce_func(func(ds))

        assert_equal_with_units(expected, actual)

    def test_resample(self, dtype):
        array1 = (
            np.linspace(-5, 5, 10 * 5).reshape(10, 5).astype(dtype) * unit_registry.degK
        )
        array2 = (
            np.linspace(10, 20, 10 * 8).reshape(10, 8).astype(dtype) * unit_registry.Pa
        )
        t = pd.date_range("10-09-2010", periods=array1.shape[0], freq="1y")
        y = np.arange(5) * unit_registry.m
        z = np.arange(8) * unit_registry.m

        ds = xr.Dataset(
            data_vars={
                "a": xr.DataArray(data=array1, dims=("time", "y")),
                "b": xr.DataArray(data=array2, dims=("time", "z")),
            },
            coords={"time": t, "y": y, "z": z},
        )
        units = extract_units(ds)

        func = method("resample", time="6m")

        expected = attach_units(func(strip_units(ds)).mean(), units)
        actual = func(ds).mean()

        assert_equal_with_units(expected, actual)

    @pytest.mark.parametrize(
        "func",
        (
            method("assign", c=lambda ds: 10 * ds.b),
            method("assign_coords", v=("x", np.arange(10) * unit_registry.s)),
            method("first"),
            method("last"),
            pytest.param(
                method("quantile", q=[0.25, 0.5, 0.75], dim="x"),
                marks=pytest.mark.xfail(reason="nanquantile not implemented"),
            ),
        ),
        ids=repr,
    )
    def test_grouped_operations(self, func, dtype):
        array1 = (
            np.linspace(-5, 5, 10 * 5).reshape(10, 5).astype(dtype) * unit_registry.degK
        )
        array2 = (
            np.linspace(10, 20, 10 * 5 * 8).reshape(10, 5, 8).astype(dtype)
            * unit_registry.Pa
        )
        x = np.arange(10) * unit_registry.m
        y = np.arange(5) * unit_registry.m
        z = np.arange(8) * unit_registry.m

        ds = xr.Dataset(
            data_vars={
                "a": xr.DataArray(data=array1, dims=("x", "y")),
                "b": xr.DataArray(data=array2, dims=("x", "y", "z")),
            },
            coords={"x": x, "y": y, "z": z},
        )
        units = extract_units(ds)
        units.update({"c": unit_registry.Pa, "v": unit_registry.s})

        stripped_kwargs = {
            name: strip_units(value) for name, value in func.kwargs.items()
        }
        expected = attach_units(
            func(strip_units(ds).groupby("y"), **stripped_kwargs), units
        )
        actual = func(ds.groupby("y"))

        assert_equal_with_units(expected, actual)

    @pytest.mark.parametrize(
        "func",
        (
            method("pipe", lambda ds: ds * 10),
            method("assign", d=lambda ds: ds.b * 10),
            method("assign_coords", y2=("y", np.arange(5) * unit_registry.mm)),
            method("assign_attrs", attr1="value"),
            method("rename", x2="x_mm"),
            method("rename_vars", c="temperature"),
            method("rename_dims", x="offset_x"),
            method("swap_dims", {"x": "x2"}),
            method("expand_dims", v=np.linspace(10, 20, 12) * unit_registry.s, axis=1),
            method("drop_vars", "x"),
            method("drop_dims", "z"),
            method("set_coords", names="c"),
            method("reset_coords", names="x2"),
            method("copy"),
        ),
        ids=repr,
    )
    def test_content_manipulation(self, func, dtype):
        array1 = (
            np.linspace(-5, 5, 10 * 5).reshape(10, 5).astype(dtype)
            * unit_registry.m ** 3
        )
        array2 = (
            np.linspace(10, 20, 10 * 5 * 8).reshape(10, 5, 8).astype(dtype)
            * unit_registry.Pa
        )
        array3 = np.linspace(0, 10, 10).astype(dtype) * unit_registry.degK

        x = np.arange(10) * unit_registry.m
        x2 = x.to(unit_registry.mm)
        y = np.arange(5) * unit_registry.m
        z = np.arange(8) * unit_registry.m

        ds = xr.Dataset(
            data_vars={
                "a": xr.DataArray(data=array1, dims=("x", "y")),
                "b": xr.DataArray(data=array2, dims=("x", "y", "z")),
                "c": xr.DataArray(data=array3, dims="x"),
            },
            coords={"x": x, "y": y, "z": z, "x2": ("x", x2)},
        )
        units = {
            **extract_units(ds),
            **{
                "y2": unit_registry.mm,
                "x_mm": unit_registry.mm,
                "offset_x": unit_registry.m,
                "d": unit_registry.Pa,
                "temperature": unit_registry.degK,
            },
        }

        stripped_kwargs = {
            key: strip_units(value) for key, value in func.kwargs.items()
        }
        expected = attach_units(func(strip_units(ds), **stripped_kwargs), units)
        actual = func(ds)

        assert_equal_with_units(expected, actual)

    @pytest.mark.parametrize(
        "unit,error",
        (
            pytest.param(1, xr.MergeError, id="no_unit"),
            pytest.param(
                unit_registry.dimensionless, xr.MergeError, id="dimensionless"
            ),
            pytest.param(unit_registry.s, xr.MergeError, id="incompatible_unit"),
            pytest.param(unit_registry.cm, xr.MergeError, id="compatible_unit"),
            pytest.param(unit_registry.m, None, id="identical_unit"),
        ),
    )
    @pytest.mark.parametrize(
        "variant",
        (
            "data",
            pytest.param(
                "dims", marks=pytest.mark.xfail(reason="indexes don't support units")
            ),
            "coords",
        ),
    )
    def test_merge(self, variant, unit, error, dtype):
        original_data_unit = unit_registry.m
        original_dim_unit = unit_registry.m
        original_coord_unit = unit_registry.m

        variants = {
            "data": (unit, original_dim_unit, original_coord_unit),
            "dims": (original_data_unit, unit, original_coord_unit),
            "coords": (original_data_unit, original_dim_unit, unit),
        }
        data_unit, dim_unit, coord_unit = variants.get(variant)

        left_array = np.arange(10).astype(dtype) * original_data_unit
        right_array = np.arange(-5, 5).astype(dtype) * data_unit

        left_dim = np.arange(10, 20) * original_dim_unit
        right_dim = np.arange(5, 15) * dim_unit

        left_coord = np.arange(-10, 0) * original_coord_unit
        right_coord = np.arange(-15, -5) * coord_unit

        left = xr.Dataset(
            data_vars={"a": ("x", left_array)},
            coords={"x": left_dim, "y": ("x", left_coord)},
        )
        right = xr.Dataset(
            data_vars={"a": ("x", right_array)},
            coords={"x": right_dim, "y": ("x", right_coord)},
        )

        units = extract_units(left)

        if error is not None:
            with pytest.raises(error):
                left.merge(right)

            return

        converted = convert_units(right, units)
        expected = attach_units(strip_units(left).merge(strip_units(converted)), units)
        actual = left.merge(right)

        assert_equal_with_units(expected, actual)
