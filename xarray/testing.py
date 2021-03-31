"""Testing functions exposed to the user API"""
import functools
import warnings
from typing import Hashable, Set, Union

import numpy as np
import pandas as pd

from xarray.core import duck_array_ops, formatting, utils
from xarray.core.dataarray import DataArray
from xarray.core.dataset import Dataset
from xarray.core.indexes import default_indexes
from xarray.core.variable import IndexVariable, Variable

__all__ = (
    "assert_allclose",
    "assert_chunks_equal",
    "assert_duckarray_equal",
    "assert_duckarray_allclose",
    "assert_equal",
    "assert_identical",
)


def ensure_warnings(func):
    # sometimes tests elevate warnings to errors
    # -> make sure that does not happen in the assert_* functions
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        __tracebackhide__ = True

        with warnings.catch_warnings():
            warnings.simplefilter("always")

            return func(*args, **kwargs)

    return wrapper


def _decode_string_data(data):
    if data.dtype.kind == "S":
        return np.core.defchararray.decode(data, "utf-8", "replace")
    return data


def _data_allclose_or_equiv(arr1, arr2, rtol=1e-05, atol=1e-08, decode_bytes=True):
    if any(arr.dtype.kind == "S" for arr in [arr1, arr2]) and decode_bytes:
        arr1 = _decode_string_data(arr1)
        arr2 = _decode_string_data(arr2)
    exact_dtypes = ["M", "m", "O", "S", "U"]
    if any(arr.dtype.kind in exact_dtypes for arr in [arr1, arr2]):
        return duck_array_ops.array_equiv(arr1, arr2)
    else:
        return duck_array_ops.allclose_or_equiv(arr1, arr2, rtol=rtol, atol=atol)


@ensure_warnings
def assert_equal(a, b):
    """Like :py:func:`numpy.testing.assert_array_equal`, but for xarray
    objects.

    Raises an AssertionError if two objects are not equal. This will match
    data values, dimensions and coordinates, but not names or attributes
    (except for Dataset objects for which the variable names must match).
    Arrays with NaN in the same location are considered equal.

    Parameters
    ----------
    a : xarray.Dataset, xarray.DataArray or xarray.Variable
        The first object to compare.
    b : xarray.Dataset, xarray.DataArray or xarray.Variable
        The second object to compare.

    See Also
    --------
    assert_identical, assert_allclose, Dataset.equals, DataArray.equals
    numpy.testing.assert_array_equal
    """
    __tracebackhide__ = True
    assert type(a) == type(b)
    if isinstance(a, (Variable, DataArray)):
        assert a.equals(b), formatting.diff_array_repr(a, b, "equals")
    elif isinstance(a, Dataset):
        assert a.equals(b), formatting.diff_dataset_repr(a, b, "equals")
    else:
        raise TypeError("{} not supported by assertion comparison".format(type(a)))


@ensure_warnings
def assert_identical(a, b):
    """Like :py:func:`xarray.testing.assert_equal`, but also matches the
    objects' names and attributes.

    Raises an AssertionError if two objects are not identical.

    Parameters
    ----------
    a : xarray.Dataset, xarray.DataArray or xarray.Variable
        The first object to compare.
    b : xarray.Dataset, xarray.DataArray or xarray.Variable
        The second object to compare.

    See Also
    --------
    assert_equal, assert_allclose, Dataset.equals, DataArray.equals
    """
    __tracebackhide__ = True
    assert type(a) == type(b)
    if isinstance(a, Variable):
        assert a.identical(b), formatting.diff_array_repr(a, b, "identical")
    elif isinstance(a, DataArray):
        assert a.name == b.name
        assert a.identical(b), formatting.diff_array_repr(a, b, "identical")
    elif isinstance(a, (Dataset, Variable)):
        assert a.identical(b), formatting.diff_dataset_repr(a, b, "identical")
    else:
        raise TypeError("{} not supported by assertion comparison".format(type(a)))


@ensure_warnings
def assert_allclose(a, b, rtol=1e-05, atol=1e-08, decode_bytes=True):
    """Like :py:func:`numpy.testing.assert_allclose`, but for xarray objects.

    Raises an AssertionError if two objects are not equal up to desired
    tolerance.

    Parameters
    ----------
    a : xarray.Dataset, xarray.DataArray or xarray.Variable
        The first object to compare.
    b : xarray.Dataset, xarray.DataArray or xarray.Variable
        The second object to compare.
    rtol : float, optional
        Relative tolerance.
    atol : float, optional
        Absolute tolerance.
    decode_bytes : bool, optional
        Whether byte dtypes should be decoded to strings as UTF-8 or not.
        This is useful for testing serialization methods on Python 3 that
        return saved strings as bytes.

    See Also
    --------
    assert_identical, assert_equal, numpy.testing.assert_allclose
    """
    __tracebackhide__ = True
    assert type(a) == type(b)

    equiv = functools.partial(
        _data_allclose_or_equiv, rtol=rtol, atol=atol, decode_bytes=decode_bytes
    )
    equiv.__name__ = "allclose"

    def compat_variable(a, b):
        a = getattr(a, "variable", a)
        b = getattr(b, "variable", b)

        return a.dims == b.dims and (a._data is b._data or equiv(a.data, b.data))

    if isinstance(a, Variable):
        allclose = compat_variable(a, b)
        assert allclose, formatting.diff_array_repr(a, b, compat=equiv)
    elif isinstance(a, DataArray):
        allclose = utils.dict_equiv(
            a.coords, b.coords, compat=compat_variable
        ) and compat_variable(a.variable, b.variable)
        assert allclose, formatting.diff_array_repr(a, b, compat=equiv)
    elif isinstance(a, Dataset):
        allclose = a._coord_names == b._coord_names and utils.dict_equiv(
            a.variables, b.variables, compat=compat_variable
        )
        assert allclose, formatting.diff_dataset_repr(a, b, compat=equiv)
    else:
        raise TypeError("{} not supported by assertion comparison".format(type(a)))


def _format_message(x, y, err_msg, verbose):
    diff = x - y
    abs_diff = max(abs(diff))
    rel_diff = "not implemented"

    n_diff = int(np.count_nonzero(diff))
    n_total = diff.size

    fraction = f"{n_diff} / {n_total}"
    percentage = float(n_diff / n_total * 100)

    parts = [
        "Arrays are not equal",
        err_msg,
        f"Mismatched elements: {fraction} ({percentage:.0f}%)",
        f"Max absolute difference: {abs_diff}",
        f"Max relative difference: {rel_diff}",
    ]
    if verbose:
        parts += [
            f" x: {x!r}",
            f" y: {y!r}",
        ]

    return "\n".join(parts)


@ensure_warnings
def assert_duckarray_allclose(
    actual, desired, rtol=1e-07, atol=0, err_msg="", verbose=True
):
    """ Like `np.testing.assert_allclose`, but for duckarrays. """
    __tracebackhide__ = True

    allclose = duck_array_ops.allclose_or_equiv(actual, desired, rtol=rtol, atol=atol)
    assert allclose, _format_message(actual, desired, err_msg=err_msg, verbose=verbose)


@ensure_warnings
def assert_duckarray_equal(x, y, err_msg="", verbose=True):
    """ Like `np.testing.assert_array_equal`, but for duckarrays """
    __tracebackhide__ = True

    if not utils.is_duck_array(x) and not utils.is_scalar(x):
        x = np.asarray(x)

    if not utils.is_duck_array(y) and not utils.is_scalar(y):
        y = np.asarray(y)

    if (utils.is_duck_array(x) and utils.is_scalar(y)) or (
        utils.is_scalar(x) and utils.is_duck_array(y)
    ):
        equiv = (x == y).all()
    else:
        equiv = duck_array_ops.array_equiv(x, y)
    assert equiv, _format_message(x, y, err_msg=err_msg, verbose=verbose)


def assert_chunks_equal(a, b):
    """
    Assert that chunksizes along chunked dimensions are equal.

    Parameters
    ----------
    a : xarray.Dataset or xarray.DataArray
        The first object to compare.
    b : xarray.Dataset or xarray.DataArray
        The second object to compare.
    """

    if isinstance(a, DataArray) != isinstance(b, DataArray):
        raise TypeError("a and b have mismatched types")

    left = a.unify_chunks()
    right = b.unify_chunks()
    assert left.chunks == right.chunks


def _assert_indexes_invariants_checks(indexes, possible_coord_variables, dims):
    assert isinstance(indexes, dict), indexes
    assert all(isinstance(v, pd.Index) for v in indexes.values()), {
        k: type(v) for k, v in indexes.items()
    }

    index_vars = {
        k for k, v in possible_coord_variables.items() if isinstance(v, IndexVariable)
    }
    assert indexes.keys() <= index_vars, (set(indexes), index_vars)

    # Note: when we support non-default indexes, these checks should be opt-in
    # only!
    defaults = default_indexes(possible_coord_variables, dims)
    assert indexes.keys() == defaults.keys(), (set(indexes), set(defaults))
    assert all(v.equals(defaults[k]) for k, v in indexes.items()), (indexes, defaults)


def _assert_variable_invariants(var: Variable, name: Hashable = None):
    if name is None:
        name_or_empty: tuple = ()
    else:
        name_or_empty = (name,)
    assert isinstance(var._dims, tuple), name_or_empty + (var._dims,)
    assert len(var._dims) == len(var._data.shape), name_or_empty + (
        var._dims,
        var._data.shape,
    )
    assert isinstance(var._encoding, (type(None), dict)), name_or_empty + (
        var._encoding,
    )
    assert isinstance(var._attrs, (type(None), dict)), name_or_empty + (var._attrs,)


def _assert_dataarray_invariants(da: DataArray):
    assert isinstance(da._variable, Variable), da._variable
    _assert_variable_invariants(da._variable)

    assert isinstance(da._coords, dict), da._coords
    assert all(isinstance(v, Variable) for v in da._coords.values()), da._coords
    assert all(set(v.dims) <= set(da.dims) for v in da._coords.values()), (
        da.dims,
        {k: v.dims for k, v in da._coords.items()},
    )
    assert all(
        isinstance(v, IndexVariable) for (k, v) in da._coords.items() if v.dims == (k,)
    ), {k: type(v) for k, v in da._coords.items()}
    for k, v in da._coords.items():
        _assert_variable_invariants(v, k)

    if da._indexes is not None:
        _assert_indexes_invariants_checks(da._indexes, da._coords, da.dims)


def _assert_dataset_invariants(ds: Dataset):
    assert isinstance(ds._variables, dict), type(ds._variables)
    assert all(isinstance(v, Variable) for v in ds._variables.values()), ds._variables
    for k, v in ds._variables.items():
        _assert_variable_invariants(v, k)

    assert isinstance(ds._coord_names, set), ds._coord_names
    assert ds._coord_names <= ds._variables.keys(), (
        ds._coord_names,
        set(ds._variables),
    )

    assert type(ds._dims) is dict, ds._dims
    assert all(isinstance(v, int) for v in ds._dims.values()), ds._dims
    var_dims: Set[Hashable] = set()
    for v in ds._variables.values():
        var_dims.update(v.dims)
    assert ds._dims.keys() == var_dims, (set(ds._dims), var_dims)
    assert all(
        ds._dims[k] == v.sizes[k] for v in ds._variables.values() for k in v.sizes
    ), (ds._dims, {k: v.sizes for k, v in ds._variables.items()})
    assert all(
        isinstance(v, IndexVariable)
        for (k, v) in ds._variables.items()
        if v.dims == (k,)
    ), {k: type(v) for k, v in ds._variables.items() if v.dims == (k,)}
    assert all(v.dims == (k,) for (k, v) in ds._variables.items() if k in ds._dims), {
        k: v.dims for k, v in ds._variables.items() if k in ds._dims
    }

    if ds._indexes is not None:
        _assert_indexes_invariants_checks(ds._indexes, ds._variables, ds._dims)

    assert isinstance(ds._encoding, (type(None), dict))
    assert isinstance(ds._attrs, (type(None), dict))


def _assert_internal_invariants(xarray_obj: Union[DataArray, Dataset, Variable]):
    """Validate that an xarray object satisfies its own internal invariants.

    This exists for the benefit of xarray's own test suite, but may be useful
    in external projects if they (ill-advisedly) create objects using xarray's
    private APIs.
    """
    if isinstance(xarray_obj, Variable):
        _assert_variable_invariants(xarray_obj)
    elif isinstance(xarray_obj, DataArray):
        _assert_dataarray_invariants(xarray_obj)
    elif isinstance(xarray_obj, Dataset):
        _assert_dataset_invariants(xarray_obj)
    else:
        raise TypeError(
            "{} is not a supported type for xarray invariant checks".format(
                type(xarray_obj)
            )
        )
