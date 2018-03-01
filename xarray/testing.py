"""Testing functions exposed to the user API"""
from __future__ import absolute_import, division, print_function

import numpy as np

from xarray.core import duck_array_ops


def _decode_string_data(data):
    if data.dtype.kind == 'S':
        return np.core.defchararray.decode(data, 'utf-8', 'replace')
    return data


def _data_allclose_or_equiv(arr1, arr2, rtol=1e-05, atol=1e-08,
                            decode_bytes=True):
    if any(arr.dtype.kind == 'S' for arr in [arr1, arr2]) and decode_bytes:
        arr1 = _decode_string_data(arr1)
        arr2 = _decode_string_data(arr2)
    exact_dtypes = ['M', 'm', 'O', 'S', 'U']
    if any(arr.dtype.kind in exact_dtypes for arr in [arr1, arr2]):
        return duck_array_ops.array_equiv(arr1, arr2)
    else:
        return duck_array_ops.allclose_or_equiv(
            arr1, arr2, rtol=rtol, atol=atol)


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

    See also
    --------
    assert_identical, assert_allclose, Dataset.equals, DataArray.equals,
    numpy.testing.assert_array_equal
    """
    import xarray as xr
    __tracebackhide__ = True  # noqa: F841
    assert type(a) == type(b)  # noqa
    if isinstance(a, (xr.Variable, xr.DataArray, xr.Dataset)):
        assert a.equals(b), '{}\n{}'.format(a, b)
    else:
        raise TypeError('{} not supported by assertion comparison'
                        .format(type(a)))


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

    See also
    --------
    assert_equal, assert_allclose, Dataset.equals, DataArray.equals
    """
    import xarray as xr
    __tracebackhide__ = True  # noqa: F841
    assert type(a) == type(b)  # noqa
    if isinstance(a, xr.DataArray):
        assert a.name == b.name
        assert_identical(a._to_temp_dataset(), b._to_temp_dataset())
    elif isinstance(a, (xr.Dataset, xr.Variable)):
        assert a.identical(b), '{}\n{}'.format(a, b)
    else:
        raise TypeError('{} not supported by assertion comparison'
                        .format(type(a)))


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

    See also
    --------
    assert_identical, assert_equal, numpy.testing.assert_allclose
    """
    import xarray as xr
    __tracebackhide__ = True  # noqa: F841
    assert type(a) == type(b)  # noqa
    kwargs = dict(rtol=rtol, atol=atol, decode_bytes=decode_bytes)
    if isinstance(a, xr.Variable):
        assert a.dims == b.dims
        allclose = _data_allclose_or_equiv(a.values, b.values, **kwargs)
        assert allclose, '{}\n{}'.format(a.values, b.values)
    elif isinstance(a, xr.DataArray):
        assert_allclose(a.variable, b.variable, **kwargs)
        assert set(a.coords) == set(b.coords)
        for v in a.coords.variables:
            # can't recurse with this function as coord is sometimes a
            # DataArray, so call into _data_allclose_or_equiv directly
            allclose = _data_allclose_or_equiv(a.coords[v].values,
                                               b.coords[v].values, **kwargs)
            assert allclose, '{}\n{}'.format(a.coords[v].values,
                                             b.coords[v].values)
    elif isinstance(a, xr.Dataset):
        assert set(a.data_vars) == set(b.data_vars)
        assert set(a.coords) == set(b.coords)
        for k in list(a.variables) + list(a.coords):
            assert_allclose(a[k], b[k], **kwargs)

    else:
        raise TypeError('{} not supported by assertion comparison'
                        .format(type(a)))
