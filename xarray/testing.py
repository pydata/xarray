"""Testing functions exposed to the user API"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from xarray.core import ops


def _decode_string_data(data):
    if data.dtype.kind == 'S':
        return np.core.defchararray.decode(data, 'utf-8', 'replace')
    return data


def data_allclose_or_equiv(arr1, arr2, rtol=1e-05, atol=1e-08,
                           decode_bytes=True):
    if any(arr.dtype.kind == 'S' for arr in [arr1, arr2]) and decode_bytes:
        arr1 = _decode_string_data(arr1)
        arr2 = _decode_string_data(arr2)
    exact_dtypes = ['M', 'm', 'O', 'U']
    if any(arr.dtype.kind in exact_dtypes for arr in [arr1, arr2]):
        return ops.array_equiv(arr1, arr2)
    else:
        return ops.allclose_or_equiv(arr1, arr2, rtol=rtol, atol=atol)


def assert_equal(a, b):
    """Like numpy.testing.assert_array_equal, but for xarray objects."""
    import xarray as xr
    ___tracebackhide__ = True  # noqa: F841
    assert type(a) == type(b)
    if isinstance(a, (xr.Variable, xr.DataArray, xr.Dataset)):
        assert a.equals(b), '{}\n{}'.format(a, b)
    else:
        raise TypeError('{} not supported by assertion comparison'
                        .format(type(a)))


def assert_identical(a, b):
    """Like assert_equal, but also checks the objects' names and attributes."""
    import xarray as xr
    ___tracebackhide__ = True  # noqa: F841
    assert type(a) == type(b)
    if isinstance(a, xr.DataArray):
        assert a.name == b.name
        assert_identical(a._to_temp_dataset(), b._to_temp_dataset())
    elif isinstance(a, (xr.Dataset, xr.Variable)):
        assert a.identical(b), '{}\n{}'.format(a, b)
    else:
        raise TypeError('{} not supported by assertion comparison'
                        .format(type(a)))


def assert_allclose(a, b, rtol=1e-05, atol=1e-08):
    """Like numpy.testing.assert_allclose, but for xarray objects."""
    import xarray as xr
    ___tracebackhide__ = True  # noqa: F841
    assert type(a) == type(b)
    if isinstance(a, xr.Variable):
        assert a.dims == b.dims
        allclose = data_allclose_or_equiv(
            a.values, b.values, rtol=rtol, atol=atol)
        assert allclose, '{}\n{}'.format(a.values, b.values)
    elif isinstance(a, xr.DataArray):
        assert_allclose(a.variable, b.variable)
        for v in a.coords.variables:
            # can't recurse with this function as coord is sometimes a
            # DataArray, so call into data_allclose_or_equiv directly
            assert set(a.coords) == set(b.coords)
            allclose = data_allclose_or_equiv(
                a.coords[v].values, b.coords[v].values, rtol=rtol, atol=atol)
            assert allclose, '{}\n{}'.format(a.coords[v].values,
                                             b.coords[v].values)
    elif isinstance(a, xr.Dataset):
        assert set(a) == set(b)
        assert set(a.coords) == set(b.coords)
        for k in list(a.variables) + list(a.coords):
            assert_allclose(a[k], b[k], rtol=rtol, atol=atol)

    else:
        raise TypeError('{} not supported by assertion comparison'
                        .format(type(a)))
