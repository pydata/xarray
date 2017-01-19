from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import warnings
from contextlib import contextmanager
from distutils.version import StrictVersion

import numpy as np
from numpy.testing import assert_array_equal
import pytest

from xarray.core import utils, nputils, ops
from xarray.core.variable import as_variable
from xarray.core.pycompat import PY3

try:
    import unittest2 as unittest
except ImportError:
    import unittest

try:
    import scipy
    has_scipy = True
except ImportError:
    has_scipy = False

try:
    import pydap.client
    has_pydap = True
except ImportError:
    has_pydap = False

try:
    import netCDF4
    has_netCDF4 = True
except ImportError:
    has_netCDF4 = False


try:
    import h5netcdf
    has_h5netcdf = True
except ImportError:
    has_h5netcdf = False


try:
    import Nio
    has_pynio = True
except ImportError:
    has_pynio = False


try:
    import dask.array
    import dask
    dask.set_options(get=dask.get)
    has_dask = True
except ImportError:
    has_dask = False


try:
    import matplotlib
    has_matplotlib = True
except ImportError:
    has_matplotlib = False


try:
    import bottleneck
    if StrictVersion(bottleneck.__version__) < StrictVersion('1.0'):
        raise ImportError('Fall back to numpy')
    has_bottleneck = True
except ImportError:
    has_bottleneck = False

# slighly simpler construction that the full functions.
# Generally `pytest.importorskip('package')` inline is even easier
requires_matplotlib = pytest.mark.skipif(not has_matplotlib, reason='requires matplotlib')


def requires_scipy(test):
    return test if has_scipy else pytest.mark.skip('requires scipy')(test)


def requires_pydap(test):
    return test if has_pydap else pytest.mark.skip('requires pydap.client')(test)


def requires_netCDF4(test):
    return test if has_netCDF4 else pytest.mark.skip('requires netCDF4')(test)


def requires_h5netcdf(test):
    return test if has_h5netcdf else pytest.mark.skip('requires h5netcdf')(test)


def requires_pynio(test):
    return test if has_pynio else pytest.mark.skip('requires pynio')(test)


def requires_scipy_or_netCDF4(test):
    return (test if has_scipy or has_netCDF4
            else pytest.mark.skip('requires scipy or netCDF4')(test))


def requires_dask(test):
    return test if has_dask else pytest.mark.skip('requires dask')(test)


def requires_bottleneck(test):
    return test if has_bottleneck else pytest.mark.skip('requires bottleneck')(test)


def decode_string_data(data):
    if data.dtype.kind == 'S':
        return np.core.defchararray.decode(data, 'utf-8', 'replace')
    return data


def data_allclose_or_equiv(arr1, arr2, rtol=1e-05, atol=1e-08):
    if any(arr.dtype.kind == 'S' for arr in [arr1, arr2]):
        arr1 = decode_string_data(arr1)
        arr2 = decode_string_data(arr2)
    exact_dtypes = ['M', 'm', 'O', 'U']
    if any(arr.dtype.kind in exact_dtypes for arr in [arr1, arr2]):
        return ops.array_equiv(arr1, arr2)
    else:
        return ops.allclose_or_equiv(arr1, arr2, rtol=rtol, atol=atol)


def assert_dataset_allclose(d1, d2, rtol=1e-05, atol=1e-08):
    assert sorted(d1, key=str) == sorted(d2, key=str)
    assert sorted(d1.coords, key=str) == sorted(d2.coords, key=str)
    for k in d1:
        v1 = d1.variables[k]
        v2 = d2.variables[k]
        assert v1.dims == v2.dims
        allclose = data_allclose_or_equiv(
            v1.values, v2.values, rtol=rtol, atol=atol)
        assert allclose, (k, v1.values, v2.values)


class TestCase(unittest.TestCase):
    if PY3:
        # Python 3 assertCountEqual is roughly equivalent to Python 2
        # assertItemsEqual
        def assertItemsEqual(self, first, second, msg=None):
            return self.assertCountEqual(first, second, msg)

    @contextmanager
    def assertWarns(self, message):
        with warnings.catch_warnings(record=True) as w:
            warnings.filterwarnings('always', message)
            yield
            assert len(w) > 0
            assert any(message in str(wi.message) for wi in w)

    def assertVariableEqual(self, v1, v2):
        assert_xarray_equal(v1, v2)

    def assertVariableIdentical(self, v1, v2):
        assert_xarray_identical(v1, v2)

    def assertVariableAllClose(self, v1, v2, rtol=1e-05, atol=1e-08):
        assert_xarray_allclose(v1, v2, rtol=rtol, atol=atol)

    def assertVariableNotEqual(self, v1, v2):
        assert not v1.equals(v2)

    def assertArrayEqual(self, a1, a2):
        assert_array_equal(a1, a2)

    def assertEqual(self, a1, a2):
        assert a1 == a2 or (a1 != a1 and a2 != a2)

    def assertDatasetEqual(self, d1, d2):
        assert_xarray_equal(d1, d2)

    def assertDatasetIdentical(self, d1, d2):
        assert_xarray_identical(d1, d2)

    def assertDatasetAllClose(self, d1, d2, rtol=1e-05, atol=1e-08):
        assert_xarray_allclose(d1, d2, rtol=rtol, atol=atol)

    def assertCoordinatesEqual(self, d1, d2):
        assert_xarray_equal(d1, d2)

    def assertDataArrayEqual(self, ar1, ar2):
        assert_xarray_equal(ar1, ar2)

    def assertDataArrayIdentical(self, ar1, ar2):
        assert_xarray_identical(ar1, ar2)

    def assertDataArrayAllClose(self, ar1, ar2, rtol=1e-05, atol=1e-08):
        assert_xarray_allclose(ar1, ar2, rtol=rtol, atol=atol)


def assert_xarray_equal(a, b):
    import xarray as xr
    ___tracebackhide__ = True  # noqa: F841
    assert type(a) == type(b)
    if isinstance(a, (xr.Variable, xr.DataArray, xr.Dataset)):
        assert a.equals(b), '{}\n{}'.format(a, b)
    else:
        raise TypeError('{} not supported by assertion comparison'
                        .format(type(a)))


def assert_xarray_identical(a, b):
    import xarray as xr
    ___tracebackhide__ = True  # noqa: F841
    assert type(a) == type(b)
    if isinstance(a, xr.DataArray):
        assert a.name == b.name
        assert_xarray_identical(a._to_temp_dataset(), b._to_temp_dataset())
    elif isinstance(a, (xr.Dataset, xr.Variable)):
        assert a.identical(b), '{}\n{}'.format(a, b)
    else:
        raise TypeError('{} not supported by assertion comparison'
                        .format(type(a)))


def assert_xarray_allclose(a, b, rtol=1e-05, atol=1e-08):
    import xarray as xr
    ___tracebackhide__ = True  # noqa: F841
    assert type(a) == type(b)
    if isinstance(a, xr.Variable):
        assert a.dims == b.dims
        allclose = data_allclose_or_equiv(
            a.values, b.values, rtol=rtol, atol=atol)
        assert allclose, '{}\n{}'.format(a.values, b.values)
    elif isinstance(a, xr.DataArray):
        assert_xarray_allclose(a.variable, b.variable)
        for v in a.coords.variables:
            # can't recurse with this function as coord is sometimes a DataArray,
            # so call into data_allclose_or_equiv directly
            allclose = data_allclose_or_equiv(
                a.coords[v].values, b.coords[v].values, rtol=rtol, atol=atol)
            assert allclose, '{}\n{}'.format(a.coords[v].values, b.coords[v].values)
    elif isinstance(a, xr.Dataset):
        assert sorted(a, key=str) == sorted(b, key=str)
        for k in list(a.variables) + list(a.coords):
            assert_xarray_allclose(a[k], b[k], rtol=rtol, atol=atol)

    else:
        raise TypeError('{} not supported by assertion comparison'
                        .format(type(a)))


class UnexpectedDataAccess(Exception):
    pass


class InaccessibleArray(utils.NDArrayMixin):
    def __init__(self, array):
        self.array = array

    def __getitem__(self, key):
        raise UnexpectedDataAccess("Tried accessing data")


class ReturnItem(object):
    def __getitem__(self, key):
        return key


def source_ndarray(array):
    """Given an ndarray, return the base object which holds its memory, or the
    object itself.
    """
    base = getattr(array, 'base', np.asarray(array).base)
    if base is None:
        base = array
    return base
