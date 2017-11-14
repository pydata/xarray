from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import warnings
from contextlib import contextmanager
from distutils.version import LooseVersion
import re
import importlib

import numpy as np
from numpy.testing import assert_array_equal
from xarray.core.duck_array_ops import allclose_or_equiv
import pytest

from xarray.core import utils
from xarray.core.pycompat import PY3
from xarray.core.indexing import ExplicitlyIndexed
from xarray.testing import assert_equal, assert_identical, assert_allclose
from xarray.plot.utils import import_seaborn

try:
    from pandas.testing import assert_frame_equal
except ImportError:
    # old location, for pandas < 0.20
    from pandas.util.testing import assert_frame_equal

try:
    import unittest2 as unittest
except ImportError:
    import unittest

try:
    from unittest import mock
except ImportError:
    import mock

# import mpl and change the backend before other mpl imports
try:
    import matplotlib as mpl
    # Order of imports is important here.
    # Using a different backend makes Travis CI work
    mpl.use('Agg')
except ImportError:
    pass


def _importorskip(modname, minversion=None):
    try:
        mod = importlib.import_module(modname)
        has = True
        if minversion is not None:
            if LooseVersion(mod.__version__) < LooseVersion(minversion):
                raise ImportError('Minimum version not satisfied')
    except ImportError:
        has = False
    # TODO: use pytest.skipif instead of unittest.skipUnless
    # Using `unittest.skipUnless` is a temporary workaround for pytest#568,
    # wherein class decorators stain inherited classes.
    # xref: xarray#1531, implemented in xarray #1557.
    func = unittest.skipUnless(has, reason='requires {}'.format(modname))
    return has, func


has_matplotlib, requires_matplotlib = _importorskip('matplotlib')
has_scipy, requires_scipy = _importorskip('scipy')
has_pydap, requires_pydap = _importorskip('pydap.client')
has_netCDF4, requires_netCDF4 = _importorskip('netCDF4')
has_h5netcdf, requires_h5netcdf = _importorskip('h5netcdf')
has_pynio, requires_pynio = _importorskip('Nio')
has_dask, requires_dask = _importorskip('dask')
has_bottleneck, requires_bottleneck = _importorskip('bottleneck')
has_rasterio, requires_rasterio = _importorskip('rasterio')
has_pathlib, requires_pathlib = _importorskip('pathlib')

# some special cases
has_scipy_or_netCDF4 = has_scipy or has_netCDF4
requires_scipy_or_netCDF4 = unittest.skipUnless(
    has_scipy_or_netCDF4, reason='requires scipy or netCDF4')
if not has_pathlib:
    has_pathlib, requires_pathlib = _importorskip('pathlib2')
if has_dask:
    import dask
    dask.set_options(get=dask.get)
try:
    import_seaborn()
    has_seaborn = True
except:
    has_seaborn = False
requires_seaborn = unittest.skipUnless(has_seaborn, reason='requires seaborn')

try:
    _SKIP_FLAKY = not pytest.config.getoption("--run-flaky")
    _SKIP_NETWORK_TESTS = not pytest.config.getoption("--run-network-tests")
except (ValueError, AttributeError):
    # Can't get config from pytest, e.g., because xarray is installed instead
    # of being run from a development version (and hence conftests.py is not
    # available). Don't run flaky tests.
    _SKIP_FLAKY = True
    _SKIP_NETWORK_TESTS = True

flaky = pytest.mark.skipif(
    _SKIP_FLAKY, reason="set --run-flaky option to run flaky tests")
network = pytest.mark.skipif(
    _SKIP_NETWORK_TESTS,
    reason="set --run-network-tests option to run tests requiring an "
    "internet connection")


class TestCase(unittest.TestCase):
    if PY3:
        # Python 3 assertCountEqual is roughly equivalent to Python 2
        # assertItemsEqual
        def assertItemsEqual(self, first, second, msg=None):
            __tracebackhide__ = True  # noqa: F841
            return self.assertCountEqual(first, second, msg)

    @contextmanager
    def assertWarns(self, message):
        __tracebackhide__ = True  # noqa: F841
        with warnings.catch_warnings(record=True) as w:
            warnings.filterwarnings('always', message)
            yield
        assert len(w) > 0
        assert any(message in str(wi.message) for wi in w)

    def assertVariableEqual(self, v1, v2):
        __tracebackhide__ = True  # noqa: F841
        assert_equal(v1, v2)

    def assertVariableIdentical(self, v1, v2):
        __tracebackhide__ = True  # noqa: F841
        assert_identical(v1, v2)

    def assertVariableAllClose(self, v1, v2, rtol=1e-05, atol=1e-08):
        __tracebackhide__ = True  # noqa: F841
        assert_allclose(v1, v2, rtol=rtol, atol=atol)

    def assertVariableNotEqual(self, v1, v2):
        __tracebackhide__ = True  # noqa: F841
        assert not v1.equals(v2)

    def assertArrayEqual(self, a1, a2):
        __tracebackhide__ = True  # noqa: F841
        assert_array_equal(a1, a2)

    def assertEqual(self, a1, a2):
        __tracebackhide__ = True  # noqa: F841
        assert a1 == a2 or (a1 != a1 and a2 != a2)

    def assertAllClose(self, a1, a2, rtol=1e-05, atol=1e-8):
        __tracebackhide__ = True  # noqa: F841
        assert allclose_or_equiv(a1, a2, rtol=rtol, atol=atol)

    def assertDatasetEqual(self, d1, d2):
        __tracebackhide__ = True  # noqa: F841
        assert_equal(d1, d2)

    def assertDatasetIdentical(self, d1, d2):
        __tracebackhide__ = True  # noqa: F841
        assert_identical(d1, d2)

    def assertDatasetAllClose(self, d1, d2, rtol=1e-05, atol=1e-08,
                              decode_bytes=True):
        __tracebackhide__ = True  # noqa: F841
        assert_allclose(d1, d2, rtol=rtol, atol=atol,
                        decode_bytes=decode_bytes)

    def assertCoordinatesEqual(self, d1, d2):
        __tracebackhide__ = True  # noqa: F841
        assert_equal(d1, d2)

    def assertDataArrayEqual(self, ar1, ar2):
        __tracebackhide__ = True  # noqa: F841
        assert_equal(ar1, ar2)

    def assertDataArrayIdentical(self, ar1, ar2):
        __tracebackhide__ = True  # noqa: F841
        assert_identical(ar1, ar2)

    def assertDataArrayAllClose(self, ar1, ar2, rtol=1e-05, atol=1e-08,
                                decode_bytes=True):
        __tracebackhide__ = True  # noqa: F841
        assert_allclose(ar1, ar2, rtol=rtol, atol=atol,
                        decode_bytes=decode_bytes)


@contextmanager
def raises_regex(error, pattern):
    __tracebackhide__ = True  # noqa: F841
    with pytest.raises(error) as excinfo:
        yield
    message = str(excinfo.value)
    if not re.search(pattern, message):
        raise AssertionError('exception %r did not match pattern %r'
                             % (excinfo.value, pattern))


class UnexpectedDataAccess(Exception):
    pass


class InaccessibleArray(utils.NDArrayMixin, ExplicitlyIndexed):

    def __init__(self, array):
        self.array = array

    def __getitem__(self, key):
        raise UnexpectedDataAccess("Tried accessing data")


class ReturnItem(object):

    def __getitem__(self, key):
        return key


class IndexerMaker(object):

    def __init__(self, indexer_cls):
        self._indexer_cls = indexer_cls

    def __getitem__(self, key):
        if not isinstance(key, tuple):
            key = (key,)
        return self._indexer_cls(key)


def source_ndarray(array):
    """Given an ndarray, return the base object which holds its memory, or the
    object itself.
    """
    base = getattr(array, 'base', np.asarray(array).base)
    if base is None:
        base = array
    return base
