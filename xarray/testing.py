"""Testing functions exposed to the user API"""
from __future__ import absolute_import, division, print_function

import numpy as np
from pandas import date_range

from xarray.core import duck_array_ops
from xarray.core import formatting
from xarray import Dataset


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
    if isinstance(a, (xr.Variable, xr.DataArray)):
        assert a.equals(b), formatting.diff_array_repr(a, b, 'equals')
    elif isinstance(a, xr.Dataset):
        assert a.equals(b), formatting.diff_dataset_repr(a, b, 'equals')
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
    if isinstance(a, xr.Variable):
        assert a.identical(b), formatting.diff_array_repr(a, b, 'identical')
    elif isinstance(a, xr.DataArray):
        assert a.name == b.name
        assert a.identical(b), formatting.diff_array_repr(a, b, 'identical')
    elif isinstance(a, (xr.Dataset, xr.Variable)):
        assert a.identical(b), formatting.diff_dataset_repr(a, b, 'identical')
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


def create_test_data(seed=None):
    """
    Creates an example dataset for use when testing functions which act on
    xarray objects.

    The dataset returned covers several possible edge cases, including
    dimensions with and without coordinates, and datetime, integer and string
    coordinate values.

    Used extensively within xarray's own test suite.

    Parameters
    ----------
    seed : int, optional
        Seed to use for random data (passed to `np.random.RandomState`),
        default is `None`.

    Returns
    -------
    dataset

    Examples
    --------
    >>> xr.create_test_data(seed=0)
    <xarray.Dataset>
    Dimensions:  (dim1: 8, dim2: 9, dim3: 10, time: 20)
    Coordinates:
      * time     (time) datetime64[ns] 2000-01-01 2000-01-02 ... 2000-01-20
      * dim2     (dim2) float64 0.0 0.5 1.0 1.5 2.0 2.5 3.0 3.5 4.0
      * dim3     (dim3) <U1 'a' 'b' 'c' 'd' 'e' 'f' 'g' 'h' 'i' 'j'
        numbers  (dim3) int64 0 1 2 0 0 1 1 2 2 3
    Dimensions without coordinates: dim1
    Data variables:
        var1     (dim1, dim2) float64 1.764 0.4002 0.9787 ...
        var2     (dim1, dim2) float64 1.139 -1.235 0.4023 ...
        var3     (dim3, dim1) float64 2.383 0.9445 -0.9128 ...
    """
    rs = np.random.RandomState(seed)
    _vars = {'var1': ['dim1', 'dim2'],
             'var2': ['dim1', 'dim2'],
             'var3': ['dim3', 'dim1']}
    _dims = {'dim1': 8, 'dim2': 9, 'dim3': 10}

    obj = Dataset()
    obj['time'] = ('time', date_range('2000-01-01', periods=20))
    obj['dim2'] = ('dim2', 0.5 * np.arange(_dims['dim2']))
    obj['dim3'] = ('dim3', list('abcdefghij'))
    for v, dims in sorted(_vars.items()):
        data = rs.normal(size=tuple(_dims[d] for d in dims))
        obj[v] = (dims, data, {'foo': 'variable'})
    obj.coords['numbers'] = ('dim3', np.array([0, 1, 2, 0, 0, 1, 1, 2, 2, 3],
                                              dtype='int64'))
    obj.encoding = {'foo': 'bar'}
    assert all(obj.data.flags.writeable for obj in obj.variables.values())
    return obj


def assert_combined_tile_ids_equal(dict1, dict2):
    assert len(dict1) == len(dict2)
    for k, v in dict1.items():
        assert k in dict2.keys()
        assert_equal(dict1[k], dict2[k])
