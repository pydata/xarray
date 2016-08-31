from collections import OrderedDict

import numpy as np
import pytest
import xarray as xr

from numpy.testing import assert_array_equal
from xarray.core.computation import (
    join_dict_keys, collect_dict_values, broadcast_compat_data, Signature)


def assert_identical(a, b):
    msg = 'not identical:\n%r\n%r' % (a, b)
    assert a.identical(b), msg


def test_parse_signature():
    assert Signature([['x']]) == Signature.parse('(x)->()')
    assert Signature([['x', 'y']]) == Signature.parse('(x,y)->()')
    assert Signature([['x'], ['y']]) == Signature.parse('(x),(y)->()')
    assert Signature([['x']], [['y']]) == Signature.parse('(x)->(y)')
    with pytest.raises(ValueError):
        Signature.parse('(x)(y)->()')
    with pytest.raises(ValueError):
        Signature.parse('(x),(y)->')
    with pytest.raises(ValueError):
        Signature.parse('((x))->(x)')


def test_join_dict_keys():
    dicts = [OrderedDict.fromkeys(keys) for keys in [['x', 'y'], ['y', 'z']]]
    assert_array_equal(join_dict_keys(dicts, 'left'), ['x', 'y'])
    assert_array_equal(join_dict_keys(dicts, 'right'), ['y', 'z'])
    assert_array_equal(join_dict_keys(dicts, 'inner'), ['y'])
    assert_array_equal(join_dict_keys(dicts, 'outer'), ['x', 'y', 'z'])


def test_collect_dict_values():
    dicts = [{'x': 1, 'y': 2, 'z': 3}, {'z': 4}]
    expected = [[1, 0], [2, 0], [3, 4]]
    collected = collect_dict_values(dicts, ['x', 'y', 'z'], fill_value=0)
    assert collected == expected


def test_apply_ufunc_identity():
    array = np.arange(10)
    variable = xr.Variable('x', array)
    data_array = xr.DataArray(variable, [('x', -array)])
    dataset = xr.Dataset({'y': variable}, {'x': -array})

    identity = lambda x: x

    output = xr.apply_ufunc(identity, array)
    assert_array_equal(output, array)

    output = xr.apply_ufunc(identity, variable)
    assert_identical(output, variable)

    output = xr.apply_ufunc(identity, data_array)
    assert_identical(output, data_array)

    output = xr.apply_ufunc(identity, dataset)
    assert_identical(output, dataset)


def test_apply_ufunc_two_outputs():
    array = np.arange(10)
    variable = xr.Variable('x', array)
    data_array = xr.DataArray(variable, [('x', -array)])
    dataset = xr.Dataset({'y': variable}, {'x': -array})

    func = lambda x: (x, 2 * x)
    signature = '()->(),()'

    out0, out1 = xr.apply_ufunc(func, array, signature=signature)
    assert_array_equal(out0, array)
    assert_array_equal(out1, 2 * array)

    out0, out1 = xr.apply_ufunc(func, variable, signature=signature)
    assert_identical(out0, variable)
    assert_identical(out1, 2 * variable)

    out0, out1 = xr.apply_ufunc(func, data_array, signature=signature)
    assert_identical(out0, data_array)
    assert_identical(out1, 2 * data_array)

    out0, out1 = xr.apply_ufunc(func, dataset, signature=signature)
    assert_identical(out0, dataset)
    assert_identical(out1, 2 * dataset)


def test_broadcast_compat_data_1d():
    data = np.arange(5)
    var = xr.Variable('x', data)

    actual = broadcast_compat_data(var, ('x',), ())
    assert_array_equal(actual, data)

    actual = broadcast_compat_data(var, (), ('x',))
    assert_array_equal(actual, data)

    actual = broadcast_compat_data(var, ('w',), ('x',))
    assert_array_equal(actual, data[None, :])

    actual = broadcast_compat_data(var, ('w', 'x', 'y'), ())
    assert_array_equal(actual, data[None, :, None])

    with pytest.raises(ValueError):
         broadcast_compat_data(var, ('x',), ('w',))


def test_broadcast_compat_data_2d():
    data = np.arange(12).reshape(3, 4)
    var = xr.Variable(['x', 'y'], data)

    actual = broadcast_compat_data(var, ('x', 'y'), ())
    assert_array_equal(actual, data)

    actual = broadcast_compat_data(var, ('x',), ('y',))
    assert_array_equal(actual, data)

    actual = broadcast_compat_data(var, (), ('x', 'y'))
    assert_array_equal(actual, data)

    actual = broadcast_compat_data(var, ('y', 'x'), ())
    assert_array_equal(actual, data.T)

    actual = broadcast_compat_data(var, ('y',), ('x',))
    assert_array_equal(actual, data.T)

    actual = broadcast_compat_data(var, ('w', 'x'), ('y',))
    assert_array_equal(actual, data[None, :, :])

    actual = broadcast_compat_data(var, ('w',), ('x', 'y'))
    assert_array_equal(actual, data[None, :, :])

    actual = broadcast_compat_data(var, ('w',), ('y', 'x'))
    assert_array_equal(actual, data.T[None, :, :])

    actual = broadcast_compat_data(var, ('w', 'x', 'y', 'z'), ())
    assert_array_equal(actual, data[None, :, :, None])

    actual = broadcast_compat_data(var, ('w', 'y', 'x', 'z'), ())
    assert_array_equal(actual, data.T[None, :, :, None])
