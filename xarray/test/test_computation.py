from collections import OrderedDict
import operator

import numpy as np
import pytest
import xarray as xr

from numpy.testing import assert_array_equal
from xarray.core.computation import (
    join_dict_keys, collect_dict_values, broadcast_compat_data, _Signature)


def assert_identical(a, b):
    msg = 'not identical:\n%r\n%r' % (a, b)
    assert a.identical(b), msg


def test_parse_signature():
    assert _Signature([['x']]) == _Signature.from_string('(x)->()')
    assert _Signature([['x', 'y']]) == _Signature.from_string('(x,y)->()')
    assert _Signature([['x'], ['y']]) == _Signature.from_string('(x),(y)->()')
    assert (_Signature([['x']], [['y'], []]) ==
            _Signature.from_string('(x)->(y),()'))
    with pytest.raises(ValueError):
        _Signature.from_string('(x)(y)->()')
    with pytest.raises(ValueError):
        _Signature.from_string('(x),(y)->')
    with pytest.raises(ValueError):
        _Signature.from_string('((x))->(x)')


def test_signature_properties():
    sig = _Signature.from_string('(x),(x,y)->(z)')
    assert sig.input_core_dims == (('x',), ('x', 'y'))
    assert sig.output_core_dims == (('z',),)
    assert sig.all_input_core_dims == frozenset(['x', 'y'])
    assert sig.all_output_core_dims == frozenset(['z'])
    assert sig.n_inputs == 2
    assert sig.n_outputs == 1


def test_join_dict_keys():
    dicts = [OrderedDict.fromkeys(keys) for keys in [['x', 'y'], ['y', 'z']]]
    assert_array_equal(join_dict_keys(dicts, 'left'), ['x', 'y'])
    assert_array_equal(join_dict_keys(dicts, 'right'), ['y', 'z'])
    assert_array_equal(join_dict_keys(dicts, 'inner'), ['y'])
    assert_array_equal(join_dict_keys(dicts, 'outer'), ['x', 'y', 'z'])


def test_collect_dict_values():
    dicts = [{'x': 1, 'y': 2, 'z': 3}, {'z': 4}, 5]
    expected = [[1, 0, 5], [2, 0, 5], [3, 4, 5]]
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


def test_apply_ufunc_two_inputs():
    array = np.array([1, 2, 3])
    variable = xr.Variable('x', array)
    data_array = xr.DataArray(variable, [('x', -array)])
    dataset = xr.Dataset({'y': variable}, {'x': -array})

    zeros_array = np.zeros_like(array)
    zeros_variable = xr.Variable('x', zeros_array)
    zeros_data_array = xr.DataArray(zeros_variable, [('x', -array)])
    zeros_dataset = xr.Dataset({'y': zeros_variable}, {'x': -array})

    add = lambda a, b: xr.apply_ufunc(operator.add, a, b)

    assert_array_equal(array, add(array, 0))
    assert_array_equal(array, add(array, zeros_array))
    assert_array_equal(array, add(0, array))
    assert_array_equal(array, add(zeros_array, array))

    assert_identical(variable, add(variable, 0))
    assert_identical(variable, add(variable, zeros_array))
    assert_identical(variable, add(variable, zeros_variable))
    assert_identical(variable, add(0, variable))
    assert_identical(variable, add(zeros_array, variable))
    assert_identical(variable, add(zeros_variable, variable))

    assert_identical(data_array, add(data_array, 0))
    assert_identical(data_array, add(data_array, zeros_array))
    assert_identical(data_array, add(data_array, zeros_variable))
    assert_identical(data_array, add(data_array, zeros_data_array))
    assert_identical(data_array, add(0, data_array))
    assert_identical(data_array, add(zeros_array, data_array))
    assert_identical(data_array, add(zeros_variable, data_array))
    assert_identical(data_array, add(zeros_data_array, data_array))

    assert_identical(dataset, add(dataset, 0))
    assert_identical(dataset, add(dataset, zeros_array))
    assert_identical(dataset, add(dataset, zeros_variable))
    assert_identical(dataset, add(dataset, zeros_data_array))
    assert_identical(dataset, add(dataset, zeros_dataset))
    assert_identical(dataset, add(0, dataset))
    assert_identical(dataset, add(zeros_array, dataset))
    assert_identical(dataset, add(zeros_variable, dataset))
    assert_identical(dataset, add(zeros_data_array, dataset))
    assert_identical(dataset, add(zeros_dataset, dataset))


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


def test_apply_ufunc_input_core_dimension():

    def first_element(obj, dim):
        func = lambda x: x[..., 0]
        sig = ([(dim,)], [(),])
        return xr.apply_ufunc(func, obj, signature=sig)

    array = np.array([[1, 2], [3, 4]])
    variable = xr.Variable(['x', 'y'], array)
    data_array = xr.DataArray(variable, {'x': ['a', 'b'], 'y': [-1, -2]})
    dataset = xr.Dataset({'data': data_array})

    expected_variable_x = xr.Variable(['y'], [1, 2])
    expected_data_array_x = xr.DataArray(expected_variable_x, {'y': [-1, -2]})
    expected_dataset_x = xr.Dataset({'data': expected_data_array_x})

    expected_variable_y = xr.Variable(['x'], [1, 3])
    expected_data_array_y = xr.DataArray(expected_variable_y, {'x': ['a', 'b']})
    expected_dataset_y = xr.Dataset({'data': expected_data_array_y})

    actual = first_element(variable, 'x')
    assert_identical(actual, expected_variable_x)
    actual = first_element(variable, 'y')
    assert_identical(actual, expected_variable_y)

    actual = first_element(data_array, 'x')
    assert_identical(actual, expected_data_array_x)
    actual = first_element(data_array, 'y')
    assert_identical(actual, expected_data_array_y)

    actual = first_element(dataset, 'x')
    assert_identical(actual, expected_dataset_x)
    actual = first_element(dataset, 'y')
    assert_identical(actual, expected_dataset_y)


def test_apply_ufunc_output_core_dimension():

    def stack_negative(obj):
        func = lambda x: xr.core.npcompat.stack([x, -x], axis=-1)
        sig = ([()], [('sign',)])
        new_coords = {'sign': [1, -1]}
        return xr.apply_ufunc(func, obj, signature=sig, new_coords=new_coords)

    array = np.array([[1, 2], [3, 4]])
    variable = xr.Variable(['x', 'y'], array)
    data_array = xr.DataArray(variable, {'x': ['a', 'b'], 'y': [-1, -2]})
    dataset = xr.Dataset({'data': data_array})

    stacked_array = np.array([[[1, -1], [2, -2]], [[3, -3], [4, -4]]])
    expected_variable = xr.Variable(['x', 'y', 'sign'], stacked_array)
    expected_coords = {'x': ['a', 'b'], 'y': [-1, -2], 'sign': [1, -1]}
    expected_data_array = xr.DataArray(expected_variable, expected_coords)
    expected_dataset = xr.Dataset({'data': expected_data_array})

    actual = stack_negative(variable)
    assert_identical(actual, expected_variable)

    actual = stack_negative(data_array)
    assert_identical(actual, expected_data_array)

    actual = stack_negative(dataset)
    assert_identical(actual, expected_dataset)

    def stack2(obj):
        func = lambda x: xr.core.npcompat.stack([x, -x], axis=-1)
        sig = ([()], [('sign',)])
        # no new_coords
        return xr.apply_ufunc(func, obj, signature=sig)

    actual = stack2(data_array)
    expected_data_array.coords['sign'] = [0, 1]
    assert_identical(actual, expected_data_array)

    actual = stack2(dataset)
    expected_dataset.coords['sign'] = [0, 1]
    assert_identical(actual, expected_dataset)


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

    with pytest.raises(ValueError):
         broadcast_compat_data(var, (), ())


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
