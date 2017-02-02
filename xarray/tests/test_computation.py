import functools
import operator
from collections import OrderedDict

import numpy as np
from numpy.testing import assert_array_equal

import pytest

import xarray as xr
from xarray.core.computation import (
    _UFuncSignature, broadcast_compat_data, collect_dict_values,
    join_dict_keys, ordered_set_intersection, ordered_set_union,
    unified_dim_sizes, apply_ufunc)

from . import requires_dask


def assert_identical(a, b):
    if hasattr(a, 'identical'):
        msg = 'not identical:\n%r\n%r' % (a, b)
        assert a.identical(b), msg
    else:
        assert_array_equal(a, b)


def test_signature_properties():
    sig = _UFuncSignature([['x'], ['x', 'y']], [['z']])
    assert sig.input_core_dims == (('x',), ('x', 'y'))
    assert sig.output_core_dims == (('z',),)
    assert sig.all_input_core_dims == frozenset(['x', 'y'])
    assert sig.all_output_core_dims == frozenset(['z'])
    assert sig.n_inputs == 2
    assert sig.n_outputs == 1
    # dimension names matter
    assert _UFuncSignature([['x']]) != _UFuncSignature([['y']])


def test_ordered_set_union():
    assert list(ordered_set_union([[1, 2]])) == [1, 2]
    assert list(ordered_set_union([[1, 2], [2, 1]])) == [1, 2]
    assert list(ordered_set_union([[0], [1, 2], [1, 3]])) == [0, 1, 2, 3]


def test_ordered_set_intersection():
    assert list(ordered_set_intersection([[1, 2]])) == [1, 2]
    assert list(ordered_set_intersection([[1, 2], [2, 1]])) == [1, 2]
    assert list(ordered_set_intersection([[1, 2], [1, 3]])) == [1]
    assert list(ordered_set_intersection([[1, 2], [2]])) == [2]


def test_join_dict_keys():
    dicts = [OrderedDict.fromkeys(keys) for keys in [['x', 'y'], ['y', 'z']]]
    assert list(join_dict_keys(dicts, 'left')) == ['x', 'y']
    assert list(join_dict_keys(dicts, 'right')) == ['y', 'z']
    assert list(join_dict_keys(dicts, 'inner')) == ['y']
    assert list(join_dict_keys(dicts, 'outer')) == ['x', 'y', 'z']
    with pytest.raises(KeyError):
        join_dict_keys(dicts, 'foobar')


def test_collect_dict_values():
    dicts = [{'x': 1, 'y': 2, 'z': 3}, {'z': 4}, 5]
    expected = [[1, 0, 5], [2, 0, 5], [3, 4, 5]]
    collected = collect_dict_values(dicts, ['x', 'y', 'z'], fill_value=0)
    assert collected == expected


def identity(x):
    return x


def test_apply_identity():
    array = np.arange(10)
    variable = xr.Variable('x', array)
    data_array = xr.DataArray(variable, [('x', -array)])
    dataset = xr.Dataset({'y': variable}, {'x': -array})

    apply_identity = functools.partial(apply_ufunc, identity)

    assert_identical(array, apply_identity(array))
    assert_identical(variable, apply_identity(variable))
    assert_identical(data_array, apply_identity(data_array))
    assert_identical(data_array, apply_identity(data_array.groupby('x')))
    assert_identical(dataset, apply_identity(dataset))
    assert_identical(dataset, apply_identity(dataset.groupby('x')))


def add(a, b):
    return apply_ufunc(operator.add, a, b)


def test_apply_two_inputs():
    array = np.array([1, 2, 3])
    variable = xr.Variable('x', array)
    data_array = xr.DataArray(variable, [('x', -array)])
    dataset = xr.Dataset({'y': variable}, {'x': -array})

    zero_array = np.zeros_like(array)
    zero_variable = xr.Variable('x', zero_array)
    zero_data_array = xr.DataArray(zero_variable, [('x', -array)])
    zero_dataset = xr.Dataset({'y': zero_variable}, {'x': -array})

    assert_identical(array, add(array, zero_array))
    assert_identical(array, add(zero_array, array))

    assert_identical(variable, add(variable, zero_array))
    assert_identical(variable, add(variable, zero_variable))
    assert_identical(variable, add(zero_array, variable))
    assert_identical(variable, add(zero_variable, variable))

    assert_identical(data_array, add(data_array, zero_array))
    assert_identical(data_array, add(data_array, zero_variable))
    assert_identical(data_array, add(data_array, zero_data_array))
    assert_identical(data_array, add(zero_array, data_array))
    assert_identical(data_array, add(zero_variable, data_array))
    assert_identical(data_array, add(zero_data_array, data_array))

    assert_identical(dataset, add(dataset, zero_array))
    assert_identical(dataset, add(dataset, zero_variable))
    assert_identical(dataset, add(dataset, zero_data_array))
    assert_identical(dataset, add(dataset, zero_dataset))
    assert_identical(dataset, add(zero_array, dataset))
    assert_identical(dataset, add(zero_variable, dataset))
    assert_identical(dataset, add(zero_data_array, dataset))
    assert_identical(dataset, add(zero_dataset, dataset))

    assert_identical(data_array, add(data_array.groupby('x'), zero_data_array))
    assert_identical(data_array, add(zero_data_array, data_array.groupby('x')))

    assert_identical(dataset, add(data_array.groupby('x'), zero_dataset))
    assert_identical(dataset, add(zero_dataset, data_array.groupby('x')))

    assert_identical(dataset, add(dataset.groupby('x'), zero_data_array))
    assert_identical(dataset, add(dataset.groupby('x'), zero_dataset))
    assert_identical(dataset, add(zero_data_array, dataset.groupby('x')))
    assert_identical(dataset, add(zero_dataset, dataset.groupby('x')))


def test_apply_1d_and_0d():
    array = np.array([1, 2, 3])
    variable = xr.Variable('x', array)
    data_array = xr.DataArray(variable, [('x', -array)])
    dataset = xr.Dataset({'y': variable}, {'x': -array})

    zero_array = 0
    zero_variable = xr.Variable((), zero_array)
    zero_data_array = xr.DataArray(zero_variable)
    zero_dataset = xr.Dataset({'y': zero_variable})

    assert_identical(array, add(array, zero_array))
    assert_identical(array, add(zero_array, array))

    assert_identical(variable, add(variable, zero_array))
    assert_identical(variable, add(variable, zero_variable))
    assert_identical(variable, add(zero_array, variable))
    assert_identical(variable, add(zero_variable, variable))

    assert_identical(data_array, add(data_array, zero_array))
    assert_identical(data_array, add(data_array, zero_variable))
    assert_identical(data_array, add(data_array, zero_data_array))
    assert_identical(data_array, add(zero_array, data_array))
    assert_identical(data_array, add(zero_variable, data_array))
    assert_identical(data_array, add(zero_data_array, data_array))

    assert_identical(dataset, add(dataset, zero_array))
    assert_identical(dataset, add(dataset, zero_variable))
    assert_identical(dataset, add(dataset, zero_data_array))
    assert_identical(dataset, add(dataset, zero_dataset))
    assert_identical(dataset, add(zero_array, dataset))
    assert_identical(dataset, add(zero_variable, dataset))
    assert_identical(dataset, add(zero_data_array, dataset))
    assert_identical(dataset, add(zero_dataset, dataset))

    assert_identical(data_array, add(data_array.groupby('x'), zero_data_array))
    assert_identical(data_array, add(zero_data_array, data_array.groupby('x')))

    assert_identical(dataset, add(data_array.groupby('x'), zero_dataset))
    assert_identical(dataset, add(zero_dataset, data_array.groupby('x')))

    assert_identical(dataset, add(dataset.groupby('x'), zero_data_array))
    assert_identical(dataset, add(dataset.groupby('x'), zero_dataset))
    assert_identical(dataset, add(zero_data_array, dataset.groupby('x')))
    assert_identical(dataset, add(zero_dataset, dataset.groupby('x')))


def test_apply_two_outputs():
    array = np.arange(5)
    variable = xr.Variable('x', array)
    data_array = xr.DataArray(variable, [('x', -array)])
    dataset = xr.Dataset({'y': variable}, {'x': -array})

    def twice(obj):
        def func(x):
            return (x, x)
        return apply_ufunc(func, obj, output_core_dims=[[], []])

    out0, out1 = twice(array)
    assert_identical(out0, array)
    assert_identical(out1, array)

    out0, out1 = twice(variable)
    assert_identical(out0, variable)
    assert_identical(out1, variable)

    out0, out1 = twice(data_array)
    assert_identical(out0, data_array)
    assert_identical(out1, data_array)

    out0, out1 = twice(dataset)
    assert_identical(out0, dataset)
    assert_identical(out1, dataset)

    out0, out1 = twice(data_array.groupby('x'))
    assert_identical(out0, data_array)
    assert_identical(out1, data_array)

    out0, out1 = twice(dataset.groupby('x'))
    assert_identical(out0, dataset)
    assert_identical(out1, dataset)


def test_apply_input_core_dimension():

    def first_element(obj, dim):
        def func(x):
            return x[..., 0]
        return apply_ufunc(func, obj, input_core_dims=[[dim]])

    array = np.array([[1, 2], [3, 4]])
    variable = xr.Variable(['x', 'y'], array)
    data_array = xr.DataArray(variable, {'x': ['a', 'b'], 'y': [-1, -2]})
    dataset = xr.Dataset({'data': data_array})

    expected_variable_x = xr.Variable(['y'], [1, 2])
    expected_data_array_x = xr.DataArray(expected_variable_x, {'y': [-1, -2]})
    expected_dataset_x = xr.Dataset({'data': expected_data_array_x})

    expected_variable_y = xr.Variable(['x'], [1, 3])
    expected_data_array_y = xr.DataArray(expected_variable_y,
                                         {'x': ['a', 'b']})
    expected_dataset_y = xr.Dataset({'data': expected_data_array_y})

    assert_identical(expected_variable_x, first_element(variable, 'x'))
    assert_identical(expected_variable_y, first_element(variable, 'y'))

    assert_identical(expected_data_array_x, first_element(data_array, 'x'))
    assert_identical(expected_data_array_y, first_element(data_array, 'y'))

    assert_identical(expected_dataset_x, first_element(dataset, 'x'))
    assert_identical(expected_dataset_y, first_element(dataset, 'y'))

    assert_identical(expected_data_array_x,
                     first_element(data_array.groupby('y'), 'x'))
    assert_identical(expected_dataset_x,
                     first_element(dataset.groupby('y'), 'x'))


def test_apply_output_core_dimension():

    def stack_negative(obj):
        def func(x):
            return xr.core.npcompat.stack([x, -x], axis=-1)
        result = apply_ufunc(func, obj, output_core_dims=[['sign']])
        if isinstance(result, (xr.Dataset, xr.DataArray)):
            result.coords['sign'] = [1, -1]
        return result

    array = np.array([[1, 2], [3, 4]])
    variable = xr.Variable(['x', 'y'], array)
    data_array = xr.DataArray(variable, {'x': ['a', 'b'], 'y': [-1, -2]})
    dataset = xr.Dataset({'data': data_array})

    stacked_array = np.array([[[1, -1], [2, -2]], [[3, -3], [4, -4]]])
    stacked_variable = xr.Variable(['x', 'y', 'sign'], stacked_array)
    stacked_coords = {'x': ['a', 'b'], 'y': [-1, -2], 'sign': [1, -1]}
    stacked_data_array = xr.DataArray(stacked_variable, stacked_coords)
    stacked_dataset = xr.Dataset({'data': stacked_data_array})

    assert_identical(stacked_array, stack_negative(array))
    assert_identical(stacked_variable, stack_negative(variable))
    assert_identical(stacked_data_array, stack_negative(data_array))
    assert_identical(stacked_dataset, stack_negative(dataset))
    assert_identical(stacked_data_array,
                     stack_negative(data_array.groupby('x')))
    assert_identical(stacked_dataset,
                     stack_negative(dataset.groupby('x')))

    def original_and_stack_negative(obj):
        def func(x):
            return (x, xr.core.npcompat.stack([x, -x], axis=-1))
        result = apply_ufunc(func, obj, output_core_dims=[[], ['sign']])
        if isinstance(result[1], (xr.Dataset, xr.DataArray)):
            result[1].coords['sign'] = [1, -1]
        return result

    out0, out1 = original_and_stack_negative(array)
    assert_identical(array, out0)
    assert_identical(stacked_array, out1)

    out0, out1 = original_and_stack_negative(variable)
    assert_identical(variable, out0)
    assert_identical(stacked_variable, out1)

    out0, out1 = original_and_stack_negative(data_array)
    assert_identical(data_array, out0)
    assert_identical(stacked_data_array, out1)

    out0, out1 = original_and_stack_negative(dataset)
    assert_identical(dataset, out0)
    assert_identical(stacked_dataset, out1)

    out0, out1 = original_and_stack_negative(data_array.groupby('x'))
    assert_identical(data_array, out0)
    assert_identical(stacked_data_array, out1)

    out0, out1 = original_and_stack_negative(dataset.groupby('x'))
    assert_identical(dataset, out0)
    assert_identical(stacked_dataset, out1)


def test_apply_exclude():

    def concatenate(objects, dim='x'):
        def func(*x):
            return np.concatenate(x, axis=-1)
        result = apply_ufunc(func, *objects,
                             input_core_dims=[[dim]] * len(objects),
                             output_core_dims=[[dim]],
                             exclude_dims={dim})
        if isinstance(result, (xr.Dataset, xr.DataArray)):
            # note: this will fail dim is not a coordinate on any input
            new_coord = np.concatenate([obj.coords[dim] for obj in objects])
            result.coords[dim] = new_coord
        return result

    arrays = [np.array([1]), np.array([2, 3])]
    variables = [xr.Variable('x', a) for a in arrays]
    data_arrays = [xr.DataArray(v, {'x': c, 'y': ('x', range(len(c)))})
                   for v, c in zip(variables, [['a'], ['b', 'c']])]
    datasets = [xr.Dataset({'data': data_array}) for data_array in data_arrays]

    expected_array = np.array([1, 2, 3])
    expected_variable = xr.Variable('x', expected_array)
    expected_data_array = xr.DataArray(expected_variable, [('x', list('abc'))])
    expected_dataset = xr.Dataset({'data': expected_data_array})

    assert_identical(expected_array, concatenate(arrays))
    assert_identical(expected_variable, concatenate(variables))
    assert_identical(expected_data_array, concatenate(data_arrays))
    assert_identical(expected_dataset, concatenate(datasets))

    # must also be a core dimension
    with pytest.raises(ValueError):
        apply_ufunc(identity, variables[0], exclude_dims={'x'})


def test_apply_groupby_add():
    array = np.arange(5)
    variable = xr.Variable('x', array)
    coords = {'x': -array, 'y': ('x', [0, 0, 1, 1, 2])}
    data_array = xr.DataArray(variable, coords, dims='x')
    dataset = xr.Dataset({'z': variable}, coords)

    other_variable = xr.Variable('y', [0, 10])
    other_data_array = xr.DataArray(other_variable, dims='y')
    other_dataset = xr.Dataset({'z': other_variable})

    expected_variable = xr.Variable('x', [0, 1, 12, 13, np.nan])
    expected_data_array = xr.DataArray(expected_variable, coords, dims='x')
    expected_dataset = xr.Dataset({'z': expected_variable}, coords)

    assert_identical(expected_data_array,
                     add(data_array.groupby('y'), other_data_array))
    assert_identical(expected_dataset,
                     add(data_array.groupby('y'), other_dataset))
    assert_identical(expected_dataset,
                     add(dataset.groupby('y'), other_data_array))
    assert_identical(expected_dataset,
                     add(dataset.groupby('y'), other_dataset))

    # cannot be performed with xarray.Variable objects that share a dimension
    with pytest.raises(ValueError):
        add(data_array.groupby('y'), other_variable)

    # if they are all grouped the same way
    with pytest.raises(ValueError):
        add(data_array.groupby('y'), data_array[:4].groupby('y'))
    with pytest.raises(ValueError):
        add(data_array.groupby('y'), data_array[1:].groupby('y'))
    with pytest.raises(ValueError):
        add(data_array.groupby('y'), other_data_array.groupby('y'))
    with pytest.raises(ValueError):
        add(data_array.groupby('y'), data_array.groupby('x'))


def test_unified_dim_sizes():
    assert unified_dim_sizes([xr.Variable((), 0)]) == OrderedDict()
    assert (unified_dim_sizes([xr.Variable('x', [1]),
                               xr.Variable('x', [1])]) ==
            OrderedDict([('x', 1)]))
    assert (unified_dim_sizes([xr.Variable('x', [1]),
                               xr.Variable('y', [1, 2])]) ==
            OrderedDict([('x', 1), ('y', 2)]))
    assert (unified_dim_sizes([xr.Variable(('x', 'z'), [[1]]),
                               xr.Variable(('y', 'z'), [[1, 2], [3, 4]])],
                              exclude_dims={'z'}) ==
            OrderedDict([('x', 1), ('y', 2)]))

    # duplicate dimensions
    with pytest.raises(ValueError):
        unified_dim_sizes([xr.Variable(('x', 'x'), [[1]])])

    # mismatched lengths
    with pytest.raises(ValueError):
        unified_dim_sizes(
            [xr.Variable('x', [1]), xr.Variable('x', [1, 2])])


def test_broadcast_compat_data_1d():
    data = np.arange(5)
    var = xr.Variable('x', data)

    assert_identical(data, broadcast_compat_data(var, ('x',), ()))
    assert_identical(data, broadcast_compat_data(var, (), ('x',)))
    assert_identical(data[:], broadcast_compat_data(var, ('w',), ('x',)))
    assert_identical(data[:, None],
                     broadcast_compat_data(var, ('w', 'x', 'y'), ()))

    with pytest.raises(ValueError):
        broadcast_compat_data(var, ('x',), ('w',))

    with pytest.raises(ValueError):
        broadcast_compat_data(var, (), ())


def test_broadcast_compat_data_2d():
    data = np.arange(12).reshape(3, 4)
    var = xr.Variable(['x', 'y'], data)

    assert_identical(data, broadcast_compat_data(var, ('x', 'y'), ()))
    assert_identical(data, broadcast_compat_data(var, ('x',), ('y',)))
    assert_identical(data, broadcast_compat_data(var, (), ('x', 'y')))
    assert_identical(data.T, broadcast_compat_data(var, ('y', 'x'), ()))
    assert_identical(data.T, broadcast_compat_data(var, ('y',), ('x',)))
    assert_identical(data, broadcast_compat_data(var, ('w', 'x'), ('y',)))
    assert_identical(data, broadcast_compat_data(var, ('w',), ('x', 'y')))
    assert_identical(data.T, broadcast_compat_data(var, ('w',), ('y', 'x')))
    assert_identical(data[:, :, None],
                     broadcast_compat_data(var, ('w', 'x', 'y', 'z'), ()))
    assert_identical(data[None, :, :].T,
                     broadcast_compat_data(var, ('w', 'y', 'x', 'z'), ()))


def test_keep_attrs():

    def add(a, b, keep_attrs):
        if keep_attrs:
            return apply_ufunc(operator.add, a, b, keep_attrs=keep_attrs)
        else:
            return apply_ufunc(operator.add, a, b)

    a = xr.DataArray([0, 1], [('x', [0, 1])])
    a.attrs['attr'] = 'da'
    b = xr.DataArray([1, 2], [('x', [0, 1])])

    actual = add(a, b, keep_attrs=False)
    assert not actual.attrs
    actual = add(a, b, keep_attrs=True)
    assert_identical(actual.attrs, a.attrs)

    a = xr.Dataset({'x': ('x', [1, 2]), 'x': [0, 1]})
    a.attrs['attr'] = 'ds'
    a.x.attrs['attr'] = 'da'
    b = xr.Dataset({'x': ('x', [1, 1]), 'x': [0, 1]})

    actual = add(a, b, keep_attrs=False)
    assert not actual.attrs
    actual = add(a, b, keep_attrs=True)
    assert_identical(actual.attrs, a.attrs)
    assert_identical(actual.x.attrs, a.x.attrs)


def test_dataset_join():
    import numpy as np
    ds0 = xr.Dataset({'a': ('x', [1, 2]), 'x': [0, 1]})
    ds1 = xr.Dataset({'a': ('x', [99, 3]), 'x': [1, 2]})

    with pytest.raises(TypeError):
        apply_ufunc(operator.add, ds0, ds1, dataset_join='outer')

    def add(a, b, join, dataset_join):
        return apply_ufunc(operator.add, a, b, join=join,
                           dataset_join=dataset_join,
                           dataset_fill_value=np.nan)

    actual = add(ds0, ds1, 'outer', 'inner')
    expected = xr.Dataset({'a': ('x', [np.nan, 101, np.nan]),
                           'x': [0, 1, 2]})
    assert_identical(actual, expected)

    actual = add(ds0, ds1, 'outer', 'outer')
    assert_identical(actual, expected)

    # if variables don't match, join will perform add with np.nan
    ds2 = xr.Dataset({'b': ('x', [99, 3]), 'x': [1, 2]})
    actual = add(ds0, ds2, 'outer', 'inner')
    expected = xr.Dataset({'x': [0, 1, 2]})
    assert_identical(actual, expected)

    actual = add(ds0, ds2, 'outer', 'outer')
    expected = xr.Dataset({'a': ('x', [np.nan, np.nan, np.nan]),
                           'b': ('x', [np.nan, np.nan, np.nan]),
                           'x': [0, 1, 2]})
    assert_identical(actual, expected)


class _NoCacheVariable(xr.Variable):
    """Subclass of Variable for testing that does not cache values."""
    # TODO: remove this class when we change the default behavior for caching
    # dask.array objects.
    def _data_cached(self):
        return np.asarray(self._data)


@requires_dask
def test_apply_dask():
    import dask.array as da

    array = da.ones((2,), chunks=2)
    variable = _NoCacheVariable('x', array)
    coords = xr.DataArray(variable).coords.variables
    data_array = xr.DataArray(variable, coords, fastpath=True)
    dataset = xr.Dataset({'y': variable})

    # encountered dask array, but did not set dask_array='allowed'
    with pytest.raises(ValueError):
        apply_ufunc(identity, array)
    with pytest.raises(ValueError):
        apply_ufunc(identity, variable)
    with pytest.raises(ValueError):
        apply_ufunc(identity, data_array)
    with pytest.raises(ValueError):
        apply_ufunc(identity, dataset)

    # unknown setting for dask array handling
    with pytest.raises(ValueError):
        apply_ufunc(identity, array, dask_array='auto')

    def dask_safe_identity(x):
        return apply_ufunc(identity, x, dask_array='allowed')

    assert array is dask_safe_identity(array)

    actual = dask_safe_identity(variable)
    assert isinstance(actual.data, da.Array)
    assert_identical(variable, actual)

    actual = dask_safe_identity(data_array)
    assert isinstance(actual.data, da.Array)
    assert_identical(data_array, actual)

    actual = dask_safe_identity(dataset)
    assert isinstance(actual['y'].data, da.Array)
    assert_identical(dataset, actual)
