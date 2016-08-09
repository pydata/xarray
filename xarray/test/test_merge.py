import numpy as np
import xarray as xr

from . import TestCase
from .test_dataset import create_test_data

from xarray.core import merge


class TestMergeInternals(TestCase):
    def test_broadcast_dimension_size(self):
        actual = merge.broadcast_dimension_size(
            [xr.Variable('x', [1]), xr.Variable('y', [2, 1])])
        assert actual == {'x': 1, 'y': 2}

        actual = merge.broadcast_dimension_size(
            [xr.Variable(('x', 'y'), [[1, 2]]), xr.Variable('y', [2, 1])])
        assert actual == {'x': 1, 'y': 2}

        with self.assertRaises(ValueError):
            actual = merge.broadcast_dimension_size(
                [xr.Variable(('x', 'y'), [[1, 2]]), xr.Variable('y', [2])])


class TestMergeFunction(TestCase):
    def test_merge_arrays(self):
        data = create_test_data()
        actual = xr.merge([data.var1, data.var2])
        expected = data[['var1', 'var2']]
        assert actual.identical(expected)

    def test_merge_datasets(self):
        data = create_test_data()

        actual = xr.merge([data[['var1']], data[['var2']]])
        expected = data[['var1', 'var2']]
        assert actual.identical(expected)

        actual = xr.merge([data, data])
        assert actual.identical(data)

    def test_merge_dataarray_unnamed(self):
        data = xr.DataArray([1, 2], dims='x')
        with self.assertRaisesRegexp(
                ValueError, 'without providing an explicit name'):
            xr.merge([data])

    def test_merge_dicts_simple(self):
        actual = xr.merge([{'foo': 0}, {'bar': 'one'}, {'baz': 3.5}])
        expected = xr.Dataset({'foo': 0, 'bar': 'one', 'baz': 3.5})
        assert actual.identical(expected)

    def test_merge_dicts_dims(self):
        actual = xr.merge([{'y': ('x', [13])}, {'x': [12]}])
        expected = xr.Dataset({'x': [12], 'y': ('x', [13])})
        assert actual.identical(expected)

    def test_merge_error(self):
        ds = xr.Dataset({'x': 0})
        with self.assertRaises(xr.MergeError):
            xr.merge([ds, ds + 1])


class TestMergeMethod(TestCase):

    def test_merge(self):
        data = create_test_data()
        ds1 = data[['var1']]
        ds2 = data[['var3']]
        expected = data[['var1', 'var3']]
        actual = ds1.merge(ds2)
        assert expected.identical(actual)

        actual = ds2.merge(ds1)
        assert expected.identical(actual)

        actual = data.merge(data)
        assert data.identical(actual)
        actual = data.reset_coords(drop=True).merge(data)
        assert data.identical(actual)
        actual = data.merge(data.reset_coords(drop=True))
        assert data.identical(actual)

        with self.assertRaises(ValueError):
            ds1.merge(ds2.rename({'var3': 'var1'}))
        with self.assertRaisesRegexp(
                ValueError, 'should be coordinates or not'):
            data.reset_coords().merge(data)
        with self.assertRaisesRegexp(
                ValueError, 'should be coordinates or not'):
            data.merge(data.reset_coords())

    def test_merge_broadcast_equals(self):
        ds1 = xr.Dataset({'x': 0})
        ds2 = xr.Dataset({'x': ('y', [0, 0])})
        actual = ds1.merge(ds2)
        assert ds2.identical(actual)

        actual = ds2.merge(ds1)
        assert ds2.identical(actual)

        actual = ds1.copy()
        actual.update(ds2)
        assert ds2.identical(actual)

        ds1 = xr.Dataset({'x': np.nan})
        ds2 = xr.Dataset({'x': ('y', [np.nan, np.nan])})
        actual = ds1.merge(ds2)
        assert ds2.identical(actual)

    def test_merge_compat(self):
        ds1 = xr.Dataset({'x': 0})
        ds2 = xr.Dataset({'x': 1})
        for compat in ['broadcast_equals', 'equals', 'identical']:
            with self.assertRaises(xr.MergeError):
                ds1.merge(ds2, compat=compat)

        ds2 = xr.Dataset({'x': [0, 0]})
        for compat in ['equals', 'identical']:
            with self.assertRaisesRegexp(
                    ValueError, 'should be coordinates or not'):
                ds1.merge(ds2, compat=compat)

        ds2 = xr.Dataset({'x': ((), 0, {'foo': 'bar'})})
        with self.assertRaises(xr.MergeError):
            ds1.merge(ds2, compat='identical')

        with self.assertRaisesRegexp(ValueError, 'compat=\S+ invalid'):
            ds1.merge(ds2, compat='foobar')

    def test_merge_auto_align(self):
        ds1 = xr.Dataset({'a': ('x', [1, 2])})
        ds2 = xr.Dataset({'b': ('x', [3, 4]), 'x': [1, 2]})
        expected = xr.Dataset({'a': ('x', [1, 2, np.nan]),
                            'b': ('x', [np.nan, 3, 4])})
        assert expected.identical(ds1.merge(ds2))
        assert expected.identical(ds2.merge(ds1))

        expected = expected.isel(x=slice(2))
        assert expected.identical(ds1.merge(ds2, join='left'))
        assert expected.identical(ds2.merge(ds1, join='right'))

        expected = expected.isel(x=slice(1, 2))
        assert expected.identical(ds1.merge(ds2, join='inner'))
        assert expected.identical(ds2.merge(ds1, join='inner'))
