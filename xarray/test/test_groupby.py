import numpy as np
import pandas as pd
import pytest

import xarray as xr
from xarray.core.groupby import _consolidate_slices

from . import TestCase, unittest
from .test_dataset import create_test_data


def test_consolidate_slices():

    assert _consolidate_slices([slice(3), slice(3, 5)]) == [slice(5)]
    assert _consolidate_slices([slice(2, 3), slice(3, 6)]) == [slice(2, 6)]
    assert (_consolidate_slices([slice(2, 3, 1), slice(3, 6, 1)])
          == [slice(2, 6, 1)])

    slices = [slice(2, 3), slice(5, 6)]
    assert _consolidate_slices(slices) == slices

    with pytest.raises(ValueError):
        _consolidate_slices([slice(3), 4])


class TestDatasetGroupBy(TestCase):

    def test_groupby(self):
        data = xr.Dataset({'z': (['x', 'y'], np.random.randn(3, 5))},
                          {'x': ('x', list('abc')),
                           'c': ('x', [0, 1, 0])})
        groupby = data.groupby('x')
        self.assertEqual(len(groupby), 3)
        expected_groups = {'a': 0, 'b': 1, 'c': 2}
        self.assertEqual(groupby.groups, expected_groups)
        expected_items = [('a', data.isel(x=0)),
                          ('b', data.isel(x=1)),
                          ('c', data.isel(x=2))]
        for actual, expected in zip(groupby, expected_items):
            self.assertEqual(actual[0], expected[0])
            self.assertDatasetEqual(actual[1], expected[1])

        identity = lambda x: x
        for k in ['x', 'c', 'y']:
            actual = data.groupby(k, squeeze=False).apply(identity)
            self.assertDatasetEqual(data, actual)

    def test_groupby_returns_new_type(self):
        data = xr.Dataset({'z': (['x', 'y'], np.random.randn(3, 5))})

        actual = data.groupby('x').apply(lambda ds: ds['z'])
        expected = data['z']
        self.assertDataArrayIdentical(expected, actual)

        actual = data['z'].groupby('x').apply(lambda x: x.to_dataset())
        expected = data
        self.assertDatasetIdentical(expected, actual)

    def test_groupby_iter(self):
        data = create_test_data()
        for n, (t, sub) in enumerate(list(data.groupby('dim1'))[:3]):
            self.assertEqual(data['dim1'][n], t)
            self.assertVariableEqual(data['var1'][n], sub['var1'])
            self.assertVariableEqual(data['var2'][n], sub['var2'])
            self.assertVariableEqual(data['var3'][:, n], sub['var3'])

    def test_groupby_errors(self):
        data = create_test_data()
        with self.assertRaisesRegexp(TypeError, 'must be'):
            data.groupby(np.arange(10))
        with self.assertRaisesRegexp(ValueError, 'must have a name'):
            data.groupby(xr.DataArray(range(10), dims='foo'))
        with self.assertRaisesRegexp(ValueError, 'length does not match'):
            data.groupby(data['dim1'][:3])
        with self.assertRaisesRegexp(TypeError, 'must be'):
            data.groupby(data.coords['dim1'].to_index())

    def test_groupby_reduce(self):
        data = xr.Dataset({'xy': (['x', 'y'], np.random.randn(3, 4)),
                           'xonly': ('x', np.random.randn(3)),
                           'yonly': ('y', np.random.randn(4)),
                           'letters': ('y', ['a', 'a', 'b', 'b'])})

        expected = data.mean('y')
        expected['yonly'] = expected['yonly'].variable.expand_dims({'x': 3})
        actual = data.groupby('x').mean()
        self.assertDatasetAllClose(expected, actual)

        actual = data.groupby('x').mean('y')
        self.assertDatasetAllClose(expected, actual)

        letters = data['letters']
        expected = xr.Dataset({'xy': data['xy'].groupby(letters).mean(),
                               'xonly': (data['xonly'].mean().variable
                                         .expand_dims({'letters': 2})),
                               'yonly': data['yonly'].groupby(letters).mean()})
        actual = data.groupby('letters').mean()
        self.assertDatasetAllClose(expected, actual)

    def test_groupby_math(self):
        reorder_dims = lambda x: x.transpose('dim1', 'dim2', 'dim3', 'time')

        ds = create_test_data()
        for squeeze in [True, False]:
            grouped = ds.groupby('dim1', squeeze=squeeze)

            expected = reorder_dims(ds + ds.coords['dim1'])
            actual = grouped + ds.coords['dim1']
            self.assertDatasetIdentical(expected, reorder_dims(actual))

            actual = ds.coords['dim1'] + grouped
            self.assertDatasetIdentical(expected, reorder_dims(actual))

            ds2 = 2 * ds
            expected = reorder_dims(ds + ds2)
            actual = grouped + ds2
            self.assertDatasetIdentical(expected, reorder_dims(actual))

            actual = ds2 + grouped
            self.assertDatasetIdentical(expected, reorder_dims(actual))

        grouped = ds.groupby('numbers')
        zeros = xr.DataArray([0, 0, 0, 0], [('numbers', range(4))])
        expected = ((ds + xr.Variable('dim3', np.zeros(10)))
                    .transpose('dim3', 'dim1', 'dim2', 'time'))
        actual = grouped + zeros
        self.assertDatasetEqual(expected, actual)

        actual = zeros + grouped
        self.assertDatasetEqual(expected, actual)

        with self.assertRaisesRegexp(ValueError, 'dimensions .* do not exist'):
            grouped + ds
        with self.assertRaisesRegexp(ValueError, 'dimensions .* do not exist'):
            ds + grouped
        with self.assertRaisesRegexp(TypeError, 'only support binary ops'):
            grouped + 1
        with self.assertRaisesRegexp(TypeError, 'only support binary ops'):
            grouped + grouped
        with self.assertRaisesRegexp(TypeError, 'in-place operations'):
            ds += grouped

        ds = xr.Dataset({'x': ('time', np.arange(100)),
                         'time': pd.date_range('2000-01-01', periods=100)})
        with self.assertRaisesRegexp(ValueError, 'incompat.* grouped binary'):
            ds + ds.groupby('time.month')

    def test_groupby_math_virtual(self):
        ds = xr.Dataset({'x': ('t', [1, 2, 3])},
                        {'t': pd.date_range('20100101', periods=3)})
        grouped = ds.groupby('t.day')
        actual = grouped - grouped.mean()
        expected = xr.Dataset({'x': ('t', [0, 0, 0])},
                              ds[['t', 't.day']])
        self.assertDatasetIdentical(actual, expected)

    def test_groupby_nan(self):
        # nan should be excluded from groupby
        ds = xr.Dataset({'foo': ('x', [1, 2, 3, 4])},
                        {'bar': ('x', [1, 1, 2, np.nan])})
        actual = ds.groupby('bar').mean()
        expected = xr.Dataset({'foo': ('bar', [1.5, 3]), 'bar': [1, 2]})
        self.assertDatasetIdentical(actual, expected)


class TestDataArrayGroupBy(TestCase):

    def make_groupby_example_array(self):
        da = xr.DataArray(np.random.RandomState(0).rand(10, 20),
                          {'abc':  ('y', ['a'] * 9 + ['c'] + ['b'] * 10),
                           'y': 20 + 100 * np.arange(20)},
                          ('x', 'y'),
                          name='foo',
                          attrs={'attr1': 'value1', 'attr2': 2929})
        return da

    def test_groupby_iter(self):
        array = self.make_groupby_example_array()
        ds = array.to_dataset()
        for ((act_x, act_dv), (exp_x, exp_ds)) in \
                zip(array.groupby('y'), ds.groupby('y')):
            self.assertEqual(exp_x, act_x)
            self.assertDataArrayIdentical(exp_ds['foo'], act_dv)
        for ((_, exp_dv), act_dv) in zip(array.groupby('x'), array):
            self.assertDataArrayIdentical(exp_dv, act_dv)

    def test_groupby_properties(self):
        grouped = self.make_groupby_example_array().groupby('abc')
        expected_unique = xr.Variable('abc', ['a', 'b', 'c'])
        self.assertVariableEqual(expected_unique, grouped.unique_coord)
        self.assertEqual(3, len(grouped))

    def test_groupby_apply_identity(self):
        expected = self.make_groupby_example_array()
        idx = expected.coords['y']

        def identity(x):
            return x

        for g in ['x', 'y', 'abc', idx]:
            for shortcut in [False, True]:
                for squeeze in [False, True]:
                    grouped = expected.groupby(g, squeeze=squeeze)
                    actual = grouped.apply(identity, shortcut=shortcut)
                    self.assertDataArrayIdentical(expected, actual)

    def test_groupby_sum(self):
        array = self.make_groupby_example_array()
        grouped = array.groupby('abc')

        expected_sum_all = xr.Dataset(
            {'foo': (['abc'], np.array([array.values[:, :9].sum(),
                                        array.values[:, 10:].sum(),
                                        array.values[:, 9:10].sum()]).T),
             'abc': (['abc'], np.array(['a', 'b', 'c']))})['foo']
        self.assertDataArrayAllClose(expected_sum_all, grouped.reduce(np.sum))
        self.assertDataArrayAllClose(expected_sum_all, grouped.sum())

        expected = xr.DataArray([array['y'].values[idx].sum() for idx
                                 in [slice(9), slice(10, None), slice(9, 10)]],
                                [['a', 'b', 'c']], ['abc'])
        actual = array['y'].groupby('abc').apply(np.sum)
        self.assertDataArrayAllClose(expected, actual)
        actual = array['y'].groupby('abc').sum()
        self.assertDataArrayAllClose(expected, actual)

        expected_sum_axis1 = xr.Dataset(
            {'foo': (['x', 'abc'], np.array([array.values[:, :9].sum(1),
                                             array.values[:, 10:].sum(1),
                                             array.values[:, 9:10].sum(1)]).T),
             'x': array['x'],
             'abc': (['abc'], np.array(['a', 'b', 'c']))})['foo']
        self.assertDataArrayAllClose(expected_sum_axis1,
                                     grouped.reduce(np.sum, 'y'))
        self.assertDataArrayAllClose(expected_sum_axis1, grouped.sum('y'))

    def test_groupby_count(self):
        array = xr.DataArray([0, 0, np.nan, np.nan, 0, 0],
                          coords={'cat': ('x', ['a', 'b', 'b', 'c', 'c', 'c'])},
                          dims='x')
        actual = array.groupby('cat').count()
        expected = xr.DataArray([1, 1, 2], coords=[('cat', ['a', 'b', 'c'])])
        self.assertDataArrayIdentical(actual, expected)

    @unittest.skip('needs to be fixed for shortcut=False, keep_attrs=False')
    def test_groupby_reduce_attrs(self):
        array = self.make_groupby_example_array()
        array.attrs['foo'] = 'bar'

        for shortcut in [True, False]:
            for keep_attrs in [True, False]:
                print('shortcut=%s, keep_attrs=%s' % (shortcut, keep_attrs))
                actual = array.groupby('abc').reduce(
                    np.mean, keep_attrs=keep_attrs, shortcut=shortcut)
                expected = array.groupby('abc').mean()
                if keep_attrs:
                    expected.attrs['foo'] = 'bar'
                self.assertDataArrayIdentical(expected, actual)

    def test_groupby_apply_center(self):
        def center(x):
            return x - np.mean(x)

        array = self.make_groupby_example_array()
        grouped = array.groupby('abc')

        expected_ds = array.to_dataset()
        exp_data = np.hstack([center(array.values[:, :9]),
                              center(array.values[:, 9:10]),
                              center(array.values[:, 10:])])
        expected_ds['foo'] = (['x', 'y'], exp_data)
        expected_centered = expected_ds['foo']
        self.assertDataArrayAllClose(expected_centered, grouped.apply(center))

    def test_groupby_apply_ndarray(self):
        # regression test for #326
        array = self.make_groupby_example_array()
        grouped = array.groupby('abc')
        actual = grouped.apply(np.asarray)
        self.assertDataArrayEqual(array, actual)

    def test_groupby_apply_changes_metadata(self):
        def change_metadata(x):
            x.coords['x'] = x.coords['x'] * 2
            x.attrs['fruit'] = 'lemon'
            return x

        array = self.make_groupby_example_array()
        grouped = array.groupby('abc')
        actual = grouped.apply(change_metadata)
        expected = array.copy()
        expected = change_metadata(expected)
        self.assertDataArrayEqual(expected, actual)

    def test_groupby_math(self):
        array = self.make_groupby_example_array()
        for squeeze in [True, False]:
            grouped = array.groupby('x', squeeze=squeeze)

            expected = array + array.coords['x']
            actual = grouped + array.coords['x']
            self.assertDataArrayIdentical(expected, actual)

            actual = array.coords['x'] + grouped
            self.assertDataArrayIdentical(expected, actual)

            ds = array.coords['x'].to_dataset('X')
            expected = array + ds
            actual = grouped + ds
            self.assertDatasetIdentical(expected, actual)

            actual = ds + grouped
            self.assertDatasetIdentical(expected, actual)

        grouped = array.groupby('abc')
        expected_agg = (grouped.mean() - np.arange(3)).rename(None)
        actual = grouped - xr.DataArray(range(3), [('abc', ['a', 'b', 'c'])])
        actual_agg = actual.groupby('abc').mean()
        self.assertDataArrayAllClose(expected_agg, actual_agg)

        with self.assertRaisesRegexp(TypeError, 'only support binary ops'):
            grouped + 1
        with self.assertRaisesRegexp(TypeError, 'only support binary ops'):
            grouped + grouped
        with self.assertRaisesRegexp(TypeError, 'in-place operations'):
            array += grouped

    def test_groupby_math_not_aligned(self):
        array = xr.DataArray(range(4), {'b': ('x', [0, 0, 1, 1])}, dims='x')
        other = xr.DataArray([10], dims='b')
        actual = array.groupby('b') + other
        expected = xr.DataArray([10, 11, np.nan, np.nan], array.coords)
        self.assertDataArrayIdentical(expected, actual)

        other = xr.DataArray([10], coords={'c': 123}, dims='b')
        actual = array.groupby('b') + other
        expected.coords['c'] = (['x'], [123] * 2 + [np.nan] * 2)
        self.assertDataArrayIdentical(expected, actual)

        other = xr.Dataset({'a': ('b', [10])})
        actual = array.groupby('b') + other
        expected = xr.Dataset({'a': ('x', [10, 11, np.nan, np.nan])},
                           array.coords)
        self.assertDatasetIdentical(expected, actual)

    def test_groupby_restore_dim_order(self):
        array = xr.DataArray(np.random.randn(5, 3),
                          coords={'a': ('x', range(5)), 'b': ('y', range(3))},
                          dims=['x', 'y'])
        for by, expected_dims in [('x', ('x', 'y')),
                                  ('y', ('x', 'y')),
                                  ('a', ('a', 'y')),
                                  ('b', ('x', 'b'))]:
            result = array.groupby(by).apply(lambda x: x.squeeze())
            self.assertEqual(result.dims, expected_dims)

    def test_groupby_first_and_last(self):
        array = xr.DataArray([1, 2, 3, 4, 5], dims='x')
        by = xr.DataArray(['a'] * 2 + ['b'] * 3, dims='x', name='ab')

        expected = xr.DataArray([1, 3], [('ab', ['a', 'b'])])
        actual = array.groupby(by).first()
        self.assertDataArrayIdentical(expected, actual)

        expected = xr.DataArray([2, 5], [('ab', ['a', 'b'])])
        actual = array.groupby(by).last()
        self.assertDataArrayIdentical(expected, actual)

        array = xr.DataArray(np.random.randn(5, 3), dims=['x', 'y'])
        expected = xr.DataArray(array[[0, 2]], {'ab': ['a', 'b']}, ['ab', 'y'])
        actual = array.groupby(by).first()
        self.assertDataArrayIdentical(expected, actual)

        actual = array.groupby('x').first()
        expected = array  # should be a no-op
        self.assertDataArrayIdentical(expected, actual)

    def make_groupby_multidim_example_array(self):
        return xr.DataArray([[[0, 1], [2, 3]], [[5, 10], [15, 20]]],
                        coords={'lon': (['ny', 'nx'], [[30., 40.], [40., 50.]]),
                                'lat': (['ny', 'nx'],
                                        [[10., 10.], [20., 20.]])},
                        dims=['time', 'ny', 'nx'])

    def test_groupby_multidim(self):
        array = self.make_groupby_multidim_example_array()
        for dim, expected_sum in [
                ('lon', xr.DataArray([5, 28, 23],
                                     coords={'lon': [30., 40., 50.]})),
                ('lat', xr.DataArray([16, 40], coords={'lat': [10., 20.]}))]:
            actual_sum = array.groupby(dim).sum()
            self.assertDataArrayIdentical(expected_sum, actual_sum)

    def test_groupby_multidim_apply(self):
        array = self.make_groupby_multidim_example_array()
        actual = array.groupby('lon').apply(
            lambda x: x - x.mean(), shortcut=False)
        expected = xr.DataArray([[[-2.5, -6.], [-5., -8.5]],
                                 [[2.5,  3.], [8., 8.5]]],
                                coords=array.coords, dims=array.dims)
        self.assertDataArrayIdentical(expected, actual)

    def test_groupby_bins(self):
        array = xr.DataArray(np.arange(4), dims='dim_0')
        # the first value should not be part of any group ("right" binning)
        array[0] = 99
        # bins follow conventions for pandas.cut
        # http://pandas.pydata.org/pandas-docs/stable/generated/pandas.cut.html
        bins = [0, 1.5, 5]
        bin_coords = ['(0, 1.5]', '(1.5, 5]']
        expected = xr.DataArray([1, 5], dims='dim_0_bins',
                                coords={'dim_0_bins': bin_coords})
        # the problem with this is that it overwrites the dimensions of array!
        #actual = array.groupby('dim_0', bins=bins).sum()
        actual = array.groupby_bins('dim_0', bins).apply(
            lambda x: x.sum(), shortcut=False)
        self.assertDataArrayIdentical(expected, actual)
        # make sure original array dims are unchanged
        # (would fail with shortcut=True above)
        self.assertEqual(len(array.dim_0), 4)

    def test_groupby_bins_multidim(self):
        array = self.make_groupby_multidim_example_array()
        bins = [0, 15, 20]
        bin_coords = ['(0, 15]', '(15, 20]']
        expected = xr.DataArray([16, 40], dims='lat_bins',
                                coords={'lat_bins': bin_coords})
        actual = array.groupby_bins('lat', bins).apply(
            lambda x: x.sum(), shortcut=False)
        self.assertDataArrayIdentical(expected, actual)
        # modify the array coordinates to be non-monotonic after unstacking
        array['lat'].data = np.array([[10., 20.], [20., 10.]])
        expected = xr.DataArray([28, 28], dims='lat_bins',
                                coords={'lat_bins': bin_coords})
        actual = array.groupby_bins('lat', bins).apply(
            lambda x: x.sum(), shortcut=False)
        self.assertDataArrayIdentical(expected, actual)


def test_multi_index_groupby_apply():
    # regression test for GH873
    ds = xr.Dataset({'foo': (('x', 'y'), np.random.randn(3, 4))},
                    {'x': ['a', 'b', 'c'], 'y': [1, 2, 3, 4]})
    doubled = 2 * ds
    group_doubled = (ds.stack(space=['x', 'y'])
                     .groupby('space')
                     .apply(lambda x: 2 * x)
                     .unstack('space'))
    assert doubled.equals(group_doubled)


def test_multi_index_groupby_sum():
    # regression test for GH873
    ds = xr.Dataset({'foo': (('x', 'y', 'z'), np.ones((3, 4, 2)))},
                    {'x': ['a', 'b', 'c'], 'y': [1, 2, 3, 4]})
    expected = ds.sum('z')
    actual = (ds.stack(space=['x', 'y'])
              .groupby('space')
              .sum('z')
              .unstack('space'))
    assert expected.equals(actual)


class TestMultipleArgumentGroupby(TestCase):

    def test_apply_identity(self):
        ds = xr.Dataset({'foo': ('x', [1, 2, 3])}, {'y': ('x', list('abc'))})
        roundtripped = ds.groupby(['x', 'y']).apply(lambda x: x)
        pandas_roundtripped = xr.Dataset.from_dataframe(
            ds.to_dataframe().reset_index().groupby(['x', 'y']).apply(lambda x: x))
        self.assertDatasetIdentical(ds, roundtripped)
        self.assertDatasetIdentical(ds, pandas_roundtripped)

    def test_sum_identity(self):
        ds = xr.Dataset({'foo': ('x', [1, 2, 3])}, {'y': ('x', list('abc'))})
        roundtripped = ds.groupby(['x', 'y']).sum()
        pandas_roundtripped = xr.Dataset.from_dataframe(
            ds.to_dataframe().reset_index().groupby(['x', 'y']).sum())
        self.assertDatasetIdentical(ds, roundtripped)
        self.assertDatasetIdentical(ds, pandas_roundtripped)
