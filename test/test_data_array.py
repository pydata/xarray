import numpy as np
import pandas as pd
from copy import deepcopy
from textwrap import dedent
from collections import OrderedDict

from xray import Dataset, DataArray, Variable, align
from xray.data_array import Indexes
from xray.pycompat import iteritems
from . import TestCase, ReturnItem, source_ndarray


class TestIndexes(TestCase):
    def test(self):
        indexes = Indexes(['a', 'b', 'c'], [0, 1, 2])
        self.assertEqual(indexes['a'], 0)
        self.assertEqual(indexes[0], 0)
        self.assertEqual(indexes[:1], Indexes(['a'], [0]))
        self.assertEqual(indexes[:], indexes)
        self.assertEqual(repr(indexes),
                         "Indexes(['a', 'b', 'c'], [0, 1, 2])")


class TestDataArray(TestCase):
    def setUp(self):
        self.attrs = {'attr1': 'value1', 'attr2': 2929}
        self.x = np.random.random((10, 20))
        self.v = Variable(['x', 'y'], self.x)
        self.va = Variable(['x', 'y'], self.x, self.attrs)
        self.ds = Dataset({'foo': self.v})
        self.dv = self.ds['foo']

    def test_repr(self):
        v = Variable(['time', 'x'], [[1, 2, 3], [4, 5, 6]], {'foo': 'bar'})
        data_array = Dataset({'my_variable': v, 'other': ([], 0)}
                             )['my_variable']
        expected = dedent("""
        <xray.DataArray 'my_variable' (time: 2, x: 3)>
        array([[1, 2, 3],
               [4, 5, 6]])
        Indexes:
            time: Int64Index([0, 1], dtype='int64')
            x: Int64Index([0, 1, 2], dtype='int64')
        Linked dataset variables:
            other
        Attributes:
            foo: bar
        """).strip()
        self.assertEqual(expected, repr(data_array))

    def test_properties(self):
        self.assertDatasetIdentical(self.dv.dataset, self.ds)
        self.assertEqual(self.dv.name, 'foo')
        self.assertVariableEqual(self.dv.variable, self.v)
        self.assertArrayEqual(self.dv.values, self.v.values)
        for attr in ['dimensions', 'dtype', 'shape', 'size', 'ndim', 'attrs']:
            self.assertEqual(getattr(self.dv, attr), getattr(self.v, attr))
        self.assertEqual(len(self.dv), len(self.v))
        self.assertVariableEqual(self.dv, self.v)
        self.assertEqual(list(self.dv.indexes), list(self.ds.indexes))
        for k, v in iteritems(self.dv.indexes):
            self.assertArrayEqual(v, self.ds.indexes[k])
        with self.assertRaises(AttributeError):
            self.dv.name = 'bar'
        with self.assertRaises(AttributeError):
            self.dv.dataset = self.ds
        self.assertIsInstance(self.ds['x'].as_index, pd.Index)
        with self.assertRaisesRegexp(ValueError, 'must be 1-dimensional'):
            self.ds['foo'].as_index

    def test_constructor(self):
        data = np.random.random((2, 3))

        actual = DataArray(data)
        expected = Dataset({None: (['dim_0', 'dim_1'], data)})[None]
        self.assertDataArrayIdentical(expected, actual)

        actual = DataArray(data, [['a', 'b'], [-1, -2, -3]])
        expected = Dataset({None: (['dim_0', 'dim_1'], data),
                            'dim_0': ('dim_0', ['a', 'b']),
                            'dim_1': ('dim_1', [-1, -2, -3])})[None]
        self.assertDataArrayIdentical(expected, actual)

        actual = DataArray(data, [pd.Index(['a', 'b'], name='x'),
                                  pd.Index([-1, -2, -3], name='y')])
        expected = Dataset({None: (['x', 'y'], data),
                            'x': ('x', ['a', 'b']),
                            'y': ('y', [-1, -2, -3])})[None]
        self.assertDataArrayIdentical(expected, actual)

        indexes = [['a', 'b'], [-1, -2, -3]]
        actual = DataArray(data, indexes, ['x', 'y'])
        self.assertDataArrayIdentical(expected, actual)

        indexes = [pd.Index(['a', 'b'], name='A'),
                   pd.Index([-1, -2, -3], name='B')]
        actual = DataArray(data, indexes, ['x', 'y'])
        self.assertDataArrayIdentical(expected, actual)

        indexes = {'x': ['a', 'b'], 'y': [-1, -2, -3]}
        actual = DataArray(data, indexes, ['x', 'y'])
        self.assertDataArrayIdentical(expected, actual)

        indexes = OrderedDict([('x', ['a', 'b']), ('y', [-1, -2, -3])])
        actual = DataArray(data, indexes)
        self.assertDataArrayIdentical(expected, actual)

        indexes = pd.Series([['a', 'b'], [-1, -2, -3]], ['x', 'y'])
        actual = DataArray(data, indexes)
        self.assertDataArrayIdentical(expected, actual)

        expected = Dataset({None: (['x', 'y'], data),
                            'x': ('x', ['a', 'b'])})[None]
        actual = DataArray(data, {'x': ['a', 'b']}, ['x', 'y'])
        self.assertDataArrayIdentical(expected, actual)

        with self.assertRaisesRegexp(ValueError, 'but data has ndim'):
            DataArray(data, [[0, 1, 2]], ['x', 'y'])

        with self.assertRaisesRegexp(ValueError, 'not array dimensions'):
            DataArray(data, {'x': [0, 1, 2]}, ['a', 'b'])

        with self.assertRaisesRegexp(ValueError, 'must have the same length'):
            DataArray(data, {'x': [0, 1, 2]})

        actual = DataArray(data, dimensions=['x', 'y'])
        expected = Dataset({None: (['x', 'y'], data)})[None]
        self.assertDataArrayIdentical(expected, actual)

        actual = DataArray(data, dimensions=['x', 'y'], name='foo')
        expected = Dataset({'foo': (['x', 'y'], data)})['foo']
        self.assertDataArrayIdentical(expected, actual)

        with self.assertRaisesRegexp(TypeError, 'is not a string'):
            DataArray(data, dimensions=['x', None])

        actual = DataArray(data, name='foo')
        expected = Dataset({'foo': (['dim_0', 'dim_1'], data)})['foo']
        self.assertDataArrayIdentical(expected, actual)

        actual = DataArray(data, dimensions=['x', 'y'], attributes={'bar': 2})
        expected = Dataset({None: (['x', 'y'], data, {'bar': 2})})[None]
        self.assertDataArrayIdentical(expected, actual)

        actual = DataArray(data, dimensions=['x', 'y'], encoding={'bar': 2})
        expected = Dataset({None: (['x', 'y'], data, {}, {'bar': 2})})[None]
        self.assertDataArrayIdentical(expected, actual)

    def test_constructor_from_self_described(self):
        data = [[-0.1, 21], [0, 2]]
        expected = DataArray(data,
                             indexes={'x': ['a', 'b'], 'y': [-1, -2]},
                             dimensions=['x', 'y'], name='foobar',
                             attributes={'bar': 2}, encoding={'foo': 3})
        actual = DataArray(expected)
        self.assertDataArrayIdentical(expected, actual)

        frame = pd.DataFrame(data, index=pd.Index(['a', 'b'], name='x'),
                             columns=pd.Index([-1, -2], name='y'))
        actual = DataArray(frame)
        self.assertDataArrayEqual(expected, actual)

        series = pd.Series(data[0], index=pd.Index([-1, -2], name='y'))
        actual = DataArray(series)
        self.assertDataArrayEqual(expected[0], actual)

        panel = pd.Panel({0: frame})
        actual = DataArray(panel)
        expected = DataArray([data], expected.indexes, ['dim_0', 'x', 'y'])
        self.assertDataArrayIdentical(expected, actual)

        expected = DataArray(['a', 'b'], name='foo')
        actual = DataArray(pd.Index(['a', 'b'], name='foo'))
        self.assertDataArrayIdentical(expected, actual)

    def test_equals_and_identical(self):
        da2 = self.dv.copy()
        self.assertTrue(self.dv.equals(da2))
        self.assertTrue(self.dv.identical(da2))

        da3 = self.dv.rename('baz')
        self.assertTrue(self.dv.equals(da3))
        self.assertFalse(self.dv.identical(da3))

        da4 = self.dv.rename({'x': 'xxx'})
        self.assertFalse(self.dv.equals(da4))
        self.assertFalse(self.dv.identical(da4))

        da5 = self.dv.copy()
        da5.attrs['foo'] = 'bar'
        self.assertTrue(self.dv.equals(da5))
        self.assertFalse(self.dv.identical(da5))

        da6 = self.dv.copy()
        da6['x'] = ('x', -np.arange(10))
        self.assertFalse(self.dv.equals(da6))
        self.assertFalse(self.dv.identical(da6))

        da2[0, 0] = np.nan
        self.dv[0, 0] = np.nan
        self.assertTrue(self.dv.equals(da2))
        self.assertTrue(self.dv.identical(da2))

        da2[:] = np.nan
        self.assertFalse(self.dv.equals(da2))
        self.assertFalse(self.dv.identical(da2))

    def test_items(self):
        # strings pull out dataarrays
        self.assertDataArrayIdentical(self.dv, self.ds['foo'])
        x = self.dv['x']
        y = self.dv['y']
        self.assertDataArrayIdentical(self.ds['x'], x)
        self.assertDataArrayIdentical(self.ds['y'], y)
        # integer indexing
        I = ReturnItem()
        for i in [I[:], I[...], I[x.values], I[x.variable], I[x], I[x, y],
                  I[x.values > -1], I[x.variable > -1], I[x > -1],
                  I[x > -1, y > -1]]:
            self.assertVariableEqual(self.dv, self.dv[i])
        for i in [I[0], I[:, 0], I[:3, :2],
                  I[x.values[:3]], I[x.variable[:3]], I[x[:3]], I[x[:3], y[:4]],
                  I[x.values > 3], I[x.variable > 3], I[x > 3], I[x > 3, y > 3]]:
            self.assertVariableEqual(self.v[i], self.dv[i])
        # make sure we always keep the array around, even if it's a scalar
        self.assertVariableEqual(self.dv[0, 0], self.dv.variable[0, 0])
        for k in ['x', 'y', 'foo']:
            self.assertIn(k, self.dv[0, 0].dataset)

    def test_isel(self):
        self.assertEqual(self.dv[0].dataset, self.ds.isel(x=0))
        self.assertEqual(self.dv[:3, :5].dataset,
                         self.ds.isel(x=slice(3), y=slice(5)))
        self.assertDataArrayIdentical(self.dv, self.dv.isel(x=slice(None)))
        self.assertDataArrayIdentical(self.dv[:3], self.dv.isel(x=slice(3)))

    def test_sel(self):
        self.ds['x'] = ('x', np.array(list('abcdefghij')))
        da = self.ds['foo']
        self.assertDataArrayIdentical(da, da.sel(x=slice(None)))
        self.assertDataArrayIdentical(da[1], da.sel(x='b'))
        self.assertDataArrayIdentical(da[:3], da.sel(x=slice('c')))

    def test_loc(self):
        self.ds['x'] = ('x', np.array(list('abcdefghij')))
        da = self.ds['foo']
        self.assertDataArrayIdentical(da[:3], da.loc[:'c'])
        self.assertDataArrayIdentical(da[1], da.loc['b'])
        self.assertDataArrayIdentical(da[:3], da.loc[['a', 'b', 'c']])
        self.assertDataArrayIdentical(da[:3, :4],
                                      da.loc[['a', 'b', 'c'], np.arange(4)])
        da.loc['a':'j'] = 0
        self.assertTrue(np.all(da.values == 0))

    def test_reindex(self):
        foo = self.dv
        bar = self.dv[:2, :2]
        self.assertDataArrayIdentical(foo.reindex_like(bar), bar)

        expected = foo.copy()
        expected[:] = np.nan
        expected[:2, :2] = bar
        self.assertDataArrayIdentical(bar.reindex_like(foo), expected)

    def test_rename(self):
        renamed = self.dv.rename('bar')
        self.assertDatasetIdentical(
            renamed.dataset, self.ds.rename({'foo': 'bar'}))
        self.assertEqual(renamed.name, 'bar')

        renamed = self.dv.rename({'foo': 'bar'})
        self.assertDatasetIdentical(
            renamed.dataset, self.ds.rename({'foo': 'bar'}))
        self.assertEqual(renamed.name, 'bar')

    def test_dataset_getitem(self):
        dv = self.ds['foo']
        self.assertDataArrayIdentical(dv, self.dv)

    def test_array_interface(self):
        self.assertArrayEqual(np.asarray(self.dv), self.x)
        # test patched in methods
        self.assertArrayEqual(self.dv.astype(float), self.v.astype(float))
        self.assertVariableEqual(self.dv.argsort(), self.v.argsort())
        self.assertVariableEqual(self.dv.clip(2, 3), self.v.clip(2, 3))
        # test ufuncs
        expected = deepcopy(self.ds)
        expected['foo'][:] = np.sin(self.x)
        self.assertDataArrayEqual(expected['foo'], np.sin(self.dv))
        self.assertDataArrayEqual(self.dv, np.maximum(self.v, self.dv))
        bar = Variable(['x', 'y'], np.zeros((10, 20)))
        self.assertDataArrayEqual(self.dv, np.maximum(self.dv, bar))

    def test_math(self):
        x = self.x
        v = self.v
        a = self.dv
        # variable math was already tested extensively, so let's just make sure
        # that all types are properly converted here
        self.assertDataArrayEqual(a, +a)
        self.assertDataArrayEqual(a, a + 0)
        self.assertDataArrayEqual(a, 0 + a)
        self.assertDataArrayEqual(a, a + 0 * v)
        self.assertDataArrayEqual(a, 0 * v + a)
        self.assertDataArrayEqual(a, a + 0 * x)
        self.assertDataArrayEqual(a, 0 * x + a)
        self.assertDataArrayEqual(a, a + 0 * a)
        self.assertDataArrayEqual(a, 0 * a + a)
        # test different indices
        ds2 = self.ds.update({'x': ('x', 3 + np.arange(10))}, inplace=False)
        b = ds2['foo']
        with self.assertRaisesRegexp(ValueError, 'not aligned'):
            a + b
        with self.assertRaisesRegexp(ValueError, 'not aligned'):
            b + a
        with self.assertRaisesRegexp(TypeError, 'datasets do not support'):
            a + a.dataset

    def test_dataset_math(self):
        # verify that mathematical operators keep around the expected variables
        # when doing math with dataset arrays from one or more aligned datasets
        obs = Dataset({'tmin': ('x', np.arange(5)),
                       'tmax': ('x', 10 + np.arange(5)),
                       'x': ('x', 0.5 * np.arange(5))})

        actual = 2 * obs['tmax']
        expected = Dataset({'tmax2': ('x', 2 * (10 + np.arange(5))),
                            'x': obs['x']})['tmax2']
        self.assertDataArrayEqual(actual, expected)

        actual = obs['tmax'] - obs['tmin']
        expected = Dataset({'trange': ('x', 10 * np.ones(5)),
                            'x': obs['x']})['trange']
        self.assertDataArrayEqual(actual, expected)

        sim = Dataset({'tmin': ('x', 1 + np.arange(5)),
                       'tmax': ('x', 11 + np.arange(5)),
                       'x': ('x', 0.5 * np.arange(5))})

        actual = sim['tmin'] - obs['tmin']
        expected = Dataset({'error': ('x', np.ones(5)),
                            'x': obs['x']})['error']
        self.assertDataArrayEqual(actual, expected)

        # in place math shouldn't remove or conflict with other variables
        actual = deepcopy(sim['tmin'])
        actual -= obs['tmin']
        expected = Dataset({'tmin': ('x', np.ones(5)),
                            'tmax': sim['tmax'],
                            'x': sim['x']})['tmin']
        self.assertDataArrayEqual(actual, expected)

    def test_math_name(self):
        # Verify that name is preserved only when it can be done unambiguously.
        # The rule (copied from pandas.Series) is keep the current name only if
        # the other object has no name attribute and this object isn't a
        # coordinate; otherwise reset to None.
        ds = self.ds
        a = self.dv
        self.assertEqual((+a).name, 'foo')
        self.assertEqual((a + 0).name, 'foo')
        self.assertIs((a + a.rename(None)).name, None)
        self.assertIs((a + a).name, None)
        self.assertIs((+ds['x']).name, None)
        self.assertIs((ds['x'] + 0).name, None)
        self.assertIs((a + ds['x']).name, None)

    def test_coord_math(self):
        ds = Dataset({'x': ('x', 1 + np.arange(3))})
        expected = ds.copy()
        expected['x2'] = ('x', np.arange(3))
        actual = ds['x'] - 1
        self.assertDataArrayEqual(expected['x2'], actual)

    def test_item_math(self):
        self.ds['x'] = ('x', np.array(list('abcdefghij')))
        self.assertVariableEqual(self.dv + self.dv[0, 0],
                               self.dv + self.dv[0, 0].values)
        new_data = self.x[0][None, :] + self.x[:, 0][:, None]
        self.assertVariableEqual(self.dv[:, 0] + self.dv[0],
                                 Variable(['x', 'y'], new_data))
        self.assertVariableEqual(self.dv[0] + self.dv[:, 0],
                                 Variable(['y', 'x'], new_data.T))

    def test_inplace_math(self):
        x = self.x
        v = self.v
        a = self.dv
        b = a
        b += 1
        self.assertIs(b, a)
        self.assertIs(b.variable, v)
        self.assertArrayEqual(b.values, x)
        self.assertIs(source_ndarray(b.values), x)
        self.assertDatasetIdentical(b.dataset, self.ds)

    def test_transpose(self):
        self.assertVariableEqual(self.dv.variable.transpose(),
                               self.dv.transpose())

    def test_squeeze(self):
        self.assertVariableEqual(self.dv.variable.squeeze(), self.dv.squeeze())

    def test_reduce(self):
        self.assertVariableEqual(self.dv.reduce(np.mean, 'x'),
                            self.v.reduce(np.mean, 'x'))
        # needs more...
        # should check which extra dimensions are dropped

    def test_reduce_keep_attrs(self):
        # Test dropped attrs
        vm = self.va.mean()
        self.assertEqual(len(vm.attrs), 0)
        self.assertEqual(vm.attrs, OrderedDict())

        # Test kept attrs
        vm = self.va.mean(keep_attrs=True)
        self.assertEqual(len(vm.attrs), len(self.attrs))
        self.assertEqual(vm.attrs, self.attrs)

    def test_drop_vars(self):
        with self.assertRaisesRegexp(ValueError, 'cannot drop the name'):
            self.dv.drop_vars('foo')
        with self.assertRaisesRegexp(ValueError, 'must be a variable in'):
            self.dv.drop_vars('y')

    def test_groupby_iter(self):
        for ((act_x, act_dv), (exp_x, exp_ds)) in \
                zip(self.dv.groupby('y'), self.ds.groupby('y')):
            self.assertEqual(exp_x, act_x)
            self.assertDataArrayIdentical(exp_ds['foo'], act_dv)
        for ((_, exp_dv), act_dv) in zip(self.dv.groupby('x'), self.dv):
            self.assertDataArrayIdentical(exp_dv, act_dv)

    def make_groupby_example_array(self):
        da = self.dv.copy()
        agg_var = Variable(['y'], np.array(['a'] * 9 + ['c'] + ['b'] * 10))
        da['abc'] = agg_var
        da['y'] = 20 + 100 * da['y']
        return da

    def test_groupby_properties(self):
        grouped = self.make_groupby_example_array().groupby('abc')
        expected_unique = Variable('abc', ['a', 'b', 'c'])
        self.assertVariableEqual(expected_unique, grouped.unique_index)
        self.assertEqual(3, len(grouped))

    def test_groupby_apply_identity(self):
        expected = self.make_groupby_example_array()
        idx = expected.indexes['y']
        identity = lambda x: x
        for g in ['x', 'y', 'abc', idx]:
            for shortcut in [False, True]:
                for squeeze in [False, True]:
                    grouped = expected.groupby(g, squeeze=squeeze)
                    actual = grouped.apply(identity, shortcut=shortcut)
                    self.assertDataArrayIdentical(expected, actual)

    def test_groupby_sum(self):
        array = self.make_groupby_example_array()
        grouped = array.groupby('abc')

        expected_sum_all = Dataset(
            {'foo': Variable(['abc'], np.array([self.x[:, :9].sum(),
                                                self.x[:, 10:].sum(),
                                                self.x[:, 9:10].sum()]).T),
             'abc': Variable(['abc'], np.array(['a', 'b', 'c']))})['foo']
        self.assertDataArrayAllClose(expected_sum_all, grouped.reduce(np.sum))
        self.assertDataArrayAllClose(expected_sum_all, grouped.sum())

        expected = DataArray([array['y'].values[idx].sum() for idx
                              in [slice(9), slice(10, None), slice(9, 10)]],
                             [['a', 'b', 'c']], ['abc'])
        actual = array['y'].groupby('abc').apply(np.sum)
        self.assertDataArrayAllClose(expected, actual)
        actual = array['y'].groupby('abc').sum()
        self.assertDataArrayAllClose(expected, actual)

        expected_sum_axis1 = Dataset(
            {'foo': (['x', 'abc'], np.array([self.x[:, :9].sum(1),
                                             self.x[:, 10:].sum(1),
                                             self.x[:, 9:10].sum(1)]).T),
             'x': self.ds.variables['x'],
             'abc': Variable(['abc'], np.array(['a', 'b', 'c']))})['foo']
        self.assertDataArrayAllClose(expected_sum_axis1,
                                     grouped.reduce(np.sum, 'y'))
        self.assertDataArrayAllClose(expected_sum_axis1, grouped.sum('y'))

    def test_groupby_apply_center(self):
        def center(x):
            return x - np.mean(x)

        array = self.make_groupby_example_array()
        grouped = array.groupby('abc')

        expected_ds = array.dataset.copy()
        exp_data = np.hstack([center(self.x[:, :9]),
                              center(self.x[:, 9:10]),
                              center(self.x[:, 10:])])
        expected_ds['foo'] = (['x', 'y'], exp_data)
        expected_centered = expected_ds['foo']
        self.assertDataArrayAllClose(expected_centered, grouped.apply(center))

    def test_concat(self):
        self.ds['bar'] = Variable(['x', 'y'], np.random.randn(10, 20))
        foo = self.ds['foo'].select_vars()
        bar = self.ds['bar'].rename('foo').select_vars()
        # from dataset array:
        self.assertVariableEqual(Variable(['w', 'x', 'y'],
                                          np.array([foo.values, bar.values])),
                                 DataArray.concat([foo, bar], 'w'))
        # from iteration:
        grouped = [g for _, g in foo.groupby('x')]
        stacked = DataArray.concat(grouped, self.ds['x'])
        self.assertDataArrayIdentical(foo.select_vars(), stacked)

    def test_align(self):
        self.ds['x'] = ('x', np.array(list('abcdefghij')))
        with self.assertRaises(ValueError):
            self.dv + self.dv[:5]
        dv1, dv2 = align(self.dv, self.dv[:5], join='inner')
        self.assertDataArrayIdentical(dv1, self.dv[:5])
        self.assertDataArrayIdentical(dv2, self.dv[:5])

    def test_to_and_from_series(self):
        expected = self.dv.to_dataframe()['foo']
        actual = self.dv.to_series()
        self.assertArrayEqual(expected.values, actual.values)
        self.assertArrayEqual(expected.index.values, actual.index.values)
        self.assertEqual('foo', actual.name)
        # test roundtrip
        self.assertDataArrayIdentical(self.dv, DataArray.from_series(actual))
        # test name is None
        actual.name = None
        expected_da = self.dv.rename(None)
        self.assertDataArrayIdentical(expected_da,
                                      DataArray.from_series(actual))
