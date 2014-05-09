import numpy as np
import pandas as pd
from copy import deepcopy
from textwrap import dedent

from xray import Dataset, DataArray, Variable, align
from xray.pycompat import iteritems
from . import TestCase, ReturnItem, source_ndarray


class TestDataArray(TestCase):
    def setUp(self):
        self.x = np.random.random((10, 20))
        self.v = Variable(['x', 'y'], self.x)
        self.ds = Dataset({'foo': self.v})
        self.dv = self.ds['foo']

    def test_repr(self):
        v = Variable(['time', 'x'], [[1, 2, 3], [4, 5, 6]], {'foo': 'bar'})
        data_array = Dataset({'my_variable': v})['my_variable']
        expected = dedent("""
        <xray.DataArray 'my_variable' (time: 2, x: 3)>
        array([[1, 2, 3],
               [4, 5, 6]])
        Attributes:
            foo: bar
        """).strip()
        self.assertEqual(expected, repr(data_array))

    def test_properties(self):
        self.assertIs(self.dv.dataset, self.ds)
        self.assertEqual(self.dv.name, 'foo')
        self.assertVariableEqual(self.dv.variable, self.v)
        self.assertArrayEqual(self.dv.values, self.v.values)
        for attr in ['dimensions', 'dtype', 'shape', 'size', 'ndim', 'attrs']:
            self.assertEqual(getattr(self.dv, attr), getattr(self.v, attr))
        self.assertEqual(len(self.dv), len(self.v))
        self.assertVariableEqual(self.dv, self.v)
        self.assertEqual(list(self.dv.coordinates), list(self.ds.coordinates))
        for k, v in iteritems(self.dv.coordinates):
            self.assertArrayEqual(v, self.ds.coordinates[k])
        with self.assertRaises(AttributeError):
            self.dv.name = 'bar'
        with self.assertRaises(AttributeError):
            self.dv.dataset = self.ds
        self.assertIsInstance(self.ds['x'].as_index, pd.Index)
        with self.assertRaisesRegexp(ValueError, 'must be 1-dimensional'):
            self.ds['foo'].as_index

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

    def test_indexed(self):
        self.assertEqual(self.dv[0].dataset, self.ds.indexed(x=0))
        self.assertEqual(self.dv[:3, :5].dataset,
                         self.ds.indexed(x=slice(3), y=slice(5)))
        self.assertDataArrayIdentical(self.dv, self.dv.indexed(x=slice(None)))
        self.assertDataArrayIdentical(self.dv[:3], self.dv.indexed(x=slice(3)))

    def test_labeled(self):
        self.ds['x'] = ('x', np.array(list('abcdefghij')))
        self.assertDataArrayIdentical(self.dv, self.dv.labeled(x=slice(None)))
        self.assertDataArrayIdentical(self.dv[1], self.dv.labeled(x='b'))
        self.assertDataArrayIdentical(self.dv[:3], self.dv.labeled(x=slice('c')))

    def test_loc(self):
        self.ds['x'] = ('x', np.array(list('abcdefghij')))
        self.assertDataArrayIdentical(self.dv[:3], self.dv.loc[:'c'])
        self.assertDataArrayIdentical(self.dv[1], self.dv.loc['b'])
        self.assertDataArrayIdentical(self.dv[:3], self.dv.loc[['a', 'b', 'c']])
        self.assertDataArrayIdentical(self.dv[:3, :4],
                                      self.dv.loc[['a', 'b', 'c'], np.arange(4)])
        self.dv.loc['a':'j'] = 0
        self.assertTrue(np.all(self.dv.values == 0))

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
        self.assertEqual(renamed.dataset, self.ds.rename({'foo': 'bar'}))
        self.assertEqual(renamed.name, 'bar')

        renamed = self.dv.rename({'foo': 'bar'})
        self.assertEqual(renamed.dataset, self.ds.rename({'foo': 'bar'}))
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
        self.assertIs(b.dataset, self.ds)

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

    def test_unselect(self):
        with self.assertRaisesRegexp(ValueError, 'cannot unselect the name'):
            self.dv.unselect('foo')
        with self.assertRaisesRegexp(ValueError, 'must be a variable in'):
            self.dv.unselect('y')

    def test_groupby_iter(self):
        for ((act_x, act_dv), (exp_x, exp_ds)) in \
                zip(self.dv.groupby('y'), self.ds.groupby('y')):
            self.assertEqual(exp_x, act_x)
            self.assertDataArrayIdentical(exp_ds['foo'], act_dv)
        for ((_, exp_dv), act_dv) in zip(self.dv.groupby('x'), self.dv):
            self.assertDataArrayIdentical(exp_dv, act_dv)

    def test_groupby(self):
        agg_var = Variable(['y'], np.array(['a'] * 9 + ['c'] + ['b'] * 10))
        self.dv['abc'] = agg_var
        self.dv['y'] = 20 + 100 * self.ds['y'].variable

        identity = lambda x: x
        for g in ['x', 'y', 'abc']:
            for shortcut in [False, True]:
                for squeeze in [False, True]:
                    expected = self.dv
                    grouped = self.dv.groupby(g, squeeze=squeeze)
                    actual = grouped.apply(identity, shortcut=shortcut)
                    self.assertDataArrayIdentical(expected, actual)

        grouped = self.dv.groupby('abc', squeeze=True)
        expected_sum_all = Dataset(
            {'foo': Variable(['abc'], np.array([self.x[:, :9].sum(),
                                                self.x[:, 10:].sum(),
                                                self.x[:, 9:10].sum()]).T),
             'abc': Variable(['abc'], np.array(['a', 'b', 'c']))})['foo']
        self.assertDataArrayAllClose(
            expected_sum_all, grouped.reduce(np.sum))
        self.assertDataArrayAllClose(
            expected_sum_all, grouped.sum())
        self.assertDataArrayAllClose(
            expected_sum_all, grouped.sum())
        expected_unique = Variable('abc', ['a', 'b', 'c'])
        self.assertVariableEqual(expected_unique, grouped.unique_coord)
        self.assertEqual(3, len(grouped))

        grouped = self.dv.groupby('abc', squeeze=False)
        self.assertDataArrayAllClose(
            expected_sum_all, grouped.sum(dimension=None))

        expected_sum_axis1 = Dataset(
            {'foo': (['x', 'abc'], np.array([self.x[:, :9].sum(1),
                                             self.x[:, 10:].sum(1),
                                             self.x[:, 9:10].sum(1)]).T),
             'x': self.ds.variables['x'],
             'abc': Variable(['abc'], np.array(['a', 'b', 'c']))})['foo']
        self.assertDataArrayAllClose(expected_sum_axis1,
                                     grouped.reduce(np.sum, 'y'))
        self.assertDataArrayAllClose(expected_sum_axis1, grouped.sum('y'))

        def center(x):
            return x - np.mean(x)

        expected_ds = self.dv.dataset.copy()
        exp_data = np.hstack([center(self.x[:, :9]),
                              center(self.x[:, 9:10]),
                              center(self.x[:, 10:])])
        expected_ds['foo'] = (['x', 'y'], exp_data)
        expected_centered = expected_ds['foo']
        self.assertDataArrayAllClose(expected_centered,
                                     grouped.apply(center))

    def test_concat(self):
        self.ds['bar'] = Variable(['x', 'y'], np.random.randn(10, 20))
        foo = self.ds['foo'].select()
        bar = self.ds['bar'].rename('foo').select()
        # from dataset array:
        self.assertVariableEqual(Variable(['w', 'x', 'y'],
                                          np.array([foo.values, bar.values])),
                                 DataArray.concat([foo, bar], 'w'))
        # from iteration:
        grouped = [g for _, g in foo.groupby('x')]
        stacked = DataArray.concat(grouped, self.ds['x'])
        self.assertDataArrayIdentical(foo.select(), stacked)

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
