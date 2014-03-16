import numpy as np
from copy import deepcopy
from textwrap import dedent

from xray import Dataset, DatasetArray, XArray, align
from . import TestCase, ReturnItem


class TestDatasetArray(TestCase):
    def assertDSArrayEqual(self, ar1, ar2):
        self.assertEqual(ar1.name, ar2.name)
        self.assertDatasetEqual(ar1.dataset, ar2.dataset)

    def assertDSArrayEquiv(self, ar1, ar2):
        self.assertIsInstance(ar1, DatasetArray)
        self.assertIsInstance(ar2, DatasetArray)
        random_name = 'randomly-renamed-variable'
        self.assertDSArrayEqual(ar1.rename(random_name),
                                ar2.rename(random_name))

    def setUp(self):
        self.x = np.random.random((10, 20))
        self.v = XArray(['x', 'y'], self.x)
        self.ds = Dataset({'foo': self.v})
        self.dv = DatasetArray(self.ds, 'foo')

    def test_repr(self):
        v = XArray(['time', 'x'], [[1, 2, 3], [4, 5, 6]], {'foo': 'bar'})
        dataset_array = Dataset({'my_variable': v})['my_variable']
        expected = dedent("""
        <xray.DatasetArray 'my_variable' (time: 2, x: 3)>
        array([[1, 2, 3],
               [4, 5, 6]])
        Attributes:
            foo: bar
        """).strip()
        self.assertEqual(expected, repr(dataset_array))

    def test_properties(self):
        self.assertIs(self.dv.dataset, self.ds)
        self.assertEqual(self.dv.name, 'foo')
        self.assertXArrayEqual(self.dv.variable, self.v)
        self.assertArrayEqual(self.dv.data, self.v.data)
        for attr in ['dimensions', 'dtype', 'shape', 'size', 'ndim',
                     'attributes']:
            self.assertEqual(getattr(self.dv, attr), getattr(self.v, attr))
        self.assertEqual(len(self.dv), len(self.v))
        self.assertXArrayEqual(self.dv, self.v)
        self.assertEqual(list(self.dv.coordinates), list(self.ds.coordinates))
        for k, v in self.dv.coordinates.iteritems():
            self.assertArrayEqual(v, self.ds.coordinates[k])
        with self.assertRaises(AttributeError):
            self.dv.name = 'bar'
        with self.assertRaises(AttributeError):
            self.dv.dataset = self.ds

    def test_items(self):
        # strings pull out dataviews
        self.assertDSArrayEqual(self.dv, self.ds['foo'])
        x = self.dv['x']
        y = self.dv['y']
        self.assertDSArrayEqual(DatasetArray(self.ds, 'x'), x)
        self.assertDSArrayEqual(DatasetArray(self.ds, 'y'), y)
        # integer indexing
        I = ReturnItem()
        for i in [I[:], I[...], I[x.data], I[x.variable], I[x], I[x, y],
                  I[x.data > -1], I[x.variable > -1], I[x > -1],
                  I[x > -1, y > -1]]:
            self.assertXArrayEqual(self.dv, self.dv[i])
        for i in [I[0], I[:, 0], I[:3, :2],
                  I[x.data[:3]], I[x.variable[:3]], I[x[:3]], I[x[:3], y[:4]],
                  I[x.data > 3], I[x.variable > 3], I[x > 3], I[x > 3, y > 3]]:
            self.assertXArrayEqual(self.v[i], self.dv[i])
        # make sure we always keep the array around, even if it's a scalar
        self.assertXArrayEqual(self.dv[0, 0], self.dv.variable[0, 0])
        self.assertEqual(self.dv[0, 0].dataset,
                         Dataset({'foo': self.dv.variable[0, 0]}))

    def test_indexed_by(self):
        self.assertEqual(self.dv[0].dataset, self.ds.indexed_by(x=0))
        self.assertEqual(self.dv[:3, :5].dataset,
                         self.ds.indexed_by(x=slice(3), y=slice(5)))
        self.assertDSArrayEqual(self.dv, self.dv.indexed_by(x=slice(None)))
        self.assertDSArrayEqual(self.dv[:3], self.dv.indexed_by(x=slice(3)))

    def test_labeled_by(self):
        self.ds['x'] = ('x', np.array(list('abcdefghij')))
        self.assertDSArrayEqual(self.dv, self.dv.labeled_by(x=slice(None)))
        self.assertDSArrayEqual(self.dv[1], self.dv.labeled_by(x='b'))
        self.assertDSArrayEqual(self.dv[:3], self.dv.labeled_by(x=slice('c')))

    def test_loc(self):
        self.ds['x'] = ('x', np.array(list('abcdefghij')))
        self.assertDSArrayEqual(self.dv[:3], self.dv.loc[:'c'])
        self.assertDSArrayEqual(self.dv[1], self.dv.loc['b'])
        self.assertDSArrayEqual(self.dv[:3], self.dv.loc[['a', 'b', 'c']])
        self.assertDSArrayEqual(self.dv[:3, :4],
                             self.dv.loc[['a', 'b', 'c'], np.arange(4)])
        self.dv.loc['a':'j'] = 0
        self.assertTrue(np.all(self.dv.data == 0))

    def test_rename(self):
        renamed = self.dv.rename('bar')
        self.assertEqual(renamed.dataset, self.ds.rename({'foo': 'bar'}))
        self.assertEqual(renamed.name, 'bar')

        renamed = self.dv.rename({'foo': 'bar'})
        self.assertEqual(renamed.dataset, self.ds.rename({'foo': 'bar'}))
        self.assertEqual(renamed.name, 'bar')

    def test_dataset_getitem(self):
        dv = self.ds['foo']
        self.assertDSArrayEqual(dv, self.dv)

    def test_array_interface(self):
        self.assertArrayEqual(np.asarray(self.dv), self.x)
        # test patched in methods
        self.assertArrayEqual(self.dv.take([2, 3]), self.v.take([2, 3]))
        self.assertXArrayEqual(self.dv.argsort(), self.v.argsort())
        self.assertXArrayEqual(self.dv.clip(2, 3), self.v.clip(2, 3))
        # test ufuncs
        expected = deepcopy(self.ds)
        expected['foo'][:] = np.sin(self.x)
        self.assertDSArrayEquiv(expected['foo'], np.sin(self.dv))
        self.assertDSArrayEquiv(self.dv, np.maximum(self.v, self.dv))
        bar = XArray(['x', 'y'], np.zeros((10, 20)))
        self.assertDSArrayEquiv(self.dv, np.maximum(self.dv, bar))

    def test_math(self):
        x = self.x
        v = self.v
        a = self.dv
        # variable math was already tested extensively, so let's just make sure
        # that all types are properly converted here
        self.assertDSArrayEquiv(a, +a)
        self.assertDSArrayEquiv(a, a + 0)
        self.assertDSArrayEquiv(a, 0 + a)
        self.assertDSArrayEquiv(a, a + 0 * v)
        self.assertDSArrayEquiv(a, 0 * v + a)
        self.assertDSArrayEquiv(a, a + 0 * x)
        self.assertDSArrayEquiv(a, 0 * x + a)
        self.assertDSArrayEquiv(a, a + 0 * a)
        self.assertDSArrayEquiv(a, 0 * a + a)
        # test different indices
        ds2 = self.ds.replace('x', XArray(['x'], 3 + np.arange(10)))
        b = DatasetArray(ds2, 'foo')
        with self.assertRaisesRegexp(ValueError, 'not aligned'):
            a + b
        with self.assertRaisesRegexp(ValueError, 'not aligned'):
            b + a

    def test_dataset_math(self):
        # verify that mathematical operators keep around the expected variables
        # when doing math with dataset arrays from one or more aligned datasets
        obs = Dataset({'tmin': ('x', np.arange(5)),
                       'tmax': ('x', 10 + np.arange(5)),
                       'x': ('x', 0.5 * np.arange(5))})

        actual = 2 * obs['tmax']
        expected = Dataset({'tmax2': ('x', 2 * (10 + np.arange(5))),
                            'x': obs['x']})['tmax2']
        self.assertDSArrayEquiv(actual, expected)

        actual = obs['tmax'] - obs['tmin']
        expected = Dataset({'trange': ('x', 10 * np.ones(5)),
                            'x': obs['x']})['trange']
        self.assertDSArrayEquiv(actual, expected)

        sim = Dataset({'tmin': ('x', 1 + np.arange(5)),
                       'tmax': ('x', 11 + np.arange(5)),
                       'x': ('x', 0.5 * np.arange(5))})

        actual = sim['tmin'] - obs['tmin']
        expected = Dataset({'error': ('x', np.ones(5)),
                            'x': obs['x']})['error']
        self.assertDSArrayEquiv(actual, expected)

        # in place math shouldn't remove or conflict with other variables
        actual = deepcopy(sim['tmin'])
        actual -= obs['tmin']
        expected = Dataset({'tmin': ('x', np.ones(5)),
                            'tmax': sim['tmax'],
                            'x': sim['x']})['tmin']
        self.assertDSArrayEquiv(actual, expected)

    def test_coord_math(self):
        ds = Dataset({'x': ('x', 1 + np.arange(3))})
        expected = ds.copy()
        expected['x2'] = ('x', np.arange(3))
        actual = ds['x'] - 1
        self.assertDSArrayEquiv(expected['x2'], actual)

    def test_item_math(self):
        self.ds['x'] = ('x', np.array(list('abcdefghij')))
        self.assertXArrayEqual(self.dv + self.dv[0, 0],
                               self.dv + self.dv[0, 0].data)
        new_data = self.x[0][None, :] + self.x[:, 0][:, None]
        self.assertXArrayEqual(self.dv[:, 0] + self.dv[0],
                               XArray(['x', 'y'], new_data))
        self.assertXArrayEqual(self.dv[0] + self.dv[:, 0],
                               XArray(['y', 'x'], new_data.T))

    def test_inplace_math(self):
        x = self.x
        v = self.v
        a = self.dv
        b = a
        b += 1
        self.assertIs(b, a)
        self.assertIs(b.variable, v)
        self.assertIs(b.data, x)
        self.assertIs(b.dataset, self.ds)

    def test_transpose(self):
        self.assertXArrayEqual(self.dv.variable.transpose(),
                               self.dv.transpose())

    def test_squeeze(self):
        self.assertXArrayEqual(self.dv.variable.squeeze(), self.dv.squeeze())

    def test_reduce(self):
        self.assertXArrayEqual(self.dv.reduce(np.mean, 'x'),
                            self.v.reduce(np.mean, 'x'))
        # needs more...
        # should check which extra dimensions are dropped

    def test_groupby_iter(self):
        for ((act_x, act_dv), (exp_x, exp_ds)) in \
                zip(self.dv.groupby('y'), self.ds.groupby('y')):
            self.assertXArrayEqual(exp_x, act_x)
            self.assertDSArrayEqual(DatasetArray(exp_ds, 'foo'), act_dv)
        for ((_, exp_dv), act_dv) in zip(self.dv.groupby('x'), self.dv):
            self.assertDSArrayEqual(exp_dv, act_dv)

    def test_groupby(self):
        agg_var = XArray(['y'], np.array(['a'] * 9 + ['c'] + ['b'] * 10))
        self.dv['abc'] = agg_var
        self.dv['y'] = 20 + 100 * self.ds['y'].variable

        identity = lambda x: x
        for g in ['x', 'y']:
            for shortcut in [False, True]:
                for squeeze in [False, True]:
                    expected = self.dv
                    actual = self.dv.groupby(g, squeeze=squeeze).apply(
                        identity, shortcut=shortcut)
                    self.assertDSArrayEqual(expected, actual)

        grouped = self.dv.groupby('abc')

        expected_sum_all = DatasetArray(Dataset(
            {'foo': XArray(['abc'], np.array([self.x[:, :9].sum(),
                                              self.x[:, 10:].sum(),
                                              self.x[:, 9:10].sum()]).T,
                           {'cell_methods': 'x: y: sum'}),
             'abc': XArray(['abc'], np.array(['a', 'b', 'c']))}), 'foo')
        self.assertDSArrayEqual(expected_sum_all,
                                grouped.reduce(np.sum, dimension=None))
        self.assertDSArrayEqual(expected_sum_all, grouped.sum(dimension=None))

        grouped = self.dv.groupby('abc', squeeze=False)
        self.assertDSArrayEqual(expected_sum_all, grouped.sum(dimension=None))

        expected_sum_axis1 = DatasetArray(Dataset(
            {'foo': XArray(['x', 'abc'], np.array([self.x[:, :9].sum(1),
                                                   self.x[:, 10:].sum(1),
                                                   self.x[:, 9:10].sum(1)]).T,
                           {'cell_methods': 'y: sum'}),
             'x': self.ds.variables['x'],
             'abc': XArray(['abc'], np.array(['a', 'b', 'c']))}), 'foo')
        self.assertDSArrayEqual(expected_sum_axis1, grouped.reduce(np.sum))
        self.assertDSArrayEqual(expected_sum_axis1, grouped.sum())

        self.assertDSArrayEqual(self.dv, grouped.apply(identity))

    def test_concat(self):
        self.ds['bar'] = XArray(['x', 'y'], np.random.randn(10, 20))
        foo = self.ds['foo'].select()
        bar = self.ds['bar'].rename('foo').select()
        # from dataset array:
        self.assertXArrayEqual(XArray(['w', 'x', 'y'],
                                      np.array([foo.data, bar.data])),
                               DatasetArray.concat([foo, bar], 'w'))
        # from xarrays:
        self.assertXArrayEqual(XArray(['w', 'x', 'y'],
                                      np.array([foo.data, bar.data])),
                               DatasetArray.concat([foo.variable,
                                                    bar.variable], 'w'))
        # from iteration:
        stacked = DatasetArray.concat((v for _, v in foo.groupby('x')),
                                          self.ds['x'])
        self.assertDSArrayEqual(foo.select(), stacked)

    def test_align(self):
        self.ds['x'] = ('x', np.array(list('abcdefghij')))
        with self.assertRaises(ValueError):
            self.dv + self.dv[:5]
        dv1, dv2 = align(self.dv, self.dv[:5])
        self.assertDSArrayEqual(dv1, self.dv[:5])
        self.assertDSArrayEqual(dv2, self.dv[:5])

    def test_to_series(self):
        expected = self.dv.to_dataframe()['foo']
        actual = self.dv.to_series()
        self.assertArrayEqual(expected.values, actual.values)
        self.assertArrayEqual(expected.index.values, actual.index.values)
        self.assertEqual('foo', actual.name)
