import numpy as np

from scidata import Dataset, DatasetArray, Array, intersection
from . import TestCase, ReturnItem


class TestDatasetArray(TestCase):
    def assertViewEqual(self, dv1, dv2):
        self.assertEqual(dv1.dataset, dv2.dataset)
        self.assertEqual(dv1.focus, dv2.focus)

    def setUp(self):
        self.x = np.random.random((10, 20))
        self.v = Array(['x', 'y'], self.x)
        self.ds = Dataset({'foo': self.v})
        self.ds.create_coordinate('x', np.arange(10))
        self.ds.create_coordinate('y', np.arange(20))
        self.dv = DatasetArray(self.ds, 'foo')

    def test_properties(self):
        self.assertIs(self.dv.dataset, self.ds)
        self.assertEqual(self.dv.focus, 'foo')
        self.assertVarEqual(self.dv.variable, self.v)
        self.assertNDArrayEqual(self.dv.data, self.v.data)
        for attr in ['dimensions', 'dtype', 'shape', 'size', 'ndim',
                     'attributes']:
            self.assertEqual(getattr(self.dv, attr), getattr(self.v, attr))
        self.assertEqual(len(self.dv), len(self.v))
        self.assertVarEqual(self.dv, self.v)
        self.assertEqual(list(self.dv.indices), list(self.ds.indices))
        for k, v in self.dv.indices.iteritems():
            self.assertNDArrayEqual(v, self.ds.indices[k])

    def test_items(self):
        # strings pull out dataviews
        self.assertViewEqual(self.dv, self.ds['foo'])
        x = self.dv['x']
        y = self.dv['y']
        self.assertViewEqual(DatasetArray(self.ds.select('x'), 'x'), x)
        self.assertViewEqual(DatasetArray(self.ds.select('y'), 'y'), y)
        # integer indexing
        I = ReturnItem()
        for i in [I[:], I[...], I[x.data], I[x.variable], I[x], I[x, y],
                  I[x.data > -1], I[x.variable > -1], I[x > -1],
                  I[x > -1, y > -1]]:
            self.assertVarEqual(self.dv, self.dv[i])
        for i in [I[0], I[:, 0], I[:3, :2],
                  I[x.data[:3]], I[x.variable[:3]], I[x[:3]], I[x[:3], y[:4]],
                  I[x.data > 3], I[x.variable > 3], I[x > 3], I[x > 3, y > 3]]:
            self.assertVarEqual(self.v[i], self.dv[i])
        # check that the new index is consistent
        self.assertEqual(list(self.dv[0].indices), ['y'])
        # we always to keep the dataview variable around
        self.assertVarEqual(self.dv[0, 0], self.dv.variable[0, 0])
        self.assertEqual(self.dv[0, 0].dataset,
                         Dataset({'foo': self.dv.variable[0, 0]}))

    def test_iteration(self):
        for ((act_x, act_dv), (exp_x, exp_ds)) in \
                zip(self.dv.iterator('y'), self.ds.iterator('y')):
            self.assertVarEqual(exp_x, act_x)
            self.assertViewEqual(DatasetArray(exp_ds, 'foo'), act_dv)
        for ((_, exp_dv), act_dv) in zip(self.dv.iterator('x'), self.dv):
            self.assertViewEqual(exp_dv, act_dv)

    def test_indexed_by(self):
        self.assertEqual(self.dv[0].dataset, self.ds.indexed_by(x=0))
        self.assertEqual(self.dv[:3, :5].dataset,
                         self.ds.indexed_by(x=slice(3), y=slice(5)))
        self.assertViewEqual(self.dv, self.dv.indexed_by(x=slice(None)))
        self.assertViewEqual(self.dv[:3], self.dv.indexed_by(x=slice(3)))

    def test_labeled_by(self):
        self.ds.set_variable('x', Array(['x'], np.array(list('abcdefghij'))))
        self.assertViewEqual(self.dv, self.dv.labeled_by(x=slice(None)))
        self.assertViewEqual(self.dv[1], self.dv.labeled_by(x='b'))
        self.assertViewEqual(self.dv[:3], self.dv.labeled_by(x=slice('c')))

    def test_loc(self):
        self.ds.set_variable('x', Array(['x'], np.array(list('abcdefghij'))))
        self.assertViewEqual(self.dv[:3], self.dv.loc[:'c'])
        self.assertViewEqual(self.dv[1], self.dv.loc['b'])
        self.assertViewEqual(self.dv[:3], self.dv.loc[['a', 'b', 'c']])
        self.assertViewEqual(self.dv[:3, :4],
                             self.dv.loc[['a', 'b', 'c'], np.arange(4)])
        self.dv.loc['a':'j'] = 0
        self.assertTrue(np.all(self.dv.data == 0))

    def test_renamed(self):
        renamed = self.dv.renamed('bar')
        self.assertEqual(renamed.dataset, self.ds.renamed({'foo': 'bar'}))
        self.assertEqual(renamed.focus, 'bar')

    def test_refocus(self):
        self.assertVarEqual(self.dv, self.dv.refocus(self.v))
        self.assertVarEqual(self.dv, self.dv.refocus(self.x))

    def test_dataset_getitem(self):
        dv = self.ds['foo']
        self.assertViewEqual(dv, self.dv)

    def test_array_interface(self):
        self.assertNDArrayEqual(np.asarray(self.dv), self.x)
        # test patched in methods
        self.assertNDArrayEqual(self.dv.take([2, 3]), self.x.take([2, 3]))
        self.assertViewEqual(self.dv.argsort(),
                             self.dv.refocus(self.x.argsort()))
        self.assertViewEqual(self.dv.clip(2, 3),
                             self.dv.refocus(self.x.clip(2, 3)))
        # test ufuncs
        self.assertViewEqual(np.sin(self.dv),
                             self.dv.refocus(np.sin(self.x)))
        self.assertViewEqual(self.dv, np.maximum(self.v, self.dv))
        self.ds['bar'] = Array(['x', 'y'], np.zeros((10, 20)))
        self.assertViewEqual(self.dv, np.maximum(self.dv, self.ds['bar']))

    def test_math(self):
        x = self.x
        v = self.v
        a = self.dv
        # variable math was already tested extensively, so let's just make sure
        # that all types are properly converted here
        self.assertViewEqual(a, +a)
        self.assertViewEqual(a, a + 0)
        self.assertViewEqual(a, 0 + a)
        self.assertViewEqual(a, a + 0 * v)
        self.assertViewEqual(a, 0 * v + a)
        self.assertViewEqual(a, a + 0 * x)
        self.assertViewEqual(a, 0 * x + a)
        self.assertViewEqual(a, a + 0 * a)
        self.assertViewEqual(a, 0 * a + a)
        # test different indices
        ds2 = self.ds.replace('x', Array(['x'], 3 + np.arange(10)))
        b = DatasetArray(ds2, 'foo')
        with self.assertRaisesRegexp(ValueError, 'not aligned'):
            a + b
        with self.assertRaisesRegexp(ValueError, 'not aligned'):
            b + a

    def test_item_math(self):
        self.ds.set_variable('x', Array(['x'], np.array(list('abcdefghij'))))
        self.assertVarEqual(self.dv + self.dv[0, 0],
                            self.dv + self.dv[0, 0].data)
        new_data = self.x[0][None, :] + self.x[:, 0][:, None]
        self.assertVarEqual(self.dv[:, 0] + self.dv[0],
                            Array(['x', 'y'], new_data))
        self.assertVarEqual(self.dv[0] + self.dv[:, 0],
                            Array(['y', 'x'], new_data.T))

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

    def test_collapse(self):
        self.assertVarEqual(self.dv.collapse(np.mean, 'x'),
                            self.v.collapse(np.mean, 'x'))
        # needs more...
        # should check which extra dimensions are dropped

    def test_aggregate(self):
        agg_var = Array(['y'], np.array(['a'] * 9 + ['c'] + ['b'] * 7 +
                                           ['c'] * 3))
        self.ds.add_variable('abc', agg_var)
        expected_unique, expected_var = \
            self.dv.variable.aggregate(np.mean, 'abc', agg_var)
        expected = DatasetArray(Dataset(
            {'foo': expected_var, 'x': self.ds.variables['x'],
             'abc': expected_unique}), 'foo')
        actual = self.dv.aggregate(np.mean, 'abc')
        self.assertViewEqual(expected, actual)
        actual = self.dv.aggregate(np.mean, self.ds['abc'])
        self.assertViewEqual(expected, actual)

    def test_from_stack(self):
        self.ds['bar'] = Array(['x', 'y'], np.random.randn(10, 20))
        foo = self.ds['foo']
        bar = self.ds['bar'].renamed('foo')
        # from dataviews:
        self.assertVarEqual(Array(['w', 'x', 'y'],
                                     np.array([foo.data, bar.data])),
                            DatasetArray.from_stack([foo, bar], 'w'))
        # from variables:
        self.assertVarEqual(Array(['w', 'x', 'y'],
                                     np.array([foo.data, bar.data])),
                            DatasetArray.from_stack([foo.variable,
                                                     bar.variable], 'w'))
        # from iteration:
        stacked = DatasetArray.from_stack((v for _, v in foo.iterator('x')),
                                      self.ds['x'])
        self.assertViewEqual(foo, stacked)

    def test_intersection(self):
        self.ds.set_variable('x', Array(['x'], np.array(list('abcdefghij'))))
        with self.assertRaises(ValueError):
            self.dv + self.dv[:5]
        dv1, dv2 = intersection(self.dv, self.dv[:5])
        self.assertViewEqual(dv1, self.dv[:5])
        self.assertViewEqual(dv2, self.dv[:5])
