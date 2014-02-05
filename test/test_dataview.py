import numpy as np

from scidata import Dataset, DataView, Variable
from . import TestCase, ReturnItem


class TestDataView(TestCase):
    def assertViewEqual(self, dv1, dv2):
        self.assertEqual(dv1.dataset, dv2.dataset)
        self.assertEqual(dv1.name, dv2.name)

    def setUp(self):
        self.x = np.random.random((10, 20))
        self.v = Variable(['x', 'y'], self.x)
        self.ds = Dataset({'foo': self.v})
        self.ds.create_coordinate('x', np.arange(10))
        self.ds.create_coordinate('y', np.arange(20))
        self.dv = DataView(self.ds, 'foo')

    def test_properties(self):
        self.assertIs(self.dv.dataset, self.ds)
        self.assertEqual(self.dv.name, 'foo')
        self.assertVarEqual(self.dv.variable, self.v)
        self.assertArrayEqual(self.dv.data, self.v.data)
        for attr in ['dimensions', 'dtype', 'shape', 'size', 'ndim',
                     'attributes']:
            self.assertEqual(getattr(self.dv, attr), getattr(self.v, attr))
        self.assertEqual(len(self.dv), len(self.v))
        self.assertVarEqual(self.dv, self.v)
        self.assertEqual(list(self.dv.indices), list(self.ds.indices))
        for k, v in self.dv.indices.iteritems():
            self.assertArrayEqual(v, self.ds.indices[k])

    def test_items(self):
        # test indexing
        x = self.ds['x']
        y = self.ds['y']
        I = ReturnItem()
        for i in [I[:], I[...], I[x.data], I[x.variable], I[x], I[x, y],
                  I[x.data > -1], I[x.variable > -1], I[x > -1],
                  I[x > -1, y > -1]]:
            self.assertVarEqual(self.dv, self.dv[i])
        for i in [I[0], I[:, 0], I[:3, :2],
                  I[x.data[:3]], I[x.variable[:3]], I[x[:3]], I[x[:3], y[:4]],
                  I[x.data > 3], I[x.variable > 3], I[x > 3], I[x > 3, y > 3]]:
            self.assertVarEqual(self.v[i], self.dv[i])
        # test index
        self.assertEqual(list(self.dv[0].indices), ['y'])
        # test matches views
        self.assertEqual(self.dv[0].dataset, self.ds.views(x=0))
        self.assertEqual(self.dv[:3, :5].dataset,
                         self.ds.views(x=slice(3), y=slice(5)))

    def test_iteration(self):
        for ((act_x, act_dv), (exp_x, exp_ds)) in \
                zip(self.dv.iterator('y'), self.ds.iterator('y')):
            self.assertVarEqual(exp_x, act_x)
            self.assertViewEqual(DataView(exp_ds, 'foo'), act_dv)
        for ((_, exp_dv), act_dv) in zip(self.dv.iterator('x'), self.dv):
            self.assertViewEqual(exp_dv, act_dv)

    def test_views(self):
        self.assertViewEqual(self.dv, self.dv.views(x=slice(None)))
        self.assertViewEqual(self.dv[:3], self.dv.views(x=slice(3)))

    def test_renamed(self):
        renamed = self.dv.renamed('bar')
        self.assertEqual(renamed.dataset, self.ds.renamed({'foo': 'bar'}))
        self.assertEqual(renamed.name, 'bar')

    def test_dataset_getitem(self):
        dv = self.ds['foo']
        self.assertViewEqual(dv, self.dv)

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
        ds2 = self.ds.replace('x', Variable(['x'], 3 + np.arange(10)))
        b = DataView(ds2, 'foo')
        with self.assertRaisesRegexp(ValueError, 'not aligned'):
            a + b
        with self.assertRaisesRegexp(ValueError, 'not aligned'):
            b + a

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

    def test_collapsed(self):
        self.assertVarEqual(self.dv.collapsed(np.mean, 'x'),
                            self.v.collapsed(np.mean, 'x'))
        # needs more...
        # should check which extra dimensions are dropped

    def test_aggregated_by(self):
        agg_var = Variable(['y'], np.array(['a'] * 9 + ['c'] + ['b'] * 7 +
                                           ['c'] * 3))
        self.ds.add_variable('abc', agg_var)
        expected_unique, expected_var = \
            self.dv.variable.aggregated_by(np.mean, 'abc', agg_var)
        expected = DataView(Dataset(
            {'foo': expected_var, 'x': self.ds.variables['x'],
             'abc': expected_unique}), 'foo')
        actual = self.dv.aggregated_by(np.mean, 'abc')
        self.assertViewEqual(expected, actual)
