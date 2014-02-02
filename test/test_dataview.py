import numpy as np

from scidata import Dataset, DataView, Variable
from . import TestCase


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

    def test_items(self):
        self.assertVarEqual(self.dv[0], self.v[0])
        self.assertEqual(self.dv[0].dataset, self.ds.views({'x': 0}))
        self.assertVarEqual(self.dv[:3, :5], self.v[:3, :5])
        self.assertEqual(self.dv[:3, :5].dataset,
                         self.ds.views({'x': slice(3), 'y': slice(5)}))

    def test_renamed(self):
        renamed = self.dv.renamed('bar')
        self.assertEqual(renamed.dataset, self.ds.renamed({'foo': 'bar'}))
        self.assertEqual(renamed.name, 'bar')

    def test_to_dataview(self):
        dv = self.ds.to_dataview('foo')
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

    def test_inplace_math(self):
        x = self.x
        v = self.v
        a = self.dv
        b = a
        b += 1
        self.assertIs(b, a)
        self.assertIs(b.variable, v)
        self.assertIs(b.data, x)
        #FIXME: this test currently fails (see DataView.variable.setter)
        # self.assertIs(b.dataset, self.ds)
