import numpy as np
import pandas as pd

from xarray import (align, broadcast, Dataset, DataArray, Variable)

from xarray.test import (TestCase, unittest)

try:
    import quantities as pq
    has_quantities = True
except ImportError:
    has_quantities = False

def requires_quantities(test):
    return test if has_quantities else unittest.skip('requires dask')(test)

@requires_quantities
class TestWithQuantities(TestCase):
    def setUp(self):
        self.x = np.arange(10) * pq.A
        self.y = np.arange(20)
        self.xp = np.arange(10) * pq.J
        self.v = np.arange(10*20).reshape(10,20) * pq.V
        self.da = DataArray(self.v,dims=['x','y'],coords=dict(x=self.x,y=self.y,xp=(['x'],self.xp)))

    def assertEqualWUnits(self,a,b):
        self.assertIsNotNone(getattr(a, 'units', None))
        self.assertIsNotNone(getattr(b, 'units', None))
        self.assertEqual(a.units,b.units)
        np.testing.assert_allclose(a.magnitude,b.magnitude)

    def test_units_in_data_and_coords(self):
        da = self.da
        self.assertEqualWUnits(da.xp.data,self.xp)
        self.assertEqualWUnits(da.data,self.v)

    def test_arithmetics(self):
        da = self.da
        f = DataArray(np.arange(10*20).reshape(10,20)*pq.A,dims=['x','y'],coords=dict(x=self.x,y=self.y))
        self.assertEqualWUnits((da*f).data, da.data*f.data)

    def test_unit_checking(self):
        da = self.da
        f = DataArray(np.arange(10*20).reshape(10,20)*pq.A,dims=['x','y'],coords=dict(x=self.x,y=self.y))
        with self.assertRaisesRegex(ValueError,'Unable to convert between units'):
            da + f

    @unittest.expectedFailure
    def test_units_in_indexes(self):
        """
        Indexes are borrowed from Pandas, and Pandas does not support units.
        Therefore, we currently don't intend to support units on indexes either.
        """
        da = self.da
        self.assertEqualWUnits(da.x.data,self.x)

    @unittest.expectedFailure
    def test_sel(self):
        self.assertEqualWUnits(self.da.sel(y=self.y[0]).values,self.v[:,0])

    @unittest.expectedFailure
    def test_mean(self):
        self.assertEqualWUnits(self.da.mean('x').values,self.v.mean(0))

