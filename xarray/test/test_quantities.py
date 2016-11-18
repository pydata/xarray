""" test_quantities: Test storing and using instances of
:py:class:`quantities.Quantity` inside :py:class`xarray.DataArray`.

It can be considered a stand-in for other :py:class:`numpy.ndarray`
subclasses, particularly other units implementations such as
astropy's. As preservation of subclasses is not a guaranteed feature of
`xarray`, some operations will discard subclasses. This test
also serves as documnetation which operations do preserve subclasses
and which don't.
"""

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
    return (
        test if has_quantities else
        unittest.skip('requires python-quantities')(test)
    )


@requires_quantities
class TestWithQuantities(TestCase):
    def setUp(self):
        self.x = np.arange(10) * pq.A
        self.y = np.arange(20)
        self.xp = np.arange(10) * pq.J
        self.v = np.arange(10 * 20).reshape(10, 20) * pq.V
        self.da = DataArray(self.v, dims=['x', 'y'],
                            coords=dict(x=self.x, y=self.y, xp=(['x'], self.xp)))

    def assertEqualWUnits(self, a, b):
        # DataArray's are supposed to preserve Quantity instances
        # but they (currently?) do not expose their behaviour.
        # We thus need to extract the contained subarray via .data
        if isinstance(a, DataArray):
            a = a.data
        if isinstance(b, DataArray):
            b = b.data
        self.assertIsNotNone(getattr(a, 'units', None))
        self.assertIsNotNone(getattr(b, 'units', None))
        self.assertEqual(a.units, b.units)
        np.testing.assert_allclose(a.magnitude, b.magnitude)

    def test_units_in_data_and_coords(self):
        da = self.da
        self.assertEqualWUnits(da.xp.data, self.xp)
        self.assertEqualWUnits(da.data, self.v)

    def test_arithmetics(self):
        x = self.x
        y = self.y
        v = self.v
        da = self.da

        f = np.arange(10 * 20).reshape(10, 20) * pq.A
        g = DataArray(f, dims=['x', 'y'], coords=dict(x=x, y=y))
        self.assertEqualWUnits(da * g, v * f)

        # swapped dimension order
        f = np.arange(20 * 10).reshape(20, 10) * pq.V
        g = DataArray(f, dims=['y', 'x'], coords=dict(x=x, y=y))
        self.assertEqualWUnits(da + g, v + f.T)

        # broadcasting
        f = np.arange(10) * pq.m
        g = DataArray(f, dims=['x'], coords=dict(x=x))
        self.assertEqualWUnits(da / g, v / f[:,None])

    def test_unit_checking(self):
        da = self.da
        f = np.arange(10 * 20).reshape(10, 20) * pq.A
        g = DataArray(f, dims=['x', 'y'], coords=dict(x=self.x, y=self.y))
        with self.assertRaisesRegexp(ValueError,
                                     'Unable to convert between units'):
            da + g

    @unittest.expectedFailure
    def test_units_in_indexes(self):
        """ Test if units survive through xarray indexes.

        Indexes are borrowed from Pandas, and Pandas does not support units.
        Therefore, we currently don't intend to support units on indexes either.
        """
        da = self.da
        self.assertEqualWUnits(da.x, self.x)

    @unittest.expectedFailure
    def test_sel(self):
        self.assertEqualWUnits(self.da.sel(y=self.y[0]), self.v[:, 0])

    @unittest.expectedFailure
    def test_mean(self):
        self.assertEqualWUnits(self.da.mean('x'), self.v.mean(0))
