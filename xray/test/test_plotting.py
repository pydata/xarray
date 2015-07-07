import sys

import numpy as np
import pandas as pd

from xray import Dataset, DataArray

from . import TestCase, requires_matplotlib

try:
    import matplotlib
    # Using a different backend makes Travis CI work.
    matplotlib.use('Agg')
    # Order of imports is important here.
    import matplotlib.pyplot as plt
except ImportError:
    pass


@requires_matplotlib
class PlotTestCase(TestCase):

    def tearDown(self):
        # Remove all matplotlib figures
        plt.close('all')


class TestPlot(PlotTestCase):

    def setUp(self):
        d = np.arange(24).reshape(2, 3, 4)
        self.darray = DataArray(d)

    def test1d(self):
        self.darray[0, 0, :].plot()

    def test2d(self):
        self.darray[0, :, :].plot()

    def test3d(self):
        self.darray.plot()


class TestPlot1D(PlotTestCase):

    def setUp(self):
        d = [0, 1, 0, 2]
        self.darray = DataArray(d, coords={'period': range(len(d))})

    def test_xlabel_is_index_name(self):
        ax = self.darray.plot()
        self.assertEqual('period', ax.get_xlabel())

    def test_ylabel_is_data_name(self):
        self.darray.name = 'temperature'
        ax = self.darray.plot()
        self.assertEqual(self.darray.name, ax.get_ylabel())

    def test_can_pass_in_axis(self):
        # TODO - add this test to for other plotting methods
        fig, axes = plt.subplots(ncols=2)
        self.darray.plot(axes[0])
        self.assertTrue(axes[0].has_data())

    def test_wrong_dims_raises_valueerror(self):
        twodims = DataArray(np.arange(10).reshape(2, 5))
        with self.assertRaises(ValueError):
            twodims.plot_line()


class TestPlot2D(PlotTestCase):

    def setUp(self):
        self.darray = DataArray(np.random.randn(10, 15), 
                dims=['y', 'x'])

    def test_contour_label_names(self):
        ax = self.darray.plot_contourf()
        self.assertEqual('x', ax.get_xlabel())
        self.assertEqual('y', ax.get_ylabel())

    def test_imshow_label_names(self):
        ax = self.darray.plot_imshow()
        self.assertEqual('x', ax.get_xlabel())
        self.assertEqual('y', ax.get_ylabel())

    def test_1d_raises_valueerror(self):
        with self.assertRaisesRegexp(ValueError, r'[Dd]im'):
            self.darray[0, :].plot_imshow()

    def test_3d_raises_valueerror(self):
        da = DataArray(np.random.randn(2, 3, 4))
        with self.assertRaisesRegexp(ValueError, r'[Dd]im'):
            da.plot_imshow()


class TestPlotHist(PlotTestCase):

    def setUp(self):
        self.darray = DataArray(np.random.randn(2, 3, 4))

    def test_3d_array(self):
        self.darray.plot_hist()

    def test_title_no_name(self):
        ax = self.darray.plot_hist()
        self.assertEqual('', ax.get_title())

    def test_title_uses_name(self):
        self.darray.name = 'randompoints'
        ax = self.darray.plot_hist()
        self.assertIn(self.darray.name, ax.get_title())

    def test_ylabel_is_count(self):
        ax = self.darray.plot_hist()
        self.assertEqual('Count', ax.get_ylabel())

    def test_can_pass_in_kwargs(self):
        nbins = 5
        ax = self.darray.plot_hist(bins=nbins)
        self.assertEqual(nbins, len(ax.patches))
