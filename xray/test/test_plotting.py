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

# TODO - Every test in this file requires matplotlib
# Hence it's redundant to have to use the decorator on every test
# How to refactor?

class PlotTestCase(TestCase):

    def tearDown(self):
        # Remove all matplotlib figures
        plt.close('all')


class TestPlot(PlotTestCase):

    def setUp(self):
        d = np.arange(24).reshape(2, 3, 4)
        self.darray = DataArray(d)

    @requires_matplotlib
    def test3d(self):
        self.darray[0, 0, :].plot()

    @requires_matplotlib
    def test2d(self):
        self.darray[0, :, :].plot()

    @requires_matplotlib
    def test3d(self):
        self.darray.plot()


class TestPlot1D(PlotTestCase):

    def setUp(self):
        d = [0, 1, 0, 2]
        self.darray = DataArray(d, coords={'period': range(len(d))})

    @requires_matplotlib
    def test_xlabel_is_index_name(self):
        self.darray.plot()
        xlabel = plt.gca().get_xlabel()
        self.assertEqual(xlabel, 'period')

    @requires_matplotlib
    def test_ylabel_is_data_name(self):
        self.darray.name = 'temperature'
        self.darray.plot()
        ylabel = plt.gca().get_ylabel()
        self.assertEqual(ylabel, self.darray.name)

    @requires_matplotlib
    def test_can_pass_in_axis(self):
        # TODO - add this test to for other plotting methods
        fig, axes = plt.subplots(ncols=2)
        self.darray.plot(axes[0])
        self.assertTrue(axes[0].has_data())

    @requires_matplotlib
    def test_wrong_dims_raises_valueerror(self):
        twodims = DataArray(np.arange(10).reshape(2, 5))
        with self.assertRaises(ValueError):
            twodims.plot_line()


class TestPlot2D(PlotTestCase):

    def setUp(self):
        self.darray = DataArray(np.random.randn(10, 15), 
                dims=['y', 'x'])

    @requires_matplotlib
    def test_contour_label_names(self):
        self.darray.plot_contourf()
        xlabel = plt.gca().get_xlabel()
        ylabel = plt.gca().get_ylabel()
        self.assertEqual(xlabel, 'x')
        self.assertEqual(ylabel, 'y')

    @requires_matplotlib
    def test_imshow_label_names(self):
        self.darray.plot_imshow()
        xlabel = plt.gca().get_xlabel()
        ylabel = plt.gca().get_ylabel()
        self.assertEqual(xlabel, 'x')
        self.assertEqual(ylabel, 'y')

    @requires_matplotlib
    def test_too_few_dims_raises_valueerror(self):
        with self.assertRaisesRegexp(ValueError, r'[Dd]im'):
            self.darray[0, :].plot_imshow()

    @requires_matplotlib
    def test_too_many_dims_raises_valueerror(self):
        da = DataArray(np.random.randn(2, 3, 4))
        with self.assertRaisesRegexp(ValueError, r'[Dd]im'):
            da.plot_imshow()


class TestPlotHist(PlotTestCase):

    def setUp(self):
        self.darray = DataArray(np.random.randn(2, 3, 4))

    @requires_matplotlib
    def test_3d_array(self):
        self.darray.plot_hist()

    @requires_matplotlib
    def test_title_uses_name(self):
        nm = 'randompoints'
        self.darray.name = nm
        self.darray.plot_hist()
        title = plt.gca().get_title()
        self.assertIn(nm, title)

    @requires_matplotlib
    def test_ylabel_is_count(self):
        self.darray.plot_hist()
        ylabel = plt.gca().get_ylabel()
        self.assertEqual(ylabel, 'Count')
