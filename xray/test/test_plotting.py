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

    @requires_matplotlib
    def test_can_pass_in_axis(self):
        # TODO
        pass


class TestPlot(PlotTestCase):

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


class TestPlotLine(PlotTestCase):

    def setUp(self):
        d = [0, 1, 0, 2]
        self.darray = DataArray(d, coords={'period': range(len(d))})

    @requires_matplotlib
    def test_wrong_dims_raises_valueerror(self):
        # TODO
        pass


class TestPlotContourf(PlotTestCase):

    def setUp(self):
        self.darray = DataArray(np.random.randn(10, 15), 
                dims=['y', 'x'])

    @requires_matplotlib
    def test_label_names(self):
        self.darray.plot_contourf()
        xlabel = plt.gca().get_xlabel()
        ylabel = plt.gca().get_ylabel()
        self.assertEqual(xlabel, 'x')
        self.assertEqual(ylabel, 'y')
