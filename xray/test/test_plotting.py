import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from xray import (Dataset, DataArray)

from . import TestCase


class PlotTestCase(TestCase):

    def tearDown(self):
        # Remove all matplotlib figures
        plt.close('all')


class TestSimpleDataArray(PlotTestCase):

    def setUp(self):
        d = [0, 1, 0, 2]
        self.darray = DataArray(d, coords={'period': range(len(d))})

    def test_xlabel_is_index_name(self):
        self.darray.plot()
        xlabel = plt.gca().get_xlabel()
        self.assertEqual(xlabel, 'period')

    def test_ylabel_is_data_name(self):
        self.darray.name = 'temperature'
        self.darray.plot()
        ylabel = plt.gca().get_ylabel()
        self.assertEqual(ylabel, self.darray.name)


class Test2dDataArray(PlotTestCase):

    def setUp(self):
        self.darray = DataArray(np.random.randn(10, 15), 
                dims=['long', 'lat'])

    def test_label_names(self):
        self.darray.plot_contourf()
        xlabel = plt.gca().get_xlabel()
        ylabel = plt.gca().get_ylabel()
        self.assertEqual(xlabel, 'long')
        self.assertEqual(ylabel, 'lat')
