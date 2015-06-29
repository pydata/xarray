import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from xray import (Dataset, DataArray)

from . import TestCase


class TestDataArray(TestCase):

    def setUp(self):
        d = [0, 1, 0, 2]
        self.darray = DataArray(d, coords={'period': range(len(d))})

    def tearDown(self):
        # Remove all matplotlib figures
        plt.close('all')

    def test_plot_exists_and_callable(self):
        self.assertTrue(callable(self.darray.plot))

    def test_xlabel_is_coordinate_name(self):
        self.darray.plot()
        xlabel = plt.gca().get_xlabel()
        self.assertEqual(xlabel, 'period')
