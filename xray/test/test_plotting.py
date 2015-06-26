import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from xray import (Dataset, DataArray)

from . import TestCase


class PlotTestCase(TestCase):

    def setUp(self):
        d = [0, 1, 0, 2]
        self.dv = DataArray(d, coords={'period': range(len(d))})

    def tearDown(self):
        # Remove all matplotlib figures
        pass


class TestBasics(PlotTestCase):

    # Not sure how to test this
    def test_matplotlib_not_imported(self):
        # Doesn't work. Keeping so I remember to change it.
        #self.assertFalse('matplotlib' in sys.modules)
        pass


class TestDataArray(PlotTestCase):

    def test_plot_exists_and_callable(self):
        self.assertTrue(callable(self.dv.plot))
