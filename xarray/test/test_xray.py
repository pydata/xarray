from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from . import TestCase

import xarray as xr


class TestXray(TestCase):
    def test_import(self):
        with self.assertWarns('xray has been renamed'):
            import xray

        assert xray.Dataset is xr.Dataset
        assert xray.__version__ == xr.__version__
