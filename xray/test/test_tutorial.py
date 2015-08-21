import os
import unittest

from xray import tutorial

from . import TestCase


@unittest.skip('TODO: make this conditional on network availability')
class Test_load_dataset(TestCase):

    def setUp(self):
        self.testfile = 'tiny.nc'
        self.testfilepath = os.path.expanduser(os.sep.join(('~',
                '.xray_tutorial_data', self.testfile)))
        try:
            os.remove(self.testfilepath)
        except OSError:
            pass

    def test_download_from_github(self):
        ds = tutorial.load_dataset(self.testfile)
        tiny = xray.DataArray(range(5), name='tiny').to_dataset()
        self.assertDatasetIdentical(ds, tiny)
