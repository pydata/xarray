from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pytest

from xarray import tutorial, DataArray
from xarray.core.pycompat import suppress

from . import TestCase, network


@network
class TestLoadDataset(TestCase):

    def setUp(self):
        self.testfile = 'tiny'
        self.testfilepath = os.path.expanduser(os.sep.join(
            ('~', '.xarray_tutorial_data', self.testfile)))
        with suppress(OSError):
            os.remove('{}.nc'.format(self.testfilepath))
        with suppress(OSError):
            os.remove('{}.md5'.format(self.testfilepath))

    def test_download_from_github(self):
        ds = tutorial.load_dataset(self.testfile)
        tiny = DataArray(range(5), name='tiny').to_dataset()
        self.assertDatasetIdentical(ds, tiny)
