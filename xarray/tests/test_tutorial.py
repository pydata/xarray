import os
from contextlib import suppress

import pytest

from xarray import DataArray, tutorial

from . import assert_identical, network


@network
class TestLoadDataset:
    @pytest.fixture(autouse=True)
    def setUp(self):
        self.testfile = "tiny"
        self.testfilepath = os.path.expanduser(
            os.sep.join(("~", ".xarray_tutorial_data", self.testfile))
        )
        with suppress(OSError):
            os.remove(f"{self.testfilepath}.nc")
        with suppress(OSError):
            os.remove(f"{self.testfilepath}.md5")

    def test_download_from_github(self):
        ds = tutorial.open_dataset(self.testfile).load()
        tiny = DataArray(range(5), name="tiny").to_dataset()
        assert_identical(ds, tiny)

    def test_download_from_github_load_without_cache(self):
        ds_nocache = tutorial.open_dataset(self.testfile, cache=False).load()
        ds_cache = tutorial.open_dataset(self.testfile).load()
        assert_identical(ds_cache, ds_nocache)
