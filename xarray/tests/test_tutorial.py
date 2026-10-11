from __future__ import annotations

import gc
import os
import pathlib
import sys

import pytest

from xarray import DataArray, DataTree, tutorial
from xarray.testing import assert_identical
from xarray.tests import (
    network,
    requires_h5netcdf_or_netCDF4,
    requires_pooch,
    requires_scipy_or_netCDF4,
)

TINY = DataArray(range(5), name="tiny").to_dataset()


@network
class TestLoadDataset:
    def test_download_from_github(self, tmp_path) -> None:
        cache_dir = tmp_path / tutorial._default_cache_dir_name
        ds = tutorial.load_dataset("tiny", cache_dir=cache_dir)
        tiny = DataArray(range(5), name="tiny").to_dataset()
        assert_identical(ds, tiny)

    def test_download_from_github_load_without_cache(self, tmp_path) -> None:
        cache_dir = tmp_path / tutorial._default_cache_dir_name
        ds_nocache = tutorial.load_dataset("tiny", cache=False, cache_dir=cache_dir)
        ds_cache = tutorial.load_dataset("tiny", cache_dir=cache_dir)
        assert_identical(ds_cache, ds_nocache)


@network
class TestLoadDataTree:
    def test_download_from_github(self, tmp_path) -> None:
        cache_dir = tmp_path / tutorial._default_cache_dir_name
        ds = tutorial.load_datatree("tiny", cache_dir=cache_dir)
        tiny = DataTree.from_dict({"/": DataArray(range(5), name="tiny").to_dataset()})
        assert_identical(ds, tiny)

    def test_download_from_github_load_without_cache(self, tmp_path) -> None:
        cache_dir = tmp_path / tutorial._default_cache_dir_name
        ds_nocache = tutorial.load_datatree("tiny", cache=False, cache_dir=cache_dir)
        ds_cache = tutorial.load_datatree("tiny", cache_dir=cache_dir)
        assert_identical(ds_cache, ds_nocache)


@pytest.fixture
def fake_retrieve(monkeypatch):
    """Replace the download by writing the tiny dataset to the cache directory."""
    import pooch

    calls = []

    def retrieve(url, known_hash, path, downloader):
        calls.append({"url": url, "path": path, "downloader": downloader})
        filepath = pathlib.Path(path) / url.rsplit("/", 1)[-1]
        filepath.parent.mkdir(parents=True, exist_ok=True)
        if not filepath.exists():
            DataTree(TINY).to_netcdf(filepath)
        return os.fspath(filepath)

    monkeypatch.setattr(pooch, "retrieve", retrieve)
    return calls


@requires_pooch
@requires_h5netcdf_or_netCDF4
class TestOffline:
    @pytest.mark.parametrize("func", [tutorial.load_dataset, tutorial.open_dataset])
    def test_open_dataset(self, func, fake_retrieve, tmp_path) -> None:
        with func("tiny", cache_dir=tmp_path) as ds:
            assert_identical(ds, TINY)
        (call,) = fake_retrieve
        assert call["url"] == f"{tutorial.base_url}/raw/{tutorial.version}/tiny.nc"
        assert call["path"] == os.fspath(tmp_path)
        user_agent = call["downloader"].kwargs["headers"]["User-Agent"]
        assert user_agent.startswith("xarray ")
        assert (tmp_path / "tiny.nc").exists()

    @pytest.mark.parametrize("func", [tutorial.load_datatree, tutorial.open_datatree])
    def test_open_datatree(self, func, fake_retrieve, tmp_path) -> None:
        with func("tiny", cache_dir=tmp_path) as tree:
            assert_identical(tree, DataTree(TINY))
        (call,) = fake_retrieve
        assert call["url"].endswith("/tiny.nc")

    @pytest.mark.parametrize("func", [tutorial.open_dataset, tutorial.open_datatree])
    # the file has to be closed before it is removed, which fails on Windows
    # otherwise and warns about deallocating an open file elsewhere
    @pytest.mark.filterwarnings("error::pytest.PytestUnraisableExceptionWarning")
    def test_without_cache(self, func, fake_retrieve, tmp_path) -> None:
        obj = func("tiny", cache=False, cache_dir=tmp_path)
        assert not (tmp_path / "tiny.nc").exists()
        assert obj["tiny"].values.tolist() == [0, 1, 2, 3, 4]
        del obj
        gc.collect()

    def test_external_url(self, fake_retrieve, tmp_path, monkeypatch) -> None:
        url = "https://example.com/data/tiny_external.nc"
        monkeypatch.setitem(tutorial.external_urls, "external", url)
        tutorial.load_dataset("external", cache_dir=tmp_path)
        assert fake_retrieve[0]["url"] == url

    def test_name_with_suffix(self, fake_retrieve, tmp_path) -> None:
        tutorial.load_dataset("tiny.nc", cache_dir=tmp_path)
        assert fake_retrieve[0]["url"].endswith("/tiny.nc")

    @pytest.mark.parametrize("func", [tutorial.open_dataset, tutorial.open_datatree])
    def test_grib_requires_cfgrib(self, func, monkeypatch, tmp_path) -> None:
        monkeypatch.setitem(sys.modules, "cfgrib", None)
        with pytest.raises(ImportError, match="requires the cfgrib package"):
            func("era5-2mt-2019-03-uk.grib", cache_dir=tmp_path)

    def test_default_cache_dir(self, fake_retrieve, monkeypatch, tmp_path) -> None:
        import pooch

        monkeypatch.setattr(pooch, "os_cache", lambda name: tmp_path / name)
        tutorial.load_dataset("tiny")
        assert fake_retrieve[0]["path"] == tmp_path / tutorial._default_cache_dir_name


@pytest.mark.parametrize(
    "func",
    [
        tutorial.open_dataset,
        tutorial.load_dataset,
        tutorial.open_datatree,
        tutorial.load_datatree,
    ],
)
def test_requires_pooch(func, monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "pooch", None)
    with pytest.raises(ImportError, match="depends on pooch"):
        func("tiny")


@pytest.mark.parametrize(
    ("name", "missing", "match"),
    [
        ("tiny", ["scipy", "netCDF4"], "requires either scipy or netCDF4"),
        ("basin_mask", ["h5netcdf", "netCDF4"], "requires either h5netcdf or netCDF4"),
    ],
)
def test_check_netcdf_engine_installed(name, missing, match, monkeypatch) -> None:
    for module in missing:
        monkeypatch.setitem(sys.modules, module, None)
    with pytest.raises(ImportError, match=match):
        tutorial._check_netcdf_engine_installed(name)


@requires_scipy_or_netCDF4
def test_check_netcdf_engine_installed_passes() -> None:
    tutorial._check_netcdf_engine_installed("tiny")
    # unknown names are not checked
    tutorial._check_netcdf_engine_installed("unknown")


def test_scatter_example_dataset() -> None:
    ds = tutorial.scatter_example_dataset(seed=42)
    assert set(ds.data_vars) == {"A", "B"}
    assert dict(ds.sizes) == {"x": 3, "y": 11, "z": 4, "w": 4}
    assert ds.A.attrs["units"] == "Aunits"
    assert_identical(ds, tutorial.scatter_example_dataset(seed=42))
