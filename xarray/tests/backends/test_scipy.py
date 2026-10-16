from __future__ import annotations

import contextlib
import gzip
import warnings
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest

import xarray as xr
from xarray import Dataset, backends, open_dataset
from xarray.backends.file_manager import _close_unless_pinned
from xarray.backends.scipy_ import ScipyArrayWrapper, ScipyBackendEntrypoint
from xarray.core import indexing
from xarray.core.indexes import PandasIndex
from xarray.tests import (
    assert_allclose,
    assert_identical,
    requires_netCDF4,
    requires_scipy,
)
from xarray.tests.backends.base import (
    CFEncodedBase,
    FileObjectNetCDF,
    InMemoryNetCDF,
    NetCDF3Only,
    _check_guess_can_open_and_open,
    create_tmp_file,
    open_example_dataset,
)
from xarray.tests.test_dataset import create_test_data

with contextlib.suppress(ImportError):
    import netCDF4 as nc4

if TYPE_CHECKING:
    from xarray.backends.api import T_NetcdfEngine


@requires_scipy
class TestScipyInMemoryData(CFEncodedBase, NetCDF3Only, InMemoryNetCDF):
    engine: T_NetcdfEngine = "scipy"

    @contextlib.contextmanager
    def create_store(self):
        fobj = BytesIO()
        yield backends.ScipyDataStore(fobj, "w")

    @contextlib.contextmanager
    def roundtrip(
        self, data, save_kwargs=None, open_kwargs=None, allow_cleanup_failure=False
    ):
        if save_kwargs is None:
            save_kwargs = {}
        if open_kwargs is None:
            open_kwargs = {}
        saved = self.save(data, path=None, **save_kwargs)
        with self.open(saved, **open_kwargs) as ds:
            yield ds

    @pytest.mark.asyncio
    @pytest.mark.skip(reason="NetCDF backends don't support async loading")
    async def test_load_async(self) -> None:
        await super().test_load_async()


@requires_scipy
class TestScipyFileObject(CFEncodedBase, NetCDF3Only, FileObjectNetCDF):
    # TODO: Consider consolidating some of these cases (e.g.,
    # test_file_remains_open) with TestH5NetCDFFileObject
    engine: T_NetcdfEngine = "scipy"

    @contextlib.contextmanager
    def create_store(self):
        fobj = BytesIO()
        yield backends.ScipyDataStore(fobj, "w")

    @contextlib.contextmanager
    def roundtrip(
        self, data, save_kwargs=None, open_kwargs=None, allow_cleanup_failure=False
    ):
        if save_kwargs is None:
            save_kwargs = {}
        if open_kwargs is None:
            open_kwargs = {}
        with create_tmp_file() as tmp_file:
            with open(tmp_file, "wb") as f:
                self.save(data, f, **save_kwargs)
            with open(tmp_file, "rb") as f:
                with self.open(f, **open_kwargs) as ds:
                    yield ds

    @pytest.mark.asyncio
    @pytest.mark.skip(reason="NetCDF backends don't support async loading")
    async def test_load_async(self) -> None:
        await super().test_load_async()

    @pytest.mark.skip(reason="cannot pickle file objects")
    def test_pickle(self) -> None:
        super().test_pickle()

    @pytest.mark.skip(reason="cannot pickle file objects")
    def test_pickle_dataarray(self) -> None:
        super().test_pickle_dataarray()

    @pytest.mark.parametrize("create_default_indexes", [True, False])
    def test_create_default_indexes(self, tmp_path, create_default_indexes) -> None:
        store_path = tmp_path / "tmp.nc"
        original_ds = xr.Dataset(
            {"data": ("x", np.arange(3))}, coords={"x": [-1, 0, 1]}
        )
        original_ds.to_netcdf(store_path, engine=self.engine, mode="w")

        with open_dataset(
            store_path,
            engine=self.engine,
            create_default_indexes=create_default_indexes,
        ) as loaded_ds:
            if create_default_indexes:
                assert list(loaded_ds.xindexes) == ["x"] and isinstance(
                    loaded_ds.xindexes["x"], PandasIndex
                )
            else:
                assert len(loaded_ds.xindexes) == 0

    def test_indexing_multiple_non_adjacent_indexers(self) -> None:
        # GH10338: mixing slices, scalars, and a non-adjacent array indexer
        # in .sel() should not shuffle dimension sizes when reading through
        # the scipy backend from a closed file object.
        data = xr.Dataset(
            {
                "x": (
                    ("a", "b", "c", "d"),
                    np.random.rand(3, 6, 5, 25),
                )
            },
            coords={
                "a": np.arange(3),
                "b": ["u", "v", "w", "x", "y", "z"],
                "c": np.arange(5),
                "d": np.arange(0, 250, 10),
            },
        )
        with self.roundtrip(data) as on_disk:
            actual = on_disk["x"].sel(
                b="w",
                d=np.arange(0, 250, 20),
            )
            expected = data["x"].sel(
                b="w",
                d=np.arange(0, 250, 20),
            )
            assert actual.dims == expected.dims
            assert actual.shape == expected.shape
            assert_allclose(actual, expected)


@requires_scipy
class TestScipyFilePath(NetCDF3Only, CFEncodedBase):
    engine: T_NetcdfEngine = "scipy"

    @contextlib.contextmanager
    def create_store(self):
        with create_tmp_file() as tmp_file:
            with backends.ScipyDataStore(tmp_file, mode="w") as store:
                yield store

    def test_array_attrs(self) -> None:
        ds = Dataset(attrs={"foo": [[1, 2], [3, 4]]})
        with pytest.raises(ValueError, match=r"must be 1-dimensional"):
            with self.roundtrip(ds):
                pass

    def test_roundtrip_example_1_netcdf_gz(self) -> None:
        with open_example_dataset("example_1.nc.gz") as expected:
            with open_example_dataset("example_1.nc") as actual:
                assert_identical(expected, actual)

    def test_netcdf3_endianness(self) -> None:
        # regression test for GH416
        with open_example_dataset("bears.nc", engine="scipy") as expected:
            for var in expected.variables.values():
                assert var.dtype.isnative

    @pytest.mark.parametrize("dtype", ["i2", "i4", "f4", "f8"])
    def test_netcdf3_endianness_attrs(self, dtype) -> None:
        # netCDF3 stores attribute arrays big-endian, just like variable data
        values = np.array([1, 2], dtype=dtype)
        original = Dataset(
            {"x": ("t", [1.0], {"valid_range": values})}, attrs={"levels": values}
        )
        with self.roundtrip(original) as actual:
            for attr in (actual["x"].attrs["valid_range"], actual.attrs["levels"]):
                assert attr.dtype == values.dtype
                np.testing.assert_array_equal(attr, values)

    @requires_netCDF4
    def test_nc4_scipy(self) -> None:
        with create_tmp_file(allow_cleanup_failure=True) as tmp_file:
            with nc4.Dataset(tmp_file, "w", format="NETCDF4") as rootgrp:
                rootgrp.createGroup("foo")

            with pytest.raises(TypeError, match=r"pip install netcdf4"):
                open_dataset(tmp_file, engine="scipy")

    @pytest.mark.parametrize(
        "key",
        [
            indexing.BasicIndexer((slice(1, 3),)),
            indexing.OuterIndexer((np.array([0, 2]),)),
        ],
        ids=["view", "copy"],
    )
    def test_read_while_evicted_closes_mmap(self, key) -> None:
        # A file evicted from the cache while being read is closed as soon as
        # the read unpins it. No reference into its memory map may be alive by
        # then, or scipy cannot close the mmap and warns.
        original = Dataset({"x": ("t", np.arange(5.0))})
        with create_tmp_file() as tmp_file:
            original.to_netcdf(tmp_file, engine="scipy")
            with backends.ScipyDataStore(tmp_file, mmap=True) as store:
                array = ScipyArrayWrapper("x", store)
                manager = store._manager
                nc = manager.acquire()

                class EvictOnAccess(dict):
                    def __getitem__(self, name):
                        # evict the file while the read has it pinned
                        manager._cache.pop(manager._key)
                        _close_unless_pinned(manager._key, nc)
                        return super().__getitem__(name)

                # bypass netcdf_file.__setattr__, which records global attributes
                nc.__dict__["variables"] = EvictOnAccess(nc.variables)

                with warnings.catch_warnings():
                    warnings.simplefilter("error", RuntimeWarning)
                    actual = array[key]

                assert nc.fp.closed
                np.testing.assert_array_equal(actual, original["x"].values[key.tuple])


@requires_scipy
def test_scipy_entrypoint(tmp_path: Path) -> None:
    entrypoint = ScipyBackendEntrypoint()
    ds = create_test_data()

    path = tmp_path / "foo"
    ds.to_netcdf(path, engine="scipy")
    _check_guess_can_open_and_open(entrypoint, path, engine="scipy", expected=ds)
    _check_guess_can_open_and_open(entrypoint, str(path), engine="scipy", expected=ds)
    with open(path, "rb") as f:
        _check_guess_can_open_and_open(entrypoint, f, engine="scipy", expected=ds)

    contents = ds.to_netcdf(engine="scipy")
    _check_guess_can_open_and_open(entrypoint, contents, engine="scipy", expected=ds)
    _check_guess_can_open_and_open(
        entrypoint, BytesIO(contents), engine="scipy", expected=ds
    )

    path = tmp_path / "foo.nc.gz"
    with gzip.open(path, mode="wb") as f:
        f.write(contents)
    _check_guess_can_open_and_open(entrypoint, path, engine="scipy", expected=ds)
    _check_guess_can_open_and_open(entrypoint, str(path), engine="scipy", expected=ds)

    assert entrypoint.guess_can_open("something-local.nc")
    assert entrypoint.guess_can_open("something-local.nc.gz")
    assert not entrypoint.guess_can_open("not-found-and-no-extension")
    assert not entrypoint.guess_can_open(b"not-a-netcdf-file")
    # Should not claim .gz files that aren't netCDF
    assert not entrypoint.guess_can_open("something.zarr.gz")
    assert not entrypoint.guess_can_open("something.tar.gz")
    assert not entrypoint.guess_can_open("something.txt.gz")
