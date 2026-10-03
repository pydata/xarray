from __future__ import annotations

import asyncio
import contextlib
import os.path
import platform
import re
import tempfile
import uuid
import warnings
from collections import ChainMap
from collections.abc import Iterator, Mapping
from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, overload
from unittest.mock import patch

import numpy as np
import pytest
from packaging.version import Version

import xarray as xr
import xarray.testing as xrt
from xarray import (
    DataArray,
    Dataset,
    backends,
    open_dataarray,
    open_dataset,
    open_mfdataset,
)
from xarray.backends.zarr import ZarrStore
from xarray.coders import CFDatetimeCoder, CFTimedeltaCoder
from xarray.coding.cftime_offsets import date_range
from xarray.coding.variables import SerializationWarning
from xarray.compat.npcompat import HAS_STRING_DTYPE
from xarray.core.indexes import PandasIndex
from xarray.tests import (
    assert_equal,
    assert_identical,
    assert_no_warnings,
    dask_array_api,
    dask_array_type,
    has_netCDF4,
    has_zarr,
    has_zarr_v3_async_oindex,
    has_zarr_v3_dtypes,
    parametrize_zarr_format,
    requires_cftime,
    requires_dask,
    requires_fsspec,
    requires_netcdf,
    requires_netCDF4,
    requires_zarr,
    requires_zarr_rectilinear_chunks,
    requires_zarr_v3,
    requires_zarr_v3_async_oindex,
    requires_zarr_v3_dtypes,
)
from xarray.tests.backends.base import (
    DATA_DIR,
    MASK_AND_SCALE_DATA,
    ON_WINDOWS,
    CFEncodedBase,
    _check_guess_can_open_and_open,
    create_boolean_data,
    create_tmp_file,
    open_example_dataset,
    skip_if_zarr_format_2,
    skip_if_zarr_format_3,
    skip_if_zip_store,
)
from xarray.tests.test_dataset import (
    create_append_string_length_mismatch_test_data,
    create_append_test_data,
    create_test_data,
)

with contextlib.suppress(ImportError):
    import netCDF4 as nc4

with contextlib.suppress(ImportError):
    import dask


if has_zarr:
    import zarr
    import zarr.codecs
    from zarr.storage import MemoryStore as KVStore
    from zarr.storage import WrapperStore
else:
    KVStore = None  # type: ignore[assignment,misc,unused-ignore]
    WrapperStore = object  # type: ignore[assignment,misc,unused-ignore]

if TYPE_CHECKING:
    from zarr.abc.store import Store as ZarrStoreABC

    from xarray import Variable


if has_netCDF4:
    NETCDFC_VERSION: Version | None = Version(
        nc4.getlibversion().split()[0].split("-development")[0]
    )
else:
    NETCDFC_VERSION = None


@requires_zarr
@pytest.mark.usefixtures("default_zarr_format")
class ZarrBase(CFEncodedBase):
    DIMENSION_KEY = "_ARRAY_DIMENSIONS"
    version_kwargs: dict[str, Any] = {}

    @pytest.mark.parametrize("decoded_fn, encoded_fn", MASK_AND_SCALE_DATA)
    @pytest.mark.parametrize(
        "dtype",
        [
            np.dtype("float64"),
            pytest.param(
                np.dtype("float32"),
                marks=pytest.mark.skip(
                    reason="float32 will be treated as float64 in zarr"
                ),
            ),
        ],
    )
    def test_roundtrip_mask_and_scale(self, decoded_fn, encoded_fn, dtype) -> None:
        super().test_roundtrip_mask_and_scale(decoded_fn, encoded_fn, dtype)

    @pytest.mark.skip(reason="No unlimited_dims handled in zarr.")
    def test_encoding_unlimited_dims(self) -> None:
        super().test_encoding_unlimited_dims()

    @skip_if_zarr_format_3(
        "endian support requires zarr-python>=3.1", condition=not has_zarr_v3_dtypes
    )
    def test_roundtrip_endian(self) -> None:
        super().test_roundtrip_endian()

    def create_zarr_target(self):
        raise NotImplementedError

    @contextlib.contextmanager
    def create_store(self, cache_members: bool = False):
        with self.create_zarr_target() as store_target:
            yield backends.ZarrStore.open_group(
                store_target,
                mode="w",
                cache_members=cache_members,
                **self.version_kwargs,
            )

    def save(self, dataset, store_target, **kwargs):  # type: ignore[override]
        return dataset.to_zarr(store=store_target, **kwargs, **self.version_kwargs)

    @contextlib.contextmanager
    def open(self, path, **kwargs):
        with xr.open_dataset(
            path, engine="zarr", mode="r", **kwargs, **self.version_kwargs
        ) as ds:
            yield ds

    @contextlib.contextmanager
    def roundtrip(
        self, data, save_kwargs=None, open_kwargs=None, allow_cleanup_failure=False
    ):
        if save_kwargs is None:
            save_kwargs = {}
        if open_kwargs is None:
            open_kwargs = {}
        with self.create_zarr_target() as store_target:
            self.save(data, store_target, **save_kwargs)
            with self.open(store_target, **open_kwargs) as ds:
                yield ds

    @pytest.mark.asyncio
    async def test_load_async(self) -> None:
        await super().test_load_async()

    def test_roundtrip_boolean_dtype(self) -> None:
        original = create_boolean_data()
        assert original["x"].dtype == "bool"
        with self.create_zarr_target() as store_target:
            self.save(original, store_target, consolidated=False)
            # Verify on-disk zarr array uses native bool dtype (not int8)
            zg = zarr.open_group(store_target, mode="r")
            zarr_arr = zg["x"]
            assert isinstance(zarr_arr, zarr.Array)
            assert zarr_arr.dtype == np.dtype("bool")
            assert "dtype" not in zarr_arr.attrs
            with self.open(
                store_target, backend_kwargs={"consolidated": False}
            ) as actual:
                assert_identical(original, actual)
                assert actual["x"].dtype == "bool"
                # Verify second roundtrip also preserves bool
                with self.roundtrip(actual) as actual2:
                    assert_identical(original, actual2)
                    assert actual2["x"].dtype == "bool"

    def test_roundtrip_boolean_dtype_legacy_int8(self) -> None:
        """Verify backward compat: old-style int8 + attrs['dtype']='bool' decodes to bool."""
        original = create_boolean_data()
        with self.create_zarr_target() as store_target:
            zg = zarr.open_group(store_target, mode="w")
            data_int8 = original["x"].values.astype("i1")
            is_v3_format = zg.metadata.zarr_format == 3
            if is_v3_format:
                arr = zg.create_array(
                    "x",
                    shape=data_int8.shape,
                    dtype=data_int8.dtype,
                    fill_value=-1,
                    dimension_names=("t", "x"),
                )
            else:
                arr = zg.create_array(
                    "x",
                    shape=data_int8.shape,
                    dtype=data_int8.dtype,
                    fill_value=-1,
                )
            arr[:] = data_int8
            arr.attrs["dtype"] = "bool"
            arr.attrs["units"] = "-"
            if not is_v3_format:
                arr.attrs["_ARRAY_DIMENSIONS"] = ["t", "x"]
            with self.open(
                store_target, backend_kwargs={"consolidated": False}
            ) as actual:
                assert actual["x"].dtype == "bool"
                np.testing.assert_array_equal(actual["x"].values, original["x"].values)

    def test_roundtrip_bytes_with_fill_value(self):
        pytest.xfail("Broken by Zarr 3.0.7")

    @pytest.mark.parametrize("consolidated", [False, True, None])
    def test_roundtrip_consolidated(self, consolidated) -> None:
        expected = create_test_data()
        with self.roundtrip(
            expected,
            save_kwargs={"consolidated": consolidated},
            open_kwargs={"backend_kwargs": {"consolidated": consolidated}},
        ) as actual:
            self.check_dtypes_roundtripped(expected, actual)
            assert_identical(expected, actual)

    def test_read_non_consolidated_warning(self) -> None:
        expected = create_test_data()
        with self.create_zarr_target() as store:
            self.save(
                expected, store_target=store, consolidated=False, **self.version_kwargs
            )
            if getattr(store, "supports_consolidated_metadata", True):
                with pytest.warns(
                    RuntimeWarning,
                    match="Failed to open Zarr store with consolidated",
                ):
                    with xr.open_zarr(store, **self.version_kwargs) as ds:
                        assert_identical(ds, expected)

    def test_non_existent_store(self) -> None:
        patterns = [
            "No such file or directory",
            "Unable to find group",
            "No group found in store",
            "does not exist",
        ]
        with pytest.raises(FileNotFoundError, match=f"({'|'.join(patterns)})"):
            xr.open_zarr(f"{uuid.uuid4()}")

    @requires_dask
    def test_auto_chunk(self) -> None:
        original = create_test_data().chunk()

        with self.roundtrip(original, open_kwargs={"chunks": None}) as actual:
            for k, v in actual.variables.items():
                # only index variables should be in memory
                assert v._in_memory == (k in actual.dims)
                # there should be no chunks
                assert v.chunks is None

        with self.roundtrip(original, open_kwargs={"chunks": {}}) as actual:
            for k, v in actual.variables.items():
                # only index variables should be in memory
                assert v._in_memory == (k in actual.dims)
                # chunk size should be the same as original
                assert v.chunks == original[k].chunks

    @requires_dask
    @pytest.mark.filterwarnings("ignore:The specified chunks separate:UserWarning")
    def test_manual_chunk(self) -> None:
        original = create_test_data().chunk({"dim1": 3, "dim2": 4, "dim3": 3})

        # Using chunks = None should return non-chunked arrays
        open_kwargs: dict[str, Any] = {"chunks": None}
        with self.roundtrip(original, open_kwargs=open_kwargs) as actual:
            for k, v in actual.variables.items():
                # only index variables should be in memory
                assert v._in_memory == (k in actual.dims)
                # there should be no chunks
                assert v.chunks is None

        # uniform arrays
        for i in range(2, 6):
            rechunked = original.chunk(chunks=i)
            open_kwargs = {"chunks": i}
            with self.roundtrip(original, open_kwargs=open_kwargs) as actual:
                for k, v in actual.variables.items():
                    # only index variables should be in memory
                    assert v._in_memory == (k in actual.dims)
                    # chunk size should be the same as rechunked
                    assert v.chunks == rechunked[k].chunks

        chunks = {"dim1": 2, "dim2": 3, "dim3": 5}
        rechunked = original.chunk(chunks=chunks)

        open_kwargs = {
            "chunks": chunks,
            "backend_kwargs": {"overwrite_encoded_chunks": True},
        }
        with self.roundtrip(original, open_kwargs=open_kwargs) as actual:
            for k, v in actual.variables.items():
                assert v.chunks == rechunked[k].chunks

            with self.roundtrip(actual) as auto:
                # encoding should have changed
                for k, v in actual.variables.items():
                    assert v.chunks == rechunked[k].chunks

                assert_identical(actual, auto)
                assert_identical(actual.load(), auto.load())

    def test_unlimited_dims_encoding_is_ignored(self) -> None:
        ds = Dataset({"x": np.arange(10)})
        ds.encoding = {"unlimited_dims": ["x"]}
        with self.roundtrip(ds) as actual:
            assert_identical(ds, actual)

    @requires_dask
    @pytest.mark.filterwarnings("ignore:.*does not have a Zarr V3 specification.*")
    def test_warning_on_bad_chunks(self) -> None:
        original = create_test_data().chunk({"dim1": 4, "dim2": 3, "dim3": 3})

        bad_chunks = (2, {"dim2": (3, 3, 2, 1)})
        for chunks in bad_chunks:
            kwargs = {"chunks": chunks}
            with pytest.warns(UserWarning):
                with self.roundtrip(original, open_kwargs=kwargs) as actual:
                    in_memory = {k: v._in_memory for k, v in actual.variables.items()}
            # only index variables should be in memory
            assert in_memory == {k: k in actual.dims for k in actual.variables}

        good_chunks: tuple[dict[str, Any], ...] = ({"dim2": 3}, {"dim3": (6, 4)}, {})
        for chunks in good_chunks:
            kwargs = {"chunks": chunks}
            with assert_no_warnings():
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        message=".*Zarr format 3 specification.*",
                        category=UserWarning,
                    )
                    with self.roundtrip(original, open_kwargs=kwargs) as actual:
                        for k, v in actual.variables.items():
                            # only index variables should be in memory
                            assert v._in_memory == (k in actual.dims)

    @requires_dask
    def test_deprecate_auto_chunk(self) -> None:
        original = create_test_data().chunk()
        with pytest.raises(TypeError):
            with self.roundtrip(original, open_kwargs={"auto_chunk": True}) as actual:
                for k, v in actual.variables.items():
                    # only index variables should be in memory
                    assert v._in_memory == (k in actual.dims)
                    # chunk size should be the same as original
                    assert v.chunks == original[k].chunks

        with pytest.raises(TypeError):
            with self.roundtrip(original, open_kwargs={"auto_chunk": False}) as actual:
                for k, v in actual.variables.items():
                    # only index variables should be in memory
                    assert v._in_memory == (k in actual.dims)
                    # there should be no chunks
                    assert v.chunks is None

    @requires_dask
    def test_write_uneven_dask_chunks(self) -> None:
        # regression for GH#2225
        original = create_test_data().chunk({"dim1": 3, "dim2": 4, "dim3": 3})
        with self.roundtrip(original, open_kwargs={"chunks": {}}) as actual:
            for k, v in actual.data_vars.items():
                assert v.chunks == actual[k].chunks

    def test_chunk_encoding(self) -> None:
        # These datasets have no dask chunks. All chunking specified in
        # encoding
        data = create_test_data()
        chunks = (5, 5)
        data["var2"].encoding.update({"chunks": chunks})

        with self.roundtrip(data) as actual:
            assert chunks == actual["var2"].encoding["chunks"]

        # expect an error with non-integer chunks
        data["var2"].encoding.update({"chunks": (5, 4.5)})
        with pytest.raises(TypeError):
            with self.roundtrip(data) as actual:
                pass

    def test_shard_encoding(self) -> None:
        # These datasets have no dask chunks. All chunking/sharding specified in
        # encoding
        if zarr.config.config["default_zarr_format"] == 3:
            data = create_test_data()
            chunks = (1, 1)
            shards = (5, 5)
            data["var2"].encoding.update({"chunks": chunks})
            data["var2"].encoding.update({"shards": shards})
            with self.roundtrip(data) as actual:
                assert shards == actual["var2"].encoding["shards"]

            # expect an error with shards not divisible by chunks
            data["var2"].encoding.update({"chunks": (2, 2)})
            with pytest.raises(ValueError):
                with self.roundtrip(data) as actual:
                    pass

    @requires_dask
    @skip_if_zarr_format_2("sharding requires zarr v3 format")
    def test_shard_encoding_with_dask(self) -> None:
        # Test that dask chunks must align with shard boundaries.
        # See https://github.com/pydata/xarray/issues/10831

        ds = xr.DataArray(np.arange(12), dims="x", name="var1").to_dataset()

        # Case 1: Dask chunks equal to shards should work
        # (zarr chunk=3, shard=6, dask chunk=6)
        ds1 = ds.chunk({"x": 6})
        ds1["var1"].encoding = {"chunks": (3,), "shards": (6,)}
        with self.roundtrip(ds1) as actual:
            assert_identical(ds, actual)

        # Case 2: Dask chunks that are multiples of shards should work
        # (zarr chunk=1, shard=3, dask chunk=6)
        ds2 = ds.chunk({"x": 6})
        ds2["var1"].encoding = {"chunks": (1,), "shards": (3,)}
        with self.roundtrip(ds2) as actual:
            assert_identical(ds, actual)

        # Case 3: Dask chunks smaller than shards should fail
        # (zarr chunk=2, shard=4, dask chunk=3) - dask chunk doesn't align with shard
        ds3 = ds.chunk({"x": 3})
        ds3["var1"].encoding = {"chunks": (2,), "shards": (4,)}
        with pytest.raises(ValueError, match=r"would overlap"):
            with self.roundtrip(ds3) as actual:
                pass

        # Case 4: Can bypass with safe_chunks=False (but data may be corrupted)
        with self.roundtrip(ds3, save_kwargs={"safe_chunks": False}) as actual:
            pass

    @requires_dask
    @pytest.mark.skipif(
        ON_WINDOWS,
        reason="Very flaky on Windows CI. Can re-enable assuming it starts consistently passing.",
    )
    def test_chunk_encoding_with_dask(self) -> None:
        # These datasets DO have dask chunks. Need to check for various
        # interactions between dask and zarr chunks
        ds = xr.DataArray((np.arange(12)), dims="x", name="var1").to_dataset()

        # - no encoding specified -
        # zarr automatically gets chunk information from dask chunks
        ds_chunk4 = ds.chunk({"x": 4})
        with self.roundtrip(ds_chunk4) as actual:
            assert (4,) == actual["var1"].encoding["chunks"]

        # should fail if dask_chunks are irregular...
        ds_chunk_irreg = ds.chunk({"x": (5, 4, 3)})
        with pytest.raises(ValueError, match=r"uniform chunk sizes."):
            with self.roundtrip(ds_chunk_irreg) as actual:
                pass

        # should fail if encoding["chunks"] clashes with dask_chunks
        badenc = ds.chunk({"x": 4})
        badenc.var1.encoding["chunks"] = (6,)
        with pytest.raises(ValueError, match=r"named 'var1' would overlap"):
            with self.roundtrip(badenc) as actual:
                pass

        # unless...
        with self.roundtrip(badenc, save_kwargs={"safe_chunks": False}) as actual:
            # don't actually check equality because the data could be corrupted
            pass

        # if dask chunks (4) are an integer multiple of zarr chunks (2) it should not fail...
        goodenc = ds.chunk({"x": 4})
        goodenc.var1.encoding["chunks"] = (2,)
        with self.roundtrip(goodenc) as actual:
            pass

        # if initial dask chunks are aligned, size of last dask chunk doesn't matter
        goodenc = ds.chunk({"x": (3, 3, 6)})
        goodenc.var1.encoding["chunks"] = (3,)
        with self.roundtrip(goodenc) as actual:
            pass

        goodenc = ds.chunk({"x": (3, 6, 3)})
        goodenc.var1.encoding["chunks"] = (3,)
        with self.roundtrip(goodenc) as actual:
            pass

        # ... also if the last chunk is irregular
        ds_chunk_irreg = ds.chunk({"x": (5, 5, 2)})
        with self.roundtrip(ds_chunk_irreg) as actual:
            assert (5,) == actual["var1"].encoding["chunks"]
        # re-save Zarr arrays
        with self.roundtrip(ds_chunk_irreg) as original:
            with self.roundtrip(original) as actual:
                assert_identical(original, actual)

        # but intermediate unaligned chunks are bad
        badenc = ds.chunk({"x": (3, 5, 3, 1)})
        badenc.var1.encoding["chunks"] = (3,)
        with pytest.raises(ValueError, match=r"would overlap multiple Dask chunks"):
            with self.roundtrip(badenc) as actual:
                pass

        # - encoding specified  -
        # specify compatible encodings
        for chunk_enc in 4, (4,):
            ds_chunk4["var1"].encoding.update({"chunks": chunk_enc})
            with self.roundtrip(ds_chunk4) as actual:
                assert (4,) == actual["var1"].encoding["chunks"]

        # TODO: remove this failure once synchronized overlapping writes are
        # supported by xarray
        ds_chunk4["var1"].encoding.update({"chunks": 5})
        with pytest.raises(ValueError, match=r"named 'var1' would overlap"):
            with self.roundtrip(ds_chunk4) as actual:
                pass
        # override option
        with self.roundtrip(ds_chunk4, save_kwargs={"safe_chunks": False}) as actual:
            # don't actually check equality because the data could be corrupted
            pass

    @requires_netcdf
    def test_drop_encoding(self):
        with open_example_dataset("example_1.nc") as ds:
            encodings = {v: {**ds[v].encoding} for v in ds.data_vars}
            with self.create_zarr_target() as store:
                ds.to_zarr(store, encoding=encodings)

    @skip_if_zarr_format_3("This test is unnecessary; no hidden Zarr keys")
    def test_hidden_zarr_keys(self) -> None:
        expected = create_test_data()
        with self.create_store() as store:
            expected.dump_to_store(store)
            zarr_group = store.ds

            # check that a variable hidden attribute is present and correct
            # JSON only has a single array type, which maps to list in Python.
            # In contrast, dims in xarray is always a tuple.
            for var in expected.variables.keys():
                dims = zarr_group[var].attrs[self.DIMENSION_KEY]
                assert dims == list(expected[var].dims)

            with xr.decode_cf(store):
                # make sure it is hidden
                for var in expected.variables.keys():
                    assert self.DIMENSION_KEY not in expected[var].attrs

            # put it back and try removing from a variable
            attrs = dict(zarr_group["var2"].attrs)
            del attrs[self.DIMENSION_KEY]
            zarr_group["var2"].attrs.put(attrs)

            with pytest.raises(KeyError):
                with xr.decode_cf(store):
                    pass

    @skip_if_zarr_format_2("No dimension names in V2")
    def test_dimension_names(self) -> None:
        expected = create_test_data()
        with self.create_store() as store:
            expected.dump_to_store(store)
            zarr_group = store.ds
            for var in zarr_group:
                assert expected[var].dims == zarr_group[var].metadata.dimension_names

    @pytest.mark.parametrize("group", [None, "group1"])
    def test_write_persistence_modes(self, group) -> None:
        original = create_test_data()

        # overwrite mode
        with self.roundtrip(
            original,
            save_kwargs={"mode": "w", "group": group},
            open_kwargs={"group": group},
        ) as actual:
            assert_identical(original, actual)

        # don't overwrite mode
        with self.roundtrip(
            original,
            save_kwargs={"mode": "w-", "group": group},
            open_kwargs={"group": group},
        ) as actual:
            assert_identical(original, actual)

        # make sure overwriting works as expected
        with self.create_zarr_target() as store:
            self.save(original, store)
            # should overwrite with no error
            self.save(original, store, mode="w", group=group)
            with self.open(store, group=group) as actual:
                assert_identical(original, actual)
                with pytest.raises((ValueError, FileExistsError)):
                    self.save(original, store, mode="w-")

        # check append mode for normal write
        with self.roundtrip(
            original,
            save_kwargs={"mode": "a", "group": group},
            open_kwargs={"group": group},
        ) as actual:
            assert_identical(original, actual)

        # check append mode for append write
        ds, ds_to_append, _ = create_append_test_data()
        with self.create_zarr_target() as store_target:
            ds.to_zarr(store_target, mode="w", group=group, **self.version_kwargs)
            ds_to_append.to_zarr(
                store_target, append_dim="time", group=group, **self.version_kwargs
            )
            original = xr.concat([ds, ds_to_append], dim="time")
            actual = xr.open_dataset(
                store_target, group=group, engine="zarr", **self.version_kwargs
            )
            assert_identical(original, actual)

    def test_compressor_encoding(self) -> None:
        # specify a custom compressor
        original = create_test_data()
        if zarr.config.config["default_zarr_format"] == 3:
            encoding_key = "compressors"
            # all parameters need to be explicitly specified in order for the comparison to pass below
            encoding = {
                "serializer": zarr.codecs.BytesCodec(endian="little"),
                encoding_key: (
                    zarr.codecs.BloscCodec(
                        cname="zstd",
                        clevel=3,
                        shuffle="shuffle",
                        typesize=8,
                        blocksize=0,
                    ),
                ),
            }
        else:
            from numcodecs.blosc import Blosc

            encoding_key = "compressors"
            comp = Blosc(cname="zstd", clevel=3, shuffle=2)
            encoding = {encoding_key: (comp,)}

        save_kwargs = dict(encoding={"var1": encoding})

        with self.roundtrip(original, save_kwargs=save_kwargs) as ds:
            enc = ds["var1"].encoding[encoding_key]
            assert enc == encoding[encoding_key]

    def test_group(self) -> None:
        original = create_test_data()
        group = "some/random/path"
        with self.roundtrip(
            original, save_kwargs={"group": group}, open_kwargs={"group": group}
        ) as actual:
            assert_identical(original, actual)

    def test_zarr_mode_w_overwrites_encoding(self) -> None:
        data = Dataset({"foo": ("x", [1.0, 1.0, 1.0])})
        with self.create_zarr_target() as store:
            data.to_zarr(
                store, **self.version_kwargs, encoding={"foo": {"add_offset": 1}}
            )
            np.testing.assert_equal(
                zarr.open_group(store, **self.version_kwargs)["foo"], data.foo.data - 1
            )
            data.to_zarr(
                store,
                **self.version_kwargs,
                encoding={"foo": {"add_offset": 0}},
                mode="w",
            )
            np.testing.assert_equal(
                zarr.open_group(store, **self.version_kwargs)["foo"], data.foo.data
            )

    def test_encoding_kwarg_fixed_width_string(self) -> None:
        # not relevant for zarr, since we don't use EncodedStringCoder
        pass

    def test_dataset_caching(self) -> None:
        super().test_dataset_caching()

    def test_append_write(self) -> None:
        super().test_append_write()

    def test_append_with_mode_rplus_success(self) -> None:
        original = Dataset({"foo": ("x", [1])})
        modified = Dataset({"foo": ("x", [2])})
        with self.create_zarr_target() as store:
            original.to_zarr(store, **self.version_kwargs)
            modified.to_zarr(store, mode="r+", **self.version_kwargs)
            with self.open(store) as actual:
                assert_identical(actual, modified)

    def test_append_with_mode_rplus_fails(self) -> None:
        original = Dataset({"foo": ("x", [1])})
        modified = Dataset({"bar": ("x", [2])})
        with self.create_zarr_target() as store:
            original.to_zarr(store, **self.version_kwargs)
            with pytest.raises(
                ValueError, match="dataset contains non-pre-existing variables"
            ):
                modified.to_zarr(store, mode="r+", **self.version_kwargs)

    def test_append_with_invalid_dim_raises(self) -> None:
        ds, ds_to_append, _ = create_append_test_data()
        with self.create_zarr_target() as store_target:
            ds.to_zarr(store_target, mode="w", **self.version_kwargs)
            with pytest.raises(
                ValueError, match="does not match any existing dataset dimensions"
            ):
                ds_to_append.to_zarr(
                    store_target, append_dim="notvalid", **self.version_kwargs
                )

    def test_append_with_no_dims_raises(self) -> None:
        with self.create_zarr_target() as store_target:
            Dataset({"foo": ("x", [1])}).to_zarr(
                store_target, mode="w", **self.version_kwargs
            )
            with pytest.raises(ValueError, match="different dimension names"):
                Dataset({"foo": ("y", [2])}).to_zarr(
                    store_target, mode="a", **self.version_kwargs
                )

    def test_append_with_append_dim_not_set_raises(self) -> None:
        ds, ds_to_append, _ = create_append_test_data()
        with self.create_zarr_target() as store_target:
            ds.to_zarr(store_target, mode="w", **self.version_kwargs)
            with pytest.raises(ValueError, match="different dimension sizes"):
                ds_to_append.to_zarr(store_target, mode="a", **self.version_kwargs)

    def test_append_with_mode_not_a_raises(self) -> None:
        ds, ds_to_append, _ = create_append_test_data()
        with self.create_zarr_target() as store_target:
            ds.to_zarr(store_target, mode="w", **self.version_kwargs)
            with pytest.raises(ValueError, match="cannot set append_dim unless"):
                ds_to_append.to_zarr(
                    store_target, mode="w", append_dim="time", **self.version_kwargs
                )

    def test_append_with_existing_encoding_raises(self) -> None:
        ds, ds_to_append, _ = create_append_test_data()
        with self.create_zarr_target() as store_target:
            ds.to_zarr(store_target, mode="w", **self.version_kwargs)
            with pytest.raises(ValueError, match="but encoding was provided"):
                ds_to_append.to_zarr(
                    store_target,
                    append_dim="time",
                    encoding={"da": {"compressor": None}},
                    **self.version_kwargs,
                )

    @skip_if_zarr_format_3(
        "This actually works fine with Zarr format 3", condition=not has_zarr_v3_dtypes
    )
    @pytest.mark.parametrize("dtype", ["U", "S"])
    def test_append_string_length_mismatch_raises(self, dtype) -> None:
        ds, ds_to_append = create_append_string_length_mismatch_test_data(dtype)
        with self.create_zarr_target() as store_target:
            ds.to_zarr(store_target, mode="w", **self.version_kwargs)
            with pytest.raises(ValueError, match="Mismatched dtypes for variable"):
                ds_to_append.to_zarr(
                    store_target, append_dim="time", **self.version_kwargs
                )

    @pytest.mark.skipif(
        has_zarr_v3_dtypes,
        reason="This works on pre ZDtype Zarr-Python, but fails after.",
    )
    # ...but it probably would work with Zarr format 2 if we used object dtype
    @skip_if_zarr_format_2("This doesn't work with Zarr format 2")
    @pytest.mark.parametrize("dtype", ["U", "S"])
    def test_append_string_length_mismatch_works(self, dtype) -> None:

        ds, ds_to_append = create_append_string_length_mismatch_test_data(dtype)
        expected = xr.concat([ds, ds_to_append], dim="time")

        with self.create_zarr_target() as store_target:
            ds.to_zarr(store_target, mode="w", **self.version_kwargs)
            ds_to_append.to_zarr(store_target, append_dim="time", **self.version_kwargs)
            actual = xr.open_dataset(store_target, engine="zarr")
            xr.testing.assert_identical(expected, actual)

    def test_check_encoding_is_consistent_after_append(self) -> None:
        ds, ds_to_append, _ = create_append_test_data()

        # check encoding consistency
        with self.create_zarr_target() as store_target:
            import numcodecs

            encoding_value: Any
            if zarr.config.config["default_zarr_format"] == 3:
                compressor = zarr.codecs.BloscCodec()
            else:
                compressor = numcodecs.Blosc()
            encoding_key = "compressors"
            encoding_value = (compressor,)

            encoding = {"da": {encoding_key: encoding_value}}
            ds.to_zarr(store_target, mode="w", encoding=encoding, **self.version_kwargs)
            original_ds = xr.open_dataset(
                store_target, engine="zarr", **self.version_kwargs
            )
            original_encoding = original_ds["da"].encoding[encoding_key]
            ds_to_append.to_zarr(store_target, append_dim="time", **self.version_kwargs)
            actual_ds = xr.open_dataset(
                store_target, engine="zarr", **self.version_kwargs
            )

            actual_encoding = actual_ds["da"].encoding[encoding_key]
            assert original_encoding == actual_encoding
            assert_identical(
                xr.open_dataset(
                    store_target, engine="zarr", **self.version_kwargs
                ).compute(),
                xr.concat([ds, ds_to_append], dim="time"),
            )

    def test_append_with_new_variable(self) -> None:
        ds, ds_to_append, ds_with_new_var = create_append_test_data()

        # check append mode for new variable
        with self.create_zarr_target() as store_target:
            combined = xr.concat([ds, ds_to_append], dim="time")
            combined.to_zarr(store_target, mode="w", **self.version_kwargs)
            assert_identical(
                combined,
                xr.open_dataset(store_target, engine="zarr", **self.version_kwargs),
            )
            ds_with_new_var.to_zarr(store_target, mode="a", **self.version_kwargs)
            combined = xr.concat([ds, ds_to_append], dim="time")
            combined["new_var"] = ds_with_new_var["new_var"]
            assert_identical(
                combined,
                xr.open_dataset(store_target, engine="zarr", **self.version_kwargs),
            )

    def test_append_with_append_dim_no_overwrite(self) -> None:
        ds, ds_to_append, _ = create_append_test_data()
        with self.create_zarr_target() as store_target:
            ds.to_zarr(store_target, mode="w", **self.version_kwargs)
            original = xr.concat([ds, ds_to_append], dim="time")
            original2 = xr.concat([original, ds_to_append], dim="time")

            # overwrite a coordinate;
            # for mode='a-', this will not get written to the store
            # because it does not have the append_dim as a dim
            lon = ds_to_append.lon.to_numpy().copy()
            lon[:] = -999
            ds_to_append["lon"] = lon
            ds_to_append.to_zarr(
                store_target, mode="a-", append_dim="time", **self.version_kwargs
            )
            actual = xr.open_dataset(store_target, engine="zarr", **self.version_kwargs)
            assert_identical(original, actual)

            # by default, mode="a" will overwrite all coordinates.
            ds_to_append.to_zarr(store_target, append_dim="time", **self.version_kwargs)
            actual = xr.open_dataset(store_target, engine="zarr", **self.version_kwargs)
            lon = original2.lon.to_numpy().copy()
            lon[:] = -999
            original2["lon"] = lon
            assert_identical(original2, actual)

    @requires_dask
    def test_to_zarr_compute_false_roundtrip(self) -> None:
        from dask.delayed import Delayed

        original = create_test_data().chunk()

        with self.create_zarr_target() as store:
            delayed_obj = self.save(original, store, compute=False)
            assert isinstance(delayed_obj, Delayed)

            # make sure target store has not been written to yet
            with pytest.raises(AssertionError):
                with self.open(store) as actual:
                    assert_identical(original, actual)

            delayed_obj.compute()

            with self.open(store) as actual:
                assert_identical(original, actual)

    @requires_dask
    # whether appending also warns about the object dtype depends on the zarr format
    @pytest.mark.filterwarnings(
        "ignore:variable None has data in the form of a dask array with dtype=object"
    )
    def test_to_zarr_append_compute_false_roundtrip(self) -> None:
        from dask.delayed import Delayed

        ds, ds_to_append, _ = create_append_test_data()
        ds, ds_to_append = ds.chunk(), ds_to_append.chunk()

        with self.create_zarr_target() as store:
            with pytest.warns(SerializationWarning):
                delayed_obj = self.save(ds, store, compute=False, mode="w")
            assert isinstance(delayed_obj, Delayed)

            with pytest.raises(AssertionError):
                with self.open(store) as actual:
                    assert_identical(ds, actual)

            delayed_obj.compute()

            with self.open(store) as actual:
                assert_identical(ds, actual)

            delayed_obj = self.save(
                ds_to_append, store, compute=False, append_dim="time"
            )
            assert isinstance(delayed_obj, Delayed)

            with pytest.raises(AssertionError):
                with self.open(store) as actual:
                    assert_identical(xr.concat([ds, ds_to_append], dim="time"), actual)

            delayed_obj.compute()

            with self.open(store) as actual:
                assert_identical(xr.concat([ds, ds_to_append], dim="time"), actual)

    def test_save_emptydim(self, use_dask) -> None:
        ds = Dataset({"x": (("a", "b"), np.empty((5, 0))), "y": ("a", [1, 2, 5, 8, 9])})
        if use_dask:
            ds = ds.chunk({})  # chunk dataset to save dask array
        with self.roundtrip(ds) as ds_reload:
            assert_identical(ds, ds_reload)

    @requires_dask
    def test_no_warning_from_open_emptydim_with_chunks(self) -> None:
        ds = Dataset({"x": (("a", "b"), np.empty((5, 0)))}).chunk({"a": 1})
        with assert_no_warnings():
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=".*Zarr format 3 specification.*",
                    category=UserWarning,
                )
                with self.roundtrip(ds, open_kwargs=dict(chunks={"a": 1})) as ds_reload:
                    assert_identical(ds, ds_reload)

    @pytest.mark.parametrize("consolidated", [False, True, None])
    @pytest.mark.parametrize(
        "compute", [pytest.param(False, marks=requires_dask), True]
    )
    @pytest.mark.parametrize("write_empty", [False, True, None])
    def test_write_region(self, consolidated, compute, use_dask, write_empty) -> None:

        zeros = Dataset({"u": (("x",), np.zeros(10))})
        nonzeros = Dataset({"u": (("x",), np.arange(1, 11))})

        if use_dask:
            zeros = zeros.chunk(2)
            nonzeros = nonzeros.chunk(2)

        with self.create_zarr_target() as store:
            zeros.to_zarr(
                store,
                consolidated=consolidated,
                compute=compute,
                encoding={"u": dict(chunks=2)},
                **self.version_kwargs,
            )
            if compute:
                with xr.open_zarr(
                    store, consolidated=consolidated, **self.version_kwargs
                ) as actual:
                    assert_identical(actual, zeros)
            for i in range(0, 10, 2):
                region = {"x": slice(i, i + 2)}
                nonzeros.isel(region).to_zarr(
                    store,
                    region=region,
                    consolidated=consolidated,
                    write_empty_chunks=write_empty,
                    **self.version_kwargs,
                )
            with xr.open_zarr(
                store, consolidated=consolidated, **self.version_kwargs
            ) as actual:
                assert_identical(actual, nonzeros)

    def test_region_scalar(self) -> None:
        ds = Dataset({"x": 0})
        with self.create_zarr_target() as store:
            ds.to_zarr(store)
            ds.to_zarr(store, region={}, mode="r+")
            with xr.open_zarr(store) as actual:
                assert_identical(actual, ds)

    @pytest.mark.parametrize("mode", [None, "r+", "a"])
    def test_write_region_mode(self, mode) -> None:
        zeros = Dataset({"u": (("x",), np.zeros(10))})
        nonzeros = Dataset({"u": (("x",), np.arange(1, 11))})
        with self.create_zarr_target() as store:
            zeros.to_zarr(store, **self.version_kwargs)
            for region in [{"x": slice(5)}, {"x": slice(5, 10)}]:
                nonzeros.isel(region).to_zarr(
                    store, region=region, mode=mode, **self.version_kwargs
                )
            with xr.open_zarr(store, **self.version_kwargs) as actual:
                assert_identical(actual, nonzeros)

    @requires_dask
    def test_write_preexisting_override_metadata(self) -> None:
        """Metadata should be overridden if mode="a" but not in mode="r+"."""
        original = Dataset(
            {"u": (("x",), np.zeros(10), {"variable": "original"})},
            attrs={"global": "original"},
        )
        both_modified = Dataset(
            {"u": (("x",), np.ones(10), {"variable": "modified"})},
            attrs={"global": "modified"},
        )
        global_modified = Dataset(
            {"u": (("x",), np.ones(10), {"variable": "original"})},
            attrs={"global": "modified"},
        )
        only_new_data = Dataset(
            {"u": (("x",), np.ones(10), {"variable": "original"})},
            attrs={"global": "original"},
        )

        with self.create_zarr_target() as store:
            original.to_zarr(store, compute=False, **self.version_kwargs)
            both_modified.to_zarr(store, mode="a", **self.version_kwargs)
            with self.open(store) as actual:
                # NOTE: this arguably incorrect -- we should probably be
                # overriding the variable metadata, too. See the TODO note in
                # ZarrStore.set_variables.
                assert_identical(actual, global_modified)

        with self.create_zarr_target() as store:
            original.to_zarr(store, compute=False, **self.version_kwargs)
            both_modified.to_zarr(store, mode="r+", **self.version_kwargs)
            with self.open(store) as actual:
                assert_identical(actual, only_new_data)

        with self.create_zarr_target() as store:
            original.to_zarr(store, compute=False, **self.version_kwargs)
            # with region, the default mode becomes r+
            both_modified.to_zarr(
                store, region={"x": slice(None)}, **self.version_kwargs
            )
            with self.open(store) as actual:
                assert_identical(actual, only_new_data)

    def test_write_region_errors(self) -> None:
        data = Dataset({"u": (("x",), np.arange(5))})
        data2 = Dataset({"u": (("x",), np.array([10, 11]))})

        @contextlib.contextmanager
        def setup_and_verify_store(expected=data):
            with self.create_zarr_target() as store:
                data.to_zarr(store, **self.version_kwargs)
                yield store
                with self.open(store) as actual:
                    assert_identical(actual, expected)

        # verify the base case works
        expected = Dataset({"u": (("x",), np.array([10, 11, 2, 3, 4]))})
        with setup_and_verify_store(expected) as store:
            data2.to_zarr(store, region={"x": slice(2)}, **self.version_kwargs)

        with setup_and_verify_store() as store:
            with pytest.raises(
                ValueError,
                match=re.escape(
                    "cannot set region unless mode='a', mode='a-', mode='r+' or mode=None"
                ),
            ):
                data.to_zarr(
                    store, region={"x": slice(None)}, mode="w", **self.version_kwargs
                )

        with setup_and_verify_store() as store:
            with pytest.raises(TypeError, match=r"must be a dict"):
                data.to_zarr(store, region=slice(None), **self.version_kwargs)  # type: ignore[call-overload]

        with setup_and_verify_store() as store:
            with pytest.raises(TypeError, match=r"must be slice objects"):
                data2.to_zarr(store, region={"x": [0, 1]}, **self.version_kwargs)  # type: ignore[dict-item]

        with setup_and_verify_store() as store:
            with pytest.raises(ValueError, match=r"step on all slices"):
                data2.to_zarr(
                    store, region={"x": slice(None, None, 2)}, **self.version_kwargs
                )

        with setup_and_verify_store() as store:
            with pytest.raises(
                ValueError,
                match=r"all keys in ``region`` are not in Dataset dimensions",
            ):
                data.to_zarr(store, region={"y": slice(None)}, **self.version_kwargs)

        with setup_and_verify_store() as store:
            with pytest.raises(
                ValueError,
                match=r"all variables in the dataset to write must have at least one dimension in common",
            ):
                data2.assign(v=2).to_zarr(
                    store, region={"x": slice(2)}, **self.version_kwargs
                )

        with setup_and_verify_store() as store:
            with pytest.raises(
                ValueError, match=r"cannot list the same dimension in both"
            ):
                data.to_zarr(
                    store,
                    region={"x": slice(None)},
                    append_dim="x",
                    **self.version_kwargs,
                )

        with setup_and_verify_store() as store:
            with pytest.raises(
                ValueError,
                match=r"variable 'u' already exists with different dimension sizes",
            ):
                data2.to_zarr(store, region={"x": slice(3)}, **self.version_kwargs)

    @requires_dask
    def test_encoding_chunksizes(self) -> None:
        # regression test for GH2278
        # see also test_encoding_chunksizes_unlimited
        nx, ny, nt = 4, 4, 5
        original = xr.Dataset(
            {},
            coords={
                "x": np.arange(nx),
                "y": np.arange(ny),
                "t": np.arange(nt),
            },
        )
        original["v"] = xr.Variable(("x", "y", "t"), np.zeros((nx, ny, nt)))
        original = original.chunk({"t": 1, "x": 2, "y": 2})

        with self.roundtrip(original) as ds1:
            assert_equal(ds1, original)
            with self.roundtrip(ds1.isel(t=0)) as ds2:
                assert_equal(ds2, original.isel(t=0))

    @requires_dask
    def test_chunk_encoding_with_partial_dask_chunks(self) -> None:
        original = xr.Dataset(
            {"x": xr.DataArray(np.random.random(size=(6, 8)), dims=("a", "b"))}
        ).chunk({"a": 3})

        with self.roundtrip(
            original, save_kwargs={"encoding": {"x": {"chunks": [3, 2]}}}
        ) as ds1:
            assert_equal(ds1, original)

    @requires_dask
    def test_chunk_encoding_with_larger_dask_chunks(self) -> None:
        original = xr.Dataset({"a": ("x", [1, 2, 3, 4])}).chunk({"x": 2})

        with self.roundtrip(
            original, save_kwargs={"encoding": {"a": {"chunks": [1]}}}
        ) as ds1:
            assert_equal(ds1, original)

    @requires_dask
    def test_chunk_auto_with_small_dask_chunks(self) -> None:
        original = Dataset({"u": (("x",), np.zeros(10))}).chunk({"x": 2})
        with self.create_zarr_target() as store:
            original.to_zarr(store, **self.version_kwargs)
            with xr.open_zarr(store, **self.version_kwargs) as default:
                assert default.chunks == {"x": (2, 2, 2, 2, 2)}
                with xr.open_zarr(store, chunks="auto", **self.version_kwargs) as auto:
                    assert_identical(auto, original)
                    assert auto.chunks == {"x": (10,)}
                    assert auto.chunks != default.chunks

    @requires_cftime
    def test_open_zarr_use_cftime(self) -> None:
        ds = create_test_data()
        with self.create_zarr_target() as store_target:
            ds.to_zarr(store_target, **self.version_kwargs)
            ds_a = xr.open_zarr(store_target, **self.version_kwargs)
            assert_identical(ds, ds_a)
            decoder = CFDatetimeCoder(use_cftime=True)
            ds_b = xr.open_zarr(
                store_target, decode_times=decoder, **self.version_kwargs
            )
            assert xr.coding.times.contains_cftime_datetimes(ds_b.time.variable)

    def test_write_read_select_write(self) -> None:
        # Test for https://github.com/pydata/xarray/issues/4084
        ds = create_test_data()

        # NOTE: using self.roundtrip, which uses open_dataset, will not trigger the bug.
        with self.create_zarr_target() as initial_store:
            ds.to_zarr(initial_store, mode="w", **self.version_kwargs)
            ds1 = xr.open_zarr(initial_store, **self.version_kwargs)

            # Combination of where+squeeze triggers error on write.
            ds_sel = ds1.where(ds1.coords["dim3"] == "a", drop=True).squeeze("dim3")
            with self.create_zarr_target() as final_store:
                ds_sel.to_zarr(final_store, mode="w", **self.version_kwargs)

    @pytest.mark.parametrize("obj", [Dataset(), DataArray(name="foo")])
    def test_attributes(self, obj) -> None:
        obj = obj.copy()

        obj.attrs["good"] = {"key": "value"}
        ds = obj if isinstance(obj, Dataset) else obj.to_dataset()
        with self.create_zarr_target() as store_target:
            ds.to_zarr(store_target, **self.version_kwargs)
            assert_identical(ds, xr.open_zarr(store_target, **self.version_kwargs))

        obj.attrs["bad"] = DataArray()
        ds = obj if isinstance(obj, Dataset) else obj.to_dataset()
        with self.create_zarr_target() as store_target:
            with pytest.raises(TypeError, match=r"Invalid attribute in Dataset.attrs."):
                ds.to_zarr(store_target, **self.version_kwargs)

    @requires_dask
    @pytest.mark.parametrize("dtype", ["datetime64[ns]", "timedelta64[ns]"])
    def test_chunked_datetime64_or_timedelta64(self, dtype) -> None:
        # Generalized from @malmans2's test in PR #8253
        original = create_test_data().astype(dtype).chunk(1)
        with self.roundtrip(
            original,
            open_kwargs={
                "chunks": {},
                "decode_timedelta": CFTimedeltaCoder(time_unit="ns"),
            },
        ) as actual:
            for name, actual_var in actual.variables.items():
                assert original[name].chunks == actual_var.chunks
            assert original.chunks == actual.chunks

    @requires_cftime
    @requires_dask
    def test_chunked_cftime_datetime(self) -> None:
        # Based on @malmans2's test in PR #8253
        times = date_range("2000", freq="D", periods=3, use_cftime=True)
        original = xr.Dataset(data_vars={"chunked_times": (["time"], times)})
        original = original.chunk({"time": 1})
        with self.roundtrip(original, open_kwargs={"chunks": {}}) as actual:
            for name, actual_var in actual.variables.items():
                assert original[name].chunks == actual_var.chunks
            assert original.chunks == actual.chunks

    def test_cache_members(self) -> None:
        """
        Ensure that if `ZarrStore` is created with `cache_members` set to `True`,
        a `ZarrStore` only inspects the underlying zarr group once,
        and that the results of that inspection are cached.

        Otherwise, `ZarrStore.members` should inspect the underlying zarr group each time it is
        invoked
        """
        with self.create_zarr_target() as store_target:
            zstore_mut = backends.ZarrStore.open_group(
                store_target, mode="w", cache_members=False
            )

            # ensure that the keys are sorted
            array_keys = sorted(("foo", "bar"))

            # create some arrays
            for ak in array_keys:
                zstore_mut.zarr_group.create(name=ak, shape=(1,), dtype="uint8")

            zstore_stat = backends.ZarrStore.open_group(
                store_target, mode="r", cache_members=True
            )

            observed_keys_0 = sorted(zstore_stat.array_keys())
            assert observed_keys_0 == array_keys

            # create a new array
            new_key = "baz"
            zstore_mut.zarr_group.create(name=new_key, shape=(1,), dtype="uint8")

            observed_keys_1 = sorted(zstore_stat.array_keys())
            assert observed_keys_1 == array_keys

            observed_keys_2 = sorted(zstore_mut.array_keys())
            assert observed_keys_2 == sorted(array_keys + [new_key])

    @requires_dask
    @pytest.mark.parametrize("dtype", [int, float])
    def test_zarr_fill_value_setting(self, dtype):
        # When zarr_format=2, _FillValue sets fill_value
        # When zarr_format=3, fill_value is set independently
        # We test this by writing a dask array with compute=False,
        # on read we should receive chunks filled with `fill_value`
        fv = -1
        da = dask_array_api
        ds = xr.Dataset(
            {
                "foo": (
                    "x",
                    da.from_array(np.array([0, 0, 0], dtype=dtype), chunks=(3,)),
                )
            }
        )
        expected = xr.Dataset({"foo": ("x", [fv] * 3)})

        zarr_format_2 = zarr.config.get("default_zarr_format") == 2
        if zarr_format_2:
            attr = "_FillValue"
            expected.foo.attrs[attr] = fv
        else:
            attr = "fill_value"
            if dtype is float:
                # for floats, Xarray inserts a default `np.nan`
                expected.foo.attrs["_FillValue"] = np.nan

        # turn off all decoding so we see what Zarr returns to us.
        # Since chunks, are not written, we should receive on `fill_value`
        open_kwargs = {
            "mask_and_scale": False,
            "consolidated": False,
            "use_zarr_fill_value_as_mask": False,
        }
        save_kwargs = dict(compute=False, consolidated=False)
        with self.roundtrip(
            ds,
            save_kwargs=ChainMap(save_kwargs, dict(encoding={"foo": {attr: fv}})),
            open_kwargs=open_kwargs,
        ) as actual:
            assert_identical(actual, expected)

        ds.foo.encoding[attr] = fv
        with self.roundtrip(
            ds, save_kwargs=save_kwargs, open_kwargs=open_kwargs
        ) as actual:
            assert_identical(actual, expected)

        if zarr_format_2:
            ds = ds.drop_encoding()
            with pytest.raises(ValueError, match="_FillValue"):
                with self.roundtrip(
                    ds,
                    save_kwargs=ChainMap(
                        save_kwargs, dict(encoding={"foo": {"fill_value": fv}})
                    ),
                    open_kwargs=open_kwargs,
                ):
                    pass
            # TODO: this doesn't fail because of the
            # ``raise_on_invalid=vn in check_encoding_set`` line in zarr.py
            # ds.foo.encoding["fill_value"] = fv

    @skip_if_zarr_format_2("fill_value is only an encoding key for zarr_format 3")
    def test_zarr_fill_value_in_encoding_on_read(self) -> None:
        # GH #10269: the Zarr array fill_value should be preserved in the
        # variable encoding on read, so that it is not lost on round-trip.
        # `fill_value` is an independent encoding key only for zarr_format 3;
        # for zarr_format 2 the fill_value is set via `_FillValue`.

        ds = xr.Dataset({"foo": ("x", [1, 2, 3])})
        ds.foo.encoding = {"fill_value": -99}

        open_kwargs = {"consolidated": False, "use_zarr_fill_value_as_mask": False}
        with self.roundtrip(ds, open_kwargs=open_kwargs) as actual:
            assert actual.foo.encoding["fill_value"] == -99

        # the fill_value must survive an open -> write -> open round-trip even
        # when the user never touches the encoding explicitly
        with self.roundtrip(ds, open_kwargs=open_kwargs) as opened:
            with self.roundtrip(opened, open_kwargs=open_kwargs) as actual:
                assert actual.foo.encoding["fill_value"] == -99


@requires_zarr
class TestInstrumentedZarrStore:
    methods = [
        "get",
        "set",
        "list_dir",
        "list_prefix",
    ]

    @contextlib.contextmanager
    def create_zarr_target(self):
        store = KVStore({}, read_only=False)  # type: ignore[arg-type,unused-ignore]
        yield store

    def make_patches(self, store):
        from unittest.mock import MagicMock

        return {
            method: MagicMock(
                f"KVStore.{method}",
                side_effect=getattr(store, method),
                autospec=True,
            )
            for method in self.methods
        }

    def summarize(self, patches):
        summary = {}
        for name, patch_ in patches.items():
            count = 0
            for call in patch_.mock_calls:
                if "zarr.json" not in call.args:
                    count += 1
            summary[name.strip("_")] = count
        return summary

    def check_requests(self, expected, patches):
        summary = self.summarize(patches)
        for k in summary:
            assert summary[k] <= expected[k], (k, summary)

    def test_append(self) -> None:
        original = Dataset({"foo": ("x", [1])}, coords={"x": [0]})
        modified = Dataset({"foo": ("x", [2])}, coords={"x": [1]})

        with self.create_zarr_target() as store:
            # TODO: verify these
            expected = {
                "set": 5,
                "get": 4,
                "list_dir": 2,
                "list_prefix": 1,
            }

            patches = self.make_patches(store)
            with patch.multiple(KVStore, **patches):
                original.to_zarr(store)
            self.check_requests(expected, patches)

            patches = self.make_patches(store)
            expected = {
                "set": 4,
                "get": 9,  # TODO: fixme upstream (should be 8)
                "list_dir": 2,  # TODO: fixme upstream (should be 2)
                "list_prefix": 0,
            }

            with patch.multiple(KVStore, **patches):
                modified.to_zarr(store, mode="a", append_dim="x")
            self.check_requests(expected, patches)

            patches = self.make_patches(store)

            expected = {
                "set": 4,
                "get": 9,  # TODO: fixme upstream (should be 8)
                "list_dir": 2,  # TODO: fixme upstream (should be 2)
                "list_prefix": 0,
            }

            with patch.multiple(KVStore, **patches):
                modified.to_zarr(store, mode="a-", append_dim="x")
            self.check_requests(expected, patches)

            with open_dataset(store, engine="zarr") as actual:
                assert_identical(
                    actual, xr.concat([original, modified, modified], dim="x")
                )

    @requires_dask
    def test_region_write(self) -> None:
        ds = Dataset({"foo": ("x", [1, 2, 3])}, coords={"x": [1, 2, 3]}).chunk()
        with self.create_zarr_target() as store:
            expected = {
                "set": 5,
                "get": 2,
                "list_dir": 2,
                "list_prefix": 4,
            }

            patches = self.make_patches(store)
            with patch.multiple(KVStore, **patches):
                ds.to_zarr(store, mode="w", compute=False)
            self.check_requests(expected, patches)

            expected = {
                "set": 1,
                "get": 3,
                "list_dir": 0,
                "list_prefix": 0,
            }

            patches = self.make_patches(store)
            with patch.multiple(KVStore, **patches):
                ds.to_zarr(store, region={"x": slice(None)})
            self.check_requests(expected, patches)

            expected = {
                "set": 1,
                "get": 4,
                "list_dir": 0,
                "list_prefix": 0,
            }

            patches = self.make_patches(store)
            with patch.multiple(KVStore, **patches):
                ds.to_zarr(store, region="auto")
            self.check_requests(expected, patches)

            expected = {
                "set": 0,
                "get": 5,
                "list_dir": 0,
                "list_prefix": 0,
            }

            patches = self.make_patches(store)
            with patch.multiple(KVStore, **patches):
                with open_dataset(store, engine="zarr") as actual:
                    assert_identical(actual, ds)
            self.check_requests(expected, patches)


@requires_zarr_v3_dtypes
@pytest.mark.skipif(not HAS_STRING_DTYPE, reason="requires StringDType")
def test_roundtrip_stringdtype_zarr_v3() -> None:
    dtype = np.dtypes.StringDType()
    data = np.array(["a", "bb", "ccc"], dtype=dtype)
    expected = Dataset(
        {
            "data": ("dim", data.copy()),
            "scalar": np.array("a", dtype=dtype),
        },
        coords={
            "dim": ("dim", data.copy()),
            "nondim": ("dim", data.copy()),
        },
    )
    store = zarr.storage.MemoryStore({}, read_only=False)

    with assert_no_warnings():
        expected.to_zarr(store, zarr_format=3, consolidated=False)
    actual = xr.open_zarr(store, consolidated=False).load()

    for name in expected.variables:
        assert zarr.open_array(store=store, path=str(name), mode="r").dtype == dtype
        assert actual[name].dtype == dtype
    assert_identical(expected, actual)


@requires_zarr_v3_dtypes
@pytest.mark.skipif(not HAS_STRING_DTYPE, reason="requires StringDType")
def test_stringdtype_with_na_object_uses_the_compatibility_path() -> None:
    dtype = np.dtypes.StringDType(na_object=np.nan)
    data = np.array(["a", "bb", "ccc"], dtype=dtype)
    expected = Dataset({"data": ("dim", data.copy())})
    store = zarr.storage.MemoryStore({}, read_only=False)

    expected.to_zarr(store, zarr_format=3, consolidated=False)
    actual = xr.open_zarr(store, consolidated=False).load()

    assert zarr.open_array(store=store, path="data", mode="r").dtype.kind != "T"
    assert (actual["data"].values == data.astype(object)).all()


@requires_zarr
class TestZarrDictStore(ZarrBase):
    @contextlib.contextmanager
    def create_zarr_target(self):
        yield zarr.storage.MemoryStore({}, read_only=False)

    def test_chunk_key_encoding_v2(self) -> None:
        encoding = {"name": "v2", "configuration": {"separator": "/"}}

        # Create a dataset with a variable name containing a period
        data = np.ones((4, 4))
        original = Dataset({"var1": (("x", "y"), data)})

        # Set up chunk key encoding with slash separator
        encoding = {
            "var1": {
                "chunk_key_encoding": encoding,
                "chunks": (2, 2),
            }
        }

        # Write to store with custom encoding
        with self.create_zarr_target() as store:
            original.to_zarr(store, encoding=encoding)

            # Read back and verify data
            with xr.open_zarr(store) as actual:
                assert_identical(original, actual)
                # Verify chunks are preserved
                assert actual["var1"].encoding["chunks"] == (2, 2)

    @pytest.mark.asyncio
    @requires_zarr_v3
    async def test_async_load_multiple_variables(self) -> None:
        target_class = zarr.AsyncArray
        method_name = "getitem"
        original_method = getattr(target_class, method_name)

        # the indexed coordinate variables is not lazy, so the create_test_dataset has 4 lazy variables in total
        N_LAZY_VARS = 4

        original = create_test_data()
        with self.create_zarr_target() as store:
            original.to_zarr(store, zarr_format=3, consolidated=False)

            with patch.object(
                target_class, method_name, side_effect=original_method, autospec=True
            ) as mocked_meth:
                # blocks upon loading the coordinate variables here
                ds = xr.open_zarr(store, consolidated=False, chunks=None)

                # TODO we're not actually testing that these indexing methods are not blocking...
                result_ds = await ds.load_async()

                mocked_meth.assert_called()
                assert mocked_meth.call_count == N_LAZY_VARS
                mocked_meth.assert_awaited()

            xrt.assert_identical(result_ds, ds.load())

    @pytest.mark.asyncio
    @requires_zarr_v3
    @pytest.mark.parametrize("cls_name", ["Variable", "DataArray", "Dataset"])
    async def test_concurrent_load_multiple_objects(
        self,
        cls_name: Literal["Variable", "DataArray", "Dataset"],
    ) -> None:
        N_OBJECTS = 5
        N_LAZY_VARS = {
            "Variable": 1,
            "DataArray": 1,
            "Dataset": 4,
        }  # specific to the create_test_data() used

        target_class = zarr.AsyncArray
        method_name = "getitem"
        original_method = getattr(target_class, method_name)

        original = create_test_data()
        with self.create_zarr_target() as store:
            original.to_zarr(store, consolidated=False, zarr_format=3)

            with patch.object(
                target_class, method_name, side_effect=original_method, autospec=True
            ) as mocked_meth:
                xr_obj = get_xr_obj(store, cls_name)

                # TODO we're not actually testing that these indexing methods are not blocking...
                coros = [xr_obj.load_async() for _ in range(N_OBJECTS)]
                results = await asyncio.gather(*coros)

                mocked_meth.assert_called()
                assert mocked_meth.call_count == N_OBJECTS * N_LAZY_VARS[cls_name]
                mocked_meth.assert_awaited()

            for result in results:
                xrt.assert_identical(result, xr_obj.load())

    @pytest.mark.asyncio
    @requires_zarr_v3
    @pytest.mark.skip_if_param(
        cls_name="Variable", method="sel", reason="Variable doesn't have a .sel method"
    )
    @pytest.mark.parametrize("cls_name", ["Variable", "DataArray", "Dataset"])
    @pytest.mark.parametrize(
        "indexer, method, target_zarr_class",
        [
            pytest.param({}, "sel", "zarr.AsyncArray", id="no-indexing-sel"),
            pytest.param({}, "isel", "zarr.AsyncArray", id="no-indexing-isel"),
            pytest.param({"dim2": 1.0}, "sel", "zarr.AsyncArray", id="basic-int-sel"),
            pytest.param({"dim2": 2}, "isel", "zarr.AsyncArray", id="basic-int-isel"),
            pytest.param(
                {"dim2": slice(1.0, 3.0)},
                "sel",
                "zarr.AsyncArray",
                id="basic-slice-sel",
            ),
            pytest.param(
                {"dim2": slice(1, 3)}, "isel", "zarr.AsyncArray", id="basic-slice-isel"
            ),
            pytest.param(
                {"dim2": [1.0, 3.0]},
                "sel",
                "zarr.core.indexing.AsyncOIndex",
                marks=requires_zarr_v3_async_oindex,
                id="outer-sel",
            ),
            pytest.param(
                {"dim2": [1, 3]},
                "isel",
                "zarr.core.indexing.AsyncOIndex",
                marks=requires_zarr_v3_async_oindex,
                id="outer-isel",
            ),
            pytest.param(
                {
                    "dim1": xr.Variable(data=[2, 3], dims="points"),
                    "dim2": xr.Variable(data=[1.0, 2.0], dims="points"),
                },
                "sel",
                "zarr.core.indexing.AsyncVIndex",
                marks=requires_zarr_v3_async_oindex,
                id="vectorized-sel",
            ),
            pytest.param(
                {
                    "dim1": xr.Variable(data=[2, 3], dims="points"),
                    "dim2": xr.Variable(data=[1, 3], dims="points"),
                },
                "isel",
                "zarr.core.indexing.AsyncVIndex",
                marks=requires_zarr_v3_async_oindex,
                id="vectorized-isel",
            ),
        ],
    )
    async def test_indexing(
        self,
        cls_name: Literal["Variable", "DataArray", "Dataset"],
        method: Literal["sel", "isel"],
        indexer,
        target_zarr_class,
    ) -> None:
        # Each type of indexing ends up calling a different zarr indexing method
        # They all use a method named .getitem, but on a different internal zarr class
        def _resolve_class_from_string(class_path: str) -> type[Any]:
            """Resolve a string class path like 'zarr.AsyncArray' to the actual class."""
            module_path, class_name = class_path.rsplit(".", 1)
            module = import_module(module_path)
            return getattr(module, class_name)

        target_class = _resolve_class_from_string(target_zarr_class)
        method_name = "getitem"
        original_method = getattr(target_class, method_name)

        original = create_test_data()
        with self.create_zarr_target() as store:
            original.to_zarr(store, consolidated=False, zarr_format=3)

            with patch.object(
                target_class, method_name, side_effect=original_method, autospec=True
            ) as mocked_meth:
                xr_obj = get_xr_obj(store, cls_name)

                # TODO we're not actually testing that these indexing methods are not blocking...
                result = await getattr(xr_obj, method)(**indexer).load_async()

                mocked_meth.assert_called()
                mocked_meth.assert_awaited()
                assert mocked_meth.call_count > 0

            expected = getattr(xr_obj, method)(**indexer).load()
            xrt.assert_identical(result, expected)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("indexer", "expected_err_msg"),
        [
            pytest.param(
                {"dim2": [1, 3]},
                "orthogonal async indexing",
                marks=pytest.mark.skipif(
                    has_zarr_v3_async_oindex,
                    reason="current version of zarr has async orthogonal indexing",
                ),
            ),  # tests oindexing
            pytest.param(
                {
                    "dim1": xr.Variable(data=[2, 3], dims="points"),
                    "dim2": xr.Variable(data=[1, 3], dims="points"),
                },
                "vectorized async indexing",
                marks=pytest.mark.skipif(
                    has_zarr_v3_async_oindex,
                    reason="current version of zarr has async vectorized indexing",
                ),
            ),  # tests vindexing
        ],
    )
    @parametrize_zarr_format
    async def test_raise_on_older_zarr_version(
        self,
        indexer,
        expected_err_msg,
        zarr_format,
    ):
        """Test that trying to use async load with insufficiently new version of zarr raises a clear error"""

        original = create_test_data()
        with self.create_zarr_target() as store:
            original.to_zarr(store, consolidated=False, zarr_format=zarr_format)

            ds = xr.open_zarr(store, consolidated=False, chunks=None)
            var = ds["var1"].variable

            with pytest.raises(NotImplementedError, match=expected_err_msg):
                await var.isel(**indexer).load_async()


@overload
def get_xr_obj(store: ZarrStoreABC, cls_name: Literal["Variable"]) -> Variable: ...


@overload
def get_xr_obj(store: ZarrStoreABC, cls_name: Literal["DataArray"]) -> DataArray: ...


@overload
def get_xr_obj(store: ZarrStoreABC, cls_name: Literal["Dataset"]) -> Dataset: ...


@overload
def get_xr_obj(
    store: ZarrStoreABC, cls_name: Literal["Variable", "DataArray", "Dataset"]
) -> Variable | DataArray | Dataset: ...


def get_xr_obj(
    store: ZarrStoreABC, cls_name: Literal["Variable", "DataArray", "Dataset"]
) -> Variable | DataArray | Dataset:
    ds = xr.open_zarr(store, consolidated=False, chunks=None)

    match cls_name:
        case "Variable":
            return ds["var1"].variable
        case "DataArray":
            return ds["var1"]
        case "Dataset":
            return ds


class NoConsolidatedMetadataSupportStore(WrapperStore):
    """
    Store that explicitly does not support consolidated metadata.

    Useful as a proxy for stores like Icechunk, see https://github.com/zarr-developers/zarr-python/pull/3119.
    """

    supports_consolidated_metadata = False

    def __init__(
        self,
        store,
        *,
        read_only: bool = False,
    ) -> None:
        self._store = store.with_read_only(read_only=read_only)

    def with_read_only(
        self, read_only: bool = False
    ) -> NoConsolidatedMetadataSupportStore:
        return type(self)(
            store=self._store,
            read_only=read_only,
        )


@requires_zarr_v3
class TestZarrNoConsolidatedMetadataSupport(ZarrBase):
    @contextlib.contextmanager
    def create_zarr_target(self):
        # TODO the zarr version would need to be >3.08 for the supports_consolidated_metadata property to have any effect
        yield NoConsolidatedMetadataSupportStore(
            zarr.storage.MemoryStore({}, read_only=False)
        )


@requires_zarr
@pytest.mark.skipif(
    ON_WINDOWS,
    reason="Very flaky on Windows CI. Can re-enable assuming it starts consistently passing.",
)
class TestZarrDirectoryStore(ZarrBase):
    @contextlib.contextmanager
    def create_zarr_target(self):
        with create_tmp_file(suffix=".zarr") as tmp:
            yield tmp


@requires_zarr
class TestZarrWriteEmpty(TestZarrDirectoryStore):
    @contextlib.contextmanager
    def temp_dir(self) -> Iterator[tuple[str, str]]:
        with tempfile.TemporaryDirectory() as d:
            store = os.path.join(d, "test.zarr")
            yield d, store

    @contextlib.contextmanager
    def roundtrip_dir(
        self,
        data,
        store,
        save_kwargs=None,
        open_kwargs=None,
        allow_cleanup_failure=False,
    ) -> Iterator[Dataset]:
        if save_kwargs is None:
            save_kwargs = {}
        if open_kwargs is None:
            open_kwargs = {}

        data.to_zarr(store, **save_kwargs, **self.version_kwargs)
        with xr.open_dataset(
            store, engine="zarr", **open_kwargs, **self.version_kwargs
        ) as ds:
            yield ds

    @requires_dask
    def test_default_zarr_fill_value(self):
        inputs = xr.Dataset({"floats": ("x", [1.0]), "ints": ("x", [1])}).chunk()
        expected = xr.Dataset({"floats": ("x", [np.nan]), "ints": ("x", [0])})
        with self.temp_dir() as (_d, store):
            inputs.to_zarr(store, compute=False)
            with open_dataset(store) as on_disk:
                assert np.isnan(on_disk.variables["floats"].encoding["_FillValue"])
                assert (
                    "_FillValue" not in on_disk.variables["ints"].encoding
                )  # use default
                assert_identical(expected, on_disk)

    @pytest.mark.parametrize("consolidated", [True, False, None])
    @pytest.mark.parametrize("write_empty", [True, False, None])
    def test_write_empty(
        self,
        consolidated: bool | None,
        write_empty: bool | None,
        use_dask: bool,
    ) -> None:
        def assert_expected_files(expected: list[str], store: str) -> None:
            """Convenience for comparing with actual files written"""
            ls = []
            test_root = os.path.join(store, "test")
            for root, _, files in os.walk(test_root):
                ls.extend(
                    [
                        os.path.join(root, f).removeprefix(test_root).lstrip("/")
                        for f in files
                    ]
                )

            assert set(expected) == {
                file.lstrip("c/")
                for file in ls
                if (file not in (".zattrs", ".zarray", "zarr.json"))
            }

        # The zarr format is set by the `default_zarr_format`
        # pytest fixture that acts on a superclass
        zarr_format_3 = zarr.config.config["default_zarr_format"] == 3
        if (write_empty is False) or (write_empty is None):
            expected = ["0.1.0"]
        else:
            expected = [
                "0.0.0",
                "0.0.1",
                "0.1.0",
                "0.1.1",
            ]

        # use nan for default fill_value behaviour
        data = np.array([np.nan, np.nan, 1.0, np.nan]).reshape((1, 2, 2))

        if zarr_format_3:
            # transform to the path style of zarr 3
            # e.g. 0/0/1
            expected = [e.replace(".", "/") for e in expected]

        ds = xr.Dataset(data_vars={"test": (("Z", "Y", "X"), data)})

        if use_dask:
            ds["test"] = ds["test"].chunk(1)
            encoding = None
        else:
            encoding = {"test": {"chunks": (1, 1, 1)}}

        with self.temp_dir() as (_d, store):
            ds.to_zarr(
                store,
                mode="w",
                encoding=encoding,
                write_empty_chunks=write_empty,
            )

            # check expected files after a write
            assert_expected_files(expected, store)

            with self.roundtrip_dir(
                ds,
                store,
                save_kwargs={
                    "mode": "a",
                    "append_dim": "Z",
                    "write_empty_chunks": write_empty,
                },
            ) as a_ds:
                expected_ds = xr.concat([ds, ds], dim="Z")

                assert_identical(a_ds, expected_ds.compute())
                # add the new files we expect to be created by the append
                # that was performed by the roundtrip_dir
                if (write_empty is False) or (write_empty is None):
                    expected.append("1.1.0")
                elif has_zarr_v3_async_oindex:
                    # this was broken from zarr 3.0.0 until 3.1.2
                    # async oindex released in 3.1.2 along with a fix
                    # for write_empty_chunks in append
                    expected.extend(
                        [
                            "1.1.0",
                            "1.0.0",
                            "1.0.1",
                            "1.1.1",
                        ]
                    )
                else:
                    expected.append("1.1.0")
                if zarr_format_3:
                    expected = [e.replace(".", "/") for e in expected]
                assert_expected_files(expected, store)

    def test_avoid_excess_metadata_calls(self) -> None:
        """Test that chunk requests do not trigger redundant metadata requests.

        This test targets logic in backends.zarr.ZarrArrayWrapper, asserting that calls
        to retrieve chunk data after initialization do not trigger additional
        metadata requests.

        https://github.com/pydata/xarray/issues/8290
        """
        ds = xr.Dataset(data_vars={"test": (("Z",), np.array([123]).reshape(1))})

        # The call to retrieve metadata performs a group lookup. We patch Group.__getitem__
        # so that we can inspect calls to this method - specifically count of calls.
        # Use of side_effect means that calls are passed through to the original method
        # rather than a mocked method.

        Group: Any = zarr.AsyncGroup
        patched = patch.object(
            Group, "getitem", side_effect=Group.getitem, autospec=True
        )

        with self.create_zarr_target() as store, patched as mock:
            ds.to_zarr(store, mode="w")

            # We expect this to request array metadata information, so call_count should be == 1,
            xrds = xr.open_zarr(store)
            call_count = mock.call_count
            assert call_count == 1

            # compute() requests array data, which should not trigger additional metadata requests
            # we assert that the number of calls has not increased after fetchhing the array
            xrds.test.compute(scheduler="sync")
            assert mock.call_count == call_count


@pytest.fixture(params=["tmp_path", "ZipStore", "Dict"])
def tmp_store(request, tmp_path):
    if request.param == "tmp_path":
        return tmp_path
    elif request.param == "ZipStore":
        from zarr.storage import ZipStore

        path = tmp_path / "store.zip"
        return ZipStore(path)
    elif request.param == "Dict":
        return dict()
    else:
        raise ValueError("not supported")


@requires_zarr
class TestDataArrayToZarr:
    @skip_if_zip_store
    def test_dataarray_to_zarr_no_name(self, tmp_store) -> None:
        original_da = DataArray(np.arange(12).reshape((3, 4)))

        original_da.to_zarr(tmp_store)

        with open_dataarray(tmp_store, engine="zarr") as loaded_da:
            assert_identical(original_da, loaded_da)

    @skip_if_zip_store
    def test_dataarray_to_zarr_with_name(self, tmp_store) -> None:
        original_da = DataArray(np.arange(12).reshape((3, 4)), name="test")

        original_da.to_zarr(tmp_store)

        with open_dataarray(tmp_store, engine="zarr") as loaded_da:
            assert_identical(original_da, loaded_da)

    @skip_if_zip_store
    def test_dataarray_to_zarr_coord_name_clash(self, tmp_store) -> None:
        original_da = DataArray(
            np.arange(12).reshape((3, 4)), dims=["x", "y"], name="x"
        )

        original_da.to_zarr(tmp_store)

        with open_dataarray(tmp_store, engine="zarr") as loaded_da:
            assert_identical(original_da, loaded_da)

    @skip_if_zip_store
    def test_open_dataarray_options(self, tmp_store) -> None:
        data = DataArray(np.arange(5), coords={"y": ("x", range(1, 6))}, dims=["x"])

        data.to_zarr(tmp_store)

        expected = data.drop_vars("y")
        with open_dataarray(tmp_store, engine="zarr", drop_variables=["y"]) as loaded:
            assert_identical(expected, loaded)

    @requires_dask
    @skip_if_zip_store
    def test_dataarray_to_zarr_compute_false(self, tmp_store) -> None:
        from dask.delayed import Delayed

        original_da = DataArray(np.arange(12).reshape((3, 4)))

        output = original_da.to_zarr(tmp_store, compute=False)
        assert isinstance(output, Delayed)
        output.compute()
        with open_dataarray(tmp_store, engine="zarr") as loaded_da:
            assert_identical(original_da, loaded_da)

    @requires_dask
    @skip_if_zip_store
    def test_dataarray_to_zarr_align_chunks_true(self, tmp_store) -> None:
        # TODO: Improve data integrity checks when using Dask.
        #   Detecting automatic alignment issues in Dask can be tricky,
        #   as unintended misalignment might lead to subtle data corruption.
        #   For now, ensure that the parameter is present, but explore
        #   more robust verification methods to confirm data consistency.

        arr = DataArray(
            np.arange(4), dims=["a"], coords={"a": np.arange(4)}, name="foo"
        ).chunk(a=(2, 1, 1))

        arr.to_zarr(
            tmp_store,
            align_chunks=True,
            encoding={"foo": {"chunks": (3,)}},
        )
        with open_dataarray(tmp_store, engine="zarr") as loaded_da:
            assert_identical(arr, loaded_da)


@requires_zarr
def test_encode_zarr_attr_value() -> None:
    # array -> list
    arr = np.array([1, 2, 3])
    expected1 = [1, 2, 3]
    actual1 = backends.zarr.encode_zarr_attr_value(arr)
    assert isinstance(actual1, list)
    assert actual1 == expected1

    # scalar array -> scalar
    sarr = np.array(1)[()]
    expected2 = 1
    actual2 = backends.zarr.encode_zarr_attr_value(sarr)
    assert isinstance(actual2, int)
    assert actual2 == expected2

    # string -> string (no change)
    expected3 = "foo"
    actual3 = backends.zarr.encode_zarr_attr_value(expected3)
    assert isinstance(actual3, str)
    assert actual3 == expected3


@requires_zarr
@pytest.mark.parametrize("dtype", [complex, np.complex64, np.complex128])
def test_fill_value_coder_complex(dtype) -> None:
    """Test that FillValueCoder round-trips complex fill values."""
    from xarray.backends.zarr import FillValueCoder

    for value in [dtype(1 + 2j), dtype(-3.5 + 4.5j), dtype(complex("nan+nanj"))]:
        encoded = FillValueCoder.encode(value, np.dtype(dtype))
        decoded = FillValueCoder.decode(encoded, np.dtype(dtype))
        np.testing.assert_equal(np.array(decoded, dtype=dtype), np.array(value))


@requires_zarr
@pytest.mark.parametrize(
    "value,dtype",
    [
        (np.float32(np.inf), np.float32),
        (np.float32(-np.inf), np.float32),
        (np.float64(np.inf), np.float64),
        (np.float64(-np.inf), np.float64),
        (np.float32(np.nan), np.float32),
        (np.float64(np.nan), np.float64),
    ],
)
def test_fill_value_coder_inf_nan(value, dtype) -> None:
    """Test that FillValueCoder round-trips inf and nan fill values."""
    from xarray.backends.zarr import FillValueCoder

    encoded = FillValueCoder.encode(value, np.dtype(dtype))
    decoded = FillValueCoder.decode(encoded, np.dtype(dtype))
    np.testing.assert_equal(
        np.array(decoded, dtype=dtype), np.array(value, dtype=dtype)
    )

@requires_zarr
def test_fill_value_coder_json_bytes() -> None:
    from xarray.backends.zarr import FillValueCoder
    decoded = FillValueCoder.decode("hello", np.dtype("S5"))
    assert decoded == b"hello"

@requires_zarr
def test_fill_value_coder_json_none() -> None:
    from xarray.backends.zarr import FillValueCoder
    decoded = FillValueCoder.decode(None, np.dtype("float32"))
    assert decoded is None

@requires_zarr
def test_fill_value_coder_json_float() -> None:
    from xarray.backends.zarr import FillValueCoder
    decoded = FillValueCoder.decode(0.0, np.dtype("float32"))
    assert decoded == np.float32(0.0)

@requires_zarr
def test_fill_value_coder_json_string() -> None:
    from xarray.backends.zarr import FillValueCoder
    decoded = FillValueCoder.decode("hello", np.dtype("U5"))
    assert decoded == "hello"

@requires_zarr
def test_extract_zarr_variable_encoding() -> None:
    var = xr.Variable("x", [1, 2])
    actual = backends.zarr.extract_zarr_variable_encoding(var, zarr_format=3)
    assert "chunks" in actual
    assert actual["chunks"] == "auto"

    var = xr.Variable("x", [1, 2], encoding={"chunks": (1,)})
    actual = backends.zarr.extract_zarr_variable_encoding(var, zarr_format=3)
    assert actual["chunks"] == (1,)

    # does not raise on invalid
    var = xr.Variable("x", [1, 2], encoding={"foo": (1,)})
    actual = backends.zarr.extract_zarr_variable_encoding(var, zarr_format=3)

    # raises on invalid
    var = xr.Variable("x", [1, 2], encoding={"foo": (1,)})
    with pytest.raises(ValueError, match=r"unexpected encoding parameters"):
        actual = backends.zarr.extract_zarr_variable_encoding(
            var, raise_on_invalid=True, zarr_format=3
        )


@requires_zarr_rectilinear_chunks
class TestZarrRectilinearChunksRead:
    """Reading rectilinear (variable-sized) zarr chunks.

    xarray can't write these yet, so the stores are created with zarr directly.
    """

    @staticmethod
    def create_zarr_array(
        store_path, shape, chunks, dimension_names, dtype, shards=None
    ):
        import zarr

        root = zarr.open_group(store_path, mode="w", zarr_format=3)
        return root.create(
            "var",
            shape=shape,
            # older zarr stubs (<3.2) don't include rectilinear chunk types
            chunks=chunks,  # type: ignore[arg-type, unused-ignore]
            shards=shards,  # type: ignore[arg-type, unused-ignore]
            dtype=dtype,
            dimension_names=dimension_names,
        )

    # expected_chunks is what xarray puts in encoding["chunks"]: an int per
    # regular dim, a tuple of sizes per rectilinear dim (zarr-python itself
    # differs between versions here). expected_dask_chunks is the expanded form.
    cases = pytest.mark.parametrize(
        "shape,chunks,dimension_names,dtype,expected_chunks,expected_dask_chunks",
        [
            pytest.param(
                (60,),
                ((10, 20, 30),),
                ("x",),
                "float32",
                ((10, 20, 30),),
                ((10, 20, 30),),
                id="1d-rectilinear",
            ),
            pytest.param(
                (6, 20),
                (2, (5, 10, 5)),
                ("x", "y"),
                "float64",
                (2, (5, 10, 5)),
                ((2, 2, 2), (5, 10, 5)),
                id="mixed-regular-and-rectilinear",
            ),
        ],
    )

    @cases
    def test_read(
        self,
        tmp_path,
        shape,
        chunks,
        dimension_names,
        dtype,
        expected_chunks,
        expected_dask_chunks,
    ) -> None:
        import zarr

        data = np.arange(np.prod(shape), dtype=dtype).reshape(shape)
        store_path = tmp_path / "source.zarr"

        with zarr.config.set({"array.rectilinear_chunks": True}):
            arr = self.create_zarr_array(
                store_path, shape, chunks, dimension_names, dtype
            )
            arr[:] = data

            roundtrip = xr.open_zarr(
                store_path, zarr_format=3, consolidated=False, chunks=None
            )
            assert roundtrip["var"].encoding["chunks"] == expected_chunks
            assert roundtrip["var"].encoding["preferred_chunks"] == dict(
                zip(dimension_names, expected_chunks, strict=True)
            )
            assert isinstance(roundtrip["var"].data, np.ndarray)
            np.testing.assert_array_equal(roundtrip["var"].values, data)

    @cases
    @requires_dask
    def test_read_dask(
        self,
        tmp_path,
        shape,
        chunks,
        dimension_names,
        dtype,
        expected_chunks,
        expected_dask_chunks,
    ) -> None:
        """Dask arrays get the exact variable-sized chunks."""
        import zarr

        data = np.arange(np.prod(shape), dtype=dtype).reshape(shape)
        store_path = tmp_path / "source.zarr"

        with zarr.config.set({"array.rectilinear_chunks": True}):
            arr = self.create_zarr_array(
                store_path, shape, chunks, dimension_names, dtype
            )
            arr[:] = data

            roundtrip = xr.open_zarr(store_path, zarr_format=3, consolidated=False)
            assert isinstance(roundtrip["var"].data, dask_array_type)
            assert roundtrip["var"].data.chunks == expected_dask_chunks
            np.testing.assert_array_equal(roundtrip["var"].values, data)

    @requires_dask
    def test_read_zero_length_dim(self, tmp_path) -> None:
        """zarr reports no chunk sizes at all along a zero-length dimension,
        which must not be passed to dask as an empty tuple."""
        import zarr

        store_path = tmp_path / "source.zarr"
        with zarr.config.set({"array.rectilinear_chunks": True}):
            self.create_zarr_array(
                store_path,
                shape=(0, 20),
                chunks=(2, (5, 10, 5)),
                dimension_names=("x", "y"),
                dtype="float32",
            )
            roundtrip = xr.open_zarr(store_path, zarr_format=3, consolidated=False)
            assert roundtrip["var"].shape == (0, 20)
            assert roundtrip["var"].encoding["chunks"][1] == (5, 10, 5)
            assert roundtrip["var"].data.chunks == ((0,), (5, 10, 5))

    def test_read_rectilinear_shards(self, tmp_path) -> None:
        """Regular inner chunks with rectilinear shards. The regular chunks
        must be reported as an int regardless of zarr-python version."""
        import zarr

        data = np.array([1.0, 2.0, 3.0], dtype="float32")
        store_path = tmp_path / "source.zarr"

        with zarr.config.set({"array.rectilinear_chunks": True}):
            arr = self.create_zarr_array(
                store_path,
                shape=(3,),
                chunks=(1,),
                shards=((1, 2),),
                dimension_names=("x",),
                dtype="float32",
            )
            arr[:] = data

            roundtrip = xr.open_zarr(
                store_path, zarr_format=3, consolidated=False, chunks=None
            )
            assert roundtrip["var"].encoding["chunks"] == (1,)
            assert roundtrip["var"].encoding["shards"] == ((1, 2),)
            np.testing.assert_array_equal(roundtrip["var"].values, data)

    def test_write_after_read_gives_helpful_error(self, tmp_path) -> None:
        """Writing isn't supported yet; the error should say so, not just
        "must be an int"."""
        import zarr

        data = np.arange(60, dtype="float32")
        store_path = tmp_path / "source.zarr"

        with zarr.config.set({"array.rectilinear_chunks": True}):
            arr = self.create_zarr_array(
                store_path,
                shape=(60,),
                chunks=((10, 20, 30),),
                dimension_names=("x",),
                dtype="float32",
            )
            arr[:] = data

            roundtrip = xr.open_zarr(store_path, zarr_format=3, consolidated=False)
            with pytest.raises(TypeError, match=r"rectilinear"):
                roundtrip.to_zarr(tmp_path / "dest.zarr", zarr_format=3, mode="w")

    def test_append_does_not_resize_before_erroring(self, tmp_path) -> None:
        """A failed append must not leave the existing array resized."""
        import zarr

        data = np.arange(60, dtype="float32")
        store_path = tmp_path / "source.zarr"

        with zarr.config.set({"array.rectilinear_chunks": True}):
            arr = self.create_zarr_array(
                store_path,
                shape=(60,),
                chunks=((10, 20, 30),),
                dimension_names=("x",),
                dtype="float32",
            )
            arr[:] = data

            ds = xr.open_zarr(store_path, zarr_format=3, consolidated=False)
            with pytest.raises(TypeError, match=r"rectilinear"):
                ds.to_zarr(
                    store_path, append_dim="x", zarr_format=3, consolidated=False
                )

            assert zarr.open_array(store_path / "var").shape == (60,)

    def test_write_rectilinear_shards_blocked(self, tmp_path) -> None:
        """Rectilinear shards must be rejected on write, like rectilinear chunks."""
        data = np.arange(60, dtype="float32")
        ds = xr.Dataset({"var": ("x", data)})
        ds["var"].encoding["chunks"] = (10,)
        ds["var"].encoding["shards"] = ((20, 10, 30),)

        import zarr

        with zarr.config.set({"array.rectilinear_chunks": True}):
            with pytest.raises(TypeError, match=r"rectilinear"):
                ds.to_zarr(tmp_path / "dest.zarr", zarr_format=3, mode="w")

    @pytest.mark.parametrize(
        "shards",
        [
            20,
            (20,),
            pytest.param(
                "auto",
                marks=pytest.mark.filterwarnings(
                    "ignore:Automatic shard shape inference is experimental"
                ),
            ),
        ],
        ids=repr,
    )
    def test_write_regular_shards_still_works(self, tmp_path, shards) -> None:
        """Every non-rectilinear shard spec zarr accepts must still write."""
        data = np.arange(60, dtype="float32")
        ds = xr.Dataset({"var": ("x", data)})
        ds["var"].encoding["chunks"] = 10
        ds["var"].encoding["shards"] = shards
        ds.to_zarr(tmp_path / "dest.zarr", zarr_format=3, mode="w")
        assert (
            xr.open_zarr(tmp_path / "dest.zarr")["var"].encoding["shards"] is not None
        )


@pytest.fixture
def fsspec_memory_zarr_stores():
    """Write two zarr stores to fsspec's global in-memory filesystem."""
    import fsspec

    ds = open_dataset(os.path.join(DATA_DIR, "example_1.nc"))

    m = fsspec.filesystem("memory")
    mm = m.get_mapper("out1.zarr")
    ds.to_zarr(mm)  # old interface
    ds0 = ds.copy()
    # pd.to_timedelta returns ns-precision, but the example data is in second precision
    # so we need to fix this
    ds0["time"] = ds.time + np.timedelta64(1, "D")
    mm = m.get_mapper("out2.zarr")
    ds0.to_zarr(mm)  # old interface

    yield ds, ds0

    for path in ("out1.zarr", "out2.zarr"):
        m.rm(path, recursive=True)


@requires_zarr
@requires_fsspec
@pytest.mark.filterwarnings("ignore:deallocating CachingFileManager")
def test_open_fsspec(fsspec_memory_zarr_stores) -> None:
    _, ds0 = fsspec_memory_zarr_stores

    # single dataset
    url = "memory://out2.zarr"
    ds2 = open_dataset(url, engine="zarr")
    xr.testing.assert_equal(ds0, ds2)

    # single dataset with caching
    url = "simplecache::memory://out2.zarr"
    ds2 = open_dataset(url, engine="zarr")
    xr.testing.assert_equal(ds0, ds2)


@requires_zarr
@requires_fsspec
@requires_dask
@pytest.mark.filterwarnings("ignore:deallocating CachingFileManager")
def test_open_mfdataset_fsspec(fsspec_memory_zarr_stores) -> None:
    ds, ds0 = fsspec_memory_zarr_stores

    # multi dataset
    url = "memory://out*.zarr"
    ds2 = open_mfdataset(url, engine="zarr")
    xr.testing.assert_equal(xr.concat([ds, ds0], dim="time"), ds2)

    # multi dataset with caching
    url = "simplecache::memory://out*.zarr"
    ds2 = open_mfdataset(url, engine="zarr")
    xr.testing.assert_equal(xr.concat([ds, ds0], dim="time"), ds2)


@requires_zarr
@requires_dask
@pytest.mark.parametrize(
    "chunks", ["auto", -1, {}, {"x": "auto"}, {"x": -1}, {"x": "auto", "y": -1}]
)
def test_open_dataset_chunking_zarr(chunks, tmp_path: Path) -> None:
    encoded_chunks = 100
    dask_arr = dask_array_api.from_array(
        np.ones((500, 500), dtype="float64"), chunks=encoded_chunks
    )
    ds = xr.Dataset(
        {
            "test": xr.DataArray(
                dask_arr,
                dims=("x", "y"),
            )
        }
    )
    ds["test"].encoding["chunks"] = encoded_chunks
    ds.to_zarr(tmp_path / "test.zarr")

    with dask.config.set({"array.chunk-size": "1MiB"}):
        expected = ds.chunk(chunks)
        with open_dataset(
            tmp_path / "test.zarr", engine="zarr", chunks=chunks
        ) as actual:
            xr.testing.assert_chunks_equal(actual, expected)


@requires_zarr
@requires_dask
@pytest.mark.parametrize(
    "chunks", ["auto", -1, {}, {"x": "auto"}, {"x": -1}, {"x": "auto", "y": -1}]
)
@pytest.mark.filterwarnings("ignore:The specified chunks separate")
def test_chunking_consistency(chunks, tmp_path: Path) -> None:
    encoded_chunks: dict[str, Any] = {}
    dask_arr = dask_array_api.from_array(
        np.ones((500, 500), dtype="float64"), chunks=encoded_chunks
    )
    ds = xr.Dataset(
        {
            "test": xr.DataArray(
                dask_arr,
                dims=("x", "y"),
            )
        }
    )
    ds["test"].encoding["chunks"] = encoded_chunks
    ds.to_zarr(tmp_path / "test.zarr")
    ds.to_netcdf(tmp_path / "test.nc")

    with dask.config.set({"array.chunk-size": "1MiB"}):
        expected = ds.chunk(chunks)
        with xr.open_dataset(
            tmp_path / "test.zarr", engine="zarr", chunks=chunks
        ) as actual:
            xr.testing.assert_chunks_equal(actual, expected)

        with xr.open_dataset(tmp_path / "test.nc", chunks=chunks) as actual:
            xr.testing.assert_chunks_equal(actual, expected)


@requires_zarr
def test_zarr_entrypoint(tmp_path: Path) -> None:
    from xarray.backends.zarr import ZarrBackendEntrypoint

    entrypoint = ZarrBackendEntrypoint()
    ds = create_test_data()

    path = tmp_path / "foo.zarr"
    ds.to_zarr(path)
    _check_guess_can_open_and_open(entrypoint, path, engine="zarr", expected=ds)
    _check_guess_can_open_and_open(entrypoint, str(path), engine="zarr", expected=ds)

    # add a trailing slash to the path and check again
    _check_guess_can_open_and_open(
        entrypoint, str(path) + "/", engine="zarr", expected=ds
    )

    # Test the new functionality: .zarr with trailing slash
    assert entrypoint.guess_can_open("something-local.zarr")
    assert entrypoint.guess_can_open("something-local.zarr/")  # With trailing slash
    assert not entrypoint.guess_can_open("something-local.nc")
    assert not entrypoint.guess_can_open("not-found-and-no-extension")
    assert not entrypoint.guess_can_open("something.zarr.txt")


@requires_zarr
@requires_netCDF4
@pytest.mark.skipif(
    NETCDFC_VERSION is not None and NETCDFC_VERSION < Version("4.8.1"),
    reason="requires netcdf-c>=4.8.1",
)
# Bug in netcdf-c==4.8.1 (typo: Nan instead of NaN)
# https://github.com/Unidata/netcdf-c/issues/2265
@pytest.mark.skipif(
    platform.system() == "Windows" and NETCDFC_VERSION == Version("4.8.1"),
    reason="netcdf-c==4.8.1 has issues on Windows",
)
class TestNCZarr:
    def _create_nczarr(self, filename):
        ds = create_test_data()
        # Drop dim3: netcdf-c does not support dtype='<U1'
        # https://github.com/Unidata/netcdf-c/issues/2259
        ds = ds.drop_vars("dim3")

        # engine="netcdf4" is not required for backwards compatibility
        ds.to_netcdf(f"file://{filename}#mode=nczarr")
        return ds

    def test_open_nczarr(self) -> None:
        with create_tmp_file(suffix=".zarr") as tmp:
            expected = self._create_nczarr(tmp)
            actual = xr.open_zarr(tmp, consolidated=False)
            assert_identical(expected, actual)

    def test_overwriting_nczarr(self) -> None:
        with create_tmp_file(suffix=".zarr") as tmp:
            ds = self._create_nczarr(tmp)
            expected = ds[["var1"]]
            expected.to_zarr(tmp, mode="w")
            actual = xr.open_zarr(tmp, consolidated=False)
            assert_identical(expected, actual)

    @pytest.mark.skipif(
        NETCDFC_VERSION is not None and NETCDFC_VERSION > Version("4.8.1"),
        reason="netcdf-c>4.8.1 adds the _ARRAY_DIMENSIONS attribute",
    )
    @pytest.mark.parametrize("mode", ["a", "r+"])
    @pytest.mark.filterwarnings("ignore:.*non-consolidated metadata.*")
    def test_raise_writing_to_nczarr(self, mode) -> None:
        with create_tmp_file(suffix=".zarr") as tmp:
            ds = self._create_nczarr(tmp)
            with pytest.raises(
                KeyError, match="missing the attribute `_ARRAY_DIMENSIONS`,"
            ):
                ds.to_zarr(tmp, mode=mode)


@requires_zarr
@pytest.mark.usefixtures("default_zarr_format")
def test_zarr_closing_internal_zip_store():
    store_name = "tmp.zarr.zip"
    original_da = DataArray(np.arange(12).reshape((3, 4)))
    original_da.to_zarr(store_name, mode="w")

    with open_dataarray(store_name, engine="zarr") as loaded_da:
        assert_identical(original_da, loaded_da)


@requires_zarr
@pytest.mark.parametrize("create_default_indexes", [True, False])
def test_zarr_create_default_indexes(tmp_path, create_default_indexes) -> None:
    store_path = tmp_path / "tmp.zarr"
    original_ds = xr.Dataset({"data": ("x", np.arange(3))}, coords={"x": [-1, 0, 1]})
    original_ds.to_zarr(store_path, mode="w")

    with open_dataset(
        store_path, engine="zarr", create_default_indexes=create_default_indexes
    ) as loaded_ds:
        if create_default_indexes:
            assert list(loaded_ds.xindexes) == ["x"] and isinstance(
                loaded_ds.xindexes["x"], PandasIndex
            )
        else:
            assert len(loaded_ds.xindexes) == 0


@requires_zarr
@pytest.mark.usefixtures("default_zarr_format")
def test_raises_key_error_on_invalid_zarr_store(tmp_path):
    root = zarr.open_group(tmp_path / "tmp.zarr")
    root.create_array("bar", shape=(3, 5), dtype=np.float32)
    with pytest.raises(KeyError, match=r"xarray to determine variable dimensions"):
        xr.open_zarr(tmp_path / "tmp.zarr", consolidated=False)


@requires_zarr
@pytest.mark.usefixtures("default_zarr_format")
class TestZarrRegionAuto:
    """These are separated out since we should not need to test this logic with every store."""

    @contextlib.contextmanager
    def create_zarr_target(self):
        with create_tmp_file(suffix=".zarr") as tmp:
            yield tmp

    @contextlib.contextmanager
    def create(self):
        x = np.arange(0, 50, 10)
        y = np.arange(0, 20, 2)
        data = np.ones((5, 10))
        ds = xr.Dataset(
            {"test": xr.DataArray(data, dims=("x", "y"), coords={"x": x, "y": y})}
        )
        with self.create_zarr_target() as target:
            self.save(target, ds)
            yield target, ds

    def save(self, target, ds, **kwargs):
        ds.to_zarr(target, **kwargs)

    @pytest.mark.parametrize(
        "region",
        [
            pytest.param("auto", id="full-auto"),
            pytest.param({"x": "auto", "y": slice(6, 8)}, id="mixed-auto"),
        ],
    )
    def test_zarr_region_auto(self, region):
        with self.create() as (target, ds):
            ds_region = 1 + ds.isel(x=slice(2, 4), y=slice(6, 8))
            self.save(target, ds_region, region=region)
            ds_updated = xr.open_zarr(target)

            expected = ds.copy()
            expected["test"][2:4, 6:8] += 1
            assert_identical(ds_updated, expected)

    def test_zarr_region_auto_noncontiguous(self):
        with self.create() as (target, ds):
            with pytest.raises(ValueError):
                self.save(target, ds.isel(x=[0, 2, 3], y=[5, 6]), region="auto")

            dsnew = ds.copy()
            dsnew["x"] = dsnew.x + 5
            with pytest.raises(KeyError):
                self.save(target, dsnew, region="auto")

    def test_zarr_region_index_write(self, tmp_path):
        region: Mapping[str, slice] | Literal["auto"]
        region_slice = dict(x=slice(2, 4), y=slice(6, 8))

        with self.create() as (target, ds):
            ds_region = 1 + ds.isel(region_slice)
            for region in [region_slice, "auto"]:  # type: ignore[assignment]
                with patch.object(
                    ZarrStore,
                    "set_variables",
                    side_effect=ZarrStore.set_variables,
                    autospec=True,
                ) as mock:
                    self.save(target, ds_region, region=region, mode="r+")

                    # should write the data vars but never the index vars with auto mode
                    for call in mock.call_args_list:
                        written_variables = call.args[1].keys()
                        assert "test" in written_variables
                        assert "x" not in written_variables
                        assert "y" not in written_variables

    def test_zarr_region_append(self):
        with self.create() as (target, ds):
            x_new = np.arange(40, 70, 10)
            data_new = np.ones((3, 10))
            ds_new = xr.Dataset(
                {
                    "test": xr.DataArray(
                        data_new,
                        dims=("x", "y"),
                        coords={"x": x_new, "y": ds.y},
                    )
                }
            )

            # Now it is valid to use auto region detection with the append mode,
            # but it is still unsafe to modify dimensions or metadata using the region
            # parameter.
            with pytest.raises(KeyError):
                self.save(target, ds_new, mode="a", append_dim="x", region="auto")

    def test_zarr_region(self):
        with self.create() as (target, ds):
            ds_transposed = ds.transpose("y", "x")
            ds_region = 1 + ds_transposed.isel(x=[0], y=[0])
            self.save(target, ds_region, region={"x": slice(0, 1), "y": slice(0, 1)})

            # Write without region
            self.save(target, ds_transposed, mode="r+")

    @requires_dask
    def test_zarr_region_chunk_partial(self):
        """
        Check that writing to partial chunks with `region` fails, assuming `safe_chunks=False`.
        """
        ds = (
            xr.DataArray(np.arange(120).reshape(4, 3, -1), dims=list("abc"))
            .rename("var1")
            .to_dataset()
        )

        with self.create_zarr_target() as target:
            self.save(target, ds.chunk(5), compute=False, mode="w")
            with pytest.raises(ValueError):
                for r in range(ds.sizes["a"]):
                    self.save(
                        target, ds.chunk(3).isel(a=[r]), region=dict(a=slice(r, r + 1))
                    )

    @requires_dask
    def test_zarr_append_chunk_partial(self):
        t_coords = np.array([np.datetime64("2020-01-01").astype("datetime64[ns]")])
        data = np.ones((10, 10))

        da = xr.DataArray(
            data.reshape((-1, 10, 10)),
            dims=["time", "x", "y"],
            coords={"time": t_coords},
            name="foo",
        )
        new_time = np.array([np.datetime64("2021-01-01").astype("datetime64[ns]")])
        da2 = xr.DataArray(
            data.reshape((-1, 10, 10)),
            dims=["time", "x", "y"],
            coords={"time": new_time},
            name="foo",
        )

        with self.create_zarr_target() as target:
            self.save(target, da, mode="w", encoding={"foo": {"chunks": (5, 5, 1)}})

            with pytest.raises(ValueError, match="encoding was provided"):
                self.save(
                    target,
                    da2,
                    append_dim="time",
                    mode="a",
                    encoding={"foo": {"chunks": (1, 1, 1)}},
                )

            # chunking with dask sidesteps the encoding check, so we need a different check
            with pytest.raises(ValueError, match="Specified Zarr chunks"):
                self.save(
                    target,
                    da2.chunk({"x": 1, "y": 1, "time": 1}),
                    append_dim="time",
                    mode="a",
                )

    @pytest.mark.xfail(
        ON_WINDOWS,
        reason="Permission errors from Zarr: https://github.com/pydata/xarray/pull/10793",
    )
    @requires_dask
    def test_zarr_region_chunk_partial_offset(self):
        # https://github.com/pydata/xarray/pull/8459#issuecomment-1819417545
        with self.create_zarr_target() as store:
            data = np.ones((30,))
            da = xr.DataArray(
                data, dims=["x"], coords={"x": range(30)}, name="foo"
            ).chunk(x=10)
            self.save(store, da, compute=False)

            self.save(store, da.isel(x=slice(10)).chunk(x=(10,)), region="auto")

            self.save(
                store,
                da.isel(x=slice(5, 25)).chunk(x=(10, 10)),
                safe_chunks=False,
                region="auto",
            )

            with pytest.raises(ValueError):
                self.save(
                    store, da.isel(x=slice(5, 25)).chunk(x=(10, 10)), region="auto"
                )

    @requires_dask
    def test_zarr_safe_chunk_append_dim(self):
        with self.create_zarr_target() as store:
            data = np.ones((20,))
            da = xr.DataArray(
                data, dims=["x"], coords={"x": range(20)}, name="foo"
            ).chunk(x=5)

            self.save(store, da.isel(x=slice(0, 7)), safe_chunks=True, mode="w")
            with pytest.raises(ValueError):
                # If the first chunk is smaller than the border size then raise an error
                self.save(
                    store,
                    da.isel(x=slice(7, 11)).chunk(x=(2, 2)),
                    append_dim="x",
                    safe_chunks=True,
                )

            self.save(store, da.isel(x=slice(0, 7)), safe_chunks=True, mode="w")
            # If the first chunk is of the size of the border size then it is valid
            self.save(
                store,
                da.isel(x=slice(7, 11)).chunk(x=(3, 1)),
                safe_chunks=True,
                append_dim="x",
            )
            assert xr.open_zarr(store)["foo"].equals(da.isel(x=slice(0, 11)))

            self.save(store, da.isel(x=slice(0, 7)), safe_chunks=True, mode="w")
            # If the first chunk is of the size of the border size + N * zchunk then it is valid
            self.save(
                store,
                da.isel(x=slice(7, 17)).chunk(x=(8, 2)),
                safe_chunks=True,
                append_dim="x",
            )
            assert xr.open_zarr(store)["foo"].equals(da.isel(x=slice(0, 17)))

            self.save(store, da.isel(x=slice(0, 7)), safe_chunks=True, mode="w")
            with pytest.raises(ValueError):
                # If the first chunk is valid but the other are not then raise an error
                self.save(
                    store,
                    da.isel(x=slice(7, 14)).chunk(x=(3, 3, 1)),
                    append_dim="x",
                    safe_chunks=True,
                )

            self.save(store, da.isel(x=slice(0, 7)), safe_chunks=True, mode="w")
            with pytest.raises(ValueError):
                # If the first chunk have a size bigger than the border size but not enough
                # to complete the size of the next chunk then an error must be raised
                self.save(
                    store,
                    da.isel(x=slice(7, 14)).chunk(x=(4, 3)),
                    append_dim="x",
                    safe_chunks=True,
                )

            self.save(store, da.isel(x=slice(0, 7)), safe_chunks=True, mode="w")
            # Append with a single chunk it's totally valid,
            # and it does not matter the size of the chunk
            self.save(
                store,
                da.isel(x=slice(7, 19)).chunk(x=-1),
                append_dim="x",
                safe_chunks=True,
            )
            assert xr.open_zarr(store)["foo"].equals(da.isel(x=slice(0, 19)))

    @requires_dask
    @pytest.mark.parametrize("mode", ["r+", "a"])
    def test_zarr_safe_chunk_region(self, mode: Literal["r+", "a"]):
        with self.create_zarr_target() as store:
            arr = xr.DataArray(
                list(range(11)), dims=["a"], coords={"a": list(range(11))}, name="foo"
            ).chunk(a=3)
            self.save(store, arr, mode="w")

            with pytest.raises(ValueError):
                # There are two Dask chunks on the same Zarr chunk,
                # which means that it is unsafe in any mode
                self.save(
                    store,
                    arr.isel(a=slice(0, 3)).chunk(a=(2, 1)),
                    region="auto",
                    mode=mode,
                )

            with pytest.raises(ValueError):
                # the first chunk is covering the border size, but it is not
                # completely covering the second chunk, which means that it is
                # unsafe in any mode
                self.save(
                    store,
                    arr.isel(a=slice(1, 5)).chunk(a=(3, 1)),
                    region="auto",
                    mode=mode,
                )

            with pytest.raises(ValueError):
                # The first chunk is safe but the other two chunks are overlapping with
                # the same Zarr chunk
                self.save(
                    store,
                    arr.isel(a=slice(0, 5)).chunk(a=(3, 1, 1)),
                    region="auto",
                    mode=mode,
                )

            # Fully update two contiguous chunks is safe in any mode
            self.save(store, arr.isel(a=slice(3, 9)), region="auto", mode=mode)

            # The last chunk is considered full based on their current size (2)
            self.save(store, arr.isel(a=slice(9, 11)), region="auto", mode=mode)
            self.save(
                store, arr.isel(a=slice(6, None)).chunk(a=-1), region="auto", mode=mode
            )

            # Write the last chunk of a region partially is safe in "a" mode
            self.save(store, arr.isel(a=slice(3, 8)), region="auto", mode="a")
            with pytest.raises(ValueError):
                # with "r+" mode it is invalid to write partial chunk
                self.save(store, arr.isel(a=slice(3, 8)), region="auto", mode="r+")

            # This is safe with mode "a", the border size is covered by the first chunk of Dask
            self.save(
                store, arr.isel(a=slice(1, 4)).chunk(a=(2, 1)), region="auto", mode="a"
            )
            with pytest.raises(ValueError):
                # This is considered unsafe in mode "r+" because it is writing in a partial chunk
                self.save(
                    store,
                    arr.isel(a=slice(1, 4)).chunk(a=(2, 1)),
                    region="auto",
                    mode="r+",
                )

            # This is safe on mode "a" because there is a single dask chunk
            self.save(
                store, arr.isel(a=slice(1, 5)).chunk(a=(4,)), region="auto", mode="a"
            )
            with pytest.raises(ValueError):
                # This is unsafe on mode "r+", because the Dask chunk is partially writing
                # in the first chunk of Zarr
                self.save(
                    store,
                    arr.isel(a=slice(1, 5)).chunk(a=(4,)),
                    region="auto",
                    mode="r+",
                )

            # The first chunk is completely covering the first Zarr chunk
            # and the last chunk is a partial one
            self.save(
                store, arr.isel(a=slice(0, 5)).chunk(a=(3, 2)), region="auto", mode="a"
            )

            with pytest.raises(ValueError):
                # The last chunk is partial, so it is considered unsafe on mode "r+"
                self.save(
                    store,
                    arr.isel(a=slice(0, 5)).chunk(a=(3, 2)),
                    region="auto",
                    mode="r+",
                )

            # The first chunk is covering the border size (2 elements)
            # and also the second chunk (3 elements), so it is valid
            self.save(
                store, arr.isel(a=slice(1, 8)).chunk(a=(5, 2)), region="auto", mode="a"
            )

            with pytest.raises(ValueError):
                # The first chunk is not fully covering the first zarr chunk
                self.save(
                    store,
                    arr.isel(a=slice(1, 8)).chunk(a=(5, 2)),
                    region="auto",
                    mode="r+",
                )

            with pytest.raises(ValueError):
                # Validate that the border condition is not affecting the "r+" mode
                self.save(store, arr.isel(a=slice(1, 9)), region="auto", mode="r+")

            self.save(store, arr.isel(a=slice(10, 11)), region="auto", mode="a")
            with pytest.raises(ValueError):
                # Validate that even if we write with a single Dask chunk on the last Zarr
                # chunk it is still unsafe if it is not fully covering it
                # (the last Zarr chunk has size 2)
                self.save(store, arr.isel(a=slice(10, 11)), region="auto", mode="r+")

            # Validate the same as the above test but in the beginning of the last chunk
            self.save(store, arr.isel(a=slice(9, 10)), region="auto", mode="a")
            with pytest.raises(ValueError):
                self.save(store, arr.isel(a=slice(9, 10)), region="auto", mode="r+")

            self.save(
                store, arr.isel(a=slice(7, None)).chunk(a=-1), region="auto", mode="a"
            )
            with pytest.raises(ValueError):
                # Test that even a Dask chunk that covers the last Zarr chunk can be unsafe
                # if it is partial covering other Zarr chunks
                self.save(
                    store,
                    arr.isel(a=slice(7, None)).chunk(a=-1),
                    region="auto",
                    mode="r+",
                )

            with pytest.raises(ValueError):
                # If the chunk is of size equal to the one in the Zarr encoding, but
                # it is partially writing in the first chunk then raise an error
                self.save(
                    store,
                    arr.isel(a=slice(8, None)).chunk(a=3),
                    region="auto",
                    mode="r+",
                )

            with pytest.raises(ValueError):
                self.save(
                    store, arr.isel(a=slice(5, -1)).chunk(a=5), region="auto", mode="r+"
                )

            # Test if the code is detecting the last chunk correctly
            data = np.random.default_rng(0).random((2920, 25, 53))
            ds = xr.Dataset({"temperature": (("time", "lat", "lon"), data)})
            chunks = {"time": 1000, "lat": 25, "lon": 53}
            self.save(store, ds.chunk(chunks), compute=False, mode="w")
            region = {"time": slice(1000, 2000, 1)}
            chunk = ds.isel(region)
            chunk = chunk.chunk()
            self.save(store, chunk.chunk(), region=region)

    @requires_dask
    def test_dataset_to_zarr_align_chunks_true(self, tmp_store) -> None:
        # This test is a replica of the one in `test_dataarray_to_zarr_align_chunks_true`
        # but for datasets
        with self.create_zarr_target() as store:
            ds = (
                DataArray(
                    np.arange(4).reshape((2, 2)),
                    dims=["a", "b"],
                    coords={
                        "a": np.arange(2),
                        "b": np.arange(2),
                    },
                )
                .chunk(a=(1, 1), b=(1, 1))
                .to_dataset(name="foo")
            )

            self.save(
                store,
                ds,
                align_chunks=True,
                encoding={"foo": {"chunks": (3, 3)}},
                mode="w",
            )
            assert_identical(ds, xr.open_zarr(store))

            ds = (
                DataArray(
                    np.arange(4, 8).reshape((2, 2)),
                    dims=["a", "b"],
                    coords={
                        "a": np.arange(2),
                        "b": np.arange(2),
                    },
                )
                .chunk(a=(1, 1), b=(1, 1))
                .to_dataset(name="foo")
            )

            self.save(
                store,
                ds,
                align_chunks=True,
                region="auto",
            )
            assert_identical(ds, xr.open_zarr(store))
