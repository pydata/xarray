from __future__ import annotations

import io
import re
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

import xarray as xr
from xarray.backends.common import _infer_dtype, robust_getitem
from xarray.tests import (
    requires_h5netcdf,
    requires_netCDF4,
    requires_pydap,
    requires_scipy,
    requires_zarr,
)


class DummyFailure(Exception):
    pass


class DummyArray:
    def __init__(self, failures):
        self.failures = failures

    def __getitem__(self, key):
        if self.failures:
            self.failures -= 1
            raise DummyFailure
        return "success"


def test_robust_getitem() -> None:
    array = DummyArray(failures=2)
    with pytest.raises(DummyFailure):
        array[...]
    result = robust_getitem(array, ..., catch=DummyFailure, initial_delay=1)
    assert result == "success"

    array = DummyArray(failures=3)
    with pytest.raises(DummyFailure):
        robust_getitem(array, ..., catch=DummyFailure, initial_delay=1, max_retries=2)


@pytest.mark.parametrize(
    "data",
    [
        np.array([["ab", "cdef", b"X"], [1, 2, "c"]], dtype=object),
        np.array([["x", 1], ["y", 2]], dtype="object"),
    ],
)
def test_infer_dtype_error_on_mixed_types(data):
    with pytest.raises(ValueError, match="unable to infer dtype on variable"):
        _infer_dtype(data, "test")


@requires_scipy
def test_encoding_failure_note():
    # Create an arbitrary value that cannot be encoded in netCDF3
    ds = xr.Dataset({"invalid": np.array([2**63 - 1], dtype=np.int64)})
    f = io.BytesIO()
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Raised while encoding variable 'invalid' with value <xarray.Variable"
        ),
    ):
        ds.to_netcdf(f, engine="scipy")


class TestCommon:
    def test_robust_getitem(self) -> None:
        class UnreliableArrayFailure(Exception):
            pass

        class UnreliableArray:
            def __init__(self, array, failures=1):
                self.array = array
                self.failures = failures

            def __getitem__(self, key):
                if self.failures > 0:
                    self.failures -= 1
                    raise UnreliableArrayFailure
                return self.array[key]

        array = UnreliableArray([0])
        with pytest.raises(UnreliableArrayFailure):
            array[0]
        assert array[0] == 0

        actual = robust_getitem(array, 0, catch=UnreliableArrayFailure, initial_delay=0)
        assert actual == 0


def _memory_store() -> Any:
    from zarr.storage import MemoryStore

    return MemoryStore()


def _store_path() -> Any:
    from zarr.storage import MemoryStore, StorePath

    return StorePath(MemoryStore())


@pytest.mark.parametrize(
    "engine",
    [
        pytest.param("netcdf4", marks=requires_netCDF4),
        pytest.param("h5netcdf", marks=requires_h5netcdf),
        pytest.param("scipy", marks=requires_scipy),
        pytest.param("pydap", marks=requires_pydap),
    ],
)
@pytest.mark.parametrize(
    "make_store",
    [
        pytest.param(dict, id="dict"),
        pytest.param(_memory_store, id="MemoryStore", marks=requires_zarr),
        pytest.param(_store_path, id="StorePath", marks=requires_zarr),
    ],
)
def test_open_zarr_store_with_other_engine(
    engine: str, make_store: Callable[[], Any]
) -> None:
    with pytest.raises(TypeError, match=r"cannot open zarr stores.*engine='zarr'"):
        xr.open_dataset(make_store(), engine=engine)
    if engine != "scipy":
        with pytest.raises(TypeError, match=r"cannot open zarr stores.*engine='zarr'"):
            xr.open_datatree(make_store(), engine=engine)
