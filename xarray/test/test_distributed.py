import numpy as np
import pytest
import xarray as xr

distributed = pytest.importorskip('distributed')
da = pytest.importorskip('dask.array')
from distributed.protocol import serialize, deserialize
from distributed.utils_test import cluster, loop, gen_cluster

from xarray.core.indexing import LazilyIndexedArray
from xarray.backends.netCDF4_ import NetCDF4ArrayWrapper
from xarray.conventions import CharToStringArray

from xarray.test.test_backends import create_tmp_file


def test_serialize_deserialize_lazily_indexed_array():
    original = LazilyIndexedArray(np.arange(10))[:5]
    restored = deserialize(*serialize(original))
    assert type(restored) is type(original)
    assert (restored.array == original.array).all()
    assert restored.key == original.key


def test_serialize_deserialize_netcdf4_array_wrapper():
    original = NetCDF4ArrayWrapper(np.arange(10), is_remote=False)
    restored = deserialize(*serialize(original))
    assert type(restored) is type(original)
    assert (restored.array == original.array).all()
    assert restored.is_remote == original.is_remote


def test_serialize_deserialize_char_to_string_array():
    original = CharToStringArray(np.array(['a', 'b', 'c'], dtype='S1'))
    restored = deserialize(*serialize(original))
    assert type(restored) is type(original)
    assert (restored.array == original.array).all()


def test_serialize_deserialize_nested_arrays():
    original = LazilyIndexedArray(NetCDF4ArrayWrapper(np.arange(5)))
    restored = deserialize(*serialize(original))
    assert (restored.array.array == original.array.array).all()


def test_dask_distributed_integration_test(loop):
    with cluster() as (s, _):
        with distributed.Client(('127.0.0.1', s['port']), loop=loop) as client:
            original = xr.Dataset({'foo': ('x', [10, 20, 30, 40, 50])})
            with create_tmp_file() as filename:
                original.to_netcdf(filename, engine='netcdf4')
                # TODO: should be able to serialize locks?
                # TODO: should be able to serialize array types from
                # xarray.conventions
                restored = xr.open_dataset(filename, chunks=3, lock=False)
                assert isinstance(restored.foo.data, da.Array)
                restored.load()
                assert original.identical(restored)


@gen_cluster(client=True)
def test_dask_distributed_integration_test_fast(c, s, a, b):
    original = xr.Dataset({'foo': ('x', [10, 20, 30, 40, 50])})
    with create_tmp_file() as filename:
        original.to_netcdf(filename, engine='netcdf4')
        # TODO: should be able to serialize locks?
        # TODO: should be able to serialize array types from
        # xarray.conventions
        restored = xr.open_dataset(filename, chunks=3, lock=False, decode_cf=False)
        print(restored.foo.data.dask)
        y = c.compute(restored.foo.data)
        y = yield y._result()
        computed = xr.Dataset({'foo': ('x', y)})
        assert computed.identical(original)
