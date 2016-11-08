import numpy as np
import pandas as pd
import pytest
import xarray as xr

distributed = pytest.importorskip('distributed')
da = pytest.importorskip('dask.array')
from distributed.protocol import serialize, deserialize
from distributed.utils_test import cluster, loop, gen_cluster

from xarray.test.test_backends import create_tmp_file
from xarray.test.test_dataset import create_test_data


def test_dask_distributed_integration_test(loop):
    with cluster() as (s, _):
        with distributed.Client(('127.0.0.1', s['port']), loop=loop) as client:
            original = create_test_data()
            # removing the line below results in a test that never times out!
            del original['time']
            with create_tmp_file() as filename:
                original.to_netcdf(filename, engine='netcdf4')
                # TODO: should be able to serialize locks?
                # TODO: should be able to serialize array types from
                # xarray.conventions
                restored = xr.open_dataset(filename, chunks=3, lock=False)
                assert isinstance(restored.var1.data, da.Array)
                restored.load()
                assert original.identical(restored)


@gen_cluster(client=True)
def test_dask_distributed_integration_test_fast(c, s, a, b):
    values = [10, 20, 30]
    values = [0.2, 1.5, 1.8]
    values = ['ab', 'cd', 'ef']
    # does not work: ValueError: cannot include dtype 'M' in a buffer
    # values = pd.date_range('2010-01-01', periods=3).values
    original = xr.Dataset({'foo': ('x', values)})
    engine = 'netcdf4'
    # does not work: we don't know how to pickle h5netcdf objects, which wrap
    # h5py datasets/files
    # engine = 'h5netcdf'
    with create_tmp_file() as filename:
        original.to_netcdf(filename, engine=engine)
        # TODO: should be able to serialize locks?
        # TODO: should be able to serialize array types from
        # xarray.conventions
        restored = xr.open_dataset(filename, chunks=5, lock=False,
                                   engine=engine)
        print(restored.foo.data.dask)
        foo = c.compute(restored.foo.data)
        foo = yield foo._result()
        computed = xr.Dataset({'foo': ('x', foo)})
        assert computed.identical(original)
