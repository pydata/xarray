from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd

try:
    import dask
    import dask.multiprocessing
except ImportError:
    pass

import xarray as xr

from . import randn, requires_dask


class IOSingleNetCDF(object):
    """
    A few examples that benchmark reading/writing a single netCDF file with
    xarray
    """

    timeout = 300.
    repeat = 1
    number = 5

    def make_ds(self):

        # single Dataset
        self.ds = xr.Dataset()
        self.nt = 1000
        self.nx = 90
        self.ny = 45

        self.block_chunks = {'time': self.nt / 4,
                             'lon': self.nx / 3,
                             'lat': self.ny / 3}

        self.time_chunks = {'time': int(self.nt / 36)}

        times = pd.date_range('1970-01-01', periods=self.nt, freq='D')
        lons = xr.DataArray(np.linspace(0, 360, self.nx), dims=('lon', ),
                            attrs={'units': 'degrees east',
                                   'long_name': 'longitude'})
        lats = xr.DataArray(np.linspace(-90, 90, self.ny), dims=('lat', ),
                            attrs={'units': 'degrees north',
                                   'long_name': 'latitude'})
        self.ds['foo'] = xr.DataArray(randn((self.nt, self.nx, self.ny),
                                            frac_nan=0.2),
                                      coords={'lon': lons, 'lat': lats,
                                              'time': times},
                                      dims=('time', 'lon', 'lat'),
                                      name='foo', encoding=None,
                                      attrs={'units': 'foo units',
                                             'description': 'a description'})
        self.ds['bar'] = xr.DataArray(randn((self.nt, self.nx, self.ny),
                                            frac_nan=0.2),
                                      coords={'lon': lons, 'lat': lats,
                                              'time': times},
                                      dims=('time', 'lon', 'lat'),
                                      name='bar', encoding=None,
                                      attrs={'units': 'bar units',
                                             'description': 'a description'})
        self.ds['baz'] = xr.DataArray(randn((self.nx, self.ny),
                                            frac_nan=0.2).astype(np.float32),
                                      coords={'lon': lons, 'lat': lats},
                                      dims=('lon', 'lat'),
                                      name='baz', encoding=None,
                                      attrs={'units': 'baz units',
                                             'description': 'a description'})

        self.ds.attrs = {'history': 'created for xarray benchmarking'}


class IOWriteSingleNetCDF3(IOSingleNetCDF):
    def setup(self):
        self.format = 'NETCDF3_64BIT'
        self.make_ds()

    def time_write_dataset_netcdf4(self):
        self.ds.to_netcdf('test_netcdf4_write.nc', engine='netcdf4',
                          format=self.format)

    def time_write_dataset_scipy(self):
        self.ds.to_netcdf('test_scipy_write.nc', engine='scipy',
                          format=self.format)


class IOReadSingleNetCDF4(IOSingleNetCDF):
    def setup(self):

        self.make_ds()

        self.filepath = 'test_single_file.nc4.nc'
        self.format = 'NETCDF4'
        self.ds.to_netcdf(self.filepath, format=self.format)

    def time_load_dataset_netcdf4(self):
        xr.open_dataset(self.filepath, engine='netcdf4').load()


class IOReadSingleNetCDF3(IOReadSingleNetCDF4):
    def setup(self):

        self.make_ds()

        self.filepath = 'test_single_file.nc3.nc'
        self.format = 'NETCDF3_64BIT'
        self.ds.to_netcdf(self.filepath, format=self.format)

    def time_load_dataset_scipy(self):
        xr.open_dataset(self.filepath, engine='scipy').load()


class IOReadSingleNetCDF4Dask(IOSingleNetCDF):
    def setup(self):

        requires_dask()

        self.make_ds()

        self.filepath = 'test_single_file.nc4.nc'
        self.format = 'NETCDF4'
        self.ds.to_netcdf(self.filepath, format=self.format)

    def time_load_dataset_netcdf4_with_block_chunks(self):
        xr.open_dataset(self.filepath, engine='netcdf4',
                        chunks=self.block_chunks).load()

    def time_load_dataset_netcdf4_with_block_chunks_multiprocessing(self):
        with dask.set_options(get=dask.multiprocessing.get):
            xr.open_dataset(self.filepath, engine='netcdf4',
                            chunks=self.block_chunks).load()

    def time_load_dataset_netcdf4_with_time_chunks(self):
        xr.open_dataset(self.filepath, engine='netcdf4',
                        chunks=self.time_chunks).load()

    def time_load_dataset_netcdf4_with_time_chunks_multiprocessing(self):
        with dask.set_options(get=dask.multiprocessing.get):
            xr.open_dataset(self.filepath, engine='netcdf4',
                            chunks=self.time_chunks).load()


class IOReadSingleNetCDF3Dask(IOReadSingleNetCDF4Dask):
    def setup(self):

        requires_dask()

        self.make_ds()

        self.filepath = 'test_single_file.nc3.nc'
        self.format = 'NETCDF3_64BIT'
        self.ds.to_netcdf(self.filepath, format=self.format)

    def time_load_dataset_scipy_with_block_chunks(self):
        with dask.set_options(get=dask.multiprocessing.get):
            xr.open_dataset(self.filepath, engine='scipy',
                            chunks=self.block_chunks).load()

    def time_load_dataset_scipy_with_time_chunks(self):
        with dask.set_options(get=dask.multiprocessing.get):
            xr.open_dataset(self.filepath, engine='scipy',
                            chunks=self.time_chunks).load()
