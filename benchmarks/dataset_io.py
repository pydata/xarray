from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd

import xarray as xr

from . import randn


class IOSingleNetCDF(object):
    """
    A few examples that benchmark reading/writing a single netCDF file with
    xarray
    """

    timeout = 300.

    def setup(self):

        # single Dataset
        self.ds = xr.Dataset()
        nt = int(4*365.25)
        nx = 90
        ny = 45

        times = pd.date_range('1970-01-01', periods=nt, freq='D')
        lons = xr.DataArray(np.linspace(0, 360, nx), dims=('lon', ),
                            attrs={'units': 'degrees east',
                                   'long_name': 'longitude'})
        lats = xr.DataArray(np.linspace(-90, 90, ny), dims=('lat', ),
                            attrs={'units': 'degrees north',
                                   'long_name': 'latitude'})
        self.ds['foo'] = xr.DataArray(randn((nt, nx, ny), frac_nan=0.2),
                                      coords={'lon': lons, 'lat': lats,
                                              'time': times},
                                      dims=('time', 'lon', 'lat'),
                                      name='foo', encoding=None,
                                      attrs={'units': 'foo units',
                                             'description': 'a description'})
        self.ds['bar'] = xr.DataArray(randn((nt, nx, ny), frac_nan=0.2),
                                      coords={'lon': lons, 'lat': lats,
                                              'time': times},
                                      dims=('time', 'lon', 'lat'),
                                      name='bar', encoding=None,
                                      attrs={'units': 'bar units',
                                             'description': 'a description'})
        self.ds['baz'] = xr.DataArray(randn((nx, ny), frac_nan=0.2).astype(
            np.float32),
                                      coords={'lon': lons, 'lat': lats},
                                      dims=('lon', 'lat'),
                                      name='baz', encoding=None,
                                      attrs={'units': 'baz units',
                                             'description': 'a description'})

        self.ds.attrs = {'history': 'created for xarray benchmarking'}

        self.filepath = 'test_single_file.nc'
        print(self.filepath, self.ds.nbytes / 1e9, 'GB')
        self.ds.to_netcdf(self.filepath, format='NETCDF3_64BIT')

    def time_load_dataset_netcdf4(self):
        xr.open_dataset(self.filepath, engine='netcdf4').load()
        pass

    def time_load_dataset_scipy(self):
        xr.open_dataset(self.filepath, engine='scipy').load()
        pass

    def time_write_dataset_netcdf4(self):
        self.ds.to_netcdf('test_netcdf4_write.nc', engine='netcdf4')
        pass

    def time_write_dataset_scipy(self):
        self.ds.to_netcdf('test_scipy_write.nc', engine='scipy')
        pass
