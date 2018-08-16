"""Backend objects for saving and loading data

DataStores provide a uniform interface for saving and loading data in different
formats. They should not be used directly, but rather through Dataset objects.
"""
from .common import AbstractDataStore
from .memory import InMemoryDataStore
from .netCDF4_ import NetCDF4DataStore
from .pydap_ import PydapDataStore
from .pynio_ import NioDataStore
from .scipy_ import ScipyDataStore
from .h5netcdf_ import H5NetCDFStore
from .pseudonetcdf_ import PseudoNetCDFDataStore
from .zarr import ZarrStore

__all__ = [
    'AbstractDataStore',
    'InMemoryDataStore',
    'NetCDF4DataStore',
    'PydapDataStore',
    'NioDataStore',
    'ScipyDataStore',
    'H5NetCDFStore',
    'ZarrStore',
    'PseudoNetCDFDataStore',
]
