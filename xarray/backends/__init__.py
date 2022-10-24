"""Backend objects for saving and loading data

DataStores provide a uniform interface for saving and loading data in different
formats. They should not be used directly, but rather through Dataset objects.
"""
from xarray.backends.cfgrib_ import CfGribDataStore
from xarray.backends.common import AbstractDataStore, BackendArray, BackendEntrypoint
from xarray.backends.file_manager import (
    CachingFileManager,
    DummyFileManager,
    FileManager,
)
from xarray.backends.h5netcdf_ import H5NetCDFStore
from xarray.backends.memory import InMemoryDataStore
from xarray.backends.netCDF4_ import NetCDF4DataStore
from xarray.backends.plugins import list_engines
from xarray.backends.pseudonetcdf_ import PseudoNetCDFDataStore
from xarray.backends.pydap_ import PydapDataStore
from xarray.backends.pynio_ import NioDataStore
from xarray.backends.scipy_ import ScipyDataStore
from xarray.backends.zarr import ZarrStore

__all__ = [
    "AbstractDataStore",
    "BackendArray",
    "BackendEntrypoint",
    "FileManager",
    "CachingFileManager",
    "CfGribDataStore",
    "DummyFileManager",
    "InMemoryDataStore",
    "NetCDF4DataStore",
    "PydapDataStore",
    "NioDataStore",
    "ScipyDataStore",
    "H5NetCDFStore",
    "ZarrStore",
    "PseudoNetCDFDataStore",
    "list_engines",
]
