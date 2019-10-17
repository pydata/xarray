"""Backend objects for saving and loading data

DataStores provide a uniform interface for saving and loading data in different
formats. They should not be used directly, but rather through Dataset objects.
"""
from .cfgrib_ import CfGribDataStore
from .common import AbstractDataStore
from .file_manager import CachingFileManager, DummyFileManager, FileManager
from .h5netcdf_ import H5NetCDFStore
from .memory import InMemoryDataStore
from .netCDF4_ import NetCDF4DataStore
from .pseudonetcdf_ import PseudoNetCDFDataStore
from .pydap_ import PydapDataStore
from .pynio_ import NioDataStore
from .scipy_ import ScipyDataStore
from .zarr import ZarrStore

__all__ = [
    "AbstractDataStore",
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
]
