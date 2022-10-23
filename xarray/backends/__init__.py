"""Backend objects for saving and loading data

DataStores provide a uniform interface for saving and loading data in different
formats. They should not be used directly, but rather through Dataset objects.
"""
from .cfgrib_ import CfGribDataStore
from .common import AbstractDataStore, BackendArray, BackendEntrypoint
from .file_manager import CachingFileManager, DummyFileManager, FileManager
from .h5netcdf_ import H5netcdfBackendEntrypoint, H5NetCDFStore
from .memory import InMemoryDataStore
from .netCDF4_ import NetCDF4BackendEntrypoint, NetCDF4DataStore
from .plugins import list_engines
from .pseudonetcdf_ import PseudoNetCDFBackendEntrypoint, PseudoNetCDFDataStore
from .pydap_ import PydapBackendEntrypoint, PydapDataStore
from .pynio_ import NioDataStore
from .scipy_ import ScipyBackendEntrypoint, ScipyDataStore
from .store import StoreBackendEntrypoint
from .zarr import ZarrBackendEntrypoint, ZarrStore

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
    "H5netcdfBackendEntrypoint",
    "NetCDF4BackendEntrypoint",
    "PseudoNetCDFBackendEntrypoint",
    "PydapBackendEntrypoint",
    "ScipyBackendEntrypoint",
    "StoreBackendEntrypoint",
    "ZarrBackendEntrypoint",
    "list_engines",
]
