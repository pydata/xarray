"""Backend objects for saving and loading data

DataStores provide a uniform interface for saving and loading data in different
formats. They should not be used directly, but rather through Dataset objects.
"""
from .common import AbstractDataStore
from .memory import InMemoryDataStore
from .netCDF4_ import NetCDF4DataStore
from .pydap_ import PydapDataStore
from .scipy_ import ScipyDataStore
from .h5netcdf_ import H5NetCDFStore
from .pynio import NioDataStore
