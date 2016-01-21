import warnings
from xarray import *
from xarray import __version__

warnings.warn('xray has been renamed to xarray! Please install xarray to get '
              'future releases, and update your imports from "import xray" to '
              '"import xarray as xr". See http://xarray.pydata.org for more '
              'details.', DeprecationWarning)
