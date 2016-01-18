import warnings
from xarray import *
from xarray import __version__

warnings.warn('xray has been renamed to xarray! Install xarray to get future '
			  'releases. This will be the last xray release. See '
			  'http://xarray.pydata.org for more details.',
			  DeprecationWarning)
