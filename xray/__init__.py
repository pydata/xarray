from .variable import Variable, Coordinate
from .dataset import Dataset, open_dataset
from .data_array import DataArray, align
from .utils import xarray_equal
from .utils import class_alias as _class_alias

from .version import version as __version__

# TODO: remove these when users of pre-release versions have switched over
DatasetArray = _class_alias(DataArray, 'DatasetArray')
XArray = _class_alias(Variable, 'XArray')

# TODO: define a global "concat" function to provide a uniform interface
