from .variable import Variable, XIndex, Index, Coordinate
from .dataset import Dataset, open_dataset
from .data_array import DataArray, align

from .version import version as __version__

# TODO: define a global "concat" function to provide a uniform interface
