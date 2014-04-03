from .xarray import as_xarray, XArray, CoordXArray, broadcast_xarrays
from .conventions import decode_cf_datetime, encode_cf_datetime
from .dataset import Dataset, open_dataset
from .dataset_array import DataArray, align
from .utils import orthogonal_indexer, xarray_equal

from .version import version as __version__

# TODO: remove this when our users have switched over
DatasetArray = DataArray

# TODO: define a global "concat" function to provide a uniform interface
