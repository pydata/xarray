from .core.alignment import align, broadcast, broadcast_arrays
from .core.combine import concat, auto_combine
from .core.variable import Variable, Coordinate
from .core.dataset import Dataset
from .core.dataarray import DataArray
from .core.options import set_options

from .backends.api import open_dataset, open_mfdataset, save_mfdataset
from .conventions import decode_cf

try:
    from .version import version as __version__
except ImportError:  # pragma: no cover
    raise ImportError('xarray not properly installed. If you are running from '
                      'the source directory, please instead create a new '
                      'virtual environment (using conda or virtualenv) and '
                      'then install it in-place by running: pip install -e .')

from . import tutorial
