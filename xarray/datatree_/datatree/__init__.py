# flake8: noqa
# Ignoring F401: imported but unused

from pkg_resources import DistributionNotFound, get_distribution

# import public API
from .datatree import DataTree
from .io import open_datatree
from .mapping import TreeIsomorphismError, map_over_subtree

try:
    __version__ = get_distribution(__name__).version
except DistributionNotFound:  # noqa: F401; pragma: no cover
    # package is not installed
    pass
