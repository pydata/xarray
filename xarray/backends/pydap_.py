from __future__ import absolute_import, division, print_function

import numpy as np

from .. import Variable
from ..core import indexing
from ..core.pycompat import integer_types
from ..core.utils import Frozen, FrozenOrderedDict, is_dict_like
from .common import AbstractDataStore, BackendArray, robust_getitem


class PydapArrayWrapper(BackendArray):
    def __init__(self, array):
        self.array = array

    @property
    def shape(self):
        return self.array.shape

    @property
    def dtype(self):
        return self.array.dtype

    def __getitem__(self, key):
        key, np_inds = indexing.decompose_indexer(
            key, self.shape, indexing.IndexingSupport.BASIC)

        # pull the data from the array attribute if possible, to avoid
        # downloading coordinate data twice
        array = getattr(self.array, 'array', self.array)
        result = robust_getitem(array, key.tuple, catch=ValueError)
        # pydap doesn't squeeze axes automatically like numpy
        axis = tuple(n for n, k in enumerate(key.tuple)
                     if isinstance(k, integer_types))
        if len(axis) > 0:
            result = np.squeeze(result, axis)

        if len(np_inds.tuple) > 0:
            result = indexing.NumpyIndexingAdapter(np.asarray(result))[np_inds]

        return result


def _fix_attributes(attributes):
    attributes = dict(attributes)
    for k in list(attributes):
        if k.lower() == 'global' or k.lower().endswith('_global'):
            # move global attributes to the top level, like the netcdf-C
            # DAP client
            attributes.update(attributes.pop(k))
        elif is_dict_like(attributes[k]):
            # Make Hierarchical attributes to a single level with a
            # dot-separated key
            attributes.update({'{}.{}'.format(k, k_child): v_child for
                               k_child, v_child in attributes.pop(k).items()})
    return attributes


class PydapDataStore(AbstractDataStore):
    """Store for accessing OpenDAP datasets with pydap.

    This store provides an alternative way to access OpenDAP datasets that may
    be useful if the netCDF4 library is not available.
    """

    def __init__(self, ds):
        """
        Parameters
        ----------
        ds : pydap DatasetType
        """
        self.ds = ds

    @classmethod
    def open(cls, url, session=None):
        import pydap.client
        ds = pydap.client.open_url(url, session=session)
        return cls(ds)

    def open_store_variable(self, var):
        data = indexing.LazilyOuterIndexedArray(PydapArrayWrapper(var))
        return Variable(var.dimensions, data,
                        _fix_attributes(var.attributes))

    def get_variables(self):
        return FrozenOrderedDict((k, self.open_store_variable(self.ds[k]))
                                 for k in self.ds.keys())

    def get_attrs(self):
        return Frozen(_fix_attributes(self.ds.attributes))

    def get_dimensions(self):
        return Frozen(self.ds.dimensions)
