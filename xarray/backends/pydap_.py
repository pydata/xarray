import numpy as np

from ..core import indexing
from ..core.pycompat import integer_types
from ..core.utils import Frozen, FrozenDict, close_on_error, is_dict_like, is_remote_uri
from ..core.variable import Variable
from .common import (
    BACKEND_ENTRYPOINTS,
    AbstractDataStore,
    BackendArray,
    BackendEntrypoint,
    robust_getitem,
)
from .store import StoreBackendEntrypoint

try:
    import pydap.client

    has_pydap = True
except ModuleNotFoundError:
    has_pydap = False


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
        return indexing.explicit_indexing_adapter(
            key, self.shape, indexing.IndexingSupport.BASIC, self._getitem
        )

    def _getitem(self, key):
        # pull the data from the array attribute if possible, to avoid
        # downloading coordinate data twice
        array = getattr(self.array, "array", self.array)
        result = robust_getitem(array, key, catch=ValueError)
        # in some cases, pydap doesn't squeeze axes automatically like numpy
        axis = tuple(n for n, k in enumerate(key) if isinstance(k, integer_types))
        if result.ndim + len(axis) != array.ndim and len(axis) > 0:
            result = np.squeeze(result, axis)

        return result


def _fix_attributes(attributes):
    attributes = dict(attributes)
    for k in list(attributes):
        if k.lower() == "global" or k.lower().endswith("_global"):
            # move global attributes to the top level, like the netcdf-C
            # DAP client
            attributes.update(attributes.pop(k))
        elif is_dict_like(attributes[k]):
            # Make Hierarchical attributes to a single level with a
            # dot-separated key
            attributes.update(
                {
                    f"{k}.{k_child}": v_child
                    for k_child, v_child in attributes.pop(k).items()
                }
            )
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

        ds = pydap.client.open_url(url, session=session)
        return cls(ds)

    def open_store_variable(self, var):
        data = indexing.LazilyOuterIndexedArray(PydapArrayWrapper(var))
        return Variable(var.dimensions, data, _fix_attributes(var.attributes))

    def get_variables(self):
        return FrozenDict(
            (k, self.open_store_variable(self.ds[k])) for k in self.ds.keys()
        )

    def get_attrs(self):
        return Frozen(_fix_attributes(self.ds.attributes))

    def get_dimensions(self):
        return Frozen(self.ds.dimensions)


class PydapBackendEntrypoint(BackendEntrypoint):
    def guess_can_open(self, store_spec):
        return isinstance(store_spec, str) and is_remote_uri(store_spec)

    def open_dataset(
        self,
        filename_or_obj,
        mask_and_scale=True,
        decode_times=True,
        concat_characters=True,
        decode_coords=True,
        drop_variables=None,
        use_cftime=None,
        decode_timedelta=None,
        session=None,
    ):
        store = PydapDataStore.open(
            filename_or_obj,
            session=session,
        )

        store_entrypoint = StoreBackendEntrypoint()
        with close_on_error(store):
            ds = store_entrypoint.open_dataset(
                store,
                mask_and_scale=mask_and_scale,
                decode_times=decode_times,
                concat_characters=concat_characters,
                decode_coords=decode_coords,
                drop_variables=drop_variables,
                use_cftime=use_cftime,
                decode_timedelta=decode_timedelta,
            )
            return ds


if has_pydap:
    BACKEND_ENTRYPOINTS["pydap"] = PydapBackendEntrypoint
