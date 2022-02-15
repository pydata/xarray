import inspect
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
        if result.ndim + len(axis) != array.ndim and axis:
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

    @staticmethod
    def _update_default_params(func: callable, **kwargs) -> dict:
        """Let pydap decide on default parameter values

        Used in :meth:`open` to validate and update deviating defaults. For
        instance pydap has some defaults set in signature of
        :func:`pydap.client.open_url` (e.g. timeout or user_charset). These
        parameters are set to None in
        :meth:`PydapBackendEntrypoint.open_dataset` since the defaults in
        pydap may change. This function will check all additional keyword
        args that were provided and those, which are None, will be updated
        with pydaps defaults.

        This workaround is needed since xarray's plugin management prohibits
        to parse *args or **kwargs to the backends.
        """
        signature = inspect.signature(func)
        params = signature.parameters
        for key, value in kwargs.items():
            if not key in params:
                raise KeyError(f'Param {key} not supported bu {func}')
            elif value is None:
                kwargs[key] = params[key].default
        return kwargs

    @classmethod
    def open(cls, url, **kwargs):
        kwargs = cls._update_default_params(pydap.client.open_url, **kwargs)
        ds = pydap.client.open_url(
            url=url,
            **kwargs
        )
        return cls(ds)

    def open_store_variable(self, var):
        data = indexing.LazilyIndexedArray(PydapArrayWrapper(var))
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
    available = has_pydap

    def guess_can_open(self, filename_or_obj):
        return isinstance(filename_or_obj, str) and is_remote_uri(filename_or_obj)

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
        application=None, # uses pydap's default if None
        session=None, # uses pydap's default if None
        output_grid=None, # uses pydap's default if None
        timeout=None, # uses pydap's default if None
        verify=None, # uses pydap's default if None
        user_charset=None # uses pydap's default if None
    ):

        store = PydapDataStore.open(
            filename_or_obj,
            application=application,
            session=session,
            output_grid=output_grid,
            timeout=timeout,
            verify=verify,
            user_charset=user_charset
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


BACKEND_ENTRYPOINTS["pydap"] = PydapBackendEntrypoint
