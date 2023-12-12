from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from xarray.backends.common import (
    BACKEND_ENTRYPOINTS,
    AbstractDataStore,
    BackendArray,
    BackendEntrypoint,
    robust_getitem,
)
from xarray.backends.store import StoreBackendEntrypoint
from xarray.core import indexing
from xarray.core.pycompat import integer_types
from xarray.core.utils import (
    Frozen,
    FrozenDict,
    close_on_error,
    is_dict_like,
    is_remote_uri,
)
from xarray.core.variable import Variable

if TYPE_CHECKING:
    import pydap.model
    import requests

    from xarray.core.dataset import Dataset
    from xarray.core.types import Self, T_XarrayCanOpen


class PydapArrayWrapper(BackendArray):
    def __init__(self, array):
        self.array = array

    @property
    def shape(self) -> tuple[int, ...]:
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
        result = np.asarray(result)
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

    ds: pydap.model.DatasetType

    def __init__(self, ds: pydap.model.DatasetType) -> None:
        """
        Parameters
        ----------
        ds : pydap DatasetType
        """
        self.ds = ds

    @classmethod
    def open(
        cls,
        url: str,
        application: Any = None,
        session: requests.Session | None = None,
        output_grid: bool | None = None,
        timeout: float | None = None,
        verify: bool | None = None,
        user_charset: str | None = None,
    ) -> Self:
        import pydap.client
        import pydap.lib

        if timeout is None:
            from pydap.lib import DEFAULT_TIMEOUT

            timeout = DEFAULT_TIMEOUT

        kwargs = {
            "url": url,
            "application": application,
            "session": session,
            "output_grid": output_grid or True,
            "timeout": timeout,
        }
        if verify is not None:
            kwargs.update({"verify": verify})
        if user_charset is not None:
            kwargs.update({"user_charset": user_charset})
        ds = pydap.client.open_url(**kwargs)
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


@dataclass(repr=False)
class PydapBackendEntrypoint(BackendEntrypoint):
    """
    Backend for steaming datasets over the internet using
    the Data Access Protocol, also known as DODS or OPeNDAP
    based on the pydap package.

    This backend is selected by default for urls.

    For more information about the underlying library, visit:
    https://www.pydap.org

    Parameters
    ----------
    application:
    session:
    output_grid:
    timeout: float or None, default: 120
        Timeout in seconds.
    verify:
    user_charset:

    See Also
    --------
    backends.PydapDataStore
    """

    description = "Open remote datasets via OPeNDAP using pydap in Xarray"
    url = "https://docs.xarray.dev/en/stable/generated/xarray.backends.PydapBackendEntrypoint.html"
    open_dataset_parameters = (
        "drop_variables",
        "mask_and_scale",
        "decode_times",
        "concat_characters",
        "use_cftime",
        "decode_timedelta",
        "decode_coords",
    )

    application: Any = None
    session: requests.Session | None = None
    output_grid: bool | None = None
    timeout: float | None = None
    verify: bool | None = None
    user_charset: str | None = None

    def guess_can_open(self, filename_or_obj: T_XarrayCanOpen) -> bool:
        return isinstance(filename_or_obj, str) and is_remote_uri(filename_or_obj)

    def open_dataset(
        self,
        filename_or_obj: T_XarrayCanOpen,
        *,
        drop_variables: str | Iterable[str] | None = None,
        mask_and_scale: bool = True,
        decode_times: bool = True,
        concat_characters: bool = True,
        use_cftime: bool | None = None,
        decode_timedelta: bool | None = None,
        decode_coords: bool | Literal["coordinates", "all"] = True,
        **kwargs: Any,
    ) -> Dataset:
        if not isinstance(filename_or_obj, str):
            raise ValueError(
                f"'filename_or_obj' must be a str (url), got {type(filename_or_obj)}."
            )
        store = PydapDataStore.open(
            url=filename_or_obj,
            application=kwargs.get("application", self.application),
            session=kwargs.get("session", self.session),
            output_grid=kwargs.get("output_grid", self.output_grid),
            timeout=kwargs.get("timeout", self.timeout),
            verify=kwargs.get("verify", self.verify),
            user_charset=kwargs.get("user_charset", self.user_charset),
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


BACKEND_ENTRYPOINTS["pydap"] = ("pydap", PydapBackendEntrypoint)
