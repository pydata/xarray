from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

from xarray import conventions
from xarray.backends.common import (
    BACKEND_ENTRYPOINTS,
    AbstractDataStore,
    BackendEntrypoint,
)
from xarray.core.dataset import Dataset

if TYPE_CHECKING:
    import os
    from io import BufferedIOBase


class StoreBackendEntrypoint(BackendEntrypoint):
    description = "Open AbstractDataStore instances in Xarray"
    url = "https://docs.xarray.dev/en/stable/generated/xarray.backends.StoreBackendEntrypoint.html"

    @classmethod
    def _set_availability(cls) -> None:
        """Resets the backends availability."""
        cls.available = True

    def guess_can_open(
        self,
        filename_or_obj: str | os.PathLike[Any] | BufferedIOBase | AbstractDataStore,
    ) -> bool:
        return isinstance(filename_or_obj, AbstractDataStore)

    def open_dataset(
        self,
        store,
        *,
        mask_and_scale=True,
        decode_times=True,
        concat_characters=True,
        decode_coords=True,
        drop_variables: str | Iterable[str] | None = None,
        use_cftime=None,
        decode_timedelta=None,
    ) -> Dataset:
        vars, attrs = store.load()
        encoding = store.get_encoding()

        vars, attrs, coord_names = conventions.decode_cf_variables(
            vars,
            attrs,
            mask_and_scale=mask_and_scale,
            decode_times=decode_times,
            concat_characters=concat_characters,
            decode_coords=decode_coords,
            drop_variables=drop_variables,
            use_cftime=use_cftime,
            decode_timedelta=decode_timedelta,
        )

        ds = Dataset(vars, attrs=attrs)
        ds = ds.set_coords(coord_names.intersection(vars))
        ds.set_close(store.close)
        ds.encoding = encoding

        return ds


BACKEND_ENTRYPOINTS["store"] = (None, StoreBackendEntrypoint)
