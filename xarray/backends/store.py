from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Literal

from xarray import conventions
from xarray.backends.common import (
    BACKEND_ENTRYPOINTS,
    AbstractDataStore,
    BackendEntrypoint,
)
from xarray.core.dataset import Dataset

if TYPE_CHECKING:
    from xarray.core.types import T_XarrayCanOpen


class StoreBackendEntrypoint(BackendEntrypoint):
    description = "Open AbstractDataStore instances in Xarray"
    url = "https://docs.xarray.dev/en/stable/generated/xarray.backends.StoreBackendEntrypoint.html"
    open_dataset_parameters = (
        "drop_variables",
        "mask_and_scale",
        "decode_times",
        "concat_characters",
        "use_cftime",
        "decode_timedelta",
        "decode_coords",
    )

    def guess_can_open(self, filename_or_obj: T_XarrayCanOpen) -> bool:
        return isinstance(filename_or_obj, AbstractDataStore)

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
    ) -> Dataset:
        assert isinstance(filename_or_obj, AbstractDataStore)

        vars, attrs = filename_or_obj.load()
        encoding = filename_or_obj.get_encoding()

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
        ds.set_close(filename_or_obj.close)
        ds.encoding = encoding

        return ds


BACKEND_ENTRYPOINTS["store"] = (None, StoreBackendEntrypoint)
