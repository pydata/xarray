from __future__ import annotations

from typing import TYPE_CHECKING, Any

from xarray import conventions
from xarray.backends.common import (
    BACKEND_ENTRYPOINTS,
    AbstractDataStore,
    BackendEntrypoint,
    CoderOptions,
)
from xarray.core.dataset import Dataset

if TYPE_CHECKING:
    import os

    from xarray.core.types import ReadBuffer


class StoreBackendEntrypoint(BackendEntrypoint):
    description = "Open AbstractDataStore instances in Xarray"
    url = "https://docs.xarray.dev/en/stable/generated/xarray.backends.StoreBackendEntrypoint.html"

    coder_class = CoderOptions

    def guess_can_open(
        self,
        filename_or_obj: str | os.PathLike[Any] | ReadBuffer | AbstractDataStore,
    ) -> bool:
        return isinstance(filename_or_obj, AbstractDataStore)

    def open_dataset(
        self,
        filename_or_obj: str | os.PathLike[Any] | ReadBuffer | AbstractDataStore,
        *,
        coder_options: CoderOptions | None = None,
    ) -> Dataset:
        assert isinstance(filename_or_obj, AbstractDataStore)

        vars, attrs = filename_or_obj.load()
        encoding = filename_or_obj.get_encoding()

        coder_options = (
            coder_options if coder_options is not None else self.coder_class()
        )
        vars, attrs, coord_names = conventions.decode_cf_variables(
            vars, attrs, **coder_options.to_kwargs()
        )

        ds = Dataset(vars, attrs=attrs)
        ds = ds.set_coords(coord_names.intersection(vars))
        ds.set_close(filename_or_obj.close)
        ds.encoding = encoding

        return ds


BACKEND_ENTRYPOINTS["store"] = (None, StoreBackendEntrypoint)
