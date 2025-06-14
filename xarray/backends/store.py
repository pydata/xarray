from __future__ import annotations

from dataclasses import asdict
from typing import TYPE_CHECKING, Any, Optional

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

    def guess_can_open(
        self,
        filename_or_obj: str | os.PathLike[Any] | ReadBuffer | AbstractDataStore,
    ) -> bool:
        return isinstance(filename_or_obj, AbstractDataStore)

    def open_dataset(
        self,
        filename_or_obj: str | os.PathLike[Any] | ReadBuffer | AbstractDataStore,
        *,
        coder_opts: Optional[CoderOptions] = None,
        **kwargs,
    ) -> Dataset:
        assert isinstance(filename_or_obj, AbstractDataStore)

        vars, attrs = filename_or_obj.load()
        encoding = filename_or_obj.get_encoding()

        coder_opts = coder_opts if coder_opts is not None else self.coder_opts
        coders_kwargs = asdict(coder_opts)
        vars, attrs, coord_names = conventions.decode_cf_variables(
            vars,
            attrs,
            **coders_kwargs,
        )

        ds = Dataset(vars, attrs=attrs)
        ds = ds.set_coords(coord_names.intersection(vars))
        ds.set_close(filename_or_obj.close)
        ds.encoding = encoding

        return ds


BACKEND_ENTRYPOINTS["store"] = (None, StoreBackendEntrypoint)
