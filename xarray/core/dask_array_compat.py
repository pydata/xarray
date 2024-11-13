from typing import Any

from xarray.namedarray.utils import module_available


def reshape_blockwise(
    x: Any,
    shape: int | tuple[int, ...],
    chunks: tuple[tuple[int, ...], ...] | None = None,
):
    if module_available("dask", "2024.08.2"):
        from dask.array import reshape_blockwise

        return reshape_blockwise(x, shape=shape, chunks=chunks)
    else:
        return x.reshape(shape)
