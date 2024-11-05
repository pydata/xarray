from xarray.namedarray.utils import module_available


def sliding_window_view(
    x, window_shape, axis=None, *, automatic_rechunk=True, **kwargs
):
    # Backcompat for handling `automatic_rechunk`, delete when dask>=2024.11.0
    # subok, writeable are unsupported by dask
    from dask.array.lib.stride_tricks import sliding_window_view

    if module_available("dask", "2024.11.0"):
        return sliding_window_view(
            x, window_shape=window_shape, axis=axis, automatic_rechunk=automatic_rechunk
        )
    else:
        # automatic_rechunk is not supported
        return sliding_window_view(x, window_shape=window_shape, axis=axis)
