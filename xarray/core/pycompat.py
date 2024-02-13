__all__ = [
    "is_chunked_array",
    "is_duck_dask_array",
]


def __getattr__(attr):
    if attr == "is_chunked_array":
        import warnings

        warnings.warn(
            "xarray.core.pycompat.is_chunked_array is deprecated. "
            "Use xarray.namedarray.pycompat.is_chunked_array instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        from xarray.namedarray.pycompat import is_chunked_array

        return is_chunked_array
    elif attr == "is_duck_dask_array":
        import warnings

        warnings.warn(
            "xarray.core.pycompat.is_duck_dask_array is deprecated. "
            "Use xarray.namedarray.pycompat.is_duck_dask_array instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        from xarray.namedarray.pycompat import is_duck_dask_array

        return is_duck_dask_array
    else:
        raise AttributeError(f"module {__name__!r} has no attribute " f"{attr!r}")


def __dir__():
    return __all__
