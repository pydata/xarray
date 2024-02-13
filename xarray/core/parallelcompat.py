__all__ = [
    "ChunkManagerEntrypoint",
]


def __getattr__(attr):
    if attr == "ChunkManagerEntrypoint":
        import warnings

        warnings.warn(
            "xarray.core.parallelcompat.ChunkManagerEntrypoint is deprecated. "
            "Use xarray.namedarray.parallelcompat.ChunkManagerEntrypoint instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        from xarray.namedarray.parallelcompat import ChunkManagerEntrypoint

        return ChunkManagerEntrypoint
    else:
        raise AttributeError(f"module {__name__!r} has no attribute " f"{attr!r}")


def __dir__():
    return __all__
