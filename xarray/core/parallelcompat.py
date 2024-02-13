__all__ = [
    "ChunkManagerEntrypoint",
]

def __getattr__(attr):
    if attr == 'ChunkManagerEntrypoint':
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
        raise AttributeError("module {!r} has no attribute "
                             "{!r}".format(__name__, attr))

def __dir__():
    return __all__
