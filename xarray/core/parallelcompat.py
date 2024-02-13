from xarray.namedarray.parallelcompat import ChunkManagerEntrypoint  # noqa
import warnings

warnings.warn(
    "xarray.core.parallelcompat.ChunkManagerEntrypoint is deprecated. "
    "Use xarray.namedarray.parallel.ChunkManagerEntrypoint instead.",
    DeprecationWarning,
    stacklevel=2,
)
