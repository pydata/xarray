import pytest

def test_parallelcompat_backward():
    # In vesion 2024.4.1 and before
    # These functions were located in the
    #    xarray.core.parallelcompat
    #    xarray.core.pycompat
    # modules
    # https://github.com/pydata/xarray/commit/d64460795e406bc4a998e2ddae0054a1029d52a9
    with pytest.warns(match="xarray.core.parallelcompat.ChunkManagerEntrypoint is deprecated."):
        from xarray.core.parallelcompat import ChunkManagerEntrypoint

    with pytest.warns(match="xarray.core.pycompat.is_chunked_array is deprecated."):
        from xarray.core.pycompat import is_chunked_array

    with pytest.warns(match="xarray.core.pycompat.is_chunked_array is deprecated."):
        from xarray.core.pycompat import is_chunked_array
