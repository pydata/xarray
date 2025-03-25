"""Test for verifying our changes don't cause import problems."""

def test_import_chunks():
    """Test importing the functions we moved."""
    from xarray.structure.chunks import _maybe_chunk, _get_chunk, unify_chunks
    assert _maybe_chunk is not None
    assert _get_chunk is not None
    assert unify_chunks is not None

def test_imports_still_work():
    """Test that we can still import from the expected places."""
    import xarray as xr
    assert xr.unify_chunks is not None