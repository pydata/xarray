from xarray.backends.plugins import get_backend, list_engines


def test_get_backend__engine():
    """Test passing in a a backend name (engine) gets us an instance of the corresponding backend."""
    engine = "netcdf4"
    engines = list_engines()
    backend = get_backend(engine)
    assert backend == engines[engine]


def test_get_backend__backend():
    """Test passing in a backend directly gets us an instance of that backend."""
    engine = "netcdf4"
    given_backend = type(list_engines()[engine])
    assert isinstance(get_backend(given_backend), given_backend)
