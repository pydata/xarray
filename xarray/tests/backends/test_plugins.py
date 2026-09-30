from __future__ import annotations

import sys
from importlib.metadata import EntryPoint, EntryPoints
from itertools import starmap
from unittest import mock

import pytest

from xarray.backends import common, plugins
from xarray.core.options import OPTIONS
from xarray.tests import (
    has_h5netcdf,
    has_netCDF4,
    has_pydap,
    has_scipy,
    has_zarr,
    requires_h5netcdf,
    requires_netCDF4,
    requires_zarr,
)

# Do not import list_engines here, this will break the lazy tests

importlib_metadata_mock = "importlib.metadata"


class DummyBackendEntrypointArgs(common.BackendEntrypoint):
    def open_dataset(filename_or_obj, *args):  # type: ignore[override]
        pass


class DummyBackendEntrypointKwargs(common.BackendEntrypoint):
    def open_dataset(filename_or_obj, **kwargs):  # type: ignore[override]
        pass


class DummyBackendEntrypoint1(common.BackendEntrypoint):
    def open_dataset(self, filename_or_obj, *, decoder):  # type: ignore[override]
        pass


class DummyBackendEntrypoint2(common.BackendEntrypoint):
    def open_dataset(self, filename_or_obj, *, decoder):  # type: ignore[override]
        pass


@pytest.fixture
def dummy_duplicated_entrypoints():
    specs = [
        ["engine1", "xarray.tests.backends.test_plugins:backend_1", "xarray.backends"],
        ["engine1", "xarray.tests.backends.test_plugins:backend_2", "xarray.backends"],
        ["engine2", "xarray.tests.backends.test_plugins:backend_1", "xarray.backends"],
        ["engine2", "xarray.tests.backends.test_plugins:backend_2", "xarray.backends"],
    ]
    eps = list(starmap(EntryPoint, specs))
    return eps


@pytest.mark.filterwarnings("ignore:Found")
def test_remove_duplicates(dummy_duplicated_entrypoints) -> None:
    with pytest.warns(RuntimeWarning):
        entrypoints = plugins.remove_duplicates(dummy_duplicated_entrypoints)
    assert len(entrypoints) == 2


def test_broken_plugin() -> None:
    broken_backend = EntryPoint(
        "broken_backend",
        "xarray.tests.backends.test_plugins:backend_1",
        "xarray.backends",
    )
    with pytest.warns(RuntimeWarning) as record:
        _ = plugins.build_engines(EntryPoints([broken_backend]))
    assert len(record) == 1
    message = str(record[0].message)
    assert "Engine 'broken_backend'" in message


def test_remove_duplicates_warnings(dummy_duplicated_entrypoints) -> None:
    with pytest.warns(RuntimeWarning) as record:
        _ = plugins.remove_duplicates(dummy_duplicated_entrypoints)

    assert len(record) == 2
    message0 = str(record[0].message)
    message1 = str(record[1].message)
    assert "entrypoints" in message0
    assert "entrypoints" in message1


@mock.patch(
    f"{importlib_metadata_mock}.EntryPoint.load", mock.MagicMock(return_value=None)
)
def test_backends_dict_from_pkg() -> None:
    specs = [
        ["engine1", "xarray.tests.backends.test_plugins:backend_1", "xarray.backends"],
        ["engine2", "xarray.tests.backends.test_plugins:backend_2", "xarray.backends"],
    ]
    entrypoints = list(starmap(EntryPoint, specs))
    engines = plugins.backends_dict_from_pkg(entrypoints)
    assert len(engines) == 2
    assert engines.keys() == {"engine1", "engine2"}


def test_set_missing_parameters() -> None:
    backend_1 = DummyBackendEntrypoint1
    backend_2 = DummyBackendEntrypoint2
    backend_2.open_dataset_parameters = ("filename_or_obj",)
    engines = {"engine_1": backend_1, "engine_2": backend_2}
    plugins.set_missing_parameters(engines)

    assert len(engines) == 2
    assert backend_1.open_dataset_parameters == ("filename_or_obj", "decoder")
    assert backend_2.open_dataset_parameters == ("filename_or_obj",)

    backend_kwargs = DummyBackendEntrypointKwargs
    backend_kwargs.open_dataset_parameters = ("filename_or_obj", "decoder")
    plugins.set_missing_parameters({"engine": backend_kwargs})
    assert backend_kwargs.open_dataset_parameters == ("filename_or_obj", "decoder")

    backend_args = DummyBackendEntrypointArgs
    backend_args.open_dataset_parameters = ("filename_or_obj", "decoder")
    plugins.set_missing_parameters({"engine": backend_args})
    assert backend_args.open_dataset_parameters == ("filename_or_obj", "decoder")

    # reset
    backend_1.open_dataset_parameters = None
    backend_1.open_dataset_parameters = None
    backend_kwargs.open_dataset_parameters = None
    backend_args.open_dataset_parameters = None


def test_set_missing_parameters_raise_error() -> None:
    backend = DummyBackendEntrypointKwargs
    with pytest.raises(TypeError):
        plugins.set_missing_parameters({"engine": backend})

    backend_args = DummyBackendEntrypointArgs
    with pytest.raises(TypeError):
        plugins.set_missing_parameters({"engine": backend_args})


@mock.patch(
    f"{importlib_metadata_mock}.EntryPoint.load",
    mock.MagicMock(return_value=DummyBackendEntrypoint1),
)
def test_build_engines() -> None:
    dummy_pkg_entrypoint = EntryPoint(
        "dummy", "xarray.tests.backends.test_plugins:backend_1", "xarray_backends"
    )
    backend_entrypoints = plugins.build_engines(EntryPoints([dummy_pkg_entrypoint]))

    assert isinstance(backend_entrypoints["dummy"], DummyBackendEntrypoint1)
    assert backend_entrypoints["dummy"].open_dataset_parameters == (
        "filename_or_obj",
        "decoder",
    )


@mock.patch(
    f"{importlib_metadata_mock}.EntryPoint.load",
    mock.MagicMock(return_value=DummyBackendEntrypoint1),
)
def test_build_engines_sorted() -> None:
    dummy_pkg_entrypoints = EntryPoints(
        [
            EntryPoint(
                "dummy2",
                "xarray.tests.backends.test_plugins:backend_1",
                "xarray.backends",
            ),
            EntryPoint(
                "dummy1",
                "xarray.tests.backends.test_plugins:backend_1",
                "xarray.backends",
            ),
        ]
    )
    backend_entrypoints = list(plugins.build_engines(dummy_pkg_entrypoints))

    indices = []
    for be in OPTIONS["netcdf_engine_order"]:
        try:
            index = backend_entrypoints.index(be)
            backend_entrypoints.pop(index)
            indices.append(index)
        except ValueError:
            pass

    assert set(indices) < {0, -1}
    assert list(backend_entrypoints) == sorted(backend_entrypoints)


@mock.patch(
    "xarray.backends.plugins.list_engines",
    mock.MagicMock(return_value={"dummy": DummyBackendEntrypointArgs()}),
)
def test_no_matching_engine_found(tmp_path) -> None:
    # Non-existent local file raises FileNotFoundError
    with pytest.raises(FileNotFoundError, match=r"No such file"):
        plugins.guess_engine("not-valid")

    # Existing file with unrecognized extension raises ValueError
    existing_file = tmp_path / "test.unknown"
    existing_file.write_bytes(b"")
    with pytest.raises(ValueError, match=r"did not find a match in any"):
        plugins.guess_engine(str(existing_file))

    # Existing file with recognized magic number raises ValueError
    nc_file = tmp_path / "foo.nc"
    nc_file.write_bytes(b"CDF\x01\x00\x00\x00\x00")
    with pytest.raises(ValueError, match=r"found the following matches with the input"):
        plugins.guess_engine(str(nc_file))


@mock.patch(
    "xarray.backends.plugins.list_engines",
    mock.MagicMock(return_value={}),
)
def test_engines_not_installed(tmp_path) -> None:
    # Non-existent local file raises FileNotFoundError
    with pytest.raises(FileNotFoundError, match=r"No such file"):
        plugins.guess_engine("not-valid")

    # Existing file with no matching engine raises ValueError
    existing_file = tmp_path / "test.unknown"
    existing_file.write_bytes(b"")
    with pytest.raises(ValueError, match=r"xarray is unable to open"):
        plugins.guess_engine(str(existing_file))

    # Existing file with recognized magic number raises ValueError
    nc_file = tmp_path / "foo.nc"
    nc_file.write_bytes(b"CDF\x01\x00\x00\x00\x00")
    with pytest.raises(ValueError, match=r"found the following matches with the input"):
        plugins.guess_engine(str(nc_file))


@mock.patch(
    "xarray.backends.plugins.list_engines",
    mock.MagicMock(return_value={"dummy": DummyBackendEntrypointArgs()}),
)
def test_guess_engine_file_not_found() -> None:
    # Non-existent local file path (string)
    with pytest.raises(
        FileNotFoundError, match=r"No such file: '/nonexistent/path.h5'"
    ):
        plugins.guess_engine("/nonexistent/path.h5")

    # Non-existent local file path (PathLike)
    from pathlib import Path

    with pytest.raises(FileNotFoundError, match=r"No such file"):
        plugins.guess_engine(Path("/nonexistent/path.h5"))

    # Remote URIs should not raise FileNotFoundError (raises ValueError instead)
    with pytest.raises(ValueError):
        plugins.guess_engine("https://example.com/missing.h5")


@pytest.mark.parametrize("engine", common.BACKEND_ENTRYPOINTS.keys())
def test_get_backend_fastpath_skips_list_engines(engine: str) -> None:
    """Test that built-in engines skip list_engines (fastpath)."""
    plugins.list_engines.cache_clear()
    initial_misses = plugins.list_engines.cache_info().misses
    plugins.get_backend(engine)
    assert plugins.list_engines.cache_info().misses == initial_misses


def test_lazy_import() -> None:
    """Test that some modules are imported in a lazy manner.

    When importing xarray these should not be imported as well.
    Only when running code for the first time that requires them.
    """
    deny_list = [
        "cubed",
        "cupy",
        # "dask",  # TODO: backends.locks is not lazy yet :(
        "dask.array",
        "dask.distributed",
        "flox",
        "h5netcdf",
        "matplotlib",
        "nc_time_axis",
        "netCDF4",
        "numbagg",
        "pint",
        "pydap",
        "scipy",
        "sparse",
        "zarr",
    ]
    # ensure that none of the above modules has been imported before
    modules_backup = {}
    for pkg in list(sys.modules.keys()):
        for mod in deny_list + ["xarray"]:
            if pkg.startswith(mod):
                modules_backup[pkg] = sys.modules[pkg]
                del sys.modules[pkg]
                break

    try:
        import xarray  # noqa: F401
        from xarray.backends import list_engines

        list_engines()

        # ensure that none of the modules that are supposed to be
        # lazy loaded are loaded when importing xarray
        is_imported = set()
        for pkg in sys.modules:
            for mod in deny_list:
                if pkg.startswith(mod):
                    is_imported.add(mod)
                    break
        assert len(is_imported) == 0, (
            f"{is_imported} have been imported but should be lazy"
        )

    finally:
        # restore original
        sys.modules.update(modules_backup)


def test_list_engines() -> None:
    from xarray.backends import list_engines

    engines = list_engines()
    assert list_engines.cache_info().currsize == 1

    assert ("scipy" in engines) == has_scipy
    assert ("h5netcdf" in engines) == has_h5netcdf
    assert ("netcdf4" in engines) == has_netCDF4
    assert ("pydap" in engines) == has_pydap
    assert ("zarr" in engines) == has_zarr
    assert "store" in engines


def test_refresh_engines() -> None:
    from xarray.backends import list_engines, refresh_engines

    EntryPointMock1 = mock.MagicMock()
    EntryPointMock1.name = "test1"
    EntryPointMock1.load.return_value = DummyBackendEntrypoint1

    return_value = EntryPoints([EntryPointMock1])

    with mock.patch("xarray.backends.plugins.entry_points", return_value=return_value):
        list_engines.cache_clear()
        engines = list_engines()
    assert "test1" in engines
    assert isinstance(engines["test1"], DummyBackendEntrypoint1)

    EntryPointMock2 = mock.MagicMock()
    EntryPointMock2.name = "test2"
    EntryPointMock2.load.return_value = DummyBackendEntrypoint2

    return_value2 = EntryPoints([EntryPointMock2])

    with mock.patch("xarray.backends.plugins.entry_points", return_value=return_value2):
        refresh_engines()
        engines = list_engines()
    assert "test1" not in engines
    assert "test2" in engines
    assert isinstance(engines["test2"], DummyBackendEntrypoint2)

    # reset to original
    refresh_engines()


@requires_h5netcdf
@requires_netCDF4
@requires_zarr
def test_remote_url_backend_auto_detection() -> None:
    """
    Test that remote URLs are correctly selected by the backend resolution system.

    This tests the fix for issue where netCDF4, h5netcdf, and pydap backends were
    claiming ALL remote URLs, preventing remote Zarr stores from being
    auto-detected.

    See: https://github.com/pydata/xarray/issues/10801
    """
    from xarray.backends.plugins import guess_engine

    # Test cases: (url, expected_backend)
    test_cases = [
        # Remote Zarr URLs
        ("https://example.com/store.zarr", "zarr"),
        ("http://example.com/data.zarr/", "zarr"),
        ("s3://bucket/path/to/data.zarr", "zarr"),
        # Remote netCDF URLs (non-DAP) - netcdf4 wins (first in order, no query params)
        ("https://example.com/file.nc", "netcdf4"),
        ("http://example.com/data.nc4", "netcdf4"),
        ("https://example.com/test.cdf", "netcdf4"),
        ("s3://bucket/path/to/data.nc", "netcdf4"),
        # Remote netCDF URLs with query params - netcdf4 wins
        # Note: Query params are typically indicative of DAP URLs (e.g., OPeNDAP constraint expressions),
        # so we prefer netcdf4 (which has DAP support) over h5netcdf (which doesn't)
        ("https://example.com/data.nc?var=temperature&time=0", "netcdf4"),
        (
            "http://test.opendap.org/opendap/dap4/StaggeredGrid.nc4?dap4.ce=/time[0:1:0]",
            "netcdf4",
        ),
        # DAP URLs with .nc extensions (no query params) - netcdf4 wins (first in order)
        ("http://test.opendap.org/opendap/dap4/StaggeredGrid.nc4", "netcdf4"),
        ("https://example.com/DAP4/data.nc", "netcdf4"),
        ("http://example.com/data/Dap4/file.nc", "netcdf4"),
    ]

    for url, expected_backend in test_cases:
        engine = guess_engine(url)
        assert engine == expected_backend, (
            f"URL {url!r} should select {expected_backend!r} but got {engine!r}"
        )

    # DAP URLs - netcdf4 should handle these (it comes first in backend order)
    # Both netcdf4 and pydap can open DAP URLs, but netcdf4 has priority
    expected_dap_backend = "netcdf4"
    dap_urls = [
        # Explicit DAP protocol schemes
        "dap2://opendap.earthdata.nasa.gov/collections/dataset",
        "dap4://opendap.earthdata.nasa.gov/collections/dataset",
        "dap://example.com/dataset",
        "DAP2://example.com/dataset",  # uppercase scheme
        "DAP4://example.com/dataset",  # uppercase scheme
        # DAP path indicators
        "https://example.com/services/DAP2/dataset",  # uppercase in path
        "http://test.opendap.org/opendap/data/nc/file.nc",  # /opendap/ path
        "https://coastwatch.pfeg.noaa.gov/erddap/griddap/erdMH1chla8day",  # ERDDAP
        "http://thredds.ucar.edu/thredds/dodsC/grib/NCEP/GFS/",  # THREDDS dodsC
        "https://disc2.gesdisc.eosdis.nasa.gov/dods/TRMM_3B42",  # GrADS /dods/
    ]

    for url in dap_urls:
        engine = guess_engine(url)
        assert engine == expected_dap_backend, (
            f"URL {url!r} should select {expected_dap_backend!r} but got {engine!r}"
        )

    # URLs with .dap suffix are claimed by netcdf4 (backward compatibility fallback)
    # Note: .dap suffix is intentionally NOT recognized as a DAP dataset URL
    fallback_urls = [
        ("http://test.opendap.org/opendap/data/nc/coads_climatology.nc.dap", "netcdf4"),
        ("https://example.com/data.dap", "netcdf4"),
    ]

    for url, expected_backend in fallback_urls:
        engine = guess_engine(url)
        assert engine == expected_backend
