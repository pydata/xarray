"""Configuration for pytest."""

import os

import pytest


def _use_dask_array(config: pytest.Config) -> bool:
    env_value = os.environ.get("XR_USE_DASK_ARRAY", "")
    return config.getoption("--use-dask-array") or env_value.lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _register_dask_array() -> None:
    try:
        import dask_array.xarray
    except ImportError as err:
        raise pytest.UsageError(
            "--use-dask-array requires dask-array to be importable"
        ) from err

    dask_array.xarray.register()
    if not dask_array.xarray.isactive():
        raise pytest.UsageError(
            "--use-dask-array registered dask-array, but it is not the active dask chunk manager"
        )


def pytest_addoption(parser: pytest.Parser):
    """Add command-line flags for pytest."""
    parser.addoption("--run-flaky", action="store_true", help="runs flaky tests")
    parser.addoption(
        "--run-network-tests",
        action="store_true",
        help="runs tests requiring a network connection",
    )
    parser.addoption("--run-mypy", action="store_true", help="runs mypy tests")
    parser.addoption(
        "--use-dask-array",
        action="store_true",
        help="register dask-array as xarray's dask chunk manager",
    )


def pytest_configure(config: pytest.Config):
    config.addinivalue_line(
        "markers",
        "skip_with_dask_array: skip when dask-array is registered as xarray's dask chunk manager",
    )
    config.addinivalue_line(
        "markers",
        "xfail_with_dask_array: xfail when dask-array is registered as xarray's dask chunk manager",
    )
    if not _use_dask_array(config):
        return

    _register_dask_array()


def pytest_runtest_setup(item):
    if _use_dask_array(item.config):
        _register_dask_array()
    # based on https://stackoverflow.com/questions/47559524
    if "flaky" in item.keywords and not item.config.getoption("--run-flaky"):
        pytest.skip("set --run-flaky option to run flaky tests")
    if "network" in item.keywords and not item.config.getoption("--run-network-tests"):
        pytest.skip(
            "set --run-network-tests to run test requiring an internet connection"
        )
    if any("mypy" in m.name for m in item.own_markers) and not item.config.getoption(
        "--run-mypy"
    ):
        pytest.skip("set --run-mypy option to run mypy tests")


# See https://docs.pytest.org/en/stable/example/markers.html#automatically-adding-markers-based-on-test-names
def pytest_collection_modifyitems(items):
    for item in items:
        if "mypy" in item.nodeid:
            # IMPORTANT: mypy type annotation tests leverage the pytest-mypy-plugins
            # plugin, and are thus written in test_*.yml files.  As such, there are
            # no explicit test functions on which we can apply a pytest.mark.mypy
            # decorator.  Therefore, we mark them via this name-based, automatic
            # marking approach, meaning that each test case must contain "mypy" in the
            # name.
            item.add_marker(pytest.mark.mypy)
        if _use_dask_array(item.config) and "skip_with_dask_array" in item.keywords:
            item.add_marker(
                pytest.mark.skip(reason="skipped with dask-array chunk manager")
            )
        if _use_dask_array(item.config) and "xfail_with_dask_array" in item.keywords:
            mark = item.get_closest_marker("xfail_with_dask_array")
            kwargs = dict(mark.kwargs) if mark is not None else {}
            kwargs.setdefault(
                "reason", "expected failure with dask-array chunk manager"
            )
            kwargs.setdefault("strict", True)
            item.add_marker(pytest.mark.xfail(**kwargs))


@pytest.fixture(autouse=True)
def set_zarr_v3_api(monkeypatch):
    """Set ZARR_V3_EXPERIMENTAL_API environment variable for all tests."""
    monkeypatch.setenv("ZARR_V3_EXPERIMENTAL_API", "1")


@pytest.fixture(autouse=True)
def add_standard_imports(doctest_namespace, tmpdir):
    import numpy as np
    import pandas as pd

    import xarray as xr

    doctest_namespace["np"] = np
    doctest_namespace["pd"] = pd
    doctest_namespace["xr"] = xr

    # always seed numpy.random to make the examples deterministic
    np.random.seed(0)

    # always switch to the temporary directory, so files get written there
    tmpdir.chdir()

    # Avoid the dask deprecation warning, can remove if CI passes without this.
    try:
        import dask
    except ImportError:
        pass
    else:
        dask.config.set({"dataframe.query-planning": True})
