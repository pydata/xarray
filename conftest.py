"""Configuration for pytest."""

import pytest


def pytest_addoption(parser):
    """Add command-line flags for pytest."""
    parser.addoption("--run-flaky", action="store_true", help="runs flaky tests")
    parser.addoption(
        "--run-network-tests",
        action="store_true",
        help="runs tests requiring a network connection",
    )


def pytest_runtest_setup(item):
    # based on https://stackoverflow.com/questions/47559524
    if "flaky" in item.keywords and not item.config.getoption("--run-flaky"):
        pytest.skip("set --run-flaky option to run flaky tests")
    if "network" in item.keywords and not item.config.getoption("--run-network-tests"):
        pytest.skip(
            "set --run-network-tests to run test requiring an internet connection"
        )


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
