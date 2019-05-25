"""Configuration for pytest."""

import pytest


def pytest_addoption(parser):
    """Add command-line flags for pytest."""
    parser.addoption("--run-flaky", action="store_true",
                     help="runs flaky tests")
    parser.addoption("--run-network-tests", action="store_true",
                     help="runs tests requiring a network connection")


def pytest_collection_modifyitems(config, items):

    if not config.getoption("--run-flaky"):
        skip_flaky = pytest.mark.skip(
            reason="set --run-flaky option to run flaky tests")
        for item in items:
            if "flaky" in item.keywords:
                item.add_marker(skip_flaky)

    if not config.getoption("--run-network-tests"):
        skip_network = pytest.mark.skip(
            reason="set --run-network-tests option to run tests requiring an"
            "internet connection")
        for item in items:
            if "network" in item.keywords:
                item.add_marker(skip_network)
