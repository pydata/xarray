"""Configuration for pytest."""


def pytest_addoption(parser):
    """Add command-line flags for pytest."""
    parser.addoption("--run-flaky", action="store_true",
                     help="runs flaky tests")
    parser.addoption("--run-network-tests", action="store_true",
                     help="runs tests requiring a network connection")
