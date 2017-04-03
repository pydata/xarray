import pytest

def pytest_addoption(parser):
    parser.addoption("--run-flaky", action="store_true",
                     help="runs flaky tests")
    parser.addoption("--skip-optional-ci", action="store_true",
                     help="skips optional tests continuous integration (CI)")
    parser.addoption("--skip-slow", action="store_true",
                     help="skips slow tests")
