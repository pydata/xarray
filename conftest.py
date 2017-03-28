import pytest

def pytest_addoption(parser):
    parser.addoption("--run-flakey", action="store_true",
                     help="runs flakey tests")
    parser.addoption("--skip-optional-ci", action="store_true",
                     help="skips optional tests continuous integration (CI)")
    parser.addoption("--skip-slow", action="store_true",
                     help="skips slow tests")
