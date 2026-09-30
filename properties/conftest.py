import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--run-slow-hypothesis",
        action="store_true",
        default=False,
        help="run slow hypothesis tests",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-slow-hypothesis"):
        return
    skip_slow_hyp = pytest.mark.skip(reason="need --run-slow-hypothesis option to run")
    for item in items:
        if "slow_hypothesis" in item.keywords:
            item.add_marker(skip_slow_hyp)


try:
    from hypothesis import settings
except ImportError:
    pass
else:
    # Run for a while - arrays are a bigger search space than usual
    settings.register_profile("ci", deadline=None, print_blob=True)
    settings.load_profile("ci")
