import pytest

from . import requires_dask


@pytest.fixture(params=["numpy", pytest.param("dask", marks=requires_dask)])
def backend(request):
    return request.param


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "apply_marks(marks): function to attach marks to tests and test variants",
    )


def always_sequence(obj):
    if not isinstance(obj, (list, tuple)):
        obj = [obj]

    return obj


def pytest_collection_modifyitems(session, config, items):
    for item in items:
        mark = item.get_closest_marker("apply_marks")
        if mark is None:
            continue

        marks = mark.args[0]
        if not isinstance(marks, dict):
            continue

        possible_marks = marks.get(item.originalname)
        if possible_marks is None:
            continue

        if not isinstance(possible_marks, dict):
            for mark in always_sequence(possible_marks):
                item.add_marker(mark)
            continue

        variant = item.name[len(item.originalname) :]
        to_attach = possible_marks.get(variant)
        if to_attach is None:
            continue

        for mark in always_sequence(to_attach):
            item.add_marker(mark)
