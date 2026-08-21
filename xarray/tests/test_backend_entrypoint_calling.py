import pytest

from xarray import (
    open_dataarray,
    open_dataset,
    open_datatree,
)
from xarray.backends import BackendEntrypoint


class ArgsCalled(Exception):
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs


class DataArrayCalled(ArgsCalled):
    pass


class DatasetCalled(ArgsCalled):
    pass


class DataTreeCalled(ArgsCalled):
    pass


class DummyBackendEntrypointDataset(BackendEntrypoint):
    def open_dataset(filename_or_obj, *args, **kwargs):
        raise DatasetCalled(*args, **kwargs)


class DummyBackendEntrypointDatasetDataTree(BackendEntrypoint):
    def open_dataset(filename_or_obj, *args, **kwargs):
        raise DatasetCalled(*args, **kwargs)

    def open_datatree(filename_or_obj, *args, **kwargs):
        raise DataTreeCalled(*args, **kwargs)


class DummyBackendEntrypointAll(BackendEntrypoint):
    def open_dataarray(filename_or_obj, *args, **kwargs):
        raise DataArrayCalled(*args, **kwargs)

    def open_dataset(filename_or_obj, *args, **kwargs):
        raise DatasetCalled(*args, **kwargs)

    def open_datatree(filename_or_obj, *args, **kwargs):
        raise DataTreeCalled(*args, **kwargs)


def test_dataset(tmp_path):

    dataset_engine = DummyBackendEntrypointDataset

    existing_file = tmp_path / "test.unknown"
    existing_file.write_bytes(b"")

    try:
        open_dataarray(existing_file, engine=dataset_engine)
    except DatasetCalled as e:
        assert e.args[0] == existing_file
        assert e.kwargs == {"drop_variables": None}

    try:
        open_dataset(existing_file, engine=dataset_engine)
    except DatasetCalled as e:
        assert e.args[0] == existing_file
        assert e.kwargs == {"drop_variables": None}

    with pytest.raises(NotImplementedError):
        open_datatree(existing_file, engine=dataset_engine)


def test_datatree(tmp_path):

    dataset_engine = DummyBackendEntrypointDatasetDataTree

    existing_file = tmp_path / "test.unknown"
    existing_file.write_bytes(b"")

    try:
        open_datatree(existing_file, engine=dataset_engine)
    except DataTreeCalled as e:
        assert e.args[0] == existing_file
        assert e.kwargs == {"drop_variables": None}


def test_dataarray(tmp_path):

    dataset_engine = DummyBackendEntrypointAll

    existing_file = tmp_path / "test.unknown"
    existing_file.write_bytes(b"")

    try:
        open_dataarray(existing_file, engine=dataset_engine)
    except DataArrayCalled as e:
        assert e.args[0] == existing_file
        assert e.kwargs == {"drop_variables": None}
