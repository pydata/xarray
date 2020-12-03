from unittest import mock

import pkg_resources
import pytest

from xarray.backends import plugins


def dummy_open_dataset_args(filename_or_obj, *args):
    pass


def dummy_open_dataset_kwargs(filename_or_obj, **kwargs):
    pass


def dummy_open_dataset(filename_or_obj, *, decoder):
    pass


@pytest.fixture
def dummy_duplicated_entrypoints():
    specs = [
        "engine1 = xarray.tests.test_plugins:backend_1",
        "engine1 = xarray.tests.test_plugins:backend_2",
        "engine2 = xarray.tests.test_plugins:backend_1",
        "engine2 = xarray.tests.test_plugins:backend_2",
    ]
    eps = [
        pkg_resources.EntryPoint.parse(spec)
        for spec in specs
    ]
    return eps


def test_remove_duplicates(dummy_duplicated_entrypoints):
    entrypoints = plugins.remove_duplicates(dummy_duplicated_entrypoints)
    assert len(entrypoints) == 2


def test_remove_duplicates_warnings(dummy_duplicated_entrypoints):

    with pytest.warns(RuntimeWarning) as record:
        _ = plugins.remove_duplicates(dummy_duplicated_entrypoints)

    assert len(record) == 2
    message0 = str(record[0].message)
    message1 = str(record[1].message)
    assert "entrypoints" in message0
    assert "entrypoints" in message1


@mock.patch("pkg_resources.EntryPoint.load", mock.MagicMock(return_value=None))
def test_create_engines_dict():
    specs = [
        "engine1 = xarray.tests.test_plugins:backend_1",
        "engine2 = xarray.tests.test_plugins:backend_2",
    ]
    entrypoints = []
    for spec in specs:
        entrypoints.append(pkg_resources.EntryPoint.parse(spec))
    engines = plugins.create_engines_dict(entrypoints)
    assert len(engines) == 2
    assert engines.keys() == set(("engine1", "engine2"))


def test_set_missing_parameters():
    backend_1 = plugins.BackendEntrypoint(dummy_open_dataset)
    backend_2 = plugins.BackendEntrypoint(dummy_open_dataset, ("filename_or_obj",))
    engines = {"engine_1": backend_1, "engine_2": backend_2}
    plugins.set_missing_parameters(engines)

    assert len(engines) == 2
    engine_1 = engines["engine_1"]
    assert engine_1.open_dataset_parameters == ("filename_or_obj", "decoder")
    engine_2 = engines["engine_2"]
    assert engine_2.open_dataset_parameters == ("filename_or_obj",)


def test_set_missing_parameters_raise_error():

    backend = plugins.BackendEntrypoint(dummy_open_dataset_args)
    with pytest.raises(TypeError):
        plugins.set_missing_parameters({"engine": backend})

    backend = plugins.BackendEntrypoint(
        dummy_open_dataset_args, ("filename_or_obj", "decoder")
    )
    plugins.set_missing_parameters({"engine": backend})

    backend = plugins.BackendEntrypoint(dummy_open_dataset_kwargs)
    with pytest.raises(TypeError):
        plugins.set_missing_parameters({"engine": backend})

    backend = plugins.BackendEntrypoint(
        dummy_open_dataset_kwargs, ("filename_or_obj", "decoder")
    )
    plugins.set_missing_parameters({"engine": backend})
