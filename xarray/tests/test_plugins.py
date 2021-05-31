from unittest import mock

import pkg_resources
import pytest

from xarray.backends import common, plugins


class DummyBackendEntrypointArgs(common.BackendEntrypoint):
    def open_dataset(filename_or_obj, *args):
        pass


class DummyBackendEntrypointKwargs(common.BackendEntrypoint):
    def open_dataset(filename_or_obj, **kwargs):
        pass


class DummyBackendEntrypoint1(common.BackendEntrypoint):
    def open_dataset(self, filename_or_obj, *, decoder):
        pass


class DummyBackendEntrypoint2(common.BackendEntrypoint):
    def open_dataset(self, filename_or_obj, *, decoder):
        pass


@pytest.fixture
def dummy_duplicated_entrypoints():
    specs = [
        "engine1 = xarray.tests.test_plugins:backend_1",
        "engine1 = xarray.tests.test_plugins:backend_2",
        "engine2 = xarray.tests.test_plugins:backend_1",
        "engine2 = xarray.tests.test_plugins:backend_2",
    ]
    eps = [pkg_resources.EntryPoint.parse(spec) for spec in specs]
    return eps


@pytest.mark.filterwarnings("ignore:Found")
def test_remove_duplicates(dummy_duplicated_entrypoints):
    with pytest.warns(RuntimeWarning):
        entrypoints = plugins.remove_duplicates(dummy_duplicated_entrypoints)
    assert len(entrypoints) == 2


def test_broken_plugin():
    broken_backend = pkg_resources.EntryPoint.parse(
        "broken_backend = xarray.tests.test_plugins:backend_1"
    )
    with pytest.warns(RuntimeWarning) as record:
        _ = plugins.build_engines([broken_backend])
    assert len(record) == 1
    message = str(record[0].message)
    assert "Engine 'broken_backend'" in message


def test_remove_duplicates_warnings(dummy_duplicated_entrypoints):

    with pytest.warns(RuntimeWarning) as record:
        _ = plugins.remove_duplicates(dummy_duplicated_entrypoints)

    assert len(record) == 2
    message0 = str(record[0].message)
    message1 = str(record[1].message)
    assert "entrypoints" in message0
    assert "entrypoints" in message1


@mock.patch("pkg_resources.EntryPoint.load", mock.MagicMock(return_value=None))
def test_backends_dict_from_pkg():
    specs = [
        "engine1 = xarray.tests.test_plugins:backend_1",
        "engine2 = xarray.tests.test_plugins:backend_2",
    ]
    entrypoints = [pkg_resources.EntryPoint.parse(spec) for spec in specs]
    engines = plugins.backends_dict_from_pkg(entrypoints)
    assert len(engines) == 2
    assert engines.keys() == set(("engine1", "engine2"))


def test_set_missing_parameters():
    backend_1 = DummyBackendEntrypoint1
    backend_2 = DummyBackendEntrypoint2
    backend_2.open_dataset_parameters = ("filename_or_obj",)
    engines = {"engine_1": backend_1, "engine_2": backend_2}
    plugins.set_missing_parameters(engines)

    assert len(engines) == 2
    assert backend_1.open_dataset_parameters == ("filename_or_obj", "decoder")
    assert backend_2.open_dataset_parameters == ("filename_or_obj",)

    backend = DummyBackendEntrypointKwargs()
    backend.open_dataset_parameters = ("filename_or_obj", "decoder")
    plugins.set_missing_parameters({"engine": backend})
    assert backend.open_dataset_parameters == ("filename_or_obj", "decoder")

    backend = DummyBackendEntrypointArgs()
    backend.open_dataset_parameters = ("filename_or_obj", "decoder")
    plugins.set_missing_parameters({"engine": backend})
    assert backend.open_dataset_parameters == ("filename_or_obj", "decoder")


def test_set_missing_parameters_raise_error():

    backend = DummyBackendEntrypointKwargs()
    with pytest.raises(TypeError):
        plugins.set_missing_parameters({"engine": backend})

    backend = DummyBackendEntrypointArgs()
    with pytest.raises(TypeError):
        plugins.set_missing_parameters({"engine": backend})


@mock.patch(
    "pkg_resources.EntryPoint.load",
    mock.MagicMock(return_value=DummyBackendEntrypoint1),
)
def test_build_engines():
    dummy_pkg_entrypoint = pkg_resources.EntryPoint.parse(
        "cfgrib = xarray.tests.test_plugins:backend_1"
    )
    backend_entrypoints = plugins.build_engines([dummy_pkg_entrypoint])

    assert isinstance(backend_entrypoints["cfgrib"], DummyBackendEntrypoint1)
    assert backend_entrypoints["cfgrib"].open_dataset_parameters == (
        "filename_or_obj",
        "decoder",
    )


@mock.patch(
    "pkg_resources.EntryPoint.load",
    mock.MagicMock(return_value=DummyBackendEntrypoint1),
)
def test_build_engines_sorted():
    dummy_pkg_entrypoints = [
        pkg_resources.EntryPoint.parse(
            "dummy2 = xarray.tests.test_plugins:backend_1",
        ),
        pkg_resources.EntryPoint.parse(
            "dummy1 = xarray.tests.test_plugins:backend_1",
        ),
    ]
    backend_entrypoints = plugins.build_engines(dummy_pkg_entrypoints)
    backend_entrypoints = list(backend_entrypoints)

    indices = []
    for be in plugins.STANDARD_BACKENDS_ORDER:
        try:
            index = backend_entrypoints.index(be)
            backend_entrypoints.pop(index)
            indices.append(index)
        except ValueError:
            pass

    assert set(indices) < {0, -1}
    assert list(backend_entrypoints) == sorted(backend_entrypoints)


@mock.patch(
    "xarray.backends.plugins.list_engines",
    mock.MagicMock(return_value={"dummy": DummyBackendEntrypointArgs()}),
)
def test_no_matching_engine_found():
    with pytest.raises(
        ValueError, match="match in any of xarray's currently installed IO"
    ):
        plugins.guess_engine("not-valid")


@mock.patch(
    "xarray.backends.plugins.list_engines",
    mock.MagicMock(return_value={}),
)
def test_no_engines_installed():
    with pytest.raises(ValueError, match="no currently installed IO backends."):
        plugins.guess_engine("not-valid")
