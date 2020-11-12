import inspect
import itertools
import typing as T
import warnings
from functools import lru_cache

import entrypoints  # type: ignore


class BackendEntrypoint:
    __slots__ = ("open_dataset", "open_dataset_parameters")

    def __init__(self, open_dataset, open_dataset_parameters=None):
        self.open_dataset = open_dataset
        self.open_dataset_parameters = open_dataset_parameters


def get_entrypoint_id(entrypoint):
    name = entrypoint.name
    obj = entrypoint.object_name or ""
    module = entrypoint.module_name
    return (name, f"{module}:{obj}")


def filter_unique_ids(backend_entrypoints):
    entrypoints_id = set()
    backend_entrypoints_unique = []
    for entrypoint in backend_entrypoints:
        id = get_entrypoint_id(entrypoint)
        if id not in entrypoints_id:
            backend_entrypoints_unique.append(entrypoint)
            entrypoints_id.add(id)
    return backend_entrypoints_unique


def warning_on_entrypoints_conflict(backend_entrypoints, backend_entrypoints_all):

    # sort and group entrypoints by name
    key_name = lambda ep: ep.name
    backend_entrypoints_all_unique_ids = sorted(backend_entrypoints_all, key=key_name)
    backend_entrypoints_ids_grouped = itertools.groupby(
        backend_entrypoints_all_unique_ids, key=key_name
    )

    # check if there are multiple entrypoints for the same name
    for name, matches in backend_entrypoints_ids_grouped:
        matches = list(matches)
        matches_len = len(matches)
        if matches_len > 1:
            selected_entrypoint = backend_entrypoints[name]
            warnings.warn(
                f"\nFound {matches_len} entrypoints for the engine name {name}:"
                f"\n {matches}.\n It will be used: {selected_entrypoint}.",
            )


def detect_parameters(open_dataset):
    signature = inspect.signature(open_dataset)
    parameters = signature.parameters
    for name, param in parameters.items():
        if param.kind in (
            inspect.Parameter.VAR_KEYWORD,
            inspect.Parameter.VAR_POSITIONAL,
        ):
            raise TypeError(
                f"All the parameters in {open_dataset!r} signature should be explicit. "
                "*args and **kwargs is not supported"
            )
    return set(parameters)


@lru_cache(maxsize=1)
def detect_engines():
    backend_entrypoints = entrypoints.get_group_named("xarray.backends")
    backend_entrypoints_all = entrypoints.get_group_all("xarray.backends")
    backend_entrypoints_all = filter_unique_ids(backend_entrypoints_all)

    if len(backend_entrypoints_all) != len(backend_entrypoints):
        warning_on_entrypoints_conflict(backend_entrypoints, backend_entrypoints_all)

    engines: T.Dict[str, T.Dict[str, T.Any]] = {}
    for name, backend in backend_entrypoints.items():
        backend = backend.load()
        engines[name] = backend

        if backend.open_dataset_parameters is None:
            open_dataset = backend.open_dataset
            backend.open_dataset_parameters = detect_parameters(open_dataset)
    return engines
