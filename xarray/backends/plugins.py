import functools
import inspect
import itertools
import logging
import warnings

import pkg_resources

from .common import BACKEND_ENTRYPOINTS


def remove_duplicates(backend_entrypoints):

    # sort and group entrypoints by name
    backend_entrypoints = sorted(backend_entrypoints, key=lambda ep: ep.name)
    backend_entrypoints_grouped = itertools.groupby(
        backend_entrypoints, key=lambda ep: ep.name
    )
    # check if there are multiple entrypoints for the same name
    unique_backend_entrypoints = []
    for name, matches in backend_entrypoints_grouped:
        matches = list(matches)
        unique_backend_entrypoints.append(matches[0])
        matches_len = len(matches)
        if matches_len > 1:
            selected_module_name = matches[0].module_name
            all_module_names = [e.module_name for e in matches]
            warnings.warn(
                f"Found {matches_len} entrypoints for the engine name {name}:"
                f"\n {all_module_names}.\n It will be used: {selected_module_name}.",
                RuntimeWarning,
            )
    return unique_backend_entrypoints


def detect_parameters(open_dataset):
    signature = inspect.signature(open_dataset)
    parameters = signature.parameters
    parameters_list = []
    for name, param in parameters.items():
        if param.kind in (
            inspect.Parameter.VAR_KEYWORD,
            inspect.Parameter.VAR_POSITIONAL,
        ):
            raise TypeError(
                f"All the parameters in {open_dataset!r} signature should be explicit. "
                "*args and **kwargs is not supported"
            )
        if name != "self":
            parameters_list.append(name)
    return tuple(parameters_list)


def create_engines_dict(backend_entrypoints):
    engines = {}
    for backend_ep in backend_entrypoints:
        name = backend_ep.name
        backend = backend_ep.load()
        engines[name] = backend
    return engines


def set_missing_parameters(backend_entrypoints):
    for name, backend in backend_entrypoints.items():
        if backend.open_dataset_parameters is None:
            open_dataset = backend.open_dataset
            backend.open_dataset_parameters = detect_parameters(open_dataset)


def build_engines(entrypoints):
    backend_entrypoints = BACKEND_ENTRYPOINTS.copy()
    pkg_entrypoints = remove_duplicates(entrypoints)
    external_backend_entrypoints = create_engines_dict(pkg_entrypoints)
    backend_entrypoints.update(external_backend_entrypoints)
    set_missing_parameters(backend_entrypoints)
    engines = {}
    for name, backend in backend_entrypoints.items():
        engines[name] = backend()
    return engines


@functools.lru_cache(maxsize=1)
def list_engines():
    entrypoints = pkg_resources.iter_entry_points("xarray.backends")
    return build_engines(entrypoints)


def guess_engine(store_spec):
    engines = list_engines()

    for engine, backend in engines.items():
        try:
            if backend.guess_can_open and backend.guess_can_open(store_spec):
                return engine
        except Exception:
            logging.exception(f"{engine!r} fails while guessing")

    raise ValueError("cannot guess the engine, try passing one explicitly")


def get_backend(engine):
    """Select open_dataset method based on current engine"""
    engines = list_engines()
    if engine not in engines:
        raise ValueError(
            f"unrecognized engine {engine} must be one of: {list(engines)}"
        )
    return engines[engine]
