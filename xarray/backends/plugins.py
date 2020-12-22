import inspect
import itertools
import logging
import warnings
from functools import lru_cache

import pkg_resources


class BackendEntrypoint:
    __slots__ = ("guess_can_open", "open_dataset", "open_dataset_parameters")

    def __init__(self, open_dataset, open_dataset_parameters=None, guess_can_open=None):
        self.open_dataset = open_dataset
        self.open_dataset_parameters = open_dataset_parameters
        self.guess_can_open = guess_can_open


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
                f"\nFound {matches_len} entrypoints for the engine name {name}:"
                f"\n {all_module_names}.\n It will be used: {selected_module_name}.",
                RuntimeWarning,
            )
    return unique_backend_entrypoints


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
    return tuple(parameters)


def create_engines_dict(backend_entrypoints):
    engines = {}
    for backend_ep in backend_entrypoints:
        name = backend_ep.name
        backend = backend_ep.load()
        engines[name] = backend
    return engines


def set_missing_parameters(engines):
    for name, backend in engines.items():
        if backend.open_dataset_parameters is None:
            open_dataset = backend.open_dataset
            backend.open_dataset_parameters = detect_parameters(open_dataset)


@lru_cache(maxsize=1)
def list_engines():
    entrypoints = pkg_resources.iter_entry_points("xarray.backends")
    backend_entrypoints = remove_duplicates(entrypoints)
    engines = create_engines_dict(backend_entrypoints)
    set_missing_parameters(engines)
    return engines


def guess_engine(store_spec):
    engines = list_engines()

    # use the pre-defined selection order for netCDF files
    for engine in ["netcdf4", "h5netcdf", "scipy"]:
        if engine in engines and engines[engine].guess_can_open(store_spec):
            return engine

    for engine, backend in engines.items():
        try:
            if backend.guess_can_open and backend.guess_can_open(store_spec):
                return engine
        except Exception:
            logging.exception(f"{engine!r} fails while guessing")

    raise ValueError("cannot guess the engine, try passing one explicitly")
