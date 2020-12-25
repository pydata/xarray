import functools
import inspect
import itertools
import logging
import typing as T
import warnings

import pkg_resources

from .cfgrib_ import cfgrib_backend
from .common import BackendEntrypoint
from .h5netcdf_ import h5netcdf_backend
from .netCDF4_ import netcdf4_backend
from .pseudonetcdf_ import pseudonetcdf_backend
from .pydap_ import pydap_backend
from .pynio_ import pynio_backend
from .scipy_ import scipy_backend
from .store import store_backend
from .zarr import zarr_backend

BACKEND_ENTRYPOINTS: T.Dict[str, BackendEntrypoint] = {
    "store": store_backend,
    "netcdf4": netcdf4_backend,
    "h5netcdf": h5netcdf_backend,
    "scipy": scipy_backend,
    "pseudonetcdf": pseudonetcdf_backend,
    "zarr": zarr_backend,
    "cfgrib": cfgrib_backend,
    "pydap": pydap_backend,
    "pynio": pynio_backend,
}


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


def build_engines(entrypoints):
    backend_entrypoints = BACKEND_ENTRYPOINTS.copy()
    pkg_entrypoints = remove_duplicates(entrypoints)
    external_backend_entrypoints = create_engines_dict(pkg_entrypoints)
    backend_entrypoints.update(external_backend_entrypoints)
    set_missing_parameters(backend_entrypoints)
    return backend_entrypoints


@functools.lru_cache(maxsize=1)
def list_engines():
    entrypoints = pkg_resources.iter_entry_points("xarray.backends")
    return build_engines(entrypoints)


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


def get_backend(engine):
    """Select open_dataset method based on current engine"""
    engines = list_engines()
    if engine not in engines:
        raise ValueError(
            f"unrecognized engine {engine} must be one of: {list(engines)}"
        )
    return engines[engine]
