import inspect
import itertools
import warnings
from functools import lru_cache

import pkg_resources

class BackendEntrypoint:
    __slots__ = ("open_dataset", "open_dataset_parameters")

    def __init__(self, open_dataset, open_dataset_parameters=None):
        self.open_dataset = open_dataset
        self.open_dataset_parameters = open_dataset_parameters


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
    from .. import backends

    backend_entrypoints = dict(
        zarr=backends.zarr.zarr_backend,
        h5netcdf=backends.h5netcdf_.h5netcdf_backend,
        cfgrib=backends.cfgrib_.cfgrib_backend,
        scipy=backends.scipy_.scipy_backend,
        pynio=backends.pynio_.pynio_backend,
        pseudonetcdf=backends.pseudonetcdf_.pseudonetcdf_backend,
        netcdf4=backends.netCDF4_.netcdf4_backend,
        store=backends.store.store_backend,
    )

    entrypoints = pkg_resources.iter_entry_points("xarray.backends")
    external_backend_entrypoints = remove_duplicates(entrypoints)
    backend_entrypoints.update(external_backend_entrypoints)
    engines = create_engines_dict(backend_entrypoints)
    set_missing_parameters(engines)
    return engines
