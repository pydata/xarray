import functools
import inspect
import itertools
import warnings

import pkg_resources

from .common import BACKEND_ENTRYPOINTS, BackendEntrypoint

STANDARD_BACKENDS_ORDER = ["netcdf4", "h5netcdf", "scipy"]


def remove_duplicates(pkg_entrypoints):

    # sort and group entrypoints by name
    pkg_entrypoints = sorted(pkg_entrypoints, key=lambda ep: ep.name)
    pkg_entrypoints_grouped = itertools.groupby(pkg_entrypoints, key=lambda ep: ep.name)
    # check if there are multiple entrypoints for the same name
    unique_pkg_entrypoints = []
    for name, matches in pkg_entrypoints_grouped:
        matches = list(matches)
        unique_pkg_entrypoints.append(matches[0])
        matches_len = len(matches)
        if matches_len > 1:
            selected_module_name = matches[0].module_name
            all_module_names = [e.module_name for e in matches]
            warnings.warn(
                f"Found {matches_len} entrypoints for the engine name {name}:"
                f"\n {all_module_names}.\n It will be used: {selected_module_name}.",
                RuntimeWarning,
            )
    return unique_pkg_entrypoints


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


def backends_dict_from_pkg(pkg_entrypoints):
    backend_entrypoints = {}
    for pkg_ep in pkg_entrypoints:
        name = pkg_ep.name
        try:
            backend = pkg_ep.load()
            backend_entrypoints[name] = backend
        except Exception as ex:
            warnings.warn(f"Engine {name!r} loading failed:\n{ex}", RuntimeWarning)
    return backend_entrypoints


def set_missing_parameters(backend_entrypoints):
    for name, backend in backend_entrypoints.items():
        if backend.open_dataset_parameters is None:
            open_dataset = backend.open_dataset
            backend.open_dataset_parameters = detect_parameters(open_dataset)


def sort_backends(backend_entrypoints):
    ordered_backends_entrypoints = {}
    for be_name in STANDARD_BACKENDS_ORDER:
        if backend_entrypoints.get(be_name, None) is not None:
            ordered_backends_entrypoints[be_name] = backend_entrypoints.pop(be_name)
    ordered_backends_entrypoints.update(
        {name: backend_entrypoints[name] for name in sorted(backend_entrypoints)}
    )
    return ordered_backends_entrypoints


def build_engines(pkg_entrypoints):
    backend_entrypoints = BACKEND_ENTRYPOINTS.copy()
    pkg_entrypoints = remove_duplicates(pkg_entrypoints)
    external_backend_entrypoints = backends_dict_from_pkg(pkg_entrypoints)
    backend_entrypoints.update(external_backend_entrypoints)
    backend_entrypoints = sort_backends(backend_entrypoints)
    set_missing_parameters(backend_entrypoints)
    return {name: backend() for name, backend in backend_entrypoints.items()}


@functools.lru_cache(maxsize=1)
def list_engines():
    pkg_entrypoints = pkg_resources.iter_entry_points("xarray.backends")
    return build_engines(pkg_entrypoints)


def guess_engine(store_spec):
    engines = list_engines()

    for engine, backend in engines.items():
        try:
            if backend.guess_can_open and backend.guess_can_open(store_spec):
                return engine
        except Exception:
            warnings.warn(f"{engine!r} fails while guessing", RuntimeWarning)

    installed = [k for k in engines if k != "store"]
    if installed:
        raise ValueError(
            "did not find a match in any of xarray's currently installed IO "
            f"backends {installed}. Consider explicitly selecting one of the "
            "installed backends via the ``engine`` parameter to "
            "xarray.open_dataset(), or installing additional IO dependencies:\n"
            "http://xarray.pydata.org/en/stable/getting-started-guide/installing.html\n"
            "http://xarray.pydata.org/en/stable/user-guide/io.html"
        )
    else:
        raise ValueError(
            "xarray is unable to open this file because it has no currently "
            "installed IO backends. Xarray's read/write support requires "
            "installing optional dependencies:\n"
            "http://xarray.pydata.org/en/stable/getting-started-guide/installing.html\n"
            "http://xarray.pydata.org/en/stable/user-guide/io.html"
        )


def get_backend(engine):
    """Select open_dataset method based on current engine."""
    if isinstance(engine, str):
        engines = list_engines()
        if engine not in engines:
            raise ValueError(
                f"unrecognized engine {engine} must be one of: {list(engines)}"
            )
        backend = engines[engine]
    elif isinstance(engine, type) and issubclass(engine, BackendEntrypoint):
        backend = engine
    else:
        raise TypeError(
            (
                "engine must be a string or a subclass of "
                f"xarray.backends.BackendEntrypoint: {engine}"
            )
        )

    return backend
