from __future__ import annotations

import functools
import inspect
import itertools
import os
import warnings
from collections.abc import Callable
from html import escape
from importlib.metadata import entry_points
from typing import TYPE_CHECKING, Any, TextIO

from xarray.backends.common import BACKEND_ENTRYPOINTS, BackendEntrypoint
from xarray.core.options import OPTIONS
from xarray.core.utils import is_remote_uri, module_available
from xarray.util._report import package_version, show_report

if TYPE_CHECKING:
    from importlib.metadata import EntryPoint, EntryPoints

    from xarray.backends.common import AbstractDataStore
    from xarray.core.types import ReadBuffer


def remove_duplicates(entrypoints: EntryPoints) -> list[EntryPoint]:
    # sort and group entrypoints by name
    entrypoints_sorted = sorted(entrypoints, key=lambda ep: ep.name)
    entrypoints_grouped = itertools.groupby(entrypoints_sorted, key=lambda ep: ep.name)
    # check if there are multiple entrypoints for the same name
    unique_entrypoints = []
    for name, _matches in entrypoints_grouped:
        # remove equal entrypoints
        matches = list(set(_matches))
        unique_entrypoints.append(matches[0])
        matches_len = len(matches)
        if matches_len > 1:
            all_module_names = [e.value.split(":")[0] for e in matches]
            selected_module_name = all_module_names[0]
            warnings.warn(
                f"Found {matches_len} entrypoints for the engine name {name}:"
                f"\n {all_module_names}.\n "
                f"The entrypoint {selected_module_name} will be used.",
                RuntimeWarning,
                stacklevel=2,
            )
    return unique_entrypoints


def detect_parameters(open_dataset: Callable) -> tuple[str, ...]:
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


def backends_dict_from_pkg(
    entrypoints: list[EntryPoint],
) -> dict[str, type[BackendEntrypoint]]:
    backend_entrypoints = {}
    for entrypoint in entrypoints:
        name = entrypoint.name
        try:
            backend = entrypoint.load()
            backend_entrypoints[name] = backend
        except Exception as ex:
            warnings.warn(
                f"Engine {name!r} loading failed:\n{ex}", RuntimeWarning, stacklevel=2
            )
    return backend_entrypoints


def set_missing_parameters(
    backend_entrypoints: dict[str, type[BackendEntrypoint]],
) -> None:
    for backend in backend_entrypoints.values():
        if backend.open_dataset_parameters is None:
            open_dataset = backend.open_dataset
            backend.open_dataset_parameters = detect_parameters(open_dataset)


def sort_backends(
    backend_entrypoints: dict[str, type[BackendEntrypoint]],
) -> dict[str, type[BackendEntrypoint]]:
    ordered_backends_entrypoints: dict[str, type[BackendEntrypoint]] = {}
    for be_name in OPTIONS["netcdf_engine_order"]:
        if backend_entrypoints.get(be_name) is not None:
            ordered_backends_entrypoints[be_name] = backend_entrypoints.pop(be_name)
    ordered_backends_entrypoints.update(
        {name: backend_entrypoints[name] for name in sorted(backend_entrypoints)}
    )
    return ordered_backends_entrypoints


def build_engines(entrypoints: EntryPoints) -> dict[str, BackendEntrypoint]:
    backend_entrypoints: dict[str, type[BackendEntrypoint]] = {}
    for backend_name, (module_name, backend) in BACKEND_ENTRYPOINTS.items():
        if module_name is None or module_available(module_name):
            backend_entrypoints[backend_name] = backend
    entrypoints_unique = remove_duplicates(entrypoints)
    external_backend_entrypoints = backends_dict_from_pkg(entrypoints_unique)
    backend_entrypoints.update(external_backend_entrypoints)
    backend_entrypoints = sort_backends(backend_entrypoints)
    set_missing_parameters(backend_entrypoints)
    return {name: backend() for name, backend in backend_entrypoints.items()}


@functools.lru_cache(maxsize=1)
def list_engines() -> dict[str, BackendEntrypoint]:
    """
    Return a dictionary of available engines and their BackendEntrypoint objects.

    Returns
    -------
    dictionary

    Notes
    -----
    This function lives in the backends namespace (``engs=xr.backends.list_engines()``).
    If available, more information is available about each backend via ``engs["eng_name"]``.
    """
    entrypoints = entry_points(group="xarray.backends")
    return build_engines(entrypoints)


def refresh_engines() -> None:
    """Refreshes the backend engines based on installed packages."""
    list_engines.cache_clear()


def _engine_sources() -> dict[str, tuple[str, str | None]]:
    """Map each engine name to its providing package and version."""
    sources: dict[str, tuple[str, str | None]] = dict.fromkeys(
        BACKEND_ENTRYPOINTS, ("xarray", None)
    )
    for ep in entry_points(group="xarray.backends"):
        if ep.dist is not None:
            sources[ep.name] = (ep.dist.name, ep.dist.version)
        else:
            sources[ep.name] = (ep.module.partition(".")[0], None)
    return sources


def show_backends(file: TextIO | None = None, markdown: bool = False) -> None:
    """Print the available backends (engines) and information about them.

    The backends are listed in the order in which they are tried when opening
    a file without specifying the ``engine``.
    Use :py:func:`xarray.backends.list_engines` to get the backend objects.
    In Jupyter notebooks the backends are displayed formatted.

    Parameters
    ----------
    file : file-like, optional
        print to the given file-like object. Defaults to sys.stdout.
    markdown : bool, default: False
        print a markdown list wrapped in a ``<details>`` block, which can be
        pasted as is into a GitHub issue.
    """
    engines = list_engines()
    sources = _engine_sources()

    text_lines = []
    md_lines = []
    html_items = []
    for name, backend in engines.items():
        package, version = sources.get(name, ("unknown", None))
        summary = f"from {package}"
        if version is not None:
            summary += f" {version}"
        details = []
        if package == "xarray" and name in BACKEND_ENTRYPOINTS:
            dependency = BACKEND_ENTRYPOINTS[name][0]
            if dependency is not None:
                details.append(f"using {dependency} {package_version(dependency)}")
        if backend.supports_groups:
            details.append("supports groups")
        if details:
            summary += f" ({', '.join(details)})"
        cls_name = type(backend).__name__
        text_lines.append(f"{name}: {cls_name} {summary}")
        md_lines.append(f"- **{name}**: `{cls_name}` {summary}")

        extra = []
        html_extra = []
        if backend.description:
            description = " ".join(backend.description.split())
            extra.append(description)
            html_extra.append(escape(description))
        if backend.url:
            extra.append(backend.url)
            url = escape(backend.url)
            html_extra.append(f'<a href="{url}" target="_blank">{url}</a>')
        text_lines.extend(f"    {line}" for line in extra)
        md_lines.extend(f"  - {line}" for line in extra)
        html_items.append(
            f"<li><strong>{escape(name)}</strong>: <code>{escape(cls_name)}</code> "
            f"{escape(summary)}<ul>{''.join(f'<li>{x}</li>' for x in html_extra)}</ul></li>"
        )

    empty = "No backends available."
    text = "\n".join(text_lines) or empty
    md = "\n".join(md_lines) or empty
    html = f"<ul>{''.join(html_items)}</ul>" if html_items else f"<p>{empty}</p>"
    show_report("AVAILABLE BACKENDS", text, md, html, file=file, as_markdown=markdown)


def guess_engine(
    store_spec: str
    | os.PathLike[Any]
    | ReadBuffer
    | bytes
    | memoryview
    | AbstractDataStore,
    must_support_groups: bool = False,
) -> str | type[BackendEntrypoint]:
    engines = list_engines()

    for engine, backend in engines.items():
        if must_support_groups and not backend.supports_groups:
            continue
        try:
            if backend.guess_can_open(store_spec):
                return engine
        except PermissionError:
            raise
        except Exception:
            warnings.warn(
                f"{engine!r} fails while guessing", RuntimeWarning, stacklevel=2
            )

    compatible_engines = []
    for engine, (_, backend_cls) in BACKEND_ENTRYPOINTS.items():
        try:
            backend = backend_cls()
            if must_support_groups and not backend.supports_groups:
                continue
            if backend.guess_can_open(store_spec):
                compatible_engines.append(engine)
        except Exception:
            warnings.warn(
                f"{engine!r} fails while guessing", RuntimeWarning, stacklevel=2
            )

    installed_engines = [k for k in engines if k != "store"]
    if not compatible_engines:
        if installed_engines:
            error_msg = (
                "did not find a match in any of xarray's currently installed IO "
                f"backends {installed_engines}. Consider explicitly selecting one of the "
                "installed engines via the ``engine`` parameter, or installing "
                "additional IO dependencies, see:\n"
                "https://docs.xarray.dev/en/stable/getting-started-guide/installing.html\n"
                "https://docs.xarray.dev/en/stable/user-guide/io.html"
            )
        elif must_support_groups:
            error_msg = (
                "xarray is unable to open this file because it has no currently "
                "installed IO backends that support reading groups (e.g., h5netcdf "
                "or netCDF4-python). Xarray's read/write support requires "
                "installing optional IO dependencies, see:\n"
                "https://docs.xarray.dev/en/stable/getting-started-guide/installing.html\n"
                "https://docs.xarray.dev/en/stable/user-guide/io"
            )
        else:
            error_msg = (
                "xarray is unable to open this file because it has no currently "
                "installed IO backends. Xarray's read/write support requires "
                "installing optional IO dependencies, see:\n"
                "https://docs.xarray.dev/en/stable/getting-started-guide/installing.html\n"
                "https://docs.xarray.dev/en/stable/user-guide/io"
            )
    else:
        error_msg = (
            "found the following matches with the input file in xarray's IO "
            f"backends: {compatible_engines}. But their dependencies may not be installed, see:\n"
            "https://docs.xarray.dev/en/stable/user-guide/io.html \n"
            "https://docs.xarray.dev/en/stable/getting-started-guide/installing.html"
        )

    if isinstance(store_spec, str | os.PathLike):
        store_spec_str = str(store_spec)
        if not is_remote_uri(store_spec_str) and not os.path.exists(store_spec_str):
            raise FileNotFoundError(f"No such file: '{store_spec_str}'")

    raise ValueError(error_msg)


def get_backend(engine: str | type[BackendEntrypoint]) -> BackendEntrypoint:
    """Select open_dataset method based on current engine."""
    if isinstance(engine, str):
        if engine in BACKEND_ENTRYPOINTS:
            # fast path for built-in engines
            backend_cls = BACKEND_ENTRYPOINTS[engine][1]
            set_missing_parameters({engine: backend_cls})
            backend = backend_cls()
        else:
            engines = list_engines()
            if engine not in engines:
                raise ValueError(
                    f"unrecognized engine '{engine}' must be one of your download engines: {list(engines)}. "
                    "To install additional dependencies, see:\n"
                    "https://docs.xarray.dev/en/stable/user-guide/io.html \n"
                    "https://docs.xarray.dev/en/stable/getting-started-guide/installing.html"
                )
            backend = engines[engine]
    elif issubclass(engine, BackendEntrypoint):
        backend = engine()
    else:
        raise TypeError(
            "engine must be a string or a subclass of "
            f"xarray.backends.BackendEntrypoint: {engine}"
        )

    return backend
