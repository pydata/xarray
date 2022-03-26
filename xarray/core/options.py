import warnings
from typing import TYPE_CHECKING, Literal, TypedDict, Union

from .utils import FrozenDict

if TYPE_CHECKING:
    try:
        from matplotlib.colors import Colormap
    except ImportError:
        Colormap = str


class T_Options(TypedDict):
    arithmetic_join: Literal["inner", "outer", "left", "right", "exact"]
    cmap_divergent: Union[str, "Colormap"]
    cmap_sequential: Union[str, "Colormap"]
    display_max_rows: int
    display_style: Literal["text", "html"]
    display_width: int
    display_expand_attrs: Literal["default", True, False]
    display_expand_coords: Literal["default", True, False]
    display_expand_data_vars: Literal["default", True, False]
    display_expand_data: Literal["default", True, False]
    enable_cftimeindex: bool
    file_cache_maxsize: int
    keep_attrs: Literal["default", True, False]
    warn_for_unclosed_files: bool
    use_bottleneck: bool


OPTIONS: T_Options = {
    "arithmetic_join": "inner",
    "cmap_divergent": "RdBu_r",
    "cmap_sequential": "viridis",
    "display_max_rows": 12,
    "display_style": "html",
    "display_width": 80,
    "display_expand_attrs": "default",
    "display_expand_coords": "default",
    "display_expand_data_vars": "default",
    "display_expand_data": "default",
    "enable_cftimeindex": True,
    "file_cache_maxsize": 128,
    "keep_attrs": "default",
    "use_bottleneck": True,
    "warn_for_unclosed_files": False,
}

_JOIN_OPTIONS = frozenset(["inner", "outer", "left", "right", "exact"])
_DISPLAY_OPTIONS = frozenset(["text", "html"])


def _positive_integer(value):
    return isinstance(value, int) and value > 0


_VALIDATORS = {
    "arithmetic_join": _JOIN_OPTIONS.__contains__,
    "display_max_rows": _positive_integer,
    "display_style": _DISPLAY_OPTIONS.__contains__,
    "display_width": _positive_integer,
    "display_expand_attrs": lambda choice: choice in [True, False, "default"],
    "display_expand_coords": lambda choice: choice in [True, False, "default"],
    "display_expand_data_vars": lambda choice: choice in [True, False, "default"],
    "display_expand_data": lambda choice: choice in [True, False, "default"],
    "enable_cftimeindex": lambda value: isinstance(value, bool),
    "file_cache_maxsize": _positive_integer,
    "keep_attrs": lambda choice: choice in [True, False, "default"],
    "use_bottleneck": lambda value: isinstance(value, bool),
    "warn_for_unclosed_files": lambda value: isinstance(value, bool),
}


def _set_file_cache_maxsize(value):
    from ..backends.file_manager import FILE_CACHE

    FILE_CACHE.maxsize = value


def _warn_on_setting_enable_cftimeindex(enable_cftimeindex):
    warnings.warn(
        "The enable_cftimeindex option is now a no-op "
        "and will be removed in a future version of xarray.",
        FutureWarning,
    )


_SETTERS = {
    "enable_cftimeindex": _warn_on_setting_enable_cftimeindex,
    "file_cache_maxsize": _set_file_cache_maxsize,
}


def _get_boolean_with_default(option, default):
    global_choice = OPTIONS[option]

    if global_choice == "default":
        return default
    elif global_choice in [True, False]:
        return global_choice
    else:
        raise ValueError(
            f"The global option {option} must be one of True, False or 'default'."
        )


def _get_keep_attrs(default):
    return _get_boolean_with_default("keep_attrs", default)


class set_options:
    """
    Set options for xarray in a controlled context.

    Parameters
    ----------
    arithmetic_join : {"inner", "outer", "left", "right", "exact"}, default: "inner"
        DataArray/Dataset alignment in binary operations.
    cmap_divergent : str or matplotlib.colors.Colormap, default: "RdBu_r"
        Colormap to use for divergent data plots. If string, must be
        matplotlib built-in colormap. Can also be a Colormap object
        (e.g. mpl.cm.magma)
    cmap_sequential : str or matplotlib.colors.Colormap, default: "viridis"
        Colormap to use for nondivergent data plots. If string, must be
        matplotlib built-in colormap. Can also be a Colormap object
        (e.g. mpl.cm.magma)
    display_expand_attrs : {"default", True, False}:
        Whether to expand the attributes section for display of
        ``DataArray`` or ``Dataset`` objects. Can be

        * ``True`` : to always expand attrs
        * ``False`` : to always collapse attrs
        * ``default`` : to expand unless over a pre-defined limit
    display_expand_coords : {"default", True, False}:
        Whether to expand the coordinates section for display of
        ``DataArray`` or ``Dataset`` objects. Can be

        * ``True`` : to always expand coordinates
        * ``False`` : to always collapse coordinates
        * ``default`` : to expand unless over a pre-defined limit
    display_expand_data : {"default", True, False}:
        Whether to expand the data section for display of ``DataArray``
        objects. Can be

        * ``True`` : to always expand data
        * ``False`` : to always collapse data
        * ``default`` : to expand unless over a pre-defined limit
    display_expand_data_vars : {"default", True, False}:
        Whether to expand the data variables section for display of
        ``Dataset`` objects. Can be

        * ``True`` : to always expand data variables
        * ``False`` : to always collapse data variables
        * ``default`` : to expand unless over a pre-defined limit
    display_max_rows : int, default: 12
        Maximum display rows.
    display_style : {"text", "html"}, default: "html"
        Display style to use in jupyter for xarray objects.
    display_width : int, default: 80
        Maximum display width for ``repr`` on xarray objects.
    file_cache_maxsize : int, default: 128
        Maximum number of open files to hold in xarray's
        global least-recently-usage cached. This should be smaller than
        your system's per-process file descriptor limit, e.g.,
        ``ulimit -n`` on Linux.
    keep_attrs : {"default", True, False}
        Whether to keep attributes on xarray Datasets/dataarrays after
        operations. Can be

        * ``True`` : to always keep attrs
        * ``False`` : to always discard attrs
        * ``default`` : to use original logic that attrs should only
          be kept in unambiguous circumstances
    use_bottleneck : bool, default: True
        Whether to use ``bottleneck`` to accelerate 1D reductions and
        1D rolling reduction operations.
    warn_for_unclosed_files : bool, default: False
        Whether or not to issue a warning when unclosed files are
        deallocated. This is mostly useful for debugging.

    Examples
    --------
    It is possible to use ``set_options`` either as a context manager:

    >>> ds = xr.Dataset({"x": np.arange(1000)})
    >>> with xr.set_options(display_width=40):
    ...     print(ds)
    ...
    <xarray.Dataset>
    Dimensions:  (x: 1000)
    Coordinates:
      * x        (x) int64 0 1 2 ... 998 999
    Data variables:
        *empty*

    Or to set global options:

    >>> xr.set_options(display_width=80)  # doctest: +ELLIPSIS
    <xarray.core.options.set_options object at 0x...>
    """

    def __init__(self, **kwargs):
        self.old = {}
        for k, v in kwargs.items():
            if k not in OPTIONS:
                raise ValueError(
                    f"argument name {k!r} is not in the set of valid options {set(OPTIONS)!r}"
                )
            if k in _VALIDATORS and not _VALIDATORS[k](v):
                if k == "arithmetic_join":
                    expected = f"Expected one of {_JOIN_OPTIONS!r}"
                elif k == "display_style":
                    expected = f"Expected one of {_DISPLAY_OPTIONS!r}"
                else:
                    expected = ""
                raise ValueError(
                    f"option {k!r} given an invalid value: {v!r}. " + expected
                )
            self.old[k] = OPTIONS[k]
        self._apply_update(kwargs)

    def _apply_update(self, options_dict):
        for k, v in options_dict.items():
            if k in _SETTERS:
                _SETTERS[k](v)
        OPTIONS.update(options_dict)

    def __enter__(self):
        return

    def __exit__(self, type, value, traceback):
        self._apply_update(self.old)


def get_options():
    """
    Get options for xarray.

    See Also
    ----------
    set_options

    """
    return FrozenDict(OPTIONS)
