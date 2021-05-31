import warnings

ARITHMETIC_JOIN = "arithmetic_join"
CMAP_DIVERGENT = "cmap_divergent"
CMAP_SEQUENTIAL = "cmap_sequential"
DISPLAY_MAX_ROWS = "display_max_rows"
DISPLAY_STYLE = "display_style"
DISPLAY_WIDTH = "display_width"
DISPLAY_EXPAND_ATTRS = "display_expand_attrs"
DISPLAY_EXPAND_COORDS = "display_expand_coords"
DISPLAY_EXPAND_DATA_VARS = "display_expand_data_vars"
DISPLAY_EXPAND_DATA = "display_expand_data"
ENABLE_CFTIMEINDEX = "enable_cftimeindex"
FILE_CACHE_MAXSIZE = "file_cache_maxsize"
KEEP_ATTRS = "keep_attrs"
WARN_FOR_UNCLOSED_FILES = "warn_for_unclosed_files"


OPTIONS = {
    ARITHMETIC_JOIN: "inner",
    CMAP_DIVERGENT: "RdBu_r",
    CMAP_SEQUENTIAL: "viridis",
    DISPLAY_MAX_ROWS: 12,
    DISPLAY_STYLE: "html",
    DISPLAY_WIDTH: 80,
    DISPLAY_EXPAND_ATTRS: "default",
    DISPLAY_EXPAND_COORDS: "default",
    DISPLAY_EXPAND_DATA_VARS: "default",
    DISPLAY_EXPAND_DATA: "default",
    ENABLE_CFTIMEINDEX: True,
    FILE_CACHE_MAXSIZE: 128,
    KEEP_ATTRS: "default",
    WARN_FOR_UNCLOSED_FILES: False,
}

_JOIN_OPTIONS = frozenset(["inner", "outer", "left", "right", "exact"])
_DISPLAY_OPTIONS = frozenset(["text", "html"])


def _positive_integer(value):
    return isinstance(value, int) and value > 0


_VALIDATORS = {
    ARITHMETIC_JOIN: _JOIN_OPTIONS.__contains__,
    DISPLAY_MAX_ROWS: _positive_integer,
    DISPLAY_STYLE: _DISPLAY_OPTIONS.__contains__,
    DISPLAY_WIDTH: _positive_integer,
    DISPLAY_EXPAND_ATTRS: lambda choice: choice in [True, False, "default"],
    DISPLAY_EXPAND_COORDS: lambda choice: choice in [True, False, "default"],
    DISPLAY_EXPAND_DATA_VARS: lambda choice: choice in [True, False, "default"],
    DISPLAY_EXPAND_DATA: lambda choice: choice in [True, False, "default"],
    ENABLE_CFTIMEINDEX: lambda value: isinstance(value, bool),
    FILE_CACHE_MAXSIZE: _positive_integer,
    KEEP_ATTRS: lambda choice: choice in [True, False, "default"],
    WARN_FOR_UNCLOSED_FILES: lambda value: isinstance(value, bool),
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
    ENABLE_CFTIMEINDEX: _warn_on_setting_enable_cftimeindex,
    FILE_CACHE_MAXSIZE: _set_file_cache_maxsize,
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
    """Set options for xarray in a controlled context.

    Currently supported options:

    - ``display_width``: maximum display width for ``repr`` on xarray objects.
      Default: ``80``.
    - ``display_max_rows``: maximum display rows. Default: ``12``.
    - ``arithmetic_join``: DataArray/Dataset alignment in binary operations.
      Default: ``'inner'``.
    - ``file_cache_maxsize``: maximum number of open files to hold in xarray's
      global least-recently-usage cached. This should be smaller than your
      system's per-process file descriptor limit, e.g., ``ulimit -n`` on Linux.
      Default: 128.
    - ``warn_for_unclosed_files``: whether or not to issue a warning when
      unclosed files are deallocated (default False). This is mostly useful
      for debugging.
    - ``cmap_sequential``: colormap to use for nondivergent data plots.
      Default: ``viridis``. If string, must be matplotlib built-in colormap.
      Can also be a Colormap object (e.g. mpl.cm.magma)
    - ``cmap_divergent``: colormap to use for divergent data plots.
      Default: ``RdBu_r``. If string, must be matplotlib built-in colormap.
      Can also be a Colormap object (e.g. mpl.cm.magma)
    - ``keep_attrs``: rule for whether to keep attributes on xarray
      Datasets/dataarrays after operations. Either ``True`` to always keep
      attrs, ``False`` to always discard them, or ``'default'`` to use original
      logic that attrs should only be kept in unambiguous circumstances.
      Default: ``'default'``.
    - ``display_style``: display style to use in jupyter for xarray objects.
      Default: ``'text'``. Other options are ``'html'``.
    - ``display_expand_attrs``: whether to expand the attributes section for
      display of ``DataArray`` or ``Dataset`` objects. Can be ``True`` to always
      expand, ``False`` to always collapse, or ``default`` to expand unless over
      a pre-defined limit. Default: ``default``.
    - ``display_expand_coords``: whether to expand the coordinates section for
      display of ``DataArray`` or ``Dataset`` objects. Can be ``True`` to always
      expand, ``False`` to always collapse, or ``default`` to expand unless over
      a pre-defined limit. Default: ``default``.
    - ``display_expand_data``: whether to expand the data section for display
      of ``DataArray`` objects. Can be ``True`` to always expand, ``False`` to
      always collapse, or ``default`` to expand unless over a pre-defined limit.
      Default: ``default``.
    - ``display_expand_data_vars``: whether to expand the data variables section
      for display of ``Dataset`` objects. Can be ``True`` to always
      expand, ``False`` to always collapse, or ``default`` to expand unless over
      a pre-defined limit. Default: ``default``.


    You can use ``set_options`` either as a context manager:

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
                if k == ARITHMETIC_JOIN:
                    expected = f"Expected one of {_JOIN_OPTIONS!r}"
                elif k == DISPLAY_STYLE:
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
