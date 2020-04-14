import warnings

DISPLAY_WIDTH = "display_width"
ARITHMETIC_JOIN = "arithmetic_join"
ENABLE_CFTIMEINDEX = "enable_cftimeindex"
FILE_CACHE_MAXSIZE = "file_cache_maxsize"
WARN_FOR_UNCLOSED_FILES = "warn_for_unclosed_files"
CMAP_SEQUENTIAL = "cmap_sequential"
CMAP_DIVERGENT = "cmap_divergent"
KEEP_ATTRS = "keep_attrs"
DISPLAY_STYLE = "display_style"


OPTIONS = {
    DISPLAY_WIDTH: 80,
    ARITHMETIC_JOIN: "inner",
    ENABLE_CFTIMEINDEX: True,
    FILE_CACHE_MAXSIZE: 128,
    WARN_FOR_UNCLOSED_FILES: False,
    CMAP_SEQUENTIAL: "viridis",
    CMAP_DIVERGENT: "RdBu_r",
    KEEP_ATTRS: "default",
    DISPLAY_STYLE: "html",
}

_JOIN_OPTIONS = frozenset(["inner", "outer", "left", "right", "exact"])
_DISPLAY_OPTIONS = frozenset(["text", "html"])


def _positive_integer(value):
    return isinstance(value, int) and value > 0


_VALIDATORS = {
    DISPLAY_WIDTH: _positive_integer,
    ARITHMETIC_JOIN: _JOIN_OPTIONS.__contains__,
    ENABLE_CFTIMEINDEX: lambda value: isinstance(value, bool),
    FILE_CACHE_MAXSIZE: _positive_integer,
    WARN_FOR_UNCLOSED_FILES: lambda value: isinstance(value, bool),
    KEEP_ATTRS: lambda choice: choice in [True, False, "default"],
    DISPLAY_STYLE: _DISPLAY_OPTIONS.__contains__,
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
    FILE_CACHE_MAXSIZE: _set_file_cache_maxsize,
    ENABLE_CFTIMEINDEX: _warn_on_setting_enable_cftimeindex,
}


def _get_keep_attrs(default):
    global_choice = OPTIONS["keep_attrs"]

    if global_choice == "default":
        return default
    elif global_choice in [True, False]:
        return global_choice
    else:
        raise ValueError(
            "The global option keep_attrs must be one of" " True, False or 'default'."
        )


class set_options:
    """Set options for xarray in a controlled context.

    Currently supported options:

    - ``display_width``: maximum display width for ``repr`` on xarray objects.
      Default: ``80``.
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


    You can use ``set_options`` either as a context manager:

    >>> ds = xr.Dataset({"x": np.arange(1000)})
    >>> with xr.set_options(display_width=40):
    ...     print(ds)
    <xarray.Dataset>
    Dimensions:  (x: 1000)
    Coordinates:
      * x        (x) int64 0 1 2 3 4 5 6 ...
    Data variables:
        *empty*

    Or to set global options:

    >>> xr.set_options(display_width=80)
    """

    def __init__(self, **kwargs):
        self.old = {}
        for k, v in kwargs.items():
            if k not in OPTIONS:
                raise ValueError(
                    "argument name %r is not in the set of valid options %r"
                    % (k, set(OPTIONS))
                )
            if k in _VALIDATORS and not _VALIDATORS[k](v):
                raise ValueError(f"option {k!r} given an invalid value: {v!r}")
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
