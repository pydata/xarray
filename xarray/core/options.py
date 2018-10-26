from __future__ import absolute_import, division, print_function

DISPLAY_WIDTH = 'display_width'
ARITHMETIC_JOIN = 'arithmetic_join'
ENABLE_CFTIMEINDEX = 'enable_cftimeindex'
FILE_CACHE_MAXSIZE = 'file_cache_maxsize'
CMAP_SEQUENTIAL = 'cmap_sequential'
CMAP_DIVERGENT = 'cmap_divergent'

OPTIONS = {
    DISPLAY_WIDTH: 80,
    ARITHMETIC_JOIN: 'inner',
    ENABLE_CFTIMEINDEX: True,
    FILE_CACHE_MAXSIZE: 128,
    CMAP_SEQUENTIAL: 'viridis',
    CMAP_DIVERGENT: 'RdBu_r',
}

_JOIN_OPTIONS = frozenset(['inner', 'outer', 'left', 'right', 'exact'])


def _positive_integer(value):
    return isinstance(value, int) and value > 0


_VALIDATORS = {
    DISPLAY_WIDTH: _positive_integer,
    ARITHMETIC_JOIN: _JOIN_OPTIONS.__contains__,
    ENABLE_CFTIMEINDEX: lambda value: isinstance(value, bool),
    FILE_CACHE_MAXSIZE: _positive_integer,
}


def _set_file_cache_maxsize(value):
    from ..backends.file_manager import FILE_CACHE
    FILE_CACHE.maxsize = value


_SETTERS = {
    FILE_CACHE_MAXSIZE: _set_file_cache_maxsize,
}


class set_options(object):
    """Set options for xarray in a controlled context.

    Currently supported options:

    - ``display_width``: maximum display width for ``repr`` on xarray objects.
      Default: ``80``.
    - ``arithmetic_join``: DataArray/Dataset alignment in binary operations.
      Default: ``'inner'``.
    - ``enable_cftimeindex``: flag to enable using a ``CFTimeIndex``
      for time indexes with non-standard calendars or dates outside the
      Timestamp-valid range. Default: ``True``.
    - ``file_cache_maxsize``: maximum number of open files to hold in xarray's
      global least-recently-usage cached. This should be smaller than your
      system's per-process file descriptor limit, e.g., ``ulimit -n`` on Linux.
      Default: 128.
    - ``cmap_sequential``: colormap to use for nondivergent data plots.
      Default: ``viridis``. If string, must be matplotlib built-in colormap.
      Can also be a Colormap object (e.g. mpl.cm.magma)
    - ``cmap_divergent``: colormap to use for divergent data plots.
      Default: ``RdBu_r``. If string, must be matplotlib built-in colormap.
      Can also be a Colormap object (e.g. mpl.cm.magma)

f    You can use ``set_options`` either as a context manager:

    >>> ds = xr.Dataset({'x': np.arange(1000)})
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
        self.old = OPTIONS.copy()
        for k, v in kwargs.items():
            if k not in OPTIONS:
                raise ValueError(
                    'argument name %r is not in the set of valid options %r'
                    % (k, set(OPTIONS)))
            if k in _VALIDATORS and not _VALIDATORS[k](v):
                raise ValueError(
                    'option %r given an invalid value: %r' % (k, v))
        self._apply_update(kwargs)

    def _apply_update(self, options_dict):
        for k, v in options_dict.items():
            if k in _SETTERS:
                _SETTERS[k](v)
        OPTIONS.update(options_dict)

    def __enter__(self):
        return

    def __exit__(self, type, value, traceback):
        OPTIONS.clear()
        self._apply_update(self.old)
