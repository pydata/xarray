from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from .pycompat import basestring

# Check which of the backend I/O engine packages are installed
_AVAILABLE_IO_ENGINES = []
try:
    import netCDF4  # noqa
    _AVAILABLE_IO_ENGINES.append('netcdf4')
except ImportError:
    pass
try:
    import pydap  # noqa
    _AVAILABLE_IO_ENGINES.append('pydap')
except ImportError:
    pass
try:
    import scipy.io.netcdf  # noqa
    _AVAILABLE_IO_ENGINES.append('scipy')
except ImportError:
    pass
try:
    import h5netcdf.legacyapi  # noqa
    _AVAILABLE_IO_ENGINES.append('h5netcdf')
except ImportError:
    pass

OPTIONS = {
    'display_width': 80,
    'arithmetic_join': 'inner',
    'io_engines': _AVAILABLE_IO_ENGINES,
}


class set_options(object):
    """Set options for xarray in a controlled context.

    Currently supported options:

    - ``display_width``: maximum display width for ``repr`` on xarray objects.
      Default: ``80``.
    - ``arithmetic_join``: DataArray/Dataset alignment in binary operations.
      Default: ``'inner'``.
    - ``io_engines``: List of backend data I/O engines to try in order when
      saving/loading data without an explicit engine. Default: All the
      installed engines on start up from this list: ['netcdf4', 'pydap',
      'scipy', 'h5netcdf'].

    You can use ``set_options`` either as a context manager:

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
        invalid_options = {k for k in kwargs if k not in OPTIONS}
        if invalid_options:
            raise ValueError('argument names %r are not in the set of valid '
                             'options %r' % (invalid_options, set(OPTIONS)))
        self.old = OPTIONS.copy()

        if 'io_engines' in kwargs:
            if isinstance(kwargs['io_engines'], basestring):
                kwargs['io_engines'] = [kwargs['io_engines']]
            new_engines = list()
            for e in kwargs['io_engines']:
                if e in _AVAILABLE_IO_ENGINES:
                    new_engines.append(e)
                else:
                    raise ValueError('I/O engine %s not installed' % e)
            if not new_engines:
                raise ValueError('No I/O engines')
            else:
                kwargs['io_engines'] = new_engines

        OPTIONS.update(kwargs)

    def __enter__(self):
        return

    def __exit__(self, type, value, traceback):
        OPTIONS.clear()
        OPTIONS.update(self.old)
