from __future__ import absolute_import, division, print_function

OPTIONS = {
    'display_width': 80,
    'arithmetic_join': 'inner',
    'enable_cftimeindex': False
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
      Timestamp-valid range. Default: ``False``.

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
        OPTIONS.update(kwargs)

    def __enter__(self):
        return

    def __exit__(self, type, value, traceback):
        OPTIONS.clear()
        OPTIONS.update(self.old)
