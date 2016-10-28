OPTIONS = {'display_width': 80}


class set_options(object):
    """Set options for xarray in a controlled context.

    Currently, the only supported option is ``display_width``, which has a
    default value of 80.

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
