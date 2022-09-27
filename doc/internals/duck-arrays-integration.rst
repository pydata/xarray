
.. _internals.duck_arrays:

Integrating with duck arrays
=============================

.. warning::

    This is a experimental feature.

Xarray can wrap custom :term:`duck array` objects as long as they define numpy's
``shape``, ``dtype`` and ``ndim`` properties and the ``__array__``,
``__array_ufunc__`` and ``__array_function__`` methods.

In certain situations (e.g. when printing the collapsed preview of
variables of a ``Dataset``), xarray will display the repr of a :term:`duck array`
in a single line, truncating it to a certain number of characters. If that
would drop too much information, the :term:`duck array` may define a
``_repr_inline_`` method that takes ``max_width`` (number of characters) as an
argument:

.. code:: python

    class MyDuckArray:
        ...

        def _repr_inline_(self, max_width):
            """format to a single line with at most max_width characters"""
            ...

        ...

To avoid duplicated information, this method must omit information about the shape and
:term:`dtype`. For example, the string representation of a ``dask`` array or a
``sparse`` matrix would be:

.. ipython:: python

    import dask.array as da
    import xarray as xr
    import sparse

    a = da.linspace(0, 1, 20, chunks=2)
    a

    b = np.eye(10)
    b[[5, 7, 3, 0], [6, 8, 2, 9]] = 2
    b = sparse.COO.from_numpy(b)
    b

    xr.Dataset(dict(a=("x", a), b=(("y", "z"), b)))
