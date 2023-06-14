
.. _internals.duckarrays:

Integrating with duck arrays
=============================

.. warning::

    This is an experimental feature. Please report any bugs or other difficulties on `xarray's issue tracker <https://github.com/pydata/xarray/issues>`_.

Xarray can wrap custom numpy-like arrays (":term:`duck array`\s") - see the :ref:`user guide documentation <userguide.duckarrays>`.
This page is intended for developers who are interested in wrapping a custom array type with xarray.

Duck array requirements
~~~~~~~~~~~~~~~~~~~~~~~

Xarray does not explicitly check that that required methods are defined by the underlying duck array object before
attempting to wrap the given array. However, a wrapped array type should at a minimum support numpy's ``shape``,
``dtype`` and ``ndim`` properties, as well as the ``__array__``, ``__array_ufunc__`` and ``__array_function__`` methods.
The array ``shape`` property needs to obey numpy's broadcasting rules.

Python Array API standard support
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

As an integration library xarray benefits greatly from the standardization of duck-array libraries' APIs, and so is a
big supporter of the python Array API Standard (link). In fact the crystallization of different array libraries' APIs towards
the standard has already helped xarray remove a lot of internal adapter code.

We aim to support any array libraries that follow the standard out-of-the-box. However, xarray does occasionally
call some numpy functions which are not (yet) part of the standard (e.g. :py:class:`DataArray.pad` calls `np.pad`,
). (link to issue)

Custom inline reprs
~~~~~~~~~~~~~~~~~~~

In certain situations (e.g. when printing the collapsed preview of
variables of a ``Dataset``), xarray will display the repr of a :term:`duck array`
in a single line, truncating it to a certain number of characters. If that
would drop too much information, the :term:`duck array` may define a
``_repr_inline_`` method that takes ``max_width`` (number of characters) as an
argument

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
