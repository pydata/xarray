.. currentmodule:: xarray

Working with numpy-like arrays
==============================

.. warning::

   This feature should be considered experimental. Please report any bug you may find on
   xarray’s github repository.

NumPy-like arrays (:term:`duck array`) extend the :py:class:`numpy.ndarray` with
additional features, like propagating physical units or a different layout in memory.

:py:class:`DataArray` and :py:class:`Dataset` objects can wrap these duck arrays, as
long as they satisfy certain conditions (see :ref:`internals.duck_arrays`).

.. note::

    For ``dask`` support see :ref:`dask`.


Missing features
----------------
Most of the API does support :term:`duck array` objects, but there are a few areas where
the code will still cast to ``numpy`` arrays:

- dimension coordinates, and thus all indexing operations:

  * :py:meth:`Dataset.sel` and :py:meth:`DataArray.sel`
  * :py:meth:`Dataset.loc` and :py:meth:`DataArray.loc`
  * :py:meth:`Dataset.drop_sel` and :py:meth:`DataArray.drop_sel`
  * :py:meth:`Dataset.reindex`, :py:meth:`Dataset.reindex_like`,
    :py:meth:`DataArray.reindex` and :py:meth:`DataArray.reindex_like`: duck arrays in
    data variables and non-dimension coordinates won't be casted

- functions and methods that depend on external libraries or features of ``numpy`` not
  covered by ``__array_function__`` / ``__array_ufunc__``:

  * :py:meth:`Dataset.ffill` and :py:meth:`DataArray.ffill` (uses ``bottleneck``)
  * :py:meth:`Dataset.bfill` and :py:meth:`DataArray.bfill` (uses ``bottleneck``)
  * :py:meth:`Dataset.interp`, :py:meth:`Dataset.interp_like`,
    :py:meth:`DataArray.interp` and :py:meth:`DataArray.interp_like` (uses ``scipy``):
    duck arrays in data variables and non-dimension coordinates will be casted in
    addition to not supporting duck arrays in dimension coordinates
  * :py:meth:`Dataset.rolling` and :py:meth:`DataArray.rolling` (requires ``numpy>=1.20``)
  * :py:meth:`Dataset.rolling_exp` and :py:meth:`DataArray.rolling_exp` (uses
    ``numbagg``)
  * :py:meth:`Dataset.interpolate_na` and :py:meth:`DataArray.interpolate_na` (uses
    :py:class:`numpy.vectorize`)
  * :py:func:`apply_ufunc` with ``vectorize=True`` (uses :py:class:`numpy.vectorize`)

- incompatibilities between different :term:`duck array` libraries:

  * :py:meth:`Dataset.chunk` and :py:meth:`DataArray.chunk`: this fails if the data was
    not already chunked and the :term:`duck array` (e.g. a ``pint`` quantity) should
    wrap the new ``dask`` array; changing the chunk sizes works.


Extensions using duck arrays
----------------------------
Here's a list of libraries extending ``xarray`` to make working with wrapped duck arrays
easier:

- `pint-xarray <https://pint-xarray.readthedocs.io>`_
- `cupy-xarray <https://cupy-xarray.readthedocs.io>`_
