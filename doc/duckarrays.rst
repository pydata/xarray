.. currentmodule:: xarray

Duck arrays
===========
This is a high-level overview, for the technical details of integrating duck arrays with
``xarray``, see :ref:`internals.duck_arrays`.

Missing features
----------------
Most of the API does support duck arrays, but there are a areas where the code
will still cast to ``numpy`` arrays:

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
  * :py:meth:`Dataset.rolling_exp` and :py:meth:`DataArray.rolling_exp` (uses
    ``numbagg``)
  * :py:meth:`Dataset.rolling` and :py:meth:`DataArray.rolling` (uses internal functions
    of ``numpy``)
  * :py:meth:`Dataset.interpolate_na` and :py:meth:`DataArray.interpolate_na` (uses
    :py:func:`numpy.vectorize`)