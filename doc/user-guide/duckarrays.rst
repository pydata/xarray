.. currentmodule:: xarray

.. _userguide.duckarrays:

Working with numpy-like arrays
==============================

NumPy-like arrays (often known as :term:`duck array`\s) are drop-in replacements for the :py:class:`numpy.ndarray`
class but with different features, such as propagating physical units or a different layout in memory.
Xarray can often wrap these array types, allowing you to use labelled dimensions and indexes whilst benefiting from the
additional features of these array libraries.

Some numpy-like array types that xarray already has some support for:

* `Cupy <https://cupy.dev/>`_ - GPU support (see `cupy-xarray <https://cupy-xarray.readthedocs.io>`_),
* `Sparse <https://sparse.pydata.org/en/stable/>`_ - for performant arrays with many zero elements,
* `Pint <https://pint.readthedocs.io/en/latest/>`_ - for tracking the physical units of your data (see `pint-xarray <https://pint-xarray.readthedocs.io>`_),
* `Dask <https://docs.dask.org/en/stable/>`_ - parallel computing on larger-than-memory arrays (see :ref:`using dask with xarray <dask>`),
* `Cubed <https://github.com/tomwhite/cubed/tree/main/cubed>`_ - another parallel computing framework that emphasises reliability (see `cubed-xarray <https://github.com/cubed-xarray>`_).

.. warning::

   This feature should be considered somewhat experimental. Please report any bugs you find on
   `xarray’s issue tracker <https://github.com/pydata/xarray/issues>`_.

.. note::

    For information on wrapping dask arrays see :ref:`dask`. Whilst xarray wraps dask arrays in a similar way to that
    described on this page, chunked array types like :py:class:`dask.array.Array` implement additional methods that require
    slightly different user code (e.g. calling ``.chunk`` or ``.compute``). See the docs on :ref:`wrapping chunked arrays <internals.chunkedarrays>`.

Why "duck"?
-----------

Why is it also called a "duck" array? This comes from a common statement of object-oriented programming -
"If it walks like a duck, and quacks like a duck, treat it like a duck". In other words, a library like xarray that
is capable of using multiple different types of arrays does not have to explicitly check that each one it encounters is
permitted (e.g. ``if dask``, ``if numpy``, ``if sparse`` etc.). Instead xarray can take the more permissive approach of simply
treating the wrapped array as valid, attempting to call the relevant methods (e.g. ``.mean()``) and only raising an
error if a problem occurs (e.g. the method is not found on the wrapped class). This is much more flexible, and allows
objects and classes from different libraries to work together more easily.

What is a numpy-like array?
---------------------------

A "numpy-like array" (also known as a "duck array") is a class that contains array-like data, and implements key
numpy-like functionality such as indexing, broadcasting, and computation methods.

For example, the `sparse <https://sparse.pydata.org/en/stable/>`_ library provides a sparse array type which is useful for representing nD array objects like sparse matrices
in a memory-efficient manner. We can create a sparse array object (of the :py:class:`sparse.COO` type) from a numpy array like this:

.. ipython:: python

    from sparse import COO

    x = np.eye(4, dtype=np.uint8)  # create diagonal identity matrix
    s = COO.from_numpy(x)
    s

This sparse object does not attempt to explicitly store every element in the array, only the non-zero elements.
This approach is much more efficient for large arrays with only a few non-zero elements (such as tri-diagonal matrices).
Sparse array objects can be converted back to a "dense" numpy array by calling :py:meth:`sparse.COO.todense`.

Just like :py:class:`numpy.ndarray` objects, :py:class:`sparse.COO` arrays support indexing

.. ipython:: python

    s[1, 1]  # diagonal elements should be ones
    s[2, 3]  # off-diagonal elements should be zero

broadcasting,

.. ipython:: python

    x2 = np.zeros(
        (4, 1), dtype=np.uint8
    )  # create second sparse array of different shape
    s2 = COO.from_numpy(x2)
    (s * s2)  # multiplication requires broadcasting

and various computation methods

.. ipython:: python

    s.sum(axis=1)

This numpy-like array also supports calling so-called `numpy ufuncs <https://numpy.org/doc/stable/reference/ufuncs.html#available-ufuncs>`_
("universal functions") on it directly:

.. ipython:: python

    np.sum(s, axis=1)


Notice that in each case the API for calling the operation on the sparse array is identical to that of calling it on the
equivalent numpy array - this is the sense in which the sparse array is "numpy-like".

.. note::

    For discussion on exactly which methods a class needs to implement to be considered "numpy-like", see :ref:`internals.duckarrays`.

Wrapping numpy-like arrays in xarray
------------------------------------

:py:class:`DataArray`, :py:class:`Dataset`, and :py:class:`Variable` objects can wrap these numpy-like arrays.

Constructing xarray objects which wrap numpy-like arrays
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The primary way to create an xarray object which wraps a numpy-like array is to pass that numpy-like array instance directly
to the constructor of the xarray class. The :ref:`page on xarray data structures <data structures>` shows how :py:class:`DataArray` and :py:class:`Dataset`
both accept data in various forms through their ``data`` argument, but in fact this data can also be any wrappable numpy-like array.

For example, we can wrap the sparse array we created earlier inside a new DataArray object:

.. ipython:: python

    s_da = xr.DataArray(s, dims=["i", "j"])
    s_da

We can see what's inside - the printable representation of our xarray object (the repr) automatically uses the printable
representation of the underlying wrapped array.

Of course our sparse array object is still there underneath - it's stored under the ``.data`` attribute of the dataarray:

.. ipython:: python

    s_da.data

Array methods
~~~~~~~~~~~~~

We saw above that numpy-like arrays provide numpy methods. Xarray automatically uses these when you call the corresponding xarray method:

.. ipython:: python

    s_da.sum(dim="j")

Converting wrapped types
~~~~~~~~~~~~~~~~~~~~~~~~

If you want to change the type inside your xarray object you can use :py:meth:`DataArray.as_numpy`:

.. ipython:: python

    s_da.as_numpy()

This returns a new :py:class:`DataArray` object, but now wrapping a normal numpy array.

If instead you want to convert to numpy and return that numpy array you can use either :py:meth:`DataArray.to_numpy` or
:py:meth:`DataArray.values`, where the former is strongly preferred. The difference is in the way they coerce to numpy - :py:meth:`~DataArray.values`
always uses :py:func:`numpy.asarray` which will fail for some array types (e.g. ``cupy``), whereas :py:meth:`~DataArray.to_numpy`
uses the correct method depending on the array type.

.. ipython:: python

    s_da.to_numpy()

.. ipython:: python
    :okexcept:

    s_da.values

This illustrates the difference between :py:meth:`~DataArray.data` and :py:meth:`~DataArray.values`,
which is sometimes a point of confusion for new xarray users.
Explicitly: :py:meth:`DataArray.data` returns the underlying numpy-like array, regardless of type, whereas
:py:meth:`DataArray.values` converts the underlying array to a numpy array before returning it.
(This is another reason to use :py:meth:`~DataArray.to_numpy` over :py:meth:`~DataArray.values` - the intention is clearer.)

Conversion to numpy as a fallback
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If a wrapped array does not implement the corresponding array method then xarray will often attempt to convert the
underlying array to a numpy array so that the operation can be performed. You may want to watch out for this behavior,
and report any instances in which it causes problems.

Most of xarray's API does support using :term:`duck array` objects, but there are a few areas where
the code will still convert to ``numpy`` arrays:

- Dimension coordinates, and thus all indexing operations:

  * :py:meth:`Dataset.sel` and :py:meth:`DataArray.sel`
  * :py:meth:`Dataset.loc` and :py:meth:`DataArray.loc`
  * :py:meth:`Dataset.drop_sel` and :py:meth:`DataArray.drop_sel`
  * :py:meth:`Dataset.reindex`, :py:meth:`Dataset.reindex_like`,
    :py:meth:`DataArray.reindex` and :py:meth:`DataArray.reindex_like`: duck arrays in
    data variables and non-dimension coordinates won't be casted

- Functions and methods that depend on external libraries or features of ``numpy`` not
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

- Incompatibilities between different :term:`duck array` libraries:

  * :py:meth:`Dataset.chunk` and :py:meth:`DataArray.chunk`: this fails if the data was
    not already chunked and the :term:`duck array` (e.g. a ``pint`` quantity) should
    wrap the new ``dask`` array; changing the chunk sizes works however.

Extensions using duck arrays
----------------------------

Whilst the features above allow many numpy-like array libraries to be used pretty seamlessly with xarray, it often also
makes sense to use an interfacing package to make certain tasks easier.

For example the `pint-xarray package <https://pint-xarray.readthedocs.io>`_ offers a custom ``.pint`` accessor (see :ref:`internals.accessors`) which provides
convenient access to information stored within the wrapped array (e.g. ``.units`` and ``.magnitude``), and makes
creating wrapped pint arrays (and especially xarray-wrapping-pint-wrapping-dask arrays) simpler for the user.

We maintain a list of libraries extending ``xarray`` to make working with particular wrapped duck arrays
easier. If you know of more that aren't on this list please raise an issue to add them!

- `pint-xarray <https://pint-xarray.readthedocs.io>`_
- `cupy-xarray <https://cupy-xarray.readthedocs.io>`_
- `cubed-xarray <https://github.com/cubed-xarray>`_
