
.. _internals.chunkedarrays:

Alternative chunked array types
===============================

.. warning::

    This is a *highly* experimental feature. Please report any bugs or other difficulties on `xarray's issue tracker <https://github.com/pydata/xarray/issues>`_.
    In particular see discussion on `xarray issue #6807 <https://github.com/pydata/xarray/issues/6807>`_

Xarray can wrap chunked dask arrays (see :ref:`dask`), but can also wrap any other chunked array type that exposes the correct interface.
This allows us to support using other frameworks for distributed and out-of-core processing, with user code still written as xarray commands.

The basic idea is that by wrapping an array that has an explicit notion of ``chunks``, xarray can expose control over
the choice of chunking scheme to users via methods like :py:meth:`DataArray.chunk` whilst the wrapped array actually
implements the handling of processing all of the chunks.


Chunked array requirements
~~~~~~~~~~~~~~~~~~~~~~~~~~

A chunked array needs to meet all the :ref:`requirements for normal duck arrays <internals.duckarrays>`, but should also implement these methods:

- ``.chunk``
- ``.rechunk``
- ``.compute``

Chunked operations as function primitives
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Xarray dispatches chunk-aware computations across arrays using function "primitives" that accept one or more arrays.
Examples include ``map_blocks``, ``blockwise``, and ``apply_gufunc``.
These primitives are generalizations of functions first implemented in :py:class:`dask.array`.
The implementation of these functions is specific to the type of arrays passed to them: :py:class:`dask.array.Array` objects
must be processed by :py:func:`dask.array.map_blocks`, whereas :py:class:`cubed.Array` objects must be processed by :py:func:`cubed.map_blocks`.

In order to use the correct function primitive for the array type encountered, xarray dispatches to the corresponding subclass of :py:class:``~xarray.core.parallelcompat.ChunkManagerEntryPoint``,
also known as a "Chunk Manager". Therefore a full list of the primitive functions that need to be defined is set by the API of the
:py:class:``~xarray.core.parallelcompat.ChunkManagerEntrypoint`` abstract base class.

:: note:

    The :py:class:``~xarray.core.parallelcompat.ChunkManagerEntrypoint`` abstract base class is mostly just acting as a
    namespace for containing the chunked-aware function primitives. Ideally in the future we would have an API standard
    for chunked array types which codified this structure, making the entrypoint system unnecessary.

ChunkManagerEntrypoint subclassing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Rather than hard-coding various chunk managers to deal with specific chunked array implementations, xarray uses an
entrypoint system to allow developers of new chunked array implementations to register their corresponding subclass of
:py:class:``~xarray.core.parallelcompat.ChunkManagerEntrypoint``.

.. autosummary::
   :toctree: generated/

   xarray.core.parallelcompat.list_chunkmanagers
   xarray.core.parallelcompat.ChunkManagerEntrypoint


User interface
~~~~~~~~~~~~~~

``chunked_array_type`` kwarg
``from_array_kwargs`` dict


Parallel processing without chunks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To use a parallel array type that does not expose a concept of chunks explicitly, none of the information on this page
is theoretically required. Such an array type could be wrapped using xarray's existing
support for `numpy-like "duck" arrays <userguide.duckarrays>`.
