
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

A chunked array Needs to meet all the :ref:`requirements for normal duck arrays <internals.duckarrays>`, but should also

- ``.chunk``
- ``.rechunk``
- ``.compute``


Chunked operations as function primitives
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Actual full list is defined in the :py:class:``~xarray.core.parallelcompat.ChunkManagerEntryPoint`` class (link to that API documentation)


ChunkManagerEntrypoint subclassing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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

Don't necessarily need all this
