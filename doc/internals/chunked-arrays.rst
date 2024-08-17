.. currentmodule:: xarray

.. _internals.chunkedarrays:

Alternative chunked array types
===============================

.. warning::

    This is a *highly* experimental feature. Please report any bugs or other difficulties on `xarray's issue tracker <https://github.com/pydata/xarray/issues>`_.
    In particular see discussion on `xarray issue #6807 <https://github.com/pydata/xarray/issues/6807>`_

Xarray can wrap chunked dask arrays (see :ref:`dask`), but can also wrap any other chunked array type that exposes the correct interface.
This allows us to support using other frameworks for distributed and out-of-core processing, with user code still written as xarray commands.
In particular xarray also supports wrapping :py:class:`cubed.Array` objects
(see `Cubed's documentation <https://tom-e-white.com/cubed/>`_ and the `cubed-xarray package <https://github.com/xarray-contrib/cubed-xarray>`_).

The basic idea is that by wrapping an array that has an explicit notion of ``.chunks``, xarray can expose control over
the choice of chunking scheme to users via methods like :py:meth:`DataArray.chunk` whilst the wrapped array actually
implements the handling of processing all of the chunks.

Chunked array methods and "core operations"
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A chunked array needs to meet all the :ref:`requirements for normal duck arrays <internals.duckarrays.requirements>`, but must also
implement additional features.

Chunked arrays have additional attributes and methods, such as ``.chunks`` and ``.rechunk``.
Furthermore, Xarray dispatches chunk-aware computations across one or more chunked arrays using special functions known
as "core operations". Examples include ``map_blocks``, ``blockwise``, and ``apply_gufunc``.

The core operations are generalizations of functions first implemented in :py:mod:`dask.array`.
The implementation of these functions is specific to the type of arrays passed to them. For example, when applying the
``map_blocks`` core operation, :py:class:`dask.array.Array` objects must be processed by :py:func:`dask.array.map_blocks`,
whereas :py:class:`cubed.Array` objects must be processed by :py:func:`cubed.map_blocks`.

In order to use the correct implementation of a core operation for the array type encountered, xarray dispatches to the
corresponding subclass of :py:class:`~xarray.namedarray.parallelcompat.ChunkManagerEntrypoint`,
also known as a "Chunk Manager". Therefore **a full list of the operations that need to be defined is set by the
API of the** :py:class:`~xarray.namedarray.parallelcompat.ChunkManagerEntrypoint` **abstract base class**. Note that chunked array
methods are also currently dispatched using this class.

Chunked array creation is also handled by this class. As chunked array objects have a one-to-one correspondence with
in-memory numpy arrays, it should be possible to create a chunked array from a numpy array by passing the desired
chunking pattern to an implementation of :py:class:`~xarray.namedarray.parallelcompat.ChunkManagerEntrypoint.from_array``.

.. note::

    The :py:class:`~xarray.namedarray.parallelcompat.ChunkManagerEntrypoint` abstract base class is mostly just acting as a
    namespace for containing the chunked-aware function primitives. Ideally in the future we would have an API standard
    for chunked array types which codified this structure, making the entrypoint system unnecessary.

.. currentmodule:: xarray.namedarray.parallelcompat

.. autoclass:: xarray.namedarray.parallelcompat.ChunkManagerEntrypoint
   :members:

Registering a new ChunkManagerEntrypoint subclass
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Rather than hard-coding various chunk managers to deal with specific chunked array implementations, xarray uses an
entrypoint system to allow developers of new chunked array implementations to register their corresponding subclass of
:py:class:`~xarray.namedarray.parallelcompat.ChunkManagerEntrypoint`.


To register a new entrypoint you need to add an entry to the ``setup.cfg`` like this::

    [options.entry_points]
    xarray.chunkmanagers =
        dask = xarray.namedarray.daskmanager:DaskManager

See also `cubed-xarray <https://github.com/xarray-contrib/cubed-xarray>`_ for another example.

To check that the entrypoint has worked correctly, you may find it useful to display the available chunkmanagers using
the internal function :py:func:`~xarray.namedarray.parallelcompat.list_chunkmanagers`.

.. autofunction:: list_chunkmanagers


User interface
~~~~~~~~~~~~~~

Once the chunkmanager subclass has been registered, xarray objects wrapping the desired array type can be created in 3 ways:

#. By manually passing the array type to the :py:class:`~xarray.DataArray` constructor, see the examples for :ref:`numpy-like arrays <userguide.duckarrays>`,

#. Calling :py:meth:`~xarray.DataArray.chunk`, passing the keyword arguments ``chunked_array_type`` and ``from_array_kwargs``,

#. Calling :py:func:`~xarray.open_dataset`, passing the keyword arguments ``chunked_array_type`` and ``from_array_kwargs``.

The latter two methods ultimately call the chunkmanager's implementation of ``.from_array``, to which they pass the ``from_array_kwargs`` dict.
The ``chunked_array_type`` kwarg selects which registered chunkmanager subclass to dispatch to. It defaults to ``'dask'``
if Dask is installed, otherwise it defaults to whichever chunkmanager is registered if only one is registered.
If multiple chunkmanagers are registered, the ``chunk_manager`` configuration option (which can be set using :py:func:`set_options`)
will be used to determine which chunkmanager to use, defaulting to ``'dask'``.

Parallel processing without chunks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To use a parallel array type that does not expose a concept of chunks explicitly, none of the information on this page
is theoretically required. Such an array type (e.g. `Ramba <https://github.com/Python-for-HPC/ramba>`_ or
`Arkouda <https://github.com/Bears-R-Us/arkouda>`_) could be wrapped using xarray's existing support for
:ref:`numpy-like "duck" arrays <userguide.duckarrays>`.
