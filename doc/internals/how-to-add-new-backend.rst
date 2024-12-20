.. _add_a_backend:

How to add a new backend
------------------------

Adding a new backend for read support to Xarray does not require
one to integrate any code in Xarray; all you need to do is:

- Create a class that inherits from Xarray :py:class:`~xarray.backends.BackendEntrypoint`
  and implements the method ``open_dataset`` see :ref:`RST backend_entrypoint`

- Declare this class as an external plugin in your project configuration, see :ref:`RST
  backend_registration`

If you also want to support lazy loading and dask see :ref:`RST lazy_loading`.

Note that the new interface for backends is available from Xarray
version >= 0.18 onwards.

You can see what backends are currently available in your working environment
with :py:class:`~xarray.backends.list_engines()`.

.. _RST backend_entrypoint:

BackendEntrypoint subclassing
+++++++++++++++++++++++++++++

Your ``BackendEntrypoint`` sub-class is the primary interface with Xarray, and
it should implement the following attributes and methods:

- the ``open_dataset`` method (mandatory)
- the ``open_dataset_parameters`` attribute (optional)
- the ``guess_can_open`` method (optional)
- the ``description`` attribute (optional)
- the ``url`` attribute (optional).

This is what a ``BackendEntrypoint`` subclass should look like:

.. code-block:: python

    from xarray.backends import BackendEntrypoint


    class MyBackendEntrypoint(BackendEntrypoint):
        def open_dataset(
            self,
            filename_or_obj,
            *,
            drop_variables=None,
            # other backend specific keyword arguments
            # `chunks` and `cache` DO NOT go here, they are handled by xarray
        ):
            return my_open_dataset(filename_or_obj, drop_variables=drop_variables)

        open_dataset_parameters = ["filename_or_obj", "drop_variables"]

        def guess_can_open(self, filename_or_obj):
            try:
                _, ext = os.path.splitext(filename_or_obj)
            except TypeError:
                return False
            return ext in {".my_format", ".my_fmt"}

        description = "Use .my_format files in Xarray"

        url = "https://link_to/your_backend/documentation"

``BackendEntrypoint`` subclass methods and attributes are detailed in the following.

.. _RST open_dataset:

open_dataset
^^^^^^^^^^^^

The backend ``open_dataset`` shall implement reading from file, the variables
decoding and it shall instantiate the output Xarray class :py:class:`~xarray.Dataset`.

The following is an example of the high level processing steps:

.. code-block:: python

    def open_dataset(
        self,
        filename_or_obj,
        *,
        drop_variables=None,
        decode_times=True,
        decode_timedelta=True,
        decode_coords=True,
        my_backend_option=None,
    ):
        vars, attrs, coords = my_reader(
            filename_or_obj,
            drop_variables=drop_variables,
            my_backend_option=my_backend_option,
        )
        vars, attrs, coords = my_decode_variables(
            vars, attrs, decode_times, decode_timedelta, decode_coords
        )  #  see also conventions.decode_cf_variables

        ds = xr.Dataset(vars, attrs=attrs, coords=coords)
        ds.set_close(my_close_method)

        return ds


The output :py:class:`~xarray.Dataset` shall implement the additional custom method
``close``, used by Xarray to ensure the related files are eventually closed. This
method shall be set by using :py:meth:`~xarray.Dataset.set_close`.


The input of ``open_dataset`` method are one argument
(``filename_or_obj``) and one keyword argument (``drop_variables``):

- ``filename_or_obj``: can be any object but usually it is a string containing a path or an instance of
  :py:class:`pathlib.Path`.
- ``drop_variables``: can be ``None`` or an iterable containing the variable
  names to be dropped when reading the data.

If it makes sense for your backend, your ``open_dataset``  method
should implement in its interface the following boolean keyword arguments, called
**decoders**, which default to ``None``:

- ``mask_and_scale``
- ``decode_times``
- ``decode_timedelta``
- ``use_cftime``
- ``concat_characters``
- ``decode_coords``

Note: all the supported decoders shall be declared explicitly
in backend ``open_dataset`` signature and adding a ``**kwargs`` is not allowed.

These keyword arguments are explicitly defined in Xarray
:py:func:`~xarray.open_dataset` signature. Xarray will pass them to the
backend only if the User explicitly sets a value different from ``None``.
For more details on decoders see :ref:`RST decoders`.

Your backend can also take as input a set of backend-specific keyword
arguments. All these keyword arguments can be passed to
:py:func:`~xarray.open_dataset` grouped either via the ``backend_kwargs``
parameter or explicitly using the syntax ``**kwargs``.


If you don't want to support the lazy loading, then the
:py:class:`~xarray.Dataset` shall contain values as a :py:class:`numpy.ndarray`
and your work is almost done.

.. _RST open_dataset_parameters:

open_dataset_parameters
^^^^^^^^^^^^^^^^^^^^^^^

``open_dataset_parameters`` is the list of backend ``open_dataset`` parameters.
It is not a mandatory parameter, and if the backend does not provide it
explicitly, Xarray creates a list of them automatically by inspecting the
backend signature.

If ``open_dataset_parameters`` is not defined, but ``**kwargs`` and ``*args``
are in the backend ``open_dataset`` signature, Xarray raises an error.
On the other hand, if the backend provides the ``open_dataset_parameters``,
then ``**kwargs`` and ``*args`` can be used in the signature.
However, this practice is discouraged unless there is a good reasons for using
``**kwargs`` or ``*args``.

.. _RST guess_can_open:

guess_can_open
^^^^^^^^^^^^^^

``guess_can_open`` is used to identify the proper engine to open your data
file automatically in case the engine is not specified explicitly. If you are
not interested in supporting this feature, you can skip this step since
:py:class:`~xarray.backends.BackendEntrypoint` already provides a
default :py:meth:`~xarray.backends.BackendEntrypoint.guess_can_open`
that always returns ``False``.

Backend ``guess_can_open`` takes as input the ``filename_or_obj`` parameter of
Xarray :py:meth:`~xarray.open_dataset`, and returns a boolean.

.. _RST properties:

description and url
^^^^^^^^^^^^^^^^^^^^

``description`` is used to provide a short text description of the backend.
``url`` is used to include a link to the backend's documentation or code.

These attributes are surfaced when a user prints :py:class:`~xarray.backends.BackendEntrypoint`.
If ``description`` or ``url`` are not defined, an empty string is returned.

.. _RST decoders:

Decoders
^^^^^^^^

The decoders implement specific operations to transform data from on-disk
representation to Xarray representation.

A classic example is the “time” variable decoding operation. In NetCDF, the
elements of the “time” variable are stored as integers, and the unit contains
an origin (for example: "seconds since 1970-1-1"). In this case, Xarray
transforms the pair integer-unit in a :py:class:`numpy.datetime64`.

The standard coders implemented in Xarray are:

- :py:class:`xarray.coding.strings.CharacterArrayCoder()`
- :py:class:`xarray.coding.strings.EncodedStringCoder()`
- :py:class:`xarray.coding.variables.UnsignedIntegerCoder()`
- :py:class:`xarray.coding.variables.CFMaskCoder()`
- :py:class:`xarray.coding.variables.CFScaleOffsetCoder()`
- :py:class:`xarray.coding.times.CFTimedeltaCoder()`
- :py:class:`xarray.coding.times.CFDatetimeCoder()`

Xarray coders all have the same interface. They have two methods: ``decode``
and ``encode``. The method ``decode`` takes a ``Variable`` in on-disk
format and returns a ``Variable`` in Xarray format. Variable
attributes no more applicable after the decoding, are dropped and stored in the
``Variable.encoding`` to make them available to the ``encode`` method, which
performs the inverse transformation.

In the following an example on how to use the coders ``decode`` method:

.. ipython:: python
    :suppress:

    import xarray as xr

.. ipython:: python

    var = xr.Variable(
        dims=("x",), data=np.arange(10.0), attrs={"scale_factor": 10, "add_offset": 2}
    )
    var

    coder = xr.coding.variables.CFScaleOffsetCoder()
    decoded_var = coder.decode(var)
    decoded_var
    decoded_var.encoding

Some of the transformations can be common to more backends, so before
implementing a new decoder, be sure Xarray does not already implement that one.

The backends can reuse Xarray’s decoders, either instantiating the coders
and using the method ``decode`` directly or using the higher-level function
:py:func:`~xarray.conventions.decode_cf_variables` that groups Xarray decoders.

In some cases, the transformation to apply strongly depends on the on-disk
data format. Therefore, you may need to implement your own decoder.

An example of such a case is when you have to deal with the time format of a
grib file. grib format is very different from the NetCDF one: in grib, the
time is stored in two attributes dataDate and dataTime as strings. Therefore,
it is not possible to reuse the Xarray time decoder, and implementing a new
one is mandatory.

Decoders can be activated or deactivated using the boolean keywords of
Xarray :py:meth:`~xarray.open_dataset` signature: ``mask_and_scale``,
``decode_times``, ``decode_timedelta``, ``use_cftime``,
``concat_characters``, ``decode_coords``.
Such keywords are passed to the backend only if the User sets a value
different from ``None``.  Note that the backend does not necessarily have to
implement all the decoders, but it shall declare in its ``open_dataset``
interface only the boolean keywords related to the supported decoders.

.. _RST backend_registration:

How to register a backend
+++++++++++++++++++++++++

Define a new entrypoint in your ``pyproject.toml`` (or ``setup.cfg/setup.py`` for older
configurations), with:

- group: ``xarray.backends``
- name: the name to be passed to :py:meth:`~xarray.open_dataset`  as ``engine``
- object reference: the reference of the class that you have implemented.

You can declare the entrypoint in your project configuration like so:

.. tab:: pyproject.toml

   .. code:: toml

      [project.entry-points."xarray.backends"]
      my_engine = "my_package.my_module:MyBackendEntrypoint"

.. tab:: pyproject.toml [Poetry]

   .. code-block:: toml

       [tool.poetry.plugins."xarray.backends"]
       my_engine = "my_package.my_module:MyBackendEntrypoint"

.. tab:: setup.cfg

   .. code-block:: cfg

       [options.entry_points]
       xarray.backends =
           my_engine = my_package.my_module:MyBackendEntrypoint

.. tab:: setup.py

   .. code-block::

       setuptools.setup(
           entry_points={
               "xarray.backends": [
                   "my_engine=my_package.my_module:MyBackendEntrypoint"
               ],
           },
       )


See the `Python Packaging User Guide
<https://packaging.python.org/specifications/entry-points/#data-model>`_ for more
information on entrypoints and details of the syntax.

If you're using Poetry, note that table name in ``pyproject.toml`` is slightly different.
See `the Poetry docs <https://python-poetry.org/docs/pyproject/#plugins>`_ for more
information on plugins.

.. _RST lazy_loading:

How to support lazy loading
+++++++++++++++++++++++++++

If you want to make your backend effective with big datasets, then you should
support lazy loading.
Basically, you shall replace the :py:class:`numpy.ndarray` inside the
variables with a custom class that supports lazy loading indexing.
See the example below:

.. code-block:: python

    backend_array = MyBackendArray()
    data = indexing.LazilyIndexedArray(backend_array)
    var = xr.Variable(dims, data, attrs=attrs, encoding=encoding)

Where:

- :py:class:`~xarray.core.indexing.LazilyIndexedArray` is a class
  provided by Xarray that manages the lazy loading.
- ``MyBackendArray`` shall be implemented by the backend and shall inherit
  from :py:class:`~xarray.backends.BackendArray`.

BackendArray subclassing
^^^^^^^^^^^^^^^^^^^^^^^^

The BackendArray subclass shall implement the following method and attributes:

- the ``__getitem__`` method that takes in input an index and returns a
  `NumPy <https://numpy.org/>`__ array
- the ``shape`` attribute
- the ``dtype`` attribute.

Xarray supports different type of :doc:`/user-guide/indexing`, that can be
grouped in three types of indexes
:py:class:`~xarray.core.indexing.BasicIndexer`,
:py:class:`~xarray.core.indexing.OuterIndexer` and
:py:class:`~xarray.core.indexing.VectorizedIndexer`.
This implies that the implementation of the method ``__getitem__`` can be tricky.
In order to simplify this task, Xarray provides a helper function,
:py:func:`~xarray.core.indexing.explicit_indexing_adapter`, that transforms
all the input indexer types (basic, outer, vectorized) in a tuple
which is interpreted correctly by your backend.

This is an example ``BackendArray`` subclass implementation:

.. code-block:: python

    from xarray.backends import BackendArray


    class MyBackendArray(BackendArray):
        def __init__(
            self,
            shape,
            dtype,
            lock,
            # other backend specific keyword arguments
        ):
            self.shape = shape
            self.dtype = dtype
            self.lock = lock

        def __getitem__(
            self, key: xarray.core.indexing.ExplicitIndexer
        ) -> np.typing.ArrayLike:
            return indexing.explicit_indexing_adapter(
                key,
                self.shape,
                indexing.IndexingSupport.BASIC,
                self._raw_indexing_method,
            )

        def _raw_indexing_method(self, key: tuple) -> np.typing.ArrayLike:
            # thread safe method that access to data on disk
            with self.lock:
                ...
                return item

Note that ``BackendArray.__getitem__`` must be thread safe to support
multi-thread processing.

The :py:func:`~xarray.core.indexing.explicit_indexing_adapter` method takes in
input the ``key``, the array ``shape`` and the following parameters:

- ``indexing_support``: the type of index supported by ``raw_indexing_method``
- ``raw_indexing_method``: a method that shall take in input a key in the form
  of a tuple and return an indexed :py:class:`numpy.ndarray`.

For more details see
:py:class:`~xarray.core.indexing.IndexingSupport` and :ref:`RST indexing`.

In order to support `Dask Distributed <https://distributed.dask.org/>`__ and
:py:mod:`multiprocessing`, ``BackendArray`` subclass should be serializable
either with :ref:`io.pickle` or
`cloudpickle <https://github.com/cloudpipe/cloudpickle>`__.
That implies that all the reference to open files should be dropped. For
opening files, we therefore suggest to use the helper class provided by Xarray
:py:class:`~xarray.backends.CachingFileManager`.

.. _RST indexing:

Indexing examples
^^^^^^^^^^^^^^^^^

**BASIC**

In the ``BASIC`` indexing support, numbers and slices are supported.

Example:

.. ipython::
    :verbatim:

    In [1]: # () shall return the full array
       ...: backend_array._raw_indexing_method(())
    Out[1]: array([[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]])

    In [2]: # shall support integers
       ...: backend_array._raw_indexing_method(1, 1)
    Out[2]: 5

    In [3]: # shall support slices
       ...: backend_array._raw_indexing_method(slice(0, 3), slice(2, 4))
    Out[3]: array([[2, 3], [6, 7], [10, 11]])

**OUTER**

The ``OUTER`` indexing shall support number, slices and in addition it shall
support also lists of integers. The the outer indexing is equivalent to
combining multiple input list with ``itertools.product()``:

.. ipython::
    :verbatim:

    In [1]: backend_array._raw_indexing_method([0, 1], [0, 1, 2])
    Out[1]: array([[0, 1, 2], [4, 5, 6]])

    # shall support integers
    In [2]: backend_array._raw_indexing_method(1, 1)
    Out[2]: 5


**OUTER_1VECTOR**

The ``OUTER_1VECTOR`` indexing shall supports number, slices and at most one
list. The behaviour with the list shall be the same of ``OUTER`` indexing.

If you support more complex indexing as explicit indexing or
numpy indexing, you can have a look to the implementation of Zarr backend and Scipy backend,
currently available in :py:mod:`~xarray.backends` module.

.. _RST preferred_chunks:

Preferred chunk sizes
^^^^^^^^^^^^^^^^^^^^^

To potentially improve performance with lazy loading, the backend may define for each
variable the chunk sizes that it prefers---that is, sizes that align with how the
variable is stored. (Note that the backend is not directly involved in `Dask
<https://dask.org/>`__ chunking, because Xarray internally manages chunking.) To define
the preferred chunk sizes, store a mapping within the variable's encoding under the key
``"preferred_chunks"`` (that is, ``var.encoding["preferred_chunks"]``). The mapping's
keys shall be the names of dimensions with preferred chunk sizes, and each value shall
be the corresponding dimension's preferred chunk sizes expressed as either an integer
(such as ``{"dim1": 1000, "dim2": 2000}``) or a tuple of integers (such as ``{"dim1":
(1000, 100), "dim2": (2000, 2000, 2000)}``).

Xarray uses the preferred chunk sizes in some special cases of the ``chunks`` argument
of the :py:func:`~xarray.open_dataset` and :py:func:`~xarray.open_mfdataset` functions.
If ``chunks`` is a ``dict``, then for any dimensions missing from the keys or whose
value is ``None``, Xarray sets the chunk sizes to the preferred sizes. If ``chunks``
equals ``"auto"``, then Xarray seeks ideal chunk sizes informed by the preferred chunk
sizes. Specifically, it determines the chunk sizes using
:py:func:`dask.array.core.normalize_chunks` with the ``previous_chunks`` argument set
according to the preferred chunk sizes.
