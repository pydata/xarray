.. _internals:

xarray Internals
================

.. currentmodule:: xarray

xarray builds upon two of the foundational libraries of the scientific Python
stack, NumPy and pandas. It is written in pure Python (no C or Cython
extensions), which makes it easy to develop and extend. Instead, we push
compiled code to :ref:`optional dependencies<installing>`.

.. ipython:: python
    :suppress:

    import dask.array as da
    import numpy as np
    import pandas as pd
    import sparse
    import xarray as xr

    np.random.seed(123456)

Variable objects
----------------

The core internal data structure in xarray is the :py:class:`~xarray.Variable`,
which is used as the basic building block behind xarray's
:py:class:`~xarray.Dataset` and :py:class:`~xarray.DataArray` types. A
``Variable`` consists of:

- ``dims``: A tuple of dimension names.
- ``data``: The N-dimensional array (typically, a NumPy or Dask array) storing
  the Variable's data. It must have the same number of dimensions as the length
  of ``dims``.
- ``attrs``: An ordered dictionary of metadata associated with this array. By
  convention, xarray's built-in operations never use this metadata.
- ``encoding``: Another ordered dictionary used to store information about how
  these variable's data is represented on disk. See :ref:`io.encoding` for more
  details.

``Variable`` has an interface similar to NumPy arrays, but extended to make use
of named dimensions. For example, it uses ``dim`` in preference to an ``axis``
argument for methods like ``mean``, and supports :ref:`compute.broadcasting`.

However, unlike ``Dataset`` and ``DataArray``, the basic ``Variable`` does not
include coordinate labels along each axis.

``Variable`` is public API, but because of its incomplete support for labeled
data, it is mostly intended for advanced uses, such as in xarray itself or for
writing new backends. You can access the variable objects that correspond to
xarray objects via the (readonly) :py:attr:`Dataset.variables
<xarray.Dataset.variables>` and
:py:attr:`DataArray.variable <xarray.DataArray.variable>` attributes.


.. _internals.duck_arrays:

Integrating with duck arrays
----------------------------

.. warning::

    This is a experimental feature.

xarray can wrap custom :term:`duck array` objects as long as they define numpy's
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
            """ format to a single line with at most max_width characters """
            ...

        ...

To avoid duplicated information, this method must omit information about the shape and
:term:`dtype`. For example, the string representation of a ``dask`` array or a
``sparse`` matrix would be:

.. ipython:: python

    a = da.linspace(0, 1, 20, chunks=2)
    a

    b = np.eye(10)
    b[[5, 7, 3, 0], [6, 8, 2, 9]] = 2
    b = sparse.COO.from_numpy(b)
    b

    xr.Dataset({"a": ("x", a), "b": (("y", "z"), b)})

Extending xarray
----------------

xarray is designed as a general purpose library, and hence tries to avoid
including overly domain specific functionality. But inevitably, the need for more
domain specific logic arises.

One standard solution to this problem is to subclass Dataset and/or DataArray to
add domain specific functionality. However, inheritance is not very robust. It's
easy to inadvertently use internal APIs when subclassing, which means that your
code may break when xarray upgrades. Furthermore, many builtin methods will
only return native xarray objects.

The standard advice is to use `composition over inheritance`__, but
reimplementing an API as large as xarray's on your own objects can be an onerous
task, even if most methods are only forwarding to xarray implementations.

__ https://github.com/pydata/xarray/issues/706

If you simply want the ability to call a function with the syntax of a
method call, then the builtin :py:meth:`~xarray.DataArray.pipe` method (copied
from pandas) may suffice.

To resolve this issue for more complex cases, xarray has the
:py:func:`~xarray.register_dataset_accessor` and
:py:func:`~xarray.register_dataarray_accessor` decorators for adding custom
"accessors" on xarray objects. Here's how you might use these decorators to
write a custom "geo" accessor implementing a geography specific extension to
xarray:

.. literalinclude:: examples/_code/accessor_example.py

In general, the only restriction on the accessor class is that the ``__init__`` method
must have a single parameter: the ``Dataset`` or ``DataArray`` object it is supposed
to work on.

This achieves the same result as if the ``Dataset`` class had a cached property
defined that returns an instance of your class:

.. code-block:: python

    class Dataset:
        ...

        @property
        def geo(self):
            return GeoAccessor(self)

However, using the register accessor decorators is preferable to simply adding
your own ad-hoc property (i.e., ``Dataset.geo = property(...)``), for several
reasons:

1. It ensures that the name of your property does not accidentally conflict with
   any other attributes or methods (including other accessors).
2. Instances of accessor object will be cached on the xarray object that creates
   them. This means you can save state on them (e.g., to cache computed
   properties).
3. Using an accessor provides an implicit namespace for your custom
   functionality that clearly identifies it as separate from built-in xarray
   methods.

.. note::

   Accessors are created once per DataArray and Dataset instance. New
   instances, like those created from arithmetic operations or when accessing
   a DataArray from a Dataset (ex. ``ds[var_name]``), will have new
   accessors created.

Back in an interactive IPython session, we can use these properties:

.. ipython:: python
    :suppress:

    exec(open("examples/_code/accessor_example.py").read())

.. ipython:: python

    ds = xr.Dataset({"longitude": np.linspace(0, 10), "latitude": np.linspace(0, 20)})
    ds.geo.center
    ds.geo.plot()

The intent here is that libraries that extend xarray could add such an accessor
to implement subclass specific functionality rather than using actual subclasses
or patching in a large number of domain specific methods. For further reading
on ways to write new accessors and the philosophy behind the approach, see
:issue:`1080`.

To help users keep things straight, please `let us know
<https://github.com/pydata/xarray/issues>`_ if you plan to write a new accessor
for an open source library. In the future, we will maintain a list of accessors
and the libraries that implement them on this page.

To make documenting accessors with ``sphinx`` and ``sphinx.ext.autosummary``
easier, you can use `sphinx-autosummary-accessors`_.

.. _sphinx-autosummary-accessors: https://sphinx-autosummary-accessors.readthedocs.io/

.. _zarr_encoding:

Zarr Encoding Specification
---------------------------

In implementing support for the `Zarr <https://zarr.readthedocs.io/>`_ storage
format, Xarray developers made some *ad hoc* choices about how to store
NetCDF data in Zarr.
Future versions of the Zarr spec will likely include a more formal convention
for the storage of the NetCDF data model in Zarr; see
`Zarr spec repo <https://github.com/zarr-developers/zarr-specs>`_ for ongoing
discussion.

First, Xarray can only read and write Zarr groups. There is currently no support
for reading / writting individual Zarr arrays. Zarr groups are mapped to
Xarray ``Dataset`` objects.

Second, from Xarray's point of view, the key difference between
NetCDF and Zarr is that all NetCDF arrays have *dimension names* while Zarr
arrays do not. Therefore, in order to store NetCDF data in Zarr, Xarray must
somehow encode and decode the name of each array's dimensions.

To accomplish this, Xarray developers decided to define a special Zarr array
attribute: ``_ARRAY_DIMENSIONS``. The value of this attribute is a list of
dimension names (strings), for example ``["time", "lon", "lat"]``. When writing
data to Zarr, Xarray sets this attribute on all variables based on the variable
dimensions. When reading a Zarr group, Xarray looks for this attribute on all
arrays, raising an error if it can't be found. The attribute is used to define
the variable dimension names and then removed from the attributes dictionary
returned to the user.

Because of these choices, Xarray cannot read arbitrary array data, but only
Zarr data with valid ``_ARRAY_DIMENSIONS`` attributes on each array.

After decoding the ``_ARRAY_DIMENSIONS`` attribute and assigning the variable
dimensions, Xarray proceeds to [optionally] decode each variable using its
standard CF decoding machinery used for NetCDF data (see :py:func:`decode_cf`).

As a concrete example, here we write a tutorial dataset to Zarr and then
re-open it directly with Zarr:

.. ipython:: python

    ds = xr.tutorial.load_dataset("rasm")
    ds.to_zarr("rasm.zarr", mode="w")
    import zarr

    zgroup = zarr.open("rasm.zarr")
    print(zgroup.tree())
    dict(zgroup["Tair"].attrs)

.. _add_a_backend:

How to add a new backend
------------------------

Adding a new backend for read support to Xarray does not require
to integrate any code in Xarray; all you need to do is:

- Create a class that inherits from Xarray :py:class:`~xarray.backends.common.BackendEntrypoint`
  and implements the method ``open_dataset`` see :ref:`RST backend_entrypoint`

- Declare this class as an external plugin in your ``setup.py``, see :ref:`RST backend_registration`

If you also want to support lazy loading and dask see :ref:`RST lazy_loading`.

Note that the new interface for backends is available from Xarray
version >= 0.18 onwards.

.. _RST backend_entrypoint:

BackendEntrypoint subclassing
+++++++++++++++++++++++++++++

Your ``BackendEntrypoint`` sub-class is the primary interface with Xarray, and
it should implement the following attributes and methods:

- the ``open_dataset`` method (mandatory)
- the ``open_dataset_parameters`` attribute (optional)
- the ``guess_can_open`` method (optional).

This is what a ``BackendEntrypoint`` subclass should look like:

.. code-block:: python

    class MyBackendEntrypoint(BackendEntrypoint):
        def open_dataset(
            self,
            filename_or_obj,
            *,
            drop_variables=None,
            # other backend specific keyword arguments
        ):
            ...
            return ds

        open_dataset_parameters = ["filename_or_obj", "drop_variables"]

        def guess_can_open(self, filename_or_obj):
            try:
                _, ext = os.path.splitext(filename_or_obj)
            except TypeError:
                return False
            return ext in {...}

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
        my_backend_param=None,
    ):
        vars, attrs, coords = my_reader(
            filename_or_obj,
            drop_variables=drop_variables,
            my_backend_param=my_backend_param,
        )
        vars, attrs, coords = my_decode_variables(
            vars, attrs, decode_times, decode_timedelta, decode_coords
        )  #  see also conventions.decode_cf_variables

        ds = xr.Dataset(vars, attrs=attrs)
        ds = ds.set_coords(coords)
        ds.set_close(store.close)

        return ds


The output :py:class:`~xarray.Dataset` shall implement the additional custom method
``close``, used by Xarray to ensure the related files are eventually closed. This
method shall be set by using :py:meth:`~xarray.Dataset.set_close`.


The input of ``open_dataset`` method are one argument
(``filename``) and one keyword argument (``drop_variables``):

- ``filename``: can be a string containing a path or an instance of
  :py:class:`pathlib.Path`.
- ``drop_variables``: can be `None` or an iterable containing the variable
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
in backend ``open_dataset`` signature.

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
:py:class:`~xarray.backends.common.BackendEntrypoint` already provides a
default :py:meth:`~xarray.backend.common.BackendEntrypoint.guess_can_open`
that always returns ``False``.

Backend ``guess_can_open`` takes as input the ``filename_or_obj`` parameter of
Xarray :py:meth:`~xarray.open_dataset`, and returns a boolean.

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
+++++++++++++++++++++++++++

Define a new entrypoint in your ``setup.py`` (or ``setup.cfg``) with:

- group: ``xarray.backend``
- name: the name to be passed to :py:meth:`~xarray.open_dataset`  as ``engine``
- object reference: the reference of the class that you have implemented.

You can declare the entrypoint in ``setup.py`` using the following syntax:

.. code-block::

    setuptools.setup(
        entry_points={
            "xarray.backends": [
                "engine_name=your_package.your_module:YourBackendEntryClass"
            ],
        },
    )

in ``setup.cfg``:

.. code-block:: cfg

    [options.entry_points]
    xarray.backends =
        engine_name = your_package.your_module:YourBackendEntryClass


See https://packaging.python.org/specifications/entry-points/#data-model
for more information

If you are using [Poetry](https://python-poetry.org/) for your build system, you can accomplish the same thing using "plugins". In this case you would need to add the following to your ``pyproject.toml`` file:

.. code-block:: toml

    [tool.poetry.plugins."xarray_backends"]
    "engine_name" = "your_package.your_module:YourBackendEntryClass"

See https://python-poetry.org/docs/pyproject/#plugins for more information on Poetry plugins.

.. _RST lazy_loading:

How to support Lazy Loading
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
  from :py:class:`~xarray.backends.common.BackendArray`.

BackendArray subclassing
^^^^^^^^^^^^^^^^^^^^^^^^

The BackendArray subclass shall implement the following method and attributes:

- the ``__getitem__`` method that takes in input an index and returns a
  `NumPy <https://numpy.org/>`__ array
- the ``shape`` attribute
- the ``dtype`` attribute.


Xarray supports different type of
`indexing <http://xarray.pydata.org/en/stable/indexing.html>`__, that can be
grouped in three types of indexes
:py:class:`~xarray.core.indexing.BasicIndexer`,
:py:class:`~xarray.core.indexing.OuterIndexer` and
:py:class:`~xarray.core.indexing.VectorizedIndexer`.
This implies that the implementation of the method ``__getitem__`` can be tricky.
In oder to simplify this task, Xarray provides a helper function,
:py:func:`~xarray.core.indexing.explicit_indexing_adapter`, that transforms
all the input  ``indexer`` types (`basic`, `outer`, `vectorized`) in a tuple
which is interpreted correctly by your backend.

This is an example ``BackendArray`` subclass implementation:

.. code-block:: python

    class MyBackendArray(BackendArray):
        def __init__(
            self,
            shape,
            dtype,
            lock,
            # other backend specific keyword arguments
        ):
            self.shape = shape
            self.dtype = lock
            self.lock = dtype

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

In order to support `Dask <http://dask.pydata.org/>`__ distributed and
:py:mod:`multiprocessing`, ``BackendArray`` subclass should be serializable
either with :ref:`io.pickle` or
`cloudpickle <https://github.com/cloudpipe/cloudpickle>`__.
That implies that all the reference to open files should be dropped. For
opening files, we therefore suggest to use the helper class provided by Xarray
:py:class:`~xarray.backends.CachingFileManager`.

.. _RST indexing:

Indexing Examples
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

If you support more complex indexing as `explicit indexing` or
`numpy indexing`, you can have a look to the implemetation of Zarr backend and Scipy backend,
currently available in :py:mod:`~xarray.backends` module.

.. _RST preferred_chunks:

Backend preferred chunks
^^^^^^^^^^^^^^^^^^^^^^^^

The backend is not directly involved in `Dask <http://dask.pydata.org/>`__
chunking, since it is internally managed by Xarray. However, the backend can
define the preferred chunk size inside the variable’s encoding
``var.encoding["preferred_chunks"]``. The ``preferred_chunks`` may be useful
to improve performances with lazy loading. ``preferred_chunks`` shall be a
dictionary specifying chunk size per dimension like
``{“dim1”: 1000, “dim2”: 2000}``  or
``{“dim1”: [1000, 100], “dim2”: [2000, 2000, 2000]]}``.

The ``preferred_chunks`` is used by Xarray to define the chunk size in some
special cases:

- if ``chunks`` along a dimension is ``None`` or not defined
- if ``chunks`` is ``"auto"``.

In the first case Xarray uses the chunks size specified in
``preferred_chunks``.
In the second case Xarray accommodates ideal chunk sizes, preserving if
possible the "preferred_chunks". The ideal chunk size is computed using
:py:func:`dask.core.normalize_chunks`, setting
``previous_chunks = preferred_chunks``.
