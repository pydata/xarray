.. _internals:

xarray Internals
================

.. currentmodule:: xarray

xarray builds upon two of the foundational libraries of the scientific Python
stack, NumPy and pandas. It is written in pure Python (no C or Cython
extensions), which makes it easy to develop and extend. Instead, we push
compiled code to :ref:`optional dependencies<installing>`.

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


Extending xarray
----------------

.. ipython:: python
    :suppress:

    import numpy as np
    import pandas as pd
    import xarray as xr

    np.random.seed(123456)

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


How to add a new backend
------------------------------------

Adding a new backend for read support to Xarray is easy, and does not require
to integrate any code in Xarray; all you need to do is approaching the
following steps:

- Create a class that inherits from Xarray py:class:`~xarray.backend.commonBackendEntrypoint`
- Implement the method ``open_dataset`` that returns an instance of :py:class:`~xarray.Dataset`
- Declare such a class as an external plugin in your setup.py.

Your ``BackendEntrypoint`` sub-class is the primary interface with Xarray, and
it should implement the following attributes and functions:

- ``open_dataset`` (mandatory)
- [``open_dataset_parameters``] (optional)
- [``guess_can_open``] (optional)

These are detailed in the following.

open_dataset
++++++++++++

Inputs
^^^^^^

The backend ``open_dataset`` method takes as input one argument
(``filename``), and one keyword argument (``drop_variables``):

- ``filename``: can be a string containing a relative path or an instance of ``pathlib.Path``.
- ``drop_variables``: can be `None` or an iterable containing the variable names to be dropped when reading the data.

If it makes sense for your backend, your ``open_dataset`` method should
implement in its interface all the following boolean keyword arguments, called
**decoders** which default to ``None``:

- ``mask_and_scale=None``
- ``decode_times=None``
- ``decode_timedelta=None``
- ``use_cftime=None``
- ``concat_characters=None``
- ``decode_coords=None``

These keyword arguments are explicitly defined in Xarray
:py:meth:`~xarray.open_dataset` signature.  Xarray will pass them to the
backend only if the User sets a value different from ``None`` explicitly.
Your backend can also take as input a set of backend-specific keyword
arguments. All these keyword arguments can be passed to
:py:meth:`~xarray.open_dataset` grouped either via the ``backend_kwargs``
parameter or explicitly using the syntax ``**kwargs``.

Output
^^^^^^
The output of the backend `open_dataset` shall be an instance of
Xarray py:class:`~xarray.Dataset` that implements the additional method ``close``,
used by Xarray to ensure the related files are eventually closed.

If you don't want to support the lazy loading, then the :py:class:`~xarray.Dataset`
shall contain :py:class:`numpy.ndarray` and your work is almost done.

open_dataset_parameters
+++++++++++++++++++++++
``open_dataset_parameters`` is the list of backend ``open_dataset`` parameters.
It is not a mandatory parameter, and if the backend does not provide it
explicitly, Xarray creates a list of them automatically by inspecting the
backend signature.

Xarray uses ``open_dataset_parameters`` only when it needs to select
the **decoders** supported by the backend.

If ``open_dataset_parameters`` is not defined, but ``**kwargs`` and ``*args`` have
been passed to the signature, Xarray raises an error.
On the other hand, if the backend provides the ``open_dataset_parameters``,
then ``**kwargs`` and `*args`` can be used in the signature.

However, this practice is discouraged unless there is a good reasons for using
`**kwargs` or `*args`.

guess_can_open
++++++++++++++
``guess_can_open`` is used to identify the proper engine to open your data
file automatically in case the engine is not specified explicitly. If you are
not interested in supporting this feature, you can skip this step since
py:class:`~xarray.backend.common.BackendEntrypoint` already provides a default
py:meth:`~xarray.backend.common BackendEntrypoint.guess_engine` that always returns ``False``.

Backend ``guess_can_open`` takes as input the ``filename_or_obj`` parameter of
Xarray :py:meth:`~xarray.open_dataset`, and returns a boolean.


How to register a backend
+++++++++++++++++++++++++++

Define in your setup.py (or setup.cfg) a new entrypoint with:

- group: ``xarray.backend``
- name: the name to be passed to :py:meth:`~xarray.open_dataset`  as ``engine``.
- object reference: the reference of the class that you have implemented.

See https://packaging.python.org/specifications/entry-points/#data-model
for more information

How to support Lazy Loading
+++++++++++++++++++++++++++
If you want to make your backend effective with big datasets, then you should support
the lazy loading.
Basically, you shall replace the :py:class:`numpy.array` inside the variables with
a custom class:

.. ipython:: python
    backend_array = YourBackendArray()
    data = indexing.LazilyOuterIndexedArray(backend_array)
    variable = Variable(..., data, ...)

Where ``YourBackendArray``is a class that inherit from
:py:class:`~xarray.backends.common.BackendArray` and
:py:class:`~xarray.core.indexing.LazilyOuterIndexedArray` is a
class of Xarray that wraps an array to make basic and outer indexing lazy.

BackendArray
^^^^^^^^^^^^

CachingFileManager
^^^^^^^^^^^^^^^^^^


Dask chunking
+++++++++++++
The backend is not directly involved in `Dask <http://dask.pydata.org/>`__ chunking, since it is managed
internally by Xarray. However, the backend can define the preferred chunk size
inside the variable’s encoding ``var.encoding["preferred_chunks"]``.
The ``preferred_chunks`` may be useful to improve performances with lazy loading.
``preferred_chunks`` shall be a  dictionary specifying chunk size per dimension
like ``{“dim1”: 1000, “dim2”: 2000}``  or
``{“dim1”: [1000, 100], “dim2”: [2000, 2000, 2000]]}``.

The ``preferred_chunks`` is used by Xarray to define the chunk size in some
special cases:

- If ``chunks`` along a dimension is None or not defined
- If ``chunks`` is “auto”

In the first case Xarray uses the chunks size specified in
``preferred_chunks``.
In the second case Xarray accommodates ideal chunk sizes, preserving if
possible the "preferred_chunks". The ideal chunk size is computed using
``dask.core.normalize function``, setting ``previus_chunks = preferred_chunks``.


Decoders
++++++++
The decoders implement specific operations to transform data from on-disk
representation to Xarray representation.

A classic example is the “time” variable decoding operation. In NetCDF, the
elements of the “time” variable are stored as integers, and the unit contains
an origin (for example: "seconds since 1970-1-1"). In this case, Xarray
transforms the pair integer-unit in a ``np.datetimes``.

The standard decoders implemented in Xarray are:
- strings.CharacterArrayCoder()
- strings.EncodedStringCoder()
- variables.UnsignedIntegerCoder()
- variables.CFMaskCoder()
- variables.CFScaleOffsetCoder()
- times.CFTimedeltaCoder()
- times.CFDatetimeCoder()

Some of the transformations can be common to more backends, so before
implementing a new decoder, be sure Xarray does not already implement that one.

The backends can reuse Xarray’s decoders, either instantiating the decoders
directly or using the higher-level function
:py:func:`~xarray.conventions.decode_cf_variables` that groups Xarray decoders.

In some cases, the transformation to apply strongly depends on the on-disk
data format. Therefore, you may need to implement your decoder.

An example of such a case is when you have to deal with the time format of a
grib file. grib format is very different from the NetCDF one: in grib, the
time is stored in two attributes dataDate and dataTime as strings. Therefore,
it is not possible to reuse the Xarray time decoder, and implementing a new
one is mandatory.

Decoders can be activated or deactivated using the boolean keywords of
:py:meth:`~xarray.open_dataset` signature: ``mask_and_scale``,
``decode_times``, ``decode_timedelta``, ``use_cftime``,
``concat_characters``, ``decode_coords``.

Such keywords are passed to the backend only if the User sets a value
different from ``None``.  Note that the backend does not necessarily have to
implement all the decoders, but it shall declare in its ``open_dataset``
interface only the boolean keywords related to the supported decoders. The
backend shall implement the deactivation and activation of the supported
decoders.
