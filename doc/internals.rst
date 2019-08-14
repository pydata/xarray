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

This achieves the same result as if the ``Dataset`` class had a cached property
defined that returns an instance of your class:

.. code-block:: python

  class Dataset:
      ...
      @property
      def geo(self)
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

    ds = xr.Dataset({'longitude': np.linspace(0, 10),
                     'latitude': np.linspace(0, 20)})
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
