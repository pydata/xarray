
Extending xarray
================

.. ipython:: python
    :suppress:

    import xarray as xr


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

.. literalinclude:: ../examples/_code/accessor_example.py

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
or patching in a large number of domain specific methods.

Parametrizing an accessor is possible by defining ``__call__``. For
example, we could use this to reimplement the API of the
:py:func:`DataArray.weighted` function:

.. ipython::
    :okwarning:

    In [1]: @xr.register_dataarray_accessor("weighted")
       ...: class WeightedAccessor:
       ...:     def __init__(self, xarray_obj):
       ...:         self._obj = xarray_obj
       ...:         self._weights = np.ones_like(xarray_obj) / xarray_obj.size
       ...:
       ...:     def __call__(self, weights):
       ...:         self._weights = weights
       ...:         return self
       ...:
       ...:     def sum(self, *args, **kwargs):
       ...:         return np.sum(self._obj * self._weights, *args, **kwargs)

    In [2]: da = xr.DataArray(data=np.linspace(1, 2, 10), dims="x")
       ...: weights = xr.DataArray(
       ...:     np.array([0.25, 0, 0, 0, 0.25, 0, 0.25, 0, 0, 0.25]),
       ...:     dims="x",
       ...: )

    In [3]: da.weighted.sum()

    In [4]: da.weighted(weights).sum()

If we want to require the parameter, the easiest way to do so is using
a wrapper function:

.. ipython::
    :okwarning:
    :okexcept:

    In [1]: class Weighted:
       ...:     def __init__(self, obj, weights):
       ...:         self._obj = obj
       ...:         self._weights = weights
       ...:
       ...:     def sum(self, *args, **kwargs):
       ...:         return np.sum(self._obj * self._weights, *args, **kwargs)

    In [2]: @xr.register_dataarray_accessor("weighted")
       ...: def weighted(obj):
       ...:     def wrapped(weights):
       ...:         return Weighted(obj, weights)
       ...:     return wrapped

    In [3]: da = xr.DataArray(data=np.linspace(1, 2, 10), dims="x")
       ...: weights = xr.DataArray(
       ...:     np.array([0.25, 0, 0, 0, 0.25, 0, 0.25, 0, 0, 0.25]),
       ...:     dims="x",
       ...: )

    In [4]: da.weighted.sum()

    In [5]: da.weighted(weights).sum()

.. note::

   Keep in mind, though, that accessors are designed to add new
   namespaces to the :py:class:`Dataset` and :py:class:`DataArray`
   objects and should not be used to add methods.

For further reading on ways to write new accessors and the philosophy
behind the approach, see :issue:`1080`.

To help users keep things straight, please `let us know
<https://github.com/pydata/xarray/issues>`_ if you plan to write a new accessor
for an open source library. In the future, we will maintain a list of accessors
and the libraries that implement them on this page.

To make documenting accessors with ``sphinx`` and ``sphinx.ext.autosummary``
easier, you can use `sphinx-autosummary-accessors`_.

.. _sphinx-autosummary-accessors: https://sphinx-autosummary-accessors.readthedocs.io/
