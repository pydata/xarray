Variable objects
================

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
