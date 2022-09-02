.. currentmodule:: xarray

How to add a custom index
=========================

.. warning::

   This feature is highly experimental. Support for custom indexes
   has been introduced in v2022.06.0 and is still incomplete.

Xarray's built-in support for label-based indexing and alignment operations
relies on :py:class:`pandas.Index` objects. It is powerful and suitable for many
applications but it also has some limitations:

- it only works with 1-dimensional coordinates
- it is hard to reuse it with irregular data for which there exist more
  efficient, tree-based structures to perform data selection
- it doesn't support extra metadata that may be required for indexing and
  alignment (e.g., a coordinate reference system)

Fortunately, Xarray now allows extending this functionality with custom indexes,
which can be implemented in 3rd-party libraries.

The Index base class
--------------------

Every Xarray index must inherit from the :py:class:`Index` base class. It is the
case of Xarray built-in ``PandasIndex`` and ``PandasMultiIndex`` subclasses,
which wrap :py:class:`pandas.Index` and :py:class:`pandas.MultiIndex`
respectively.
