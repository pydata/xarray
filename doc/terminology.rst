.. _terminology:

Terminology
===========

*Xarray terminology differs slightly from CF, mathematical conventions, and pandas; and therefore using xarray, understanding the documentation, and parsing error messages is easier once key terminology is defined. This glossary was designed so that more fundamental concepts come first. Thus for new users, this page is best read top-to-bottom. Throughout the glossary,* ``arr`` *will refer to an xarray* :py:class:`DataArray` *in any small examples. For more complete examples, please consult the relevant documentation.*

----

**DataArray:** A multi-dimensional array with labeled or named dimensions. ``DataArray`` objects add metadata such as dimension names, coordinates, and attributes (defined below) to underlying "unlabeled" data structures such as numpy and Dask arrays. If its optional ``name`` property is set, it is a *named DataArray*.

----

**Dataset:** A dict-like collection of ``DataArray`` objects with aligned dimensions. Thus, most operations that can be performed on the dimensions of a single ``DataArray`` can be performed on a dataset. Datasets have data variables (see **Variable** below), dimensions, coordinates, and attributes.

----

**Variable:** A `NetCDF-like variable <https://www.unidata.ucar.edu/software/netcdf/netcdf/Variables.html>`_ consisting of dimensions, data, and attributes which describe a single array. The main functional difference between variables and numpy arrays is that numerical operations on variables implement array broadcasting by dimension name. Each ``DataArray`` has an underlying variable that can be accessed via ``arr.variable``. However, a variable is not fully described outside of either a ``Dataset`` or a ``DataArray``.

.. note::

    The :py:class:`Variable` class is low-level interface and can typically be ignored. However, the word "variable" appears often enough in the code and documentation that is useful to understand.

----

**Dimension:** In mathematics, the *dimension* of data is loosely the number of degrees of freedom for it. A *dimension axis* is a set of all points in which all but one of these degrees of freedom is fixed. We can think of each dimension axis as having a name, for example the "x dimension".  In xarray, a ``DataArray`` object's *dimensions* are its named dimension axes, and the name of the ``i``-th dimension is ``arr.dims[i]``. If an array is created without dimensions, the default dimension names are ``dim_0``, ``dim_1``, and so forth.

----

**Coordinate:** An array that labels a dimension of another ``DataArray``. Loosely, the coordinate array's values can be thought of as tick labels along a dimension. There are two types of coordinate arrays: *dimension coordinates* and *non-dimension coordinates* (see below). A coordinate named ``x`` can be retrieved from ``arr.coords[x]``. A ``DataArray`` can have more coordinates than dimensions because a single dimension can be assigned multiple coordinate arrays. However, only one coordinate array can be a assigned as a particular dimension's dimension coordinate array. As a consequence, ``len(arr.dims) <= len(arr.coords)`` in general.

----

**Dimension coordinate:** A coordinate array assigned to ``arr`` with both a name and dimension name in ``arr.dims``. Dimension coordinates are used for label-based indexing and alignment, like the index found on a :py:class:`pandas.DataFrame` or :py:class:`pandas.Series`. In fact, dimension coordinates use :py:class:`pandas.Index` objects under the hood for efficient computation. Dimension coordinates are marked by ``*`` when printing a ``DataArray`` or ``Dataset``.

----

**Non-dimension coordinate:** A coordinate array assigned to ``arr`` with a name in ``arr.dims`` but a dimension name *not* in ``arr.dims``. These coordinate arrays are useful for auxiliary labeling. However, non-dimension coordinates are not indexed, and any operation on non-dimension coordinates that leverages indexing will fail. Printing ``arr.coords`` will print all of ``arr``'s coordinate names, with the assigned dimensions in parentheses. For example, ``coord_name   (dim_name) 1 2 3 ...``.

----

**Index:** An *index* is a data structure optimized for efficient selecting and slicing of an associated array. Xarray creates indexes for dimension coordinates so that operations along dimensions are fast, while non-dimension coordinates are not indexed. Under the hood, indexes are implemented as :py:class:`pandas.Index` objects. The index associated with dimension name ``x`` can be retrieved by ``arr.indexes[x]``. By construction, ``len(arr.dims) == len(arr.indexes)``