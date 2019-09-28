.. _terminology:

.. https://github.com/pydata/xarray/issues/2410
.. https://github.com/pydata/xarray/issues/1295

Terminology
===========

*Xarray terminology differs slightly from CF and mathematical conventions, and therefore using xarray, understanding the documentation, and parsing error messages is easier once key terminology is defined. This glossary was designed so that more fundamental concepts come first. Thus for new users, this page is best read top-to-bottom. Throughout the glossary,* ``arr`` *will refer to an xarray* :py:class:`DataArray` *in any small examples. For more complete examples, please consult the relevant documentation.*

----

**DataArray:** A multi-dimensional array with labeled or named dimensions. If its optional ``name`` property is set, it is a *named DataArray*.

----

**Dimension / dimensions:** A *dimension* is a nonnegative number for the dimensionality of the underlying data, while an array's *dimensions* are a set of dimension names. The name of the ``i``-th dimension is ``arr.dims[i]``. If an array is created without dimensions, the default dimension names are ``dim_0``, ``dim_1``, and so forth.

----

**Coordinate:** A one-dimensional array that labels a dimension of another ``DataArray``. There are two types of coordinate arrays: *dimension coordinates* and *non-dimension coordinates* (see below). A coordinate named ``x`` can be retrieved from ``arr.coords[x]``. A ``DataArray`` can have more coordinates than dimensions because a single dimension can be assigned multiple coordinate arrays. However, only one coordinate array can be a assigned as a particular dimension's dimension coordinate    array. As a consequence, ``len(arr.dims) <= len(arr.coords)`` in general.

----

**Dimension coordinate:** A coordinate array assigned to ``arr`` with both a name and dimension name in ``arr.dims`` (see **Name matching rules** below). Dimension coordinates are used for label-based indexing and alignment, like the index found on a :py:class:`pandas.DataFrame` or :py:class:`pandas.Series`. In fact, dimension coordinates use :py:class:`pandas.Index` objects under the hood for efficient computation. Dimension coordinates are marked by ``*`` when printing a ``DataArray`` or ``Dataset``.

----

**Non-dimension coordinate:** A coordinate array assigned to `arr`` with a name in ``arr.dims`` but a dimension name *not* in ``arr.dims`` (see **Name matching rules** below). These coordinate arrays are useful for auxiliary labeling. However, non-dimension coordinates are not indexed, and any operation on non-dimension coordinates that leverages indexing will fail. Printing ``arr.coords`` will print all of ``arr``'s coordinate names, with the assigned dimensions in parentheses. For example, ``coord_name   (dim_name) 1 2 3 ...``.

.. note::

    **Name matching rules:** Xarray follows simple but important-to-grok name matching rules for dimensions and coordinates. Let ``arr`` be an array with an existing dimension ``x`` and assigned new coordinates ``new_coords``. If ``new_coords`` is a list-like collection, then they must be assigned a name that matches an existing dimension. For example, if ``arr.assign_coords({'x': new_coords}).``

    However, if ``new_coords`` is a one-dimensional ``DataArray``, then the rules are slightly more complex. In this case, if both ``new_coords``'s name and only dimension match any dimension name in ``arr.dims``, it is assigned as a dimension coordinate to ``arr``. If ``new_coords``'s name matches a name in ``arr.dims`` but its own dimension name does not, it is assigned as a non-dimension coordinate with name ``new_coords.dims[0]`` to ``arr``. Otherwise, an exception is raised.

----

**Index:** An *index* is a :py:class:`pandas.Index` that indexes the values in a dimension coordinate. Non-dimension coordinates are not indexed. The index associated with dimension name ``x`` can be retrieved by ``arr.indexes[x]``. By construction, ``len(arr.dims) == len(arr.indexes)``

----

**Dataset:** A dict-like collection of ``DataArray`` objects with aligned dimensions. Thus, most operations that can be performed on the dimensions of a single ``DataArray`` can be performed on a dataset.

----

**Variable:** A `NetCDF-like variable <https://www.unidata.ucar.edu/software/netcdf/netcdf/Variables.html>`_ consisting of dimensions, data, and attributes which describe a single array. The main functional difference between variables and numpy arrays is that numerical operations on variables implement array broadcasting by dimension name. Each ``DataArray`` has an underlying variable that can be accessed via ``arr.variable``. However, a variable is not fully described outside of either a ``Dataset`` or a ``DataArray``.

.. note::

    The :py:class:`Variable` class is low-level interface and can typically be ignored. However, the word "variable" appears often enough in the code and documentation that is useful to understand.
