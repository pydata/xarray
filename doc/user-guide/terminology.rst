.. currentmodule:: xarray
.. _terminology:

Terminology
===========

*Xarray terminology differs slightly from CF, mathematical conventions, and
pandas; so we've put together a glossary of its terms. Here,* ``arr``
*refers to an xarray* :py:class:`DataArray` *in the examples. For more
complete examples, please consult the relevant documentation.*

.. glossary::

    DataArray
        A multi-dimensional array with labeled or named
        dimensions. ``DataArray`` objects add metadata such as dimension names,
        coordinates, and attributes (defined below) to underlying "unlabeled"
        data structures such as numpy and Dask arrays. If its optional ``name``
        property is set, it is a *named DataArray*.

    Dataset
        A dict-like collection of ``DataArray`` objects with aligned
        dimensions. Thus, most operations that can be performed on the
        dimensions of a single ``DataArray`` can be performed on a
        dataset. Datasets have data variables (see **Variable** below),
        dimensions, coordinates, and attributes.

    Variable
        A `NetCDF-like variable
        <https://docs.unidata.ucar.edu/nug/current/netcdf_data_set_components.html#variables>`_
        consisting of dimensions, data, and attributes which describe a single
        array. The main functional difference between variables and numpy arrays
        is that numerical operations on variables implement array broadcasting
        by dimension name. Each ``DataArray`` has an underlying variable that
        can be accessed via ``arr.variable``. However, a variable is not fully
        described outside of either a ``Dataset`` or a ``DataArray``.

        .. note::

            The :py:class:`Variable` class is low-level interface and can
            typically be ignored. However, the word "variable" appears often
            enough in the code and documentation that is useful to understand.

    Dimension
        In mathematics, the *dimension* of data is loosely the number of degrees
        of freedom for it. A *dimension axis* is a set of all points in which
        all but one of these degrees of freedom is fixed. We can think of each
        dimension axis as having a name, for example the "x dimension".  In
        xarray, a ``DataArray`` object's *dimensions* are its named dimension
        axes, and the name of the ``i``-th dimension is ``arr.dims[i]``. If an
        array is created without dimension names, the default dimension names are
        ``dim_0``, ``dim_1``, and so forth.

    Coordinate
        An array that labels a dimension or set of dimensions of another
        ``DataArray``. In the usual one-dimensional case, the coordinate array's
        values can loosely be thought of as tick labels along a dimension. We
        distinguish :term:`Dimension coordinate` vs. :term:`Non-dimension
        coordinate` and :term:`Indexed coordinate` vs. :term:`Non-indexed
        coordinate`. A coordinate named ``x`` can be retrieved from
        ``arr.coords[x]``. A ``DataArray`` can have more coordinates than
        dimensions because a single dimension can be labeled by multiple
        coordinate arrays. However, only one coordinate array can be a assigned
        as a particular dimension's dimension coordinate array. As a
        consequence, ``len(arr.dims) <= len(arr.coords)`` in general.

    Dimension coordinate
        A one-dimensional coordinate array assigned to ``arr`` with both a name
        and dimension name in ``arr.dims``. Usually (but not always), a
        dimension coordinate is also an :term:`Indexed coordinate` so that it can
        be used for label-based indexing and alignment, like the index found on
        a :py:class:`pandas.DataFrame` or :py:class:`pandas.Series`.

    Non-dimension coordinate
        A coordinate array assigned to ``arr`` with a name in ``arr.coords`` but
        *not* in ``arr.dims``. These coordinates arrays can be one-dimensional
        or multidimensional, and they are useful for auxiliary labeling. As an
        example, multidimensional coordinates are often used in geoscience
        datasets when :doc:`the data's physical coordinates (such as latitude
        and longitude) differ from their logical coordinates
        <../examples/multidimensional-coords>`. Printing ``arr.coords`` will
        print all of ``arr``'s coordinate names, with the corresponding
        dimension(s) in parentheses. For example, ``coord_name (dim_name) 1 2 3
        ...``.

    Indexed coordinate
        A coordinate which has an associated :term:`Index`. Generally this means
        that the coordinate labels can be used for indexing (selection) and/or
        alignment. An indexed coordinate may have one or more arbitrary
        dimensions although in most cases it is also a :term:`Dimension
        coordinate`. It may or may not be grouped with other indexed coordinates
        depending on whether they share the same index. Indexed coordinates are
        marked by ``*`` when printing a ``DataArray`` or ``Dataset``.

    Non-indexed coordinate
        A coordinate which has no associated :term:`Index`. It may still
        represent fixed labels along one or more dimensions but it cannot be
        used for label-based indexing and alignment.

    Index
        An *index* is a data structure optimized for efficient data selection
        and alignment within a discrete or continuous space that is defined by
        coordinate labels (unless it is a functional index). By default, Xarray
        creates a :py:class:`~xarray.indexes.PandasIndex` object (i.e., a
        :py:class:`pandas.Index` wrapper) for each :term:`Dimension coordinate`.
        For more advanced use cases (e.g., staggered or irregular grids,
        geospatial indexes), Xarray also accepts any instance of a specialized
        :py:class:`~xarray.indexes.Index` subclass that is associated to one or
        more arbitrary coordinates. The index associated with the coordinate
        ``x`` can be retrieved by ``arr.xindexes[x]`` (or ``arr.indexes["x"]``
        if the index is convertible to a :py:class:`pandas.Index` object). If
        two coordinates ``x`` and ``y`` share the same index,
        ``arr.xindexes[x]`` and ``arr.xindexes[y]`` both return the same
        :py:class:`~xarray.indexes.Index` object.

    name
        The names of dimensions, coordinates, DataArray objects and data
        variables can be anything as long as they are :term:`hashable`. However,
        it is preferred to use :py:class:`str` typed names.

    scalar
        By definition, a scalar is not an :term:`array` and when converted to
        one, it has 0 dimensions. That means that, e.g., :py:class:`int`,
        :py:class:`float`, and :py:class:`str` objects are "scalar" while
        :py:class:`list` or :py:class:`tuple` are not.

    duck array
        `Duck arrays`__ are array implementations that behave
        like numpy arrays. They have to define the ``shape``, ``dtype`` and
        ``ndim`` properties. For integration with ``xarray``, the ``__array__``,
        ``__array_ufunc__`` and ``__array_function__`` protocols are also required.

        __ https://numpy.org/neps/nep-0022-ndarray-duck-typing-overview.html
