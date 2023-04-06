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
        values can loosely be thought of as tick labels along a dimension. There
        are two types of coordinate arrays: *dimension coordinates* and
        *non-dimension coordinates* (see below). A coordinate named ``x`` can be
        retrieved from ``arr.coords[x]``. A ``DataArray`` can have more
        coordinates than dimensions because a single dimension can be labeled by
        multiple coordinate arrays. However, only one coordinate array can be a
        assigned as a particular dimension's dimension coordinate array. As a
        consequence, ``len(arr.dims) <= len(arr.coords)`` in general.

    Dimension coordinate
        A one-dimensional coordinate array assigned to ``arr`` with both a name
        and dimension name in ``arr.dims``. Dimension coordinates are used for
        label-based indexing and alignment, like the index found on a
        :py:class:`pandas.DataFrame` or :py:class:`pandas.Series`. In fact,
        dimension coordinates use :py:class:`pandas.Index` objects under the
        hood for efficient computation. Dimension coordinates are marked by
        ``*`` when printing a ``DataArray`` or ``Dataset``.

    Non-dimension coordinate
        A coordinate array assigned to ``arr`` with a name in ``arr.coords`` but
        *not* in ``arr.dims``. These coordinates arrays can be one-dimensional
        or multidimensional, and they are useful for auxiliary labeling. As an
        example, multidimensional coordinates are often used in geoscience
        datasets when :doc:`the data's physical coordinates (such as latitude
        and longitude) differ from their logical coordinates
        <../examples/multidimensional-coords>`. However, non-dimension coordinates
        are not indexed, and any operation on non-dimension coordinates that
        leverages indexing will fail. Printing ``arr.coords`` will print all of
        ``arr``'s coordinate names, with the corresponding dimension(s) in
        parentheses. For example, ``coord_name (dim_name) 1 2 3 ...``.

    Index
        An *index* is a data structure optimized for efficient selecting and
        slicing of an associated array. Xarray creates indexes for dimension
        coordinates so that operations along dimensions are fast, while
        non-dimension coordinates are not indexed. Under the hood, indexes are
        implemented as :py:class:`pandas.Index` objects. The index associated
        with dimension name ``x`` can be retrieved by ``arr.indexes[x]``. By
        construction, ``len(arr.dims) == len(arr.indexes)``

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

    Aligning

        Aligning refers to the process of making sure that the dimensions
        and coordinates of two or more DataArrays or Datasets are consistent with each
        other so that they can be combined or compared properly.

        For example, if you have two DataArrays that represent temperature measurements at
        different times, but one has a time coordinate in seconds and the other has a time
        coordinate in hours, you would need to align the two arrays by converting the time
        coordinate in one of the arrays to match the other array. Once the arrays are aligned,
        you can perform operations on them that require matching dimensions or coordinates,
        such as taking the difference between the two arrays or calculating the mean across time.

    Broadcasting

        Broadcasting is a technique that allows operations to be performed on arrays
        with different shapes and dimensions. When performing operations on arrays with different
        shapes and dimensions, xarray will automatically broadcast the arrays to a common shape
        before the operation is applied.

        For example, if you have two arrays with different shapes, xarray will try to match the
        dimensions of the arrays and add new dimensions as necessary. This allows for easy element-wise
        operations on arrays that might otherwise have incompatible shapes.

    Merging

        Merging refers to the process of combining multiple DataArrays or Dataset objects
        along one or more dimensions to create a new Dataset.

        The merge() function allows you to combine multiple DataArrays or Dataset objects into a
        single ``Dataset`` along one or more shared dimensions. If the input objects have different values for
        the same coordinate, merge() will create a new coordinate with the union of the values from the input objects.

        Suppose you have two datasets, both containing temperature data from different weather
        stations over the same time period. You want to combine these two datasets into a single dataset.
        Assuming that both datasets have the same coordinates (time, latitude, and longitude), you can merge
        them using merge() function.

    Concatenating

        Concatenating refers to the process of combining two or more arrays along a given dimension
        to create a new array. The resulting array has the same shape as the input arrays, except for the dimension
        along which the concatenation was performed, which is expanded to include the data from all input arrays.

        Concatenation is commonly used when working with multi-dimensional arrays that represent data over time or space.
        For example, if you have daily temperature data for multiple years, you can concatenate the arrays along the time
        dimension to create a single array with all the data.

    Combining

        Combining refers to the process of merging multiple DataArrays or Datasets along a shared dimension
        to create a new object. This can be useful when working with data that has been split into multiple files, or
        when wanting to combine data from different sources.

        Suppose we have one dataset containing temperature data and another dataset containing precipitation data,
        both measured at the same set of locations and times. We can combine these two datasets using the ``combine_by_coords``
        method in xarray to create a single dataset with both temperature and precipitation variables. The resulting dataset
        will have the same coordinates as the original datasets and the variables will be combined based on their coordinates.
        This allows us to easily analyze and visualize both variables together in a single dataset.
