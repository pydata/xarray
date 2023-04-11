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

.. ipython:: python
    :suppress:

    import xarray as xr
    import numpy as np

Aligning
    Aligning refers to the process of ensuring that two or more DataArrays or Datasets
    have the same dimensions and coordinates, so that they can be combined or compared properly.

.. ipython:: python

    # Two DataArrays with different time coordinates
    time1 = np.arange("2022-01-01", "2022-01-06", dtype="datetime64")
    time2 = np.arange("2022-01-03", "2022-01-08", dtype="datetime64")

    # Two DataArrays of random temperature values, each with time as a coordinate
    temp1 = xr.DataArray(
        np.random.rand(len(time1)), coords=[("time", time1)], name="temp"
    )
    temp2 = xr.DataArray(
        np.random.rand(len(time2)), coords=[("time", time2)], name="temp"
    )

    # Align the two DataArrays along the time dimension using the 'outer' join method
    temp1_aligned, temp2_aligned = xr.align(temp1, temp2, join="outer")

    # Print the resulting DataArrays
    print(temp1_aligned)
    print(temp2_aligned)

There are two DataArrays 'temp1' and 'temp2' with different time coordinates. We then use the align
method to align the two DataArrays along the time dimension. The join parameter is set to 'outer', which means that
the resulting DataArrays will have all time values that are present in either temp1 or temp2.
The align method returns two new DataArrays, temp1_aligned and temp2_aligned now have the same length, and their time
coordinates span the entire range from '2022-01-01' to '2022-01-07'. Any missing values are filled with NaNs.

Broadcasting
    Broadcasting is a technique that allows operations to be performed on arrays with different shapes and dimensions.
    When performing operations on arrays with different shapes and dimensions, xarray will automatically broadcast the
    arrays to a common shape before the operation is applied.

.. ipython:: python

    a = xr.DataArray(np.array([1, 2, 3]), dims=["x"])
    b = xr.DataArray(np.array([4, 5, 6, 7]), dims=["y"])
    result = a + b
    print(result)

In this example, 'a' has shape (3,) and 'b' has shape (4,).
If we try to add these two arrays, xarray will automatically broadcast the arrays to a common shape before performing
the addition. It will extend 'a' along the new 'y' dimension and extend 'b' along the new 'x' dimension so that both
arrays have shape (3, 4).
The result is a 2D array with shape (3, 4) where each element is the sum of the corresponding elements
in 'a' and 'b'. Note that xarray has also automatically added coordinates for the new dimensions 'x' and 'y'.

**In xarray, "merging", "concatenating", and "combining" are all operations used to combine two or more DataArrays or
Datasets into a single** ``DataArray`` **or** ``Dataset``. **However, each of these operations has a slightly different meaning and
purpose.**

Merging
    Merging is used to combine two or more Datasets or DataArrays that have different variables or coordinates along
    the same dimensions. When merging, xarray aligns the variables and coordinates of the different datasets along
    the specified dimensions and creates a new ``Dataset`` containing all the variables and coordinates.

.. ipython:: python

    # create two 1D arrays with names
    arr1 = xr.DataArray([1, 2, 3], dims=["x"], coords={"x": [10, 20, 30]}, name="arr1")
    arr2 = xr.DataArray([4, 5, 6], dims=["x"], coords={"x": [20, 30, 40]}, name="arr2")

    # merge the two arrays into a new dataset
    merged_ds = xr.Dataset({"arr1": arr1, "arr2": arr2})

    # print the merged dataset
    print(merged_ds)

Both arrays 'arr1' and 'arr2' have one dimension 'x', which has three coordinate values each.
This code creates a new ``dataset`` 'merged_ds' by merging the two arrays 'arr1' and 'arr2'.
The ``merge()`` function allows you to combine multiple ``DataArray`` or ``Dataset`` objects into a single ``Dataset``
along one or more shared dimensions. If the input objects have different values for the same coordinate,
``merge()`` will create a new coordinate with the union of the values from the input objects.

Concatenating
    Concatenating is used to combine two or more Datasets or DataArrays along a new dimension. When concatenating,
    xarray stacks the datasets or dataarrays along a new dimension, and the resulting ``Dataset`` or ``Dataarray``
    will have the same variables and coordinates along the other dimensions.

.. ipython:: python

    a = xr.DataArray([[1, 2], [3, 4]], dims=("x", "y"))
    b = xr.DataArray([[5, 6], [7, 8]], dims=("x", "y"))
    c = xr.concat([a, b], dim="c")
    print(c)

This code creates two 2D arrays 'a' and 'b'. Both arrays have two dimensions "x" and "y", and contain the numbers 1 to 4
and 5 to 8, respectively.
This code concatenates the two arrays 'a' and 'b' along a new dimension "c" using the ``xr.concat()`` function. The resulting
array 'c' has three dimensions "c", "x", and "y", and contains the numbers 1 to 8 arranged in two 2D arrays.

Combining
    Combining in xarray is a general term used to describe the process of combining two or more DataArrays or Datasets
    into a single ``DataArray`` or ``Dataset``. This can include both merging and concatenating, as well as other operations
    like arithmetic operations (e.g., adding two arrays together) or stacking (e.g., stacking two arrays along a new
    dimension).

.. ipython:: python

    # create the first dataset
    ds1 = xr.Dataset(
        {"data": xr.DataArray([[1, 2], [3, 4]], dims=("x", "y"))},
        coords={"x": [1, 2], "y": [3, 4]},
    )

    # create the second dataset
    ds2 = xr.Dataset(
        {"data": xr.DataArray([[5, 6], [7, 8]], dims=("x", "y"))},
        coords={"x": [2, 3], "y": [4, 5]},
    )

    # combine the datasets
    combined_ds = xr.combine_by_coords([ds1, ds2])

    # print the combined dataset
    print(combined_ds)

This code creates two datasets, ds1 and ds2, each containing a 2D array of data with dimensions 'x' and 'y' and
corresponding coordinate arrays. The datasets have overlapping coordinates on dimension 'x', with values [1, 2] in
ds1 and [2, 3] in ds2, but no overlapping coordinates on dimension 'y'.

The ``xr.combine_by_coords`` function is then used to combine the datasets by their coordinates. This function combines
datasets with non-overlapping dimensions and concatenates arrays along overlapping dimensions. In this case, it will
concatenate the data arrays along dimension 'x' and create a new coordinate array with values [1, 2, 3]. The resulting
combined dataset 'combined_ds' will have dimensions 'x' and 'y'.
