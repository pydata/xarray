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
        axes ``da.dims``, and the name of the ``i``-th dimension is ``da.dims[i]``.
        If an array is created without specifying dimension names, the default dimension
        names will be ``dim_0``, ``dim_1``, and so forth.

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
        as a particular dimension's dimension coordinate array.

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
        marked by an asterisk ``*`` when printing a ``DataArray`` or ``Dataset``.

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

        .. ipython:: python
            :suppress:

            import numpy as np
            import xarray as xr

    Aligning
        Aligning refers to the process of ensuring that two or more DataArrays or Datasets
        have the same dimensions and coordinates, so that they can be combined or compared properly.

        .. ipython:: python

            x = xr.DataArray(
                [[25, 35], [10, 24]],
                dims=("lat", "lon"),
                coords={"lat": [35.0, 40.0], "lon": [100.0, 120.0]},
            )
            y = xr.DataArray(
                [[20, 5], [7, 13]],
                dims=("lat", "lon"),
                coords={"lat": [35.0, 42.0], "lon": [100.0, 120.0]},
            )
            x
            y

    Broadcasting
        A technique that allows operations to be performed on arrays with different shapes and dimensions.
        When performing operations on arrays with different shapes and dimensions, xarray will automatically attempt to broadcast the
        arrays to a common shape before the operation is applied.

        .. ipython:: python

            # 'a' has shape (3,) and 'b' has shape (4,)
            a = xr.DataArray(np.array([1, 2, 3]), dims=["x"])
            b = xr.DataArray(np.array([4, 5, 6, 7]), dims=["y"])

            # 2D array with shape (3, 4)
            a + b

    Merging
        Merging is used to combine two or more Datasets or DataArrays that have different variables or coordinates along
        the same dimensions. When merging, xarray aligns the variables and coordinates of the different datasets along
        the specified dimensions and creates a new ``Dataset`` containing all the variables and coordinates.

        .. ipython:: python

            # create two 1D arrays with names
            arr1 = xr.DataArray(
                [1, 2, 3], dims=["x"], coords={"x": [10, 20, 30]}, name="arr1"
            )
            arr2 = xr.DataArray(
                [4, 5, 6], dims=["x"], coords={"x": [20, 30, 40]}, name="arr2"
            )

            # merge the two arrays into a new dataset
            merged_ds = xr.Dataset({"arr1": arr1, "arr2": arr2})
            merged_ds

    Concatenating
        Concatenating is used to combine two or more Datasets or DataArrays along a dimension. When concatenating,
        xarray arranges the datasets or dataarrays along a new dimension, and the resulting ``Dataset`` or ``Dataarray``
        will have the same variables and coordinates along the other dimensions.

        .. ipython:: python

            a = xr.DataArray([[1, 2], [3, 4]], dims=("x", "y"))
            b = xr.DataArray([[5, 6], [7, 8]], dims=("x", "y"))
            c = xr.concat([a, b], dim="c")
            c

    Combining
        Combining is the process of arranging two or more DataArrays or Datasets into a single ``DataArray`` or
        ``Dataset`` using some combination of merging and concatenation operations.

        .. ipython:: python

            ds1 = xr.Dataset(
                {"data": xr.DataArray([[1, 2], [3, 4]], dims=("x", "y"))},
                coords={"x": [1, 2], "y": [3, 4]},
            )
            ds2 = xr.Dataset(
                {"data": xr.DataArray([[5, 6], [7, 8]], dims=("x", "y"))},
                coords={"x": [2, 3], "y": [4, 5]},
            )

            # combine the datasets
            combined_ds = xr.combine_by_coords([ds1, ds2])
            combined_ds

    lazy
        Lazily-evaluated operations do not load data into memory until necessary. Instead of doing calculations
        right away, xarray lets you plan what calculations you want to do, like finding the
        average temperature in a dataset. This planning is called "lazy evaluation." Later, when
        you're ready to see the final result, you tell xarray, "Okay, go ahead and do those calculations now!"
        That's when xarray starts working through the steps you planned and gives you the answer you wanted. This
        lazy approach helps save time and memory because xarray only does the work when you actually need the
        results.

    labeled
        Labeled data has metadata describing the context of the data, not just the raw data values.
        This contextual information can be labels for array axes (i.e. dimension names) tick labels along axes (stored as Coordinate variables) or unique names for each array. These labels
        provide context and meaning to the data, making it easier to understand and work with. If you have
        temperature data for different cities over time. Using xarray, you can label the dimensions: one for
        cities and another for time.

    serialization
        Serialization is the process of converting your data into a format that makes it easy to save and share.
        When you serialize data in xarray, you're taking all those temperature measurements, along with their
        labels and other information, and turning them into a format that can be stored in a file or sent over
        the internet. xarray objects can be serialized into formats which store the labels alongside the data.
        Some supported serialization formats are files that can then be stored or transferred (e.g. netCDF),
        whilst others are protocols that allow for data access over a network (e.g. Zarr).

    indexing
        :ref:`Indexing` is how you select subsets of your data which you are interested in.

        - Label-based Indexing: Selecting data by passing a specific label and comparing it to the labels
          stored in the associated coordinates. You can use labels to specify what you want like "Give me the
          temperature for New York on July 15th."

        - Positional Indexing: You can use numbers to refer to positions in the data like "Give me the third temperature value" This is useful when you know the order of your data but don't need to remember the exact labels.

        - Slicing: You can take a "slice" of your data, like you might want all temperatures from July 1st
          to July 10th. xarray supports slicing for both positional and label-based indexing.

    DataTree
        A tree-like collection of ``Dataset`` objects. A *tree* is made up of one or more *nodes*,
        each of which can store the same information as a single ``Dataset`` (accessed via ``.dataset``).
        This data is stored in the same way as in a ``Dataset``, i.e. in the form of data
        :term:`variables<Variable>`, :term:`dimensions<Dimension>`, :term:`coordinates<Coordinate>`,
        and attributes.

       The nodes in a tree are linked to one another, and each node is its own instance of
        ``DataTree`` object. Each node can have zero or more *children* (stored in a dictionary-like
        manner under their corresponding *names*), and those child nodes can themselves have
        children. If a node is a child of another node that other node is said to be its *parent*.
        Nodes can have a maximum of one parent, and if a node has no parent it is said to be the
        *root* node of that *tree*.

    Subtree
        A section of a *tree*, consisting of a *node* along with all the child nodes below it
        (and the child nodes below them, i.e. all so-called *descendant* nodes).
        Excludes the parent node and all nodes above.

    Group
        Another word for a subtree, reflecting how the hierarchical structure of a ``DataTree``
        allows for grouping related data together.
        Analogous to a single
        `netCDF group <https://www.unidata.ucar.edu/software/netcdf/workshops/2011/groups-types/GroupsIntro.html>`_
        or `Zarr group <https://zarr.readthedocs.io/en/stable/tutorial.html#groups>`_.
