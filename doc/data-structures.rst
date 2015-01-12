.. _data structures:

Data Structures
===============

.. ipython:: python
   :suppress:

    import numpy as np
    import pandas as pd
    import xray
    np.random.seed(123456)
    np.set_printoptions(threshold=10)

DataArray
---------

:py:class:`xray.DataArray` is xray's implementation of a labeled,
multi-dimensional array. It has several key properties:

- ``values``: a :py:class:`numpy.ndarray` holding the array's values
- ``dims``: dimension names for each axis (e.g., ``('x', 'y', 'z')``)
- ``coords``: a dict-like container of arrays (*coordinates*) that label each
  point (e.g., 1-dimensional arrays of numbers, datetime objects or
  strings)
- ``attrs``: an ``OrderedDict`` to hold arbitrary metadata (*attributes*)

xray uses ``dims`` and ``coords`` to enable its core metadata aware operations.
Dimensions provide names that xray uses instead of the ``axis`` argument found
in many numpy functions. Coordinates enable fast label based indexing and
alignment, building on the functionality of the ``index`` found on a pandas
:py:class:`~pandas.DataFrame` or :py:class:`~pandas.Series`.

DataArray objects also can have a ``name`` and can hold arbitrary metadata in
the form of their ``attrs`` property (an ordered dictionary). Names and
attributes are strictly for users and user-written code: xray makes no attempt
to interpret them, and propagates them only in unambiguous cases (see FAQ,
:ref:`approach to metadata`).

.. _creating a dataarray:

Creating a DataArray
~~~~~~~~~~~~~~~~~~~~

The :py:class:`~xray.DataArray` constructor takes a multi-dimensional array of
values (e.g., a numpy ndarray), a list or dictionary of coordinates label and
a list of dimension names:

.. ipython:: python

    data = np.random.rand(4, 3)
    locs = ['IA', 'IL', 'IN']
    times = pd.date_range('2000-01-01', periods=4)
    foo = xray.DataArray(data, coords=[times, locs], dims=['time', 'space'])
    foo

All of these arguments (except for ``data``) are optional, and will be filled
in with default values:

.. ipython:: python

    xray.DataArray(data)

As you can see, dimensions and coordinate arrays corresponding to each
dimension are always present. This behavior is similar to pandas, which fills
in index values in the same way.

The data array constructor also supports supplying ``coords`` as a list of
``(dim, ticks[, attrs])`` pairs with length equal to the number of dimensions:

.. ipython:: python

    xray.DataArray(data, coords=[('time', times), ('space', locs)])

Yet another option is to supply ``coords`` in the form of a dictionary where
the values are scaler values, 1D arrays or tuples (in the same form as the
`dataarray constructor`_). This form lets you supply other coordinates than
those corresponding to dimensions (more on these later):

.. ipython:: python

    xray.DataArray(data, coords={'time': times, 'space': locs, 'const': 42,
                                 'ranking': ('space', [1, 2, 3])},
                   dims=['time', 'space'])

You can also create a ``DataArray`` by supplying a pandas
:py:class:`~pandas.Series`, :py:class:`~pandas.DataFrame` or
:py:class:`~pandas.Panel`, in which case any non-specified arguments in the
``DataArray`` constructor will be filled in from the pandas object:

.. ipython:: python

    df = pd.DataFrame({'x': [0, 1], 'y': [2, 3]}, index=['a', 'b'])
    df.index.name = 'abc'
    df.columns.name = 'xyz'
    df
    xray.DataArray(df)

xray does not (yet!) support labeling coordinate values with a
:py:class:`pandas.MultiIndex` (see :issue:`164`).
However, the alternate ``from_series`` constructor will automatically unpack
any hierarchical indexes it encounters by expanding the series into a
multi-dimensional array, as described in :doc:`pandas`.

DataArray properties
~~~~~~~~~~~~~~~~~~~~

Let's take a look at the important properties on our array:

.. ipython:: python

    foo.values
    foo.dims
    foo.coords
    foo.attrs
    print(foo.name)

You can even modify ``values`` inplace:

.. ipython:: python

   foo.values = 1.0 * foo.values

.. note::

    The array values in a :py:class:`~xray.DataArray` have a single
    (homogeneous) data type. To work with heterogeneous or structured data
    types in xray, use coordinates, or put separate ``DataArray`` objects in a
    single :py:class:`~xray.Dataset`.

Now fill in some of that missing metadata:

.. ipython:: python

    foo.name = 'foo'
    foo.attrs['units'] = 'meters'
    foo

The :py:meth:`~xray.DataArray.rename` method is another option, returning a
new data array:

.. ipython:: python

   foo.rename('bar')

DataArray Coordinates
~~~~~~~~~~~~~~~~~~~~~

The ``coords`` property is ``dict`` like. Individual coordinates can be
accessed from the coordinates by name, or even by indexing the data array
itself:

.. ipython:: python

    foo.coords['time']
    foo['time']

These are also :py:class:`~xray.DataArray` objects, which contain tick-labels
for each dimension.

Coordinates can also be set or removed by using the dictionary like syntax:

.. ipython:: python

    foo['ranking'] = ('space', [1, 2, 3])
    foo.coords
    del foo['ranking']
    foo.coords

xray also supports a notion of "virtual" or "derived" coordinates for
`datetime components`__ implemented by pandas, including "year", "month",
"day", "hour", "minute", "second", "dayofyear", "week", "dayofweek", "weekday"
and "quarter":

__ http://pandas.pydata.org/pandas-docs/stable/api.html#time-date-components

.. ipython:: python

    foo['time.month']
    foo['time.dayofyear']


Dataset
-------

:py:class:`xray.Dataset` is xray's multi-dimensional equivalent of a
:py:class:`~pandas.DataFrame`. It is a dict-like
container of labeled arrays (:py:class:`~xray.DataArray` objects) with aligned
dimensions. It is designed as an in-memory representation of the data model
from the `netCDF`__ file format.

__ http://www.unidata.ucar.edu/software/netcdf/

In addition to the dict-like interface of the dataset itself, which can be used
to access any array in a dataset, datasets have four key properties:

- ``dims``: a dictionary mapping from dimension names to the fixed length of
  each dimension (e.g., ``{'x': 6, 'y': 6, 'time': 8}``)
- ``vars``: a dict-like container of arrays (`variables`)
- ``coords``: another dict-like container of arrays intended to label points
  used in ``vars`` (e.g., 1-dimensional arrays of numbers, datetime objects or
  strings)
- ``attrs``: an ``OrderedDict`` to hold arbitrary metadata

The distinction between whether an array falls in variables or coordinates is
**mostly semantic**: coordinates are intended for constant/fixed/independent
quantities, unlike the varying/measured/dependent quantities that belong in
variables. Dictionary like access on a dataset will supply arrays found in
either category. However, the distinction does have important implications for
indexing and computation.

Here is an example of how we might structure a dataset for a weather forecast:

.. image:: _static/dataset-diagram.png

In this example, it would be natural to call ``temperature`` and
``precipitation`` "variables" and all the other arrays "coordinates" because
they label the points along the dimensions. (see [1]_ for more background on
this example).

.. _dataarray constructor:

Creating a Dataset
~~~~~~~~~~~~~~~~~~

To make an :py:class:`~xray.Dataset` from scratch, supply dictionaries for any
variables coordinates and attributes you  would like to insert into the
dataset.

For the ``vars`` and ``coords`` arguments, keys should be the name of the
variable or coordinate, and values should be scalars, 1d arrays or tuples of
the form ``(dims, data[, attrs])`` sufficient to label each array:

- ``dims`` should be a sequence of strings.
- ``data`` should be a numpy.ndarray (or array-like object) that has a
  dimensionality equal to the length of ``dims``.
- ``attrs`` is an arbitrary Python dictionary for storing metadata associated
  with a particular array.

Let's create some fake data for the example we show above:

.. ipython:: python

    temp = 15 + 8 * np.random.randn(2, 2, 3)
    precip = 10 * np.random.rand(2, 2, 3)
    lon = [[-99.83, -99.32], [-99.79, -99.23]]
    lat = [[42.25, 42.21], [42.63, 42.59]]

    # for real use cases, its good practice to supply array attributes such as
    # units, but we won't bother here for the sake of brevity
    ds = xray.Dataset({'temperature': (['x', 'y', 'time'],  temp),
                       'precipitation': (['x', 'y', 'time'], precip)},
                      coords={'lon': (['x', 'y'], lon),
                              'lat': (['x', 'y'], lat),
                              'time': pd.date_range('2014-09-06', periods=3),
                              'reference_time': pd.Timestamp('2014-09-05')})
    ds

Notice that we did not explicitly include coordinates for the "x" or "y"
dimensions, so they were filled in array of ascending integers of the proper
length.

We can also pass :py:class:`xray.DataArray` objects as values in the dictionary
instead of tuples:

.. ipython:: python

    xray.Dataset({'bar': foo})

Or directly convert a data array into a dataset:

.. ipython::

    foo.to_dataset(name='bar')

You can also create an dataset from a :py:class:`pandas.DataFrame` with
:py:meth:`Dataset.from_dataframe <xray.Dataset.from_dataframe>` or from a
netCDF file on disk with :py:func:`~xray.open_dataset`. See
:ref:`pandas` and :ref:`io`.

Dataset contents
~~~~~~~~~~~~~~~~

:py:class:`~xray.Dataset` implements the Python dictionary interface, with
values given by :py:class:`xray.DataArray` objects:

.. ipython:: python

    'temperature' in ds

    ds.keys()

    ds['temperature']

The valid keys include each listed coordinate and variable.

Variables and coordinates are also contained separately in the
:py:attr:`~xray.Dataset.vars` and :py:attr:`~xray.Dataset.coords`
dictionary-like attributes:

.. ipython:: python

    ds.vars
    ds.coords

Finally, like data arrays, datasets also store arbitrary metadata in the form
of `attributes`:

.. ipython:: python

    ds.attrs

    ds.attrs['title'] = 'example attribute'
    ds

xray does not enforce any restrictions on attributes, but serialization to
some file formats may fail if you use objects that are not strings, numbers
or :py:class:`numpy.ndarray` objects.

Dictionary like methods
~~~~~~~~~~~~~~~~~~~~~~~

We can update a dataset in-place using Python's standard dictionary syntax. For
example, to create this example dataset from scratch, we could have written:

.. ipython:: python

    ds = xray.Dataset()
    ds['temperature'] = (('x', 'y', 'time'), temp)
    ds['precipitation'] = (('x', 'y', 'time'), precip)
    ds.coords['lat'] = (('x', 'y'), lat)
    ds.coords['lon'] = (('x', 'y'), lon)
    ds.coords['time'] = pd.date_range('2014-09-06', periods=3)
    ds.coords['reference_time'] = pd.Timestamp('2014-09-05')

To change the variables in a ``Dataset``, you can use all the standard dictionary
methods, including ``values``, ``items``, ``__delitem__``, ``get`` and
:py:meth:`~xray.Dataset.update``.

Creating modified modified datasets
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can copy a ``Dataset`` by using the :py:meth:`~xray.Dataset.copy` method:

.. ipython:: python

    ds2 = ds.copy()
    del ds2['time']
    ds2

By default, the copy is shallow, so only the container will be copied: the
arrays in the ``Dataset`` will still be stored in the same underlying
:py:class:`numpy.ndarray` objects. You can copy all data by supplying the
argument ``deep=True``.

You also can select and drop an explicit list of variables by using the
by indexing with a list of names or using the
:py:meth:`~xray.Dataset.drop_vars` methods to return a new ``Dataset``. These
operations keep around coordinates:

.. ipython:: python

    ds[['temperature']]
    ds[['x']]

If a dimension name is given as an argument to `drop_vars`, it also drops all
variables that use that dimension:

.. ipython:: python

    ds.drop_vars('time')

Another useful option is the ability to rename the variables in a dataset:

.. ipython:: python

    ds.rename({'temperature': 'temp', 'precipitation': 'precip'})

.. _coordinates:

Coordinates
-----------

Coordinates are ancillary arrays stored for ``DataArray`` and ``Dataset``
objects in the ``coords`` attribute:

.. ipython:: python

    ds.coords

Unlike attributes, xray *does* interpret and persist coordinates in
operations that transform xray objects.

One dimensional coordinates with a name equal to their sole dimension (marked
by ``*`` when printing a dataset or data array) take on a special meaning in
xray. They are used for label based indexing and alignment,
like the ``index`` found on a pandas :py:class:`~pandas.DataFrame` or
:py:class:`~pandas.Series`. Indeed, these "dimension" coordinates use a
:py:class:`pandas.Index` internally to store their values.

Other than for indexing, xray does not make any direct use of the values
associated with coordinates. Coordinates with names not matching a dimension
are not used for alignment or indexing, nor are they required to match when
doing arithmetic (see :ref:`alignment and coordinates`).

Converting to ``pandas.Index``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To convert a coordinate (or any ``DataArray``) into an actual
:py:class:`pandas.Index`, use the :py:meth:`~xray.DataArray.to_index` method:

.. ipython:: python

    ds['time'].to_index()

A useful shortcut is the ``indexes`` property (on both ``DataArray`` and
``Dataset``), which lazily constructs a dictionary whose keys are given by each
dimension and whose the values are ``Index`` objects:

.. ipython:: python

    ds.indexes

Switching between coordinates and variables
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To entirely add or removing coordinate arrays, you can use dictionary like
syntax, as shown above. To convert back and forth between coordinates and
variables, use the the :py:meth:`~xray.Dataset.set_coords` and
:py:meth:`~xray.Dataset.reset_coords` methods:

.. ipython:: python

    ds.reset_coords()
    ds.set_coords(['temperature', 'precipitation'])
    ds['temperature'].reset_coords(drop=True)

Notice that these operations skip coordinates with names given by dimensions,
as used for indexing. This mostly because we are not entirely sure how to
design the interface around the fact that xray cannot store a coordinate and
variable with the name but different values in the same dictionary. But we do
recognize that supporting something like this would be useful.

Converting into datasets
~~~~~~~~~~~~~~~~~~~~~~~~

``Coordinates`` objects also have a few useful methods, mostly for converting
them into dataset objects:

.. ipython:: python

    ds.coords.to_dataset()

The merge method is particularly interesting, because it implements the same
logic used for merging coordinates in arithmetic operations
(see :ref:`comput`):

.. ipython:: python

    alt = xray.Dataset(coords={'z': [10], 'lat': 0, 'lon': 0})
    ds.coords.merge(alt.coords)

The ``coords.merge`` method may be useful if you want to implement your own
binary operations that act on xray objects. In the future, we hope to write
more helper functions so that you can easily make your functions act like
xray's built-in arithmetic.


.. [1] Latitude and longitude are 2D arrays because the dataset uses
   `projected coordinates`__. ``reference_time`` refers to the reference time
   at which the forecast was made, rather than ``time`` which is the valid time
   for which the forecast applies.

__ http://en.wikipedia.org/wiki/Map_projection

