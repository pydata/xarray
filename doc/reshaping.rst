Reshaping and reorganizing data
===============================

.. ipython:: python
   :suppress:

    import numpy as np
    import pandas as pd
    import xray
    np.random.seed(123456)
    np.set_printoptions(threshold=10)

We'll return to our example dataset from :ref:`data structures`:

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

Converting between Dataset and DataArray
----------------------------------------

To convert from a Dataset to a DataArray, use :py:meth:`~xray.Dataset.to_array`:

.. ipython:: python

    arr = ds.to_array()
    arr

This method broadcasts all data variables in the dataset against each other,
then concatenates them along a new dimension into a new array while preserving
coordinates.

To convert back from a DataArray to a Dataset, use
:py:meth:`~xray.Dataset.to_dataset`:

.. ipython:: python

    arr.to_dataset(dim='variable')

.. note::

    The broadcasting behavior of ``to_array`` means that the resulting array
    includes the union of data variable dimensions:

    .. ipython:: python

        ds2 = xray.Dataset({'a': 0, 'b': ('x', [3, 4, 5])})

        # the input dataset has 4 elements
        ds2

        # the resulting array has 6 elements
        ds2.to_array()

    Otherwise, the result could not be represented as an orthogonal array.

If you use ``to_dataset`` without supplying the ``dim`` argument, the DataArray will be converted into a Dataset of one variable:

.. ipython:: python

    arr.to_dataset(name='combined')

Coordinate variables
--------------------

To entirely add or removing coordinate arrays, you can use dictionary like
syntax, as shown in . To convert back and forth between data and
coordinates, use the the :py:meth:`~xray.Dataset.set_coords` and
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

