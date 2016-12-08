.. _reshape:

###############################
Reshaping and reorganizing data
###############################

These methods allow you to reorganize 

.. ipython:: python
   :suppress:

    import numpy as np
    import pandas as pd
    import xarray as xr
    np.random.seed(123456)

Reordering dimensions
---------------------

To reorder dimensions on a :py:class:`~xarray.DataArray` or across all variables
on a :py:class:`~xarray.Dataset`, use :py:meth:`~xarray.DataArray.transpose` or the
``.T`` property:

.. ipython:: python

    ds = xr.Dataset({'foo': (('x', 'y', 'z'), [[[42]]]), 'bar': (('y', 'z'), [[24]])})
    ds.transpose('y', 'z', 'x')
    ds.T

Converting between datasets and arrays
--------------------------------------

To convert from a Dataset to a DataArray, use :py:meth:`~xarray.Dataset.to_array`:

.. ipython:: python

    arr = ds.to_array()
    arr

This method broadcasts all data variables in the dataset against each other,
then concatenates them along a new dimension into a new array while preserving
coordinates.

To convert back from a DataArray to a Dataset, use
:py:meth:`~xarray.DataArray.to_dataset`:

.. ipython:: python

    arr.to_dataset(dim='variable')

The broadcasting behavior of ``to_array`` means that the resulting array
includes the union of data variable dimensions:

.. ipython:: python

    ds2 = xr.Dataset({'a': 0, 'b': ('x', [3, 4, 5])})

    # the input dataset has 4 elements
    ds2

    # the resulting array has 6 elements
    ds2.to_array()

Otherwise, the result could not be represented as an orthogonal array.

If you use ``to_dataset`` without supplying the ``dim`` argument, the DataArray will be converted into a Dataset of one variable:

.. ipython:: python

    arr.to_dataset(name='combined')

.. _reshape.stack:

Stack and unstack
-----------------

As part of xarray's nascent support for :py:class:`pandas.MultiIndex`, we have
implemented :py:meth:`~xarray.DataArray.stack` and
:py:meth:`~xarray.DataArray.unstack` method, for combining or splitting dimensions:

.. ipython:: python

    array = xr.DataArray(np.random.randn(2, 3),
                         coords=[('x', ['a', 'b']), ('y', [0, 1, 2])])
    stacked = array.stack(z=('x', 'y'))
    stacked
    stacked.unstack('z')

These methods are modeled on the :py:class:`pandas.DataFrame` methods of the
same name, although in xarray they always create new dimensions rather than
adding to the existing index or columns.

Like :py:meth:`DataFrame.unstack<pandas.DataFrame.unstack>`, xarray's ``unstack``
always succeeds, even if the multi-index being unstacked does not contain all
possible levels. Missing levels are filled in with ``NaN`` in the resulting object:

.. ipython:: python
    
    stacked2 = stacked[::2]
    stacked2    
    stacked2.unstack('z')

However, xarray's ``stack`` has an important difference from pandas: unlike
pandas, it does not automatically drop missing values. Compare:

.. ipython:: python
    
    array = xr.DataArray([[np.nan, 1], [2, 3]], dims=['x', 'y'])
    array.stack(z=('x', 'y'))    
    array.to_pandas().stack()

We departed from pandas's behavior here because predictable shapes for new
array dimensions is necessary for :ref:`dask`.

Shift and roll
--------------

To adjust coordinate labels, you can use the :py:meth:`~xarray.Dataset.shift` and
:py:meth:`~xarray.Dataset.roll` methods:

.. ipython:: python

	array = xr.DataArray([1, 2, 3, 4], dims='x')
	array.shift(x=2)
	array.roll(x=2)
