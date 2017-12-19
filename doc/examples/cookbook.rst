.. _examples.cookbook:

Cookbook
========

This is a repository for short and sweet examples and links for useful xarray recipes. We encourage users to add to this documentation.

Adding interesting links and/or inline examples to this section is a great First Pull Request.

Pandas (pd), Numpy (np) and xarray (xr) are the only abbreviated imported modules. The rest are kept explicitly imported for newer users.


.. ipython:: python
   :suppress:

    import numpy as np
    import pandas as pd
    import xarray as xr

Example recipe
--------------

Flip array dimension
....................

It's very easy to flip a ``DataArray`` along a named dimension, for example to reverse a decreasing coordinate, by indexing the ``dims`` attribute:

.. ipython:: python

   da = xr.DataArray(np.random.rand(4, 5), dims=['x', 'y'],
                     coords=dict(x=[40, 30, 20, 10],
                                 y=pd.date_range('2000-01-01', periods=5)))

   da

   np.flip(da, da.dims.index('x'))
