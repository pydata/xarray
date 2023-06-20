.. currentmodule:: xarray
.. _numpy-to-xarray:

====================
From numpy to xarray
====================

.. ipython:: python
    :suppress:

    import numpy as np
    import pandas as pd
    import xarray as xr

    np.random.seed(123456)

Xarray data structures wrap numpy arrays (or :ref:`numpy-like arrays<duckarrays>`).
This page is intended for new users of xarray who are used to working directly with numpy arrays.


Differences and similarities
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Similarities:
- Multi-dimensional arrays
- Method-chaining

Differences:
- Named dimensions
- Transpose invariance


Advantages of xarray over pure numpy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Advantages:
- Clearer code
- No magic axis numbers
- Generalizes to higher dimensions
- Simpler expression of complicated operations (e.g. groupby, coarsen)
- Metadata
- Automatic NaN-handling
- I/O

Disadvantages:
- Sometimes speed
- Some tricks not available

Remember you are not locked in to using xarray!
You can always get back the underlying numpy array via :py:meth:`~xarray.DataArray.values`,
and if you want to apply numpy-level functions to your data you can wrap them using :py:func:`xarray.apply_ufunc`.
It's very common to start dipping your toes into using xarray by dropping back down to numpy occasionally,
but once you become more proficient you should find that you rarely if ever actually _need_ to do this.


What's the xarray version of this function?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

There are a large number of numpy functions and methods whose xarray equivalent has an identical name
(e.g. :py:func:`~xarray.broadcast()` or :py:meth:`~xarray.DataArray.transpose()`).
Make sure you search the API page for the name of the function you are after first.

If you can't find the xarray equivalent of a particular numpy function then this handy table might help you:

.. list-table::
   :header-rows: 1
   :widths: 60 60 60

   * - numpy function
     - xarray function
     - Additional reading
   * - ``np.concatenate``, ``np.stack``
     - :py:func:`xarray.concat`
     - :ref:`Concatenating data<concatenate>`
   * - ``np.block``
     - :py:func:`xarray.combine_nested`
     - :ref:`Combining along multiple dimensions<combining.multi>`
   * - ``np.apply_along_axis``
     - :py:func:`xarray.apply_ufunc`
     - :ref:`Wrapping custom computation<comput.wrapping-custom>`
   * - ``np.moveaxis``
     - :py:meth:`DataArray.transpose()`
     -

If the numpy function you are after is not available in xarray,
but you think it should be, raise an issue to suggest adding it!


Other things you should know about
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you came from numpy but are only just learning about xarray, there are some other tools you should know about.
The tools listed here provide alternative ways to handle tasks that you may
