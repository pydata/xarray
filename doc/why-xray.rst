Why xray?
=========

Adding dimensions names and coordinate indexes to numpy's ndarray_ makes many
powerful array operations possible:

-  Apply operations over dimensions by name: ``x.sum('time')``.
-  Select values by label instead of integer location:
   ``x.loc['2014-01-01']`` or ``x.sel(time='2014-01-01')``.
-  Mathematical operations (e.g., ``x - y``) vectorize across multiple
   dimensions (known in numpy as "broadcasting") based on dimension
   names, not array shape.
-  Flexible split-apply-combine operations with groupby:
   ``x.groupby('time.dayofyear').mean()``.
-  Database like aligment based on coordinate labels that smoothly
   handles missing values: ``x, y = xray.align(x, y, join='outer')``.
-  Keep track of arbitrary metadata in the form of a Python dictionary:
   ``x.attrs``.

pandas_ excels at working with tabular data. That suffices for many statistical
analyses, but physical scientists rely on N-dimensional arrays -- which is
where **xray** comes in.

**xray** aims to provide a data analysis toolkit as powerful as pandas_ but
designed for working with homogeneous N-dimensional arrays
instead of tabular data. When possible, we copy the pandas API and rely on
pandas's highly optimized internals (in particular, for fast indexing).

Because **xray** implements the same data model as the NetCDF file
format, xray datasets have a natural and portable serialization format.
But it's also easy to robustly convert an xray ``DataArray`` to and from
a numpy ``ndarray`` or a pandas ``DataFrame`` or ``Series``, providing
compatibility with the full `PyData ecosystem <http://pydata.org/>`__.

.. _ndarray: http://docs.scipy.org/doc/numpy/reference/arrays.ndarray.html
.. _pandas: http://pandas.pydata.org

.. warning::

    xray is still very new – it is only a few months old. Although we will make
    a best effort to maintain compatibility with the current API, it is likely
    that the API will change in future versions as xray matures. Some changes
    are already anticipated, as called out in the `Tutorial <tutorial>`_.
