Data structures
===============

``xray``'s core data structures are the ``Dataset``, ``Variable`` and
``DataArray``.

Dataset
-------

``Dataset`` is netcdf-like object consisting of **variables** (a dictionary of
Variable objects) and **attributes** (an ordered dictionary) which together
form a self-describing data set.

Variable
--------

``Variable`` implements **xray's** basic extended array object. It supports the
numpy ndarray interface, but is extended to support and use metadata. It
consists of:

1. **dimensions**: A tuple of dimension names.
2. **data**: The n-dimensional array (typically, of type ``numpy.ndarray``)
   storing the array's data. It must have the same number of dimensions as the
   length of the "dimensions" attribute.
3. **attributes**: An ordered dictionary of additional metadata to associate
   with this array.

The main functional difference between Variables and numpy.ndarrays is that
numerical operations on Variables implement array broadcasting by dimension
name. For example, adding an Variable with dimensions `('time',)` to another
Variable with dimensions `('space',)` results in a new Variable with dimensions
`('time', 'space')`. Furthermore, numpy reduce operations like ``mean`` or
``sum`` are overwritten to take a "dimension" argument instead of an "axis".

Variables are light-weight objects used as the building block for datasets.
However, usually manipulating data in the form of a DataArray should be
preferred (see below), because they can use more complete metadata in the full
of other dataset variables.

DataArray
---------

``DataArray`` is a flexible hybrid of Dataset and Variable that attempts to
provide the best of both in a single object. Under the covers, DataArrays
are simply pointers to a dataset (the ``dataset`` attribute) and the name of a
"focus variable" in the dataset (the ``focus`` attribute), which indicates to
which variable array operations should be applied.

DataArray objects implement the broadcasting rules of Variable objects, but
also use and maintain coordinates (aka "indices"). This means you can do
intelligent (and fast!) label based indexing on DataArrays (via the
``.loc`` attribute), do flexibly split-apply-combine operations with
``groupby`` and also easily export them to ``pandas.DataFrame`` or
``pandas.Series`` objects.