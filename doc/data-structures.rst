Data structures
===============

xray's core data structures are the :py:class:`~xray.Dataset`,
the :py:class:`~xray.Variable` (including its subclass
:py:class:`~xray.Coordinate`) and the :py:class:`~xray.DataArray`.

The document is intended as a technical summary of the xray data model. It
should be mostly of interest to advanced users interested in extending or
contributing to xray internals.

Dataset
-------

:py:class:`~xray.Dataset` is a Python object representing a fully self-
described dataset of labeled N-dimensional arrays. It consists of:

1. **variables**: A dictionary of Variable objects.
2. **dimensions**: A dictionary giving the lengths of shared dimensions, which
   are required to be consistent across all variables in a Dataset.
3. **attributes**: An ordered dictionary of metadata.

The design of the Dataset is based by the
`NetCDF <http://www.unidata.ucar.edu/software/netcdf/>`__ file format for
self-described scientific data. This is a data model that has become very
successful and widely used in the geosciences.

The Dataset is an intelligent container. It allows for simultaneous integer
or label based indexing of all of its variables, supports split-apply-combine
operations with groupby, and can be converted to and from
:py:class:`pandas.DataFrame` objects.

Variable
--------

:py:class:`~xray.Variable` implements xray's basic extended array object. It
supports the numpy ndarray interface, but is extended to support and use
basic metadata (not including coordinate values). It consists of:

1. **dimensions**: A tuple of dimension names.
2. **data**: The N-dimensional array (for example, of type
   :py:class:`numpy.ndarray`) storing the array's data. It must have the same
   number of dimensions as the length of the "dimensions" attribute.
3. **attributes**: An ordered dictionary of additional metadata to associate
   with this array.

The main functional difference between Variables and numpy arrays is that
numerical operations on Variables implement array broadcasting by dimension
name. For example, adding an Variable with dimensions `('time',)` to another
Variable with dimensions `('space',)` results in a new Variable with dimensions
`('time', 'space')`. Furthermore, numpy reduce operations like ``mean`` or
``sum`` are overwritten to take a "dimension" argument instead of an "axis".

Variables are light-weight objects used as the building block for datasets.
**However, manipulating data in the form of a Dataset or DataArray should
almost always be preferred** (see below), because they can use more complete
metadata in context of coordinate labels.

DataArray
---------

A :py:class:`~xray.DataArray` object is a multi-dimensional array with labeled
dimensions and coordinates. Coordinate labels give it additional power over the
Variable object, so it should be preferred for all high-level use.

Under the covers, DataArrays are simply pointers to a dataset (the ``dataset``
attribute) and the name of a variable in the dataset (the ``name`` attribute),
which indicates to which variable array operations should be applied.

DataArray objects implement the broadcasting rules of Variable objects, but
also use and maintain coordinates (aka "indices"). This means you can do
intelligent (and fast!) label based indexing on DataArrays (via the
``.loc`` attribute), do flexibly split-apply-combine operations with
``groupby`` and convert them to or from :py:class:`pandas.Series` objects.
