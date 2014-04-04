# xray: extended arrays for working with scientific datasets in Python

[![travis-ci build status](https://travis-ci.org/akleeman/xray.png)][travis]

**xray** is a Python package for working with aligned sets of homogeneous,
n-dimensional arrays. It implements flexible array operations and dataset
manipulation for in-memory datasets within the [Common Data Model][cdm] widely
used for self-describing scientific data (e.g., the NetCDF file format).

[travis]: https://travis-ci.org/akleeman/xray
[cdm]: http://www.unidata.ucar.edu/software/thredds/current/netcdf-java/CDM/

## Why xray?

Adding dimensions names and coordinate values to numpy's [ndarray][ndarray]
makes many powerful array operations easy:

  - Apply operations over dimensions by name: `x.sum('time')`.
  - Select values by label instead of integer location: `x.loc['2014-01-01']`
    or `x.labeled_by(time='2014-01-01')`.
  - Mathematical operations (e.g., `x - y`) vectorize across multiple
    dimensions (known in numpy as "broadcasting") based on dimension names,
    regardless of their original order.
  - Flexible split-apply-combine operations with groupby:
    `x.groupby('time.dayofyear').apply(lambda y: y - y.mean())`.
  - Database like aligment based on coordinate labels that smoothly
    handles missing values: `x, y = xray.align(x, y, join='outer')`.
  - Keep track of arbitrary metadata in the form of a Python dictionary:
    `x.attributes`.

**xray** aims to provide a data analysis toolkit as powerful as
[pandas][pandas] but designed for working with homogeneous N-dimensional
arrays instead of tabular data. Indeed, much of its design and internal
functionality (in particular, fast indexing) is shamelessly borrowed from
pandas.

Because **xray** implements the same data model as the NetCDF file format,
xray datasets have a natural and portable serialization format. But it's
also easy to robustly convert an xray `DataArray` to and from a numpy
`ndarray` or a pandas `DataFrame` or `Series`, providing compatibility with
the full [scientific-python ecosystem][scipy].

[pandas]: http://pandas.pydata.org/
[scipy]: http://scipy.org/
[ndarray]: http://docs.scipy.org/doc/numpy/reference/arrays.ndarray.html

## Why not pandas?

[pandas][pandas], thanks to its unrivaled speed and flexibility, has emerged
as the premier python package for working with labeled arrays. So why are we
contributing to further [fragmentation][fragmentation] in the ecosystem for
working with data arrays in Python?

**xray** provides two data-structures that are missing in pandas:

  1. An extended array object (with labels) that is truly n-dimensional.
  2. A dataset object for holding a collection of these extended arrays
     aligned along shared coordinates.

Sometimes, we really want to work with collections of higher dimensional array
(`ndim > 2`), or arrays for which the order of dimensions (e.g., columns vs
rows) shouldn't really matter. This is particularly common when working with
climate and weather data, which is often natively expressed in 4 or more
dimensions.

The use of datasets, which allow for simultaneous manipulation and indexing of
many varibles, actually handles most of the use-cases for heterogeneously
typed arrays. For example, if you want to keep track of latitude and longitude
coordinates (numbers) as well as place names (strings) along your "location"
dimension, you can simply toss both arrays into your dataset.

This is a proven data model: the netCDF format has been around
[for decades][netcdf-background].

Pandas does support [N-dimensional panels][ndpanel], but the implementation
is very limited:

  - You need to create a new factory type for each dimensionality.
  - You can't do math between NDPanels with different dimensionality.
  - Each dimension in a NDPanel has a name (e.g., 'labels', 'items',
    'major_axis', etc.) but the dimension names refer to order, not their
    meaning. You can't specify an operation as to be applied along the "time"
    axis.

Fundamentally, the N-dimensional panel is limited by its context in the pandas
data model, which treats 2D `DataFrame`s as collections of 1D `Series`, 3D
`Panel`s as a collection of  2D `DataFrame`s, and so on. Quite simply, we
think the [Common Data Model][cdm] implemented in xray is better suited for
working with many scientific datasets.

[fragmentation]: http://wesmckinney.com/blog/?p=77
[netcdf-background]: http://www.unidata.ucar.edu/software/netcdf/docs/background.html
[ndpanel]: http://pandas.pydata.org/pandas-docs/stable/dsintro.html#panelnd-experimental

## Why not Iris?

[Iris][iris] (supported by the UK Met office) is a similar package designed
for working with weather data in Python. Iris provided much of the inspiration
for xray (xray's `DataArray` is largely based on the Iris `Cube`), but it has
several limitations that led us to build xray instead of extending Iris:

  1. Iris has essentially one first-class object (the `Cube`) on which it
     attempts to build all functionality (`Coord` supports a much more
     limited set of functionality). xray has its equivalent of the Cube
     (the `DataArray` object), but under the hood it is only thin wrapper
     on the more primitive building blocks of Dataset and XArray objects.
  2. Iris has a strict interpretation of [CF conventions][cf], which,
     although a principled choice, we have found to be impractical for
     everyday uses. With Iris, every quantity has physical (SI) units, all
     coordinates have cell-bounds, and all metadata (units, cell-bounds and
     other attributes) is required to match before merging or doing
     operations with on multiple cubes. This means that a lot of time with
     Iris is spent figuring out why cubes are incompatible and explicitly
     removing possibly conflicting metadata.
  3. Iris can be slow and complex. Strictly interpretting metadata requires
     a lot of work and (in our experience) can be difficult to build mental
     models of how Iris functions work. Moreover, it means that a lot of
     logic (e.g., constraint handling) uses non-vectorized operations. For
     example, extracting all times within a range can be surprisingly slow
     (e.g., 0.3 seconds vs 3 milliseconds in xray to select along a time
     dimension with 10000 elements).

[iris]: http://scitools.org.uk/iris/
[cf]: http://cf-pcmdi.llnl.gov/documents/cf-conventions/1.6/cf-conventions.html

## Other prior art

[netCDF4-python][nc4] provides a low level interface for working with
NetCDF and OpenDAP datasets in Python. We use netCDF4-python internally in
xray, and have contributed a number of improvements and fixes upstream.

[larry][larry] is another package that implements labeled arrays for
Python built on top of numpy that provided some guidance for xray.

[nc4]: https://github.com/Unidata/netcdf4-python
[larry]: https://pypi.python.org/pypi/la

## Broader design goals

  - Whenever possible, build on top of and interoperate with pandas and the
    rest of the awesome [scientific python stack][scipy].
  - Be fast. There shouldn't be a significant overhead for metadata aware
    manipulation of n-dimensional arrays, as long as the arrays are large
    enough. The goal is to be as fast as pandas or raw numpy.
  - Provide a uniform API for loading and saving scientific data in a variety
    of formats (including streaming data).
  - Understand metadata according to [Climate and Forecast Conventions][cf]
    when appropriate, but don't strictly enforce them. Conflicting attributes
    (e.g., units) should be silently dropped instead of causing errors. The
    onus is on the user to make sure that operations make sense.

## Getting started

For more details, see the **[full documentation][docs]** (still a work in
progress) or the source code. **xray** is rapidly maturing, but it is still in
its early development phase. ***Expect the API to change.***

xray currently requires Python 2.7 and the python libraries `numpy>=1.8.0`,
`scipy>=0.13`, `pandas>=0.13.1` and `netCDF4>=1.0.6`. The easiest way to get
these installed from scratch is to use [Anaconda][anaconda]. Eventually these
requirements will be relaxed: python libraries (other than numpy and pandas)
will be required only for the relevant functionality, and we will support
Python 3.

It is not yet available on the Python package index (prior to its initial
release). For now, you need to install it from source:

    git clone https://github.com/akleeman/xray.git
    pip install -e xray

Don't forget to `git fetch` regular updates!

[docs]: http://xray.readthedocs.org/
[anaconda]: https://store.continuum.io/cshop/anaconda/

## About xray

xray is an evolution of an internal tool developed at
[The Climate Corporation][tcc], and was written by current and former Climate
Corp researchers Stephan Hoyer, Alex Kleeman and Eugene Brevdo. It is
available under the open source Apache License.

[tcc]: http://climate.com/
