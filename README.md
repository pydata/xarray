# scidata: objects for working with scientific data in Python

**scidata** is a Python package for working with aligned sets of homogeneous,
n-dimensional arrays. It implements flexible array operations and dataset
manipulation for in-memory datasets within the [Common Data Model][cdm] widely
used for self-describing scientific data (netCDF, OpenDAP, etc.).

## Main Feaures

  - A `DataView` object that is compatible with NumPy's ndarray and ufuncs
    but keeps ancilliary variables and metadata intact.
  - Array broadcasting based on dimension names and coordinate indices
    instead of only shapes.
  - Aggregate variables across dimensions or grouped by other variables.
  - Fast label-based indexing and time-series functionality built on
    [pandas][pandas].

## Design Goals

  - Provide a data analysis toolkit as fast and powerful as pandas but
    designed for working with datasets of aligned, homogeneous N-dimensional
    arrays.
  - Whenever possible, build on top of and interoperate with pandas and the
    rest of the awesome [scientific python stack][scipy].
  - Provide a uniform API for loading and saving scientific data in a variety
    of formats (including streaming data).
  - Use metadata according to [conventions][cf] when appropriate, but don't
    strictly enforce them. Conflicting attributes (e.g., units) should be
    silently dropped instead of causing errors. The onus is on the user to
    make sure that operations make sense.

## Prior Art

  - [Iris][iris] is an awesome package for working with meteorological data with
    unfortunately complex data-structures and strict enforcement of metadata
    conventions. Scidata's `DataView` is largely based on the Iris `Cube`.
  - [netCDF4-python][nc4] provides scidata's primary interface for working with
    netCDF and OpenDAP datasets.
  - [pandas][pandas] is fast and powerful but oriented around working with
    tabular datasets. pandas has experimental N-dimensional panels, but they
    don't support aligned math with other objects. We believe the `DataView`/
    `Cube` model is better suited to working with scientific datasets.

[pandas]: http://pandas.pydata.org/
[cdm]: http://www.unidata.ucar.edu/software/thredds/current/netcdf-java/CDM/
[cf]: http://cf-pcmdi.llnl.gov/documents/cf-conventions/1.6/cf-conventions.html
[scipy]: http://scipy.org/
[nc4]: http://netcdf4-python.googlecode.com/svn/trunk/docs/netCDF4-module.html
[iris]: http://scitools.org.uk/iris/
