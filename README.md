# xray: extended arrays for working with scientific datasets in Python

**xray** is a Python package for working with aligned sets of homogeneous,
n-dimensional arrays. It implements flexible array operations and dataset
manipulation for in-memory datasets within the [Common Data Model][cdm] widely
used for self-describing scientific data (netCDF, OpenDAP, etc.).

***Warning: xray is still in its early development phase. Expect the API to
change.***

## Main Feaures

  - Extended array objects (`XArray` and `DatasetArray`) that are compatible
    with NumPy's ndarray and ufuncs but that keep ancilliary variables and
    metadata intact.
  - Flexible array broadcasting based on dimension names and coordinate indices.
  - Lazily load arrays from netCDF files on disk or OpenDAP URLs.
  - Flexible split-apply-combine functionality with the array `groupby` method
    (patterned after [pandas][pandas]).
  - Fast label-based indexing and (limited) time-series functionality built on
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

  - [Iris][iris] (supported by the UK Met office) is a similar package
    designed for working with geophysical datasets in Python. Iris provided
    much of the inspiration for xray (e.g., xray's `DatasetArray` is largely
    based on the Iris `Cube`), but it has several limitations that led us to
    build xray instead of extending Iris:
    1. Iris has essentially one first-class object (the `Cube`) on which it
       attempts to build all functionality (`Coord` supports a much more
       limited set of functionality). xray has its equivalent of the Cube
       (the `DatasetArray` object), but it is only a thin wrapper on the more
       primitive building blocks of Dataset and Array objects.
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
  - [pandas][pandas] is fast and powerful but oriented around working with
    tabular datasets. pandas has experimental N-dimensional panels, but they
    don't support aligned math with other objects. We believe the
    `DatasetArray`/ `Cube` model is better suited to working with scientific
    datasets. We use pandas internally in xray to support fast indexing.
  - [netCDF4-python][nc4] provides xray's primary interface for working with
    netCDF and OpenDAP datasets.

[pandas]: http://pandas.pydata.org/
[cdm]: http://www.unidata.ucar.edu/software/thredds/current/netcdf-java/CDM/
[cf]: http://cf-pcmdi.llnl.gov/documents/cf-conventions/1.6/cf-conventions.html
[scipy]: http://scipy.org/
[nc4]: http://netcdf4-python.googlecode.com/svn/trunk/docs/netCDF4-module.html
[iris]: http://scitools.org.uk/iris/
