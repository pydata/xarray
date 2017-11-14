.. currentmodule:: xarray

What's New
==========

.. ipython:: python
   :suppress:

    import numpy as np
    import pandas as pd
    import xarray as xray
    import xarray
    import xarray as xr
    np.random.seed(123456)

.. _whats-new.0.10.0:

v0.10.0 rc2 (13 November 2017)
------------------------------

.. caution::

  You're reached the documentation for the **pre-release version of xarray.**
  Please test out this release candidate and report any issues on GitHub. If
  all goes well, the final v0.10.0 release will come out in about a week. To
  install, use::

    pip install --pre --upgrade --upgrade-strategy=only-if-needed xarray

This is a major release that includes bug fixes, new features and a few
backwards incompatible changes. Highlights include:

- Indexing now supports broadcasting over dimensions, similar to NumPy's
  vectorized indexing (but better!).
- :py:meth:`~DataArray.resample` has a new groupby-like API like pandas.
- :py:func:`~xarray.apply_ufunc` facilitates wrapping and parallelizing
  functions written for NumPy arrays.
- Performance improvements, particularly for dask and :py:func:`open_mfdataset`.

Breaking changes
~~~~~~~~~~~~~~~~

- xarray now supports a form of vectorized indexing with broadcasting, where
  the result of indexing depends on dimensions of indexers,
  e.g., ``array.sel(x=ind)`` with ``ind.dims == ('y',)``. Alignment between
  coordinates on indexed and indexing objects is also now enforced.
  Due to these changes, existing uses of xarray objects to index other xarray
  objects will break in some cases.

  The new indexing API is much more powerful, supporting outer, diagonal and
  vectorized indexing in a single interface.
  The ``isel_points`` and ``sel_points`` methods are deprecated, since they are
  now redundant with the ``isel`` / ``sel`` methods.
  See :ref:`vectorized_indexing` for the details (:issue:`1444`,
  :issue:`1436`).
  By `Keisuke Fujii <https://github.com/fujiisoup>`_ and
  `Stephan Hoyer <https://github.com/shoyer>`_.

- A new resampling interface to match pandas' groupby-like API was added to
  :py:meth:`Dataset.resample` and :py:meth:`DataArray.resample`
  (:issue:`1272`). :ref:`Timeseries resampling <resampling>` is
  fully supported for data  with arbitrary dimensions as is both downsampling
  and upsampling (including linear, quadratic, cubic, and spline interpolation).

  Old syntax:

  .. ipython::
    :verbatim:

    In [1]: ds.resample('24H', dim='time', how='max')
    Out[1]:
    <xarray.Dataset>
    [...]

  New syntax:

  .. ipython::
    :verbatim:

    In [1]: ds.resample(time='24H').max()
    Out[1]:
    <xarray.Dataset>
    [...]

  Note that both versions are currently supported, but using the old syntax will
  produce a warning encouraging users to adopt the new syntax.
  By `Daniel Rothenberg <https://github.com/darothen>`_.

- Calling ``repr()`` or printing xarray objects at the command line or in a
  Jupyter Notebook will not longer automatically compute dask variables or
  load data on arrays lazily loaded from disk (:issue:`1522`).
  By `Guido Imperiale <https://github.com/crusaderky>`_.

- Supplying ``coords`` as a dictionary to the ``DataArray`` constructor without
  also supplying an explicit ``dims`` argument is no longer supported. This
  behavior was deprecated in version 0.9 but will now raise an error
  (:issue:`727`).

- Several existing features have been deprecated and will change to new
  behavior in xarray v0.11. If you use any of them with xarray v0.10, you
  should see a ``FutureWarning`` that describes how to update your code:

  - ``Dataset.T`` has been deprecated an alias for ``Dataset.transpose()``
    (:issue:`1232`). In the next major version of xarray, it will provide short-
    cut lookup for variables or attributes with name ``'T'``.
  - ``DataArray.__contains__`` (e.g., ``key in data_array``) currently checks
    for membership in ``DataArray.coords``. In the next major version of
    xarray, it will check membership in the array data found in
    ``DataArray.values`` instead (:issue:`1267`).
  - Direct iteration over and counting a ``Dataset`` (e.g., ``[k for k in ds]``,
    ``ds.keys()``, ``ds.values()``, ``len(ds)`` and ``if ds``) currently
    includes all variables, both data and coordinates. For improved usability
    and consistency with pandas, in the next major version of xarray these will
    change to only include data variables (:issue:`884`). Use ``ds.variables``,
    ``ds.data_vars`` or ``ds.coords`` as alternatives.

- Changes to minimum versions of dependencies:

  - Old numpy < 1.11 and pandas < 0.18 are no longer supported (:issue:`1512`).
    By `Keisuke Fujii <https://github.com/fujiisoup>`_.
  - The minimum supported version bottleneck has increased to 1.1
    (:issue:`1279`).
    By `Joe Hamman <https://github.com/jhamman>`_.

Enhancements
~~~~~~~~~~~~

**New functions/methods**

- New helper function :py:func:`~xarray.apply_ufunc` for wrapping functions
  written to work on NumPy arrays to support labels on xarray objects
  (:issue:`770`). ``apply_ufunc`` also support automatic parallelization for
  many functions with dask. See :ref:`comput.wrapping-custom` and
  :ref:`dask.automatic-parallelization` for details.
  By `Stephan Hoyer <https://github.com/shoyer>`_.

- Added new method :py:meth:`Dataset.to_dask_dataframe`, convert a dataset into
  a dask dataframe.
  This allows lazy loading of data from a dataset containing dask arrays (:issue:`1462`).
  By `James Munroe <https://github.com/jmunroe>`_.

- New function :py:func:`~xarray.where` for conditionally switching between
  values in xarray objects, like :py:func:`numpy.where`:

  .. ipython::
    :verbatim:

    In [1]: import xarray as xr

    In [2]: arr = xr.DataArray([[1, 2, 3], [4, 5, 6]], dims=('x', 'y'))

    In [3]: xr.where(arr % 2, 'even', 'odd')
    Out[3]:
    <xarray.DataArray (x: 2, y: 3)>
    array([['even', 'odd', 'even'],
           ['odd', 'even', 'odd']],
          dtype='<U4')
    Dimensions without coordinates: x, y

  Equivalently, the :py:meth:`~xarray.Dataset.where` method also now supports
  the ``other`` argument, for filling with a value other than ``NaN``
  (:issue:`576`).
  By `Stephan Hoyer <https://github.com/shoyer>`_.

- Added :py:func:`~xarray.show_versions` function to aid in debugging
  (:issue:`1485`).
  By `Joe Hamman <https://github.com/jhamman>`_.

**Performance improvements**

- :py:func:`~xarray.concat` was computing variables that aren't in memory
  (e.g. dask-based) multiple times; :py:func:`~xarray.open_mfdataset`
  was loading them multiple times from disk. Now, both functions will instead
  load them at most once and, if they do, store them in memory in the
  concatenated array/dataset (:issue:`1521`).
  By `Guido Imperiale <https://github.com/crusaderky>`_.

- Speed-up (x 100) of :py:func:`~xarray.conventions.decode_cf_datetime`.
  By `Christian Chwala <https://github.com/cchwala>`_.

**IO related improvements**

- Unicode strings (``str`` on Python 3) are now round-tripped successfully even
  when written as character arrays (e.g., as netCDF3 files or when using
  ``engine='scipy'``) (:issue:`1638`). This is controlled by the ``_Encoding``
  attribute convention, which is also understood directly by the netCDF4-Python
  interface. See :ref:`io.string-encoding` for full details.
  By `Stephan Hoyer <https://github.com/shoyer>`_.

- Support for ``data_vars`` and ``coords`` keywords from
  :py:func:`~xarray.concat` added to :py:func:`~xarray.open_mfdataset`
  (:issue:`438`). Using these keyword arguments can significantly reduce
  memory usage and increase speed.
  By `Oleksandr Huziy <https://github.com/guziy>`_.

- Support for :py:class:`pathlib.Path` objects added to
  :py:func:`~xarray.open_dataset`, :py:func:`~xarray.open_mfdataset`,
  :py:func:`~xarray.to_netcdf`, and :py:func:`~xarray.save_mfdataset`
  (:issue:`799`):

  .. ipython::
    :verbatim:

    In [2]: from pathlib import Path  # In Python 2, use pathlib2!

    In [3]: data_dir = Path("data/")

    In [4]: one_file = data_dir / "dta_for_month_01.nc"

    In [5]: xr.open_dataset(one_file)
    Out[5]:
    <xarray.Dataset>
    [...]

  By `Willi Rath <https://github.com/willirath>`_.

- You can now explicitly disable any default ``_FillValue`` (``NaN`` for
  floating point values) by passing the enconding ``{'_FillValue': None}``
  (:issue:`1598`).
  By `Stephan Hoyer <https://github.com/shoyer>`_.

- More attributes available in :py:attr:`~xarray.Dataset.attrs` dictionary when
  raster files are opened with :py:func:`~xarray.open_rasterio`.
  By `Greg Brener <https://github.com/gbrener>`_.

- Support for NetCDF files using an ``_Unsigned`` attribute to indicate that a
  a signed integer data type should be interpreted as unsigned bytes
  (:issue:`1444`).
  By `Eric Bruning <https://github.com/deeplycloudy>`_.

- Support using an existing, opened netCDF4 ``Dataset`` with
  :py:class:`~xarray.backends.NetCDF4DataStore`. This permits creating an
  :py:class:`~xarray.Dataset` from a netCDF4 ``Dataset`` that has been opened using
  other means (:issue:`1459`).
  By `Ryan May <https://github.com/dopplershift>`_.

- Changed :py:class:`~xarray.backends.PydapDataStore` to take a Pydap dataset.
  This permits opening Opendap datasets that require authentication, by
  instantiating a Pydap dataset with a session object. Also added
  :py:meth:`xarray.backends.PydapDataStore.open` which takes a url and session
  object (:issue:`1068`).
  By `Philip Graae <https://github.com/mrpgraae>`_.

- Support reading and writing unlimited dimensions with h5netcdf (:issue:`1636`).
  By `Joe Hamman <https://github.com/jhamman>`_.

**Other improvements**

- Added ``_ipython_key_completions_`` to xarray objects, to enable
  autocompletion for dictionary-like access in IPython, e.g.,
  ``ds['tem`` + tab -> ``ds['temperature']`` (:issue:`1628`).
  By `Keisuke Fujii <https://github.com/fujiisoup>`_.

- Support passing keyword arguments to ``load``, ``compute``, and ``persist``
  methods. Any keyword arguments supplied to these methods are passed on to
  the corresponding dask function (:issue:`1523`).
  By `Joe Hamman <https://github.com/jhamman>`_.

- Encoding attributes are now preserved when xarray objects are concatenated.
  The encoding is copied from the first object  (:issue:`1297`).
  By `Joe Hamman <https://github.com/jhamman>`_ and
  `Gerrit Holl <https://github.com/gerritholl>`_.

- Support applying rolling window operations using bottleneck's moving window
  functions on data stored as dask arrays (:issue:`1279`).
  By `Joe Hamman <https://github.com/jhamman>`_.

- Experimental support for the Dask collection interface (:issue:`1674`).
  By `Matthew Rocklin <https://github.com/mrocklin>`_.

Bug fixes
~~~~~~~~~

- Suppress ``RuntimeWarning`` issued by ``numpy`` for "invalid value comparisons"
  (e.g. ``NaN``). Xarray now behaves similarly to Pandas in its treatment of
  binary and unary operations on objects with NaNs (:issue:`1657`).
  By `Joe Hamman <https://github.com/jhamman>`_.

- Unsigned int support for reduce methods with ``skipna=True``
  (:issue:`1562`).
  By `Keisuke Fujii <https://github.com/fujiisoup>`_.

- Fixes to ensure xarray works properly with pandas 0.21:

  - Fix :py:meth:`~xarray.DataArray.isnull` method (:issue:`1549`).
  - :py:meth:`~xarray.DataArray.to_series` and
    :py:meth:`~xarray.Dataset.to_dataframe` should not return a ``pandas.MultiIndex``
    for 1D data (:issue:`1548`).
  - Fix plotting with datetime64 axis labels (:issue:`1661`).

  By `Stephan Hoyer <https://github.com/shoyer>`_.

- :py:func:`~xarray.open_rasterio` method now shifts the rasterio
  coordinates so that they are centered in each pixel (:issue:`1468`).
  By `Greg Brener <https://github.com/gbrener>`_.

- :py:meth:`~xarray.Dataset.rename` method now doesn't throw errors
  if some ``Variable`` is renamed to the same name as another ``Variable``
  as long as that other ``Variable`` is also renamed (:issue:`1477`). This
  method now does throw when two ``Variables`` would end up with the same name
  after the rename (since one of them would get overwritten in this case).
  By `Prakhar Goel <https://github.com/newt0311>`_.

- Fix :py:func:`xarray.testing.assert_allclose` to actually use ``atol`` and
  ``rtol`` arguments when called on ``DataArray`` objects (:issue:`1488`).
  By `Stephan Hoyer <https://github.com/shoyer>`_.

- xarray ``quantile`` methods now properly raise a ``TypeError`` when applied to
  objects with data stored as ``dask`` arrays (:issue:`1529`).
  By `Joe Hamman <https://github.com/jhamman>`_.

- Fix positional indexing to allow the use of unsigned integers (:issue:`1405`).
  By `Joe Hamman <https://github.com/jhamman>`_ and
  `Gerrit Holl <https://github.com/gerritholl>`_.

- Creating a :py:class:`Dataset` now raises ``MergeError`` if a coordinate
  shares a name with a dimension but is comprised of arbitrary dimensions
  (:issue:`1120`).
  By `Joe Hamman <https://github.com/jhamman>`_.

- :py:func:`~xarray.open_rasterio` method now skips rasterio's ``crs``
  attribute if its value is ``None`` (:issue:`1520`).
  By `Leevi Annala <https://github.com/leevei>`_.

- Fix :py:func:`xarray.DataArray.to_netcdf` to return bytes when no path is
  provided (:issue:`1410`).
  By `Joe Hamman <https://github.com/jhamman>`_.

- Fix :py:func:`xarray.save_mfdataset` to properly raise an informative error
  when objects other than  ``Dataset`` are provided (:issue:`1555`).
  By `Joe Hamman <https://github.com/jhamman>`_.

- :py:func:`xarray.Dataset.copy` would not preserve the encoding property
  (:issue:`1586`).
  By `Guido Imperiale <https://github.com/crusaderky>`_.

- :py:func:`xarray.concat` would eagerly load dask variables into memory if
  the first argument was a numpy variable (:issue:`1588`).
  By `Guido Imperiale <https://github.com/crusaderky>`_.

- Fix bug in :py:meth:`~xarray.Dataset.to_netcdf` when writing in append mode
  (:issue:`1215`).
  By `Joe Hamman <https://github.com/jhamman>`_.

- Fix ``netCDF4`` backend to properly roundtrip the ``shuffle`` encoding option
  (:issue:`1606`).
  By `Joe Hamman <https://github.com/jhamman>`_.

- Fix bug when using ``pytest`` class decorators to skiping certain unittests.
  The previous behavior unintentionally causing additional tests to be skipped
  (:issue:`1531`). By `Joe Hamman <https://github.com/jhamman>`_.

- Fix pynio backend for upcoming release of pynio with Python 3 support
  (:issue:`1611`). By `Ben Hillman <https://github/brhillman>`_.

- Fix ``seaborn`` import warning for Seaborn versions 0.8 and newer when the
  ``apionly`` module was deprecated.
  (:issue:`1633`). By `Joe Hamman <https://github.com/jhamman>`_.

- Fix ``rasterio`` backend for Rasterio versions 1.0alpha10 and newer.
  (:issue:`1641`). By `Chris Holden <https://github.com/ceholden>`_.

Bug fixes after rc1
~~~~~~~~~~~~~~~~~~~

- Suppress warning in IPython autocompletion, related to the deprecation
  of ``.T`` attributes (:issue:`1675`).
  By `Keisuke Fujii <https://github.com/fujiisoup>`_.

- Fix a bug in lazily-indexing netCDF array. (:issue:`1688`)
  By `Keisuke Fujii <https://github.com/fujiisoup>`_.

- (Internal bug) MemoryCachedArray now supports the orthogonal indexing.
  Also made some internal cleanups around array wrappers (:issue:`1429`).
  By `Keisuke Fujii <https://github.com/fujiisoup>`_.

- (Internal bug) MemoryCachedArray now always wraps ``np.ndarray`` by
  ``NumpyIndexingAdapter``. (:issue:`1694`)
  By `Keisuke Fujii <https://github.com/fujiisoup>`_.

- Fix importing xarray when running Python with ``-OO`` (:issue:`1706`).
  By `Stephan Hoyer <https://github.com/shoyer>`_.

- Saving a netCDF file with a coordinates with a spaces in its names now raises
  an appropriate warning (:issue:`1689`).
  By `Stephan Hoyer <https://github.com/shoyer>`_.

- Fix two bugs that were preventing dask arrays from being specified as
  coordinates in the DataArray constructor (:issue:`1684`).
  By `Joe Hamman <https://github.com/jhamman>`_.

- Fixed ``apply_ufunc`` with ``dask='parallelized'`` for scalar arguments
  (:issue:`1697`).
  By `Stephan Hoyer <https://github.com/shoyer>`_.

- Fix "Chunksize cannot exceed dimension size" error when writing netCDF4 files
  loaded from disk (:issue:`1225`).
  By `Stephan Hoyer <https://github.com/shoyer>`_.

- Validate the shape of coordinates with names matching dimensions in the
  DataArray constructor (:issue:`1709`).
  By `Stephan Hoyer <https://github.com/shoyer>`_.

- Raise ``NotImplementedError`` when attempting to save a MultiIndex to a
  netCDF file (:issue:`1547`).
  By `Stephan Hoyer <https://github.com/shoyer>`_.

- Remove netCDF dependency from rasterio backend tests.
  By `Matti Eskelinen <https://github.com/maaleske>`_

.. _whats-new.0.9.6:

v0.9.6 (8 June 2017)
--------------------

This release includes a number of backwards compatible enhancements and bug
fixes.

Enhancements
~~~~~~~~~~~~

- New :py:meth:`~xarray.Dataset.sortby` method to ``Dataset`` and ``DataArray``
  that enable sorting along dimensions (:issue:`967`).
  See :ref:`the docs <reshape.sort>` for examples.
  By `Chun-Wei Yuan <https://github.com/chunweiyuan>`_ and
  `Kyle Heuton <https://github.com/kheuton>`_.

- Add ``.dt`` accessor to DataArrays for computing datetime-like properties
  for the values they contain, similar to ``pandas.Series`` (:issue:`358`).
  By `Daniel Rothenberg <https://github.com/darothen>`_.

- Renamed internal dask arrays created by ``open_dataset`` to match new dask
  conventions (:issue:`1343`).
  By `Ryan Abernathey <https://github.com/rabernat>`_.

- :py:meth:`~xarray.as_variable` is now part of the public API (:issue:`1303`).
  By `Benoit Bovy <https://github.com/benbovy>`_.

- :py:func:`~xarray.align` now supports ``join='exact'``, which raises
  an error instead of aligning when indexes to be aligned are not equal.
  By `Stephan Hoyer <https://github.com/shoyer>`_.

- New function :py:func:`~xarray.open_rasterio` for opening raster files with
  the `rasterio <https://mapbox.github.io/rasterio/>`_ library.
  See :ref:`the docs <io.rasterio>` for details.
  By `Joe Hamman <https://github.com/jhamman>`_,
  `Nic Wayand <https://github.com/NicWayand>`_ and
  `Fabien Maussion <https://github.com/fmaussion>`_

Bug fixes
~~~~~~~~~

- Fix error from repeated indexing of datasets loaded from disk (:issue:`1374`).
  By `Stephan Hoyer <https://github.com/shoyer>`_.

- Fix a bug where ``.isel_points`` wrongly assigns unselected coordinate to
  ``data_vars``.
  By `Keisuke Fujii <https://github.com/fujiisoup>`_.

- Tutorial datasets are now checked against a reference MD5 sum to confirm
  successful download (:issue:`1392`). By `Matthew Gidden
  <https://github.com/gidden>`_.

- ``DataArray.chunk()`` now accepts dask specific kwargs like
  ``Dataset.chunk()`` does. By `Fabien Maussion <https://github.com/fmaussion>`_.

- Support for ``engine='pydap'`` with recent releases of Pydap (3.2.2+),
  including on Python 3 (:issue:`1174`).

Documentation
~~~~~~~~~~~~~

- A new `gallery <http://xarray.pydata.org/en/latest/auto_gallery/index.html>`_
  allows to add interactive examples to the documentation.
  By `Fabien Maussion <https://github.com/fmaussion>`_.

Testing
~~~~~~~

- Fix test suite failure caused by changes to ``pandas.cut`` function
  (:issue:`1386`).
  By `Ryan Abernathey <https://github.com/rabernat>`_.

- Enhanced tests suite by use of ``@network`` decorator, which is
  controlled via ``--run-network-tests`` command line argument
  to ``py.test`` (:issue:`1393`).
  By `Matthew Gidden <https://github.com/gidden>`_.

.. _whats-new.0.9.5:

v0.9.5 (17 April, 2017)
-----------------------

Remove an inadvertently introduced print statement.

.. _whats-new.0.9.3:

v0.9.3 (16 April, 2017)
-----------------------

This minor release includes bug-fixes and backwards compatible enhancements.

Enhancements
~~~~~~~~~~~~

- New :py:meth:`~xarray.DataArray.persist` method to Datasets and DataArrays to
  enable persisting data in distributed memory when using Dask (:issue:`1344`).
  By `Matthew Rocklin <https://github.com/mrocklin>`_.

- New :py:meth:`~xarray.DataArray.expand_dims` method for ``DataArray`` and
  ``Dataset`` (:issue:`1326`).
  By `Keisuke Fujii <https://github.com/fujiisoup>`_.

Bug fixes
~~~~~~~~~

- Fix ``.where()`` with ``drop=True`` when arguments do not have indexes
  (:issue:`1350`). This bug, introduced in v0.9, resulted in xarray producing
  incorrect results in some cases.
  By `Stephan Hoyer <https://github.com/shoyer>`_.

- Fixed writing to file-like objects with :py:meth:`~xarray.Dataset.to_netcdf`
  (:issue:`1320`).
  `Stephan Hoyer <https://github.com/shoyer>`_.

- Fixed explicitly setting ``engine='scipy'`` with ``to_netcdf`` when not
  providing a path (:issue:`1321`).
  `Stephan Hoyer <https://github.com/shoyer>`_.

- Fixed open_dataarray does not pass properly its parameters to open_dataset
  (:issue:`1359`).
  `Stephan Hoyer <https://github.com/shoyer>`_.

- Ensure test suite works when runs from an installed version of xarray
  (:issue:`1336`). Use ``@pytest.mark.slow`` instead of a custom flag to mark
  slow tests.
  By `Stephan Hoyer <https://github.com/shoyer>`_

.. _whats-new.0.9.2:

v0.9.2 (2 April 2017)
---------------------

The minor release includes bug-fixes and backwards compatible enhancements.

Enhancements
~~~~~~~~~~~~

- ``rolling`` on Dataset is now supported (:issue:`859`).

- ``.rolling()`` on Dataset is now supported (:issue:`859`).
  By `Keisuke Fujii <https://github.com/fujiisoup>`_.

- When bottleneck version 1.1 or later is installed, use bottleneck for rolling
  ``var``, ``argmin``, ``argmax``, and ``rank`` computations. Also, rolling
  median now accepts a ``min_periods`` argument (:issue:`1276`).
  By `Joe Hamman <https://github.com/jhamman>`_.

- When ``.plot()`` is called on a 2D DataArray and only one dimension is
  specified with ``x=`` or ``y=``, the other dimension is now guessed
  (:issue:`1291`).
  By `Vincent Noel <https://github.com/vnoel>`_.

- Added new method :py:meth:`~Dataset.assign_attrs` to ``DataArray`` and
  ``Dataset``, a chained-method compatible implementation of the
  ``dict.update`` method on attrs (:issue:`1281`).
  By `Henry S. Harrison <https://hsharrison.github.io>`_.

- Added new ``autoclose=True`` argument to
  :py:func:`~xarray.open_mfdataset` to explicitly close opened files when not in
  use to prevent occurrence of an OS Error related to too many open files
  (:issue:`1198`).
  Note, the default is ``autoclose=False``, which is consistent with
  previous xarray behavior.
  By `Phillip J. Wolfram <https://github.com/pwolfram>`_.

- The ``repr()`` of ``Dataset`` and ``DataArray`` attributes uses a similar
  format to coordinates and variables, with vertically aligned entries
  truncated to fit on a single line (:issue:`1319`).  Hopefully this will stop
  people writing ``data.attrs = {}`` and discarding metadata in notebooks for
  the sake of cleaner output.  The full metadata is still available as
  ``data.attrs``.
  By `Zac Hatfield-Dodds <https://github.com/Zac-HD>`_.

- Enhanced tests suite by use of ``@slow`` and ``@flaky`` decorators, which are
  controlled via ``--run-flaky`` and ``--skip-slow`` command line arguments
  to ``py.test`` (:issue:`1336`).
  By `Stephan Hoyer <https://github.com/shoyer>`_ and
  `Phillip J. Wolfram <https://github.com/pwolfram>`_.

- New aggregation on rolling objects :py:meth:`DataArray.rolling(...).count()`
  which providing a rolling count of valid values (:issue:`1138`).

Bug fixes
~~~~~~~~~
- Rolling operations now keep preserve original dimension order (:issue:`1125`).
  By `Keisuke Fujii <https://github.com/fujiisoup>`_.

- Fixed ``sel`` with ``method='nearest'`` on Python 2.7 and 64-bit Windows
  (:issue:`1140`).
  `Stephan Hoyer <https://github.com/shoyer>`_.

- Fixed ``where`` with ``drop='True'`` for empty masks (:issue:`1341`).
  By `Stephan Hoyer <https://github.com/shoyer>`_ and
  `Phillip J. Wolfram <https://github.com/pwolfram>`_.

.. _whats-new.0.9.1:

v0.9.1 (30 January 2017)
------------------------

Renamed the "Unindexed dimensions" section in the ``Dataset`` and
``DataArray`` repr (added in v0.9.0) to "Dimensions without coordinates"
(:issue:`1199`).

.. _whats-new.0.9.0:

v0.9.0 (25 January 2017)
------------------------

This major release includes five months worth of enhancements and bug fixes from
24 contributors, including some significant changes that are not fully backwards
compatible. Highlights include:

- Coordinates are now *optional* in the xarray data model, even for dimensions.
- Changes to caching, lazy loading and pickling to improve xarray's experience
  for parallel computing.
- Improvements for accessing and manipulating ``pandas.MultiIndex`` levels.
- Many new methods and functions, including
  :py:meth:`~DataArray.quantile`,
  :py:meth:`~DataArray.cumsum`,
  :py:meth:`~DataArray.cumprod`
  :py:attr:`~DataArray.combine_first`
  :py:meth:`~DataArray.set_index`,
  :py:meth:`~DataArray.reset_index`,
  :py:meth:`~DataArray.reorder_levels`,
  :py:func:`~xarray.full_like`,
  :py:func:`~xarray.zeros_like`,
  :py:func:`~xarray.ones_like`
  :py:func:`~xarray.open_dataarray`,
  :py:meth:`~DataArray.compute`,
  :py:meth:`Dataset.info`,
  :py:func:`testing.assert_equal`,
  :py:func:`testing.assert_identical`, and
  :py:func:`testing.assert_allclose`.

Breaking changes
~~~~~~~~~~~~~~~~

- Index coordinates for each dimensions are now optional, and no longer created
  by default :issue:`1017`. You can identify such dimensions without coordinates
  by their appearance in list of "Dimensions without coordinates" in the
  ``Dataset`` or ``DataArray`` repr:

  .. ipython::
    :verbatim:

    In [1]: xr.Dataset({'foo': (('x', 'y'), [[1, 2]])})
    Out[1]:
    <xarray.Dataset>
    Dimensions:  (x: 1, y: 2)
    Dimensions without coordinates: x, y
    Data variables:
        foo      (x, y) int64 1 2

  This has a number of implications:

  - :py:func:`~align` and :py:meth:`~Dataset.reindex` can now error, if
    dimensions labels are missing and dimensions have different sizes.
  - Because pandas does not support missing indexes, methods such as
    ``to_dataframe``/``from_dataframe`` and ``stack``/``unstack`` no longer
    roundtrip faithfully on all inputs. Use :py:meth:`~Dataset.reset_index` to
    remove undesired indexes.
  - ``Dataset.__delitem__`` and :py:meth:`~Dataset.drop` no longer delete/drop
    variables that have dimensions matching a deleted/dropped variable.
  - ``DataArray.coords.__delitem__`` is now allowed on variables matching
    dimension names.
  - ``.sel`` and ``.loc`` now handle indexing along a dimension without
    coordinate labels by doing integer based indexing. See
    :ref:`indexing.missing_coordinates` for an example.
  - :py:attr:`~Dataset.indexes` is no longer guaranteed to include all
    dimensions names as keys. The new method :py:meth:`~Dataset.get_index` has
    been added to get an index for a dimension guaranteed, falling back to
    produce a default ``RangeIndex`` if necessary.

- The default behavior of ``merge`` is now ``compat='no_conflicts'``, so some
  merges will now succeed in cases that previously raised
  ``xarray.MergeError``. Set ``compat='broadcast_equals'`` to restore the
  previous default. See :ref:`combining.no_conflicts` for more details.

- Reading :py:attr:`~DataArray.values` no longer always caches values in a NumPy
  array :issue:`1128`. Caching of ``.values`` on variables read from netCDF
  files on disk is still the default when :py:func:`open_dataset` is called with
  ``cache=True``.
  By `Guido Imperiale <https://github.com/crusaderky>`_ and
  `Stephan Hoyer <https://github.com/shoyer>`_.
- Pickling a ``Dataset`` or ``DataArray`` linked to a file on disk no longer
  caches its values into memory before pickling (:issue:`1128`). Instead, pickle
  stores file paths and restores objects by reopening file references. This
  enables preliminary, experimental use of xarray for opening files with
  `dask.distributed <https://distributed.readthedocs.io>`_.
  By `Stephan Hoyer <https://github.com/shoyer>`_.
- Coordinates used to index a dimension are now loaded eagerly into
  :py:class:`pandas.Index` objects, instead of loading the values lazily.
  By `Guido Imperiale <https://github.com/crusaderky>`_.
- Automatic levels for 2d plots are now guaranteed to land on ``vmin`` and
  ``vmax`` when these kwargs are explicitly provided (:issue:`1191`). The
  automated level selection logic also slightly changed.
  By `Fabien Maussion <https://github.com/fmaussion>`_.

- ``DataArray.rename()`` behavior changed to strictly change the ``DataArray.name``
  if called with string argument, or strictly change coordinate names if called with
  dict-like argument.
  By `Markus Gonser <https://github.com/magonser>`_.

- By default ``to_netcdf()`` add a ``_FillValue = NaN`` attributes to float types.
  By `Frederic Laliberte <https://github.com/laliberte>`_.

- ``repr`` on ``DataArray`` objects uses an shortened display for NumPy array
  data that is less likely to overflow onto multiple pages (:issue:`1207`).
  By `Stephan Hoyer <https://github.com/shoyer>`_.

- xarray no longer supports python 3.3, versions of dask prior to v0.9.0,
  or versions of bottleneck prior to v1.0.

Deprecations
~~~~~~~~~~~~

- Renamed the ``Coordinate`` class from xarray's low level API to
  :py:class:`~xarray.IndexVariable`. ``Variable.to_variable`` and
  ``Variable.to_coord`` have been renamed to
  :py:meth:`~xarray.Variable.to_base_variable` and
  :py:meth:`~xarray.Variable.to_index_variable`.
- Deprecated supplying ``coords`` as a dictionary to the ``DataArray``
  constructor without also supplying an explicit ``dims`` argument. The old
  behavior encouraged relying on the iteration order of dictionaries, which is
  a bad practice (:issue:`727`).
- Removed a number of methods deprecated since v0.7.0 or earlier:
  ``load_data``, ``vars``, ``drop_vars``, ``dump``, ``dumps`` and the
  ``variables`` keyword argument to ``Dataset``.
- Removed the dummy module that enabled ``import xray``.

Enhancements
~~~~~~~~~~~~

- Added new method :py:meth:`~DataArray.combine_first` to ``DataArray`` and
  ``Dataset``, based on the pandas method of the same name (see :ref:`combine`).
  By `Chun-Wei Yuan <https://github.com/chunweiyuan>`_.

- Added the ability to change default automatic alignment (arithmetic_join="inner")
  for binary operations via :py:func:`~xarray.set_options()`
  (see :ref:`math automatic alignment`).
  By `Chun-Wei Yuan <https://github.com/chunweiyuan>`_.

- Add checking of ``attr`` names and values when saving to netCDF, raising useful
  error messages if they are invalid. (:issue:`911`).
  By `Robin Wilson <https://github.com/robintw>`_.
- Added ability to save ``DataArray`` objects directly to netCDF files using
  :py:meth:`~xarray.DataArray.to_netcdf`, and to load directly from netCDF files
  using :py:func:`~xarray.open_dataarray` (:issue:`915`). These remove the need
  to convert a ``DataArray`` to a ``Dataset`` before saving as a netCDF file,
  and deals with names to ensure a perfect 'roundtrip' capability.
  By `Robin Wilson <https://github.com/robintw>`_.
- Multi-index levels are now accessible as "virtual" coordinate variables,
  e.g., ``ds['time']`` can pull out the ``'time'`` level of a multi-index
  (see :ref:`coordinates`). ``sel`` also accepts providing multi-index levels
  as keyword arguments, e.g., ``ds.sel(time='2000-01')``
  (see :ref:`multi-level indexing`).
  By `Benoit Bovy <https://github.com/benbovy>`_.
- Added ``set_index``, ``reset_index`` and ``reorder_levels`` methods to
  easily create and manipulate (multi-)indexes (see :ref:`reshape.set_index`).
  By `Benoit Bovy <https://github.com/benbovy>`_.
- Added the ``compat`` option ``'no_conflicts'`` to ``merge``, allowing the
  combination of xarray objects with disjoint (:issue:`742`) or
  overlapping (:issue:`835`) coordinates as long as all present data agrees.
  By `Johnnie Gray <https://github.com/jcmgray>`_. See
  :ref:`combining.no_conflicts` for more details.
- It is now possible to set ``concat_dim=None`` explicitly in
  :py:func:`~xarray.open_mfdataset` to disable inferring a dimension along
  which to concatenate.
  By `Stephan Hoyer <https://github.com/shoyer>`_.
- Added methods :py:meth:`DataArray.compute`, :py:meth:`Dataset.compute`, and
  :py:meth:`Variable.compute` as a non-mutating alternative to
  :py:meth:`~DataArray.load`.
  By `Guido Imperiale <https://github.com/crusaderky>`_.
- Adds DataArray and Dataset methods :py:meth:`~xarray.DataArray.cumsum` and
  :py:meth:`~xarray.DataArray.cumprod`.  By `Phillip J. Wolfram
  <https://github.com/pwolfram>`_.

- New properties :py:attr:`Dataset.sizes` and :py:attr:`DataArray.sizes` for
  providing consistent access to dimension length on both ``Dataset`` and
  ``DataArray`` (:issue:`921`).
  By `Stephan Hoyer <https://github.com/shoyer>`_.
- New keyword argument ``drop=True`` for :py:meth:`~DataArray.sel`,
  :py:meth:`~DataArray.isel` and :py:meth:`~DataArray.squeeze` for dropping
  scalar coordinates that arise from indexing.
  ``DataArray`` (:issue:`242`).
  By `Stephan Hoyer <https://github.com/shoyer>`_.

- New top-level functions :py:func:`~xarray.full_like`,
  :py:func:`~xarray.zeros_like`, and :py:func:`~xarray.ones_like`
  By `Guido Imperiale <https://github.com/crusaderky>`_.
- Overriding a preexisting attribute with
  :py:func:`~xarray.register_dataset_accessor` or
  :py:func:`~xarray.register_dataarray_accessor` now issues a warning instead of
  raising an error (:issue:`1082`).
  By `Stephan Hoyer <https://github.com/shoyer>`_.
- Options for axes sharing between subplots are exposed to
  :py:class:`FacetGrid` and :py:func:`~xarray.plot.plot`, so axes
  sharing can be disabled for polar plots.
  By `Bas Hoonhout <https://github.com/hoonhout>`_.
- New utility functions :py:func:`~xarray.testing.assert_equal`,
  :py:func:`~xarray.testing.assert_identical`, and
  :py:func:`~xarray.testing.assert_allclose` for asserting relationships
  between xarray objects, designed for use in a pytest test suite.
- ``figsize``, ``size`` and ``aspect`` plot arguments are now supported for all
  plots (:issue:`897`). See :ref:`plotting.figsize` for more details.
  By `Stephan Hoyer <https://github.com/shoyer>`_ and
  `Fabien Maussion <https://github.com/fmaussion>`_.
- New :py:meth:`~Dataset.info` method to summarize ``Dataset`` variables
  and attributes. The method prints to a buffer (e.g. ``stdout``) with output
  similar to what the command line utility ``ncdump -h`` produces (:issue:`1150`).
  By `Joe Hamman <https://github.com/jhamman>`_.
- Added the ability write unlimited netCDF dimensions with the ``scipy`` and
  ``netcdf4`` backends via the new :py:attr:`~xray.Dataset.encoding` attribute
  or via the ``unlimited_dims`` argument to :py:meth:`~xray.Dataset.to_netcdf`.
  By `Joe Hamman <https://github.com/jhamman>`_.
- New :py:meth:`~DataArray.quantile` method to calculate quantiles from
  DataArray objects (:issue:`1187`).
  By `Joe Hamman <https://github.com/jhamman>`_.


Bug fixes
~~~~~~~~~
- ``groupby_bins`` now restores empty bins by default (:issue:`1019`).
  By `Ryan Abernathey <https://github.com/rabernat>`_.

- Fix issues for dates outside the valid range of pandas timestamps
  (:issue:`975`). By `Mathias Hauser <https://github.com/mathause>`_.

- Unstacking produced flipped array after stacking decreasing coordinate values
  (:issue:`980`).
  By `Stephan Hoyer <https://github.com/shoyer>`_.

- Setting ``dtype`` via the ``encoding`` parameter of ``to_netcdf`` failed if
  the encoded dtype was the same as the dtype of the original array
  (:issue:`873`).
  By `Stephan Hoyer <https://github.com/shoyer>`_.

- Fix issues with variables where both attributes ``_FillValue`` and
  ``missing_value`` are set to ``NaN`` (:issue:`997`).
  By `Marco Zühlke <https://github.com/mzuehlke>`_.

- ``.where()`` and ``.fillna()`` now preserve attributes (:issue:`1009`).
  By `Fabien Maussion <https://github.com/fmaussion>`_.

- Applying :py:func:`broadcast()` to an xarray object based on the dask backend
  won't accidentally convert the array from dask to numpy anymore (:issue:`978`).
  By `Guido Imperiale <https://github.com/crusaderky>`_.

- ``Dataset.concat()`` now preserves variables order (:issue:`1027`).
  By `Fabien Maussion <https://github.com/fmaussion>`_.

- Fixed an issue with pcolormesh (:issue:`781`). A new
  ``infer_intervals`` keyword gives control on whether the cell intervals
  should be computed or not.
  By `Fabien Maussion <https://github.com/fmaussion>`_.

- Grouping over an dimension with non-unique values with ``groupby`` gives
  correct groups.
  By `Stephan Hoyer <https://github.com/shoyer>`_.

- Fixed accessing coordinate variables with non-string names from ``.coords``.
  By `Stephan Hoyer <https://github.com/shoyer>`_.

- :py:meth:`~xarray.DataArray.rename` now simultaneously renames the array and
  any coordinate with the same name, when supplied via a :py:class:`dict`
  (:issue:`1116`).
  By `Yves Delley <https://github.com/burnpanck>`_.

- Fixed sub-optimal performance in certain operations with object arrays (:issue:`1121`).
  By `Yves Delley <https://github.com/burnpanck>`_.

- Fix ``.groupby(group)`` when ``group`` has datetime dtype (:issue:`1132`).
  By `Jonas Sølvsteen <https://github.com/j08lue>`_.

- Fixed a bug with facetgrid (the ``norm`` keyword was ignored, :issue:`1159`).
  By `Fabien Maussion <https://github.com/fmaussion>`_.

- Resolved a concurrency bug that could cause Python to crash when
  simultaneously reading and writing netCDF4 files with dask (:issue:`1172`).
  By `Stephan Hoyer <https://github.com/shoyer>`_.

- Fix to make ``.copy()`` actually copy dask arrays, which will be relevant for
  future releases of dask in which dask arrays will be mutable (:issue:`1180`).
  By `Stephan Hoyer <https://github.com/shoyer>`_.

- Fix opening NetCDF files with multi-dimensional time variables
  (:issue:`1229`).
  By `Stephan Hoyer <https://github.com/shoyer>`_.

Performance improvements
~~~~~~~~~~~~~~~~~~~~~~~~

- :py:meth:`~xarray.Dataset.isel_points` and
  :py:meth:`~xarray.Dataset.sel_points` now use vectorised indexing in numpy
  and dask (:issue:`1161`), which can result in several orders of magnitude
  speedup.
  By `Jonathan Chambers <https://github.com/mangecoeur>`_.

.. _whats-new.0.8.2:

v0.8.2 (18 August 2016)
-----------------------

This release includes a number of bug fixes and minor enhancements.

Breaking changes
~~~~~~~~~~~~~~~~

- :py:func:`~xarray.broadcast` and :py:func:`~xarray.concat` now auto-align
  inputs, using ``join=outer``. Previously, these functions raised
  ``ValueError`` for non-aligned inputs.
  By `Guido Imperiale <https://github.com/crusaderky>`_.

Enhancements
~~~~~~~~~~~~

- New documentation on :ref:`panel transition`. By
  `Maximilian Roos <https://github.com/maximilianr>`_.
- New ``Dataset`` and ``DataArray`` methods :py:meth:`~xarray.Dataset.to_dict`
  and :py:meth:`~xarray.Dataset.from_dict` to allow easy conversion between
  dictionaries and xarray objects (:issue:`432`). See
  :ref:`dictionary IO<dictionary io>` for more details.
  By `Julia Signell <https://github.com/jsignell>`_.
- Added ``exclude`` and ``indexes`` optional parameters to :py:func:`~xarray.align`,
  and ``exclude`` optional parameter to :py:func:`~xarray.broadcast`.
  By `Guido Imperiale <https://github.com/crusaderky>`_.
- Better error message when assigning variables without dimensions
  (:issue:`971`). By `Stephan Hoyer <https://github.com/shoyer>`_.
- Better error message when reindex/align fails due to duplicate index values
  (:issue:`956`). By `Stephan Hoyer <https://github.com/shoyer>`_.

Bug fixes
~~~~~~~~~

- Ensure xarray works with h5netcdf v0.3.0 for arrays with ``dtype=str``
  (:issue:`953`). By `Stephan Hoyer <https://github.com/shoyer>`_.
- ``Dataset.__dir__()`` (i.e. the method python calls to get autocomplete
  options) failed if one of the dataset's keys was not a string (:issue:`852`).
  By `Maximilian Roos <https://github.com/maximilianr>`_.
- ``Dataset`` constructor can now take arbitrary objects as values
  (:issue:`647`). By `Maximilian Roos <https://github.com/maximilianr>`_.
- Clarified ``copy`` argument for :py:meth:`~xarray.DataArray.reindex` and
  :py:func:`~xarray.align`, which now consistently always return new xarray
  objects (:issue:`927`).
- Fix ``open_mfdataset`` with ``engine='pynio'`` (:issue:`936`).
  By `Stephan Hoyer <https://github.com/shoyer>`_.
- ``groupby_bins`` sorted bin labels as strings (:issue:`952`).
  By `Stephan Hoyer <https://github.com/shoyer>`_.
- Fix bug introduced by v0.8.0 that broke assignment to datasets when both the
  left and right side have the same non-unique index values (:issue:`956`).

.. _whats-new.0.8.1:

v0.8.1 (5 August 2016)
----------------------

Bug fixes
~~~~~~~~~

- Fix bug in v0.8.0 that broke assignment to Datasets with non-unique
  indexes (:issue:`943`). By `Stephan Hoyer <https://github.com/shoyer>`_.

.. _whats-new.0.8.0:

v0.8.0 (2 August 2016)
----------------------

This release includes four months of new features and bug fixes, including
several breaking changes.

.. _v0.8.0.breaking:

Breaking changes
~~~~~~~~~~~~~~~~

- Dropped support for Python 2.6 (:issue:`855`).
- Indexing on multi-index now drop levels, which is consistent with pandas.
  It also changes the name of the dimension / coordinate when the multi-index is
  reduced to a single index (:issue:`802`).
- Contour plots no longer add a colorbar per default (:issue:`866`). Filled
  contour plots are unchanged.
- ``DataArray.values`` and ``.data`` now always returns an NumPy array-like
  object, even for 0-dimensional arrays with object dtype (:issue:`867`).
  Previously, ``.values`` returned native Python objects in such cases. To
  convert the values of scalar arrays to Python objects, use the ``.item()``
  method.

Enhancements
~~~~~~~~~~~~

- Groupby operations now support grouping over multidimensional variables. A new
  method called :py:meth:`~xarray.Dataset.groupby_bins` has also been added to
  allow users to specify bins for grouping. The new features are described in
  :ref:`groupby.multidim` and :ref:`examples.multidim`.
  By `Ryan Abernathey <https://github.com/rabernat>`_.

- DataArray and Dataset method :py:meth:`where` now supports a ``drop=True``
  option that clips coordinate elements that are fully masked.  By
  `Phillip J. Wolfram <https://github.com/pwolfram>`_.

- New top level :py:func:`merge` function allows for combining variables from
  any number of ``Dataset`` and/or ``DataArray`` variables. See :ref:`merge`
  for more details. By `Stephan Hoyer <https://github.com/shoyer>`_.

- DataArray and Dataset method :py:meth:`resample` now supports the
  ``keep_attrs=False`` option that determines whether variable and dataset
  attributes are retained in the resampled object. By
  `Jeremy McGibbon <https://github.com/mcgibbon>`_.

- Better multi-index support in DataArray and Dataset :py:meth:`sel` and
  :py:meth:`loc` methods, which now behave more closely to pandas and which
  also accept dictionaries for indexing based on given level names and labels
  (see :ref:`multi-level indexing`). By
  `Benoit Bovy <https://github.com/benbovy>`_.

- New (experimental) decorators :py:func:`~xarray.register_dataset_accessor` and
  :py:func:`~xarray.register_dataarray_accessor` for registering custom xarray
  extensions without subclassing. They are described in the new documentation
  page on :ref:`internals`. By `Stephan Hoyer <https://github.com/shoyer>`_.

- Round trip boolean datatypes. Previously, writing boolean datatypes to netCDF
  formats would raise an error since netCDF does not have a `bool` datatype.
  This feature reads/writes a `dtype` attribute to boolean variables in netCDF
  files. By `Joe Hamman <https://github.com/jhamman>`_.

- 2D plotting methods now have two new keywords (`cbar_ax` and `cbar_kwargs`),
  allowing more control on the colorbar (:issue:`872`).
  By `Fabien Maussion <https://github.com/fmaussion>`_.

- New Dataset method :py:meth:`filter_by_attrs`, akin to
  ``netCDF4.Dataset.get_variables_by_attributes``, to easily filter
  data variables using its attributes.
  `Filipe Fernandes <https://github.com/ocefpaf>`_.

Bug fixes
~~~~~~~~~

- Attributes were being retained by default for some resampling
  operations when they should not. With the ``keep_attrs=False`` option, they
  will no longer be retained by default. This may be backwards-incompatible
  with some scripts, but the attributes may be kept by adding the
  ``keep_attrs=True`` option. By
  `Jeremy McGibbon <https://github.com/mcgibbon>`_.

- Concatenating xarray objects along an axis with a MultiIndex or PeriodIndex
  preserves the nature of the index (:issue:`875`). By
  `Stephan Hoyer <https://github.com/shoyer>`_.

- Fixed bug in arithmetic operations on DataArray objects whose dimensions
  are numpy structured arrays or recarrays :issue:`861`, :issue:`837`. By
  `Maciek Swat <https://github.com/maciekswat>`_.

- ``decode_cf_timedelta`` now accepts arrays with ``ndim`` >1 (:issue:`842`).
   This fixes issue :issue:`665`.
   `Filipe Fernandes <https://github.com/ocefpaf>`_.

- Fix a bug where `xarray.ufuncs` that take two arguments would incorrectly
  use to numpy functions instead of dask.array functions (:issue:`876`). By
  `Stephan Hoyer <https://github.com/shoyer>`_.

- Support for pickling functions from  ``xarray.ufuncs`` (:issue:`901`). By
  `Stephan Hoyer <https://github.com/shoyer>`_.

- ``Variable.copy(deep=True)`` no longer converts MultiIndex into a base Index
  (:issue:`769`). By `Benoit Bovy <https://github.com/benbovy>`_.

- Fixes for groupby on dimensions with a multi-index (:issue:`867`). By
  `Stephan Hoyer <https://github.com/shoyer>`_.

- Fix printing datasets with unicode attributes on Python 2 (:issue:`892`). By
  `Stephan Hoyer <https://github.com/shoyer>`_.

- Fixed incorrect test for dask version (:issue:`891`). By
  `Stephan Hoyer <https://github.com/shoyer>`_.

- Fixed `dim` argument for `isel_points`/`sel_points` when a `pandas.Index` is
  passed. By `Stephan Hoyer <https://github.com/shoyer>`_.

- :py:func:`~xarray.plot.contour` now plots the correct number of contours
  (:issue:`866`). By `Fabien Maussion <https://github.com/fmaussion>`_.

.. _whats-new.0.7.2:

v0.7.2 (13 March 2016)
----------------------

This release includes two new, entirely backwards compatible features and
several bug fixes.

Enhancements
~~~~~~~~~~~~

- New DataArray method :py:meth:`DataArray.dot` for calculating the dot
  product of two DataArrays along shared dimensions. By
  `Dean Pospisil <https://github.com/deanpospisil>`_.

- Rolling window operations on DataArray objects are now supported via a new
  :py:meth:`DataArray.rolling` method. For example:

  .. ipython::
    :verbatim:

    In [1]: import xarray as xr; import numpy as np

    In [2]: arr = xr.DataArray(np.arange(0, 7.5, 0.5).reshape(3, 5),
                               dims=('x', 'y'))

    In [3]: arr
    Out[3]:
    <xarray.DataArray (x: 3, y: 5)>
    array([[ 0. ,  0.5,  1. ,  1.5,  2. ],
           [ 2.5,  3. ,  3.5,  4. ,  4.5],
           [ 5. ,  5.5,  6. ,  6.5,  7. ]])
    Coordinates:
      * x        (x) int64 0 1 2
      * y        (y) int64 0 1 2 3 4

    In [4]: arr.rolling(y=3, min_periods=2).mean()
    Out[4]:
    <xarray.DataArray (x: 3, y: 5)>
    array([[  nan,  0.25,  0.5 ,  1.  ,  1.5 ],
           [  nan,  2.75,  3.  ,  3.5 ,  4.  ],
           [  nan,  5.25,  5.5 ,  6.  ,  6.5 ]])
    Coordinates:
      * x        (x) int64 0 1 2
      * y        (y) int64 0 1 2 3 4

  See :ref:`comput.rolling` for more details. By
  `Joe Hamman <https://github.com/jhamman>`_.

Bug fixes
~~~~~~~~~

- Fixed an issue where plots using pcolormesh and Cartopy axes were being distorted
  by the inference of the axis interval breaks. This change chooses not to modify
  the coordinate variables when the axes have the attribute ``projection``, allowing
  Cartopy to handle the extent of pcolormesh plots (:issue:`781`). By
  `Joe Hamman <https://github.com/jhamman>`_.

- 2D plots now better handle additional coordinates which are not ``DataArray``
  dimensions (:issue:`788`). By `Fabien Maussion <https://github.com/fmaussion>`_.


.. _whats-new.0.7.1:

v0.7.1 (16 February 2016)
-------------------------

This is a bug fix release that includes two small, backwards compatible enhancements.
We recommend that all users upgrade.

Enhancements
~~~~~~~~~~~~

- Numerical operations now return empty objects on no overlapping labels rather
  than raising ``ValueError`` (:issue:`739`).
- :py:class:`~pd.Series` is now supported as valid input to the ``Dataset``
  constructor (:issue:`740`).

Bug fixes
~~~~~~~~~

- Restore checks for shape consistency between data and coordinates in the
  DataArray constructor (:issue:`758`).
- Single dimension variables no longer transpose as part of a broader
  ``.transpose``. This  behavior was causing ``pandas.PeriodIndex`` dimensions
  to lose their type (:issue:`749`)
- :py:class:`~xarray.Dataset` labels remain as their native type on ``.to_dataset``.
  Previously they were coerced to strings (:issue:`745`)
- Fixed a bug where replacing a ``DataArray`` index coordinate would improperly
  align the coordinate (:issue:`725`).
- ``DataArray.reindex_like`` now maintains the dtype of complex numbers when
  reindexing leads to NaN values (:issue:`738`).
- ``Dataset.rename`` and ``DataArray.rename`` support the old and new names
  being the same (:issue:`724`).
- Fix :py:meth:`~xarray.Dataset.from_dataset` for DataFrames with Categorical
  column and a MultiIndex index (:issue:`737`).
- Fixes to ensure xarray works properly after the upcoming pandas v0.18 and
  NumPy v1.11 releases.

Acknowledgments
~~~~~~~~~~~~~~~

The following individuals contributed to this release:

- Edward Richards
- Maximilian Roos
- Rafael Guedes
- Spencer Hill
- Stephan Hoyer

.. _whats-new.0.7.0:

v0.7.0 (21 January 2016)
------------------------

This major release includes redesign of :py:class:`~xarray.DataArray`
internals, as well as new methods for reshaping, rolling and shifting
data. It includes preliminary support for :py:class:`pandas.MultiIndex`,
as well as a number of other features and bug fixes, several of which
offer improved compatibility with pandas.

New name
~~~~~~~~

The project formerly known as "xray" is now "xarray", pronounced "x-array"!
This avoids a namespace conflict with the entire field of x-ray science. Renaming
our project seemed like the right thing to do, especially because some
scientists who work with actual x-rays are interested in using this project in
their work. Thanks for your understanding and patience in this transition. You
can now find our documentation and code repository at new URLs:

- http://xarray.pydata.org
- http://github.com/pydata/xarray/

To ease the transition, we have simultaneously released v0.7.0 of both
``xray`` and ``xarray`` on the Python Package Index. These packages are
identical. For now, ``import xray`` still works, except it issues a
deprecation warning. This will be the last xray release. Going forward, we
recommend switching your import statements to ``import xarray as xr``.

.. _v0.7.0.breaking:

Breaking changes
~~~~~~~~~~~~~~~~

- The internal data model used by :py:class:`~xray.DataArray` has been
  rewritten to fix several outstanding issues (:issue:`367`, :issue:`634`,
  `this stackoverflow report`_). Internally, ``DataArray`` is now implemented
  in terms of ``._variable`` and ``._coords`` attributes instead of holding
  variables in a ``Dataset`` object.

  This refactor ensures that if a DataArray has the
  same name as one of its coordinates, the array and the coordinate no longer
  share the same data.

  In practice, this means that creating a DataArray with the same ``name`` as
  one of its dimensions no longer automatically uses that array to label the
  corresponding coordinate. You will now need to provide coordinate labels
  explicitly. Here's the old behavior:

  .. ipython::
    :verbatim:

    In [2]: xray.DataArray([4, 5, 6], dims='x', name='x')
    Out[2]:
    <xray.DataArray 'x' (x: 3)>
    array([4, 5, 6])
    Coordinates:
      * x        (x) int64 4 5 6

  and the new behavior (compare the values of the ``x`` coordinate):

  .. ipython::
    :verbatim:

    In [2]: xray.DataArray([4, 5, 6], dims='x', name='x')
    Out[2]:
    <xray.DataArray 'x' (x: 3)>
    array([4, 5, 6])
    Coordinates:
      * x        (x) int64 0 1 2

- It is no longer possible to convert a DataArray to a Dataset with
  :py:meth:`xray.DataArray.to_dataset` if it is unnamed. This will now
  raise ``ValueError``. If the array is unnamed, you need to supply the
  ``name`` argument.

.. _this stackoverflow report: http://stackoverflow.com/questions/33158558/python-xray-extract-first-and-last-time-value-within-each-month-of-a-timeseries

Enhancements
~~~~~~~~~~~~

- Basic support for :py:class:`~pandas.MultiIndex` coordinates on xray objects, including
  indexing, :py:meth:`~DataArray.stack` and :py:meth:`~DataArray.unstack`:

  .. ipython::
    :verbatim:

    In [7]: df = pd.DataFrame({'foo': range(3),
       ...:                    'x': ['a', 'b', 'b'],
       ...:                    'y': [0, 0, 1]})

    In [8]: s = df.set_index(['x', 'y'])['foo']

    In [12]: arr = xray.DataArray(s, dims='z')

    In [13]: arr
    Out[13]:
    <xray.DataArray 'foo' (z: 3)>
    array([0, 1, 2])
    Coordinates:
      * z        (z) object ('a', 0) ('b', 0) ('b', 1)

    In [19]: arr.indexes['z']
    Out[19]:
    MultiIndex(levels=[[u'a', u'b'], [0, 1]],
               labels=[[0, 1, 1], [0, 0, 1]],
               names=[u'x', u'y'])

    In [14]: arr.unstack('z')
    Out[14]:
    <xray.DataArray 'foo' (x: 2, y: 2)>
    array([[  0.,  nan],
           [  1.,   2.]])
    Coordinates:
      * x        (x) object 'a' 'b'
      * y        (y) int64 0 1

    In [26]: arr.unstack('z').stack(z=('x', 'y'))
    Out[26]:
    <xray.DataArray 'foo' (z: 4)>
    array([  0.,  nan,   1.,   2.])
    Coordinates:
      * z        (z) object ('a', 0) ('a', 1) ('b', 0) ('b', 1)

  See :ref:`reshape.stack` for more details.

  .. warning::

      xray's MultiIndex support is still experimental, and we have a long to-
      do list of desired additions (:issue:`719`), including better display of
      multi-index levels when printing a ``Dataset``, and support for saving
      datasets with a MultiIndex to a netCDF file. User contributions in this
      area would be greatly appreciated.

- Support for reading GRIB, HDF4 and other file formats via PyNIO_. See
  :ref:`io.pynio` for more details.
- Better error message when a variable is supplied with the same name as
  one of its dimensions.
- Plotting: more control on colormap parameters (:issue:`642`). ``vmin`` and
  ``vmax`` will not be silently ignored anymore. Setting ``center=False``
  prevents automatic selection of a divergent colormap.
- New :py:meth:`~xray.Dataset.shift` and :py:meth:`~xray.Dataset.roll` methods
  for shifting/rotating datasets or arrays along a dimension:

  .. ipython:: python

      array = xray.DataArray([5, 6, 7, 8], dims='x')
      array.shift(x=2)
      array.roll(x=2)

  Notice that ``shift`` moves data independently of coordinates, but ``roll``
  moves both data and coordinates.
- Assigning a ``pandas`` object directly as a ``Dataset`` variable is now permitted. Its
  index names correspond to the ``dims`` of the ``Dataset``, and its data is aligned.
- Passing a :py:class:`pandas.DataFrame` or :py:class:`pandas.Panel` to a Dataset constructor
  is now permitted.
- New function :py:func:`~xray.broadcast` for explicitly broadcasting
  ``DataArray`` and ``Dataset`` objects against each other. For example:

  .. ipython:: python

      a = xray.DataArray([1, 2, 3], dims='x')
      b = xray.DataArray([5, 6], dims='y')
      a
      b
      a2, b2 = xray.broadcast(a, b)
      a2
      b2

.. _PyNIO: https://www.pyngl.ucar.edu/Nio.shtml

Bug fixes
~~~~~~~~~

- Fixes for several issues found on ``DataArray`` objects with the same name
  as one of their coordinates (see :ref:`v0.7.0.breaking` for more details).
- ``DataArray.to_masked_array`` always returns masked array with mask being an
  array (not a scalar value) (:issue:`684`)
- Allows for (imperfect) repr of Coords when underlying index is PeriodIndex (:issue:`645`).
- Fixes for several issues found on ``DataArray`` objects with the same name
  as one of their coordinates (see :ref:`v0.7.0.breaking` for more details).
- Attempting to assign a ``Dataset`` or ``DataArray`` variable/attribute using
  attribute-style syntax (e.g., ``ds.foo = 42``) now raises an error rather
  than silently failing (:issue:`656`, :issue:`714`).
- You can now pass pandas objects with non-numpy dtypes (e.g., ``categorical``
  or ``datetime64`` with a timezone) into xray without an error
  (:issue:`716`).

Acknowledgments
~~~~~~~~~~~~~~~

The following individuals contributed to this release:

- Antony Lee
- Fabien Maussion
- Joe Hamman
- Maximilian Roos
- Stephan Hoyer
- Takeshi Kanmae
- femtotrader

v0.6.1 (21 October 2015)
------------------------

This release contains a number of bug and compatibility fixes, as well
as enhancements to plotting, indexing and writing files to disk.

Note that the minimum required version of dask for use with xray is now
version 0.6.

API Changes
~~~~~~~~~~~

- The handling of colormaps and discrete color lists for 2D plots in
  :py:meth:`~xray.DataArray.plot` was changed to provide more compatibility
  with matplotlib's ``contour`` and ``contourf`` functions (:issue:`538`).
  Now discrete lists of colors should be specified using ``colors`` keyword,
  rather than ``cmap``.

Enhancements
~~~~~~~~~~~~

- Faceted plotting through :py:class:`~xray.plot.FacetGrid` and the
  :py:meth:`~xray.plot.plot` method. See :ref:`plotting.faceting` for more details
  and examples.
- :py:meth:`~xray.Dataset.sel` and :py:meth:`~xray.Dataset.reindex` now support
  the ``tolerance`` argument for controlling nearest-neighbor selection
  (:issue:`629`):

  .. ipython::
    :verbatim:

    In [5]: array = xray.DataArray([1, 2, 3], dims='x')

    In [6]: array.reindex(x=[0.9, 1.5], method='nearest', tolerance=0.2)
    Out[6]:
    <xray.DataArray (x: 2)>
    array([  2.,  nan])
    Coordinates:
      * x        (x) float64 0.9 1.5

  This feature requires pandas v0.17 or newer.
- New ``encoding`` argument in :py:meth:`~xray.Dataset.to_netcdf` for writing
  netCDF files with compression, as described in the new documentation
  section on :ref:`io.netcdf.writing_encoded`.
- Add :py:attr:`~xray.Dataset.real` and :py:attr:`~xray.Dataset.imag`
  attributes to Dataset and DataArray (:issue:`553`).
- More informative error message with :py:meth:`~xray.Dataset.from_dataframe`
  if the frame has duplicate columns.
- xray now uses deterministic names for dask arrays it creates or opens from
  disk. This allows xray users to take advantage of dask's nascent support for
  caching intermediate computation results. See :issue:`555` for an example.

Bug fixes
~~~~~~~~~

- Forwards compatibility with the latest pandas release (v0.17.0). We were
  using some internal pandas routines for datetime conversion, which
  unfortunately have now changed upstream (:issue:`569`).
- Aggregation functions now correctly skip ``NaN`` for data for ``complex128``
  dtype (:issue:`554`).
- Fixed indexing 0d arrays with unicode dtype (:issue:`568`).
- :py:meth:`~xray.DataArray.name` and Dataset keys must be a string or None to
  be written to netCDF (:issue:`533`).
- :py:meth:`~xray.DataArray.where` now uses dask instead of numpy if either the
  array or ``other`` is a dask array. Previously, if ``other`` was a numpy array
  the method was evaluated eagerly.
- Global attributes are now handled more consistently when loading remote
  datasets using ``engine='pydap'`` (:issue:`574`).
- It is now possible to assign to the ``.data`` attribute of DataArray objects.
- ``coordinates`` attribute is now kept in the encoding dictionary after
  decoding (:issue:`610`).
- Compatibility with numpy 1.10 (:issue:`617`).

Acknowledgments
~~~~~~~~~~~~~~~

The following individuals contributed to this release:

- Ryan Abernathey
- Pete Cable
- Clark Fitzgerald
- Joe Hamman
- Stephan Hoyer
- Scott Sinclair

v0.6.0 (21 August 2015)
-----------------------

This release includes numerous bug fixes and enhancements. Highlights
include the introduction of a plotting module and the new Dataset and DataArray
methods :py:meth:`~xray.Dataset.isel_points`, :py:meth:`~xray.Dataset.sel_points`,
:py:meth:`~xray.Dataset.where` and :py:meth:`~xray.Dataset.diff`. There are no
breaking changes from v0.5.2.

Enhancements
~~~~~~~~~~~~

- Plotting methods have been implemented on DataArray objects
  :py:meth:`~xray.DataArray.plot` through integration with matplotlib
  (:issue:`185`). For an introduction, see :ref:`plotting`.
- Variables in netCDF files with multiple missing values are now decoded as NaN
  after issuing a warning if open_dataset is called with mask_and_scale=True.
- We clarified our rules for when the result from an xray operation is a copy
  vs. a view (see :ref:`copies vs views` for more details).
- Dataset variables are now written to netCDF files in order of appearance
  when using the netcdf4 backend (:issue:`479`).

- Added :py:meth:`~xray.Dataset.isel_points` and :py:meth:`~xray.Dataset.sel_points`
  to support pointwise indexing of Datasets and DataArrays (:issue:`475`).

  .. ipython::
    :verbatim:

    In [1]: da = xray.DataArray(np.arange(56).reshape((7, 8)),
       ...:                     coords={'x': list('abcdefg'),
       ...:                             'y': 10 * np.arange(8)},
       ...:                     dims=['x', 'y'])

    In [2]: da
    Out[2]:
    <xray.DataArray (x: 7, y: 8)>
    array([[ 0,  1,  2,  3,  4,  5,  6,  7],
           [ 8,  9, 10, 11, 12, 13, 14, 15],
           [16, 17, 18, 19, 20, 21, 22, 23],
           [24, 25, 26, 27, 28, 29, 30, 31],
           [32, 33, 34, 35, 36, 37, 38, 39],
           [40, 41, 42, 43, 44, 45, 46, 47],
           [48, 49, 50, 51, 52, 53, 54, 55]])
    Coordinates:
    * y        (y) int64 0 10 20 30 40 50 60 70
    * x        (x) |S1 'a' 'b' 'c' 'd' 'e' 'f' 'g'

    # we can index by position along each dimension
    In [3]: da.isel_points(x=[0, 1, 6], y=[0, 1, 0], dim='points')
    Out[3]:
    <xray.DataArray (points: 3)>
    array([ 0,  9, 48])
    Coordinates:
        y        (points) int64 0 10 0
        x        (points) |S1 'a' 'b' 'g'
      * points   (points) int64 0 1 2

    # or equivalently by label
    In [9]: da.sel_points(x=['a', 'b', 'g'], y=[0, 10, 0], dim='points')
    Out[9]:
    <xray.DataArray (points: 3)>
    array([ 0,  9, 48])
    Coordinates:
        y        (points) int64 0 10 0
        x        (points) |S1 'a' 'b' 'g'
      * points   (points) int64 0 1 2

- New :py:meth:`~xray.Dataset.where` method for masking xray objects according
  to some criteria. This works particularly well with multi-dimensional data:

  .. ipython:: python

    ds = xray.Dataset(coords={'x': range(100), 'y': range(100)})
    ds['distance'] = np.sqrt(ds.x ** 2 + ds.y ** 2)

    @savefig where_example.png width=4in height=4in
    ds.distance.where(ds.distance < 100).plot()

- Added new methods :py:meth:`DataArray.diff <xray.DataArray.diff>`
  and :py:meth:`Dataset.diff <xray.Dataset.diff>` for finite
  difference calculations along a given axis.

- New :py:meth:`~xray.DataArray.to_masked_array` convenience method for
  returning a numpy.ma.MaskedArray.

  .. ipython:: python

    da = xray.DataArray(np.random.random_sample(size=(5, 4)))
    da.where(da < 0.5)
    da.where(da < 0.5).to_masked_array(copy=True)

- Added new flag "drop_variables" to :py:meth:`~xray.open_dataset` for
  excluding variables from being parsed. This may be useful to drop
  variables with problems or inconsistent values.

Bug fixes
~~~~~~~~~

- Fixed aggregation functions (e.g., sum and mean) on big-endian arrays when
  bottleneck is installed (:issue:`489`).
- Dataset aggregation functions dropped variables with unsigned integer dtype
  (:issue:`505`).
- ``.any()`` and ``.all()`` were not lazy when used on xray objects containing
  dask arrays.
- Fixed an error when attempting to saving datetime64 variables to netCDF
  files when the first element is ``NaT`` (:issue:`528`).
- Fix pickle on DataArray objects (:issue:`515`).
- Fixed unnecessary coercion of float64 to float32 when using netcdf3 and
  netcdf4_classic formats (:issue:`526`).

v0.5.2 (16 July 2015)
---------------------

This release contains bug fixes, several additional options for opening and
saving netCDF files, and a backwards incompatible rewrite of the advanced
options for ``xray.concat``.

Backwards incompatible changes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- The optional arguments ``concat_over`` and ``mode`` in :py:func:`~xray.concat` have
  been removed and replaced by ``data_vars`` and ``coords``. The new arguments are both
  more easily understood and more robustly implemented, and allowed us to fix a bug
  where ``concat`` accidentally loaded data into memory. If you set values for
  these optional arguments manually, you will need to update your code. The default
  behavior should be unchanged.

Enhancements
~~~~~~~~~~~~

- :py:func:`~xray.open_mfdataset` now supports a ``preprocess`` argument for
  preprocessing datasets prior to concatenaton. This is useful if datasets
  cannot be otherwise merged automatically, e.g., if the original datasets
  have conflicting index coordinates (:issue:`443`).
- :py:func:`~xray.open_dataset` and :py:func:`~xray.open_mfdataset` now use a
  global thread lock by default for reading from netCDF files with dask. This
  avoids possible segmentation faults for reading from netCDF4 files when HDF5
  is not configured properly for concurrent access (:issue:`444`).
- Added support for serializing arrays of complex numbers with `engine='h5netcdf'`.
- The new :py:func:`~xray.save_mfdataset` function allows for saving multiple
  datasets to disk simultaneously. This is useful when processing large datasets
  with dask.array. For example, to save a dataset too big to fit into memory
  to one file per year, we could write:

  .. ipython::
    :verbatim:

    In [1]: years, datasets = zip(*ds.groupby('time.year'))

    In [2]: paths = ['%s.nc' % y for y in years]

    In [3]: xray.save_mfdataset(datasets, paths)

Bug fixes
~~~~~~~~~

- Fixed ``min``, ``max``, ``argmin`` and ``argmax`` for arrays with string or
  unicode types (:issue:`453`).
- :py:func:`~xray.open_dataset` and :py:func:`~xray.open_mfdataset` support
  supplying chunks as a single integer.
- Fixed a bug in serializing scalar datetime variable to netCDF.
- Fixed a bug that could occur in serialization of 0-dimensional integer arrays.
- Fixed a bug where concatenating DataArrays was not always lazy (:issue:`464`).
- When reading datasets with h5netcdf, bytes attributes are decoded to strings.
  This allows conventions decoding to work properly on Python 3 (:issue:`451`).

v0.5.1 (15 June 2015)
---------------------

This minor release fixes a few bugs and an inconsistency with pandas. It also
adds the ``pipe`` method, copied from pandas.

Enhancements
~~~~~~~~~~~~

- Added :py:meth:`~xray.Dataset.pipe`, replicating the `new pandas method`_ in version
  0.16.2. See :ref:`transforming datasets` for more details.
- :py:meth:`~xray.Dataset.assign` and :py:meth:`~xray.Dataset.assign_coords`
  now assign new variables in sorted (alphabetical) order, mirroring the
  behavior in pandas. Previously, the order was arbitrary.

.. _new pandas method: http://pandas.pydata.org/pandas-docs/version/0.16.2/whatsnew.html#pipe

Bug fixes
~~~~~~~~~

- ``xray.concat`` fails in an edge case involving identical coordinate variables (:issue:`425`)
- We now decode variables loaded from netCDF3 files with the scipy engine using native
  endianness (:issue:`416`). This resolves an issue when aggregating these arrays with
  bottleneck installed.

v0.5 (1 June 2015)
------------------

Highlights
~~~~~~~~~~

The headline feature in this release is experimental support for out-of-core
computing (data that doesn't fit into memory) with dask_. This includes a new
top-level function :py:func:`~xray.open_mfdataset` that makes it easy to open
a collection of netCDF (using dask) as a single ``xray.Dataset`` object. For
more on dask, read the `blog post introducing xray + dask`_ and the new
documentation section :doc:`dask`.

.. _blog post introducing xray + dask: https://www.anaconda.com/blog/developer-blog/xray-dask-out-core-labeled-arrays-python/

Dask makes it possible to harness parallelism and manipulate gigantic datasets
with xray. It is currently an optional dependency, but it may become required
in the future.

Backwards incompatible changes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- The logic used for choosing which variables are concatenated with
  :py:func:`~xray.concat` has changed. Previously, by default any variables
  which were equal across a dimension were not concatenated. This lead to some
  surprising behavior, where the behavior of groupby and concat operations
  could depend on runtime values (:issue:`268`). For example:

  .. ipython::
    :verbatim:

    In [1]: ds = xray.Dataset({'x': 0})

    In [2]: xray.concat([ds, ds], dim='y')
    Out[2]:
    <xray.Dataset>
    Dimensions:  ()
    Coordinates:
        *empty*
    Data variables:
        x        int64 0

  Now, the default always concatenates data variables:

  .. ipython:: python
    :suppress:

    ds = xray.Dataset({'x': 0})

  .. ipython:: python

    xray.concat([ds, ds], dim='y')

  To obtain the old behavior, supply the argument ``concat_over=[]``.

Enhancements
~~~~~~~~~~~~

- New :py:meth:`~xray.Dataset.to_array` and enhanced
  :py:meth:`~xray.DataArray.to_dataset` methods make it easy to switch back
  and forth between arrays and datasets:

  .. ipython:: python

      ds = xray.Dataset({'a': 1, 'b': ('x', [1, 2, 3])},
                        coords={'c': 42}, attrs={'Conventions': 'None'})
      ds.to_array()
      ds.to_array().to_dataset(dim='variable')

- New :py:meth:`~xray.Dataset.fillna` method to fill missing values, modeled
  off the pandas method of the same name:

  .. ipython:: python

      array = xray.DataArray([np.nan, 1, np.nan, 3], dims='x')
      array.fillna(0)

  ``fillna`` works on both ``Dataset`` and ``DataArray`` objects, and uses
  index based alignment and broadcasting like standard binary operations. It
  also can be applied by group, as illustrated in
  :ref:`fill with climatology`.
- New :py:meth:`~xray.Dataset.assign` and :py:meth:`~xray.Dataset.assign_coords`
  methods patterned off the new :py:meth:`DataFrame.assign <pandas.DataFrame.assign>`
  method in pandas:

  .. ipython:: python

      ds = xray.Dataset({'y': ('x', [1, 2, 3])})
      ds.assign(z = lambda ds: ds.y ** 2)
      ds.assign_coords(z = ('x', ['a', 'b', 'c']))

  These methods return a new Dataset (or DataArray) with updated data or
  coordinate variables.
- :py:meth:`~xray.Dataset.sel` now supports the ``method`` parameter, which works
  like the paramter of the same name on :py:meth:`~xray.Dataset.reindex`. It
  provides a simple interface for doing nearest-neighbor interpolation:

  .. use verbatim because I can't seem to install pandas 0.16.1 on RTD :(

  .. ipython::
      :verbatim:

      In [12]: ds.sel(x=1.1, method='nearest')
      Out[12]:
      <xray.Dataset>
      Dimensions:  ()
      Coordinates:
          x        int64 1
      Data variables:
          y        int64 2

      In [13]: ds.sel(x=[1.1, 2.1], method='pad')
      Out[13]:
      <xray.Dataset>
      Dimensions:  (x: 2)
      Coordinates:
        * x        (x) int64 1 2
      Data variables:
          y        (x) int64 2 3

  See :ref:`nearest neighbor lookups` for more details.
- You can now control the underlying backend used for accessing remote
  datasets (via OPeNDAP) by specifying ``engine='netcdf4'`` or
  ``engine='pydap'``.
- xray now provides experimental support for reading and writing netCDF4 files directly
  via `h5py`_ with the `h5netcdf`_ package, avoiding the netCDF4-Python package. You
  will need to install h5netcdf and specify ``engine='h5netcdf'`` to try this
  feature.
- Accessing data from remote datasets now has retrying logic (with exponential
  backoff) that should make it robust to occasional bad responses from DAP
  servers.
- You can control the width of the Dataset repr with :py:class:`xray.set_options`.
  It can be used either as a context manager, in which case the default is restored
  outside the context:

  .. ipython:: python

      ds = xray.Dataset({'x': np.arange(1000)})
      with xray.set_options(display_width=40):
          print(ds)

  Or to set a global option:

  .. ipython::
      :verbatim:

      In [1]: xray.set_options(display_width=80)

  The default value for the ``display_width`` option is 80.

.. _h5py: http://www.h5py.org/
.. _h5netcdf: https://github.com/shoyer/h5netcdf

Deprecations
~~~~~~~~~~~~

- The method ``load_data()`` has been renamed to the more succinct
  :py:meth:`~xray.Dataset.load`.

v0.4.1 (18 March 2015)
----------------------

The release contains bug fixes and several new features. All changes should be
fully backwards compatible.

Enhancements
~~~~~~~~~~~~

- New documentation sections on :ref:`time-series` and
  :ref:`combining multiple files`.
- :py:meth:`~xray.Dataset.resample` lets you resample a dataset or data array to
  a new temporal resolution. The syntax is the `same as pandas`_, except you
  need to supply the time dimension explicitly:

  .. ipython:: python

      time = pd.date_range('2000-01-01', freq='6H', periods=10)
      array = xray.DataArray(np.arange(10), [('time', time)])
      array.resample('1D', dim='time')

  You can specify how to do the resampling with the ``how`` argument and other
  options such as ``closed`` and ``label`` let you control labeling:

  .. ipython:: python

      array.resample('1D', dim='time', how='sum', label='right')

  If the desired temporal resolution is higher than the original data
  (upsampling), xray will insert missing values:

  .. ipython:: python

      array.resample('3H', 'time')

- ``first`` and ``last`` methods on groupby objects let you take the first or
  last examples from each group along the grouped axis:

  .. ipython:: python

      array.groupby('time.day').first()

  These methods combine well with ``resample``:

  .. ipython:: python

      array.resample('1D', dim='time', how='first')


- :py:meth:`~xray.Dataset.swap_dims` allows for easily swapping one dimension
  out for another:

  .. ipython:: python

       ds = xray.Dataset({'x': range(3), 'y': ('x', list('abc'))})
       ds
       ds.swap_dims({'x': 'y'})

  This was possible in earlier versions of xray, but required some contortions.
- :py:func:`~xray.open_dataset` and :py:meth:`~xray.Dataset.to_netcdf` now
  accept an ``engine`` argument to explicitly select which underlying library
  (netcdf4 or scipy) is used for reading/writing a netCDF file.

.. _same as pandas: http://pandas.pydata.org/pandas-docs/stable/timeseries.html#up-and-downsampling

Bug fixes
~~~~~~~~~

- Fixed a bug where data netCDF variables read from disk with
  ``engine='scipy'`` could still be associated with the file on disk, even
  after closing the file (:issue:`341`). This manifested itself in warnings
  about mmapped arrays and segmentation faults (if the data was accessed).
- Silenced spurious warnings about all-NaN slices when using nan-aware
  aggregation methods (:issue:`344`).
- Dataset aggregations with ``keep_attrs=True`` now preserve attributes on
  data variables, not just the dataset itself.
- Tests for xray now pass when run on Windows (:issue:`360`).
- Fixed a regression in v0.4 where saving to netCDF could fail with the error
  ``ValueError: could not automatically determine time units``.

v0.4 (2 March, 2015)
--------------------

This is one of the biggest releases yet for xray: it includes some major
changes that may break existing code, along with the usual collection of minor
enhancements and bug fixes. On the plus side, this release includes all
hitherto planned breaking changes, so the upgrade path for xray should be
smoother going forward.

Breaking changes
~~~~~~~~~~~~~~~~

- We now automatically align index labels in arithmetic, dataset construction,
  merging and updating. This means the need for manually invoking methods like
  :py:func:`~xray.align` and :py:meth:`~xray.Dataset.reindex_like` should be
  vastly reduced.

  :ref:`For arithmetic<math automatic alignment>`, we align
  based on the **intersection** of labels:

  .. ipython:: python

      lhs = xray.DataArray([1, 2, 3], [('x', [0, 1, 2])])
      rhs = xray.DataArray([2, 3, 4], [('x', [1, 2, 3])])
      lhs + rhs

  :ref:`For dataset construction and merging<merge>`, we align based on the
  **union** of labels:

  .. ipython:: python

      xray.Dataset({'foo': lhs, 'bar': rhs})

  :ref:`For update and __setitem__<update>`, we align based on the **original**
  object:

  .. ipython:: python

      lhs.coords['rhs'] = rhs
      lhs

- Aggregations like ``mean`` or ``median`` now skip missing values by default:

  .. ipython:: python

      xray.DataArray([1, 2, np.nan, 3]).mean()

  You can turn this behavior off by supplying the keyword arugment
  ``skipna=False``.

  These operations are lightning fast thanks to integration with bottleneck_,
  which is a new optional dependency for xray (numpy is used if bottleneck is
  not installed).
- Scalar coordinates no longer conflict with constant arrays with the same
  value (e.g., in arithmetic, merging datasets and concat), even if they have
  different shape (:issue:`243`). For example, the coordinate ``c`` here
  persists through arithmetic, even though it has different shapes on each
  DataArray:

  .. ipython:: python

      a = xray.DataArray([1, 2], coords={'c': 0}, dims='x')
      b = xray.DataArray([1, 2], coords={'c': ('x', [0, 0])}, dims='x')
      (a + b).coords

  This functionality can be controlled through the ``compat`` option, which
  has also been added to the :py:class:`~xray.Dataset` constructor.
- Datetime shortcuts such as ``'time.month'`` now return a ``DataArray`` with
  the name ``'month'``, not ``'time.month'`` (:issue:`345`). This makes it
  easier to index the resulting arrays when they are used with ``groupby``:

  .. ipython:: python

      time = xray.DataArray(pd.date_range('2000-01-01', periods=365),
                            dims='time', name='time')
      counts = time.groupby('time.month').count()
      counts.sel(month=2)

  Previously, you would need to use something like
  ``counts.sel(**{'time.month': 2}})``, which is much more awkward.
- The ``season`` datetime shortcut now returns an array of string labels
  such `'DJF'`:

  .. ipython:: python

      ds = xray.Dataset({'t': pd.date_range('2000-01-01', periods=12, freq='M')})
      ds['t.season']

  Previously, it returned numbered seasons 1 through 4.
- We have updated our use of the terms of "coordinates" and "variables". What
  were known in previous versions of xray as "coordinates" and "variables" are
  now referred to throughout the documentation as "coordinate variables" and
  "data variables". This brings xray in closer alignment to `CF Conventions`_.
  The only visible change besides the documentation is that ``Dataset.vars``
  has been renamed ``Dataset.data_vars``.
- You will need to update your code if you have been ignoring deprecation
  warnings: methods and attributes that were deprecated in xray v0.3 or earlier
  (e.g., ``dimensions``, ``attributes```) have gone away.

.. _bottleneck: https://github.com/kwgoodman/bottleneck

Enhancements
~~~~~~~~~~~~

- Support for :py:meth:`~xray.Dataset.reindex` with a fill method. This
  provides a useful shortcut for upsampling:

  .. ipython:: python

      data = xray.DataArray([1, 2, 3], [('x', range(3))])
      data.reindex(x=[0.5, 1, 1.5, 2, 2.5], method='pad')

  This will be especially useful once pandas 0.16 is released, at which point
  xray will immediately support reindexing with
  `method='nearest' <https://github.com/pydata/pandas/pull/9258>`_.
- Use functions that return generic ndarrays with DataArray.groupby.apply and
  Dataset.apply (:issue:`327` and :issue:`329`). Thanks Jeff Gerard!
- Consolidated the functionality of ``dumps`` (writing a dataset to a netCDF3
  bytestring) into :py:meth:`~xray.Dataset.to_netcdf` (:issue:`333`).
- :py:meth:`~xray.Dataset.to_netcdf` now supports writing to groups in netCDF4
  files (:issue:`333`). It also finally has a full docstring -- you should read
  it!
- :py:func:`~xray.open_dataset` and :py:meth:`~xray.Dataset.to_netcdf` now
  work on netCDF3 files when netcdf4-python is not installed as long as scipy
  is available (:issue:`333`).
- The new :py:meth:`Dataset.drop <xray.Dataset.drop>` and
  :py:meth:`DataArray.drop <xray.DataArray.drop>` methods makes it easy to drop
  explicitly listed variables or index labels:

  .. ipython:: python

      # drop variables
      ds = xray.Dataset({'x': 0, 'y': 1})
      ds.drop('x')

      # drop index labels
      arr = xray.DataArray([1, 2, 3], coords=[('x', list('abc'))])
      arr.drop(['a', 'c'], dim='x')

- :py:meth:`~xray.Dataset.broadcast_equals` has been added to correspond to
  the new ``compat`` option.
- Long attributes are now truncated at 500 characters when printing a dataset
  (:issue:`338`). This should make things more convenient for working with
  datasets interactively.
- Added a new documentation example, :ref:`monthly means example`. Thanks Joe
  Hamman!

Bug fixes
~~~~~~~~~

- Several bug fixes related to decoding time units from netCDF files
  (:issue:`316`, :issue:`330`). Thanks Stefan Pfenninger!
- xray no longer requires ``decode_coords=False`` when reading datasets with
  unparseable coordinate attributes (:issue:`308`).
- Fixed ``DataArray.loc`` indexing with ``...`` (:issue:`318`).
- Fixed an edge case that resulting in an error when reindexing
  multi-dimensional variables (:issue:`315`).
- Slicing with negative step sizes (:issue:`312`).
- Invalid conversion of string arrays to numeric dtype (:issue:`305`).
- Fixed``repr()`` on dataset objects with non-standard dates (:issue:`347`).

Deprecations
~~~~~~~~~~~~

- ``dump`` and ``dumps`` have been deprecated in favor of
  :py:meth:`~xray.Dataset.to_netcdf`.
- ``drop_vars`` has been deprecated in favor of :py:meth:`~xray.Dataset.drop`.

Future plans
~~~~~~~~~~~~

The biggest feature I'm excited about working toward in the immediate future
is supporting out-of-core operations in xray using Dask_, a part of the Blaze_
project. For a preview of using Dask with weather data, read
`this blog post`_ by Matthew Rocklin. See :issue:`328` for more details.

.. _Dask: http://dask.pydata.org
.. _Blaze: http://blaze.pydata.org
.. _this blog post: http://matthewrocklin.com/blog/work/2015/02/13/Towards-OOC-Slicing-and-Stacking/

v0.3.2 (23 December, 2014)
--------------------------

This release focused on bug-fixes, speedups and resolving some niggling
inconsistencies.

There are a few cases where the behavior of xray differs from the previous
version. However, I expect that in almost all cases your code will continue to
run unmodified.

.. warning::

    xray now requires pandas v0.15.0 or later. This was necessary for
    supporting TimedeltaIndex without too many painful hacks.

Backwards incompatible changes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Arrays of :py:class:`datetime.datetime` objects are now automatically cast to
  ``datetime64[ns]`` arrays when stored in an xray object, using machinery
  borrowed from pandas:

  .. ipython:: python

      from datetime import datetime
      xray.Dataset({'t': [datetime(2000, 1, 1)]})

- xray now has support (including serialization to netCDF) for
  :py:class:`~pandas.TimedeltaIndex`. :py:class:`datetime.timedelta` objects
  are thus accordingly cast to ``timedelta64[ns]`` objects when appropriate.
- Masked arrays are now properly coerced to use ``NaN`` as a sentinel value
  (:issue:`259`).

Enhancements
~~~~~~~~~~~~

- Due to popular demand, we have added experimental attribute style access as
  a shortcut for dataset variables, coordinates and attributes:

  .. ipython:: python

     ds = xray.Dataset({'tmin': ([], 25, {'units': 'celcius'})})
     ds.tmin.units

  Tab-completion for these variables should work in editors such as IPython.
  However, setting variables or attributes in this fashion is not yet
  supported because there are some unresolved ambiguities (:issue:`300`).
- You can now use a dictionary for indexing with labeled dimensions. This
  provides a safe way to do assignment with labeled dimensions:

  .. ipython:: python

      array = xray.DataArray(np.zeros(5), dims=['x'])
      array[dict(x=slice(3))] = 1
      array

- Non-index coordinates can now be faithfully written to and restored from
  netCDF files. This is done according to CF conventions when possible by
  using the ``coordinates`` attribute on a data variable. When not possible,
  xray defines a global ``coordinates`` attribute.
- Preliminary support for converting ``xray.DataArray`` objects to and from
  CDAT_ ``cdms2`` variables.
- We sped up any operation that involves creating a new Dataset or DataArray
  (e.g., indexing, aggregation, arithmetic) by a factor of 30 to 50%. The full
  speed up requires cyordereddict_ to be installed.

.. _CDAT: http://uvcdat.llnl.gov/
.. _cyordereddict: https://github.com/shoyer/cyordereddict

Bug fixes
~~~~~~~~~

- Fix for ``to_dataframe()`` with 0d string/object coordinates (:issue:`287`)
- Fix for ``to_netcdf`` with 0d string variable (:issue:`284`)
- Fix writing datetime64 arrays to netcdf if NaT is present (:issue:`270`)
- Fix align silently upcasts data arrays when NaNs are inserted (:issue:`264`)

Future plans
~~~~~~~~~~~~

- I am contemplating switching to the terms "coordinate variables" and "data
  variables" instead of the (currently used) "coordinates" and "variables",
  following their use in `CF Conventions`_ (:issue:`293`). This would mostly
  have implications for the documentation, but I would also change the
  ``Dataset`` attribute ``vars`` to ``data``.
- I no longer certain that automatic label alignment for arithmetic would be a
  good idea for xray -- it is a feature from pandas that I have not missed
  (:issue:`186`).
- The main API breakage that I *do* anticipate in the next release is finally
  making all aggregation operations skip missing values by default
  (:issue:`130`). I'm pretty sick of writing ``ds.reduce(np.nanmean, 'time')``.
- The next version of xray (0.4) will remove deprecated features and aliases
  whose use currently raises a warning.

If you have opinions about any of these anticipated changes, I would love to
hear them -- please add a note to any of the referenced GitHub issues.

.. _CF Conventions: http://cfconventions.org/Data/cf-conventions/cf-conventions-1.6/build/cf-conventions.html

v0.3.1 (22 October, 2014)
-------------------------

This is mostly a bug-fix release to make xray compatible with the latest
release of pandas (v0.15).

We added several features to better support working with missing values and
exporting xray objects to pandas. We also reorganized the internal API for
serializing and deserializing datasets, but this change should be almost
entirely transparent to users.

Other than breaking the experimental DataStore API, there should be no
backwards incompatible changes.

New features
~~~~~~~~~~~~

- Added :py:meth:`~xray.Dataset.count` and :py:meth:`~xray.Dataset.dropna`
  methods, copied from pandas, for working with missing values (:issue:`247`,
  :issue:`58`).
- Added :py:meth:`DataArray.to_pandas <xray.DataArray.to_pandas>` for
  converting a data array into the pandas object with the same dimensionality
  (1D to Series, 2D to DataFrame, etc.) (:issue:`255`).
- Support for reading gzipped netCDF3 files (:issue:`239`).
- Reduced memory usage when writing netCDF files (:issue:`251`).
- 'missing_value' is now supported as an alias for the '_FillValue' attribute
  on netCDF variables (:issue:`245`).
- Trivial indexes, equivalent to ``range(n)`` where ``n`` is the length of the
  dimension, are no longer written to disk (:issue:`245`).

Bug fixes
~~~~~~~~~

- Compatibility fixes for pandas v0.15 (:issue:`262`).
- Fixes for display and indexing of ``NaT`` (not-a-time) (:issue:`238`,
  :issue:`240`)
- Fix slicing by label was an argument is a data array (:issue:`250`).
- Test data is now shipped with the source distribution (:issue:`253`).
- Ensure order does not matter when doing arithmetic with scalar data arrays
  (:issue:`254`).
- Order of dimensions preserved with ``DataArray.to_dataframe`` (:issue:`260`).

v0.3 (21 September 2014)
------------------------

New features
~~~~~~~~~~~~

- **Revamped coordinates**: "coordinates" now refer to all arrays that are not
  used to index a dimension. Coordinates are intended to allow for keeping track
  of arrays of metadata that describe the grid on which the points in "variable"
  arrays lie. They are preserved (when unambiguous) even though mathematical
  operations.
- **Dataset math** :py:class:`~xray.Dataset` objects now support all arithmetic
  operations directly. Dataset-array operations map across all dataset
  variables; dataset-dataset operations act on each pair of variables with the
  same name.
- **GroupBy math**: This provides a convenient shortcut for normalizing by the
  average value of a group.
- The dataset ``__repr__`` method has been entirely overhauled; dataset
  objects now show their values when printed.
- You can now index a dataset with a list of variables to return a new dataset:
  ``ds[['foo', 'bar']]``.

Backwards incompatible changes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- ``Dataset.__eq__`` and ``Dataset.__ne__`` are now element-wise operations
  instead of comparing all values to obtain a single boolean. Use the method
  :py:meth:`~xray.Dataset.equals` instead.

Deprecations
~~~~~~~~~~~~

- ``Dataset.noncoords`` is deprecated: use ``Dataset.vars`` instead.
- ``Dataset.select_vars`` deprecated: index a ``Dataset`` with a list of
  variable names instead.
- ``DataArray.select_vars`` and ``DataArray.drop_vars`` deprecated: use
  :py:meth:`~xray.DataArray.reset_coords` instead.

v0.2 (14 August 2014)
---------------------

This is major release that includes some new features and quite a few bug
fixes. Here are the highlights:

- There is now a direct constructor for ``DataArray`` objects, which makes it
  possible to create a DataArray without using a Dataset. This is highlighted
  in the refreshed :doc:`tutorial`.
- You can perform aggregation operations like ``mean`` directly on
  :py:class:`~xray.Dataset` objects, thanks to Joe Hamman. These aggregation
  methods also worked on grouped datasets.
- xray now works on Python 2.6, thanks to Anna Kuznetsova.
- A number of methods and attributes were given more sensible (usually shorter)
  names: ``labeled`` -> ``sel``,  ``indexed`` -> ``isel``, ``select`` ->
  ``select_vars``, ``unselect`` -> ``drop_vars``, ``dimensions`` -> ``dims``,
  ``coordinates`` -> ``coords``, ``attributes`` -> ``attrs``.
- New :py:meth:`~xray.Dataset.load_data` and :py:meth:`~xray.Dataset.close`
  methods for datasets facilitate lower level of control of data loaded from
  disk.

v0.1.1 (20 May 2014)
--------------------

xray 0.1.1 is a bug-fix release that includes changes that should be almost
entirely backwards compatible with v0.1:

- Python 3 support (:issue:`53`)
- Required numpy version relaxed to 1.7 (:issue:`129`)
- Return numpy.datetime64 arrays for non-standard calendars (:issue:`126`)
- Support for opening datasets associated with NetCDF4 groups (:issue:`127`)
- Bug-fixes for concatenating datetime arrays (:issue:`134`)

Special thanks to new contributors Thomas Kluyver, Joe Hamman and Alistair
Miles.

v0.1 (2 May 2014)
-----------------

Initial release.
