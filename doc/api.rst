.. currentmodule:: xarray

#############
API reference
#############

This page provides an auto-generated summary of xarray's API. For more details
and examples, refer to the relevant chapters in the main part of the
documentation.

See also: :ref:`public api`

Top-level functions
===================

.. autosummary::
   :toctree: generated/

   apply_ufunc
   align
   broadcast
   concat
   merge
   combine_by_coords
   combine_nested
   where
   set_options
   infer_freq
   full_like
   zeros_like
   ones_like
   cov
   corr
   dot
   polyval
   map_blocks
   show_versions
   set_options

Dataset
=======

Creating a dataset
------------------

.. autosummary::
   :toctree: generated/

   Dataset
   decode_cf

Attributes
----------

.. autosummary::
   :toctree: generated/

   Dataset.dims
   Dataset.sizes
   Dataset.data_vars
   Dataset.coords
   Dataset.attrs
   Dataset.encoding
   Dataset.indexes
   Dataset.get_index
   Dataset.chunks
   Dataset.nbytes

Dictionary interface
--------------------

Datasets implement the mapping interface with keys given by variable names
and values given by ``DataArray`` objects.

.. autosummary::
   :toctree: generated/

   Dataset.__getitem__
   Dataset.__setitem__
   Dataset.__delitem__
   Dataset.update
   Dataset.get
   Dataset.items
   Dataset.keys
   Dataset.values

Dataset contents
----------------

.. autosummary::
   :toctree: generated/

   Dataset.copy
   Dataset.assign
   Dataset.assign_coords
   Dataset.assign_attrs
   Dataset.pipe
   Dataset.merge
   Dataset.rename
   Dataset.rename_vars
   Dataset.rename_dims
   Dataset.swap_dims
   Dataset.expand_dims
   Dataset.drop_vars
   Dataset.drop_dims
   Dataset.set_coords
   Dataset.reset_coords

Comparisons
-----------

.. autosummary::
   :toctree: generated/

   Dataset.equals
   Dataset.identical
   Dataset.broadcast_equals

Indexing
--------

.. autosummary::
   :toctree: generated/

   Dataset.loc
   Dataset.isel
   Dataset.sel
   Dataset.drop_sel
   Dataset.head
   Dataset.tail
   Dataset.thin
   Dataset.squeeze
   Dataset.interp
   Dataset.interp_like
   Dataset.reindex
   Dataset.reindex_like
   Dataset.set_index
   Dataset.reset_index
   Dataset.reorder_levels

Missing value handling
----------------------

.. autosummary::
   :toctree: generated/

   Dataset.isnull
   Dataset.notnull
   Dataset.combine_first
   Dataset.count
   Dataset.dropna
   Dataset.fillna
   Dataset.ffill
   Dataset.bfill
   Dataset.interpolate_na
   Dataset.where
   Dataset.isin

Computation
-----------

.. autosummary::
   :toctree: generated/

   Dataset.map
   Dataset.reduce
   Dataset.groupby
   Dataset.groupby_bins
   Dataset.rolling
   Dataset.rolling_exp
   Dataset.weighted
   Dataset.coarsen
   Dataset.resample
   Dataset.diff
   Dataset.quantile
   Dataset.differentiate
   Dataset.integrate
   Dataset.map_blocks
   Dataset.polyfit

**Aggregation**:
:py:attr:`~Dataset.all`
:py:attr:`~Dataset.any`
:py:attr:`~Dataset.argmax`
:py:attr:`~Dataset.argmin`
:py:attr:`~Dataset.idxmax`
:py:attr:`~Dataset.idxmin`
:py:attr:`~Dataset.max`
:py:attr:`~Dataset.mean`
:py:attr:`~Dataset.median`
:py:attr:`~Dataset.min`
:py:attr:`~Dataset.prod`
:py:attr:`~Dataset.sum`
:py:attr:`~Dataset.std`
:py:attr:`~Dataset.var`

**ndarray methods**:
:py:attr:`~Dataset.astype`
:py:attr:`~Dataset.argsort`
:py:attr:`~Dataset.clip`
:py:attr:`~Dataset.conj`
:py:attr:`~Dataset.conjugate`
:py:attr:`~Dataset.imag`
:py:attr:`~Dataset.round`
:py:attr:`~Dataset.real`
:py:attr:`~Dataset.cumsum`
:py:attr:`~Dataset.cumprod`
:py:attr:`~Dataset.rank`

**Grouped operations**:
:py:attr:`~core.groupby.DatasetGroupBy.assign`
:py:attr:`~core.groupby.DatasetGroupBy.assign_coords`
:py:attr:`~core.groupby.DatasetGroupBy.first`
:py:attr:`~core.groupby.DatasetGroupBy.last`
:py:attr:`~core.groupby.DatasetGroupBy.fillna`
:py:attr:`~core.groupby.DatasetGroupBy.where`
:py:attr:`~core.groupby.DatasetGroupBy.quantile`

Reshaping and reorganizing
--------------------------

.. autosummary::
   :toctree: generated/

   Dataset.transpose
   Dataset.stack
   Dataset.unstack
   Dataset.to_stacked_array
   Dataset.shift
   Dataset.roll
   Dataset.pad
   Dataset.sortby
   Dataset.broadcast_like

Plotting
--------

.. autosummary::
   :toctree: generated/
   :template: autosummary/accessor_method.rst

   Dataset.plot.scatter

DataArray
=========

.. autosummary::
   :toctree: generated/

   DataArray

Attributes
----------

.. autosummary::
   :toctree: generated/

   DataArray.values
   DataArray.data
   DataArray.coords
   DataArray.dims
   DataArray.sizes
   DataArray.name
   DataArray.attrs
   DataArray.encoding
   DataArray.indexes
   DataArray.get_index

**ndarray attributes**:
:py:attr:`~DataArray.ndim`
:py:attr:`~DataArray.shape`
:py:attr:`~DataArray.size`
:py:attr:`~DataArray.dtype`
:py:attr:`~DataArray.nbytes`
:py:attr:`~DataArray.chunks`

DataArray contents
------------------

.. autosummary::
   :toctree: generated/

   DataArray.assign_coords
   DataArray.assign_attrs
   DataArray.pipe
   DataArray.rename
   DataArray.swap_dims
   DataArray.expand_dims
   DataArray.drop_vars
   DataArray.reset_coords
   DataArray.copy

**ndarray methods**:
:py:attr:`~DataArray.astype`
:py:attr:`~DataArray.item`


Indexing
--------

.. autosummary::
   :toctree: generated/

   DataArray.__getitem__
   DataArray.__setitem__
   DataArray.loc
   DataArray.isel
   DataArray.sel
   DataArray.drop_sel
   DataArray.head
   DataArray.tail
   DataArray.thin
   DataArray.squeeze
   DataArray.interp
   DataArray.interp_like
   DataArray.reindex
   DataArray.reindex_like
   DataArray.set_index
   DataArray.reset_index
   DataArray.reorder_levels

Missing value handling
----------------------

.. autosummary::
  :toctree: generated/

  DataArray.isnull
  DataArray.notnull
  DataArray.combine_first
  DataArray.count
  DataArray.dropna
  DataArray.fillna
  DataArray.ffill
  DataArray.bfill
  DataArray.interpolate_na
  DataArray.where
  DataArray.isin

Comparisons
-----------

.. autosummary::
   :toctree: generated/

   DataArray.equals
   DataArray.identical
   DataArray.broadcast_equals

Computation
-----------

.. autosummary::
   :toctree: generated/

   DataArray.reduce
   DataArray.groupby
   DataArray.groupby_bins
   DataArray.rolling
   DataArray.rolling_exp
   DataArray.weighted
   DataArray.coarsen
   DataArray.resample
   DataArray.get_axis_num
   DataArray.diff
   DataArray.dot
   DataArray.quantile
   DataArray.differentiate
   DataArray.integrate
   DataArray.polyfit
   DataArray.map_blocks


**Aggregation**:
:py:attr:`~DataArray.all`
:py:attr:`~DataArray.any`
:py:attr:`~DataArray.argmax`
:py:attr:`~DataArray.argmin`
:py:attr:`~DataArray.idxmax`
:py:attr:`~DataArray.idxmin`
:py:attr:`~DataArray.max`
:py:attr:`~DataArray.mean`
:py:attr:`~DataArray.median`
:py:attr:`~DataArray.min`
:py:attr:`~DataArray.prod`
:py:attr:`~DataArray.sum`
:py:attr:`~DataArray.std`
:py:attr:`~DataArray.var`

**ndarray methods**:
:py:attr:`~DataArray.argsort`
:py:attr:`~DataArray.clip`
:py:attr:`~DataArray.conj`
:py:attr:`~DataArray.conjugate`
:py:attr:`~DataArray.imag`
:py:attr:`~DataArray.searchsorted`
:py:attr:`~DataArray.round`
:py:attr:`~DataArray.real`
:py:attr:`~DataArray.T`
:py:attr:`~DataArray.cumsum`
:py:attr:`~DataArray.cumprod`
:py:attr:`~DataArray.rank`

**Grouped operations**:
:py:attr:`~core.groupby.DataArrayGroupBy.assign_coords`
:py:attr:`~core.groupby.DataArrayGroupBy.first`
:py:attr:`~core.groupby.DataArrayGroupBy.last`
:py:attr:`~core.groupby.DataArrayGroupBy.fillna`
:py:attr:`~core.groupby.DataArrayGroupBy.where`
:py:attr:`~core.groupby.DataArrayGroupBy.quantile`


String manipulation
-------------------

.. autosummary::
   :toctree: generated/
   :template: autosummary/accessor_method.rst

   DataArray.str.capitalize
   DataArray.str.center
   DataArray.str.contains
   DataArray.str.count
   DataArray.str.decode
   DataArray.str.encode
   DataArray.str.endswith
   DataArray.str.find
   DataArray.str.get
   DataArray.str.index
   DataArray.str.isalnum
   DataArray.str.isalpha
   DataArray.str.isdecimal
   DataArray.str.isdigit
   DataArray.str.isnumeric
   DataArray.str.isspace
   DataArray.str.istitle
   DataArray.str.isupper
   DataArray.str.len
   DataArray.str.ljust
   DataArray.str.lower
   DataArray.str.lstrip
   DataArray.str.match
   DataArray.str.pad
   DataArray.str.repeat
   DataArray.str.replace
   DataArray.str.rfind
   DataArray.str.rindex
   DataArray.str.rjust
   DataArray.str.rstrip
   DataArray.str.slice
   DataArray.str.slice_replace
   DataArray.str.startswith
   DataArray.str.strip
   DataArray.str.swapcase
   DataArray.str.title
   DataArray.str.translate
   DataArray.str.upper
   DataArray.str.wrap
   DataArray.str.zfill

Datetimelike properties
-----------------------

**Datetime properties**:

.. autosummary::
   :toctree: generated/
   :template: autosummary/accessor_attribute.rst

   DataArray.dt.year
   DataArray.dt.month
   DataArray.dt.day
   DataArray.dt.hour
   DataArray.dt.minute
   DataArray.dt.second
   DataArray.dt.microsecond
   DataArray.dt.nanosecond
   DataArray.dt.weekofyear
   DataArray.dt.week
   DataArray.dt.dayofweek
   DataArray.dt.weekday
   DataArray.dt.weekday_name
   DataArray.dt.dayofyear
   DataArray.dt.quarter
   DataArray.dt.days_in_month
   DataArray.dt.daysinmonth
   DataArray.dt.season
   DataArray.dt.time
   DataArray.dt.is_month_start
   DataArray.dt.is_month_end
   DataArray.dt.is_quarter_end
   DataArray.dt.is_year_start
   DataArray.dt.is_leap_year

**Datetime methods**:

.. autosummary::
   :toctree: generated/
   :template: autosummary/accessor_method.rst

   DataArray.dt.floor
   DataArray.dt.ceil
   DataArray.dt.round
   DataArray.dt.strftime

**Timedelta properties**:

.. autosummary::
   :toctree: generated/
   :template: autosummary/accessor_attribute.rst

   DataArray.dt.days
   DataArray.dt.seconds
   DataArray.dt.microseconds
   DataArray.dt.nanoseconds

**Timedelta methods**:

.. autosummary::
   :toctree: generated/
   :template: autosummary/accessor_method.rst

   DataArray.dt.floor
   DataArray.dt.ceil
   DataArray.dt.round


Reshaping and reorganizing
--------------------------

.. autosummary::
   :toctree: generated/

   DataArray.transpose
   DataArray.stack
   DataArray.unstack
   DataArray.to_unstacked_dataset
   DataArray.shift
   DataArray.roll
   DataArray.pad
   DataArray.sortby
   DataArray.broadcast_like

Plotting
--------

.. autosummary::
   :toctree: generated/
   :template: autosummary/accessor_callable.rst

   DataArray.plot

.. autosummary::
   :toctree: generated/
   :template: autosummary/accessor_method.rst

   DataArray.plot.contourf
   DataArray.plot.contour
   DataArray.plot.hist
   DataArray.plot.imshow
   DataArray.plot.line
   DataArray.plot.pcolormesh
   DataArray.plot.step

.. _api.ufuncs:

Universal functions
===================

.. warning::

   With recent versions of numpy, dask and xarray, NumPy ufuncs are now
   supported directly on all xarray and dask objects. This obviates the need
   for the ``xarray.ufuncs`` module, which should not be used for new code
   unless compatibility with versions of NumPy prior to v1.13 is
   required. They will be removed once support for NumPy prior to
   v1.17 is dropped.

These functions are copied from NumPy, but extended to work on NumPy arrays,
dask arrays and all xarray objects. You can find them in the ``xarray.ufuncs``
module:

:py:attr:`~ufuncs.angle`
:py:attr:`~ufuncs.arccos`
:py:attr:`~ufuncs.arccosh`
:py:attr:`~ufuncs.arcsin`
:py:attr:`~ufuncs.arcsinh`
:py:attr:`~ufuncs.arctan`
:py:attr:`~ufuncs.arctan2`
:py:attr:`~ufuncs.arctanh`
:py:attr:`~ufuncs.ceil`
:py:attr:`~ufuncs.conj`
:py:attr:`~ufuncs.copysign`
:py:attr:`~ufuncs.cos`
:py:attr:`~ufuncs.cosh`
:py:attr:`~ufuncs.deg2rad`
:py:attr:`~ufuncs.degrees`
:py:attr:`~ufuncs.exp`
:py:attr:`~ufuncs.expm1`
:py:attr:`~ufuncs.fabs`
:py:attr:`~ufuncs.fix`
:py:attr:`~ufuncs.floor`
:py:attr:`~ufuncs.fmax`
:py:attr:`~ufuncs.fmin`
:py:attr:`~ufuncs.fmod`
:py:attr:`~ufuncs.fmod`
:py:attr:`~ufuncs.frexp`
:py:attr:`~ufuncs.hypot`
:py:attr:`~ufuncs.imag`
:py:attr:`~ufuncs.iscomplex`
:py:attr:`~ufuncs.isfinite`
:py:attr:`~ufuncs.isinf`
:py:attr:`~ufuncs.isnan`
:py:attr:`~ufuncs.isreal`
:py:attr:`~ufuncs.ldexp`
:py:attr:`~ufuncs.log`
:py:attr:`~ufuncs.log10`
:py:attr:`~ufuncs.log1p`
:py:attr:`~ufuncs.log2`
:py:attr:`~ufuncs.logaddexp`
:py:attr:`~ufuncs.logaddexp2`
:py:attr:`~ufuncs.logical_and`
:py:attr:`~ufuncs.logical_not`
:py:attr:`~ufuncs.logical_or`
:py:attr:`~ufuncs.logical_xor`
:py:attr:`~ufuncs.maximum`
:py:attr:`~ufuncs.minimum`
:py:attr:`~ufuncs.nextafter`
:py:attr:`~ufuncs.rad2deg`
:py:attr:`~ufuncs.radians`
:py:attr:`~ufuncs.real`
:py:attr:`~ufuncs.rint`
:py:attr:`~ufuncs.sign`
:py:attr:`~ufuncs.signbit`
:py:attr:`~ufuncs.sin`
:py:attr:`~ufuncs.sinh`
:py:attr:`~ufuncs.sqrt`
:py:attr:`~ufuncs.square`
:py:attr:`~ufuncs.tan`
:py:attr:`~ufuncs.tanh`
:py:attr:`~ufuncs.trunc`

IO / Conversion
===============

Dataset methods
---------------

.. autosummary::
   :toctree: generated/

   open_dataset
   load_dataset
   open_mfdataset
   open_rasterio
   open_zarr
   Dataset.to_netcdf
   Dataset.to_zarr
   save_mfdataset
   Dataset.to_array
   Dataset.to_dataframe
   Dataset.to_dask_dataframe
   Dataset.to_dict
   Dataset.from_dataframe
   Dataset.from_dict
   Dataset.close
   Dataset.compute
   Dataset.persist
   Dataset.load
   Dataset.chunk
   Dataset.unify_chunks
   Dataset.filter_by_attrs
   Dataset.info

DataArray methods
-----------------

.. autosummary::
   :toctree: generated/

   open_dataarray
   load_dataarray
   DataArray.to_dataset
   DataArray.to_netcdf
   DataArray.to_pandas
   DataArray.to_series
   DataArray.to_dataframe
   DataArray.to_index
   DataArray.to_masked_array
   DataArray.to_cdms2
   DataArray.to_iris
   DataArray.from_iris
   DataArray.to_dict
   DataArray.from_series
   DataArray.from_cdms2
   DataArray.from_dict
   DataArray.close
   DataArray.compute
   DataArray.persist
   DataArray.load
   DataArray.chunk
   DataArray.unify_chunks

Coordinates objects
===================

.. autosummary::
   :toctree: generated/

   core.coordinates.DataArrayCoordinates
   core.coordinates.DatasetCoordinates

GroupBy objects
===============

.. autosummary::
   :toctree: generated/

   core.groupby.DataArrayGroupBy
   core.groupby.DataArrayGroupBy.map
   core.groupby.DataArrayGroupBy.reduce
   core.groupby.DatasetGroupBy
   core.groupby.DatasetGroupBy.map
   core.groupby.DatasetGroupBy.reduce

Rolling objects
===============

.. autosummary::
   :toctree: generated/

   core.rolling.DataArrayRolling
   core.rolling.DataArrayRolling.construct
   core.rolling.DataArrayRolling.reduce
   core.rolling.DatasetRolling
   core.rolling.DatasetRolling.construct
   core.rolling.DatasetRolling.reduce
   core.rolling_exp.RollingExp

Weighted objects
================

.. autosummary::
   :toctree: generated/

   core.weighted.DataArrayWeighted
   core.weighted.DataArrayWeighted.mean
   core.weighted.DataArrayWeighted.sum
   core.weighted.DataArrayWeighted.sum_of_weights
   core.weighted.DatasetWeighted
   core.weighted.DatasetWeighted.mean
   core.weighted.DatasetWeighted.sum
   core.weighted.DatasetWeighted.sum_of_weights


Coarsen objects
===============

.. autosummary::
   :toctree: generated/

   core.rolling.DataArrayCoarsen
   core.rolling.DatasetCoarsen


Resample objects
================

Resample objects also implement the GroupBy interface
(methods like ``map()``, ``reduce()``, ``mean()``, ``sum()``, etc.).

.. autosummary::
   :toctree: generated/

   core.resample.DataArrayResample
   core.resample.DataArrayResample.asfreq
   core.resample.DataArrayResample.backfill
   core.resample.DataArrayResample.interpolate
   core.resample.DataArrayResample.nearest
   core.resample.DataArrayResample.pad
   core.resample.DatasetResample
   core.resample.DatasetResample.asfreq
   core.resample.DatasetResample.backfill
   core.resample.DatasetResample.interpolate
   core.resample.DatasetResample.nearest
   core.resample.DatasetResample.pad

Accessors
=========

.. autosummary::
   :toctree: generated/

   core.accessor_dt.DatetimeAccessor
   core.accessor_dt.TimedeltaAccessor
   core.accessor_str.StringAccessor

Custom Indexes
==============
.. autosummary::
   :toctree: generated/

   CFTimeIndex

Creating custom indexes
-----------------------
.. autosummary::
   :toctree: generated/

   cftime_range

Faceting
--------
.. autosummary::
   :toctree: generated/

   plot.FacetGrid
   plot.FacetGrid.add_colorbar
   plot.FacetGrid.add_legend
   plot.FacetGrid.map
   plot.FacetGrid.map_dataarray
   plot.FacetGrid.map_dataarray_line
   plot.FacetGrid.map_dataset
   plot.FacetGrid.set_axis_labels
   plot.FacetGrid.set_ticks
   plot.FacetGrid.set_titles
   plot.FacetGrid.set_xlabels
   plot.FacetGrid.set_ylabels

Tutorial
========

.. autosummary::
   :toctree: generated/

   tutorial.open_dataset
   tutorial.load_dataset

Testing
=======

.. autosummary::
   :toctree: generated/

   testing.assert_equal
   testing.assert_identical
   testing.assert_allclose
   testing.assert_chunks_equal

Exceptions
==========

.. autosummary::
   :toctree: generated/

   MergeError
   SerializationWarning

Advanced API
============

.. autosummary::
   :toctree: generated/

   Dataset.variables
   DataArray.variable
   Variable
   IndexVariable
   as_variable
   register_dataset_accessor
   register_dataarray_accessor

These backends provide a low-level interface for lazily loading data from
external file-formats or protocols, and can be manually invoked to create
arguments for the ``load_store`` and ``dump_to_store`` Dataset methods:

.. autosummary::
   :toctree: generated/

   backends.NetCDF4DataStore
   backends.H5NetCDFStore
   backends.PydapDataStore
   backends.ScipyDataStore
   backends.FileManager
   backends.CachingFileManager
   backends.DummyFileManager

Deprecated / Pending Deprecation
================================

.. autosummary::
   :toctree: generated/

   Dataset.drop
   DataArray.drop
   Dataset.apply
   core.groupby.DataArrayGroupBy.apply
   core.groupby.DatasetGroupBy.apply
