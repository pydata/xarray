.. currentmodule:: xarray

.. _api:

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
   infer_freq
   full_like
   zeros_like
   ones_like
   cov
   corr
   cross
   dot
   polyval
   map_blocks
   show_versions
   set_options
   get_options
   unify_chunks

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
   Dataset.dtypes
   Dataset.data_vars
   Dataset.coords
   Dataset.attrs
   Dataset.encoding
   Dataset.indexes
   Dataset.chunks
   Dataset.chunksizes
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
   Dataset.drop_indexes
   Dataset.drop_duplicates
   Dataset.drop_dims
   Dataset.drop_encoding
   Dataset.set_coords
   Dataset.reset_coords
   Dataset.convert_calendar
   Dataset.interp_calendar
   Dataset.get_index

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
   Dataset.drop_isel
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
   Dataset.set_xindex
   Dataset.reorder_levels
   Dataset.query

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
   Dataset.curvefit

Aggregation
-----------

.. autosummary::
   :toctree: generated/

   Dataset.all
   Dataset.any
   Dataset.argmax
   Dataset.argmin
   Dataset.count
   Dataset.idxmax
   Dataset.idxmin
   Dataset.max
   Dataset.min
   Dataset.mean
   Dataset.median
   Dataset.prod
   Dataset.sum
   Dataset.std
   Dataset.var
   Dataset.cumsum
   Dataset.cumprod

ndarray methods
---------------

.. autosummary::
   :toctree: generated/

   Dataset.argsort
   Dataset.astype
   Dataset.clip
   Dataset.conj
   Dataset.conjugate
   Dataset.imag
   Dataset.round
   Dataset.real
   Dataset.rank

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
   DataArray.chunksizes

ndarray attributes
------------------

.. autosummary::
   :toctree: generated/

   DataArray.ndim
   DataArray.nbytes
   DataArray.shape
   DataArray.size
   DataArray.dtype
   DataArray.chunks


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
   DataArray.drop_indexes
   DataArray.drop_duplicates
   DataArray.drop_encoding
   DataArray.reset_coords
   DataArray.copy
   DataArray.convert_calendar
   DataArray.interp_calendar
   DataArray.get_index
   DataArray.astype
   DataArray.item

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
   DataArray.drop_isel
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
   DataArray.set_xindex
   DataArray.reorder_levels
   DataArray.query

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
   DataArray.curvefit

Aggregation
-----------

.. autosummary::
   :toctree: generated/

   DataArray.all
   DataArray.any
   DataArray.argmax
   DataArray.argmin
   DataArray.count
   DataArray.idxmax
   DataArray.idxmin
   DataArray.max
   DataArray.min
   DataArray.mean
   DataArray.median
   DataArray.prod
   DataArray.sum
   DataArray.std
   DataArray.var
   DataArray.cumsum
   DataArray.cumprod

ndarray methods
---------------

.. autosummary::
   :toctree: generated/

   DataArray.argsort
   DataArray.clip
   DataArray.conj
   DataArray.conjugate
   DataArray.imag
   DataArray.searchsorted
   DataArray.round
   DataArray.real
   DataArray.T
   DataArray.rank


String manipulation
-------------------

.. autosummary::
   :toctree: generated/
   :template: autosummary/accessor.rst

   DataArray.str

.. autosummary::
   :toctree: generated/
   :template: autosummary/accessor_method.rst

   DataArray.str.capitalize
   DataArray.str.casefold
   DataArray.str.cat
   DataArray.str.center
   DataArray.str.contains
   DataArray.str.count
   DataArray.str.decode
   DataArray.str.encode
   DataArray.str.endswith
   DataArray.str.extract
   DataArray.str.extractall
   DataArray.str.find
   DataArray.str.findall
   DataArray.str.format
   DataArray.str.get
   DataArray.str.get_dummies
   DataArray.str.index
   DataArray.str.isalnum
   DataArray.str.isalpha
   DataArray.str.isdecimal
   DataArray.str.isdigit
   DataArray.str.islower
   DataArray.str.isnumeric
   DataArray.str.isspace
   DataArray.str.istitle
   DataArray.str.isupper
   DataArray.str.join
   DataArray.str.len
   DataArray.str.ljust
   DataArray.str.lower
   DataArray.str.lstrip
   DataArray.str.match
   DataArray.str.normalize
   DataArray.str.pad
   DataArray.str.partition
   DataArray.str.repeat
   DataArray.str.replace
   DataArray.str.rfind
   DataArray.str.rindex
   DataArray.str.rjust
   DataArray.str.rpartition
   DataArray.str.rsplit
   DataArray.str.rstrip
   DataArray.str.slice
   DataArray.str.slice_replace
   DataArray.str.split
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
   DataArray.dt.dayofweek
   DataArray.dt.weekday
   DataArray.dt.weekday_name
   DataArray.dt.dayofyear
   DataArray.dt.quarter
   DataArray.dt.days_in_month
   DataArray.dt.daysinmonth
   DataArray.dt.season
   DataArray.dt.time
   DataArray.dt.date
   DataArray.dt.calendar
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
   DataArray.dt.isocalendar
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

IO / Conversion
===============

Dataset methods
---------------

.. autosummary::
   :toctree: generated/

   load_dataset
   open_dataset
   open_mfdataset
   open_zarr
   save_mfdataset
   Dataset.as_numpy
   Dataset.from_dataframe
   Dataset.from_dict
   Dataset.to_array
   Dataset.to_dataframe
   Dataset.to_dask_dataframe
   Dataset.to_dict
   Dataset.to_netcdf
   Dataset.to_pandas
   Dataset.to_zarr
   Dataset.chunk
   Dataset.close
   Dataset.compute
   Dataset.filter_by_attrs
   Dataset.info
   Dataset.load
   Dataset.persist
   Dataset.unify_chunks

DataArray methods
-----------------

.. autosummary::
   :toctree: generated/

   load_dataarray
   open_dataarray
   DataArray.as_numpy
   DataArray.from_cdms2
   DataArray.from_dict
   DataArray.from_iris
   DataArray.from_series
   DataArray.to_cdms2
   DataArray.to_dask_dataframe
   DataArray.to_dataframe
   DataArray.to_dataset
   DataArray.to_dict
   DataArray.to_index
   DataArray.to_iris
   DataArray.to_masked_array
   DataArray.to_netcdf
   DataArray.to_numpy
   DataArray.to_pandas
   DataArray.to_series
   DataArray.to_zarr
   DataArray.chunk
   DataArray.close
   DataArray.compute
   DataArray.persist
   DataArray.load
   DataArray.unify_chunks

Coordinates objects
===================

Dataset
-------

.. autosummary::
   :toctree: generated/

   core.coordinates.DatasetCoordinates
   core.coordinates.DatasetCoordinates.dtypes

DataArray
---------

.. autosummary::
   :toctree: generated/

   core.coordinates.DataArrayCoordinates
   core.coordinates.DataArrayCoordinates.dtypes

Plotting
========

Dataset
-------

.. autosummary::
   :toctree: generated/
   :template: autosummary/accessor_method.rst

   Dataset.plot.scatter
   Dataset.plot.quiver
   Dataset.plot.streamplot

DataArray
---------

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
   DataArray.plot.scatter
   DataArray.plot.surface


Faceting
--------
.. autosummary::
   :toctree: generated/

   plot.FacetGrid
   plot.FacetGrid.add_colorbar
   plot.FacetGrid.add_legend
   plot.FacetGrid.add_quiverkey
   plot.FacetGrid.map
   plot.FacetGrid.map_dataarray
   plot.FacetGrid.map_dataarray_line
   plot.FacetGrid.map_dataset
   plot.FacetGrid.map_plot1d
   plot.FacetGrid.set_axis_labels
   plot.FacetGrid.set_ticks
   plot.FacetGrid.set_titles
   plot.FacetGrid.set_xlabels
   plot.FacetGrid.set_ylabels



GroupBy objects
===============

.. currentmodule:: xarray.core.groupby

Dataset
-------

.. autosummary::
   :toctree: generated/

   DatasetGroupBy
   DatasetGroupBy.map
   DatasetGroupBy.reduce
   DatasetGroupBy.assign
   DatasetGroupBy.assign_coords
   DatasetGroupBy.first
   DatasetGroupBy.last
   DatasetGroupBy.fillna
   DatasetGroupBy.quantile
   DatasetGroupBy.where
   DatasetGroupBy.all
   DatasetGroupBy.any
   DatasetGroupBy.count
   DatasetGroupBy.cumsum
   DatasetGroupBy.cumprod
   DatasetGroupBy.max
   DatasetGroupBy.mean
   DatasetGroupBy.median
   DatasetGroupBy.min
   DatasetGroupBy.prod
   DatasetGroupBy.std
   DatasetGroupBy.sum
   DatasetGroupBy.var
   DatasetGroupBy.dims
   DatasetGroupBy.groups

DataArray
---------

.. autosummary::
   :toctree: generated/

   DataArrayGroupBy
   DataArrayGroupBy.map
   DataArrayGroupBy.reduce
   DataArrayGroupBy.assign_coords
   DataArrayGroupBy.first
   DataArrayGroupBy.last
   DataArrayGroupBy.fillna
   DataArrayGroupBy.quantile
   DataArrayGroupBy.where
   DataArrayGroupBy.all
   DataArrayGroupBy.any
   DataArrayGroupBy.count
   DataArrayGroupBy.cumsum
   DataArrayGroupBy.cumprod
   DataArrayGroupBy.max
   DataArrayGroupBy.mean
   DataArrayGroupBy.median
   DataArrayGroupBy.min
   DataArrayGroupBy.prod
   DataArrayGroupBy.std
   DataArrayGroupBy.sum
   DataArrayGroupBy.var
   DataArrayGroupBy.dims
   DataArrayGroupBy.groups


Rolling objects
===============

.. currentmodule:: xarray.core.rolling

Dataset
-------

.. autosummary::
   :toctree: generated/

   DatasetRolling
   DatasetRolling.construct
   DatasetRolling.reduce
   DatasetRolling.argmax
   DatasetRolling.argmin
   DatasetRolling.count
   DatasetRolling.max
   DatasetRolling.mean
   DatasetRolling.median
   DatasetRolling.min
   DatasetRolling.prod
   DatasetRolling.std
   DatasetRolling.sum
   DatasetRolling.var

DataArray
---------

.. autosummary::
   :toctree: generated/

   DataArrayRolling
   DataArrayRolling.__iter__
   DataArrayRolling.construct
   DataArrayRolling.reduce
   DataArrayRolling.argmax
   DataArrayRolling.argmin
   DataArrayRolling.count
   DataArrayRolling.max
   DataArrayRolling.mean
   DataArrayRolling.median
   DataArrayRolling.min
   DataArrayRolling.prod
   DataArrayRolling.std
   DataArrayRolling.sum
   DataArrayRolling.var

Coarsen objects
===============

Dataset
-------

.. autosummary::
   :toctree: generated/

   DatasetCoarsen
   DatasetCoarsen.all
   DatasetCoarsen.any
   DatasetCoarsen.construct
   DatasetCoarsen.count
   DatasetCoarsen.max
   DatasetCoarsen.mean
   DatasetCoarsen.median
   DatasetCoarsen.min
   DatasetCoarsen.prod
   DatasetCoarsen.reduce
   DatasetCoarsen.std
   DatasetCoarsen.sum
   DatasetCoarsen.var

DataArray
---------

.. autosummary::
   :toctree: generated/

   DataArrayCoarsen
   DataArrayCoarsen.all
   DataArrayCoarsen.any
   DataArrayCoarsen.construct
   DataArrayCoarsen.count
   DataArrayCoarsen.max
   DataArrayCoarsen.mean
   DataArrayCoarsen.median
   DataArrayCoarsen.min
   DataArrayCoarsen.prod
   DataArrayCoarsen.reduce
   DataArrayCoarsen.std
   DataArrayCoarsen.sum
   DataArrayCoarsen.var

Exponential rolling objects
===========================

.. currentmodule:: xarray.core.rolling_exp

.. autosummary::
   :toctree: generated/

   RollingExp
   RollingExp.mean
   RollingExp.sum

Weighted objects
================

.. currentmodule:: xarray.core.weighted

Dataset
-------

.. autosummary::
   :toctree: generated/

   DatasetWeighted
   DatasetWeighted.mean
   DatasetWeighted.quantile
   DatasetWeighted.sum
   DatasetWeighted.std
   DatasetWeighted.var
   DatasetWeighted.sum_of_weights
   DatasetWeighted.sum_of_squares

DataArray
---------

.. autosummary::
   :toctree: generated/

   DataArrayWeighted
   DataArrayWeighted.mean
   DataArrayWeighted.quantile
   DataArrayWeighted.sum
   DataArrayWeighted.std
   DataArrayWeighted.var
   DataArrayWeighted.sum_of_weights
   DataArrayWeighted.sum_of_squares

Resample objects
================

.. currentmodule:: xarray.core.resample

Dataset
-------

.. autosummary::
   :toctree: generated/

   DatasetResample
   DatasetResample.asfreq
   DatasetResample.backfill
   DatasetResample.interpolate
   DatasetResample.nearest
   DatasetResample.pad
   DatasetResample.all
   DatasetResample.any
   DatasetResample.apply
   DatasetResample.assign
   DatasetResample.assign_coords
   DatasetResample.bfill
   DatasetResample.count
   DatasetResample.ffill
   DatasetResample.fillna
   DatasetResample.first
   DatasetResample.last
   DatasetResample.map
   DatasetResample.max
   DatasetResample.mean
   DatasetResample.median
   DatasetResample.min
   DatasetResample.prod
   DatasetResample.quantile
   DatasetResample.reduce
   DatasetResample.std
   DatasetResample.sum
   DatasetResample.var
   DatasetResample.where
   DatasetResample.dims
   DatasetResample.groups


DataArray
---------

.. autosummary::
   :toctree: generated/

   DataArrayResample
   DataArrayResample.asfreq
   DataArrayResample.backfill
   DataArrayResample.interpolate
   DataArrayResample.nearest
   DataArrayResample.pad
   DataArrayResample.all
   DataArrayResample.any
   DataArrayResample.apply
   DataArrayResample.assign_coords
   DataArrayResample.bfill
   DataArrayResample.count
   DataArrayResample.ffill
   DataArrayResample.fillna
   DataArrayResample.first
   DataArrayResample.last
   DataArrayResample.map
   DataArrayResample.max
   DataArrayResample.mean
   DataArrayResample.median
   DataArrayResample.min
   DataArrayResample.prod
   DataArrayResample.quantile
   DataArrayResample.reduce
   DataArrayResample.std
   DataArrayResample.sum
   DataArrayResample.var
   DataArrayResample.where
   DataArrayResample.dims
   DataArrayResample.groups

Accessors
=========

.. currentmodule:: xarray

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
   date_range
   date_range_like

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

   Coordinates
   Dataset.variables
   DataArray.variable
   Variable
   IndexVariable
   as_variable
   Index
   IndexSelResult
   Context
   register_dataset_accessor
   register_dataarray_accessor
   Dataset.set_close
   backends.BackendArray
   backends.BackendEntrypoint
   backends.list_engines
   backends.refresh_engines

Default, pandas-backed indexes built-in Xarray:

   indexes.PandasIndex
   indexes.PandasMultiIndex

These backends provide a low-level interface for lazily loading data from
external file-formats or protocols, and can be manually invoked to create
arguments for the ``load_store`` and ``dump_to_store`` Dataset methods:

.. autosummary::
   :toctree: generated/

   backends.NetCDF4DataStore
   backends.H5NetCDFStore
   backends.PseudoNetCDFDataStore
   backends.PydapDataStore
   backends.ScipyDataStore
   backends.ZarrStore
   backends.FileManager
   backends.CachingFileManager
   backends.DummyFileManager

These BackendEntrypoints provide a basic interface to the most commonly
used filetypes in the xarray universe.

.. autosummary::
   :toctree: generated/

   backends.NetCDF4BackendEntrypoint
   backends.H5netcdfBackendEntrypoint
   backends.PseudoNetCDFBackendEntrypoint
   backends.PydapBackendEntrypoint
   backends.ScipyBackendEntrypoint
   backends.StoreBackendEntrypoint
   backends.ZarrBackendEntrypoint

Deprecated / Pending Deprecation
================================

.. autosummary::
   :toctree: generated/

   Dataset.drop
   DataArray.drop
   Dataset.apply
   core.groupby.DataArrayGroupBy.apply
   core.groupby.DatasetGroupBy.apply

.. autosummary::
   :toctree: generated/
   :template: autosummary/accessor_attribute.rst

   DataArray.dt.weekofyear
   DataArray.dt.week
