.. currentmodule:: xarray

DataArray
=========

.. autosummary::
   :toctree: ../generated/

   DataArray

Attributes
----------

.. autosummary::
   :toctree: ../generated/

   DataArray.values
   DataArray.data
   DataArray.coords
   DataArray.dims
   DataArray.sizes
   DataArray.name
   DataArray.attrs
   DataArray.encoding
   DataArray.indexes
   DataArray.xindexes
   DataArray.chunksizes

ndarray attributes
------------------

.. autosummary::
   :toctree: ../generated/

   DataArray.ndim
   DataArray.nbytes
   DataArray.shape
   DataArray.size
   DataArray.dtype
   DataArray.chunks


DataArray contents
------------------

.. autosummary::
   :toctree: ../generated/

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
   DataArray.drop_attrs
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
   :toctree: ../generated/

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
  :toctree: ../generated/

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
   :toctree: ../generated/

   DataArray.equals
   DataArray.identical
   DataArray.broadcast_equals

Computation
-----------

.. autosummary::
   :toctree: ../generated/

   DataArray.reduce
   DataArray.groupby
   DataArray.groupby_bins
   DataArray.rolling
   DataArray.rolling_exp
   DataArray.cumulative
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
   :toctree: ../generated/

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
   :toctree: ../generated/

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
   :toctree: ../generated/
   :template: autosummary/accessor.rst

   DataArray.str

.. autosummary::
   :toctree: ../generated/
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
   :toctree: ../generated/
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
   DataArray.dt.dayofyear
   DataArray.dt.quarter
   DataArray.dt.days_in_month
   DataArray.dt.daysinmonth
   DataArray.dt.days_in_year
   DataArray.dt.season
   DataArray.dt.time
   DataArray.dt.date
   DataArray.dt.decimal_year
   DataArray.dt.calendar
   DataArray.dt.is_month_start
   DataArray.dt.is_month_end
   DataArray.dt.is_quarter_end
   DataArray.dt.is_year_start
   DataArray.dt.is_leap_year

**Datetime methods**:

.. autosummary::
   :toctree: ../generated/
   :template: autosummary/accessor_method.rst

   DataArray.dt.floor
   DataArray.dt.ceil
   DataArray.dt.isocalendar
   DataArray.dt.round
   DataArray.dt.strftime

**Timedelta properties**:

.. autosummary::
   :toctree: ../generated/
   :template: autosummary/accessor_attribute.rst

   DataArray.dt.days
   DataArray.dt.seconds
   DataArray.dt.microseconds
   DataArray.dt.nanoseconds
   DataArray.dt.total_seconds

**Timedelta methods**:

.. autosummary::
   :toctree: ../generated/
   :template: autosummary/accessor_method.rst

   DataArray.dt.floor
   DataArray.dt.ceil
   DataArray.dt.round


Reshaping and reorganizing
--------------------------

.. autosummary::
   :toctree: ../generated/

   DataArray.transpose
   DataArray.stack
   DataArray.unstack
   DataArray.to_unstacked_dataset
   DataArray.shift
   DataArray.roll
   DataArray.pad
   DataArray.sortby
   DataArray.broadcast_like
