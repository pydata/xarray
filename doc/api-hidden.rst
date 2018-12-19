.. Generate API reference pages, but don't display these in tables.
.. This extra page is a work around for sphinx not having any support for
.. hiding an autosummary table.

.. currentmodule:: xarray

.. autosummary::
   :toctree: generated/

   auto_combine

   Dataset.nbytes
   Dataset.chunks

   Dataset.all
   Dataset.any
   Dataset.argmax
   Dataset.argmin
   Dataset.max
   Dataset.min
   Dataset.mean
   Dataset.median
   Dataset.prod
   Dataset.sum
   Dataset.std
   Dataset.var

   core.groupby.DatasetGroupBy.assign
   core.groupby.DatasetGroupBy.assign_coords
   core.groupby.DatasetGroupBy.first
   core.groupby.DatasetGroupBy.last
   core.groupby.DatasetGroupBy.fillna
   core.groupby.DatasetGroupBy.where

   Dataset.argsort
   Dataset.clip
   Dataset.conj
   Dataset.conjugate
   Dataset.imag
   Dataset.round
   Dataset.real
   Dataset.cumsum
   Dataset.cumprod
   Dataset.rank

   DataArray.ndim
   DataArray.nbytes
   DataArray.shape
   DataArray.size
   DataArray.dtype
   DataArray.nbytes
   DataArray.chunks

   DataArray.astype
   DataArray.item

   DataArray.all
   DataArray.any
   DataArray.argmax
   DataArray.argmin
   DataArray.max
   DataArray.min
   DataArray.mean
   DataArray.median
   DataArray.prod
   DataArray.sum
   DataArray.std
   DataArray.var

   core.groupby.DataArrayGroupBy.assign_coords
   core.groupby.DataArrayGroupBy.first
   core.groupby.DataArrayGroupBy.last
   core.groupby.DataArrayGroupBy.fillna
   core.groupby.DataArrayGroupBy.where

   DataArray.argsort
   DataArray.clip
   DataArray.conj
   DataArray.conjugate
   DataArray.imag
   DataArray.searchsorted
   DataArray.round
   DataArray.real
   DataArray.T
   DataArray.cumsum
   DataArray.cumprod
   DataArray.rank

   ufuncs.angle
   ufuncs.arccos
   ufuncs.arccosh
   ufuncs.arcsin
   ufuncs.arcsinh
   ufuncs.arctan
   ufuncs.arctan2
   ufuncs.arctanh
   ufuncs.ceil
   ufuncs.conj
   ufuncs.copysign
   ufuncs.cos
   ufuncs.cosh
   ufuncs.deg2rad
   ufuncs.degrees
   ufuncs.exp
   ufuncs.expm1
   ufuncs.fabs
   ufuncs.fix
   ufuncs.floor
   ufuncs.fmax
   ufuncs.fmin
   ufuncs.fmod
   ufuncs.fmod
   ufuncs.frexp
   ufuncs.hypot
   ufuncs.imag
   ufuncs.iscomplex
   ufuncs.isfinite
   ufuncs.isinf
   ufuncs.isnan
   ufuncs.isreal
   ufuncs.ldexp
   ufuncs.log
   ufuncs.log10
   ufuncs.log1p
   ufuncs.log2
   ufuncs.logaddexp
   ufuncs.logaddexp2
   ufuncs.logical_and
   ufuncs.logical_not
   ufuncs.logical_or
   ufuncs.logical_xor
   ufuncs.maximum
   ufuncs.minimum
   ufuncs.nextafter
   ufuncs.rad2deg
   ufuncs.radians
   ufuncs.real
   ufuncs.rint
   ufuncs.sign
   ufuncs.signbit
   ufuncs.sin
   ufuncs.sinh
   ufuncs.sqrt
   ufuncs.square
   ufuncs.tan
   ufuncs.tanh
   ufuncs.trunc

   plot.FacetGrid.map_dataarray
   plot.FacetGrid.set_titles
   plot.FacetGrid.set_ticks
   plot.FacetGrid.map

   CFTimeIndex.shift
   CFTimeIndex.to_datetimeindex
