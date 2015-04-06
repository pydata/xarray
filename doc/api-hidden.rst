.. Generate API reference pages, but don't display these in tables.
.. This extra page is a work around for sphinx not having any support for
.. hiding an autosummary table.

.. currentmodule:: xray

.. autosummary::
   :toctree: generated/

   auto_combine

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

   Dataset.isnull
   Dataset.notnull
   Dataset.count
   Dataset.dropna
   Dataset.fillna

   core.groupby.DatasetGroupBy.assign
   core.groupby.DatasetGroupBy.assign_coords
   core.groupby.DatasetGroupBy.first
   core.groupby.DatasetGroupBy.last
   core.groupby.DatasetGroupBy.fillna

   Dataset.argsort
   Dataset.clip
   Dataset.conj
   Dataset.conjugate
   Dataset.round
   Dataset.T

   DataArray.ndim
   DataArray.shape
   DataArray.size
   DataArray.dtype
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

   DataArray.isnull
   DataArray.notnull
   DataArray.count
   DataArray.dropna
   DataArray.fillna

   core.groupby.DataArrayGroupBy.assign_coords
   core.groupby.DataArrayGroupBy.first
   core.groupby.DataArrayGroupBy.last
   core.groupby.DataArrayGroupBy.fillna

   DataArray.argsort
   DataArray.clip
   DataArray.conj
   DataArray.conjugate
   DataArray.searchsorted
   DataArray.round
   DataArray.T

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
