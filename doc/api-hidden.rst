.. Generate API reference pages, but don't display these in tables.
.. This extra page is a work around for sphinx not having any support for
.. hiding an autosummary table.

.. currentmodule:: xray

.. autosummary::
   :toctree: generated/

   Dataset.all
   Dataset.any
   Dataset.argmax
   Dataset.argmin
   Dataset.max
   Dataset.min
   Dataset.mean
   Dataset.prod
   Dataset.sum
   Dataset.std
   Dataset.var

   Dataset.isnull
   Dataset.notnull
   Dataset.count
   Dataset.dropna

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
   DataArray.prod
   DataArray.sum
   DataArray.std
   DataArray.var

   DataArray.isnull
   DataArray.notnull
   DataArray.count
   DataArray.dropna

   DataArray.argsort
   DataArray.clip
   DataArray.conj
   DataArray.conjugate
   DataArray.searchsorted
   DataArray.round
   DataArray.T
