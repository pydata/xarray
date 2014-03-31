.. currentmodule:: xray

API reference
=============

Dataset
-------

Creating a dataset
~~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: generated/

   Dataset
   open_dataset
   Dataset.concat

Attributes and underlying data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   Dataset.variables
   Dataset.virtual_variables
   Dataset.coordinates
   Dataset.noncoordinates
   Dataset.dimensions
   Dataset.attributes

Dataset contents
~~~~~~~~~~~~~~~~

Datasets implement the mapping interface with keys given by variable names
and values given by ``DataArray`` objects.

.. autosummary::
   :toctree: generated/

   Dataset.__getitem__
   Dataset.__setitem__
   Dataset.__delitem__
   Dataset.update
   Dataset.merge
   Dataset.copy
   Dataset.iteritems
   Dataset.itervalues

Selecting
~~~~~~~~~

.. autosummary::
   :toctree: generated/

   Dataset.indexed_by
   Dataset.labeled_by
   Dataset.reindex
   Dataset.reindex_like
   Dataset.rename
   Dataset.select
   Dataset.unselect
   Dataset.squeeze
   Dataset.groupby

IO / Conversion
~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   Dataset.dump
   Dataset.dumps
   Dataset.dump_to_store
   Dataset.to_dataframe
   Dataset.from_dataframe


DataArray
---------

.. autosummary::
   :toctree: generated/

   DataArray

Selecting
~~~~~~~~~

.. autosummary::
   :toctree: generated/

   DataArray.loc
   DataArray.indexed_by
   DataArray.labeled_by
   DataArray.reindex
   DataArray.reindex_like
   DataArray.rename
   DataArray.select
   DataArray.unselect
   DataArray.squeeze

Group operations
~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   DataArray.groupby
   DataArray.concat

Manipulating data
~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   DataArray.transpose
   DataArray.T
   DataArray.reduce
   DataArray.all
   DataArray.any
   DataArray.argmax
   DataArray.argmin
   DataArray.cumprod
   DataArray.cumsum
   DataArray.max
   DataArray.min
   DataArray.mean
   DataArray.prod
   DataArray.ptp
   DataArray.std
   DataArray.sum
   DataArray.var

IO / Conversion
~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   DataArray.to_dataframe
   DataArray.to_series
   DataArray.from_series
   DataArray.copy


XArray
------

`XArray` objects provide a low-level interface for manipulating the contents
of `Dataset` objects. Essentially, they are `DatasetArray`s without coordinate
labels.

.. autosummary::
   :toctree: generated/

   XArray

Top-level functions
-------------------

.. autosummary::
   :toctree: generated/

   as_xarray
   broadcast_xarrays
   xarray_equal
   align
   encode_cf_datetime
   decode_cf_datetime
