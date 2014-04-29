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

Attributes and underlying data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   Dataset.variables
   Dataset.virtual_variables
   Dataset.coordinates
   Dataset.noncoordinates
   Dataset.dimensions
   Dataset.attrs

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
   Dataset.concat
   Dataset.copy
   Dataset.iteritems
   Dataset.itervalues

Comparisons
~~~~~~~~~~~

   Dataset.equals
   Dataset.identical

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

Backends (experimental)
~~~~~~~~~~~~~~~~~~~~~~~

These backends provide a low-level interface for lazily loading data from
external file-formats or protocols, and can be manually invoked to create
arguments for the `from_store` and `dump_to_store` Dataset methods.

.. autosummary::
   :toctree: generated/

   backends.NetCDF4DataStore
   backends.PydapDataStore
   backends.ScipyDataStore

DataArray
---------

.. autosummary::
   :toctree: generated/

   DataArray

Attributes and underlying data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   DataArray.values
   DataArray.coordinates
   DataArray.name
   DataArray.dataset
   DataArray.attrs

Selecting
~~~~~~~~~

.. autosummary::
   :toctree: generated/

   DataArray.__getitem__
   DataArray.__setitem__
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

Computations
~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   DataArray.transpose
   DataArray.T
   DataArray.reduce
   DataArray.all
   DataArray.any
   DataArray.argmax
   DataArray.argmin
   DataArray.max
   DataArray.min
   DataArray.mean
   DataArray.std
   DataArray.sum
   DataArray.var

Comparisons
~~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   DataArray.equals
   DataArray.identical

IO / Conversion
~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   DataArray.to_dataframe
   DataArray.to_series
   DataArray.from_series
   DataArray.copy


Top-level functions
-------------------

.. autosummary::
   :toctree: generated/

   align
