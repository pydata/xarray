.. currentmodule:: xray

API reference
=============

Dataset
-------

Creating a dataset
~~~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: generated/

   Dataset
   open_dataset

Attributes and underlying data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   Dataset.coords
   Dataset.noncoordinates
   Dataset.dims
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
   Dataset.load_data
   Dataset.iteritems
   Dataset.itervalues
   Dataset.virtual_variables

Comparisons
~~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   Dataset.equals
   Dataset.identical

Selecting
~~~~~~~~~

.. autosummary::
   :toctree: generated/

   Dataset.isel
   Dataset.sel
   Dataset.reindex
   Dataset.reindex_like
   Dataset.rename
   Dataset.select_vars
   Dataset.drop_vars
   Dataset.squeeze
   Dataset.groupby

Computations
~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   Dataset.apply
   Dataset.reduce
   Dataset.all
   Dataset.any
   Dataset.argmax
   Dataset.argmin
   Dataset.max
   Dataset.min
   Dataset.mean
   Dataset.std
   Dataset.sum
   Dataset.var

IO / Conversion
~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   Dataset.to_netcdf
   Dataset.dumps
   Dataset.close
   Dataset.to_dataframe
   Dataset.from_dataframe

Dataset internals
~~~~~~~~~~~~~~~~~

These attributes and classes provide a low-level interface for working
with Dataset variables. In general you should use the Dataset dictionary-
like interface instead and working with DataArray objects:

.. autosummary::
   :toctree: generated/

   Dataset.variables
   Variable
   Index

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
   DataArray.as_index
   DataArray.coords
   DataArray.dims
   DataArray.name
   DataArray.dataset
   DataArray.attrs
   DataArray.encoding
   DataArray.variable

NDArray attributes
~~~~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: generated/

    DataArray.ndim
    DataArray.shape
    DataArray.size
    DataArray.dtype

Selecting
~~~~~~~~~

.. autosummary::
   :toctree: generated/

   DataArray.__getitem__
   DataArray.__setitem__
   DataArray.loc
   DataArray.isel
   DataArray.sel
   DataArray.reindex
   DataArray.reindex_like
   DataArray.rename
   DataArray.select_vars
   DataArray.drop_vars
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
   DataArray.get_axis_num
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
   DataArray.isnull
   DataArray.notnull


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
   DataArray.load_data


Top-level functions
-------------------

.. autosummary::
   :toctree: generated/

   align
