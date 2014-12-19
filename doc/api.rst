.. currentmodule:: xray

#############
API reference
#############

This page provides an auto-generated summary of xray's API. For more details
and examples, refer to the relevant chapter in the main part of the
documentation.

Top-level functions
===================

.. autosummary::
   :toctree: generated/

   align
   concat

Dataset
=======

Creating a dataset
------------------

.. autosummary::
   :toctree: generated/

   Dataset
   open_dataset
   decode_cf

Attributes
----------

.. autosummary::
   :toctree: generated/

   Dataset.dims
   Dataset.vars
   Dataset.coords
   Dataset.attrs

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
   Dataset.iteritems
   Dataset.itervalues

Dataset contents
----------------

.. autosummary::
   :toctree: generated/

   Dataset.copy
   Dataset.merge
   Dataset.rename
   Dataset.drop_vars
   Dataset.set_coords
   Dataset.reset_coords

Comparisons
-----------

.. autosummary::
   :toctree: generated/

   Dataset.equals
   Dataset.identical

Indexing
--------

.. autosummary::
   :toctree: generated/

   Dataset.loc
   Dataset.isel
   Dataset.sel
   Dataset.squeeze
   Dataset.reindex
   Dataset.reindex_like

Computation
-----------

.. autosummary::
   :toctree: generated/

   Dataset.apply
   Dataset.reduce
   Dataset.groupby
   Dataset.transpose

**Aggregation**:
:py:attr:`~Dataset.all`
:py:attr:`~Dataset.any`
:py:attr:`~Dataset.argmax`
:py:attr:`~Dataset.argmin`
:py:attr:`~Dataset.max`
:py:attr:`~Dataset.mean`
:py:attr:`~Dataset.min`
:py:attr:`~Dataset.prod`
:py:attr:`~Dataset.sum`
:py:attr:`~Dataset.std`
:py:attr:`~Dataset.var`

**Missing values**:
:py:attr:`~Dataset.isnull`
:py:attr:`~Dataset.notnull`
:py:attr:`~Dataset.count`
:py:attr:`~Dataset.dropna`

**ndarray methods**:
:py:attr:`~Dataset.argsort`
:py:attr:`~Dataset.clip`
:py:attr:`~Dataset.conj`
:py:attr:`~Dataset.conjugate`
:py:attr:`~Dataset.round`
:py:attr:`~Dataset.T`

IO / Conversion
---------------

.. autosummary::
   :toctree: generated/

   Dataset.to_netcdf
   Dataset.to_dataframe
   Dataset.from_dataframe
   Dataset.close
   Dataset.load_data

Backends (experimental)
-----------------------

These backends provide a low-level interface for lazily loading data from
external file-formats or protocols, and can be manually invoked to create
arguments for the ``from_store`` and ``dump_to_store`` Dataset methods.

.. autosummary::
   :toctree: generated/

   backends.NetCDF4DataStore
   backends.PydapDataStore
   backends.ScipyDataStore

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
   DataArray.coords
   DataArray.dims
   DataArray.name
   DataArray.attrs
   DataArray.encoding

**ndarray attributes**:
:py:attr:`~DataArray.ndim`
:py:attr:`~DataArray.shape`
:py:attr:`~DataArray.size`
:py:attr:`~DataArray.dtype`

DataArray contents
------------------

.. autosummary::
   :toctree: generated/

   DataArray.rename
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
   DataArray.squeeze
   DataArray.reindex
   DataArray.reindex_like

Computation
-----------

.. autosummary::
   :toctree: generated/

   DataArray.reduce
   DataArray.groupby
   DataArray.transpose
   DataArray.get_axis_num

**Aggregation**:
:py:attr:`~DataArray.all`
:py:attr:`~DataArray.any`
:py:attr:`~DataArray.argmax`
:py:attr:`~DataArray.argmin`
:py:attr:`~DataArray.max`
:py:attr:`~DataArray.mean`
:py:attr:`~DataArray.min`
:py:attr:`~DataArray.prod`
:py:attr:`~DataArray.sum`
:py:attr:`~DataArray.std`
:py:attr:`~DataArray.var`

**Missing values**:
:py:attr:`~DataArray.isnull`
:py:attr:`~DataArray.notnull`
:py:attr:`~DataArray.count`
:py:attr:`~DataArray.dropna`

**ndarray methods**:
:py:attr:`~DataArray.argsort`
:py:attr:`~DataArray.clip`
:py:attr:`~DataArray.conj`
:py:attr:`~DataArray.conjugate`
:py:attr:`~DataArray.searchsorted`
:py:attr:`~DataArray.round`
:py:attr:`~DataArray.T`

Comparisons
-----------

.. autosummary::
   :toctree: generated/

   DataArray.equals
   DataArray.identical

IO / Conversion
---------------

.. autosummary::
   :toctree: generated/

   DataArray.to_dataset
   DataArray.to_pandas
   DataArray.to_series
   DataArray.to_dataframe
   DataArray.to_index
   DataArray.to_cdms2
   DataArray.from_series
   DataArray.from_cdms2
   DataArray.load_data
