.. currentmodule:: xarray

Coordinates
===========

Creating coordinates
--------------------

.. autosummary::
   :toctree: ../generated/

   Coordinates
   Coordinates.from_xindex
   Coordinates.from_pandas_multiindex

Attributes
----------

.. autosummary::
   :toctree: ../generated/

   Coordinates.dims
   Coordinates.sizes
   Coordinates.dtypes
   Coordinates.variables
   Coordinates.indexes
   Coordinates.xindexes

Dictionary Interface
--------------------

Coordinates implement the mapping interface with keys given by variable names
and values given by ``DataArray`` objects.

.. autosummary::
   :toctree: ../generated/

   Coordinates.__getitem__
   Coordinates.__setitem__
   Coordinates.__delitem__
   Coordinates.update
   Coordinates.get
   Coordinates.items
   Coordinates.keys
   Coordinates.values

Coordinates contents
--------------------

.. autosummary::
   :toctree: ../generated/

   Coordinates.to_dataset
   Coordinates.to_index
   Coordinates.assign
   Coordinates.merge
   Coordinates.copy

Comparisons
-----------

.. autosummary::
   :toctree: ../generated/

   Coordinates.equals
   Coordinates.identical

Proxies
-------

.. currentmodule:: xarray.core.coordinates

Coordinates that are accessed from the ``coords`` property of Dataset, DataArray
and DataTree objects, respectively.

.. autosummary::
   :toctree: ../generated/

   DatasetCoordinates
   DataArrayCoordinates
   DataTreeCoordinates
