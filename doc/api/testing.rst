.. currentmodule:: xarray

Testing
=======

.. autosummary::
   :toctree: ../generated/

   testing.assert_equal
   testing.assert_identical
   testing.assert_allclose
   testing.assert_chunks_equal

Test that two ``DataTree`` objects are similar.

.. autosummary::
   :toctree: ../generated/

   testing.assert_isomorphic
   testing.assert_equal
   testing.assert_identical

Hypothesis Testing Strategies
=============================

.. currentmodule:: xarray

See the :ref:`documentation page on testing <testing.hypothesis>` for a guide on how to use these strategies.

.. warning::
    These strategies should be considered highly experimental, and liable to change at any time.

.. autosummary::
   :toctree: ../generated/

   testing.strategies.supported_dtypes
   testing.strategies.names
   testing.strategies.dimension_names
   testing.strategies.dimension_sizes
   testing.strategies.attrs
   testing.strategies.variables
   testing.strategies.unique_subset_of
