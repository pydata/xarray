import pytest

pint = pytest.importorskip("pint")

# make sure scalars are converted to 0d arrays so quantities can
# always be treated like ndarrays
unit_registry = pint.UnitRegistry(force_ndarray_like=True)
