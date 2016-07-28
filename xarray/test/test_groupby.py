from xarray.core.groupby import _consolidate_slices

import pytest


def test_consolidate_slices():

  assert _consolidate_slices([slice(3), slice(3, 5)]) == [slice(5)]
  assert _consolidate_slices([slice(2, 3), slice(3, 6)]) == [slice(2, 6)]
  assert (_consolidate_slices([slice(2, 3, 1), slice(3, 6, 1)])
          == [slice(2, 6, 1)])

  slices = [slice(2, 3), slice(5, 6)]
  assert _consolidate_slices(slices) == slices

  with pytest.raises(ValueError):
    _consolidate_slices([slice(3), 4])


# TODO: move other groupby tests from test_dataset and test_dataarray over here
