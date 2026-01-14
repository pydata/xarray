import numpy as np
import pytest

from xarray.indexes.nd_point_index import ScipyKDTreeAdapter
from xarray.tests import has_scipy


@pytest.mark.skipif(has_scipy, reason="requires scipy to be missing")
def test_scipy_kdtree_adapter_missing_scipy():
    points = np.random.rand(4, 2)

    with pytest.raises(ImportError, match=r"scipy"):
        ScipyKDTreeAdapter(points, options={})
