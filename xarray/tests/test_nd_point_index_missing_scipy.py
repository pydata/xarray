import sys
import numpy as np
import pytest

from xarray.indexes.nd_point_index import ScipyKDTreeAdapter


def test_scipy_kdtree_adapter_missing_scipy(monkeypatch):
    # Simulate scipy not being installed (fully, including submodules)
    monkeypatch.setitem(sys.modules, "scipy", None)
    monkeypatch.setitem(sys.modules, "scipy.spatial", None)

    # Minimal valid points array: 4 points in 2D
    points = np.random.rand(4, 2)

    with pytest.raises(
        ImportError,
        match=r"scipy.*ScipyKDTreeAdapter|ScipyKDTreeAdapter.*scipy",
    ):
        ScipyKDTreeAdapter(points, options={})
