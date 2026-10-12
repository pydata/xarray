from __future__ import annotations

import warnings

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from xarray.core.nputils import NumpyVIndexAdapter, _is_contiguous, nanquantile


def test_is_contiguous() -> None:
    assert _is_contiguous([1])
    assert _is_contiguous([1, 2, 3])
    assert not _is_contiguous([1, 3])


def test_vindex() -> None:
    x = np.arange(3 * 4 * 5).reshape((3, 4, 5))
    vindex = NumpyVIndexAdapter(x)

    # getitem
    assert_array_equal(vindex[0], x[0])
    assert_array_equal(vindex[[1, 2], [1, 2]], x[([1, 2], [1, 2])])
    assert vindex[[0, 1], [0, 1], :].shape == (2, 5)
    assert vindex[[0, 1], :, [0, 1]].shape == (2, 4)
    assert vindex[:, [0, 1], [0, 1]].shape == (2, 3)

    # setitem
    vindex[:] = 0
    assert_array_equal(x, np.zeros_like(x))
    # assignment should not raise
    vindex[[0, 1], [0, 1], :] = vindex[[0, 1], [0, 1], :]
    vindex[[0, 1], :, [0, 1]] = vindex[[0, 1], :, [0, 1]]
    vindex[:, [0, 1], [0, 1]] = vindex[:, [0, 1], [0, 1]]


def test_nanquantile_all_nan_slice_does_not_warn() -> None:
    # median_unbiased skips the numbagg optimization and exercises NumPy.
    values = np.array([[np.nan, 1.0], [np.nan, 3.0]])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = nanquantile(values, axis=0, q=0.5, method="median_unbiased")

    assert_array_equal(result, [np.nan, 2.0])
    assert not any("All-NaN slice encountered" in str(w.message) for w in caught)


def test_nanquantile_other_runtime_warnings_not_suppressed(monkeypatch) -> None:
    def warning_quantile(*args, **kwargs):
        warnings.warn("unrelated quantile warning", RuntimeWarning, stacklevel=2)
        return np.array([1.5])

    monkeypatch.setattr(np, "nanquantile", warning_quantile)
    with pytest.warns(RuntimeWarning, match="unrelated quantile warning"):
        nanquantile(np.array([[1.0], [2.0]]), axis=0, q=0.5, method="median_unbiased")
