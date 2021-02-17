import numpy as np
import pytest
from numpy.testing import assert_array_equal

from xarray.core.nputils import NumpyVIndexAdapter, _is_contiguous, rolling_window


def test_is_contiguous():
    assert _is_contiguous([1])
    assert _is_contiguous([1, 2, 3])
    assert not _is_contiguous([1, 3])


def test_vindex():
    x = np.arange(3 * 4 * 5).reshape((3, 4, 5))
    vindex = NumpyVIndexAdapter(x)

    # getitem
    assert_array_equal(vindex[0], x[0])
    assert_array_equal(vindex[[1, 2], [1, 2]], x[[1, 2], [1, 2]])
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


def test_rolling():
    x = np.array([1, 2, 3, 4], dtype=float)

    actual = rolling_window(x, axis=-1, window=3, center=True, fill_value=np.nan)
    expected = np.array(
        [[np.nan, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, np.nan]], dtype=float
    )
    assert_array_equal(actual, expected)

    actual = rolling_window(x, axis=-1, window=3, center=False, fill_value=0.0)
    expected = np.array([[0, 0, 1], [0, 1, 2], [1, 2, 3], [2, 3, 4]], dtype=float)
    assert_array_equal(actual, expected)

    x = np.stack([x, x * 1.1])
    actual = rolling_window(x, axis=-1, window=3, center=False, fill_value=0.0)
    expected = np.stack([expected, expected * 1.1], axis=0)
    assert_array_equal(actual, expected)


@pytest.mark.parametrize("center", [[True, True], [False, False]])
@pytest.mark.parametrize("axis", [(0, 1), (1, 2), (2, 0)])
def test_nd_rolling(center, axis):
    x = np.arange(7 * 6 * 8).reshape(7, 6, 8).astype(float)
    window = [3, 3]
    actual = rolling_window(
        x, axis=axis, window=window, center=center, fill_value=np.nan
    )
    expected = x
    for ax, win, cent in zip(axis, window, center):
        expected = rolling_window(
            expected, axis=ax, window=win, center=cent, fill_value=np.nan
        )
    assert_array_equal(actual, expected)
