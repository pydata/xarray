import copy

import pytest

marray = pytest.importorskip("marray")

import numpy as np

import xarray as xr
from xarray.coding.mask import CfMarrayCoder

try:
    sentinel  # noqa: B018
except NameError:
    # backwards compatibility with python <= 3.14
    from typing_extensions import sentinel

MASKED = sentinel("MASKED")
mxp = marray.masked_namespace(np)


def construct_marray(data, fill_value, dtype):
    mask = np.asarray([v is MASKED for v in data], dtype=bool)
    data = np.asarray([v if v is not MASKED else fill_value for v in data], dtype=dtype)

    return mxp.asarray(data, mask=mask)


@pytest.mark.parametrize(
    ["coding", "data", "expected"],
    (
        pytest.param(
            {"missing_value": 0},
            np.asarray([5, 0, 2, 3, 0, 10], dtype=np.int8),
            construct_marray(
                [5, MASKED, 2, 3, MASKED, 10], dtype=np.int8, fill_value=0
            ),
            id="missing_value",
        ),
        pytest.param(
            {"valid_min": 3},
            np.asarray([5, 0, 2, 3, 0, 10], dtype=np.int8),
            construct_marray(
                [5, MASKED, MASKED, 3, MASKED, 10], dtype=np.int8, fill_value=0
            ),
            id="valid_min",
        ),
        pytest.param(
            {"valid_max": 3},
            np.asarray([5, 0, 2, 3, 0, 10], dtype=np.int8),
            construct_marray([MASKED, 0, 2, 3, 0, MASKED], dtype=np.int8, fill_value=0),
            id="valid_max",
        ),
        pytest.param(
            {"valid_min": 2, "valid_max": 5},
            np.asarray([5, 0, 2, 3, 0, 10], dtype=np.int8),
            construct_marray(
                [5, MASKED, 2, 3, MASKED, MASKED], dtype=np.int8, fill_value=0
            ),
            id="valid_minmax",
        ),
        pytest.param(
            {"valid_range": [2, 5]},
            np.asarray([5, 0, 2, 3, 0, 10], dtype=np.int8),
            construct_marray(
                [5, MASKED, 2, 3, MASKED, MASKED], dtype=np.int8, fill_value=0
            ),
            id="valid_range",
        ),
        pytest.param(
            {"missing_value": 0, "valid_range": [2, 5]},
            np.asarray([5, 0, 2, 3, 0, 10], dtype=np.int8),
            construct_marray(
                [5, MASKED, 2, 3, MASKED, 10], dtype=np.int8, fill_value=0
            ),
            id="valid_range-multiple",
        ),
    ),
)
def test_decode(coding, data, expected) -> None:
    original_coding = copy.deepcopy(coding)
    var = xr.Variable("x", data, coding)

    coder = CfMarrayCoder()

    actual = coder.decode(var)
    actual_data = actual.data.data
    actual_mask = actual.data.mask

    np.testing.assert_equal(actual_mask, expected.mask)
    np.testing.assert_equal(actual_data[~actual_mask], expected.data[~expected.mask])

    assert var.attrs == original_coding
    assert actual.attrs == {}
    assert actual.encoding.get("mask", {}) == original_coding
