from __future__ import annotations

import datetime

import numpy as np
import pandas as pd

from xarray.core.variable import _possibly_convert_objects


def test_possibly_convert_objects_datetime():
    # Array with datetime.datetime
    data = np.array([datetime.datetime(2020, 1, 1), None], dtype=object)
    result = _possibly_convert_objects(data)
    assert result.dtype.kind == "M"
    assert result[0] == np.datetime64("2020-01-01T00:00:00")


def test_possibly_convert_objects_timedelta():
    # Array with datetime.timedelta
    data = np.array([datetime.timedelta(days=1), None], dtype=object)
    result = _possibly_convert_objects(data)
    assert result.dtype.kind == "m"
    assert result[0] == np.timedelta64(1, "D")


def test_possibly_convert_objects_np_datetime():
    # Array with np.datetime64
    data = np.array([np.datetime64("2020-01-01"), None], dtype=object)
    result = _possibly_convert_objects(data)
    assert result.dtype.kind == "M"


def test_possibly_convert_objects_np_timedelta():
    # Array with np.timedelta64
    data = np.array([np.timedelta64(1, "D"), None], dtype=object)
    result = _possibly_convert_objects(data)
    assert result.dtype.kind == "m"


def test_possibly_convert_objects_strings():
    # Array with strings (should NOT be converted/modified)
    data = np.array(["a", "b", "c"], dtype=object)
    result = _possibly_convert_objects(data)
    assert result is data  # Should return the exact same array object
    assert result.dtype == object


def test_possibly_convert_objects_mixed_non_datetime():
    # Array with mixed types, no datetimelike (should NOT be converted/modified)
    data = np.array([1, "a", 2.5, None], dtype=object)
    result = _possibly_convert_objects(data)
    assert result is data
    assert result.dtype == object


def test_possibly_convert_objects_all_nulls_small():
    # Small array with only non-NaT nulls (should NOT be converted/modified)
    data = np.array([None, np.nan], dtype=object)
    result = _possibly_convert_objects(data)
    assert result is data
    assert result.dtype == object

    # But if it has pd.NaT, it should be converted to datetime64
    data_with_nat = np.array([None, np.nan, pd.NaT], dtype=object)
    result_with_nat = _possibly_convert_objects(data_with_nat)
    assert result_with_nat.dtype.kind == "M"


def test_possibly_convert_objects_all_nulls_large():
    # Large array (exceeding the 10000 limit) with only nulls (should NOT be converted/modified)
    data = np.array([None] * 10005, dtype=object)
    result = _possibly_convert_objects(data)
    assert result is data
    assert result.dtype == object


def test_possibly_convert_objects_large_with_datetime_at_end():
    # Large array (exceeding 10000 limit) with nulls first, and a datetime at the end
    data = np.array([None] * 10000 + [datetime.datetime(2020, 1, 1)], dtype=object)
    result = _possibly_convert_objects(data)
    assert result.dtype.kind == "M"
    assert result[-1] == np.datetime64("2020-01-01T00:00:00")


def test_possibly_convert_objects_large_with_non_datetime_at_end():
    # Large array (exceeding 10000 limit) with nulls first, and a string at the end
    data = np.array([None] * 10000 + ["a"], dtype=object)
    result = _possibly_convert_objects(data)
    assert result is data
    assert result.dtype == object


def test_possibly_convert_objects_non_object_dtype():
    # Non-object dtype array (should be converted to a numpy array via pd.Series)
    data = np.array([1, 2, 3], dtype=int)
    result = _possibly_convert_objects(data)
    np.testing.assert_array_equal(result, data)
