import numpy as np
import pandas as pd
import pytest

from xarray.core.indexes import PandasIndex, PandasMultiIndex, _asarray_tuplesafe


def test_asarray_tuplesafe():
    res = _asarray_tuplesafe(("a", 1))
    assert isinstance(res, np.ndarray)
    assert res.ndim == 0
    assert res.item() == ("a", 1)

    res = _asarray_tuplesafe([(0,), (1,)])
    assert res.shape == (2,)
    assert res[0] == (0,)
    assert res[1] == (1,)


class TestPandasIndex:
    def test_query(self):
        # TODO: add tests that aren't just for edge cases
        index = PandasIndex(pd.Index([1, 2, 3]))
        with pytest.raises(KeyError, match=r"not all values found"):
            index.query({"x": [0]})
        with pytest.raises(KeyError):
            index.query({"x": 0})
        with pytest.raises(ValueError, match=r"does not have a MultiIndex"):
            index.query({"x": {"one": 0}})

    def test_query_datetime(self):
        index = PandasIndex(pd.to_datetime(["2000-01-01", "2001-01-01", "2002-01-01"]))
        actual = index.query({"x": "2001-01-01"})
        expected = (1, None)
        assert actual == expected

        actual = index.query({"x": index.to_pandas_index().to_numpy()[1]})
        assert actual == expected

    def test_query_unsorted_datetime_index_raises(self):
        index = PandasIndex(pd.to_datetime(["2001", "2000", "2002"]))
        with pytest.raises(KeyError):
            # pandas will try to convert this into an array indexer. We should
            # raise instead, so we can be sure the result of indexing with a
            # slice is always a view.
            index.query({"x": slice("2001", "2002")})


class TestPandasMultiIndex:
    def test_query(self):
        index = PandasMultiIndex(
            pd.MultiIndex.from_product([["a", "b"], [1, 2]], names=("one", "two"))
        )
        # test tuples inside slice are considered as scalar indexer values
        assert index.query({"x": slice(("a", 1), ("b", 2))}) == (slice(0, 4), None)

        with pytest.raises(KeyError, match=r"not all values found"):
            index.query({"x": [0]})
        with pytest.raises(KeyError):
            index.query({"x": 0})
        with pytest.raises(ValueError, match=r"cannot provide labels for both.*"):
            index.query({"one": 0, "x": "a"})
        with pytest.raises(ValueError, match=r"invalid multi-index level names"):
            index.query({"x": {"three": 0}})
        with pytest.raises(IndexError):
            index.query({"x": (slice(None), 1, "no_level")})
