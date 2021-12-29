import numpy as np
import pandas as pd
import pytest

import xarray as xr
from xarray.core.indexes import PandasIndex, PandasMultiIndex, _asarray_tuplesafe
from xarray.core.variable import IndexVariable


def test_asarray_tuplesafe() -> None:
    res = _asarray_tuplesafe(("a", 1))
    assert isinstance(res, np.ndarray)
    assert res.ndim == 0
    assert res.item() == ("a", 1)

    res = _asarray_tuplesafe([(0,), (1,)])
    assert res.shape == (2,)
    assert res[0] == (0,)
    assert res[1] == (1,)


class TestPandasIndex:
    def test_constructor(self) -> None:
        pd_idx = pd.Index([1, 2, 3])
        index = PandasIndex(pd_idx, "x")

        assert index.index is pd_idx
        assert index.dim == "x"

    def test_from_variables(self) -> None:
        var = xr.Variable(
            "x", [1, 2, 3], attrs={"unit": "m"}, encoding={"dtype": np.int32}
        )

        index, index_vars = PandasIndex.from_variables({"x": var})
        xr.testing.assert_identical(var.to_index_variable(), index_vars["x"])
        assert index.dim == "x"
        assert index.index.equals(index_vars["x"].to_index())

        var2 = xr.Variable(("x", "y"), [[1, 2, 3], [4, 5, 6]])
        with pytest.raises(ValueError, match=r".*only accepts one variable.*"):
            PandasIndex.from_variables({"x": var, "foo": var2})

        with pytest.raises(
            ValueError, match=r".*only accepts a 1-dimensional variable.*"
        ):
            PandasIndex.from_variables({"foo": var2})

    def test_from_pandas_index(self) -> None:
        pd_idx = pd.Index([1, 2, 3], name="foo")

        index, index_vars = PandasIndex.from_pandas_index(pd_idx, "x")

        assert index.dim == "x"
        assert index.index is pd_idx
        assert index.index.name == "foo"
        xr.testing.assert_identical(index_vars["foo"], IndexVariable("x", [1, 2, 3]))

        # test no name set for pd.Index
        pd_idx.name = None
        index, index_vars = PandasIndex.from_pandas_index(pd_idx, "x")
        assert "x" in index_vars
        assert index.index is not pd_idx
        assert index.index.name == "x"

    def to_pandas_index(self):
        pd_idx = pd.Index([1, 2, 3], name="foo")
        index = PandasIndex(pd_idx, "x")
        assert index.to_pandas_index() is pd_idx

    def test_query(self) -> None:
        # TODO: add tests that aren't just for edge cases
        index = PandasIndex(pd.Index([1, 2, 3]), "x")
        with pytest.raises(KeyError, match=r"not all values found"):
            index.query({"x": [0]})
        with pytest.raises(KeyError):
            index.query({"x": 0})
        with pytest.raises(ValueError, match=r"does not have a MultiIndex"):
            index.query({"x": {"one": 0}})

    def test_query_datetime(self) -> None:
        index = PandasIndex(
            pd.to_datetime(["2000-01-01", "2001-01-01", "2002-01-01"]), "x"
        )
        actual = index.query({"x": "2001-01-01"})
        expected = (1, None)
        assert actual == expected

        actual = index.query({"x": index.to_pandas_index().to_numpy()[1]})
        assert actual == expected

    def test_query_unsorted_datetime_index_raises(self) -> None:
        index = PandasIndex(pd.to_datetime(["2001", "2000", "2002"]), "x")
        with pytest.raises(KeyError):
            # pandas will try to convert this into an array indexer. We should
            # raise instead, so we can be sure the result of indexing with a
            # slice is always a view.
            index.query({"x": slice("2001", "2002")})

    def test_equals(self) -> None:
        index1 = PandasIndex([1, 2, 3], "x")
        index2 = PandasIndex([1, 2, 3], "x")
        assert index1.equals(index2) is True

    def test_union(self) -> None:
        index1 = PandasIndex([1, 2, 3], "x")
        index2 = PandasIndex([4, 5, 6], "y")
        actual = index1.union(index2)
        assert actual.index.equals(pd.Index([1, 2, 3, 4, 5, 6]))
        assert actual.dim == "x"

    def test_intersection(self) -> None:
        index1 = PandasIndex([1, 2, 3], "x")
        index2 = PandasIndex([2, 3, 4], "y")
        actual = index1.intersection(index2)
        assert actual.index.equals(pd.Index([2, 3]))
        assert actual.dim == "x"

    def test_copy(self) -> None:
        expected = PandasIndex([1, 2, 3], "x")
        actual = expected.copy()

        assert actual.index.equals(expected.index)
        assert actual.index is not expected.index
        assert actual.dim == expected.dim

    def test_getitem(self) -> None:
        pd_idx = pd.Index([1, 2, 3])
        expected = PandasIndex(pd_idx, "x")
        actual = expected[1:]

        assert actual.index.equals(pd_idx[1:])
        assert actual.dim == expected.dim


class TestPandasMultiIndex:
    def test_from_variables(self) -> None:
        v_level1 = xr.Variable(
            "x", [1, 2, 3], attrs={"unit": "m"}, encoding={"dtype": np.int32}
        )
        v_level2 = xr.Variable(
            "x", ["a", "b", "c"], attrs={"unit": "m"}, encoding={"dtype": "U"}
        )

        index, index_vars = PandasMultiIndex.from_variables(
            {"level1": v_level1, "level2": v_level2}
        )

        expected_idx = pd.MultiIndex.from_arrays([v_level1.data, v_level2.data])
        assert index.dim == "x"
        assert index.index.equals(expected_idx)

        assert list(index_vars) == ["x", "level1", "level2"]
        xr.testing.assert_equal(xr.IndexVariable("x", expected_idx), index_vars["x"])
        xr.testing.assert_identical(v_level1.to_index_variable(), index_vars["level1"])
        xr.testing.assert_identical(v_level2.to_index_variable(), index_vars["level2"])

        var = xr.Variable(("x", "y"), [[1, 2, 3], [4, 5, 6]])
        with pytest.raises(
            ValueError, match=r".*only accepts 1-dimensional variables.*"
        ):
            PandasMultiIndex.from_variables({"var": var})

        v_level3 = xr.Variable("y", [4, 5, 6])
        with pytest.raises(ValueError, match=r"unmatched dimensions for variables.*"):
            PandasMultiIndex.from_variables({"level1": v_level1, "level3": v_level3})

    def test_from_pandas_index(self) -> None:
        pd_idx = pd.MultiIndex.from_arrays([[1, 2, 3], [4, 5, 6]], names=("foo", "bar"))

        index, index_vars = PandasMultiIndex.from_pandas_index(pd_idx, "x")

        assert index.dim == "x"
        assert index.index is pd_idx
        assert index.index.names == ("foo", "bar")
        xr.testing.assert_identical(index_vars["x"], IndexVariable("x", pd_idx))
        xr.testing.assert_identical(index_vars["foo"], IndexVariable("x", [1, 2, 3]))
        xr.testing.assert_identical(index_vars["bar"], IndexVariable("x", [4, 5, 6]))

    def test_query(self) -> None:
        index = PandasMultiIndex(
            pd.MultiIndex.from_product([["a", "b"], [1, 2]], names=("one", "two")), "x"
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
