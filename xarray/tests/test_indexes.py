from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import pytest

import xarray as xr
from xarray.core.indexes import (
    Index,
    Indexes,
    PandasIndex,
    PandasMultiIndex,
    _asarray_tuplesafe,
)
from xarray.core.variable import IndexVariable, Variable

from . import assert_equal, assert_identical


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
        # pandas has only Float64Index but variable dtype should be preserved
        data = np.array([1.1, 2.2, 3.3], dtype=np.float32)
        var = xr.Variable(
            "x", data, attrs={"unit": "m"}, encoding={"dtype": np.float64}
        )

        index, index_vars = PandasIndex.from_variables({"x": var})
        assert_identical(var.to_index_variable(), index_vars["x"])
        assert index_vars["x"].dtype == var.dtype
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
        assert_identical(index_vars["foo"], IndexVariable("x", [1, 2, 3]))

        # test no name set for pd.Index
        pd_idx.name = None
        index, index_vars = PandasIndex.from_pandas_index(pd_idx, "x")
        assert "x" in index_vars
        assert index.index is not pd_idx
        assert index.index.name == "x"

    def test_create_variables(self) -> None:
        pd_idx = pd.Index([1, 2, 3], name="foo")
        index, _ = PandasIndex.from_pandas_index(pd_idx, "x")
        index_vars = {
            "foo": IndexVariable(
                "x", pd_idx, attrs={"unit": "m"}, encoding={"fill_value": 0}
            )
        }

        actual = index.create_variables(index_vars)
        assert_identical(actual["foo"], index_vars["foo"])

    def test_to_pandas_index(self) -> None:
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

    def test_query_boolean(self) -> None:
        # index should be ignored and indexer dtype should not be coerced
        # see https://github.com/pydata/xarray/issues/5727
        index = PandasIndex(pd.Index([0.0, 2.0, 1.0, 3.0]), "x")
        actual = index.query({"x": [False, True, False, True]})
        expected_dim_indexers = {"x": [False, True, False, True]}
        np.testing.assert_array_equal(
            actual.dim_indexers["x"], expected_dim_indexers["x"]
        )

    def test_query_datetime(self) -> None:
        index = PandasIndex(
            pd.to_datetime(["2000-01-01", "2001-01-01", "2002-01-01"]), "x"
        )
        actual = index.query({"x": "2001-01-01"})
        expected_dim_indexers = {"x": 1}
        assert actual.dim_indexers == expected_dim_indexers

        actual = index.query({"x": index.to_pandas_index().to_numpy()[1]})
        assert actual.dim_indexers == expected_dim_indexers

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

    def test_join(self) -> None:
        index1 = PandasIndex(["a", "aa", "aaa"], "x", coord_dtype="<U3")
        index2 = PandasIndex(["aa", "aaa", "aaaa"], "x", coord_dtype="<U4")

        expected = PandasIndex(["aa", "aaa"], "x")
        actual = index1.join(index2)
        print(actual.index)
        assert actual.equals(expected)
        assert actual.coord_dtype == "<U4"

        expected = PandasIndex(["a", "aa", "aaa", "aaaa"], "x")
        actual = index1.join(index2, how="outer")
        print(actual.index)
        assert actual.equals(expected)
        assert actual.coord_dtype == "<U4"

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

    def test_reindex_like(self) -> None:
        index1 = PandasIndex([0, 1, 2], "x")
        index2 = PandasIndex([1, 2, 3, 4], "x")

        expected = {"x": [1, 2, -1, -1]}
        actual = index1.reindex_like(index2)
        assert actual.keys() == expected.keys()
        np.testing.assert_array_equal(actual["x"], expected["x"])

        index3 = PandasIndex([1, 1, 2], "x")
        with pytest.raises(ValueError, match=r".*index has duplicate values"):
            index3.reindex_like(index2)

    def test_rename(self) -> None:
        index = PandasIndex(pd.Index([1, 2, 3], name="a"), "x", coord_dtype=np.int32)

        # shortcut
        new_index, index_vars = index.rename({}, {})
        assert new_index is index
        assert index_vars == {}

        new_index, index_vars = index.rename({"a": "b"}, {})
        assert new_index.index.name == "b"
        assert new_index.dim == "x"
        assert new_index.coord_dtype == np.int32
        assert_identical(index_vars["b"], IndexVariable("x", [1, 2, 3]))

        new_index, index_vars = index.rename({}, {"x": "y"})
        assert new_index.index.name == "a"
        assert new_index.dim == "y"
        assert new_index.coord_dtype == np.int32
        assert_identical(index_vars["a"], IndexVariable("y", [1, 2, 3]))

    def test_copy(self) -> None:
        expected = PandasIndex([1, 2, 3], "x", coord_dtype=np.int32)
        actual = expected.copy()

        assert actual.index.equals(expected.index)
        assert actual.index is not expected.index
        assert actual.dim == expected.dim
        assert actual.coord_dtype == expected.coord_dtype

    def test_getitem(self) -> None:
        pd_idx = pd.Index([1, 2, 3])
        expected = PandasIndex(pd_idx, "x", coord_dtype=np.int32)
        actual = expected[1:]

        assert actual.index.equals(pd_idx[1:])
        assert actual.dim == expected.dim
        assert actual.coord_dtype == expected.coord_dtype


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
        assert index.index.name == "x"
        assert index.index.names == ["level1", "level2"]

        assert list(index_vars) == ["x", "level1", "level2"]
        assert_equal(xr.IndexVariable("x", expected_idx), index_vars["x"])
        assert_identical(v_level1.to_index_variable(), index_vars["level1"])
        assert_identical(v_level2.to_index_variable(), index_vars["level2"])

        var = xr.Variable(("x", "y"), [[1, 2, 3], [4, 5, 6]])
        with pytest.raises(
            ValueError, match=r".*only accepts 1-dimensional variables.*"
        ):
            PandasMultiIndex.from_variables({"var": var})

        v_level3 = xr.Variable("y", [4, 5, 6])
        with pytest.raises(
            ValueError, match=r"unmatched dimensions for multi-index variables.*"
        ):
            PandasMultiIndex.from_variables({"level1": v_level1, "level3": v_level3})

    def test_from_product_variables(self) -> None:
        prod_vars = {
            "x": xr.Variable("x", pd.Index(["b", "a"]), attrs={"foo": "bar"}),
            "y": xr.Variable("y", pd.Index([1, 3, 2])),
        }

        index, index_vars = PandasMultiIndex.from_product_variables(prod_vars, "z")

        assert index.dim == "z"
        assert index.index.names == ["x", "y"]
        np.testing.assert_array_equal(
            index.index.codes, [[0, 0, 0, 1, 1, 1], [0, 1, 2, 0, 1, 2]]
        )

        assert list(index_vars) == ["z", "x", "y"]
        midx = pd.MultiIndex.from_product([["b", "a"], [1, 3, 2]])
        assert_equal(xr.IndexVariable("z", midx), index_vars["z"])
        assert_identical(
            xr.IndexVariable("z", ["b", "b", "b", "a", "a", "a"], attrs={"foo": "bar"}),
            index_vars["x"],
        )
        assert_identical(xr.IndexVariable("z", [1, 3, 2, 1, 3, 2]), index_vars["y"])

        with pytest.raises(
            ValueError, match=r"conflicting dimensions for multi-index product.*"
        ):
            PandasMultiIndex.from_product_variables(
                {"x": xr.Variable("x", ["a", "b"]), "x2": xr.Variable("x", [1, 2])},
                "z",
            )

    def test_from_product_variables_non_unique(self) -> None:
        prod_vars = {
            "x": xr.Variable("x", pd.Index(["b", "a"]), attrs={"foo": "bar"}),
            "y": xr.Variable("y", pd.Index([1, 1, 2])),
        }

        index, _ = PandasMultiIndex.from_product_variables(prod_vars, "z")

        np.testing.assert_array_equal(
            index.index.codes, [[0, 0, 0, 1, 1, 1], [0, 0, 1, 0, 0, 1]]
        )
        np.testing.assert_array_equal(index.index.levels[0], ["b", "a"])
        np.testing.assert_array_equal(index.index.levels[1], [1, 2])

    def test_from_pandas_index(self) -> None:
        foo_data = np.array([0, 0, 1], dtype="int")
        bar_data = np.array([1.1, 1.2, 1.3], dtype="float64")
        pd_idx = pd.MultiIndex.from_arrays([foo_data, bar_data], names=("foo", "bar"))

        index, index_vars = PandasMultiIndex.from_pandas_index(pd_idx, "x")

        assert index.dim == "x"
        assert index.index.equals(pd_idx)
        assert index.index.names == ("foo", "bar")
        assert index.index.name == "x"
        assert_identical(index_vars["x"], IndexVariable("x", pd_idx))
        assert_identical(index_vars["foo"], IndexVariable("x", foo_data))
        assert_identical(index_vars["bar"], IndexVariable("x", bar_data))
        assert index_vars["foo"].dtype == foo_data.dtype
        assert index_vars["bar"].dtype == bar_data.dtype

        with pytest.raises(ValueError, match=".*conflicting multi-index level name.*"):
            PandasMultiIndex.from_pandas_index(pd_idx, "foo")

    def test_create_variables(self) -> None:
        foo_data = np.array([0, 0, 1], dtype="int")
        bar_data = np.array([1.1, 1.2, 1.3], dtype="float64")
        pd_idx = pd.MultiIndex.from_arrays([foo_data, bar_data], names=("foo", "bar"))
        index_vars = {
            "x": IndexVariable("x", pd_idx),
            "foo": IndexVariable("x", foo_data, attrs={"unit": "m"}),
            "bar": IndexVariable("x", bar_data, encoding={"fill_value": 0}),
        }

        index, _ = PandasMultiIndex.from_pandas_index(pd_idx, "x")
        actual = index.create_variables(index_vars)

        for k, expected in index_vars.items():
            assert_identical(actual[k], expected)

    def test_query(self) -> None:
        index = PandasMultiIndex(
            pd.MultiIndex.from_product([["a", "b"], [1, 2]], names=("one", "two")), "x"
        )

        # test tuples inside slice are considered as scalar indexer values
        actual = index.query({"x": slice(("a", 1), ("b", 2))})
        expected_dim_indexers = {"x": slice(0, 4)}
        assert actual.dim_indexers == expected_dim_indexers

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

    def test_join(self):
        midx = pd.MultiIndex.from_product([["a", "aa"], [1, 2]], names=("one", "two"))
        level_coords_dtype = {"one": "<U2", "two": "i"}
        index1 = PandasMultiIndex(midx, "x", level_coords_dtype=level_coords_dtype)
        index2 = PandasMultiIndex(midx[0:2], "x", level_coords_dtype=level_coords_dtype)

        actual = index1.join(index2)
        assert actual.equals(index2)
        assert actual.level_coords_dtype == level_coords_dtype

        actual = index1.join(index2, how="outer")
        assert actual.equals(index1)
        assert actual.level_coords_dtype == level_coords_dtype

    def test_rename(self) -> None:
        level_coords_dtype = {"one": "<U1", "two": np.int32}
        index = PandasMultiIndex(
            pd.MultiIndex.from_product([["a", "b"], [1, 2]], names=("one", "two")),
            "x",
            level_coords_dtype=level_coords_dtype,
        )

        # shortcut
        new_index, index_vars = index.rename({}, {})
        assert new_index is index
        assert index_vars == {}

        new_index, index_vars = index.rename({"two": "three"}, {})
        assert new_index.index.names == ["one", "three"]
        assert new_index.dim == "x"
        assert new_index.level_coords_dtype == {"one": "<U1", "three": np.int32}
        assert list(index_vars.keys()) == ["x", "one", "three"]
        for v in index_vars.values():
            assert v.dims == ("x",)

        new_index, index_vars = index.rename({}, {"x": "y"})
        assert new_index.index.names == ["one", "two"]
        assert new_index.dim == "y"
        assert new_index.level_coords_dtype == level_coords_dtype
        assert list(index_vars.keys()) == ["y", "one", "two"]
        for v in index_vars.values():
            assert v.dims == ("y",)

    def test_copy(self) -> None:
        level_coords_dtype = {"one": "U<1", "two": np.int32}
        expected = PandasMultiIndex(
            pd.MultiIndex.from_product([["a", "b"], [1, 2]], names=("one", "two")),
            "x",
            level_coords_dtype=level_coords_dtype,
        )
        actual = expected.copy()

        assert actual.index.equals(expected.index)
        assert actual.index is not expected.index
        assert actual.dim == expected.dim
        assert actual.level_coords_dtype == expected.level_coords_dtype


class TestIndexes:
    @pytest.fixture
    def unique_indexes(self) -> List[PandasIndex]:
        x_idx = PandasIndex(pd.Index([1, 2, 3], name="x"), "x")
        y_idx = PandasIndex(pd.Index([4, 5, 6], name="y"), "y")
        z_pd_midx = pd.MultiIndex.from_product(
            [["a", "b"], [1, 2]], names=["one", "two"]
        )
        z_midx = PandasMultiIndex(z_pd_midx, "z")

        return [x_idx, y_idx, z_midx]

    @pytest.fixture
    def indexes(self, unique_indexes) -> Indexes[Index]:
        x_idx, y_idx, z_midx = unique_indexes
        indexes: Dict[Any, Index] = {
            "x": x_idx,
            "y": y_idx,
            "z": z_midx,
            "one": z_midx,
            "two": z_midx,
        }
        variables: Dict[Any, Variable] = {}
        for idx in unique_indexes:
            variables.update(idx.create_variables())

        return Indexes(indexes, variables)

    def test_interface(self, unique_indexes, indexes) -> None:
        x_idx = unique_indexes[0]
        assert list(indexes) == ["x", "y", "z", "one", "two"]
        assert len(indexes) == 5
        assert "x" in indexes
        assert indexes["x"] is x_idx

    def test_variables(self, indexes) -> None:
        assert tuple(indexes.variables) == ("x", "y", "z", "one", "two")

    def test_dims(self, indexes) -> None:
        assert indexes.dims == {"x": 3, "y": 3, "z": 4}

    def test_get_unique(self, unique_indexes, indexes) -> None:
        assert indexes.get_unique() == unique_indexes

    def test_is_multi(self, indexes) -> None:
        assert indexes.is_multi("one") is True
        assert indexes.is_multi("x") is False

    def test_get_all_coords(self, indexes) -> None:
        expected = {
            "z": indexes.variables["z"],
            "one": indexes.variables["one"],
            "two": indexes.variables["two"],
        }
        assert indexes.get_all_coords("one") == expected

        with pytest.raises(ValueError, match="errors must be.*"):
            indexes.get_all_coords("x", errors="invalid")

        with pytest.raises(ValueError, match="no index found.*"):
            indexes.get_all_coords("no_coord")

        assert indexes.get_all_coords("no_coord", errors="ignore") == {}

    def test_group_by_index(self, unique_indexes, indexes):
        expected = [
            (unique_indexes[0], {"x": indexes.variables["x"]}),
            (unique_indexes[1], {"y": indexes.variables["y"]}),
            (
                unique_indexes[2],
                {
                    "z": indexes.variables["z"],
                    "one": indexes.variables["one"],
                    "two": indexes.variables["two"],
                },
            ),
        ]

        assert indexes.group_by_index() == expected

    def test_to_pandas_indexes(self, indexes) -> None:
        pd_indexes = indexes.to_pandas_indexes()
        assert isinstance(pd_indexes, Indexes)
        assert all([isinstance(idx, pd.Index) for idx in pd_indexes.values()])
        assert indexes.variables == pd_indexes.variables
