from datetime import datetime
from typing import Hashable

import numpy as np
import pandas as pd
import pytest

from xarray.coding.cftimeindex import CFTimeIndex
from xarray.core import duck_array_ops, utils
from xarray.core.utils import either_dict_or_kwargs

from . import assert_array_equal, requires_cftime, requires_dask
from .test_coding_times import _all_cftime_date_types


class TestAlias:
    def test(self):
        def new_method():
            pass

        old_method = utils.alias(new_method, "old_method")
        assert "deprecated" in old_method.__doc__
        with pytest.warns(Warning, match="deprecated"):
            old_method()


def test_safe_cast_to_index():
    dates = pd.date_range("2000-01-01", periods=10)
    x = np.arange(5)
    td = x * np.timedelta64(1, "D")
    for expected, array in [
        (dates, dates.values),
        (pd.Index(x, dtype=object), x.astype(object)),
        (pd.Index(td), td),
        (pd.Index(td, dtype=object), td.astype(object)),
    ]:
        actual = utils.safe_cast_to_index(array)
        assert_array_equal(expected, actual)
        assert expected.dtype == actual.dtype


@pytest.mark.parametrize(
    "a, b, expected", [["a", "b", np.array(["a", "b"])], [1, 2, pd.Index([1, 2])]]
)
def test_maybe_coerce_to_str(a, b, expected):

    a = np.array([a])
    b = np.array([b])
    index = pd.Index(a).append(pd.Index(b))

    actual = utils.maybe_coerce_to_str(index, [a, b])

    assert_array_equal(expected, actual)
    assert expected.dtype == actual.dtype


def test_maybe_coerce_to_str_minimal_str_dtype():

    a = np.array(["a", "a_long_string"])
    index = pd.Index(["a"])

    actual = utils.maybe_coerce_to_str(index, [a])
    expected = np.array("a")

    assert_array_equal(expected, actual)
    assert expected.dtype == actual.dtype


@requires_cftime
def test_safe_cast_to_index_cftimeindex():
    date_types = _all_cftime_date_types()
    for date_type in date_types.values():
        dates = [date_type(1, 1, day) for day in range(1, 20)]
        expected = CFTimeIndex(dates)
        actual = utils.safe_cast_to_index(np.array(dates))
        assert_array_equal(expected, actual)
        assert expected.dtype == actual.dtype
        assert isinstance(actual, type(expected))


# Test that datetime.datetime objects are never used in a CFTimeIndex
@requires_cftime
def test_safe_cast_to_index_datetime_datetime():
    dates = [datetime(1, 1, day) for day in range(1, 20)]

    expected = pd.Index(dates)
    actual = utils.safe_cast_to_index(np.array(dates))
    assert_array_equal(expected, actual)
    assert isinstance(actual, pd.Index)


def test_multiindex_from_product_levels():
    result = utils.multiindex_from_product_levels(
        [pd.Index(["b", "a"]), pd.Index([1, 3, 2])]
    )
    np.testing.assert_array_equal(
        result.codes, [[0, 0, 0, 1, 1, 1], [0, 1, 2, 0, 1, 2]]
    )
    np.testing.assert_array_equal(result.levels[0], ["b", "a"])
    np.testing.assert_array_equal(result.levels[1], [1, 3, 2])

    other = pd.MultiIndex.from_product([["b", "a"], [1, 3, 2]])
    np.testing.assert_array_equal(result.values, other.values)


def test_multiindex_from_product_levels_non_unique():
    result = utils.multiindex_from_product_levels(
        [pd.Index(["b", "a"]), pd.Index([1, 1, 2])]
    )
    np.testing.assert_array_equal(
        result.codes, [[0, 0, 0, 1, 1, 1], [0, 0, 1, 0, 0, 1]]
    )
    np.testing.assert_array_equal(result.levels[0], ["b", "a"])
    np.testing.assert_array_equal(result.levels[1], [1, 2])


class TestArrayEquiv:
    def test_0d(self):
        # verify our work around for pd.isnull not working for 0-dimensional
        # object arrays
        assert duck_array_ops.array_equiv(0, np.array(0, dtype=object))
        assert duck_array_ops.array_equiv(np.nan, np.array(np.nan, dtype=object))
        assert not duck_array_ops.array_equiv(0, np.array(1, dtype=object))


class TestDictionaries:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.x = {"a": "A", "b": "B"}
        self.y = {"c": "C", "b": "B"}
        self.z = {"a": "Z"}

    def test_equivalent(self):
        assert utils.equivalent(0, 0)
        assert utils.equivalent(np.nan, np.nan)
        assert utils.equivalent(0, np.array(0.0))
        assert utils.equivalent([0], np.array([0]))
        assert utils.equivalent(np.array([0]), [0])
        assert utils.equivalent(np.arange(3), 1.0 * np.arange(3))
        assert not utils.equivalent(0, np.zeros(3))

    def test_safe(self):
        # should not raise exception:
        utils.update_safety_check(self.x, self.y)

    def test_unsafe(self):
        with pytest.raises(ValueError):
            utils.update_safety_check(self.x, self.z)

    def test_compat_dict_intersection(self):
        assert {"b": "B"} == utils.compat_dict_intersection(self.x, self.y)
        assert {} == utils.compat_dict_intersection(self.x, self.z)

    def test_compat_dict_union(self):
        assert {"a": "A", "b": "B", "c": "C"} == utils.compat_dict_union(self.x, self.y)
        with pytest.raises(
            ValueError,
            match=r"unsafe to merge dictionaries without "
            "overriding values; conflicting key",
        ):
            utils.compat_dict_union(self.x, self.z)

    def test_dict_equiv(self):
        x = {}
        x["a"] = 3
        x["b"] = np.array([1, 2, 3])
        y = {}
        y["b"] = np.array([1.0, 2.0, 3.0])
        y["a"] = 3
        assert utils.dict_equiv(x, y)  # two nparrays are equal
        y["b"] = [1, 2, 3]  # np.array not the same as a list
        assert utils.dict_equiv(x, y)  # nparray == list
        x["b"] = [1.0, 2.0, 3.0]
        assert utils.dict_equiv(x, y)  # list vs. list
        x["c"] = None
        assert not utils.dict_equiv(x, y)  # new key in x
        x["c"] = np.nan
        y["c"] = np.nan
        assert utils.dict_equiv(x, y)  # as intended, nan is nan
        x["c"] = np.inf
        y["c"] = np.inf
        assert utils.dict_equiv(x, y)  # inf == inf
        y = dict(y)
        assert utils.dict_equiv(x, y)  # different dictionary types are fine
        y["b"] = 3 * np.arange(3)
        assert not utils.dict_equiv(x, y)  # not equal when arrays differ

    def test_frozen(self):
        x = utils.Frozen(self.x)
        with pytest.raises(TypeError):
            x["foo"] = "bar"
        with pytest.raises(TypeError):
            del x["a"]
        with pytest.raises(AttributeError):
            x.update(self.y)
        assert x.mapping == self.x
        assert repr(x) in (
            "Frozen({'a': 'A', 'b': 'B'})",
            "Frozen({'b': 'B', 'a': 'A'})",
        )

    def test_sorted_keys_dict(self):
        x = {"a": 1, "b": 2, "c": 3}
        y = utils.SortedKeysDict(x)
        assert list(y) == ["a", "b", "c"]
        assert repr(utils.SortedKeysDict()) == "SortedKeysDict({})"


def test_repr_object():
    obj = utils.ReprObject("foo")
    assert repr(obj) == "foo"
    assert isinstance(obj, Hashable)
    assert not isinstance(obj, str)


def test_repr_object_magic_methods():
    o1 = utils.ReprObject("foo")
    o2 = utils.ReprObject("foo")
    o3 = utils.ReprObject("bar")
    o4 = "foo"
    assert o1 == o2
    assert o1 != o3
    assert o1 != o4
    assert hash(o1) == hash(o2)
    assert hash(o1) != hash(o3)
    assert hash(o1) != hash(o4)


def test_is_remote_uri():
    assert utils.is_remote_uri("http://example.com")
    assert utils.is_remote_uri("https://example.com")
    assert not utils.is_remote_uri(" http://example.com")
    assert not utils.is_remote_uri("example.nc")


class Test_is_uniform_and_sorted:
    def test_sorted_uniform(self):
        assert utils.is_uniform_spaced(np.arange(5))

    def test_sorted_not_uniform(self):
        assert not utils.is_uniform_spaced([-2, 1, 89])

    def test_not_sorted_uniform(self):
        assert not utils.is_uniform_spaced([1, -1, 3])

    def test_not_sorted_not_uniform(self):
        assert not utils.is_uniform_spaced([4, 1, 89])

    def test_two_numbers(self):
        assert utils.is_uniform_spaced([0, 1.7])

    def test_relative_tolerance(self):
        assert utils.is_uniform_spaced([0, 0.97, 2], rtol=0.1)


class Test_hashable:
    def test_hashable(self):
        for v in [False, 1, (2,), (3, 4), "four"]:
            assert utils.hashable(v)
        for v in [[5, 6], ["seven", "8"], {9: "ten"}]:
            assert not utils.hashable(v)


@requires_dask
def test_dask_array_is_scalar():
    # regression test for GH1684
    import dask.array as da

    y = da.arange(8, chunks=4)
    assert not utils.is_scalar(y)


def test_hidden_key_dict():
    hidden_key = "_hidden_key"
    data = {"a": 1, "b": 2, hidden_key: 3}
    data_expected = {"a": 1, "b": 2}
    hkd = utils.HiddenKeyDict(data, [hidden_key])
    assert len(hkd) == 2
    assert hidden_key not in hkd
    for k, v in data_expected.items():
        assert hkd[k] == v
    with pytest.raises(KeyError):
        hkd[hidden_key]
    with pytest.raises(KeyError):
        del hkd[hidden_key]


def test_either_dict_or_kwargs():

    result = either_dict_or_kwargs(dict(a=1), None, "foo")
    expected = dict(a=1)
    assert result == expected

    result = either_dict_or_kwargs(None, dict(a=1), "foo")
    expected = dict(a=1)
    assert result == expected

    with pytest.raises(ValueError, match=r"foo"):
        result = either_dict_or_kwargs(dict(a=1), dict(a=1), "foo")


@pytest.mark.parametrize(
    ["supplied", "all_", "expected"],
    [
        (list("abc"), list("abc"), list("abc")),
        (["a", ..., "c"], list("abc"), list("abc")),
        (["a", ...], list("abc"), list("abc")),
        (["c", ...], list("abc"), list("cab")),
        ([..., "b"], list("abc"), list("acb")),
        ([...], list("abc"), list("abc")),
    ],
)
def test_infix_dims(supplied, all_, expected):
    result = list(utils.infix_dims(supplied, all_))
    assert result == expected


@pytest.mark.parametrize(
    ["supplied", "all_"], [([..., ...], list("abc")), ([...], list("aac"))]
)
def test_infix_dims_errors(supplied, all_):
    with pytest.raises(ValueError):
        list(utils.infix_dims(supplied, all_))
