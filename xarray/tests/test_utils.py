from __future__ import absolute_import, division, print_function

import numpy as np
import pandas as pd
import pytest

from datetime import datetime
from xarray.coding.cftimeindex import CFTimeIndex
from xarray.core import duck_array_ops, utils
from xarray.core.options import set_options
from xarray.core.pycompat import OrderedDict
from .test_coding_times import _all_cftime_date_types
from . import (TestCase, requires_dask, assert_array_equal,
               has_cftime_or_netCDF4, has_cftime)


class TestAlias(TestCase):
    def test(self):
        def new_method():
            pass
        old_method = utils.alias(new_method, 'old_method')
        assert 'deprecated' in old_method.__doc__
        with pytest.warns(Warning, match='deprecated'):
            old_method()


def test_safe_cast_to_index():
    dates = pd.date_range('2000-01-01', periods=10)
    x = np.arange(5)
    td = x * np.timedelta64(1, 'D')
    for expected, array in [
            (dates, dates.values),
            (pd.Index(x, dtype=object), x.astype(object)),
            (pd.Index(td), td),
            (pd.Index(td, dtype=object), td.astype(object)),
    ]:
        actual = utils.safe_cast_to_index(array)
        assert_array_equal(expected, actual)
        assert expected.dtype == actual.dtype


@pytest.mark.skipif(not has_cftime_or_netCDF4, reason='cftime not installed')
@pytest.mark.parametrize('enable_cftimeindex', [False, True])
def test_safe_cast_to_index_cftimeindex(enable_cftimeindex):
    date_types = _all_cftime_date_types()
    for date_type in date_types.values():
        dates = [date_type(1, 1, day) for day in range(1, 20)]

        if enable_cftimeindex and has_cftime:
            expected = CFTimeIndex(dates)
        else:
            expected = pd.Index(dates)

        with set_options(enable_cftimeindex=enable_cftimeindex):
            actual = utils.safe_cast_to_index(np.array(dates))
        assert_array_equal(expected, actual)
        assert expected.dtype == actual.dtype
        assert isinstance(actual, type(expected))


# Test that datetime.datetime objects are never used in a CFTimeIndex
@pytest.mark.skipif(not has_cftime_or_netCDF4, reason='cftime not installed')
@pytest.mark.parametrize('enable_cftimeindex', [False, True])
def test_safe_cast_to_index_datetime_datetime(enable_cftimeindex):
    dates = [datetime(1, 1, day) for day in range(1, 20)]

    expected = pd.Index(dates)
    with set_options(enable_cftimeindex=enable_cftimeindex):
        actual = utils.safe_cast_to_index(np.array(dates))
    assert_array_equal(expected, actual)
    assert isinstance(actual, pd.Index)


def test_multiindex_from_product_levels():
    result = utils.multiindex_from_product_levels([['b', 'a'], [1, 3, 2]])
    np.testing.assert_array_equal(
        result.labels, [[0, 0, 0, 1, 1, 1], [0, 1, 2, 0, 1, 2]])
    np.testing.assert_array_equal(result.levels[0], ['b', 'a'])
    np.testing.assert_array_equal(result.levels[1], [1, 3, 2])

    other = pd.MultiIndex.from_product([['b', 'a'], [1, 3, 2]])
    np.testing.assert_array_equal(result.values, other.values)


class TestArrayEquiv(TestCase):
    def test_0d(self):
        # verify our work around for pd.isnull not working for 0-dimensional
        # object arrays
        assert duck_array_ops.array_equiv(0, np.array(0, dtype=object))
        assert duck_array_ops.array_equiv(np.nan,
                                          np.array(np.nan, dtype=object))
        assert not duck_array_ops.array_equiv(0, np.array(1, dtype=object))


class TestDictionaries(TestCase):
    def setUp(self):
        self.x = {'a': 'A', 'b': 'B'}
        self.y = {'c': 'C', 'b': 'B'}
        self.z = {'a': 'Z'}

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

    def test_ordered_dict_intersection(self):
        assert {'b': 'B'} == \
            utils.ordered_dict_intersection(self.x, self.y)
        assert {} == utils.ordered_dict_intersection(self.x, self.z)

    def test_dict_equiv(self):
        x = OrderedDict()
        x['a'] = 3
        x['b'] = np.array([1, 2, 3])
        y = OrderedDict()
        y['b'] = np.array([1.0, 2.0, 3.0])
        y['a'] = 3
        assert utils.dict_equiv(x, y)  # two nparrays are equal
        y['b'] = [1, 2, 3]  # np.array not the same as a list
        assert utils.dict_equiv(x, y)  # nparray == list
        x['b'] = [1.0, 2.0, 3.0]
        assert utils.dict_equiv(x, y)  # list vs. list
        x['c'] = None
        assert not utils.dict_equiv(x, y)  # new key in x
        x['c'] = np.nan
        y['c'] = np.nan
        assert utils.dict_equiv(x, y)  # as intended, nan is nan
        x['c'] = np.inf
        y['c'] = np.inf
        assert utils.dict_equiv(x, y)  # inf == inf
        y = dict(y)
        assert utils.dict_equiv(x, y)  # different dictionary types are fine
        y['b'] = 3 * np.arange(3)
        assert not utils.dict_equiv(x, y)  # not equal when arrays differ

    def test_frozen(self):
        x = utils.Frozen(self.x)
        with pytest.raises(TypeError):
            x['foo'] = 'bar'
        with pytest.raises(TypeError):
            del x['a']
        with pytest.raises(AttributeError):
            x.update(self.y)
        assert x.mapping == self.x
        assert repr(x) in ("Frozen({'a': 'A', 'b': 'B'})",
                           "Frozen({'b': 'B', 'a': 'A'})")

    def test_sorted_keys_dict(self):
        x = {'a': 1, 'b': 2, 'c': 3}
        y = utils.SortedKeysDict(x)
        self.assertItemsEqual(y, ['a', 'b', 'c'])
        assert repr(utils.SortedKeysDict()) == \
            "SortedKeysDict({})"

    def test_chain_map(self):
        m = utils.ChainMap({'x': 0, 'y': 1}, {'x': -100, 'z': 2})
        assert 'x' in m
        assert 'y' in m
        assert 'z' in m
        assert m['x'] == 0
        assert m['y'] == 1
        assert m['z'] == 2
        m['x'] = 100
        assert m['x'] == 100
        assert m.maps[0]['x'] == 100
        self.assertItemsEqual(['x', 'y', 'z'], m)


def test_repr_object():
    obj = utils.ReprObject('foo')
    assert repr(obj) == 'foo'


class Test_is_uniform_and_sorted(TestCase):

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


class Test_hashable(TestCase):

    def test_hashable(self):
        for v in [False, 1, (2, ), (3, 4), 'four']:
            assert utils.hashable(v)
        for v in [[5, 6], ['seven', '8'], {9: 'ten'}]:
            assert not utils.hashable(v)


@requires_dask
def test_dask_array_is_scalar():
    # regression test for GH1684
    import dask.array as da

    y = da.arange(8, chunks=4)
    assert not utils.is_scalar(y)


def test_hidden_key_dict():
    hidden_key = '_hidden_key'
    data = {'a': 1, 'b': 2, hidden_key: 3}
    data_expected = {'a': 1, 'b': 2}
    hkd = utils.HiddenKeyDict(data, [hidden_key])
    assert len(hkd) == 2
    assert hidden_key not in hkd
    for k, v in data_expected.items():
        assert hkd[k] == v
    with pytest.raises(KeyError):
        hkd[hidden_key]
    with pytest.raises(KeyError):
        del hkd[hidden_key]
