from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import pickle
import pytest

import numpy as np
import pandas as pd

from xarray.core import duck_array_ops, utils
from xarray.core.pycompat import OrderedDict
from . import TestCase, requires_dask


class TestAlias(TestCase):
    def test(self):
        def new_method():
            pass
        old_method = utils.alias(new_method, 'old_method')
        assert 'deprecated' in old_method.__doc__
        with self.assertWarns('deprecated'):
            old_method()


class TestSafeCastToIndex(TestCase):
    def test(self):
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
            self.assertArrayEqual(expected, actual)
            self.assertEqual(expected.dtype, actual.dtype)


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
        self.assertTrue(
            duck_array_ops.array_equiv(0, np.array(0, dtype=object)))
        self.assertTrue(
            duck_array_ops.array_equiv(np.nan, np.array(np.nan, dtype=object)))
        self.assertFalse(
            duck_array_ops.array_equiv(0, np.array(1, dtype=object)))


class TestDictionaries(TestCase):
    def setUp(self):
        self.x = {'a': 'A', 'b': 'B'}
        self.y = {'c': 'C', 'b': 'B'}
        self.z = {'a': 'Z'}

    def test_equivalent(self):
        self.assertTrue(utils.equivalent(0, 0))
        self.assertTrue(utils.equivalent(np.nan, np.nan))
        self.assertTrue(utils.equivalent(0, np.array(0.0)))
        self.assertTrue(utils.equivalent([0], np.array([0])))
        self.assertTrue(utils.equivalent(np.array([0]), [0]))
        self.assertTrue(utils.equivalent(np.arange(3), 1.0 * np.arange(3)))
        self.assertFalse(utils.equivalent(0, np.zeros(3)))

    def test_safe(self):
        # should not raise exception:
        utils.update_safety_check(self.x, self.y)

    def test_unsafe(self):
        with pytest.raises(ValueError):
            utils.update_safety_check(self.x, self.z)

    def test_ordered_dict_intersection(self):
        self.assertEqual({'b': 'B'},
                         utils.ordered_dict_intersection(self.x, self.y))
        self.assertEqual({}, utils.ordered_dict_intersection(self.x, self.z))

    def test_dict_equiv(self):
        x = OrderedDict()
        x['a'] = 3
        x['b'] = np.array([1, 2, 3])
        y = OrderedDict()
        y['b'] = np.array([1.0, 2.0, 3.0])
        y['a'] = 3
        self.assertTrue(utils.dict_equiv(x, y))  # two nparrays are equal
        y['b'] = [1, 2, 3]  # np.array not the same as a list
        self.assertTrue(utils.dict_equiv(x, y))  # nparray == list
        x['b'] = [1.0, 2.0, 3.0]
        self.assertTrue(utils.dict_equiv(x, y))  # list vs. list
        x['c'] = None
        self.assertFalse(utils.dict_equiv(x, y))  # new key in x
        x['c'] = np.nan
        y['c'] = np.nan
        self.assertTrue(utils.dict_equiv(x, y))  # as intended, nan is nan
        x['c'] = np.inf
        y['c'] = np.inf
        self.assertTrue(utils.dict_equiv(x, y))  # inf == inf
        y = dict(y)
        self.assertTrue(utils.dict_equiv(x, y))  # different dictionary types are fine
        y['b'] = 3 * np.arange(3)
        self.assertFalse(utils.dict_equiv(x, y))  # not equal when arrays differ

    def test_frozen(self):
        x = utils.Frozen(self.x)
        with pytest.raises(TypeError):
            x['foo'] = 'bar'
        with pytest.raises(TypeError):
            del x['a']
        with pytest.raises(AttributeError):
            x.update(self.y)
        self.assertEqual(x.mapping, self.x)
        self.assertIn(repr(x), ("Frozen({'a': 'A', 'b': 'B'})",
                                "Frozen({'b': 'B', 'a': 'A'})"))

    def test_sorted_keys_dict(self):
        x = {'a': 1, 'b': 2, 'c': 3}
        y = utils.SortedKeysDict(x)
        self.assertItemsEqual(y, ['a', 'b', 'c'])
        self.assertEqual(repr(utils.SortedKeysDict()),
                         "SortedKeysDict({})")

    def test_chain_map(self):
        m = utils.ChainMap({'x': 0, 'y': 1}, {'x': -100, 'z': 2})
        self.assertIn('x', m)
        self.assertIn('y', m)
        self.assertIn('z', m)
        self.assertEqual(m['x'], 0)
        self.assertEqual(m['y'], 1)
        self.assertEqual(m['z'], 2)
        m['x'] = 100
        self.assertEqual(m['x'], 100)
        self.assertEqual(m.maps[0]['x'], 100)
        self.assertItemsEqual(['x', 'y', 'z'], m)


def test_repr_object():
    obj = utils.ReprObject('foo')
    assert repr(obj) == 'foo'


class Test_is_uniform_and_sorted(TestCase):

    def test_sorted_uniform(self):
        self.assertTrue(utils.is_uniform_spaced(np.arange(5)))

    def test_sorted_not_uniform(self):
        self.assertEqual(False, utils.is_uniform_spaced([-2, 1, 89]))

    def test_not_sorted_uniform(self):
        self.assertEqual(False, utils.is_uniform_spaced([1, -1, 3]))

    def test_not_sorted_not_uniform(self):
        self.assertEqual(False, utils.is_uniform_spaced([4, 1, 89]))

    def test_two_numbers(self):
        self.assertTrue(utils.is_uniform_spaced([0, 1.7]))

    def test_relative_tolerance(self):
        self.assertTrue(utils.is_uniform_spaced([0, 0.97, 2], rtol=0.1))


class Test_hashable(TestCase):

    def test_hashable(self):
        for v in [False, 1, (2, ), (3, 4), 'four']:
            self.assertTrue(utils.hashable(v))
        for v in [[5, 6], ['seven', '8'], {9: 'ten'}]:
            self.assertFalse(utils.hashable(v))


@requires_dask
def test_dask_array_is_scalar():
    # regression test for GH1684
    import dask.array as da

    y = da.arange(8, chunks=4)
    assert not utils.is_scalar(y)
