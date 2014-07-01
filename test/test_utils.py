import numpy as np
import pandas as pd

from xray import utils
from xray.pycompat import OrderedDict
from . import TestCase


class TestSafeCastToIndex(TestCase):
    def test(self):
        dates = pd.date_range('2000-01-01', periods=10)
        x = np.arange(5)
        timedeltas = x * np.timedelta64(1, 'D')
        for expected, array in [
                (dates, dates.values),
                (pd.Index(x, dtype=object), x.astype(object)),
                (pd.Index(timedeltas, dtype=object), timedeltas),
                ]:
            actual = utils.safe_cast_to_index(array)
            self.assertArrayEqual(expected, actual)
            self.assertEqual(expected.dtype, actual.dtype)


class TestArrayEquiv(TestCase):
    def test_0d(self):
        # verify our work around for pd.isnull not working for 0-dimensional
        # object arrays
        self.assertTrue(utils.array_equiv(0, np.array(0, dtype=object)))
        self.assertTrue(
            utils.array_equiv(np.nan, np.array(np.nan, dtype=object)))
        self.assertFalse(
            utils.array_equiv(0, np.array(1, dtype=object)))


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
        with self.assertRaises(ValueError):
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
        self.assertTrue(utils.dict_equiv(x, y)) # two nparrays are equal
        y['b'] = [1, 2, 3] # np.array not the same as a list
        self.assertTrue(utils.dict_equiv(x, y)) # nparray == list
        x['b'] = [1.0, 2.0, 3.0]
        self.assertTrue(utils.dict_equiv(x, y)) # list vs. list
        x['c'] = None
        self.assertFalse(utils.dict_equiv(x, y)) # new key in x
        x['c'] = np.nan
        y['c'] = np.nan
        self.assertTrue(utils.dict_equiv(x, y)) # as intended, nan is nan
        x['c'] = np.inf
        y['c'] = np.inf
        self.assertTrue(utils.dict_equiv(x, y)) # inf == inf
        y = dict(y)
        self.assertTrue(utils.dict_equiv(x, y)) # different dictionary types are fine
        y['b'] = 3 * np.arange(3)
        self.assertFalse(utils.dict_equiv(x, y)) # not equal when arrays differ

    def test_frozen(self):
        x = utils.Frozen(self.x)
        with self.assertRaises(TypeError):
            x['foo'] = 'bar'
        with self.assertRaises(TypeError):
            del x['a']
        with self.assertRaises(AttributeError):
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
