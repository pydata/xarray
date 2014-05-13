from collections import OrderedDict
import datetime
import numpy as np
import pandas as pd

from xray import utils
from . import TestCase


class TestAsSafeArray(TestCase):
    def test_as_safe_array(self):
        values = np.arange(5.)
        safe_values = utils.as_safe_array(values)
        safe_values[0] = 5.
        self.assertEqual(values[0], safe_values[0])

        dates = [datetime.datetime(2010, 1, i + 1) for i in range(5)]
        values = np.array(dates).astype('<M8[ns]')
        safe_values = utils.as_safe_array(values)
        safe_values[0] = datetime.datetime(1982, 11, 20)
        self.assertEqual(values[0], safe_values[0])

    def test_as_safe_array_datetime(self):
        dates = [datetime.datetime(2010, 1, i + 1) for i in range(5)]
        values = np.array(dates)
        safe_values = utils.as_safe_array(values)
        safe_values[0] = datetime.datetime(1982, 11, 20)
        # Note that this will fail, because as_safe_array converts
        # datetime obecjts to datetime64 objects, which requires copying
        #self.assertEqual(values.astype('<M8[ns]')[0], safe_values[0])
        self.assertEqual(safe_values.dtype, '<M8[ns]')


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


class TestDictionaries(TestCase):
    def setUp(self):
        self.x = {'a': 'A', 'b': 'B'}
        self.y = {'c': 'C', 'b': 'B'}
        self.z = {'a': 'Z'}

    def test_safe(self):
        # should not raise exception:
        utils.update_safety_check(self.x, self.y)

    def test_unsafe(self):
        with self.assertRaises(ValueError):
            utils.update_safety_check(self.x, self.z)

    def test_ordered_dict_intersection(self):
        self.assertEqual({'a': 'A', 'b': 'B'},
                         utils.ordered_dict_intersection(self.x, self.y))
        self.assertEqual({'b': 'B'},
                          utils.ordered_dict_intersection(self.x, self.z))

    def test_dict_equal(self):
        x = OrderedDict()
        x['a'] = 3
        x['b'] = np.array([1, 2, 3])
        y = OrderedDict()
        y['b'] = np.array([1.0, 2.0, 3.0])
        y['a'] = 3
        self.assertTrue(utils.dict_equal(x, y)) # two nparrays are equal
        y['b'] = [1, 2, 3] # np.array not the same as a list
        self.assertTrue(utils.dict_equal(x, y)) # nparray == list
        x['b'] = [1.0, 2.0, 3.0]
        self.assertTrue(utils.dict_equal(x, y)) # list vs. list
        x['c'] = None
        self.assertFalse(utils.dict_equal(x, y)) # new key in x
        x['c'] = np.nan
        y['c'] = np.nan
        self.assertFalse(utils.dict_equal(x, y)) # as intended, nan != nan
        x['c'] = np.inf
        y['c'] = np.inf
        self.assertTrue(utils.dict_equal(x, y)) # inf == inf
        y = dict(y)
        self.assertTrue(utils.dict_equal(x, y)) # different dictionary types are fine
        y['b'] = 3 * np.arange(3)
        self.assertFalse(utils.dict_equal(x, y)) # not equal when arrays differ

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
