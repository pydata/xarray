from collections import OrderedDict
import netCDF4 as nc4
import numpy as np
import pandas as pd

from xray import utils, XArray
from . import TestCase, ReturnItem


class TestIndexers(TestCase):
    def set_to_zero(self, x, i):
        x = x.copy()
        x[i] = 0
        return x

    def test_expanded_indexer(self):
        x = np.random.randn(10, 11, 12, 13, 14)
        y = np.arange(5)
        I = ReturnItem()
        for i in [I[:], I[...], I[0, :, 10], I[..., 10], I[:5, ..., 0],
                  I[y], I[y, y], I[..., y, y], I[..., 0, 1, 2, 3, 4]]:
            j = utils.expanded_indexer(i, x.ndim)
            self.assertArrayEqual(x[i], x[j])
            self.assertArrayEqual(self.set_to_zero(x, i),
                                  self.set_to_zero(x, j))

    def test_orthogonal_indexer(self):
        x = np.random.randn(10, 11, 12, 13, 14)
        y = np.arange(5)
        I = ReturnItem()
        # orthogonal and numpy indexing should be equivalent, because we only
        # use at most one array and it never in between two slice objects
        # (i.e., we try to avoid numpy's mind-boggling "partial indexing"
        # http://docs.scipy.org/doc/numpy/reference/arrays.indexing.html)
        for i in [I[:], I[0], I[0, 0], I[:5], I[2:5], I[2:5:-1], I[:3, :4],
                  I[:3, 0, :4], I[:3, 0, :4, 0], I[y], I[:, y], I[0, y],
                  I[:2, :3, y], I[0, y, :, :4, 0]]:
            j = utils.orthogonal_indexer(i, x.shape)
            self.assertArrayEqual(x[i], x[j])
            self.assertArrayEqual(self.set_to_zero(x, i),
                                  self.set_to_zero(x, j))
        # for more complicated cases, check orthogonal indexing is still
        # equivalent to slicing
        z = np.arange(2, 8, 2)
        for i, j, shape in [
                (I[y, y], I[:5, :5], (5, 5, 12, 13, 14)),
                (I[y, z], I[:5, 2:8:2], (5, 3, 12, 13, 14)),
                (I[0, y, y], I[0, :5, :5], (5, 5, 13, 14)),
                (I[y, 0, z], I[:5, 0, 2:8:2], (5, 3, 13, 14)),
                (I[y, :, z], I[:5, :, 2:8:2], (5, 11, 3, 13, 14)),
                (I[0, :2, y, y, 0], I[0, :2, :5, :5, 0], (2, 5, 5)),
                (I[0, :, y, :, 0], I[0, :, :5, :, 0], (11, 5, 13)),
                (I[:, :, y, :, 0], I[:, :, :5, :, 0], (10, 11, 5, 13)),
                (I[:, :, y, z, :], I[:, :, :5, 2:8:2], (10, 11, 5, 3, 14))]:
            k = utils.orthogonal_indexer(i, x.shape)
            self.assertEqual(shape, x[k].shape)
            self.assertArrayEqual(x[j], x[k])
            self.assertArrayEqual(self.set_to_zero(x, j),
                                  self.set_to_zero(x, k))
        # standard numpy (non-orthogonal) indexing doesn't work anymore
        with self.assertRaisesRegexp(ValueError, 'only supports 1d'):
            utils.orthogonal_indexer(x > 0, x.shape)

    def test_remap_loc_indexers(self):
        # TODO: fill in more tests!
        indices = {'x': XArray(['x'], pd.Index([1, 2, 3]))}
        test_indexer = lambda x: utils.remap_loc_indexers(indices, {'x': x})
        self.assertEqual({'x': 0}, test_indexer(1))
        self.assertEqual({'x': 0}, test_indexer(np.int32(1)))
        self.assertEqual({'x': 0}, test_indexer(XArray([], 1)))


class TestDatetime(TestCase):
    def test_cf_datetime(self):
        for num_dates, units in [
                (np.arange(100), 'days since 2000-01-01'),
                (np.arange(100).reshape(10, 10), 'days since 2000-01-01'),
                (12300 + np.arange(50), 'hours since 1680-01-01 00:00:00'),
                (10, 'days since 2000-01-01'),
                ([10], 'days since 2000-01-01'),
                ([[10]], 'days since 2000-01-01'),
                ([10, 10], 'days since 2000-01-01'),
                (0, 'days since 1000-01-01'),
                ([0], 'days since 1000-01-01'),
                ([[0]], 'days since 1000-01-01'),
                (np.arange(20), 'days since 1000-01-01'),
                (np.arange(0, 100000, 10000), 'days since 1900-01-01')
                ]:
            for calendar in ['standard', 'gregorian', 'proleptic_gregorian']:
                expected = nc4.num2date(num_dates, units, calendar)
                actual = utils.decode_cf_datetime(num_dates, units, calendar)
                if (isinstance(actual, np.ndarray)
                        and np.issubdtype(actual.dtype, np.datetime64)):
                    self.assertEqual(actual.dtype, np.dtype('M8[ns]'))
                    # For some reason, numpy 1.8 does not compare ns precision
                    # datetime64 arrays as equal to arrays of datetime objects,
                    # but it works for us precision. Thus, convert to us
                    # precision for the actual array equal comparison...
                    actual_cmp = actual.astype('M8[us]')
                else:
                    actual_cmp = actual
                self.assertArrayEqual(expected, actual_cmp)
                encoded, _, _ = utils.encode_cf_datetime(actual, units, calendar)
                self.assertArrayEqual(num_dates, np.around(encoded))
                if (hasattr(num_dates, 'ndim') and num_dates.ndim == 1
                        and '1000' not in units):
                    # verify that wrapping with a pandas.Index works
                    # note that it *does not* currently work to even put
                    # non-datetime64 compatible dates into a pandas.Index :(
                    encoded, _, _ = utils.encode_cf_datetime(
                        pd.Index(actual), units, calendar)
                    self.assertArrayEqual(num_dates, np.around(encoded))

    def test_guess_time_units(self):
        for dates, expected in [(pd.date_range('1900-01-01', periods=5),
                                 'days since 1900-01-01 00:00:00'),
                                (pd.date_range('1900-01-01 12:00:00', freq='H',
                                               periods=2),
                                 'hours since 1900-01-01 12:00:00'),
                                (['1900-01-01', '1900-01-02',
                                  '1900-01-02 00:00:01'],
                                 'seconds since 1900-01-01 00:00:00')]:
            self.assertEquals(expected, utils.guess_time_units(dates))


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
        self.assertEquals({'a': 'A', 'b': 'B'},
                          utils.ordered_dict_intersection(self.x, self.y))
        self.assertEquals({'b': 'B'},
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
        self.assertFalse(utils.dict_equal(x, y)) # nparray != list
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

    def test_frozen(self):
        x = utils.Frozen(self.x)
        with self.assertRaises(TypeError):
            x['foo'] = 'bar'
        with self.assertRaises(TypeError):
            del x['a']
        with self.assertRaises(AttributeError):
            x.update(self.y)
        self.assertEquals(x.mapping, self.x)

    def test_sorted_keys_dict(self):
        x = {'a': 1, 'b': 2, 'c': 3}
        y = utils.SortedKeysDict(x)
        self.assertItemsEqual(y, ['a', 'b', 'c'])
