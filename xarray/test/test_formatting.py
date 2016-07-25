# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

from xarray.core import formatting
from xarray.core.pycompat import PY3

from . import TestCase


class TestFormatting(TestCase):

    def test_get_indexer_at_least_n_items(self):
        cases = [
            ((20,), (slice(10),)),
            ((3, 20,), (0, slice(10))),
            ((2, 10,), (0, slice(10))),
            ((2, 5,), (slice(2), slice(None))),
            ((1, 2, 5,), (0, slice(2), slice(None))),
            ((2, 3, 5,), (0, slice(2), slice(None))),
            ((1, 10, 1,), (0, slice(10), slice(None))),
            ((2, 5, 1,), (slice(2), slice(None), slice(None))),
            ((2, 5, 3,), (0, slice(4), slice(None))),
            ((2, 3, 3,), (slice(2), slice(None), slice(None))),
            ]
        for shape, expected in cases:
            actual = formatting._get_indexer_at_least_n_items(shape, 10)
            self.assertEqual(expected, actual)

    def test_first_n_items(self):
        array = np.arange(100).reshape(10, 5, 2)
        for n in [3, 10, 13, 100, 200]:
            actual = formatting.first_n_items(array, n)
            expected = array.flat[:n]
            self.assertItemsEqual(expected, actual)

        with self.assertRaisesRegexp(ValueError, 'at least one item'):
            formatting.first_n_items(array, 0)

    def test_format_item(self):
        cases = [
            (pd.Timestamp('2000-01-01T12'), '2000-01-01T12:00:00'),
            (pd.Timestamp('2000-01-01'), '2000-01-01'),
            (pd.Timestamp('NaT'), 'NaT'),
            (pd.Timedelta('10 days 1 hour'), '10 days 01:00:00'),
            (pd.Timedelta('-3 days'), '-3 days +00:00:00'),
            (pd.Timedelta('3 hours'), '0 days 03:00:00'),
            (pd.Timedelta('NaT'), 'NaT'),
            ('foo', "'foo'"),
            (u'foo', "'foo'" if PY3 else "u'foo'"),
            (b'foo', "b'foo'" if PY3 else "'foo'"),
            (1, '1'),
            (1.0, '1.0'),
        ]
        for item, expected in cases:
            actual = formatting.format_item(item)
            self.assertEqual(expected, actual)

    def test_format_items(self):
        cases = [
            (np.arange(4) * np.timedelta64(1, 'D'),
             '0 days 1 days 2 days 3 days'),
            (np.arange(4) * np.timedelta64(3, 'h'),
             '00:00:00 03:00:00 06:00:00 09:00:00'),
            (np.arange(4) * np.timedelta64(500, 'ms'),
             '00:00:00 00:00:00.500000 00:00:01 00:00:01.500000'),
            (pd.to_timedelta(['NaT', '0s', '1s', 'NaT']),
             'NaT 00:00:00 00:00:01 NaT'),
            (pd.to_timedelta(['1 day 1 hour', '1 day', '0 hours']),
             '1 days 01:00:00 1 days 00:00:00 0 days 00:00:00'),
            ([1, 2, 3], '1 2 3'),
        ]
        for item, expected in cases:
            actual = ' '.join(formatting.format_items(item))
            self.assertEqual(expected, actual)


    def test_format_array_flat(self):
        actual = formatting.format_array_flat(np.arange(100), 13)
        expected = '0 1 2 3 4 ...'
        self.assertEqual(expected, actual)

        actual = formatting.format_array_flat(np.arange(100.0), 11)
        expected = '0.0 1.0 ...'
        self.assertEqual(expected, actual)

        actual = formatting.format_array_flat(np.arange(100.0), 1)
        expected = '0.0 ...'
        self.assertEqual(expected, actual)

        actual = formatting.format_array_flat(np.arange(3), 5)
        expected = '0 1 2'
        self.assertEqual(expected, actual)

        actual = formatting.format_array_flat(np.arange(4.0), 11)
        expected = '0.0 1.0 ...'
        self.assertEqual(expected, actual)

        actual = formatting.format_array_flat(np.arange(4), 0)
        expected = '0 ...'
        self.assertEqual(expected, actual)

    def test_pretty_print(self):
        self.assertEqual(formatting.pretty_print('abcdefghij', 8), 'abcde...')
        self.assertEqual(formatting.pretty_print(u'ß', 1), u'ß')

    def test_maybe_truncate(self):
        self.assertEqual(formatting.maybe_truncate(u'ß', 10), u'ß')
