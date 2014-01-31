import numpy as np

from polyglot import utils
from . import TestCase


class ReturnItem(object):
    def __getitem__(self, key):
        return key


class TestExpandedIndexer(TestCase):
    def test(self):
        x = np.random.randn(10, 20, 30)
        i = ReturnItem()
        for i in [i[:], i[...], i[0, :, 10], i[:5, ...], i[np.arange(5)]]:
            j = utils.expanded_indexer(i, 3)
            self.assertArrayEqual(x[i], x[j])


class TestSafeMerge(TestCase):
    def setUp(self):
        self.x = {'a': 'A', 'b': 'B'}
        self.y = {'c': 'C', 'b': 'B'}

    def test_good_merge(self):
        actual = utils.safe_merge(self.x, self.y)
        self.x.update(self.y)
        self.assertEqual(self.x, actual)

    def test_bad_merge(self):
        with self.assertRaises(ValueError):
            utils.safe_merge(self.x, {'a': 'Z'})
