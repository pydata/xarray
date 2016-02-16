import numpy as np
import pandas as pd

from xarray import Dataset, Variable
from xarray.core import indexing
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
                  I[..., 0, :], I[y], I[y, y], I[..., y, y],
                  I[..., 0, 1, 2, 3, 4]]:
            j = indexing.expanded_indexer(i, x.ndim)
            self.assertArrayEqual(x[i], x[j])
            self.assertArrayEqual(self.set_to_zero(x, i),
                                  self.set_to_zero(x, j))
        with self.assertRaisesRegexp(IndexError, 'too many indices'):
            indexing.expanded_indexer(I[1, 2, 3], 2)

    def test_orthogonal_indexer(self):
        x = np.random.randn(10, 11, 12, 13, 14)
        y = np.arange(5)
        I = ReturnItem()
        # orthogonal and numpy indexing should be equivalent, because we only
        # use at most one array and it never in between two slice objects
        # (i.e., we try to avoid numpy's mind-boggling "partial indexing"
        # http://docs.scipy.org/doc/numpy/reference/arrays.indexing.html)
        for i in [I[:], I[0], I[0, 0], I[:5], I[5:], I[2:5], I[3:-3], I[::-1],
                  I[::-2], I[5::-2], I[:3:-2], I[2:5:-1], I[7:3:-2], I[:3, :4],
                  I[:3, 0, :4], I[:3, 0, :4, 0], I[y], I[:, y], I[0, y],
                  I[:2, :3, y], I[0, y, :, :4, 0]]:
            j = indexing.orthogonal_indexer(i, x.shape)
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
                (I[0, :, z], I[0, :, 2:8:2], (11, 3, 13, 14)),
                (I[0, :2, y, y, 0], I[0, :2, :5, :5, 0], (2, 5, 5)),
                (I[0, :, y, :, 0], I[0, :, :5, :, 0], (11, 5, 13)),
                (I[:, :, y, :, 0], I[:, :, :5, :, 0], (10, 11, 5, 13)),
                (I[:, :, y, z, :], I[:, :, :5, 2:8:2], (10, 11, 5, 3, 14))]:
            k = indexing.orthogonal_indexer(i, x.shape)
            self.assertEqual(shape, x[k].shape)
            self.assertArrayEqual(x[j], x[k])
            self.assertArrayEqual(self.set_to_zero(x, j),
                                  self.set_to_zero(x, k))
        # standard numpy (non-orthogonal) indexing doesn't work anymore
        with self.assertRaisesRegexp(ValueError, 'only supports 1d'):
            indexing.orthogonal_indexer(x > 0, x.shape)
        with self.assertRaisesRegexp(ValueError, 'invalid subkey'):
            print(indexing.orthogonal_indexer((1.5 * y, 1.5 * y), x.shape))

    def test_asarray_tuplesafe(self):
        res = indexing._asarray_tuplesafe(('a', 1))
        assert isinstance(res, np.ndarray)
        assert res.ndim == 0
        assert res.item() == ('a', 1)

        res = indexing._asarray_tuplesafe([(0,), (1,)])
        assert res.shape == (2,)
        assert res[0] == (0,)
        assert res[1] == (1,)

    def test_convert_label_indexer(self):
        # TODO: add tests that aren't just for edge cases
        index = pd.Index([1, 2, 3])
        with self.assertRaisesRegexp(KeyError, 'not all values found'):
            indexing.convert_label_indexer(index, [0])
        with self.assertRaises(KeyError):
            indexing.convert_label_indexer(index, 0)

    def test_convert_unsorted_datetime_index_raises(self):
        index = pd.to_datetime(['2001', '2000', '2002'])
        with self.assertRaises(KeyError):
            # pandas will try to convert this into an array indexer. We should
            # raise instead, so we can be sure the result of indexing with a
            # slice is always a view.
            indexing.convert_label_indexer(index, slice('2001', '2002'))

    def test_remap_label_indexers(self):
        # TODO: fill in more tests!
        data = Dataset({'x': ('x', [1, 2, 3])})

        def test_indexer(x):
            return indexing.remap_label_indexers(data, {'x': x})
        self.assertEqual({'x': 0}, test_indexer(1))
        self.assertEqual({'x': 0}, test_indexer(np.int32(1)))
        self.assertEqual({'x': 0}, test_indexer(Variable([], 1)))


class TestLazyArray(TestCase):
    def test_slice_slice(self):
        I = ReturnItem()
        x = np.arange(100)
        slices = [I[:3], I[:4], I[2:4], I[:1], I[:-1], I[5:-1], I[-5:-1],
                  I[::-1], I[5::-1], I[:3:-1], I[:30:-1], I[10:4:], I[::4],
                  I[4:4:4], I[:4:-4]]
        for i in slices:
            for j in slices:
                expected = x[i][j]
                new_slice = indexing.slice_slice(i, j, size=100)
                actual = x[new_slice]
                self.assertArrayEqual(expected, actual)

    def test_lazily_indexed_array(self):
        x = indexing.NumpyIndexingAdapter(np.random.rand(10, 20, 30))
        lazy = indexing.LazilyIndexedArray(x)
        I = ReturnItem()
        # test orthogonally applied indexers
        indexers = [I[:], 0, -2, I[:3], [0, 1, 2, 3], np.arange(10) < 5]
        for i in indexers:
            for j in indexers:
                for k in indexers:
                    expected = np.asarray(x[i, j, k])
                    for actual in [lazy[i, j, k],
                                   lazy[:, j, k][i],
                                   lazy[:, :, k][:, j][i]]:
                        self.assertEqual(expected.shape, actual.shape)
                        self.assertArrayEqual(expected, actual)
        # test sequentially applied indexers
        indexers = [(3, 2), (I[:], 0), (I[:2], -1), (I[:4], [0]), ([4, 5], 0),
                    ([0, 1, 2], [0, 1]), ([0, 3, 5], I[:2])]
        for i, j in indexers:
            expected = np.asarray(x[i][j])
            actual = lazy[i][j]
            self.assertEqual(expected.shape, actual.shape)
            self.assertArrayEqual(expected, actual)
