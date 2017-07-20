from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import pandas as pd
import pytest

from xarray import Dataset, DataArray, Variable
from xarray.core import indexing, utils
from . import TestCase, ReturnItem


class NumpyOrthogonalIndexingAdapter(utils.NDArrayMixin):
    """Wrap a NumPy array to use orthogonal indexing (array indexing
    accesses different dimensions independently, like netCDF4-python variables)
    """
    # note: this object is somewhat similar to biggus.NumpyArrayAdapter in that
    # it implements orthogonal indexing, except it casts to a numpy array,
    # isn't lazy and supports writing values.
    def __init__(self, array):
        self.array = np.asarray(array)

    def __array__(self, dtype=None):
        return np.asarray(self.array, dtype=dtype)

    def _convert_key(self, key):
        key = indexing.expanded_indexer(key, self.ndim)
        if any(not isinstance(k, indexing.integer_types + (slice,))
                for k in key):
            # key would trigger fancy indexing
            key = indexing.orthogonal_indexer(key, self.shape)
        return key

    def _ensure_ndarray(self, value):
        # We always want the result of indexing to be a NumPy array. If it's
        # not, then it really should be a 0d array. Doing the coercion here
        # instead of inside variable.as_compatible_data makes it less error
        # prone.
        if not isinstance(value, np.ndarray):
            value = utils.to_0d_array(value)
        return value

    def __getitem__(self, key):
        key = self._convert_key(key)
        return type(self)(self._ensure_ndarray(self.array[key]))

    def __setitem__(self, key, value):
        key = self._convert_key(key)
        self.array[key] = value


class TestNumpyOrthogonalIndexingAdapter(TestCase):
    def test_basic(self):
        def maybe_boolean_array(array, size):
            """ Map boolean array to size 'size' by appendin False in its tail
            """
            if hasattr(array, 'dtype') and array.dtype.kind == 'b':
                array_new = np.ndarray(size, dtype='?')
                array_new[:array.size] = array
                array_new[array.size:] = False
                return array_new
            return array

        original = np.random.rand(10, 20, 30)
        orthogonal = NumpyOrthogonalIndexingAdapter(original)
        I = ReturnItem()
        # test broadcasted indexers
        indexers = [I[:], 0, -2, I[:3], [0, 1, 2, 3], [0], np.arange(10) < 5]
        for i in indexers:
            for j in indexers:
                for k in indexers:
                    actual = orthogonal[i, j, k]
                    j = maybe_boolean_array(j, 20)
                    k = maybe_boolean_array(k, 30)
                    if isinstance(i, int):
                        if isinstance(j, int):
                            expected = original[i][j][k]
                        else:
                            expected = original[i][j][:, k]
                    else:
                        if isinstance(j, int):
                            expected = original[i][:, j][:, k]
                        else:
                            expected = original[i][:, j][:, :, k]
                    self.assertArrayEqual(actual, expected)
        # indivisual testing
        self.assertTrue(orthogonal[np.array([0]), :, :].shape == (1, 20, 30))
        self.assertArrayEqual(orthogonal[[0], :, :], original[[0], :, :])


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

    def test_maybe_convert_to_slice(self):

        cases = [
            (1,),
            (1, 1),
            (1, 2),
            (10,),
            (0, 10),
            (5, 10),
            (5, 8),
            (None, 5),
            (None, -3),
            (0, 10, 2),
            (10, None, -1),
            (7, 3, -2),
        ]
        for case in cases:
            slice_obj = slice(*case)
            base_array = np.arange(*slice_obj.indices(10))
            for array in [base_array, base_array - 10]:
                actual = indexing.maybe_convert_to_slice(array, 10)
                self.assertArrayEqual(np.arange(10)[actual],
                                      np.arange(10)[slice_obj])

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
        with self.assertRaisesRegexp(ValueError, 'does not have a MultiIndex'):
            indexing.convert_label_indexer(index, {'one': 0})

        mindex = pd.MultiIndex.from_product([['a', 'b'], [1, 2]],
                                            names=('one', 'two'))
        with self.assertRaisesRegexp(KeyError, 'not all values found'):
            indexing.convert_label_indexer(mindex, [0])
        with self.assertRaises(KeyError):
            indexing.convert_label_indexer(mindex, 0)
        with self.assertRaises(ValueError):
            indexing.convert_label_indexer(index, {'three': 0})
        with self.assertRaisesRegexp(KeyError, 'index to be fully lexsorted'):
            indexing.convert_label_indexer(mindex, (slice(None), 1, 'no_level'))

    def test_convert_unsorted_datetime_index_raises(self):
        index = pd.to_datetime(['2001', '2000', '2002'])
        with self.assertRaises(KeyError):
            # pandas will try to convert this into an array indexer. We should
            # raise instead, so we can be sure the result of indexing with a
            # slice is always a view.
            indexing.convert_label_indexer(index, slice('2001', '2002'))

    def test_get_dim_indexers(self):
        mindex = pd.MultiIndex.from_product([['a', 'b'], [1, 2]],
                                            names=('one', 'two'))
        mdata = DataArray(range(4), [('x', mindex)])

        dim_indexers = indexing.get_dim_indexers(mdata, {'one': 'a', 'two': 1})
        self.assertEqual(dim_indexers, {'x': {'one': 'a', 'two': 1}})

        with self.assertRaisesRegexp(ValueError, 'cannot combine'):
            indexing.get_dim_indexers(mdata, {'x': 'a', 'two': 1})

        with self.assertRaisesRegexp(ValueError, 'do not exist'):
            indexing.get_dim_indexers(mdata, {'y': 'a'})

        with self.assertRaisesRegexp(ValueError, 'do not exist'):
            indexing.get_dim_indexers(mdata, {'four': 1})

    def test_remap_label_indexers(self):
        def test_indexer(data, x, expected_pos, expected_idx=None):
            pos, idx = indexing.remap_label_indexers(data, {'x': x})
            self.assertArrayEqual(pos.get('x'), expected_pos)
            self.assertArrayEqual(idx.get('x'), expected_idx)

        data = Dataset({'x': ('x', [1, 2, 3])})
        mindex = pd.MultiIndex.from_product([['a', 'b'], [1, 2], [-1, -2]],
                                            names=('one', 'two', 'three'))
        mdata = DataArray(range(8), [('x', mindex)])

        test_indexer(data, 1, 0)
        test_indexer(data, np.int32(1), 0)
        test_indexer(data, Variable([], 1), 0)
        test_indexer(mdata, ('a', 1, -1), 0)
        test_indexer(mdata, ('a', 1),
                     [True,  True, False, False, False, False, False, False],
                     [-1, -2])
        test_indexer(mdata, 'a', slice(0, 4, None),
                     pd.MultiIndex.from_product([[1, 2], [-1, -2]]))
        test_indexer(mdata, ('a',),
                     [True,  True,  True,  True, False, False, False, False],
                     pd.MultiIndex.from_product([[1, 2], [-1, -2]]))
        test_indexer(mdata, [('a', 1, -1), ('b', 2, -2)], [0, 7])
        test_indexer(mdata, slice('a', 'b'), slice(0, 8, None))
        test_indexer(mdata, slice(('a', 1), ('b', 1)), slice(0, 6, None))
        test_indexer(mdata, {'one': 'a', 'two': 1, 'three': -1}, 0)
        test_indexer(mdata, {'one': 'a', 'two': 1},
                     [True,  True, False, False, False, False, False, False],
                     [-1, -2])
        test_indexer(mdata, {'one': 'a', 'three': -1},
                     [True,  False, True, False, False, False, False, False],
                     [1, 2])
        test_indexer(mdata, {'one': 'a'},
                     [True,  True,  True,  True, False, False, False, False],
                     pd.MultiIndex.from_product([[1, 2], [-1, -2]]))


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
        x = NumpyOrthogonalIndexingAdapter(np.random.rand(10, 20, 30))
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


class TestCopyOnWriteArray(TestCase):
    def test_setitem(self):
        original = np.arange(10)
        wrapped = indexing.CopyOnWriteArray(original)
        wrapped[:] = 0
        self.assertArrayEqual(original, np.arange(10))
        self.assertArrayEqual(wrapped, np.zeros(10))

    def test_sub_array(self):
        original = np.arange(10)
        wrapped = indexing.CopyOnWriteArray(original)
        child = wrapped[:5]
        self.assertIsInstance(child, indexing.CopyOnWriteArray)
        child[:] = 0
        self.assertArrayEqual(original, np.arange(10))
        self.assertArrayEqual(wrapped, np.arange(10))
        self.assertArrayEqual(child, np.zeros(5))

    def test_index_scalar(self):
        # regression test for GH1374
        x = indexing.CopyOnWriteArray(np.array(['foo', 'bar']))
        assert np.array(x[0][()]) == 'foo'


class TestMemoryCachedArray(TestCase):
    def test_wrapper(self):
        original = indexing.LazilyIndexedArray(np.arange(10))
        wrapped = indexing.MemoryCachedArray(original)
        self.assertArrayEqual(wrapped, np.arange(10))
        self.assertIsInstance(wrapped.array, np.ndarray)

    def test_sub_array(self):
        original = indexing.LazilyIndexedArray(np.arange(10))
        wrapped = indexing.MemoryCachedArray(original)
        child = wrapped[:5]
        self.assertIsInstance(child, indexing.MemoryCachedArray)
        self.assertArrayEqual(child, np.arange(5))
        self.assertIsInstance(child.array, np.ndarray)
        self.assertIsInstance(wrapped.array, indexing.LazilyIndexedArray)

    def test_setitem(self):
        original = np.arange(10)
        wrapped = indexing.MemoryCachedArray(original)
        wrapped[:] = 0
        self.assertArrayEqual(original, np.zeros(10))

    def test_index_scalar(self):
        # regression test for GH1374
        x = indexing.MemoryCachedArray(np.array(['foo', 'bar']))
        assert np.array(x[0][()]) == 'foo'


class Test_orthogonalize_indexers(TestCase):
    def assert1dIndexEqual(self, x, y, size):
        """ Compare 1d vector, slice, array """
        def vectorize(array):
            if isinstance(array, slice):  # slice
                return np.arange(*array.indices(size))
            if hasattr(array, 'dtype') and array.dtype.kind == 'b':
                # boolean array
                return np.arange(len(array))[array]
            return np.arange(size)[array]

        self.assertArrayEqual(vectorize(x), vectorize(y))
        self.assertArrayEqual(vectorize(x).shape, vectorize(y).shape)

    def test(self):
        original = np.random.rand(10, 20, 30)
        v = Variable(('i', 'j', 'k'), original)
        I = ReturnItem()
        # test broadcasted indexers
        indexers = [I[:], 0, -2, I[:3], [4, 1, 2, 3], [0], np.arange(10) < 5]
        for i in indexers:
            for j in indexers:
                for k in indexers:
                    dims, indexer = v._broadcast_indexes((i, j, k))
                    orthogonalized = indexing.orthogonalize_indexers(
                                                        indexer, v.shape)
                    dim_new, indexer_new = v._broadcast_indexes(orthogonalized)

                    self.assertArrayEqual(original[indexer],
                                          original[indexer_new])
                    orthogonalized_new = indexing.orthogonalize_indexers(
                                                        indexer_new, v.shape)
                    self.assertArrayEqual(orthogonalized[0],
                                          orthogonalized_new[0])
                    self.assertArrayEqual(orthogonalized[0],
                                          orthogonalized_new[0])

    def test_error(self):
        with self.assertRaisesRegexp(IndexError, 'Indexer cannot be'):
            indexing.orthogonalize_indexers((np.ones((2, 2)), np.ones((2, 1))),
                                            shape=(3, 2))
        with self.assertRaisesRegexp(IndexError, 'Indexer cannot be'):
            indexing.orthogonalize_indexers((np.ones((1, 2)), np.ones((2, 1))),
                                            shape=(3, 2))


class TestBroadcastedIndexingAdapter(TestCase):
    def test_basic(self):
        original = np.random.rand(10, 20, 30)
        v = Variable(('i', 'j', 'k'), original)
        orthogonal = NumpyOrthogonalIndexingAdapter(original)
        wrapped = indexing.BroadcastedIndexingAdapter(orthogonal)
        I = ReturnItem()
        # test broadcasted indexers
        indexers = [I[:], 0, -2, I[:3], [0, 1, 2, 3], [0], np.arange(10) < 5]
        for i in indexers:
            for j in indexers:
                for k in indexers:
                    actual_ortho = orthogonal[i, j, k]
                    dims, indexer = v._broadcast_indexes((i, j, k))
                    expected = original[indexer]
                    actual = wrapped[indexer]
                    self.assertEqual(expected.shape, actual_ortho.shape)
                    self.assertArrayEqual(expected, actual_ortho)
                    self.assertEqual(expected.shape, actual.shape)
                    self.assertArrayEqual(expected, actual)
