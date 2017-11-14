from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import itertools

import pytest

import numpy as np
import pandas as pd

from xarray import Dataset, DataArray, Variable
from xarray.core import indexing
from xarray.core import nputils
from xarray.core.pycompat import native_int_types
from . import TestCase, ReturnItem, raises_regex, IndexerMaker


B = IndexerMaker(indexing.BasicIndexer)


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
        with raises_regex(IndexError, 'too many indices'):
            indexing.expanded_indexer(I[1, 2, 3], 2)

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
        with raises_regex(KeyError, 'not all values found'):
            indexing.convert_label_indexer(index, [0])
        with pytest.raises(KeyError):
            indexing.convert_label_indexer(index, 0)
        with raises_regex(ValueError, 'does not have a MultiIndex'):
            indexing.convert_label_indexer(index, {'one': 0})

        mindex = pd.MultiIndex.from_product([['a', 'b'], [1, 2]],
                                            names=('one', 'two'))
        with raises_regex(KeyError, 'not all values found'):
            indexing.convert_label_indexer(mindex, [0])
        with pytest.raises(KeyError):
            indexing.convert_label_indexer(mindex, 0)
        with pytest.raises(ValueError):
            indexing.convert_label_indexer(index, {'three': 0})
        with pytest.raises((KeyError, IndexError)):
            # pandas 0.21 changed this from KeyError to IndexError
            indexing.convert_label_indexer(
                mindex, (slice(None), 1, 'no_level'))

    def test_convert_unsorted_datetime_index_raises(self):
        index = pd.to_datetime(['2001', '2000', '2002'])
        with pytest.raises(KeyError):
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

        with raises_regex(ValueError, 'cannot combine'):
            indexing.get_dim_indexers(mdata, {'x': 'a', 'two': 1})

        with raises_regex(ValueError, 'do not exist'):
            indexing.get_dim_indexers(mdata, {'y': 'a'})

        with raises_regex(ValueError, 'do not exist'):
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
        original = np.random.rand(10, 20, 30)
        x = indexing.NumpyIndexingAdapter(original)
        v = Variable(['i', 'j', 'k'], original)
        lazy = indexing.LazilyIndexedArray(x)
        v_lazy = Variable(['i', 'j', 'k'], lazy)
        I = ReturnItem()
        # test orthogonally applied indexers
        indexers = [I[:], 0, -2, I[:3], [0, 1, 2, 3], [0], np.arange(10) < 5]
        for i in indexers:
            for j in indexers:
                for k in indexers:
                    if isinstance(j, np.ndarray) and j.dtype.kind == 'b':
                        j = np.arange(20) < 5
                    if isinstance(k, np.ndarray) and k.dtype.kind == 'b':
                        k = np.arange(30) < 5
                    expected = np.asarray(v[i, j, k])
                    for actual in [v_lazy[i, j, k],
                                   v_lazy[:, j, k][i],
                                   v_lazy[:, :, k][:, j][i]]:
                        self.assertEqual(expected.shape, actual.shape)
                        self.assertArrayEqual(expected, actual)
                        assert isinstance(actual._data,
                                          indexing.LazilyIndexedArray)

                        # make sure actual.key is appropriate type
                        if all(isinstance(k, native_int_types + (slice, ))
                               for k in v_lazy._data.key.tuple):
                            assert isinstance(v_lazy._data.key,
                                              indexing.BasicIndexer)
                        else:
                            assert isinstance(v_lazy._data.key,
                                              indexing.OuterIndexer)

        # test sequentially applied indexers
        indexers = [(3, 2), (I[:], 0), (I[:2], -1), (I[:4], [0]), ([4, 5], 0),
                    ([0, 1, 2], [0, 1]), ([0, 3, 5], I[:2])]
        for i, j in indexers:
            expected = np.asarray(v[i][j])
            actual = v_lazy[i][j]
            self.assertEqual(expected.shape, actual.shape)
            self.assertArrayEqual(expected, actual)
            assert isinstance(actual._data, indexing.LazilyIndexedArray)
            assert isinstance(actual._data.array,
                              indexing.NumpyIndexingAdapter)


class TestCopyOnWriteArray(TestCase):
    def test_setitem(self):
        original = np.arange(10)
        wrapped = indexing.CopyOnWriteArray(original)
        wrapped[B[:]] = 0
        self.assertArrayEqual(original, np.arange(10))
        self.assertArrayEqual(wrapped, np.zeros(10))

    def test_sub_array(self):
        original = np.arange(10)
        wrapped = indexing.CopyOnWriteArray(original)
        child = wrapped[B[:5]]
        self.assertIsInstance(child, indexing.CopyOnWriteArray)
        child[B[:]] = 0
        self.assertArrayEqual(original, np.arange(10))
        self.assertArrayEqual(wrapped, np.arange(10))
        self.assertArrayEqual(child, np.zeros(5))

    def test_index_scalar(self):
        # regression test for GH1374
        x = indexing.CopyOnWriteArray(np.array(['foo', 'bar']))
        assert np.array(x[B[0]][B[()]]) == 'foo'


class TestMemoryCachedArray(TestCase):
    def test_wrapper(self):
        original = indexing.LazilyIndexedArray(np.arange(10))
        wrapped = indexing.MemoryCachedArray(original)
        self.assertArrayEqual(wrapped, np.arange(10))
        self.assertIsInstance(wrapped.array, indexing.NumpyIndexingAdapter)

    def test_sub_array(self):
        original = indexing.LazilyIndexedArray(np.arange(10))
        wrapped = indexing.MemoryCachedArray(original)
        child = wrapped[B[:5]]
        self.assertIsInstance(child, indexing.MemoryCachedArray)
        self.assertArrayEqual(child, np.arange(5))
        self.assertIsInstance(child.array, indexing.NumpyIndexingAdapter)
        self.assertIsInstance(wrapped.array, indexing.LazilyIndexedArray)

    def test_setitem(self):
        original = np.arange(10)
        wrapped = indexing.MemoryCachedArray(original)
        wrapped[B[:]] = 0
        self.assertArrayEqual(original, np.zeros(10))

    def test_index_scalar(self):
        # regression test for GH1374
        x = indexing.MemoryCachedArray(np.array(['foo', 'bar']))
        assert np.array(x[B[0]][B[()]]) == 'foo'


def test_base_explicit_indexer():
    with pytest.raises(TypeError):
        indexing.ExplicitIndexer(())

    class Subclass(indexing.ExplicitIndexer):
        pass

    value = Subclass((1, 2, 3))
    assert value.tuple == (1, 2, 3)
    assert repr(value) == 'Subclass((1, 2, 3))'


@pytest.mark.parametrize('indexer_cls', [indexing.BasicIndexer,
                                         indexing.OuterIndexer,
                                         indexing.VectorizedIndexer])
def test_invalid_for_all(indexer_cls):
    with pytest.raises(TypeError):
        indexer_cls(None)
    with pytest.raises(TypeError):
        indexer_cls(([],))
    with pytest.raises(TypeError):
        indexer_cls((None,))
    with pytest.raises(TypeError):
        indexer_cls(('foo',))
    with pytest.raises(TypeError):
        indexer_cls((1.0,))
    with pytest.raises(TypeError):
        indexer_cls((slice('foo'),))
    with pytest.raises(TypeError):
        indexer_cls((np.array(['foo']),))


def check_integer(indexer_cls):
    value = indexer_cls((1, np.uint64(2),)).tuple
    assert all(isinstance(v, int) for v in value)
    assert value == (1, 2)


def check_slice(indexer_cls):
    (value,) = indexer_cls((slice(1, None, np.int64(2)),)).tuple
    assert value == slice(1, None, 2)
    assert isinstance(value.step, native_int_types)


def check_array1d(indexer_cls):
    (value,) = indexer_cls((np.arange(3, dtype=np.int32),)).tuple
    assert value.dtype == np.int64
    np.testing.assert_array_equal(value, [0, 1, 2])


def check_array2d(indexer_cls):
    array = np.array([[1, 2], [3, 4]], dtype=np.int64)
    (value,) = indexer_cls((array,)).tuple
    assert value.dtype == np.int64
    np.testing.assert_array_equal(value, array)


def test_basic_indexer():
    check_integer(indexing.BasicIndexer)
    check_slice(indexing.BasicIndexer)
    with pytest.raises(TypeError):
        check_array1d(indexing.BasicIndexer)
    with pytest.raises(TypeError):
        check_array2d(indexing.BasicIndexer)


def test_outer_indexer():
    check_integer(indexing.OuterIndexer)
    check_slice(indexing.OuterIndexer)
    check_array1d(indexing.OuterIndexer)
    with pytest.raises(TypeError):
        check_array2d(indexing.OuterIndexer)


def test_vectorized_indexer():
    with pytest.raises(TypeError):
        check_integer(indexing.VectorizedIndexer)
    check_slice(indexing.VectorizedIndexer)
    check_array1d(indexing.VectorizedIndexer)
    check_array2d(indexing.VectorizedIndexer)


def test_unwrap_explicit_indexer():
    indexer = indexing.BasicIndexer((1, 2))
    target = None

    unwrapped = indexing.unwrap_explicit_indexer(
        indexer, target, allow=indexing.BasicIndexer)
    assert unwrapped == (1, 2)

    with raises_regex(NotImplementedError, 'Load your data'):
        indexing.unwrap_explicit_indexer(
            indexer, target, allow=indexing.OuterIndexer)

    with raises_regex(TypeError, 'unexpected key type'):
        indexing.unwrap_explicit_indexer(
            indexer.tuple, target, allow=indexing.OuterIndexer)


def test_implicit_indexing_adapter():
    array = np.arange(10)
    implicit = indexing.ImplicitToExplicitIndexingAdapter(
        indexing.NumpyIndexingAdapter(array), indexing.BasicIndexer)
    np.testing.assert_array_equal(array, np.asarray(implicit))
    np.testing.assert_array_equal(array, implicit[:])


def test_outer_indexer_consistency_with_broadcast_indexes_vectorized():
    def nonzero(x):
        if isinstance(x, np.ndarray) and x.dtype.kind == 'b':
            x = x.nonzero()[0]
        return x

    original = np.random.rand(10, 20, 30)
    v = Variable(['i', 'j', 'k'], original)
    I = ReturnItem()
    # test orthogonally applied indexers
    indexers = [I[:], 0, -2, I[:3], np.array([0, 1, 2, 3]), np.array([0]),
                np.arange(10) < 5]
    for i, j, k in itertools.product(indexers, repeat=3):

        if isinstance(j, np.ndarray) and j.dtype.kind == 'b':  # match size
            j = np.arange(20) < 4
        if isinstance(k, np.ndarray) and k.dtype.kind == 'b':
            k = np.arange(30) < 8

        _, expected, new_order = v._broadcast_indexes_vectorized((i, j, k))
        expected_data = nputils.NumpyVIndexAdapter(v.data)[expected.tuple]
        if new_order:
            old_order = range(len(new_order))
            expected_data = np.moveaxis(expected_data, old_order,
                                        new_order)

        outer_index = (nonzero(i), nonzero(j), nonzero(k))
        actual = indexing._outer_to_numpy_indexer(outer_index, v.shape)
        actual_data = v.data[actual]
        np.testing.assert_array_equal(actual_data, expected_data)
