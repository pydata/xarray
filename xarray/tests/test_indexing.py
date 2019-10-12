import itertools

import numpy as np
import pandas as pd
import pytest

from xarray import DataArray, Dataset, Variable
from xarray.core import indexing, nputils

from . import IndexerMaker, ReturnItem, assert_array_equal, raises_regex

B = IndexerMaker(indexing.BasicIndexer)


class TestIndexers:
    def set_to_zero(self, x, i):
        x = x.copy()
        x[i] = 0
        return x

    def test_expanded_indexer(self):
        x = np.random.randn(10, 11, 12, 13, 14)
        y = np.arange(5)
        arr = ReturnItem()
        for i in [
            arr[:],
            arr[...],
            arr[0, :, 10],
            arr[..., 10],
            arr[:5, ..., 0],
            arr[..., 0, :],
            arr[y],
            arr[y, y],
            arr[..., y, y],
            arr[..., 0, 1, 2, 3, 4],
        ]:
            j = indexing.expanded_indexer(i, x.ndim)
            assert_array_equal(x[i], x[j])
            assert_array_equal(self.set_to_zero(x, i), self.set_to_zero(x, j))
        with raises_regex(IndexError, "too many indices"):
            indexing.expanded_indexer(arr[1, 2, 3], 2)

    def test_asarray_tuplesafe(self):
        res = indexing._asarray_tuplesafe(("a", 1))
        assert isinstance(res, np.ndarray)
        assert res.ndim == 0
        assert res.item() == ("a", 1)

        res = indexing._asarray_tuplesafe([(0,), (1,)])
        assert res.shape == (2,)
        assert res[0] == (0,)
        assert res[1] == (1,)

    def test_stacked_multiindex_min_max(self):
        data = np.random.randn(3, 23, 4)
        da = DataArray(
            data,
            name="value",
            dims=["replicate", "rsample", "exp"],
            coords=dict(
                replicate=[0, 1, 2], exp=["a", "b", "c", "d"], rsample=list(range(23))
            ),
        )
        da2 = da.stack(sample=("replicate", "rsample"))
        s = da2.sample
        assert_array_equal(da2.loc["a", s.max()], data[2, 22, 0])
        assert_array_equal(da2.loc["b", s.min()], data[0, 0, 1])

    def test_convert_label_indexer(self):
        # TODO: add tests that aren't just for edge cases
        index = pd.Index([1, 2, 3])
        with raises_regex(KeyError, "not all values found"):
            indexing.convert_label_indexer(index, [0])
        with pytest.raises(KeyError):
            indexing.convert_label_indexer(index, 0)
        with raises_regex(ValueError, "does not have a MultiIndex"):
            indexing.convert_label_indexer(index, {"one": 0})

        mindex = pd.MultiIndex.from_product([["a", "b"], [1, 2]], names=("one", "two"))
        with raises_regex(KeyError, "not all values found"):
            indexing.convert_label_indexer(mindex, [0])
        with pytest.raises(KeyError):
            indexing.convert_label_indexer(mindex, 0)
        with pytest.raises(ValueError):
            indexing.convert_label_indexer(index, {"three": 0})
        with pytest.raises(IndexError):
            indexing.convert_label_indexer(mindex, (slice(None), 1, "no_level"))

    def test_convert_unsorted_datetime_index_raises(self):
        index = pd.to_datetime(["2001", "2000", "2002"])
        with pytest.raises(KeyError):
            # pandas will try to convert this into an array indexer. We should
            # raise instead, so we can be sure the result of indexing with a
            # slice is always a view.
            indexing.convert_label_indexer(index, slice("2001", "2002"))

    def test_get_dim_indexers(self):
        mindex = pd.MultiIndex.from_product([["a", "b"], [1, 2]], names=("one", "two"))
        mdata = DataArray(range(4), [("x", mindex)])

        dim_indexers = indexing.get_dim_indexers(mdata, {"one": "a", "two": 1})
        assert dim_indexers == {"x": {"one": "a", "two": 1}}

        with raises_regex(ValueError, "cannot combine"):
            indexing.get_dim_indexers(mdata, {"x": "a", "two": 1})

        with raises_regex(ValueError, "do not exist"):
            indexing.get_dim_indexers(mdata, {"y": "a"})

        with raises_regex(ValueError, "do not exist"):
            indexing.get_dim_indexers(mdata, {"four": 1})

    def test_remap_label_indexers(self):
        def test_indexer(data, x, expected_pos, expected_idx=None):
            pos, idx = indexing.remap_label_indexers(data, {"x": x})
            assert_array_equal(pos.get("x"), expected_pos)
            assert_array_equal(idx.get("x"), expected_idx)

        data = Dataset({"x": ("x", [1, 2, 3])})
        mindex = pd.MultiIndex.from_product(
            [["a", "b"], [1, 2], [-1, -2]], names=("one", "two", "three")
        )
        mdata = DataArray(range(8), [("x", mindex)])

        test_indexer(data, 1, 0)
        test_indexer(data, np.int32(1), 0)
        test_indexer(data, Variable([], 1), 0)
        test_indexer(mdata, ("a", 1, -1), 0)
        test_indexer(
            mdata,
            ("a", 1),
            [True, True, False, False, False, False, False, False],
            [-1, -2],
        )
        test_indexer(
            mdata,
            "a",
            slice(0, 4, None),
            pd.MultiIndex.from_product([[1, 2], [-1, -2]]),
        )
        test_indexer(
            mdata,
            ("a",),
            [True, True, True, True, False, False, False, False],
            pd.MultiIndex.from_product([[1, 2], [-1, -2]]),
        )
        test_indexer(mdata, [("a", 1, -1), ("b", 2, -2)], [0, 7])
        test_indexer(mdata, slice("a", "b"), slice(0, 8, None))
        test_indexer(mdata, slice(("a", 1), ("b", 1)), slice(0, 6, None))
        test_indexer(mdata, {"one": "a", "two": 1, "three": -1}, 0)
        test_indexer(
            mdata,
            {"one": "a", "two": 1},
            [True, True, False, False, False, False, False, False],
            [-1, -2],
        )
        test_indexer(
            mdata,
            {"one": "a", "three": -1},
            [True, False, True, False, False, False, False, False],
            [1, 2],
        )
        test_indexer(
            mdata,
            {"one": "a"},
            [True, True, True, True, False, False, False, False],
            pd.MultiIndex.from_product([[1, 2], [-1, -2]]),
        )

    def test_read_only_view(self):

        arr = DataArray(
            np.random.rand(3, 3),
            coords={"x": np.arange(3), "y": np.arange(3)},
            dims=("x", "y"),
        )  # Create a 2D DataArray
        arr = arr.expand_dims({"z": 3}, -1)  # New dimension 'z'
        arr["z"] = np.arange(3)  # New coords to dimension 'z'
        with pytest.raises(ValueError, match="Do you want to .copy()"):
            arr.loc[0, 0, 0] = 999


class TestLazyArray:
    def test_slice_slice(self):
        arr = ReturnItem()
        for size in [100, 99]:
            # We test even/odd size cases
            x = np.arange(size)
            slices = [
                arr[:3],
                arr[:4],
                arr[2:4],
                arr[:1],
                arr[:-1],
                arr[5:-1],
                arr[-5:-1],
                arr[::-1],
                arr[5::-1],
                arr[:3:-1],
                arr[:30:-1],
                arr[10:4:],
                arr[::4],
                arr[4:4:4],
                arr[:4:-4],
                arr[::-2],
            ]
            for i in slices:
                for j in slices:
                    expected = x[i][j]
                    new_slice = indexing.slice_slice(i, j, size=size)
                    actual = x[new_slice]
                    assert_array_equal(expected, actual)

    def test_lazily_indexed_array(self):
        original = np.random.rand(10, 20, 30)
        x = indexing.NumpyIndexingAdapter(original)
        v = Variable(["i", "j", "k"], original)
        lazy = indexing.LazilyOuterIndexedArray(x)
        v_lazy = Variable(["i", "j", "k"], lazy)
        arr = ReturnItem()
        # test orthogonally applied indexers
        indexers = [arr[:], 0, -2, arr[:3], [0, 1, 2, 3], [0], np.arange(10) < 5]
        for i in indexers:
            for j in indexers:
                for k in indexers:
                    if isinstance(j, np.ndarray) and j.dtype.kind == "b":
                        j = np.arange(20) < 5
                    if isinstance(k, np.ndarray) and k.dtype.kind == "b":
                        k = np.arange(30) < 5
                    expected = np.asarray(v[i, j, k])
                    for actual in [
                        v_lazy[i, j, k],
                        v_lazy[:, j, k][i],
                        v_lazy[:, :, k][:, j][i],
                    ]:
                        assert expected.shape == actual.shape
                        assert_array_equal(expected, actual)
                        assert isinstance(
                            actual._data, indexing.LazilyOuterIndexedArray
                        )

                        # make sure actual.key is appropriate type
                        if all(
                            isinstance(k, (int, slice)) for k in v_lazy._data.key.tuple
                        ):
                            assert isinstance(v_lazy._data.key, indexing.BasicIndexer)
                        else:
                            assert isinstance(v_lazy._data.key, indexing.OuterIndexer)

        # test sequentially applied indexers
        indexers = [
            (3, 2),
            (arr[:], 0),
            (arr[:2], -1),
            (arr[:4], [0]),
            ([4, 5], 0),
            ([0, 1, 2], [0, 1]),
            ([0, 3, 5], arr[:2]),
        ]
        for i, j in indexers:
            expected = v[i][j]
            actual = v_lazy[i][j]
            assert expected.shape == actual.shape
            assert_array_equal(expected, actual)

            # test transpose
            if actual.ndim > 1:
                order = np.random.choice(actual.ndim, actual.ndim)
                order = np.array(actual.dims)
                transposed = actual.transpose(*order)
                assert_array_equal(expected.transpose(*order), transposed)
                assert isinstance(
                    actual._data,
                    (
                        indexing.LazilyVectorizedIndexedArray,
                        indexing.LazilyOuterIndexedArray,
                    ),
                )

            assert isinstance(actual._data, indexing.LazilyOuterIndexedArray)
            assert isinstance(actual._data.array, indexing.NumpyIndexingAdapter)

    def test_vectorized_lazily_indexed_array(self):
        original = np.random.rand(10, 20, 30)
        x = indexing.NumpyIndexingAdapter(original)
        v_eager = Variable(["i", "j", "k"], x)
        lazy = indexing.LazilyOuterIndexedArray(x)
        v_lazy = Variable(["i", "j", "k"], lazy)
        arr = ReturnItem()

        def check_indexing(v_eager, v_lazy, indexers):
            for indexer in indexers:
                actual = v_lazy[indexer]
                expected = v_eager[indexer]
                assert expected.shape == actual.shape
                assert isinstance(
                    actual._data,
                    (
                        indexing.LazilyVectorizedIndexedArray,
                        indexing.LazilyOuterIndexedArray,
                    ),
                )
                assert_array_equal(expected, actual)
                v_eager = expected
                v_lazy = actual

        # test orthogonal indexing
        indexers = [(arr[:], 0, 1), (Variable("i", [0, 1]),)]
        check_indexing(v_eager, v_lazy, indexers)

        # vectorized indexing
        indexers = [
            (Variable("i", [0, 1]), Variable("i", [0, 1]), slice(None)),
            (slice(1, 3, 2), 0),
        ]
        check_indexing(v_eager, v_lazy, indexers)

        indexers = [
            (slice(None, None, 2), 0, slice(None, 10)),
            (Variable("i", [3, 2, 4, 3]), Variable("i", [3, 2, 1, 0])),
            (Variable(["i", "j"], [[0, 1], [1, 2]]),),
        ]
        check_indexing(v_eager, v_lazy, indexers)

        indexers = [
            (Variable("i", [3, 2, 4, 3]), Variable("i", [3, 2, 1, 0])),
            (Variable(["i", "j"], [[0, 1], [1, 2]]),),
        ]
        check_indexing(v_eager, v_lazy, indexers)


class TestCopyOnWriteArray:
    def test_setitem(self):
        original = np.arange(10)
        wrapped = indexing.CopyOnWriteArray(original)
        wrapped[B[:]] = 0
        assert_array_equal(original, np.arange(10))
        assert_array_equal(wrapped, np.zeros(10))

    def test_sub_array(self):
        original = np.arange(10)
        wrapped = indexing.CopyOnWriteArray(original)
        child = wrapped[B[:5]]
        assert isinstance(child, indexing.CopyOnWriteArray)
        child[B[:]] = 0
        assert_array_equal(original, np.arange(10))
        assert_array_equal(wrapped, np.arange(10))
        assert_array_equal(child, np.zeros(5))

    def test_index_scalar(self):
        # regression test for GH1374
        x = indexing.CopyOnWriteArray(np.array(["foo", "bar"]))
        assert np.array(x[B[0]][B[()]]) == "foo"


class TestMemoryCachedArray:
    def test_wrapper(self):
        original = indexing.LazilyOuterIndexedArray(np.arange(10))
        wrapped = indexing.MemoryCachedArray(original)
        assert_array_equal(wrapped, np.arange(10))
        assert isinstance(wrapped.array, indexing.NumpyIndexingAdapter)

    def test_sub_array(self):
        original = indexing.LazilyOuterIndexedArray(np.arange(10))
        wrapped = indexing.MemoryCachedArray(original)
        child = wrapped[B[:5]]
        assert isinstance(child, indexing.MemoryCachedArray)
        assert_array_equal(child, np.arange(5))
        assert isinstance(child.array, indexing.NumpyIndexingAdapter)
        assert isinstance(wrapped.array, indexing.LazilyOuterIndexedArray)

    def test_setitem(self):
        original = np.arange(10)
        wrapped = indexing.MemoryCachedArray(original)
        wrapped[B[:]] = 0
        assert_array_equal(original, np.zeros(10))

    def test_index_scalar(self):
        # regression test for GH1374
        x = indexing.MemoryCachedArray(np.array(["foo", "bar"]))
        assert np.array(x[B[0]][B[()]]) == "foo"


def test_base_explicit_indexer():
    with pytest.raises(TypeError):
        indexing.ExplicitIndexer(())

    class Subclass(indexing.ExplicitIndexer):
        pass

    value = Subclass((1, 2, 3))
    assert value.tuple == (1, 2, 3)
    assert repr(value) == "Subclass((1, 2, 3))"


@pytest.mark.parametrize(
    "indexer_cls",
    [indexing.BasicIndexer, indexing.OuterIndexer, indexing.VectorizedIndexer],
)
def test_invalid_for_all(indexer_cls):
    with pytest.raises(TypeError):
        indexer_cls(None)
    with pytest.raises(TypeError):
        indexer_cls(([],))
    with pytest.raises(TypeError):
        indexer_cls((None,))
    with pytest.raises(TypeError):
        indexer_cls(("foo",))
    with pytest.raises(TypeError):
        indexer_cls((1.0,))
    with pytest.raises(TypeError):
        indexer_cls((slice("foo"),))
    with pytest.raises(TypeError):
        indexer_cls((np.array(["foo"]),))


def check_integer(indexer_cls):
    value = indexer_cls((1, np.uint64(2))).tuple
    assert all(isinstance(v, int) for v in value)
    assert value == (1, 2)


def check_slice(indexer_cls):
    (value,) = indexer_cls((slice(1, None, np.int64(2)),)).tuple
    assert value == slice(1, None, 2)
    assert isinstance(value.step, int)


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
    with raises_regex(ValueError, "numbers of dimensions"):
        indexing.VectorizedIndexer(
            (np.array(1, dtype=np.int64), np.arange(5, dtype=np.int64))
        )


class Test_vectorized_indexer:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.data = indexing.NumpyIndexingAdapter(np.random.randn(10, 12, 13))
        self.indexers = [
            np.array([[0, 3, 2]]),
            np.array([[0, 3, 3], [4, 6, 7]]),
            slice(2, -2, 2),
            slice(2, -2, 3),
            slice(None),
        ]

    def test_arrayize_vectorized_indexer(self):
        for i, j, k in itertools.product(self.indexers, repeat=3):
            vindex = indexing.VectorizedIndexer((i, j, k))
            vindex_array = indexing._arrayize_vectorized_indexer(
                vindex, self.data.shape
            )
            np.testing.assert_array_equal(self.data[vindex], self.data[vindex_array])

        actual = indexing._arrayize_vectorized_indexer(
            indexing.VectorizedIndexer((slice(None),)), shape=(5,)
        )
        np.testing.assert_array_equal(actual.tuple, [np.arange(5)])

        actual = indexing._arrayize_vectorized_indexer(
            indexing.VectorizedIndexer((np.arange(5),) * 3), shape=(8, 10, 12)
        )
        expected = np.stack([np.arange(5)] * 3)
        np.testing.assert_array_equal(np.stack(actual.tuple), expected)

        actual = indexing._arrayize_vectorized_indexer(
            indexing.VectorizedIndexer((np.arange(5), slice(None))), shape=(8, 10)
        )
        a, b = actual.tuple
        np.testing.assert_array_equal(a, np.arange(5)[:, np.newaxis])
        np.testing.assert_array_equal(b, np.arange(10)[np.newaxis, :])

        actual = indexing._arrayize_vectorized_indexer(
            indexing.VectorizedIndexer((slice(None), np.arange(5))), shape=(8, 10)
        )
        a, b = actual.tuple
        np.testing.assert_array_equal(a, np.arange(8)[np.newaxis, :])
        np.testing.assert_array_equal(b, np.arange(5)[:, np.newaxis])


def get_indexers(shape, mode):
    if mode == "vectorized":
        indexed_shape = (3, 4)
        indexer = tuple(np.random.randint(0, s, size=indexed_shape) for s in shape)
        return indexing.VectorizedIndexer(indexer)

    elif mode == "outer":
        indexer = tuple(np.random.randint(0, s, s + 2) for s in shape)
        return indexing.OuterIndexer(indexer)

    elif mode == "outer_scalar":
        indexer = (np.random.randint(0, 3, 4), 0, slice(None, None, 2))
        return indexing.OuterIndexer(indexer[: len(shape)])

    elif mode == "outer_scalar2":
        indexer = (np.random.randint(0, 3, 4), -2, slice(None, None, 2))
        return indexing.OuterIndexer(indexer[: len(shape)])

    elif mode == "outer1vec":
        indexer = [slice(2, -3) for s in shape]
        indexer[1] = np.random.randint(0, shape[1], shape[1] + 2)
        return indexing.OuterIndexer(tuple(indexer))

    elif mode == "basic":  # basic indexer
        indexer = [slice(2, -3) for s in shape]
        indexer[0] = 3
        return indexing.BasicIndexer(tuple(indexer))

    elif mode == "basic1":  # basic indexer
        return indexing.BasicIndexer((3,))

    elif mode == "basic2":  # basic indexer
        indexer = [0, 2, 4]
        return indexing.BasicIndexer(tuple(indexer[: len(shape)]))

    elif mode == "basic3":  # basic indexer
        indexer = [slice(None) for s in shape]
        indexer[0] = slice(-2, 2, -2)
        indexer[1] = slice(1, -1, 2)
        return indexing.BasicIndexer(tuple(indexer[: len(shape)]))


@pytest.mark.parametrize("size", [100, 99])
@pytest.mark.parametrize(
    "sl", [slice(1, -1, 1), slice(None, -1, 2), slice(-1, 1, -1), slice(-1, 1, -2)]
)
def test_decompose_slice(size, sl):
    x = np.arange(size)
    slice1, slice2 = indexing._decompose_slice(sl, size)
    expected = x[sl]
    actual = x[slice1][slice2]
    assert_array_equal(expected, actual)


@pytest.mark.parametrize("shape", [(10, 5, 8), (10, 3)])
@pytest.mark.parametrize(
    "indexer_mode",
    [
        "vectorized",
        "outer",
        "outer_scalar",
        "outer_scalar2",
        "outer1vec",
        "basic",
        "basic1",
        "basic2",
        "basic3",
    ],
)
@pytest.mark.parametrize(
    "indexing_support",
    [
        indexing.IndexingSupport.BASIC,
        indexing.IndexingSupport.OUTER,
        indexing.IndexingSupport.OUTER_1VECTOR,
        indexing.IndexingSupport.VECTORIZED,
    ],
)
def test_decompose_indexers(shape, indexer_mode, indexing_support):
    data = np.random.randn(*shape)
    indexer = get_indexers(shape, indexer_mode)

    backend_ind, np_ind = indexing.decompose_indexer(indexer, shape, indexing_support)

    expected = indexing.NumpyIndexingAdapter(data)[indexer]
    array = indexing.NumpyIndexingAdapter(data)[backend_ind]
    if len(np_ind.tuple) > 0:
        array = indexing.NumpyIndexingAdapter(array)[np_ind]
    np.testing.assert_array_equal(expected, array)

    if not all(isinstance(k, indexing.integer_types) for k in np_ind.tuple):
        combined_ind = indexing._combine_indexers(backend_ind, shape, np_ind)
        array = indexing.NumpyIndexingAdapter(data)[combined_ind]
        np.testing.assert_array_equal(expected, array)


def test_implicit_indexing_adapter():
    array = np.arange(10, dtype=np.int64)
    implicit = indexing.ImplicitToExplicitIndexingAdapter(
        indexing.NumpyIndexingAdapter(array), indexing.BasicIndexer
    )
    np.testing.assert_array_equal(array, np.asarray(implicit))
    np.testing.assert_array_equal(array, implicit[:])


def test_implicit_indexing_adapter_copy_on_write():
    array = np.arange(10, dtype=np.int64)
    implicit = indexing.ImplicitToExplicitIndexingAdapter(
        indexing.CopyOnWriteArray(array)
    )
    assert isinstance(implicit[:], indexing.ImplicitToExplicitIndexingAdapter)


def test_outer_indexer_consistency_with_broadcast_indexes_vectorized():
    def nonzero(x):
        if isinstance(x, np.ndarray) and x.dtype.kind == "b":
            x = x.nonzero()[0]
        return x

    original = np.random.rand(10, 20, 30)
    v = Variable(["i", "j", "k"], original)
    arr = ReturnItem()
    # test orthogonally applied indexers
    indexers = [
        arr[:],
        0,
        -2,
        arr[:3],
        np.array([0, 1, 2, 3]),
        np.array([0]),
        np.arange(10) < 5,
    ]
    for i, j, k in itertools.product(indexers, repeat=3):

        if isinstance(j, np.ndarray) and j.dtype.kind == "b":  # match size
            j = np.arange(20) < 4
        if isinstance(k, np.ndarray) and k.dtype.kind == "b":
            k = np.arange(30) < 8

        _, expected, new_order = v._broadcast_indexes_vectorized((i, j, k))
        expected_data = nputils.NumpyVIndexAdapter(v.data)[expected.tuple]
        if new_order:
            old_order = range(len(new_order))
            expected_data = np.moveaxis(expected_data, old_order, new_order)

        outer_index = indexing.OuterIndexer((nonzero(i), nonzero(j), nonzero(k)))
        actual = indexing._outer_to_numpy_indexer(outer_index, v.shape)
        actual_data = v.data[actual]
        np.testing.assert_array_equal(actual_data, expected_data)


def test_create_mask_outer_indexer():
    indexer = indexing.OuterIndexer((np.array([0, -1, 2]),))
    expected = np.array([False, True, False])
    actual = indexing.create_mask(indexer, (5,))
    np.testing.assert_array_equal(expected, actual)

    indexer = indexing.OuterIndexer((1, slice(2), np.array([0, -1, 2])))
    expected = np.array(2 * [[False, True, False]])
    actual = indexing.create_mask(indexer, (5, 5, 5))
    np.testing.assert_array_equal(expected, actual)


def test_create_mask_vectorized_indexer():
    indexer = indexing.VectorizedIndexer((np.array([0, -1, 2]), np.array([0, 1, -1])))
    expected = np.array([False, True, True])
    actual = indexing.create_mask(indexer, (5,))
    np.testing.assert_array_equal(expected, actual)

    indexer = indexing.VectorizedIndexer(
        (np.array([0, -1, 2]), slice(None), np.array([0, 1, -1]))
    )
    expected = np.array([[False, True, True]] * 2).T
    actual = indexing.create_mask(indexer, (5, 2))
    np.testing.assert_array_equal(expected, actual)


def test_create_mask_basic_indexer():
    indexer = indexing.BasicIndexer((-1,))
    actual = indexing.create_mask(indexer, (3,))
    np.testing.assert_array_equal(True, actual)

    indexer = indexing.BasicIndexer((0,))
    actual = indexing.create_mask(indexer, (3,))
    np.testing.assert_array_equal(False, actual)


def test_create_mask_dask():
    da = pytest.importorskip("dask.array")

    indexer = indexing.OuterIndexer((1, slice(2), np.array([0, -1, 2])))
    expected = np.array(2 * [[False, True, False]])
    actual = indexing.create_mask(
        indexer, (5, 5, 5), da.empty((2, 3), chunks=((1, 1), (2, 1)))
    )
    assert actual.chunks == ((1, 1), (2, 1))
    np.testing.assert_array_equal(expected, actual)

    indexer = indexing.VectorizedIndexer(
        (np.array([0, -1, 2]), slice(None), np.array([0, 1, -1]))
    )
    expected = np.array([[False, True, True]] * 2).T
    actual = indexing.create_mask(
        indexer, (5, 2), da.empty((3, 2), chunks=((3,), (2,)))
    )
    assert isinstance(actual, da.Array)
    np.testing.assert_array_equal(expected, actual)

    with pytest.raises(ValueError):
        indexing.create_mask(indexer, (5, 2), da.empty((5,), chunks=(1,)))


def test_create_mask_error():
    with raises_regex(TypeError, "unexpected key type"):
        indexing.create_mask((1, 2), (3, 4))


@pytest.mark.parametrize(
    "indices, expected",
    [
        (np.arange(5), np.arange(5)),
        (np.array([0, -1, -1]), np.array([0, 0, 0])),
        (np.array([-1, 1, -1]), np.array([1, 1, 1])),
        (np.array([-1, -1, 2]), np.array([2, 2, 2])),
        (np.array([-1]), np.array([0])),
        (np.array([0, -1, 1, -1, -1]), np.array([0, 0, 1, 1, 1])),
        (np.array([0, -1, -1, -1, 1]), np.array([0, 0, 0, 0, 1])),
    ],
)
def test_posify_mask_subindexer(indices, expected):
    actual = indexing._posify_mask_subindexer(indices)
    np.testing.assert_array_equal(expected, actual)
