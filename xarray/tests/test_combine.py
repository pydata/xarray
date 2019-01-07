from __future__ import absolute_import, division, print_function

from copy import deepcopy
from itertools import product

import numpy as np
import numpy.testing as npt
import pandas as pd
import pytest

from xarray import DataArray, Dataset, concat, auto_combine, manual_combine
from xarray.core.combine import (
    _new_tile_id, _check_shape_tile_ids,
    _combine_all_along_first_dim, _combine_nd, _all_arrays_equal,
    _infer_concat_order_from_positions, _infer_tile_ids_from_nested_list,
    _infer_concat_order_from_coords, _infer_order_1d)
from xarray.core.pycompat import OrderedDict

from . import (assert_combined_tile_ids_equal, assert_identical, assert_equal,
               raises_regex)
from .test_dataset import create_test_data


class TestTileIDsFromNestedList(object):
    def test_1d(self):
        ds = create_test_data
        input = [ds(0), ds(1)]

        expected = {(0,): ds(0), (1,): ds(1)}
        actual = dict(_infer_tile_ids_from_nested_list(input, ()))
        assert_combined_tile_ids_equal(expected, actual)

    def test_2d(self):
        ds = create_test_data
        input = [[ds(0), ds(1)], [ds(2), ds(3)], [ds(4), ds(5)]]

        expected = {(0, 0): ds(0), (0, 1): ds(1),
                    (1, 0): ds(2), (1, 1): ds(3),
                    (2, 0): ds(4), (2, 1): ds(5)}
        actual = dict(_infer_tile_ids_from_nested_list(input, ()))
        assert_combined_tile_ids_equal(expected, actual)

    def test_3d(self):
        ds = create_test_data
        input = [[[ds(0), ds(1)], [ds(2), ds(3)], [ds(4), ds(5)]],
                 [[ds(6), ds(7)], [ds(8), ds(9)], [ds(10), ds(11)]]]

        expected = {(0, 0, 0): ds(0), (0, 0, 1): ds(1),
                    (0, 1, 0): ds(2), (0, 1, 1): ds(3),
                    (0, 2, 0): ds(4), (0, 2, 1): ds(5),
                    (1, 0, 0): ds(6), (1, 0, 1): ds(7),
                    (1, 1, 0): ds(8), (1, 1, 1): ds(9),
                    (1, 2, 0): ds(10), (1, 2, 1): ds(11)}
        actual = dict(_infer_tile_ids_from_nested_list(input, ()))
        assert_combined_tile_ids_equal(expected, actual)

    def test_single_dataset(self):
        ds = create_test_data(0)
        input = [ds]

        expected = {(0,): ds}
        actual = dict(_infer_tile_ids_from_nested_list(input, ()))
        assert_combined_tile_ids_equal(expected, actual)

    def test_redundant_nesting(self):
        ds = create_test_data
        input = [[ds(0)], [ds(1)]]

        expected = {(0, 0): ds(0), (1, 0): ds(1)}
        actual = dict(_infer_tile_ids_from_nested_list(input, ()))
        assert_combined_tile_ids_equal(expected, actual)

    def test_ignore_empty_list(self):
        ds = create_test_data(0)
        input = [ds, []]
        expected = {(0,): ds}
        actual = dict(_infer_tile_ids_from_nested_list(input, ()))
        assert_combined_tile_ids_equal(expected, actual)

    def test_uneven_depth_input(self):
        # Auto_combine won't work on ragged input
        # but this is just to increase test coverage
        ds = create_test_data
        input = [ds(0), [ds(1), ds(2)]]

        expected = {(0,): ds(0), (1, 0): ds(1), (1, 1): ds(2)}
        actual = dict(_infer_tile_ids_from_nested_list(input, ()))
        assert_combined_tile_ids_equal(expected, actual)

    def test_uneven_length_input(self):
        # Auto_combine won't work on ragged input
        # but this is just to increase test coverage
        ds = create_test_data
        input = [[ds(0)], [ds(1), ds(2)]]

        expected = {(0, 0): ds(0), (1, 0): ds(1), (1, 1): ds(2)}
        actual = dict(_infer_tile_ids_from_nested_list(input, ()))
        assert_combined_tile_ids_equal(expected, actual)

    def test_infer_from_datasets(self):
        ds = create_test_data
        input = [ds(0), ds(1)]

        expected = {(0,): ds(0), (1,): ds(1)}
        actual, concat_dims = _infer_concat_order_from_positions(input, [
                                                                 'dim1'])
        assert_combined_tile_ids_equal(expected, actual)

        input = [ds(0), ds(1)]
        with pytest.raises(ValueError):
            _infer_concat_order_from_positions(input, ['dim1', 'extra_dim'])


class TestInferOrder1D(object):
    def test_arrays(self):
        npt.assert_equal(_infer_order_1d([3, 1, 2, 7]), np.array([2, 0, 1, 3]))
        npt.assert_equal(_infer_order_1d([5, 7, 8, 8]), np.array([0, 1, 2, 2]))
        npt.assert_equal(_infer_order_1d([2, 2, 0]),    np.array([1, 1, 0]))
        npt.assert_equal(_infer_order_1d([2, 5, 5, 1]), np.array([1, 2, 2, 0]))

    @pytest.mark.xfail
    def test_strings(self):
        npt.assert_equal(_infer_order_1d(['b', 'a']), np.array([1, 0]))
        npt.assert_equal(_infer_order_1d(['aa', 'a']), np.array([1, 0]))
        npt.assert_equal(_infer_order_1d(['c1', 'c0']), np.array([1, 0]))

        npt.assert_equal(_infer_order_1d(['c1', 'c0', 'c0']),
                         np.array([1, 0, 0]))

        # Natural sorting
        npt.assert_equal(_infer_order_1d(['c1', 'c0', 'c10']),
                         np.array([1, 0, 2]))

    @pytest.mark.skip
    def test_datetimes(self):
        pass


def test_all_arrays_equal():
    assert _all_arrays_equal([np.array([1,2,3]),
                              np.array([1,2,3]),
                              np.array([1,2,3])])
    assert not _all_arrays_equal([np.array([1, 2, 3]),
                                  np.array([1, 2, 3]),
                                  np.array([1, 2, 4])])


class TestTileIDsFromCoords(object):
    def test_1d(self):
        ds0 = Dataset({'x': [0, 1]})
        ds1 = Dataset({'x': [2, 3]})

        expected = {(0,): ds0, (1,): ds1}
        actual, concat_dims = _infer_concat_order_from_coords([ds1, ds0])
        assert_combined_tile_ids_equal(expected, actual)
        assert concat_dims == ['x']

    def test_2d(self):
        ds0 = Dataset({'x': [0, 1], 'y': [10, 20, 30]})
        ds1 = Dataset({'x': [2, 3], 'y': [10, 20, 30]})
        ds2 = Dataset({'x': [0, 1], 'y': [40, 50, 60]})
        ds3 = Dataset({'x': [2, 3], 'y': [40, 50, 60]})
        ds4 = Dataset({'x': [0, 1], 'y': [70, 80, 90]})
        ds5 = Dataset({'x': [2, 3], 'y': [70, 80, 90]})

        expected = {(0, 0): ds0, (1, 0): ds1,
                    (0, 1): ds2, (1, 1): ds3,
                    (0, 2): ds4, (1, 2): ds5}
        actual, concat_dims = _infer_concat_order_from_coords([ds1, ds0, ds3,
                                                               ds5, ds2, ds4])
        assert_combined_tile_ids_equal(expected, actual)
        assert concat_dims == ['x', 'y']

    def test_no_dimension_coords(self):
        ds0 = Dataset({'foo': ('x', [0, 1])})
        ds1 = Dataset({'foo': ('x', [2, 3])})
        with raises_regex(ValueError, "Could not find any dimension"):
            _infer_concat_order_from_coords([ds1, ds0])

    def test_coord_not_monotonic(self):
        ds0 = Dataset({'x': [0, 1]})
        ds1 = Dataset({'x': [3, 2]})
        with raises_regex(ValueError, "Coordinate variable x is not "
                                      "monotonically increasing"):
            _infer_concat_order_from_coords([ds1, ds0])

    # TODO implement this error message
    @pytest.mark.xfail
    def test_check_for_impossible_ordering(self):
        ds0 = Dataset({'x': [0, 1, 5]})
        ds1 = Dataset({'x': [2, 3]})
        with raises_regex(ValueError, "Unable to arrange datasets such that "
                                      "coordinate values along dimension x are"
                                      " monotonically increasing"):
            _infer_concat_order_from_coords([ds1, ds0])

    def test_no_concatenation_needed(self):
        ds = Dataset({'foo': ('x', [0, 1])})
        expected = {(): ds}
        actual, concat_dims = _infer_concat_order_from_coords([ds])
        assert_combined_tile_ids_equal(expected, actual)
        assert concat_dims == []

    def test_2d_plus_bystander_dim(self):
        ds0 = Dataset({'x': [0, 1], 'y': [10, 20, 30], 't': [0.1, 0.2]})
        ds1 = Dataset({'x': [2, 3], 'y': [10, 20, 30], 't': [0.1, 0.2]})
        ds2 = Dataset({'x': [0, 1], 'y': [40, 50, 60], 't': [0.1, 0.2]})
        ds3 = Dataset({'x': [2, 3], 'y': [40, 50, 60], 't': [0.1, 0.2]})

        expected = {(0, 0): ds0, (1, 0): ds1,
                    (0, 1): ds2, (1, 1): ds3}
        actual, concat_dims = _infer_concat_order_from_coords([ds1, ds0,
                                                               ds3, ds2])
        assert_combined_tile_ids_equal(expected, actual)
        assert concat_dims == ['x', 'y']

    @pytest.mark.skip
    def test_string_coord(self):
        pass

    @pytest.mark.skip
    def test_datetime_coord(self):
        pass


@pytest.fixture(scope='module')
def create_combined_ids():
    return _create_combined_ids


def _create_combined_ids(shape):
    tile_ids = _create_tile_ids(shape)
    nums = range(len(tile_ids))
    return {tile_id: create_test_data(num)
            for tile_id, num in zip(tile_ids, nums)}


def _create_tile_ids(shape):
    tile_ids = product(*(range(i) for i in shape))
    return list(tile_ids)


class TestNewTileIDs(object):
    @pytest.mark.parametrize("old_id, new_id", [((3, 0, 1), (0, 1)),
                                                ((0, 0), (0,)),
                                                ((1,), ()),
                                                ((0,), ()),
                                                ((1, 0), (0,))])
    def test_new_tile_id(self, old_id, new_id):
        ds = create_test_data
        assert _new_tile_id((old_id, ds)) == new_id

    def test_get_new_tile_ids(self, create_combined_ids):
        shape = (1, 2, 3)
        combined_ids = create_combined_ids(shape)

        expected_tile_ids = sorted(combined_ids.keys())
        actual_tile_ids = _create_tile_ids(shape)
        assert expected_tile_ids == actual_tile_ids


class TestCombineND(object):
    @pytest.mark.parametrize("concat_dim", ['dim1', 'new_dim'])
    def test_concat_once(self, create_combined_ids, concat_dim):
        shape = (2,)
        combined_ids = create_combined_ids(shape)
        ds = create_test_data
        result = _combine_all_along_first_dim(combined_ids, dim=concat_dim,
                                              data_vars='all',
                                              coords='different',
                                              compat='no_conflicts')

        expected_ds = concat([ds(0), ds(1)], dim=concat_dim)
        assert_combined_tile_ids_equal(result, {(): expected_ds})

    def test_concat_only_first_dim(self, create_combined_ids):
        shape = (2, 3)
        combined_ids = create_combined_ids(shape)
        result = _combine_all_along_first_dim(combined_ids,
                                                   dim='dim1',
                                                   data_vars='all',
                                                   coords='different',
                                                   compat='no_conflicts')

        ds = create_test_data
        partway1 = concat([ds(0), ds(3)], dim='dim1')
        partway2 = concat([ds(1), ds(4)], dim='dim1')
        partway3 = concat([ds(2), ds(5)], dim='dim1')
        expected_datasets = [partway1, partway2, partway3]
        expected = {(i,): ds for i, ds in enumerate(expected_datasets)}

        assert_combined_tile_ids_equal(result, expected)

    @pytest.mark.parametrize("concat_dim", ['dim1', 'new_dim'])
    def test_concat_twice(self, create_combined_ids, concat_dim):
        shape = (2, 3)
        combined_ids = create_combined_ids(shape)
        result = _combine_nd(combined_ids, concat_dims=['dim1', concat_dim])

        ds = create_test_data
        partway1 = concat([ds(0), ds(3)], dim='dim1')
        partway2 = concat([ds(1), ds(4)], dim='dim1')
        partway3 = concat([ds(2), ds(5)], dim='dim1')
        expected = concat([partway1, partway2, partway3], dim=concat_dim)

        assert_equal(result, expected)


class TestCheckShapeTileIDs(object):
    def test_check_depths(self):
        ds = create_test_data(0)
        combined_tile_ids = {(0,): ds, (0, 1): ds}
        with raises_regex(ValueError, 'sub-lists do not have '
                                      'consistent depths'):
            _check_shape_tile_ids(combined_tile_ids)

    def test_check_lengths(self):
        ds = create_test_data(0)
        combined_tile_ids = {(0, 0): ds, (0, 1): ds, (0, 2): ds,
                             (1, 0): ds, (1, 1): ds}
        with raises_regex(ValueError, 'sub-lists do not have '
                                      'consistent lengths'):
            _check_shape_tile_ids(combined_tile_ids)


class TestManualCombine(object):
    def test_manual_concat(self):
        objs = [Dataset({'x': [0]}), Dataset({'x': [1]})]
        expected = Dataset({'x': [0, 1]})
        actual = manual_combine(objs, concat_dim='x')
        assert_identical(expected, actual)
        actual = manual_combine(objs, concat_dim=['x'])
        assert_identical(expected, actual)

        actual = manual_combine([actual], concat_dim=None)
        assert_identical(expected, actual)

        objs = [Dataset({'x': [0, 1]}), Dataset({'x': [2]})]
        actual = manual_combine(objs, concat_dim='x')
        expected = Dataset({'x': [0, 1, 2]})
        assert_identical(expected, actual)

        # ensure manual_combine handles non-sorted variables
        objs = [Dataset(OrderedDict([('x', ('a', [0])), ('y', ('a', [0]))])),
                Dataset(OrderedDict([('y', ('a', [1])), ('x', ('a', [1]))]))]
        actual = manual_combine(objs, concat_dim='a')
        expected = Dataset({'x': ('a', [0, 1]), 'y': ('a', [0, 1])})
        assert_identical(expected, actual)

        objs = [Dataset({'x': [0], 'y': [0]}), Dataset({'x': [0]})]
        with pytest.raises(KeyError):
            manual_combine(objs, concat_dim='x')

    # TODO weird error from auto_concat on both of these when it tries to infer dimension?
    @pytest.mark.xfail
    def test_manual_concat_too_many_dims_at_once(self):
        objs = [Dataset({'x': [0], 'y': [0]}), Dataset({'y': [1], 'x': [1]})]
        with raises_regex(ValueError, 'too many .* dimensions'):
           manual_combine(objs)

        objs = [Dataset({'x': 0}), Dataset({'x': 1})]
        with raises_regex(ValueError, 'cannot infer dimension'):
            manual_combine(objs)

    def test_manual_concat_along_new_dim(self):
        objs = [Dataset({'a': ('x', [10]), 'x': [0]}),
                Dataset({'a': ('x', [20]), 'x': [0]})]
        expected = Dataset({'a': (('t', 'x'), [[10], [20]]), 'x': [0]})
        actual = manual_combine(objs, concat_dim='t')
        assert_identical(expected, actual)

        # Same but with a DataArray as new dim, see GH #1988 and #2647
        dim = DataArray([100, 150], name='baz', dims='baz')
        expected = Dataset({'a': (('baz', 'x'), [[10], [20]]),
                            'x': [0], 'baz': [100, 150]})
        actual = manual_combine(objs, concat_dim=dim)
        assert_identical(expected, actual)

    def test_manual_merge(self):
        data = Dataset({'x': 0})
        actual = manual_combine([data, data, data], concat_dim=None)
        assert_identical(data, actual)

        ds1 = Dataset({'a': ('x', [1, 2]), 'x': [0, 1]})
        ds2 = Dataset({'a': ('x', [2, 3]), 'x': [1, 2]})
        expected = Dataset({'a': ('x', [1, 2, 3]), 'x': [0, 1, 2]})
        actual = manual_combine([ds1, ds2], concat_dim=None)
        assert_identical(expected, actual)
        actual = manual_combine([ds1, ds2], concat_dim=[None])
        assert_identical(expected, actual)

        tmp1 = Dataset({'x': 0})
        tmp2 = Dataset({'x': np.nan})
        actual = manual_combine([tmp1, tmp2], concat_dim=None)
        assert_identical(tmp1, actual)
        actual = manual_combine([tmp1, tmp2], concat_dim=[None])
        assert_identical(tmp1, actual)

        # Single object, with a concat_dim explicitly provided
        # Test the issue reported in GH #1988
        objs = [Dataset({'x': 0, 'y': 1})]
        dim = DataArray([100], name='baz', dims='baz')
        actual = manual_combine(objs, concat_dim=[dim])
        expected = Dataset({'x': ('baz', [0]), 'y': ('baz', [1])},
                           {'baz': [100]})
        assert_identical(expected, actual)

        # Just making sure that auto_combine is doing what is
        # expected for non-scalar values, too.
        objs = [Dataset({'x': ('z', [0, 1]), 'y': ('z', [1, 2])})]
        dim = DataArray([100], name='baz', dims='baz')
        actual = manual_combine(objs, concat_dim=[dim])
        expected = Dataset({'x': (('baz', 'z'), [[0, 1]]),
                            'y': (('baz', 'z'), [[1, 2]])},
                           {'baz': [100]})
        assert_identical(expected, actual)

    def test_concat_multiple_dims(self):
        objs = [[Dataset({'a': (('x', 'y'), [[0]])}),
                 Dataset({'a': (('x', 'y'), [[1]])})],
                [Dataset({'a': (('x', 'y'), [[2]])}),
                 Dataset({'a': (('x', 'y'), [[3]])})]]
        actual = manual_combine(objs, concat_dim=['x', 'y'])
        expected = Dataset({'a': (('x', 'y'), [[0, 1], [2, 3]])})
        assert_identical(expected, actual)

    def test_concat_one_dim_merge_another(self):
        data = create_test_data()
        data1 = data.copy(deep=True)
        data2 = data.copy(deep=True)

        objs = [[data1.var1.isel(dim2=slice(4)),
                 data2.var1.isel(dim2=slice(4, 9))],
                [data1.var2.isel(dim2=slice(4)),
                 data2.var2.isel(dim2=slice(4, 9))]]

        expected = data[['var1', 'var2']]
        actual = manual_combine(objs, concat_dim=[None, 'dim2'])
        assert expected.identical(actual)

    def test_auto_combine_2d(self):
        ds = create_test_data

        partway1 = concat([ds(0), ds(3)], dim='dim1')
        partway2 = concat([ds(1), ds(4)], dim='dim1')
        partway3 = concat([ds(2), ds(5)], dim='dim1')
        expected = concat([partway1, partway2, partway3], dim='dim2')

        datasets = [[ds(0), ds(1), ds(2)], [ds(3), ds(4), ds(5)]]
        result = manual_combine(datasets, concat_dim=['dim1', 'dim2'])
        assert_equal(result, expected)

    def test_manual_combine_missing_data_new_dim(self):
        # Your data includes "time" and "station" dimensions, and each year's
        # data has a different set of stations.
        datasets = [Dataset({'a': ('x', [2, 3]), 'x': [1, 2]}),
                    Dataset({'a': ('x', [1, 2]), 'x': [0, 1]})]
        expected = Dataset({'a': (('t', 'x'),
                                  [[np.nan, 2, 3], [1, 2, np.nan]])},
                           {'x': [0, 1, 2]})
        actual = manual_combine(datasets, concat_dim='t')
        assert_identical(expected, actual)

    def test_invalid_hypercube_input(self):
        ds = create_test_data

        datasets = [[ds(0), ds(1), ds(2)], [ds(3), ds(4)]]
        with raises_regex(ValueError, 'sub-lists do not have '
                                      'consistent lengths'):
            manual_combine(datasets, concat_dim=['dim1', 'dim2'])

        datasets = [[ds(0), ds(1)], [[ds(3), ds(4)]]]
        with raises_regex(ValueError, 'sub-lists do not have '
                                      'consistent depths'):
            manual_combine(datasets, concat_dim=['dim1', 'dim2'])

        datasets = [[ds(0), ds(1)], [ds(3), ds(4)]]
        with raises_regex(ValueError, 'concat_dims has length'):
            manual_combine(datasets, concat_dim=['dim1'])

    def test_merge_one_dim_concat_another(self):
        objs = [[Dataset({'foo': ('x', [0, 1])}),
                 Dataset({'bar': ('x', [10, 20])})],
                [Dataset({'foo': ('x', [2, 3])}),
                 Dataset({'bar': ('x', [30, 40])})]]
        expected = Dataset({'foo': ('x', [0, 1, 2, 3]),
                            'bar': ('x', [10, 20, 30, 40])})

        actual = manual_combine(objs, concat_dim=['x', None])
        assert_identical(expected, actual)

        # Proving it works symmetrically
        objs = [[Dataset({'foo': ('x', [0, 1])}),
                 Dataset({'foo': ('x', [2, 3])})],
                [Dataset({'bar': ('x', [10, 20])}),
                 Dataset({'bar': ('x', [30, 40])})]]
        actual = manual_combine(objs, concat_dim=[None, 'x'])
        assert_identical(expected, actual)

    def test_combine_concat_over_redundant_nesting(self):
        objs = [[Dataset({'x': [0]}), Dataset({'x': [1]})]]
        actual = manual_combine(objs, concat_dim=[None, 'x'])
        expected = Dataset({'x': [0, 1]})
        assert_identical(expected, actual)

        objs = [[Dataset({'x': [0]})], [Dataset({'x': [1]})]]
        actual = manual_combine(objs, concat_dim=['x', None])
        expected = Dataset({'x': [0, 1]})
        assert_identical(expected, actual)

        objs = [[Dataset({'x': [0]})]]
        actual = manual_combine(objs, concat_dim=[None, None])
        expected = Dataset({'x': [0]})
        assert_identical(expected, actual)


class TestAutoCombine(object):
    def test_auto_combine(self):
        objs = [Dataset({'x': [0]}), Dataset({'x': [1]})]
        actual = auto_combine(objs)
        expected = Dataset({'x': [0, 1]})
        assert_identical(expected, actual)

        actual = auto_combine([actual])
        assert_identical(expected, actual)

        objs = [Dataset({'x': [0, 1]}), Dataset({'x': [2]})]
        actual = auto_combine(objs)
        expected = Dataset({'x': [0, 1, 2]})
        assert_identical(expected, actual)

        # ensure auto_combine handles non-sorted variables
        objs = [Dataset({'x': ('a', [0]), 'y': ('a', [0]), 'a': [0]}),
                Dataset({'x': ('a', [1]), 'y': ('a', [1]), 'a': [1]})]
        actual = auto_combine(objs)
        expected = Dataset({'x': ('a', [0, 1]), 'y': ('a', [0, 1]), 'a': [0, 1]})
        assert_identical(expected, actual)

        # TODO check this is the desired behaviour
        objs = [Dataset({'x': [0], 'y': [0]}), Dataset({'y': [1], 'x': [1]})]
        actual = auto_combine(objs)
        expected = Dataset({'x': [0, 1], 'y': [0, 1]})
        assert_equal(actual, expected)

        objs = [Dataset({'x': 0}), Dataset({'x': 1})]
        with raises_regex(ValueError, 'Could not find any dimension '
                                      'coordinates'):
            auto_combine(objs)

        objs = [Dataset({'x': [0], 'y': [0]}), Dataset({'x': [0]})]
        with pytest.raises(KeyError):
            auto_combine(objs)

    def test_infer_order_from_coords(self):
        data = create_test_data()
        objs = [data.isel(dim2=slice(4, 9)), data.isel(dim2=slice(4))]
        actual = auto_combine(objs)
        expected = data
        assert expected.broadcast_equals(actual)

    def test_auto_combine_previously_failed(self):
        # In the above scenario, one file is missing, containing the data for
        # one year's data for one variable.
        datasets = [Dataset({'a': ('x', [0]), 'x': [0]}),
                    Dataset({'b': ('x', [0]), 'x': [0]}),
                    Dataset({'a': ('x', [1]), 'x': [1]})]
        expected = Dataset({'a': ('x', [0, 1]), 'b': ('x', [0, np.nan])},
                           {'x': [0, 1]})
        actual = auto_combine(datasets)
        assert_identical(expected, actual)

    def test_auto_combine_still_fails(self):
        # concat can't handle new variables (yet):
        # https://github.com/pydata/xarray/issues/508
        datasets = [Dataset({'x': 0}, {'y': 0}),
                    Dataset({'x': 1}, {'y': 1, 'z': 1})]
        with pytest.raises(ValueError):
            auto_combine(datasets, 'y')

    def test_auto_combine_no_concat(self):
        objs = [Dataset({'x': 0}), Dataset({'y': 1})]
        actual = auto_combine(objs)
        expected = Dataset({'x': 0, 'y': 1})
        assert_identical(expected, actual)

        objs = [Dataset({'x': 0, 'y': 1}), Dataset({'y': np.nan, 'z': 2})]
        actual = auto_combine(objs)
        expected = Dataset({'x': 0, 'y': 1, 'z': 2})
        assert_identical(expected, actual)
