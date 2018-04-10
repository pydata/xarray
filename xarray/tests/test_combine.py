from __future__ import absolute_import, division, print_function

from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from xarray import DataArray, Dataset, Variable, auto_combine, concat
from xarray.core.pycompat import OrderedDict, iteritems

from . import (
    InaccessibleArray, TestCase, assert_array_equal, assert_equal,
    assert_identical, raises_regex, requires_dask)
from .test_dataset import create_test_data


class TestConcatDataset(TestCase):
    def test_concat(self):
        # TODO: simplify and split this test case

        # drop the third dimension to keep things relatively understandable
        data = create_test_data()
        for k in list(data.variables):
            if 'dim3' in data[k].dims:
                del data[k]

        split_data = [data.isel(dim1=slice(3)),
                      data.isel(dim1=slice(3, None))]
        assert_identical(data, concat(split_data, 'dim1'))

        def rectify_dim_order(dataset):
            # return a new dataset with all variable dimensions transposed into
            # the order in which they are found in `data`
            return Dataset(dict((k, v.transpose(*data[k].dims))
                                for k, v in iteritems(dataset.data_vars)),
                           dataset.coords, attrs=dataset.attrs)

        for dim in ['dim1', 'dim2']:
            datasets = [g for _, g in data.groupby(dim, squeeze=False)]
            assert_identical(data, concat(datasets, dim))

        dim = 'dim2'
        assert_identical(
            data, concat(datasets, data[dim]))
        assert_identical(
            data, concat(datasets, data[dim], coords='minimal'))

        datasets = [g for _, g in data.groupby(dim, squeeze=True)]
        concat_over = [k for k, v in iteritems(data.coords)
                       if dim in v.dims and k != dim]
        actual = concat(datasets, data[dim], coords=concat_over)
        assert_identical(data, rectify_dim_order(actual))

        actual = concat(datasets, data[dim], coords='different')
        assert_identical(data, rectify_dim_order(actual))

        # make sure the coords argument behaves as expected
        data.coords['extra'] = ('dim4', np.arange(3))
        for dim in ['dim1', 'dim2']:
            datasets = [g for _, g in data.groupby(dim, squeeze=True)]
            actual = concat(datasets, data[dim], coords='all')
            expected = np.array([data['extra'].values
                                 for _ in range(data.dims[dim])])
            assert_array_equal(actual['extra'].values, expected)

            actual = concat(datasets, data[dim], coords='different')
            assert_equal(data['extra'], actual['extra'])
            actual = concat(datasets, data[dim], coords='minimal')
            assert_equal(data['extra'], actual['extra'])

        # verify that the dim argument takes precedence over
        # concatenating dataset variables of the same name
        dim = (2 * data['dim1']).rename('dim1')
        datasets = [g for _, g in data.groupby('dim1', squeeze=False)]
        expected = data.copy()
        expected['dim1'] = dim
        assert_identical(expected, concat(datasets, dim))

    def test_concat_data_vars(self):
        data = Dataset({'foo': ('x', np.random.randn(10))})
        objs = [data.isel(x=slice(5)), data.isel(x=slice(5, None))]
        for data_vars in ['minimal', 'different', 'all', [], ['foo']]:
            actual = concat(objs, dim='x', data_vars=data_vars)
            assert_identical(data, actual)

    def test_concat_coords(self):
        data = Dataset({'foo': ('x', np.random.randn(10))})
        expected = data.assign_coords(c=('x', [0] * 5 + [1] * 5))
        objs = [data.isel(x=slice(5)).assign_coords(c=0),
                data.isel(x=slice(5, None)).assign_coords(c=1)]
        for coords in ['different', 'all', ['c']]:
            actual = concat(objs, dim='x', coords=coords)
            assert_identical(expected, actual)
        for coords in ['minimal', []]:
            with raises_regex(ValueError, 'not equal across'):
                concat(objs, dim='x', coords=coords)

    def test_concat_constant_index(self):
        # GH425
        ds1 = Dataset({'foo': 1.5}, {'y': 1})
        ds2 = Dataset({'foo': 2.5}, {'y': 1})
        expected = Dataset({'foo': ('y', [1.5, 2.5]), 'y': [1, 1]})
        for mode in ['different', 'all', ['foo']]:
            actual = concat([ds1, ds2], 'y', data_vars=mode)
            assert_identical(expected, actual)
        with raises_regex(ValueError, 'not equal across datasets'):
            concat([ds1, ds2], 'y', data_vars='minimal')

    def test_concat_size0(self):
        data = create_test_data()
        split_data = [data.isel(dim1=slice(0, 0)), data]
        actual = concat(split_data, 'dim1')
        assert_identical(data, actual)

        actual = concat(split_data[::-1], 'dim1')
        assert_identical(data, actual)

    def test_concat_autoalign(self):
        ds1 = Dataset({'foo': DataArray([1, 2], coords=[('x', [1, 2])])})
        ds2 = Dataset({'foo': DataArray([1, 2], coords=[('x', [1, 3])])})
        actual = concat([ds1, ds2], 'y')
        expected = Dataset({'foo': DataArray([[1, 2, np.nan], [1, np.nan, 2]],
                                             dims=['y', 'x'],
                                             coords={'x': [1, 2, 3]})})
        assert_identical(expected, actual)

    def test_concat_errors(self):
        data = create_test_data()
        split_data = [data.isel(dim1=slice(3)),
                      data.isel(dim1=slice(3, None))]

        with raises_regex(ValueError, 'must supply at least one'):
            concat([], 'dim1')

        with raises_regex(ValueError, 'are not coordinates'):
            concat([data, data], 'new_dim', coords=['not_found'])

        with raises_regex(ValueError, 'global attributes not'):
            data0, data1 = deepcopy(split_data)
            data1.attrs['foo'] = 'bar'
            concat([data0, data1], 'dim1', compat='identical')
        assert_identical(
            data, concat([data0, data1], 'dim1', compat='equals'))

        with raises_regex(ValueError, 'encountered unexpected'):
            data0, data1 = deepcopy(split_data)
            data1['foo'] = ('bar', np.random.randn(10))
            concat([data0, data1], 'dim1')

        with raises_regex(ValueError, 'compat.* invalid'):
            concat(split_data, 'dim1', compat='foobar')

        with raises_regex(ValueError, 'unexpected value for'):
            concat([data, data], 'new_dim', coords='foobar')

        with raises_regex(
                ValueError, 'coordinate in some datasets but not others'):
            concat([Dataset({'x': 0}), Dataset({'x': [1]})], dim='z')

        with raises_regex(
                ValueError, 'coordinate in some datasets but not others'):
            concat([Dataset({'x': 0}), Dataset({}, {'x': 1})], dim='z')

        with raises_regex(ValueError, 'no longer a valid'):
            concat([data, data], 'new_dim', mode='different')
        with raises_regex(ValueError, 'no longer a valid'):
            concat([data, data], 'new_dim', concat_over='different')

    def test_concat_promote_shape(self):
        # mixed dims within variables
        objs = [Dataset({}, {'x': 0}), Dataset({'x': [1]})]
        actual = concat(objs, 'x')
        expected = Dataset({'x': [0, 1]})
        assert_identical(actual, expected)

        objs = [Dataset({'x': [0]}), Dataset({}, {'x': 1})]
        actual = concat(objs, 'x')
        assert_identical(actual, expected)

        # mixed dims between variables
        objs = [Dataset({'x': [2], 'y': 3}), Dataset({'x': [4], 'y': 5})]
        actual = concat(objs, 'x')
        expected = Dataset({'x': [2, 4], 'y': ('x', [3, 5])})
        assert_identical(actual, expected)

        # mixed dims in coord variable
        objs = [Dataset({'x': [0]}, {'y': -1}),
                Dataset({'x': [1]}, {'y': ('x', [-2])})]
        actual = concat(objs, 'x')
        expected = Dataset({'x': [0, 1]}, {'y': ('x', [-1, -2])})
        assert_identical(actual, expected)

        # scalars with mixed lengths along concat dim -- values should repeat
        objs = [Dataset({'x': [0]}, {'y': -1}),
                Dataset({'x': [1, 2]}, {'y': -2})]
        actual = concat(objs, 'x')
        expected = Dataset({'x': [0, 1, 2]}, {'y': ('x', [-1, -2, -2])})
        assert_identical(actual, expected)

        # broadcast 1d x 1d -> 2d
        objs = [Dataset({'z': ('x', [-1])}, {'x': [0], 'y': [0]}),
                Dataset({'z': ('y', [1])}, {'x': [1], 'y': [0]})]
        actual = concat(objs, 'x')
        expected = Dataset({'z': (('x', 'y'), [[-1], [1]])},
                           {'x': [0, 1], 'y': [0]})
        assert_identical(actual, expected)

    def test_concat_do_not_promote(self):
        # GH438
        objs = [Dataset({'y': ('t', [1])}, {'x': 1, 't': [0]}),
                Dataset({'y': ('t', [2])}, {'x': 1, 't': [0]})]
        expected = Dataset({'y': ('t', [1, 2])}, {'x': 1, 't': [0, 0]})
        actual = concat(objs, 't')
        assert_identical(expected, actual)

        objs = [Dataset({'y': ('t', [1])}, {'x': 1, 't': [0]}),
                Dataset({'y': ('t', [2])}, {'x': 2, 't': [0]})]
        with pytest.raises(ValueError):
            concat(objs, 't', coords='minimal')

    def test_concat_dim_is_variable(self):
        objs = [Dataset({'x': 0}), Dataset({'x': 1})]
        coord = Variable('y', [3, 4])
        expected = Dataset({'x': ('y', [0, 1]), 'y': [3, 4]})
        actual = concat(objs, coord)
        assert_identical(actual, expected)

    def test_concat_multiindex(self):
        x = pd.MultiIndex.from_product([[1, 2, 3], ['a', 'b']])
        expected = Dataset({'x': x})
        actual = concat([expected.isel(x=slice(2)),
                         expected.isel(x=slice(2, None))], 'x')
        assert expected.equals(actual)
        assert isinstance(actual.x.to_index(), pd.MultiIndex)


class TestConcatDataArray(TestCase):
    def test_concat(self):
        ds = Dataset({'foo': (['x', 'y'], np.random.random((2, 3))),
                      'bar': (['x', 'y'], np.random.random((2, 3)))},
                     {'x': [0, 1]})
        foo = ds['foo']
        bar = ds['bar']

        # from dataset array:
        expected = DataArray(np.array([foo.values, bar.values]),
                             dims=['w', 'x', 'y'], coords={'x': [0, 1]})
        actual = concat([foo, bar], 'w')
        assert_equal(expected, actual)
        # from iteration:
        grouped = [g for _, g in foo.groupby('x')]
        stacked = concat(grouped, ds['x'])
        assert_identical(foo, stacked)
        # with an index as the 'dim' argument
        stacked = concat(grouped, ds.indexes['x'])
        assert_identical(foo, stacked)

        actual = concat([foo[0], foo[1]], pd.Index([0, 1])
                        ).reset_coords(drop=True)
        expected = foo[:2].rename({'x': 'concat_dim'})
        assert_identical(expected, actual)

        actual = concat([foo[0], foo[1]], [0, 1]).reset_coords(drop=True)
        expected = foo[:2].rename({'x': 'concat_dim'})
        assert_identical(expected, actual)

        with raises_regex(ValueError, 'not identical'):
            concat([foo, bar], dim='w', compat='identical')

        with raises_regex(ValueError, 'not a valid argument'):
            concat([foo, bar], dim='w', data_vars='minimal')

    def test_concat_encoding(self):
        # Regression test for GH1297
        ds = Dataset({'foo': (['x', 'y'], np.random.random((2, 3))),
                      'bar': (['x', 'y'], np.random.random((2, 3)))},
                     {'x': [0, 1]})
        foo = ds['foo']
        foo.encoding = {"complevel": 5}
        ds.encoding = {"unlimited_dims": 'x'}
        assert concat([foo, foo], dim="x").encoding == foo.encoding
        assert concat([ds, ds], dim="x").encoding == ds.encoding

    @requires_dask
    def test_concat_lazy(self):
        import dask.array as da

        arrays = [DataArray(
            da.from_array(InaccessibleArray(np.zeros((3, 3))), 3),
            dims=['x', 'y']) for _ in range(2)]
        # should not raise
        combined = concat(arrays, dim='z')
        assert combined.shape == (2, 3, 3)
        assert combined.dims == ('z', 'x', 'y')


class TestAutoCombine(TestCase):

    @requires_dask  # only for toolz
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
        objs = [Dataset(OrderedDict([('x', ('a', [0])), ('y', ('a', [0]))])),
                Dataset(OrderedDict([('y', ('a', [1])), ('x', ('a', [1]))]))]
        actual = auto_combine(objs)
        expected = Dataset({'x': ('a', [0, 1]), 'y': ('a', [0, 1])})
        assert_identical(expected, actual)

        objs = [Dataset({'x': [0], 'y': [0]}), Dataset({'y': [1], 'x': [1]})]
        with raises_regex(ValueError, 'too many .* dimensions'):
            auto_combine(objs)

        objs = [Dataset({'x': 0}), Dataset({'x': 1})]
        with raises_regex(ValueError, 'cannot infer dimension'):
            auto_combine(objs)

        objs = [Dataset({'x': [0], 'y': [0]}), Dataset({'x': [0]})]
        with pytest.raises(KeyError):
            auto_combine(objs)

    @requires_dask  # only for toolz
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

        # Your data includes "time" and "station" dimensions, and each year's
        # data has a different set of stations.
        datasets = [Dataset({'a': ('x', [2, 3]), 'x': [1, 2]}),
                    Dataset({'a': ('x', [1, 2]), 'x': [0, 1]})]
        expected = Dataset({'a': (('t', 'x'),
                                  [[np.nan, 2, 3], [1, 2, np.nan]])},
                           {'x': [0, 1, 2]})
        actual = auto_combine(datasets, concat_dim='t')
        assert_identical(expected, actual)

    @requires_dask  # only for toolz
    def test_auto_combine_still_fails(self):
        # concat can't handle new variables (yet):
        # https://github.com/pydata/xarray/issues/508
        datasets = [Dataset({'x': 0}, {'y': 0}),
                    Dataset({'x': 1}, {'y': 1, 'z': 1})]
        with pytest.raises(ValueError):
            auto_combine(datasets, 'y')

    @requires_dask  # only for toolz
    def test_auto_combine_no_concat(self):
        objs = [Dataset({'x': 0}), Dataset({'y': 1})]
        actual = auto_combine(objs)
        expected = Dataset({'x': 0, 'y': 1})
        assert_identical(expected, actual)

        objs = [Dataset({'x': 0, 'y': 1}), Dataset({'y': np.nan, 'z': 2})]
        actual = auto_combine(objs)
        expected = Dataset({'x': 0, 'y': 1, 'z': 2})
        assert_identical(expected, actual)

        data = Dataset({'x': 0})
        actual = auto_combine([data, data, data], concat_dim=None)
        assert_identical(data, actual)

        # Single object, with a concat_dim explicitly provided
        # Test the issue reported in GH #1988
        objs = [Dataset({'x': 0, 'y': 1})]
        dim = DataArray([100], name='baz', dims='baz')
        actual = auto_combine(objs, concat_dim=dim)
        expected = Dataset({'x': ('baz', [0]), 'y': ('baz', [1])},
                           {'baz': [100]})
        assert_identical(expected, actual)

        # Just making sure that auto_combine is doing what is
        # expected for non-scalar values, too.
        objs = [Dataset({'x': ('z', [0, 1]), 'y': ('z', [1, 2])})]
        dim = DataArray([100], name='baz', dims='baz')
        actual = auto_combine(objs, concat_dim=dim)
        expected = Dataset({'x': (('baz', 'z'), [[0, 1]]),
                            'y': (('baz', 'z'), [[1, 2]])},
                           {'baz': [100]})
        assert_identical(expected, actual)
