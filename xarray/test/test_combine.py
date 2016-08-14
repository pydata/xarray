from copy import deepcopy

import numpy as np
import pandas as pd

from xarray import Dataset, DataArray, auto_combine, concat, Variable
from xarray.core.pycompat import iteritems, OrderedDict

from . import TestCase, InaccessibleArray, requires_dask
from .test_dataset import create_test_data


class TestConcatDataset(TestCase):
    def test_concat(self):
        # TODO: simplify and split this test case

        # drop the third dimension to keep things relatively understandable
        data = create_test_data().drop('dim3')

        split_data = [data.isel(dim1=slice(3)),
                      data.isel(dim1=slice(3, None))]
        self.assertDatasetIdentical(data, concat(split_data, 'dim1'))

        def rectify_dim_order(dataset):
            # return a new dataset with all variable dimensions transposed into
            # the order in which they are found in `data`
            return Dataset(dict((k, v.transpose(*data[k].dims))
                                for k, v in iteritems(dataset.data_vars)),
                           dataset.coords, attrs=dataset.attrs)

        for dim in ['dim1', 'dim2']:
            datasets = [g for _, g in data.groupby(dim, squeeze=False)]
            self.assertDatasetIdentical(data, concat(datasets, dim))
            self.assertDatasetIdentical(
                data, concat(datasets, data[dim]))
            self.assertDatasetIdentical(
                data, concat(datasets, data[dim], coords='minimal'))

            datasets = [g for _, g in data.groupby(dim, squeeze=True)]
            concat_over = [k for k, v in iteritems(data.coords)
                           if dim in v.dims and k != dim]
            actual = concat(datasets, data[dim], coords=concat_over)
            self.assertDatasetIdentical(data, rectify_dim_order(actual))

            actual = concat(datasets, data[dim], coords='different')
            self.assertDatasetIdentical(data, rectify_dim_order(actual))

        # make sure the coords argument behaves as expected
        data.coords['extra'] = ('dim4', np.arange(3))
        for dim in ['dim1', 'dim2']:
            datasets = [g for _, g in data.groupby(dim, squeeze=True)]
            actual = concat(datasets, data[dim], coords='all')
            expected = np.array([data['extra'].values
                                 for _ in range(data.dims[dim])])
            self.assertArrayEqual(actual['extra'].values, expected)

            actual = concat(datasets, data[dim], coords='different')
            self.assertDataArrayEqual(data['extra'], actual['extra'])
            actual = concat(datasets, data[dim], coords='minimal')
            self.assertDataArrayEqual(data['extra'], actual['extra'])

        # verify that the dim argument takes precedence over
        # concatenating dataset variables of the same name
        dim = (2 * data['dim1']).rename('dim1')
        datasets = [g for _, g in data.groupby('dim1', squeeze=False)]
        expected = data.copy()
        expected['dim1'] = dim
        self.assertDatasetIdentical(expected, concat(datasets, dim))

    def test_concat_data_vars(self):
        data = Dataset({'foo': ('x', np.random.randn(10))})
        objs = [data.isel(x=slice(5)), data.isel(x=slice(5, None))]
        for data_vars in ['minimal', 'different', 'all', [], ['foo']]:
            actual = concat(objs, dim='x', data_vars=data_vars)
            self.assertDatasetIdentical(data, actual)

    def test_concat_coords(self):
        data = Dataset({'foo': ('x', np.random.randn(10))})
        expected = data.assign_coords(c=('x', [0] * 5 + [1] * 5))
        objs = [data.isel(x=slice(5)).assign_coords(c=0),
                data.isel(x=slice(5, None)).assign_coords(c=1)]
        for coords in ['different', 'all', ['c']]:
            actual = concat(objs, dim='x', coords=coords)
            self.assertDatasetIdentical(expected, actual)
        for coords in ['minimal', []]:
            with self.assertRaisesRegexp(ValueError, 'not equal across'):
                concat(objs, dim='x', coords=coords)

    def test_concat_constant_index(self):
        # GH425
        ds1 = Dataset({'foo': 1.5}, {'y': 1})
        ds2 = Dataset({'foo': 2.5}, {'y': 1})
        expected = Dataset({'foo': ('y', [1.5, 2.5]), 'y': [1, 1]})
        for mode in ['different', 'all', ['foo']]:
            actual = concat([ds1, ds2], 'y', data_vars=mode)
            self.assertDatasetIdentical(expected, actual)
        with self.assertRaisesRegexp(ValueError, 'not equal across datasets'):
            concat([ds1, ds2], 'y', data_vars='minimal')

    def test_concat_size0(self):
        data = create_test_data()
        split_data = [data.isel(dim1=slice(0, 0)), data]
        actual = concat(split_data, 'dim1')
        self.assertDatasetIdentical(data, actual)

        actual = concat(split_data[::-1], 'dim1')
        self.assertDatasetIdentical(data, actual)

    def test_concat_autoalign(self):
        ds1 = Dataset({'foo': DataArray([1, 2], coords={'x': [1, 2]})})
        ds2 = Dataset({'foo': DataArray([1, 2], coords={'x': [1, 3]})})        
        actual = concat([ds1, ds2], 'y')
        expected = Dataset({'foo': DataArray([[1, 2, np.nan], [1, np.nan, 2]],
                                             dims=['y', 'x'], coords={'y': [0, 1], 'x': [1, 2, 3]})})
        self.assertDatasetIdentical(expected, actual)

    def test_concat_errors(self):
        data = create_test_data()
        split_data = [data.isel(dim1=slice(3)),
                      data.isel(dim1=slice(3, None))]

        with self.assertRaisesRegexp(ValueError, 'must supply at least one'):
            concat([], 'dim1')

        with self.assertRaisesRegexp(ValueError, 'are not coordinates'):
            concat([data, data], 'new_dim', coords=['not_found'])

        with self.assertRaisesRegexp(ValueError, 'global attributes not'):
            data0, data1 = deepcopy(split_data)
            data1.attrs['foo'] = 'bar'
            concat([data0, data1], 'dim1', compat='identical')
        self.assertDatasetIdentical(
            data, concat([data0, data1], 'dim1', compat='equals'))

        with self.assertRaisesRegexp(ValueError, 'encountered unexpected'):
            data0, data1 = deepcopy(split_data)
            data1['foo'] = ('bar', np.random.randn(10))
            concat([data0, data1], 'dim1')

        with self.assertRaisesRegexp(ValueError, 'compat.* invalid'):
            concat(split_data, 'dim1', compat='foobar')

        with self.assertRaisesRegexp(ValueError, 'unexpected value for'):
            concat([data, data], 'new_dim', coords='foobar')

        with self.assertRaisesRegexp(
                ValueError, 'coordinate in some datasets but not others'):
            concat([Dataset({'x': 0}), Dataset({'x': [1]})], dim='z')

        with self.assertRaisesRegexp(
                ValueError, 'coordinate in some datasets but not others'):
            concat([Dataset({'x': 0}), Dataset({}, {'x': 1})], dim='z')

        with self.assertRaisesRegexp(ValueError, 'no longer a valid'):
            concat([data, data], 'new_dim', mode='different')
        with self.assertRaisesRegexp(ValueError, 'no longer a valid'):
            concat([data, data], 'new_dim', concat_over='different')

    def test_concat_promote_shape(self):
        # mixed dims within variables
        objs = [Dataset({}, {'x': 0}), Dataset({'x': [1]})]
        actual = concat(objs, 'x')
        expected = Dataset({'x': [0, 1]})
        self.assertDatasetIdentical(actual, expected)

        objs = [Dataset({'x': [0]}), Dataset({}, {'x': 1})]
        actual = concat(objs, 'x')
        self.assertDatasetIdentical(actual, expected)

        # mixed dims between variables
        objs = [Dataset({'x': [2], 'y': 3}), Dataset({'x': [4], 'y': 5})]
        actual = concat(objs, 'x')
        expected = Dataset({'x': [2, 4], 'y': ('x', [3, 5])})
        self.assertDatasetIdentical(actual, expected)

        # mixed dims in coord variable
        objs = [Dataset({'x': [0]}, {'y': -1}),
                Dataset({'x': [1]}, {'y': ('x', [-2])})]
        actual = concat(objs, 'x')
        expected = Dataset({'x': [0, 1]}, {'y': ('x', [-1, -2])})
        self.assertDatasetIdentical(actual, expected)

        # scalars with mixed lengths along concat dim -- values should repeat
        objs = [Dataset({'x': [0]}, {'y': -1}),
                Dataset({'x': [1, 2]}, {'y': -2})]
        actual = concat(objs, 'x')
        expected = Dataset({}, {'y': ('x', [-1, -2, -2])})
        self.assertDatasetIdentical(actual, expected)

        # broadcast 1d x 1d -> 2d
        objs = [Dataset({'z': ('x', [-1])}, {'x': [0], 'y': [0]}),
                Dataset({'z': ('y', [1])}, {'x': [1], 'y': [0]})]
        actual = concat(objs, 'x')
        expected = Dataset({'z': (('x', 'y'), [[-1], [1]])})
        self.assertDatasetIdentical(actual, expected)

    def test_concat_do_not_promote(self):
        # GH438
        objs = [Dataset({'y': ('t', [1])}, {'x': 1}),
                Dataset({'y': ('t', [2])}, {'x': 1})]
        expected = Dataset({'y': ('t', [1, 2])}, {'x': 1, 't': [0, 0]})
        actual = concat(objs, 't')
        self.assertDatasetIdentical(expected, actual)

        objs = [Dataset({'y': ('t', [1])}, {'x': 1}),
                Dataset({'y': ('t', [2])}, {'x': 2})]
        with self.assertRaises(ValueError):
            concat(objs, 't', coords='minimal')

    def test_concat_dim_is_variable(self):
        objs = [Dataset({'x': 0}), Dataset({'x': 1})]
        coord = Variable('y', [3, 4])
        expected = Dataset({'x': ('y', [0, 1]), 'y': [3, 4]})
        actual = concat(objs, coord)
        self.assertDatasetIdentical(actual, expected)

    def test_concat_multiindex(self):
        x = pd.MultiIndex.from_product([[1, 2, 3], ['a', 'b']])
        expected = Dataset({'x': x})
        actual = concat([expected.isel(x=slice(2)),
                         expected.isel(x=slice(2, None))], 'x')
        assert expected.equals(actual)
        assert isinstance(actual.x.to_index(), pd.MultiIndex)

    @requires_dask  # only for toolz
    def test_auto_combine(self):
        objs = [Dataset({'x': [0]}), Dataset({'x': [1]})]
        actual = auto_combine(objs)
        expected = Dataset({'x': [0, 1]})
        self.assertDatasetIdentical(expected, actual)

        actual = auto_combine([actual])
        self.assertDatasetIdentical(expected, actual)

        objs = [Dataset({'x': [0, 1]}), Dataset({'x': [2]})]
        actual = auto_combine(objs)
        expected = Dataset({'x': [0, 1, 2]})
        self.assertDatasetIdentical(expected, actual)

        # ensure auto_combine handles non-sorted dimensions
        objs = [Dataset(OrderedDict([('x', ('a', [0])), ('y', ('a', [0]))])),
                Dataset(OrderedDict([('y', ('a', [1])), ('x', ('a', [1]))]))]
        actual = auto_combine(objs)
        expected = Dataset({'x': ('a', [0, 1]), 'y': ('a', [0, 1]), 'a': [0, 0]})
        self.assertDatasetIdentical(expected, actual)

        objs = [Dataset({'x': [0], 'y': [0]}), Dataset({'y': [1], 'x': [1]})]
        with self.assertRaisesRegexp(ValueError, 'too many .* dimensions'):
            auto_combine(objs)

        objs = [Dataset({'x': 0}), Dataset({'x': 1})]
        with self.assertRaisesRegexp(ValueError, 'cannot infer dimension'):
            auto_combine(objs)

        objs = [Dataset({'x': [0], 'y': [0]}), Dataset({'x': [0]})]
        with self.assertRaises(KeyError):
            auto_combine(objs)


class TestConcatDataArray(TestCase):
    def test_concat(self):
        ds = Dataset({'foo': (['x', 'y'], np.random.random((10, 20))),
                      'bar': (['x', 'y'], np.random.random((10, 20)))})
        foo = ds['foo']
        bar = ds['bar']

        # from dataset array:
        expected = DataArray(np.array([foo.values, bar.values]),
                             dims=['w', 'x', 'y'])
        actual = concat([foo, bar], 'w')
        self.assertDataArrayEqual(expected, actual)
        # from iteration:
        grouped = [g for _, g in foo.groupby('x')]
        stacked = concat(grouped, ds['x'])
        self.assertDataArrayIdentical(foo, stacked)
        # with an index as the 'dim' argument
        stacked = concat(grouped, ds.indexes['x'])
        self.assertDataArrayIdentical(foo, stacked)

        actual = concat([foo[0], foo[1]], pd.Index([0, 1])).reset_coords(drop=True)
        expected = foo[:2].rename({'x': 'concat_dim'})
        self.assertDataArrayIdentical(expected, actual)

        actual = concat([foo[0], foo[1]], [0, 1]).reset_coords(drop=True)
        expected = foo[:2].rename({'x': 'concat_dim'})
        self.assertDataArrayIdentical(expected, actual)

        with self.assertRaisesRegexp(ValueError, 'not identical'):
            concat([foo, bar], dim='w', compat='identical')

        with self.assertRaisesRegexp(ValueError, 'not a valid argument'):
            concat([foo, bar], dim='w', data_vars='minimal')

    @requires_dask
    def test_concat_lazy(self):
        import dask.array as da

        arrays = [DataArray(
            da.from_array(InaccessibleArray(np.zeros((3, 3))), 3),
            dims=['x', 'y']) for _ in range(2)]
        # should not raise
        combined = concat(arrays, dim='z')
        self.assertEqual(combined.shape, (2, 3, 3))
        self.assertEqual(combined.dims, ('z', 'x', 'y'))
