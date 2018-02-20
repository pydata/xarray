from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import pickle

import numpy as np

import xarray.ufuncs as xu
import xarray as xr

from . import (
    TestCase, raises_regex, assert_identical, assert_array_equal)


class TestOps(TestCase):
    def assert_identical(self, a, b):
        assert type(a) is type(b) or (float(a) == float(b))  # noqa
        try:
            assert a.identical(b), (a, b)
        except AttributeError:
            assert_array_equal(a, b)

    def test_unary(self):
        args = [0,
                np.zeros(2),
                xr.Variable(['x'], [0, 0]),
                xr.DataArray([0, 0], dims='x'),
                xr.Dataset({'y': ('x', [0, 0])})]
        for a in args:
            self.assert_identical(a + 1, xu.cos(a))

    def test_binary(self):
        args = [0,
                np.zeros(2),
                xr.Variable(['x'], [0, 0]),
                xr.DataArray([0, 0], dims='x'),
                xr.Dataset({'y': ('x', [0, 0])})]
        for n, t1 in enumerate(args):
            for t2 in args[n:]:
                self.assert_identical(t2 + 1, xu.maximum(t1, t2 + 1))
                self.assert_identical(t2 + 1, xu.maximum(t2, t1 + 1))
                self.assert_identical(t2 + 1, xu.maximum(t1 + 1, t2))
                self.assert_identical(t2 + 1, xu.maximum(t2 + 1, t1))

    def test_groupby(self):
        ds = xr.Dataset({'a': ('x', [0, 0, 0])}, {'c': ('x', [0, 0, 1])})
        ds_grouped = ds.groupby('c')
        group_mean = ds_grouped.mean('x')
        arr_grouped = ds['a'].groupby('c')

        assert_identical(ds, xu.maximum(ds_grouped, group_mean))
        assert_identical(ds, xu.maximum(group_mean, ds_grouped))

        assert_identical(ds, xu.maximum(arr_grouped, group_mean))
        assert_identical(ds, xu.maximum(group_mean, arr_grouped))

        assert_identical(ds, xu.maximum(ds_grouped, group_mean['a']))
        assert_identical(ds, xu.maximum(group_mean['a'], ds_grouped))

        assert_identical(ds.a, xu.maximum(arr_grouped, group_mean.a))
        assert_identical(ds.a, xu.maximum(group_mean.a, arr_grouped))

        with raises_regex(TypeError, 'only support binary ops'):
            xu.maximum(ds.a.variable, ds_grouped)

    def test_pickle(self):
        a = 1.0
        cos_pickled = pickle.loads(pickle.dumps(xu.cos))
        self.assert_identical(cos_pickled(a), xu.cos(a))
