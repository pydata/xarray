import numpy as np

import xarray as xr
from xarray.testing import assert_identical, assert_allclose


class TestMap:
    def test_2_dim(self):
        da = xr.DataArray(np.random.randn(2, 3))
        ds1 = xr.Dataset({"foo": da, "bar": ("x", [-1, 2])})
        ds2 = xr.Dataset({"foo": da + 1, "bar": ("x", [0, 3])})
        print(ds1)
        print(ds2)

        f = lambda a, b: b - a
        r = xr.map([ds1, ds2], f)
        assert_allclose(r, xr.ones_like(ds1))

    def test_different_variables_with_overlap(self):
        ds1 = xr.Dataset({"foo": ("x", [1, 2]), "bar": ("x", [-1, 2]),
                          "oof" : ("x", [3, 4])})
        ds2 = xr.Dataset({"foo": ("x", [11, 22]), "oof": ("x", [-1, 3])})
        ds3 = xr.Dataset({"bar": ("x", [11, 22]), "oof": ("x", [-1, 2])})

        ds_out = xr.Dataset({"oof": ("x", [3, 5])})
        f = lambda x, y, z: x + y - z
        r = xr.map([ds1, ds2, ds3], f)
        assert_identical(r, ds_out)

    def test_no_variable_overlap(self):
        ds1 = xr.Dataset({"foo": ("x", [1, 2]), "oof": ("x", [3, 4])})
        ds2 = xr.Dataset({"bar": ("x", [11, 22]), "rab": ("x", [-1, 3])})

        ds_out = xr.Dataset()
        f = lambda x, y: x + y
        r = xr.map([ds1, ds2], f)
        assert_identical(r, ds_out)

    def test_with_args_and_kwargs(self):
        ds1 = xr.Dataset({"foo": ("x", [1, 1]), "oof": ("x", [3, 3])})
        ds2 = xr.Dataset({"foo": ("x", [2, 2]), "oof": ("x", [4, 4])})

        ds_out = xr.Dataset({"foo": ("x", [8, 8]), "oof": ("x", [18, 18])})

        def f(da1, da2, multiplier_1, multiplier_2):
            return multiplier_1 * da1 + multiplier_2 * da2

        r = xr.map([ds1, ds2], f, args=[2], kwargs={'multiplier_2' : 3})
        assert_identical(r, ds_out)

    def test_keep_attrs(self):
        ds1 = xr.Dataset({"foo": ("x", [1, 1]), "oof": ("x", [3, 3])}, attrs={'value': 1})
        ds2 = xr.Dataset({"foo": ("x", [2, 2]), "oof": ("x", [4, 4])}, attrs={'value': 2})
        ds3 = xr.Dataset({"foo": ("x", [3, 3]), "oof": ("x", [5, 5])}, attrs={'value': 3})

        def f(da1, da2, da3, selector):
            return [da1, da2, da3][selector]

        r = xr.map([ds1, ds2, ds3], f, args=[2], keep_attrs=0)
        assert r.attrs['value'] == 1

        r = xr.map([ds1, ds2, ds3], f, args=[0], keep_attrs=1)
        assert r.attrs['value'] == 2
