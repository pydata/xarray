import numpy as np
import pytest

import xarray as xr
from xarray.core import dtypes, merge

from . import raises_regex
from .test_dataset import create_test_data


class TestMergeInternals:
    def test_broadcast_dimension_size(self):
        actual = merge.broadcast_dimension_size(
            [xr.Variable("x", [1]), xr.Variable("y", [2, 1])]
        )
        assert actual == {"x": 1, "y": 2}

        actual = merge.broadcast_dimension_size(
            [xr.Variable(("x", "y"), [[1, 2]]), xr.Variable("y", [2, 1])]
        )
        assert actual == {"x": 1, "y": 2}

        with pytest.raises(ValueError):
            merge.broadcast_dimension_size(
                [xr.Variable(("x", "y"), [[1, 2]]), xr.Variable("y", [2])]
            )


class TestMergeFunction:
    def test_merge_arrays(self):
        data = create_test_data()
        actual = xr.merge([data.var1, data.var2])
        expected = data[["var1", "var2"]]
        assert actual.identical(expected)

    def test_merge_datasets(self):
        data = create_test_data()

        actual = xr.merge([data[["var1"]], data[["var2"]]])
        expected = data[["var1", "var2"]]
        assert actual.identical(expected)

        actual = xr.merge([data, data])
        assert actual.identical(data)

    def test_merge_dataarray_unnamed(self):
        data = xr.DataArray([1, 2], dims="x")
        with raises_regex(ValueError, "without providing an explicit name"):
            xr.merge([data])

    def test_merge_dicts_simple(self):
        actual = xr.merge([{"foo": 0}, {"bar": "one"}, {"baz": 3.5}])
        expected = xr.Dataset({"foo": 0, "bar": "one", "baz": 3.5})
        assert actual.identical(expected)

    def test_merge_dicts_dims(self):
        actual = xr.merge([{"y": ("x", [13])}, {"x": [12]}])
        expected = xr.Dataset({"x": [12], "y": ("x", [13])})
        assert actual.identical(expected)

    def test_merge_error(self):
        ds = xr.Dataset({"x": 0})
        with pytest.raises(xr.MergeError):
            xr.merge([ds, ds + 1])

    def test_merge_alignment_error(self):
        ds = xr.Dataset(coords={"x": [1, 2]})
        other = xr.Dataset(coords={"x": [2, 3]})
        with raises_regex(ValueError, "indexes .* not equal"):
            xr.merge([ds, other], join="exact")

    def test_merge_wrong_input_error(self):
        with raises_regex(TypeError, "objects must be an iterable"):
            xr.merge([1])
        ds = xr.Dataset(coords={"x": [1, 2]})
        with raises_regex(TypeError, "objects must be an iterable"):
            xr.merge({"a": ds})
        with raises_regex(TypeError, "objects must be an iterable"):
            xr.merge([ds, 1])

    def test_merge_no_conflicts_single_var(self):
        ds1 = xr.Dataset({"a": ("x", [1, 2]), "x": [0, 1]})
        ds2 = xr.Dataset({"a": ("x", [2, 3]), "x": [1, 2]})
        expected = xr.Dataset({"a": ("x", [1, 2, 3]), "x": [0, 1, 2]})
        assert expected.identical(xr.merge([ds1, ds2], compat="no_conflicts"))
        assert expected.identical(xr.merge([ds2, ds1], compat="no_conflicts"))
        assert ds1.identical(xr.merge([ds1, ds2], compat="no_conflicts", join="left"))
        assert ds2.identical(xr.merge([ds1, ds2], compat="no_conflicts", join="right"))
        expected = xr.Dataset({"a": ("x", [2]), "x": [1]})
        assert expected.identical(
            xr.merge([ds1, ds2], compat="no_conflicts", join="inner")
        )

        with pytest.raises(xr.MergeError):
            ds3 = xr.Dataset({"a": ("x", [99, 3]), "x": [1, 2]})
            xr.merge([ds1, ds3], compat="no_conflicts")

        with pytest.raises(xr.MergeError):
            ds3 = xr.Dataset({"a": ("y", [2, 3]), "y": [1, 2]})
            xr.merge([ds1, ds3], compat="no_conflicts")

    def test_merge_no_conflicts_multi_var(self):
        data = create_test_data()
        data1 = data.copy(deep=True)
        data2 = data.copy(deep=True)

        expected = data[["var1", "var2"]]
        actual = xr.merge([data1.var1, data2.var2], compat="no_conflicts")
        assert expected.identical(actual)

        data1["var1"][:, :5] = np.nan
        data2["var1"][:, 5:] = np.nan
        data1["var2"][:4, :] = np.nan
        data2["var2"][4:, :] = np.nan
        del data2["var3"]

        actual = xr.merge([data1, data2], compat="no_conflicts")
        assert data.equals(actual)

    def test_merge_no_conflicts_preserve_attrs(self):
        data = xr.Dataset({"x": ([], 0, {"foo": "bar"})})
        actual = xr.merge([data, data])
        assert data.identical(actual)

    def test_merge_no_conflicts_broadcast(self):
        datasets = [xr.Dataset({"x": ("y", [0])}), xr.Dataset({"x": np.nan})]
        actual = xr.merge(datasets)
        expected = xr.Dataset({"x": ("y", [0])})
        assert expected.identical(actual)

        datasets = [xr.Dataset({"x": ("y", [np.nan])}), xr.Dataset({"x": 0})]
        actual = xr.merge(datasets)
        assert expected.identical(actual)


class TestMergeMethod:
    def test_merge(self):
        data = create_test_data()
        ds1 = data[["var1"]]
        ds2 = data[["var3"]]
        expected = data[["var1", "var3"]]
        actual = ds1.merge(ds2)
        assert expected.identical(actual)

        actual = ds2.merge(ds1)
        assert expected.identical(actual)

        actual = data.merge(data)
        assert data.identical(actual)
        actual = data.reset_coords(drop=True).merge(data)
        assert data.identical(actual)
        actual = data.merge(data.reset_coords(drop=True))
        assert data.identical(actual)

        with pytest.raises(ValueError):
            ds1.merge(ds2.rename({"var3": "var1"}))
        with raises_regex(ValueError, "should be coordinates or not"):
            data.reset_coords().merge(data)
        with raises_regex(ValueError, "should be coordinates or not"):
            data.merge(data.reset_coords())

    def test_merge_broadcast_equals(self):
        ds1 = xr.Dataset({"x": 0})
        ds2 = xr.Dataset({"x": ("y", [0, 0])})
        actual = ds1.merge(ds2)
        assert ds2.identical(actual)

        actual = ds2.merge(ds1)
        assert ds2.identical(actual)

        actual = ds1.copy()
        actual.update(ds2)
        assert ds2.identical(actual)

        ds1 = xr.Dataset({"x": np.nan})
        ds2 = xr.Dataset({"x": ("y", [np.nan, np.nan])})
        actual = ds1.merge(ds2)
        assert ds2.identical(actual)

    def test_merge_compat(self):
        ds1 = xr.Dataset({"x": 0})
        ds2 = xr.Dataset({"x": 1})
        for compat in ["broadcast_equals", "equals", "identical", "no_conflicts"]:
            with pytest.raises(xr.MergeError):
                ds1.merge(ds2, compat=compat)

        ds2 = xr.Dataset({"x": [0, 0]})
        for compat in ["equals", "identical"]:
            with raises_regex(ValueError, "should be coordinates or not"):
                ds1.merge(ds2, compat=compat)

        ds2 = xr.Dataset({"x": ((), 0, {"foo": "bar"})})
        with pytest.raises(xr.MergeError):
            ds1.merge(ds2, compat="identical")

        with raises_regex(ValueError, "compat=.* invalid"):
            ds1.merge(ds2, compat="foobar")

        assert ds1.identical(ds1.merge(ds2, compat="override"))

    def test_merge_auto_align(self):
        ds1 = xr.Dataset({"a": ("x", [1, 2]), "x": [0, 1]})
        ds2 = xr.Dataset({"b": ("x", [3, 4]), "x": [1, 2]})
        expected = xr.Dataset(
            {"a": ("x", [1, 2, np.nan]), "b": ("x", [np.nan, 3, 4])}, {"x": [0, 1, 2]}
        )
        assert expected.identical(ds1.merge(ds2))
        assert expected.identical(ds2.merge(ds1))

        expected = expected.isel(x=slice(2))
        assert expected.identical(ds1.merge(ds2, join="left"))
        assert expected.identical(ds2.merge(ds1, join="right"))

        expected = expected.isel(x=slice(1, 2))
        assert expected.identical(ds1.merge(ds2, join="inner"))
        assert expected.identical(ds2.merge(ds1, join="inner"))

    @pytest.mark.parametrize("fill_value", [dtypes.NA, 2, 2.0])
    def test_merge_fill_value(self, fill_value):
        ds1 = xr.Dataset({"a": ("x", [1, 2]), "x": [0, 1]})
        ds2 = xr.Dataset({"b": ("x", [3, 4]), "x": [1, 2]})
        if fill_value == dtypes.NA:
            # if we supply the default, we expect the missing value for a
            # float array
            fill_value = np.nan
        expected = xr.Dataset(
            {"a": ("x", [1, 2, fill_value]), "b": ("x", [fill_value, 3, 4])},
            {"x": [0, 1, 2]},
        )
        assert expected.identical(ds1.merge(ds2, fill_value=fill_value))
        assert expected.identical(ds2.merge(ds1, fill_value=fill_value))
        assert expected.identical(xr.merge([ds1, ds2], fill_value=fill_value))

    def test_merge_no_conflicts(self):
        ds1 = xr.Dataset({"a": ("x", [1, 2]), "x": [0, 1]})
        ds2 = xr.Dataset({"a": ("x", [2, 3]), "x": [1, 2]})
        expected = xr.Dataset({"a": ("x", [1, 2, 3]), "x": [0, 1, 2]})

        assert expected.identical(ds1.merge(ds2, compat="no_conflicts"))
        assert expected.identical(ds2.merge(ds1, compat="no_conflicts"))

        assert ds1.identical(ds1.merge(ds2, compat="no_conflicts", join="left"))

        assert ds2.identical(ds1.merge(ds2, compat="no_conflicts", join="right"))

        expected2 = xr.Dataset({"a": ("x", [2]), "x": [1]})
        assert expected2.identical(ds1.merge(ds2, compat="no_conflicts", join="inner"))

        with pytest.raises(xr.MergeError):
            ds3 = xr.Dataset({"a": ("x", [99, 3]), "x": [1, 2]})
            ds1.merge(ds3, compat="no_conflicts")

        with pytest.raises(xr.MergeError):
            ds3 = xr.Dataset({"a": ("y", [2, 3]), "y": [1, 2]})
            ds1.merge(ds3, compat="no_conflicts")
