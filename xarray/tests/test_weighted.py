import numpy as np
import pytest

import xarray as xr
from xarray import DataArray
from xarray.tests import assert_allclose, assert_equal, raises_regex

from . import raise_if_dask_computes, requires_cftime, requires_dask


@pytest.mark.parametrize("as_dataset", (True, False))
def test_weighted_non_DataArray_weights(as_dataset):

    data = DataArray([1, 2])
    if as_dataset:
        data = data.to_dataset(name="data")

    with raises_regex(ValueError, "`weights` must be a DataArray"):
        data.weighted([1, 2])


@pytest.mark.parametrize("as_dataset", (True, False))
@pytest.mark.parametrize("weights", ([np.nan, 2], [np.nan, np.nan]))
def test_weighted_weights_nan_raises(as_dataset, weights):

    data = DataArray([1, 2])
    if as_dataset:
        data = data.to_dataset(name="data")

    with pytest.raises(ValueError, match="`weights` cannot contain missing values."):
        data.weighted(DataArray(weights))


@requires_dask
@pytest.mark.parametrize("as_dataset", (True, False))
@pytest.mark.parametrize("weights", ([np.nan, 2], [np.nan, np.nan]))
def test_weighted_weights_nan_raises_dask(as_dataset, weights):

    data = DataArray([1, 2]).chunk({"dim_0": -1})
    if as_dataset:
        data = data.to_dataset(name="data")

    weights = DataArray(weights).chunk({"dim_0": -1})

    with raise_if_dask_computes():
        weighted = data.weighted(weights)

    with pytest.raises(ValueError, match="`weights` cannot contain missing values."):
        weighted.sum().load()


@requires_cftime
@requires_dask
@pytest.mark.parametrize("time_chunks", (1, 5))
@pytest.mark.parametrize("resample_spec", ("1AS", "5AS", "10AS"))
def test_weighted_lazy_resample(time_chunks, resample_spec):
    # https://github.com/pydata/xarray/issues/4625

    # simple customized weighted mean function
    def mean_func(ds):
        return ds.weighted(ds.weights).mean("time")

    # example dataset
    t = xr.cftime_range(start="2000", periods=20, freq="1AS")
    weights = xr.DataArray(np.random.rand(len(t)), dims=["time"], coords={"time": t})
    data = xr.DataArray(
        np.random.rand(len(t)), dims=["time"], coords={"time": t, "weights": weights}
    )
    ds = xr.Dataset({"data": data}).chunk({"time": time_chunks})

    with raise_if_dask_computes():
        ds.resample(time=resample_spec).map(mean_func)


@pytest.mark.parametrize(
    ("weights", "expected"),
    (([1, 2], 3), ([2, 0], 2), ([0, 0], np.nan), ([-1, 1], np.nan)),
)
def test_weighted_sum_of_weights_no_nan(weights, expected):

    da = DataArray([1, 2])
    weights = DataArray(weights)
    result = da.weighted(weights).sum_of_weights()

    expected = DataArray(expected)

    assert_equal(expected, result)


@pytest.mark.parametrize(
    ("weights", "expected"),
    (([1, 2], 2), ([2, 0], np.nan), ([0, 0], np.nan), ([-1, 1], 1)),
)
def test_weighted_sum_of_weights_nan(weights, expected):

    da = DataArray([np.nan, 2])
    weights = DataArray(weights)
    result = da.weighted(weights).sum_of_weights()

    expected = DataArray(expected)

    assert_equal(expected, result)


def test_weighted_sum_of_weights_bool():
    # https://github.com/pydata/xarray/issues/4074

    da = DataArray([1, 2])
    weights = DataArray([True, True])
    result = da.weighted(weights).sum_of_weights()

    expected = DataArray(2)

    assert_equal(expected, result)


@pytest.mark.parametrize("da", ([1.0, 2], [1, np.nan], [np.nan, np.nan]))
@pytest.mark.parametrize("factor", [0, 1, 3.14])
@pytest.mark.parametrize("skipna", (True, False))
def test_weighted_sum_equal_weights(da, factor, skipna):
    # if all weights are 'f'; weighted sum is f times the ordinary sum

    da = DataArray(da)
    weights = xr.full_like(da, factor)

    expected = da.sum(skipna=skipna) * factor
    result = da.weighted(weights).sum(skipna=skipna)

    assert_equal(expected, result)


@pytest.mark.parametrize(
    ("weights", "expected"), (([1, 2], 5), ([0, 2], 4), ([0, 0], 0))
)
def test_weighted_sum_no_nan(weights, expected):

    da = DataArray([1, 2])

    weights = DataArray(weights)
    result = da.weighted(weights).sum()
    expected = DataArray(expected)

    assert_equal(expected, result)


@pytest.mark.parametrize(
    ("weights", "expected"), (([1, 2], 4), ([0, 2], 4), ([1, 0], 0), ([0, 0], 0))
)
@pytest.mark.parametrize("skipna", (True, False))
def test_weighted_sum_nan(weights, expected, skipna):

    da = DataArray([np.nan, 2])

    weights = DataArray(weights)
    result = da.weighted(weights).sum(skipna=skipna)

    if skipna:
        expected = DataArray(expected)
    else:
        expected = DataArray(np.nan)

    assert_equal(expected, result)


@pytest.mark.filterwarnings("error")
@pytest.mark.parametrize("da", ([1.0, 2], [1, np.nan], [np.nan, np.nan]))
@pytest.mark.parametrize("skipna", (True, False))
@pytest.mark.parametrize("factor", [1, 2, 3.14])
def test_weighted_mean_equal_weights(da, skipna, factor):
    # if all weights are equal (!= 0), should yield the same result as mean

    da = DataArray(da)

    # all weights as 1.
    weights = xr.full_like(da, factor)

    expected = da.mean(skipna=skipna)
    result = da.weighted(weights).mean(skipna=skipna)

    assert_equal(expected, result)


@pytest.mark.parametrize(
    ("weights", "expected"), (([4, 6], 1.6), ([1, 0], 1.0), ([0, 0], np.nan))
)
def test_weighted_mean_no_nan(weights, expected):

    da = DataArray([1, 2])
    weights = DataArray(weights)
    expected = DataArray(expected)

    result = da.weighted(weights).mean()

    assert_equal(expected, result)


@pytest.mark.parametrize(
    ("weights", "expected"), (([4, 6], 2.0), ([1, 0], np.nan), ([0, 0], np.nan))
)
@pytest.mark.parametrize("skipna", (True, False))
def test_weighted_mean_nan(weights, expected, skipna):

    da = DataArray([np.nan, 2])
    weights = DataArray(weights)

    if skipna:
        expected = DataArray(expected)
    else:
        expected = DataArray(np.nan)

    result = da.weighted(weights).mean(skipna=skipna)

    assert_equal(expected, result)


def test_weighted_mean_bool():
    # https://github.com/pydata/xarray/issues/4074
    da = DataArray([1, 1])
    weights = DataArray([True, True])
    expected = DataArray(1)

    result = da.weighted(weights).mean()

    assert_equal(expected, result)


def expected_weighted(da, weights, dim, skipna, operation):
    """
    Generate expected result using ``*`` and ``sum``. This is checked against
    the result of da.weighted which uses ``dot``
    """

    weighted_sum = (da * weights).sum(dim=dim, skipna=skipna)

    if operation == "sum":
        return weighted_sum

    masked_weights = weights.where(da.notnull())
    sum_of_weights = masked_weights.sum(dim=dim, skipna=True)
    valid_weights = sum_of_weights != 0
    sum_of_weights = sum_of_weights.where(valid_weights)

    if operation == "sum_of_weights":
        return sum_of_weights

    weighted_mean = weighted_sum / sum_of_weights

    if operation == "mean":
        return weighted_mean


def check_weighted_operations(data, weights, dim, skipna):

    # check sum of weights
    result = data.weighted(weights).sum_of_weights(dim)
    expected = expected_weighted(data, weights, dim, skipna, "sum_of_weights")
    assert_allclose(expected, result)

    # check weighted sum
    result = data.weighted(weights).sum(dim, skipna=skipna)
    expected = expected_weighted(data, weights, dim, skipna, "sum")
    assert_allclose(expected, result)

    # check weighted mean
    result = data.weighted(weights).mean(dim, skipna=skipna)
    expected = expected_weighted(data, weights, dim, skipna, "mean")
    assert_allclose(expected, result)


@pytest.mark.parametrize("dim", ("a", "b", "c", ("a", "b"), ("a", "b", "c"), None))
@pytest.mark.parametrize("add_nans", (True, False))
@pytest.mark.parametrize("skipna", (None, True, False))
def test_weighted_operations_3D(dim, add_nans, skipna):

    dims = ("a", "b", "c")
    coords = dict(a=[0, 1, 2, 3], b=[0, 1, 2, 3], c=[0, 1, 2, 3])

    weights = DataArray(np.random.randn(4, 4, 4), dims=dims, coords=coords)

    data = np.random.randn(4, 4, 4)

    # add approximately 25 % NaNs (https://stackoverflow.com/a/32182680/3010700)
    if add_nans:
        c = int(data.size * 0.25)
        data.ravel()[np.random.choice(data.size, c, replace=False)] = np.NaN

    data = DataArray(data, dims=dims, coords=coords)

    check_weighted_operations(data, weights, dim, skipna)

    data = data.to_dataset(name="data")
    check_weighted_operations(data, weights, dim, skipna)


def test_weighted_operations_nonequal_coords():

    weights = DataArray(np.random.randn(4), dims=("a",), coords=dict(a=[0, 1, 2, 3]))
    data = DataArray(np.random.randn(4), dims=("a",), coords=dict(a=[1, 2, 3, 4]))

    check_weighted_operations(data, weights, dim="a", skipna=None)

    data = data.to_dataset(name="data")
    check_weighted_operations(data, weights, dim="a", skipna=None)


@pytest.mark.parametrize("shape_data", ((4,), (4, 4), (4, 4, 4)))
@pytest.mark.parametrize("shape_weights", ((4,), (4, 4), (4, 4, 4)))
@pytest.mark.parametrize("add_nans", (True, False))
@pytest.mark.parametrize("skipna", (None, True, False))
def test_weighted_operations_different_shapes(
    shape_data, shape_weights, add_nans, skipna
):

    weights = DataArray(np.random.randn(*shape_weights))

    data = np.random.randn(*shape_data)

    # add approximately 25 % NaNs
    if add_nans:
        c = int(data.size * 0.25)
        data.ravel()[np.random.choice(data.size, c, replace=False)] = np.NaN

    data = DataArray(data)

    check_weighted_operations(data, weights, "dim_0", skipna)
    check_weighted_operations(data, weights, None, skipna)

    data = data.to_dataset(name="data")
    check_weighted_operations(data, weights, "dim_0", skipna)
    check_weighted_operations(data, weights, None, skipna)


@pytest.mark.parametrize("operation", ("sum_of_weights", "sum", "mean"))
@pytest.mark.parametrize("as_dataset", (True, False))
@pytest.mark.parametrize("keep_attrs", (True, False, None))
def test_weighted_operations_keep_attr(operation, as_dataset, keep_attrs):

    weights = DataArray(np.random.randn(2, 2), attrs=dict(attr="weights"))
    data = DataArray(np.random.randn(2, 2))

    if as_dataset:
        data = data.to_dataset(name="data")

    data.attrs = dict(attr="weights")

    result = getattr(data.weighted(weights), operation)(keep_attrs=True)

    if operation == "sum_of_weights":
        assert weights.attrs == result.attrs
    else:
        assert data.attrs == result.attrs

    result = getattr(data.weighted(weights), operation)(keep_attrs=None)
    assert not result.attrs

    result = getattr(data.weighted(weights), operation)(keep_attrs=False)
    assert not result.attrs


@pytest.mark.parametrize("operation", ("sum", "mean"))
def test_weighted_operations_keep_attr_da_in_ds(operation):
    # GH #3595

    weights = DataArray(np.random.randn(2, 2))
    data = DataArray(np.random.randn(2, 2), attrs=dict(attr="data"))
    data = data.to_dataset(name="a")

    result = getattr(data.weighted(weights), operation)(keep_attrs=True)

    assert data.a.attrs == result.a.attrs
