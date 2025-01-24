from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import pytest

import xarray as xr
from xarray.core.missing import (
    NumpyInterpolator,
    ScipyInterpolator,
    SplineInterpolator,
    _get_gap_dist_to_left_edge,
    _get_gap_dist_to_right_edge,
    _get_gap_left_edge,
    _get_gap_right_edge,
    _get_limit_area_mask,
    _get_limit_fill_mask,
    _get_nan_block_lengths,
    get_clean_interp_index,
)
from xarray.namedarray.pycompat import array_type
from xarray.tests import (
    _CFTIME_CALENDARS,
    assert_allclose,
    assert_array_equal,
    assert_equal,
    raise_if_dask_computes,
    requires_bottleneck,
    requires_cftime,
    requires_dask,
    requires_numbagg,
    requires_numbagg_or_bottleneck,
    requires_scipy,
)

dask_array_type = array_type("dask")


@pytest.fixture
def da():
    return xr.DataArray([0, np.nan, 1, 2, np.nan, 3, 4, 5, np.nan, 6, 7], dims="time")


@pytest.fixture
def cf_da():
    def _cf_da(calendar, freq="1D"):
        times = xr.cftime_range(
            start="1970-01-01", freq=freq, periods=10, calendar=calendar
        )
        values = np.arange(10)
        return xr.DataArray(values, dims=("time",), coords={"time": times})

    return _cf_da


@pytest.fixture
def ds():
    ds = xr.Dataset()
    ds["var1"] = xr.DataArray(
        [0, np.nan, 1, 2, np.nan, 3, 4, 5, np.nan, 6, 7], dims="time"
    )
    ds["var2"] = xr.DataArray(
        [10, np.nan, 11, 12, np.nan, 13, 14, 15, np.nan, 16, 17], dims="x"
    )
    return ds


def make_interpolate_example_data(shape, frac_nan, seed=12345, non_uniform=False):
    rs = np.random.default_rng(seed)
    vals = rs.normal(size=shape)
    if frac_nan == 1:
        vals[:] = np.nan
    elif frac_nan == 0:
        pass
    else:
        n_missing = int(vals.size * frac_nan)

        ys = np.arange(shape[0])
        xs = np.arange(shape[1])
        if n_missing:
            np.random.shuffle(ys)
            ys = ys[:n_missing]

            np.random.shuffle(xs)
            xs = xs[:n_missing]

            vals[ys, xs] = np.nan

    if non_uniform:
        # construct a datetime index that has irregular spacing
        deltas = pd.to_timedelta(rs.normal(size=shape[0], scale=10), unit="D")
        coords = {"time": (pd.Timestamp("2000-01-01") + deltas).sort_values()}
    else:
        coords = {"time": pd.date_range("2000-01-01", freq="D", periods=shape[0])}
    da = xr.DataArray(vals, dims=("time", "x"), coords=coords)
    df = da.to_pandas()

    return da, df


@pytest.mark.parametrize("fill_value", [None, np.nan, 47.11])
@pytest.mark.parametrize(
    "method", ["linear", "nearest", "zero", "slinear", "quadratic", "cubic"]
)
@pytest.mark.parametrize("dim", ["time", "x"])
@pytest.mark.parametrize("shape", [(8, 8), (1, 20), (20, 1), (100, 100)])
@pytest.mark.parametrize("frac_nan", [0, 0.5, 1])
@requires_scipy
def test_interpolate_pd_compat(method, fill_value, dim, shape, frac_nan) -> None:
    da, df = make_interpolate_example_data(shape, frac_nan)

    actual = da.interpolate_na(method=method, dim=dim, fill_value=fill_value)
    expected = df.interpolate(
        method=method,
        axis=da.get_axis_num(dim),
        fill_value=fill_value,
        limit_direction="both",
    )

    if method == "linear":
        # Note, Pandas does not take left/right fill_value into account
        # for the numpy linear methods.
        # see https://github.com/pandas-dev/pandas/issues/55144
        # This aligns the pandas output with the xarray output
        fixed = expected.values.copy()
        fixed[pd.isnull(actual.values)] = np.nan
        fixed[actual.values == fill_value] = fill_value
    else:
        fixed = expected.values
    np.testing.assert_allclose(actual.values, fixed)


@requires_scipy
@pytest.mark.parametrize("method", ["barycentric", "krogh", "pchip", "spline", "akima"])
def test_scipy_methods_function(method) -> None:
    # Note: Pandas does some wacky things with these methods and the full
    # integration tests won't work.
    da, _ = make_interpolate_example_data((25, 25), 0.4, non_uniform=True)
    if method == "spline":
        with pytest.warns(PendingDeprecationWarning):
            actual = da.interpolate_na(method=method, dim="time")
    else:
        actual = da.interpolate_na(method=method, dim="time")
    assert (da.count("time") <= actual.count("time")).all()


@requires_scipy
def test_interpolate_pd_compat_non_uniform_index():
    shapes = [(8, 8), (1, 20), (20, 1), (100, 100)]
    frac_nans = [0, 0.5, 1]
    methods = ["time", "index", "values"]

    for shape, frac_nan, method in itertools.product(shapes, frac_nans, methods):
        da, df = make_interpolate_example_data(shape, frac_nan, non_uniform=True)
        for dim in ["time", "x"]:
            if method == "time" and dim != "time":
                continue
            actual = da.interpolate_na(
                method="linear", dim=dim, use_coordinate=True, fill_value=np.nan
            )
            expected = df.interpolate(
                method=method,
                axis=da.get_axis_num(dim),
            )

            # Note, Pandas does some odd things with the left/right fill_value
            # for the linear methods. This next line inforces the xarray
            # fill_value convention on the pandas output. Therefore, this test
            # only checks that interpolated values are the same (not nans)
            expected_values = expected.values.copy()
            expected_values[pd.isnull(actual.values)] = np.nan

            np.testing.assert_allclose(actual.values, expected_values)


@requires_scipy
def test_interpolate_pd_compat_polynomial():
    shapes = [(8, 8), (1, 20), (20, 1), (100, 100)]
    frac_nans = [0, 0.5, 1]
    orders = [1, 2, 3]

    for shape, frac_nan, order in itertools.product(shapes, frac_nans, orders):
        da, df = make_interpolate_example_data(shape, frac_nan)

        for dim in ["time", "x"]:
            actual = da.interpolate_na(
                method="polynomial", order=order, dim=dim, use_coordinate=False
            )
            expected = df.interpolate(
                method="polynomial", order=order, axis=da.get_axis_num(dim)
            )
            np.testing.assert_allclose(actual.values, expected.values)


@requires_scipy
def test_interpolate_pd_compat_limits():
    shapes = [(7, 7)]
    method = "slinear"  # need slinear, since pandas does constant extrapolation for methods 'time', 'index', 'values'
    frac_nan = 0.5
    limits = [
        None,
        1,
        3,
    ]  # pandas 2.1.4 is currently unable to handle coordinate based limits!
    limit_directions = ["forward", "backward"]
    limit_areas = [None, "outside", "inside"]

    for shape, limit, limit_direction, limit_area in itertools.product(
        shapes, limits, limit_directions, limit_areas
    ):
        da, df = make_interpolate_example_data(shape, frac_nan, non_uniform=True)
        for dim in ["time", "x"]:
            actual = da.fill_gaps(
                dim=dim,
                limit=limit,
                limit_direction=limit_direction,
                limit_area=limit_area,
                use_coordinate=False,
            ).interpolate_na(
                dim=dim,
                method=method,
                use_coordinate=True,
                fill_value="extrapolate",
            )
            expected = df.interpolate(
                method=method,
                axis=da.get_axis_num(dim),
                limit=limit,
                limit_direction=limit_direction,
                limit_area=limit_area,
                fill_value="extrapolate",
            )
            np.testing.assert_allclose(actual.values, expected.values)


@requires_bottleneck
def test_fill_pd_compat_limits():
    shapes = [(7, 7)]
    frac_nan = 0.5
    limits = [
        None,
        1,
        3,
    ]  # pandas 2.1.4 is currently unable to handle coordinate based limits!
    limit_areas = [None, "outside", "inside"]

    for shape, limit, limit_area in itertools.product(shapes, limits, limit_areas):
        da, df = make_interpolate_example_data(shape, frac_nan, non_uniform=True)
        for dim in ["time", "x"]:
            masked = da.fill_gaps(
                dim=dim,
                limit=limit,
                limit_area=limit_area,
                use_coordinate=False,
            )
            filled_forward = masked.ffill()
            filled_backward = masked.bfill()
            if (
                pd.__version__ >= "2.2.0"
            ):  # limit_area was introduced in pandas ffill in v2.2.0
                expected_forward = df.ffill(
                    axis=da.get_axis_num(dim), limit=limit, limit_area=limit_area
                )
                expected_backward = df.bfill(
                    axis=da.get_axis_num(dim), limit=limit, limit_area=limit_area
                )
                np.testing.assert_allclose(
                    filled_forward.values, expected_forward.values
                )
                np.testing.assert_allclose(
                    filled_backward.values, expected_backward.values
                )


@requires_scipy
def test_interpolate_unsorted_index_raises():
    vals = np.array([1, 2, 3], dtype=np.float64)
    expected = xr.DataArray(vals, dims="x", coords={"x": [2, 1, 3]})
    with pytest.raises(ValueError, match=r"Index 'x' must be monotonically increasing"):
        expected.interpolate_na(dim="x", method="index")


def test_interpolate_invalid_interpolator_raises():
    da = xr.DataArray(np.array([1, 2, np.nan, 5], dtype=np.float64), dims="x")
    with pytest.raises(ValueError, match=r"not a valid"):
        da.interpolate_na(dim="x", method="foo")


def test_interpolate_duplicate_values_raises():
    data = np.random.randn(2, 3)
    da = xr.DataArray(data, coords=[("x", ["a", "a"]), ("y", [0, 1, 2])])
    with pytest.raises(ValueError, match=r"Index 'x' has duplicate values"):
        da.interpolate_na(dim="x", method="foo")


def test_interpolate_multiindex_raises():
    data = np.random.randn(2, 3)
    data[1, 1] = np.nan
    da = xr.DataArray(data, coords=[("x", ["a", "b"]), ("y", [0, 1, 2])])
    das = da.stack(z=("x", "y"))
    with pytest.raises(TypeError, match=r"Index 'z' must be castable to float64"):
        das.interpolate_na(dim="z")


def test_interpolate_2d_coord_raises():
    coords = {
        "x": xr.Variable(("a", "b"), np.arange(6).reshape(2, 3)),
        "y": xr.Variable(("a", "b"), np.arange(6).reshape(2, 3)) * 2,
    }

    data = np.random.randn(2, 3)
    data[1, 1] = np.nan
    da = xr.DataArray(data, dims=("a", "b"), coords=coords)
    with pytest.raises(ValueError, match=r"interpolation must be 1D"):
        da.interpolate_na(dim="a", use_coordinate="x")


@requires_scipy
def test_interpolate_kwargs():
    da = xr.DataArray(np.array([4, 5, np.nan], dtype=np.float64), dims="x")
    expected = xr.DataArray(np.array([4, 5, 6], dtype=np.float64), dims="x")
    actual = da.interpolate_na(dim="x", fill_value="extrapolate")
    assert_equal(actual, expected)

    expected = xr.DataArray(np.array([4, 5, -999], dtype=np.float64), dims="x")
    actual = da.interpolate_na(dim="x", fill_value=-999)
    assert_equal(actual, expected)


def test_interpolate_keep_attrs():
    vals = np.array([1, 2, 3, 4, 5, 6], dtype=np.float64)
    mvals = vals.copy()
    mvals[2] = np.nan
    missing = xr.DataArray(mvals, dims="x")
    missing.attrs = {"test": "value"}

    actual = missing.interpolate_na(dim="x", keep_attrs=True)
    assert actual.attrs == {"test": "value"}


def test_interpolate():
    vals = np.array([1, 2, 3, 4, 5, 6], dtype=np.float64)
    expected = xr.DataArray(vals, dims="x")
    mvals = vals.copy()
    mvals[2] = np.nan
    missing = xr.DataArray(mvals, dims="x")

    actual = missing.interpolate_na(dim="x")

    assert_equal(actual, expected)


@requires_scipy
@pytest.mark.parametrize(
    "method,vals",
    [
        pytest.param(method, vals, id=f"{desc}:{method}")
        for method in [
            "linear",
            "nearest",
            "zero",
            "slinear",
            "quadratic",
            "cubic",
            "polynomial",
        ]
        for (desc, vals) in [
            ("no nans", np.array([1, 2, 3, 4, 5, 6], dtype=np.float64)),
            ("one nan", np.array([1, np.nan, np.nan], dtype=np.float64)),
            ("all nans", np.full(6, np.nan, dtype=np.float64)),
        ]
    ],
)
def test_interp1d_fastrack(method, vals):
    expected = xr.DataArray(vals, dims="x")
    actual = expected.interpolate_na(dim="x", method=method)

    assert_equal(actual, expected)


@requires_bottleneck
def test_interpolate_limits():
    n = np.nan
    times = pd.date_range("2000-01-01", periods=9, freq="2h")
    coords = {"yt": ("y", times)}
    da = xr.DataArray([n, n, 3, n, n, 6, n, 8, n], dims=["y"], coords=coords)

    actual = da.interpolate_na(dim="y", limit=None, fill_value="extrapolate")
    # With no limit, all gaps should be interpolated (forward+backward, including boundaries). Introduced in xarray due to a bug (GH7665), but kept for backward compatibility. With limit, do only forward interpolation.
    expected = da.copy(data=[1, 2, 3, 4, 5, 6, 7, 8, 9])
    assert_equal(actual, expected)

    actual = da.interpolate_na(dim="y", limit=None, max_gap=2, fill_value="extrapolate")
    expected = da.copy(data=[1, 2, 3, n, n, 6, 7, 8, 9])
    assert_equal(actual, expected)

    actual = da.interpolate_na(dim="y", limit=1, fill_value="extrapolate")
    expected = da.copy(data=[n, n, 3, 4, n, 6, 7, 8, 9])
    assert_equal(actual, expected)

    actual = da.interpolate_na(dim="y", limit=1, max_gap=2, fill_value="extrapolate")
    expected = da.copy(data=[n, n, 3, n, n, 6, 7, 8, 9])
    assert_equal(actual, expected)


@requires_bottleneck
def test_interpolate_double_coordinate():
    # Check if max_gap is able to handle string coordinate names
    # Limit is always refering to an index
    n = np.nan
    da = xr.DataArray(
        [[1, n, n, 4, n, 6, 7], [1, n, n, n, 5, n, n]],
        dims=["x", "y"],
        coords={"y1": ("y", np.arange(7)), "y2": ("y", np.arange(7) * 2)},
    )
    actual = da.interpolate_na(
        dim="y", limit=1, max_gap=4, use_coordinate="y1", fill_value="extrapolate"
    )
    expected = da.copy(data=[[1, 2, n, 4, 5, 6, 7], [1, 2, n, n, 5, 6, n]])
    assert_equal(actual, expected)

    actual = da.interpolate_na(
        "y",
        limit=2,
        max_gap=4,
        use_coordinate="y2",
        fill_value="extrapolate",
    )
    expected = da.copy(data=[[1, n, n, 4, 5, 6, 7], [1, n, n, n, 5, 6, 7]])
    assert_equal(actual, expected)


@requires_scipy
def test_interpolate_methods():
    for method in ["linear", "nearest", "zero", "slinear", "quadratic", "cubic"]:
        kwargs = {}
        da = xr.DataArray(
            np.array([0, 1, 2, np.nan, np.nan, np.nan, 6, 7, 8], dtype=np.float64),
            dims="x",
        )
        actual = da.interpolate_na("x", method=method, **kwargs)
        assert actual.isnull().sum() == 0

        actual = da.interpolate_na("x", method=method, limit=2, **kwargs)
        assert actual.isnull().sum() == 1


@requires_scipy
def test_interpolators():
    for method, interpolator in [
        ("linear", NumpyInterpolator),
        ("linear", ScipyInterpolator),
        ("spline", SplineInterpolator),
    ]:
        xi = np.array([-1, 0, 1, 2, 5], dtype=np.float64)
        yi = np.array([-10, 0, 10, 20, 50], dtype=np.float64)
        x = np.array([3, 4], dtype=np.float64)

        f = interpolator(xi, yi, method=method)
        out = f(x)
        assert pd.isnull(out).sum() == 0


def test_interpolate_use_coordinate():
    xc = xr.Variable("x", [100, 200, 300, 400, 500, 600])
    da = xr.DataArray(
        np.array([1, 2, np.nan, np.nan, np.nan, 6], dtype=np.float64),
        dims="x",
        coords={"xc": xc},
    )

    # use_coordinate == False is same as using the default index
    actual = da.interpolate_na(dim="x", use_coordinate=False)
    expected = da.interpolate_na(dim="x")
    assert_equal(actual, expected)

    # possible to specify non index coordinate
    actual = da.interpolate_na(dim="x", use_coordinate="xc")
    expected = da.interpolate_na(dim="x")
    assert_equal(actual, expected)

    # possible to specify index coordinate by name
    actual = da.interpolate_na(dim="x", use_coordinate="x")
    expected = da.interpolate_na(dim="x")
    assert_equal(actual, expected)


@requires_dask
def test_interpolate_dask():
    da, _ = make_interpolate_example_data((40, 40), 0.5)
    da = da.chunk({"x": 5})
    actual = da.interpolate_na("time")
    expected = da.load().interpolate_na("time")
    assert isinstance(actual.data, dask_array_type)
    assert_equal(actual.compute(), expected)

    # with limit
    da = da.chunk({"x": 5})
    actual = da.interpolate_na("time", limit=3)
    expected = da.load().interpolate_na("time", limit=3)
    assert isinstance(actual.data, dask_array_type)
    assert_equal(actual, expected)


@requires_dask
def test_interpolate_dask_raises_for_invalid_chunk_dim():
    da, _ = make_interpolate_example_data((40, 40), 0.5)
    da = da.chunk({"time": 5})
    # this checks for ValueError in dask.array.apply_gufunc
    with pytest.raises(ValueError, match=r"consists of multiple chunks"):
        da.interpolate_na("time")


@requires_dask
@requires_scipy
@pytest.mark.parametrize("dtype, method", [(int, "linear"), (int, "nearest")])
def test_interpolate_dask_expected_dtype(dtype, method):
    da = xr.DataArray(
        data=np.array([0, 1], dtype=dtype),
        dims=["time"],
        coords=dict(time=np.array([0, 1])),
    ).chunk(dict(time=2))
    da = da.interp(time=np.array([0, 0.5, 1, 2]), method=method)

    assert da.dtype == da.compute().dtype


@requires_numbagg_or_bottleneck
def test_ffill():
    da = xr.DataArray(np.array([4, 5, np.nan], dtype=np.float64), dims="x")
    expected = xr.DataArray(np.array([4, 5, 5], dtype=np.float64), dims="x")
    actual = da.ffill("x")
    assert_equal(actual, expected)


@pytest.mark.parametrize("compute_backend", [None], indirect=True)
@pytest.mark.parametrize("method", ["ffill", "bfill"])
def test_b_ffill_use_bottleneck_numbagg(method, compute_backend):
    """
    bfill & ffill fail if both bottleneck and numba are disabled
    """
    da = xr.DataArray(np.array([4, 5, np.nan], dtype=np.float64), dims="x")
    with pytest.raises(RuntimeError):
        getattr(da, method)("x")


@requires_dask
@pytest.mark.parametrize("compute_backend", [None], indirect=True)
@pytest.mark.parametrize("method", ["ffill", "bfill"])
def test_b_ffill_use_bottleneck_dask(method, compute_backend):
    """
    ffill fails if both bottleneck and numba are disabled, on dask arrays
    """
    da = xr.DataArray(np.array([4, 5, np.nan], dtype=np.float64), dims="x")
    with pytest.raises(RuntimeError):
        getattr(da, method)("x")


@requires_numbagg
@requires_dask
@pytest.mark.parametrize("compute_backend", ["numbagg"], indirect=True)
def test_ffill_use_numbagg_dask(compute_backend):
    da = xr.DataArray(np.array([4, 5, np.nan], dtype=np.float64), dims="x")
    da = da.chunk(x=-1)
    # Succeeds with a single chunk:
    _ = da.ffill("x").compute()


@requires_bottleneck
@requires_dask
@pytest.mark.parametrize("method", ["ffill", "bfill"])
def test_ffill_bfill_dask(method):
    da, _ = make_interpolate_example_data((40, 40), 0.5)
    da = da.chunk({"x": 5})

    dask_method = getattr(da, method)
    numpy_method = getattr(da.compute(), method)
    # unchunked axis
    with raise_if_dask_computes():
        actual = dask_method("time")
    expected = numpy_method("time")
    assert_equal(actual, expected)

    # chunked axis
    with raise_if_dask_computes():
        actual = dask_method("x")
    expected = numpy_method("x")
    assert_equal(actual, expected)

    # with limit
    with raise_if_dask_computes():
        actual = dask_method("time", limit=3)
    expected = numpy_method("time", limit=3)
    assert_equal(actual, expected)

    # limit < axis size
    with raise_if_dask_computes():
        actual = dask_method("x", limit=2)
    expected = numpy_method("x", limit=2)
    assert_equal(actual, expected)

    # limit > axis size
    with raise_if_dask_computes():
        actual = dask_method("x", limit=41)
    expected = numpy_method("x", limit=41)
    assert_equal(actual, expected)


@requires_bottleneck
def test_ffill_bfill_nonans():
    vals = np.array([1, 2, 3, 4, 5, 6], dtype=np.float64)
    expected = xr.DataArray(vals, dims="x")

    actual = expected.ffill(dim="x")
    assert_equal(actual, expected)

    actual = expected.bfill(dim="x")
    assert_equal(actual, expected)


@requires_bottleneck
def test_ffill_bfill_allnans():
    vals = np.full(6, np.nan, dtype=np.float64)
    expected = xr.DataArray(vals, dims="x")

    actual = expected.ffill(dim="x")
    assert_equal(actual, expected)

    actual = expected.bfill(dim="x")
    assert_equal(actual, expected)


@requires_bottleneck
def test_ffill_functions(da):
    result = da.ffill("time")
    assert result.isnull().sum() == 0


@requires_bottleneck
def test_ffill_limit():
    da = xr.DataArray(
        [0, np.nan, np.nan, np.nan, np.nan, 3, 4, 5, np.nan, 6, 7], dims="time"
    )
    result = da.ffill("time")
    expected = xr.DataArray([0, 0, 0, 0, 0, 3, 4, 5, 5, 6, 7], dims="time")
    assert_array_equal(result, expected)

    result = da.ffill("time", limit=1)
    expected = xr.DataArray(
        [0, 0, np.nan, np.nan, np.nan, 3, 4, 5, 5, 6, 7], dims="time"
    )
    assert_array_equal(result, expected)


def test_interpolate_dataset(ds):
    actual = ds.interpolate_na(dim="time")
    # no missing values in var1
    assert actual["var1"].count("time") == actual.sizes["time"]

    # var2 should be the same as it was
    assert_array_equal(actual["var2"], ds["var2"])


@requires_bottleneck
def test_ffill_dataset(ds):
    ds.ffill(dim="time")


@requires_bottleneck
def test_bfill_dataset(ds):
    ds.ffill(dim="time")


@requires_bottleneck
def test_get_gap_left_edge():
    n = np.nan
    arr = [
        [n, 1, n, n, n, n, n, n, 4],
        [n, n, n, 1, n, n, 4, n, n],
    ]

    y = np.arange(9) * 3
    da = xr.DataArray(arr, dims=["x", "y"], coords={"x": [0, 1], "y": y})
    index = get_clean_interp_index(da, dim="y", use_coordinate=True)
    actual = _get_gap_left_edge(da, dim="y", index=index)
    expected = da.copy(
        data=[[n, 3, 3, 3, 3, 3, 3, 3, 24], [n, n, n, 9, 9, 9, 18, 18, 18]]
    )
    assert_equal(actual, expected)

    actual = _get_gap_left_edge(da, dim="y", index=index, outside=True)
    expected = da.copy(
        data=[[0, 3, 3, 3, 3, 3, 3, 3, 24], [0, 0, 0, 9, 9, 9, 18, 18, 18]]
    )
    assert_equal(actual, expected)

    y = [0, 2, 5, 6, 7, 8, 10, 12, 14]
    da = xr.DataArray(arr, dims=["x", "y"], coords={"x": [0, 1], "y": y})
    index = get_clean_interp_index(da, dim="y", use_coordinate=True)
    actual = _get_gap_left_edge(da, dim="y", index=index)
    expected = da.copy(
        data=[[n, 2, 2, 2, 2, 2, 2, 2, 14], [n, n, n, 6, 6, 6, 10, 10, 10]]
    )


@requires_bottleneck
def test_get_gap_right_edge():
    n = np.nan
    arr = [
        [n, 1, n, n, n, n, n, n, 4],
        [n, n, n, 1, n, n, 4, n, n],
    ]

    y = np.arange(9) * 3
    da = xr.DataArray(arr, dims=["x", "y"], coords={"x": [0, 1], "y": y})
    index = get_clean_interp_index(da, dim="y", use_coordinate=True)
    actual = _get_gap_right_edge(da, dim="y", index=index)
    expected = da.copy(
        data=[[3, 3, 24, 24, 24, 24, 24, 24, 24], [9, 9, 9, 9, 18, 18, 18, n, n]]
    )
    assert_equal(actual, expected)

    actual = _get_gap_right_edge(da, dim="y", index=index, outside=True)
    expected = da.copy(
        data=[[3, 3, 24, 24, 24, 24, 24, 24, 24], [9, 9, 9, 9, 18, 18, 18, 24, 24]]
    )
    assert_equal(actual, expected)

    y = [0, 2, 5, 6, 7, 8, 10, 12, 14]
    da = xr.DataArray(arr, dims=["x", "y"], coords={"x": [0, 1], "y": y})
    index = get_clean_interp_index(da, dim="y", use_coordinate=True)
    actual = _get_gap_right_edge(da, dim="y", index=index)
    expected = da.copy(
        data=[[2, 2, 14, 14, 14, 14, 14, 14, 14], [6, 6, 6, 6, 10, 10, 10, n, n]]
    )


@requires_bottleneck
def test_get_gap_dist_to_left_edge():
    n = np.nan
    arr = [
        [n, 1, n, n, n, n, n, n, 4],
        [n, n, n, 1, n, n, 4, n, n],
    ]

    y = np.arange(9) * 3
    da = xr.DataArray(arr, dims=["x", "y"], coords={"x": [0, 1], "y": y})
    index = get_clean_interp_index(da, dim="y", use_coordinate=True)
    actual = _get_gap_dist_to_left_edge(da, dim="y", index=index)
    expected = da.copy(
        data=[[n, 0, 3, 6, 9, 12, 15, 18, 0], [n, n, n, 0, 3, 6, 0, 3, 6]]
    )
    assert_equal(actual, expected)

    y = [0, 2, 5, 6, 7, 8, 10, 12, 14]
    da = xr.DataArray(arr, dims=["x", "y"], coords={"x": [0, 1], "y": y})
    index = get_clean_interp_index(da, dim="y", use_coordinate=True)
    actual = _get_gap_dist_to_left_edge(da, dim="y", index=index)
    expected = da.copy(data=[[n, 0, 3, 4, 5, 6, 8, 10, 0], [n, n, n, 0, 1, 2, 0, 2, 4]])


@requires_bottleneck
def test_get_gap_dist_to_right_edge():
    n = np.nan
    arr = [
        [n, 1, n, n, n, n, n, n, 4],
        [n, n, n, 1, n, n, 4, n, n],
    ]

    y = np.arange(9) * 3
    da = xr.DataArray(arr, dims=["x", "y"], coords={"x": [0, 1], "y": y})
    index = get_clean_interp_index(da, dim="y", use_coordinate=True)
    actual = _get_gap_dist_to_right_edge(da, dim="y", index=index)
    expected = da.copy(
        data=[[3, 0, 18, 15, 12, 9, 6, 3, 0], [9, 6, 3, 0, 6, 3, 0, n, n]]
    )
    assert_equal(actual, expected)

    y = [0, 2, 5, 6, 7, 8, 10, 12, 14]
    da = xr.DataArray(arr, dims=["x", "y"], coords={"x": [0, 1], "y": y})
    index = get_clean_interp_index(da, dim="y", use_coordinate=True)
    actual = _get_gap_dist_to_right_edge(da, dim="y", index=index)
    expected = da.copy(data=[[2, 0, 9, 8, 7, 6, 4, 2, 0], [5, 3, 0, 4, 3, 2, 0, n, n]])


@requires_bottleneck
@pytest.mark.parametrize(
    "y, lengths_expected",
    [
        [np.arange(9), [[1, 0, 7, 7, 7, 7, 7, 7, 0], [3, 3, 3, 0, 3, 3, 0, 2, 2]]],
        [
            np.arange(9) * 3,
            [[3, 0, 21, 21, 21, 21, 21, 21, 0], [9, 9, 9, 0, 9, 9, 0, 6, 6]],
        ],
        [
            [0, 2, 5, 6, 7, 8, 10, 12, 14],
            [[2, 0, 12, 12, 12, 12, 12, 12, 0], [6, 6, 6, 0, 4, 4, 0, 4, 4]],
        ],
    ],
)
def test_get_nan_block_lengths(y, lengths_expected):
    arr = [
        [np.nan, 1, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 4],
        [np.nan, np.nan, np.nan, 1, np.nan, np.nan, 4, np.nan, np.nan],
    ]
    da = xr.DataArray(arr, dims=["x", "y"], coords={"x": [0, 1], "y": y})
    index = get_clean_interp_index(da, dim="y", use_coordinate=True)
    actual = _get_nan_block_lengths(da, dim="y", index=index)
    expected = da.copy(data=lengths_expected)
    assert_equal(actual, expected)


@requires_bottleneck
def test_get_nan_block_lengths_2d():
    n = np.nan
    da = xr.DataArray(
        [
            [1, 2, 3, 4, n, 6, n, n, n, 10, 11, n],
            [n, n, 3, n, n, 6, n, n, n, 10, n, n],
            [n, n, 3, n, n, 6, n, n, n, 10, n, n],
            [n, 2, 3, 4, n, 6, n, n, n, 10, 11, n],
        ],
        dims=["x", "y"],
        coords={"x": np.arange(4), "y": np.arange(12) ** 2},
    )
    index = get_clean_interp_index(da, dim="y", use_coordinate=False)
    actual = _get_nan_block_lengths(da, dim="y", index=index)
    expected_y = da.copy(
        data=[
            [0, 0, 0, 0, 2, 0, 4, 4, 4, 0, 0, 1],
            [2, 2, 0, 3, 3, 0, 4, 4, 4, 0, 2, 2],
            [2, 2, 0, 3, 3, 0, 4, 4, 4, 0, 2, 2],
            [1, 0, 0, 0, 2, 0, 4, 4, 4, 0, 0, 1],
        ]
    )
    assert_equal(actual, expected_y)
    index = get_clean_interp_index(da, dim="y", use_coordinate=True)
    actual = _get_nan_block_lengths(da, dim="y", index=index)
    expected_y = da.copy(
        data=[
            [0, 0, 0, 0, 16, 0, 56, 56, 56, 0, 0, 21],
            [4, 4, 0, 21, 21, 0, 56, 56, 56, 0, 40, 40],
            [4, 4, 0, 21, 21, 0, 56, 56, 56, 0, 40, 40],
            [1, 0, 0, 0, 16, 0, 56, 56, 56, 0, 0, 21],
        ]
    )
    assert_equal(actual, expected_y)


@requires_bottleneck
def test_get_limit_fill_mask():
    T = True
    F = False
    n = np.nan
    arr = [
        [n, 1, n, n, n, n, n, n, 4],
        [n, n, n, 1, n, n, 4, n, n],
    ]
    y = [0, 2, 5, 6, 7, 8, 10, 12, 14]
    da = xr.DataArray(arr, dims=["x", "y"], coords={"x": [0, 1], "y": y})
    index = get_clean_interp_index(da, dim="y", use_coordinate=True)

    with pytest.raises(ValueError, match=r"limit_direction must be one of"):
        _get_limit_fill_mask(da, dim="y", index=index, limit=3, limit_direction="cat")

    actual = _get_limit_fill_mask(
        da, dim="y", index=index, limit=3, limit_direction="forward"
    )
    expected = da.copy(data=[[T, F, F, T, T, T, T, T, F], [T, T, T, F, F, F, F, F, T]])
    assert_equal(actual, expected)

    actual = _get_limit_fill_mask(
        da, dim="y", index=index, limit=3, limit_direction="backward"
    )
    expected = da.copy(data=[[F, F, T, T, T, T, T, F, F], [T, T, F, F, F, F, F, T, T]])
    assert_equal(actual, expected)

    actual = _get_limit_fill_mask(
        da, dim="y", index=index, limit=3, limit_direction="both"
    )
    expected = da.copy(data=[[F, F, F, T, T, T, T, F, F], [T, T, F, F, F, F, F, F, T]])
    assert_equal(actual, expected)

    actual = _get_limit_fill_mask(
        da, dim="y", index=index, limit=1, limit_direction="forward"
    )
    expected = da.copy(data=[[T, F, T, T, T, T, T, T, F], [T, T, T, F, F, T, F, T, T]])
    assert_equal(actual, expected)

    actual = _get_limit_fill_mask(
        da, dim="y", index=index, limit=1, limit_direction="backward"
    )
    expected = da.copy(data=[[T, F, T, T, T, T, T, T, F], [T, T, F, F, T, T, F, T, T]])
    assert_equal(actual, expected)

    actual = _get_limit_fill_mask(
        da, dim="y", index=index, limit=1, limit_direction="both"
    )
    expected = da.copy(data=[[T, F, T, T, T, T, T, T, F], [T, T, F, F, F, T, F, T, T]])
    assert_equal(actual, expected)


@requires_bottleneck
def test_get_area_mask():
    T = True
    F = False
    n = np.nan
    arr = [
        [n, 1, n, n, 5, n, n, n, 4],
        [n, n, n, 1, n, n, 4, n, n],
    ]
    y = [0, 2, 5, 6, 7, 8, 10, 12, 14]
    da = xr.DataArray(arr, dims=["x", "y"], coords={"x": [0, 1], "y": y})
    index = get_clean_interp_index(da, dim="y", use_coordinate=True)

    with pytest.raises(ValueError, match=r"limit_area must be one of"):
        _get_limit_area_mask(da, dim="y", index=index, limit_area="cow")

    actual = _get_limit_area_mask(da, dim="y", index=index, limit_area="inside")
    expected = da.copy(data=[[T, F, F, F, F, F, F, F, F], [T, T, T, F, F, F, F, T, T]])
    assert_equal(actual, expected)

    actual = _get_limit_area_mask(da, dim="y", index=index, limit_area="outside")
    expected = da.copy(data=[[F, F, T, T, F, T, T, T, F], [F, F, F, F, T, T, F, F, F]])
    assert_equal(actual, expected)


@requires_cftime
@pytest.mark.parametrize("calendar", _CFTIME_CALENDARS)
def test_get_clean_interp_index_cf_calendar(cf_da, calendar):
    """The index for CFTimeIndex is in units of days. This means that if two series using a 360 and 365 days
    calendar each have a trend of .01C/year, the linear regression coefficients will be different because they
    have different number of days.

    Another option would be to have an index in units of years, but this would likely create other difficulties.
    """
    i = get_clean_interp_index(cf_da(calendar), dim="time")
    np.testing.assert_array_equal(i, np.arange(10) * 1e9 * 86400)


@requires_cftime
@pytest.mark.parametrize("calendar", ["gregorian", "proleptic_gregorian"])
@pytest.mark.parametrize("freq", ["1D", "1ME", "1YE"])
def test_get_clean_interp_index_dt(cf_da, calendar, freq):
    """In the gregorian case, the index should be proportional to normal datetimes."""
    g = cf_da(calendar, freq=freq)
    g["stime"] = xr.Variable(data=g.time.to_index().to_datetimeindex(), dims=("time",))

    gi = get_clean_interp_index(g, "time")
    si = get_clean_interp_index(g, "time", use_coordinate="stime")
    np.testing.assert_array_equal(gi, si)


@requires_cftime
def test_get_clean_interp_index_potential_overflow():
    da = xr.DataArray(
        [0, 1, 2],
        dims=("time",),
        coords={"time": xr.cftime_range("0000-01-01", periods=3, calendar="360_day")},
    )
    get_clean_interp_index(da, "time")


@pytest.mark.parametrize("index", ([0, 2, 1], [0, 1, 1]))
def test_get_clean_interp_index_strict(index):
    da = xr.DataArray([0, 1, 2], dims=("x",), coords={"x": index})

    with pytest.raises(ValueError):
        get_clean_interp_index(da, "x")

    clean = get_clean_interp_index(da, "x", strict=False)
    np.testing.assert_array_equal(index, clean)
    assert clean.dtype == np.float64


def test_get_clean_interp_index_double_coordinate():
    da = xr.DataArray(
        np.ones((2, 7)),
        dims=["x", "y"],
        coords={
            "x": ("x", [10, 20]),
            "y1": ("y", np.arange(7) * 2),
            "y2": ("y", np.arange(7) * 3),
        },
    )
    with pytest.raises(ValueError, match=r"not a valid dimension"):
        get_clean_interp_index(da, "y1", use_coordinate=True)

    actual = get_clean_interp_index(da, "y", use_coordinate=True)
    expected = xr.Variable(["y"], np.arange(7))
    assert_equal(actual, expected)

    actual = get_clean_interp_index(da, "y", use_coordinate="y1")
    expected = xr.Variable(["y"], np.arange(7) * 2)
    assert_equal(actual, expected)

    with pytest.raises(ValueError, match=r"must have dimension"):
        get_clean_interp_index(da, "x", use_coordinate="y1")


@pytest.fixture
def da_time():
    return xr.DataArray(
        [np.nan, 1, 2, np.nan, np.nan, 5, np.nan, np.nan, np.nan, np.nan, 10],
        dims=["t"],
    )


def test_interpolate_na_max_gap_errors(da_time):
    with pytest.raises(ValueError, match=r"max_gap must be a scalar."):
        da_time.interpolate_na("t", max_gap=(1,))

    da_time["t"] = pd.date_range("2001-01-01", freq="h", periods=11)
    with pytest.raises(TypeError, match=r"Expected value of type str"):
        da_time.interpolate_na("t", max_gap=1)

    with pytest.raises(TypeError, match=r"Expected integer or floating point"):
        da_time.interpolate_na("t", max_gap="1h", use_coordinate=False)

    with pytest.raises(ValueError, match=r"Could not convert 'huh' to timedelta64"):
        da_time.interpolate_na("t", max_gap="huh")


@requires_bottleneck
@pytest.mark.parametrize(
    "time_range_func",
    [pd.date_range, pytest.param(xr.cftime_range, marks=requires_cftime)],
)
@pytest.mark.parametrize("transform", [lambda x: x, lambda x: x.to_dataset(name="a")])
@pytest.mark.parametrize(
    "max_gap", ["3h", np.timedelta64(3, "h"), pd.to_timedelta("3h")]
)
def test_interpolate_na_max_gap_time_specifier(
    da_time, max_gap, transform, time_range_func
):
    da_time["t"] = time_range_func("2001-01-01", freq="h", periods=11)
    expected = transform(
        da_time.copy(data=[np.nan, 1, 2, 3, 4, 5, np.nan, np.nan, np.nan, np.nan, 10])
    )
    actual = transform(da_time).interpolate_na("t", max_gap=max_gap)
    assert_allclose(actual, expected)


@requires_bottleneck
@pytest.mark.parametrize(
    "coords",
    [
        None,
        {"x": np.arange(4), "y": np.arange(12)},
    ],
)
def test_interpolate_na_max_gap_2d(coords):
    n = np.nan
    if coords is None:
        use_coordinate = False
    else:
        use_coordinate = True
    da = xr.DataArray(
        [
            [1, 2, 3, 4, n, 6, n, n, n, 10, 11, n],
            [n, n, 3, n, n, 6, n, n, n, 10, n, n],
            [n, n, 3, n, n, 6, n, n, n, 10, n, n],
            [n, 2, 3, 4, n, 6, n, n, n, 10, 11, n],
        ],
        dims=["x", "y"],
        coords=coords,
    )

    actual = da.interpolate_na(
        "y", use_coordinate=use_coordinate, max_gap=2, fill_value="extrapolate"
    )
    expected_y = da.copy(
        data=[
            [1, 2, 3, 4, 5, 6, n, n, n, 10, 11, 12],
            [1, 2, 3, n, n, 6, n, n, n, 10, 11, 12],
            [1, 2, 3, n, n, 6, n, n, n, 10, 11, 12],
            [1, 2, 3, 4, 5, 6, n, n, n, 10, 11, 12],
        ]
    )
    assert_equal(actual, expected_y)

    actual = da.interpolate_na(
        "y", use_coordinate=use_coordinate, max_gap=1, fill_value="extrapolate"
    )
    expected_y_extra = da.copy(
        data=[
            [1, 2, 3, 4, n, 6, n, n, n, 10, 11, 12],
            [n, n, 3, n, n, 6, n, n, n, 10, n, n],
            [n, n, 3, n, n, 6, n, n, n, 10, n, n],
            [1, 2, 3, 4, n, 6, n, n, n, 10, 11, 12],
        ]
    )
    assert_equal(actual, expected_y_extra)

    actual = da.interpolate_na(
        "x", use_coordinate=use_coordinate, max_gap=3, fill_value="extrapolate"
    )
    expected_x = xr.DataArray(
        [
            [1, 2, 3, 4, n, 6, n, n, n, 10, 11, n],
            [n, 2, 3, 4, n, 6, n, n, n, 10, 11, n],
            [n, 2, 3, 4, n, 6, n, n, n, 10, 11, n],
            [n, 2, 3, 4, n, 6, n, n, n, 10, 11, n],
        ],
        dims=["x", "y"],
        coords=coords,
    )
    assert_equal(actual, expected_x)


@requires_bottleneck
def test_interpolate_na_limit_2d():
    n = np.nan
    times = pd.date_range("2000-01-01", periods=12, freq="3h")
    coords = {
        "x": np.arange(3) * 2,
        "time": (times),
    }
    da = xr.DataArray(
        [
            [1, 2, 3, 4, n, 6, n, n, n, 10, 11, n],
            [n, n, 3, n, n, 6, n, n, n, 10, n, n],
            [n, 2, 3, 4, n, 6, n, n, n, 10, 11, n],
        ],
        coords=coords,
    )

    actual = da.interpolate_na("time", limit=1, fill_value="extrapolate")
    expected = da.copy(
        data=[
            [1, 2, 3, 4, 5, 6, 7, n, n, 10, 11, 12],
            [n, n, 3, 4, n, 6, 7, n, n, 10, 11, n],
            [n, 2, 3, 4, 5, 6, 7, n, n, 10, 11, 12],
        ]
    )
    assert_equal(actual, expected)


@requires_scipy
def test_interpolators_complex_out_of_bounds():
    """Ensure complex nans are used for complex data"""

    xi = np.array([-1, 0, 1, 2, 5], dtype=np.float64)
    yi = np.exp(1j * xi)
    x = np.array([-2, 1, 6], dtype=np.float64)

    expected = np.array(
        [np.nan + np.nan * 1j, np.exp(1j), np.nan + np.nan * 1j], dtype=yi.dtype
    )

    for method, interpolator in [
        ("linear", NumpyInterpolator),
        ("linear", ScipyInterpolator),
    ]:
        f = interpolator(xi, yi, method=method)
        actual = f(x)
        assert_array_equal(actual, expected)


####Masking Functionality
@requires_bottleneck
def test_fill_gaps_limit():
    n = np.nan
    times = pd.date_range("2000-01-01", periods=8, freq="2h")
    coords = {"yt": ("y", times)}
    da = xr.DataArray([n, n, 2, n, n, 5, n, n], dims=["y"], coords=coords)

    mask = da.fill_gaps(dim="y", limit_direction="backward")
    with pytest.raises(
        ValueError,
        match=r"limit_direction='backward' is not allowed with ffill, must be 'forward'",
    ):
        mask.ffill()
    mask = da.fill_gaps(dim="y", limit_direction="both")
    with pytest.raises(
        ValueError,
        match=r"limit_direction='both' is not allowed with ffill, must be 'forward'",
    ):
        mask.ffill()
    mask = da.fill_gaps(dim="y", limit_direction="forward")
    with pytest.raises(
        ValueError,
        match=r"limit_direction='forward' is not allowed with bfill, must be 'backward'",
    ):
        mask.bfill()

    actual = da.fill_gaps(dim="y", limit=None).interpolate_na(
        dim="y", fill_value="extrapolate"
    )
    expected = da.copy(data=[0, 1, 2, 3, 4, 5, 6, 7])
    assert_equal(actual, expected)

    actual = da.fill_gaps(dim="y", limit=1).interpolate_na(
        dim="y", fill_value="extrapolate"
    )
    expected = da.copy(data=[n, 1, 2, 3, 4, 5, 6, n])
    assert_equal(actual, expected)

    actual = da.fill_gaps(
        dim="y",
        limit=pd.Timedelta("3h"),
        use_coordinate="yt",
    ).interpolate_na(dim="y", fill_value="extrapolate")
    expected = da.copy(data=[n, 1, 2, 3, 4, 5, 6, n])
    assert_equal(actual, expected)

    actual = da.fill_gaps(
        dim="y",
        limit=pd.Timedelta("3h"),
        limit_direction="backward",
        use_coordinate="yt",
    ).interpolate_na(dim="y", fill_value="extrapolate")
    expected = da.copy(data=[n, 1, 2, n, 4, 5, n, n])
    assert_equal(actual, expected)


@requires_bottleneck
def test_mask_gap_limit_2d():
    n = np.nan
    times = pd.date_range("2000-01-01", periods=12, freq="3h")
    coords = {
        "x": np.arange(3) * 2,
        "time": (times),
    }
    da = xr.DataArray(
        [
            [1, 2, 3, 4, n, 6, n, n, n, 10, 11, n],
            [n, n, 3, n, n, 6, n, n, n, 10, n, n],
            [n, 2, 3, 4, n, 6, n, n, n, 10, 11, n],
        ],
        coords=coords,
    )

    mask = da.fill_gaps("time", limit=1, use_coordinate=False)
    actual = mask.interpolate_na("time", fill_value="extrapolate")
    expected = da.copy(
        data=[
            [1, 2, 3, 4, 5, 6, 7, n, 9, 10, 11, 12],
            [n, 2, 3, 4, 5, 6, 7, n, 9, 10, 11, n],
            [1, 2, 3, 4, 5, 6, 7, n, 9, 10, 11, 12],
        ]
    )
    assert_equal(actual, expected)
    actual = mask.ffill()
    expected = da.copy(
        data=[
            [1, 2, 3, 4, 4, 6, 6, n, n, 10, 11, 11],
            [n, n, 3, 3, n, 6, 6, n, n, 10, 10, n],
            [n, 2, 3, 4, 4, 6, 6, n, n, 10, 11, 11],
        ]
    )
    assert_equal(actual, expected)
    actual = mask.bfill()
    expected = da.copy(
        data=[
            [1, 2, 3, 4, 6, 6, n, n, 10, 10, 11, n],
            [n, 3, 3, n, 6, 6, n, n, 10, 10, n, n],
            [2, 2, 3, 4, 6, 6, n, n, 10, 10, 11, n],
        ]
    )
    assert_equal(actual, expected)
    actual = mask.fillna(0)
    expected = da.copy(
        data=[
            [1, 2, 3, 4, 0, 6, 0, n, 0, 10, 11, 0],
            [n, 0, 3, 0, 0, 6, 0, n, 0, 10, 0, n],
            [0, 2, 3, 4, 0, 6, 0, n, 0, 10, 11, 0],
        ]
    )
    assert_equal(actual, expected)

    actual = da.fill_gaps(
        "time", limit=2, use_coordinate=False, limit_direction="backward"
    ).interpolate_na("time", fill_value="extrapolate")
    expected = da.copy(
        data=[
            [1, 2, 3, 4, 5, 6, n, 8, 9, 10, 11, n],
            [1, 2, 3, 4, 5, 6, n, 8, 9, 10, n, n],
            [1, 2, 3, 4, 5, 6, n, 8, 9, 10, 11, n],
        ]
    )
    assert_equal(actual, expected)

    actual = da.fill_gaps(
        "time",
        limit=pd.Timedelta("3h"),
        limit_direction="backward",
        limit_area="inside",
        use_coordinate=True,
    ).interpolate_na(
        "time",
        fill_value="extrapolate",
    )
    expected = da.copy(
        data=[
            [1, 2, 3, 4, 5, 6, n, n, 9, 10, 11, n],
            [n, n, 3, n, 5, 6, n, n, 9, 10, n, n],
            [n, 2, 3, 4, 5, 6, n, n, 9, 10, 11, n],
        ]
    )

    actual = da.fill_gaps(
        "time",
        limit=pd.Timedelta("3h"),
        limit_direction="backward",
        limit_area="outside",
        use_coordinate=True,
    ).interpolate_na(
        "time",
        fill_value="extrapolate",
    )
    expected = da.copy(
        data=[
            [1, 2, 3, 4, n, 6, n, n, n, 10, 11, n],
            [n, 2, 3, n, n, 6, n, n, n, 10, n, n],
            [1, 2, 3, 4, n, 6, n, n, n, 10, 11, n],
        ]
    )
    assert_equal(actual, expected)

    actual = da.fill_gaps(
        "time",
        limit=None,
        limit_direction="backward",
        limit_area="outside",
        use_coordinate=True,
    ).interpolate_na(
        "time",
        fill_value=8,
    )
    expected = da.copy(
        data=[
            [1, 2, 3, 4, n, 6, n, n, n, 10, 11, n],
            [8, 8, 3, n, n, 6, n, n, n, 10, n, n],
            [8, 2, 3, 4, n, 6, n, n, n, 10, 11, n],
        ]
    )
    assert_equal(actual, expected)

    da = xr.DataArray(
        [
            [1, 1, n, n, 1, 1],
            [n, 2, 2, n, 2, n],
            [n, n, 3, 3, n, n],
            [n, n, n, 4, 4, 4],
        ],
        dims=["x", "y"],
        coords={"x": np.arange(4) * 2},
    )
    mask = da.fill_gaps(
        dim="x",
        limit=3,
        limit_direction="forward",
        limit_area=None,
        use_coordinate=True,
    )
    actual = mask.interpolate_na(
        "x",
        fill_value="extrapolate",
        method="linear",
    )
    expected = da.copy(
        data=[
            [1, 1, n, n, 1, 1],
            [n, 2, 2, n, 2, 2],
            [n, 3, 3, 3, 3, n],
            [n, n, 4, 4, 4, 4],
        ]
    )
    assert_equal(actual, expected)
    # Test: Dim argument from mask should be used
    actual = mask.interpolate_na(
        fill_value="extrapolate",
        method="linear",
    )
    expected = da.copy(
        data=[
            [1, 1, n, n, 1, 1],
            [n, 2, 2, n, 2, 2],
            [n, 3, 3, 3, 3, n],
            [n, n, 4, 4, 4, 4],
        ]
    )
    assert_equal(actual, expected)


@requires_bottleneck
def test_mask_gap_max_gap_2d():
    n = np.nan
    times = pd.date_range("2000-01-01", periods=12, freq="3h")
    coords = {
        "x": np.arange(3) * 2,
        "time": (times),
    }
    da = xr.DataArray(
        [
            [1, 2, 3, 4, n, 6, n, n, n, 10, 11, n],
            [n, n, 3, n, n, 6, n, n, n, 10, n, n],
            [n, 2, 3, 4, n, 6, n, n, n, 10, 11, n],
        ],
        coords=coords,
    )

    mask = da.fill_gaps("time", max_gap=1, use_coordinate=False)
    actual = mask.interpolate_na("time", fill_value="extrapolate")
    expected = da.copy(
        data=[
            [1, 2, 3, 4, n, 6, n, n, n, 10, 11, 12],
            [n, n, 3, n, n, 6, n, n, n, 10, n, n],
            [1, 2, 3, 4, n, 6, n, n, n, 10, 11, 12],
        ]
    )
    assert_equal(actual, expected)
    mask = da.fill_gaps("time", max_gap=2, use_coordinate=False)
    actual = mask.interpolate_na("time", fill_value="extrapolate")
    expected = da.copy(
        data=[
            [1, 2, 3, 4, 5, 6, n, n, n, 10, 11, 12],
            [1, 2, 3, n, n, 6, n, n, n, 10, 11, 12],
            [1, 2, 3, 4, 5, 6, n, n, n, 10, 11, 12],
        ]
    )
    assert_equal(actual, expected)

    mask = da.fill_gaps("time", max_gap=pd.Timedelta("3h"), use_coordinate=True)
    actual = mask.interpolate_na("time", fill_value="extrapolate")
    expected = da.copy(
        data=[
            [1, 2, 3, 4, n, 6, n, n, n, 10, 11, 12],
            [n, n, 3, n, n, 6, n, n, n, 10, n, n],
            [1, 2, 3, 4, n, 6, n, n, n, 10, 11, 12],
        ]
    )
    assert_equal(actual, expected)


@requires_bottleneck
def test_mask_double_coordinate():
    # Check if limit and max_gap are able to handle string coordinate names
    n = np.nan
    da = xr.DataArray(
        [[1, n, n, 4, n, 6, 7], [1, n, n, n, 5, n, n]],
        dims=["x", "y"],
        coords={"y1": ("y", np.arange(7)), "y2": ("y", np.arange(7) * 2)},
    )
    actual = da.fill_gaps(
        "y",
        limit=1,
        max_gap=4,
        use_coordinate="y1",
    ).interpolate_na("y", fill_value="extrapolate")
    expected = da.copy(data=[[1, 2, 3, 4, 5, 6, 7], [1, 2, n, 4, 5, 6, n]])
    assert_equal(actual, expected)

    actual = da.fill_gaps("y", limit=2, max_gap=4, use_coordinate="y2").interpolate_na(
        "y",
        fill_value="extrapolate",
    )
    expected = da.copy(data=[[1, n, n, 4, 5, 6, 7], [1, n, n, n, 5, 6, n]])
    assert_equal(actual, expected)
