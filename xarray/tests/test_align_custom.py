import xarray as xr


def test_align_basic_case():
    a = xr.DataArray([10, 20], dims="x", coords={"x": [0, 1]})
    b = xr.DataArray([30, 40], dims="x", coords={"x": [1, 2]})
    a_aligned, b_aligned = xr.align(a, b, join="outer", fill_value=-1)
    assert list(a_aligned.values) == [10, 20, -1]
    assert list(b_aligned.values) == [-1, 30, 40]


def test_align_inner_case():
    a = xr.DataArray([10, 20], dims="x", coords={"x": [0, 1]})
    b = xr.DataArray([30, 40], dims="x", coords={"x": [1, 2]})
    a_aligned, b_aligned = xr.align(a, b, join="inner")
    assert list(a_aligned.values) == [20]
    assert list(b_aligned.values) == [30]


def test_align_left_case():
    a = xr.DataArray([10, 20], dims="x", coords={"x": [0, 1]})
    b = xr.DataArray([30, 40], dims="x", coords={"x": [1, 2]})
    a_aligned, b_aligned = xr.align(a, b, join="left", fill_value=999)
    assert list(a_aligned.values) == [10, 20]
    assert list(b_aligned.values) == [999, 30]


def test_align_right_case():
    a = xr.DataArray([10, 20], dims="x", coords={"x": [0, 1]})
    b = xr.DataArray([30, 40], dims="x", coords={"x": [1, 2]})
    a_aligned, b_aligned = xr.align(a, b, join="right", fill_value=888)
    assert list(a_aligned.values) == [20, 888]
    assert list(b_aligned.values) == [30, 40]
