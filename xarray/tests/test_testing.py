import warnings

import numpy as np
import pytest

import xarray as xr

from . import has_dask

try:
    from dask.array import from_array as dask_from_array
except ImportError:
    dask_from_array = lambda x: x

try:
    import pint

    unit_registry = pint.UnitRegistry(force_ndarray_like=True)

    def quantity(x):
        return unit_registry.Quantity(x, "m")

    has_pint = True
except ImportError:

    def quantity(x):
        return x

    has_pint = False


def test_allclose_regression() -> None:
    x = xr.DataArray(1.01)
    y = xr.DataArray(1.02)
    xr.testing.assert_allclose(x, y, atol=0.01)


@pytest.mark.parametrize(
    "obj1,obj2",
    (
        pytest.param(
            xr.Variable("x", [1e-17, 2]), xr.Variable("x", [0, 3]), id="Variable"
        ),
        pytest.param(
            xr.DataArray([1e-17, 2], dims="x"),
            xr.DataArray([0, 3], dims="x"),
            id="DataArray",
        ),
        pytest.param(
            xr.Dataset({"a": ("x", [1e-17, 2]), "b": ("y", [-2e-18, 2])}),
            xr.Dataset({"a": ("x", [0, 2]), "b": ("y", [0, 1])}),
            id="Dataset",
        ),
    ),
)
def test_assert_allclose(obj1, obj2) -> None:
    with pytest.raises(AssertionError):
        xr.testing.assert_allclose(obj1, obj2)


@pytest.mark.filterwarnings("error")
@pytest.mark.parametrize(
    "duckarray",
    (
        pytest.param(np.array, id="numpy"),
        pytest.param(
            dask_from_array,
            id="dask",
            marks=pytest.mark.skipif(not has_dask, reason="requires dask"),
        ),
        pytest.param(
            quantity,
            id="pint",
            marks=pytest.mark.skipif(not has_pint, reason="requires pint"),
        ),
    ),
)
@pytest.mark.parametrize(
    ["obj1", "obj2"],
    (
        pytest.param([1e-10, 2], [0.0, 2.0], id="both arrays"),
        pytest.param([1e-17, 2], 0.0, id="second scalar"),
        pytest.param(0.0, [1e-17, 2], id="first scalar"),
    ),
)
def test_assert_duckarray_equal_failing(duckarray, obj1, obj2) -> None:
    # TODO: actually check the repr
    a = duckarray(obj1)
    b = duckarray(obj2)
    with pytest.raises(AssertionError):
        xr.testing.assert_duckarray_equal(a, b)


@pytest.mark.filterwarnings("error")
@pytest.mark.parametrize(
    "duckarray",
    (
        pytest.param(
            np.array,
            id="numpy",
        ),
        pytest.param(
            dask_from_array,
            id="dask",
            marks=pytest.mark.skipif(not has_dask, reason="requires dask"),
        ),
        pytest.param(
            quantity,
            id="pint",
            marks=pytest.mark.skipif(not has_pint, reason="requires pint"),
        ),
    ),
)
@pytest.mark.parametrize(
    ["obj1", "obj2"],
    (
        pytest.param([0, 2], [0.0, 2.0], id="both arrays"),
        pytest.param([0, 0], 0.0, id="second scalar"),
        pytest.param(0.0, [0, 0], id="first scalar"),
    ),
)
def test_assert_duckarray_equal(duckarray, obj1, obj2) -> None:
    a = duckarray(obj1)
    b = duckarray(obj2)

    xr.testing.assert_duckarray_equal(a, b)


@pytest.mark.parametrize(
    "func",
    [
        "assert_equal",
        "assert_identical",
        "assert_allclose",
        "assert_duckarray_equal",
        "assert_duckarray_allclose",
    ],
)
def test_ensure_warnings_not_elevated(func) -> None:
    # make sure warnings are not elevated to errors in the assertion functions
    # e.g. by @pytest.mark.filterwarnings("error")
    # see https://github.com/pydata/xarray/pull/4760#issuecomment-774101639

    # define a custom Variable class that raises a warning in assert_*
    class WarningVariable(xr.Variable):
        @property  # type: ignore[misc]
        def dims(self):
            warnings.warn("warning in test")
            return super().dims

        def __array__(self):
            warnings.warn("warning in test")
            return super().__array__()

    a = WarningVariable("x", [1])
    b = WarningVariable("x", [2])

    with warnings.catch_warnings(record=True) as w:
        # elevate warnings to errors
        warnings.filterwarnings("error")
        with pytest.raises(AssertionError):
            getattr(xr.testing, func)(a, b)

        assert len(w) > 0
