from __future__ import annotations

import warnings

import numpy as np
import pytest

import xarray as xr
from xarray.tests import has_dask

try:
    from dask.array import from_array as dask_from_array
except ImportError:
    dask_from_array = lambda x: x  # type: ignore[assignment, misc]

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
        pytest.param(
            xr.DataArray(np.array("a", dtype="|S1")),
            xr.DataArray(np.array("b", dtype="|S1")),
            id="DataArray_with_character_dtype",
        ),
        pytest.param(
            xr.Coordinates({"x": [1e-17, 2]}),
            xr.Coordinates({"x": [0, 3]}),
            id="Coordinates",
        ),
    ),
)
def test_assert_allclose(obj1, obj2) -> None:
    with pytest.raises(AssertionError):
        xr.testing.assert_allclose(obj1, obj2)
    with pytest.raises(AssertionError):
        xr.testing.assert_allclose(obj1, obj2, check_dim_order=False)


@pytest.mark.parametrize("func", ["assert_equal", "assert_allclose"])
def test_assert_allclose_equal_transpose(func) -> None:
    """Transposed DataArray raises assertion unless check_dim_order=False."""
    obj1 = xr.DataArray([[0, 1, 2], [2, 3, 4]], dims=["a", "b"])
    obj2 = xr.DataArray([[0, 2], [1, 3], [2, 4]], dims=["b", "a"])
    with pytest.raises(AssertionError):
        getattr(xr.testing, func)(obj1, obj2)
    getattr(xr.testing, func)(obj1, obj2, check_dim_order=False)
    ds1 = obj1.to_dataset(name="varname")
    ds1["var2"] = obj1
    ds2 = obj1.to_dataset(name="varname")
    ds2["var2"] = obj1.transpose()
    with pytest.raises(AssertionError):
        getattr(xr.testing, func)(ds1, ds2)
    getattr(xr.testing, func)(ds1, ds2, check_dim_order=False)


def test_assert_equal_transpose_datatree() -> None:
    """Ensure `check_dim_order=False` works for transposed DataTree"""
    ds = xr.Dataset(data_vars={"data": (("x", "y"), [[1, 2]])})

    a = xr.DataTree.from_dict({"node": ds})
    b = xr.DataTree.from_dict({"node": ds.transpose("y", "x")})

    with pytest.raises(AssertionError):
        xr.testing.assert_equal(a, b)

    xr.testing.assert_equal(a, b, check_dim_order=False)

    # Test with mixed dimension orders in datasets (the tricky case)
    import numpy as np

    ds_mixed = xr.Dataset(
        {
            "foo": xr.DataArray(np.zeros([4, 5]), dims=("a", "b")),
            "bar": xr.DataArray(np.ones([5, 4]), dims=("b", "a")),
        }
    )
    ds_mixed2 = xr.Dataset(
        {
            "foo": xr.DataArray(np.zeros([5, 4]), dims=("b", "a")),
            "bar": xr.DataArray(np.ones([4, 5]), dims=("a", "b")),
        }
    )

    tree1 = xr.DataTree.from_dict({"node": ds_mixed})
    tree2 = xr.DataTree.from_dict({"node": ds_mixed2})

    # Should work with check_dim_order=False
    xr.testing.assert_equal(tree1, tree2, check_dim_order=False)


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
            warnings.warn("warning in test", stacklevel=2)
            return super().dims

        def __array__(
            self,
            dtype: np.typing.DTypeLike | None = None,
            /,
            *,
            copy: bool | None = None,
        ) -> np.ndarray:
            warnings.warn("warning in test", stacklevel=2)
            return super().__array__(dtype, copy=copy)

    a = WarningVariable("x", [1])
    b = WarningVariable("x", [2])

    with warnings.catch_warnings(record=True) as w:
        # elevate warnings to errors
        warnings.filterwarnings("error")
        with pytest.raises(AssertionError):
            getattr(xr.testing, func)(a, b)

        assert len(w) > 0

        # ensure warnings still raise outside of assert_*
        with pytest.raises(UserWarning):
            warnings.warn("test", stacklevel=2)

    # ensure warnings stay ignored in assert_*
    with warnings.catch_warnings(record=True) as w:
        # ignore warnings
        warnings.filterwarnings("ignore")
        with pytest.raises(AssertionError):
            getattr(xr.testing, func)(a, b)

        assert len(w) == 0


def test_assert_equal_dataset_check_dim_order():
    """Test for issue #10704 - check_dim_order=False with Datasets containing mixed dimension orders."""
    import numpy as np

    # Dataset with variables having different dimension orders
    dataset_1 = xr.Dataset(
        {
            "foo": xr.DataArray(np.zeros([4, 5]), dims=("a", "b")),
            "bar": xr.DataArray(np.ones([5, 4]), dims=("b", "a")),
        }
    )

    dataset_2 = xr.Dataset(
        {
            "foo": xr.DataArray(np.zeros([5, 4]), dims=("b", "a")),
            "bar": xr.DataArray(np.ones([4, 5]), dims=("a", "b")),
        }
    )

    # These should be equal when ignoring dimension order
    xr.testing.assert_equal(dataset_1, dataset_2, check_dim_order=False)
    xr.testing.assert_allclose(dataset_1, dataset_2, check_dim_order=False)

    # Should also work when comparing dataset to itself
    xr.testing.assert_equal(dataset_1, dataset_1, check_dim_order=False)
    xr.testing.assert_allclose(dataset_1, dataset_1, check_dim_order=False)

    # But should fail with check_dim_order=True
    with pytest.raises(AssertionError):
        xr.testing.assert_equal(dataset_1, dataset_2, check_dim_order=True)
    with pytest.raises(AssertionError):
        xr.testing.assert_allclose(dataset_1, dataset_2, check_dim_order=True)

    # Test with non-sortable dimension names (int and str)
    dataset_mixed_1 = xr.Dataset(
        {
            "foo": xr.DataArray(np.zeros([4, 5]), dims=(1, "b")),
            "bar": xr.DataArray(np.ones([5, 4]), dims=("b", 1)),
        }
    )

    dataset_mixed_2 = xr.Dataset(
        {
            "foo": xr.DataArray(np.zeros([5, 4]), dims=("b", 1)),
            "bar": xr.DataArray(np.ones([4, 5]), dims=(1, "b")),
        }
    )

    # Should work with mixed types when ignoring dimension order
    xr.testing.assert_equal(dataset_mixed_1, dataset_mixed_2, check_dim_order=False)
    xr.testing.assert_equal(dataset_mixed_1, dataset_mixed_1, check_dim_order=False)


def test_assert_equal_no_common_dims():
    """Test assert_equal when objects have no common dimensions."""
    import numpy as np

    # DataArrays with completely different dimensions
    da1 = xr.DataArray(np.zeros([4, 5]), dims=("x", "y"))
    da2 = xr.DataArray(np.zeros([3, 2]), dims=("a", "b"))

    # Should fail even with check_dim_order=False since dims are different
    with pytest.raises(AssertionError):
        xr.testing.assert_equal(da1, da2, check_dim_order=False)

    # Datasets with no common dimensions
    ds1 = xr.Dataset(
        {
            "foo": xr.DataArray(np.zeros([4]), dims=("x",)),
            "bar": xr.DataArray(np.ones([5]), dims=("y",)),
        }
    )
    ds2 = xr.Dataset(
        {
            "foo": xr.DataArray(np.zeros([3]), dims=("a",)),
            "bar": xr.DataArray(np.ones([2]), dims=("b",)),
        }
    )

    # Should fail since dimensions are completely different
    with pytest.raises(AssertionError):
        xr.testing.assert_equal(ds1, ds2, check_dim_order=False)


def test_assert_equal_variable_transpose():
    """Test assert_equal with transposed Variable objects."""
    import numpy as np

    # Variables with transposed dimensions
    var1 = xr.Variable(("x", "y"), np.zeros([4, 5]))
    var2 = xr.Variable(("y", "x"), np.zeros([5, 4]))

    # Should fail with check_dim_order=True
    with pytest.raises(AssertionError):
        xr.testing.assert_equal(var1, var2, check_dim_order=True)

    # Should pass with check_dim_order=False
    xr.testing.assert_equal(var1, var2, check_dim_order=False)
    xr.testing.assert_allclose(var1, var2, check_dim_order=False)
