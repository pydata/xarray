import numpy as np
import pytest

from xarray import DataArray, set_options

try:
    import quantities as pq

    has_quantities = True
except ImportError:
    has_quantities = False

pytestmark = pytest.mark.skipif(not has_quantities, reason="requires python-quantities")


set_options(enable_experimental_ndarray_subclass_support=True)


def assert_equal_with_units(a, b):
    a = a if not isinstance(a, DataArray) else a.data
    b = b if not isinstance(b, DataArray) else b.data

    assert (hasattr(a, "units") and hasattr(b, "units")) and a.units == b.units

    assert (hasattr(a, "magnitude") and hasattr(b, "magnitude")) and np.allclose(
        a.magnitude, b.magnitude
    )


def test_without_subclass_support():
    with set_options(enable_experimental_ndarray_subclass_support=False):
        data_array = DataArray(data=np.arange(10) * pq.m)
        assert not hasattr(data_array.data, "units")


@pytest.mark.filterwarnings("ignore:the matrix subclass:PendingDeprecationWarning")
def test_matrix():
    matrix = np.matrix([[1, 2], [3, 4]])
    da = DataArray(matrix)

    assert not isinstance(da.data, np.matrix)


def test_masked_array():
    masked = np.ma.array([[1, 2], [3, 4]], mask=[[0, 1], [1, 0]])
    da = DataArray(masked)
    assert not isinstance(da.data, np.ma.MaskedArray)


def test_units_in_data_and_coords():
    data = np.arange(10) * pq.m
    x = np.arange(10) * pq.s
    xp = x.rescale(pq.ms)
    data_array = DataArray(data=data, coords={"x": x, "xp": ("x", xp)}, dims=["x"])

    assert_equal_with_units(data, data_array)
    assert_equal_with_units(xp, data_array.xp)


def test_arithmetics():
    x = np.arange(10)
    y = np.arange(20)

    f = (np.arange(10 * 20).reshape(10, 20) + 1) * pq.V
    g = np.arange(10 * 20).reshape(10, 20) * pq.A

    a = DataArray(data=f, coords={"x": x, "y": y}, dims=("x", "y"))
    b = DataArray(data=g, coords={"x": x, "y": y}, dims=("x", "y"))

    assert_equal_with_units(a * b, f * g)

    # swapped dimension order
    g = np.arange(20 * 10).reshape(20, 10) * pq.V
    b = DataArray(data=g, coords={'x': x, 'y': y}, dims=("y", "x"))
    assert_equal_with_units(a + b, f + g.T)

    # broadcasting
    g = (np.arange(10) + 1) * pq.m
    b = DataArray(data=g, coords={'x': x}, dims=["x"])
    assert_equal_with_units(a / b, f / g[:, None])


@pytest.mark.xfail(reason="units don't survive through combining yet")
def test_combine():
    from xarray import concat

    data = (np.arange(15) + 10) * pq.m
    y = np.arange(len(data))

    data_array = DataArray(data=data, coords={"y": y}, dims=["y"])
    a = data_array[:5]
    b = data_array[5:]

    assert_equal_with_units(concat([a, b], dim="y"), data_array)


def test_unit_checking():
    coords = {"x": np.arange(10), "y": np.arange(20)}

    f = np.arange(10 * 20).reshape(10, 20) * pq.A
    g = np.arange(10 * 20).reshape(10, 20) * pq.V

    a = DataArray(f, coords=coords, dims=("x", "y"))
    b = DataArray(g, coords=coords, dims=("x", "y"))
    with pytest.raises(ValueError, match="Unable to convert between units"):
        a + b


@pytest.mark.xfail(reason="units in indexes not supported")
def test_units_in_indexes():
    """ Test if units survive through xarray indexes.
    Indexes are borrowed from Pandas, and Pandas does not support
    units. Therefore, we currently don't intend to support units on
    indexes either.
    """
    data = np.arange(15) * 10
    x = np.arange(len(data)) * pq.A

    data_array = DataArray(data=data, coords={"x": x}, dims=["x"])
    assert_equal_with_units(data_array.x, x)


def test_sel():
    data = np.arange(10 * 20).reshape(10, 20) * pq.m / pq.s
    x = np.arange(data.shape[0]) * pq.m
    y = np.arange(data.shape[1]) * pq.s

    data_array = DataArray(data=data, coords={"x": x, "y": y}, dims=("x", "y"))
    assert_equal_with_units(data_array.sel(y=y[0]), data[:, 0])


def test_mean():
    data = np.arange(10 * 20).reshape(10, 20) * pq.V
    x = np.arange(data.shape[0])
    y = np.arange(data.shape[1])

    data_array = DataArray(data=data, coords={"x": x, "y": y}, dims=("x", "y"))
    assert_equal_with_units(data_array.mean("x"), data.mean(0))
