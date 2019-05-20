import numpy as np
import pytest

from xarray import DataArray, set_options

try:
    import quantities as pq
    has_quantities = True
except ImportError:
    has_quantities = False

pytestmark = pytest.mark.skipif(
    not has_quantities,
    reason="requires python-quantities",
)


set_options(enable_experimental_ndarray_subclass_support=True)


def assert_equal_with_units(a, b):
    a = a if not isinstance(a, DataArray) else a.data
    b = b if not isinstance(b, DataArray) else b.data

    assert (
        (hasattr(a, "units") and hasattr(b, "units"))
        and a.units == b.units
    )

    assert (
        (hasattr(a, "magnitude") and hasattr(b, "magnitude"))
        and np.allclose(a.magnitude, b.magnitude)
    )


def create_data():
    return (np.arange(10 * 20).reshape(10, 20) + 1) * pq.V


def create_coord_arrays():
    x = (np.arange(10) + 1) * pq.A
    y = np.arange(20) + 1
    xp = (np.arange(10) + 1) * pq.J
    return x, y, xp


def create_coords():
    x, y, xp = create_coord_arrays()
    coords = dict(
        x=x,
        y=y,
        xp=(['x'], xp),
    )
    return coords


def create_data_array():
    data = create_data()
    coords = create_coords()
    return DataArray(
        data,
        dims=('x', 'y'),
        coords=coords,
    )


def with_keys(mapping, keys):
    return {
        key: value
        for key, value in mapping.items()
        if key in keys
    }


def test_without_subclass_support():
    with set_options(enable_experimental_ndarray_subclass_support=False):
        data_array = create_data_array()
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
    data_array = create_data_array()

    assert_equal_with_units(data_array.data, data_array)
    assert_equal_with_units(data_array.xp.data, data_array.xp)


def test_arithmetics():
    v = create_data()
    coords = create_coords()
    da = create_data_array()

    f = np.arange(10 * 20).reshape(10, 20) * pq.A
    g = DataArray(f, dims=['x', 'y'], coords=with_keys(coords, ['x', 'y']))
    assert_equal_with_units(da * g, v * f)

    # swapped dimension order
    f = np.arange(20 * 10).reshape(20, 10) * pq.V
    g = DataArray(f, dims=['y', 'x'], coords=with_keys(coords, ['x', 'y']))
    assert_equal_with_units(da + g, v + f.T)

    # broadcasting
    f = (np.arange(10) + 1) * pq.m
    g = DataArray(f, dims=['x'], coords=with_keys(coords, ['x']))
    assert_equal_with_units(da / g, v / f[:, None])


@pytest.mark.xfail(reason="units don't survive through combining yet")
def test_combine():
    from xarray import concat

    data_array = create_data_array()

    a = data_array[:, :10]
    b = data_array[:, 10:]

    assert_equal_with_units(concat([a, b], dim='y'), data_array)


def test_unit_checking():
    coords = create_coords()
    da = create_data_array()

    f = np.arange(10 * 20).reshape(10, 20) * pq.A
    g = DataArray(f, dims=['x', 'y'], coords=with_keys(coords, ['x', 'y']))
    with pytest.raises(ValueError,
                       match="Unable to convert between units"):
        da + g


@pytest.mark.xfail(reason="units in indexes not supported")
def test_units_in_indexes():
    """ Test if units survive through xarray indexes.
    Indexes are borrowed from Pandas, and Pandas does not support
    units. Therefore, we currently don't intend to support units on
    indexes either.
    """
    x, *_ = create_coord_arrays()
    data_array = create_data_array()
    assert_equal_with_units(data_array.x, x)


def test_sel():
    data = create_data()
    _, y, _ = create_coord_arrays()
    data_array = create_data_array()
    assert_equal_with_units(data_array.sel(y=y[0]), data[:, 0])


def test_mean():
    data = create_data()
    data_array = create_data_array()
    assert_equal_with_units(data_array.mean('x'), data.mean(0))
