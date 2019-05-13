import numpy as np

from xarray import DataArray

import pytest


try:
    import quantities as pq
    has_quantities = True
except ImportError:
    has_quantities = False


def requires_quantities(test):
    return (
        test
        if has_quantities
        else pytest.mark.skip('requires python-quantities')(test)
    )


def equal_with_units(a, b):
    a = a if not isinstance(a, DataArray) else a.data
    b = b if not isinstance(b, DataArray) else b.data

    return (
        (hasattr(a, "units") and hasattr(b, "units"))
        and a.units == b.units
        and np.allclose(a.magnitude, b.magnitude)
    )


@pytest.fixture
def data():
    return (np.arange(10 * 20).reshape(10, 20) + 1) * pq.V


@pytest.fixture
def coords():
    return {
        'x': (np.arange(10) + 1) * pq.A,
        'y': np.arange(20) + 1,
        'xp': (np.arange(10) + 1) * pq.J,
    }


@pytest.fixture
def data_array(data, coords):
    coords['xp'] = (['x'], coords['xp'])
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


@requires_quantities
class TestWithQuantities(object):
    def test_units_in_data_and_coords(self, data_array):
        assert equal_with_units(data_array.xp.data, data_array.xp)

    def test_arithmetics(self, data_array, data, coords):
        v = data
        da = data_array

        f = np.arange(10 * 20).reshape(10, 20) * pq.A
        g = DataArray(f, dims=['x', 'y'], coords=with_keys(coords, ['x', 'y']))
        assert equal_with_units(da * g, v * f)

        # swapped dimension order
        f = np.arange(20 * 10).reshape(20, 10) * pq.V
        g = DataArray(f, dims=['y', 'x'], coords=with_keys(coords, ['x', 'y']))
        assert equal_with_units(da + g, v + f.T)

        # broadcasting
        f = (np.arange(10) + 1) * pq.m
        g = DataArray(f, dims=['x'], coords=with_keys(coords, ['x']))
        assert equal_with_units(da / g, v / f[:, None])

    @pytest.mark.xfail(reason="units don't survive through combining yet")
    def test_combine(self, data_array):
        from xarray import concat

        a = data_array[:, :10]
        b = data_array[:, 10:]

        assert equal_with_units(concat([a, b], dim='y'), data_array)

    def test_unit_checking(self, data_array, coords):
        da = data_array

        f = np.arange(10 * 20).reshape(10, 20) * pq.A
        g = DataArray(f, dims=['x', 'y'], coords=with_keys(coords, ['x', 'y']))
        with pytest.raises(ValueError,
                           match="Unable to convert between units"):
            da + g

    @pytest.mark.xfail(reason="units in indexes not supported")
    def test_units_in_indexes(self, data_array, coords):
        """ Test if units survive through xarray indexes.
        Indexes are borrowed from Pandas, and Pandas does not support
        units. Therefore, we currently don't intend to support units on
        indexes either.
        """
        assert equal_with_units(data_array.x, coords['x'])

    def test_sel(self, data_array, coords, data):
        assert equal_with_units(data_array.sel(y=coords['y'][0]), data[:, 0])

    @pytest.mark.xfail
    def test_mean(self, data_array, data):
        assert equal_with_units(data_array.mean('x'), data.mean(0))
