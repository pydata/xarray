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


@pytest.fixture
def sample_data():
    from collections import namedtuple

    x = (np.arange(10) + 1) * pq.A
    y = (np.arange(20) + 1)
    xp = (np.arange(10) + 1) * pq.J
    v = (np.arange(10 * 20).reshape(10, 20) + 1) * pq.V
    da = DataArray(
        v,
        dims=('x', 'y'),
        coords=dict(x=x, y=y, xp=(['x'], xp)),
    )

    data = namedtuple(
        'SampleData',
        ['x', 'y', 'xp', 'v', 'da'],
    )

    return data(x=x, y=y, xp=xp, v=v, da=da)


def equal_with_units(a, b):
    a = a if not isinstance(a, DataArray) else a.data
    b = b if not isinstance(b, DataArray) else b.data

    return (
        (hasattr(a, "units") and hasattr(b, "units"))
        and a.units == b.units
        and np.allclose(a.magnitude, b.magnitude)
    )


@requires_quantities
class TestWithQuantities(object):
    def test_units_in_data_and_coords(self, sample_data):
        assert equal_with_units(sample_data.da.xp.data, sample_data.xp)

    def test_arithmetics(self, sample_data):
        x = sample_data.x
        y = sample_data.y
        v = sample_data.v
        da = sample_data.da

        f = np.arange(10 * 20).reshape(10, 20) * pq.A
        g = DataArray(f, dims=['x', 'y'], coords=dict(x=x, y=y))
        assert equal_with_units(da * g, v * f)

        # swapped dimension order
        f = np.arange(20 * 10).reshape(20, 10) * pq.V
        g = DataArray(f, dims=['y', 'x'], coords=dict(x=x, y=y))
        assert equal_with_units(da + g, v + f.T)

        # broadcasting
        f = np.arange(10) * pq.m
        g = DataArray(f, dims=['x'], coords=dict(x=x))
        assert equal_with_units(da / g, v / f[:, None])
        
    def test_unit_checking(self, sample_data):
        da = sample_data.da
        x = sample_data.x
        y = sample_data.y
        f = np.arange(10 * 20).reshape(10, 20) * pq.A
        g = DataArray(f, dims=['x', 'y'], coords=dict(x=x, y=y))
        with pytest.raises(ValueError,
                           match="Unable to convert between units"):
            da + g

    @pytest.mark.xfail(reason="units in indexes not supported")
    def test_units_in_indexes(self, sample_data):
        """ Test if units survive through xarray indexes.
        Indexes are borrowed from Pandas, and Pandas does not support
        units. Therefore, we currently don't inted to support units on
        indexes either.
        """
        x = sample_data.x
        da = sample_data.da
        assert equal_with_units(da.x, x)

    @pytest.mark.xfail
    def test_sel(self, sample_data):
        y = sample_data.y
        v = sample_data.v
        da = sample_data.da
        assert equal_with_units(da.sel(y=y[0]), v[:, 0])

    @pytest.mark.xfail
    def test_mean(self, sample_data):
        da = sample_data.da
        v = sample_data.v
        assert equal_with_units(da.mean('x'), v.mean(0))
