"""
Property-based tests for round-tripping data to netCDF
"""
import hypothesis.extra.numpy as npst
import hypothesis.strategies as st
from hypothesis import given, settings

import xarray as xr

# Run for a while - arrays are a bigger search space than usual
settings.register_profile("ci", deadline=None)
settings.load_profile("ci")


an_array = npst.arrays(
    dtype=st.one_of(
        npst.unsigned_integer_dtypes(), npst.integer_dtypes(),
        # NetCDF does not support float16
        # https://www.unidata.ucar.edu/software/netcdf/docs/data_type.html
        npst.floating_dtypes(sizes=(32, 64))
    ),
    shape=npst.array_shapes(max_side=3),  # max_side specified for performance
)

compatible_names = st.text(
    alphabet=st.characters(
        whitelist_categories=('Ll', 'Lu', 'Nd'),
        # It looks like netCDF should allow unicode names, but removing
        # this causes a failure with 'ά'
        max_codepoint=255
    ),
    min_size=1
)

@given(st.data(), an_array)
def test_netcdf_roundtrip(tmp_path, data, arr):
    names = data.draw(
        st.lists(compatible_names, min_size=arr.ndim, max_size=arr.ndim, unique=True).map(
            tuple
        )
    )
    var = xr.Variable(names, arr)
    original = xr.Dataset({'data': var})
    original.to_netcdf(tmp_path / 'test.nc')

    roundtripped = xr.open_dataset(tmp_path / 'test.nc')
    xr.testing.assert_identical(original, roundtripped)
