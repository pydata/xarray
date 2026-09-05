"""
Property-based tests for round-tripping data to netCDF
"""
import hypothesis.extra.numpy as npst
import hypothesis.strategies as st
from hypothesis import given

import xarray as xr


an_array = npst.arrays(
    dtype=st.one_of(
        npst.unsigned_integer_dtypes(),
        npst.integer_dtypes(),
        # NetCDF does not support float16
        # https://www.unidata.ucar.edu/software/netcdf/docs/data_type.html
        npst.floating_dtypes(sizes=(32, 64)),
        npst.byte_string_dtypes(),
        npst.unicode_string_dtypes(),
        npst.datetime64_dtypes(),
        npst.timedelta64_dtypes(),
    ),
    shape=npst.array_shapes(max_side=3),  # max_side specified for performance
)

compatible_names = st.text(
    alphabet=st.characters(
        # Limit characters to upper & lowercase letters and decimal digits
        whitelist_categories=("Ll", "Lu", "Nd"),
        # It looks like netCDF should allow unicode names, but removing
        # this causes a failure with 'ά'
        # https://www.unidata.ucar.edu/software/netcdf/docs/netcdf_data_set_components.html#Permitted
        max_codepoint=255,
    ),
    min_size=1,
)


@given(st.data(), an_array)
def test_netcdf_roundtrip(tmp_path, data, arr):
    names = data.draw(
        st.lists(
            compatible_names, min_size=arr.ndim, max_size=arr.ndim, unique=True
        ).map(tuple)
    )
    var = xr.Variable(names, arr)
    original = xr.Dataset({"data": var})
    original.to_netcdf(tmp_path / "test.nc")

    with xr.open_dataset(tmp_path / "test.nc") as roundtripped:
        xr.testing.assert_identical(original, roundtripped)
