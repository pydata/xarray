import numpy as np

import xarray as xr
import pytest

from xarray.backends.chunks import grid_rechunk


# TODO: Not sure if it would be good to add a test for the other functions inside the chunks module
#     at the end they are already being used internally by the grid_rechunk

@pytest.mark.parametrize(
    "enc_chunks, region, nd_var_chunks, expected_chunks",
    [
        ((3,), (slice(2, 14),), ((6, 6),), ((4, 6, 2,), )),
        ((6,), (slice(0, 13),), ((6, 7),), ((6, 7,),)),
        ((6,), (slice(0, 13),), ((6, 6, 1),), ((6, 6, 1),)),
        ((3,), (slice(2, 14),), ((1, 3, 2, 6),), ((1, 3, 6, 2),)),
        ((3,), (slice(2, 14),), ((2, 2, 2, 6),), ((4, 6, 2),)),
        ((3,), (slice(2, 14),), ((3, 1, 3, 5),), ((4, 3, 5),)),
        ((4,), (slice(1, 13),), ((1, 1, 1, 4, 3, 2),), ((3, 4, 4, 1),)),
        ((5,), (slice(4, 16),), ((5, 7),), ((6, 6),)),

        # ND cases
        ((3, 6), (slice(2, 14), slice(0, 13)), ((6, 6) ,(6, 7)), ((4, 6, 2,), (6, 7,))),
    ],
)
def test_grid_rechunk(enc_chunks, region, nd_var_chunks, expected_chunks):
    dims = [f"dim_{i}" for i in range(len(region))]
    coords = {dim: list(range(r.start, r.stop)) for dim, r in zip(dims, region)}
    shape = tuple(r.stop - r.start for r in region)
    arr = np.arange(np.prod(shape)).reshape(shape)
    arr = xr.DataArray(
        arr,
        dims=dims,
        coords=coords,
    )
    arr = arr.chunk(dict(zip(dims, nd_var_chunks)))

    result = grid_rechunk(
        arr.variable,
        enc_chunks=enc_chunks,
        region=region,
    )
    assert result.chunks == expected_chunks
