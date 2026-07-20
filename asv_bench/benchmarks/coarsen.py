import numpy as np

import xarray as xr

from . import randn

# Sizes chosen to test padding optimization
nx_padded = 4003  # Not divisible by 10 - requires padding
ny_padded = 4007  # Not divisible by 10 - requires padding

nx_exact = 4000  # Divisible by 10 - no padding needed
ny_exact = 4000  # Divisible by 10 - no padding needed

window = 10


class Coarsen:
    def setup(self, *args, **kwargs):
        # Case 1: Requires padding on both dimensions
        self.da_padded = xr.DataArray(
            randn((nx_padded, ny_padded)),
            dims=("x", "y"),
            coords={"x": np.arange(nx_padded), "y": np.arange(ny_padded)},
        )

        # Case 2: No padding required
        self.da_exact = xr.DataArray(
            randn((nx_exact, ny_exact)),
            dims=("x", "y"),
            coords={"x": np.arange(nx_exact), "y": np.arange(ny_exact)},
        )

    def time_coarsen_with_padding(self):
        """Coarsen 2D array where both dimensions require padding."""
        self.da_padded.coarsen(x=window, y=window, boundary="pad").mean()

    def time_coarsen_no_padding(self):
        """Coarsen 2D array where dimensions are exact multiples (no padding)."""
        self.da_exact.coarsen(x=window, y=window, boundary="pad").mean()

    def peakmem_coarsen_with_padding(self):
        """Peak memory for coarsening with padding on both dimensions."""
        self.da_padded.coarsen(x=window, y=window, boundary="pad").mean()

    def peakmem_coarsen_no_padding(self):
        """Peak memory for coarsening without padding."""
        self.da_exact.coarsen(x=window, y=window, boundary="pad").mean()
