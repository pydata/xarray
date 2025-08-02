import jax.numpy as jnp

import xarray as xr

data = jnp.ones((2, 3))
da = xr.DataArray(data, dims=["x", "y"])
print(da)
