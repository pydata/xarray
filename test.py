import xarray as xr
import jax.numpy as jnp

data = jnp.ones((2, 3))
da = xr.DataArray(data, dims=["x", "y"])
print(da)
