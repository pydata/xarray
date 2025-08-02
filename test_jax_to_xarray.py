import jax.numpy as jnp
import xarray as xr

data = jnp.ones((2, 3))
da = xr.DataArray(jnp.asarray(data), dims=["x", "y"])  # Optional: jnp.asarray

print(da)
