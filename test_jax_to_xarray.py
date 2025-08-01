from jax_to_xarray import jax_to_xarray
import jax.numpy as jnp

data = jnp.ones((2, 3))
da = jax_to_xarray(data, dims=["x", "y"])

print(da)
