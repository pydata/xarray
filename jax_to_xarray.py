import xarray as xr


def jax_to_xarray(jax_array, dims=None):
    """
    Convert a JAX array into an Xarray DataArray with optional dimension names.

    Parameters:
        jax_array (jax.numpy.ndarray): The JAX array to convert
        dims (list of str): Optional dimension names

    Returns:
        xarray.DataArray: The wrapped array
    """
    if dims is None:
        dims = [f"dim_{i}" for i in range(jax_array.ndim)]
    return xr.DataArray(jax_array, dims=dims)
