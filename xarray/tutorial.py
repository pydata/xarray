"""
Useful for:

* users learning xarray
* building tutorials in the documentation.

"""
import os
import pathlib

import numpy as np

from .backends.api import open_dataset as _open_dataset
from .core.dataarray import DataArray
from .core.dataset import Dataset

_default_cache_dir_name = "xarray_tutorial_data"
base_url = "https://github.com/pydata/xarray-data"
version = "master"

external_urls = {
    "RGB.byte": "https://github.com/mapbox/rasterio/raw/master/tests/data/RGB.byte.tif",
}


# idea borrowed from Seaborn
def open_dataset(
    name,
    cache=True,
    cache_dir=None,
    **kws,
):
    """
    Open a dataset from the online repository (requires internet).

    If a local copy is found then always use that to avoid network traffic.

    Parameters
    ----------
    name : str
        Name of the file containing the dataset. If no suffix is given, assumed
        to be netCDF ('.nc' is appended)
        e.g. 'air_temperature'
    cache_dir : path-like, optional
        The directory in which to search for and write cached data.
    cache : bool, optional
        If True, then cache data locally for use on subsequent calls
    kws : dict, optional
        Passed to xarray.open_dataset

    See Also
    --------
    xarray.open_dataset
    xarray.tutorial.open_rasterio

    """
    try:
        import pooch
    except ImportError:
        raise ImportError("using the tutorial data requires pooch")

    if isinstance(cache_dir, pathlib.Path):
        cache_dir = os.fspath(cache_dir)
    elif cache_dir is None:
        cache_dir = pooch.os_cache(_default_cache_dir_name)

    # process the name
    default_extension = ".nc"
    path = pathlib.Path(name)
    if not path.suffix:
        path = path.with_suffix(default_extension)

    # retrieve the file
    filepath = pooch.retrieve(
        url=f"{base_url}/raw/{version}/{path.name}",
        known_hash=None,
        path=cache_dir,
    )
    ds = _open_dataset(filepath, **kws)
    if not cache:
        ds = ds.load()
        pathlib.Path(filepath).unlink()

    return ds


def load_dataset(*args, **kwargs):
    """
    Open, load into memory, and close a dataset from the online repository
    (requires internet).

    See Also
    --------
    open_dataset
    """
    with open_dataset(*args, **kwargs) as ds:
        return ds.load()


def scatter_example_dataset():
    A = DataArray(
        np.zeros([3, 11, 4, 4]),
        dims=["x", "y", "z", "w"],
        coords=[
            np.arange(3),
            np.linspace(0, 1, 11),
            np.arange(4),
            0.1 * np.random.randn(4),
        ],
    )
    B = 0.1 * A.x ** 2 + A.y ** 2.5 + 0.1 * A.z * A.w
    A = -0.1 * A.x + A.y / (5 + A.z) + A.w
    ds = Dataset({"A": A, "B": B})
    ds["w"] = ["one", "two", "three", "five"]

    ds.x.attrs["units"] = "xunits"
    ds.y.attrs["units"] = "yunits"
    ds.z.attrs["units"] = "zunits"
    ds.w.attrs["units"] = "wunits"

    ds.A.attrs["units"] = "Aunits"
    ds.B.attrs["units"] = "Bunits"

    return ds
