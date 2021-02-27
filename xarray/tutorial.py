"""
Useful for:

* users learning xarray
* building tutorials in the documentation.

"""
import hashlib
import os as _os
from urllib.request import urlretrieve

import numpy as np

from .backends.api import open_dataset as _open_dataset
from .core.dataarray import DataArray
from .core.dataset import Dataset

_default_cache_dir = _os.sep.join(("~", ".xarray_tutorial_data"))


def file_md5_checksum(fname):
    hash_md5 = hashlib.md5()
    with open(fname, "rb") as f:
        hash_md5.update(f.read())
    return hash_md5.hexdigest()


# idea borrowed from Seaborn
def open_dataset(
    name,
    cache=True,
    cache_dir=_default_cache_dir,
    github_url="https://github.com/pydata/xarray-data",
    branch="master",
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
    cache_dir : str, optional
        The directory in which to search for and write cached data.
    cache : bool, optional
        If True, then cache data locally for use on subsequent calls
    github_url : str
        Github repository where the data is stored
    branch : str
        The git branch to download from
    kws : dict, optional
        Passed to xarray.open_dataset

    See Also
    --------
    xarray.open_dataset

    """
    root, ext = _os.path.splitext(name)
    if not ext:
        ext = ".nc"
    fullname = root + ext
    longdir = _os.path.expanduser(cache_dir)
    localfile = _os.sep.join((longdir, fullname))
    md5name = fullname + ".md5"
    md5file = _os.sep.join((longdir, md5name))

    if not _os.path.exists(localfile):

        # This will always leave this directory on disk.
        # May want to add an option to remove it.
        if not _os.path.isdir(longdir):
            _os.mkdir(longdir)

        url = "/".join((github_url, "raw", branch, fullname))
        urlretrieve(url, localfile)
        url = "/".join((github_url, "raw", branch, md5name))
        urlretrieve(url, md5file)

        localmd5 = file_md5_checksum(localfile)
        with open(md5file) as f:
            remotemd5 = f.read()
        if localmd5 != remotemd5:
            _os.remove(localfile)
            msg = """
            MD5 checksum does not match, try downloading dataset again.
            """
            raise OSError(msg)

    ds = _open_dataset(localfile, **kws)

    if not cache:
        ds = ds.load()
        _os.remove(localfile)

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
