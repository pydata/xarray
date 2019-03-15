'''
Useful for:

* users learning xarray
* building tutorials in the documentation.

'''
import hashlib
import os as _os
import warnings
from urllib.request import urlretrieve

from .backends.api import open_dataset as _open_dataset

_default_cache_dir = _os.sep.join(('~', '.xarray_tutorial_data'))


def file_md5_checksum(fname):
    hash_md5 = hashlib.md5()
    with open(fname, "rb") as f:
        hash_md5.update(f.read())
    return hash_md5.hexdigest()


# idea borrowed from Seaborn
def open_dataset(name, cache=True, cache_dir=_default_cache_dir,
                 github_url='https://github.com/pydata/xarray-data',
                 branch='master', **kws):
    """
    Load a dataset from the online repository (requires internet).

    If a local copy is found then always use that to avoid network traffic.

    Parameters
    ----------
    name : str
        Name of the netcdf file containing the dataset
        ie. 'air_temperature'
    cache_dir : string, optional
        The directory in which to search for and write cached data.
    cache : boolean, optional
        If True, then cache data locally for use on subsequent calls
    github_url : string
        Github repository where the data is stored
    branch : string
        The git branch to download from
    kws : dict, optional
        Passed to xarray.open_dataset

    See Also
    --------
    xarray.open_dataset

    """
    longdir = _os.path.expanduser(cache_dir)
    fullname = name + '.nc'
    localfile = _os.sep.join((longdir, fullname))
    md5name = name + '.md5'
    md5file = _os.sep.join((longdir, md5name))

    if not _os.path.exists(localfile):

        # This will always leave this directory on disk.
        # May want to add an option to remove it.
        if not _os.path.isdir(longdir):
            _os.mkdir(longdir)

        url = '/'.join((github_url, 'raw', branch, fullname))
        urlretrieve(url, localfile)
        url = '/'.join((github_url, 'raw', branch, md5name))
        urlretrieve(url, md5file)

        localmd5 = file_md5_checksum(localfile)
        with open(md5file, 'r') as f:
            remotemd5 = f.read()
        if localmd5 != remotemd5:
            _os.remove(localfile)
            msg = """
            MD5 checksum does not match, try downloading dataset again.
            """
            raise IOError(msg)

    ds = _open_dataset(localfile, **kws)

    if not cache:
        ds = ds.load()
        _os.remove(localfile)

    return ds


def load_dataset(*args, **kwargs):
    """
    `load_dataset` will be removed a future version of xarray. The current
    behavior of this function can be achived by using
    `tutorial.open_dataset(...).load()`.

    See Also
    --------
    open_dataset
    """
    warnings.warn(
        "load_dataset` will be removed in a future version of xarray. The "
        "current behavior of this function can be achived by using "
        "`tutorial.open_dataset(...).load()`.",
        DeprecationWarning, stacklevel=2)
    return open_dataset(*args, **kwargs).load()
