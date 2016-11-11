'''
Useful for:

* users learning xarray
* building tutorials in the documentation.

'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os as _os

from .backends.api import open_dataset as _open_dataset
from .core.pycompat import urlretrieve as _urlretrieve


_default_cache_dir = _os.sep.join(('~', '.xarray_tutorial_data'))


# idea borrowed from Seaborn
def load_dataset(name, cache=True, cache_dir=_default_cache_dir,
                 github_url='https://github.com/pydata/xarray-data',  **kws):
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
    kws : dict, optional
        Passed to xarray.open_dataset

    """
    longdir = _os.path.expanduser(cache_dir)
    fullname = name + '.nc'
    localfile = _os.sep.join((longdir, fullname))

    if not _os.path.exists(localfile):

        # This will always leave this directory on disk.
        # May want to add an option to remove it.
        if not _os.path.isdir(longdir):
            _os.mkdir(longdir)

        url = '/'.join((github_url, 'raw', 'master', fullname))
        _urlretrieve(url, localfile)

    ds = _open_dataset(localfile, **kws).load()

    if not cache:
        _os.remove(localfile)

    return ds
