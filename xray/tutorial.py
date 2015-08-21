'''
Useful for:

* users learning xray
* building tutorials in the documentation.

'''

import os as _os

from .backends.api import open_dataset as _open_dataset
from .core.pycompat import urlretrieve as _urlretrieve


# Borrowed from Seaborn
def load_dataset(name, cache=True, cache_dir='.xray_tutorial_data',
                 github_url='https://github.com/xray/xray-data',  **kws):
    """
    Load a dataset from the online repository (requires internet).

    If a local copy is found then always use that to avoid network traffic.

    Parameters
    ----------
    name : str, optional
        Name of the netcdf file containing the dataset
        ie. 'air_temperature'
    cache_dir : string, optional
        The directory in which to search for and write cached data.
        Relative to user's home directory
    cache : boolean, optional
        If True, then cache data locally for use on subsequent calls
    github_url : string
        Github repository where the data is stored
    kws : dict, optional
        Passed to xray.open_dataset

    """
    longdir = _os.path.expanduser(_os.sep.join(('~', cache_dir)))
    localfile = _os.sep.join((longdir, name + '.nc'))

    if not _os.path.exists(localfile):

        # This will always leave this directory on disk.
        # May want to add an option to remove it.
        if not _os.path.isdir(longdir):
            _os.mkdir(longdir)

        url = '/'.join((github_url, 'raw', 'master', name))
        _urlretrieve(url, localfile)

    ds = _open_dataset(localfile, **kws).load()

    if not cache:
        _os.remove(localfile)

    return ds
