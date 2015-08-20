'''
Useful for:

* users learning xray
* building tutorials in the documentation.

'''

import os

import requests

from .backends.api import open_dataset


# Borrowed from Seaborn
def load_dataset(name='ncep_temperature_north-america_2013-14.nc',
                 cache=True, cache_dir='.xray_tutorial_data',
                 github_url='https://github.com/xray/xray-data',  **kws):
    """
    Load a dataset from the online repository (requires internet).

    If a local copy is found then always use that to avoid network traffic.

    Parameters
    ----------
    name : str, optional
        Name of the netcdf file containing the dataset
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
    longdir = os.path.expanduser(os.sep.join(('~', cache_dir)))
    localfile = os.sep.join((longdir, name))

    if not os.path.exists(localfile):

        url = '/'.join((github_url, 'raw', 'master', name))
        response = requests.get(url)

        # This will always leave this directory on disk.
        # May want to add an option to remove it.
        if not os.path.isdir(longdir):
            os.mkdir(longdir)

        with open(localfile, 'wb') as f:
            f.write(response.content)

    ds = open_dataset(localfile, **kws)

    if not cache:
        os.remove(localfile)

    return ds
