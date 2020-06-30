"""
Useful for:

* users learning xarray
* building tutorials in the documentation.

"""
import hashlib
import os
import pathlib
import shutil
import sys
import tempfile
from contextlib import contextmanager

import numpy as np
import requests

from .backends.api import open_dataset as _open_dataset
from .backends.rasterio_ import open_rasterio as _open_rasterio
from .core.dataarray import DataArray
from .core.dataset import Dataset

# vendored from appdirs
if sys.platform.startswith("java"):
    import platform

    os_name = platform.java_ver()[3][0]
    if os_name.startswith("Windows"):  # "Windows XP", "Windows 7", etc.
        system = "win32"
    elif os_name.startswith("Mac"):  # "Mac OS X", etc.
        system = "darwin"
    else:  # "Linux", "SunOS", "FreeBSD", etc.
        # Setting this to "linux2" is not ideal, but only Windows or Mac
        # are actually checked for and the rest of the module expects
        # *sys.platform* style strings.
        system = "linux2"
else:
    system = sys.platform


def _get_win_folder_from_registry(csidl_name):
    """This is a fallback technique at best. I'm not sure if using the
    registry for this guarantees us the correct answer for all CSIDL_*
    names.
    """
    import winreg

    shell_folder_name = {
        "CSIDL_APPDATA": "AppData",
        "CSIDL_COMMON_APPDATA": "Common AppData",
        "CSIDL_LOCAL_APPDATA": "Local AppData",
    }[csidl_name]

    key = winreg.OpenKey(
        winreg.HKEY_CURRENT_USER,
        r"Software\Microsoft\Windows\CurrentVersion\Explorer\Shell Folders",
    )
    dir, type = winreg.QueryValueEx(key, shell_folder_name)
    return dir


def _get_win_folder_with_ctypes(csidl_name):
    import ctypes

    csidl_const = {
        "CSIDL_APPDATA": 26,
        "CSIDL_COMMON_APPDATA": 35,
        "CSIDL_LOCAL_APPDATA": 28,
    }[csidl_name]

    buf = ctypes.create_unicode_buffer(1024)
    ctypes.windll.shell32.SHGetFolderPathW(None, csidl_const, None, 0, buf)

    # Downgrade to short path name if have highbit chars. See
    # <http://bugs.activestate.com/show_bug.cgi?id=85099>.
    has_high_char = False
    for c in buf:
        if ord(c) > 255:
            has_high_char = True
            break
    if has_high_char:
        buf2 = ctypes.create_unicode_buffer(1024)
        if ctypes.windll.kernel32.GetShortPathNameW(buf.value, buf2, 1024):
            buf = buf2

    return buf.value


def _get_win_folder_with_jna(csidl_name):
    import array
    from com.sun import jna
    from com.sun.jna.platform import win32

    buf_size = win32.WinDef.MAX_PATH * 2
    buf = array.zeros("c", buf_size)
    shell = win32.Shell32.INSTANCE
    shell.SHGetFolderPath(
        None,
        getattr(win32.ShlObj, csidl_name),
        None,
        win32.ShlObj.SHGFP_TYPE_CURRENT,
        buf,
    )
    dir = jna.Native.toString(buf.tostring()).rstrip("\0")

    # Downgrade to short path name if have highbit chars. See
    # <http://bugs.activestate.com/show_bug.cgi?id=85099>.
    has_high_char = False
    for c in dir:
        if ord(c) > 255:
            has_high_char = True
            break
    if has_high_char:
        buf = array.zeros("c", buf_size)
        kernel = win32.Kernel32.INSTANCE
        if kernel.GetShortPathName(dir, buf, buf_size):
            dir = jna.Native.toString(buf.tostring()).rstrip("\0")

    return dir


if system == "win32":
    try:
        from ctypes import windll  # noqa: F401
    except ImportError:
        try:
            import com.sun.jna  # noqa: F401
        except ImportError:
            _get_win_folder = _get_win_folder_from_registry
        else:
            _get_win_folder = _get_win_folder_with_jna
    else:
        _get_win_folder = _get_win_folder_with_ctypes


def user_cache_dir(appname=None, appauthor=None, version=None, opinion=True):
    r"""Return full path to the user-specific cache dir for this application.
        "appname" is the name of application.
            If None, just the system directory is returned.
        "appauthor" (only used on Windows) is the name of the
            appauthor or distributing body for this application. Typically
            it is the owning company name. This falls back to appname. You may
            pass False to disable it.
        "version" is an optional version path element to append to the
            path. You might want to use this if you want multiple versions
            of your app to be able to run independently. If used, this
            would typically be "<major>.<minor>".
            Only applied when appname is present.
        "opinion" (boolean) can be False to disable the appending of
            "Cache" to the base app data dir for Windows. See
            discussion below.
    Typical user cache directories are:
        Mac OS X:   ~/Library/Caches/<AppName>
        Unix:       ~/.cache/<AppName> (XDG default)
        Win XP:     C:\Documents and Settings\<username>\Local Settings\Application Data\<AppAuthor>\<AppName>\Cache
        Vista:      C:\Users\<username>\AppData\Local\<AppAuthor>\<AppName>\Cache
    On Windows the only suggestion in the MSDN docs is that local settings go in
    the `CSIDL_LOCAL_APPDATA` directory. This is identical to the non-roaming
    app data dir (the default returned by `user_data_dir` above). Apps typically
    put cache data somewhere *under* the given dir here. Some examples:
        ...\Mozilla\Firefox\Profiles\<ProfileName>\Cache
        ...\Acme\SuperApp\Cache\1.0
    OPINION: This function appends "Cache" to the `CSIDL_LOCAL_APPDATA` value.
    This can be disabled with the `opinion=False` option.
    """
    if system == "win32":
        if appauthor is None:
            appauthor = appname
        path = os.path.normpath(_get_win_folder("CSIDL_LOCAL_APPDATA"))
        if appname:
            if appauthor is not False:
                path = os.path.join(path, appauthor, appname)
            else:
                path = os.path.join(path, appname)
            if opinion:
                path = os.path.join(path, "Cache")
    elif system == "darwin":
        path = os.path.expanduser("~/Library/Caches")
        if appname:
            path = os.path.join(path, appname)
    else:
        path = os.getenv("XDG_CACHE_HOME", os.path.expanduser("~/.cache"))
        if appname:
            path = os.path.join(path, appname)
    if appname and version:
        path = os.path.join(path, version)

    return pathlib.Path(path)


_default_cache_dir = user_cache_dir("xarray_tutorial_data")


def check_md5sum(content, checksum):
    md5 = hashlib.md5()
    md5.update(content)
    md5sum = md5.hexdigest()

    return md5sum == checksum


# based on https://stackoverflow.com/a/29491523
@contextmanager
def open_atomic(path, mode=None):
    folder = path.parent
    prefix = f".{path.name}"

    try:
        f = tempfile.NamedTemporaryFile(dir=folder, prefix=prefix, delete=False)
        temporary_path = pathlib.Path(f.name)

        yield f

        temporary_path.rename(path)
    except Exception:
        temporary_path.unlink()
        raise
    finally:
        pass


def download_to(url, path):
    # based on https://stackoverflow.com/a/39217788
    with open_atomic(path, mode="wb") as f:
        with requests.get(url, stream=True) as r:
            if r.status_code != 200:
                raise OSError(f"download failed: {r.reason}")

            r.raw.decode_content = True
            shutil.copyfileobj(r.raw, f)


def open_rasterio(
    name,
    cache=True,
    cache_dir=_default_cache_dir,
    github_url="https://github.com/mapbox/rasterio",
    branch="master",
    **kws,
):
    if not cache_dir.is_dir():
        cache_dir.mkdir()

    default_extension = ".tif"

    if cache:
        path = cache_dir / name
        # need to always do that, otherwise we might close the
        # path using the context manager
        cache_dir = pathlib.Path(cache_dir)
    else:
        cache_dir = tempfile.TemporaryDirectory()
        path = pathlib.Path(cache_dir.name) / name

    if not path.suffix:
        path = path.with_suffix(default_extension)
    elif path.suffix == ".byte":
        path = path.with_name(name + default_extension)

    if cache and path.is_file():
        return _open_rasterio(path, **kws)

    url = f"{github_url}/raw/{branch}/tests/data/{path.name}"
    # if cache_dir points to a temporary directory, make sure it is
    # deleted afterwards
    with cache_dir:
        download_to(url, path)
        return _open_rasterio(path, **kws)


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

    def construct_url(full_name):
        return f"{github_url}/raw/{branch}/{full_name}"

    if not cache_dir.is_dir():
        cache_dir.mkdir()

    default_extension = ".nc"

    if cache:
        path = cache_dir / name
        # need to always do that, otherwise we might close the
        # path using the context manager
        cache_dir = pathlib.Path(cache_dir)
    else:
        cache_dir = tempfile.TemporaryDirectory()
        path = pathlib.Path(cache_dir.name) / name

    if not path.suffix:
        path = path.with_suffix(default_extension)

    if cache and path.is_file():
        return _open_dataset(path, **kws)

    # if cache_dir points to a temporary directory, make sure it is
    # deleted afterwards
    with cache_dir:
        download_to(construct_url(path.name), path)

        # verify the checksum (md5 guards only against transport corruption)
        md5_path = path.with_name(path.name + ".md5")
        download_to(construct_url(md5_path.name), md5_path)
        if not check_md5sum(path.read_bytes(), md5_path.read_text()):
            path.unlink()
            md5_path.unlink()
            msg = """
            MD5 checksum does not match, try downloading dataset again.
            """
            raise OSError(msg)

        return _open_dataset(path, **kws)


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
