"""Utility functions for printing version information."""

import contextlib
import locale
import os
import platform
import struct
import subprocess
import sys
from typing import TextIO

from xarray.util._report import markdown_table, package_version, show_report


def get_sys_info():
    """Returns system information as a dict"""

    blob = []

    # get full commit hash
    commit = None
    if os.path.isdir(".git") and os.path.isdir("xarray"):
        try:
            pipe = subprocess.Popen(
                ("git", "log", '--format="%H"', "-n", "1"),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            so, _ = pipe.communicate()
        except Exception:
            pass
        else:
            if pipe.returncode == 0:
                commit = so
                with contextlib.suppress(ValueError):
                    commit = so.decode("utf-8")
                commit = commit.strip().strip('"')

    blob.append(("commit", commit))

    try:
        (sysname, _nodename, release, _version, machine, processor) = platform.uname()
        blob.extend(
            [
                ("python", sys.version),
                ("python-bits", struct.calcsize("P") * 8),
                ("OS", f"{sysname}"),
                ("OS-release", f"{release}"),
                # ("Version", f"{version}"),
                ("machine", f"{machine}"),
                ("processor", f"{processor}"),
                ("byteorder", f"{sys.byteorder}"),
                ("LC_ALL", f"{os.environ.get('LC_ALL', 'None')}"),
                ("LANG", f"{os.environ.get('LANG', 'None')}"),
                ("LOCALE", f"{locale.getlocale()}"),
            ]
        )
    except Exception:
        pass

    return blob


def netcdf_and_hdf5_versions():
    libhdf5_version = None
    libnetcdf_version = None
    try:
        import netCDF4

        libhdf5_version = netCDF4.__hdf5libversion__
        libnetcdf_version = netCDF4.__netcdf4libversion__
    except ImportError:
        try:
            import h5py

            libhdf5_version = h5py.version.hdf5_version
        except ImportError:
            pass
    return [("libhdf5", libhdf5_version), ("libnetcdf", libnetcdf_version)]


DEPENDENCIES = [
    "xarray",
    "pandas",
    "numpy",
    "scipy",
    # xarray optionals
    "netCDF4",
    "pydap",
    "h5netcdf",
    "h5py",
    "zarr",
    "cftime",
    "nc_time_axis",
    "iris",
    "bottleneck",
    "dask",
    "distributed",
    "matplotlib",
    "cartopy",
    "seaborn",
    "numbagg",
    "fsspec",
    "cupy",
    "pint",
    "sparse",
    "flox",
    "numpy_groupies",
    # xarray setup/test
    "setuptools",
    "pip",
    "conda",
    "pytest",
    "mypy",
    # Misc.
    "IPython",
    "sphinx",
]


def show_versions(file: TextIO | None = None) -> None:
    """Print the versions of xarray and its dependencies.

    The output is markdown that can be pasted as is into a GitHub issue.
    In Jupyter notebooks it is displayed as formatted tables instead; pass
    ``file=sys.stdout`` to get the markdown to copy.

    Parameters
    ----------
    file : file-like, optional
        print to the given file-like object. Defaults to sys.stdout.
    """
    sys_info = get_sys_info()

    try:
        sys_info.extend(netcdf_and_hdf5_versions())
    except Exception as e:
        sys_info.append(("libhdf5 / libnetcdf", f"error: {e}"))

    versions = {name: package_version(name) for name in DEPENDENCIES}
    installed = [(name, ver) for name, ver in versions.items() if ver is not None]
    missing = [name for name, ver in versions.items() if ver is None]

    body = "\n\n".join(
        [
            markdown_table(
                ("system", "info"),
                [(k, v) for k, v in sys_info if v not in (None, "")],
            ),
            markdown_table(("package", "version"), installed),
            f"Not installed: {', '.join(missing) or 'none'}",
        ]
    )
    show_report("INSTALLED VERSIONS", body, file=file)


if __name__ == "__main__":
    show_versions()
