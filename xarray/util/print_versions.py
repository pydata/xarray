"""Utility functions for printing version information."""

import contextlib
import locale
import os
import platform
import struct
import subprocess
import sys
from html import escape
from typing import TextIO

from xarray.util._report import (
    html_columns,
    html_table,
    markdown_table,
    package_version,
    show_report,
)


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


def show_versions(file: TextIO | None = None, markdown: bool = False) -> None:
    """Print the versions of xarray and its dependencies.

    In Jupyter notebooks the versions are displayed as formatted tables.

    Parameters
    ----------
    file : file-like, optional
        print to the given file-like object. Defaults to sys.stdout.
    markdown : bool, default: False
        print markdown tables wrapped in a ``<details>`` block, which can be
        pasted as is into a GitHub issue.
    """
    sys_info = get_sys_info()

    try:
        sys_info.extend(netcdf_and_hdf5_versions())
    except Exception as e:
        sys_info.append(("libhdf5 / libnetcdf", f"error: {e}"))

    sys_info = [(k, v) for k, v in sys_info if v not in (None, "")]
    versions = {name: package_version(name) for name in DEPENDENCIES}
    installed = [(name, ver) for name, ver in versions.items() if ver is not None]
    missing = ", ".join(name for name, ver in versions.items() if ver is None)
    not_installed = f"Not installed: {missing or 'none'}"

    text = "\n\n".join(
        [
            "\n".join(f"{k}: {v}" for k, v in sys_info),
            "\n".join(f"{k}: {v}" for k, v in installed),
            not_installed,
        ]
    )
    md = "\n\n".join(
        [
            markdown_table(("system", "info"), sys_info),
            markdown_table(("package", "version"), installed),
            not_installed,
        ]
    )
    html = (
        html_columns(
            html_table("System", sys_info),
            html_table("Packages", installed),
        )
        + f"<p>{escape(not_installed)}</p>"
    )
    show_report(
        "INSTALLED VERSIONS", text, md, file=file, as_markdown=markdown, html=html
    )


if __name__ == "__main__":
    show_versions()
