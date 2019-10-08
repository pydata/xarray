#!/usr/bin/env python
import sys

import versioneer
from setuptools import find_packages, setup

DISTNAME = "xarray"
LICENSE = "Apache"
AUTHOR = "xarray Developers"
AUTHOR_EMAIL = "xarray@googlegroups.com"
URL = "https://github.com/pydata/xarray"
CLASSIFIERS = [
    "Development Status :: 5 - Production/Stable",
    "License :: OSI Approved :: Apache Software License",
    "Operating System :: OS Independent",
    "Intended Audience :: Science/Research",
    "Programming Language :: Python",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.6",
    "Programming Language :: Python :: 3.7",
    "Topic :: Scientific/Engineering",
]

PYTHON_REQUIRES = ">=3.6"
INSTALL_REQUIRES = ["numpy >= 1.14", "pandas >= 0.24"]
needs_pytest = {"pytest", "test", "ptr"}.intersection(sys.argv)
SETUP_REQUIRES = ["pytest-runner >= 4.2"] if needs_pytest else []
TESTS_REQUIRE = ["pytest >= 2.7.1"]

DESCRIPTION = "N-D labeled arrays and datasets in Python"
LONG_DESCRIPTION = """
**xarray** (formerly **xray**) is an open source project and Python package
that makes working with labelled multi-dimensional arrays simple,
efficient, and fun!

Xarray introduces labels in the form of dimensions, coordinates and
attributes on top of raw NumPy_-like arrays, which allows for a more
intuitive, more concise, and less error-prone developer experience.
The package includes a large and growing library of domain-agnostic functions
for advanced analytics and visualization with these data structures.

Xarray was inspired by and borrows heavily from pandas_, the popular data
analysis package focused on labelled tabular data.
It is particularly tailored to working with netCDF_ files, which were the
source of xarray's data model, and integrates tightly with dask_ for parallel
computing.

.. _NumPy: https://www.numpy.org
.. _pandas: https://pandas.pydata.org
.. _dask: https://dask.org
.. _netCDF: https://www.unidata.ucar.edu/software/netcdf

Why xarray?
-----------

Multi-dimensional (a.k.a. N-dimensional, ND) arrays (sometimes called
"tensors") are an essential part of computational science.
They are encountered in a wide range of fields, including physics, astronomy,
geoscience, bioinformatics, engineering, finance, and deep learning.
In Python, NumPy_ provides the fundamental data structure and API for
working with raw ND arrays.
However, real-world datasets are usually more than just raw numbers;
they have labels which encode information about how the array values map
to locations in space, time, etc.

Xarray doesn't just keep track of labels on arrays -- it uses them to provide a
powerful and concise interface. For example:

-  Apply operations over dimensions by name: ``x.sum('time')``.
-  Select values by label instead of integer location:
   ``x.loc['2014-01-01']`` or ``x.sel(time='2014-01-01')``.
-  Mathematical operations (e.g., ``x - y``) vectorize across multiple
   dimensions (array broadcasting) based on dimension names, not shape.
-  Flexible split-apply-combine operations with groupby:
   ``x.groupby('time.dayofyear').mean()``.
-  Database like alignment based on coordinate labels that smoothly
   handles missing values: ``x, y = xr.align(x, y, join='outer')``.
-  Keep track of arbitrary metadata in the form of a Python dictionary:
   ``x.attrs``.

Learn more
----------

- Documentation: http://xarray.pydata.org
- Issue tracker: http://github.com/pydata/xarray/issues
- Source code: http://github.com/pydata/xarray
- SciPy2015 talk: https://www.youtube.com/watch?v=X0pAhJgySxk
"""


setup(
    name=DISTNAME,
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    license=LICENSE,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    classifiers=CLASSIFIERS,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    python_requires=PYTHON_REQUIRES,
    install_requires=INSTALL_REQUIRES,
    setup_requires=SETUP_REQUIRES,
    tests_require=TESTS_REQUIRE,
    url=URL,
    packages=find_packages(),
    package_data={"xarray": ["py.typed", "tests/data/*"]},
)
