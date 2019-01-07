#!/usr/bin/env python
import sys

import versioneer
from setuptools import find_packages, setup

DISTNAME = 'xarray'
LICENSE = 'Apache'
AUTHOR = 'xarray Developers'
AUTHOR_EMAIL = 'xarray@googlegroups.com'
URL = 'https://github.com/pydata/xarray'
CLASSIFIERS = [
    'Development Status :: 4 - Beta',
    'License :: OSI Approved :: Apache Software License',
    'Operating System :: OS Independent',
    'Intended Audience :: Science/Research',
    'Programming Language :: Python',
    'Programming Language :: Python :: 2',
    'Programming Language :: Python :: 2.7',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
    'Topic :: Scientific/Engineering',
]

PYTHON_REQUIRES = '>=3.5'
INSTALL_REQUIRES = ['numpy >= 1.12', 'pandas >= 0.19.2']
needs_pytest = {'pytest', 'test', 'ptr'}.intersection(sys.argv)
SETUP_REQUIRES = ['pytest-runner >= 4.2'] if needs_pytest else []
TESTS_REQUIRE = ['pytest >= 2.7.1']
if sys.version_info[0] < 3:
    TESTS_REQUIRE.append('mock')

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

.. _NumPy: http://www.numpy.org/
.. _pandas: http://pandas.pydata.org
.. _netCDF: http://www.unidata.ucar.edu/software/netcdf

Important links
---------------

- HTML documentation: http://xarray.pydata.org
- Issue tracker: http://github.com/pydata/xarray/issues
- Source code: http://github.com/pydata/xarray
- SciPy2015 talk: https://www.youtube.com/watch?v=X0pAhJgySxk
"""  # noqa


setup(name=DISTNAME,
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
      package_data={'xarray': ['tests/data/*']})
