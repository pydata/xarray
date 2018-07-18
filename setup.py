#!/usr/bin/env python
import sys

from setuptools import find_packages, setup

import versioneer


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
    'Topic :: Scientific/Engineering',
]

INSTALL_REQUIRES = ['numpy >= 1.12', 'pandas >= 0.19.2']
TESTS_REQUIRE = ['pytest >= 2.7.1']
if sys.version_info[0] < 3:
    TESTS_REQUIRE.append('mock')

DESCRIPTION = "N-D labeled arrays and datasets in Python"
LONG_DESCRIPTION = """
**xarray** (formerly **xray**) is an open source project and Python package
that aims to bring the labeled data power of pandas_ to the physical sciences,
by providing N-dimensional variants of the core pandas data structures.

Our goal is to provide a pandas-like and pandas-compatible toolkit for
analytics on multi-dimensional arrays, rather than the tabular data for which
pandas excels. Our approach adopts the `Common Data Model`_ for self-
describing scientific data in widespread use in the Earth sciences:
``xarray.Dataset`` is an in-memory representation of a netCDF file.

.. _pandas: http://pandas.pydata.org
.. _Common Data Model: http://www.unidata.ucar.edu/software/thredds/current/netcdf-java/CDM
.. _netCDF: http://www.unidata.ucar.edu/software/netcdf
.. _OPeNDAP: http://www.opendap.org/

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
      install_requires=INSTALL_REQUIRES,
      tests_require=TESTS_REQUIRE,
      url=URL,
      packages=find_packages(),
      package_data={'xarray': ['tests/data/*']})
