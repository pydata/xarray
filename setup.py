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
that makes working with labelled multi-dimensional arrays simple,
efficient, and fun!

Multi-dimensional (a.k.a. N-dimensional, ND) arrays (sometimes called
"tensors") are an essential part of computational science.
They are encountered in a wide range of fields, including physics, astronomy,
geoscience, bioinformatics, engineering, finance, and deep learning.
In Python, NumPy_ provides the fundamental data structure and API for
working with raw ND arrays.
However, real-world datasets are usually more than just raw numbers;
they have labels which encode information about how the array values map
to locations in space, time, etc.

By introducing *dimensions*, *coordinates*, and *attributes* on top of raw
NumPy-like arrays, xarray is able to understand these labels and use them to
provide a more intuitive, more concise, and less error-prone experience.
Xarray also provides a large and growing library of functions for advanced
analytics and visualization with these data structures.
Xarray was inspired by and borrows heavily from pandas_, the popular data
analysis package focused on labelled tabular data.
Xarray can read and write data from most common labeled ND-array storage
formats and is particularly tailored to working with netCDF_ files, which were
the source of xarray's data model.

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
      install_requires=INSTALL_REQUIRES,
      tests_require=TESTS_REQUIRE,
      url=URL,
      packages=find_packages(),
      package_data={'xarray': ['tests/data/*']})
