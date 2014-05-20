 #!/usr/bin/env python
import os
import re
import sys
import warnings
try:
    from setuptools import setup
except:
    from distutils.core import setup

MAJOR = 0
MINOR = 1
MICRO = 1
ISRELEASED = True
VERSION = '%d.%d.%d' % (MAJOR, MINOR, MICRO)
QUALIFIER = ''


DISTNAME = 'xray'
LICENSE = 'Apache'
AUTHOR = 'xray Developers'
AUTHOR_EMAIL = 'xray-discussion@googlegroups.com'
URL = 'https://github.com/xray/xray'
CLASSIFIERS = [
    'Development Status :: 3 - Alpha',
    'License :: OSI Approved :: Apache Software License',
    'Operating System :: OS Independent',
    'Intended Audience :: Science/Research',
    'Programming Language :: Python',
    'Programming Language :: Python :: 2',
    'Programming Language :: Python :: 2.7',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.3',
    'Topic :: Scientific/Engineering',
]


DESCRIPTION = "Extended arrays for working with scientific datasets in Python"
LONG_DESCRIPTION = """
**xray** is a Python package for working with aligned sets of
homogeneous, n-dimensional arrays. It implements flexible array
operations and dataset manipulation for in-memory datasets within the
`Common Data
Model <http://www.unidata.ucar.edu/software/thredds/current/netcdf-java/CDM/>`__
widely used for self-describing scientific data (e.g., the NetCDF file
format).

Why xray?
---------

Adding dimensions names and coordinate values to numpy's
`ndarray <http://docs.scipy.org/doc/numpy/reference/arrays.ndarray.html>`__
makes many powerful array operations possible:

-  Apply operations over dimensions by name: ``x.sum('time')``.
-  Select values by label instead of integer location:
   ``x.loc['2014-01-01']`` or ``x.labeled(time='2014-01-01')``.
-  Mathematical operations (e.g., ``x - y``) vectorize across multiple
   dimensions (known in numpy as "broadcasting") based on dimension
   names, regardless of their original order.
-  Flexible split-apply-combine operations with groupby:
   ``x.groupby('time.dayofyear').mean()``.
-  Database like aligment based on coordinate labels that smoothly
   handles missing values: ``x, y = xray.align(x, y, join='outer')``.
-  Keep track of arbitrary metadata in the form of a Python dictionary:
   ``x.attrs``.

**xray** aims to provide a data analysis toolkit as powerful as
`pandas <http://pandas.pydata.org/>`__ but designed for working with
homogeneous N-dimensional arrays instead of tabular data. Indeed, much
of its design and internal functionality (in particular, fast indexing)
is shamelessly borrowed from pandas.

Because **xray** implements the same data model as the NetCDF file
format, xray datasets have a natural and portable serialization format.
But it's also easy to robustly convert an xray ``DataArray`` to and from
a numpy ``ndarray`` or a pandas ``DataFrame`` or ``Series``, providing
compatibility with the full `PyData ecosystem <http://pydata.org/>`__.

For more about **xray**, see the project's `GitHub page
<https://github.com/xray/xray>`__ and `documentation
<http://xray.readthedocs.org>`__
"""

# code to extract and write the version copied from pandas, which is available
# under the BSD license:
FULLVERSION = VERSION
write_version = True

if not ISRELEASED:
    import subprocess
    FULLVERSION += '.dev'

    pipe = None
    for cmd in ['git', 'git.cmd']:
        try:
            pipe = subprocess.Popen(
                [cmd, "describe", "--always", "--match", "v[0-9]*"],
                stdout=subprocess.PIPE)
            (so, serr) = pipe.communicate()
            if pipe.returncode == 0:
                break
        except:
            pass

    if pipe is None or pipe.returncode != 0:
        # no git, or not in git dir
        if os.path.exists('xray/version.py'):
            warnings.warn("WARNING: Couldn't get git revision, using existing xray/version.py")
            write_version = False
        else:
            warnings.warn("WARNING: Couldn't get git revision, using generic version string")
    else:
        # have git, in git dir, but may have used a shallow clone (travis does this)
        rev = so.strip()
        # makes distutils blow up on Python 2.7
        if sys.version_info[0] >= 3:
            rev = rev.decode('ascii')

        if not rev.startswith('v') and re.match("[a-zA-Z0-9]{7,9}", rev):
            # partial clone, manually construct version string
            # this is the format before we started using git-describe
            # to get an ordering on dev version strings.
            rev = "v%s.dev-%s" % (VERSION, rev)

        # Strip leading v from tags format "vx.y.z" to get th version string
        FULLVERSION = rev.lstrip('v')

else:
    FULLVERSION += QUALIFIER


def write_version_py(filename=None):
    cnt = """\
version = '%s'
short_version = '%s'
"""
    if not filename:
        filename = os.path.join(
            os.path.dirname(__file__), 'xray', 'version.py')

    a = open(filename, 'w')
    try:
        a.write(cnt % (FULLVERSION, VERSION))
    finally:
        a.close()

if write_version:
    write_version_py()


setup(name=DISTNAME,
      version=FULLVERSION,
      license=LICENSE,
      author=AUTHOR,
      author_email=AUTHOR_EMAIL,
      classifiers=CLASSIFIERS,
      description=DESCRIPTION,
      long_description=LONG_DESCRIPTION,
      install_requires=['numpy >= 1.7', 'pandas >= 0.13.1'],
      tests_require=['nose >= 1.0'],
      url=URL,
      test_suite='nose.collector',
      packages=['xray', 'xray.backends'])
