xray: N-D labeled arrays and datasets
=====================================

.. image:: https://travis-ci.org/xray/xray.svg?branch=master
   :target: https://travis-ci.org/xray/xray
.. image:: https://ci.appveyor.com/api/projects/status/github/xray/xray?svg=true&passingText=passing&failingText=failing&pendingText=pending
   :target: https://ci.appveyor.com/project/shoyer/xray
.. image:: https://coveralls.io/repos/xray/xray/badge.svg
   :target: https://coveralls.io/r/xray/xray
.. image:: https://img.shields.io/pypi/v/xray.svg
   :target: https://pypi.python.org/pypi/xray/
.. image:: https://badges.gitter.im/Join%20Chat.svg
   :target: https://gitter.im/xray/xray

**xray** is an open source project and Python package that aims to bring the
labeled data power of pandas_ to the physical sciences, by providing
N-dimensional variants of the core pandas data structures.

Our goal is to provide a pandas-like and pandas-compatible toolkit for
analytics on multi-dimensional arrays, rather than the tabular data for which
pandas excels. Our approach adopts the `Common Data Model`_ for self-
describing scientific data in widespread use in the Earth sciences:
``xray.Dataset`` is an in-memory representation of a netCDF file.

.. _pandas: http://pandas.pydata.org
.. _Common Data Model: http://www.unidata.ucar.edu/software/thredds/current/netcdf-java/CDM
.. _netCDF: http://www.unidata.ucar.edu/software/netcdf
.. _OPeNDAP: http://www.opendap.org/

Why xray?
---------

Adding dimensions names and coordinate indexes to numpy's ndarray_ makes many
powerful array operations possible:

-  Apply operations over dimensions by name: ``x.sum('time')``.
-  Select values by label instead of integer location:
   ``x.loc['2014-01-01']`` or ``x.sel(time='2014-01-01')``.
-  Mathematical operations (e.g., ``x - y``) vectorize across multiple
   dimensions (array broadcasting) based on dimension names, not shape.
-  Flexible split-apply-combine operations with groupby:
   ``x.groupby('time.dayofyear').mean()``.
-  Database like alignment based on coordinate labels that smoothly
   handles missing values: ``x, y = xray.align(x, y, join='outer')``.
-  Keep track of arbitrary metadata in the form of a Python dictionary:
   ``x.attrs``.

pandas_ provides many of these features, but it does not make use of dimension
names, and its core data structures are fixed dimensional arrays.

Why isn't pandas enough?
------------------------

pandas_ excels at working with tabular data. That suffices for many statistical
analyses, but physical scientists rely on N-dimensional arrays -- which is
where xray comes in.

xray aims to provide a data analysis toolkit as powerful as pandas_ but
designed for working with homogeneous N-dimensional arrays
instead of tabular data. When possible, we copy the pandas API and rely on
pandas's highly optimized internals (in particular, for fast indexing).

Why netCDF?
-----------

Because xray implements the same data model as the netCDF_ file format,
xray datasets have a natural and portable serialization format. But it is also
easy to robustly convert an xray ``DataArray`` to and from a numpy ``ndarray``
or a pandas ``DataFrame`` or ``Series``, providing compatibility with the full
`PyData ecosystem <http://pydata.org/>`__.

Our target audience is anyone who needs N-dimensional labeled arrays, but we
are particularly focused on the data analysis needs of physical scientists --
especially geoscientists who already know and love netCDF_.

.. _ndarray: http://docs.scipy.org/doc/numpy/reference/arrays.ndarray.html
.. _pandas: http://pandas.pydata.org
.. _netCDF: http://www.unidata.ucar.edu/software/netcdf

Documentation
-------------

The official documentation is hosted on ReadTheDocs: http://xray.readthedocs.org/

Get in touch
------------

- GitHub issue tracker: https://github.com/xray/xray/issues/
- Mailing list: https://groups.google.com/forum/#!forum/xray-dev
- Twitter: http://twitter.com/shoyer

History
-------

xray is an evolution of an internal tool developed at `The Climate
Corporation`__, and was originally written by current and former Climate Corp
researchers Stephan Hoyer, Alex Kleeman and Eugene Brevdo.

__ http://climate.com/

License
-------

Copyright 2014, xray Developers

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

xray includes portions of pandas, NumPy and Seaborn. Their licenses are
included in the licenses directory.
