xray: N-D labeled arrays and datasets
=====================================

.. image:: https://travis-ci.org/xray/xray.svg?branch=master
   :target: https://travis-ci.org/xray/xray
.. image:: https://coveralls.io/repos/xray/xray/badge.svg
   :target: https://coveralls.io/r/xray/xray
.. image:: https://img.shields.io/pypi/v/xray.svg
   :target: https://pypi.python.org/pypi/xray/
.. image:: https://readthedocs.org/projects/xray/badge/?version=stable
   :target: https://readthedocs.org/projects/xray/?badge=stable

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

Documentation:
   http://xray.readthedocs.org/

Get in touch:
   - Mailing list: https://groups.google.com/forum/#!forum/xray-dev
   - Twitter: http://twitter.com/shoyer

History:
   xray is an evolution of an internal tool developed at `The Climate
   Corporation`__, and was originally written by current and former Climate Corp
   researchers Stephan Hoyer, Alex Kleeman and Eugene Brevdo.

__ http://climate.com/

License:
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
   
   xray includes portions of pandas and numpy. Their licenses are included in the
   licenses directory.
