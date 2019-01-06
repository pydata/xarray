xarray: N-D labeled arrays and datasets
=======================================

.. image:: https://travis-ci.org/pydata/xarray.svg?branch=master
   :target: https://travis-ci.org/pydata/xarray
.. image:: https://ci.appveyor.com/api/projects/status/github/pydata/xarray?svg=true&passingText=passing&failingText=failing&pendingText=pending
   :target: https://ci.appveyor.com/project/shoyer/xray
.. image:: https://coveralls.io/repos/pydata/xarray/badge.svg
   :target: https://coveralls.io/r/pydata/xarray
.. image:: https://readthedocs.org/projects/xray/badge/?version=latest
   :target: http://xarray.pydata.org/
.. image:: https://img.shields.io/pypi/v/xarray.svg
   :target: https://pypi.python.org/pypi/xarray/
.. image:: https://zenodo.org/badge/13221727.svg
  :target: https://zenodo.org/badge/latestdoi/13221727
.. image:: http://img.shields.io/badge/benchmarked%20by-asv-green.svg?style=flat
  :target: http://pandas.pydata.org/speed/xarray/
.. image:: https://img.shields.io/badge/powered%20by-NumFOCUS-orange.svg?style=flat&colorA=E1523D&colorB=007D8A
  :target: http://numfocus.org

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

Why xarray?
-----------

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
   handles missing values: ``x, y = xr.align(x, y, join='outer')``.
-  Keep track of arbitrary metadata in the form of a Python dictionary:
   ``x.attrs``.

pandas_ provides many of these features, but it does not make use of dimension
names, and its core data structures are fixed dimensional arrays.

Why isn't pandas enough?
------------------------

pandas_ excels at working with tabular data. That suffices for many statistical
analyses, but physical scientists rely on N-dimensional arrays -- which is
where xarray comes in.

xarray aims to provide a data analysis toolkit as powerful as pandas_ but
designed for working with homogeneous N-dimensional arrays
instead of tabular data. When possible, we copy the pandas API and rely on
pandas's highly optimized internals (in particular, for fast indexing).

Why netCDF?
-----------

Because xarray implements the same data model as the netCDF_ file format,
xarray datasets have a natural and portable serialization format. But it is also
easy to robustly convert an xarray ``DataArray`` to and from a numpy ``ndarray``
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

The official documentation is hosted on ReadTheDocs at http://xarray.pydata.org/

Contributing
------------

You can find information about contributing to xarray at our `Contributing page <http://xarray.pydata.org/en/latest/contributing.html#>`_.

Get in touch
------------

- Ask usage questions ("How do I?") on `StackOverflow`_.
- Report bugs, suggest features or view the source code `on GitHub`_.
- For less well defined questions or ideas, or to announce other projects of
  interest to xarray users, use the `mailing list`_.

.. _StackOverFlow: http://stackoverflow.com/questions/tagged/python-xarray
.. _mailing list: https://groups.google.com/forum/#!forum/xarray
.. _on GitHub: http://github.com/pydata/xarray

NumFOCUS
--------

.. image:: https://numfocus.org/wp-content/uploads/2017/07/NumFocus_LRG.png
   :scale: 25 %
   :target: https://numfocus.org/

Xarray is a fiscally sponsored project of NumFOCUS_, a nonprofit dedicated
to supporting the open source scientific computing community. If you like
Xarray and want to support our mission, please consider making a donation_
to support our efforts.

.. _donation: https://www.flipcause.com/secure/cause_pdetails/NDE2NTU=

History
-------

xarray is an evolution of an internal tool developed at `The Climate
Corporation`__. It was originally written by Climate Corp researchers Stephan
Hoyer, Alex Kleeman and Eugene Brevdo and was released as open source in
May 2014. The project was renamed from "xray" in January 2016. Xarray became a
fiscally sponsored project of NumFOCUS_ in August 2018.

__ http://climate.com/
.. _NumFOCUS: https://numfocus.org

License
-------

Copyright 2014-2018, xarray Developers

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

xarray bundles portions of pandas, NumPy and Seaborn, all of which are available
under a "3-clause BSD" license:
- pandas: setup.py, xarray/util/print_versions.py
- NumPy: xarray/core/npcompat.py
- Seaborn: _determine_cmap_params in xarray/core/plot/utils.py

xarray also bundles portions of CPython, which is available under the "Python
Software Foundation License" in xarray/core/pycompat.py.

The full text of these licenses are included in the licenses directory.
