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
.. image:: http://img.shields.io/badge/benchmarked%20by-asv-green.svg?style=flat
  :target: http://pandas.pydata.org/speed/xarray/
.. image:: https://img.shields.io/pypi/v/xarray.svg
   :target: https://pypi.python.org/pypi/xarray/

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

.. _NumPy: http://www.numpy.org
.. _pandas: http://pandas.pydata.org
.. _dask: http://dask.org
.. _netCDF: http://www.unidata.ucar.edu/software/netcdf

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

Documentation
-------------

Learn more about xarray in its official documentation at http://xarray.pydata.org/

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

.. _donation: https://numfocus.salsalabs.org/donate-to-xarray/

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

Copyright 2014-2019, xarray Developers

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
