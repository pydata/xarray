
.. image:: _static/dataset-diagram-logo.png
   :width: 300 px
   :align: center

|

N-D labeled arrays and datasets in Python
=========================================

**xarray** (formerly **xray**) is an open source project and Python package
that aims to bring the labeled data power of pandas_ to the physical sciences,
by providing N-dimensional variants of the core pandas data structures.

Our goal is to provide a pandas-like and pandas-compatible toolkit for
analytics on multi-dimensional arrays, rather than the tabular data for which
pandas excels. Our approach adopts the `Common Data Model`_ for self-
describing scientific data in widespread use in the Earth sciences:
``xarray.Dataset`` is an in-memory representation of a netCDF file.

.. note::

   xray is now xarray! See :ref:`the v0.7.0 release notes<whats-new.0.7.0>`
   for more details. The preferred URL for these docs is now
   http://xarray.pydata.org.

.. _pandas: http://pandas.pydata.org
.. _Common Data Model: http://www.unidata.ucar.edu/software/thredds/current/netcdf-java/CDM
.. _netCDF: http://www.unidata.ucar.edu/software/netcdf
.. _OPeNDAP: http://www.opendap.org/

Documentation
-------------

.. toctree::
   :maxdepth: 1

   whats-new
   why-xarray
   faq
   examples
   installing
   data-structures
   indexing
   computation
   groupby
   reshaping
   combining
   time-series
   pandas
   io
   dask
   plotting
   api
   internals

See also
--------

- Stephan Hoyer's `SciPy2015 talk`_ introducing xarray to a general audience.
- Stephan Hoyer's `2015 Unidata Users Workshop talk`_ and `tutorial`_ (`with answers`_) introducing
  xarray to users familiar with netCDF.
- `Nicolas Fauchereau's tutorial`_ on xarray for netCDF users.

.. _SciPy2015 talk: https://www.youtube.com/watch?v=X0pAhJgySxk
.. _2015 Unidata Users Workshop talk: https://www.youtube.com/watch?v=J9ypQOnt5l8
.. _tutorial: https://github.com/Unidata/unidata-users-workshop/blob/master/notebooks/xray-tutorial.ipynb
.. _with answers: https://github.com/Unidata/unidata-users-workshop/blob/master/notebooks/xray-tutorial-with-answers.ipynb
.. _Nicolas Fauchereau's tutorial: http://nbviewer.ipython.org/github/nicolasfauchereau/metocean/blob/master/notebooks/xray.ipynb

Get in touch
------------

- Ask usage questions on `StackOverflow`_.
- Report bugs, suggest features or view the source code `on GitHub`_.
- For less well defined questions or ideas, use the `mailing list`_.
- You can also try our chatroom `on Gitter`_.

.. _StackOverFlow: http://stackoverflow.com/questions/tagged/python-xarray
.. _mailing list: https://groups.google.com/forum/#!forum/xarray
.. _on Gitter: https://gitter.im/pydata/xarray
.. _on GitHub: http://github.com/pydata/xarray

License
-------

xarray is available under the open source `Apache License`__.

__ http://www.apache.org/licenses/LICENSE-2.0.html

History
-------

xarray is an evolution of an internal tool developed at `The Climate
Corporation`__. It was originally written by Climate Corp researchers Stephan
Hoyer, Alex Kleeman and Eugene Brevdo and was released as open source in
May 2014. The project was renamed from "xray" in January 2016.

__ http://climate.com/
