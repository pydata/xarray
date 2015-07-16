
.. image:: _static/dataset-diagram-logo.png
   :width: 300 px
   :align: center

|

N-D labeled arrays and datasets in Python
=========================================

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

Documentation
-------------

.. toctree::
   :maxdepth: 1

   why-xray
   examples
   installing
   data-structures
   indexing
   computation
   groupby
   combining
   time-series
   pandas
   io
   dask
   api
   faq
   whats-new

See also
--------

- Stephan Hoyer's `SciPy2015 talk`_ introducing xray to a general audience.
- Stephan Hoyer's `2015 Unidata Users Workshop talk`_ and `tutorial`_ (`with answers`_) introducing
  xray to users familiar with netCDF.
- `Nicolas Fauchereau's tutorial`_ on xray for netCDF users.

.. _SciPy2015 talk: https://www.youtube.com/watch?v=X0pAhJgySxk
.. _2015 Unidata Users Workshop talk: https://www.youtube.com/watch?v=J9ypQOnt5l8
.. _tutorial: https://github.com/Unidata/unidata-users-workshop/blob/master/notebooks/xray-tutorial.ipynb
.. _with answers: https://github.com/Unidata/unidata-users-workshop/blob/master/notebooks/xray-tutorial-with-answers.ipynb
.. _Nicolas Fauchereau's tutorial: http://nbviewer.ipython.org/github/nicolasfauchereau/metocean/blob/master/notebooks/xray.ipynb

Get in touch
------------

- To ask questions or discuss xray, use the `mailing list`_.
- Report bugs, suggest feature ideas or view the source code `on GitHub`_.
- For interactive discussion, we have a chatroom `on Gitter`_.
- You can also get in touch `on Twitter`_.

.. _mailing list: https://groups.google.com/forum/#!forum/xray-dev
.. _on Gitter: https://gitter.im/xray/xray
.. _on GitHub: http://github.com/xray/xray
.. _on Twitter: http://twitter.com/shoyer

License
-------

xray is available under the open source `Apache License`__.

__ http://www.apache.org/licenses/LICENSE-2.0.html

History
-------

xray is an evolution of an internal tool developed at `The Climate
Corporation`__, and was originally written by current and former Climate Corp
researchers Stephan Hoyer, Alex Kleeman and Eugene Brevdo.

__ http://climate.com/
