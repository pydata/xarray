xarray: N-D labeled arrays and datasets in Python
=================================================

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

Documentation
-------------

**Getting Started**

* :doc:`why-xarray`
* :doc:`faq`
* :doc:`examples`
* :doc:`installing`

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Getting Started

   why-xarray
   faq
   examples
   installing

**User Guide**

* :doc:`data-structures`
* :doc:`indexing`
* :doc:`interpolation`
* :doc:`computation`
* :doc:`groupby`
* :doc:`reshaping`
* :doc:`combining`
* :doc:`time-series`
* :doc:`pandas`
* :doc:`io`
* :doc:`dask`
* :doc:`plotting`

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: User Guide

   data-structures
   indexing
   interpolation
   computation
   groupby
   reshaping
   combining
   time-series
   pandas
   io
   dask
   plotting

**Help & reference**

* :doc:`whats-new`
* :doc:`api`
* :doc:`internals`
* :doc:`roadmap`
* :doc:`contributing`
* :doc:`related-projects`

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Help & reference

   whats-new
   api
   internals
   roadmap
   contributing
   related-projects

See also
--------

- Stephan Hoyer and Joe Hamman's `Journal of Open Research Software paper`_ describing the xarray project.
- The `UW eScience Institute's Geohackweek`_ tutorial on xarray for geospatial data scientists.
- Stephan Hoyer's `SciPy2015 talk`_ introducing xarray to a general audience.
- Stephan Hoyer's `2015 Unidata Users Workshop talk`_ and `tutorial`_ (`with answers`_) introducing
  xarray to users familiar with netCDF.
- `Nicolas Fauchereau's tutorial`_ on xarray for netCDF users.

.. _Journal of Open Research Software paper: http://doi.org/10.5334/jors.148
.. _UW eScience Institute's Geohackweek : https://geohackweek.github.io/nDarrays/
.. _SciPy2015 talk: https://www.youtube.com/watch?v=X0pAhJgySxk
.. _2015 Unidata Users Workshop talk: https://www.youtube.com/watch?v=J9ypQOnt5l8
.. _tutorial: https://github.com/Unidata/unidata-users-workshop/blob/master/notebooks/xray-tutorial.ipynb
.. _with answers: https://github.com/Unidata/unidata-users-workshop/blob/master/notebooks/xray-tutorial-with-answers.ipynb
.. _Nicolas Fauchereau's tutorial: http://nbviewer.ipython.org/github/nicolasfauchereau/metocean/blob/master/notebooks/xray.ipynb

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

.. image:: _static/numfocus_logo.png
   :scale: 50 %
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

xarray is available under the open source `Apache License`__.

__ http://www.apache.org/licenses/LICENSE-2.0.html
