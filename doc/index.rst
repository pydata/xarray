xray: pandas for N-dimensional arrays
=====================================

**xray** is an open source project and Python package that aims to bring the
labeled data power of pandas_ to the physical sciences, by providing
N-dimensional variants of the core pandas_ data structures, ``Series`` and
``DataFrame``: the xray ``DataArray`` and ``Dataset``.

**xray** aims to provide a pandas-like and pandas-compatible toolkit for
analytics on *multi-dimensional* labeled arrays. Our approach also builds on
the `Common Data Model`_ for self-describing scientific data (e.g., `NetCDF`_
and OpenDAP), in widespread use in the Earth sciences: ``xray.Dataset``
is an in-memory representation of a NetCDF file.

.. _pandas: http://pandas.pydata.org
.. _PyData: http://pydata.org
.. _Common Data Model: http://www.unidata.ucar.edu/software/thredds/current/netcdf-java/CDM
.. _NetCDF: http://www.unidata.ucar.edu/software/netcdf

Documentation
-------------

.. toctree::
   :maxdepth: 1

   why-xray
   installing
   tutorial
   examples
   faq
   api
   whats-new

Important links
---------------

- HTML documentation: http://xray.readthedocs.org
- Issue tracker: http://github.com/xray/xray/issues
- Source code: http://github.com/xray/xray
- PyData talk: https://www.youtube.com/watch?v=T5CZyNwBa9c

Get in touch
------------

- Mailing list: https://groups.google.com/forum/#!forum/xray-dev
- Twitter: http://twitter.com/shoyer

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
