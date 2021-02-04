xarray: N-D labeled arrays and datasets in Python
=================================================

**xarray** (formerly **xray**) is an open source project and Python package
that makes working with labelled multi-dimensional arrays simple,
efficient, and fun!

Xarray introduces labels in the form of dimensions, coordinates and
attributes on top of raw NumPy_-like arrays, which allows for a more
intuitive, more concise, and less error-prone developer experience.
The package includes a large and growing library of domain-agnostic functions
for advanced analytics and visualization with these data structures.

Xarray is inspired by and borrows heavily from pandas_, the popular data
analysis package focused on labelled tabular data.
It is particularly tailored to working with netCDF_ files, which were the
source of xarray's data model, and integrates tightly with dask_ for parallel
computing.

.. _NumPy: http://www.numpy.org
.. _pandas: http://pandas.pydata.org
.. _dask: http://dask.org
.. _netCDF: http://www.unidata.ucar.edu/software/netcdf


.. panels::
    :container: full-width
    :column: text-center col-lg-6 col-md-6 col-sm-12 col-xs-12 p-2
    :card: +my-2
    :body: d-none

    ---
    :fa:`walking, fa-9x`

    Getting started guide
    ^^^^^^^^^^^^^^^^^^^^^
    +++
    The getting started guide aims to get you using xarray productively as quickly as possible.
    It is designed as an entry point for new users, and it provided an introduction to xarray's main concepts.

    .. link-button:: getting-started-guide/index
        :type: ref
        :text: To the getting started guide
        :classes: btn-outline-dark btn-block stretched-link

    ---
    :fa:`book-reader, fa-9x`

    User guide
    ^^^^^^^^^^

    +++
    In this user guide, you will find detailed descriptions and
    examples that describe many common tasks that you can accomplish with xarray.

    .. link-button:: user-guide/index
        :type: ref
        :text: To the user guide
        :classes: btn-outline-dark btn-block stretched-link

    ---
    :fa:`laptop-code, fa-9x`

    Developer guide
    ^^^^^^^^^^^^^^^

    +++
    Contributions are highly welcomed and appreciated. Every little help counts, so do not hesitate!
    The contribution guide explains how to structure your contributions.

    .. link-button:: reference-guide/index
        :type: ref
        :text: To the development guide
        :classes: btn-outline-dark btn-block stretched-link


.. toctree::
   :maxdepth: 1
   :caption: Documentation
   :hidden:

   Getting Started <getting-started-guide/index>
   Gallery <gallery.rst>
   User Guide <user-guide/index>
   Ecosystem <related-projects.rst>
   API Reference <reference-guide/index>
   Development <dev-guide/index>


See also
--------

- `Xarray's Tutorial`_ presented at the 2020 SciPy Conference (`video recording`_).
- Stephan Hoyer and Joe Hamman's `Journal of Open Research Software paper`_ describing the xarray project.
- The `UW eScience Institute's Geohackweek`_ tutorial on xarray for geospatial data scientists.
- Stephan Hoyer's `SciPy2015 talk`_ introducing xarray to a general audience.
- Stephan Hoyer's `2015 Unidata Users Workshop talk`_ and `tutorial`_ (`with answers`_) introducing
  xarray to users familiar with netCDF.
- `Nicolas Fauchereau's tutorial`_ on xarray for netCDF users.

.. _Xarray's Tutorial: https://xarray-contrib.github.io/xarray-tutorial/
.. _video recording: https://youtu.be/mecN-Ph_-78
.. _Journal of Open Research Software paper: http://doi.org/10.5334/jors.148
.. _UW eScience Institute's Geohackweek : https://geohackweek.github.io/nDarrays/
.. _SciPy2015 talk: https://www.youtube.com/watch?v=X0pAhJgySxk
.. _2015 Unidata Users Workshop talk: https://www.youtube.com/watch?v=J9ypQOnt5l8
.. _tutorial: https://github.com/Unidata/unidata-users-workshop/blob/master/notebooks/xray-tutorial.ipynb
.. _with answers: https://github.com/Unidata/unidata-users-workshop/blob/master/notebooks/xray-tutorial-with-answers.ipynb
.. _Nicolas Fauchereau's tutorial: http://nbviewer.iPython.org/github/nicolasfauchereau/metocean/blob/master/notebooks/xray.ipynb

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

xarray is available under the open source `Apache License`__.

__ http://www.apache.org/licenses/LICENSE-2.0.html
