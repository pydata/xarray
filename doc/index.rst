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

.. _NumPy: https://www.numpy.org
.. _pandas: https://pandas.pydata.org
.. _dask: https://dask.org
.. _netCDF: https://www.unidata.ucar.edu/software/netcdf


.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: For users

   Getting Started <getting-started-guide/index>
   User Guide <user-guide/index>
   Gallery <gallery>
   Tutorials & Videos <tutorials-and-videos>
   API Reference <api>
   How do I ... <howdoi>
   Ecosystem <ecosystem>

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: For developers/contributors

   Contributing Guide <contributing>
   Xarray Internals <internals/index>
   Development Roadmap <roadmap>
   Team <team>
   Developers Meeting <developers-meeting>
   What’s New <whats-new>
   GitHub repository <https://github.com/pydata/xarray>

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Community

   GitHub discussions <https://github.com/pydata/xarray/discussions>
   StackOverflow <https://stackoverflow.com/questions/tagged/python-xarray>




Get in touch
------------

- If you have a question like "How do I concatenate a list of datasets?", ask on `GitHub discussions`_ or `StackOverflow`_.
  Please include a self-contained reproducible example if possible.
- Report bugs, suggest features or view the source code `on GitHub`_.
- For less well defined questions or ideas, or to announce other projects of
  interest to xarray users, use `GitHub discussions`_ or the `mailing list`_.

.. _StackOverFlow: https://stackoverflow.com/questions/tagged/python-xarray
.. _Github discussions: https://github.com/pydata/xarray/discussions
.. _mailing list: https://groups.google.com/forum/#!forum/xarray
.. _on GitHub: https://github.com/pydata/xarray

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

Xarray is an evolution of an internal tool developed at `The Climate
Corporation`__. It was originally written by Climate Corp researchers Stephan
Hoyer, Alex Kleeman and Eugene Brevdo and was released as open source in
May 2014. The project was renamed from "xray" in January 2016. Xarray became a
fiscally sponsored project of NumFOCUS_ in August 2018.

__ https://climate.com/
.. _NumFOCUS: https://numfocus.org

License
-------

Xarray is available under the open source `Apache License`__.

__ https://www.apache.org/licenses/LICENSE-2.0.html
