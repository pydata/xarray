.. _faq:

Frequently Asked Questions
==========================

.. ipython:: python
   :suppress:

    import numpy as np
    import pandas as pd
    import xarray as xr
    np.random.seed(123456)

Why is pandas not enough?
-------------------------

pandas is a fantastic library for analysis of low-dimensional labelled data -
if it can be sensibly described as "rows and columns", pandas is probably the
right choice.  However, sometimes we want to use higher dimensional arrays
(`ndim > 2`), or arrays for which the order of dimensions (e.g., columns vs
rows) shouldn't really matter. For example, climate and weather data is often
natively expressed in 4 or more dimensions: time, x, y and z.

Pandas has historically supported N-dimensional panels, but deprecated them in
version 0.20 in favor of Xarray data structures.  There are now built-in methods
on both sides to convert between pandas and Xarray, allowing for more focussed
development effort.  Xarray objects have a much richer model of dimensionality -
if you were using Panels:

- You need to create a new factory type for each dimensionality.
- You can't do math between NDPanels with different dimensionality.
- Each dimension in a NDPanel has a name (e.g., 'labels', 'items',
  'major_axis', etc.) but the dimension names refer to order, not their
  meaning. You can't specify an operation as to be applied along the "time"
  axis.
- You often have to manually convert collections of pandas arrays
  (Series, DataFrames, etc) to have the same number of dimensions.
  In contrast, this sort of data structure fits very naturally in an
  xarray ``Dataset``.

You can :ref:`read about switching from Panels to Xarray here <panel transition>`.
Pandas gets a lot of things right, but scientific users need fully multi-
dimensional data structures.


How do xarray data structures differ from those found in pandas?
----------------------------------------------------------------

The main distinguishing feature of xarray's ``DataArray`` over labeled arrays in
pandas is that dimensions can have names (e.g., "time", "latitude",
"longitude"). Names are much easier to keep track of than axis numbers, and
xarray uses dimension names for indexing, aggregation and broadcasting. Not only
can you write ``x.sel(time='2000-01-01')`` and  ``x.mean(dim='time')``, but
operations like ``x - x.mean(dim='time')`` always work, no matter the order
of the "time" dimension. You never need to reshape arrays (e.g., with
``np.newaxis``) to align them for arithmetic operations in xarray.


Should I use xarray instead of pandas?
--------------------------------------

It's not an either/or choice! xarray provides robust support for converting
back and forth between the tabular data-structures of pandas and its own
multi-dimensional data-structures.

That said, you should only bother with xarray if some aspect of data is
fundamentally multi-dimensional. If your data is unstructured or
one-dimensional, stick with pandas.


Why don't aggregations return Python scalars?
---------------------------------------------

xarray tries hard to be self-consistent: operations on a ``DataArray`` (resp.
``Dataset``) return another ``DataArray`` (resp. ``Dataset``) object. In
particular, operations returning scalar values (e.g. indexing or aggregations
like ``mean`` or ``sum`` applied to all axes) will also return xarray objects.

Unfortunately, this means we sometimes have to explicitly cast our results from
xarray when using them in other libraries. As an illustration, the following
code fragment

.. ipython:: python

    arr = xr.DataArray([1, 2, 3])
    pd.Series({'x': arr[0], 'mean': arr.mean(), 'std': arr.std()})

does not yield the pandas DataFrame we expected. We need to specify the type
conversion ourselves:

.. ipython:: python

    pd.Series({'x': arr[0], 'mean': arr.mean(), 'std': arr.std()}, dtype=float)

Alternatively, we could use the ``item`` method or the ``float`` constructor to
convert values one at a time

.. ipython:: python

    pd.Series({'x': arr[0].item(), 'mean': float(arr.mean())})


.. _approach to metadata:

What is your approach to metadata?
----------------------------------

We are firm believers in the power of labeled data! In addition to dimensions
and coordinates, xarray supports arbitrary metadata in the form of global
(Dataset) and variable specific (DataArray) attributes (``attrs``).

Automatic interpretation of labels is powerful but also reduces flexibility.
With xarray, we draw a firm line between labels that the library understands
(``dims`` and ``coords``) and labels for users and user code (``attrs``). For
example, we do not automatically interpret and enforce units or `CF
conventions`_. (An exception is serialization to and from netCDF files.)

.. _CF conventions: http://cfconventions.org/latest.html

An implication of this choice is that we do not propagate ``attrs`` through
most operations unless explicitly flagged (some methods have a ``keep_attrs``
option). Similarly, xarray does not check for conflicts between ``attrs`` when
combining arrays and datasets, unless explicitly requested with the option
``compat='identical'``. The guiding principle is that metadata should not be
allowed to get in the way.


What other netCDF related Python libraries should I know about?
---------------------------------------------------------------

`netCDF4-python`__ provides a lower level interface for working with
netCDF and OpenDAP datasets in Python. We use netCDF4-python internally in
xarray, and have contributed a number of improvements and fixes upstream. xarray
does not yet support all of netCDF4-python's features, such as modifying files
on-disk.

__ https://github.com/Unidata/netcdf4-python

Iris_ (supported by the UK Met office) provides similar tools for in-
memory manipulation of labeled arrays, aimed specifically at weather and
climate data needs. Indeed, the Iris :py:class:`~iris.cube.Cube` was direct
inspiration for xarray's :py:class:`~xarray.DataArray`. xarray and Iris take very
different approaches to handling metadata: Iris strictly interprets
`CF conventions`_. Iris particularly shines at mapping, thanks to its
integration with Cartopy_.

.. _Iris: http://scitools.org.uk/iris/
.. _Cartopy: http://scitools.org.uk/cartopy/docs/latest/

`UV-CDAT`__ is another Python library that implements in-memory netCDF-like
variables and `tools for working with climate data`__.

__ http://uvcdat.llnl.gov/
__ http://drclimate.wordpress.com/2014/01/02/a-beginners-guide-to-scripting-with-uv-cdat/

We think the design decisions we have made for xarray (namely, basing it on
pandas) make it a faster and more flexible data analysis tool. That said, Iris
and CDAT have some great domain specific functionality, and xarray includes
methods for converting back and forth between xarray and these libraries. See
:py:meth:`~xarray.DataArray.to_iris` and :py:meth:`~xarray.DataArray.to_cdms2`
for more details.

.. _faq.other_projects:

What other projects leverage xarray?
------------------------------------

Here are several existing libraries that build functionality upon xarray.

Geosciences
~~~~~~~~~~~

- `aospy <https://aospy.readthedocs.io>`_: Automated analysis and management of gridded climate data.
- `infinite-diff <https://github.com/spencerahill/infinite-diff>`_: xarray-based finite-differencing, focused on gridded climate/meterology data
- `marc_analysis <https://github.com/darothen/marc_analysis>`_: Analysis package for CESM/MARC experiments and output.
- `MPAS-Analysis <http://mpas-analysis.readthedocs.io>`_: Analysis for simulations produced with Model for Prediction Across Scales (MPAS) components and the Accelerated Climate Model for Energy (ACME).
- `OGGM <http://oggm.org/>`_: Open Global Glacier Model
- `Oocgcm <https://oocgcm.readthedocs.io/>`_: Analysis of large gridded geophysical datasets
- `Open Data Cube <https://www.opendatacube.org/>`_: Analysis toolkit of continental scale Earth Observation data from satellites.
- `Pangaea: <https://pangaea.readthedocs.io/en/latest/>`_: xarray extension for gridded land surface & weather model output).
- `Pangeo <https://pangeo-data.github.io>`_: A community effort for big data geoscience in the cloud.
- `PyGDX <https://pygdx.readthedocs.io/en/latest/>`_: Python 3 package for
  accessing data stored in GAMS Data eXchange (GDX) files. Also uses a custom
  subclass.
- `Regionmask <https://regionmask.readthedocs.io/>`_: plotting and creation of masks of spatial regions
- `salem <https://salem.readthedocs.io>`_: Adds geolocalised subsetting, masking, and plotting operations to xarray's data structures via accessors.
- `Spyfit <https://spyfit.readthedocs.io/en/master/>`_: FTIR spectroscopy of the atmosphere
- `windspharm <https://ajdawson.github.io/windspharm/index.html>`_: Spherical
  harmonic wind analysis in Python.
- `wrf-python <https://wrf-python.readthedocs.io/>`_: A collection of diagnostic and interpolation routines for use with output of the Weather Research and Forecasting (WRF-ARW) Model.
- `xarray-simlab <https://xarray-simlab.readthedocs.io>`_: xarray extension for computer model simulations.
- `xarray-topo <https://gitext.gfz-potsdam.de/sec55-public/xarray-topo>`_: xarray extension for topographic analysis and modelling.
- `xbpch <https://github.com/darothen/xbpch>`_: xarray interface for bpch files.
- `xESMF <https://xesmf.readthedocs.io>`_: Universal Regridder for Geospatial Data.
- `xgcm <https://xgcm.readthedocs.io/>`_: Extends the xarray data model to understand finite volume grid cells (common in General Circulation Models) and provides interpolation and difference operations for such grids.
- `xmitgcm <http://xgcm.readthedocs.io/>`_: a python package for reading `MITgcm <http://mitgcm.org/>`_ binary MDS files into xarray data structures.
- `xshape <https://xshape.readthedocs.io/>`_: Tools for working with shapefiles, topographies, and polygons in xarray.

Machine Learning
~~~~~~~~~~~~~~~~
- `cesium <http://cesium-ml.org/>`_: machine learning for time series analysis
- `Elm <https://ensemble-learning-models.readthedocs.io>`_: Parallel machine learning on xarray data structures
- `sklearn-xarray (1) <https://phausamann.github.io/sklearn-xarray>`_: Combines scikit-learn and xarray (1).
- `sklearn-xarray (2) <https://sklearn-xarray.readthedocs.io/en/latest/>`_: Combines scikit-learn and xarray (2).

Extend xarray capabilities
~~~~~~~~~~~~~~~~~~~~~~~~~~
- `Collocate <https://github.com/cistools/collocate>`_: Collocate xarray trajectories in arbitrary physical dimensions
- `eofs <https://ajdawson.github.io/eofs/>`_: EOF analysis in Python.
- `xarray_extras <https://github.com/crusaderky/xarray_extras>`_: Advanced algorithms for xarray objects (e.g. intergrations/interpolations).
- `xrft <https://github.com/rabernat/xrft>`_: Fourier transforms for xarray data.
- `xr-scipy <https://xr-scipy.readthedocs.io>`_: A lightweight scipy wrapper for xarray.
- `X-regression <https://github.com/kuchaale/X-regression>`_: Multiple linear regression from Statsmodels library coupled with Xarray library.
- `xyzpy <http://xyzpy.readthedocs.io>`_: Easily generate high dimensional data, including parallelization.

Visualization
~~~~~~~~~~~~~
- `Datashader <https://datashader.org>`_, `geoviews <http://geo.holoviews.org>`_, `holoviews <http://holoviews.org/>`_, : visualization packages for large data
- `psyplot <https://psyplot.readthedocs.io>`_: Interactive data visualization with python.

Other
~~~~~
- `ptsa <https://pennmem.github.io/ptsa_new/html/index.html>`_: EEG Time Series Analysis
- `pycalphad <https://pycalphad.org/docs/latest/>`_: Computational Thermodynamics in Python

More projects can be found at the `"xarray" Github topic <https://github.com/topics/xarray>`_.

How should I cite xarray?
-------------------------

If you are using xarray and would like to cite it in academic publication, we
would certainly appreciate it. We recommend two citations.

  1. At a minimum, we recommend citing the xarray overview journal article,
     published in the Journal of Open Research Software.

     - Hoyer, S. & Hamman, J., (2017). xarray: N-D labeled Arrays and
       Datasets in Python. Journal of Open Research Software. 5(1), p.10.
       DOI: http://doi.org/10.5334/jors.148

       Here’s an example of a BibTeX entry::

           @article{hoyer2017xarray,
             title     = {xarray: {N-D} labeled arrays and datasets in {Python}},
             author    = {Hoyer, S. and J. Hamman},
             journal   = {Journal of Open Research Software},
             volume    = {5},
             number    = {1},
             year      = {2017},
             publisher = {Ubiquity Press},
             doi       = {10.5334/jors.148},
             url       = {http://doi.org/10.5334/jors.148}
           }

  2. You may also want to cite a specific version of the xarray package. We
     provide a `Zenodo citation and DOI <https://doi.org/10.5281/zenodo.598201>`_
     for this purpose:

        .. image:: https://zenodo.org/badge/doi/10.5281/zenodo.598201.svg
           :target: https://doi.org/10.5281/zenodo.598201

       An example BibTeX entry::

           @misc{xarray_v0_8_0,
                 author = {Stephan Hoyer and Clark Fitzgerald and Joe Hamman and others},
                 title  = {xarray: v0.8.0},
                 month  = aug,
                 year   = 2016,
                 doi    = {10.5281/zenodo.59499},
                 url    = {https://doi.org/10.5281/zenodo.59499}
                }
