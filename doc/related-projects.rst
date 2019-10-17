.. _related-projects:

Xarray related projects
-----------------------

Here below is a list of existing open source projects that build
functionality upon xarray. See also section :ref:`internals` for more
details on how to build xarray extensions.

Geosciences
~~~~~~~~~~~

- `aospy <https://aospy.readthedocs.io>`_: Automated analysis and management of gridded climate data.
- `climpred <https://climpred.readthedocs.io>`_: Analysis of ensemble forecast models for climate prediction.
- `geocube <https://corteva.github.io/geocube>`_: Tool to convert geopandas vector data into rasterized xarray data.
- `infinite-diff <https://github.com/spencerahill/infinite-diff>`_: xarray-based finite-differencing, focused on gridded climate/meterology data
- `marc_analysis <https://github.com/darothen/marc_analysis>`_: Analysis package for CESM/MARC experiments and output.
- `MetPy <https://unidata.github.io/MetPy/dev/index.html>`_: A collection of tools in Python for reading, visualizing, and performing calculations with weather data.
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
- `rioxarray <https://corteva.github.io/rioxarray>`_: geospatial xarray extension powered by rasterio
- `salem <https://salem.readthedocs.io>`_: Adds geolocalised subsetting, masking, and plotting operations to xarray's data structures via accessors.
- `SatPy <https://satpy.readthedocs.io/>`_ : Library for reading and manipulating meteorological remote sensing data and writing it to various image and data file formats.
- `Spyfit <https://spyfit.readthedocs.io/en/master/>`_: FTIR spectroscopy of the atmosphere
- `windspharm <https://ajdawson.github.io/windspharm/index.html>`_: Spherical
  harmonic wind analysis in Python.
- `wrf-python <https://wrf-python.readthedocs.io/>`_: A collection of diagnostic and interpolation routines for use with output of the Weather Research and Forecasting (WRF-ARW) Model.
- `xarray-simlab <https://xarray-simlab.readthedocs.io>`_: xarray extension for computer model simulations.
- `xarray-topo <https://gitext.gfz-potsdam.de/sec55-public/xarray-topo>`_: xarray extension for topographic analysis and modelling.
- `xbpch <https://github.com/darothen/xbpch>`_: xarray interface for bpch files.
- `xclim <https://xclim.readthedocs.io/>`_: A library for calculating climate science indices with unit handling built from xarray and dask.
- `xESMF <https://xesmf.readthedocs.io>`_: Universal Regridder for Geospatial Data.
- `xgcm <https://xgcm.readthedocs.io/>`_: Extends the xarray data model to understand finite volume grid cells (common in General Circulation Models) and provides interpolation and difference operations for such grids.
- `xmitgcm <http://xgcm.readthedocs.io/>`_: a python package for reading `MITgcm <http://mitgcm.org/>`_ binary MDS files into xarray data structures.
- `xshape <https://xshape.readthedocs.io/>`_: Tools for working with shapefiles, topographies, and polygons in xarray.

Machine Learning
~~~~~~~~~~~~~~~~
- `ArviZ <https://arviz-devs.github.io/arviz/>`_: Exploratory analysis of Bayesian models, built on top of xarray.
- `Elm <https://ensemble-learning-models.readthedocs.io>`_: Parallel machine learning on xarray data structures
- `sklearn-xarray (1) <https://phausamann.github.io/sklearn-xarray>`_: Combines scikit-learn and xarray (1).
- `sklearn-xarray (2) <https://sklearn-xarray.readthedocs.io/en/latest/>`_: Combines scikit-learn and xarray (2).

Other domains
~~~~~~~~~~~~~
- `ptsa <https://pennmem.github.io/ptsa_new/html/index.html>`_: EEG Time Series Analysis
- `pycalphad <https://pycalphad.org/docs/latest/>`_: Computational Thermodynamics in Python

Extend xarray capabilities
~~~~~~~~~~~~~~~~~~~~~~~~~~
- `Collocate <https://github.com/cistools/collocate>`_: Collocate xarray trajectories in arbitrary physical dimensions
- `eofs <https://ajdawson.github.io/eofs/>`_: EOF analysis in Python.
- `hypothesis-gufunc <https://hypothesis-gufunc.readthedocs.io/en/latest/>`_: Extension to hypothesis. Makes it easy to write unit tests with xarray objects as input.
- `xarray_extras <https://github.com/crusaderky/xarray_extras>`_: Advanced algorithms for xarray objects (e.g. integrations/interpolations).
- `xrft <https://github.com/rabernat/xrft>`_: Fourier transforms for xarray data.
- `xr-scipy <https://xr-scipy.readthedocs.io>`_: A lightweight scipy wrapper for xarray.
- `X-regression <https://github.com/kuchaale/X-regression>`_: Multiple linear regression from Statsmodels library coupled with Xarray library.
- `xskillscore <https://github.com/raybellwaves/xskillscore>`_: Metrics for verifying forecasts.
- `xyzpy <http://xyzpy.readthedocs.io>`_: Easily generate high dimensional data, including parallelization.

Visualization
~~~~~~~~~~~~~
- `Datashader <https://datashader.org>`_, `geoviews <http://geo.holoviews.org>`_, `holoviews <http://holoviews.org/>`_, : visualization packages for large data.
- `hvplot <https://hvplot.pyviz.org/>`_ : A high-level plotting API for the PyData ecosystem built on HoloViews.
- `psyplot <https://psyplot.readthedocs.io>`_: Interactive data visualization with python.

Non-Python projects
~~~~~~~~~~~~~~~~~~~
- `xframe <https://github.com/QuantStack/xframe>`_: C++ data structures inspired by xarray.
- `AxisArrays <https://github.com/JuliaArrays/AxisArrays.jl>`_ and
  `NamedArrays <https://github.com/davidavdav/NamedArrays.jl>`_: similar data structures for Julia.

More projects can be found at the `"xarray" Github topic <https://github.com/topics/xarray>`_.
