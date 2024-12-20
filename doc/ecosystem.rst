.. _ecosystem:

Xarray related projects
-----------------------

Below is a list of existing open source projects that build
functionality upon xarray. See also section :ref:`internals` for more
details on how to build xarray extensions. We also maintain the
`xarray-contrib <https://github.com/xarray-contrib>`_ GitHub organization
as a place to curate projects that build upon xarray.

Geosciences
~~~~~~~~~~~

- `aospy <https://aospy.readthedocs.io>`_: Automated analysis and management of gridded climate data.
- `argopy <https://github.com/euroargodev/argopy>`_: xarray-based Argo data access, manipulation and visualisation for standard users as well as Argo experts.
- `climpred <https://climpred.readthedocs.io>`_: Analysis of ensemble forecast models for climate prediction.
- `geocube <https://corteva.github.io/geocube>`_: Tool to convert geopandas vector data into rasterized xarray data.
- `GeoWombat <https://github.com/jgrss/geowombat>`_: Utilities for analysis of remotely sensed and gridded raster data at scale (easily tame Landsat, Sentinel, Quickbird, and PlanetScope).
- `grib2io <https://github.com/NOAA-MDL/grib2io>`_: Utility to work with GRIB2 files including an xarray backend, DASK support for parallel reading in open_mfdataset, lazy loading of data, editing of GRIB2 attributes and GRIB2IO DataArray attrs, and spatial interpolation and reprojection of GRIB2 messages and GRIB2IO Datasets/DataArrays for both grid to grid and grid to stations.
- `gsw-xarray <https://github.com/DocOtak/gsw-xarray>`_: a wrapper around `gsw <https://teos-10.github.io/GSW-Python>`_ that adds CF compliant attributes when possible, units, name.
- `infinite-diff <https://github.com/spencerahill/infinite-diff>`_: xarray-based finite-differencing, focused on gridded climate/meteorology data
- `marc_analysis <https://github.com/darothen/marc_analysis>`_: Analysis package for CESM/MARC experiments and output.
- `MetPy <https://unidata.github.io/MetPy/dev/index.html>`_: A collection of tools in Python for reading, visualizing, and performing calculations with weather data.
- `MPAS-Analysis <https://mpas-dev.github.io/MPAS-Analysis>`_: Analysis for simulations produced with Model for Prediction Across Scales (MPAS) components and the Accelerated Climate Model for Energy (ACME).
- `OGGM <https://oggm.org/>`_: Open Global Glacier Model
- `Oocgcm <https://oocgcm.readthedocs.io/>`_: Analysis of large gridded geophysical datasets
- `Open Data Cube <https://www.opendatacube.org/>`_: Analysis toolkit of continental scale Earth Observation data from satellites.
- `Pangaea <https://pangaea.readthedocs.io/en/latest/>`_: xarray extension for gridded land surface & weather model output).
- `Pangeo <https://pangeo.io>`_: A community effort for big data geoscience in the cloud.
- `PyGDX <https://pygdx.readthedocs.io/en/latest/>`_: Python 3 package for
  accessing data stored in GAMS Data eXchange (GDX) files. Also uses a custom
  subclass.
- `pyinterp <https://pangeo-pyinterp.readthedocs.io/en/latest/>`_: Python 3 package for interpolating geo-referenced data used in the field of geosciences.
- `pyXpcm <https://pyxpcm.readthedocs.io>`_: xarray-based Profile Classification Modelling (PCM), mostly for ocean data.
- `Regionmask <https://regionmask.readthedocs.io/>`_: plotting and creation of masks of spatial regions
- `rioxarray <https://corteva.github.io/rioxarray>`_: geospatial xarray extension powered by rasterio
- `salem <https://salem.readthedocs.io>`_: Adds geolocalised subsetting, masking, and plotting operations to xarray's data structures via accessors.
- `SatPy <https://satpy.readthedocs.io/>`_ : Library for reading and manipulating meteorological remote sensing data and writing it to various image and data file formats.
- `SARXarray <https://tudelftgeodesy.github.io/sarxarray/>`_: xarray extension for reading and processing large Synthetic Aperture Radar (SAR) data stacks.
- `Spyfit <https://spyfit.readthedocs.io/en/master/>`_: FTIR spectroscopy of the atmosphere
- `windspharm <https://ajdawson.github.io/windspharm/index.html>`_: Spherical
  harmonic wind analysis in Python.
- `wradlib <https://wradlib.org/>`_: An Open Source Library for Weather Radar Data Processing.
- `wrf-python <https://wrf-python.readthedocs.io/>`_: A collection of diagnostic and interpolation routines for use with output of the Weather Research and Forecasting (WRF-ARW) Model.
- `xarray-regrid <https://github.com/EXCITED-CO2/xarray-regrid>`_: xarray extension for regridding rectilinear data.
- `xarray-simlab <https://xarray-simlab.readthedocs.io>`_: xarray extension for computer model simulations.
- `xarray-spatial <https://xarray-spatial.org/>`_: Numba-accelerated raster-based spatial processing tools (NDVI, curvature, zonal-statistics, proximity, hillshading, viewshed, etc.)
- `xarray-topo <https://xarray-topo.readthedocs.io/>`_: xarray extension for topographic analysis and modelling.
- `xbpch <https://github.com/darothen/xbpch>`_: xarray interface for bpch files.
- `xCDAT <https://xcdat.readthedocs.io/>`_: An extension of xarray for climate data analysis on structured grids.
- `xclim <https://xclim.readthedocs.io/>`_: A library for calculating climate science indices with unit handling built from xarray and dask.
- `xESMF <https://pangeo-xesmf.readthedocs.io/>`_: Universal regridder for geospatial data.
- `xgcm <https://xgcm.readthedocs.io/>`_: Extends the xarray data model to understand finite volume grid cells (common in General Circulation Models) and provides interpolation and difference operations for such grids.
- `xmitgcm <https://xmitgcm.readthedocs.io/>`_: a python package for reading `MITgcm <https://mitgcm.org/>`_ binary MDS files into xarray data structures.
- `xnemogcm <https://github.com/rcaneill/xnemogcm/>`_: a package to read `NEMO <https://nemo-ocean.eu/>`_ output files and add attributes to interface with xgcm.

Machine Learning
~~~~~~~~~~~~~~~~
- `ArviZ <https://arviz-devs.github.io/arviz/>`_: Exploratory analysis of Bayesian models, built on top of xarray.
- `Darts <https://github.com/unit8co/darts/>`_: User-friendly modern machine learning for time series in Python.
- `Elm <https://ensemble-learning-models.readthedocs.io>`_: Parallel machine learning on xarray data structures
- `sklearn-xarray (1) <https://phausamann.github.io/sklearn-xarray>`_: Combines scikit-learn and xarray (1).
- `sklearn-xarray (2) <https://sklearn-xarray.readthedocs.io/en/latest/>`_: Combines scikit-learn and xarray (2).
- `xbatcher <https://xbatcher.readthedocs.io>`_: Batch Generation from Xarray Datasets.

Other domains
~~~~~~~~~~~~~
- `ptsa <https://pennmem.github.io/ptsa/html/index.html>`_: EEG Time Series Analysis
- `pycalphad <https://pycalphad.org/docs/latest/>`_: Computational Thermodynamics in Python
- `pyomeca <https://pyomeca.github.io/>`_: Python framework for biomechanical analysis

Extend xarray capabilities
~~~~~~~~~~~~~~~~~~~~~~~~~~
- `Collocate <https://github.com/cistools/collocate>`_: Collocate xarray trajectories in arbitrary physical dimensions
- `eofs <https://ajdawson.github.io/eofs/>`_: EOF analysis in Python.
- `hypothesis-gufunc <https://hypothesis-gufunc.readthedocs.io/en/latest/>`_: Extension to hypothesis. Makes it easy to write unit tests with xarray objects as input.
- `ntv-pandas <https://github.com/loco-philippe/ntv-pandas>`_ : A tabular analyzer and a semantic, compact and reversible converter for multidimensional and tabular data
- `nxarray <https://github.com/nxarray/nxarray>`_: NeXus input/output capability for xarray.
- `xarray-compare <https://github.com/astropenguin/xarray-compare>`_: xarray extension for data comparison.
- `xarray-dataclasses <https://github.com/astropenguin/xarray-dataclasses>`_: xarray extension for typed DataArray and Dataset creation.
- `xarray_einstats <https://xarray-einstats.readthedocs.io>`_: Statistics, linear algebra and einops for xarray
- `xarray_extras <https://github.com/crusaderky/xarray_extras>`_: Advanced algorithms for xarray objects (e.g. integrations/interpolations).
- `xeofs <https://github.com/nicrie/xeofs>`_: PCA/EOF analysis and related techniques, integrated with xarray and Dask for efficient handling of large-scale data.
- `xpublish <https://xpublish.readthedocs.io/>`_: Publish Xarray Datasets via a Zarr compatible REST API.
- `xrft <https://github.com/rabernat/xrft>`_: Fourier transforms for xarray data.
- `xr-scipy <https://xr-scipy.readthedocs.io>`_: A lightweight scipy wrapper for xarray.
- `X-regression <https://github.com/kuchaale/X-regression>`_: Multiple linear regression from Statsmodels library coupled with Xarray library.
- `xskillscore <https://github.com/xarray-contrib/xskillscore>`_: Metrics for verifying forecasts.
- `xyzpy <https://xyzpy.readthedocs.io>`_: Easily generate high dimensional data, including parallelization.

Visualization
~~~~~~~~~~~~~
- `datashader <https://datashader.org>`_, `geoviews <https://geoviews.org>`_, `holoviews <https://holoviews.org/>`_, : visualization packages for large data.
- `hvplot <https://hvplot.pyviz.org/>`_ : A high-level plotting API for the PyData ecosystem built on HoloViews.
- `psyplot <https://psyplot.readthedocs.io>`_: Interactive data visualization with python.
- `xarray-leaflet <https://github.com/davidbrochart/xarray_leaflet>`_: An xarray extension for tiled map plotting based on ipyleaflet.
- `xtrude <https://github.com/davidbrochart/xtrude>`_: An xarray extension for 3D terrain visualization based on pydeck.
- `pyvista-xarray <https://github.com/pyvista/pyvista-xarray>`_: xarray DataArray accessor for 3D visualization with `PyVista <https://github.com/pyvista/pyvista>`_ and DataSet engines for reading VTK data formats.

Non-Python projects
~~~~~~~~~~~~~~~~~~~
- `xframe <https://github.com/xtensor-stack/xframe>`_: C++ data structures inspired by xarray.
- `AxisArrays <https://github.com/JuliaArrays/AxisArrays.jl>`_, `NamedArrays <https://github.com/davidavdav/NamedArrays.jl>`_ and `YAXArrays.jl <https://github.com/JuliaDataCubes/YAXArrays.jl>`_: similar data structures for Julia.

More projects can be found at the `"xarray" Github topic <https://github.com/topics/xarray>`_.
