.. _faq:

Frequently Asked Questions
==========================

.. ipython:: python
   :suppress:

    import numpy as np
    import pandas as pd
    import xarray as xr
    np.random.seed(123456)


Your documentation keeps mentioning pandas. What is pandas?
-----------------------------------------------------------

pandas_ is a very popular data analysis package in Python
with wide usage in many fields. Our API is heavily inspired by pandas —
this is why there are so many references to pandas.

.. _pandas: https://pandas.pydata.org


Do I need to know pandas to use xarray?
---------------------------------------

No! Our API is heavily inspired by pandas so while knowing pandas will let you
become productive more quickly, knowledge of pandas is not necessary to use xarray.


Should I use xarray instead of pandas?
--------------------------------------

It's not an either/or choice! xarray provides robust support for converting
back and forth between the tabular data-structures of pandas and its own
multi-dimensional data-structures.

That said, you should only bother with xarray if some aspect of data is
fundamentally multi-dimensional. If your data is unstructured or
one-dimensional, pandas is usually the right choice: it has better performance
for common operations such as ``groupby`` and you'll find far more usage
examples online.


Why is pandas not enough?
-------------------------

pandas is a fantastic library for analysis of low-dimensional labelled data -
if the data can be sensibly described as "rows and columns", pandas is probably the
right choice.  However, xarray has some distinct advantages: higher dimensionality
and an explicit, production-ready API.

Higher dimensionality
~~~~~~~~~~~~~~~~~~~~~

Sometimes we want to use higher dimensional arrays, or arrays for 
which the names of dimensions have some semantic significance we want to retain.
if it can be sensibly described as "rows and columns", pandas is probably the
right choice.  However, sometimes we want to use higher dimensional arrays
(`ndim > 2`), or arrays for which the order of dimensions (e.g., columns vs
rows) shouldn't really matter. For example, the images of a movie can be
natively represented as an array with four dimensions: time, row, column and
color.

Pandas has historically supported N-dimensional panels, but deprecated them in
version 0.20 in favor of Xarray data structures. There are now built-in methods
on both sides to convert between pandas and Xarray, allowing for more focused
development effort. Xarray objects have a much richer model of dimensionality -
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
Pandas gets a lot of things right, but many science, engineering and complex
analytics use cases need fully multi-dimensional data structures.

For example, consider the images of a movie, represented as an array of pixels with 
four dimensions: time, row, column and color; xarray makes working with this 
much easier than if it were stacked into two dimensions:

  - *Name not position*: The dimensions are better represented by their name 
    than their position. By allowing us to use their names, our code becomes 
    easier to write and much easier to read; 
    e.g. ``da.isel(time=0)`` rather than ``da[:, :, 0]``.

  - *Broadcasting*: Xarray can broadcast operations to an array, meaning it
    applies the operation similarly regardless of the number of dimensions. 
    This allows our code to be more abstract, requiring far fewer special cases. 

    For example, ``da - 1`` would make every pixel slightly darker, whether 
    we're dealing with a single or multiple frames, color or grayscale, etc. 
    
    Similarly ``da - da.mean(dim=['row', 'column'])`` would normalize the values 
    based on the mean value in its width and height, again regardless of whether
    we have a single or multiple frames, color or grayscale, etc.

  - *Alignment*: Xarray can package arrays which share some, but not all, of
    their dimensions in a dataset and align the dimensions they do share. 
    For example, if we calculated the brightness of our movie images
    using ``da.mean(dim=['row', 'column', 'color'])``, we'd have an array with a time
    dimension. When we add that to a dataset and select the first point in time,
    ``ds.isel(time=0)``, we'll receive both the frame and the first brightness
    value. 

    Even following best-pracitices around tidy tabular data, this approach is
    very difficult or at least unintuitive to replicate using tabular data.

Pandas previously supported N-dimensional panels, but started removing them in
version 0.20 in favor of xarray data structures. There are now built-in methods
on both sides to convert between pandas and xarray, allowing for more focused
development effort.  You can :ref:`read about switching from Panels to Xarray 
here <panel transition>`.

Explicit API
~~~~~~~~~~~~

pandas has the benefit of almost a decade of cumulative improvements, and allows
a number of . Along with that accumulated experience comes 
# FIXME finish


Should I use xarray instead of pandas?
--------------------------------------

It's not an either/or choice! xarray provides robust support for converting
back and forth between the tabular data-structures of pandas and its own
multi-dimensional data-structures.

That said, you should only bother with xarray if some aspect of data is
fundamentally multi-dimensional. If your data is unstructured,
one-dimensional, or `tidy <https://vita.had.co.nz/papers/tidy-data.pdf >`_,
pandas is usually the right choice: it has better performance
for common operations such as ``groupby`` and you'll find far more usage
examples online.


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
option, and there is a global flag for setting this to be always True or
False). Similarly, xarray does not check for conflicts between ``attrs`` when
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

What other projects leverage xarray?
------------------------------------

See section :ref:`related-projects`.

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

.. _public api:

What parts of xarray are considered public API?
-----------------------------------------------

As a rule, only functions/methods documented in our :ref:`api` are considered
part of xarray's public API. Everything else (in particular, everything in
``xarray.core`` that is not also exposed in the top level ``xarray`` namespace)
is considered a private implementation detail that may change at any time.

Objects that exist to facilitate xarray's fluent interface on ``DataArray`` and
``Dataset`` objects are a special case. For convenience, we document them in
the API docs, but only their methods and the ``DataArray``/``Dataset``
methods/properties to construct them (e.g., ``.plot()``, ``.groupby()``,
``.str``) are considered public API. Constructors and other details of the
internal classes used to implemented them (i.e.,
``xarray.plot.plotting._PlotMethods``, ``xarray.core.groupby.DataArrayGroupBy``,
``xarray.core.accessor_str.StringAccessor``) are not.
