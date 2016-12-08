Frequently Asked Questions
==========================

Why is pandas not enough?
-------------------------

pandas, thanks to its unrivaled speed and flexibility, has emerged
as the premier python package for working with labeled arrays. So why are we
contributing to further fragmentation__ in the ecosystem for
working with data arrays in Python?

__ http://wesmckinney.com/blog/a-roadmap-for-rich-scientific-data-structures-in-python/

Sometimes, we really want to work with collections of higher dimensional arrays
(`ndim > 2`), or arrays for which the order of dimensions (e.g., columns vs
rows) shouldn't really matter. For example, climate and weather data is often
natively expressed in 4 or more dimensions: time, x, y and z.

Pandas does support `N-dimensional panels`__, but the implementation
is very limited:

__ http://pandas.pydata.org/pandas-docs/stable/dsintro.html#panelnd-experimental

  - You need to create a new factory type for each dimensionality.
  - You can't do math between NDPanels with different dimensionality.
  - Each dimension in a NDPanel has a name (e.g., 'labels', 'items',
    'major_axis', etc.) but the dimension names refer to order, not their
    meaning. You can't specify an operation as to be applied along the "time"
    axis.

Fundamentally, the N-dimensional panel is limited by its context in pandas's
tabular model, which treats a 2D ``DataFrame`` as a collections of 1D
``Series``, a 3D ``Panel`` as a collection of 2D ``DataFrame``, and so on. In
my experience, it usually easier to work with a DataFrame with a hierarchical
index rather than to use higher dimensional (*N > 3*) data structures in
pandas.

Another use case is handling collections of arrays with different numbers of
dimensions. For example, suppose you have a 2D array and a handful of
associated 1D arrays that share one of the same axes. Storing these in one
pandas object is possible but awkward -- you can either upcast all the 1D
arrays to 2D and store everything in a ``Panel``, or put everything in a
``DataFrame``, where the first few columns have a different meaning than the
other columns. In contrast, this sort of data structure fits very naturally in
an xarray ``Dataset``.

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
one-dimensional, stick with pandas, which is a more developed toolkit for doing
data analysis in Python.


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
does not yet support all of netCDF4-python's features, such as writing to
netCDF groups or modifying files on-disk.

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
and CDAT have some great domain specific functionality, and we would love to
have support for converting their native objects to and from xarray (see
:issue:`37` and :issue:`133`)

How should I cite xarray?
-------------------------

If you are using xarray and would like to cite it in academic publication, we
would certainly appreciate it. We recommend two citations.

  1. At a minimum, we recommend citing the xarray overview journal article,
     submitted to the Journal of Open Research Software.

     - Hoyer, S., Hamman, J. (In review). Xarray: N-D labeled arrays and
       datasets in Python. Journal of Open Research Software.

       Here’s an example of a BibTeX entry::

           @article{hoyer2016xarray,
             title   = {xarray: {N-D} labeled arrays and datasets in {Python}},
             author  = {Hoyer, S. and J. Hamman},
             journal = {In review, J. Open Res. Software},
             year    = {2016}
           }

  2. You may also want to cite a specific version of the xarray package. We
     provide a `Zenodo`__ citation and DOI for this purpose:

        .. image:: https://zenodo.org/badge/7766/pydata/xarray.svg
            :target: https://zenodo.org/badge/latestdoi/7766/pydata/xarray.

       An example BibTeX entry::

           @misc{xarray_v0_8_0,
                 author = {Stephan Hoyer and Clark Fitzgerald and Joe Hamman and others},
                 title  = {xarray: v0.8.0},
                 month  = aug,
                 year   = 2016,
                 doi    = {10.5281/zenodo.59499},
                 url    = {http://dx.doi.org/10.5281/zenodo.59499}
                }

__ https://zenodo.org/badge/latestdoi/7766/pydata/xarray
