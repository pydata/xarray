.. currentmodule:: xarray
.. _io:

Reading and writing files
=========================

Xarray supports direct serialization and IO to several file formats, from
simple :ref:`io.pickle` files to the more flexible :ref:`io.netcdf`
format (recommended).

.. ipython:: python
    :suppress:

    import os

    import numpy as np
    import pandas as pd
    import xarray as xr

    np.random.seed(123456)

You can read different types of files in ``xr.open_dataset`` by specifying the engine to be used:

.. code:: python

    xr.open_dataset("example.nc", engine="netcdf4")

The "engine" provides a set of instructions that tells xarray how
to read the data and pack them into a ``Dataset`` (or ``Dataarray``).
These instructions are stored in an underlying "backend".

Xarray comes with several backends that cover many common data formats.
Many more backends are available via external libraries, or you can `write your own <https://docs.xarray.dev/en/stable/internals/how-to-add-new-backend.html>`_.
This diagram aims to help you determine - based on the format of the file you'd like to read -
which type of backend you're using and how to use it.

Text and boxes are clickable for more information.
Following the diagram is detailed information on many popular backends.
You can learn more about using and developing backends in the
`Xarray tutorial JupyterBook <https://tutorial.xarray.dev/advanced/backends/backends.html>`_.

.. mermaid::
    :alt: Flowchart illustrating how to choose the right backend engine to read your data

    flowchart LR
        built-in-eng["""Is your data stored in one of these formats?
            - netCDF4 (<code>netcdf4</code>)
            - netCDF3 (<code>scipy</code>)
            - Zarr (<code>zarr</code>)
            - DODS/OPeNDAP (<code>pydap</code>)
            - HDF5 (<code>h5netcdf</code>)
            """]

        built-in("""You're in luck! Xarray bundles a backend for this format.
            Open data using <code>xr.open_dataset()</code>. We recommend
            always setting the engine you want to use.""")

        installed-eng["""One of these formats?
            - <a href='https://github.com/ecmwf/cfgrib'>GRIB (<code>cfgrib</code>)
            - <a href='https://tiledb-inc.github.io/TileDB-CF-Py/documentation/index.html'>TileDB (<code>tiledb</code>)
            - <a href='https://corteva.github.io/rioxarray/stable/getting_started/getting_started.html#rioxarray'>GeoTIFF, JPEG-2000, ESRI-hdf (<code>rioxarray</code>, via GDAL)
            - <a href='https://www.bopen.eu/xarray-sentinel-open-source-library/'>Sentinel-1 SAFE (<code>xarray-sentinel</code>)
            """]

        installed("""Install the package indicated in parentheses to your
            Python environment. Restart the kernel and use
            <code>xr.open_dataset(files, engine='rioxarray')</code>.""")

        other("""Ask around to see if someone in your data community
            has created an Xarray backend for your data type.
            If not, you may need to create your own or consider
            exporting your data to a more common format.""")

        built-in-eng -->|Yes| built-in
        built-in-eng -->|No| installed-eng

        installed-eng -->|Yes| installed
        installed-eng -->|No| other

        click built-in-eng "https://docs.xarray.dev/en/stable/getting-started-guide/faq.html#how-do-i-open-format-x-file-as-an-xarray-dataset"
        click other "https://docs.xarray.dev/en/stable/internals/how-to-add-new-backend.html"

        classDef quesNodefmt fill:#9DEEF4,stroke:#206C89,text-align:left
        class built-in-eng,installed-eng quesNodefmt

        classDef ansNodefmt fill:#FFAA05,stroke:#E37F17,text-align:left,white-space:nowrap
        class built-in,installed,other ansNodefmt

        linkStyle default font-size:20pt,color:#206C89


.. _io.netcdf:

netCDF
------

The recommended way to store xarray data structures is `netCDF`__, which
is a binary file format for self-described datasets that originated
in the geosciences. Xarray is based on the netCDF data model, so netCDF files
on disk directly correspond to :py:class:`Dataset` objects (more accurately,
a group in a netCDF file directly corresponds to a :py:class:`Dataset` object.
See :ref:`io.netcdf_groups` for more.)

NetCDF is supported on almost all platforms, and parsers exist
for the vast majority of scientific programming languages. Recent versions of
netCDF are based on the even more widely used HDF5 file-format.

__ https://www.unidata.ucar.edu/software/netcdf/

.. tip::

    If you aren't familiar with this data format, the `netCDF FAQ`_ is a good
    place to start.

.. _netCDF FAQ: https://www.unidata.ucar.edu/software/netcdf/docs/faq.html#What-Is-netCDF

Reading and writing netCDF files with xarray requires scipy, h5netcdf, or the
`netCDF4-Python`__ library to be installed. SciPy only supports reading and writing
of netCDF V3 files.

__ https://github.com/Unidata/netcdf4-python

We can save a Dataset to disk using the
:py:meth:`Dataset.to_netcdf` method:

.. ipython:: python

    ds = xr.Dataset(
        {"foo": (("x", "y"), np.random.rand(4, 5))},
        coords={
            "x": [10, 20, 30, 40],
            "y": pd.date_range("2000-01-01", periods=5),
            "z": ("x", list("abcd")),
        },
    )

    ds.to_netcdf("saved_on_disk.nc")

By default, the file is saved as netCDF4 (assuming netCDF4-Python is
installed). You can control the format and engine used to write the file with
the ``format`` and ``engine`` arguments.

.. tip::

   Using the `h5netcdf <https://github.com/h5netcdf/h5netcdf>`_  package
   by passing ``engine='h5netcdf'`` to :py:meth:`open_dataset` can
   sometimes be quicker than the default ``engine='netcdf4'`` that uses the
   `netCDF4 <https://github.com/Unidata/netcdf4-python>`_ package.


We can load netCDF files to create a new Dataset using
:py:func:`open_dataset`:

.. ipython:: python

    ds_disk = xr.open_dataset("saved_on_disk.nc")
    ds_disk

.. ipython:: python
    :suppress:

    # Close "saved_on_disk.nc", but retain the file until after closing or deleting other
    # datasets that will refer to it.
    ds_disk.close()

Similarly, a DataArray can be saved to disk using the
:py:meth:`DataArray.to_netcdf` method, and loaded
from disk using the :py:func:`open_dataarray` function. As netCDF files
correspond to :py:class:`Dataset` objects, these functions internally
convert the ``DataArray`` to a ``Dataset`` before saving, and then convert back
when loading, ensuring that the ``DataArray`` that is loaded is always exactly
the same as the one that was saved.

A dataset can also be loaded or written to a specific group within a netCDF
file. To load from a group, pass a ``group`` keyword argument to the
``open_dataset`` function. The group can be specified as a path-like
string, e.g., to access subgroup 'bar' within group 'foo' pass
'/foo/bar' as the ``group`` argument. When writing multiple groups in one file,
pass ``mode='a'`` to ``to_netcdf`` to ensure that each call does not delete the
file.

.. tip::

    It is recommended to use :py:class:`~xarray.DataTree` to represent
    hierarchical data, and to use the :py:meth:`xarray.DataTree.to_netcdf` method
    when writing hierarchical data to a netCDF file.

Data is *always* loaded lazily from netCDF files. You can manipulate, slice and subset
Dataset and DataArray objects, and no array values are loaded into memory until
you try to perform some sort of actual computation. For an example of how these
lazy arrays work, see the OPeNDAP section below.

There may be minor differences in the :py:class:`Dataset` object returned
when reading a NetCDF file with different engines.

It is important to note that when you modify values of a Dataset, even one
linked to files on disk, only the in-memory copy you are manipulating in xarray
is modified: the original file on disk is never touched.

.. tip::

    Xarray's lazy loading of remote or on-disk datasets is often but not always
    desirable. Before performing computationally intense operations, it is
    often a good idea to load a Dataset (or DataArray) entirely into memory by
    invoking the :py:meth:`Dataset.load` method.

Datasets have a :py:meth:`Dataset.close` method to close the associated
netCDF file. However, it's often cleaner to use a ``with`` statement:

.. ipython:: python

    # this automatically closes the dataset after use
    with xr.open_dataset("saved_on_disk.nc") as ds:
        print(ds.keys())

Although xarray provides reasonable support for incremental reads of files on
disk, it does not support incremental writes, which can be a useful strategy
for dealing with datasets too big to fit into memory. Instead, xarray integrates
with dask.array (see :ref:`dask`), which provides a fully featured engine for
streaming computation.

It is possible to append or overwrite netCDF variables using the ``mode='a'``
argument. When using this option, all variables in the dataset will be written
to the original netCDF file, regardless if they exist in the original dataset.


.. _io.netcdf_groups:

Groups
~~~~~~

Whilst netCDF groups can only be loaded individually as ``Dataset`` objects, a
whole file of many nested groups can be loaded as a single
:py:class:`xarray.DataTree` object. To open a whole netCDF file as a tree of groups
use the :py:func:`xarray.open_datatree` function. To save a DataTree object as a
netCDF file containing many groups, use the :py:meth:`xarray.DataTree.to_netcdf` method.


.. _netcdf.root_group.note:

.. note::
    Due to file format specifications the on-disk root group name is always ``"/"``,
    overriding any given ``DataTree`` root node name.

.. _netcdf.group.warning:

.. warning::
    ``DataTree`` objects do not follow the exact same data model as netCDF
    files, which means that perfect round-tripping is not always possible.

    In particular in the netCDF data model dimensions are entities that can
    exist regardless of whether any variable possesses them. This is in contrast
    to `xarray's data model <https://docs.xarray.dev/en/stable/user-guide/data-structures.html>`_
    (and hence :ref:`DataTree's data model <data structures>`) in which the
    dimensions of a (Dataset/Tree) object are simply the set of dimensions
    present across all variables in that dataset.

    This means that if a netCDF file contains dimensions but no variables which
    possess those dimensions, these dimensions will not be present when that
    file is opened as a DataTree object.
    Saving this DataTree object to file will therefore not preserve these
    "unused" dimensions.

.. _io.encoding:

Reading encoded data
~~~~~~~~~~~~~~~~~~~~

NetCDF files follow some conventions for encoding datetime arrays (as numbers
with a "units" attribute) and for packing and unpacking data (as
described by the "scale_factor" and "add_offset" attributes). If the argument
``decode_cf=True`` (default) is given to :py:func:`open_dataset`, xarray will attempt
to automatically decode the values in the netCDF objects according to
`CF conventions`_. Sometimes this will fail, for example, if a variable
has an invalid "units" or "calendar" attribute. For these cases, you can
turn this decoding off manually.

.. _CF conventions: https://cfconventions.org/

You can view this encoding information (among others) in the
:py:attr:`DataArray.encoding` and
:py:attr:`DataArray.encoding` attributes:

.. ipython:: python

    ds_disk["y"].encoding
    ds_disk.encoding

Note that all operations that manipulate variables other than indexing
will remove encoding information.

In some cases it is useful to intentionally reset a dataset's original encoding values.
This can be done with either the :py:meth:`Dataset.drop_encoding` or
:py:meth:`DataArray.drop_encoding` methods.

.. ipython:: python

    ds_no_encoding = ds_disk.drop_encoding()
    ds_no_encoding.encoding

.. _combining multiple files:

Reading multi-file datasets
...........................

NetCDF files are often encountered in collections, e.g., with different files
corresponding to different model runs or one file per timestamp.
Xarray can straightforwardly combine such files into a single Dataset by making use of
:py:func:`concat`, :py:func:`merge`, :py:func:`combine_nested` and
:py:func:`combine_by_coords`. For details on the difference between these
functions see :ref:`combining data`.

Xarray includes support for manipulating datasets that don't fit into memory
with dask_. If you have dask installed, you can open multiple files
simultaneously in parallel using :py:func:`open_mfdataset`::

    xr.open_mfdataset('my/files/*.nc', parallel=True)

This function automatically concatenates and merges multiple files into a
single xarray dataset.
It is the recommended way to open multiple files with xarray.
For more details on parallel reading, see :ref:`combining.multi`, :ref:`dask.io` and a
`blog post`_ by Stephan Hoyer.
:py:func:`open_mfdataset` takes many kwargs that allow you to
control its behaviour (for e.g. ``parallel``, ``combine``, ``compat``, ``join``, ``concat_dim``).
See its docstring for more details.


.. note::

    A common use-case involves a dataset distributed across a large number of files with
    each file containing a large number of variables. Commonly, a few of these variables
    need to be concatenated along a dimension (say ``"time"``), while the rest are equal
    across the datasets (ignoring floating point differences). The following command
    with suitable modifications (such as ``parallel=True``) works well with such datasets::

         xr.open_mfdataset('my/files/*.nc', concat_dim="time", combine="nested",
     	              	   data_vars='minimal', coords='minimal', compat='override')

    This command concatenates variables along the ``"time"`` dimension, but only those that
    already contain the ``"time"`` dimension (``data_vars='minimal', coords='minimal'``).
    Variables that lack the ``"time"`` dimension are taken from the first dataset
    (``compat='override'``).


.. _dask: https://www.dask.org
.. _blog post: https://stephanhoyer.com/2015/06/11/xray-dask-out-of-core-labeled-arrays/

Sometimes multi-file datasets are not conveniently organized for easy use of :py:func:`open_mfdataset`.
One can use the ``preprocess`` argument to provide a function that takes a dataset
and returns a modified Dataset.
:py:func:`open_mfdataset` will call ``preprocess`` on every dataset
(corresponding to each file) prior to combining them.


If :py:func:`open_mfdataset` does not meet your needs, other approaches are possible.
The general pattern for parallel reading of multiple files
using dask, modifying those datasets and then combining into a single ``Dataset`` is::

     def modify(ds):
         # modify ds here
         return ds


     # this is basically what open_mfdataset does
     open_kwargs = dict(decode_cf=True, decode_times=False)
     open_tasks = [dask.delayed(xr.open_dataset)(f, **open_kwargs) for f in file_names]
     tasks = [dask.delayed(modify)(task) for task in open_tasks]
     datasets = dask.compute(tasks)  # get a list of xarray.Datasets
     combined = xr.combine_nested(datasets)  # or some combination of concat, merge


As an example, here's how we could approximate ``MFDataset`` from the netCDF4
library::

    from glob import glob
    import xarray as xr

    def read_netcdfs(files, dim):
        # glob expands paths with * to a list of files, like the unix shell
        paths = sorted(glob(files))
        datasets = [xr.open_dataset(p) for p in paths]
        combined = xr.concat(datasets, dim)
        return combined

    combined = read_netcdfs('/all/my/files/*.nc', dim='time')

This function will work in many cases, but it's not very robust. First, it
never closes files, which means it will fail if you need to load more than
a few thousand files. Second, it assumes that you want all the data from each
file and that it can all fit into memory. In many situations, you only need
a small subset or an aggregated summary of the data from each file.

Here's a slightly more sophisticated example of how to remedy these
deficiencies::

    def read_netcdfs(files, dim, transform_func=None):
        def process_one_path(path):
            # use a context manager, to ensure the file gets closed after use
            with xr.open_dataset(path) as ds:
                # transform_func should do some sort of selection or
                # aggregation
                if transform_func is not None:
                    ds = transform_func(ds)
                # load all data from the transformed dataset, to ensure we can
                # use it after closing each original file
                ds.load()
                return ds

        paths = sorted(glob(files))
        datasets = [process_one_path(p) for p in paths]
        combined = xr.concat(datasets, dim)
        return combined

    # here we suppose we only care about the combined mean of each file;
    # you might also use indexing operations like .sel to subset datasets
    combined = read_netcdfs('/all/my/files/*.nc', dim='time',
                            transform_func=lambda ds: ds.mean())

This pattern works well and is very robust. We've used similar code to process
tens of thousands of files constituting 100s of GB of data.


.. _io.netcdf.writing_encoded:

Writing encoded data
~~~~~~~~~~~~~~~~~~~~

Conversely, you can customize how xarray writes netCDF files on disk by
providing explicit encodings for each dataset variable. The ``encoding``
argument takes a dictionary with variable names as keys and variable specific
encodings as values. These encodings are saved as attributes on the netCDF
variables on disk, which allows xarray to faithfully read encoded data back into
memory.

It is important to note that using encodings is entirely optional: if you do not
supply any of these encoding options, xarray will write data to disk using a
default encoding, or the options in the ``encoding`` attribute, if set.
This works perfectly fine in most cases, but encoding can be useful for
additional control, especially for enabling compression.

In the file on disk, these encodings are saved as attributes on each variable, which
allow xarray and other CF-compliant tools for working with netCDF files to correctly
read the data.

Scaling and type conversions
............................

These encoding options (based on `CF Conventions on packed data`_) work on any
version of the netCDF file format:

- ``dtype``: Any valid NumPy dtype or string convertible to a dtype, e.g., ``'int16'``
  or ``'float32'``. This controls the type of the data written on disk.
- ``_FillValue``:  Values of ``NaN`` in xarray variables are remapped to this value when
  saved on disk. This is important when converting floating point with missing values
  to integers on disk, because ``NaN`` is not a valid value for integer dtypes. By
  default, variables with float types are attributed a ``_FillValue`` of ``NaN`` in the
  output file, unless explicitly disabled with an encoding ``{'_FillValue': None}``.
- ``scale_factor`` and ``add_offset``: Used to convert from encoded data on disk to
  to the decoded data in memory, according to the formula
  ``decoded = scale_factor * encoded + add_offset``. Please note that ``scale_factor``
  and ``add_offset`` must be of same type and determine the type of the decoded data.

These parameters can be fruitfully combined to compress discretized data on disk. For
example, to save the variable ``foo`` with a precision of 0.1 in 16-bit integers while
converting ``NaN`` to ``-9999``, we would use
``encoding={'foo': {'dtype': 'int16', 'scale_factor': 0.1, '_FillValue': -9999}}``.
Compression and decompression with such discretization is extremely fast.

.. _CF Conventions on packed data: https://cfconventions.org/cf-conventions/cf-conventions.html#packed-data

.. _io.string-encoding:

String encoding
...............

Xarray can write unicode strings to netCDF files in two ways:

- As variable length strings. This is only supported on netCDF4 (HDF5) files.
- By encoding strings into bytes, and writing encoded bytes as a character
  array. The default encoding is UTF-8.

By default, we use variable length strings for compatible files and fall-back
to using encoded character arrays. Character arrays can be selected even for
netCDF4 files by setting the ``dtype`` field in ``encoding`` to ``S1``
(corresponding to NumPy's single-character bytes dtype).

If character arrays are used:

- The string encoding that was used is stored on
  disk in the ``_Encoding`` attribute, which matches an ad-hoc convention
  `adopted by the netCDF4-Python library <https://github.com/Unidata/netcdf4-python/pull/665>`_.
  At the time of this writing (October 2017), a standard convention for indicating
  string encoding for character arrays in netCDF files was
  `still under discussion <https://github.com/Unidata/netcdf-c/issues/402>`_.
  Technically, you can use
  `any string encoding recognized by Python <https://docs.python.org/3/library/codecs.html#standard-encodings>`_ if you feel the need to deviate from UTF-8,
  by setting the ``_Encoding`` field in ``encoding``. But
  `we don't recommend it <https://utf8everywhere.org/>`_.
- The character dimension name can be specified by the ``char_dim_name`` field of a variable's
  ``encoding``. If the name of the character dimension is not specified, the default is
  ``f'string{data.shape[-1]}'``. When decoding character arrays from existing files, the
  ``char_dim_name`` is added to the variables ``encoding`` to preserve if encoding happens, but
  the field can be edited by the user.

.. warning::

  Missing values in bytes or unicode string arrays (represented by ``NaN`` in
  xarray) are currently written to disk as empty strings ``''``. This means
  missing values will not be restored when data is loaded from disk.
  This behavior is likely to change in the future (:issue:`1647`).
  Unfortunately, explicitly setting a ``_FillValue`` for string arrays to handle
  missing values doesn't work yet either, though we also hope to fix this in the
  future.

Chunk based compression
.......................

``zlib``, ``complevel``, ``fletcher32``, ``contiguous`` and ``chunksizes``
can be used for enabling netCDF4/HDF5's chunk based compression, as described
in the `documentation for createVariable`_ for netCDF4-Python. This only works
for netCDF4 files and thus requires using ``format='netCDF4'`` and either
``engine='netcdf4'`` or ``engine='h5netcdf'``.

.. _documentation for createVariable: https://unidata.github.io/netcdf4-python/#netCDF4.Dataset.createVariable

Chunk based gzip compression can yield impressive space savings, especially
for sparse data, but it comes with significant performance overhead. HDF5
libraries can only read complete chunks back into memory, and maximum
decompression speed is in the range of 50-100 MB/s. Worse, HDF5's compression
and decompression currently cannot be parallelized with dask. For these reasons, we
recommend trying discretization based compression (described above) first.

Time units
..........

The ``units`` and ``calendar`` attributes control how xarray serializes ``datetime64`` and
``timedelta64`` arrays to datasets on disk as numeric values. The ``units`` encoding
should be a string like ``'days since 1900-01-01'`` for ``datetime64`` data or a string
like ``'days'`` for ``timedelta64`` data. ``calendar`` should be one of the calendar types
supported by netCDF4-python: 'standard', 'gregorian', 'proleptic_gregorian' 'noleap',
'365_day', '360_day', 'julian', 'all_leap', '366_day'.

By default, xarray uses the ``'proleptic_gregorian'`` calendar and units of the smallest time
difference between values, with a reference time of the first time value.


.. _io.coordinates:

Coordinates
...........

You can control the ``coordinates`` attribute written to disk by specifying ``DataArray.encoding["coordinates"]``.
If not specified, xarray automatically sets ``DataArray.encoding["coordinates"]`` to a space-delimited list
of names of coordinate variables that share dimensions with the ``DataArray`` being written.
This allows perfect roundtripping of xarray datasets but may not be desirable.
When an xarray ``Dataset`` contains non-dimensional coordinates that do not share dimensions with any of
the variables, these coordinate variable names are saved under a "global" ``"coordinates"`` attribute.
This is not CF-compliant but again facilitates roundtripping of xarray datasets.

Invalid netCDF files
~~~~~~~~~~~~~~~~~~~~

The library ``h5netcdf`` allows writing some dtypes that aren't
allowed in netCDF4 (see
`h5netcdf documentation <https://github.com/h5netcdf/h5netcdf#invalid-netcdf-files>`_).
This feature is available through :py:meth:`DataArray.to_netcdf` and
:py:meth:`Dataset.to_netcdf` when used with ``engine="h5netcdf"``
and currently raises a warning unless ``invalid_netcdf=True`` is set.

.. warning::

  Note that this produces a file that is likely to be not readable by other netCDF
  libraries!

.. _io.hdf5:

HDF5
----
`HDF5`_ is both a file format and a data model for storing information. HDF5 stores
data hierarchically, using groups to create a nested structure. HDF5 is a more
general version of the netCDF4 data model, so the nested structure is one of many
similarities between the two data formats.

Reading HDF5 files in xarray requires the ``h5netcdf`` engine, which can be installed
with ``conda install h5netcdf``. Once installed we can use xarray to open HDF5 files:

.. code:: python

    xr.open_dataset("/path/to/my/file.h5")

The similarities between HDF5 and netCDF4 mean that HDF5 data can be written with the
same :py:meth:`Dataset.to_netcdf` method as used for netCDF4 data:

.. ipython:: python

    ds = xr.Dataset(
        {"foo": (("x", "y"), np.random.rand(4, 5))},
        coords={
            "x": [10, 20, 30, 40],
            "y": pd.date_range("2000-01-01", periods=5),
            "z": ("x", list("abcd")),
        },
    )

    ds.to_netcdf("saved_on_disk.h5")

Groups
~~~~~~

If you have multiple or highly nested groups, xarray by default may not read the group
that you want. A particular group of an HDF5 file can be specified using the ``group``
argument:

.. code:: python

    xr.open_dataset("/path/to/my/file.h5", group="/my/group")

While xarray cannot interrogate an HDF5 file to determine which groups are available,
the HDF5 Python reader `h5py`_ can be used instead.

Natively the xarray data structures can only handle one level of nesting, organized as
DataArrays inside of Datasets. If your HDF5 file has additional levels of hierarchy you
can only access one group and a time and will need to specify group names.

.. _HDF5: https://hdfgroup.github.io/hdf5/index.html
.. _h5py: https://www.h5py.org/


.. _io.zarr:

Zarr
----

`Zarr`_ is a Python package that provides an implementation of chunked, compressed,
N-dimensional arrays.
Zarr has the ability to store arrays in a range of ways, including in memory,
in files, and in cloud-based object storage such as `Amazon S3`_ and
`Google Cloud Storage`_.
Xarray's Zarr backend allows xarray to leverage these capabilities, including
the ability to store and analyze datasets far too large fit onto disk
(particularly :ref:`in combination with dask <dask>`).

Xarray can't open just any zarr dataset, because xarray requires special
metadata (attributes) describing the dataset dimensions and coordinates.
At this time, xarray can only open zarr datasets with these special attributes,
such as zarr datasets written by xarray,
`netCDF <https://docs.unidata.ucar.edu/nug/current/nczarr_head.html>`_,
or `GDAL <https://gdal.org/drivers/raster/zarr.html>`_.
For implementation details, see :ref:`zarr_encoding`.

To write a dataset with zarr, we use the :py:meth:`Dataset.to_zarr` method.

To write to a local directory, we pass a path to a directory:

.. ipython:: python
    :suppress:

    ! rm -rf path/to/directory.zarr

.. ipython:: python

    ds = xr.Dataset(
        {"foo": (("x", "y"), np.random.rand(4, 5))},
        coords={
            "x": [10, 20, 30, 40],
            "y": pd.date_range("2000-01-01", periods=5),
            "z": ("x", list("abcd")),
        },
    )
    ds.to_zarr("path/to/directory.zarr")

(The suffix ``.zarr`` is optional--just a reminder that a zarr store lives
there.) If the directory does not exist, it will be created. If a zarr
store is already present at that path, an error will be raised, preventing it
from being overwritten. To override this behavior and overwrite an existing
store, add ``mode='w'`` when invoking :py:meth:`~Dataset.to_zarr`.

DataArrays can also be saved to disk using the :py:meth:`DataArray.to_zarr` method,
and loaded from disk using the :py:func:`open_dataarray` function with ``engine='zarr'``.
Similar to :py:meth:`DataArray.to_netcdf`, :py:meth:`DataArray.to_zarr` will
convert the ``DataArray`` to a ``Dataset`` before saving, and then convert back
when loading, ensuring that the ``DataArray`` that is loaded is always exactly
the same as the one that was saved.

.. note::

    xarray does not write `NCZarr <https://docs.unidata.ucar.edu/nug/current/nczarr_head.html>`_ attributes.
    Therefore, NCZarr data must be opened in read-only mode.

To store variable length strings, convert them to object arrays first with
``dtype=object``.

To read back a zarr dataset that has been created this way, we use the
:py:func:`open_zarr` method:

.. ipython:: python

    ds_zarr = xr.open_zarr("path/to/directory.zarr")
    ds_zarr

Cloud Storage Buckets
~~~~~~~~~~~~~~~~~~~~~

It is possible to read and write xarray datasets directly from / to cloud
storage buckets using zarr. This example uses the `gcsfs`_ package to provide
an interface to `Google Cloud Storage`_.

General `fsspec`_ URLs, those that begin with ``s3://`` or ``gcs://`` for example,
are parsed and the store set up for you automatically when reading.
You should include any arguments to the storage backend as the
key ```storage_options``, part of ``backend_kwargs``.

.. code:: python

    ds_gcs = xr.open_dataset(
        "gcs://<bucket-name>/path.zarr",
        backend_kwargs={
            "storage_options": {"project": "<project-name>", "token": None}
        },
        engine="zarr",
    )


This also works with ``open_mfdataset``, allowing you to pass a list of paths or
a URL to be interpreted as a glob string.

For writing, you must explicitly set up a ``MutableMapping``
instance and pass this, as follows:

.. code:: python

    import gcsfs

    fs = gcsfs.GCSFileSystem(project="<project-name>", token=None)
    gcsmap = gcsfs.mapping.GCSMap("<bucket-name>", gcs=fs, check=True, create=False)
    # write to the bucket
    ds.to_zarr(store=gcsmap)
    # read it back
    ds_gcs = xr.open_zarr(gcsmap)

(or use the utility function ``fsspec.get_mapper()``).

.. _fsspec: https://filesystem-spec.readthedocs.io/en/latest/
.. _Zarr: https://zarr.readthedocs.io/
.. _Amazon S3: https://aws.amazon.com/s3/
.. _Google Cloud Storage: https://cloud.google.com/storage/
.. _gcsfs: https://github.com/fsspec/gcsfs

.. _io.zarr.distributed_writes:

Distributed writes
~~~~~~~~~~~~~~~~~~

Xarray will natively use dask to write in parallel to a zarr store, which should
satisfy most moderately sized datasets. For more flexible parallelization, we
can use ``region`` to write to limited regions of arrays in an existing Zarr
store.

To scale this up to writing large datasets, first create an initial Zarr store
without writing all of its array data. This can be done by first creating a
``Dataset`` with dummy values stored in :ref:`dask <dask>`, and then calling
``to_zarr`` with ``compute=False`` to write only metadata (including ``attrs``)
to Zarr:

.. ipython:: python
    :suppress:

    ! rm -rf path/to/directory.zarr

.. ipython:: python

    import dask.array

    # The values of this dask array are entirely irrelevant; only the dtype,
    # shape and chunks are used
    dummies = dask.array.zeros(30, chunks=10)
    ds = xr.Dataset({"foo": ("x", dummies)}, coords={"x": np.arange(30)})
    path = "path/to/directory.zarr"
    # Now we write the metadata without computing any array values
    ds.to_zarr(path, compute=False)

Now, a Zarr store with the correct variable shapes and attributes exists that
can be filled out by subsequent calls to ``to_zarr``.
Setting ``region="auto"`` will open the existing store and determine the
correct alignment of the new data with the existing dimensions, or as an
explicit mapping from dimension names to Python ``slice`` objects indicating
where the data should be written (in index space, not label space), e.g.,

.. ipython:: python

    # For convenience, we'll slice a single dataset, but in the real use-case
    # we would create them separately possibly even from separate processes.
    ds = xr.Dataset({"foo": ("x", np.arange(30))}, coords={"x": np.arange(30)})
    # Any of the following region specifications are valid
    ds.isel(x=slice(0, 10)).to_zarr(path, region="auto")
    ds.isel(x=slice(10, 20)).to_zarr(path, region={"x": "auto"})
    ds.isel(x=slice(20, 30)).to_zarr(path, region={"x": slice(20, 30)})

Concurrent writes with ``region`` are safe as long as they modify distinct
chunks in the underlying Zarr arrays (or use an appropriate ``lock``).

As a safety check to make it harder to inadvertently override existing values,
if you set ``region`` then *all* variables included in a Dataset must have
dimensions included in ``region``. Other variables (typically coordinates)
need to be explicitly dropped and/or written in a separate calls to ``to_zarr``
with ``mode='a'``.

Zarr Compressors and Filters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

There are many different `options for compression and filtering possible with
zarr <https://zarr.readthedocs.io/en/stable/tutorial.html#compressors>`_.

These options can be passed to the ``to_zarr`` method as variable encoding.
For example:

.. ipython:: python
    :suppress:

    ! rm -rf foo.zarr

.. ipython:: python

    import zarr
    from numcodecs.blosc import Blosc

    compressor = Blosc(cname="zstd", clevel=3, shuffle=2)
    ds.to_zarr("foo.zarr", encoding={"foo": {"compressor": compressor}})

.. note::

    Not all native zarr compression and filtering options have been tested with
    xarray.

.. _io.zarr.appending:

Modifying existing Zarr stores
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Xarray supports several ways of incrementally writing variables to a Zarr
store. These options are useful for scenarios when it is infeasible or
undesirable to write your entire dataset at once.

1. Use ``mode='a'`` to add or overwrite entire variables,
2. Use ``append_dim`` to resize and append to existing variables, and
3. Use ``region`` to write to limited regions of existing arrays.

.. tip::

    For ``Dataset`` objects containing dask arrays, a
    single call to ``to_zarr()`` will write all of your data in parallel.

.. warning::

    Alignment of coordinates is currently not checked when modifying an
    existing Zarr store. It is up to the user to ensure that coordinates are
    consistent.

To add or overwrite entire variables, simply call :py:meth:`~Dataset.to_zarr`
with ``mode='a'`` on a Dataset containing the new variables, passing in an
existing Zarr store or path to a Zarr store.

To resize and then append values along an existing dimension in a store, set
``append_dim``. This is a good option if data always arrives in a particular
order, e.g., for time-stepping a simulation:

.. ipython:: python
    :suppress:

    ! rm -rf path/to/directory.zarr

.. ipython:: python

    ds1 = xr.Dataset(
        {"foo": (("x", "y", "t"), np.random.rand(4, 5, 2))},
        coords={
            "x": [10, 20, 30, 40],
            "y": [1, 2, 3, 4, 5],
            "t": pd.date_range("2001-01-01", periods=2),
        },
    )
    ds1.to_zarr("path/to/directory.zarr")
    ds2 = xr.Dataset(
        {"foo": (("x", "y", "t"), np.random.rand(4, 5, 2))},
        coords={
            "x": [10, 20, 30, 40],
            "y": [1, 2, 3, 4, 5],
            "t": pd.date_range("2001-01-03", periods=2),
        },
    )
    ds2.to_zarr("path/to/directory.zarr", append_dim="t")

.. _io.zarr.writing_chunks:

Specifying chunks in a zarr store
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Chunk sizes may be specified in one of three ways when writing to a zarr store:

1. Manual chunk sizing through the use of the ``encoding`` argument in :py:meth:`Dataset.to_zarr`:
2. Automatic chunking based on chunks in dask arrays
3. Default chunk behavior determined by the zarr library

The resulting chunks will be determined based on the order of the above list; dask
chunks will be overridden by manually-specified chunks in the encoding argument,
and the presence of either dask chunks or chunks in the ``encoding`` attribute will
supersede the default chunking heuristics in zarr.

Importantly, this logic applies to every array in the zarr store individually,
including coordinate arrays. Therefore, if a dataset contains one or more dask
arrays, it may still be desirable to specify a chunk size for the coordinate arrays
(for example, with a chunk size of ``-1`` to include the full coordinate).

To specify chunks manually using the ``encoding`` argument, provide a nested
dictionary with the structure ``{'variable_or_coord_name': {'chunks': chunks_tuple}}``.

.. note::

    The positional ordering of the chunks in the encoding argument must match the
    positional ordering of the dimensions in each array. Watch out for arrays with
    differently-ordered dimensions within a single Dataset.

For example, let's say we're working with a dataset with dimensions
``('time', 'x', 'y')``, a variable ``Tair`` which is chunked in ``x`` and ``y``,
and two multi-dimensional coordinates ``xc`` and ``yc``:

.. ipython:: python

    ds = xr.tutorial.open_dataset("rasm")

    ds["Tair"] = ds["Tair"].chunk({"x": 100, "y": 100})

    ds

These multi-dimensional coordinates are only two-dimensional and take up very little
space on disk or in memory, yet when writing to disk the default zarr behavior is to
split them into chunks:

.. ipython:: python

    ds.to_zarr("path/to/directory.zarr", mode="w")
    ! ls -R path/to/directory.zarr


This may cause unwanted overhead on some systems, such as when reading from a cloud
storage provider. To disable this chunking, we can specify a chunk size equal to the
length of each dimension by using the shorthand chunk size ``-1``:

.. ipython:: python

    ds.to_zarr(
        "path/to/directory.zarr",
        encoding={"xc": {"chunks": (-1, -1)}, "yc": {"chunks": (-1, -1)}},
        mode="w",
    )
    ! ls -R path/to/directory.zarr


The number of chunks on Tair matches our dask chunks, while there is now only a single
chunk in the directory stores of each coordinate.

Groups
~~~~~~

Nested groups in zarr stores can be represented by loading the store as a
:py:class:`xarray.DataTree` object, similarly to netCDF. To open a whole zarr store as
a tree of groups use the :py:func:`open_datatree` function. To save a
``DataTree`` object as a zarr store containing many groups, use the
:py:meth:`xarray.DataTree.to_zarr()` method.

.. note::
    Note that perfect round-tripping should always be possible with a zarr
    store (:ref:`unlike for netCDF files <netcdf.group.warning>`), as zarr does
    not support "unused" dimensions.

    For the root group the same restrictions (:ref:`as for netCDF files <netcdf.root_group.note>`) apply.
    Due to file format specifications the on-disk root group name is always ``"/"``
    overriding any given ``DataTree`` root node name.


.. _io.zarr.consolidated_metadata:

Consolidated Metadata
~~~~~~~~~~~~~~~~~~~~~


Xarray needs to read all of the zarr metadata when it opens a dataset.
In some storage mediums, such as with cloud object storage (e.g. `Amazon S3`_),
this can introduce significant overhead, because two separate HTTP calls to the
object store must be made for each variable in the dataset.
By default Xarray uses a feature called
*consolidated metadata*, storing all metadata for the entire dataset with a
single key (by default called ``.zmetadata``). This typically drastically speeds
up opening the store. (For more information on this feature, consult the
`zarr docs on consolidating metadata <https://zarr.readthedocs.io/en/latest/tutorial.html#consolidating-metadata>`_.)

By default, xarray writes consolidated metadata and attempts to read stores
with consolidated metadata, falling back to use non-consolidated metadata for
reads. Because this fall-back option is so much slower, xarray issues a
``RuntimeWarning`` with guidance when reading with consolidated metadata fails:

    Failed to open Zarr store with consolidated metadata, falling back to try
    reading non-consolidated metadata. This is typically much slower for
    opening a dataset. To silence this warning, consider:

    1. Consolidating metadata in this existing store with
       :py:func:`zarr.consolidate_metadata`.
    2. Explicitly setting ``consolidated=False``, to avoid trying to read
       consolidate metadata.
    3. Explicitly setting ``consolidated=True``, to raise an error in this case
       instead of falling back to try reading non-consolidated metadata.


.. _io.kerchunk:

Kerchunk
--------

`Kerchunk <https://fsspec.github.io/kerchunk/index.html>`_ is a Python library
that allows you to access chunked and compressed data formats (such as NetCDF3, NetCDF4, HDF5, GRIB2, TIFF & FITS),
many of which are primary data formats for many data archives, by viewing the
whole archive as an ephemeral `Zarr`_ dataset which allows for parallel, chunk-specific access.

Instead of creating a new copy of the dataset in the Zarr spec/format or
downloading the files locally, Kerchunk reads through the data archive and extracts the
byte range and compression information of each chunk and saves as a ``reference``.
These references are then saved as ``json`` files or ``parquet`` (more efficient)
for later use. You can view some of these stored in the ``references``
directory `here <https://github.com/pydata/xarray-data>`_.


.. note::
    These references follow this `specification <https://fsspec.github.io/kerchunk/spec.html>`_.
    Packages like `kerchunk`_ and `virtualizarr <https://github.com/zarr-developers/VirtualiZarr>`_
    help in creating and reading these references.


Reading these data archives becomes really easy with ``kerchunk`` in combination
with ``xarray``, especially when these archives are large in size. A single combined
reference can refer to thousands of the original data files present in these archives.
You can view the whole dataset with from this combined reference using the above packages.

The following example shows opening a combined references generated from a ``.hdf`` file stored locally.

.. ipython:: python

    storage_options = {
        "target_protocol": "file",
    }

    # add the `remote_protocol` key in `storage_options` if you're accessing a file remotely

    ds1 = xr.open_dataset(
        "./combined.json",
        engine="kerchunk",
        storage_options=storage_options,
    )

    ds1

.. note::

    You can refer to the `project pythia kerchunk cookbook <https://projectpythia.org/kerchunk-cookbook/README.html>`_
    and the `pangeo guide on kerchunk <https://guide.cloudnativegeo.org/kerchunk/intro.html>`_ for more information.


.. _io.iris:

Iris
----

The Iris_ tool allows easy reading of common meteorological and climate model formats
(including GRIB and UK MetOffice PP files) into ``Cube`` objects which are in many ways very
similar to ``DataArray`` objects, while enforcing a CF-compliant data model. If iris is
installed, xarray can convert a ``DataArray`` into a ``Cube`` using
:py:meth:`DataArray.to_iris`:

.. ipython:: python

    da = xr.DataArray(
        np.random.rand(4, 5),
        dims=["x", "y"],
        coords=dict(x=[10, 20, 30, 40], y=pd.date_range("2000-01-01", periods=5)),
    )

    cube = da.to_iris()
    cube

Conversely, we can create a new ``DataArray`` object from a ``Cube`` using
:py:meth:`DataArray.from_iris`:

.. ipython:: python

    da_cube = xr.DataArray.from_iris(cube)
    da_cube


.. _Iris: https://scitools.org.uk/iris


OPeNDAP
-------

Xarray includes support for `OPeNDAP`__ (via the netCDF4 library or Pydap), which
lets us access large datasets over HTTP.

__ https://www.opendap.org/

For example, we can open a connection to GBs of weather data produced by the
`PRISM`__ project, and hosted by `IRI`__ at Columbia:

__ https://www.prism.oregonstate.edu/
__ https://iri.columbia.edu/

.. ipython source code for this section
   we don't use this to avoid hitting the DAP server on every doc build.

   remote_data = xr.open_dataset(
       'http://iridl.ldeo.columbia.edu/SOURCES/.OSU/.PRISM/.monthly/dods',
       decode_times=False)
   tmax = remote_data.tmax[:500, ::3, ::3]
   tmax

   @savefig opendap-prism-tmax.png
   tmax[0].plot()

.. ipython::
    :verbatim:

    In [3]: remote_data = xr.open_dataset(
       ...:     "http://iridl.ldeo.columbia.edu/SOURCES/.OSU/.PRISM/.monthly/dods",
       ...:     decode_times=False,
       ...: )

    In [4]: remote_data
    Out[4]:
    <xarray.Dataset>
    Dimensions:  (T: 1422, X: 1405, Y: 621)
    Coordinates:
      * X        (X) float32 -125.0 -124.958 -124.917 -124.875 -124.833 -124.792 -124.75 ...
      * T        (T) float32 -779.5 -778.5 -777.5 -776.5 -775.5 -774.5 -773.5 -772.5 -771.5 ...
      * Y        (Y) float32 49.9167 49.875 49.8333 49.7917 49.75 49.7083 49.6667 49.625 ...
    Data variables:
        ppt      (T, Y, X) float64 ...
        tdmean   (T, Y, X) float64 ...
        tmax     (T, Y, X) float64 ...
        tmin     (T, Y, X) float64 ...
    Attributes:
        Conventions: IRIDL
        expires: 1375315200

.. TODO: update this example to show off decode_cf?

.. note::

    Like many real-world datasets, this dataset does not entirely follow
    `CF conventions`_. Unexpected formats will usually cause xarray's automatic
    decoding to fail. The way to work around this is to either set
    ``decode_cf=False`` in ``open_dataset`` to turn off all use of CF
    conventions, or by only disabling the troublesome parser.
    In this case, we set ``decode_times=False`` because the time axis here
    provides the calendar attribute in a format that xarray does not expect
    (the integer ``360`` instead of a string like ``'360_day'``).

We can select and slice this data any number of times, and nothing is loaded
over the network until we look at particular values:

.. ipython::
    :verbatim:

    In [4]: tmax = remote_data["tmax"][:500, ::3, ::3]

    In [5]: tmax
    Out[5]:
    <xarray.DataArray 'tmax' (T: 500, Y: 207, X: 469)>
    [48541500 values with dtype=float64]
    Coordinates:
      * Y        (Y) float32 49.9167 49.7917 49.6667 49.5417 49.4167 49.2917 ...
      * X        (X) float32 -125.0 -124.875 -124.75 -124.625 -124.5 -124.375 ...
      * T        (T) float32 -779.5 -778.5 -777.5 -776.5 -775.5 -774.5 -773.5 ...
    Attributes:
        pointwidth: 120
        standard_name: air_temperature
        units: Celsius_scale
        expires: 1443657600

    # the data is downloaded automatically when we make the plot
    In [6]: tmax[0].plot()

.. image:: ../_static/opendap-prism-tmax.png

Some servers require authentication before we can access the data. For this
purpose we can explicitly create a :py:class:`backends.PydapDataStore`
and pass in a `Requests`__ session object. For example for
HTTP Basic authentication::

    import xarray as xr
    import requests

    session = requests.Session()
    session.auth = ('username', 'password')

    store = xr.backends.PydapDataStore.open('http://example.com/data',
                                            session=session)
    ds = xr.open_dataset(store)

`Pydap's cas module`__ has functions that generate custom sessions for
servers that use CAS single sign-on. For example, to connect to servers
that require NASA's URS authentication::

  import xarray as xr
  from pydata.cas.urs import setup_session

  ds_url = 'https://gpm1.gesdisc.eosdis.nasa.gov/opendap/hyrax/example.nc'

  session = setup_session('username', 'password', check_url=ds_url)
  store = xr.backends.PydapDataStore.open(ds_url, session=session)

  ds = xr.open_dataset(store)

__ https://docs.python-requests.org
__ https://www.pydap.org/en/latest/client.html#authentication

.. _io.pickle:

Pickle
------

The simplest way to serialize an xarray object is to use Python's built-in pickle
module:

.. ipython:: python

    import pickle

    # use the highest protocol (-1) because it is way faster than the default
    # text based pickle format
    pkl = pickle.dumps(ds, protocol=-1)

    pickle.loads(pkl)

Pickling is important because it doesn't require any external libraries
and lets you use xarray objects with Python modules like
:py:mod:`multiprocessing` or :ref:`Dask <dask>`. However, pickling is
**not recommended for long-term storage**.

Restoring a pickle requires that the internal structure of the types for the
pickled data remain unchanged. Because the internal design of xarray is still
being refined, we make no guarantees (at this point) that objects pickled with
this version of xarray will work in future versions.

.. note::

  When pickling an object opened from a NetCDF file, the pickle file will
  contain a reference to the file on disk. If you want to store the actual
  array values, load it into memory first with :py:meth:`Dataset.load`
  or :py:meth:`Dataset.compute`.

.. _dictionary io:

Dictionary
----------

We can convert a ``Dataset`` (or a ``DataArray``) to a dict using
:py:meth:`Dataset.to_dict`:

.. ipython:: python

    ds = xr.Dataset({"foo": ("x", np.arange(30))})
    ds

    d = ds.to_dict()
    d

We can create a new xarray object from a dict using
:py:meth:`Dataset.from_dict`:

.. ipython:: python

    ds_dict = xr.Dataset.from_dict(d)
    ds_dict

Dictionary support allows for flexible use of xarray objects. It doesn't
require external libraries and dicts can easily be pickled, or converted to
json, or geojson. All the values are converted to lists, so dicts might
be quite large.

To export just the dataset schema without the data itself, use the
``data=False`` option:

.. ipython:: python

    ds.to_dict(data=False)

.. ipython:: python
    :suppress:

    # We're now done with the dataset named `ds`.  Although the `with` statement closed
    # the dataset, displaying the unpickled pickle of `ds` re-opened "saved_on_disk.nc".
    # However, `ds` (rather than the unpickled dataset) refers to the open file.  Delete
    # `ds` to close the file.
    del ds
    os.remove("saved_on_disk.nc")

This can be useful for generating indices of dataset contents to expose to
search indices or other automated data discovery tools.

.. _io.rasterio:

Rasterio
--------

GDAL readable raster data using `rasterio`_  such as GeoTIFFs can be opened using the `rioxarray`_ extension.
`rioxarray`_ can also handle geospatial related tasks such as re-projecting and clipping.

.. ipython::
    :verbatim:

    In [1]: import rioxarray

    In [2]: rds = rioxarray.open_rasterio("RGB.byte.tif")

    In [3]: rds
    Out[3]:
    <xarray.DataArray (band: 3, y: 718, x: 791)>
    [1703814 values with dtype=uint8]
    Coordinates:
      * band         (band) int64 1 2 3
      * y            (y) float64 2.827e+06 2.826e+06 ... 2.612e+06 2.612e+06
      * x            (x) float64 1.021e+05 1.024e+05 ... 3.389e+05 3.392e+05
        spatial_ref  int64 0
    Attributes:
        STATISTICS_MAXIMUM:  255
        STATISTICS_MEAN:     29.947726688477
        STATISTICS_MINIMUM:  0
        STATISTICS_STDDEV:   52.340921626611
        transform:           (300.0379266750948, 0.0, 101985.0, 0.0, -300.0417827...
        _FillValue:          0.0
        scale_factor:        1.0
        add_offset:          0.0
        grid_mapping:        spatial_ref

    In [4]: rds.rio.crs
    Out[4]: CRS.from_epsg(32618)

    In [5]: rds4326 = rds.rio.reproject("epsg:4326")

    In [6]: rds4326.rio.crs
    Out[6]: CRS.from_epsg(4326)

    In [7]: rds4326.rio.to_raster("RGB.byte.4326.tif")


.. _rasterio: https://rasterio.readthedocs.io/en/latest/
.. _rioxarray: https://corteva.github.io/rioxarray/stable/
.. _test files: https://github.com/rasterio/rasterio/blob/master/tests/data/RGB.byte.tif
.. _pyproj: https://github.com/pyproj4/pyproj

.. _io.cfgrib:

.. ipython:: python
    :suppress:

    import shutil

    shutil.rmtree("foo.zarr")
    shutil.rmtree("path/to/directory.zarr")

GRIB format via cfgrib
----------------------

Xarray supports reading GRIB files via ECMWF cfgrib_ python driver,
if it is installed. To open a GRIB file supply ``engine='cfgrib'``
to :py:func:`open_dataset` after installing cfgrib_:

.. ipython::
    :verbatim:

    In [1]: ds_grib = xr.open_dataset("example.grib", engine="cfgrib")

We recommend installing cfgrib via conda::

    conda install -c conda-forge cfgrib

.. _cfgrib: https://github.com/ecmwf/cfgrib


CSV and other formats supported by pandas
-----------------------------------------

For more options (tabular formats and CSV files in particular), consider
exporting your objects to pandas and using its broad range of `IO tools`_.
For CSV files, one might also consider `xarray_extras`_.

.. _xarray_extras: https://xarray-extras.readthedocs.io/en/latest/api/csv.html

.. _IO tools: https://pandas.pydata.org/pandas-docs/stable/user_guide/io.html


Third party libraries
---------------------

More formats are supported by extension libraries:

- `xarray-mongodb <https://xarray-mongodb.readthedocs.io/en/latest/>`_: Store xarray objects on MongoDB
