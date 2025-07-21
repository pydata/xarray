.. currentmodule:: xarray
.. _io.netcdf:

netCDF and HDF5
================

.. jupyter-execute::
    :hide-code:

    import os

    import iris
    import ncdata.iris_xarray
    import numpy as np
    import pandas as pd
    import xarray as xr

    np.random.seed(123456)

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

.. jupyter-execute::

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

.. jupyter-execute::

    ds_disk = xr.open_dataset("saved_on_disk.nc")
    ds_disk

.. jupyter-execute::
    :hide-code:

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

.. jupyter-execute::

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

.. jupyter-execute::

    ds_disk["y"].encoding

.. jupyter-execute::

    ds_disk.encoding

Note that all operations that manipulate variables other than indexing
will remove encoding information.

In some cases it is useful to intentionally reset a dataset's original encoding values.
This can be done with either the :py:meth:`Dataset.drop_encoding` or
:py:meth:`DataArray.drop_encoding` methods.

.. jupyter-execute::

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
supported by netCDF4-python: ``'standard'``, ``'gregorian'``, ``'proleptic_gregorian'``, ``'noleap'``,
``'365_day'``, ``'360_day'``, ``'julian'``, ``'all_leap'``, ``'366_day'``.

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
~~~~~~~~~~~~~~~~~~~~~

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

.. jupyter-execute::

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

.. _complex:

Complex Data Types
------------------

Xarray leverages NumPy to seamlessly handle complex numbers in :py:class:`~xarray.DataArray` and :py:class:`~xarray.Dataset` objects.

In the examples below, we are using a DataArray named ``da`` with complex elements (of :math:`\mathbb{C}`):

.. jupyter-execute::

    data = np.array([[1 + 2j, 3 + 4j], [5 + 6j, 7 + 8j]])
    da = xr.DataArray(
        data,
        dims=["x", "y"],
        coords={"x": ["a", "b"], "y": [1, 2]},
        name="complex_nums",
    )

You can access real and imaginary components using the ``.real`` and ``.imag`` attributes. Most NumPy universal functions (ufuncs) like :py:doc:`numpy.abs <numpy:reference/generated/numpy.absolute>` or :py:doc:`numpy.angle <numpy:reference/generated/numpy.angle>` work directly.

.. jupyter-execute::

    da.real

.. jupyter-execute::

    np.abs(da)

.. note::
    Like NumPy, ``.real`` and ``.imag`` typically return *views*, not copies, of the original data.

Writing complex data to NetCDF files is supported via :py:meth:`~xarray.DataArray.to_netcdf` using specific backend engines that handle complex types:

.. tab:: h5netcdf

   This requires the `h5netcdf <https://h5netcdf.org>`_ library to be installed.

   .. jupyter-execute::

       # write the data to disk
       da.to_netcdf("complex_nums_h5.nc", engine="h5netcdf")
       # read the file back into memory
       ds_h5 = xr.open_dataset("complex_nums_h5.nc", engine="h5netcdf")
       # check the dtype
       ds_h5[da.name].dtype

.. tab:: netcdf4

   Requires the `netcdf4-python (>= 1.7.1) <https://github.com/Unidata/netcdf4-python>`_ library and you have to enable ``auto_complex=True``.

   .. jupyter-execute::

       # write the data to disk
       da.to_netcdf("complex_nums_nc4.nc", engine="netcdf4", auto_complex=True)
       # read the file back into memory
       ds_nc4 = xr.open_dataset(
           "complex_nums_nc4.nc", engine="netcdf4", auto_complex=True
       )
       # check the dtype
       ds_nc4[da.name].dtype

.. warning::
   The ``scipy`` engine only supports NetCDF V3 and does *not* support complex arrays; writing with ``engine="scipy"`` raises a ``TypeError``.

If direct writing is not supported (e.g., targeting NetCDF3), you can manually
split the complex array into separate real and imaginary variables before saving:

.. jupyter-execute::

    # Write data to file
    ds_manual = xr.Dataset(
        {
            f"{da.name}_real": da.real,
            f"{da.name}_imag": da.imag,
        }
    )
    ds_manual.to_netcdf("complex_manual.nc", engine="scipy")  # Example

    # Read data from file
    ds = xr.open_dataset("complex_manual.nc", engine="scipy")
    reconstructed = ds[f"{da.name}_real"] + 1j * ds[f"{da.name}_imag"]

**Recommendations:**

- Use ``engine="netcdf4"`` with ``auto_complex=True`` for full compliance and ease.
- Use ``h5netcdf`` for HDF5-based storage when interoperability with HDF5 is desired.
- For maximum legacy support (NetCDF3), manually handle real/imaginary components.

.. jupyter-execute::
    :hide-code:

    # Cleanup
    import os

    for f in ["complex_nums_nc4.nc", "complex_nums_h5.nc", "complex_manual.nc"]:
        if os.path.exists(f):
            os.remove(f)
