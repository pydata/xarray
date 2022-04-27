# xarray: N-D labeled arrays and datasets

[![image](https://github.com/pydata/xarray/workflows/CI/badge.svg?branch=main)](https://github.com/pydata/xarray/actions?query=workflow%3ACI)
[![image](https://codecov.io/gh/pydata/xarray/branch/main/graph/badge.svg)](https://codecov.io/gh/pydata/xarray)
[![image](https://readthedocs.org/projects/xray/badge/?version=latest)](https://docs.xarray.dev/)
[![image](https://img.shields.io/badge/benchmarked%20by-asv-green.svg?style=flat)](https://pandas.pydata.org/speed/xarray/)
[![image](https://img.shields.io/pypi/v/xarray.svg)](https://pypi.python.org/pypi/xarray/)
[![image](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/python/black)
[![image](https://zenodo.org/badge/DOI/10.5281/zenodo.598201.svg)](https://doi.org/10.5281/zenodo.598201)
[![image](https://img.shields.io/twitter/follow/xarray_dev?style=social)](https://twitter.com/xarray_dev)

**xarray** (formerly **xray**) is an open source project and Python
package that makes working with labelled multi-dimensional arrays
simple, efficient, and fun!

Xarray introduces labels in the form of dimensions, coordinates and
attributes on top of raw [NumPy](https://www.numpy.org)-like arrays,
which allows for a more intuitive, more concise, and less error-prone
developer experience. The package includes a large and growing library
of domain-agnostic functions for advanced analytics and visualization
with these data structures.

Xarray was inspired by and borrows heavily from
[pandas](https://pandas.pydata.org), the popular data analysis package
focused on labelled tabular data. It is particularly tailored to working
with [netCDF](https://www.unidata.ucar.edu/software/netcdf) files, which
were the source of xarray\'s data model, and integrates tightly with
[dask](https://dask.org) for parallel computing.

## Why xarray?

Multi-dimensional (a.k.a. N-dimensional, ND) arrays (sometimes called
"tensors") are an essential part of computational science. They are
encountered in a wide range of fields, including physics, astronomy,
geoscience, bioinformatics, engineering, finance, and deep learning. In
Python, [NumPy](https://www.numpy.org) provides the fundamental data
structure and API for working with raw ND arrays. However, real-world
datasets are usually more than just raw numbers; they have labels which
encode information about how the array values map to locations in space,
time, etc.

Xarray doesn\'t just keep track of labels on arrays \-- it uses them to
provide a powerful and concise interface. For example:

- Apply operations over dimensions by name: `x.sum('time')`.
- Select values by label instead of integer location:
    `x.loc['2014-01-01']` or `x.sel(time='2014-01-01')`.
- Mathematical operations (e.g., `x - y`) vectorize across multiple
    dimensions (array broadcasting) based on dimension names, not shape.
- Flexible split-apply-combine operations with groupby:
    `x.groupby('time.dayofyear').mean()`.
- Database like alignment based on coordinate labels that smoothly
    handles missing values: `x, y = xr.align(x, y, join='outer')`.
- Keep track of arbitrary metadata in the form of a Python dictionary:
    `x.attrs`.

## Documentation

Learn more about xarray in its official documentation at
<https://docs.xarray.dev/>.

Try out an [interactive Jupyter
notebook](https://mybinder.org/v2/gh/pydata/xarray/main?urlpath=lab/tree/doc/examples/weather-data.ipynb).

## Contributing

You can find information about contributing to xarray at our
[Contributing
page](https://docs.xarray.dev/en/latest/contributing.html#).

## Get in touch

- Report bugs, suggest features or view the source code [on
  GitHub](https://github.com/pydata/xarray).

## NumFOCUS

[![image](https://numfocus.org/wp-content/uploads/2017/07/NumFocus_LRG.png)](https://numfocus.org/)

Xarray is a fiscally sponsored project of
[NumFOCUS](https://numfocus.org), a nonprofit dedicated to supporting
the open source scientific computing community. If you like Xarray and
want to support our mission, please consider making a
[donation](https://numfocus.salsalabs.org/donate-to-xarray/) to support
our efforts.

## History

Xarray is an evolution of an internal tool developed at [The Climate
Corporation](http://climate.com/). It was originally written by Climate
Corp researchers Stephan Hoyer, Alex Kleeman and Eugene Brevdo and was
released as open source in May 2014. The project was renamed from
"xray" in January 2016. Xarray became a fiscally sponsored project of
[NumFOCUS](https://numfocus.org) in August 2018.
