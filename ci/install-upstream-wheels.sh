#!/usr/bin/env bash

# install cython for building cftime without build isolation
micromamba install "cython>=0.29.20" py-cpuinfo setuptools-scm
# temporarily (?) remove numbagg and numba
micromamba remove -y numba numbagg sparse
# temporarily remove numexpr
micromamba remove -y numexpr
# temporarily remove backends
micromamba remove -y cf_units hdf5 h5py netcdf4
# forcibly remove packages to avoid artifacts
micromamba remove -y --force \
    numpy \
    scipy \
    pandas \
    distributed \
    fsspec \
    zarr \
    cftime \
    packaging \
    pint \
    bottleneck \
    flox \
    numcodecs
# to limit the runtime of Upstream CI
python -m pip install \
    -i https://pypi.anaconda.org/scientific-python-nightly-wheels/simple \
    --no-deps \
    --pre \
    --upgrade \
    numpy \
    scipy \
    matplotlib \
    pandas
# for some reason pandas depends on pyarrow already.
# Remove once a `pyarrow` version compiled with `numpy>=2.0` is on `conda-forge`
python -m pip install \
    -i https://pypi.fury.io/arrow-nightlies/ \
    --prefer-binary \
    --no-deps \
    --pre \
    --upgrade \
    pyarrow
# without build isolation for packages compiling against numpy
# TODO: remove once there are `numpy>=2.0` builds for these
python -m pip install \
    --no-deps \
    --upgrade \
    --no-build-isolation \
    git+https://github.com/Unidata/cftime
python -m pip install \
    --no-deps \
    --upgrade \
    --no-build-isolation \
    git+https://github.com/zarr-developers/numcodecs
python -m pip install \
    --no-deps \
    --upgrade \
    --no-build-isolation \
    git+https://github.com/pydata/bottleneck
python -m pip install \
    --no-deps \
    --upgrade \
    git+https://github.com/dask/dask \
    git+https://github.com/dask/dask-expr \
    git+https://github.com/dask/distributed \
    git+https://github.com/zarr-developers/zarr \
    git+https://github.com/pypa/packaging \
    git+https://github.com/hgrecco/pint \
    git+https://github.com/intake/filesystem_spec \
    git+https://github.com/SciTools/nc-time-axis \
    git+https://github.com/xarray-contrib/flox \
    git+https://github.com/dgasmith/opt_einsum
    # git+https://github.com/pydata/sparse
    # git+https://github.com/h5netcdf/h5netcdf
