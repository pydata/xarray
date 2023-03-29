#!/usr/bin/env bash

# temporarily (?) remove numbagg and numba
pip uninstall -y numbagg
conda uninstall -y numba
# forcibly remove packages to avoid artifacts
conda uninstall -y --force \
    numpy \
    scipy \
    pandas \
    matplotlib \
    dask \
    distributed \
    fsspec \
    zarr \
    cftime \
    packaging \
    pint \
    bottleneck \
    sparse \
    flox \
    h5netcdf \
    xarray
# to limit the runtime of Upstream CI
python -m pip install \
    -i https://pypi.anaconda.org/scipy-wheels-nightly/simple \
    --no-deps \
    --pre \
    --upgrade \
    numpy \
    scipy \
    matplotlib \
    pandas
python -m pip install \
    --no-deps \
    --upgrade \
    git+https://github.com/dask/dask \
    git+https://github.com/dask/distributed \
    git+https://github.com/zarr-developers/zarr \
    git+https://github.com/Unidata/cftime \
    git+https://github.com/pypa/packaging \
    git+https://github.com/hgrecco/pint \
    git+https://github.com/pydata/bottleneck \
    git+https://github.com/pydata/sparse \
    git+https://github.com/intake/filesystem_spec \
    git+https://github.com/SciTools/nc-time-axis \
    git+https://github.com/xarray-contrib/flox \
    git+https://github.com/h5netcdf/h5netcdf
