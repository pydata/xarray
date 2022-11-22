#!/usr/bin/env bash

conda uninstall -y --force \
    bottleneck \
    cftime \
    dask \
    distributed \
    flox \
    fsspec \
    h5netcdf \
    matplotlib \
    numpy \
    numba \
    packaging \
    pandas \
    pint \
    rasterio \
    scipy \
    sparse \
    xarray
    zarr
# new matplotlib dependency
python -m pip install --upgrade contourpy
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
    git+https://github.com/h5netcdf/h5netcdf
    git+https://github.com/hgrecco/pint \
    git+https://github.com/numba/numba \
    git+https://github.com/intake/filesystem_spec \
    git+https://github.com/pydata/bottleneck \
    git+https://github.com/pydata/sparse \
    git+https://github.com/pypa/packaging \
    git+https://github.com/rasterio/rasterio \
    git+https://github.com/SciTools/nc-time-axis \
    git+https://github.com/Unidata/cftime \
    git+https://github.com/xarray-contrib/flox \
    git+https://github.com/zarr-developers/zarr
python -m pip install pytest-timeout
