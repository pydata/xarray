#!/usr/bin/env bash

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
    rasterio \
    pint \
    bottleneck \
    sparse \
    xarray
# to limit the runtime of Upstream CI
python -m pip install pytest-timeout
python -m pip install \
    -i https://pypi.anaconda.org/scipy-wheels-nightly/simple \
    --no-deps \
    --pre \
    --upgrade \
    numpy \
    scipy \
    pandas
python -m pip install \
    -f https://7933911d6844c6c53a7d-47bd50c35cd79bd838daf386af554a83.ssl.cf2.rackcdn.com \
    --no-deps \
    --pre \
    --upgrade \
    matplotlib
python -m pip install \
    --no-deps \
    --upgrade \
    git+https://github.com/dask/dask \
    git+https://github.com/dask/distributed \
    git+https://github.com/zarr-developers/zarr \
    git+https://github.com/Unidata/cftime \
    git+https://github.com/mapbox/rasterio \
    git+https://github.com/hgrecco/pint \
    git+https://github.com/pydata/bottleneck \
    git+https://github.com/pydata/sparse \
    git+https://github.com/intake/filesystem_spec \
    git+https://github.com/SciTools/nc-time-axis
