#!/usr/bin/env bash

# TODO: add sparse back in, once Numba works with the development version of
# NumPy again: https://github.com/pydata/xarray/issues/4146

conda uninstall -y --force \
    numpy \
    scipy \
    pandas \
    matplotlib \
    dask \
    distributed \
    zarr \
    cftime \
    rasterio \
    pint \
    bottleneck \
    sparse
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
    git+https://github.com/pydata/bottleneck # \
    # git+https://github.com/pydata/sparse
