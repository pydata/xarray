#!/usr/bin/env bash

if which micromamba >/dev/null; then
    conda=micromamba
elif which mamba >/dev/null; then
    conda=mamba
else
    conda=conda
fi

# temporarily (?) remove sparse
$conda remove -y sparse
# temporarily remove numexpr
$conda remove -y numexpr
# temporarily remove backends
$conda remove -y cf_units hdf5 h5py netcdf4 pydap
# forcibly remove packages to avoid artifacts
$conda remove -y --force \
    bottleneck \
    cftime \
    distributed \
    flox \
    fsspec \
    numbagg \
    numpy \
    packaging \
    pandas \
    scipy \
    zarr
# pint

# to limit the runtime of Upstream CI
python -m pip install \
    -i https://pypi.anaconda.org/scientific-python-nightly-wheels/simple \
    --no-deps \
    --pre \
    --upgrade \
    numpy \
    scipy \
    matplotlib \
    pandas \
    h5py
# for some reason pandas depends on pyarrow already.
# Remove once a `pyarrow` version compiled with `numpy>=2.0` is on `conda-forge`
python -m pip install \
    -i https://pypi.fury.io/arrow-nightlies/ \
    --prefer-binary \
    --no-deps \
    --pre \
    --upgrade \
    pyarrow
# manually install `pint` to pull in new dependencies
python -m pip install --upgrade pint

python -m pip install \
    --no-deps \
    --upgrade \
    git+https://github.com/dask/dask \
    git+https://github.com/dask/dask-expr \
    git+https://github.com/dask/distributed \
    git+https://github.com/dgasmith/opt_einsum \
    git+https://github.com/h5netcdf/h5netcdf \
    git+https://github.com/hgrecco/pint \
    git+https://github.com/intake/filesystem_spec \
    git+https://github.com/numbagg/numbagg \
    git+https://github.com/pydata/bottleneck \
    git+https://github.com/pypa/packaging \
    git+https://github.com/SciTools/nc-time-axis \
    git+https://github.com/Unidata/cftime \
    git+https://github.com/xarray-contrib/flox \
    git+https://github.com/zarr-developers/zarr.git@main
# git+https://github.com/pydata/sparse
