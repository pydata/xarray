# datatree

| CI          | [![GitHub Workflow Status][github-ci-badge]][github-ci-link] [![Code Coverage Status][codecov-badge]][codecov-link] [![pre-commit.ci status][pre-commit.ci-badge]][pre-commit.ci-link] |
| :---------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| **Docs**    |                                                                     [![Documentation Status][rtd-badge]][rtd-link]                                                                     |
| **Package** |                                                          [![Conda][conda-badge]][conda-link] [![PyPI][pypi-badge]][pypi-link]                                                          |
| **License** |                                                                         [![License][license-badge]][repo-link]                                                                         |


WIP implementation of a tree-like hierarchical data structure for xarray.

This aims to create the data structure discussed in [xarray issue #4118](https://github.com/pydata/xarray/issues/4118), and therefore extend xarray's data model to be able to [handle arbitrarily nested netCDF4 groups](https://github.com/pydata/xarray/issues/1092#issuecomment-868324949).


The approach used here is based on benbovy's [`DatasetNode` example](https://gist.github.com/benbovy/92e7c76220af1aaa4b3a0b65374e233a) - the basic idea is that each tree node wraps a up to a single `xarray.Dataset`. The differences are that this effort:
- Uses a node structure inspired by [anytree](https://github.com/TomNicholas/datatree/issues/7) for the tree,
- Implements path-like getting and setting,
- Has functions for mapping user-supplied functions over every node in the tree,
- Automatically dispatches *some* of `xarray.Dataset`'s API over every node in the tree (such as `.isel`),
- Has a bunch of tests,
- Has a printable representation that currently looks like this:
<img src="https://user-images.githubusercontent.com/35968931/130657849-577faa00-1b8b-4e33-a45c-4f389ce325b2.png" alt="drawing" width="500"/>

You can create a `DataTree` object in 3 ways:
1) Load from a netCDF file (or Zarr store) that has groups via `open_datatree()`.
2) Using the init method of `DataTree`, which creates an individual node.
  You can then specify the nodes' relationships to one other, either by setting `.parent` and `.chlldren` attributes,
  or through `__get/setitem__` access, e.g. `dt['path/to/node'] = DataTree()`.
3) Create a tree from a dictionary of paths to datasets using `DataTree.from_dict()`.



[github-ci-badge]: https://img.shields.io/github/workflow/status/xarray-contrib/datatree/CI?label=CI&logo=github
[github-ci-link]: https://github.com/xarray-contrib/datatree/actions?query=workflow%3ACI
[codecov-badge]: https://img.shields.io/codecov/c/github/xarray-contrib/datatree.svg?logo=codecov
[codecov-link]: https://codecov.io/gh/xarray-contrib/datatree
[rtd-badge]: https://img.shields.io/readthedocs/xarray-datatree/latest.svg
[rtd-link]: https://xarray-datatree.readthedocs.io/en/latest/?badge=latest
[pypi-badge]: https://img.shields.io/pypi/v/xarray-datatree?logo=pypi
[pypi-link]: https://pypi.org/project/xarray-datatree
[conda-badge]: https://img.shields.io/conda/vn/conda-forge/xarray-datatree?logo=anaconda
[conda-link]: https://anaconda.org/conda-forge/xarray-datatree
[license-badge]: https://img.shields.io/github/license/xarray-contrib/datatree
[repo-link]: https://github.com/xarray-contrib/datatree
[pre-commit.ci-badge]: https://results.pre-commit.ci/badge/github/xarray-contrib/datatree/main.svg
[pre-commit.ci-link]: https://results.pre-commit.ci/latest/github/xarray-contrib/datatree/main
