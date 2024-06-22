# datatree

| CI          | [![GitHub Workflow Status][github-ci-badge]][github-ci-link] [![Code Coverage Status][codecov-badge]][codecov-link] [![pre-commit.ci status][pre-commit.ci-badge]][pre-commit.ci-link] |
| :---------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| **Docs**    |                                                                     [![Documentation Status][rtd-badge]][rtd-link]                                                                     |
| **Package** |                                                          [![Conda][conda-badge]][conda-link] [![PyPI][pypi-badge]][pypi-link]                                                          |
| **License** |                                                                         [![License][license-badge]][repo-link]                                                                         |


**Datatree is a prototype implementation of a tree-like hierarchical data structure for xarray.**

Datatree was born after the xarray team recognised a [need for a new hierarchical data structure](https://github.com/pydata/xarray/issues/4118),
that was more flexible than a single `xarray.Dataset` object.
The initial motivation was to represent netCDF files / Zarr stores with multiple nested groups in a single in-memory object,
but `datatree.DataTree` objects have many other uses.

### DEPRECATION NOTICE

Datatree is in the process of being merged upstream into xarray (as of [v0.0.14](https://github.com/xarray-contrib/datatree/releases/tag/v0.0.14), see xarray issue [#8572](https://github.com/pydata/xarray/issues/8572)). We are aiming to preserve the record of contributions to this repository during the migration process. However whilst we will hapily accept new PRs to this repository, this repo will be deprecated and any PRs since [v0.0.14](https://github.com/xarray-contrib/datatree/releases/tag/v0.0.14) might be later copied across to xarray without full git attribution.

Hopefully for users the disruption will be minimal - and just mean that in some future version of xarray you only need to do `from xarray import DataTree` rather than `from datatree import DataTree`. Once the migration is complete this repository will be archived.

### Installation
You can install datatree via pip:
```shell
pip install xarray-datatree
```

or via conda-forge
```shell
conda install -c conda-forge xarray-datatree
```

### Why Datatree?

You might want to use datatree for:

- Organising many related datasets, e.g. results of the same experiment with different parameters, or simulations of the same system using different models,
- Analysing similar data at multiple resolutions simultaneously, such as when doing a convergence study,
- Comparing heterogenous but related data, such as experimental and theoretical data,
- I/O with nested data formats such as netCDF / Zarr groups.

[**Talk slides on Datatree from AMS-python 2023**](https://speakerdeck.com/tomnicholas/xarray-datatree-hierarchical-data-structures-for-multi-model-science)

### Features

The approach used here is based on benbovy's [`DatasetNode` example](https://gist.github.com/benbovy/92e7c76220af1aaa4b3a0b65374e233a) - the basic idea is that each tree node wraps a up to a single `xarray.Dataset`. The differences are that this effort:
- Uses a node structure inspired by [anytree](https://github.com/xarray-contrib/datatree/issues/7) for the tree,
- Implements path-like getting and setting,
- Has functions for mapping user-supplied functions over every node in the tree,
- Automatically dispatches *some* of `xarray.Dataset`'s API over every node in the tree (such as `.isel`),
- Has a bunch of tests,
- Has a printable representation that currently looks like this:
<img src="https://user-images.githubusercontent.com/35968931/130657849-577faa00-1b8b-4e33-a45c-4f389ce325b2.png" alt="drawing" width="500"/>

### Get Started

You can create a `DataTree` object in 3 ways:
1) Load from a netCDF file (or Zarr store) that has groups via `open_datatree()`.
2) Using the init method of `DataTree`, which creates an individual node.
  You can then specify the nodes' relationships to one other, either by setting `.parent` and `.children` attributes,
  or through `__get/setitem__` access, e.g. `dt['path/to/node'] = DataTree()`.
3) Create a tree from a dictionary of paths to datasets using `DataTree.from_dict()`.

### Development Roadmap

Datatree currently lives in a separate repository to the main xarray package.
This allows the datatree developers to make changes to it, experiment, and improve it faster.

Eventually we plan to fully integrate datatree upstream into xarray's main codebase, at which point the [github.com/xarray-contrib/datatree](https://github.com/xarray-contrib/datatree>) repository will be archived.
This should not cause much disruption to code that depends on datatree - you will likely only have to change the import line (i.e. from ``from datatree import DataTree`` to ``from xarray import DataTree``).

However, until this full integration occurs, datatree's API should not be considered to have the same [level of stability as xarray's](https://docs.xarray.dev/en/stable/contributing.html#backwards-compatibility).

### User Feedback

We really really really want to hear your opinions on datatree!
At this point in development, user feedback is critical to help us create something that will suit everyone's needs.
Please raise any thoughts, issues, suggestions or bugs, no matter how small or large, on the [github issue tracker](https://github.com/xarray-contrib/datatree/issues).


[github-ci-badge]: https://img.shields.io/github/actions/workflow/status/xarray-contrib/datatree/main.yaml?branch=main&label=CI&logo=github
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
