.. currentmodule:: datatree

============
Installation
============

Datatree can be installed in three ways:

Using the `conda <https://conda.io/>`__ package manager that comes with the
Anaconda/Miniconda distribution:

.. code:: bash

    $ conda install xarray-datatree --channel conda-forge

Using the `pip <https://pypi.org/project/pip/>`__ package manager:

.. code:: bash

    $ python -m pip install xarray-datatree

To install a development version from source:

.. code:: bash

    $ git clone https://github.com/xarray-contrib/datatree
    $ cd datatree
    $ python -m pip install -e .


You will just need xarray as a required dependency, with netcdf4, zarr, and h5netcdf as optional dependencies to allow file I/O.

.. note::

    Datatree is very much still in the early stages of development. There may be functions that are present but whose
    internals are not yet implemented, or significant changes to the API in future.
    That said, if you try it out and find some behaviour that looks like a bug to you, please report it on the
    `issue tracker <https://github.com/xarray-contrib/datatree/issues>`_!
