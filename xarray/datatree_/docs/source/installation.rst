============
Installation
============

Datatree is not yet available on pypi or via conda, so for now you will have to install it from source.

``git clone https://github.com/TomNicholas/datatree.git```

``pip install -e ./datatree/``

The main branch will be kept up-to-date, so if you clone main and run the test suite with ``pytest datatree`` and get no failures,
then you have the most up-to-date version.

You will need xarray and `anytree <https://github.com/c0fec0de/anytree>`_
as dependencies, with netcdf4, zarr, and h5netcdf as optional dependencies to allow file I/O.

.. note::

    Datatree is very much still in the early stages of development. There may be functions that are present but whose
    internals are not yet implemented, or significant changes to the API in future.
    That said, if you try it out and find some behaviour that looks like a bug to you, please report it on the
    `issue tracker <https://github.com/TomNicholas/datatree/issues>`_!
