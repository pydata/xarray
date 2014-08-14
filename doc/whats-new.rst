What's New
==========

v0.2.0 (14 August 2014)
-----------------------

This is major release that includes some new features and quite a few bug
fixes. Here are the highlights:

- There is now a direct constructor for ``DataArray`` objects, which makes it
  possible to create a DataArray without using a Dataset. This is highlighted
  in the refreshed :doc:`tutorial`.
- You can perform aggregation operations like ``mean`` directly on
  :py:class:`~xray.Dataset` objects, thanks to Joe Hamman. These aggregation
  methods also worked on grouped datasets.
- xray now works on Python 2.6, thanks to Anna Kuznetsova.
- A number of methods and attributes were given more sensible (usually shorter)
  names: ``labeled`` -> ``sel``,  ``indexed`` -> ``isel``, ``select`` ->
  ``select_vars``, ``unselect`` -> ``drop_vars``, ``dimensions`` -> ``dims``,
  ``coordinates`` -> ``coords``, ``attributes`` -> ``attrs``.
- New :py:meth`~xray.Dataset.load_data` and :py:meth`~xray.Dataset.close`
  methods for datasets facilitate lower level of control of data loaded from
  disk.

v0.1.1 (20 May 2014)
--------------------

xray 0.1.1 is a bug-fix release that includes changes that should be almost
entirely backwards compatible with v0.1:

- Python 3 support (:issue:`53`)
- Required numpy version relaxed to 1.7 (:issue:`129`)
- Return numpy.datetime64 arrays for non-standard calendars (:issue:`126`)
- Support for opening datasets associated with NetCDF4 groups (:issue:`127`)
- Bug-fixes for concatenating datetime arrays (:issue:`134`)

Special thanks to new contributors Thomas Kluyver, Joe Hamman and Alistair
Miles.

v0.1 (2 May 2014)
-----------------

Initial release.
