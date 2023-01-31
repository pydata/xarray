from xarray.tests import requires_datatree


@requires_datatree
def test_import_datatree():
    """Just test importing datatree package from xarray-contrib repo"""
    from xarray import DataTree, open_datatree

    DataTree()
    