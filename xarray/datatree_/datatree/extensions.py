from xarray.core.extensions import _register_accessor

from .datatree import DataTree


def register_datatree_accessor(name):
    """Register a custom accessor on DataTree objects.

    Parameters
    ----------
    name : str
        Name under which the accessor should be registered. A warning is issued
        if this name conflicts with a preexisting attribute.

    See Also
    --------
    xarray.register_dataarray_accessor
    xarray.register_dataset_accessor
    """
    return _register_accessor(name, DataTree)
