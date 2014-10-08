from ..core.pycompat import OrderedDict
import copy

from .common import AbstractWritableDataStore


class InMemoryDataStore(AbstractWritableDataStore):
    """
    Stores dimensions, variables and attributes
    in ordered dictionaries, making this store
    fast compared to stores which save to disk.
    """
    def __init__(self, dict_store=None):
        if dict_store is None:
            dict_store = {}
            dict_store['variables'] = OrderedDict()
            dict_store['attributes'] = OrderedDict()
        self.ds = dict_store

    def get_attrs(self):
        return self.ds['attributes']

    def get_variables(self):
        return self.ds['variables']

    def set_variable(self, k, v):
        new_var = copy.deepcopy(v)
        # we copy the variable and stuff all encodings in the
        # attributes to imitate what happens when writing to disk.
        new_var.attrs.update(new_var.encoding)
        new_var.encoding.clear()
        print self.ds['variables'].keys()
        self.ds['variables'][k] = new_var

    def set_attribute(self, k, v):
        # copy to imitate writing to disk.
        self.ds['attributes'][k] = copy.deepcopy(v)

    def set_dimension(self, d, l):
        # in this model, dimensions are accounted for in the variables
        pass