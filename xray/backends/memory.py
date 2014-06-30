from xray.pycompat import OrderedDict

from .common import AbstractWritableDataStore


class InMemoryDataStore(AbstractWritableDataStore):
    """
    Stores dimensions, variables and attributes
    in ordered dictionaries, making this store
    fast compared to stores which store to disk.
    """
    def __init__(self):
        self.dimensions = OrderedDict()
        self.variables = OrderedDict()
        self.attributes = OrderedDict()

    def set_dimension(self, name, length):
        self.dimensions[name] = length

    def set_attribute(self, key, value):
        self.attributes[key] = value

    def set_variable(self, name, variable):
        self.variables[name] = variable
        return self.variables[name]

    def del_attribute(self, key):
        del self.attributes[key]
