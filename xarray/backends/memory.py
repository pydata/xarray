from __future__ import absolute_import, division, print_function

import copy

import numpy as np

from ..core.pycompat import OrderedDict
from ..core.variable import Variable
from .common import AbstractWritableDataStore


class InMemoryDataStore(AbstractWritableDataStore):
    """
    Stores dimensions, variables and attributes in ordered dictionaries, making
    this store fast compared to stores which save to disk.

    This store exists purely for internal testing purposes.
    """

    def __init__(self, variables=None, attributes=None, writer=None):
        self._variables = OrderedDict() if variables is None else variables
        self._attributes = OrderedDict() if attributes is None else attributes
        super(InMemoryDataStore, self).__init__(writer)

    def get_attrs(self):
        return self._attributes

    def get_variables(self):
        return self._variables

    def get_dimensions(self):
        dims = OrderedDict()
        for v in self._variables.values():
            for d, s in v.dims.items():
                dims[d] = s
        return dims

    def prepare_variable(self, k, v, *args, **kwargs):
        new_var = Variable(v.dims, np.empty_like(v), v.attrs)
        self._variables[k] = new_var
        return new_var, v.data

    def set_attribute(self, k, v):
        # copy to imitate writing to disk.
        self._attributes[k] = copy.deepcopy(v)

    def set_dimension(self, d, l, unlimited_dims=None):
        # in this model, dimensions are accounted for in the variables
        pass
