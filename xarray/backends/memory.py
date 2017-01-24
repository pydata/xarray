from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import copy

import numpy as np

from ..core.variable import Variable
from ..core.pycompat import OrderedDict

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

    def prepare_variable(self, k, v, *args, **kwargs):
        new_var = Variable(v.dims, np.empty_like(v), v.attrs)
        # we copy the variable and stuff all encodings in the
        # attributes to imitate what happens when writing to disk.
        new_var.attrs.update(v.encoding)
        self._variables[k] = new_var
        return new_var, v.data

    def set_attribute(self, k, v):
        # copy to imitate writing to disk.
        self._attributes[k] = copy.deepcopy(v)

    def set_dimension(self, d, l):
        # in this model, dimensions are accounted for in the variables
        pass
