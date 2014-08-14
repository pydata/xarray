import numpy as np

from xray.utils import FrozenOrderedDict
from xray.pycompat import iteritems


NONE_VAR_NAME = '__values__'


def _encode_variable_name(name):
    if name is None:
        name = NONE_VAR_NAME
    return name


def _decode_variable_name(name):
    if name == NONE_VAR_NAME:
        name = None
    return name


class AbstractDataStore(object):
    def open_store_variable(self, v):
        raise NotImplementedError

    @property
    def store_variables(self):
        return self.ds.variables

    @property
    def variables(self):
        return FrozenOrderedDict((_decode_variable_name(k),
                                  self.open_store_variable(v))
                                 for k, v in iteritems(self.store_variables))

    def sync(self):
        pass

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, tracebook):
        self.close()


class AbstractWritableDataStore(AbstractDataStore):
    def set_dimensions(self, dimensions):
        for d, l in iteritems(dimensions):
            self.set_dimension(d, l)

    def set_attributes(self, attributes):
        for k, v in iteritems(attributes):
            self.set_attribute(k, v)

    def set_variables(self, variables):
        for vn, v in iteritems(variables):
            self.set_variable(_encode_variable_name(vn), v)

    def set_necessary_dimensions(self, variable):
        for d, l in zip(variable.dims, variable.shape):
            if d not in self.ds.dimensions:
                self.set_dimension(d, l)
