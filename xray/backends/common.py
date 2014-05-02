from xray.utils import FrozenOrderedDict


class AbstractDataStore(object):
    def open_store_variable(self, v):
        raise NotImplementedError

    @property
    def store_variables(self):
        return self.ds.variables

    @property
    def variables(self):
        return FrozenOrderedDict((k, self.open_store_variable(v))
                                 for k, v in self.store_variables.iteritems())


class AbstractWritableDataStore(AbstractDataStore):
    def set_dimensions(self, dimensions):
        for d, l in dimensions.iteritems():
            self.set_dimension(d, l)

    def set_attributes(self, attributes):
        for k, v in attributes.iteritems():
            self.set_attribute(k, v)

    def set_variables(self, variables):
        for vn, v in variables.iteritems():
            self.set_variable(vn, v)

    def set_necessary_dimensions(self, variable):
        for d, l in zip(variable.dimensions, variable.shape):
            if d not in self.ds.dimensions:
                self.set_dimension(d, l)
