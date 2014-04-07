class AbstractDataStore(object):
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
