import numpy as np
import itertools

from collections import Mapping

from ..core.utils import FrozenOrderedDict
from ..core.pycompat import iteritems
from ..core.variable import Coordinate


NONE_VAR_NAME = '__values__'


def _encode_variable_name(name):
    if name is None:
        name = NONE_VAR_NAME
    return name


def _decode_variable_name(name):
    if name == NONE_VAR_NAME:
        name = None
    return name


def is_trivial_index(var):
    """
    Determines if in index is 'trivial' meaning that it is
    equivalent to np.arange().  This is determined by
    checking if there are any attributes or encodings,
    if ndims is one, dtype is int and finally by comparing
    the actual values to np.arange()
    """
    # if either attributes or encodings are defined
    # the index is not trival.
    if len(var.attrs) or len(var.encoding):
        return False
    # if the index is not a 1d integer array
    if var.ndim > 1 or not var.dtype.kind == 'i':
        return False
    arange = np.arange(var.size, dtype=var.dtype)
    return np.all(var.values == arange)


class AbstractDataStore(Mapping):

    def __iter__(self):
        return iter(self.variables)

    def __getitem__(self, key):
        return self.variables[key]

    def __len__(self):
        return len(self.variables)

    def get_attrs(self):
        raise NotImplementedError

    def get_variables(self):
        raise NotImplementedError

    def load(self):
        """
        This loads the variables and attributes simultaneously.
        A centralized loading function makes it easier to create
        data stores that do automatic encoding/decoding.

        For example:

            class SuffixAppendingDataStore(AbstractDataStore):

                def load(self):
                    variables, attributes = AbstractDataStore.load(self)
                    variables = {'%s_suffix' % k: v
                                 for k, v in iteritems(variables)}
                    attributes = {'%s_suffix' % k: v
                                  for k, v in iteritems(attributes)}
                    return variables, attributes

        This function will be called anytime variables or attributes
        are requested, so care should be taken to make sure its fast.
        """
        variables = FrozenOrderedDict((_decode_variable_name(k), v)
                                      for k, v in iteritems(self.get_variables()))
        attributes = FrozenOrderedDict(self.get_attrs())
        return variables, attributes

    def get_dimensions(self):
        return list(itertools.chain(*[x.dims
                                      for x in self.variables.values()]))

    @property
    def variables(self):
        # Because encoding/decoding might happen which may require both the
        # attributes and the variables, and because a store may be updated
        # we need to load both the attributes and variables
        # anytime either one is requested.
        variables, _ = self.load()
        return variables

    @property
    def attrs(self):
        # Because encoding/decoding might happen which may require both the
        # attributes and the variables, and because a store may be updated
        # we need to load both the attributes and variables
        # anytime either one is requested.
        _, attributes = self.load()
        return attributes

    @property
    def dimensions(self):
        return self.get_dimensions()

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, tracebook):
        self.close()


class AbstractWritableDataStore(AbstractDataStore):

    def set_dimension(self, d, l):
        raise NotImplementedError

    def set_attribute(self, k, v):
        raise NotImplementedError

    def set_variable(self, k, v):
        raise NotImplementedError

    def sync(self):
        pass

    def store_dataset(self, dataset):
        # in stores variables are all variables AND coordinates
        # in xray.Dataset variables are variables NOT coordinates,
        # so here we pass the whole dataset in instead of doing
        # dataset.variables
        self.store(dataset, dataset.attrs)

    def store(self, variables, attributes):
        self.set_attributes(attributes)
        neccesary_dims = [v.dims for v in variables.values()]
        neccesary_dims = set(itertools.chain(*neccesary_dims))
        # set all non-indexes and any index which is not trivial.
        variables = dict((k, v) for k, v in iteritems(variables)
                         if not (k in neccesary_dims and is_trivial_index(v)))
        self.set_variables(variables)

    def set_dimensions(self, dimensions):
        for d, l in iteritems(dimensions):
            self.set_dimension(d, l)

    def set_attributes(self, attributes):
        for k, v in iteritems(attributes):
            self.set_attribute(k, v)

    def set_variables(self, variables):
        for vn, v in iteritems(variables):
            self.set_variable(_encode_variable_name(vn), v)
            self.set_necessary_dimensions(v)

    def set_necessary_dimensions(self, variable):
        for d, l in zip(variable.dims, variable.shape):
            if d not in self.dimensions:
                self.set_dimension(d, l)
