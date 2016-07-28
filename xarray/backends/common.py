import numpy as np
import itertools
import logging
import time
import traceback
import threading
from collections import Mapping
from distutils.version import StrictVersion

from ..conventions import cf_encoder
from ..core.utils import FrozenOrderedDict
from ..core.pycompat import iteritems, dask_array_type, OrderedDict

# Create a logger object, but don't add any handlers. Leave that to user code.
logger = logging.getLogger(__name__)


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
    # the index is not trivial.
    if len(var.attrs) or len(var.encoding):
        return False
    # if the index is not a 1d integer array
    if var.ndim > 1 or not var.dtype.kind == 'i':
        return False
    arange = np.arange(var.size, dtype=var.dtype)
    return np.all(var.values == arange)


def robust_getitem(array, key, catch=Exception, max_retries=6,
                   initial_delay=500):
    """
    Robustly index an array, using retry logic with exponential backoff if any
    of the errors ``catch`` are raised. The initial_delay is measured in ms.

    With the default settings, the maximum delay will be in the range of 32-64
    seconds.
    """
    assert max_retries >= 0
    for n in range(max_retries + 1):
        try:
            return array[key]
        except catch:
            if n == max_retries:
                raise
            base_delay = initial_delay * 2 ** n
            next_delay = base_delay + np.random.randint(base_delay)
            msg = ('getitem failed, waiting %s ms before trying again '
                   '(%s tries remaining). Full traceback: %s' %
                   (next_delay, max_retries - n, traceback.format_exc()))
            logger.debug(msg)
            time.sleep(1e-3 * next_delay)


class AbstractDataStore(Mapping):

    def __iter__(self):
        return iter(self.variables)

    def __getitem__(self, key):
        return self.variables[key]

    def __len__(self):
        return len(self.variables)

    def get_attrs(self):  # pragma: no cover
        raise NotImplementedError

    def get_variables(self):  # pragma: no cover
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

    def __exit__(self, exception_type, exception_value, traceback):
        self.close()


class ArrayWriter(object):
    def __init__(self):
        self.sources = []
        self.targets = []

    def add(self, source, target):
        if isinstance(source, dask_array_type):
            self.sources.append(source)
            self.targets.append(target)
        else:
            target[...] = source

    def sync(self):
        if self.sources:
            import dask.array as da
            import dask
            if StrictVersion(dask.__version__) > StrictVersion('0.8.1'):
                da.store(self.sources, self.targets, lock=threading.Lock())
            else:
                da.store(self.sources, self.targets)
            self.sources = []
            self.targets = []


class AbstractWritableDataStore(AbstractDataStore):
    def __init__(self, writer=None):
        if writer is None:
            writer = ArrayWriter()
        self.writer = writer

    def set_dimension(self, d, l):  # pragma: no cover
        raise NotImplementedError

    def set_attribute(self, k, v):  # pragma: no cover
        raise NotImplementedError

    def set_variable(self, k, v):  # pragma: no cover
        raise NotImplementedError

    def sync(self):
        self.writer.sync()

    def store_dataset(self, dataset):
        # in stores variables are all variables AND coordinates
        # in xarray.Dataset variables are variables NOT coordinates,
        # so here we pass the whole dataset in instead of doing
        # dataset.variables
        self.store(dataset, dataset.attrs)

    def store(self, variables, attributes, check_encoding_set=frozenset()):
        self.set_attributes(attributes)
        neccesary_dims = [v.dims for v in variables.values()]
        neccesary_dims = set(itertools.chain(*neccesary_dims))
        # set all non-indexes and any index which is not trivial.
        variables = OrderedDict((k, v) for k, v in iteritems(variables)
                                if not (k in neccesary_dims and
                                        is_trivial_index(v)))
        self.set_variables(variables, check_encoding_set)

    def set_attributes(self, attributes):
        for k, v in iteritems(attributes):
            self.set_attribute(k, v)

    def set_variables(self, variables, check_encoding_set):
        for vn, v in iteritems(variables):
            name = _encode_variable_name(vn)
            check = vn in check_encoding_set
            target, source = self.prepare_variable(name, v, check)
            self.writer.add(source, target)

    def set_necessary_dimensions(self, variable):
        for d, l in zip(variable.dims, variable.shape):
            if d not in self.dimensions:
                self.set_dimension(d, l)


class WritableCFDataStore(AbstractWritableDataStore):
    def store(self, variables, attributes, check_encoding_set=frozenset()):
        # All NetCDF files get CF encoded by default, without this attempting
        # to write times, for example, would fail.
        cf_variables, cf_attrs = cf_encoder(variables, attributes)
        AbstractWritableDataStore.store(self, cf_variables, cf_attrs,
                                        check_encoding_set)
