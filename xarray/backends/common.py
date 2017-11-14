from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import logging
import time
import traceback
import contextlib
from collections import Mapping
from distutils.version import LooseVersion

from ..conventions import cf_encoder
from ..core import indexing
from ..core.utils import FrozenOrderedDict, NdimSizeLenMixin
from ..core.pycompat import iteritems, dask_array_type

try:
    from dask.utils import SerializableLock as Lock
except ImportError:
    from threading import Lock


# Create a logger object, but don't add any handlers. Leave that to user code.
logger = logging.getLogger(__name__)


NONE_VAR_NAME = '__values__'


# dask.utils.SerializableLock if available, otherwise just a threading.Lock
GLOBAL_LOCK = Lock()


def _encode_variable_name(name):
    if name is None:
        name = NONE_VAR_NAME
    return name


def _decode_variable_name(name):
    if name == NONE_VAR_NAME:
        name = None
    return name


def find_root(ds):
    """
    Helper function to find the root of a netcdf or h5netcdf dataset.
    """
    while ds.parent is not None:
        ds = ds.parent
    return ds


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


class BackendArray(NdimSizeLenMixin, indexing.ExplicitlyIndexed):

    def __array__(self, dtype=None):
        key = indexing.BasicIndexer((slice(None),) * self.ndim)
        return np.asarray(self[key], dtype=dtype)


class AbstractDataStore(Mapping):
    _autoclose = False

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

    def get_encoding(self):
        return {}

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
                                      for k, v in self.get_variables().items())
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
            try:
                target[...] = source
            except TypeError:
                # workaround for GH: scipy/scipy#6880
                target[:] = source

    def sync(self):
        if self.sources:
            import dask.array as da
            import dask
            if LooseVersion(dask.__version__) > LooseVersion('0.8.1'):
                da.store(self.sources, self.targets, lock=GLOBAL_LOCK)
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

    def store(self, variables, attributes, check_encoding_set=frozenset(),
              unlimited_dims=None):
        self.set_attributes(attributes)
        self.set_variables(variables, check_encoding_set,
                           unlimited_dims=unlimited_dims)

    def set_attributes(self, attributes):
        for k, v in iteritems(attributes):
            self.set_attribute(k, v)

    def set_variables(self, variables, check_encoding_set,
                      unlimited_dims=None):
        for vn, v in iteritems(variables):
            name = _encode_variable_name(vn)
            check = vn in check_encoding_set
            if vn not in self.variables:
                target, source = self.prepare_variable(
                    name, v, check, unlimited_dims=unlimited_dims)
            else:
                target, source = self.ds.variables[name], v.data

            self.writer.add(source, target)

    def set_necessary_dimensions(self, variable, unlimited_dims=None):
        if unlimited_dims is None:
            unlimited_dims = set()
        for d, l in zip(variable.dims, variable.shape):
            if d not in self.dimensions:
                is_unlimited = d in unlimited_dims
                self.set_dimension(d, l, is_unlimited)


class WritableCFDataStore(AbstractWritableDataStore):

    def store(self, variables, attributes, *args, **kwargs):
        # All NetCDF files get CF encoded by default, without this attempting
        # to write times, for example, would fail.
        cf_variables, cf_attrs = cf_encoder(variables, attributes)
        AbstractWritableDataStore.store(self, cf_variables, cf_attrs,
                                        *args, **kwargs)


class DataStorePickleMixin(object):
    """Subclasses must define `ds`, `_opener` and `_mode` attributes.

    Do not subclass this class: it is not part of xarray's external API.
    """

    def __getstate__(self):
        state = self.__dict__.copy()
        del state['ds']
        if self._mode == 'w':
            # file has already been created, don't override when restoring
            state['_mode'] = 'a'
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.ds = self._opener(mode=self._mode)

    @contextlib.contextmanager
    def ensure_open(self, autoclose):
        """
        Helper function to make sure datasets are closed and opened
        at appropriate times to avoid too many open file errors.

        Use requires `autoclose=True` argument to `open_mfdataset`.
        """
        if self._autoclose and not self._isopen:
            try:
                self.ds = self._opener()
                self._isopen = True
                yield
            finally:
                if autoclose:
                    self.close()
        else:
            yield

    def assert_open(self):
        if not self._isopen:
            raise AssertionError('internal failure: file must be open '
                                 'if `autoclose=True` is used.')
