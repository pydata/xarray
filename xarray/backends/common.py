from __future__ import absolute_import, division, print_function

import contextlib
import logging
import multiprocessing
import threading
import time
import traceback
import warnings
from collections import Mapping, OrderedDict
from functools import partial

import numpy as np

from ..conventions import cf_encoder
from ..core import indexing
from ..core.pycompat import dask_array_type, iteritems
from ..core.utils import FrozenOrderedDict, NdimSizeLenMixin

# Import default lock
try:
    from dask.utils import SerializableLock
    HDF5_LOCK = SerializableLock()
except ImportError:
    HDF5_LOCK = threading.Lock()

# Create a logger object, but don't add any handlers. Leave that to user code.
logger = logging.getLogger(__name__)


NONE_VAR_NAME = '__values__'


def _get_scheduler(get=None, collection=None):
    """ Determine the dask scheduler that is being used.

    None is returned if not dask scheduler is active.

    See also
    --------
    dask.base.get_scheduler
    """
    try:
        # dask 0.18.1 and later
        from dask.base import get_scheduler
        actual_get = get_scheduler(get, collection)
    except ImportError:
        try:
            from dask.utils import effective_get
            actual_get = effective_get(get, collection)
        except ImportError:
            return None

    try:
        from dask.distributed import Client
        if isinstance(actual_get.__self__, Client):
            return 'distributed'
    except (ImportError, AttributeError):
        try:
            import dask.multiprocessing
            if actual_get == dask.multiprocessing.get:
                return 'multiprocessing'
            else:
                return 'threaded'
        except ImportError:
            return 'threaded'


def _get_scheduler_lock(scheduler, path_or_file=None):
    """ Get the appropriate lock for a certain situation based onthe dask
       scheduler used.

    See Also
    --------
    dask.utils.get_scheduler_lock
    """

    if scheduler == 'distributed':
        from dask.distributed import Lock
        return Lock(path_or_file)
    elif scheduler == 'multiprocessing':
        return multiprocessing.Lock()
    elif scheduler == 'threaded':
        from dask.utils import SerializableLock
        return SerializableLock()
    else:
        return threading.Lock()


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


class CombinedLock(object):
    """A combination of multiple locks.

    Like a locked door, a CombinedLock is locked if any of its constituent
    locks are locked.
    """

    def __init__(self, locks):
        self.locks = tuple(set(locks))  # remove duplicates

    def acquire(self, *args):
        return all(lock.acquire(*args) for lock in self.locks)

    def release(self, *args):
        for lock in self.locks:
            lock.release(*args)

    def __enter__(self):
        for lock in self.locks:
            lock.__enter__()

    def __exit__(self, *args):
        for lock in self.locks:
            lock.__exit__(*args)

    @property
    def locked(self):
        return any(lock.locked for lock in self.locks)

    def __repr__(self):
        return "CombinedLock(%r)" % list(self.locks)


class BackendArray(NdimSizeLenMixin, indexing.ExplicitlyIndexed):

    def __array__(self, dtype=None):
        key = indexing.BasicIndexer((slice(None),) * self.ndim)
        return np.asarray(self[key], dtype=dtype)


class AbstractDataStore(Mapping):
    _autoclose = None
    _ds = None
    _isopen = False

    def __iter__(self):
        return iter(self.variables)

    def __getitem__(self, key):
        return self.variables[key]

    def __len__(self):
        return len(self.variables)

    def get_dimensions(self):  # pragma: no cover
        raise NotImplementedError

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

        For example::

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
        warnings.warn('The ``variables`` property has been deprecated and '
                      'will be removed in xarray v0.11.',
                      FutureWarning, stacklevel=2)
        variables, _ = self.load()
        return variables

    @property
    def attrs(self):
        warnings.warn('The ``attrs`` property has been deprecated and '
                      'will be removed in xarray v0.11.',
                      FutureWarning, stacklevel=2)
        _, attrs = self.load()
        return attrs

    @property
    def dimensions(self):
        warnings.warn('The ``dimensions`` property has been deprecated and '
                      'will be removed in xarray v0.11.',
                      FutureWarning, stacklevel=2)
        return self.get_dimensions()

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        self.close()


class ArrayWriter(object):
    def __init__(self, lock=HDF5_LOCK):
        self.sources = []
        self.targets = []
        self.lock = lock

    def add(self, source, target):
        if isinstance(source, dask_array_type):
            self.sources.append(source)
            self.targets.append(target)
        else:
            target[...] = source

    def sync(self, compute=True):
        if self.sources:
            import dask.array as da
            delayed_store = da.store(self.sources, self.targets,
                                     lock=self.lock, compute=compute,
                                     flush=True)
            self.sources = []
            self.targets = []
            return delayed_store


class AbstractWritableDataStore(AbstractDataStore):
    def __init__(self, writer=None, lock=HDF5_LOCK):
        if writer is None:
            writer = ArrayWriter(lock=lock)
        self.writer = writer
        self.delayed_store = None

    def encode(self, variables, attributes):
        """
        Encode the variables and attributes in this store

        Parameters
        ----------
        variables : dict-like
            Dictionary of key/value (variable name / xr.Variable) pairs
        attributes : dict-like
            Dictionary of key/value (attribute name / attribute) pairs

        Returns
        -------
        variables : dict-like
        attributes : dict-like

        """
        variables = OrderedDict([(k, self.encode_variable(v))
                                 for k, v in variables.items()])
        attributes = OrderedDict([(k, self.encode_attribute(v))
                                  for k, v in attributes.items()])
        return variables, attributes

    def encode_variable(self, v):
        """encode one variable"""
        return v

    def encode_attribute(self, a):
        """encode one attribute"""
        return a

    def set_dimension(self, d, l):  # pragma: no cover
        raise NotImplementedError

    def set_attribute(self, k, v):  # pragma: no cover
        raise NotImplementedError

    def set_variable(self, k, v):  # pragma: no cover
        raise NotImplementedError

    def sync(self, compute=True):
        if self._isopen and self._autoclose:
            # datastore will be reopened during write
            self.close()
        self.delayed_store = self.writer.sync(compute=compute)

    def store_dataset(self, dataset):
        """
        in stores, variables are all variables AND coordinates
        in xarray.Dataset variables are variables NOT coordinates,
        so here we pass the whole dataset in instead of doing
        dataset.variables
        """
        self.store(dataset, dataset.attrs)

    def store(self, variables, attributes, check_encoding_set=frozenset(),
              unlimited_dims=None):
        """
        Top level method for putting data on this store, this method:
          - encodes variables/attributes
          - sets dimensions
          - sets variables

        Parameters
        ----------
        variables : dict-like
            Dictionary of key/value (variable name / xr.Variable) pairs
        attributes : dict-like
            Dictionary of key/value (attribute name / attribute) pairs
        check_encoding_set : list-like
            List of variables that should be checked for invalid encoding
            values
        unlimited_dims : list-like
            List of dimension names that should be treated as unlimited
            dimensions.
        """

        variables, attributes = self.encode(variables, attributes)

        self.set_attributes(attributes)
        self.set_dimensions(variables, unlimited_dims=unlimited_dims)
        self.set_variables(variables, check_encoding_set,
                           unlimited_dims=unlimited_dims)

    def set_attributes(self, attributes):
        """
        This provides a centralized method to set the dataset attributes on the
        data store.

        Parameters
        ----------
        attributes : dict-like
            Dictionary of key/value (attribute name / attribute) pairs
        """
        for k, v in iteritems(attributes):
            self.set_attribute(k, v)

    def set_variables(self, variables, check_encoding_set,
                      unlimited_dims=None):
        """
        This provides a centralized method to set the variables on the data
        store.

        Parameters
        ----------
        variables : dict-like
            Dictionary of key/value (variable name / xr.Variable) pairs
        check_encoding_set : list-like
            List of variables that should be checked for invalid encoding
            values
        unlimited_dims : list-like
            List of dimension names that should be treated as unlimited
            dimensions.
        """

        for vn, v in iteritems(variables):
            name = _encode_variable_name(vn)
            check = vn in check_encoding_set
            target, source = self.prepare_variable(
                name, v, check, unlimited_dims=unlimited_dims)

            self.writer.add(source, target)

    def set_dimensions(self, variables, unlimited_dims=None):
        """
        This provides a centralized method to set the dimensions on the data
        store.

        Parameters
        ----------
        variables : dict-like
            Dictionary of key/value (variable name / xr.Variable) pairs
        unlimited_dims : list-like
            List of dimension names that should be treated as unlimited
            dimensions.
        """
        if unlimited_dims is None:
            unlimited_dims = set()

        existing_dims = self.get_dimensions()

        dims = OrderedDict()
        for v in unlimited_dims:  # put unlimited_dims first
            dims[v] = None
        for v in variables.values():
            dims.update(dict(zip(v.dims, v.shape)))

        for dim, length in dims.items():
            if dim in existing_dims and length != existing_dims[dim]:
                raise ValueError(
                    "Unable to update size for existing dimension"
                    "%r (%d != %d)" % (dim, length, existing_dims[dim]))
            elif dim not in existing_dims:
                is_unlimited = dim in unlimited_dims
                self.set_dimension(dim, length, is_unlimited)


class WritableCFDataStore(AbstractWritableDataStore):

    def encode(self, variables, attributes):
        # All NetCDF files get CF encoded by default, without this attempting
        # to write times, for example, would fail.
        variables, attributes = cf_encoder(variables, attributes)
        variables = OrderedDict([(k, self.encode_variable(v))
                                 for k, v in variables.items()])
        attributes = OrderedDict([(k, self.encode_attribute(v))
                                  for k, v in attributes.items()])
        return variables, attributes


class DataStorePickleMixin(object):
    """Subclasses must define `ds`, `_opener` and `_mode` attributes.

    Do not subclass this class: it is not part of xarray's external API.
    """

    def __getstate__(self):
        state = self.__dict__.copy()
        del state['_ds']
        del state['_isopen']
        if self._mode == 'w':
            # file has already been created, don't override when restoring
            state['_mode'] = 'a'
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._ds = None
        self._isopen = False

    @property
    def ds(self):
        if self._ds is not None and self._isopen:
            return self._ds
        ds = self._opener(mode=self._mode)
        self._isopen = True
        return ds

    @contextlib.contextmanager
    def ensure_open(self, autoclose=None):
        """
        Helper function to make sure datasets are closed and opened
        at appropriate times to avoid too many open file errors.

        Use requires `autoclose=True` argument to `open_mfdataset`.
        """

        if autoclose is None:
            autoclose = self._autoclose

        if not self._isopen:
            try:
                self._ds = self._opener()
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


class PickleByReconstructionWrapper(object):

    def __init__(self, opener, file, mode='r', **kwargs):
        self.opener = partial(opener, file, mode=mode, **kwargs)
        self.mode = mode
        self._ds = None

    @property
    def value(self):
        self._ds = self.opener()
        return self._ds

    def __getstate__(self):
        state = self.__dict__.copy()
        del state['_ds']
        if self.mode == 'w':
            # file has already been created, don't override when restoring
            state['mode'] = 'a'
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    def close(self):
        self._ds.close()
