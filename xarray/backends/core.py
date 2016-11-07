from collections import Mapping, defaultdict
import copy
import os.path
import numpy as np
import pickle
import threading
from xarray.core.pycompat import OrderedDict, dask_array_type

from xarray import Variable


class AbstractDecoder(object):
    """Callable for decoding variables and attributes into an xarray.Dataset.

    Because we need to put Decoders in tokens for use with dask, they should be
    pickleable.
    """

    def __call__(self, variables, attrs):
        # type: (Mapping[Any, xarray.Variable], Mapping) -> Dataset
        """Convert variables and attributes into an xarray.Dataset.

        Parameters
        ----------
        variables : Mapping[Any, xarray.Variable]
            Mapping of xarray.Variable objects to use in the Dataset.
        attrs : Mapping
            Mapping of global attributes.

        Returns
        -------
        xarray.Dataset with the decoded contents of ``variables`` and ``attrs``.
        """
        raise NotImplementedError


class AbstractEncoder(object):
    """Callable for decoding variables and attributes into an xarray.Dataset.

    For use with dask, Encoders should be pickleable.
    """

    def __call__(self, dataset):
        # type: (xarray.Dataset,) -> (Mapping[Any, xarray.Variable], Mapping)
        """Convert an xarray.Dataset into maps of variables and attributes.

        Recognized encoding options may be removed from
        `variable.encodings` and used to encode the data (generally, they should
        then be put into `variable.attrs`. Unrecognized encodings should be
        passed on unchanged.

        Parameters
        ----------
        dataset : xarray.Dataset
            Dataset to encode.

        Returns
        -------
        variables : Mapping[Any, xarray.Variable]
            Mapping of variables to save into a DataStore. Should only have
            dtypes supported by the DataStore.
        attrs : Mapping
            Mapping of global attributes for the DataStore.
        """
        raise NotImplementedError


class DummyDecoder(AbstractDecoder):
    def __call__(self, variables, attrs):
        return xarray.Dataset(variables, attrs=attrs)


class DummyEncoder(AbstractDecoder):
    def __call__(self, dataset):
        return dataset.variables, dataset.attrs


# CFDecoder and CFEncoder would probably live in conventions.py:

class CFDecoder(AbstractDecoder):
    def __init__(self, concat_characters=True, mask_and_scale=True,
                 decode_times=True, decode_endianness=True,
                 drop_variables=None):
        self._kwargs = {
            'concat_characters': concat_characters,
            'mask_and_scale': mask_and_scale,
            'decode_times': decode_times,
            'decode_endianness': decode_endianness,
            'drop_variables': drop_variables,
        }

    def __call__(self, variables, attrs):
        return conventions.decode_cf_variables(
            variables, attrs, **self._kwargs)


class CFEncoder(AbstractDecoder):
    def __call__(self, dataset):
        variables, attributes = conventions.encode_dataset_coordinates(dataset)
        variables = OrderedDict((k, conventions.encode_cf_variable(v, name=k))
                                for k, v in variables.items())
        return variables, attributes


# Now, we get to the DataStore API

class AbstractDataStore(Mapping):
    """An abstract interface for implementing datastores.

    Every method is something that should potentially be implemented by
    developers of new datastores.
    """

    def get_variables(self):
        # type: (Any,) -> Mapping[Any, xarray.Variable]
        """Return a map from names to xarray.Variable objects.

        Consider returning Variables whose data is non-eagerly evaluated, e.g.,
        by wrapping with xarray.core.utils.LazilyIndexedArray or
        using dask.array.

        # TODO: move xarray.core.utils.LazilyIndexedArray to public API.
        """
        raise NotImplementedError

    def get_attributes(self):
        # type: () -> Mapping
        """Return a map of global attributes on the DataStore."""
        raise NotImplementedError

    def close(self):
        """Close any resources associated with this DataStore."""
        pass

    def get_read_lock(self, name, region=Ellipsis):
        # type: (Hashable, Union[Ellipsis, Tuple[slice, ...]]) -> object
        """Return a lock for reading a region of a given variable.

        This method may be useful for DataStores that from which data is read in
        parallel (e.g., with dask).

        Parameters
        ----------
        name : Hashable
            Variable name.
        region : Union[Ellipsis, Tuple[slice, ...]], optional
            Region in the variable, e.g., valid key argument to `data[key]`,
            for which to get a lock.

        Returns
        -------
        threading.Lock() ducktype (that is, an object with `acquire` and
        `release` methods), or None, if no lock is necessary.
        """
        # Do we actually want the region argument?
        # For a library such as HDF5, this will simply return a global lock for
        # all files.
        return None

    def get_token(self):
        """Return a token identifier suitable for use by dask."""
        return None

    def get_name(self):
        """Return a user-friendly name for prefixing names of dask arrays.

        Not required to be unique.
        """
        return ''


class OnDiskMixin(object):

    def get_token(self):
        return (self.filename, os.path.getmtime(self.filename))

    def get_name(self):
        return self.filename


class AbstractWritableDataStore(AbstractDataStore):
    """An abstract interface for implementing writable datastores."""

    def create_variable(self, name, variable, check_encoding=False):
        # type: (Hashable, xarray.Variable) -> WritableDuckArray
        """Create a new variable for writing into the DataStore.

        This method is responsible for setting up a variable to write. It
        SHOULD NOT actually write array values, but rather create an array to
        which xarray itself will write.

        If the given variable cannot be stored on the DataStore, this method
        MUST raise an error.

        Parameters
        ----------
        name : Hashable
            Variable name. If a variable with this name already exists in the
            DataStore, this method MAY raise an error.
        variable : xarray.Variable
            Variable to copy into the DataStore. `variable.encodings` provides
            a dictionary of DataStore specific options for how to save
            variables.
        check_encoding : bool, optional
            If True, this method SHOULD raise an error for any unexpected keys
            or invalid values in `variable.encoding`.

        Returns
        -------
        Array-like object that writes data to the store when assigning a NumPy
        array to a tuple of slice objects, e.g., ``x[key] = value``, where
        ``key`` has type ``Tuple[slice, ...]`` and length equal to the
        dimensionality of the array, and ``value`` is a ``numpy.ndarray``.
        """
        raise NotImplementedError

    def get_writable_array(self, name):
        # type: (Hashable,) -> WritableDuckArray
        """Return a writable array corresponding to an existing variable.

        This method is only needed if you want the DataStore to support partial
        writes, e.g., appending to existing variables.

        Parameters
        ----------
        name : Hashable
            Variable name. How to handle non-existing names is up to the
            DataStore class. However, xarray will never call this method unless
            a variable with the given name has already been verified to exist
            as a member of the mapping returned by `get_variables()`.

        Returns
        -------
        Writable array-like object, see `create_variable` for details.
        """
        # Note: this mostly exists for the benefit of future support for partial
        # reads -- we don't actually make use of this in the current version of
        # xarray.
        raise NotImplementedError

    def set_attribute(self, name, value):
        # type: (Hashable, Any) -> None
        """Set a global attribute on the DataStore."""
        raise NotImplementedError

    def sync(self):
        """Synchronize writes to this DataStore."""
        pass

    def get_write_lock(self, name, region=Ellipsis):
        # type: (Hashable, Union[Ellipsis, Tuple[slice, ...]]) -> object
        """Return a lock for writing a given variable.

        This method may be useful for DataStores that from which data is
        written in parallel (e.g., with dask).

        Parameters
        ----------
        name : Hashable
            Variable name.
        region : Union[Ellipsis, Tuple[slice, ...]], optional
            Region in the variable, e.g., valid key argument to `data[key]`,
            for which to get a lock.

        Returns
        -------
        threading.Lock() ducktype (that is, an object with `acquire` and
        `release` methods), or None, if no lock is necessary.
        """
        # Again, we actually have a use for the region argument? Could be useful
        # to ensure writes to zarr are safe.
        return None


class InMemoryDataStore(AbstractWritableDataStore):
    """Stores variables and attributes directly in OrderedDicts.

    This store exists internal testing purposes, e.g., for integration tests
    with dask.array that will not need to write actual data to disk.
    """
    def __init__(self):
        self._variables = OrderedDict()
        self._attributes = OrderedDict()
        # do we need locks? are writes to NumPy arrays thread-safe?
        # this is a dumb but safe approach.
        self._write_locks = defaultdict(threading.Lock)

    def get_variables(self):
        return self._variables

    def get_attributes(self):
        return self._attributes

    def get_read_lock(self, name, region=Ellipsis):
        return None

    def create_variable(self, name, variable, check_encoding=False):
        if check_encoding and variable.encoding:
            raise ValueError('encoding must be empty')
        store_variable = Variable(variable.dims,
                                  np.empty_like(variable),
                                  copy.deepcopy(variable.attrs))
        self._variables[name] = store_variable
        return store_variable.values

    def get_writable_array(self, name):
        return self._variables[name].values

    def set_attribute(self, name, value):
        self._attributes[name] = copy.deepcopy(value)

    def get_write_lock(self, name, region=Ellipsis):
        return self._write_locks[name]


# Reading and writing with DataStores

# This would live in xarray/backends/api.py, and would be user-facing as part
# of the developer API. Note that it is entirely domain agnostic:

def read_datastore(store, decode=None, chunks=None, close_on_error=False):

    # open_dataset now simply opens an appropriate store, creates a default
    # CFDecoder, and passes the arguments off to read_datastore.

    if decode is None:
        decode = coders.DummyDecoder()

    try:
        variables = store.get_variables()
        attrs = store.get_attributes()
        dataset = decode(variables, attrs)

        if chunks is not None:
            try:
                from dask.base import tokenize
            except ImportError:
                import dask  # raise the usual error if dask is entirely missing

            lock = {k: store.get_lock(k) for k in variables}
            token = tokenize(store.get_token(), pickle.dumps(decoder))
            prefix = store.get_name() + '/'
            dataset2 = dataset.chunk(
                chunks, name_prefix=prefix, token=token, lock=lock)
            dataset2._file_obj = store
        else:
            dataset2 = dataset

    except Exception:
        # This exists for the sake of testing (especially on Windows, where
        # unclosed files lead to errors)
        if close_on_error:
            store.close()

    return dataset2


class ArrayWriter(object):
    def __init__(self):
        self.sources = []
        self.targets = []
        self.locks = []

    def add(self, source, target, lock=None):
        if isinstance(source, dask_array_type):
            self.sources.append(source)
            self.targets.append(target)
            self.locks.append(lock)
        else:
            target[...] = source

    def sync(self):
        if self.sources:
            import dask.array as da
            # TODO: dask.array.store needs to be able to accept a list of Lock
            # objects.
            da.store(self.sources, self.targets, lock=self.lock)


# write_datastore also becomes developer facing API. Dataset.dump_to_store
# and the various other machinery is deprecated.

def write_datastore(dataset, store, encode=None, encoding=None,
                    close_on_error=False):
    # TODO: add compute keyword argument to allow for returning futures, like
    # dask.array.store.
    # TODO: add support for writing to regions of variables (using
    # store.get_writable_array)

    # to_netcdf now simply opens an appropriate store, creates a default
    # CFEncoder, and passes the arguments off to write_datastore.

    if encode is None:
        encode = DummyEncoder()

    if encoding is None:
        encoding = {}

    if encoding:
        # shallow copy variables so we can overwrite encoding
        dataset = dataset.copy(deep=False)
        for name, var_encoding in encoding.items():
            dataset.variables[name].encoding = var_encoding

    variables, attrs = encode(dataset)

    try:
        for k, v in attrs.item():
            store.set_attribute(k, v)

        writer = ArrayWriter()
        for name, variable in variables.items():
            check = name in encoding
            target, source = store.create_variable(name, variable, check)
            lock = store.get_write_lock(name)
            writer.add(source, target, lock)
        writer.sync()

    except Exception:
        if close_on_error:
            store.sync()
            store.close()
    else:
        store.sync()
        store.close()
