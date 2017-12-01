from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import functools
from itertools import product
# import operator

import numpy as np

from .. import Variable
from ..core import indexing
from ..core.utils import FrozenOrderedDict, HiddenKeyDict
from ..core.pycompat import iteritems, OrderedDict

from .common import (AbstractWritableDataStore, DataStorePickleMixin,
                     BackendArray)

from .. import conventions

# this is a private method but we need it for open_zar
from .api import _protect_dataset_variables_inplace

# need some special secret attributes to tell us the dimensions
_DIMENSION_KEY = '_ARRAY_DIMENSIONS'


# zarr attributes have to be serializable as json
# many xarray datasets / variables have numpy arrays and values
# these functions handle encoding / decoding of such items
def _encode_zarr_attr_value(value):
    # what is the most duck-type friendly way to do this check
    if isinstance(value, np.ndarray):
        return value.tolist()
    # I don't know how to check generically if something is a numpy scalar
    # i.e. np.float32 or np.int8, etc. without checking against each dtype
    # manually. This was the best I could come up with
    elif isinstance(value, np.generic):
        # np.string_('X').item() returns a type `bytes`
        # zarr still doesn't like that
        # causes some fill_value encoding to fail
        return value.item()
    else:
        return value


def _ensure_valid_fill_value(value, dtype):
    if dtype.type == np.string_ and type(value) == bytes:
        valid = value.decode('ascii')
    else:
        valid = value
    return _encode_zarr_attr_value(valid)


# TODO: cleanup/combine these next two functions
def _decode_zarr_attr_value(value):
    # what happens if we just don't decode anything?
    # does it matter that we don't convert back to numpy types?
    return value


def _decode_zarr_attrs(attrs):
    return OrderedDict([(k, _decode_zarr_attr_value(v))
                        for k, v in attrs.items()])


# untested, but I think this does the appropriate shape munging to make slices
# appear as the last axes of the result array
def _replace_slices_with_arrays(key, shape):
    num_slices = sum(1 for k in key if isinstance(k, slice))
    num_arrays = len(shape) - num_slices
    new_key = []
    slice_count = 0
    for k, size in zip(key, shape):
        if isinstance(k, slice):
            array = np.arange(*k.indices(size))
            sl = [np.newaxis] * len(shape)
            sl[num_arrays + slice_count] = np.newaxis
            k = array[sl]
            slice_count += 1
        else:
            assert isinstance(k, np.ndarray)
            k = k[(slice(None),) * num_arrays + (np.newaxis,) * num_slices]
        new_key.append(k)
    return tuple(new_key)


class ZarrArrayWrapper(BackendArray):
    def __init__(self, variable_name, datastore):
        self.datastore = datastore
        self.variable_name = variable_name

        array = self.get_array()
        self.shape = array.shape

        dtype = array.dtype
        if dtype is str:
            # use object dtype because that's the only way in numpy to
            # represent variable length strings; it also prevents automatic
            # string concatenation via conventions.decode_cf_variable
            dtype = np.dtype('O')
        self.dtype = dtype

    def get_array(self):
        return self.datastore.ds[self.variable_name]

    def __getitem__(self, key):
        array = self.get_array()
        if isinstance(key, indexing.BasicIndexer):
            return array[key.tuple]
        elif isinstance(key, indexing.VectorizedIndexer):
            return array.vindex[_replace_slices_with_arrays(key.tuple,
                                                            self.shape)]
        else:
            assert isinstance(key, indexing.OuterIndexer)
            return array.oindex[key.tuple]
        # if self.ndim == 0:
        # could possibly have a work-around for 0d data here


def _open_zarr_group(store=None, overwrite=None, synchronizer=None,
                     group=None, mode=None):
    '''Wrap zarr.open_group'''

    import zarr
    zarr_group = zarr.open_group(store=store, mode=mode,
                                 synchronizer=synchronizer, path=group)
    return zarr_group


def _dask_chunks_to_zarr_chunks(chunks):
    '''this function dask chunks syntax to zarr chunks'''
    if chunks is None:
        return chunks

    all_chunks = product(*chunks)
    first_chunk = next(all_chunks)
    for this_chunk in all_chunks:
        if not (this_chunk == first_chunk):
            raise ValueError("zarr requires uniform chunk sizes, found %r" %
                             chunks)
    return first_chunk


def _determine_zarr_chunks(enc_chunks, var_chunks, ndim):
    """
    Given encoding chunks (possibly None) and variable chunks (possibly None)
    """

    # zarr chunk spec:
    # chunks : int or tuple of ints, optional
    #   Chunk shape. If not provided, will be guessed from shape and dtype.

    # if there are no chunks in encoding and the variable data is a numpy
    # array, then we let zarr use its own heuristics to pick the chunks
    if var_chunks is None and enc_chunks is None:
        return None

    # if there are no chunks in encoding but there are dask chunks, we try to
    # use the same chunks in zarr
    # However, zarr chunks needs to be uniform for each array
    # http://zarr.readthedocs.io/en/latest/spec/v1.html#chunks
    # while dask chunks can be variable sized
    # http://dask.pydata.org/en/latest/array-design.html#chunks
    if var_chunks and enc_chunks is None:
        all_var_chunks = product(*var_chunks)
        first_var_chunk = next(all_var_chunks)
        for this_chunk in all_var_chunks:
            if not (this_chunk == first_var_chunk):
                raise ValueError(
                    "zarr requires uniform chunk sizes, but variable has "
                    "non-uniform chunks %r. Consider rechunking the data "
                    "using `chunk()`." % var_chunks)
        return first_var_chunk

    # from here on, we are dealing with user-specified chunks in encoding
    # zarr allows chunks to be an integer, in which case it uses the same chunk
    # size on each dimension.
    # Here we re-implement this expansion ourselves. That makes the logic of
    # checking chunk compatibility easier

    # this coerces a single int to a tuple but leaves a tuple as is
    enc_chunks_tuple = tuple(enc_chunks)
    if len(enc_chunks_tuple) == 1:
        enc_chunks_tuple = ndim * enc_chunks_tuple

    if not len(enc_chunks_tuple) == ndim:
        raise ValueError("zarr chunks tuple %r must have same length as "
                         "variable.ndim %g" %
                         (enc_chunks_tuple, _DIMENSION_KEY))

    for x in enc_chunks_tuple:
        if not isinstance(x, int):
            raise ValueError("zarr chunks must be an int or a tuple of ints. "
                             "Instead found %r" % (enc_chunks_tuple,))

    # if there are chunks in encoding and the variabile data is a numpy array,
    # we use the specified chunks
    if enc_chunks_tuple and var_chunks is None:
        return enc_chunks_tuple

    # the hard case
    # DESIGN CHOICE: do not allow multiple dask chunks on a single zarr chunk
    # this avoids the need to get involved in zarr synchronization / locking
    # From zarr docs:
    #  "If each worker in a parallel computation is writing to a separate
    #   region of the array, and if region boundaries are perfectly aligned
    #   with chunk boundaries, then no synchronization is required."
    if var_chunks and enc_chunks_tuple:
        for zchunk, dchunks in zip(enc_chunks_tuple, var_chunks):
            for dchunk in dchunks:
                if not dchunk % zchunk == 0:
                    raise ValueError(
                        "Specified zarr chunks %r would overlap multiple dask "
                        "chunks %r. Consider rechunking the data using "
                        "`chunk()` or specifying different chunks in encoding."
                        % (enc_chunks_tuple, var_chunks))
        return enc_chunks_tuple

    raise RuntimeError(
        "We should never get here. Function logic must be wrong.")


def _get_zarr_dims_and_attrs(zarr_obj, dimension_key):
    # Zarr arrays do not have dimenions. To get around this problem, we add
    # an attribute that specifies the dimension. We have to hide this attribute
    # when we send the attributes to the user.
    # zarr_obj can be either a zarr group or zarr array
    dimensions = zarr_obj.attrs.get(dimension_key)
    attributes = HiddenKeyDict(zarr_obj.attrs, dimension_key)
    return dimensions, attributes


def _extract_zarr_variable_encoding(variable, raise_on_invalid=False):
    encoding = variable.encoding.copy()

    valid_encodings = set(['chunks', 'compressor', 'filters',
                           'cache_metadata'])

    if raise_on_invalid:
        invalid = [k for k in encoding if k not in valid_encodings]
        if invalid:
            raise ValueError('unexpected encoding parameters for zarr '
                             'backend:  %r' % invalid)
    else:
        for k in list(encoding):
            if k not in valid_encodings:
                del encoding[k]

    chunks = _determine_zarr_chunks(encoding.get('chunks'), variable.chunks,
                                    variable.ndim)
    encoding['chunks'] = chunks

    # TODO: figure out how to serialize compressor and filters options
    # in zarr these are python objects, not strings

    return encoding

# copied from conventions.encode_cf_variable

def encode_zarr_variable(var, needs_copy=True, name=None):
    """
    Converts an Variable into an Variable which follows some
    of the CF conventions:

        - Nans are masked using _FillValue (or the deprecated missing_value)
        - Rescaling via: scale_factor and add_offset
        - datetimes are converted to the CF 'units since time' format
        - dtype encodings are enforced.

    Parameters
    ----------
    var : xarray.Variable
        A variable holding un-encoded data.

    Returns
    -------
    out : xarray.Variable
        A variable which has been encoded as described above.
    """
    var = conventions.maybe_encode_datetime(var, name=name)
    var = conventions.maybe_encode_timedelta(var, name=name)
    var, needs_copy = conventions.maybe_encode_offset_and_scale(var,
                                                        needs_copy, name=name)
    var, needs_copy = conventions.maybe_encode_fill_value(var, needs_copy,
                                                          name=name)
    var = conventions.maybe_encode_nonstring_dtype(var, name=name)
    var = conventions.maybe_default_fill_value(var)
    var = conventions.maybe_encode_bools(var)
    var = conventions.ensure_dtype_not_object(var, name=name)
    var = conventions.maybe_encode_string_dtype(var, name=name)
    return var


class ZarrStore(AbstractWritableDataStore, DataStorePickleMixin):
    """Store for reading and writing data via zarr
    """

    def __init__(self, store=None, mode='a', synchronizer=None, group=None,
                 writer=None, autoclose=None):
        self._mode = mode
        self._synchronizer = synchronizer
        self._group = group

        # zarr stores don't need to be opened, closed, or synced.
        # So what do we do with all this logical about openers?
        if autoclose:
            raise NotImplementedError('autoclose=True is not implemented '
                                      'for the zarr backend')
        self._autoclose = False
        self._isopen = True

        opener = functools.partial(_open_zarr_group, store=store,
                                   synchronizer=synchronizer, group=group)
        self._opener = opener
        self.ds = self._opener(mode=mode)

        # initialize hidden dimension attribute
        if _DIMENSION_KEY not in self.ds.attrs:
            self.ds.attrs[_DIMENSION_KEY] = {}

        # do we need to define attributes for all of the opener keyword args?
        super(ZarrStore, self).__init__(writer)

    def open_store_variable(self, name, zarr_array):
        # I don't see why it is necessary to wrap self.ds[name]
        # zarr seems to implement the required ndarray interface
        # TODO: possibly wrap zarr array in dask with aligned chunks
        data = indexing.LazilyIndexedArray(ZarrArrayWrapper(name, self))
        dimensions, attributes = _get_zarr_dims_and_attrs(zarr_array,
                                                          _DIMENSION_KEY)
        attributes = _decode_zarr_attrs(attributes)
        encoding = {'chunks': zarr_array.chunks,
                    'compressor': zarr_array.compressor,
                    'filters': zarr_array.filters}
        # _FillValue needs to be in attributes, not encoding, so it will get
        # picked up by decode_cf
        if getattr(zarr_array, 'fill_value') is not None:
            attributes['_FillValue'] = zarr_array.fill_value

        return Variable(dimensions, data, attributes, encoding)

    def get_variables(self):
        with self.ensure_open(autoclose=False):
            return FrozenOrderedDict((k, self.open_store_variable(k, v))
                                     for k, v in self.ds.arrays())

    def get_attrs(self):
        with self.ensure_open(autoclose=True):
            _, attributes = _get_zarr_dims_and_attrs(self.ds, _DIMENSION_KEY)
            return _decode_zarr_attrs(attributes)

    def get_dimensions(self):
        with self.ensure_open(autoclose=True):
            dimensions, _ = _get_zarr_dims_and_attrs(self.ds, _DIMENSION_KEY)
            return dimensions

    def set_dimension(self, name, length, is_unlimited=False):
        if is_unlimited:
            raise NotImplementedError(
                "Zarr backend doesn't know how to handle unlimited dimensions")
        with self.ensure_open(autoclose=False):
            self.ds.attrs[_DIMENSION_KEY][name] = length

    def set_attribute(self, key, value):
        with self.ensure_open(autoclose=False):
            _, attributes = _get_zarr_dims_and_attrs(self.ds, _DIMENSION_KEY)
            attributes[key] = _encode_zarr_attr_value(value)

    def prepare_variable(self, name, variable, check_encoding=False,
                         unlimited_dims=None):

        attrs = variable.attrs.copy()
        dims = variable.dims
        dtype = variable.dtype
        shape = variable.shape

        # TODO: figure out how zarr should deal with unlimited dimensions
        self.set_necessary_dimensions(variable, unlimited_dims=unlimited_dims)

        fill_value = _ensure_valid_fill_value(attrs.pop('_FillValue', None),
                                              dtype)

        # TODO: figure out what encoding is needed for zarr
        encoding = _extract_zarr_variable_encoding(
            variable, raise_on_invalid=check_encoding)

        # arguments for zarr.create:
        # zarr.creation.create(shape, chunks=None, dtype=None,
        # compressor='default', fill_value=0, order='C', store=None,
        # synchronizer=None, overwrite=False, path=None, chunk_store=None,
        # filters=None, cache_metadata=True, **kwargs)
        print('creating', name, shape, dtype)
        zarr_array = self.ds.create(name, shape=shape, dtype=dtype,
                                    fill_value=fill_value, **encoding)
        # decided not to explicity enumerate encoding options because we
        # risk overriding zarr's defaults (e.g. if we specificy
        # cache_metadata=None instead of True). Alternative is to have lots of
        # logic in _extract_zarr_variable encoding to duplicate zarr defaults.
        #                            chunks=encoding.get('chunks'),
        #                            compressor=encoding.get('compressor'),
        #                            filters=encodings.get('filters'),
        #                            cache_metadata=encoding.get('cache_metadata'))

        # the magic for storing the hidden dimension data
        zarr_array.attrs[_DIMENSION_KEY] = dims
        _, attributes = _get_zarr_dims_and_attrs(zarr_array, _DIMENSION_KEY)

        for k, v in iteritems(attrs):
            attributes[k] = _encode_zarr_attr_value(v)

        return zarr_array, variable.data


    def store(self, variables, attributes, *args, **kwargs):
        new_vars = OrderedDict((k, encode_zarr_variable(v, name=k))
                               for k, v in iteritems(variables))
        AbstractWritableDataStore.store(self, new_vars, attributes,
                                        *args, **kwargs)
    # sync() and close() methods should not be needed with zarr


# from zarr docs

# Zarr arrays can be used as either the source or sink for data in parallel
# computations. Both multi-threaded and multi-process parallelism are
# supported. The Python global interpreter lock (GIL) is released for both
# compression and decompression operations, so Zarr will not block other Python
# threads from running.
#
# A Zarr array can be read concurrently by multiple threads or processes. No
# synchronization (i.e., locking) is required for concurrent reads.
#
# A Zarr array can also be written to concurrently by multiple threads or
# processes. Some synchronization may be required, depending on the way the
# data is being written.

# If each worker in a parallel computation is writing to a separate region of
# the array, and if region boundaries are perfectly aligned with chunk
# boundaries, then no synchronization is required. However, if region and chunk
# boundaries are not perfectly aligned, then synchronization is required to
# avoid two workers attempting to modify the same chunk at the same time.


def open_zarr(store, mode='r+', group=None, synchronizer=None, auto_chunk=True,
              decode_cf=True, mask_and_scale=True, decode_times=True,
              autoclose=False, concat_characters=True, decode_coords=True,
              cache=False, drop_variables=None):
    """Load and decode a dataset from a file or file-like object.

    Parameters
    ----------
    store : MutableMapping or str
        Store or path to directory in file system.
    mode : {‘r’, ‘r+’}
        Persistence mode: ‘r’ means read only (must exist); ‘r+’ means
        read/write (must exist)
    synchronizer : object, optional
        Array synchronizer
    group : str, obtional
        Group path. (a.k.a. `path` in zarr terminology.)
    auto_chunk : bool, optional
        Whether to automatically create dask chunks corresponding to each
        variable's zarr chunks. If False, zarr array data will lazily convert
        to numpy arrays upon access.
    decode_cf : bool, optional
        Whether to decode these variables, assuming they were saved according
        to CF conventions.
    mask_and_scale : bool, optional
        If True, replace array values equal to `_FillValue` with NA and scale
        values according to the formula `original_values * scale_factor +
        add_offset`, where `_FillValue`, `scale_factor` and `add_offset` are
        taken from variable attributes (if they exist).  If the `_FillValue` or
        `missing_value` attribute contains multiple values a warning will be
        issued and all array values matching one of the multiple values will
        be replaced by NA.
    decode_times : bool, optional
        If True, decode times encoded in the standard NetCDF datetime format
        into datetime objects. Otherwise, leave them encoded as numbers.
    autoclose : bool, optional
        If True, automatically close files to avoid OS Error of too many files
        being open.  However, this option doesn't work with streams, e.g.,
        BytesIO.
    concat_characters : bool, optional
        If True, concatenate along the last dimension of character arrays to
        form string arrays. Dimensions will only be concatenated over (and
        removed) if they have no corresponding variable and if they are only
        used as the last dimension of character arrays.
    decode_coords : bool, optional
        If True, decode the 'coordinates' attribute to identify coordinates in
        the resulting dataset.
    cache : bool, optional
        If True, cache data loaded from the underlying datastore in memory as
        NumPy arrays when accessed to avoid reading from the underlying data-
        store multiple times. Defaults to True unless you specify the `chunks`
        argument to use dask, in which case it defaults to False. Does not
        change the behavior of coordinates corresponding to dimensions, which
        always load their data from disk into a ``pandas.Index``.
    drop_variables: string or iterable, optional
        A variable or list of variables to exclude from being parsed from the
        dataset. This may be useful to drop variables with problems or
        inconsistent values.

    Returns
    -------
    dataset : Dataset
        The newly created dataset.

    See Also
    --------
    open_dataset
    """
    if not decode_cf:
        mask_and_scale = False
        decode_times = False
        concat_characters = False
        decode_coords = False

    def maybe_decode_store(store, lock=False):
        ds = conventions.decode_cf(
            store, mask_and_scale=mask_and_scale, decode_times=decode_times,
            concat_characters=concat_characters, decode_coords=decode_coords,
            drop_variables=drop_variables)

        # this is how we would apply caching
        # but do we want it for zarr stores?
        _protect_dataset_variables_inplace(ds, cache)

        return ds

    zarr_store = ZarrStore(store=store, mode=mode, synchronizer=synchronizer,
                           group=group)
    ds = maybe_decode_store(zarr_store)

    # auto chunking needs to be here and not in ZarrStore because variable
    # chunks do not survive decode_cf
    if auto_chunk:
        # adapted from Dataset.Chunk()
        def maybe_chunk(name, var):
            from dask.base import tokenize
            chunks = var.encoding.get('chunks')
            print('chunks', chunks)
            if (var.ndim > 0) and (chunks is not None):
                # does this cause any data to be read?
                token2 = tokenize(name, var._data)
                name2 = 'zarr-%s-%s' % (name, token2)
                return var.chunk(chunks, name=name2, lock=None)
            else:
                return var

        variables = OrderedDict([(k, maybe_chunk(k, v))
                                 for k, v in ds.variables.items()])
        return ds._replace_vars_and_dims(variables)
    else:
        return ds
