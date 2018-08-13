from __future__ import absolute_import, division, print_function

from distutils.version import LooseVersion

import numpy as np

from .. import Variable, coding, conventions
from ..core import indexing
from ..core.pycompat import OrderedDict, integer_types, iteritems
from ..core.utils import FrozenOrderedDict, HiddenKeyDict
from .common import AbstractWritableDataStore, ArrayWriter, BackendArray

# need some special secret attributes to tell us the dimensions
_DIMENSION_KEY = '_ARRAY_DIMENSIONS'


# zarr attributes have to be serializable as json
# many xarray datasets / variables have numpy arrays and values
# these functions handle encoding / decoding of such items
def _encode_zarr_attr_value(value):
    if isinstance(value, np.ndarray):
        encoded = value.tolist()
    # this checks if it's a scalar number
    elif isinstance(value, np.generic):
        encoded = value.item()
    else:
        encoded = value
    return encoded


class ZarrArrayWrapper(BackendArray):
    def __init__(self, variable_name, datastore):
        self.datastore = datastore
        self.variable_name = variable_name

        array = self.get_array()
        self.shape = array.shape

        dtype = array.dtype
        self.dtype = dtype

    def get_array(self):
        return self.datastore.ds[self.variable_name]

    def __getitem__(self, key):
        array = self.get_array()
        if isinstance(key, indexing.BasicIndexer):
            return array[key.tuple]
        elif isinstance(key, indexing.VectorizedIndexer):
            return array.vindex[indexing._arrayize_vectorized_indexer(
                key.tuple, self.shape).tuple]
        else:
            assert isinstance(key, indexing.OuterIndexer)
            return array.oindex[key.tuple]
        # if self.ndim == 0:
        # could possibly have a work-around for 0d data here


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
        if any(len(set(chunks[:-1])) > 1 for chunks in var_chunks):
            raise ValueError(
                "Zarr requires uniform chunk sizes excpet for final chunk."
                " Variable %r has incompatible chunks. Consider "
                "rechunking using `chunk()`." % (var_chunks,))
        if any((chunks[0] < chunks[-1]) for chunks in var_chunks):
            raise ValueError(
                "Final chunk of Zarr array must be smaller than first. "
                "Variable %r has incompatible chunks. Consider rechunking "
                "using `chunk()`." % var_chunks)
        # return the first chunk for each dimension
        return tuple(chunk[0] for chunk in var_chunks)

    # from here on, we are dealing with user-specified chunks in encoding
    # zarr allows chunks to be an integer, in which case it uses the same chunk
    # size on each dimension.
    # Here we re-implement this expansion ourselves. That makes the logic of
    # checking chunk compatibility easier

    if isinstance(enc_chunks, integer_types):
        enc_chunks_tuple = ndim * (enc_chunks,)
    else:
        enc_chunks_tuple = tuple(enc_chunks)

    if len(enc_chunks_tuple) != ndim:
        # throw away encoding chunks, start over
        return _determine_zarr_chunks(None, var_chunks, ndim)

    for x in enc_chunks_tuple:
        if not isinstance(x, int):
            raise TypeError("zarr chunks must be an int or a tuple of ints. "
                            "Instead found %r" % (enc_chunks_tuple,))

    # if there are chunks in encoding and the variable data is a numpy array,
    # we use the specified chunks
    if var_chunks is None:
        return enc_chunks_tuple

    # the hard case
    # DESIGN CHOICE: do not allow multiple dask chunks on a single zarr chunk
    # this avoids the need to get involved in zarr synchronization / locking
    # From zarr docs:
    #  "If each worker in a parallel computation is writing to a separate
    #   region of the array, and if region boundaries are perfectly aligned
    #   with chunk boundaries, then no synchronization is required."
    # TODO: incorporate synchronizer to allow writes from multiple dask
    # threads
    if var_chunks and enc_chunks_tuple:
        for zchunk, dchunks in zip(enc_chunks_tuple, var_chunks):
            for dchunk in dchunks:
                if dchunk % zchunk:
                    raise NotImplementedError(
                        "Specified zarr chunks %r would overlap multiple dask "
                        "chunks %r. This is not implemented in xarray yet. "
                        " Consider rechunking the data using "
                        "`chunk()` or specifying different chunks in encoding."
                        % (enc_chunks_tuple, var_chunks))
        return enc_chunks_tuple

    raise AssertionError(
        "We should never get here. Function logic must be wrong.")


def _get_zarr_dims_and_attrs(zarr_obj, dimension_key):
    # Zarr arrays do not have dimenions. To get around this problem, we add
    # an attribute that specifies the dimension. We have to hide this attribute
    # when we send the attributes to the user.
    # zarr_obj can be either a zarr group or zarr array
    try:
        dimensions = zarr_obj.attrs[dimension_key]
    except KeyError:
        raise KeyError("Zarr object is missing the attribute `%s`, which is "
                       "required for xarray to determine variable dimensions."
                       % (dimension_key))
    attributes = HiddenKeyDict(zarr_obj.attrs, [dimension_key])
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
    return encoding


# Function below is copied from conventions.encode_cf_variable.
# The only change is to raise an error for object dtypes.
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

    var = conventions.encode_cf_variable(var, name=name)

    # zarr allows unicode, but not variable-length strings, so it's both
    # simpler and more compact to always encode as UTF-8 explicitly.
    # TODO: allow toggling this explicitly via dtype in encoding.
    coder = coding.strings.EncodedStringCoder(allows_unicode=False)
    var = coder.encode(var, name=name)
    var = coding.strings.ensure_fixed_length_bytes(var)

    return var


class ZarrStore(AbstractWritableDataStore):
    """Store for reading and writing data via zarr
    """

    @classmethod
    def open_group(cls, store, mode='r', synchronizer=None, group=None,
                   writer=None):
        import zarr
        min_zarr = '2.2'

        if LooseVersion(zarr.__version__) < min_zarr:  # pragma: no cover
            raise NotImplementedError("Zarr version %s or greater is "
                                      "required by xarray. See zarr "
                                      "installation "
                                      "http://zarr.readthedocs.io/en/stable/"
                                      "#installation" % min_zarr)
        zarr_group = zarr.open_group(store=store, mode=mode,
                                     synchronizer=synchronizer, path=group)
        return cls(zarr_group, writer=writer)

    def __init__(self, zarr_group, writer=None):
        self.ds = zarr_group
        self._read_only = self.ds.read_only
        self._synchronizer = self.ds.synchronizer
        self._group = self.ds.path

        if writer is None:
            # by default, we should not need a lock for writing zarr because
            # we do not (yet) allow overlapping chunks during write
            zarr_writer = ArrayWriter(lock=False)
        else:
            zarr_writer = writer

        # do we need to define attributes for all of the opener keyword args?
        super(ZarrStore, self).__init__(zarr_writer)

    def open_store_variable(self, name, zarr_array):
        data = indexing.LazilyOuterIndexedArray(ZarrArrayWrapper(name, self))
        dimensions, attributes = _get_zarr_dims_and_attrs(zarr_array,
                                                          _DIMENSION_KEY)
        attributes = OrderedDict(attributes)
        encoding = {'chunks': zarr_array.chunks,
                    'compressor': zarr_array.compressor,
                    'filters': zarr_array.filters}
        # _FillValue needs to be in attributes, not encoding, so it will get
        # picked up by decode_cf
        if getattr(zarr_array, 'fill_value') is not None:
            attributes['_FillValue'] = zarr_array.fill_value

        return Variable(dimensions, data, attributes, encoding)

    def get_variables(self):
        return FrozenOrderedDict((k, self.open_store_variable(k, v))
                                 for k, v in self.ds.arrays())

    def get_attrs(self):
        attributes = OrderedDict(self.ds.attrs.asdict())
        return attributes

    def get_dimensions(self):
        dimensions = OrderedDict()
        for k, v in self.ds.arrays():
            try:
                for d, s in zip(v.attrs[_DIMENSION_KEY], v.shape):
                    if d in dimensions and dimensions[d] != s:
                        raise ValueError(
                            'found conflicting lengths for dimension %s '
                            '(%d != %d)' % (d, s, dimensions[d]))
                    dimensions[d] = s

            except KeyError:
                raise KeyError("Zarr object is missing the attribute `%s`, "
                               "which is required for xarray to determine "
                               "variable dimensions." % (_DIMENSION_KEY))
        return dimensions

    def set_dimensions(self, variables, unlimited_dims=None):
        if unlimited_dims is not None:
            raise NotImplementedError(
                "Zarr backend doesn't know how to handle unlimited dimensions")

    def set_attributes(self, attributes):
        self.ds.attrs.put(attributes)

    def encode_variable(self, variable):
        variable = encode_zarr_variable(variable)
        return variable

    def encode_attribute(self, a):
        return _encode_zarr_attr_value(a)

    def prepare_variable(self, name, variable, check_encoding=False,
                         unlimited_dims=None):

        attrs = variable.attrs.copy()
        dims = variable.dims
        dtype = variable.dtype
        shape = variable.shape

        fill_value = attrs.pop('_FillValue', None)
        if variable.encoding == {'_FillValue': None} and fill_value is None:
            variable.encoding = {}

        encoding = _extract_zarr_variable_encoding(
            variable, raise_on_invalid=check_encoding)

        encoded_attrs = OrderedDict()
        # the magic for storing the hidden dimension data
        encoded_attrs[_DIMENSION_KEY] = dims
        for k, v in iteritems(attrs):
            encoded_attrs[k] = self.encode_attribute(v)

        zarr_array = self.ds.create(name, shape=shape, dtype=dtype,
                                    fill_value=fill_value, **encoding)
        zarr_array.attrs.put(encoded_attrs)

        return zarr_array, variable.data

    def store(self, variables, attributes, *args, **kwargs):
        AbstractWritableDataStore.store(self, variables, attributes,
                                        *args, **kwargs)

    def sync(self, compute=True):
        self.delayed_store = self.writer.sync(compute=compute)


def open_zarr(store, group=None, synchronizer=None, auto_chunk=True,
              decode_cf=True, mask_and_scale=True, decode_times=True,
              concat_characters=True, decode_coords=True,
              drop_variables=None):
    """Load and decode a dataset from a Zarr store.

    .. note:: Experimental
              The Zarr backend is new and experimental. Please report any
              unexpected behavior via github issues.

    The `store` object should be a valid store for a Zarr group. `store`
    variables must contain dimension metadata encoded in the
    `_ARRAY_DIMENSIONS` attribute.

    Parameters
    ----------
    store : MutableMapping or str
        A MutableMapping where a Zarr Group has been stored or a path to a
        directory in file system where a Zarr DirectoryStore has been stored.
    synchronizer : object, optional
        Array synchronizer provided to zarr
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
    concat_characters : bool, optional
        If True, concatenate along the last dimension of character arrays to
        form string arrays. Dimensions will only be concatenated over (and
        removed) if they have no corresponding variable and if they are only
        used as the last dimension of character arrays.
    decode_coords : bool, optional
        If True, decode the 'coordinates' attribute to identify coordinates in
        the resulting dataset.
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

    References
    ----------
    http://zarr.readthedocs.io/
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

        # TODO: this is where we would apply caching

        return ds

    # Zarr supports a wide range of access modes, but for now xarray either
    # reads or writes from a store, never both. For open_zarr, we only read
    mode = 'r'
    zarr_store = ZarrStore.open_group(store, mode=mode,
                                      synchronizer=synchronizer,
                                      group=group)
    ds = maybe_decode_store(zarr_store)

    # auto chunking needs to be here and not in ZarrStore because variable
    # chunks do not survive decode_cf
    if auto_chunk:
        # adapted from Dataset.Chunk()
        def maybe_chunk(name, var):
            from dask.base import tokenize
            chunks = var.encoding.get('chunks')
            if (var.ndim > 0) and (chunks is not None):
                # does this cause any data to be read?
                token2 = tokenize(name, var._data)
                name2 = 'zarr-%s' % token2
                return var.chunk(chunks, name=name2, lock=None)
            else:
                return var

        variables = OrderedDict([(k, maybe_chunk(k, v))
                                 for k, v in ds.variables.items()])
        return ds._replace_vars_and_dims(variables)
    else:
        return ds
