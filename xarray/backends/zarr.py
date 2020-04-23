import warnings

import numpy as np
from glob import glob

from .. import coding, conventions
from ..core import indexing
from ..core.pycompat import integer_types
from ..core.utils import FrozenDict, HiddenKeyDict, is_remote_uri
from ..core.variable import Variable
from .common import AbstractWritableDataStore, BackendArray, _encode_variable_name
from ..core.combine import (
    _infer_concat_order_from_positions,
    _nested_combine,
    auto_combine,
    combine_by_coords,
)
# need some special secret attributes to tell us the dimensions
DIMENSION_KEY = "_ARRAY_DIMENSIONS"


def encode_zarr_attr_value(value):
    """
    Encode a attribute value as something that can be serialized as json

    Many xarray datasets / variables have numpy arrays and values. This
    function handles encoding / decoding of such items.

    ndarray -> list
    scalar array -> scalar
    other -> other (no change)
    """
    if isinstance(value, np.ndarray):
        encoded = value.tolist()
    # this checks if it's a scalar number
    elif isinstance(value, np.generic):
        encoded = value.item()
    else:
        encoded = value
    return encoded


class ZarrArrayWrapper(BackendArray):
    __slots__ = ("datastore", "dtype", "shape", "variable_name")

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
            return array.vindex[
                indexing._arrayize_vectorized_indexer(key, self.shape).tuple
            ]
        else:
            assert isinstance(key, indexing.OuterIndexer)
            return array.oindex[key.tuple]
        # if self.ndim == 0:
        # could possibly have a work-around for 0d data here


def _determine_zarr_chunks(enc_chunks, var_chunks, ndim, name):
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
                "Zarr requires uniform chunk sizes except for final chunk. "
                f"Variable named {name!r} has incompatible dask chunks: {var_chunks!r}. "
                "Consider rechunking using `chunk()`."
            )
        if any((chunks[0] < chunks[-1]) for chunks in var_chunks):
            raise ValueError(
                "Final chunk of Zarr array must be the same size or smaller "
                f"than the first. Variable named {name!r} has incompatible Dask chunks {var_chunks!r}."
                "Consider either rechunking using `chunk()` or instead deleting "
                "or modifying `encoding['chunks']`."
            )
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
        return _determine_zarr_chunks(None, var_chunks, ndim, name)

    for x in enc_chunks_tuple:
        if not isinstance(x, int):
            raise TypeError(
                "zarr chunk sizes specified in `encoding['chunks']` "
                "must be an int or a tuple of ints. "
                f"Instead found encoding['chunks']={enc_chunks_tuple!r} "
                f"for variable named {name!r}."
            )

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
            for dchunk in dchunks[:-1]:
                if dchunk % zchunk:
                    raise NotImplementedError(
                        f"Specified zarr chunks encoding['chunks']={enc_chunks_tuple!r} for "
                        f"variable named {name!r} would overlap multiple dask chunks {var_chunks!r}. "
                        "This is not implemented in xarray yet. "
                        "Consider either rechunking using `chunk()` or instead deleting "
                        "or modifying `encoding['chunks']`."
                    )
            if dchunks[-1] > zchunk:
                raise ValueError(
                    "Final chunk of Zarr array must be the same size or "
                    "smaller than the first. "
                    f"Specified Zarr chunk encoding['chunks']={enc_chunks_tuple}, "
                    f"for variable named {name!r} "
                    f"but {dchunks} in the variable's Dask chunks {var_chunks} is "
                    "incompatible with this encoding. "
                    "Consider either rechunking using `chunk()` or instead deleting "
                    "or modifying `encoding['chunks']`."
                )
        return enc_chunks_tuple

    raise AssertionError("We should never get here. Function logic must be wrong.")


def _get_zarr_dims_and_attrs(zarr_obj, dimension_key):
    # Zarr arrays do not have dimenions. To get around this problem, we add
    # an attribute that specifies the dimension. We have to hide this attribute
    # when we send the attributes to the user.
    # zarr_obj can be either a zarr group or zarr array
    try:
        dimensions = zarr_obj.attrs[dimension_key]
    except KeyError:
        raise KeyError(
            "Zarr object is missing the attribute `%s`, which is "
            "required for xarray to determine variable dimensions." % (dimension_key)
        )
    attributes = HiddenKeyDict(zarr_obj.attrs, [dimension_key])
    return dimensions, attributes


def extract_zarr_variable_encoding(variable, raise_on_invalid=False, name=None):
    """
    Extract zarr encoding dictionary from xarray Variable

    Parameters
    ----------
    variable : xarray.Variable
    raise_on_invalid : bool, optional

    Returns
    -------
    encoding : dict
        Zarr encoding for `variable`
    """
    encoding = variable.encoding.copy()

    valid_encodings = {"chunks", "compressor", "filters", "cache_metadata"}

    if raise_on_invalid:
        invalid = [k for k in encoding if k not in valid_encodings]
        if invalid:
            raise ValueError(
                "unexpected encoding parameters for zarr " "backend:  %r" % invalid
            )
    else:
        for k in list(encoding):
            if k not in valid_encodings:
                del encoding[k]

    chunks = _determine_zarr_chunks(
        encoding.get("chunks"), variable.chunks, variable.ndim, name
    )
    encoding["chunks"] = chunks
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
    coder = coding.strings.EncodedStringCoder(allows_unicode=True)
    var = coder.encode(var, name=name)
    var = coding.strings.ensure_fixed_length_bytes(var)

    return var


class ZarrStore(AbstractWritableDataStore):
    """Store for reading and writing data via zarr
    """

    __slots__ = (
        "append_dim",
        "ds",
        "_consolidate_on_close",
        "_group",
        "_read_only",
        "_synchronizer",
    )

    @classmethod
    def open_group(
        cls,
        store,
        mode="r",
        synchronizer=None,
        group=None,
        consolidated=False,
        consolidate_on_close=False,
    ):
        import zarr

        open_kwargs = dict(mode=mode, synchronizer=synchronizer, path=group)
        if consolidated:
            # TODO: an option to pass the metadata_key keyword
            zarr_group = zarr.open_consolidated(store, **open_kwargs)
        else:
            zarr_group = zarr.open_group(store, **open_kwargs)
        return cls(zarr_group, consolidate_on_close)

    def __init__(self, zarr_group, consolidate_on_close=False):
        self.ds = zarr_group
        self._read_only = self.ds.read_only
        self._synchronizer = self.ds.synchronizer
        self._group = self.ds.path
        self._consolidate_on_close = consolidate_on_close
        self.append_dim = None

    def open_store_variable(self, name, zarr_array):
        data = indexing.LazilyOuterIndexedArray(ZarrArrayWrapper(name, self))
        dimensions, attributes = _get_zarr_dims_and_attrs(zarr_array, DIMENSION_KEY)
        attributes = dict(attributes)
        encoding = {
            "chunks": zarr_array.chunks,
            "compressor": zarr_array.compressor,
            "filters": zarr_array.filters,
        }
        # _FillValue needs to be in attributes, not encoding, so it will get
        # picked up by decode_cf
        if getattr(zarr_array, "fill_value") is not None:
            attributes["_FillValue"] = zarr_array.fill_value

        return Variable(dimensions, data, attributes, encoding)

    def get_variables(self):
        return FrozenDict(
            (k, self.open_store_variable(k, v)) for k, v in self.ds.arrays()
        )

    def get_attrs(self):
        attributes = dict(self.ds.attrs.asdict())
        return attributes

    def get_dimensions(self):
        dimensions = {}
        for k, v in self.ds.arrays():
            try:
                for d, s in zip(v.attrs[DIMENSION_KEY], v.shape):
                    if d in dimensions and dimensions[d] != s:
                        raise ValueError(
                            "found conflicting lengths for dimension %s "
                            "(%d != %d)" % (d, s, dimensions[d])
                        )
                    dimensions[d] = s

            except KeyError:
                raise KeyError(
                    "Zarr object is missing the attribute `%s`, "
                    "which is required for xarray to determine "
                    "variable dimensions." % (DIMENSION_KEY)
                )
        return dimensions

    def set_dimensions(self, variables, unlimited_dims=None):
        if unlimited_dims is not None:
            raise NotImplementedError(
                "Zarr backend doesn't know how to handle unlimited dimensions"
            )

    def set_attributes(self, attributes):
        self.ds.attrs.put(attributes)

    def encode_variable(self, variable):
        variable = encode_zarr_variable(variable)
        return variable

    def encode_attribute(self, a):
        return encode_zarr_attr_value(a)

    def store(
        self,
        variables,
        attributes,
        check_encoding_set=frozenset(),
        writer=None,
        unlimited_dims=None,
    ):
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
        writer : ArrayWriter
        unlimited_dims : list-like
            List of dimension names that should be treated as unlimited
            dimensions.
            dimension on which the zarray will be appended
            only needed in append mode
        """

        existing_variables = {
            vn for vn in variables if _encode_variable_name(vn) in self.ds
        }
        new_variables = set(variables) - existing_variables
        variables_without_encoding = {vn: variables[vn] for vn in new_variables}
        variables_encoded, attributes = self.encode(
            variables_without_encoding, attributes
        )

        if len(existing_variables) > 0:
            # there are variables to append
            # their encoding must be the same as in the store
            ds = open_zarr(self.ds.store, group=self.ds.path, chunks=None)
            variables_with_encoding = {}
            for vn in existing_variables:
                variables_with_encoding[vn] = variables[vn].copy(deep=False)
                variables_with_encoding[vn].encoding = ds[vn].encoding
            variables_with_encoding, _ = self.encode(variables_with_encoding, {})
            variables_encoded.update(variables_with_encoding)

        self.set_attributes(attributes)
        self.set_dimensions(variables_encoded, unlimited_dims=unlimited_dims)
        self.set_variables(
            variables_encoded, check_encoding_set, writer, unlimited_dims=unlimited_dims
        )

    def sync(self):
        pass

    def set_variables(self, variables, check_encoding_set, writer, unlimited_dims=None):
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
        writer :
        unlimited_dims : list-like
            List of dimension names that should be treated as unlimited
            dimensions.
        """

        for vn, v in variables.items():
            name = _encode_variable_name(vn)
            check = vn in check_encoding_set
            attrs = v.attrs.copy()
            dims = v.dims
            dtype = v.dtype
            shape = v.shape

            fill_value = attrs.pop("_FillValue", None)
            if v.encoding == {"_FillValue": None} and fill_value is None:
                v.encoding = {}
            if name in self.ds:
                zarr_array = self.ds[name]
                if self.append_dim in dims:
                    # this is the DataArray that has append_dim as a
                    # dimension
                    append_axis = dims.index(self.append_dim)
                    new_shape = list(zarr_array.shape)
                    new_shape[append_axis] += v.shape[append_axis]
                    new_region = [slice(None)] * len(new_shape)
                    new_region[append_axis] = slice(zarr_array.shape[append_axis], None)
                    zarr_array.resize(new_shape)
                    writer.add(v.data, zarr_array, region=tuple(new_region))
            else:
                # new variable
                encoding = extract_zarr_variable_encoding(
                    v, raise_on_invalid=check, name=vn
                )
                encoded_attrs = {}
                # the magic for storing the hidden dimension data
                encoded_attrs[DIMENSION_KEY] = dims
                for k2, v2 in attrs.items():
                    encoded_attrs[k2] = self.encode_attribute(v2)

                if coding.strings.check_vlen_dtype(dtype) == str:
                    dtype = str
                zarr_array = self.ds.create(
                    name, shape=shape, dtype=dtype, fill_value=fill_value, **encoding
                )
                zarr_array.attrs.put(encoded_attrs)
                writer.add(v.data, zarr_array)

    def close(self):
        if self._consolidate_on_close:
            import zarr

            zarr.consolidate_metadata(self.ds.store)


def open_zarr(
    store,
    group=None,
    synchronizer=None,
    chunks="auto",
    decode_cf=True,
    mask_and_scale=True,
    decode_times=True,
    concat_characters=True,
    decode_coords=True,
    drop_variables=None,
    consolidated=False,
    overwrite_encoded_chunks=False,
    **kwargs,
):
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
    group : str, optional
        Group path. (a.k.a. `path` in zarr terminology.)
    chunks : int or dict or tuple or {None, 'auto'}, optional
        Chunk sizes along each dimension, e.g., ``5`` or
        ``{'x': 5, 'y': 5}``. If `chunks='auto'`, dask chunks are created
        based on the variable's zarr chunks. If `chunks=None`, zarr array
        data will lazily convert to numpy arrays upon access. This accepts
        all the chunk specifications as Dask does.
    overwrite_encoded_chunks: bool, optional
        Whether to drop the zarr chunks encoded for each variable when a
        dataset is loaded with specified chunk sizes (default: False)
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
    drop_variables : string or iterable, optional
        A variable or list of variables to exclude from being parsed from the
        dataset. This may be useful to drop variables with problems or
        inconsistent values.
    consolidated : bool, optional
        Whether to open the store using zarr's consolidated metadata
        capability. Only works for stores that have already been consolidated.

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
    if "auto_chunk" in kwargs:
        auto_chunk = kwargs.pop("auto_chunk")
        if auto_chunk:
            chunks = "auto"  # maintain backwards compatibility
        else:
            chunks = None

        warnings.warn(
            "auto_chunk is deprecated. Use chunks='auto' instead.",
            FutureWarning,
            stacklevel=2,
        )

    if kwargs:
        raise TypeError(
            "open_zarr() got unexpected keyword arguments " + ",".join(kwargs.keys())
        )

    if not isinstance(chunks, (int, dict)):
        if chunks != "auto" and chunks is not None:
            raise ValueError(
                "chunks must be an int, dict, 'auto', or None. "
                "Instead found %s. " % chunks
            )

    if chunks == "auto":
        try:
            import dask.array  # noqa
        except ImportError:
            chunks = None

    if not decode_cf:
        mask_and_scale = False
        decode_times = False
        concat_characters = False
        decode_coords = False

    def maybe_decode_store(store, lock=False):
        ds = conventions.decode_cf(
            store,
            mask_and_scale=mask_and_scale,
            decode_times=decode_times,
            concat_characters=concat_characters,
            decode_coords=decode_coords,
            drop_variables=drop_variables,
        )

        # TODO: this is where we would apply caching

        return ds

    # Zarr supports a wide range of access modes, but for now xarray either
    # reads or writes from a store, never both. For open_zarr, we only read
    mode = "r"
    zarr_store = ZarrStore.open_group(
        store,
        mode=mode,
        synchronizer=synchronizer,
        group=group,
        consolidated=consolidated,
    )
    ds = maybe_decode_store(zarr_store)

    # auto chunking needs to be here and not in ZarrStore because variable
    # chunks do not survive decode_cf
    # return trivial case
    if not chunks:
        return ds

    # adapted from Dataset.Chunk()
    if isinstance(chunks, int):
        chunks = dict.fromkeys(ds.dims, chunks)

    if isinstance(chunks, tuple) and len(chunks) == len(ds.dims):
        chunks = dict(zip(ds.dims, chunks))

    def get_chunk(name, var, chunks):
        chunk_spec = dict(zip(var.dims, var.encoding.get("chunks")))

        # Coordinate labels aren't chunked
        if var.ndim == 1 and var.dims[0] == name:
            return chunk_spec

        if chunks == "auto":
            return chunk_spec

        for dim in var.dims:
            if dim in chunks:
                spec = chunks[dim]
                if isinstance(spec, int):
                    spec = (spec,)
                if isinstance(spec, (tuple, list)) and chunk_spec[dim]:
                    if any(s % chunk_spec[dim] for s in spec):
                        warnings.warn(
                            "Specified Dask chunks %r would "
                            "separate Zarr chunk shape %r for "
                            "dimension %r. This significantly "
                            "degrades performance. Consider "
                            "rechunking after loading instead."
                            % (chunks[dim], chunk_spec[dim], dim),
                            stacklevel=2,
                        )
                chunk_spec[dim] = chunks[dim]
        return chunk_spec

    def maybe_chunk(name, var, chunks):
        from dask.base import tokenize

        chunk_spec = get_chunk(name, var, chunks)

        if (var.ndim > 0) and (chunk_spec is not None):
            # does this cause any data to be read?
            token2 = tokenize(name, var._data)
            name2 = "zarr-%s" % token2
            var = var.chunk(chunk_spec, name=name2, lock=None)
            if overwrite_encoded_chunks and var.chunks is not None:
                var.encoding["chunks"] = tuple(x[0] for x in var.chunks)
            return var
        else:
            return var

    variables = {k: maybe_chunk(k, v, chunks) for k, v in ds.variables.items()}
    return ds._replace_vars_and_dims(variables)


def open_mzarr(
    paths,
    chunks=None,
    concat_dim="_not_supplied",
    compat='no_conflicts',
    preprocesses=None,
    lock=None,
    data_vars="all",
    coords='different',
    combine="_old_auto",
    autoclose=None,
    parallel=False,
    join="outer",
    attrs_file=None,
    **kwargs,
):
    """Open multiple zarr files as a single dataset.


    If combine="by_coords" then the function ``combine_by_coords`` is used to
    combine the datasets into one before returning the result, and if
    combined="nested" then ``combine_nested`` is used. The filepaths must be
    structured according to which combining function is used, the details of
    which are given in the documentation ``combine_by_coords`` and
    ``combine_nested``. Requires dask to be installed. Global attributes from
    the ``attrs_file`` are used for the combined dataset.

    Parameters
    ----------
    paths : str of sequence
        Either a string glob in the form ``"path/to/my/files/*.zarr"``,
        ``"path/to/my/files/*"`` (asumming the only directory is a zarr
        store), or a explicit list of files to open. Paths can be given as
        strings or as pathlib Paths.
    chunks : int or dict, optional
        Dictionary with keys given by dimension names and values given by
        chunk sizes. In general, these should divide the dimensions of each
        dataset. If int, chunk each dimension by ``chunks``. By default,
        chunks will be chosen to load entire input files into memory at once.
        This has major impact on performance: please see the full
        documentation for more details [2]_.
    concat_dim : str, or list of str, DataArray, Index or None, optional
        Dimensions to concatenate files along. You only need to provide this
        argument if any of the dimensions along which you want to concatenate
        is not a dimension of the original dataset, e.g. you want to stack a
        collection of 2D arrays along a third dimension. Set
        ``concat_dim=[..., None, ...]`` explicitly to disable concatenation
        along a particular dimension.
    combine : {'by_coords', 'nested'}, optional
        Whether ``xarray.combine_by_coords`` or ``xarray.combine_nested`` is
        used to combine all the data. If this argument is not provided,
        ``xarray.combine_by_coords`` is set by default.
    compat : {'identical', 'equals', 'broadcast_equals',
              'no_conflicts','override'}, optional
        String indicating how to compare variables of the same name for
        potential conflicts when merging:

        * 'broadcast_equals': all values must be equal when variables are
        broadcast against each other to ensure common dimensions.
        * 'equals': all values and dimensions must be the same.
        * 'identical': all values, dimensions and attributes must be the same.
        * 'no_conflicts': only values which are not null in both datasets
        must be equal. The returned dataset then contains the combination of
        all non-null values.
        * 'override': skip comparing and pick variable from first dataset.

    preprocess : callable, optional
        If provided, call this function on each dataset prior to concatenation.
        You can find the file-name from which each dataset was loaded in
        ``ds.encoding[source]``.
    lock: False or duck threading.Lock, optional
        Resource lock to use when reading data from disk. Only relevant when
        using dask or another form of parallelism. By default, appropriate
        locks are chosen to safely read and write files with the currently
        active dask scheduler.
    data_vars : {'minimal', 'different', 'all' or list of str}, optional
        These data variables will be concatenated together:
          * 'minimal': Only data variables in which the dimension already
            appears are included.
          * 'different': Data variables which are not equal (ignoring
            attributes) across all datasets are also concatenated (as well as
            all for which dimension already appears). Beware: this option may
            load the data payload of data variables into memory if they are not
            already loaded.
          * 'all': All data variables will be concatenated.
          * list of str: The listed data variables will be concatenated, in
            addition to the 'minimal' data variables.
     coords : {'minimal', 'different', 'all' or list of str}, optional
        These coordinate variables will be concatenated together:
          * 'minimal': Only coordinates in which the dimension already appears
            are included.
          * 'different': Coordinates which are not equal (ignoring attributes)
            across all datasets are also concatenated (as well as all for which
            dimension already appears). Beware: this option may load the data
            payload of coordinate variables into memory if they are not already
            loaded.
          * 'all': All coordinate variables will be concatenated, except
            those corresponding to other dimensions.
          * list of str: The listed coordinate variables will be concatenated,
            in addition the 'minimal' coordinates.
        parallel : bool, optional
            If True, the open and preprocess steps of this function will be
            performed in parallel using ``dask.delayed``. Default is False.
    join : {'outer', 'inner', 'left', 'right', 'exact, 'override'}, optional
        String indicating how to combine differing indexes
        (excluding concat_dim) in objects
        - 'outer': use the union of object indexes
        - 'inner': use the intersection of object indexes
        - 'left': use indexes from the first object with each dimension
        - 'right': use indexes from the last object with each dimension
        - 'exact': instead of aligning, raise `ValueError` when indexes to be
          aligned are not equal
        - 'override': if indexes are of same size, rewrite indexes to be
          those of the first object with that dimension. Indexes for the same
          dimension must have the same size in all objects.
    attrs_file : str or pathlib.Path, optional
        Path of the file used to read global attributes from.
        By default global attributes are read from the first file provided,
        with wildcard matches sorted by filename.
    **kwargs : optional
        Additional arguments passed on to :py:func:`xarray.open_zarr`.


    Returns
    -------
    xarray.Dataset
    Notes
    -----
    ``open_mfdataset`` opens files with read-only access. When you modify
    values
    of a Dataset, even one linked to files on disk, only the in-memory copy you
    are manipulating in xarray is modified: the original file on disk is never
    touched.
    See Also
    --------
    combine_by_coords
    combine_nested
    auto_combine
    open_dataset
    References
    ----------
    .. [1] http://xarray.pydata.org/en/stable/dask.html
    .. [2] http://xarray.pydata.org/en/stable/dask.html#chunking-and-performance
    """
    if isinstance(paths, str):
        if is_remote_uri(paths):
            raise ValueError(
                "cannot do wild-card matching for paths that are remote URLs: "
                "{!r}. Instead, supply paths as an explicit list of strings.".format(
                    paths
                )
            )
        paths = sorted(glob(paths))
    else:
        paths = [str(p) if isinstance(p, Path) else p for p in paths]
    if not paths:
        raise OSError("no files to open")

    # If combine='by_coords' then this is unnecessary, but quick.
    # If combine='nested' then this creates a flat list which is easier to
    # iterate over, while saving the originally-supplied structure as "ids"
    if combine == "nested":
        if str(concat_dim) == "_not_supplied":
            raise ValueError("Must supply concat_dim when using "
                             "combine='nested'")
        else:
            if isinstance(concat_dim, (str, DataArray)) or concat_dim is None:
                concat_dim = [concat_dim]
    combined_ids_paths = _infer_concat_order_from_positions(paths)
    ids, paths = (list(combined_ids_paths.keys()),
                  list(combined_ids_paths.values()))

    # provide with open_kwargs here specific to zarr stores
    open_kwargs = dict(
        chunks=chunks or {}, lock=lock, autoclose=autoclose, **kwargs
    )

    if parallel:
        import dask

        # wrap the open_dataset, getattr, and preprocess with delayed
        open_ = dask.delayed(open_zarr)
        getattr_ = dask.delayed(getattr)
        if preprocess is not None:
            preprocess = dask.delayed(preprocess)
    else:
        open_ = open_zarr
        getattr_ = getattr

    datasets = [open_(p, **open_kwargs) for p in paths]
    file_objs = [getattr_(ds, "_file_obj") for ds in datasets]
    if preprocess is not None:
        datasets = [preprocess(ds) for ds in datasets]

    if parallel:
        # calling compute here will return the datasets/file_objs lists,
        # the underlying datasets will still be stored as dask arrays
        datasets, file_objs = dask.compute(datasets, file_objs)

    # Combine all datasets, closing them in case of a ValueError
    try:
        if combine == "_old_auto":
            # Use the old auto_combine for now
            # Remove this after deprecation cycle from #2616 is complete
            basic_msg = dedent(
                """\
            In xarray version 0.15 the default behaviour of `open_mfdataset`
            will change. To retain the existing behavior, pass
            combine='nested'. To use future default behavior, pass
            combine='by_coords'. See
            http://xarray.pydata.org/en/stable/combining.html#combining-multi
            """
            )
            warnings.warn(basic_msg, FutureWarning, stacklevel=2)

            combined = auto_combine(
                datasets,
                concat_dim=concat_dim,
                compat=compat,
                data_vars=data_vars,
                coords=coords,
                join=join,
                from_openmfds=True,
            )
        elif combine == "nested":
            # Combined nested list by successive concat and merge operations
            # along each dimension, using structure given by "ids"
            combined = _nested_combine(
                datasets,
                concat_dims=concat_dim,
                compat=compat,
                data_vars=data_vars,
                coords=coords,
                ids=ids,
                join=join,
            )
        elif combine == "by_coords":
            # Redo ordering from coordinates, ignoring how they were ordered
            # previously
            combined = combine_by_coords(datasets,
                                         compat=compat, data_vars=data_vars,
                                         coords=coords, join=join)
        else:
            raise ValueError(
                "{} is an invalid option for the keyword argument"
                " ``combine``".format(combine)
            )
    except ValueError:
        for ds in datasets:
            ds.close()
        raise

    combined._file_obj = _MultiFileCloser(file_objs)

    # read global attributes from the attrs_file or from the first dataset
    if attrs_file is not None:
        if isinstance(attrs_file, Path):
            attrs_file = str(attrs_file)
        combined.attrs = datasets[paths.index(attrs_file)].attrs
    else:
        combined.attrs = datasets[0].attrs

    return combined
