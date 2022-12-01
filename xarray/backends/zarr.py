from __future__ import annotations

import json
import os
import warnings

import numpy as np

from .. import coding, conventions
from ..core import indexing
from ..core.pycompat import integer_types
from ..core.utils import FrozenDict, HiddenKeyDict, close_on_error, module_available
from ..core.variable import Variable
from .common import (
    BACKEND_ENTRYPOINTS,
    AbstractWritableDataStore,
    BackendArray,
    BackendEntrypoint,
    _encode_variable_name,
    _normalize_path,
)
from .store import StoreBackendEntrypoint

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
        return self.datastore.zarr_group[self.variable_name]

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


def _determine_zarr_chunks(enc_chunks, var_chunks, ndim, name, safe_chunks):
    """
    Given encoding chunks (possibly None or []) and variable chunks
    (possibly None or []).
    """

    # zarr chunk spec:
    # chunks : int or tuple of ints, optional
    #   Chunk shape. If not provided, will be guessed from shape and dtype.

    # if there are no chunks in encoding and the variable data is a numpy
    # array, then we let zarr use its own heuristics to pick the chunks
    if not var_chunks and not enc_chunks:
        return None

    # if there are no chunks in encoding but there are dask chunks, we try to
    # use the same chunks in zarr
    # However, zarr chunks needs to be uniform for each array
    # http://zarr.readthedocs.io/en/latest/spec/v1.html#chunks
    # while dask chunks can be variable sized
    # http://dask.pydata.org/en/latest/array-design.html#chunks
    if var_chunks and not enc_chunks:
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
        return _determine_zarr_chunks(None, var_chunks, ndim, name, safe_chunks)

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
    if not var_chunks:
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
                    base_error = (
                        f"Specified zarr chunks encoding['chunks']={enc_chunks_tuple!r} for "
                        f"variable named {name!r} would overlap multiple dask chunks {var_chunks!r}. "
                        f"Writing this array in parallel with dask could lead to corrupted data."
                    )
                    if safe_chunks:
                        raise NotImplementedError(
                            base_error
                            + " Consider either rechunking using `chunk()`, deleting "
                            "or modifying `encoding['chunks']`, or specify `safe_chunks=False`."
                        )
        return enc_chunks_tuple

    raise AssertionError("We should never get here. Function logic must be wrong.")


def _get_zarr_dims_and_attrs(zarr_obj, dimension_key, try_nczarr):
    # Zarr arrays do not have dimensions. To get around this problem, we add
    # an attribute that specifies the dimension. We have to hide this attribute
    # when we send the attributes to the user.
    # zarr_obj can be either a zarr group or zarr array
    try:
        # Xarray-Zarr
        dimensions = zarr_obj.attrs[dimension_key]
    except KeyError as e:
        if not try_nczarr:
            raise KeyError(
                f"Zarr object is missing the attribute `{dimension_key}`, which is "
                "required for xarray to determine variable dimensions."
            ) from e

        # NCZarr defines dimensions through metadata in .zarray
        zarray_path = os.path.join(zarr_obj.path, ".zarray")
        zarray = json.loads(zarr_obj.store[zarray_path])
        try:
            # NCZarr uses Fully Qualified Names
            dimensions = [
                os.path.basename(dim) for dim in zarray["_NCZARR_ARRAY"]["dimrefs"]
            ]
        except KeyError as e:
            raise KeyError(
                f"Zarr object is missing the attribute `{dimension_key}` and the NCZarr metadata, "
                "which are required for xarray to determine variable dimensions."
            ) from e

    nc_attrs = [attr for attr in zarr_obj.attrs if attr.startswith("_NC")]
    attributes = HiddenKeyDict(zarr_obj.attrs, [dimension_key] + nc_attrs)
    return dimensions, attributes


def extract_zarr_variable_encoding(
    variable, raise_on_invalid=False, name=None, safe_chunks=True
):
    """
    Extract zarr encoding dictionary from xarray Variable

    Parameters
    ----------
    variable : Variable
    raise_on_invalid : bool, optional

    Returns
    -------
    encoding : dict
        Zarr encoding for `variable`
    """
    encoding = variable.encoding.copy()

    valid_encodings = {
        "chunks",
        "compressor",
        "filters",
        "cache_metadata",
        "write_empty_chunks",
    }

    if raise_on_invalid:
        invalid = [k for k in encoding if k not in valid_encodings]
        if invalid:
            raise ValueError(
                f"unexpected encoding parameters for zarr backend:  {invalid!r}"
            )
    else:
        for k in list(encoding):
            if k not in valid_encodings:
                del encoding[k]

    chunks = _determine_zarr_chunks(
        encoding.get("chunks"), variable.chunks, variable.ndim, name, safe_chunks
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
    var : Variable
        A variable holding un-encoded data.

    Returns
    -------
    out : Variable
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


def _validate_existing_dims(var_name, new_var, existing_var, region, append_dim):
    if new_var.dims != existing_var.dims:
        raise ValueError(
            f"variable {var_name!r} already exists with different "
            f"dimension names {existing_var.dims} != "
            f"{new_var.dims}, but changing variable "
            f"dimensions is not supported by to_zarr()."
        )

    existing_sizes = {}
    for dim, size in existing_var.sizes.items():
        if region is not None and dim in region:
            start, stop, stride = region[dim].indices(size)
            assert stride == 1  # region was already validated
            size = stop - start
        if dim != append_dim:
            existing_sizes[dim] = size

    new_sizes = {dim: size for dim, size in new_var.sizes.items() if dim != append_dim}
    if existing_sizes != new_sizes:
        raise ValueError(
            f"variable {var_name!r} already exists with different "
            f"dimension sizes: {existing_sizes} != {new_sizes}. "
            f"to_zarr() only supports changing dimension sizes when "
            f"explicitly appending, but append_dim={append_dim!r}."
        )


def _put_attrs(zarr_obj, attrs):
    """Raise a more informative error message for invalid attrs."""
    try:
        zarr_obj.attrs.put(attrs)
    except TypeError as e:
        raise TypeError("Invalid attribute in Dataset.attrs.") from e
    return zarr_obj


class ZarrStore(AbstractWritableDataStore):
    """Store for reading and writing data via zarr"""

    __slots__ = (
        "zarr_group",
        "_append_dim",
        "_consolidate_on_close",
        "_group",
        "_mode",
        "_read_only",
        "_synchronizer",
        "_write_region",
        "_safe_chunks",
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
        chunk_store=None,
        storage_options=None,
        append_dim=None,
        write_region=None,
        safe_chunks=True,
        stacklevel=2,
        zarr_version=None,
    ):
        import zarr

        # zarr doesn't support pathlib.Path objects yet. zarr-python#601
        if isinstance(store, os.PathLike):
            store = os.fspath(store)

        if zarr_version is None:
            # default to 2 if store doesn't specify it's version (e.g. a path)
            zarr_version = getattr(store, "_store_version", 2)

        open_kwargs = dict(
            mode=mode,
            synchronizer=synchronizer,
            path=group,
        )
        open_kwargs["storage_options"] = storage_options
        if zarr_version > 2:
            open_kwargs["zarr_version"] = zarr_version

            if consolidated or consolidate_on_close:
                raise ValueError(
                    "consolidated metadata has not been implemented for zarr "
                    f"version {zarr_version} yet. Set consolidated=False for "
                    f"zarr version {zarr_version}. See also "
                    "https://github.com/zarr-developers/zarr-specs/issues/136"
                )

            if consolidated is None:
                consolidated = False

        if chunk_store:
            open_kwargs["chunk_store"] = chunk_store
            if consolidated is None:
                consolidated = False

        if consolidated is None:
            try:
                zarr_group = zarr.open_consolidated(store, **open_kwargs)
            except KeyError:
                try:
                    zarr_group = zarr.open_group(store, **open_kwargs)
                    warnings.warn(
                        "Failed to open Zarr store with consolidated metadata, "
                        "but successfully read with non-consolidated metadata. "
                        "This is typically much slower for opening a dataset. "
                        "To silence this warning, consider:\n"
                        "1. Consolidating metadata in this existing store with "
                        "zarr.consolidate_metadata().\n"
                        "2. Explicitly setting consolidated=False, to avoid trying "
                        "to read consolidate metadata, or\n"
                        "3. Explicitly setting consolidated=True, to raise an "
                        "error in this case instead of falling back to try "
                        "reading non-consolidated metadata.",
                        RuntimeWarning,
                        stacklevel=stacklevel,
                    )
                except zarr.errors.GroupNotFoundError:
                    raise FileNotFoundError(f"No such file or directory: '{store}'")
        elif consolidated:
            # TODO: an option to pass the metadata_key keyword
            zarr_group = zarr.open_consolidated(store, **open_kwargs)
        else:
            zarr_group = zarr.open_group(store, **open_kwargs)
        return cls(
            zarr_group,
            mode,
            consolidate_on_close,
            append_dim,
            write_region,
            safe_chunks,
        )

    def __init__(
        self,
        zarr_group,
        mode=None,
        consolidate_on_close=False,
        append_dim=None,
        write_region=None,
        safe_chunks=True,
    ):
        self.zarr_group = zarr_group
        self._read_only = self.zarr_group.read_only
        self._synchronizer = self.zarr_group.synchronizer
        self._group = self.zarr_group.path
        self._mode = mode
        self._consolidate_on_close = consolidate_on_close
        self._append_dim = append_dim
        self._write_region = write_region
        self._safe_chunks = safe_chunks

    @property
    def ds(self):
        # TODO: consider deprecating this in favor of zarr_group
        return self.zarr_group

    def open_store_variable(self, name, zarr_array):
        data = indexing.LazilyIndexedArray(ZarrArrayWrapper(name, self))
        try_nczarr = self._mode == "r"
        dimensions, attributes = _get_zarr_dims_and_attrs(
            zarr_array, DIMENSION_KEY, try_nczarr
        )
        attributes = dict(attributes)

        # TODO: this should not be needed once
        # https://github.com/zarr-developers/zarr-python/issues/1269 is resolved.
        attributes.pop("filters", None)

        encoding = {
            "chunks": zarr_array.chunks,
            "preferred_chunks": dict(zip(dimensions, zarr_array.chunks)),
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
            (k, self.open_store_variable(k, v)) for k, v in self.zarr_group.arrays()
        )

    def get_attrs(self):
        return {
            k: v
            for k, v in self.zarr_group.attrs.asdict().items()
            if not k.startswith("_NC")
        }

    def get_dimensions(self):
        try_nczarr = self._mode == "r"
        dimensions = {}
        for k, v in self.zarr_group.arrays():
            dim_names, _ = _get_zarr_dims_and_attrs(v, DIMENSION_KEY, try_nczarr)
            for d, s in zip(dim_names, v.shape):
                if d in dimensions and dimensions[d] != s:
                    raise ValueError(
                        f"found conflicting lengths for dimension {d} "
                        f"({s} != {dimensions[d]})"
                    )
                dimensions[d] = s
        return dimensions

    def set_dimensions(self, variables, unlimited_dims=None):
        if unlimited_dims is not None:
            raise NotImplementedError(
                "Zarr backend doesn't know how to handle unlimited dimensions"
            )

    def set_attributes(self, attributes):
        _put_attrs(self.zarr_group, attributes)

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
        import zarr

        existing_variable_names = {
            vn for vn in variables if _encode_variable_name(vn) in self.zarr_group
        }
        new_variables = set(variables) - existing_variable_names
        variables_without_encoding = {vn: variables[vn] for vn in new_variables}
        variables_encoded, attributes = self.encode(
            variables_without_encoding, attributes
        )

        if existing_variable_names:
            # Decode variables directly, without going via xarray.Dataset to
            # avoid needing to load index variables into memory.
            # TODO: consider making loading indexes lazy again?
            existing_vars, _, _ = conventions.decode_cf_variables(
                self.get_variables(), self.get_attrs()
            )
            # Modified variables must use the same encoding as the store.
            vars_with_encoding = {}
            for vn in existing_variable_names:
                vars_with_encoding[vn] = variables[vn].copy(deep=False)
                vars_with_encoding[vn].encoding = existing_vars[vn].encoding
            vars_with_encoding, _ = self.encode(vars_with_encoding, {})
            variables_encoded.update(vars_with_encoding)

            for var_name in existing_variable_names:
                new_var = variables_encoded[var_name]
                existing_var = existing_vars[var_name]
                _validate_existing_dims(
                    var_name,
                    new_var,
                    existing_var,
                    self._write_region,
                    self._append_dim,
                )

        if self._mode not in ["r", "r+"]:
            self.set_attributes(attributes)
            self.set_dimensions(variables_encoded, unlimited_dims=unlimited_dims)

        self.set_variables(
            variables_encoded, check_encoding_set, writer, unlimited_dims=unlimited_dims
        )
        if self._consolidate_on_close:
            zarr.consolidate_metadata(self.zarr_group.store)

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
        writer
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

            if name in self.zarr_group:
                # existing variable
                # TODO: if mode="a", consider overriding the existing variable
                # metadata. This would need some case work properly with region
                # and append_dim.
                zarr_array = self.zarr_group[name]
            else:
                # new variable
                encoding = extract_zarr_variable_encoding(
                    v, raise_on_invalid=check, name=vn, safe_chunks=self._safe_chunks
                )
                encoded_attrs = {}
                # the magic for storing the hidden dimension data
                encoded_attrs[DIMENSION_KEY] = dims
                for k2, v2 in attrs.items():
                    encoded_attrs[k2] = self.encode_attribute(v2)

                if coding.strings.check_vlen_dtype(dtype) == str:
                    dtype = str
                zarr_array = self.zarr_group.create(
                    name, shape=shape, dtype=dtype, fill_value=fill_value, **encoding
                )
                zarr_array = _put_attrs(zarr_array, encoded_attrs)

            write_region = self._write_region if self._write_region is not None else {}
            write_region = {dim: write_region.get(dim, slice(None)) for dim in dims}

            if self._append_dim is not None and self._append_dim in dims:
                # resize existing variable
                append_axis = dims.index(self._append_dim)
                assert write_region[self._append_dim] == slice(None)
                write_region[self._append_dim] = slice(
                    zarr_array.shape[append_axis], None
                )

                new_shape = list(zarr_array.shape)
                new_shape[append_axis] += v.shape[append_axis]
                zarr_array.resize(new_shape)

            region = tuple(write_region[dim] for dim in dims)
            writer.add(v.data, zarr_array, region)

    def close(self):
        pass


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
    consolidated=None,
    overwrite_encoded_chunks=False,
    chunk_store=None,
    storage_options=None,
    decode_timedelta=None,
    use_cftime=None,
    zarr_version=None,
    **kwargs,
):
    """Load and decode a dataset from a Zarr store.

    The `store` object should be a valid store for a Zarr group. `store`
    variables must contain dimension metadata encoded in the
    `_ARRAY_DIMENSIONS` attribute or must have NCZarr format.

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
    overwrite_encoded_chunks : bool, optional
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
    drop_variables : str or iterable, optional
        A variable or list of variables to exclude from being parsed from the
        dataset. This may be useful to drop variables with problems or
        inconsistent values.
    consolidated : bool, optional
        Whether to open the store using zarr's consolidated metadata
        capability. Only works for stores that have already been consolidated.
        By default (`consolidate=None`), attempts to read consolidated metadata,
        falling back to read non-consolidated metadata if that fails.

        When the experimental ``zarr_version=3``, ``consolidated`` must be
        either be ``None`` or ``False``.
    chunk_store : MutableMapping, optional
        A separate Zarr store only for chunk data.
    storage_options : dict, optional
        Any additional parameters for the storage backend (ignored for local
        paths).
    decode_timedelta : bool, optional
        If True, decode variables and coordinates with time units in
        {'days', 'hours', 'minutes', 'seconds', 'milliseconds', 'microseconds'}
        into timedelta objects. If False, leave them encoded as numbers.
        If None (default), assume the same value of decode_time.
    use_cftime : bool, optional
        Only relevant if encoded dates come from a standard calendar
        (e.g. "gregorian", "proleptic_gregorian", "standard", or not
        specified).  If None (default), attempt to decode times to
        ``np.datetime64[ns]`` objects; if this is not possible, decode times to
        ``cftime.datetime`` objects. If True, always decode times to
        ``cftime.datetime`` objects, regardless of whether or not they can be
        represented using ``np.datetime64[ns]`` objects.  If False, always
        decode times to ``np.datetime64[ns]`` objects; if this is not possible
        raise an error.
    zarr_version : int or None, optional
        The desired zarr spec version to target (currently 2 or 3). The default
        of None will attempt to determine the zarr version from ``store`` when
        possible, otherwise defaulting to 2.

    Returns
    -------
    dataset : Dataset
        The newly created dataset.

    See Also
    --------
    open_dataset
    open_mfdataset

    References
    ----------
    http://zarr.readthedocs.io/
    """
    from .api import open_dataset

    if chunks == "auto":
        try:
            import dask.array  # noqa

            chunks = {}
        except ImportError:
            chunks = None

    if kwargs:
        raise TypeError(
            "open_zarr() got unexpected keyword arguments " + ",".join(kwargs.keys())
        )

    backend_kwargs = {
        "synchronizer": synchronizer,
        "consolidated": consolidated,
        "overwrite_encoded_chunks": overwrite_encoded_chunks,
        "chunk_store": chunk_store,
        "storage_options": storage_options,
        "stacklevel": 4,
        "zarr_version": zarr_version,
    }

    ds = open_dataset(
        filename_or_obj=store,
        group=group,
        decode_cf=decode_cf,
        mask_and_scale=mask_and_scale,
        decode_times=decode_times,
        concat_characters=concat_characters,
        decode_coords=decode_coords,
        engine="zarr",
        chunks=chunks,
        drop_variables=drop_variables,
        backend_kwargs=backend_kwargs,
        decode_timedelta=decode_timedelta,
        use_cftime=use_cftime,
        zarr_version=zarr_version,
    )
    return ds


class ZarrBackendEntrypoint(BackendEntrypoint):
    """
    Backend for ".zarr" files based on the zarr package.

    For more information about the underlying library, visit:
    https://zarr.readthedocs.io/en/stable

    See Also
    --------
    backends.ZarrStore
    """

    available = module_available("zarr")
    description = "Open zarr files (.zarr) using zarr in Xarray"
    url = "https://docs.xarray.dev/en/stable/generated/xarray.backends.ZarrBackendEntrypoint.html"

    def guess_can_open(self, filename_or_obj):
        try:
            _, ext = os.path.splitext(filename_or_obj)
        except TypeError:
            return False
        return ext in {".zarr"}

    def open_dataset(
        self,
        filename_or_obj,
        mask_and_scale=True,
        decode_times=True,
        concat_characters=True,
        decode_coords=True,
        drop_variables=None,
        use_cftime=None,
        decode_timedelta=None,
        group=None,
        mode="r",
        synchronizer=None,
        consolidated=None,
        chunk_store=None,
        storage_options=None,
        stacklevel=3,
        zarr_version=None,
    ):

        filename_or_obj = _normalize_path(filename_or_obj)
        store = ZarrStore.open_group(
            filename_or_obj,
            group=group,
            mode=mode,
            synchronizer=synchronizer,
            consolidated=consolidated,
            consolidate_on_close=False,
            chunk_store=chunk_store,
            storage_options=storage_options,
            stacklevel=stacklevel + 1,
            zarr_version=zarr_version,
        )

        store_entrypoint = StoreBackendEntrypoint()
        with close_on_error(store):
            ds = store_entrypoint.open_dataset(
                store,
                mask_and_scale=mask_and_scale,
                decode_times=decode_times,
                concat_characters=concat_characters,
                decode_coords=decode_coords,
                drop_variables=drop_variables,
                use_cftime=use_cftime,
                decode_timedelta=decode_timedelta,
            )
        return ds


BACKEND_ENTRYPOINTS["zarr"] = ZarrBackendEntrypoint
