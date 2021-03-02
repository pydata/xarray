"""Module for the TileDB array backend."""

from collections import defaultdict
from typing import Tuple

import numpy as np

from ..core.indexing import ExplicitIndexer, LazilyOuterIndexedArray
from ..core.pycompat import integer_types
from ..core.utils import FrozenDict, close_on_error
from ..core.variable import Variable
from .common import (
    BACKEND_ENTRYPOINTS,
    AbstractDataStore,
    BackendArray,
    BackendEntrypoint,
)
from .store import StoreBackendEntrypoint

_ATTR_PREFIX = "__tiledb_attr."
_DIM_PREFIX = "__tiledb_dim."
_COORD_SUFFIX = ".axis_data"


class TileDBIndexConverter:
    """Converter from xarray-style indices to TileDB-style coordinates.

    This class converts the values contained in an xarray ExplicitIndexer tuple to
    values usable for indexing a TileDB Dimension. The xarray ExplicitIndexer uses
    standard 0-based integer indices to look-up values in DataArrays and variables;
    whereas, Tiledb accesses values directly from the dimension coordinates (analogous
    to looking-up values by "location" in xarray).

    The following is assumed about xarray indices:
       * An index may be an integer, a slice, or a Numpy array of integer indices.
       * An integer index or component of an array is such that -size <= value < size.
         * Non-negative values are a standard zero-based index.
         * Negative values count backwards from the end of the array with the last value
           of the array starting at -1.
    """

    __slots__ = (
        "name",
        "dtype",
        "min_value",
        "max_value",
        "size",
        "shape",
        "delta_dtype",
    )

    def __init__(self, dim):
        dtype_kind = dim.dtype.kind
        if dtype_kind not in ("i", "u", "M"):
            raise NotImplementedError(
                f"support for reading TileDB arrays with a dimension of type "
                f"{dim.dtype} is not implemented"
            )
        self.name = dim.name
        self.dtype = dim.dtype
        self.min_value = dim.domain[0]
        self.max_value = dim.domain[1]
        self.size = dim.size
        self.shape = dim.shape
        if dtype_kind == "M":
            unit, count = np.datetime_data(self.dtype)
            self.delta_dtype = np.dtype(f"timedelta64[{count}{unit}]")
        else:
            self.delta_dtype = self.dtype

    def __getitem__(self, index):
        """Converts an xarray integer, array, or slice to an index object usable by the
            TileDB multi_index function.

        Parameters
        ----------
        index : Union[int, np.array, slice]
            An integer index, array of integer indices, or a slice for indexing an xarray
            dimension.

        Returns
        -------
        new_index : Union[self.dtype, List[self.dtype], slice]
            A value of type `self.dtype`, a list of values, or a slice for indexing a
            TileDB dimension using mulit_index.
        """
        if isinstance(index, integer_types):
            # Convert xarray index to TileDB dimension coordinate
            if not -self.size <= index < self.size:
                raise IndexError(f"index {index} out of bounds for {type(self)}")
            return self.to_coordinate(index)

        if isinstance(index, slice) and index.step in (1, None):
            # Convert from index slice to coordinate slice (note that xarray
            # includes the starting point and excludes the ending point vs. TileDB
            # multi_index which includes both the staring point and ending point).
            start, stop = index.start, index.stop
            return slice(
                self.to_coordinate(start) if start is not None else None,
                self.to_coordinate(stop - 1) if stop is not None else None,
            )

        # Convert slice or array of xarray indices to list of TileDB dimension
        # coordinates
        return list(self.to_coordinates(index))

    def to_coordinate(self, index):
        """Converts an xarray index to a coordinate for the TileDB dimension.

        Parameters
        ----------
        index : int
            An integer index for indexing an xarray dimension.

        Returns
        -------
        new_index : self.dtype
            A `self.dtype` coordinate for indexing a TileDB dimension.
        """
        return self._to_delta(index) + (
            self.min_value if index >= 0 else self.max_value
        )

    def to_coordinates(self, index):
        """
        Converts an xarray-style slice or Numpy array of indices to an array of
        coordinates for the TileDB dimension.

        Parameters
        ----------
        index : Union[slice, np.ndarray]
            A slice or an array of integer indices for indexing an xarray dimension.

        Returns
        -------
        new_index : Union[np.ndarray]
            An array of `self.dtype` coordinates for indexing a TileDB dimension.
        """
        if isinstance(index, slice):
            # Using range handles negative start/stop, out-of-bounds, and None values.
            index = range(self.size)[index]
            start = self.to_coordinate(index.start)
            stop = self.to_coordinate(index.stop)
            step = self._to_delta(index.step)
            return np.arange(start, stop, step, dtype=self.dtype)

        if isinstance(index, np.ndarray):
            if index.ndim != 1:
                raise TypeError(
                    f"invalid indexer array for {type(self)}; input array index must "
                    f"have exactly 1 dimension"
                )
            # vectorized version of self.to_coordinate
            if not ((-self.size <= index).all() and (index < self.size).all()):
                raise IndexError(f"index {index} out of bounds for {type(self)}")
            return self._to_delta(index) + np.where(
                index >= 0, self.min_value, self.max_value
            )

        raise TypeError(f"unexpected indexer type for {type(self)}")

    def _to_delta(self, i):
        delta = np.asarray(i, self.delta_dtype)
        return delta[()] if np.isscalar(i) else delta


class TileDBCoordinateWrapper(BackendArray):
    """A backend array wrapper for TileDB dimensions.

    This class is not intended to accessed directly. Instead it should be used
    through a :class:`LazilyOuterIndexedArray` object.
    """

    __slots__ = ("_converter",)

    def __init__(self, index_converter: TileDBIndexConverter):
        """
        Parameters
        ----------
        index_converter : TileDBIndexConverter
            Converter from xarray index to the dimension this CoordinateWrapper
            wraps.
        """
        self._converter = index_converter

    def __getitem__(self, indexer: ExplicitIndexer):
        key = indexer.tuple
        if len(key) != 1:
            raise ValueError(
                f"indexer with {len(key)} cannot be used for variable with 1 dimension"
            )
        index = key[0]
        if isinstance(index, integer_types):
            return self._converter.to_coordinate(index)
        else:
            return self._converter.to_coordinates(index)

    @property
    def dtype(self) -> np.dtype:
        """Data type of the backend array."""
        return self._converter.dtype

    @property
    def shape(self) -> Tuple[int, ...]:
        """Shape of the backend array."""
        return self._converter.shape


class TileDBDenseArrayWrapper(BackendArray):
    """A backend array wrapper for a TileDB attribute.

    This class is not intended to accessed directly. Instead it should be used
    through a :class:`LazilyOuterIndexedArray` object.
    """

    __slots__ = ("_array_args", "_index_converters", "shape", "dtype")

    def __init__(self, attr, uri, key, timestamp, index_converters):
        """
        Parameters
        ----------
        attr : tiledb.Attr
            The TileDB attribute being wrapped by this routine. Must be one of the
            attributes in the array at the provided URI.
        uri : str
            Uniform Resoure Identifier (URI) for TileDB array. May be a path to a
            local TileDB array or a URI for a remote resource.
        key : Optional[str]
            If not None, the key for accessing the TileDB array at the provided URI.
        timestamp : Optional[int]
            If not None, time in milliseconds to open the array at.
        ctx : Optional[tiledb.Ctx]
            If not None, a TileDB context manager object.
        index_converters : Tuple[TileDBIndexConverter, ...]
            The TileDBIndexConverters needed to convert each xarray index to its
            corresponding TileDB coordinates.
        """
        if attr.isanon:
            raise NotImplementedError(
                "Support for anonymnous TileDB attributes has not been implemented."
            )
        self.dtype = attr.dtype
        self._array_args = (uri, "r", key, timestamp, attr.name)
        self._index_converters = index_converters
        self.shape = tuple(converter.size for converter in index_converters)

    def __getitem__(self, indexer: ExplicitIndexer):
        xarray_indices = indexer.tuple
        if len(xarray_indices) != len(self._index_converters):
            raise ValueError(
                f"key of length {len(xarray_indices)} cannot be used for a TileDB array"
                f" of length {len(self._index_converters)}"
            )
        shape = tuple(
            len(range(converter.size)[index] if isinstance(index, slice) else index)
            for index, converter in zip(xarray_indices, self._index_converters)
            if not isinstance(index, integer_types)
        )
        # TileDB multi_index does not except empty arrays/slices. If a dimension is
        # length zero, return an empty numpy array of the correct length.
        if 0 in shape:
            return np.zeros(shape)
        tiledb_indices = tuple(
            converter[index]
            for index, converter in zip(xarray_indices, self._index_converters)
        )
        with tiledb.DenseArray(*self._array_args) as array:
            result = array.multi_index[tiledb_indices][self._array_args[-1]]
        # Note: TileDB multi_index returns the same number of dimensions as the initial
        # array. To match the expected xarray output, we need to reshape the result to
        # remove any dimensions corresponding to scalar-valued input.
        return result.reshape(shape)


class TileDBDataStore(AbstractDataStore):
    """Data store for reading TileDB arrays."""

    __slots__ = ("_key", "_timestamp", "_uri")

    def __init__(
        self,
        uri,
        key=None,
        timestamp=None,
    ):
        """
        Parameters
        ----------
        uri : str
            Uniform Resoure Identifier (URI) for TileDB array. May be a path to a
            local TileDB array or a URI for a remote resource.
        key : Optional[str]
            If not None, the key for accessing the TileDB array at the provided URI.
        timestamp : Optional[int]
            If not None, time in milliseconds to open the array at.
        """
        if tiledb.object_type(uri) != "array":
            raise ValueError(
                f"Unable to read from URI '{uri}'. URI is not a TileDB array."
            )
        self._uri = uri
        self._key = key
        self._timestamp = timestamp

    def get_dimensions(self):
        """Returns a dictionary of dimension names to sizes."""
        schema = tiledb.ArraySchema.load(self._uri, key=self._key)
        return FrozenDict({dim.name: dim.size for dim in schema.domain})

    def get_attrs(self):
        """Returns a dictionary of metadata stored in the array.

        Note that xarray attributes are roughly equivalent to TileDB metadata. The
        metadata returned here metadata for the dataset, but excludes encoding data for
        TileDB and attribute metadata.
        """
        with tiledb.open(self._uri, key=self._key, mode="r") as array:
            attrs = {
                key: array.meta[key]
                for key in array.meta.keys()
                if not key.startswith((_ATTR_PREFIX, _DIM_PREFIX))
            }
        return FrozenDict(attrs)

    def get_variables(self):
        """Returns a dictionary of variables.

        Return a dictionary of variables (by name) stored in the TileDB array. Each
        variable is generated from a TileDB attributes, TileDB encoding metadata,
        metadata belonging to the attribute, and the name and size of the dimensions of
        the array.
        """
        variable_metadata = self.get_variable_metadata()
        schema = tiledb.ArraySchema.load(self._uri, key=self._key)
        index_converters = tuple(map(TileDBIndexConverter, schema.domain))
        variables = {}
        # Add TileDB dimensions as xarray variables (these are the coordinates for the
        # DataArray) for all dimensions that are not "simple" 0-based integer indexes.
        for converter in index_converters:
            if converter.dtype.kind == "M" or converter.min_value:
                variables[converter.name] = Variable(
                    {converter.name: converter.size},
                    LazilyOuterIndexedArray(TileDBCoordinateWrapper(converter)),
                    variable_metadata.get(converter.name),
                )
        # Add TileDB attributes as variables.
        dims = {indexer.name: indexer.size for indexer in index_converters}
        for attr in schema:
            variable_name = attr.name
            if variable_name.endswith(_COORD_SUFFIX):
                variable_name = variable_name[: -len(_COORD_SUFFIX)]
            data = LazilyOuterIndexedArray(
                TileDBDenseArrayWrapper(
                    attr,
                    self._uri,
                    self._key,
                    self._timestamp,
                    index_converters,
                )
            )
            metadata = variable_metadata.get(attr.name)
            if attr.fill is not None:
                if metadata is None:
                    metadata = {"_FillValue": attr.fill}
                elif metadata.get("_FillValue") is not None:
                    metadata["_FillValue"] = attr.fill
            variables[variable_name] = Variable(dims, data, metadata)
        return FrozenDict(variables)

    def get_variable_metadata(self):
        """Returns a dict of dicts for attribute metadata.

        This uses the convention that attribute and dimension metadata are stored
        using the convention ``__tiledb_attr.{attribute_name}.{key} = {value}``
        for attributes and ``__tiledb_dim.{dimension_name}.{key} = {value}`` for
        dimensions.
        """
        variable_metadata = defaultdict(dict)
        with tiledb.open(self._uri, key=self._key, mode="r") as array:
            for key in array.meta.keys():
                if key.startswith((_ATTR_PREFIX, _DIM_PREFIX)):
                    last_dot_ix = key.rindex(".")
                    attr_name = key[key.index(".") + 1 : last_dot_ix]
                    if not attr_name:
                        raise RuntimeError(
                            f"cannot parse attribute metadata '{key}' with missing name"
                            " or key value."
                        )
                    attr_key = key[last_dot_ix + 1 :]
                    variable_metadata[attr_name][attr_key] = array.meta[key]
        return variable_metadata


class TileDBBackendEntrypoint(BackendEntrypoint):
    def open_dataset(
        self,
        filename_or_obj,
        *,
        mask_and_scale=True,
        decode_times=True,
        concat_characters=None,
        decode_coords=None,
        drop_variables=None,
        use_cftime=None,
        decode_timedelta=None,
        key=None,
        timestamp=None,
    ):
        datastore = TileDBDataStore(filename_or_obj, key, timestamp)
        store_entrypoint = StoreBackendEntrypoint()
        with close_on_error(datastore):
            dataset = store_entrypoint.open_dataset(
                datastore,
                mask_and_scale=mask_and_scale,
                decode_times=decode_times,
                concat_characters=concat_characters,
                decode_coords=decode_coords,
                drop_variables=drop_variables,
                use_cftime=use_cftime,
                decode_timedelta=decode_timedelta,
            )
        return dataset


try:
    import tiledb

    BACKEND_ENTRYPOINTS["tiledb"] = TileDBBackendEntrypoint
except ModuleNotFoundError:
    pass
