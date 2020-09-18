import datetime
import functools
import warnings
from numbers import Number
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Hashable,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
    cast,
)

import numpy as np
import pandas as pd

from ..plot.plot import _PlotMethods
from . import (
    computation,
    dtypes,
    groupby,
    indexing,
    ops,
    pdcompat,
    resample,
    rolling,
    utils,
    weighted,
)
from .accessor_dt import CombinedDatetimelikeAccessor
from .accessor_str import StringAccessor
from .alignment import (
    _broadcast_helper,
    _get_broadcast_dims_map_common_coords,
    align,
    reindex_like_indexers,
)
from .common import AbstractArray, DataWithCoords
from .coordinates import (
    DataArrayCoordinates,
    LevelCoordinatesSource,
    assert_coordinate_consistent,
    remap_label_indexers,
)
from .dataset import Dataset, split_indexes
from .formatting import format_item
from .indexes import Indexes, default_indexes, propagate_indexes
from .indexing import is_fancy_indexer
from .merge import PANDAS_TYPES, MergeError, _extract_indexes_from_coords
from .options import OPTIONS
from .utils import Default, ReprObject, _check_inplace, _default, either_dict_or_kwargs
from .variable import (
    IndexVariable,
    Variable,
    as_compatible_data,
    as_variable,
    assert_unique_multiindex_level_names,
)

if TYPE_CHECKING:
    T_DSorDA = TypeVar("T_DSorDA", "DataArray", Dataset)

    try:
        from dask.delayed import Delayed
    except ImportError:
        Delayed = None
    try:
        from cdms2 import Variable as cdms2_Variable
    except ImportError:
        cdms2_Variable = None
    try:
        from iris.cube import Cube as iris_Cube
    except ImportError:
        iris_Cube = None


def _infer_coords_and_dims(
    shape, coords, dims
) -> "Tuple[Dict[Any, Variable], Tuple[Hashable, ...]]":
    """All the logic for creating a new DataArray"""

    if (
        coords is not None
        and not utils.is_dict_like(coords)
        and len(coords) != len(shape)
    ):
        raise ValueError(
            "coords is not dict-like, but it has %s items, "
            "which does not match the %s dimensions of the "
            "data" % (len(coords), len(shape))
        )

    if isinstance(dims, str):
        dims = (dims,)

    if dims is None:
        dims = ["dim_%s" % n for n in range(len(shape))]
        if coords is not None and len(coords) == len(shape):
            # try to infer dimensions from coords
            if utils.is_dict_like(coords):
                # deprecated in GH993, removed in GH1539
                raise ValueError(
                    "inferring DataArray dimensions from "
                    "dictionary like ``coords`` is no longer "
                    "supported. Use an explicit list of "
                    "``dims`` instead."
                )
            for n, (dim, coord) in enumerate(zip(dims, coords)):
                coord = as_variable(coord, name=dims[n]).to_index_variable()
                dims[n] = coord.name
        dims = tuple(dims)
    elif len(dims) != len(shape):
        raise ValueError(
            "different number of dimensions on data "
            "and dims: %s vs %s" % (len(shape), len(dims))
        )
    else:
        for d in dims:
            if not isinstance(d, str):
                raise TypeError("dimension %s is not a string" % d)

    new_coords: Dict[Any, Variable] = {}

    if utils.is_dict_like(coords):
        for k, v in coords.items():
            new_coords[k] = as_variable(v, name=k)
    elif coords is not None:
        for dim, coord in zip(dims, coords):
            var = as_variable(coord, name=dim)
            var.dims = (dim,)
            new_coords[dim] = var.to_index_variable()

    sizes = dict(zip(dims, shape))
    for k, v in new_coords.items():
        if any(d not in dims for d in v.dims):
            raise ValueError(
                "coordinate %s has dimensions %s, but these "
                "are not a subset of the DataArray "
                "dimensions %s" % (k, v.dims, dims)
            )

        for d, s in zip(v.dims, v.shape):
            if s != sizes[d]:
                raise ValueError(
                    "conflicting sizes for dimension %r: "
                    "length %s on the data but length %s on "
                    "coordinate %r" % (d, sizes[d], s, k)
                )

        if k in sizes and v.shape != (sizes[k],):
            raise ValueError(
                "coordinate %r is a DataArray dimension, but "
                "it has shape %r rather than expected shape %r "
                "matching the dimension size" % (k, v.shape, (sizes[k],))
            )

    assert_unique_multiindex_level_names(new_coords)

    return new_coords, dims


def _check_data_shape(data, coords, dims):
    if data is dtypes.NA:
        data = np.nan
    if coords is not None and utils.is_scalar(data, include_0d=False):
        if utils.is_dict_like(coords):
            if dims is None:
                return data
            else:
                data_shape = tuple(
                    as_variable(coords[k], k).size if k in coords.keys() else 1
                    for k in dims
                )
        else:
            data_shape = tuple(as_variable(coord, "foo").size for coord in coords)
        data = np.full(data_shape, data)
    return data


class _LocIndexer:
    __slots__ = ("data_array",)

    def __init__(self, data_array: "DataArray"):
        self.data_array = data_array

    def __getitem__(self, key) -> "DataArray":
        if not utils.is_dict_like(key):
            # expand the indexer so we can handle Ellipsis
            labels = indexing.expanded_indexer(key, self.data_array.ndim)
            key = dict(zip(self.data_array.dims, labels))
        return self.data_array.sel(**key)

    def __setitem__(self, key, value) -> None:
        if not utils.is_dict_like(key):
            # expand the indexer so we can handle Ellipsis
            labels = indexing.expanded_indexer(key, self.data_array.ndim)
            key = dict(zip(self.data_array.dims, labels))

        pos_indexers, _ = remap_label_indexers(self.data_array, key)
        self.data_array[pos_indexers] = value


# Used as the key corresponding to a DataArray's variable when converting
# arbitrary DataArray objects to datasets
_THIS_ARRAY = ReprObject("<this-array>")


class DataArray(AbstractArray, DataWithCoords):
    """N-dimensional array with labeled coordinates and dimensions.

    DataArray provides a wrapper around numpy ndarrays that uses labeled
    dimensions and coordinates to support metadata aware operations. The API is
    similar to that for the pandas Series or DataFrame, but DataArray objects
    can have any number of dimensions, and their contents have fixed data
    types.

    Additional features over raw numpy arrays:

    - Apply operations over dimensions by name: ``x.sum('time')``.
    - Select or assign values by integer location (like numpy): ``x[:10]``
      or by label (like pandas): ``x.loc['2014-01-01']`` or
      ``x.sel(time='2014-01-01')``.
    - Mathematical operations (e.g., ``x - y``) vectorize across multiple
      dimensions (known in numpy as "broadcasting") based on dimension names,
      regardless of their original order.
    - Keep track of arbitrary metadata in the form of a Python dictionary:
      ``x.attrs``
    - Convert to a pandas Series: ``x.to_series()``.

    Getting items from or doing mathematical operations with a DataArray
    always returns another DataArray.
    """

    _cache: Dict[str, Any]
    _coords: Dict[Any, Variable]
    _indexes: Optional[Dict[Hashable, pd.Index]]
    _name: Optional[Hashable]
    _variable: Variable

    __slots__ = (
        "_cache",
        "_coords",
        "_file_obj",
        "_indexes",
        "_name",
        "_variable",
        "__weakref__",
    )

    _groupby_cls = groupby.DataArrayGroupBy
    _rolling_cls = rolling.DataArrayRolling
    _coarsen_cls = rolling.DataArrayCoarsen
    _resample_cls = resample.DataArrayResample
    _weighted_cls = weighted.DataArrayWeighted

    dt = utils.UncachedAccessor(CombinedDatetimelikeAccessor)

    def __init__(
        self,
        data: Any = dtypes.NA,
        coords: Union[Sequence[Tuple], Mapping[Hashable, Any], None] = None,
        dims: Union[Hashable, Sequence[Hashable], None] = None,
        name: Hashable = None,
        attrs: Mapping = None,
        # internal parameters
        indexes: Dict[Hashable, pd.Index] = None,
        fastpath: bool = False,
    ):
        """
        Parameters
        ----------
        data : array_like
            Values for this array. Must be an ``numpy.ndarray``, ndarray like,
            or castable to an ``ndarray``. If a self-described xarray or pandas
            object, attempts are made to use this array's metadata to fill in
            other unspecified arguments. A view of the array's data is used
            instead of a copy if possible.
        coords : sequence or dict of array_like, optional
            Coordinates (tick labels) to use for indexing along each dimension.
            The following notations are accepted:

            - mapping {dimension name: array-like}
            - sequence of tuples that are valid arguments for xarray.Variable()
              - (dims, data)
              - (dims, data, attrs)
              - (dims, data, attrs, encoding)

            Additionally, it is possible to define a coord whose name
            does not match the dimension name, or a coord based on multiple
            dimensions, with one of the following notations:

            - mapping {coord name: DataArray}
            - mapping {coord name: Variable}
            - mapping {coord name: (dimension name, array-like)}
            - mapping {coord name: (tuple of dimension names, array-like)}

        dims : hashable or sequence of hashable, optional
            Name(s) of the data dimension(s). Must be either a hashable (only
            for 1D data) or a sequence of hashables with length equal to the
            number of dimensions. If this argument is omitted, dimension names
            default to ``['dim_0', ... 'dim_n']``.
        name : str or None, optional
            Name of this array.
        attrs : dict_like or None, optional
            Attributes to assign to the new instance. By default, an empty
            attribute dictionary is initialized.
        """
        if fastpath:
            variable = data
            assert dims is None
            assert attrs is None
        else:
            # try to fill in arguments from data if they weren't supplied
            if coords is None:

                if isinstance(data, DataArray):
                    coords = data.coords
                elif isinstance(data, pd.Series):
                    coords = [data.index]
                elif isinstance(data, pd.DataFrame):
                    coords = [data.index, data.columns]
                elif isinstance(data, (pd.Index, IndexVariable)):
                    coords = [data]
                elif isinstance(data, pdcompat.Panel):
                    coords = [data.items, data.major_axis, data.minor_axis]

            if dims is None:
                dims = getattr(data, "dims", getattr(coords, "dims", None))
            if name is None:
                name = getattr(data, "name", None)
            if attrs is None and not isinstance(data, PANDAS_TYPES):
                attrs = getattr(data, "attrs", None)

            data = _check_data_shape(data, coords, dims)
            data = as_compatible_data(data)
            coords, dims = _infer_coords_and_dims(data.shape, coords, dims)
            variable = Variable(dims, data, attrs, fastpath=True)
            indexes = dict(
                _extract_indexes_from_coords(coords)
            )  # needed for to_dataset

        # These fully describe a DataArray
        self._variable = variable
        assert isinstance(coords, dict)
        self._coords = coords
        self._name = name

        # TODO(shoyer): document this argument, once it becomes part of the
        # public interface.
        self._indexes = indexes

        self._file_obj = None

    def _replace(
        self,
        variable: Variable = None,
        coords=None,
        name: Union[Hashable, None, Default] = _default,
        indexes=None,
    ) -> "DataArray":
        if variable is None:
            variable = self.variable
        if coords is None:
            coords = self._coords
        if name is _default:
            name = self.name
        return type(self)(variable, coords, name=name, fastpath=True, indexes=indexes)

    def _replace_maybe_drop_dims(
        self, variable: Variable, name: Union[Hashable, None, Default] = _default
    ) -> "DataArray":
        if variable.dims == self.dims and variable.shape == self.shape:
            coords = self._coords.copy()
            indexes = self._indexes
        elif variable.dims == self.dims:
            # Shape has changed (e.g. from reduce(..., keepdims=True)
            new_sizes = dict(zip(self.dims, variable.shape))
            coords = {
                k: v
                for k, v in self._coords.items()
                if v.shape == tuple(new_sizes[d] for d in v.dims)
            }
            changed_dims = [
                k for k in variable.dims if variable.sizes[k] != self.sizes[k]
            ]
            indexes = propagate_indexes(self._indexes, exclude=changed_dims)
        else:
            allowed_dims = set(variable.dims)
            coords = {
                k: v for k, v in self._coords.items() if set(v.dims) <= allowed_dims
            }
            indexes = propagate_indexes(
                self._indexes, exclude=(set(self.dims) - allowed_dims)
            )
        return self._replace(variable, coords, name, indexes=indexes)

    def _overwrite_indexes(self, indexes: Mapping[Hashable, Any]) -> "DataArray":
        if not len(indexes):
            return self
        coords = self._coords.copy()
        for name, idx in indexes.items():
            coords[name] = IndexVariable(name, idx)
        obj = self._replace(coords=coords)

        # switch from dimension to level names, if necessary
        dim_names: Dict[Any, str] = {}
        for dim, idx in indexes.items():
            if not isinstance(idx, pd.MultiIndex) and idx.name != dim:
                dim_names[dim] = idx.name
        if dim_names:
            obj = obj.rename(dim_names)
        return obj

    def _to_temp_dataset(self) -> Dataset:
        return self._to_dataset_whole(name=_THIS_ARRAY, shallow_copy=False)

    def _from_temp_dataset(
        self, dataset: Dataset, name: Union[Hashable, None, Default] = _default
    ) -> "DataArray":
        variable = dataset._variables.pop(_THIS_ARRAY)
        coords = dataset._variables
        indexes = dataset._indexes
        return self._replace(variable, coords, name, indexes=indexes)

    def _to_dataset_split(self, dim: Hashable) -> Dataset:
        """ splits dataarray along dimension 'dim' """

        def subset(dim, label):
            array = self.loc[{dim: label}]
            array.attrs = {}
            return as_variable(array)

        variables = {label: subset(dim, label) for label in self.get_index(dim)}
        variables.update({k: v for k, v in self._coords.items() if k != dim})
        indexes = propagate_indexes(self._indexes, exclude=dim)
        coord_names = set(self._coords) - {dim}
        dataset = Dataset._construct_direct(
            variables, coord_names, indexes=indexes, attrs=self.attrs
        )
        return dataset

    def _to_dataset_whole(
        self, name: Hashable = None, shallow_copy: bool = True
    ) -> Dataset:
        if name is None:
            name = self.name
        if name is None:
            raise ValueError(
                "unable to convert unnamed DataArray to a "
                "Dataset without providing an explicit name"
            )
        if name in self.coords:
            raise ValueError(
                "cannot create a Dataset from a DataArray with "
                "the same name as one of its coordinates"
            )
        # use private APIs for speed: this is called by _to_temp_dataset(),
        # which is used in the guts of a lot of operations (e.g., reindex)
        variables = self._coords.copy()
        variables[name] = self.variable
        if shallow_copy:
            for k in variables:
                variables[k] = variables[k].copy(deep=False)
        indexes = self._indexes

        coord_names = set(self._coords)
        dataset = Dataset._construct_direct(variables, coord_names, indexes=indexes)
        return dataset

    def to_dataset(
        self,
        dim: Hashable = None,
        *,
        name: Hashable = None,
        promote_attrs: bool = False,
    ) -> Dataset:
        """Convert a DataArray to a Dataset.

        Parameters
        ----------
        dim : hashable, optional
            Name of the dimension on this array along which to split this array
            into separate variables. If not provided, this array is converted
            into a Dataset of one variable.
        name : hashable, optional
            Name to substitute for this array's name. Only valid if ``dim`` is
            not provided.
        promote_attrs : bool, default: False
            Set to True to shallow copy attrs of DataArray to returned Dataset.

        Returns
        -------
        dataset : Dataset
        """
        if dim is not None and dim not in self.dims:
            raise TypeError(
                f"{dim} is not a dim. If supplying a ``name``, pass as a kwarg."
            )

        if dim is not None:
            if name is not None:
                raise TypeError("cannot supply both dim and name arguments")
            result = self._to_dataset_split(dim)
        else:
            result = self._to_dataset_whole(name)

        if promote_attrs:
            result.attrs = dict(self.attrs)

        return result

    @property
    def name(self) -> Optional[Hashable]:
        """The name of this array."""
        return self._name

    @name.setter
    def name(self, value: Optional[Hashable]) -> None:
        self._name = value

    @property
    def variable(self) -> Variable:
        """Low level interface to the Variable object for this DataArray."""
        return self._variable

    @property
    def dtype(self) -> np.dtype:
        return self.variable.dtype

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.variable.shape

    @property
    def size(self) -> int:
        return self.variable.size

    @property
    def nbytes(self) -> int:
        return self.variable.nbytes

    @property
    def ndim(self) -> int:
        return self.variable.ndim

    def __len__(self) -> int:
        return len(self.variable)

    @property
    def data(self) -> Any:
        """The array's data as a dask or numpy array"""
        return self.variable.data

    @data.setter
    def data(self, value: Any) -> None:
        self.variable.data = value

    @property
    def values(self) -> np.ndarray:
        """The array's data as a numpy.ndarray"""
        return self.variable.values

    @values.setter
    def values(self, value: Any) -> None:
        self.variable.values = value

    @property
    def _in_memory(self) -> bool:
        return self.variable._in_memory

    def to_index(self) -> pd.Index:
        """Convert this variable to a pandas.Index. Only possible for 1D
        arrays.
        """
        return self.variable.to_index()

    @property
    def dims(self) -> Tuple[Hashable, ...]:
        """Tuple of dimension names associated with this array.

        Note that the type of this property is inconsistent with
        `Dataset.dims`.  See `Dataset.sizes` and `DataArray.sizes` for
        consistently named properties.
        """
        return self.variable.dims

    @dims.setter
    def dims(self, value):
        raise AttributeError(
            "you cannot assign dims on a DataArray. Use "
            ".rename() or .swap_dims() instead."
        )

    def _item_key_to_dict(self, key: Any) -> Mapping[Hashable, Any]:
        if utils.is_dict_like(key):
            return key
        else:
            key = indexing.expanded_indexer(key, self.ndim)
            return dict(zip(self.dims, key))

    @property
    def _level_coords(self) -> Dict[Hashable, Hashable]:
        """Return a mapping of all MultiIndex levels and their corresponding
        coordinate name.
        """
        level_coords: Dict[Hashable, Hashable] = {}

        for cname, var in self._coords.items():
            if var.ndim == 1 and isinstance(var, IndexVariable):
                level_names = var.level_names
                if level_names is not None:
                    (dim,) = var.dims
                    level_coords.update({lname: dim for lname in level_names})
        return level_coords

    def _getitem_coord(self, key):
        from .dataset import _get_virtual_variable

        try:
            var = self._coords[key]
        except KeyError:
            dim_sizes = dict(zip(self.dims, self.shape))
            _, key, var = _get_virtual_variable(
                self._coords, key, self._level_coords, dim_sizes
            )

        return self._replace_maybe_drop_dims(var, name=key)

    def __getitem__(self, key: Any) -> "DataArray":
        if isinstance(key, str):
            return self._getitem_coord(key)
        else:
            # xarray-style array indexing
            return self.isel(indexers=self._item_key_to_dict(key))

    def __setitem__(self, key: Any, value: Any) -> None:
        if isinstance(key, str):
            self.coords[key] = value
        else:
            # Coordinates in key, value and self[key] should be consistent.
            # TODO Coordinate consistency in key is checked here, but it
            # causes unnecessary indexing. It should be optimized.
            obj = self[key]
            if isinstance(value, DataArray):
                assert_coordinate_consistent(value, obj.coords.variables)
            # DataArray key -> Variable key
            key = {
                k: v.variable if isinstance(v, DataArray) else v
                for k, v in self._item_key_to_dict(key).items()
            }
            self.variable[key] = value

    def __delitem__(self, key: Any) -> None:
        del self.coords[key]

    @property
    def _attr_sources(self) -> List[Mapping[Hashable, Any]]:
        """List of places to look-up items for attribute-style access"""
        return self._item_sources + [self.attrs]

    @property
    def _item_sources(self) -> List[Mapping[Hashable, Any]]:
        """List of places to look-up items for key-completion"""
        return [
            self.coords,
            {d: self.coords[d] for d in self.dims},
            LevelCoordinatesSource(self),
        ]

    def __contains__(self, key: Any) -> bool:
        return key in self.data

    @property
    def loc(self) -> _LocIndexer:
        """Attribute for location based indexing like pandas."""
        return _LocIndexer(self)

    @property
    def attrs(self) -> Dict[Hashable, Any]:
        """Dictionary storing arbitrary metadata with this array."""
        return self.variable.attrs

    @attrs.setter
    def attrs(self, value: Mapping[Hashable, Any]) -> None:
        # Disable type checking to work around mypy bug - see mypy#4167
        self.variable.attrs = value  # type: ignore

    @property
    def encoding(self) -> Dict[Hashable, Any]:
        """Dictionary of format-specific settings for how this array should be
        serialized."""
        return self.variable.encoding

    @encoding.setter
    def encoding(self, value: Mapping[Hashable, Any]) -> None:
        self.variable.encoding = value

    @property
    def indexes(self) -> Indexes:
        """Mapping of pandas.Index objects used for label based indexing"""
        if self._indexes is None:
            self._indexes = default_indexes(self._coords, self.dims)
        return Indexes(self._indexes)

    @property
    def coords(self) -> DataArrayCoordinates:
        """Dictionary-like container of coordinate arrays."""
        return DataArrayCoordinates(self)

    def reset_coords(
        self,
        names: Union[Iterable[Hashable], Hashable, None] = None,
        drop: bool = False,
        inplace: bool = None,
    ) -> Union[None, "DataArray", Dataset]:
        """Given names of coordinates, reset them to become variables.

        Parameters
        ----------
        names : hashable or iterable of hashable, optional
            Name(s) of non-index coordinates in this dataset to reset into
            variables. By default, all non-index coordinates are reset.
        drop : bool, optional
            If True, remove coordinates instead of converting them into
            variables.

        Returns
        -------
        Dataset, or DataArray if ``drop == True``
        """
        _check_inplace(inplace)
        if names is None:
            names = set(self.coords) - set(self.dims)
        dataset = self.coords.to_dataset().reset_coords(names, drop)
        if drop:
            return self._replace(coords=dataset._variables)
        else:
            if self.name is None:
                raise ValueError(
                    "cannot reset_coords with drop=False on an unnamed DataArrray"
                )
            dataset[self.name] = self.variable
            return dataset

    def __dask_tokenize__(self):
        from dask.base import normalize_token

        return normalize_token((type(self), self._variable, self._coords, self._name))

    def __dask_graph__(self):
        return self._to_temp_dataset().__dask_graph__()

    def __dask_keys__(self):
        return self._to_temp_dataset().__dask_keys__()

    def __dask_layers__(self):
        return self._to_temp_dataset().__dask_layers__()

    @property
    def __dask_optimize__(self):
        return self._to_temp_dataset().__dask_optimize__

    @property
    def __dask_scheduler__(self):
        return self._to_temp_dataset().__dask_scheduler__

    def __dask_postcompute__(self):
        func, args = self._to_temp_dataset().__dask_postcompute__()
        return self._dask_finalize, (func, args, self.name)

    def __dask_postpersist__(self):
        func, args = self._to_temp_dataset().__dask_postpersist__()
        return self._dask_finalize, (func, args, self.name)

    @staticmethod
    def _dask_finalize(results, func, args, name):
        ds = func(results, *args)
        variable = ds._variables.pop(_THIS_ARRAY)
        coords = ds._variables
        return DataArray(variable, coords, name=name, fastpath=True)

    def load(self, **kwargs) -> "DataArray":
        """Manually trigger loading of this array's data from disk or a
        remote source into memory and return this array.

        Normally, it should not be necessary to call this method in user code,
        because all xarray functions should either work on deferred data or
        load data automatically. However, this method can be necessary when
        working with many file objects on disk.

        Parameters
        ----------
        **kwargs : dict
            Additional keyword arguments passed on to ``dask.array.compute``.

        See Also
        --------
        dask.array.compute
        """
        ds = self._to_temp_dataset().load(**kwargs)
        new = self._from_temp_dataset(ds)
        self._variable = new._variable
        self._coords = new._coords
        return self

    def compute(self, **kwargs) -> "DataArray":
        """Manually trigger loading of this array's data from disk or a
        remote source into memory and return a new array. The original is
        left unaltered.

        Normally, it should not be necessary to call this method in user code,
        because all xarray functions should either work on deferred data or
        load data automatically. However, this method can be necessary when
        working with many file objects on disk.

        Parameters
        ----------
        **kwargs : dict
            Additional keyword arguments passed on to ``dask.array.compute``.

        See Also
        --------
        dask.array.compute
        """
        new = self.copy(deep=False)
        return new.load(**kwargs)

    def persist(self, **kwargs) -> "DataArray":
        """Trigger computation in constituent dask arrays

        This keeps them as dask arrays but encourages them to keep data in
        memory.  This is particularly useful when on a distributed machine.
        When on a single machine consider using ``.compute()`` instead.

        Parameters
        ----------
        **kwargs : dict
            Additional keyword arguments passed on to ``dask.persist``.

        See Also
        --------
        dask.persist
        """
        ds = self._to_temp_dataset().persist(**kwargs)
        return self._from_temp_dataset(ds)

    def copy(self, deep: bool = True, data: Any = None) -> "DataArray":
        """Returns a copy of this array.

        If `deep=True`, a deep copy is made of the data array.
        Otherwise, a shallow copy is made, and the returned data array's
        values are a new view of this data array's values.

        Use `data` to create a new object with the same structure as
        original but entirely new data.

        Parameters
        ----------
        deep : bool, optional
            Whether the data array and its coordinates are loaded into memory
            and copied onto the new object. Default is True.
        data : array_like, optional
            Data to use in the new object. Must have same shape as original.
            When `data` is used, `deep` is ignored for all data variables,
            and only used for coords.

        Returns
        -------
        object : DataArray
            New object with dimensions, attributes, coordinates, name,
            encoding, and optionally data copied from original.

        Examples
        --------

        Shallow versus deep copy

        >>> array = xr.DataArray([1, 2, 3], dims="x", coords={"x": ["a", "b", "c"]})
        >>> array.copy()
        <xarray.DataArray (x: 3)>
        array([1, 2, 3])
        Coordinates:
          * x        (x) <U1 'a' 'b' 'c'
        >>> array_0 = array.copy(deep=False)
        >>> array_0[0] = 7
        >>> array_0
        <xarray.DataArray (x: 3)>
        array([7, 2, 3])
        Coordinates:
          * x        (x) <U1 'a' 'b' 'c'
        >>> array
        <xarray.DataArray (x: 3)>
        array([7, 2, 3])
        Coordinates:
          * x        (x) <U1 'a' 'b' 'c'

        Changing the data using the ``data`` argument maintains the
        structure of the original object, but with the new data. Original
        object is unaffected.

        >>> array.copy(data=[0.1, 0.2, 0.3])
        <xarray.DataArray (x: 3)>
        array([0.1, 0.2, 0.3])
        Coordinates:
          * x        (x) <U1 'a' 'b' 'c'
        >>> array
        <xarray.DataArray (x: 3)>
        array([7, 2, 3])
        Coordinates:
          * x        (x) <U1 'a' 'b' 'c'

        See Also
        --------
        pandas.DataFrame.copy
        """
        variable = self.variable.copy(deep=deep, data=data)
        coords = {k: v.copy(deep=deep) for k, v in self._coords.items()}
        if self._indexes is None:
            indexes = self._indexes
        else:
            indexes = {k: v.copy(deep=deep) for k, v in self._indexes.items()}
        return self._replace(variable, coords, indexes=indexes)

    def __copy__(self) -> "DataArray":
        return self.copy(deep=False)

    def __deepcopy__(self, memo=None) -> "DataArray":
        # memo does nothing but is required for compatibility with
        # copy.deepcopy
        return self.copy(deep=True)

    # mutable objects should not be hashable
    # https://github.com/python/mypy/issues/4266
    __hash__ = None  # type: ignore

    @property
    def chunks(self) -> Optional[Tuple[Tuple[int, ...], ...]]:
        """Block dimensions for this array's data or None if it's not a dask
        array.
        """
        return self.variable.chunks

    def chunk(
        self,
        chunks: Union[
            None,
            Number,
            Tuple[Number, ...],
            Tuple[Tuple[Number, ...], ...],
            Mapping[Hashable, Union[None, Number, Tuple[Number, ...]]],
        ] = None,
        name_prefix: str = "xarray-",
        token: str = None,
        lock: bool = False,
    ) -> "DataArray":
        """Coerce this array's data into a dask arrays with the given chunks.

        If this variable is a non-dask array, it will be converted to dask
        array. If it's a dask array, it will be rechunked to the given chunk
        sizes.

        If neither chunks is not provided for one or more dimensions, chunk
        sizes along that dimension will not be updated; non-dask arrays will be
        converted into dask arrays with a single block.

        Parameters
        ----------
        chunks : int, tuple of int or mapping of hashable to int, optional
            Chunk sizes along each dimension, e.g., ``5``, ``(5, 5)`` or
            ``{'x': 5, 'y': 5}``.
        name_prefix : str, optional
            Prefix for the name of the new dask array.
        token : str, optional
            Token uniquely identifying this array.
        lock : optional
            Passed on to :py:func:`dask.array.from_array`, if the array is not
            already as dask array.

        Returns
        -------
        chunked : xarray.DataArray
        """
        if isinstance(chunks, (tuple, list)):
            chunks = dict(zip(self.dims, chunks))

        ds = self._to_temp_dataset().chunk(
            chunks, name_prefix=name_prefix, token=token, lock=lock
        )
        return self._from_temp_dataset(ds)

    def isel(
        self,
        indexers: Mapping[Hashable, Any] = None,
        drop: bool = False,
        missing_dims: str = "raise",
        **indexers_kwargs: Any,
    ) -> "DataArray":
        """Return a new DataArray whose data is given by integer indexing
        along the specified dimension(s).

        Parameters
        ----------
        indexers : dict, optional
            A dict with keys matching dimensions and values given
            by integers, slice objects or arrays.
            indexer can be a integer, slice, array-like or DataArray.
            If DataArrays are passed as indexers, xarray-style indexing will be
            carried out. See :ref:`indexing` for the details.
            One of indexers or indexers_kwargs must be provided.
        drop : bool, optional
            If ``drop=True``, drop coordinates variables indexed by integers
            instead of making them scalar.
        missing_dims : {"raise", "warn", "ignore"}, default: "raise"
            What to do if dimensions that should be selected from are not present in the
            DataArray:
            - "raise": raise an exception
            - "warning": raise a warning, and ignore the missing dimensions
            - "ignore": ignore the missing dimensions
        **indexers_kwargs : {dim: indexer, ...}, optional
            The keyword arguments form of ``indexers``.

        See Also
        --------
        Dataset.isel
        DataArray.sel
        """

        indexers = either_dict_or_kwargs(indexers, indexers_kwargs, "isel")

        if any(is_fancy_indexer(idx) for idx in indexers.values()):
            ds = self._to_temp_dataset()._isel_fancy(
                indexers, drop=drop, missing_dims=missing_dims
            )
            return self._from_temp_dataset(ds)

        # Much faster algorithm for when all indexers are ints, slices, one-dimensional
        # lists, or zero or one-dimensional np.ndarray's

        variable = self._variable.isel(indexers, missing_dims=missing_dims)

        coords = {}
        for coord_name, coord_value in self._coords.items():
            coord_indexers = {
                k: v for k, v in indexers.items() if k in coord_value.dims
            }
            if coord_indexers:
                coord_value = coord_value.isel(coord_indexers)
                if drop and coord_value.ndim == 0:
                    continue
            coords[coord_name] = coord_value

        return self._replace(variable=variable, coords=coords)

    def sel(
        self,
        indexers: Mapping[Hashable, Any] = None,
        method: str = None,
        tolerance=None,
        drop: bool = False,
        **indexers_kwargs: Any,
    ) -> "DataArray":
        """Return a new DataArray whose data is given by selecting index
        labels along the specified dimension(s).

        In contrast to `DataArray.isel`, indexers for this method should use
        labels instead of integers.

        Under the hood, this method is powered by using pandas's powerful Index
        objects. This makes label based indexing essentially just as fast as
        using integer indexing.

        It also means this method uses pandas's (well documented) logic for
        indexing. This means you can use string shortcuts for datetime indexes
        (e.g., '2000-01' to select all values in January 2000). It also means
        that slices are treated as inclusive of both the start and stop values,
        unlike normal Python indexing.

        .. warning::

          Do not try to assign values when using any of the indexing methods
          ``isel`` or ``sel``::

            da = xr.DataArray([0, 1, 2, 3], dims=['x'])
            # DO NOT do this
            da.isel(x=[0, 1, 2])[1] = -1

          Assigning values with the chained indexing using ``.sel`` or
          ``.isel`` fails silently.

        Parameters
        ----------
        indexers : dict, optional
            A dict with keys matching dimensions and values given
            by scalars, slices or arrays of tick labels. For dimensions with
            multi-index, the indexer may also be a dict-like object with keys
            matching index level names.
            If DataArrays are passed as indexers, xarray-style indexing will be
            carried out. See :ref:`indexing` for the details.
            One of indexers or indexers_kwargs must be provided.
        method : {None, "nearest", "pad", "ffill", "backfill", "bfill"}, optional
            Method to use for inexact matches:

            * None (default): only exact matches
            * pad / ffill: propagate last valid index value forward
            * backfill / bfill: propagate next valid index value backward
            * nearest: use nearest valid index value
        tolerance : optional
            Maximum distance between original and new labels for inexact
            matches. The values of the index at the matching locations must
            satisfy the equation ``abs(index[indexer] - target) <= tolerance``.
        drop : bool, optional
            If ``drop=True``, drop coordinates variables in `indexers` instead
            of making them scalar.
        **indexers_kwargs : {dim: indexer, ...}, optional
            The keyword arguments form of ``indexers``.
            One of indexers or indexers_kwargs must be provided.

        Returns
        -------
        obj : DataArray
            A new DataArray with the same contents as this DataArray, except the
            data and each dimension is indexed by the appropriate indexers.
            If indexer DataArrays have coordinates that do not conflict with
            this object, then these coordinates will be attached.
            In general, each array's data will be a view of the array's data
            in this DataArray, unless vectorized indexing was triggered by using
            an array indexer, in which case the data will be a copy.

        See Also
        --------
        Dataset.sel
        DataArray.isel

        """
        ds = self._to_temp_dataset().sel(
            indexers=indexers,
            drop=drop,
            method=method,
            tolerance=tolerance,
            **indexers_kwargs,
        )
        return self._from_temp_dataset(ds)

    def head(
        self,
        indexers: Union[Mapping[Hashable, int], int] = None,
        **indexers_kwargs: Any,
    ) -> "DataArray":
        """Return a new DataArray whose data is given by the the first `n`
        values along the specified dimension(s). Default `n` = 5

        See Also
        --------
        Dataset.head
        DataArray.tail
        DataArray.thin
        """
        ds = self._to_temp_dataset().head(indexers, **indexers_kwargs)
        return self._from_temp_dataset(ds)

    def tail(
        self,
        indexers: Union[Mapping[Hashable, int], int] = None,
        **indexers_kwargs: Any,
    ) -> "DataArray":
        """Return a new DataArray whose data is given by the the last `n`
        values along the specified dimension(s). Default `n` = 5

        See Also
        --------
        Dataset.tail
        DataArray.head
        DataArray.thin
        """
        ds = self._to_temp_dataset().tail(indexers, **indexers_kwargs)
        return self._from_temp_dataset(ds)

    def thin(
        self,
        indexers: Union[Mapping[Hashable, int], int] = None,
        **indexers_kwargs: Any,
    ) -> "DataArray":
        """Return a new DataArray whose data is given by each `n` value
        along the specified dimension(s).

        See Also
        --------
        Dataset.thin
        DataArray.head
        DataArray.tail
        """
        ds = self._to_temp_dataset().thin(indexers, **indexers_kwargs)
        return self._from_temp_dataset(ds)

    def broadcast_like(
        self, other: Union["DataArray", Dataset], exclude: Iterable[Hashable] = None
    ) -> "DataArray":
        """Broadcast this DataArray against another Dataset or DataArray.

        This is equivalent to xr.broadcast(other, self)[1]

        xarray objects are broadcast against each other in arithmetic
        operations, so this method is not be necessary for most uses.

        If no change is needed, the input data is returned to the output
        without being copied.

        If new coords are added by the broadcast, their values are
        NaN filled.

        Parameters
        ----------
        other : Dataset or DataArray
            Object against which to broadcast this array.
        exclude : iterable of hashable, optional
            Dimensions that must not be broadcasted

        Returns
        -------
        new_da : DataArray
            The caller broadcasted against ``other``.

        Examples
        --------

        >>> arr1 = xr.DataArray(
        ...     np.random.randn(2, 3),
        ...     dims=("x", "y"),
        ...     coords={"x": ["a", "b"], "y": ["a", "b", "c"]},
        ... )
        >>> arr2 = xr.DataArray(
        ...     np.random.randn(3, 2),
        ...     dims=("x", "y"),
        ...     coords={"x": ["a", "b", "c"], "y": ["a", "b"]},
        ... )
        >>> arr1
        <xarray.DataArray (x: 2, y: 3)>
        array([[ 1.76405235,  0.40015721,  0.97873798],
               [ 2.2408932 ,  1.86755799, -0.97727788]])
        Coordinates:
          * x        (x) <U1 'a' 'b'
          * y        (y) <U1 'a' 'b' 'c'
        >>> arr2
        <xarray.DataArray (x: 3, y: 2)>
        array([[ 0.95008842, -0.15135721],
               [-0.10321885,  0.4105985 ],
               [ 0.14404357,  1.45427351]])
        Coordinates:
          * x        (x) <U1 'a' 'b' 'c'
          * y        (y) <U1 'a' 'b'
        >>> arr1.broadcast_like(arr2)
        <xarray.DataArray (x: 3, y: 3)>
        array([[ 1.76405235,  0.40015721,  0.97873798],
               [ 2.2408932 ,  1.86755799, -0.97727788],
               [        nan,         nan,         nan]])
        Coordinates:
          * x        (x) object 'a' 'b' 'c'
          * y        (y) object 'a' 'b' 'c'
        """
        if exclude is None:
            exclude = set()
        else:
            exclude = set(exclude)
        args = align(other, self, join="outer", copy=False, exclude=exclude)

        dims_map, common_coords = _get_broadcast_dims_map_common_coords(args, exclude)

        return _broadcast_helper(args[1], exclude, dims_map, common_coords)

    def reindex_like(
        self,
        other: Union["DataArray", Dataset],
        method: str = None,
        tolerance=None,
        copy: bool = True,
        fill_value=dtypes.NA,
    ) -> "DataArray":
        """Conform this object onto the indexes of another object, filling in
        missing values with ``fill_value``. The default fill value is NaN.

        Parameters
        ----------
        other : Dataset or DataArray
            Object with an 'indexes' attribute giving a mapping from dimension
            names to pandas.Index objects, which provides coordinates upon
            which to index the variables in this dataset. The indexes on this
            other object need not be the same as the indexes on this
            dataset. Any mis-matched index values will be filled in with
            NaN, and any mis-matched dimension names will simply be ignored.
        method : {None, "nearest", "pad", "ffill", "backfill", "bfill"}, optional
            Method to use for filling index values from other not found on this
            data array:

            * None (default): don't fill gaps
            * pad / ffill: propagate last valid index value forward
            * backfill / bfill: propagate next valid index value backward
            * nearest: use nearest valid index value
        tolerance : optional
            Maximum distance between original and new labels for inexact
            matches. The values of the index at the matching locations must
            satisfy the equation ``abs(index[indexer] - target) <= tolerance``.
        copy : bool, optional
            If ``copy=True``, data in the return value is always copied. If
            ``copy=False`` and reindexing is unnecessary, or can be performed
            with only slice operations, then the output may share memory with
            the input. In either case, a new xarray object is always returned.
        fill_value : scalar or dict-like, optional
            Value to use for newly missing values. If a dict-like, maps
            variable names (including coordinates) to fill values. Use this
            data array's name to refer to the data array's values.

        Returns
        -------
        reindexed : DataArray
            Another dataset array, with this array's data but coordinates from
            the other object.

        See Also
        --------
        DataArray.reindex
        align
        """
        indexers = reindex_like_indexers(self, other)
        return self.reindex(
            indexers=indexers,
            method=method,
            tolerance=tolerance,
            copy=copy,
            fill_value=fill_value,
        )

    def reindex(
        self,
        indexers: Mapping[Hashable, Any] = None,
        method: str = None,
        tolerance=None,
        copy: bool = True,
        fill_value=dtypes.NA,
        **indexers_kwargs: Any,
    ) -> "DataArray":
        """Conform this object onto the indexes of another object, filling in
        missing values with ``fill_value``. The default fill value is NaN.

        Parameters
        ----------
        indexers : dict, optional
            Dictionary with keys given by dimension names and values given by
            arrays of coordinates tick labels. Any mis-matched coordinate
            values will be filled in with NaN, and any mis-matched dimension
            names will simply be ignored.
            One of indexers or indexers_kwargs must be provided.
        copy : bool, optional
            If ``copy=True``, data in the return value is always copied. If
            ``copy=False`` and reindexing is unnecessary, or can be performed
            with only slice operations, then the output may share memory with
            the input. In either case, a new xarray object is always returned.
        method : {None, 'nearest', 'pad'/'ffill', 'backfill'/'bfill'}, optional
            Method to use for filling index values in ``indexers`` not found on
            this data array:

            * None (default): don't fill gaps
            * pad / ffill: propagate last valid index value forward
            * backfill / bfill: propagate next valid index value backward
            * nearest: use nearest valid index value
        tolerance : optional
            Maximum distance between original and new labels for inexact
            matches. The values of the index at the matching locations must
            satisfy the equation ``abs(index[indexer] - target) <= tolerance``.
        fill_value : scalar or dict-like, optional
            Value to use for newly missing values. If a dict-like, maps
            variable names (including coordinates) to fill values. Use this
            data array's name to refer to the data array's values.
        **indexers_kwargs : {dim: indexer, ...}, optional
            The keyword arguments form of ``indexers``.
            One of indexers or indexers_kwargs must be provided.

        Returns
        -------
        reindexed : DataArray
            Another dataset array, with this array's data but replaced
            coordinates.

        See Also
        --------
        DataArray.reindex_like
        align
        """
        indexers = either_dict_or_kwargs(indexers, indexers_kwargs, "reindex")
        if isinstance(fill_value, dict):
            fill_value = fill_value.copy()
            sentinel = object()
            value = fill_value.pop(self.name, sentinel)
            if value is not sentinel:
                fill_value[_THIS_ARRAY] = value

        ds = self._to_temp_dataset().reindex(
            indexers=indexers,
            method=method,
            tolerance=tolerance,
            copy=copy,
            fill_value=fill_value,
        )
        return self._from_temp_dataset(ds)

    def interp(
        self,
        coords: Mapping[Hashable, Any] = None,
        method: str = "linear",
        assume_sorted: bool = False,
        kwargs: Mapping[str, Any] = None,
        **coords_kwargs: Any,
    ) -> "DataArray":
        """Multidimensional interpolation of variables.

        Parameters
        ----------
        coords : dict, optional
            Mapping from dimension names to the new coordinates.
            New coordinate can be an scalar, array-like or DataArray.
            If DataArrays are passed as new coordinates, their dimensions are
            used for the broadcasting. Missing values are skipped.
        method : str, default: "linear"
            The method used to interpolate. Choose from

            - {"linear", "nearest"} for multidimensional array,
            - {"linear", "nearest", "zero", "slinear", "quadratic", "cubic"} for 1-dimensional array.
        assume_sorted : bool, optional
            If False, values of x can be in any order and they are sorted
            first. If True, x has to be an array of monotonically increasing
            values.
        kwargs : dict
            Additional keyword arguments passed to scipy's interpolator. Valid
            options and their behavior depend on if 1-dimensional or
            multi-dimensional interpolation is used.
        **coords_kwargs : {dim: coordinate, ...}, optional
            The keyword arguments form of ``coords``.
            One of coords or coords_kwargs must be provided.

        Returns
        -------
        interpolated : DataArray
            New dataarray on the new coordinates.

        Notes
        -----
        scipy is required.

        See Also
        --------
        scipy.interpolate.interp1d
        scipy.interpolate.interpn

        Examples
        --------
        >>> da = xr.DataArray([1, 3], [("x", np.arange(2))])
        >>> da.interp(x=0.5)
        <xarray.DataArray ()>
        array(2.)
        Coordinates:
            x        float64 0.5
        """
        if self.dtype.kind not in "uifc":
            raise TypeError(
                "interp only works for a numeric type array. "
                "Given {}.".format(self.dtype)
            )
        ds = self._to_temp_dataset().interp(
            coords,
            method=method,
            kwargs=kwargs,
            assume_sorted=assume_sorted,
            **coords_kwargs,
        )
        return self._from_temp_dataset(ds)

    def interp_like(
        self,
        other: Union["DataArray", Dataset],
        method: str = "linear",
        assume_sorted: bool = False,
        kwargs: Mapping[str, Any] = None,
    ) -> "DataArray":
        """Interpolate this object onto the coordinates of another object,
        filling out of range values with NaN.

        Parameters
        ----------
        other : Dataset or DataArray
            Object with an 'indexes' attribute giving a mapping from dimension
            names to an 1d array-like, which provides coordinates upon
            which to index the variables in this dataset. Missing values are skipped.
        method : str, default: "linear"
            The method used to interpolate. Choose from

            - {"linear", "nearest"} for multidimensional array,
            - {"linear", "nearest", "zero", "slinear", "quadratic", "cubic"} for 1-dimensional array.
        assume_sorted : bool, optional
            If False, values of coordinates that are interpolated over can be
            in any order and they are sorted first. If True, interpolated
            coordinates are assumed to be an array of monotonically increasing
            values.
        kwargs : dict, optional
            Additional keyword passed to scipy's interpolator.

        Returns
        -------
        interpolated : DataArray
            Another dataarray by interpolating this dataarray's data along the
            coordinates of the other object.

        Notes
        -----
        scipy is required.
        If the dataarray has object-type coordinates, reindex is used for these
        coordinates instead of the interpolation.

        See Also
        --------
        DataArray.interp
        DataArray.reindex_like
        """
        if self.dtype.kind not in "uifc":
            raise TypeError(
                "interp only works for a numeric type array. "
                "Given {}.".format(self.dtype)
            )
        ds = self._to_temp_dataset().interp_like(
            other, method=method, kwargs=kwargs, assume_sorted=assume_sorted
        )
        return self._from_temp_dataset(ds)

    def rename(
        self,
        new_name_or_name_dict: Union[Hashable, Mapping[Hashable, Hashable]] = None,
        **names: Hashable,
    ) -> "DataArray":
        """Returns a new DataArray with renamed coordinates or a new name.

        Parameters
        ----------
        new_name_or_name_dict : str or dict-like, optional
            If the argument is dict-like, it used as a mapping from old
            names to new names for coordinates. Otherwise, use the argument
            as the new name for this array.
        **names : hashable, optional
            The keyword arguments form of a mapping from old names to
            new names for coordinates.
            One of new_name_or_name_dict or names must be provided.

        Returns
        -------
        renamed : DataArray
            Renamed array or array with renamed coordinates.

        See Also
        --------
        Dataset.rename
        DataArray.swap_dims
        """
        if names or utils.is_dict_like(new_name_or_name_dict):
            new_name_or_name_dict = cast(
                Mapping[Hashable, Hashable], new_name_or_name_dict
            )
            name_dict = either_dict_or_kwargs(new_name_or_name_dict, names, "rename")
            dataset = self._to_temp_dataset().rename(name_dict)
            return self._from_temp_dataset(dataset)
        else:
            new_name_or_name_dict = cast(Hashable, new_name_or_name_dict)
            return self._replace(name=new_name_or_name_dict)

    def swap_dims(self, dims_dict: Mapping[Hashable, Hashable]) -> "DataArray":
        """Returns a new DataArray with swapped dimensions.

        Parameters
        ----------
        dims_dict : dict-like
            Dictionary whose keys are current dimension names and whose values
            are new names.

        Returns
        -------
        swapped : DataArray
            DataArray with swapped dimensions.

        Examples
        --------

        >>> arr = xr.DataArray(
        ...     data=[0, 1],
        ...     dims="x",
        ...     coords={"x": ["a", "b"], "y": ("x", [0, 1])},
        ... )
        >>> arr
        <xarray.DataArray (x: 2)>
        array([0, 1])
        Coordinates:
          * x        (x) <U1 'a' 'b'
            y        (x) int64 0 1

        >>> arr.swap_dims({"x": "y"})
        <xarray.DataArray (y: 2)>
        array([0, 1])
        Coordinates:
            x        (y) <U1 'a' 'b'
          * y        (y) int64 0 1

        >>> arr.swap_dims({"x": "z"})
        <xarray.DataArray (z: 2)>
        array([0, 1])
        Coordinates:
            x        (z) <U1 'a' 'b'
            y        (z) int64 0 1
        Dimensions without coordinates: z

        See Also
        --------

        DataArray.rename
        Dataset.swap_dims
        """
        ds = self._to_temp_dataset().swap_dims(dims_dict)
        return self._from_temp_dataset(ds)

    def expand_dims(
        self,
        dim: Union[None, Hashable, Sequence[Hashable], Mapping[Hashable, Any]] = None,
        axis=None,
        **dim_kwargs: Any,
    ) -> "DataArray":
        """Return a new object with an additional axis (or axes) inserted at
        the corresponding position in the array shape. The new object is a
        view into the underlying array, not a copy.


        If dim is already a scalar coordinate, it will be promoted to a 1D
        coordinate consisting of a single value.

        Parameters
        ----------
        dim : hashable, sequence of hashable, dict, or None, optional
            Dimensions to include on the new variable.
            If provided as str or sequence of str, then dimensions are inserted
            with length 1. If provided as a dict, then the keys are the new
            dimensions and the values are either integers (giving the length of
            the new dimensions) or sequence/ndarray (giving the coordinates of
            the new dimensions).
        axis : int, list of int or tuple of int, or None, default: None
            Axis position(s) where new axis is to be inserted (position(s) on
            the result array). If a list (or tuple) of integers is passed,
            multiple axes are inserted. In this case, dim arguments should be
            same length list. If axis=None is passed, all the axes will be
            inserted to the start of the result array.
        **dim_kwargs : int or sequence or ndarray
            The keywords are arbitrary dimensions being inserted and the values
            are either the lengths of the new dims (if int is given), or their
            coordinates. Note, this is an alternative to passing a dict to the
            dim kwarg and will only be used if dim is None.

        Returns
        -------
        expanded : same type as caller
            This object, but with an additional dimension(s).
        """
        if isinstance(dim, int):
            raise TypeError("dim should be hashable or sequence/mapping of hashables")
        elif isinstance(dim, Sequence) and not isinstance(dim, str):
            if len(dim) != len(set(dim)):
                raise ValueError("dims should not contain duplicate values.")
            dim = dict.fromkeys(dim, 1)
        elif dim is not None and not isinstance(dim, Mapping):
            dim = {cast(Hashable, dim): 1}

        dim = either_dict_or_kwargs(dim, dim_kwargs, "expand_dims")
        ds = self._to_temp_dataset().expand_dims(dim, axis)
        return self._from_temp_dataset(ds)

    def set_index(
        self,
        indexes: Mapping[Hashable, Union[Hashable, Sequence[Hashable]]] = None,
        append: bool = False,
        inplace: bool = None,
        **indexes_kwargs: Union[Hashable, Sequence[Hashable]],
    ) -> Optional["DataArray"]:
        """Set DataArray (multi-)indexes using one or more existing
        coordinates.

        Parameters
        ----------
        indexes : {dim: index, ...}
            Mapping from names matching dimensions and values given
            by (lists of) the names of existing coordinates or variables to set
            as new (multi-)index.
        append : bool, optional
            If True, append the supplied index(es) to the existing index(es).
            Otherwise replace the existing index(es) (default).
        **indexes_kwargs : optional
            The keyword arguments form of ``indexes``.
            One of indexes or indexes_kwargs must be provided.

        Returns
        -------
        obj : DataArray
            Another DataArray, with this data but replaced coordinates.

        Examples
        --------
        >>> arr = xr.DataArray(
        ...     data=np.ones((2, 3)),
        ...     dims=["x", "y"],
        ...     coords={"x": range(2), "y": range(3), "a": ("x", [3, 4])},
        ... )
        >>> arr
        <xarray.DataArray (x: 2, y: 3)>
        array([[1., 1., 1.],
               [1., 1., 1.]])
        Coordinates:
          * x        (x) int64 0 1
          * y        (y) int64 0 1 2
            a        (x) int64 3 4
        >>> arr.set_index(x="a")
        <xarray.DataArray (x: 2, y: 3)>
        array([[1., 1., 1.],
               [1., 1., 1.]])
        Coordinates:
          * x        (x) int64 3 4
          * y        (y) int64 0 1 2

        See Also
        --------
        DataArray.reset_index
        """
        ds = self._to_temp_dataset().set_index(
            indexes, append=append, inplace=inplace, **indexes_kwargs
        )
        return self._from_temp_dataset(ds)

    def reset_index(
        self,
        dims_or_levels: Union[Hashable, Sequence[Hashable]],
        drop: bool = False,
        inplace: bool = None,
    ) -> Optional["DataArray"]:
        """Reset the specified index(es) or multi-index level(s).

        Parameters
        ----------
        dims_or_levels : hashable or sequence of hashable
            Name(s) of the dimension(s) and/or multi-index level(s) that will
            be reset.
        drop : bool, optional
            If True, remove the specified indexes and/or multi-index levels
            instead of extracting them as new coordinates (default: False).

        Returns
        -------
        obj : DataArray
            Another dataarray, with this dataarray's data but replaced
            coordinates.

        See Also
        --------
        DataArray.set_index
        """
        _check_inplace(inplace)
        coords, _ = split_indexes(
            dims_or_levels, self._coords, set(), self._level_coords, drop=drop
        )
        return self._replace(coords=coords)

    def reorder_levels(
        self,
        dim_order: Mapping[Hashable, Sequence[int]] = None,
        inplace: bool = None,
        **dim_order_kwargs: Sequence[int],
    ) -> "DataArray":
        """Rearrange index levels using input order.

        Parameters
        ----------
        dim_order : optional
            Mapping from names matching dimensions and values given
            by lists representing new level orders. Every given dimension
            must have a multi-index.
        **dim_order_kwargs : optional
            The keyword arguments form of ``dim_order``.
            One of dim_order or dim_order_kwargs must be provided.

        Returns
        -------
        obj : DataArray
            Another dataarray, with this dataarray's data but replaced
            coordinates.
        """
        _check_inplace(inplace)
        dim_order = either_dict_or_kwargs(dim_order, dim_order_kwargs, "reorder_levels")
        replace_coords = {}
        for dim, order in dim_order.items():
            coord = self._coords[dim]
            index = coord.to_index()
            if not isinstance(index, pd.MultiIndex):
                raise ValueError("coordinate %r has no MultiIndex" % dim)
            replace_coords[dim] = IndexVariable(coord.dims, index.reorder_levels(order))
        coords = self._coords.copy()
        coords.update(replace_coords)
        return self._replace(coords=coords)

    def stack(
        self,
        dimensions: Mapping[Hashable, Sequence[Hashable]] = None,
        **dimensions_kwargs: Sequence[Hashable],
    ) -> "DataArray":
        """
        Stack any number of existing dimensions into a single new dimension.

        New dimensions will be added at the end, and the corresponding
        coordinate variables will be combined into a MultiIndex.

        Parameters
        ----------
        dimensions : mapping of hashable to sequence of hashable
            Mapping of the form `new_name=(dim1, dim2, ...)`.
            Names of new dimensions, and the existing dimensions that they
            replace. An ellipsis (`...`) will be replaced by all unlisted dimensions.
            Passing a list containing an ellipsis (`stacked_dim=[...]`) will stack over
            all dimensions.
        **dimensions_kwargs
            The keyword arguments form of ``dimensions``.
            One of dimensions or dimensions_kwargs must be provided.

        Returns
        -------
        stacked : DataArray
            DataArray with stacked data.

        Examples
        --------

        >>> arr = xr.DataArray(
        ...     np.arange(6).reshape(2, 3),
        ...     coords=[("x", ["a", "b"]), ("y", [0, 1, 2])],
        ... )
        >>> arr
        <xarray.DataArray (x: 2, y: 3)>
        array([[0, 1, 2],
               [3, 4, 5]])
        Coordinates:
          * x        (x) <U1 'a' 'b'
          * y        (y) int64 0 1 2
        >>> stacked = arr.stack(z=("x", "y"))
        >>> stacked.indexes["z"]
        MultiIndex([('a', 0),
                    ('a', 1),
                    ('a', 2),
                    ('b', 0),
                    ('b', 1),
                    ('b', 2)],
                   names=['x', 'y'])

        See Also
        --------
        DataArray.unstack
        """
        ds = self._to_temp_dataset().stack(dimensions, **dimensions_kwargs)
        return self._from_temp_dataset(ds)

    def unstack(
        self,
        dim: Union[Hashable, Sequence[Hashable], None] = None,
        fill_value: Any = dtypes.NA,
        sparse: bool = False,
    ) -> "DataArray":
        """
        Unstack existing dimensions corresponding to MultiIndexes into
        multiple new dimensions.

        New dimensions will be added at the end.

        Parameters
        ----------
        dim : hashable or sequence of hashable, optional
            Dimension(s) over which to unstack. By default unstacks all
            MultiIndexes.
        fill_value : scalar or dict-like, default: nan
            value to be filled. If a dict-like, maps variable names to
            fill values. Use the data array's name to refer to its
            name. If not provided or if the dict-like does not contain
            all variables, the dtype's NA value will be used.
        sparse : bool, default: False
            use sparse-array if True

        Returns
        -------
        unstacked : DataArray
            Array with unstacked data.

        Examples
        --------

        >>> arr = xr.DataArray(
        ...     np.arange(6).reshape(2, 3),
        ...     coords=[("x", ["a", "b"]), ("y", [0, 1, 2])],
        ... )
        >>> arr
        <xarray.DataArray (x: 2, y: 3)>
        array([[0, 1, 2],
               [3, 4, 5]])
        Coordinates:
          * x        (x) <U1 'a' 'b'
          * y        (y) int64 0 1 2
        >>> stacked = arr.stack(z=("x", "y"))
        >>> stacked.indexes["z"]
        MultiIndex([('a', 0),
                    ('a', 1),
                    ('a', 2),
                    ('b', 0),
                    ('b', 1),
                    ('b', 2)],
                   names=['x', 'y'])
        >>> roundtripped = stacked.unstack()
        >>> arr.identical(roundtripped)
        True

        See Also
        --------
        DataArray.stack
        """
        ds = self._to_temp_dataset().unstack(dim, fill_value, sparse)
        return self._from_temp_dataset(ds)

    def to_unstacked_dataset(self, dim, level=0):
        """Unstack DataArray expanding to Dataset along a given level of a
        stacked coordinate.

        This is the inverse operation of Dataset.to_stacked_array.

        Parameters
        ----------
        dim : str
            Name of existing dimension to unstack
        level : int or str
            The MultiIndex level to expand to a dataset along. Can either be
            the integer index of the level or its name.
        label : int, default: 0
            Label of the level to expand dataset along. Overrides the label
            argument if given.

        Returns
        -------
        unstacked: Dataset

        Examples
        --------
        >>> import xarray as xr
        >>> arr = xr.DataArray(
        ...     np.arange(6).reshape(2, 3),
        ...     coords=[("x", ["a", "b"]), ("y", [0, 1, 2])],
        ... )
        >>> data = xr.Dataset({"a": arr, "b": arr.isel(y=0)})
        >>> data
        <xarray.Dataset>
        Dimensions:  (x: 2, y: 3)
        Coordinates:
          * x        (x) <U1 'a' 'b'
          * y        (y) int64 0 1 2
        Data variables:
            a        (x, y) int64 0 1 2 3 4 5
            b        (x) int64 0 3
        >>> stacked = data.to_stacked_array("z", ["x"])
        >>> stacked.indexes["z"]
        MultiIndex([('a', 0.0),
                    ('a', 1.0),
                    ('a', 2.0),
                    ('b', nan)],
                   names=['variable', 'y'])
        >>> roundtripped = stacked.to_unstacked_dataset(dim="z")
        >>> data.identical(roundtripped)
        True

        See Also
        --------
        Dataset.to_stacked_array
        """

        idx = self.indexes[dim]
        if not isinstance(idx, pd.MultiIndex):
            raise ValueError(f"'{dim}' is not a stacked coordinate")

        level_number = idx._get_level_number(level)
        variables = idx.levels[level_number]
        variable_dim = idx.names[level_number]

        # pull variables out of datarray
        data_dict = {}
        for k in variables:
            data_dict[k] = self.sel({variable_dim: k}, drop=True).squeeze(drop=True)

        # unstacked dataset
        return Dataset(data_dict)

    def transpose(self, *dims: Hashable, transpose_coords: bool = True) -> "DataArray":
        """Return a new DataArray object with transposed dimensions.

        Parameters
        ----------
        *dims : hashable, optional
            By default, reverse the dimensions. Otherwise, reorder the
            dimensions to this order.
        transpose_coords : bool, default: True
            If True, also transpose the coordinates of this DataArray.

        Returns
        -------
        transposed : DataArray
            The returned DataArray's array is transposed.

        Notes
        -----
        This operation returns a view of this array's data. It is
        lazy for dask-backed DataArrays but not for numpy-backed DataArrays
        -- the data will be fully loaded.

        See Also
        --------
        numpy.transpose
        Dataset.transpose
        """
        if dims:
            dims = tuple(utils.infix_dims(dims, self.dims))
        variable = self.variable.transpose(*dims)
        if transpose_coords:
            coords: Dict[Hashable, Variable] = {}
            for name, coord in self.coords.items():
                coord_dims = tuple(dim for dim in dims if dim in coord.dims)
                coords[name] = coord.variable.transpose(*coord_dims)
            return self._replace(variable, coords)
        else:
            return self._replace(variable)

    @property
    def T(self) -> "DataArray":
        return self.transpose()

    def drop_vars(
        self, names: Union[Hashable, Iterable[Hashable]], *, errors: str = "raise"
    ) -> "DataArray":
        """Returns an array with dropped variables.

        Parameters
        ----------
        names : hashable or iterable of hashable
            Name(s) of variables to drop.
        errors: {"raise", "ignore"}, optional
            If 'raise' (default), raises a ValueError error if any of the variable
            passed are not in the dataset. If 'ignore', any given names that are in the
            DataArray are dropped and no error is raised.

        Returns
        -------
        dropped : Dataset
            New Dataset copied from `self` with variables removed.
        """
        ds = self._to_temp_dataset().drop_vars(names, errors=errors)
        return self._from_temp_dataset(ds)

    def drop(
        self,
        labels: Mapping = None,
        dim: Hashable = None,
        *,
        errors: str = "raise",
        **labels_kwargs,
    ) -> "DataArray":
        """Backward compatible method based on `drop_vars` and `drop_sel`

        Using either `drop_vars` or `drop_sel` is encouraged

        See Also
        --------
        DataArray.drop_vars
        DataArray.drop_sel
        """
        ds = self._to_temp_dataset().drop(labels, dim, errors=errors)
        return self._from_temp_dataset(ds)

    def drop_sel(
        self,
        labels: Mapping[Hashable, Any] = None,
        *,
        errors: str = "raise",
        **labels_kwargs,
    ) -> "DataArray":
        """Drop index labels from this DataArray.

        Parameters
        ----------
        labels : mapping of hashable to Any
            Index labels to drop
        errors : {"raise", "ignore"}, optional
            If 'raise' (default), raises a ValueError error if
            any of the index labels passed are not
            in the dataset. If 'ignore', any given labels that are in the
            dataset are dropped and no error is raised.
        **labels_kwargs : {dim: label, ...}, optional
            The keyword arguments form of ``dim`` and ``labels``

        Returns
        -------
        dropped : DataArray
        """
        if labels_kwargs or isinstance(labels, dict):
            labels = either_dict_or_kwargs(labels, labels_kwargs, "drop")

        ds = self._to_temp_dataset().drop_sel(labels, errors=errors)
        return self._from_temp_dataset(ds)

    def dropna(
        self, dim: Hashable, how: str = "any", thresh: int = None
    ) -> "DataArray":
        """Returns a new array with dropped labels for missing values along
        the provided dimension.

        Parameters
        ----------
        dim : hashable
            Dimension along which to drop missing values. Dropping along
            multiple dimensions simultaneously is not yet supported.
        how : {"any", "all"}, optional
            * any : if any NA values are present, drop that label
            * all : if all values are NA, drop that label
        thresh : int, default: None
            If supplied, require this many non-NA values.

        Returns
        -------
        DataArray
        """
        ds = self._to_temp_dataset().dropna(dim, how=how, thresh=thresh)
        return self._from_temp_dataset(ds)

    def fillna(self, value: Any) -> "DataArray":
        """Fill missing values in this object.

        This operation follows the normal broadcasting and alignment rules that
        xarray uses for binary arithmetic, except the result is aligned to this
        object (``join='left'``) instead of aligned to the intersection of
        index coordinates (``join='inner'``).

        Parameters
        ----------
        value : scalar, ndarray or DataArray
            Used to fill all matching missing values in this array. If the
            argument is a DataArray, it is first aligned with (reindexed to)
            this array.

        Returns
        -------
        DataArray
        """
        if utils.is_dict_like(value):
            raise TypeError(
                "cannot provide fill value as a dictionary with "
                "fillna on a DataArray"
            )
        out = ops.fillna(self, value)
        return out

    def interpolate_na(
        self,
        dim: Hashable = None,
        method: str = "linear",
        limit: int = None,
        use_coordinate: Union[bool, str] = True,
        max_gap: Union[
            int, float, str, pd.Timedelta, np.timedelta64, datetime.timedelta
        ] = None,
        keep_attrs: bool = None,
        **kwargs: Any,
    ) -> "DataArray":
        """Fill in NaNs by interpolating according to different methods.

        Parameters
        ----------
        dim : str
            Specifies the dimension along which to interpolate.
        method : str, optional
            String indicating which method to use for interpolation:

            - 'linear': linear interpolation (Default). Additional keyword
              arguments are passed to :py:func:`numpy.interp`
            - 'nearest', 'zero', 'slinear', 'quadratic', 'cubic', 'polynomial':
              are passed to :py:func:`scipy.interpolate.interp1d`. If
              ``method='polynomial'``, the ``order`` keyword argument must also be
              provided.
            - 'barycentric', 'krog', 'pchip', 'spline', 'akima': use their
              respective :py:class:`scipy.interpolate` classes.

        use_coordinate : bool or str, default: True
            Specifies which index to use as the x values in the interpolation
            formulated as `y = f(x)`. If False, values are treated as if
            eqaully-spaced along ``dim``. If True, the IndexVariable `dim` is
            used. If ``use_coordinate`` is a string, it specifies the name of a
            coordinate variariable to use as the index.
        limit : int, default: None
            Maximum number of consecutive NaNs to fill. Must be greater than 0
            or None for no limit. This filling is done regardless of the size of
            the gap in the data. To only interpolate over gaps less than a given length,
            see ``max_gap``.
        max_gap: int, float, str, pandas.Timedelta, numpy.timedelta64, datetime.timedelta, default: None
            Maximum size of gap, a continuous sequence of NaNs, that will be filled.
            Use None for no limit. When interpolating along a datetime64 dimension
            and ``use_coordinate=True``, ``max_gap`` can be one of the following:

            - a string that is valid input for pandas.to_timedelta
            - a :py:class:`numpy.timedelta64` object
            - a :py:class:`pandas.Timedelta` object
            - a :py:class:`datetime.timedelta` object

            Otherwise, ``max_gap`` must be an int or a float. Use of ``max_gap`` with unlabeled
            dimensions has not been implemented yet. Gap length is defined as the difference
            between coordinate values at the first data point after a gap and the last value
            before a gap. For gaps at the beginning (end), gap length is defined as the difference
            between coordinate values at the first (last) valid data point and the first (last) NaN.
            For example, consider::

                <xarray.DataArray (x: 9)>
                array([nan, nan, nan,  1., nan, nan,  4., nan, nan])
                Coordinates:
                  * x        (x) int64 0 1 2 3 4 5 6 7 8

            The gap lengths are 3-0 = 3; 6-3 = 3; and 8-6 = 2 respectively
        keep_attrs : bool, default: True
            If True, the dataarray's attributes (`attrs`) will be copied from
            the original object to the new one.  If False, the new
            object will be returned without attributes.
        kwargs : dict, optional
            parameters passed verbatim to the underlying interpolation function

        Returns
        -------
        interpolated: DataArray
            Filled in DataArray.

        See also
        --------
        numpy.interp
        scipy.interpolate
        """
        from .missing import interp_na

        return interp_na(
            self,
            dim=dim,
            method=method,
            limit=limit,
            use_coordinate=use_coordinate,
            max_gap=max_gap,
            keep_attrs=keep_attrs,
            **kwargs,
        )

    def ffill(self, dim: Hashable, limit: int = None) -> "DataArray":
        """Fill NaN values by propogating values forward

        *Requires bottleneck.*

        Parameters
        ----------
        dim : hashable
            Specifies the dimension along which to propagate values when
            filling.
        limit : int, default: None
            The maximum number of consecutive NaN values to forward fill. In
            other words, if there is a gap with more than this number of
            consecutive NaNs, it will only be partially filled. Must be greater
            than 0 or None for no limit.

        Returns
        -------
        DataArray
        """
        from .missing import ffill

        return ffill(self, dim, limit=limit)

    def bfill(self, dim: Hashable, limit: int = None) -> "DataArray":
        """Fill NaN values by propogating values backward

        *Requires bottleneck.*

        Parameters
        ----------
        dim : str
            Specifies the dimension along which to propagate values when
            filling.
        limit : int, default: None
            The maximum number of consecutive NaN values to backward fill. In
            other words, if there is a gap with more than this number of
            consecutive NaNs, it will only be partially filled. Must be greater
            than 0 or None for no limit.

        Returns
        -------
        DataArray
        """
        from .missing import bfill

        return bfill(self, dim, limit=limit)

    def combine_first(self, other: "DataArray") -> "DataArray":
        """Combine two DataArray objects, with union of coordinates.

        This operation follows the normal broadcasting and alignment rules of
        ``join='outer'``.  Default to non-null values of array calling the
        method.  Use np.nan to fill in vacant cells after alignment.

        Parameters
        ----------
        other : DataArray
            Used to fill all matching missing values in this array.

        Returns
        -------
        DataArray
        """
        return ops.fillna(self, other, join="outer")

    def reduce(
        self,
        func: Callable[..., Any],
        dim: Union[None, Hashable, Sequence[Hashable]] = None,
        axis: Union[None, int, Sequence[int]] = None,
        keep_attrs: bool = None,
        keepdims: bool = False,
        **kwargs: Any,
    ) -> "DataArray":
        """Reduce this array by applying `func` along some dimension(s).

        Parameters
        ----------
        func : callable
            Function which can be called in the form
            `f(x, axis=axis, **kwargs)` to return the result of reducing an
            np.ndarray over an integer valued axis.
        dim : hashable or sequence of hashable, optional
            Dimension(s) over which to apply `func`.
        axis : int or sequence of int, optional
            Axis(es) over which to repeatedly apply `func`. Only one of the
            'dim' and 'axis' arguments can be supplied. If neither are
            supplied, then the reduction is calculated over the flattened array
            (by calling `f(x)` without an axis argument).
        keep_attrs : bool, optional
            If True, the variable's attributes (`attrs`) will be copied from
            the original object to the new one.  If False (default), the new
            object will be returned without attributes.
        keepdims : bool, default: False
            If True, the dimensions which are reduced are left in the result
            as dimensions of size one. Coordinates that use these dimensions
            are removed.
        **kwargs : dict
            Additional keyword arguments passed on to `func`.

        Returns
        -------
        reduced : DataArray
            DataArray with this object's array replaced with an array with
            summarized data and the indicated dimension(s) removed.
        """

        var = self.variable.reduce(func, dim, axis, keep_attrs, keepdims, **kwargs)
        return self._replace_maybe_drop_dims(var)

    def to_pandas(self) -> Union["DataArray", pd.Series, pd.DataFrame]:
        """Convert this array into a pandas object with the same shape.

        The type of the returned object depends on the number of DataArray
        dimensions:

        * 0D -> `xarray.DataArray`
        * 1D -> `pandas.Series`
        * 2D -> `pandas.DataFrame`

        Only works for arrays with 2 or fewer dimensions.

        The DataArray constructor performs the inverse transformation.
        """
        # TODO: consolidate the info about pandas constructors and the
        # attributes that correspond to their indexes into a separate module?
        constructors = {0: lambda x: x, 1: pd.Series, 2: pd.DataFrame}
        try:
            constructor = constructors[self.ndim]
        except KeyError:
            raise ValueError(
                "cannot convert arrays with %s dimensions into "
                "pandas objects" % self.ndim
            )
        indexes = [self.get_index(dim) for dim in self.dims]
        return constructor(self.values, *indexes)

    def to_dataframe(
        self, name: Hashable = None, dim_order: List[Hashable] = None
    ) -> pd.DataFrame:
        """Convert this array and its coordinates into a tidy pandas.DataFrame.

        The DataFrame is indexed by the Cartesian product of index coordinates
        (in the form of a :py:class:`pandas.MultiIndex`).

        Other coordinates are included as columns in the DataFrame.

        Parameters
        ----------
        name
            Name to give to this array (required if unnamed).
        dim_order
            Hierarchical dimension order for the resulting dataframe.
            Array content is transposed to this order and then written out as flat
            vectors in contiguous order, so the last dimension in this list
            will be contiguous in the resulting DataFrame. This has a major
            influence on which operations are efficient on the resulting
            dataframe.

            If provided, must include all dimensions of this DataArray. By default,
            dimensions are sorted according to the DataArray dimensions order.

        Returns
        -------
        result
            DataArray as a pandas DataFrame.

        """
        if name is None:
            name = self.name
        if name is None:
            raise ValueError(
                "cannot convert an unnamed DataArray to a "
                "DataFrame: use the ``name`` parameter"
            )

        # By using a unique name, we can convert a DataArray into a DataFrame
        # even if it shares a name with one of its coordinates.
        # I would normally use unique_name = object() but that results in a
        # dataframe with columns in the wrong order, for reasons I have not
        # been able to debug (possibly a pandas bug?).
        unique_name = "__unique_name_identifier_z98xfz98xugfg73ho__"
        ds = self._to_dataset_whole(name=unique_name)

        if dim_order is None:
            ordered_dims = dict(zip(self.dims, self.shape))
        else:
            ordered_dims = ds._normalize_dim_order(dim_order=dim_order)

        df = ds._to_dataframe(ordered_dims)
        df.columns = [name if c == unique_name else c for c in df.columns]
        return df

    def to_series(self) -> pd.Series:
        """Convert this array into a pandas.Series.

        The Series is indexed by the Cartesian product of index coordinates
        (in the form of a :py:class:`pandas.MultiIndex`).
        """
        index = self.coords.to_index()
        return pd.Series(self.values.reshape(-1), index=index, name=self.name)

    def to_masked_array(self, copy: bool = True) -> np.ma.MaskedArray:
        """Convert this array into a numpy.ma.MaskedArray

        Parameters
        ----------
        copy : bool, default: True
            If True make a copy of the array in the result. If False,
            a MaskedArray view of DataArray.values is returned.

        Returns
        -------
        result : MaskedArray
            Masked where invalid values (nan or inf) occur.
        """
        values = self.values  # only compute lazy arrays once
        isnull = pd.isnull(values)
        return np.ma.MaskedArray(data=values, mask=isnull, copy=copy)

    def to_netcdf(self, *args, **kwargs) -> Union[bytes, "Delayed", None]:
        """Write DataArray contents to a netCDF file.

        All parameters are passed directly to :py:meth:`xarray.Dataset.to_netcdf`.

        Notes
        -----
        Only xarray.Dataset objects can be written to netCDF files, so
        the xarray.DataArray is converted to a xarray.Dataset object
        containing a single variable. If the DataArray has no name, or if the
        name is the same as a coordinate name, then it is given the name
        ``"__xarray_dataarray_variable__"``.

        See Also
        --------
        Dataset.to_netcdf
        """
        from ..backends.api import DATAARRAY_NAME, DATAARRAY_VARIABLE

        if self.name is None:
            # If no name is set then use a generic xarray name
            dataset = self.to_dataset(name=DATAARRAY_VARIABLE)
        elif self.name in self.coords or self.name in self.dims:
            # The name is the same as one of the coords names, which netCDF
            # doesn't support, so rename it but keep track of the old name
            dataset = self.to_dataset(name=DATAARRAY_VARIABLE)
            dataset.attrs[DATAARRAY_NAME] = self.name
        else:
            # No problems with the name - so we're fine!
            dataset = self.to_dataset()

        return dataset.to_netcdf(*args, **kwargs)

    def to_dict(self, data: bool = True) -> dict:
        """
        Convert this xarray.DataArray into a dictionary following xarray
        naming conventions.

        Converts all variables and attributes to native Python objects.
        Useful for converting to json. To avoid datetime incompatibility
        use decode_times=False kwarg in xarrray.open_dataset.

        Parameters
        ----------
        data : bool, optional
            Whether to include the actual data in the dictionary. When set to
            False, returns just the schema.

        See also
        --------
        DataArray.from_dict
        """
        d = self.variable.to_dict(data=data)
        d.update({"coords": {}, "name": self.name})
        for k in self.coords:
            d["coords"][k] = self.coords[k].variable.to_dict(data=data)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "DataArray":
        """
        Convert a dictionary into an xarray.DataArray

        Input dict can take several forms:

        .. code:: python

            d = {"dims": ("t"), "data": x}

            d = {
                "coords": {"t": {"dims": "t", "data": t, "attrs": {"units": "s"}}},
                "attrs": {"title": "air temperature"},
                "dims": "t",
                "data": x,
                "name": "a",
            }

        where "t" is the name of the dimesion, "a" is the name of the array,
        and  x and t are lists, numpy.arrays, or pandas objects.

        Parameters
        ----------
        d : dict
            Mapping with a minimum structure of {"dims": [...], "data": [...]}

        Returns
        -------
        obj : xarray.DataArray

        See also
        --------
        DataArray.to_dict
        Dataset.from_dict
        """
        coords = None
        if "coords" in d:
            try:
                coords = {
                    k: (v["dims"], v["data"], v.get("attrs"))
                    for k, v in d["coords"].items()
                }
            except KeyError as e:
                raise ValueError(
                    "cannot convert dict when coords are missing the key "
                    "'{dims_data}'".format(dims_data=str(e.args[0]))
                )
        try:
            data = d["data"]
        except KeyError:
            raise ValueError("cannot convert dict without the key 'data''")
        else:
            obj = cls(data, coords, d.get("dims"), d.get("name"), d.get("attrs"))
        return obj

    @classmethod
    def from_series(cls, series: pd.Series, sparse: bool = False) -> "DataArray":
        """Convert a pandas.Series into an xarray.DataArray.

        If the series's index is a MultiIndex, it will be expanded into a
        tensor product of one-dimensional coordinates (filling in missing
        values with NaN). Thus this operation should be the inverse of the
        `to_series` method.

        If sparse=True, creates a sparse array instead of a dense NumPy array.
        Requires the pydata/sparse package.

        See also
        --------
        xarray.Dataset.from_dataframe
        """
        temp_name = "__temporary_name"
        df = pd.DataFrame({temp_name: series})
        ds = Dataset.from_dataframe(df, sparse=sparse)
        result = cast(DataArray, ds[temp_name])
        result.name = series.name
        return result

    def to_cdms2(self) -> "cdms2_Variable":
        """Convert this array into a cdms2.Variable"""
        from ..convert import to_cdms2

        return to_cdms2(self)

    @classmethod
    def from_cdms2(cls, variable: "cdms2_Variable") -> "DataArray":
        """Convert a cdms2.Variable into an xarray.DataArray"""
        from ..convert import from_cdms2

        return from_cdms2(variable)

    def to_iris(self) -> "iris_Cube":
        """Convert this array into a iris.cube.Cube"""
        from ..convert import to_iris

        return to_iris(self)

    @classmethod
    def from_iris(cls, cube: "iris_Cube") -> "DataArray":
        """Convert a iris.cube.Cube into an xarray.DataArray"""
        from ..convert import from_iris

        return from_iris(cube)

    def _all_compat(self, other: "DataArray", compat_str: str) -> bool:
        """Helper function for equals, broadcast_equals, and identical"""

        def compat(x, y):
            return getattr(x.variable, compat_str)(y.variable)

        return utils.dict_equiv(self.coords, other.coords, compat=compat) and compat(
            self, other
        )

    def broadcast_equals(self, other: "DataArray") -> bool:
        """Two DataArrays are broadcast equal if they are equal after
        broadcasting them against each other such that they have the same
        dimensions.

        See Also
        --------
        DataArray.equals
        DataArray.identical
        """
        try:
            return self._all_compat(other, "broadcast_equals")
        except (TypeError, AttributeError):
            return False

    def equals(self, other: "DataArray") -> bool:
        """True if two DataArrays have the same dimensions, coordinates and
        values; otherwise False.

        DataArrays can still be equal (like pandas objects) if they have NaN
        values in the same locations.

        This method is necessary because `v1 == v2` for ``DataArray``
        does element-wise comparisons (like numpy.ndarrays).

        See Also
        --------
        DataArray.broadcast_equals
        DataArray.identical
        """
        try:
            return self._all_compat(other, "equals")
        except (TypeError, AttributeError):
            return False

    def identical(self, other: "DataArray") -> bool:
        """Like equals, but also checks the array name and attributes, and
        attributes on all coordinates.

        See Also
        --------
        DataArray.broadcast_equals
        DataArray.equals
        """
        try:
            return self.name == other.name and self._all_compat(other, "identical")
        except (TypeError, AttributeError):
            return False

    def _result_name(self, other: Any = None) -> Optional[Hashable]:
        # use the same naming heuristics as pandas:
        # https://github.com/ContinuumIO/blaze/issues/458#issuecomment-51936356
        other_name = getattr(other, "name", _default)
        if other_name is _default or other_name == self.name:
            return self.name
        else:
            return None

    def __array_wrap__(self, obj, context=None) -> "DataArray":
        new_var = self.variable.__array_wrap__(obj, context)
        return self._replace(new_var)

    def __matmul__(self, obj):
        return self.dot(obj)

    def __rmatmul__(self, other):
        # currently somewhat duplicative, as only other DataArrays are
        # compatible with matmul
        return computation.dot(other, self)

    @staticmethod
    def _unary_op(f: Callable[..., Any]) -> Callable[..., "DataArray"]:
        @functools.wraps(f)
        def func(self, *args, **kwargs):
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", r"All-NaN (slice|axis) encountered")
                warnings.filterwarnings(
                    "ignore", r"Mean of empty slice", category=RuntimeWarning
                )
                with np.errstate(all="ignore"):
                    return self.__array_wrap__(f(self.variable.data, *args, **kwargs))

        return func

    @staticmethod
    def _binary_op(
        f: Callable[..., Any],
        reflexive: bool = False,
        join: str = None,  # see xarray.align
        **ignored_kwargs,
    ) -> Callable[..., "DataArray"]:
        @functools.wraps(f)
        def func(self, other):
            if isinstance(other, (Dataset, groupby.GroupBy)):
                return NotImplemented
            if isinstance(other, DataArray):
                align_type = OPTIONS["arithmetic_join"] if join is None else join
                self, other = align(self, other, join=align_type, copy=False)
            other_variable = getattr(other, "variable", other)
            other_coords = getattr(other, "coords", None)

            variable = (
                f(self.variable, other_variable)
                if not reflexive
                else f(other_variable, self.variable)
            )
            coords, indexes = self.coords._merge_raw(other_coords)
            name = self._result_name(other)

            return self._replace(variable, coords, name, indexes=indexes)

        return func

    @staticmethod
    def _inplace_binary_op(f: Callable) -> Callable[..., "DataArray"]:
        @functools.wraps(f)
        def func(self, other):
            if isinstance(other, groupby.GroupBy):
                raise TypeError(
                    "in-place operations between a DataArray and "
                    "a grouped object are not permitted"
                )
            # n.b. we can't align other to self (with other.reindex_like(self))
            # because `other` may be converted into floats, which would cause
            # in-place arithmetic to fail unpredictably. Instead, we simply
            # don't support automatic alignment with in-place arithmetic.
            other_coords = getattr(other, "coords", None)
            other_variable = getattr(other, "variable", other)
            try:
                with self.coords._merge_inplace(other_coords):
                    f(self.variable, other_variable)
            except MergeError as exc:
                raise MergeError(
                    "Automatic alignment is not supported for in-place operations.\n"
                    "Consider aligning the indices manually or using a not-in-place operation.\n"
                    "See https://github.com/pydata/xarray/issues/3910 for more explanations."
                ) from exc
            return self

        return func

    def _copy_attrs_from(self, other: Union["DataArray", Dataset, Variable]) -> None:
        self.attrs = other.attrs

    plot = utils.UncachedAccessor(_PlotMethods)

    def _title_for_slice(self, truncate: int = 50) -> str:
        """
        If the dataarray has 1 dimensional coordinates or comes from a slice
        we can show that info in the title

        Parameters
        ----------
        truncate : int, default: 50
            maximum number of characters for title

        Returns
        -------
        title : string
            Can be used for plot titles

        """
        one_dims = []
        for dim, coord in self.coords.items():
            if coord.size == 1:
                one_dims.append(
                    "{dim} = {v}".format(dim=dim, v=format_item(coord.values))
                )

        title = ", ".join(one_dims)
        if len(title) > truncate:
            title = title[: (truncate - 3)] + "..."

        return title

    def diff(self, dim: Hashable, n: int = 1, label: Hashable = "upper") -> "DataArray":
        """Calculate the n-th order discrete difference along given axis.

        Parameters
        ----------
        dim : hashable
            Dimension over which to calculate the finite difference.
        n : int, optional
            The number of times values are differenced.
        label : hashable, optional
            The new coordinate in dimension ``dim`` will have the
            values of either the minuend's or subtrahend's coordinate
            for values 'upper' and 'lower', respectively.  Other
            values are not supported.

        Returns
        -------
        difference : same type as caller
            The n-th order finite difference of this object.

        .. note::

            `n` matches numpy's behavior and is different from pandas' first
            argument named `periods`.


        Examples
        --------
        >>> arr = xr.DataArray([5, 5, 6, 6], [[1, 2, 3, 4]], ["x"])
        >>> arr.diff("x")
        <xarray.DataArray (x: 3)>
        array([0, 1, 0])
        Coordinates:
          * x        (x) int64 2 3 4
        >>> arr.diff("x", 2)
        <xarray.DataArray (x: 2)>
        array([ 1, -1])
        Coordinates:
          * x        (x) int64 3 4

        See Also
        --------
        DataArray.differentiate
        """
        ds = self._to_temp_dataset().diff(n=n, dim=dim, label=label)
        return self._from_temp_dataset(ds)

    def shift(
        self,
        shifts: Mapping[Hashable, int] = None,
        fill_value: Any = dtypes.NA,
        **shifts_kwargs: int,
    ) -> "DataArray":
        """Shift this array by an offset along one or more dimensions.

        Only the data is moved; coordinates stay in place. Values shifted from
        beyond array bounds are replaced by NaN. This is consistent with the
        behavior of ``shift`` in pandas.

        Parameters
        ----------
        shifts : mapping of hashable to int, optional
            Integer offset to shift along each of the given dimensions.
            Positive offsets shift to the right; negative offsets shift to the
            left.
        fill_value: scalar, optional
            Value to use for newly missing values
        **shifts_kwargs
            The keyword arguments form of ``shifts``.
            One of shifts or shifts_kwargs must be provided.

        Returns
        -------
        shifted : DataArray
            DataArray with the same coordinates and attributes but shifted
            data.

        See also
        --------
        roll

        Examples
        --------

        >>> arr = xr.DataArray([5, 6, 7], dims="x")
        >>> arr.shift(x=1)
        <xarray.DataArray (x: 3)>
        array([nan,  5.,  6.])
        Dimensions without coordinates: x
        """
        variable = self.variable.shift(
            shifts=shifts, fill_value=fill_value, **shifts_kwargs
        )
        return self._replace(variable=variable)

    def roll(
        self,
        shifts: Mapping[Hashable, int] = None,
        roll_coords: bool = None,
        **shifts_kwargs: int,
    ) -> "DataArray":
        """Roll this array by an offset along one or more dimensions.

        Unlike shift, roll may rotate all variables, including coordinates
        if specified. The direction of rotation is consistent with
        :py:func:`numpy.roll`.

        Parameters
        ----------
        shifts : mapping of hashable to int, optional
            Integer offset to rotate each of the given dimensions.
            Positive offsets roll to the right; negative offsets roll to the
            left.
        roll_coords : bool
            Indicates whether to  roll the coordinates by the offset
            The current default of roll_coords (None, equivalent to True) is
            deprecated and will change to False in a future version.
            Explicitly pass roll_coords to silence the warning.
        **shifts_kwargs
            The keyword arguments form of ``shifts``.
            One of shifts or shifts_kwargs must be provided.

        Returns
        -------
        rolled : DataArray
            DataArray with the same attributes but rolled data and coordinates.

        See also
        --------
        shift

        Examples
        --------

        >>> arr = xr.DataArray([5, 6, 7], dims="x")
        >>> arr.roll(x=1)
        <xarray.DataArray (x: 3)>
        array([7, 5, 6])
        Dimensions without coordinates: x
        """
        ds = self._to_temp_dataset().roll(
            shifts=shifts, roll_coords=roll_coords, **shifts_kwargs
        )
        return self._from_temp_dataset(ds)

    @property
    def real(self) -> "DataArray":
        return self._replace(self.variable.real)

    @property
    def imag(self) -> "DataArray":
        return self._replace(self.variable.imag)

    def dot(
        self, other: "DataArray", dims: Union[Hashable, Sequence[Hashable], None] = None
    ) -> "DataArray":
        """Perform dot product of two DataArrays along their shared dims.

        Equivalent to taking taking tensordot over all shared dims.

        Parameters
        ----------
        other : DataArray
            The other array with which the dot product is performed.
        dims : ..., hashable or sequence of hashable, optional
            Which dimensions to sum over. Ellipsis (`...`) sums over all dimensions.
            If not specified, then all the common dimensions are summed over.

        Returns
        -------
        result : DataArray
            Array resulting from the dot product over all shared dimensions.

        See also
        --------
        dot
        numpy.tensordot

        Examples
        --------

        >>> da_vals = np.arange(6 * 5 * 4).reshape((6, 5, 4))
        >>> da = xr.DataArray(da_vals, dims=["x", "y", "z"])
        >>> dm_vals = np.arange(4)
        >>> dm = xr.DataArray(dm_vals, dims=["z"])

        >>> dm.dims
        ('z',)

        >>> da.dims
        ('x', 'y', 'z')

        >>> dot_result = da.dot(dm)
        >>> dot_result.dims
        ('x', 'y')

        """
        if isinstance(other, Dataset):
            raise NotImplementedError(
                "dot products are not yet supported with Dataset objects."
            )
        if not isinstance(other, DataArray):
            raise TypeError("dot only operates on DataArrays.")

        return computation.dot(self, other, dims=dims)

    def sortby(
        self,
        variables: Union[Hashable, "DataArray", Sequence[Union[Hashable, "DataArray"]]],
        ascending: bool = True,
    ) -> "DataArray":
        """Sort object by labels or values (along an axis).

        Sorts the dataarray, either along specified dimensions,
        or according to values of 1-D dataarrays that share dimension
        with calling object.

        If the input variables are dataarrays, then the dataarrays are aligned
        (via left-join) to the calling object prior to sorting by cell values.
        NaNs are sorted to the end, following Numpy convention.

        If multiple sorts along the same dimension is
        given, numpy's lexsort is performed along that dimension:
        https://docs.scipy.org/doc/numpy/reference/generated/numpy.lexsort.html
        and the FIRST key in the sequence is used as the primary sort key,
        followed by the 2nd key, etc.

        Parameters
        ----------
        variables : hashable, DataArray, or sequence of hashable or DataArray
            1D DataArray objects or name(s) of 1D variable(s) in
            coords whose values are used to sort this array.
        ascending : bool, optional
            Whether to sort by ascending or descending order.

        Returns
        -------
        sorted : DataArray
            A new dataarray where all the specified dims are sorted by dim
            labels.

        Examples
        --------

        >>> da = xr.DataArray(
        ...     np.random.rand(5),
        ...     coords=[pd.date_range("1/1/2000", periods=5)],
        ...     dims="time",
        ... )
        >>> da
        <xarray.DataArray (time: 5)>
        array([0.5488135 , 0.71518937, 0.60276338, 0.54488318, 0.4236548 ])
        Coordinates:
          * time     (time) datetime64[ns] 2000-01-01 2000-01-02 ... 2000-01-05

        >>> da.sortby(da)
        <xarray.DataArray (time: 5)>
        array([0.4236548 , 0.54488318, 0.5488135 , 0.60276338, 0.71518937])
        Coordinates:
          * time     (time) datetime64[ns] 2000-01-05 2000-01-04 ... 2000-01-02
        """
        ds = self._to_temp_dataset().sortby(variables, ascending=ascending)
        return self._from_temp_dataset(ds)

    def quantile(
        self,
        q: Any,
        dim: Union[Hashable, Sequence[Hashable], None] = None,
        interpolation: str = "linear",
        keep_attrs: bool = None,
        skipna: bool = True,
    ) -> "DataArray":
        """Compute the qth quantile of the data along the specified dimension.

        Returns the qth quantiles(s) of the array elements.

        Parameters
        ----------
        q : float or array-like of float
            Quantile to compute, which must be between 0 and 1 inclusive.
        dim : hashable or sequence of hashable, optional
            Dimension(s) over which to apply quantile.
        interpolation : {"linear", "lower", "higher", "midpoint", "nearest"}, default: "linear"
            This optional parameter specifies the interpolation method to
            use when the desired quantile lies between two data points
            ``i < j``:

                - linear: ``i + (j - i) * fraction``, where ``fraction`` is
                  the fractional part of the index surrounded by ``i`` and
                  ``j``.
                - lower: ``i``.
                - higher: ``j``.
                - nearest: ``i`` or ``j``, whichever is nearest.
                - midpoint: ``(i + j) / 2``.
        keep_attrs : bool, optional
            If True, the dataset's attributes (`attrs`) will be copied from
            the original object to the new one.  If False (default), the new
            object will be returned without attributes.
        skipna : bool, optional
            Whether to skip missing values when aggregating.

        Returns
        -------
        quantiles : DataArray
            If `q` is a single quantile, then the result
            is a scalar. If multiple percentiles are given, first axis of
            the result corresponds to the quantile and a quantile dimension
            is added to the return array. The other dimensions are the
            dimensions that remain after the reduction of the array.

        See Also
        --------
        numpy.nanquantile, numpy.quantile, pandas.Series.quantile, Dataset.quantile

        Examples
        --------

        >>> da = xr.DataArray(
        ...     data=[[0.7, 4.2, 9.4, 1.5], [6.5, 7.3, 2.6, 1.9]],
        ...     coords={"x": [7, 9], "y": [1, 1.5, 2, 2.5]},
        ...     dims=("x", "y"),
        ... )
        >>> da.quantile(0)  # or da.quantile(0, dim=...)
        <xarray.DataArray ()>
        array(0.7)
        Coordinates:
            quantile  float64 0.0
        >>> da.quantile(0, dim="x")
        <xarray.DataArray (y: 4)>
        array([0.7, 4.2, 2.6, 1.5])
        Coordinates:
          * y         (y) float64 1.0 1.5 2.0 2.5
            quantile  float64 0.0
        >>> da.quantile([0, 0.5, 1])
        <xarray.DataArray (quantile: 3)>
        array([0.7, 3.4, 9.4])
        Coordinates:
          * quantile  (quantile) float64 0.0 0.5 1.0
        >>> da.quantile([0, 0.5, 1], dim="x")
        <xarray.DataArray (quantile: 3, y: 4)>
        array([[0.7 , 4.2 , 2.6 , 1.5 ],
               [3.6 , 5.75, 6.  , 1.7 ],
               [6.5 , 7.3 , 9.4 , 1.9 ]])
        Coordinates:
          * y         (y) float64 1.0 1.5 2.0 2.5
          * quantile  (quantile) float64 0.0 0.5 1.0
        """

        ds = self._to_temp_dataset().quantile(
            q,
            dim=dim,
            keep_attrs=keep_attrs,
            interpolation=interpolation,
            skipna=skipna,
        )
        return self._from_temp_dataset(ds)

    def rank(
        self, dim: Hashable, pct: bool = False, keep_attrs: bool = None
    ) -> "DataArray":
        """Ranks the data.

        Equal values are assigned a rank that is the average of the ranks that
        would have been otherwise assigned to all of the values within that
        set.  Ranks begin at 1, not 0. If pct, computes percentage ranks.

        NaNs in the input array are returned as NaNs.

        The `bottleneck` library is required.

        Parameters
        ----------
        dim : hashable
            Dimension over which to compute rank.
        pct : bool, optional
            If True, compute percentage ranks, otherwise compute integer ranks.
        keep_attrs : bool, optional
            If True, the dataset's attributes (`attrs`) will be copied from
            the original object to the new one.  If False (default), the new
            object will be returned without attributes.

        Returns
        -------
        ranked : DataArray
            DataArray with the same coordinates and dtype 'float64'.

        Examples
        --------

        >>> arr = xr.DataArray([5, 6, 7], dims="x")
        >>> arr.rank("x")
        <xarray.DataArray (x: 3)>
        array([1., 2., 3.])
        Dimensions without coordinates: x
        """

        ds = self._to_temp_dataset().rank(dim, pct=pct, keep_attrs=keep_attrs)
        return self._from_temp_dataset(ds)

    def differentiate(
        self, coord: Hashable, edge_order: int = 1, datetime_unit: str = None
    ) -> "DataArray":
        """ Differentiate the array with the second order accurate central
        differences.

        .. note::
            This feature is limited to simple cartesian geometry, i.e. coord
            must be one dimensional.

        Parameters
        ----------
        coord : hashable
            The coordinate to be used to compute the gradient.
        edge_order : {1, 2}, default: 1
            N-th order accurate differences at the boundaries.
        datetime_unit : {"Y", "M", "W", "D", "h", "m", "s", "ms", \
                         "us", "ns", "ps", "fs", "as"} or None, optional
            Unit to compute gradient. Only valid for datetime coordinate.

        Returns
        -------
        differentiated: DataArray

        See also
        --------
        numpy.gradient: corresponding numpy function

        Examples
        --------

        >>> da = xr.DataArray(
        ...     np.arange(12).reshape(4, 3),
        ...     dims=["x", "y"],
        ...     coords={"x": [0, 0.1, 1.1, 1.2]},
        ... )
        >>> da
        <xarray.DataArray (x: 4, y: 3)>
        array([[ 0,  1,  2],
               [ 3,  4,  5],
               [ 6,  7,  8],
               [ 9, 10, 11]])
        Coordinates:
          * x        (x) float64 0.0 0.1 1.1 1.2
        Dimensions without coordinates: y
        >>>
        >>> da.differentiate("x")
        <xarray.DataArray (x: 4, y: 3)>
        array([[30.        , 30.        , 30.        ],
               [27.54545455, 27.54545455, 27.54545455],
               [27.54545455, 27.54545455, 27.54545455],
               [30.        , 30.        , 30.        ]])
        Coordinates:
          * x        (x) float64 0.0 0.1 1.1 1.2
        Dimensions without coordinates: y
        """
        ds = self._to_temp_dataset().differentiate(coord, edge_order, datetime_unit)
        return self._from_temp_dataset(ds)

    def integrate(
        self, dim: Union[Hashable, Sequence[Hashable]], datetime_unit: str = None
    ) -> "DataArray":
        """ integrate the array with the trapezoidal rule.

        .. note::
            This feature is limited to simple cartesian geometry, i.e. dim
            must be one dimensional.

        Parameters
        ----------
        dim : hashable, or sequence of hashable
            Coordinate(s) used for the integration.
        datetime_unit : {"Y", "M", "W", "D", "h", "m", "s", "ms", "us", "ns", \
                         "ps", "fs", "as"}, optional
            Can be used to specify the unit if datetime coordinate is used.

        Returns
        -------
        integrated: DataArray

        See also
        --------
        numpy.trapz: corresponding numpy function

        Examples
        --------

        >>> da = xr.DataArray(
        ...     np.arange(12).reshape(4, 3),
        ...     dims=["x", "y"],
        ...     coords={"x": [0, 0.1, 1.1, 1.2]},
        ... )
        >>> da
        <xarray.DataArray (x: 4, y: 3)>
        array([[ 0,  1,  2],
               [ 3,  4,  5],
               [ 6,  7,  8],
               [ 9, 10, 11]])
        Coordinates:
          * x        (x) float64 0.0 0.1 1.1 1.2
        Dimensions without coordinates: y
        >>>
        >>> da.integrate("x")
        <xarray.DataArray (y: 3)>
        array([5.4, 6.6, 7.8])
        Dimensions without coordinates: y
        """
        ds = self._to_temp_dataset().integrate(dim, datetime_unit)
        return self._from_temp_dataset(ds)

    def unify_chunks(self) -> "DataArray":
        """Unify chunk size along all chunked dimensions of this DataArray.

        Returns
        -------

        DataArray with consistent chunk sizes for all dask-array variables

        See Also
        --------

        dask.array.core.unify_chunks
        """
        ds = self._to_temp_dataset().unify_chunks()
        return self._from_temp_dataset(ds)

    def map_blocks(
        self,
        func: "Callable[..., T_DSorDA]",
        args: Sequence[Any] = (),
        kwargs: Mapping[str, Any] = None,
        template: Union["DataArray", "Dataset"] = None,
    ) -> "T_DSorDA":
        """
        Apply a function to each block of this DataArray.

        .. warning::
            This method is experimental and its signature may change.

        Parameters
        ----------
        func : callable
            User-provided function that accepts a DataArray as its first
            parameter. The function will receive a subset or 'block' of this DataArray (see below),
            corresponding to one chunk along each chunked dimension. ``func`` will be
            executed as ``func(subset_dataarray, *subset_args, **kwargs)``.

            This function must return either a single DataArray or a single Dataset.

            This function cannot add a new chunked dimension.
        args : sequence
            Passed to func after unpacking and subsetting any xarray objects by blocks.
            xarray objects in args must be aligned with this object, otherwise an error is raised.
        kwargs : mapping
            Passed verbatim to func after unpacking. xarray objects, if any, will not be
            subset to blocks. Passing dask collections in kwargs is not allowed.
        template : DataArray or Dataset, optional
            xarray object representing the final result after compute is called. If not provided,
            the function will be first run on mocked-up data, that looks like this object but
            has sizes 0, to determine properties of the returned object such as dtype,
            variable names, attributes, new dimensions and new indexes (if any).
            ``template`` must be provided if the function changes the size of existing dimensions.
            When provided, ``attrs`` on variables in `template` are copied over to the result. Any
            ``attrs`` set by ``func`` will be ignored.

        Returns
        -------
        A single DataArray or Dataset with dask backend, reassembled from the outputs of the
        function.

        Notes
        -----
        This function is designed for when ``func`` needs to manipulate a whole xarray object
        subset to each block. In the more common case where ``func`` can work on numpy arrays, it is
        recommended to use ``apply_ufunc``.

        If none of the variables in this object is backed by dask arrays, calling this function is
        equivalent to calling ``func(obj, *args, **kwargs)``.

        See Also
        --------
        dask.array.map_blocks, xarray.apply_ufunc, xarray.Dataset.map_blocks,
        xarray.DataArray.map_blocks

        Examples
        --------

        Calculate an anomaly from climatology using ``.groupby()``. Using
        ``xr.map_blocks()`` allows for parallel operations with knowledge of ``xarray``,
        its indices, and its methods like ``.groupby()``.

        >>> def calculate_anomaly(da, groupby_type="time.month"):
        ...     gb = da.groupby(groupby_type)
        ...     clim = gb.mean(dim="time")
        ...     return gb - clim
        >>> time = xr.cftime_range("1990-01", "1992-01", freq="M")
        >>> month = xr.DataArray(time.month, coords={"time": time}, dims=["time"])
        >>> np.random.seed(123)
        >>> array = xr.DataArray(
        ...     np.random.rand(len(time)),
        ...     dims=["time"],
        ...     coords={"time": time, "month": month},
        ... ).chunk()
        >>> array.map_blocks(calculate_anomaly, template=array).compute()
        <xarray.DataArray (time: 24)>
        array([ 0.12894847,  0.11323072, -0.0855964 , -0.09334032,  0.26848862,
                0.12382735,  0.22460641,  0.07650108, -0.07673453, -0.22865714,
               -0.19063865,  0.0590131 , -0.12894847, -0.11323072,  0.0855964 ,
                0.09334032, -0.26848862, -0.12382735, -0.22460641, -0.07650108,
                0.07673453,  0.22865714,  0.19063865, -0.0590131 ])
        Coordinates:
          * time     (time) object 1990-01-31 00:00:00 ... 1991-12-31 00:00:00
            month    (time) int64 1 2 3 4 5 6 7 8 9 10 11 12 1 2 3 4 5 6 7 8 9 10 11 12

        Note that one must explicitly use ``args=[]`` and ``kwargs={}`` to pass arguments
        to the function being applied in ``xr.map_blocks()``:

        >>> array.map_blocks(
        ...     calculate_anomaly, kwargs={"groupby_type": "time.year"}, template=array
        ... )  # doctest: +ELLIPSIS
        <xarray.DataArray (time: 24)>
        dask.array<calculate_anomaly-...-<this, shape=(24,), dtype=float64, chunksize=(24,), chunktype=numpy.ndarray>
        Coordinates:
          * time     (time) object 1990-01-31 00:00:00 ... 1991-12-31 00:00:00
            month    (time) int64 dask.array<chunksize=(24,), meta=np.ndarray>
        """
        from .parallel import map_blocks

        return map_blocks(func, self, args, kwargs, template)

    def polyfit(
        self,
        dim: Hashable,
        deg: int,
        skipna: bool = None,
        rcond: float = None,
        w: Union[Hashable, Any] = None,
        full: bool = False,
        cov: bool = False,
    ):
        """
        Least squares polynomial fit.

        This replicates the behaviour of `numpy.polyfit` but differs by skipping
        invalid values when `skipna = True`.

        Parameters
        ----------
        dim : hashable
            Coordinate along which to fit the polynomials.
        deg : int
            Degree of the fitting polynomial.
        skipna : bool, optional
            If True, removes all invalid values before fitting each 1D slices of the array.
            Default is True if data is stored in a dask.array or if there is any
            invalid values, False otherwise.
        rcond : float, optional
            Relative condition number to the fit.
        w : hashable or array-like, optional
            Weights to apply to the y-coordinate of the sample points.
            Can be an array-like object or the name of a coordinate in the dataset.
        full : bool, optional
            Whether to return the residuals, matrix rank and singular values in addition
            to the coefficients.
        cov : bool or str, optional
            Whether to return to the covariance matrix in addition to the coefficients.
            The matrix is not scaled if `cov='unscaled'`.

        Returns
        -------
        polyfit_results : Dataset
            A single dataset which contains:

            polyfit_coefficients
                The coefficients of the best fit.
            polyfit_residuals
                The residuals of the least-square computation (only included if `full=True`).
                When the matrix rank is deficient, np.nan is returned.
            [dim]_matrix_rank
                The effective rank of the scaled Vandermonde coefficient matrix (only included if `full=True`)
            [dim]_singular_value
                The singular values of the scaled Vandermonde coefficient matrix (only included if `full=True`)
            polyfit_covariance
                The covariance matrix of the polynomial coefficient estimates (only included if `full=False` and `cov=True`)

        See also
        --------
        numpy.polyfit
        """
        return self._to_temp_dataset().polyfit(
            dim, deg, skipna=skipna, rcond=rcond, w=w, full=full, cov=cov
        )

    def pad(
        self,
        pad_width: Mapping[Hashable, Union[int, Tuple[int, int]]] = None,
        mode: str = "constant",
        stat_length: Union[
            int, Tuple[int, int], Mapping[Hashable, Tuple[int, int]]
        ] = None,
        constant_values: Union[
            int, Tuple[int, int], Mapping[Hashable, Tuple[int, int]]
        ] = None,
        end_values: Union[
            int, Tuple[int, int], Mapping[Hashable, Tuple[int, int]]
        ] = None,
        reflect_type: str = None,
        **pad_width_kwargs: Any,
    ) -> "DataArray":
        """Pad this array along one or more dimensions.

        .. warning::
            This function is experimental and its behaviour is likely to change
            especially regarding padding of dimension coordinates (or IndexVariables).

        When using one of the modes ("edge", "reflect", "symmetric", "wrap"),
        coordinates will be padded with the same mode, otherwise coordinates
        are padded using the "constant" mode with fill_value dtypes.NA.

        Parameters
        ----------
        pad_width : mapping of hashable to tuple of int
            Mapping with the form of {dim: (pad_before, pad_after)}
            describing the number of values padded along each dimension.
            {dim: pad} is a shortcut for pad_before = pad_after = pad
        mode : str, default: "constant"
            One of the following string values (taken from numpy docs)

            'constant' (default)
                Pads with a constant value.
            'edge'
                Pads with the edge values of array.
            'linear_ramp'
                Pads with the linear ramp between end_value and the
                array edge value.
            'maximum'
                Pads with the maximum value of all or part of the
                vector along each axis.
            'mean'
                Pads with the mean value of all or part of the
                vector along each axis.
            'median'
                Pads with the median value of all or part of the
                vector along each axis.
            'minimum'
                Pads with the minimum value of all or part of the
                vector along each axis.
            'reflect'
                Pads with the reflection of the vector mirrored on
                the first and last values of the vector along each
                axis.
            'symmetric'
                Pads with the reflection of the vector mirrored
                along the edge of the array.
            'wrap'
                Pads with the wrap of the vector along the axis.
                The first values are used to pad the end and the
                end values are used to pad the beginning.
        stat_length : int, tuple or mapping of hashable to tuple, default: None
            Used in 'maximum', 'mean', 'median', and 'minimum'.  Number of
            values at edge of each axis used to calculate the statistic value.
            {dim_1: (before_1, after_1), ... dim_N: (before_N, after_N)} unique
            statistic lengths along each dimension.
            ((before, after),) yields same before and after statistic lengths
            for each dimension.
            (stat_length,) or int is a shortcut for before = after = statistic
            length for all axes.
            Default is ``None``, to use the entire axis.
        constant_values : scalar, tuple or mapping of hashable to tuple, default: 0
            Used in 'constant'.  The values to set the padded values for each
            axis.
            ``{dim_1: (before_1, after_1), ... dim_N: (before_N, after_N)}`` unique
            pad constants along each dimension.
            ``((before, after),)`` yields same before and after constants for each
            dimension.
            ``(constant,)`` or ``constant`` is a shortcut for ``before = after = constant`` for
            all dimensions.
            Default is 0.
        end_values : scalar, tuple or mapping of hashable to tuple, default: 0
            Used in 'linear_ramp'.  The values used for the ending value of the
            linear_ramp and that will form the edge of the padded array.
            ``{dim_1: (before_1, after_1), ... dim_N: (before_N, after_N)}`` unique
            end values along each dimension.
            ``((before, after),)`` yields same before and after end values for each
            axis.
            ``(constant,)`` or ``constant`` is a shortcut for ``before = after = constant`` for
            all axes.
            Default is 0.
        reflect_type : {"even", "odd"}, optional
            Used in "reflect", and "symmetric".  The "even" style is the
            default with an unaltered reflection around the edge value.  For
            the "odd" style, the extended part of the array is created by
            subtracting the reflected values from two times the edge value.
        **pad_width_kwargs
            The keyword arguments form of ``pad_width``.
            One of ``pad_width`` or ``pad_width_kwargs`` must be provided.

        Returns
        -------
        padded : DataArray
            DataArray with the padded coordinates and data.

        See also
        --------
        DataArray.shift, DataArray.roll, DataArray.bfill, DataArray.ffill, numpy.pad, dask.array.pad

        Notes
        -----
        By default when ``mode="constant"`` and ``constant_values=None``, integer types will be
        promoted to ``float`` and padded with ``np.nan``. To avoid type promotion
        specify ``constant_values=np.nan``

        Examples
        --------

        >>> arr = xr.DataArray([5, 6, 7], coords=[("x", [0, 1, 2])])
        >>> arr.pad(x=(1, 2), constant_values=0)
        <xarray.DataArray (x: 6)>
        array([0, 5, 6, 7, 0, 0])
        Coordinates:
          * x        (x) float64 nan 0.0 1.0 2.0 nan nan

        >>> da = xr.DataArray(
        ...     [[0, 1, 2, 3], [10, 11, 12, 13]],
        ...     dims=["x", "y"],
        ...     coords={"x": [0, 1], "y": [10, 20, 30, 40], "z": ("x", [100, 200])},
        ... )
        >>> da.pad(x=1)
        <xarray.DataArray (x: 4, y: 4)>
        array([[nan, nan, nan, nan],
               [ 0.,  1.,  2.,  3.],
               [10., 11., 12., 13.],
               [nan, nan, nan, nan]])
        Coordinates:
          * x        (x) float64 nan 0.0 1.0 nan
          * y        (y) int64 10 20 30 40
            z        (x) float64 nan 100.0 200.0 nan
        >>> da.pad(x=1, constant_values=np.nan)
        <xarray.DataArray (x: 4, y: 4)>
        array([[-9223372036854775808, -9223372036854775808, -9223372036854775808,
                -9223372036854775808],
               [                   0,                    1,                    2,
                                   3],
               [                  10,                   11,                   12,
                                  13],
               [-9223372036854775808, -9223372036854775808, -9223372036854775808,
                -9223372036854775808]])
        Coordinates:
          * x        (x) float64 nan 0.0 1.0 nan
          * y        (y) int64 10 20 30 40
            z        (x) float64 nan 100.0 200.0 nan
        """
        ds = self._to_temp_dataset().pad(
            pad_width=pad_width,
            mode=mode,
            stat_length=stat_length,
            constant_values=constant_values,
            end_values=end_values,
            reflect_type=reflect_type,
            **pad_width_kwargs,
        )
        return self._from_temp_dataset(ds)

    def idxmin(
        self,
        dim: Hashable = None,
        skipna: bool = None,
        fill_value: Any = dtypes.NA,
        keep_attrs: bool = None,
    ) -> "DataArray":
        """Return the coordinate label of the minimum value along a dimension.

        Returns a new `DataArray` named after the dimension with the values of
        the coordinate labels along that dimension corresponding to minimum
        values along that dimension.

        In comparison to :py:meth:`~DataArray.argmin`, this returns the
        coordinate label while :py:meth:`~DataArray.argmin` returns the index.

        Parameters
        ----------
        dim : str, optional
            Dimension over which to apply `idxmin`.  This is optional for 1D
            arrays, but required for arrays with 2 or more dimensions.
        skipna : bool or None, default: None
            If True, skip missing values (as marked by NaN). By default, only
            skips missing values for ``float``, ``complex``, and ``object``
            dtypes; other dtypes either do not have a sentinel missing value
            (``int``) or ``skipna=True`` has not been implemented
            (``datetime64`` or ``timedelta64``).
        fill_value : Any, default: NaN
            Value to be filled in case all of the values along a dimension are
            null.  By default this is NaN.  The fill value and result are
            automatically converted to a compatible dtype if possible.
            Ignored if ``skipna`` is False.
        keep_attrs : bool, default: False
            If True, the attributes (``attrs``) will be copied from the
            original object to the new one.  If False (default), the new object
            will be returned without attributes.

        Returns
        -------
        reduced : DataArray
            New `DataArray` object with `idxmin` applied to its data and the
            indicated dimension removed.

        See also
        --------
        Dataset.idxmin, DataArray.idxmax, DataArray.min, DataArray.argmin

        Examples
        --------

        >>> array = xr.DataArray(
        ...     [0, 2, 1, 0, -2], dims="x", coords={"x": ["a", "b", "c", "d", "e"]}
        ... )
        >>> array.min()
        <xarray.DataArray ()>
        array(-2)
        >>> array.argmin()
        <xarray.DataArray ()>
        array(4)
        >>> array.idxmin()
        <xarray.DataArray 'x' ()>
        array('e', dtype='<U1')

        >>> array = xr.DataArray(
        ...     [
        ...         [2.0, 1.0, 2.0, 0.0, -2.0],
        ...         [-4.0, np.NaN, 2.0, np.NaN, -2.0],
        ...         [np.NaN, np.NaN, 1.0, np.NaN, np.NaN],
        ...     ],
        ...     dims=["y", "x"],
        ...     coords={"y": [-1, 0, 1], "x": np.arange(5.0) ** 2},
        ... )
        >>> array.min(dim="x")
        <xarray.DataArray (y: 3)>
        array([-2., -4.,  1.])
        Coordinates:
          * y        (y) int64 -1 0 1
        >>> array.argmin(dim="x")
        <xarray.DataArray (y: 3)>
        array([4, 0, 2])
        Coordinates:
          * y        (y) int64 -1 0 1
        >>> array.idxmin(dim="x")
        <xarray.DataArray 'x' (y: 3)>
        array([16.,  0.,  4.])
        Coordinates:
          * y        (y) int64 -1 0 1
        """
        return computation._calc_idxminmax(
            array=self,
            func=lambda x, *args, **kwargs: x.argmin(*args, **kwargs),
            dim=dim,
            skipna=skipna,
            fill_value=fill_value,
            keep_attrs=keep_attrs,
        )

    def idxmax(
        self,
        dim: Hashable = None,
        skipna: bool = None,
        fill_value: Any = dtypes.NA,
        keep_attrs: bool = None,
    ) -> "DataArray":
        """Return the coordinate label of the maximum value along a dimension.

        Returns a new `DataArray` named after the dimension with the values of
        the coordinate labels along that dimension corresponding to maximum
        values along that dimension.

        In comparison to :py:meth:`~DataArray.argmax`, this returns the
        coordinate label while :py:meth:`~DataArray.argmax` returns the index.

        Parameters
        ----------
        dim : hashable, optional
            Dimension over which to apply `idxmax`.  This is optional for 1D
            arrays, but required for arrays with 2 or more dimensions.
        skipna : bool or None, default: None
            If True, skip missing values (as marked by NaN). By default, only
            skips missing values for ``float``, ``complex``, and ``object``
            dtypes; other dtypes either do not have a sentinel missing value
            (``int``) or ``skipna=True`` has not been implemented
            (``datetime64`` or ``timedelta64``).
        fill_value : Any, default: NaN
            Value to be filled in case all of the values along a dimension are
            null.  By default this is NaN.  The fill value and result are
            automatically converted to a compatible dtype if possible.
            Ignored if ``skipna`` is False.
        keep_attrs : bool, default: False
            If True, the attributes (``attrs``) will be copied from the
            original object to the new one.  If False (default), the new object
            will be returned without attributes.

        Returns
        -------
        reduced : DataArray
            New `DataArray` object with `idxmax` applied to its data and the
            indicated dimension removed.

        See also
        --------
        Dataset.idxmax, DataArray.idxmin, DataArray.max, DataArray.argmax

        Examples
        --------

        >>> array = xr.DataArray(
        ...     [0, 2, 1, 0, -2], dims="x", coords={"x": ["a", "b", "c", "d", "e"]}
        ... )
        >>> array.max()
        <xarray.DataArray ()>
        array(2)
        >>> array.argmax()
        <xarray.DataArray ()>
        array(1)
        >>> array.idxmax()
        <xarray.DataArray 'x' ()>
        array('b', dtype='<U1')

        >>> array = xr.DataArray(
        ...     [
        ...         [2.0, 1.0, 2.0, 0.0, -2.0],
        ...         [-4.0, np.NaN, 2.0, np.NaN, -2.0],
        ...         [np.NaN, np.NaN, 1.0, np.NaN, np.NaN],
        ...     ],
        ...     dims=["y", "x"],
        ...     coords={"y": [-1, 0, 1], "x": np.arange(5.0) ** 2},
        ... )
        >>> array.max(dim="x")
        <xarray.DataArray (y: 3)>
        array([2., 2., 1.])
        Coordinates:
          * y        (y) int64 -1 0 1
        >>> array.argmax(dim="x")
        <xarray.DataArray (y: 3)>
        array([0, 2, 2])
        Coordinates:
          * y        (y) int64 -1 0 1
        >>> array.idxmax(dim="x")
        <xarray.DataArray 'x' (y: 3)>
        array([0., 4., 4.])
        Coordinates:
          * y        (y) int64 -1 0 1
        """
        return computation._calc_idxminmax(
            array=self,
            func=lambda x, *args, **kwargs: x.argmax(*args, **kwargs),
            dim=dim,
            skipna=skipna,
            fill_value=fill_value,
            keep_attrs=keep_attrs,
        )

    def argmin(
        self,
        dim: Union[Hashable, Sequence[Hashable]] = None,
        axis: int = None,
        keep_attrs: bool = None,
        skipna: bool = None,
    ) -> Union["DataArray", Dict[Hashable, "DataArray"]]:
        """Index or indices of the minimum of the DataArray over one or more dimensions.

        If a sequence is passed to 'dim', then result returned as dict of DataArrays,
        which can be passed directly to isel(). If a single str is passed to 'dim' then
        returns a DataArray with dtype int.

        If there are multiple minima, the indices of the first one found will be
        returned.

        Parameters
        ----------
        dim : hashable, sequence of hashable or ..., optional
            The dimensions over which to find the minimum. By default, finds minimum over
            all dimensions - for now returning an int for backward compatibility, but
            this is deprecated, in future will return a dict with indices for all
            dimensions; to return a dict with all dimensions now, pass '...'.
        axis : int, optional
            Axis over which to apply `argmin`. Only one of the 'dim' and 'axis' arguments
            can be supplied.
        keep_attrs : bool, optional
            If True, the attributes (`attrs`) will be copied from the original
            object to the new one.  If False (default), the new object will be
            returned without attributes.
        skipna : bool, optional
            If True, skip missing values (as marked by NaN). By default, only
            skips missing values for float dtypes; other dtypes either do not
            have a sentinel missing value (int) or skipna=True has not been
            implemented (object, datetime64 or timedelta64).

        Returns
        -------
        result : DataArray or dict of DataArray

        See also
        --------
        Variable.argmin, DataArray.idxmin

        Examples
        --------
        >>> array = xr.DataArray([0, 2, -1, 3], dims="x")
        >>> array.min()
        <xarray.DataArray ()>
        array(-1)
        >>> array.argmin()
        <xarray.DataArray ()>
        array(2)
        >>> array.argmin(...)
        {'x': <xarray.DataArray ()>
        array(2)}
        >>> array.isel(array.argmin(...))
        <xarray.DataArray ()>
        array(-1)

        >>> array = xr.DataArray(
        ...     [[[3, 2, 1], [3, 1, 2], [2, 1, 3]], [[1, 3, 2], [2, -5, 1], [2, 3, 1]]],
        ...     dims=("x", "y", "z"),
        ... )
        >>> array.min(dim="x")
        <xarray.DataArray (y: 3, z: 3)>
        array([[ 1,  2,  1],
               [ 2, -5,  1],
               [ 2,  1,  1]])
        Dimensions without coordinates: y, z
        >>> array.argmin(dim="x")
        <xarray.DataArray (y: 3, z: 3)>
        array([[1, 0, 0],
               [1, 1, 1],
               [0, 0, 1]])
        Dimensions without coordinates: y, z
        >>> array.argmin(dim=["x"])
        {'x': <xarray.DataArray (y: 3, z: 3)>
        array([[1, 0, 0],
               [1, 1, 1],
               [0, 0, 1]])
        Dimensions without coordinates: y, z}
        >>> array.min(dim=("x", "z"))
        <xarray.DataArray (y: 3)>
        array([ 1, -5,  1])
        Dimensions without coordinates: y
        >>> array.argmin(dim=["x", "z"])
        {'x': <xarray.DataArray (y: 3)>
        array([0, 1, 0])
        Dimensions without coordinates: y, 'z': <xarray.DataArray (y: 3)>
        array([2, 1, 1])
        Dimensions without coordinates: y}
        >>> array.isel(array.argmin(dim=["x", "z"]))
        <xarray.DataArray (y: 3)>
        array([ 1, -5,  1])
        Dimensions without coordinates: y
        """
        result = self.variable.argmin(dim, axis, keep_attrs, skipna)
        if isinstance(result, dict):
            return {k: self._replace_maybe_drop_dims(v) for k, v in result.items()}
        else:
            return self._replace_maybe_drop_dims(result)

    def argmax(
        self,
        dim: Union[Hashable, Sequence[Hashable]] = None,
        axis: int = None,
        keep_attrs: bool = None,
        skipna: bool = None,
    ) -> Union["DataArray", Dict[Hashable, "DataArray"]]:
        """Index or indices of the maximum of the DataArray over one or more dimensions.

        If a sequence is passed to 'dim', then result returned as dict of DataArrays,
        which can be passed directly to isel(). If a single str is passed to 'dim' then
        returns a DataArray with dtype int.

        If there are multiple maxima, the indices of the first one found will be
        returned.

        Parameters
        ----------
        dim : hashable, sequence of hashable or ..., optional
            The dimensions over which to find the maximum. By default, finds maximum over
            all dimensions - for now returning an int for backward compatibility, but
            this is deprecated, in future will return a dict with indices for all
            dimensions; to return a dict with all dimensions now, pass '...'.
        axis : int, optional
            Axis over which to apply `argmax`. Only one of the 'dim' and 'axis' arguments
            can be supplied.
        keep_attrs : bool, optional
            If True, the attributes (`attrs`) will be copied from the original
            object to the new one.  If False (default), the new object will be
            returned without attributes.
        skipna : bool, optional
            If True, skip missing values (as marked by NaN). By default, only
            skips missing values for float dtypes; other dtypes either do not
            have a sentinel missing value (int) or skipna=True has not been
            implemented (object, datetime64 or timedelta64).

        Returns
        -------
        result : DataArray or dict of DataArray

        See also
        --------
        Variable.argmax, DataArray.idxmax

        Examples
        --------
        >>> array = xr.DataArray([0, 2, -1, 3], dims="x")
        >>> array.max()
        <xarray.DataArray ()>
        array(3)
        >>> array.argmax()
        <xarray.DataArray ()>
        array(3)
        >>> array.argmax(...)
        {'x': <xarray.DataArray ()>
        array(3)}
        >>> array.isel(array.argmax(...))
        <xarray.DataArray ()>
        array(3)

        >>> array = xr.DataArray(
        ...     [[[3, 2, 1], [3, 1, 2], [2, 1, 3]], [[1, 3, 2], [2, 5, 1], [2, 3, 1]]],
        ...     dims=("x", "y", "z"),
        ... )
        >>> array.max(dim="x")
        <xarray.DataArray (y: 3, z: 3)>
        array([[3, 3, 2],
               [3, 5, 2],
               [2, 3, 3]])
        Dimensions without coordinates: y, z
        >>> array.argmax(dim="x")
        <xarray.DataArray (y: 3, z: 3)>
        array([[0, 1, 1],
               [0, 1, 0],
               [0, 1, 0]])
        Dimensions without coordinates: y, z
        >>> array.argmax(dim=["x"])
        {'x': <xarray.DataArray (y: 3, z: 3)>
        array([[0, 1, 1],
               [0, 1, 0],
               [0, 1, 0]])
        Dimensions without coordinates: y, z}
        >>> array.max(dim=("x", "z"))
        <xarray.DataArray (y: 3)>
        array([3, 5, 3])
        Dimensions without coordinates: y
        >>> array.argmax(dim=["x", "z"])
        {'x': <xarray.DataArray (y: 3)>
        array([0, 1, 0])
        Dimensions without coordinates: y, 'z': <xarray.DataArray (y: 3)>
        array([0, 1, 2])
        Dimensions without coordinates: y}
        >>> array.isel(array.argmax(dim=["x", "z"]))
        <xarray.DataArray (y: 3)>
        array([3, 5, 3])
        Dimensions without coordinates: y
        """
        result = self.variable.argmax(dim, axis, keep_attrs, skipna)
        if isinstance(result, dict):
            return {k: self._replace_maybe_drop_dims(v) for k, v in result.items()}
        else:
            return self._replace_maybe_drop_dims(result)

    # this needs to be at the end, or mypy will confuse with `str`
    # https://mypy.readthedocs.io/en/latest/common_issues.html#dealing-with-conflicting-names
    str = utils.UncachedAccessor(StringAccessor)


# priority most be higher than Variable to properly work with binary ufuncs
ops.inject_all_ops_and_reduce_methods(DataArray, priority=60)
