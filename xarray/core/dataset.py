import copy
import functools
import sys
import warnings
from collections import OrderedDict, defaultdict
from distutils.version import LooseVersion
from numbers import Number
from typing import (Any, Dict, Hashable, Iterable, Iterator, List,
                    Mapping, Optional, Sequence, Set, Tuple, Union, cast)

import numpy as np
import pandas as pd
import xarray as xr

from ..coding.cftimeindex import _parse_array_of_cftime_strings
from . import (alignment, dtypes, duck_array_ops, formatting, groupby,
               indexing, ops, pdcompat, resample, rolling, utils)
from .alignment import align
from .common import (ALL_DIMS, DataWithCoords, ImplementsDatasetReduce,
                     _contains_datetime_like_objects)
from .coordinates import (DatasetCoordinates, LevelCoordinatesSource,
                          assert_coordinate_consistent, remap_label_indexers)
from .duck_array_ops import datetime_to_numeric
from .indexes import (
    Indexes, default_indexes, isel_variable_and_index, roll_index,
)
from .merge import (
    dataset_merge_method, dataset_update_method, merge_data_and_coords,
    merge_variables)
from .options import OPTIONS, _get_keep_attrs
from .pycompat import TYPE_CHECKING, dask_array_type
from .utils import (Frozen, SortedKeysDict, _check_inplace,
                    decode_numpy_dict_values, either_dict_or_kwargs, hashable,
                    maybe_wrap_array)
from .variable import IndexVariable, Variable, as_variable, broadcast_variables

# Support for Python 3.5.0 ~ 3.5.1
try:
    from .pycompat import Mapping  # noqa: F811
except ImportError:
    pass
try:
    from typing import DefaultDict
except ImportError:
    pass

if TYPE_CHECKING:
    from ..backends import AbstractDataStore
    from .dataarray import DataArray


# list of attributes of pd.DatetimeIndex that are ndarrays of time info
_DATETIMEINDEX_COMPONENTS = ['year', 'month', 'day', 'hour', 'minute',
                             'second', 'microsecond', 'nanosecond', 'date',
                             'time', 'dayofyear', 'weekofyear', 'dayofweek',
                             'quarter']


def _get_virtual_variable(variables, key: Hashable,
                          level_vars: Optional[Mapping] = None,
                          dim_sizes: Optional[Mapping] = None,
                          ) -> Tuple[Hashable, Hashable, Variable]:
    """Get a virtual variable (e.g., 'time.year' or a MultiIndex level)
    from a dict of xarray.Variable objects (if possible)
    """
    if level_vars is None:
        level_vars = {}
    if dim_sizes is None:
        dim_sizes = {}

    if key in dim_sizes:
        data = pd.Index(range(dim_sizes[key]), name=key)
        variable = IndexVariable((key,), data)
        return key, key, variable

    if not isinstance(key, str):
        raise KeyError(key)

    split_key = key.split('.', 1)
    if len(split_key) == 2:
        ref_name, var_name = split_key  # type: str, Optional[str]
    elif len(split_key) == 1:
        ref_name, var_name = key, None
    else:
        raise KeyError(key)

    if ref_name in level_vars:
        dim_var = variables[level_vars[ref_name]]
        ref_var = dim_var.to_index_variable().get_level_variable(ref_name)
    else:
        ref_var = variables[ref_name]

    if var_name is None:
        virtual_var = ref_var
        var_name = key
    else:
        if _contains_datetime_like_objects(ref_var):
            ref_var = xr.DataArray(ref_var)
            data = getattr(ref_var.dt, var_name).data
        else:
            data = getattr(ref_var, var_name).data
        virtual_var = Variable(ref_var.dims, data)

    return ref_name, var_name, virtual_var


def calculate_dimensions(
    variables: Mapping[Hashable, Variable]
) -> 'Dict[Any, int]':
    """Calculate the dimensions corresponding to a set of variables.

    Returns dictionary mapping from dimension names to sizes. Raises ValueError
    if any of the dimension sizes conflict.
    """
    dims = {}  # type: Dict[Any, int]
    last_used = {}
    scalar_vars = set(k for k, v in variables.items() if not v.dims)
    for k, var in variables.items():
        for dim, size in zip(var.dims, var.shape):
            if dim in scalar_vars:
                raise ValueError('dimension %r already exists as a scalar '
                                 'variable' % dim)
            if dim not in dims:
                dims[dim] = size
                last_used[dim] = k
            elif dims[dim] != size:
                raise ValueError('conflicting sizes for dimension %r: '
                                 'length %s on %r and length %s on %r' %
                                 (dim, size, k, dims[dim], last_used[dim]))
    return dims


def merge_indexes(
    indexes: Mapping[Hashable, Union[Hashable, Sequence[Hashable]]],
    variables: Mapping[Hashable, Variable],
    coord_names: Set[Hashable],
    append: bool = False
) -> 'Tuple[OrderedDict[Any, Variable], Set[Hashable]]':
    """Merge variables into multi-indexes.

    Not public API. Used in Dataset and DataArray set_index
    methods.
    """
    vars_to_replace = {}  # Dict[Any, Variable]
    vars_to_remove = []  # type: list

    for dim, var_names in indexes.items():
        if isinstance(var_names, str) or not isinstance(var_names, Sequence):
            var_names = [var_names]

        names, codes, levels = [], [], []  # type: (list, list, list)
        current_index_variable = variables.get(dim)

        for n in var_names:
            var = variables[n]
            if (current_index_variable is not None
                    and var.dims != current_index_variable.dims):
                raise ValueError(
                    "dimension mismatch between %r %s and %r %s"
                    % (dim, current_index_variable.dims, n, var.dims))

        if current_index_variable is not None and append:
            current_index = current_index_variable.to_index()
            if isinstance(current_index, pd.MultiIndex):
                try:
                    current_codes = current_index.codes
                except AttributeError:
                    # fpr pandas<0.24
                    current_codes = current_index.labels
                names.extend(current_index.names)
                codes.extend(current_codes)
                levels.extend(current_index.levels)
            else:
                names.append('%s_level_0' % dim)
                cat = pd.Categorical(current_index.values, ordered=True)
                codes.append(cat.codes)
                levels.append(cat.categories)

        if not len(names) and len(var_names) == 1:
            idx = pd.Index(variables[var_names[0]].values)

        else:
            for n in var_names:
                names.append(n)
                var = variables[n]
                cat = pd.Categorical(var.values, ordered=True)
                codes.append(cat.codes)
                levels.append(cat.categories)

            idx = pd.MultiIndex(levels, codes, names=names)

        vars_to_replace[dim] = IndexVariable(dim, idx)
        vars_to_remove.extend(var_names)

    new_variables = OrderedDict([(k, v) for k, v in variables.items()
                                 if k not in vars_to_remove])
    new_variables.update(vars_to_replace)
    new_coord_names = coord_names | set(vars_to_replace)
    new_coord_names -= set(vars_to_remove)

    return new_variables, new_coord_names


def split_indexes(
    dims_or_levels: Union[Hashable, Sequence[Hashable]],
    variables: Mapping[Hashable, Variable],
    coord_names: Set[Hashable],
    level_coords: Mapping[Hashable, Hashable],
    drop: bool = False,
) -> 'Tuple[OrderedDict[Any, Variable], Set[Hashable]]':
    """Extract (multi-)indexes (levels) as variables.

    Not public API. Used in Dataset and DataArray reset_index
    methods.
    """
    if (isinstance(dims_or_levels, str) or
            not isinstance(dims_or_levels, Sequence)):
        dims_or_levels = [dims_or_levels]

    dim_levels \
        = defaultdict(list)  # type: DefaultDict[Any, List[Hashable]]
    dims = []
    for k in dims_or_levels:
        if k in level_coords:
            dim_levels[level_coords[k]].append(k)
        else:
            dims.append(k)

    vars_to_replace = {}
    vars_to_create = OrderedDict()  # type: OrderedDict[Any, Variable]
    vars_to_remove = []

    for d in dims:
        index = variables[d].to_index()
        if isinstance(index, pd.MultiIndex):
            dim_levels[d] = index.names
        else:
            vars_to_remove.append(d)
            if not drop:
                vars_to_create[str(d) + '_'] = Variable(d, index)

    for d, levs in dim_levels.items():
        index = variables[d].to_index()
        if len(levs) == index.nlevels:
            vars_to_remove.append(d)
        else:
            vars_to_replace[d] = IndexVariable(d, index.droplevel(levs))

        if not drop:
            for lev in levs:
                idx = index.get_level_values(lev)
                vars_to_create[idx.name] = Variable(d, idx)

    new_variables = OrderedDict(variables)
    for v in set(vars_to_remove):
        del new_variables[v]
    new_variables.update(vars_to_replace)
    new_variables.update(vars_to_create)
    new_coord_names = (coord_names | set(vars_to_create)) - set(vars_to_remove)

    return new_variables, new_coord_names


def _assert_empty(args: tuple, msg: str = '%s') -> None:
    if args:
        raise ValueError(msg % args)


def as_dataset(obj: Any) -> 'Dataset':
    """Cast the given object to a Dataset.

    Handles Datasets, DataArrays and dictionaries of variables. A new Dataset
    object is only created if the provided object is not already one.
    """
    if hasattr(obj, 'to_dataset'):
        obj = obj.to_dataset()
    if not isinstance(obj, Dataset):
        obj = Dataset(obj)
    return obj


class DataVariables(Mapping[Hashable, 'Union[DataArray, Dataset]']):
    def __init__(self, dataset: 'Dataset'):
        self._dataset = dataset

    def __iter__(self) -> Iterator[Hashable]:
        return (key for key in self._dataset._variables
                if key not in self._dataset._coord_names)

    def __len__(self) -> int:
        return len(self._dataset._variables) - len(self._dataset._coord_names)

    def __contains__(self, key) -> bool:
        return (key in self._dataset._variables
                and key not in self._dataset._coord_names)

    def __getitem__(self, key) -> 'Union[DataArray, Dataset]':
        if key not in self._dataset._coord_names:
            return self._dataset[key]
        else:
            raise KeyError(key)

    def __repr__(self) -> str:
        return formatting.data_vars_repr(self)

    @property
    def variables(self) -> Mapping[Hashable, Variable]:
        all_variables = self._dataset.variables
        return Frozen(OrderedDict((k, all_variables[k]) for k in self))

    def _ipython_key_completions_(self):
        """Provide method for the key-autocompletions in IPython. """
        return [key for key in self._dataset._ipython_key_completions_()
                if key not in self._dataset._coord_names]


class _LocIndexer:
    def __init__(self, dataset: 'Dataset'):
        self.dataset = dataset

    def __getitem__(self, key: Mapping[str, Any]) -> 'Dataset':
        if not utils.is_dict_like(key):
            raise TypeError('can only lookup dictionaries from Dataset.loc')
        return self.dataset.sel(**key)


class Dataset(Mapping, ImplementsDatasetReduce, DataWithCoords):
    """A multi-dimensional, in memory, array database.

    A dataset resembles an in-memory representation of a NetCDF file, and
    consists of variables, coordinates and attributes which together form a
    self describing dataset.

    Dataset implements the mapping interface with keys given by variable names
    and values given by DataArray objects for each variable name.

    One dimensional variables with name equal to their dimension are index
    coordinates used for label based indexing.
    """
    _groupby_cls = groupby.DatasetGroupBy
    _rolling_cls = rolling.DatasetRolling
    _coarsen_cls = rolling.DatasetCoarsen
    _resample_cls = resample.DatasetResample

    def __init__(
        self,
        # could make a VariableArgs to use more generally, and refine these
        # categories
        data_vars: Optional[Mapping[Hashable, Union[
            'DataArray',
            Variable,
            Tuple[Hashable, Any],
            Tuple[Sequence[Hashable], Any],
        ]]] = None,
        coords: Optional[Mapping[Hashable, Any]] = None,
        attrs: Optional[Mapping] = None,
        compat=None,
    ):
        """To load data from a file or file-like object, use the `open_dataset`
        function.

        Parameters
        ----------
        data_vars : dict-like, optional
            A mapping from variable names to :py:class:`~xarray.DataArray`
            objects, :py:class:`~xarray.Variable` objects or to tuples of the
            form ``(dims, data[, attrs])`` which can be used as arguments to
            create a new ``Variable``. Each dimension must have the same length
            in all variables in which it appears.

            The following notations are accepted:

            - mapping {var name: DataArray}
            - mapping {var name: Variable}
            - mapping {var name: (dimension name, array-like)}
            - mapping {var name: (tuple of dimension names, array-like)}

            Each dimension must have the same length in all variables in which
            it appears.
        coords : dict-like, optional
            Another mapping in similar form as the `data_vars` argument,
            except the each item is saved on the dataset as a "coordinate".
            These variables have an associated meaning: they describe
            constant/fixed/independent quantities, unlike the
            varying/measured/dependent quantities that belong in `variables`.
            Coordinates values may be given by 1-dimensional arrays or scalars,
            in which case `dims` do not need to be supplied: 1D arrays will be
            assumed to give index values along the dimension with the same
            name.

            The following notations are accepted:

            - mapping {coord name: DataArray}
            - mapping {coord name: Variable}
            - mapping {coord name: (dimension name, array-like)}
            - mapping {coord name: (tuple of dimension names, array-like)}
            - mapping {dimension name: array-like}

            The last notation implies that the coord name is the same as the
            dimension name.

        attrs : dict-like, optional
            Global attributes to save on this dataset.
        compat : deprecated
        """
        if compat is not None:
            warnings.warn(
                'The `compat` argument to Dataset is deprecated and will be '
                'removed in 0.13.'
                'Instead, use `merge` to control how variables are combined',
                FutureWarning, stacklevel=2)
        else:
            compat = 'broadcast_equals'

        self._variables = \
            OrderedDict()  # type: OrderedDict[Any, Variable]
        self._coord_names = set()  # type: Set[Hashable]
        self._dims = {}  # type: Dict[Any, int]
        self._attrs = None  # type: Optional[OrderedDict]
        self._file_obj = None
        if data_vars is None:
            data_vars = {}
        if coords is None:
            coords = {}
        self._set_init_vars_and_dims(data_vars, coords, compat)

        # TODO(shoyer): expose indexes as a public argument in __init__
        self._indexes = None  # type: Optional[OrderedDict[Any, pd.Index]]

        if attrs is not None:
            self.attrs = attrs
        self._encoding = None  # type: Optional[Dict]
        self._initialized = True

    def _set_init_vars_and_dims(self, data_vars, coords, compat):
        """Set the initial value of Dataset variables and dimensions
        """
        both_data_and_coords = [k for k in data_vars if k in coords]
        if both_data_and_coords:
            raise ValueError('variables %r are found in both data_vars and '
                             'coords' % both_data_and_coords)

        if isinstance(coords, Dataset):
            coords = coords.variables

        variables, coord_names, dims = merge_data_and_coords(
            data_vars, coords, compat=compat)

        self._variables = variables
        self._coord_names = coord_names
        self._dims = dims

    @classmethod
    def load_store(cls, store, decoder=None) -> 'Dataset':
        """Create a new dataset from the contents of a backends.*DataStore
        object
        """
        variables, attributes = store.load()
        if decoder:
            variables, attributes = decoder(variables, attributes)
        obj = cls(variables, attrs=attributes)
        obj._file_obj = store
        return obj

    @property
    def variables(self) -> Mapping[Hashable, Variable]:
        """Low level interface to Dataset contents as dict of Variable objects.

        This ordered dictionary is frozen to prevent mutation that could
        violate Dataset invariants. It contains all variable objects
        constituting the Dataset, including both data variables and
        coordinates.
        """
        return Frozen(self._variables)

    @property
    def attrs(self) -> 'OrderedDict[Any, Any]':
        """Dictionary of global attributes on this dataset
        """
        if self._attrs is None:
            self._attrs = OrderedDict()
        return self._attrs

    @attrs.setter
    def attrs(self, value: Mapping[Hashable, Any]) -> None:
        self._attrs = OrderedDict(value)

    @property
    def encoding(self) -> Dict:
        """Dictionary of global encoding attributes on this dataset
        """
        if self._encoding is None:
            self._encoding = {}
        return self._encoding

    @encoding.setter
    def encoding(self, value: Mapping) -> None:
        self._encoding = dict(value)

    @property
    def dims(self) -> Mapping[Hashable, int]:
        """Mapping from dimension names to lengths.

        Cannot be modified directly, but is updated when adding new variables.

        Note that type of this object differs from `DataArray.dims`.
        See `Dataset.sizes` and `DataArray.sizes` for consistently named
        properties.
        """
        return Frozen(SortedKeysDict(self._dims))

    @property
    def sizes(self) -> Mapping[Hashable, int]:
        """Mapping from dimension names to lengths.

        Cannot be modified directly, but is updated when adding new variables.

        This is an alias for `Dataset.dims` provided for the benefit of
        consistency with `DataArray.sizes`.

        See also
        --------
        DataArray.sizes
        """
        return self.dims

    def load(self, **kwargs) -> 'Dataset':
        """Manually trigger loading of this dataset's data from disk or a
        remote source into memory and return this dataset.

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
        # access .data to coerce everything to numpy or dask arrays
        lazy_data = {k: v._data for k, v in self.variables.items()
                     if isinstance(v._data, dask_array_type)}
        if lazy_data:
            import dask.array as da

            # evaluate all the dask arrays simultaneously
            evaluated_data = da.compute(*lazy_data.values(), **kwargs)

            for k, data in zip(lazy_data, evaluated_data):
                self.variables[k].data = data

        # load everything else sequentially
        for k, v in self.variables.items():
            if k not in lazy_data:
                v.load()

        return self

    def __dask_graph__(self):
        graphs = {k: v.__dask_graph__() for k, v in self.variables.items()}
        graphs = {k: v for k, v in graphs.items() if v is not None}
        if not graphs:
            return None
        else:
            try:
                from dask.highlevelgraph import HighLevelGraph
                return HighLevelGraph.merge(*graphs.values())
            except ImportError:
                from dask import sharedict
                return sharedict.merge(*graphs.values())

    def __dask_keys__(self):
        import dask
        return [v.__dask_keys__() for v in self.variables.values()
                if dask.is_dask_collection(v)]

    def __dask_layers__(self):
        import dask
        return sum([v.__dask_layers__() for v in self.variables.values() if
                    dask.is_dask_collection(v)], ())

    @property
    def __dask_optimize__(self):
        import dask.array as da
        return da.Array.__dask_optimize__

    @property
    def __dask_scheduler__(self):
        import dask.array as da
        return da.Array.__dask_scheduler__

    def __dask_postcompute__(self):
        import dask
        info = [(True, k, v.__dask_postcompute__())
                if dask.is_dask_collection(v) else
                (False, k, v) for k, v in self._variables.items()]
        args = (
            info,
            self._coord_names,
            self._dims,
            self._attrs,
            self._indexes,
            self._encoding,
            self._file_obj,
        )
        return self._dask_postcompute, args

    def __dask_postpersist__(self):
        import dask
        info = [(True, k, v.__dask_postpersist__())
                if dask.is_dask_collection(v) else
                (False, k, v) for k, v in self._variables.items()]
        args = (
            info,
            self._coord_names,
            self._dims,
            self._attrs,
            self._indexes,
            self._encoding,
            self._file_obj,
        )
        return self._dask_postpersist, args

    @staticmethod
    def _dask_postcompute(results, info, *args):
        variables = OrderedDict()
        results2 = list(results[::-1])
        for is_dask, k, v in info:
            if is_dask:
                func, args2 = v
                r = results2.pop()
                result = func(r, *args2)
            else:
                result = v
            variables[k] = result

        final = Dataset._construct_direct(variables, *args)
        return final

    @staticmethod
    def _dask_postpersist(dsk, info, *args):
        variables = OrderedDict()
        for is_dask, k, v in info:
            if is_dask:
                func, args2 = v
                result = func(dsk, *args2)
            else:
                result = v
            variables[k] = result

        return Dataset._construct_direct(variables, *args)

    def compute(self, **kwargs) -> 'Dataset':
        """Manually trigger loading of this dataset's data from disk or a
        remote source into memory and return a new dataset. The original is
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

    def _persist_inplace(self, **kwargs) -> 'Dataset':
        """Persist all Dask arrays in memory
        """
        # access .data to coerce everything to numpy or dask arrays
        lazy_data = {k: v._data for k, v in self.variables.items()
                     if isinstance(v._data, dask_array_type)}
        if lazy_data:
            import dask

            # evaluate all the dask arrays simultaneously
            evaluated_data = dask.persist(*lazy_data.values(), **kwargs)

            for k, data in zip(lazy_data, evaluated_data):
                self.variables[k].data = data

        return self

    def persist(self, **kwargs) -> 'Dataset':
        """ Trigger computation, keeping data as dask arrays

        This operation can be used to trigger computation on underlying dask
        arrays, similar to ``.compute()``.  However this operation keeps the
        data as dask arrays.  This is particularly useful when using the
        dask.distributed scheduler and you want to load a large amount of data
        into distributed memory.

        Parameters
        ----------
        **kwargs : dict
            Additional keyword arguments passed on to ``dask.persist``.

        See Also
        --------
        dask.persist
        """
        new = self.copy(deep=False)
        return new._persist_inplace(**kwargs)

    @classmethod
    def _construct_direct(cls, variables, coord_names, dims, attrs=None,
                          indexes=None, encoding=None, file_obj=None):
        """Shortcut around __init__ for internal use when we want to skip
        costly validation
        """
        obj = object.__new__(cls)
        obj._variables = variables
        obj._coord_names = coord_names
        obj._dims = dims
        obj._indexes = indexes
        obj._attrs = attrs
        obj._file_obj = file_obj
        obj._encoding = encoding
        obj._initialized = True
        return obj

    __default = object()

    @classmethod
    def _from_vars_and_coord_names(cls, variables, coord_names, attrs=None):
        dims = calculate_dimensions(variables)
        return cls._construct_direct(variables, coord_names, dims, attrs)

    # TODO(shoyer): renable type checking on this signature when pytype has a
    # good way to handle defaulting arguments to a sentinel value:
    # https://github.com/python/mypy/issues/1803
    def _replace(  # type: ignore
        self,
        variables: 'OrderedDict[Any, Variable]' = None,
        coord_names: Optional[Set[Hashable]] = None,
        dims: Dict[Any, int] = None,
        attrs: 'Optional[OrderedDict]' = __default,
        indexes: 'Optional[OrderedDict[Any, pd.Index]]' = __default,
        encoding: Optional[dict] = __default,
        inplace: bool = False,
    ) -> 'Dataset':
        """Fastpath constructor for internal use.

        Returns an object with optionally with replaced attributes.

        Explicitly passed arguments are *not* copied when placed on the new
        dataset. It is up to the caller to ensure that they have the right type
        and are not used elsewhere.
        """
        if inplace:
            if variables is not None:
                self._variables = variables
            if coord_names is not None:
                self._coord_names = coord_names
            if dims is not None:
                self._dims = dims
            if attrs is not self.__default:
                self._attrs = attrs
            if indexes is not self.__default:
                self._indexes = indexes
            if encoding is not self.__default:
                self._encoding = encoding
            obj = self
        else:
            if variables is None:
                variables = self._variables.copy()
            if coord_names is None:
                coord_names = self._coord_names.copy()
            if dims is None:
                dims = self._dims.copy()
            if attrs is self.__default:
                attrs = copy.copy(self._attrs)
            if indexes is self.__default:
                indexes = copy.copy(self._indexes)
            if encoding is self.__default:
                encoding = copy.copy(self._encoding)
            obj = self._construct_direct(
                variables, coord_names, dims, attrs, indexes, encoding)
        return obj

    def _replace_with_new_dims(  # type: ignore
        self,
        variables: 'OrderedDict[Any, Variable]',
        coord_names: set = None,
        attrs: 'Optional[OrderedDict]' = __default,
        indexes: 'Optional[OrderedDict[Any, pd.Index]]' = __default,
        inplace: bool = False,
    ) -> 'Dataset':
        """Replace variables with recalculated dimensions."""
        dims = calculate_dimensions(variables)
        return self._replace(
            variables, coord_names, dims, attrs, indexes, inplace=inplace)

    def _replace_vars_and_dims(  # type: ignore
        self,
        variables: 'OrderedDict[Any, Variable]',
        coord_names: set = None,
        dims: 'Optional[Dict[Any, int]]' = None,
        attrs: 'Optional[OrderedDict]' = __default,
        inplace: bool = False,
    ) -> 'Dataset':
        """Deprecated version of _replace_with_new_dims().

        Unlike _replace_with_new_dims(), this method always recalculates
        indexes from variables.
        """
        if dims is None:
            dims = calculate_dimensions(variables)
        return self._replace(
            variables, coord_names, dims, attrs, indexes=None, inplace=inplace)

    def _overwrite_indexes(self, indexes: Mapping[Any, pd.Index]) -> 'Dataset':
        if not indexes:
            return self

        variables = self._variables.copy()
        new_indexes = OrderedDict(self.indexes)
        for name, idx in indexes.items():
            variables[name] = IndexVariable(name, idx)
            new_indexes[name] = idx
        obj = self._replace(variables, indexes=new_indexes)

        # switch from dimension to level names, if necessary
        dim_names = {}  # type: Dict[Hashable, str]
        for dim, idx in indexes.items():
            if not isinstance(idx, pd.MultiIndex) and idx.name != dim:
                dim_names[dim] = idx.name
        if dim_names:
            obj = obj.rename(dim_names)
        return obj

    def copy(self, deep: bool = False, data: Mapping = None) -> 'Dataset':
        """Returns a copy of this dataset.

        If `deep=True`, a deep copy is made of each of the component variables.
        Otherwise, a shallow copy of each of the component variable is made, so
        that the underlying memory region of the new dataset is the same as in
        the original dataset.

        Use `data` to create a new object with the same structure as
        original but entirely new data.

        Parameters
        ----------
        deep : bool, optional
            Whether each component variable is loaded into memory and copied onto
            the new object. Default is False.
        data : dict-like, optional
            Data to use in the new object. Each item in `data` must have same
            shape as corresponding data variable in original. When `data` is
            used, `deep` is ignored for the data variables and only used for
            coords.

        Returns
        -------
        object : Dataset
            New object with dimensions, attributes, coordinates, name, encoding,
            and optionally data copied from original.

        Examples
        --------

        Shallow copy versus deep copy

        >>> da = xr.DataArray(np.random.randn(2, 3))
        >>> ds = xr.Dataset({'foo': da, 'bar': ('x', [-1, 2])},
                            coords={'x': ['one', 'two']})
        >>> ds.copy()
        <xarray.Dataset>
        Dimensions:  (dim_0: 2, dim_1: 3, x: 2)
        Coordinates:
        * x        (x) <U3 'one' 'two'
        Dimensions without coordinates: dim_0, dim_1
        Data variables:
            foo      (dim_0, dim_1) float64 -0.8079 0.3897 -1.862 -0.6091 -1.051 -0.3003
            bar      (x) int64 -1 2
        >>> ds_0 = ds.copy(deep=False)
        >>> ds_0['foo'][0, 0] = 7
        >>> ds_0
        <xarray.Dataset>
        Dimensions:  (dim_0: 2, dim_1: 3, x: 2)
        Coordinates:
        * x        (x) <U3 'one' 'two'
        Dimensions without coordinates: dim_0, dim_1
        Data variables:
            foo      (dim_0, dim_1) float64 7.0 0.3897 -1.862 -0.6091 -1.051 -0.3003
            bar      (x) int64 -1 2
        >>> ds
        <xarray.Dataset>
        Dimensions:  (dim_0: 2, dim_1: 3, x: 2)
        Coordinates:
        * x        (x) <U3 'one' 'two'
        Dimensions without coordinates: dim_0, dim_1
        Data variables:
            foo      (dim_0, dim_1) float64 7.0 0.3897 -1.862 -0.6091 -1.051 -0.3003
            bar      (x) int64 -1 2

        Changing the data using the ``data`` argument maintains the
        structure of the original object, but with the new data. Original
        object is unaffected.

        >>> ds.copy(data={'foo': np.arange(6).reshape(2, 3), 'bar': ['a', 'b']})
        <xarray.Dataset>
        Dimensions:  (dim_0: 2, dim_1: 3, x: 2)
        Coordinates:
        * x        (x) <U3 'one' 'two'
        Dimensions without coordinates: dim_0, dim_1
        Data variables:
            foo      (dim_0, dim_1) int64 0 1 2 3 4 5
            bar      (x) <U1 'a' 'b'
        >>> ds
        <xarray.Dataset>
        Dimensions:  (dim_0: 2, dim_1: 3, x: 2)
        Coordinates:
        * x        (x) <U3 'one' 'two'
        Dimensions without coordinates: dim_0, dim_1
        Data variables:
            foo      (dim_0, dim_1) float64 7.0 0.3897 -1.862 -0.6091 -1.051 -0.3003
            bar      (x) int64 -1 2

        See Also
        --------
        pandas.DataFrame.copy
        """  # noqa
        if data is None:
            variables = OrderedDict((k, v.copy(deep=deep))
                                    for k, v in self._variables.items())
        elif not utils.is_dict_like(data):
            raise ValueError('Data must be dict-like')
        else:
            var_keys = set(self.data_vars.keys())
            data_keys = set(data.keys())
            keys_not_in_vars = data_keys - var_keys
            if keys_not_in_vars:
                raise ValueError(
                    'Data must only contain variables in original '
                    'dataset. Extra variables: {}'
                    .format(keys_not_in_vars))
            keys_missing_from_data = var_keys - data_keys
            if keys_missing_from_data:
                raise ValueError(
                    'Data must contain all variables in original '
                    'dataset. Data is missing {}'
                    .format(keys_missing_from_data))
            variables = OrderedDict((k, v.copy(deep=deep, data=data.get(k)))
                                    for k, v in self._variables.items())

        attrs = copy.deepcopy(self._attrs) if deep else copy.copy(self._attrs)

        return self._replace(variables, attrs=attrs)

    @property
    def _level_coords(self) -> 'OrderedDict[str, Hashable]':
        """Return a mapping of all MultiIndex levels and their corresponding
        coordinate name.
        """
        level_coords = OrderedDict()  # type: OrderedDict[str, Hashable]
        for name, index in self.indexes.items():
            if isinstance(index, pd.MultiIndex):
                level_names = index.names
                (dim,) = self.variables[name].dims
                level_coords.update({lname: dim for lname in level_names})
        return level_coords

    def _copy_listed(self, names: Iterable[Hashable]) -> 'Dataset':
        """Create a new Dataset with the listed variables from this dataset and
        the all relevant coordinates. Skips all validation.
        """
        variables = OrderedDict()  # type: OrderedDict[Any, Variable]
        coord_names = set()
        indexes = OrderedDict()  # type: OrderedDict[Any, pd.Index]

        for name in names:
            try:
                variables[name] = self._variables[name]
            except KeyError:
                ref_name, var_name, var = _get_virtual_variable(
                    self._variables, name, self._level_coords, self.dims)
                variables[var_name] = var
                if ref_name in self._coord_names or ref_name in self.dims:
                    coord_names.add(var_name)
                if (var_name,) == var.dims:
                    indexes[var_name] = var.to_index()

        needed_dims = set()  # type: set
        for v in variables.values():
            needed_dims.update(v.dims)

        dims = dict((k, self.dims[k]) for k in needed_dims)

        for k in self._coord_names:
            if set(self.variables[k].dims) <= needed_dims:
                variables[k] = self._variables[k]
                coord_names.add(k)
                if k in self.indexes:
                    indexes[k] = self.indexes[k]

        return self._replace(variables, coord_names, dims, indexes=indexes)

    def _construct_dataarray(self, name: Hashable) -> 'DataArray':
        """Construct a DataArray by indexing this dataset
        """
        from .dataarray import DataArray

        try:
            variable = self._variables[name]
        except KeyError:
            _, name, variable = _get_virtual_variable(
                self._variables, name, self._level_coords, self.dims)

        needed_dims = set(variable.dims)

        coords = OrderedDict()  # type: OrderedDict[Any, Variable]
        for k in self.coords:
            if set(self.variables[k].dims) <= needed_dims:
                coords[k] = self.variables[k]

        if self._indexes is None:
            indexes = None
        else:
            indexes = OrderedDict((k, v) for k, v in self._indexes.items()
                                  if k in coords)

        return DataArray(variable, coords, name=name, indexes=indexes,
                         fastpath=True)

    def __copy__(self) -> 'Dataset':
        return self.copy(deep=False)

    def __deepcopy__(self, memo=None) -> 'Dataset':
        # memo does nothing but is required for compatibility with
        # copy.deepcopy
        return self.copy(deep=True)

    @property
    def _attr_sources(self) -> List[Mapping[Hashable, Any]]:
        """List of places to look-up items for attribute-style access
        """
        return self._item_sources + [self.attrs]

    @property
    def _item_sources(self) -> List[Mapping[Hashable, Any]]:
        """List of places to look-up items for key-completion
        """
        return [self.data_vars, self.coords, {d: self[d] for d in self.dims},
                LevelCoordinatesSource(self)]

    def __contains__(self, key: object) -> bool:
        """The 'in' operator will return true or false depending on whether
        'key' is an array in the dataset or not.
        """
        return key in self._variables

    def __len__(self) -> int:
        return len(self.data_vars)

    def __bool__(self) -> bool:
        return bool(self.data_vars)

    def __iter__(self) -> Iterator[Hashable]:
        return iter(self.data_vars)

    def __array__(self, dtype=None):
        raise TypeError('cannot directly convert an xarray.Dataset into a '
                        'numpy array. Instead, create an xarray.DataArray '
                        'first, either with indexing on the Dataset or by '
                        'invoking the `to_array()` method.')

    @property
    def nbytes(self) -> int:
        return sum(v.nbytes for v in self.variables.values())

    @property
    def loc(self) -> _LocIndexer:
        """Attribute for location based indexing. Only supports __getitem__,
        and only when the key is a dict of the form {dim: labels}.
        """
        return _LocIndexer(self)

    def __getitem__(self, key: object) -> 'Union[DataArray, Dataset]':
        """Access variables or coordinates this dataset as a
        :py:class:`~xarray.DataArray`.

        Indexing with a list of names will return a new ``Dataset`` object.
        """
        if utils.is_dict_like(key):
            return self.isel(**cast(Mapping, key))

        if hashable(key):
            return self._construct_dataarray(key)
        else:
            return self._copy_listed(np.asarray(key))

    def __setitem__(self, key: Hashable, value) -> None:
        """Add an array to this dataset.

        If value is a `DataArray`, call its `select_vars()` method, rename it
        to `key` and merge the contents of the resulting dataset into this
        dataset.

        If value is an `Variable` object (or tuple of form
        ``(dims, data[, attrs])``), add it to this dataset as a new
        variable.
        """
        if utils.is_dict_like(key):
            raise NotImplementedError('cannot yet use a dictionary as a key '
                                      'to set Dataset values')

        self.update({key: value})

    def __delitem__(self, key: Hashable) -> None:
        """Remove a variable from this dataset.
        """
        del self._variables[key]
        self._coord_names.discard(key)
        self._dims = calculate_dimensions(self._variables)

    # mutable objects should not be hashable
    # https://github.com/python/mypy/issues/4266
    __hash__ = None  # type: ignore

    def _all_compat(self, other: 'Dataset', compat_str: str) -> bool:
        """Helper function for equals and identical
        """

        # some stores (e.g., scipy) do not seem to preserve order, so don't
        # require matching order for equality
        def compat(x: Variable, y: Variable) -> bool:
            return getattr(x, compat_str)(y)

        return (self._coord_names == other._coord_names and
                utils.dict_equiv(self._variables, other._variables,
                                 compat=compat))

    def broadcast_equals(self, other: 'Dataset') -> bool:
        """Two Datasets are broadcast equal if they are equal after
        broadcasting all variables against each other.

        For example, variables that are scalar in one dataset but non-scalar in
        the other dataset can still be broadcast equal if the the non-scalar
        variable is a constant.

        See Also
        --------
        Dataset.equals
        Dataset.identical
        """
        try:
            return self._all_compat(other, 'broadcast_equals')
        except (TypeError, AttributeError):
            return False

    def equals(self, other: 'Dataset') -> bool:
        """Two Datasets are equal if they have matching variables and
        coordinates, all of which are equal.

        Datasets can still be equal (like pandas objects) if they have NaN
        values in the same locations.

        This method is necessary because `v1 == v2` for ``Dataset``
        does element-wise comparisons (like numpy.ndarrays).

        See Also
        --------
        Dataset.broadcast_equals
        Dataset.identical
        """
        try:
            return self._all_compat(other, 'equals')
        except (TypeError, AttributeError):
            return False

    def identical(self, other: 'Dataset') -> bool:
        """Like equals, but also checks all dataset attributes and the
        attributes on all variables and coordinates.

        See Also
        --------
        Dataset.broadcast_equals
        Dataset.equals
        """
        try:
            return (utils.dict_equiv(self.attrs, other.attrs)
                    and self._all_compat(other, 'identical'))
        except (TypeError, AttributeError):
            return False

    @property
    def indexes(self) -> 'Mapping[Any, pd.Index]':
        """Mapping of pandas.Index objects used for label based indexing
        """
        if self._indexes is None:
            self._indexes = default_indexes(self._variables, self._dims)
        return Indexes(self._indexes)

    @property
    def coords(self) -> DatasetCoordinates:
        """Dictionary of xarray.DataArray objects corresponding to coordinate
        variables
        """
        return DatasetCoordinates(self)

    @property
    def data_vars(self) -> DataVariables:
        """Dictionary of DataArray objects corresponding to data variables
        """
        return DataVariables(self)

    def set_coords(
        self,
        names: 'Union[Hashable, Iterable[Hashable]]',
        inplace: bool = None
    ) -> 'Dataset':
        """Given names of one or more variables, set them as coordinates

        Parameters
        ----------
        names : hashable or iterable of hashables
            Name(s) of variables in this dataset to convert into coordinates.
        inplace : bool, optional
            If True, modify this dataset inplace. Otherwise, create a new
            object.

        Returns
        -------
        Dataset

        See also
        --------
        Dataset.swap_dims
        """
        # TODO: allow inserting new coordinates with this method, like
        # DataFrame.set_index?
        # nb. check in self._variables, not self.data_vars to insure that the
        # operation is idempotent
        inplace = _check_inplace(inplace)
        if isinstance(names, str) or not isinstance(names, Iterable):
            names = [names]
        else:
            names = list(names)
        self._assert_all_in_dataset(names)
        obj = self if inplace else self.copy()
        obj._coord_names.update(names)
        return obj

    def reset_coords(
        self,
        names: 'Union[Hashable, Iterable[Hashable], None]' = None,
        drop: bool = False,
        inplace: bool = None
    ) -> 'Dataset':
        """Given names of coordinates, reset them to become variables

        Parameters
        ----------
        names : hashable or iterable of hashables, optional
            Name(s) of non-index coordinates in this dataset to reset into
            variables. By default, all non-index coordinates are reset.
        drop : bool, optional
            If True, remove coordinates instead of converting them into
            variables.
        inplace : bool, optional
            If True, modify this dataset inplace. Otherwise, create a new
            object.

        Returns
        -------
        Dataset
        """
        inplace = _check_inplace(inplace)
        if names is None:
            names = self._coord_names - set(self.dims)
        else:
            if isinstance(names, str) or not isinstance(names, Iterable):
                names = [names]
            else:
                names = list(names)
            self._assert_all_in_dataset(names)
            bad_coords = set(names) & set(self.dims)
            if bad_coords:
                raise ValueError(
                    'cannot remove index coordinates with reset_coords: %s'
                    % bad_coords)
        obj = self if inplace else self.copy()
        obj._coord_names.difference_update(names)
        if drop:
            for name in names:
                del obj._variables[name]
        return obj

    def dump_to_store(self, store: 'AbstractDataStore', **kwargs) -> None:
        """Store dataset contents to a backends.*DataStore object.
        """
        from ..backends.api import dump_to_store
        # TODO: rename and/or cleanup this method to make it more consistent
        # with to_netcdf()
        dump_to_store(self, store, **kwargs)

    def to_netcdf(self, path=None, mode='w', format=None, group=None,
                  engine=None, encoding=None, unlimited_dims=None,
                  compute=True):
        """Write dataset contents to a netCDF file.

        Parameters
        ----------
        path : str, Path or file-like object, optional
            Path to which to save this dataset. File-like objects are only
            supported by the scipy engine. If no path is provided, this
            function returns the resulting netCDF file as bytes; in this case,
            we need to use scipy, which does not support netCDF version 4 (the
            default format becomes NETCDF3_64BIT).
        mode : {'w', 'a'}, optional
            Write ('w') or append ('a') mode. If mode='w', any existing file at
            this location will be overwritten. If mode='a', existing variables
            will be overwritten.
        format : {'NETCDF4', 'NETCDF4_CLASSIC', 'NETCDF3_64BIT',
                  'NETCDF3_CLASSIC'}, optional
            File format for the resulting netCDF file:

            * NETCDF4: Data is stored in an HDF5 file, using netCDF4 API
              features.
            * NETCDF4_CLASSIC: Data is stored in an HDF5 file, using only
              netCDF 3 compatible API features.
            * NETCDF3_64BIT: 64-bit offset version of the netCDF 3 file format,
              which fully supports 2+ GB files, but is only compatible with
              clients linked against netCDF version 3.6.0 or later.
            * NETCDF3_CLASSIC: The classic netCDF 3 file format. It does not
              handle 2+ GB files very well.

            All formats are supported by the netCDF4-python library.
            scipy.io.netcdf only supports the last two formats.

            The default format is NETCDF4 if you are saving a file to disk and
            have the netCDF4-python library available. Otherwise, xarray falls
            back to using scipy to write netCDF files and defaults to the
            NETCDF3_64BIT format (scipy does not support netCDF4).
        group : str, optional
            Path to the netCDF4 group in the given file to open (only works for
            format='NETCDF4'). The group(s) will be created if necessary.
        engine : {'netcdf4', 'scipy', 'h5netcdf'}, optional
            Engine to use when writing netCDF files. If not provided, the
            default engine is chosen based on available dependencies, with a
            preference for 'netcdf4' if writing to a file on disk.
        encoding : dict, optional
            Nested dictionary with variable names as keys and dictionaries of
            variable specific encodings as values, e.g.,
            ``{'my_variable': {'dtype': 'int16', 'scale_factor': 0.1,
                               'zlib': True}, ...}``

            The `h5netcdf` engine supports both the NetCDF4-style compression
            encoding parameters ``{'zlib': True, 'complevel': 9}`` and the h5py
            ones ``{'compression': 'gzip', 'compression_opts': 9}``.
            This allows using any compression plugin installed in the HDF5
            library, e.g. LZF.

        unlimited_dims : sequence of str, optional
            Dimension(s) that should be serialized as unlimited dimensions.
            By default, no dimensions are treated as unlimited dimensions.
            Note that unlimited_dims may also be set via
            ``dataset.encoding['unlimited_dims']``.
        compute: boolean
            If true compute immediately, otherwise return a
            ``dask.delayed.Delayed`` object that can be computed later.
        """
        if encoding is None:
            encoding = {}
        from ..backends.api import to_netcdf
        return to_netcdf(self, path, mode, format=format, group=group,
                         engine=engine, encoding=encoding,
                         unlimited_dims=unlimited_dims,
                         compute=compute)

    def to_zarr(self, store=None, mode='w-', synchronizer=None, group=None,
                encoding=None, compute=True, consolidated=False,
                append_dim=None):
        """Write dataset contents to a zarr group.

        .. note:: Experimental
                  The Zarr backend is new and experimental. Please report any
                  unexpected behavior via github issues.

        Parameters
        ----------
        store : MutableMapping or str, optional
            Store or path to directory in file system.
        mode : {'w', 'w-', 'a'}
            Persistence mode: 'w' means create (overwrite if exists);
            'w-' means create (fail if exists);
            'a' means append (create if does not exist).
        synchronizer : object, optional
            Array synchronizer
        group : str, obtional
            Group path. (a.k.a. `path` in zarr terminology.)
        encoding : dict, optional
            Nested dictionary with variable names as keys and dictionaries of
            variable specific encodings as values, e.g.,
            ``{'my_variable': {'dtype': 'int16', 'scale_factor': 0.1,}, ...}``
        compute: bool, optional
            If True compute immediately, otherwise return a
            ``dask.delayed.Delayed`` object that can be computed later.
        consolidated: bool, optional
            If True, apply zarr's `consolidate_metadata` function to the store
            after writing.
        append_dim: str, optional
            If mode='a', the dimension on which the data will be appended.

        References
        ----------
        https://zarr.readthedocs.io/
        """
        if encoding is None:
            encoding = {}
        if mode not in ['w', 'w-', 'a']:
            # TODO: figure out how to handle 'r+'
            raise ValueError("The only supported options for mode are 'w',"
                             "'w-' and 'a'.")
        from ..backends.api import to_zarr
        return to_zarr(self, store=store, mode=mode, synchronizer=synchronizer,
                       group=group, encoding=encoding, compute=compute,
                       consolidated=consolidated, append_dim=append_dim)

    def __repr__(self):
        return formatting.dataset_repr(self)

    def info(self, buf=None):
        """
        Concise summary of a Dataset variables and attributes.

        Parameters
        ----------
        buf : writable buffer, defaults to sys.stdout

        See Also
        --------
        pandas.DataFrame.assign
        ncdump: netCDF's ncdump
        """

        if buf is None:  # pragma: no cover
            buf = sys.stdout

        lines = []
        lines.append('xarray.Dataset {')
        lines.append('dimensions:')
        for name, size in self.dims.items():
            lines.append('\t{name} = {size} ;'.format(name=name, size=size))
        lines.append('\nvariables:')
        for name, da in self.variables.items():
            dims = ', '.join(da.dims)
            lines.append('\t{type} {name}({dims}) ;'.format(
                type=da.dtype, name=name, dims=dims))
            for k, v in da.attrs.items():
                lines.append('\t\t{name}:{k} = {v} ;'.format(name=name, k=k,
                                                             v=v))
        lines.append('\n// global attributes:')
        for k, v in self.attrs.items():
            lines.append('\t:{k} = {v} ;'.format(k=k, v=v))
        lines.append('}')

        buf.write('\n'.join(lines))

    @property
    def chunks(self):
        """Block dimensions for this dataset's data or None if it's not a dask
        array.
        """
        chunks = {}
        for v in self.variables.values():
            if v.chunks is not None:
                for dim, c in zip(v.dims, v.chunks):
                    if dim in chunks and c != chunks[dim]:
                        raise ValueError('inconsistent chunks')
                    chunks[dim] = c
        return Frozen(SortedKeysDict(chunks))

    def chunk(self, chunks=None, name_prefix='xarray-', token=None,
              lock=False):
        """Coerce all arrays in this dataset into dask arrays with the given
        chunks.

        Non-dask arrays in this dataset will be converted to dask arrays. Dask
        arrays will be rechunked to the given chunk sizes.

        If neither chunks is not provided for one or more dimensions, chunk
        sizes along that dimension will not be updated; non-dask arrays will be
        converted into dask arrays with a single block.

        Parameters
        ----------
        chunks : int or dict, optional
            Chunk sizes along each dimension, e.g., ``5`` or
            ``{'x': 5, 'y': 5}``.
        name_prefix : str, optional
            Prefix for the name of any new dask arrays.
        token : str, optional
            Token uniquely identifying this dataset.
        lock : optional
            Passed on to :py:func:`dask.array.from_array`, if the array is not
            already as dask array.

        Returns
        -------
        chunked : xarray.Dataset
        """
        try:
            from dask.base import tokenize
        except ImportError:
            # raise the usual error if dask is entirely missing
            import dask  # noqa
            raise ImportError('xarray requires dask version 0.9 or newer')

        if isinstance(chunks, Number):
            chunks = dict.fromkeys(self.dims, chunks)

        if chunks is not None:
            bad_dims = [d for d in chunks if d not in self.dims]
            if bad_dims:
                raise ValueError('some chunks keys are not dimensions on this '
                                 'object: %s' % bad_dims)

        def selkeys(dict_, keys):
            if dict_ is None:
                return None
            return dict((d, dict_[d]) for d in keys if d in dict_)

        def maybe_chunk(name, var, chunks):
            chunks = selkeys(chunks, var.dims)
            if not chunks:
                chunks = None
            if var.ndim > 0:
                token2 = tokenize(name, token if token else var._data)
                name2 = '%s%s-%s' % (name_prefix, name, token2)
                return var.chunk(chunks, name=name2, lock=lock)
            else:
                return var

        variables = OrderedDict([(k, maybe_chunk(k, v, chunks))
                                 for k, v in self.variables.items()])
        return self._replace(variables)

    def _validate_indexers(
        self, indexers: Mapping,
    ) -> List[Tuple[Any, Union[slice, Variable]]]:
        """ Here we make sure
        + indexer has a valid keys
        + indexer is in a valid data type
        + string indexers are cast to the appropriate date type if the
          associated index is a DatetimeIndex or CFTimeIndex
        """
        from .dataarray import DataArray

        invalid = [k for k in indexers if k not in self.dims]
        if invalid:
            raise ValueError("dimensions %r do not exist" % invalid)

        # all indexers should be int, slice, np.ndarrays, or Variable
        indexers_list = []  # type: List[Tuple[Any, Union[slice, Variable]]]
        for k, v in indexers.items():
            if isinstance(v, slice):
                indexers_list.append((k, v))
                continue

            if isinstance(v, Variable):
                pass
            elif isinstance(v, DataArray):
                v = v.variable
            elif isinstance(v, tuple):
                v = as_variable(v)
            elif isinstance(v, Dataset):
                raise TypeError('cannot use a Dataset as an indexer')
            elif isinstance(v, Sequence) and len(v) == 0:
                v = IndexVariable((k, ), np.zeros((0,), dtype='int64'))
            else:
                v = np.asarray(v)

                if v.dtype.kind == 'U' or v.dtype.kind == 'S':
                    index = self.indexes[k]
                    if isinstance(index, pd.DatetimeIndex):
                        v = v.astype('datetime64[ns]')
                    elif isinstance(index, xr.CFTimeIndex):
                        v = _parse_array_of_cftime_strings(v, index.date_type)

                if v.ndim == 0:
                    v = Variable((), v)
                elif v.ndim == 1:
                    v = IndexVariable((k,), v)
                else:
                    raise IndexError(
                        "Unlabeled multi-dimensional array cannot be "
                        "used for indexing: {}".format(k))

            if v.ndim == 1:
                v = v.to_index_variable()

            indexers_list.append((k, v))

        return indexers_list

    def _get_indexers_coords_and_indexes(self, indexers):
        """  Extract coordinates from indexers.
        Returns an OrderedDict mapping from coordinate name to the
        coordinate variable.

        Only coordinate with a name different from any of self.variables will
        be attached.
        """
        from .dataarray import DataArray

        coord_list = []
        indexes = OrderedDict()
        for k, v in indexers.items():
            if isinstance(v, DataArray):
                v_coords = v.coords
                if v.dtype.kind == 'b':
                    if v.ndim != 1:  # we only support 1-d boolean array
                        raise ValueError(
                            '{:d}d-boolean array is used for indexing along '
                            'dimension {!r}, but only 1d boolean arrays are '
                            'supported.'.format(v.ndim, k))
                    # Make sure in case of boolean DataArray, its
                    # coordinate also should be indexed.
                    v_coords = v[v.values.nonzero()[0]].coords

                coord_list.append({d: v_coords[d].variable for d in v.coords})
                indexes.update(v.indexes)

        # we don't need to call align() explicitly or check indexes for
        # alignment, because merge_variables already checks for exact alignment
        # between dimension coordinates
        coords = merge_variables(coord_list)
        assert_coordinate_consistent(self, coords)

        # silently drop the conflicted variables.
        attached_coords = OrderedDict(
            (k, v) for k, v in coords.items() if k not in self._variables
        )
        attached_indexes = OrderedDict(
            (k, v) for k, v in indexes.items() if k not in self._variables
        )
        return attached_coords, attached_indexes

    def isel(self, indexers=None, drop=False, **indexers_kwargs):
        """Returns a new dataset with each array indexed along the specified
        dimension(s).

        This method selects values from each array using its `__getitem__`
        method, except this method does not require knowing the order of
        each array's dimensions.

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
        **indexers_kwarg : {dim: indexer, ...}, optional
            The keyword arguments form of ``indexers``.
            One of indexers or indexers_kwargs must be provided.

        Returns
        -------
        obj : Dataset
            A new Dataset with the same contents as this dataset, except each
            array and dimension is indexed by the appropriate indexers.
            If indexer DataArrays have coordinates that do not conflict with
            this object, then these coordinates will be attached.
            In general, each array's data will be a view of the array's data
            in this dataset, unless vectorized indexing was triggered by using
            an array indexer, in which case the data will be a copy.

        See Also
        --------
        Dataset.sel
        DataArray.isel
        """

        indexers = either_dict_or_kwargs(indexers, indexers_kwargs, 'isel')

        indexers_list = self._validate_indexers(indexers)

        variables = OrderedDict()
        indexes = OrderedDict()
        for name, var in self.variables.items():
            var_indexers = {k: v for k, v in indexers_list if k in var.dims}
            if drop and name in var_indexers:
                continue  # drop this variable

            if name in self.indexes:
                new_var, new_index = isel_variable_and_index(
                    name, var, self.indexes[name], var_indexers)
                if new_index is not None:
                    indexes[name] = new_index
            else:
                new_var = var.isel(indexers=var_indexers)

            variables[name] = new_var

        coord_names = set(variables).intersection(self._coord_names)
        selected = self._replace_with_new_dims(
            variables, coord_names, indexes)

        # Extract coordinates from indexers
        coord_vars, new_indexes = (
            selected._get_indexers_coords_and_indexes(indexers))
        variables.update(coord_vars)
        indexes.update(new_indexes)
        coord_names = (set(variables)
                       .intersection(self._coord_names)
                       .union(coord_vars))
        return self._replace_with_new_dims(
            variables, coord_names, indexes=indexes)

    def sel(self, indexers=None, method=None, tolerance=None, drop=False,
            **indexers_kwargs):
        """Returns a new dataset with each array indexed by tick labels
        along the specified dimension(s).

        In contrast to `Dataset.isel`, indexers for this method should use
        labels instead of integers.

        Under the hood, this method is powered by using pandas's powerful Index
        objects. This makes label based indexing essentially just as fast as
        using integer indexing.

        It also means this method uses pandas's (well documented) logic for
        indexing. This means you can use string shortcuts for datetime indexes
        (e.g., '2000-01' to select all values in January 2000). It also means
        that slices are treated as inclusive of both the start and stop values,
        unlike normal Python indexing.

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
        method : {None, 'nearest', 'pad'/'ffill', 'backfill'/'bfill'}, optional
            Method to use for inexact matches (requires pandas>=0.16):

            * None (default): only exact matches
            * pad / ffill: propagate last valid index value forward
            * backfill / bfill: propagate next valid index value backward
            * nearest: use nearest valid index value
        tolerance : optional
            Maximum distance between original and new labels for inexact
            matches. The values of the index at the matching locations must
            satisfy the equation ``abs(index[indexer] - target) <= tolerance``.
            Requires pandas>=0.17.
        drop : bool, optional
            If ``drop=True``, drop coordinates variables in `indexers` instead
            of making them scalar.
        **indexers_kwarg : {dim: indexer, ...}, optional
            The keyword arguments form of ``indexers``.
            One of indexers or indexers_kwargs must be provided.

        Returns
        -------
        obj : Dataset
            A new Dataset with the same contents as this dataset, except each
            variable and dimension is indexed by the appropriate indexers.
            If indexer DataArrays have coordinates that do not conflict with
            this object, then these coordinates will be attached.
            In general, each array's data will be a view of the array's data
            in this dataset, unless vectorized indexing was triggered by using
            an array indexer, in which case the data will be a copy.


        See Also
        --------
        Dataset.isel
        DataArray.sel
        """
        indexers = either_dict_or_kwargs(indexers, indexers_kwargs, 'sel')
        pos_indexers, new_indexes = remap_label_indexers(
            self, indexers=indexers, method=method, tolerance=tolerance)
        result = self.isel(indexers=pos_indexers, drop=drop)
        return result._overwrite_indexes(new_indexes)

    def isel_points(self, dim='points', **indexers):
        # type: (...) -> Dataset
        """Returns a new dataset with each array indexed pointwise along the
        specified dimension(s).

        This method selects pointwise values from each array and is akin to
        the NumPy indexing behavior of `arr[[0, 1], [0, 1]]`, except this
        method does not require knowing the order of each array's dimensions.

        Parameters
        ----------
        dim : str or DataArray or pandas.Index or other list-like object, optional
            Name of the dimension to concatenate along. If dim is provided as a
            string, it must be a new dimension name, in which case it is added
            along axis=0. If dim is provided as a DataArray or Index or
            list-like object, its name, which must not be present in the
            dataset, is used as the dimension to concatenate along and the
            values are added as a coordinate.
        **indexers : {dim: indexer, ...}
            Keyword arguments with names matching dimensions and values given
            by array-like objects. All indexers must be the same length and
            1 dimensional.

        Returns
        -------
        obj : Dataset
            A new Dataset with the same contents as this dataset, except each
            array and dimension is indexed by the appropriate indexers. With
            pointwise indexing, the new Dataset will always be a copy of the
            original.

        See Also
        --------
        Dataset.sel
        Dataset.isel
        Dataset.sel_points
        DataArray.isel_points
        """  # noqa
        warnings.warn('Dataset.isel_points is deprecated: use Dataset.isel()'
                      'instead.', DeprecationWarning, stacklevel=2)

        indexer_dims = set(indexers)

        def take(variable, slices):
            # Note: remove helper function when once when numpy
            # supports vindex https://github.com/numpy/numpy/pull/6075
            if hasattr(variable.data, 'vindex'):
                # Special case for dask backed arrays to use vectorised list
                # indexing
                sel = variable.data.vindex[slices]
            else:
                # Otherwise assume backend is numpy array with 'fancy' indexing
                sel = variable.data[slices]
            return sel

        def relevant_keys(mapping):
            return [k for k, v in mapping.items()
                    if any(d in indexer_dims for d in v.dims)]

        coords = relevant_keys(self.coords)
        indexers = [(k, np.asarray(v))  # type: ignore
                    for k, v in indexers.items()]
        indexers_dict = dict(indexers)
        non_indexed_dims = set(self.dims) - indexer_dims
        non_indexed_coords = set(self.coords) - set(coords)

        # All the indexers should be iterables
        # Check that indexers are valid dims, integers, and 1D
        for k, v in indexers:
            if k not in self.dims:
                raise ValueError("dimension %s does not exist" % k)
            if v.dtype.kind != 'i':  # type: ignore
                raise TypeError('Indexers must be integers')
            if v.ndim != 1:  # type: ignore
                raise ValueError('Indexers must be 1 dimensional')

        # all the indexers should have the same length
        lengths = set(len(v) for k, v in indexers)
        if len(lengths) > 1:
            raise ValueError('All indexers must be the same length')

        # Existing dimensions are not valid choices for the dim argument
        if isinstance(dim, str):
            if dim in self.dims:
                # dim is an invalid string
                raise ValueError('Existing dimension names are not valid '
                                 'choices for the dim argument in sel_points')

        elif hasattr(dim, 'dims'):
            # dim is a DataArray or Coordinate
            if dim.name in self.dims:
                # dim already exists
                raise ValueError('Existing dimensions are not valid choices '
                                 'for the dim argument in sel_points')

        # Set the new dim_name, and optionally the new dim coordinate
        # dim is either an array-like or a string
        if not utils.is_scalar(dim):
            # dim is array like get name or assign 'points', get as variable
            dim_name = 'points' if not hasattr(dim, 'name') else dim.name
            dim_coord = as_variable(dim, name=dim_name)
        else:
            # dim is a string
            dim_name = dim
            dim_coord = None  # type: ignore

        reordered = self.transpose(
            *list(indexer_dims), *list(non_indexed_dims))

        variables = OrderedDict()  # type: ignore

        for name, var in reordered.variables.items():
            if name in indexers_dict or any(
                    d in indexer_dims for d in var.dims):
                # slice if var is an indexer or depends on an indexed dim
                slc = [indexers_dict[k]
                       if k in indexers_dict
                       else slice(None) for k in var.dims]

                var_dims = [dim_name] + [d for d in var.dims
                                         if d in non_indexed_dims]
                selection = take(var, tuple(slc))
                var_subset = type(var)(var_dims, selection, var.attrs)
                variables[name] = var_subset
            else:
                # If not indexed just add it back to variables or coordinates
                variables[name] = var

        coord_names = (set(coords) & set(variables)) | non_indexed_coords

        dset = self._replace_vars_and_dims(variables, coord_names=coord_names)
        # Add the dim coord to the new dset. Must be done after creation
        # because_replace_vars_and_dims can only access existing coords,
        # not add new ones
        if dim_coord is not None:
            dset.coords[dim_name] = dim_coord
        return dset

    def sel_points(self, dim='points', method=None, tolerance=None,
                   **indexers):
        """Returns a new dataset with each array indexed pointwise by tick
        labels along the specified dimension(s).

        In contrast to `Dataset.isel_points`, indexers for this method should
        use labels instead of integers.

        In contrast to `Dataset.sel`, this method selects points along the
        diagonal of multi-dimensional arrays, not the intersection.

        Parameters
        ----------
        dim : str or DataArray or pandas.Index or other list-like object, optional
            Name of the dimension to concatenate along. If dim is provided as a
            string, it must be a new dimension name, in which case it is added
            along axis=0. If dim is provided as a DataArray or Index or
            list-like object, its name, which must not be present in the
            dataset, is used as the dimension to concatenate along and the
            values are added as a coordinate.
        method : {None, 'nearest', 'pad'/'ffill', 'backfill'/'bfill'}, optional
            Method to use for inexact matches (requires pandas>=0.16):

            * None (default): only exact matches
            * pad / ffill: propagate last valid index value forward
            * backfill / bfill: propagate next valid index value backward
            * nearest: use nearest valid index value
        tolerance : optional
            Maximum distance between original and new labels for inexact
            matches. The values of the index at the matching locations must
            satisfy the equation ``abs(index[indexer] - target) <= tolerance``.
            Requires pandas>=0.17.
        **indexers : {dim: indexer, ...}
            Keyword arguments with names matching dimensions and values given
            by array-like objects. All indexers must be the same length and
            1 dimensional.

        Returns
        -------
        obj : Dataset
            A new Dataset with the same contents as this dataset, except each
            array and dimension is indexed by the appropriate indexers. With
            pointwise indexing, the new Dataset will always be a copy of the
            original.

        See Also
        --------
        Dataset.sel
        Dataset.isel
        Dataset.isel_points
        DataArray.sel_points
        """  # noqa
        warnings.warn('Dataset.sel_points is deprecated: use Dataset.sel()'
                      'instead.', DeprecationWarning, stacklevel=2)

        pos_indexers, _ = indexing.remap_label_indexers(
            self, indexers, method=method, tolerance=tolerance
        )
        return self.isel_points(dim=dim, **pos_indexers)

    def reindex_like(self, other, method=None, tolerance=None, copy=True,
                     fill_value=dtypes.NA):
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
        method : {None, 'nearest', 'pad'/'ffill', 'backfill'/'bfill'}, optional
            Method to use for filling index values from other not found in this
            dataset:

            * None (default): don't fill gaps
            * pad / ffill: propagate last valid index value forward
            * backfill / bfill: propagate next valid index value backward
            * nearest: use nearest valid index value (requires pandas>=0.16)
        tolerance : optional
            Maximum distance between original and new labels for inexact
            matches. The values of the index at the matching locations must
            satisfy the equation ``abs(index[indexer] - target) <= tolerance``.
            Requires pandas>=0.17.
        copy : bool, optional
            If ``copy=True``, data in the return value is always copied. If
            ``copy=False`` and reindexing is unnecessary, or can be performed
            with only slice operations, then the output may share memory with
            the input. In either case, a new xarray object is always returned.
        fill_value : scalar, optional
            Value to use for newly missing values

        Returns
        -------
        reindexed : Dataset
            Another dataset, with this dataset's data but coordinates from the
            other object.

        See Also
        --------
        Dataset.reindex
        align
        """
        indexers = alignment.reindex_like_indexers(self, other)
        return self.reindex(indexers=indexers, method=method, copy=copy,
                            fill_value=fill_value, tolerance=tolerance)

    def reindex(self, indexers=None, method=None, tolerance=None, copy=True,
                fill_value=dtypes.NA, **indexers_kwargs):
        """Conform this object onto a new set of indexes, filling in
        missing values with ``fill_value``. The default fill value is NaN.

        Parameters
        ----------
        indexers : dict. optional
            Dictionary with keys given by dimension names and values given by
            arrays of coordinates tick labels. Any mis-matched coordinate
            values will be filled in with NaN, and any mis-matched dimension
            names will simply be ignored.
            One of indexers or indexers_kwargs must be provided.
        method : {None, 'nearest', 'pad'/'ffill', 'backfill'/'bfill'}, optional
            Method to use for filling index values in ``indexers`` not found in
            this dataset:

            * None (default): don't fill gaps
            * pad / ffill: propagate last valid index value forward
            * backfill / bfill: propagate next valid index value backward
            * nearest: use nearest valid index value (requires pandas>=0.16)
        tolerance : optional
            Maximum distance between original and new labels for inexact
            matches. The values of the index at the matching locations must
            satisfy the equation ``abs(index[indexer] - target) <= tolerance``.
            Requires pandas>=0.17.
        copy : bool, optional
            If ``copy=True``, data in the return value is always copied. If
            ``copy=False`` and reindexing is unnecessary, or can be performed
            with only slice operations, then the output may share memory with
            the input. In either case, a new xarray object is always returned.
        fill_value : scalar, optional
            Value to use for newly missing values
        **indexers_kwarg : {dim: indexer, ...}, optional
            Keyword arguments in the same form as ``indexers``.
            One of indexers or indexers_kwargs must be provided.

        Returns
        -------
        reindexed : Dataset
            Another dataset, with this dataset's data but replaced coordinates.

        See Also
        --------
        Dataset.reindex_like
        align
        pandas.Index.get_indexer
        """
        indexers = utils.either_dict_or_kwargs(indexers, indexers_kwargs,
                                               'reindex')

        bad_dims = [d for d in indexers if d not in self.dims]
        if bad_dims:
            raise ValueError('invalid reindex dimensions: %s' % bad_dims)

        variables, indexes = alignment.reindex_variables(
            self.variables, self.sizes, self.indexes, indexers, method,
            tolerance, copy=copy, fill_value=fill_value)
        coord_names = set(self._coord_names)
        coord_names.update(indexers)
        return self._replace_with_new_dims(
            variables, coord_names, indexes=indexes)

    def interp(self, coords=None, method='linear', assume_sorted=False,
               kwargs=None, **coords_kwargs):
        """ Multidimensional interpolation of Dataset.

        Parameters
        ----------
        coords : dict, optional
            Mapping from dimension names to the new coordinates.
            New coordinate can be a scalar, array-like or DataArray.
            If DataArrays are passed as new coordates, their dimensions are
            used for the broadcasting.
        method: string, optional.
            {'linear', 'nearest'} for multidimensional array,
            {'linear', 'nearest', 'zero', 'slinear', 'quadratic', 'cubic'}
            for 1-dimensional array. 'linear' is used by default.
        assume_sorted: boolean, optional
            If False, values of coordinates that are interpolated over can be
            in any order and they are sorted first. If True, interpolated
            coordinates are assumed to be an array of monotonically increasing
            values.
        kwargs: dictionary, optional
            Additional keyword passed to scipy's interpolator.
        **coords_kwarg : {dim: coordinate, ...}, optional
            The keyword arguments form of ``coords``.
            One of coords or coords_kwargs must be provided.

        Returns
        -------
        interpolated: xr.Dataset
            New dataset on the new coordinates.

        Notes
        -----
        scipy is required.

        See Also
        --------
        scipy.interpolate.interp1d
        scipy.interpolate.interpn
        """
        from . import missing

        if kwargs is None:
            kwargs = {}
        coords = either_dict_or_kwargs(coords, coords_kwargs, 'interp')
        indexers = OrderedDict(self._validate_indexers(coords))

        obj = self if assume_sorted else self.sortby([k for k in coords])

        def maybe_variable(obj, k):
            # workaround to get variable for dimension without coordinate.
            try:
                return obj._variables[k]
            except KeyError:
                return as_variable((k, range(obj.dims[k])))

        def _validate_interp_indexer(x, new_x):
            # In the case of datetimes, the restrictions placed on indexers
            # used with interp are stronger than those which are placed on
            # isel, so we need an additional check after _validate_indexers.
            if (_contains_datetime_like_objects(x)
                    and not _contains_datetime_like_objects(new_x)):
                raise TypeError('When interpolating over a datetime-like '
                                'coordinate, the coordinates to '
                                'interpolate to must be either datetime '
                                'strings or datetimes. '
                                'Instead got\n{}'.format(new_x))
            else:
                return (x, new_x)

        variables = OrderedDict()
        for name, var in obj._variables.items():
            if name not in indexers:
                if var.dtype.kind in 'uifc':
                    var_indexers = {
                        k: _validate_interp_indexer(maybe_variable(obj, k), v)
                        for k, v in indexers.items()
                        if k in var.dims
                    }
                    variables[name] = missing.interp(
                        var, var_indexers, method, **kwargs)
                elif all(d not in indexers for d in var.dims):
                    # keep unrelated object array
                    variables[name] = var

        coord_names = set(variables).intersection(obj._coord_names)
        indexes = OrderedDict(
            (k, v) for k, v in obj.indexes.items() if k not in indexers)
        selected = self._replace_with_new_dims(
            variables.copy(), coord_names, indexes=indexes)

        # attach indexer as coordinate
        variables.update(indexers)
        indexes.update(
            (k, v.to_index()) for k, v in indexers.items() if v.dims == (k,)
        )

        # Extract coordinates from indexers
        coord_vars, new_indexes = (
            selected._get_indexers_coords_and_indexes(coords))
        variables.update(coord_vars)
        indexes.update(new_indexes)

        coord_names = (set(variables)
                       .intersection(obj._coord_names)
                       .union(coord_vars))
        return self._replace_with_new_dims(
            variables, coord_names, indexes=indexes)

    def interp_like(self, other, method='linear', assume_sorted=False,
                    kwargs=None):
        """Interpolate this object onto the coordinates of another object,
        filling the out of range values with NaN.

        Parameters
        ----------
        other : Dataset or DataArray
            Object with an 'indexes' attribute giving a mapping from dimension
            names to an 1d array-like, which provides coordinates upon
            which to index the variables in this dataset.
        method: string, optional.
            {'linear', 'nearest'} for multidimensional array,
            {'linear', 'nearest', 'zero', 'slinear', 'quadratic', 'cubic'}
            for 1-dimensional array. 'linear' is used by default.
        assume_sorted: boolean, optional
            If False, values of coordinates that are interpolated over can be
            in any order and they are sorted first. If True, interpolated
            coordinates are assumed to be an array of monotonically increasing
            values.
        kwargs: dictionary, optional
            Additional keyword passed to scipy's interpolator.

        Returns
        -------
        interpolated: xr.Dataset
            Another dataset by interpolating this dataset's data along the
            coordinates of the other object.

        Notes
        -----
        scipy is required.
        If the dataset has object-type coordinates, reindex is used for these
        coordinates instead of the interpolation.

        See Also
        --------
        Dataset.interp
        Dataset.reindex_like
        """
        if kwargs is None:
            kwargs = {}
        coords = alignment.reindex_like_indexers(self, other)

        numeric_coords = OrderedDict()
        object_coords = OrderedDict()
        for k, v in coords.items():
            if v.dtype.kind in 'uifcMm':
                numeric_coords[k] = v
            else:
                object_coords[k] = v

        ds = self
        if object_coords:
            # We do not support interpolation along object coordinate.
            # reindex instead.
            ds = self.reindex(object_coords)
        return ds.interp(numeric_coords, method, assume_sorted, kwargs)

    # Helper methods for rename()
    def _rename_vars(self, name_dict, dims_dict):
        variables = OrderedDict()
        coord_names = set()
        for k, v in self.variables.items():
            var = v.copy(deep=False)
            var.dims = tuple(dims_dict.get(dim, dim) for dim in v.dims)
            name = name_dict.get(k, k)
            if name in variables:
                raise ValueError('the new name %r conflicts' % (name,))
            variables[name] = var
            if k in self._coord_names:
                coord_names.add(name)
        return variables, coord_names

    def _rename_dims(self, name_dict):
        return {name_dict.get(k, k): v for k, v in self.dims.items()}

    def _rename_indexes(self, name_dict):
        if self._indexes is None:
            return None
        indexes = OrderedDict()
        for k, v in self.indexes.items():
            new_name = name_dict.get(k, k)
            if isinstance(v, pd.MultiIndex):
                new_names = [name_dict.get(k, k) for k in v.names]
                index = pd.MultiIndex(v.levels, v.labels, v.sortorder,
                                      names=new_names, verify_integrity=False)
            else:
                index = pd.Index(v, name=new_name)
            indexes[new_name] = index
        return indexes

    def _rename_all(self, name_dict, dims_dict):
        variables, coord_names = self._rename_vars(name_dict, dims_dict)
        dims = self._rename_dims(dims_dict)
        indexes = self._rename_indexes(name_dict)
        return variables, coord_names, dims, indexes

    def rename(self, name_dict=None, inplace=None, **names):
        """Returns a new object with renamed variables and dimensions.

        Parameters
        ----------
        name_dict : dict-like, optional
            Dictionary whose keys are current variable or dimension names and
            whose values are the desired names.
        inplace : bool, optional
            If True, rename variables and dimensions in-place. Otherwise,
            return a new dataset object.
        **names, optional
            Keyword form of ``name_dict``.
            One of name_dict or names must be provided.

        Returns
        -------
        renamed : Dataset
            Dataset with renamed variables and dimensions.

        See Also
        --------
        Dataset.swap_dims
        Dataset.rename_vars
        Dataset.rename_dims
        DataArray.rename
        """
        inplace = _check_inplace(inplace)
        name_dict = either_dict_or_kwargs(name_dict, names, 'rename')
        for k in name_dict.keys():
            if k not in self and k not in self.dims:
                raise ValueError("cannot rename %r because it is not a "
                                 "variable or dimension in this dataset" % k)

        variables, coord_names, dims, indexes = self._rename_all(
            name_dict=name_dict, dims_dict=name_dict)
        return self._replace(variables, coord_names, dims=dims,
                             indexes=indexes, inplace=inplace)

    def rename_dims(self, dims_dict=None, **dims):
        """Returns a new object with renamed dimensions only.

        Parameters
        ----------
        dims_dict : dict-like, optional
            Dictionary whose keys are current dimension names and
            whose values are the desired names.
        **dims, optional
            Keyword form of ``dims_dict``.
            One of dims_dict or dims must be provided.

        Returns
        -------
        renamed : Dataset
            Dataset with renamed dimensions.

        See Also
        --------
        Dataset.swap_dims
        Dataset.rename
        Dataset.rename_vars
        DataArray.rename
        """
        dims_dict = either_dict_or_kwargs(dims_dict, dims, 'rename_dims')
        for k in dims_dict:
            if k not in self.dims:
                raise ValueError("cannot rename %r because it is not a "
                                 "dimension in this dataset" % k)

        variables, coord_names, dims, indexes = self._rename_all(
            name_dict={}, dims_dict=dims_dict)
        return self._replace(variables, coord_names, dims=dims,
                             indexes=indexes)

    def rename_vars(self, name_dict=None, **names):
        """Returns a new object with renamed variables including coordinates

        Parameters
        ----------
        name_dict : dict-like, optional
            Dictionary whose keys are current variable or coordinate names and
            whose values are the desired names.
        **names, optional
            Keyword form of ``name_dict``.
            One of name_dict or names must be provided.

        Returns
        -------
        renamed : Dataset
            Dataset with renamed variables including coordinates

        See Also
        --------
        Dataset.swap_dims
        Dataset.rename
        Dataset.rename_dims
        DataArray.rename
        """
        name_dict = either_dict_or_kwargs(name_dict, names, 'rename_vars')
        for k in name_dict:
            if k not in self:
                raise ValueError("cannot rename %r because it is not a "
                                 "variable or coordinate in this dataset" % k)
        variables, coord_names, dims, indexes = self._rename_all(
            name_dict=name_dict, dims_dict={})
        return self._replace(variables, coord_names, dims=dims,
                             indexes=indexes)

    def swap_dims(self, dims_dict, inplace=None):
        """Returns a new object with swapped dimensions.

        Parameters
        ----------
        dims_dict : dict-like
            Dictionary whose keys are current dimension names and whose values
            are new names. Each value must already be a variable in the
            dataset.
        inplace : bool, optional
            If True, swap dimensions in-place. Otherwise, return a new dataset
            object.

        Returns
        -------
        renamed : Dataset
            Dataset with swapped dimensions.

        See Also
        --------

        Dataset.rename
        DataArray.swap_dims
        """
        # TODO: deprecate this method in favor of a (less confusing)
        # rename_dims() method that only renames dimensions.
        inplace = _check_inplace(inplace)
        for k, v in dims_dict.items():
            if k not in self.dims:
                raise ValueError('cannot swap from dimension %r because it is '
                                 'not an existing dimension' % k)
            if self.variables[v].dims != (k,):
                raise ValueError('replacement dimension %r is not a 1D '
                                 'variable along the old dimension %r'
                                 % (v, k))

        result_dims = set(dims_dict.get(dim, dim) for dim in self.dims)

        coord_names = self._coord_names.copy()
        coord_names.update(dims_dict.values())

        variables = OrderedDict()
        indexes = OrderedDict()
        for k, v in self.variables.items():
            dims = tuple(dims_dict.get(dim, dim) for dim in v.dims)
            if k in result_dims:
                var = v.to_index_variable()
                if k in self.indexes:
                    indexes[k] = self.indexes[k]
                else:
                    indexes[k] = var.to_index()
            else:
                var = v.to_base_variable()
            var.dims = dims
            variables[k] = var

        return self._replace_with_new_dims(variables, coord_names,
                                           indexes=indexes, inplace=inplace)

    def expand_dims(self, dim=None, axis=None, **dim_kwargs):
        """Return a new object with an additional axis (or axes) inserted at
        the corresponding position in the array shape.

        If dim is already a scalar coordinate, it will be promoted to a 1D
        coordinate consisting of a single value.

        Parameters
        ----------
        dim : str, sequence of str, dict, or None
            Dimensions to include on the new variable.
            If provided as str or sequence of str, then dimensions are inserted
            with length 1. If provided as a dict, then the keys are the new
            dimensions and the values are either integers (giving the length of
            the new dimensions) or sequence/ndarray (giving the coordinates of
            the new dimensions). **WARNING** for python 3.5, if ``dim`` is
            dict-like, then it must be an ``OrderedDict``. This is to ensure
            that the order in which the dims are given is maintained.
        axis : integer, list (or tuple) of integers, or None
            Axis position(s) where new axis is to be inserted (position(s) on
            the result array). If a list (or tuple) of integers is passed,
            multiple axes are inserted. In this case, dim arguments should be
            same length list. If axis=None is passed, all the axes will be
            inserted to the start of the result array.
        **dim_kwargs : int or sequence/ndarray
            The keywords are arbitrary dimensions being inserted and the values
            are either the lengths of the new dims (if int is given), or their
            coordinates. Note, this is an alternative to passing a dict to the
            dim kwarg and will only be used if dim is None. **WARNING** for
            python 3.5 ``dim_kwargs`` is not available.

        Returns
        -------
        expanded : same type as caller
            This object, but with an additional dimension(s).
        """
        if isinstance(dim, int):
            raise TypeError('dim should be str or sequence of strs or dict')
        elif isinstance(dim, str):
            dim = OrderedDict(((dim, 1),))
        elif isinstance(dim, (list, tuple)):
            if len(dim) != len(set(dim)):
                raise ValueError('dims should not contain duplicate values.')
            dim = OrderedDict(((d, 1) for d in dim))

        # TODO: get rid of the below code block when python 3.5 is no longer
        #   supported.
        python36_plus = sys.version_info[0] == 3 and sys.version_info[1] > 5
        not_ordereddict = dim is not None and not isinstance(dim, OrderedDict)
        if not python36_plus and not_ordereddict:
            raise TypeError("dim must be an OrderedDict for python <3.6")
        elif not python36_plus and dim_kwargs:
            raise ValueError("dim_kwargs isn't available for python <3.6")

        dim = either_dict_or_kwargs(dim, dim_kwargs, 'expand_dims')

        if axis is not None and not isinstance(axis, (list, tuple)):
            axis = [axis]

        if axis is None:
            axis = list(range(len(dim)))

        if len(dim) != len(axis):
            raise ValueError('lengths of dim and axis should be identical.')
        for d in dim:
            if d in self.dims:
                raise ValueError(
                    'Dimension {dim} already exists.'.format(dim=d))
            if (d in self._variables
                    and not utils.is_scalar(self._variables[d])):
                raise ValueError(
                    '{dim} already exists as coordinate or'
                    ' variable name.'.format(dim=d))

        variables = OrderedDict()
        coord_names = self._coord_names.copy()
        # If dim is a dict, then ensure that the values are either integers
        # or iterables.
        for k, v in dim.items():
            if hasattr(v, "__iter__"):
                # If the value for the new dimension is an iterable, then
                # save the coordinates to the variables dict, and set the
                # value within the dim dict to the length of the iterable
                # for later use.
                variables[k] = xr.IndexVariable((k,), v)
                coord_names.add(k)
                dim[k] = variables[k].size
            elif isinstance(v, int):
                pass  # Do nothing if the dimensions value is just an int
            else:
                raise TypeError('The value of new dimension {k} must be '
                                'an iterable or an int'.format(k=k))

        for k, v in self._variables.items():
            if k not in dim:
                if k in coord_names:  # Do not change coordinates
                    variables[k] = v
                else:
                    result_ndim = len(v.dims) + len(axis)
                    for a in axis:
                        if a < -result_ndim or result_ndim - 1 < a:
                            raise IndexError(
                                'Axis {a} is out of bounds of the expanded'
                                ' dimension size {dim}.'.format(
                                    a=a, v=k, dim=result_ndim))

                    axis_pos = [a if a >= 0 else result_ndim + a
                                for a in axis]
                    if len(axis_pos) != len(set(axis_pos)):
                        raise ValueError('axis should not contain duplicate'
                                         ' values.')
                    # We need to sort them to make sure `axis` equals to the
                    # axis positions of the result array.
                    zip_axis_dim = sorted(zip(axis_pos, dim.items()))

                    all_dims = list(zip(v.dims, v.shape))
                    for d, c in zip_axis_dim:
                        all_dims.insert(d, c)
                    all_dims = OrderedDict(all_dims)

                    variables[k] = v.set_dims(all_dims)
            else:
                # If dims includes a label of a non-dimension coordinate,
                # it will be promoted to a 1D coordinate with a single value.
                variables[k] = v.set_dims(k).to_index_variable()

        new_dims = self._dims.copy()
        new_dims.update(dim)

        return self._replace_vars_and_dims(
            variables, dims=new_dims, coord_names=coord_names)

    def set_index(self, indexes=None, append=False, inplace=None,
                  **indexes_kwargs):
        """Set Dataset (multi-)indexes using one or more existing coordinates
        or variables.

        Parameters
        ----------
        indexes : {dim: index, ...}
            Mapping from names matching dimensions and values given
            by (lists of) the names of existing coordinates or variables to set
            as new (multi-)index.
        append : bool, optional
            If True, append the supplied index(es) to the existing index(es).
            Otherwise replace the existing index(es) (default).
        inplace : bool, optional
            If True, set new index(es) in-place. Otherwise, return a new
            Dataset object.
        **indexes_kwargs: optional
            The keyword arguments form of ``indexes``.
            One of indexes or indexes_kwargs must be provided.

        Returns
        -------
        obj : Dataset
            Another dataset, with this dataset's data but replaced coordinates.

        See Also
        --------
        Dataset.reset_index
        Dataset.swap_dims
        """
        inplace = _check_inplace(inplace)
        indexes = either_dict_or_kwargs(indexes, indexes_kwargs, 'set_index')
        variables, coord_names = merge_indexes(indexes, self._variables,
                                               self._coord_names,
                                               append=append)
        return self._replace_vars_and_dims(variables, coord_names=coord_names,
                                           inplace=inplace)

    def reset_index(self, dims_or_levels, drop=False, inplace=None):
        """Reset the specified index(es) or multi-index level(s).

        Parameters
        ----------
        dims_or_levels : str or list
            Name(s) of the dimension(s) and/or multi-index level(s) that will
            be reset.
        drop : bool, optional
            If True, remove the specified indexes and/or multi-index levels
            instead of extracting them as new coordinates (default: False).
        inplace : bool, optional
            If True, modify the dataset in-place. Otherwise, return a new
            Dataset object.

        Returns
        -------
        obj : Dataset
            Another dataset, with this dataset's data but replaced coordinates.

        See Also
        --------
        Dataset.set_index
        """
        inplace = _check_inplace(inplace)
        variables, coord_names = split_indexes(dims_or_levels, self._variables,
                                               self._coord_names,
                                               self._level_coords, drop=drop)
        return self._replace_vars_and_dims(variables, coord_names=coord_names,
                                           inplace=inplace)

    def reorder_levels(self, dim_order=None, inplace=None,
                       **dim_order_kwargs):
        """Rearrange index levels using input order.

        Parameters
        ----------
        dim_order : optional
            Mapping from names matching dimensions and values given
            by lists representing new level orders. Every given dimension
            must have a multi-index.
        inplace : bool, optional
            If True, modify the dataset in-place. Otherwise, return a new
            DataArray object.
        **dim_order_kwargs: optional
            The keyword arguments form of ``dim_order``.
            One of dim_order or dim_order_kwargs must be provided.

        Returns
        -------
        obj : Dataset
            Another dataset, with this dataset's data but replaced
            coordinates.
        """
        inplace = _check_inplace(inplace)
        dim_order = either_dict_or_kwargs(dim_order, dim_order_kwargs,
                                          'reorder_levels')
        variables = self._variables.copy()
        indexes = OrderedDict(self.indexes)
        for dim, order in dim_order.items():
            coord = self._variables[dim]
            index = self.indexes[dim]
            if not isinstance(index, pd.MultiIndex):
                raise ValueError("coordinate %r has no MultiIndex" % dim)
            new_index = index.reorder_levels(order)
            variables[dim] = IndexVariable(coord.dims, new_index)
            indexes[dim] = new_index

        return self._replace(variables, indexes=indexes, inplace=inplace)

    def _stack_once(self, dims, new_dim):
        variables = OrderedDict()
        for name, var in self.variables.items():
            if name not in dims:
                if any(d in var.dims for d in dims):
                    add_dims = [d for d in dims if d not in var.dims]
                    vdims = list(var.dims) + add_dims
                    shape = [self.dims[d] for d in vdims]
                    exp_var = var.set_dims(vdims, shape)
                    stacked_var = exp_var.stack(**{new_dim: dims})
                    variables[name] = stacked_var
                else:
                    variables[name] = var.copy(deep=False)

        # consider dropping levels that are unused?
        levels = [self.get_index(dim) for dim in dims]
        if LooseVersion(pd.__version__) < LooseVersion('0.19.0'):
            # RangeIndex levels in a MultiIndex are broken for appending in
            # pandas before v0.19.0
            levels = [pd.Int64Index(level)
                      if isinstance(level, pd.RangeIndex)
                      else level
                      for level in levels]
        idx = utils.multiindex_from_product_levels(levels, names=dims)
        variables[new_dim] = IndexVariable(new_dim, idx)

        coord_names = set(self._coord_names) - set(dims) | set([new_dim])

        indexes = OrderedDict((k, v) for k, v in self.indexes.items()
                              if k not in dims)
        indexes[new_dim] = idx

        return self._replace_with_new_dims(
            variables, coord_names=coord_names, indexes=indexes)

    def stack(self, dimensions=None, **dimensions_kwargs):
        """
        Stack any number of existing dimensions into a single new dimension.

        New dimensions will be added at the end, and the corresponding
        coordinate variables will be combined into a MultiIndex.

        Parameters
        ----------
        dimensions : Mapping of the form new_name=(dim1, dim2, ...)
            Names of new dimensions, and the existing dimensions that they
            replace.
        **dimensions_kwargs:
            The keyword arguments form of ``dimensions``.
            One of dimensions or dimensions_kwargs must be provided.

        Returns
        -------
        stacked : Dataset
            Dataset with stacked data.

        See also
        --------
        Dataset.unstack
        """
        dimensions = either_dict_or_kwargs(dimensions, dimensions_kwargs,
                                           'stack')
        result = self
        for new_dim, dims in dimensions.items():
            result = result._stack_once(dims, new_dim)
        return result

    def to_stacked_array(self, new_dim, sample_dims, variable_dim='variable',
                         name=None):
        """Combine variables of differing dimensionality into a DataArray
        without broadcasting.

        This method is similar to Dataset.to_array but does not broadcast the
        variables.

        Parameters
        ----------
        new_dim : str
            Name of the new stacked coordinate
        sample_dims : Sequence[str]
            Dimensions that **will not** be stacked. Each array in the dataset
            must share these dimensions. For machine learning applications,
            these define the dimensions over which samples are drawn.
        variable_dim : str, optional
            Name of the level in the stacked coordinate which corresponds to
            the variables.
        name : str, optional
            Name of the new data array.

        Returns
        -------
        stacked : DataArray
            DataArray with the specified dimensions and data variables
            stacked together. The stacked coordinate is named ``new_dim``
            and represented by a MultiIndex object with a level containing the
            data variable names. The name of this level is controlled using
            the ``variable_dim`` argument.

        See Also
        --------
        Dataset.to_array
        Dataset.stack
        DataArray.to_unstacked_dataset

        Examples
        --------
        >>> data = Dataset(
        ...     data_vars={'a': (('x', 'y'), [[0, 1, 2], [3, 4, 5]]),
        ...                'b': ('x', [6, 7])},
        ...     coords={'y': ['u', 'v', 'w']}
        ... )

        >>> data
        <xarray.Dataset>
        Dimensions:  (x: 2, y: 3)
        Coordinates:
        * y        (y) <U1 'u' 'v' 'w'
        Dimensions without coordinates: x
        Data variables:
            a        (x, y) int64 0 1 2 3 4 5
            b        (x) int64 6 7

        >>> data.to_stacked_array("z", sample_dims=['x'])
        <xarray.DataArray (x: 2, z: 4)>
        array([[0, 1, 2, 6],
            [3, 4, 5, 7]])
        Coordinates:
        * z         (z) MultiIndex
        - variable  (z) object 'a' 'a' 'a' 'b'
        - y         (z) object 'u' 'v' 'w' nan
        Dimensions without coordinates: x

        """
        stacking_dims = tuple(dim for dim in self.dims
                              if dim not in sample_dims)

        for variable in self:
            dims = self[variable].dims
            dims_include_sample_dims = set(sample_dims) <= set(dims)
            if not dims_include_sample_dims:
                raise ValueError(
                    "All variables in the dataset must contain the "
                    "dimensions {}.".format(dims)
                )

        def ensure_stackable(val):
            assign_coords = {variable_dim: val.name}
            for dim in stacking_dims:
                if dim not in val.dims:
                    assign_coords[dim] = None

            expand_dims = set(stacking_dims).difference(set(val.dims))
            expand_dims.add(variable_dim)
            # must be list for .expand_dims
            expand_dims = list(expand_dims)

            return (val.assign_coords(**assign_coords)
                    .expand_dims(expand_dims)
                    .stack({new_dim: (variable_dim,) + stacking_dims}))

        # concatenate the arrays
        stackable_vars = [ensure_stackable(self[key])
                          for key in self.data_vars]
        data_array = xr.concat(stackable_vars, dim=new_dim)

        # coerce the levels of the MultiIndex to have the same type as the
        # input dimensions. This code is messy, so it might be better to just
        # input a dummy value for the singleton dimension.
        idx = data_array.indexes[new_dim]
        levels = ([idx.levels[0]]
                  + [level.astype(self[level.name].dtype)
                     for level in idx.levels[1:]])
        new_idx = idx.set_levels(levels)
        data_array[new_dim] = IndexVariable(new_dim, new_idx)

        if name is not None:
            data_array.name = name

        return data_array

    def _unstack_once(self, dim):
        index = self.get_index(dim)
        # GH2619. For MultiIndex, we need to call remove_unused.
        if LooseVersion(pd.__version__) >= "0.20":
            index = index.remove_unused_levels()
        else:  # for pandas 0.19
            index = pdcompat.remove_unused_levels(index)

        full_idx = pd.MultiIndex.from_product(index.levels, names=index.names)

        # take a shortcut in case the MultiIndex was not modified.
        if index.equals(full_idx):
            obj = self
        else:
            obj = self.reindex({dim: full_idx}, copy=False)

        new_dim_names = index.names
        new_dim_sizes = [lev.size for lev in index.levels]

        variables = OrderedDict()
        indexes = OrderedDict(
            (k, v) for k, v in self.indexes.items() if k != dim)

        for name, var in obj.variables.items():
            if name != dim:
                if dim in var.dims:
                    new_dims = OrderedDict(zip(new_dim_names, new_dim_sizes))
                    variables[name] = var.unstack({dim: new_dims})
                else:
                    variables[name] = var

        for name, lev in zip(new_dim_names, index.levels):
            variables[name] = IndexVariable(name, lev)
            indexes[name] = lev

        coord_names = set(self._coord_names) - set([dim]) | set(new_dim_names)

        return self._replace_with_new_dims(
            variables, coord_names=coord_names, indexes=indexes)

    def unstack(self, dim=None):
        """
        Unstack existing dimensions corresponding to MultiIndexes into
        multiple new dimensions.

        New dimensions will be added at the end.

        Parameters
        ----------
        dim : str or sequence of str, optional
            Dimension(s) over which to unstack. By default unstacks all
            MultiIndexes.

        Returns
        -------
        unstacked : Dataset
            Dataset with unstacked data.

        See also
        --------
        Dataset.stack
        """

        if dim is None:
            dims = [d for d in self.dims if isinstance(self.get_index(d),
                                                       pd.MultiIndex)]
        else:
            dims = [dim] if isinstance(dim, str) else dim

            missing_dims = [d for d in dims if d not in self.dims]
            if missing_dims:
                raise ValueError('Dataset does not contain the dimensions: %s'
                                 % missing_dims)

            non_multi_dims = [d for d in dims if not
                              isinstance(self.get_index(d), pd.MultiIndex)]
            if non_multi_dims:
                raise ValueError('cannot unstack dimensions that do not '
                                 'have a MultiIndex: %s' % non_multi_dims)

        result = self.copy(deep=False)
        for dim in dims:
            result = result._unstack_once(dim)
        return result

    def update(
        self,
        other: Union[
            'Dataset',
            Mapping[Hashable, Union[
                'DataArray',
                Variable,
                Tuple[Hashable, Any],
                Tuple[Sequence[Hashable], Any],
            ]],
        ],
        inplace: bool = None
    ) -> 'Dataset':
        """Update this dataset's variables with those from another dataset.

        Parameters
        ----------
        other : Dataset or castable to Dataset
            Variables with which to update this dataset. One of:

            - Dataset
            - mapping {var name: DataArray}
            - mapping {var name: Variable}
            - mapping {var name: (dimension name, array-like)}
            - mapping {var name: (tuple of dimension names, array-like)}

        inplace : bool, optional
            If True, merge the other dataset into this dataset in-place.
            Otherwise, return a new dataset object.

        Returns
        -------
        updated : Dataset
            Updated dataset.

        Raises
        ------
        ValueError
            If any dimensions would have inconsistent sizes in the updated
            dataset.
        """
        inplace = _check_inplace(inplace, default=True)
        variables, coord_names, dims = dataset_update_method(self, other)

        return self._replace_vars_and_dims(variables, coord_names, dims,
                                           inplace=inplace)

    def merge(self, other, inplace=None, overwrite_vars=frozenset(),
              compat='no_conflicts', join='outer', fill_value=dtypes.NA):
        """Merge the arrays of two datasets into a single dataset.

        This method generally not allow for overriding data, with the exception
        of attributes, which are ignored on the second dataset. Variables with
        the same name are checked for conflicts via the equals or identical
        methods.

        Parameters
        ----------
        other : Dataset or castable to Dataset
            Dataset or variables to merge with this dataset.
        inplace : bool, optional
            If True, merge the other dataset into this dataset in-place.
            Otherwise, return a new dataset object.
        overwrite_vars : str or sequence, optional
            If provided, update variables of these name(s) without checking for
            conflicts in this dataset.
        compat : {'broadcast_equals', 'equals', 'identical',
                  'no_conflicts'}, optional
            String indicating how to compare variables of the same name for
            potential conflicts:
            - 'broadcast_equals': all values must be equal when variables are
              broadcast against each other to ensure common dimensions.
            - 'equals': all values and dimensions must be the same.
            - 'identical': all values, dimensions and attributes must be the
              same.
            - 'no_conflicts': only values which are not null in both datasets
              must be equal. The returned dataset then contains the combination
              of all non-null values.
        join : {'outer', 'inner', 'left', 'right', 'exact'}, optional
            Method for joining ``self`` and ``other`` along shared dimensions:

            - 'outer': use the union of the indexes
            - 'inner': use the intersection of the indexes
            - 'left': use indexes from ``self``
            - 'right': use indexes from ``other``
            - 'exact': error instead of aligning non-equal indexes
        fill_value: scalar, optional
            Value to use for newly missing values

        Returns
        -------
        merged : Dataset
            Merged dataset.

        Raises
        ------
        MergeError
            If any variables conflict (see ``compat``).
        """
        inplace = _check_inplace(inplace)
        variables, coord_names, dims = dataset_merge_method(
            self, other, overwrite_vars=overwrite_vars, compat=compat,
            join=join, fill_value=fill_value)

        return self._replace_vars_and_dims(variables, coord_names, dims,
                                           inplace=inplace)

    def _assert_all_in_dataset(self, names, virtual_okay=False):
        bad_names = set(names) - set(self._variables)
        if virtual_okay:
            bad_names -= self.virtual_variables
        if bad_names:
            raise ValueError('One or more of the specified variables '
                             'cannot be found in this dataset')

    def drop(self, labels, dim=None, *, errors='raise'):
        """Drop variables or index labels from this dataset.

        Parameters
        ----------
        labels : scalar or list of scalars
            Name(s) of variables or index labels to drop.
        dim : None or str, optional
            Dimension along which to drop index labels. By default (if
            ``dim is None``), drops variables rather than index labels.
        errors: {'raise', 'ignore'}, optional
            If 'raise' (default), raises a ValueError error if
            any of the variable or index labels passed are not
            in the dataset. If 'ignore', any given labels that are in the
            dataset are dropped and no error is raised.

        Returns
        -------
        dropped : Dataset
        """
        if errors not in ['raise', 'ignore']:
            raise ValueError('errors must be either "raise" or "ignore"')
        if utils.is_scalar(labels):
            labels = [labels]
        if dim is None:
            return self._drop_vars(labels, errors=errors)
        else:
            try:
                index = self.indexes[dim]
            except KeyError:
                raise ValueError(
                    'dimension %r does not have coordinate labels' % dim)
            new_index = index.drop(labels, errors=errors)
            return self.loc[{dim: new_index}]

    def _drop_vars(self, names, errors='raise'):
        if errors == 'raise':
            self._assert_all_in_dataset(names)
        drop = set(names)
        variables = OrderedDict((k, v) for k, v in self._variables.items()
                                if k not in drop)
        coord_names = set(k for k in self._coord_names if k in variables)
        indexes = OrderedDict((k, v) for k, v in self.indexes.items()
                              if k not in drop)
        return self._replace_with_new_dims(
            variables, coord_names=coord_names, indexes=indexes)

    def drop_dims(self, drop_dims, *, errors='raise'):
        """Drop dimensions and associated variables from this dataset.

        Parameters
        ----------
        drop_dims : str or list
            Dimension or dimensions to drop.

        Returns
        -------
        obj : Dataset
            The dataset without the given dimensions (or any variables
            containing those dimensions)
        errors: {'raise', 'ignore'}, optional
            If 'raise' (default), raises a ValueError error if
            any of the dimensions passed are not
            in the dataset. If 'ignore', any given dimensions that are in the
            dataset are dropped and no error is raised.
        """
        if errors not in ['raise', 'ignore']:
            raise ValueError('errors must be either "raise" or "ignore"')

        if utils.is_scalar(drop_dims):
            drop_dims = [drop_dims]

        if errors == 'raise':
            missing_dimensions = [d for d in drop_dims if d not in self.dims]
            if missing_dimensions:
                raise ValueError('Dataset does not contain the dimensions: %s'
                                 % missing_dimensions)

        drop_vars = set(k for k, v in self._variables.items()
                        for d in v.dims if d in drop_dims)
        return self._drop_vars(drop_vars)

    def transpose(self, *dims):
        """Return a new Dataset object with all array dimensions transposed.

        Although the order of dimensions on each array will change, the dataset
        dimensions themselves will remain in fixed (sorted) order.

        Parameters
        ----------
        *dims : str, optional
            By default, reverse the dimensions on each array. Otherwise,
            reorder the dimensions to this order.

        Returns
        -------
        transposed : Dataset
            Each array in the dataset (including) coordinates will be
            transposed to the given order.

        Notes
        -----
        This operation returns a view of each array's data. It is
        lazy for dask-backed DataArrays but not for numpy-backed DataArrays
        -- the data will be fully loaded into memory.

        See Also
        --------
        numpy.transpose
        DataArray.transpose
        """
        if dims:
            if set(dims) ^ set(self.dims):
                raise ValueError('arguments to transpose (%s) must be '
                                 'permuted dataset dimensions (%s)'
                                 % (dims, tuple(self.dims)))
        ds = self.copy()
        for name, var in self._variables.items():
            var_dims = tuple(dim for dim in dims if dim in var.dims)
            ds._variables[name] = var.transpose(*var_dims)
        return ds

    def dropna(self, dim, how='any', thresh=None, subset=None):
        """Returns a new dataset with dropped labels for missing values along
        the provided dimension.

        Parameters
        ----------
        dim : str
            Dimension along which to drop missing values. Dropping along
            multiple dimensions simultaneously is not yet supported.
        how : {'any', 'all'}, optional
            * any : if any NA values are present, drop that label
            * all : if all values are NA, drop that label
        thresh : int, default None
            If supplied, require this many non-NA values.
        subset : sequence, optional
            Subset of variables to check for missing values. By default, all
            variables in the dataset are checked.

        Returns
        -------
        Dataset
        """
        # TODO: consider supporting multiple dimensions? Or not, given that
        # there are some ugly edge cases, e.g., pandas's dropna differs
        # depending on the order of the supplied axes.

        if dim not in self.dims:
            raise ValueError('%s must be a single dataset dimension' % dim)

        if subset is None:
            subset = list(self.data_vars)

        count = np.zeros(self.dims[dim], dtype=np.int64)
        size = 0

        for k in subset:
            array = self._variables[k]
            if dim in array.dims:
                dims = [d for d in array.dims if d != dim]
                count += np.asarray(array.count(dims))
                size += np.prod([self.dims[d] for d in dims])

        if thresh is not None:
            mask = count >= thresh
        elif how == 'any':
            mask = count == size
        elif how == 'all':
            mask = count > 0
        elif how is not None:
            raise ValueError('invalid how option: %s' % how)
        else:
            raise TypeError('must specify how or thresh')

        return self.isel({dim: mask})

    def fillna(self, value):
        """Fill missing values in this object.

        This operation follows the normal broadcasting and alignment rules that
        xarray uses for binary arithmetic, except the result is aligned to this
        object (``join='left'``) instead of aligned to the intersection of
        index coordinates (``join='inner'``).

        Parameters
        ----------
        value : scalar, ndarray, DataArray, dict or Dataset
            Used to fill all matching missing values in this dataset's data
            variables. Scalars, ndarrays or DataArrays arguments are used to
            fill all data with aligned coordinates (for DataArrays).
            Dictionaries or datasets match data variables and then align
            coordinates if necessary.

        Returns
        -------
        Dataset
        """
        if utils.is_dict_like(value):
            value_keys = getattr(value, 'data_vars', value).keys()
            if not set(value_keys) <= set(self.data_vars.keys()):
                raise ValueError('all variables in the argument to `fillna` '
                                 'must be contained in the original dataset')
        out = ops.fillna(self, value)
        return out

    def interpolate_na(self, dim=None, method='linear', limit=None,
                       use_coordinate=True,
                       **kwargs):
        """Interpolate values according to different methods.

        Parameters
        ----------
        dim : str
            Specifies the dimension along which to interpolate.
        method : {'linear', 'nearest', 'zero', 'slinear', 'quadratic', 'cubic',
                  'polynomial', 'barycentric', 'krog', 'pchip',
                  'spline'}, optional
            String indicating which method to use for interpolation:

            - 'linear': linear interpolation (Default). Additional keyword
              arguments are passed to ``numpy.interp``
            - 'nearest', 'zero', 'slinear', 'quadratic', 'cubic',
              'polynomial': are passed to ``scipy.interpolate.interp1d``. If
              method=='polynomial', the ``order`` keyword argument must also be
              provided.
            - 'barycentric', 'krog', 'pchip', 'spline': use their respective
              ``scipy.interpolate`` classes.
        use_coordinate : boolean or str, default True
            Specifies which index to use as the x values in the interpolation
            formulated as `y = f(x)`. If False, values are treated as if
            eqaully-spaced along `dim`. If True, the IndexVariable `dim` is
            used. If use_coordinate is a string, it specifies the name of a
            coordinate variariable to use as the index.
        limit : int, default None
            Maximum number of consecutive NaNs to fill. Must be greater than 0
            or None for no limit.

        Returns
        -------
        Dataset

        See also
        --------
        numpy.interp
        scipy.interpolate
        """
        from .missing import interp_na, _apply_over_vars_with_dim

        new = _apply_over_vars_with_dim(interp_na, self, dim=dim,
                                        method=method, limit=limit,
                                        use_coordinate=use_coordinate,
                                        **kwargs)
        return new

    def ffill(self, dim, limit=None):
        '''Fill NaN values by propogating values forward

        *Requires bottleneck.*

        Parameters
        ----------
        dim : str
            Specifies the dimension along which to propagate values when
            filling.
        limit : int, default None
            The maximum number of consecutive NaN values to forward fill. In
            other words, if there is a gap with more than this number of
            consecutive NaNs, it will only be partially filled. Must be greater
            than 0 or None for no limit.

        Returns
        -------
        Dataset
        '''
        from .missing import ffill, _apply_over_vars_with_dim

        new = _apply_over_vars_with_dim(ffill, self, dim=dim, limit=limit)
        return new

    def bfill(self, dim, limit=None):
        '''Fill NaN values by propogating values backward

        *Requires bottleneck.*

        Parameters
        ----------
        dim : str
            Specifies the dimension along which to propagate values when
            filling.
        limit : int, default None
            The maximum number of consecutive NaN values to backward fill. In
            other words, if there is a gap with more than this number of
            consecutive NaNs, it will only be partially filled. Must be greater
            than 0 or None for no limit.

        Returns
        -------
        Dataset
        '''
        from .missing import bfill, _apply_over_vars_with_dim

        new = _apply_over_vars_with_dim(bfill, self, dim=dim, limit=limit)
        return new

    def combine_first(self, other):
        """Combine two Datasets, default to data_vars of self.

        The new coordinates follow the normal broadcasting and alignment rules
        of ``join='outer'``.  Vacant cells in the expanded coordinates are
        filled with np.nan.

        Parameters
        ----------
        other : DataArray
            Used to fill all matching missing values in this array.

        Returns
        -------
        DataArray
        """
        out = ops.fillna(self, other, join="outer", dataset_join="outer")
        return out

    def reduce(self, func, dim=None, keep_attrs=None, keepdims=False,
               numeric_only=False, allow_lazy=False, **kwargs):
        """Reduce this dataset by applying `func` along some dimension(s).

        Parameters
        ----------
        func : function
            Function which can be called in the form
            `f(x, axis=axis, **kwargs)` to return the result of reducing an
            np.ndarray over an integer valued axis.
        dim : str or sequence of str, optional
            Dimension(s) over which to apply `func`.  By default `func` is
            applied over all dimensions.
        keep_attrs : bool, optional
            If True, the dataset's attributes (`attrs`) will be copied from
            the original object to the new one.  If False (default), the new
            object will be returned without attributes.
        keepdims : bool, default False
            If True, the dimensions which are reduced are left in the result
            as dimensions of size one. Coordinates that use these dimensions
            are removed.
        numeric_only : bool, optional
            If True, only apply ``func`` to variables with a numeric dtype.
        **kwargs : dict
            Additional keyword arguments passed on to ``func``.

        Returns
        -------
        reduced : Dataset
            Dataset with this object's DataArrays replaced with new DataArrays
            of summarized data and the indicated dimension(s) removed.
        """
        if dim is ALL_DIMS:
            dim = None
        if isinstance(dim, str):
            dims = set([dim])
        elif dim is None:
            dims = set(self.dims)
        else:
            dims = set(dim)

        missing_dimensions = [d for d in dims if d not in self.dims]
        if missing_dimensions:
            raise ValueError('Dataset does not contain the dimensions: %s'
                             % missing_dimensions)

        if keep_attrs is None:
            keep_attrs = _get_keep_attrs(default=False)

        variables = OrderedDict()
        for name, var in self._variables.items():
            reduce_dims = [d for d in var.dims if d in dims]
            if name in self.coords:
                if not reduce_dims:
                    variables[name] = var
            else:
                if (not numeric_only
                        or np.issubdtype(var.dtype, np.number)
                        or (var.dtype == np.bool_)):
                    if len(reduce_dims) == 1:
                        # unpack dimensions for the benefit of functions
                        # like np.argmin which can't handle tuple arguments
                        reduce_dims, = reduce_dims
                    elif len(reduce_dims) == var.ndim:
                        # prefer to aggregate over axis=None rather than
                        # axis=(0, 1) if they will be equivalent, because
                        # the former is often more efficient
                        reduce_dims = None
                    variables[name] = var.reduce(func, dim=reduce_dims,
                                                 keep_attrs=keep_attrs,
                                                 keepdims=keepdims,
                                                 allow_lazy=allow_lazy,
                                                 **kwargs)

        coord_names = set(k for k in self.coords if k in variables)
        indexes = OrderedDict((k, v) for k, v in self.indexes.items()
                              if k in variables)
        attrs = self.attrs if keep_attrs else None
        return self._replace_with_new_dims(
            variables, coord_names=coord_names, attrs=attrs, indexes=indexes)

    def apply(self, func, keep_attrs=None, args=(), **kwargs):
        """Apply a function over the data variables in this dataset.

        Parameters
        ----------
        func : function
            Function which can be called in the form `func(x, *args, **kwargs)`
            to transform each DataArray `x` in this dataset into another
            DataArray.
        keep_attrs : bool, optional
            If True, the dataset's attributes (`attrs`) will be copied from
            the original object to the new one. If False, the new object will
            be returned without attributes.
        args : tuple, optional
            Positional arguments passed on to `func`.
        **kwargs : dict
            Keyword arguments passed on to `func`.

        Returns
        -------
        applied : Dataset
            Resulting dataset from applying ``func`` over each data variable.

        Examples
        --------
        >>> da = xr.DataArray(np.random.randn(2, 3))
        >>> ds = xr.Dataset({'foo': da, 'bar': ('x', [-1, 2])})
        >>> ds
        <xarray.Dataset>
        Dimensions:  (dim_0: 2, dim_1: 3, x: 2)
        Dimensions without coordinates: dim_0, dim_1, x
        Data variables:
            foo      (dim_0, dim_1) float64 -0.3751 -1.951 -1.945 0.2948 0.711 -0.3948
            bar      (x) int64 -1 2
        >>> ds.apply(np.fabs)
        <xarray.Dataset>
        Dimensions:  (dim_0: 2, dim_1: 3, x: 2)
        Dimensions without coordinates: dim_0, dim_1, x
        Data variables:
            foo      (dim_0, dim_1) float64 0.3751 1.951 1.945 0.2948 0.711 0.3948
            bar      (x) float64 1.0 2.0
        """  # noqa
        variables = OrderedDict(
            (k, maybe_wrap_array(v, func(v, *args, **kwargs)))
            for k, v in self.data_vars.items())
        if keep_attrs is None:
            keep_attrs = _get_keep_attrs(default=False)
        attrs = self.attrs if keep_attrs else None
        return type(self)(variables, attrs=attrs)

    def assign(self, variables=None, **variables_kwargs):
        """Assign new data variables to a Dataset, returning a new object
        with all the original variables in addition to the new ones.

        Parameters
        ----------
        variables : mapping, value pairs
            Mapping from variables names to the new values. If the new values
            are callable, they are computed on the Dataset and assigned to new
            data variables. If the values are not callable, (e.g. a DataArray,
            scalar, or array), they are simply assigned.
        **variables_kwargs:
            The keyword arguments form of ``variables``.
            One of variables or variables_kwarg must be provided.

        Returns
        -------
        ds : Dataset
            A new Dataset with the new variables in addition to all the
            existing variables.

        Notes
        -----
        Since ``kwargs`` is a dictionary, the order of your arguments may not
        be preserved, and so the order of the new variables is not well
        defined. Assigning multiple variables within the same ``assign`` is
        possible, but you cannot reference other variables created within the
        same ``assign`` call.

        See Also
        --------
        pandas.DataFrame.assign
        """
        variables = either_dict_or_kwargs(
            variables, variables_kwargs, 'assign')
        data = self.copy()
        # do all calculations first...
        results = data._calc_assign_results(variables)
        # ... and then assign
        data.update(results)
        return data

    def to_array(self, dim='variable', name=None):
        """Convert this dataset into an xarray.DataArray

        The data variables of this dataset will be broadcast against each other
        and stacked along the first axis of the new array. All coordinates of
        this dataset will remain coordinates.

        Parameters
        ----------
        dim : str, optional
            Name of the new dimension.
        name : str, optional
            Name of the new data array.

        Returns
        -------
        array : xarray.DataArray
        """
        from .dataarray import DataArray

        data_vars = [self.variables[k] for k in self.data_vars]
        broadcast_vars = broadcast_variables(*data_vars)
        data = duck_array_ops.stack([b.data for b in broadcast_vars], axis=0)

        coords = dict(self.coords)
        coords[dim] = list(self.data_vars)

        dims = (dim,) + broadcast_vars[0].dims

        return DataArray(data, coords, dims, attrs=self.attrs, name=name)

    def _to_dataframe(self, ordered_dims):
        columns = [k for k in self.variables if k not in self.dims]
        data = [self._variables[k].set_dims(ordered_dims).values.reshape(-1)
                for k in columns]
        index = self.coords.to_index(ordered_dims)
        return pd.DataFrame(OrderedDict(zip(columns, data)), index=index)

    def to_dataframe(self):
        """Convert this dataset into a pandas.DataFrame.

        Non-index variables in this dataset form the columns of the
        DataFrame. The DataFrame is be indexed by the Cartesian product of
        this dataset's indices.
        """
        return self._to_dataframe(self.dims)

    @classmethod
    def from_dataframe(cls, dataframe):
        """Convert a pandas.DataFrame into an xarray.Dataset

        Each column will be converted into an independent variable in the
        Dataset. If the dataframe's index is a MultiIndex, it will be expanded
        into a tensor product of one-dimensional indices (filling in missing
        values with NaN). This method will produce a Dataset very similar to
        that on which the 'to_dataframe' method was called, except with
        possibly redundant dimensions (since all dataset variables will have
        the same dimensionality).
        """
        # TODO: Add an option to remove dimensions along which the variables
        # are constant, to enable consistent serialization to/from a dataframe,
        # even if some variables have different dimensionality.

        if not dataframe.columns.is_unique:
            raise ValueError(
                'cannot convert DataFrame with non-unique columns')

        idx = dataframe.index
        obj = cls()

        if isinstance(idx, pd.MultiIndex):
            # it's a multi-index
            # expand the DataFrame to include the product of all levels
            full_idx = pd.MultiIndex.from_product(idx.levels, names=idx.names)
            dataframe = dataframe.reindex(full_idx)
            dims = [name if name is not None else 'level_%i' % n
                    for n, name in enumerate(idx.names)]
            for dim, lev in zip(dims, idx.levels):
                obj[dim] = (dim, lev)
            shape = [lev.size for lev in idx.levels]
        else:
            dims = (idx.name if idx.name is not None else 'index',)
            obj[dims[0]] = (dims, idx)
            shape = -1

        for name, series in dataframe.items():
            data = np.asarray(series).reshape(shape)
            obj[name] = (dims, data)
        return obj

    def to_dask_dataframe(self, dim_order=None, set_index=False):
        """
        Convert this dataset into a dask.dataframe.DataFrame.

        The dimensions, coordinates and data variables in this dataset form
        the columns of the DataFrame.

        Parameters
        ----------
        dim_order : list, optional
            Hierarchical dimension order for the resulting dataframe. All
            arrays are transposed to this order and then written out as flat
            vectors in contiguous order, so the last dimension in this list
            will be contiguous in the resulting DataFrame. This has a major
            influence on which operations are efficient on the resulting dask
            dataframe.

            If provided, must include all dimensions on this dataset. By
            default, dimensions are sorted alphabetically.
        set_index : bool, optional
            If set_index=True, the dask DataFrame is indexed by this dataset's
            coordinate. Since dask DataFrames to not support multi-indexes,
            set_index only works if the dataset only contains one dimension.

        Returns
        -------
        dask.dataframe.DataFrame
        """

        import dask.array as da
        import dask.dataframe as dd

        if dim_order is None:
            dim_order = list(self.dims)
        elif set(dim_order) != set(self.dims):
            raise ValueError(
                'dim_order {} does not match the set of dimensions on this '
                'Dataset: {}'.format(dim_order, list(self.dims)))

        ordered_dims = OrderedDict((k, self.dims[k]) for k in dim_order)

        columns = list(ordered_dims)
        columns.extend(k for k in self.coords if k not in self.dims)
        columns.extend(self.data_vars)

        series_list = []
        for name in columns:
            try:
                var = self.variables[name]
            except KeyError:
                # dimension without a matching coordinate
                size = self.dims[name]
                data = da.arange(size, chunks=size, dtype=np.int64)
                var = Variable((name,), data)

            # IndexVariable objects have a dummy .chunk() method
            if isinstance(var, IndexVariable):
                var = var.to_base_variable()

            dask_array = var.set_dims(ordered_dims).chunk(self.chunks).data
            series = dd.from_array(dask_array.reshape(-1), columns=[name])
            series_list.append(series)

        df = dd.concat(series_list, axis=1)

        if set_index:
            if len(dim_order) == 1:
                (dim,) = dim_order
                df = df.set_index(dim)
            else:
                # triggers an error about multi-indexes, even if only one
                # dimension is passed
                df = df.set_index(dim_order)

        return df

    def to_dict(self, data=True):
        """
        Convert this dataset to a dictionary following xarray naming
        conventions.

        Converts all variables and attributes to native Python objects
        Useful for coverting to json. To avoid datetime incompatibility
        use decode_times=False kwarg in xarrray.open_dataset.

        Parameters
        ----------
        data : bool, optional
            Whether to include the actual data in the dictionary. When set to
            False, returns just the schema.

        See also
        --------
        Dataset.from_dict
        """
        d = {'coords': {}, 'attrs': decode_numpy_dict_values(self.attrs),
             'dims': dict(self.dims), 'data_vars': {}}
        for k in self.coords:
            d['coords'].update({k: self[k].variable.to_dict(data=data)})
        for k in self.data_vars:
            d['data_vars'].update({k: self[k].variable.to_dict(data=data)})
        return d

    @classmethod
    def from_dict(cls, d):
        """
        Convert a dictionary into an xarray.Dataset.

        Input dict can take several forms::

            d = {'t': {'dims': ('t'), 'data': t},
                 'a': {'dims': ('t'), 'data': x},
                 'b': {'dims': ('t'), 'data': y}}

            d = {'coords': {'t': {'dims': 't', 'data': t,
                                  'attrs': {'units':'s'}}},
                 'attrs': {'title': 'air temperature'},
                 'dims': 't',
                 'data_vars': {'a': {'dims': 't', 'data': x, },
                               'b': {'dims': 't', 'data': y}}}

        where 't' is the name of the dimesion, 'a' and 'b' are names of data
        variables and t, x, and y are lists, numpy.arrays or pandas objects.

        Parameters
        ----------
        d : dict, with a minimum structure of {'var_0': {'dims': [..], \
                                                         'data': [..]}, \
                                               ...}

        Returns
        -------
        obj : xarray.Dataset

        See also
        --------
        Dataset.to_dict
        DataArray.from_dict
        """

        if not set(['coords', 'data_vars']).issubset(set(d)):
            variables = d.items()
        else:
            import itertools
            variables = itertools.chain(d.get('coords', {}).items(),
                                        d.get('data_vars', {}).items())
        try:
            variable_dict = OrderedDict([(k, (v['dims'],
                                              v['data'],
                                              v.get('attrs'))) for
                                         k, v in variables])
        except KeyError as e:
            raise ValueError(
                "cannot convert dict without the key "
                "'{dims_data}'".format(dims_data=str(e.args[0])))
        obj = cls(variable_dict)

        # what if coords aren't dims?
        coords = set(d.get('coords', {})) - set(d.get('dims', {}))
        obj = obj.set_coords(coords)

        obj.attrs.update(d.get('attrs', {}))

        return obj

    @staticmethod
    def _unary_op(f, keep_attrs=False):
        @functools.wraps(f)
        def func(self, *args, **kwargs):
            variables = OrderedDict()
            for k, v in self._variables.items():
                if k in self._coord_names:
                    variables[k] = v
                else:
                    variables[k] = f(v, *args, **kwargs)
            attrs = self._attrs if keep_attrs else None
            return self._replace_with_new_dims(variables, attrs=attrs)

        return func

    @staticmethod
    def _binary_op(f, reflexive=False, join=None):
        @functools.wraps(f)
        def func(self, other):
            from .dataarray import DataArray

            if isinstance(other, groupby.GroupBy):
                return NotImplemented
            align_type = OPTIONS['arithmetic_join'] if join is None else join
            if isinstance(other, (DataArray, Dataset)):
                self, other = align(self, other, join=align_type, copy=False)
            g = f if not reflexive else lambda x, y: f(y, x)
            ds = self._calculate_binary_op(g, other, join=align_type)
            return ds

        return func

    @staticmethod
    def _inplace_binary_op(f):
        @functools.wraps(f)
        def func(self, other):
            from .dataarray import DataArray

            if isinstance(other, groupby.GroupBy):
                raise TypeError('in-place operations between a Dataset and '
                                'a grouped object are not permitted')
            # we don't actually modify arrays in-place with in-place Dataset
            # arithmetic -- this lets us automatically align things
            if isinstance(other, (DataArray, Dataset)):
                other = other.reindex_like(self, copy=False)
            g = ops.inplace_to_noninplace_op(f)
            ds = self._calculate_binary_op(g, other, inplace=True)
            self._replace_with_new_dims(ds._variables, ds._coord_names,
                                        attrs=ds._attrs, indexes=ds._indexes,
                                        inplace=True)
            return self

        return func

    def _calculate_binary_op(self, f, other, join='inner',
                             inplace=False):
        def apply_over_both(lhs_data_vars, rhs_data_vars, lhs_vars, rhs_vars):
            if inplace and set(lhs_data_vars) != set(rhs_data_vars):
                raise ValueError('datasets must have the same data variables '
                                 'for in-place arithmetic operations: %s, %s'
                                 % (list(lhs_data_vars), list(rhs_data_vars)))

            dest_vars = OrderedDict()

            for k in lhs_data_vars:
                if k in rhs_data_vars:
                    dest_vars[k] = f(lhs_vars[k], rhs_vars[k])
                elif join in ["left", "outer"]:
                    dest_vars[k] = f(lhs_vars[k], np.nan)
            for k in rhs_data_vars:
                if k not in dest_vars and join in ["right", "outer"]:
                    dest_vars[k] = f(rhs_vars[k], np.nan)
            return dest_vars

        if utils.is_dict_like(other) and not isinstance(other, Dataset):
            # can't use our shortcut of doing the binary operation with
            # Variable objects, so apply over our data vars instead.
            new_data_vars = apply_over_both(self.data_vars, other,
                                            self.data_vars, other)
            return Dataset(new_data_vars)

        other_coords = getattr(other, 'coords', None)
        ds = self.coords.merge(other_coords)

        if isinstance(other, Dataset):
            new_vars = apply_over_both(self.data_vars, other.data_vars,
                                       self.variables, other.variables)
        else:
            other_variable = getattr(other, 'variable', other)
            new_vars = OrderedDict((k, f(self.variables[k], other_variable))
                                   for k in self.data_vars)
        ds._variables.update(new_vars)
        ds._dims = calculate_dimensions(ds._variables)
        return ds

    def _copy_attrs_from(self, other):
        self.attrs = other.attrs
        for v in other.variables:
            if v in self.variables:
                self.variables[v].attrs = other.variables[v].attrs

    def diff(self, dim, n=1, label='upper'):
        """Calculate the n-th order discrete difference along given axis.

        Parameters
        ----------
        dim : str, optional
            Dimension over which to calculate the finite difference.
        n : int, optional
            The number of times values are differenced.
        label : str, optional
            The new coordinate in dimension ``dim`` will have the
            values of either the minuend's or subtrahend's coordinate
            for values 'upper' and 'lower', respectively.  Other
            values are not supported.

        Returns
        -------
        difference : same type as caller
            The n-th order finite difference of this object.

        Examples
        --------
        >>> ds = xr.Dataset({'foo': ('x', [5, 5, 6, 6])})
        >>> ds.diff('x')
        <xarray.Dataset>
        Dimensions:  (x: 3)
        Coordinates:
          * x        (x) int64 1 2 3
        Data variables:
            foo      (x) int64 0 1 0
        >>> ds.diff('x', 2)
        <xarray.Dataset>
        Dimensions:  (x: 2)
        Coordinates:
        * x        (x) int64 2 3
        Data variables:
        foo      (x) int64 1 -1

        See Also
        --------
        Dataset.differentiate
        """
        if n == 0:
            return self
        if n < 0:
            raise ValueError('order `n` must be non-negative but got {0}'
                             ''.format(n))

        # prepare slices
        kwargs_start = {dim: slice(None, -1)}
        kwargs_end = {dim: slice(1, None)}

        # prepare new coordinate
        if label == 'upper':
            kwargs_new = kwargs_end
        elif label == 'lower':
            kwargs_new = kwargs_start
        else:
            raise ValueError('The \'label\' argument has to be either '
                             '\'upper\' or \'lower\'')

        variables = OrderedDict()

        for name, var in self.variables.items():
            if dim in var.dims:
                if name in self.data_vars:
                    variables[name] = (var.isel(**kwargs_end)
                                       - var.isel(**kwargs_start))
                else:
                    variables[name] = var.isel(**kwargs_new)
            else:
                variables[name] = var

        indexes = OrderedDict(self.indexes)
        if dim in indexes:
            indexes[dim] = indexes[dim][kwargs_new[dim]]

        difference = self._replace_with_new_dims(variables, indexes=indexes)

        if n > 1:
            return difference.diff(dim, n - 1)
        else:
            return difference

    def shift(self, shifts=None, fill_value=dtypes.NA, **shifts_kwargs):
        """Shift this dataset by an offset along one or more dimensions.

        Only data variables are moved; coordinates stay in place. This is
        consistent with the behavior of ``shift`` in pandas.

        Parameters
        ----------
        shifts : Mapping with the form of {dim: offset}
            Integer offset to shift along each of the given dimensions.
            Positive offsets shift to the right; negative offsets shift to the
            left.
        fill_value: scalar, optional
            Value to use for newly missing values
        **shifts_kwargs:
            The keyword arguments form of ``shifts``.
            One of shifts or shifts_kwarg must be provided.

        Returns
        -------
        shifted : Dataset
            Dataset with the same coordinates and attributes but shifted data
            variables.

        See also
        --------
        roll

        Examples
        --------

        >>> ds = xr.Dataset({'foo': ('x', list('abcde'))})
        >>> ds.shift(x=2)
        <xarray.Dataset>
        Dimensions:  (x: 5)
        Coordinates:
          * x        (x) int64 0 1 2 3 4
        Data variables:
            foo      (x) object nan nan 'a' 'b' 'c'
        """
        shifts = either_dict_or_kwargs(shifts, shifts_kwargs, 'shift')
        invalid = [k for k in shifts if k not in self.dims]
        if invalid:
            raise ValueError("dimensions %r do not exist" % invalid)

        variables = OrderedDict()
        for name, var in self.variables.items():
            if name in self.data_vars:
                var_shifts = {k: v for k, v in shifts.items()
                              if k in var.dims}
                variables[name] = var.shift(
                    fill_value=fill_value, shifts=var_shifts)
            else:
                variables[name] = var

        return self._replace(variables)

    def roll(self, shifts=None, roll_coords=None, **shifts_kwargs):
        """Roll this dataset by an offset along one or more dimensions.

        Unlike shift, roll may rotate all variables, including coordinates
        if specified. The direction of rotation is consistent with
        :py:func:`numpy.roll`.

        Parameters
        ----------

        shifts : dict, optional
            A dict with keys matching dimensions and values given
            by integers to rotate each of the given dimensions. Positive
            offsets roll to the right; negative offsets roll to the left.
        roll_coords : bool
            Indicates whether to  roll the coordinates by the offset
            The current default of roll_coords (None, equivalent to True) is
            deprecated and will change to False in a future version.
            Explicitly pass roll_coords to silence the warning.
        **shifts_kwargs : {dim: offset, ...}, optional
            The keyword arguments form of ``shifts``.
            One of shifts or shifts_kwargs must be provided.
        Returns
        -------
        rolled : Dataset
            Dataset with the same coordinates and attributes but rolled
            variables.

        See also
        --------
        shift

        Examples
        --------

        >>> ds = xr.Dataset({'foo': ('x', list('abcde'))})
        >>> ds.roll(x=2)
        <xarray.Dataset>
        Dimensions:  (x: 5)
        Coordinates:
          * x        (x) int64 3 4 0 1 2
        Data variables:
            foo      (x) object 'd' 'e' 'a' 'b' 'c'
        """
        shifts = either_dict_or_kwargs(shifts, shifts_kwargs, 'roll')
        invalid = [k for k in shifts if k not in self.dims]
        if invalid:
            raise ValueError("dimensions %r do not exist" % invalid)

        if roll_coords is None:
            warnings.warn("roll_coords will be set to False in the future."
                          " Explicitly set roll_coords to silence warning.",
                          FutureWarning, stacklevel=2)
            roll_coords = True

        unrolled_vars = () if roll_coords else self.coords

        variables = OrderedDict()
        for k, v in self.variables.items():
            if k not in unrolled_vars:
                variables[k] = v.roll(**{k: s for k, s in shifts.items()
                                         if k in v.dims})
            else:
                variables[k] = v

        if roll_coords:
            indexes = OrderedDict()
            for k, v in self.indexes.items():
                (dim,) = self.variables[k].dims
                if dim in shifts:
                    indexes[k] = roll_index(v, shifts[dim])
        else:
            indexes = OrderedDict(self.indexes)

        return self._replace(variables, indexes=indexes)

    def sortby(self, variables, ascending=True):
        """
        Sort object by labels or values (along an axis).

        Sorts the dataset, either along specified dimensions,
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
        variables: str, DataArray, or list of either
            1D DataArray objects or name(s) of 1D variable(s) in
            coords/data_vars whose values are used to sort the dataset.
        ascending: boolean, optional
            Whether to sort by ascending or descending order.

        Returns
        -------
        sorted: Dataset
            A new dataset where all the specified dims are sorted by dim
            labels.
        """
        from .dataarray import DataArray

        if not isinstance(variables, list):
            variables = [variables]
        else:
            variables = variables
        variables = [v if isinstance(v, DataArray) else self[v]
                     for v in variables]
        aligned_vars = align(self, *variables, join='left')
        aligned_self = aligned_vars[0]
        aligned_other_vars = aligned_vars[1:]
        vars_by_dim = defaultdict(list)
        for data_array in aligned_other_vars:
            if data_array.ndim != 1:
                raise ValueError("Input DataArray is not 1-D.")
            if (data_array.dtype == object
                    and LooseVersion(np.__version__) < LooseVersion('1.11.0')):
                raise NotImplementedError(
                    'sortby uses np.lexsort under the hood, which requires '
                    'numpy 1.11.0 or later to support object data-type.')
            (key,) = data_array.dims
            vars_by_dim[key].append(data_array)

        indices = {}
        for key, arrays in vars_by_dim.items():
            order = np.lexsort(tuple(reversed(arrays)))
            indices[key] = order if ascending else order[::-1]
        return aligned_self.isel(**indices)

    def quantile(self, q, dim=None, interpolation='linear',
                 numeric_only=False, keep_attrs=None):
        """Compute the qth quantile of the data along the specified dimension.

        Returns the qth quantiles(s) of the array elements for each variable
        in the Dataset.

        Parameters
        ----------
        q : float in range of [0,1] or array-like of floats
            Quantile to compute, which must be between 0 and 1 inclusive.
        dim : str or sequence of str, optional
            Dimension(s) over which to apply quantile.
        interpolation : {'linear', 'lower', 'higher', 'midpoint', 'nearest'}
            This optional parameter specifies the interpolation method to
            use when the desired quantile lies between two data points
            ``i < j``:

                * linear: ``i + (j - i) * fraction``, where ``fraction`` is
                  the fractional part of the index surrounded by ``i`` and
                  ``j``.
                * lower: ``i``.
                * higher: ``j``.
                * nearest: ``i`` or ``j``, whichever is nearest.
                * midpoint: ``(i + j) / 2``.
        keep_attrs : bool, optional
            If True, the dataset's attributes (`attrs`) will be copied from
            the original object to the new one.  If False (default), the new
            object will be returned without attributes.
        numeric_only : bool, optional
            If True, only apply ``func`` to variables with a numeric dtype.

        Returns
        -------
        quantiles : Dataset
            If `q` is a single quantile, then the result is a scalar for each
            variable in data_vars. If multiple percentiles are given, first
            axis of the result corresponds to the quantile and a quantile
            dimension is added to the return Dataset. The other dimensions are
            the dimensions that remain after the reduction of the array.

        See Also
        --------
        numpy.nanpercentile, pandas.Series.quantile, DataArray.quantile
        """

        if isinstance(dim, str):
            dims = set([dim])
        elif dim is None:
            dims = set(self.dims)
        else:
            dims = set(dim)

        _assert_empty([d for d in dims if d not in self.dims],
                      'Dataset does not contain the dimensions: %s')

        q = np.asarray(q, dtype=np.float64)

        variables = OrderedDict()
        for name, var in self.variables.items():
            reduce_dims = [d for d in var.dims if d in dims]
            if reduce_dims or not var.dims:
                if name not in self.coords:
                    if (not numeric_only
                        or np.issubdtype(var.dtype, np.number)
                            or var.dtype == np.bool_):
                        if len(reduce_dims) == var.ndim:
                            # prefer to aggregate over axis=None rather than
                            # axis=(0, 1) if they will be equivalent, because
                            # the former is often more efficient
                            reduce_dims = None
                        variables[name] = var.quantile(
                            q, dim=reduce_dims, interpolation=interpolation)

            else:
                variables[name] = var

        # construct the new dataset
        coord_names = set(k for k in self.coords if k in variables)
        indexes = OrderedDict(
            (k, v) for k, v in self.indexes.items() if k in variables
        )
        if keep_attrs is None:
            keep_attrs = _get_keep_attrs(default=False)
        attrs = self.attrs if keep_attrs else None
        new = self._replace_with_new_dims(
            variables, coord_names=coord_names, attrs=attrs, indexes=indexes)
        if 'quantile' in new.dims:
            new.coords['quantile'] = Variable('quantile', q)
        else:
            new.coords['quantile'] = q
        return new

    def rank(self, dim, pct=False, keep_attrs=None):
        """Ranks the data.

        Equal values are assigned a rank that is the average of the ranks that
        would have been otherwise assigned to all of the values within
        that set.
        Ranks begin at 1, not 0. If pct is True, computes percentage ranks.

        NaNs in the input array are returned as NaNs.

        The `bottleneck` library is required.

        Parameters
        ----------
        dim : str
            Dimension over which to compute rank.
        pct : bool, optional
            If True, compute percentage ranks, otherwise compute integer ranks.
        keep_attrs : bool, optional
            If True, the dataset's attributes (`attrs`) will be copied from
            the original object to the new one.  If False (default), the new
            object will be returned without attributes.

        Returns
        -------
        ranked : Dataset
            Variables that do not depend on `dim` are dropped.
        """
        if dim not in self.dims:
            raise ValueError(
                'Dataset does not contain the dimension: %s' % dim)

        variables = OrderedDict()
        for name, var in self.variables.items():
            if name in self.data_vars:
                if dim in var.dims:
                    variables[name] = var.rank(dim, pct=pct)
            else:
                variables[name] = var

        coord_names = set(self.coords)
        if keep_attrs is None:
            keep_attrs = _get_keep_attrs(default=False)
        attrs = self.attrs if keep_attrs else None
        return self._replace(variables, coord_names, attrs=attrs)

    def differentiate(self, coord, edge_order=1, datetime_unit=None):
        """ Differentiate with the second order accurate central
        differences.

        .. note::
            This feature is limited to simple cartesian geometry, i.e. coord
            must be one dimensional.

        Parameters
        ----------
        coord: str
            The coordinate to be used to compute the gradient.
        edge_order: 1 or 2. Default 1
            N-th order accurate differences at the boundaries.
        datetime_unit: None or any of {'Y', 'M', 'W', 'D', 'h', 'm', 's', 'ms',
            'us', 'ns', 'ps', 'fs', 'as'}
            Unit to compute gradient. Only valid for datetime coordinate.

        Returns
        -------
        differentiated: Dataset

        See also
        --------
        numpy.gradient: corresponding numpy function
        """
        from .variable import Variable

        if coord not in self.variables and coord not in self.dims:
            raise ValueError('Coordinate {} does not exist.'.format(coord))

        coord_var = self[coord].variable
        if coord_var.ndim != 1:
            raise ValueError('Coordinate {} must be 1 dimensional but is {}'
                             ' dimensional'.format(coord, coord_var.ndim))

        dim = coord_var.dims[0]
        if _contains_datetime_like_objects(coord_var):
            if coord_var.dtype.kind in 'mM' and datetime_unit is None:
                datetime_unit, _ = np.datetime_data(coord_var.dtype)
            elif datetime_unit is None:
                datetime_unit = 's'  # Default to seconds for cftime objects
            coord_var = coord_var._to_numeric(datetime_unit=datetime_unit)

        variables = OrderedDict()
        for k, v in self.variables.items():
            if (k in self.data_vars and dim in v.dims
                    and k not in self.coords):
                if _contains_datetime_like_objects(v):
                    v = v._to_numeric(datetime_unit=datetime_unit)
                grad = duck_array_ops.gradient(
                    v.data, coord_var, edge_order=edge_order,
                    axis=v.get_axis_num(dim))
                variables[k] = Variable(v.dims, grad)
            else:
                variables[k] = v
        return self._replace(variables)

    def integrate(self, coord, datetime_unit=None):
        """ integrate the array with the trapezoidal rule.

        .. note::
            This feature is limited to simple cartesian geometry, i.e. coord
            must be one dimensional.

        Parameters
        ----------
        dim: str, or a sequence of str
            Coordinate(s) used for the integration.
        datetime_unit
            Can be specify the unit if datetime coordinate is used. One of
            {'Y', 'M', 'W', 'D', 'h', 'm', 's', 'ms', 'us', 'ns', 'ps', 'fs',
             'as'}

        Returns
        -------
        integrated: Dataset

        See also
        --------
        DataArray.integrate
        numpy.trapz: corresponding numpy function
        """
        if not isinstance(coord, (list, tuple)):
            coord = (coord, )
        result = self
        for c in coord:
            result = result._integrate_one(c, datetime_unit=datetime_unit)
        return result

    def _integrate_one(self, coord, datetime_unit=None):
        from .variable import Variable

        if coord not in self.variables and coord not in self.dims:
            raise ValueError('Coordinate {} does not exist.'.format(coord))

        coord_var = self[coord].variable
        if coord_var.ndim != 1:
            raise ValueError('Coordinate {} must be 1 dimensional but is {}'
                             ' dimensional'.format(coord, coord_var.ndim))

        dim = coord_var.dims[0]
        if _contains_datetime_like_objects(coord_var):
            if coord_var.dtype.kind in 'mM' and datetime_unit is None:
                datetime_unit, _ = np.datetime_data(coord_var.dtype)
            elif datetime_unit is None:
                datetime_unit = 's'  # Default to seconds for cftime objects
            coord_var = datetime_to_numeric(
                coord_var, datetime_unit=datetime_unit)

        variables = OrderedDict()
        coord_names = set()
        for k, v in self.variables.items():
            if k in self.coords:
                if dim not in v.dims:
                    variables[k] = v
                    coord_names.add(k)
            else:
                if k in self.data_vars and dim in v.dims:
                    if _contains_datetime_like_objects(v):
                        v = datetime_to_numeric(v, datetime_unit=datetime_unit)
                    integ = duck_array_ops.trapz(
                        v.data, coord_var.data, axis=v.get_axis_num(dim))
                    v_dims = list(v.dims)
                    v_dims.remove(dim)
                    variables[k] = Variable(v_dims, integ)
                else:
                    variables[k] = v
        indexes = OrderedDict(
            (k, v) for k, v in self.indexes.items() if k in variables
        )
        return self._replace_with_new_dims(
            variables, coord_names=coord_names, indexes=indexes)

    @property
    def real(self):
        return self._unary_op(lambda x: x.real,
                              keep_attrs=True)(self)

    @property
    def imag(self):
        return self._unary_op(lambda x: x.imag,
                              keep_attrs=True)(self)

    def filter_by_attrs(self, **kwargs):
        """Returns a ``Dataset`` with variables that match specific conditions.

        Can pass in ``key=value`` or ``key=callable``.  A Dataset is returned
        containing only the variables for which all the filter tests pass.
        These tests are either ``key=value`` for which the attribute ``key``
        has the exact value ``value`` or the callable passed into
        ``key=callable`` returns True. The callable will be passed a single
        value, either the value of the attribute ``key`` or ``None`` if the
        DataArray does not have an attribute with the name ``key``.

        Parameters
        ----------
        **kwargs : key=value
            key : str
                Attribute name.
            value : callable or obj
                If value is a callable, it should return a boolean in the form
                of bool = func(attr) where attr is da.attrs[key].
                Otherwise, value will be compared to the each
                DataArray's attrs[key].

        Returns
        -------
        new : Dataset
            New dataset with variables filtered by attribute.

        Examples
        --------
        >>> # Create an example dataset:
        >>> import numpy as np
        >>> import pandas as pd
        >>> import xarray as xr
        >>> temp = 15 + 8 * np.random.randn(2, 2, 3)
        >>> precip = 10 * np.random.rand(2, 2, 3)
        >>> lon = [[-99.83, -99.32], [-99.79, -99.23]]
        >>> lat = [[42.25, 42.21], [42.63, 42.59]]
        >>> dims = ['x', 'y', 'time']
        >>> temp_attr = dict(standard_name='air_potential_temperature')
        >>> precip_attr = dict(standard_name='convective_precipitation_flux')
        >>> ds = xr.Dataset({
        ...         'temperature': (dims,  temp, temp_attr),
        ...         'precipitation': (dims, precip, precip_attr)},
        ...                 coords={
        ...         'lon': (['x', 'y'], lon),
        ...         'lat': (['x', 'y'], lat),
        ...         'time': pd.date_range('2014-09-06', periods=3),
        ...         'reference_time': pd.Timestamp('2014-09-05')})
        >>> # Get variables matching a specific standard_name.
        >>> ds.filter_by_attrs(standard_name='convective_precipitation_flux')
        <xarray.Dataset>
        Dimensions:         (time: 3, x: 2, y: 2)
        Coordinates:
          * x               (x) int64 0 1
          * time            (time) datetime64[ns] 2014-09-06 2014-09-07 2014-09-08
            lat             (x, y) float64 42.25 42.21 42.63 42.59
          * y               (y) int64 0 1
            reference_time  datetime64[ns] 2014-09-05
            lon             (x, y) float64 -99.83 -99.32 -99.79 -99.23
        Data variables:
            precipitation   (x, y, time) float64 4.178 2.307 6.041 6.046 0.06648 ...
        >>> # Get all variables that have a standard_name attribute.
        >>> standard_name = lambda v: v is not None
        >>> ds.filter_by_attrs(standard_name=standard_name)
        <xarray.Dataset>
        Dimensions:         (time: 3, x: 2, y: 2)
        Coordinates:
            lon             (x, y) float64 -99.83 -99.32 -99.79 -99.23
            lat             (x, y) float64 42.25 42.21 42.63 42.59
          * x               (x) int64 0 1
          * y               (y) int64 0 1
          * time            (time) datetime64[ns] 2014-09-06 2014-09-07 2014-09-08
            reference_time  datetime64[ns] 2014-09-05
        Data variables:
            temperature     (x, y, time) float64 25.86 20.82 6.954 23.13 10.25 11.68 ...
            precipitation   (x, y, time) float64 5.702 0.9422 2.075 1.178 3.284 ...

        """  # noqa
        selection = []
        for var_name, variable in self.data_vars.items():
            has_value_flag = False
            for attr_name, pattern in kwargs.items():
                attr_value = variable.attrs.get(attr_name)
                if ((callable(pattern) and pattern(attr_value))
                        or attr_value == pattern):
                    has_value_flag = True
                else:
                    has_value_flag = False
                    break
            if has_value_flag is True:
                selection.append(var_name)
        return self[selection]


ops.inject_all_ops_and_reduce_methods(Dataset, array_only=False)
