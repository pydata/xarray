from __future__ import annotations

import copy
import datetime
import inspect
import sys
import warnings
from collections import defaultdict
from html import escape
from numbers import Number
from operator import methodcaller
from os import PathLike
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Collection,
    DefaultDict,
    Hashable,
    Iterable,
    Iterator,
    Literal,
    Mapping,
    MutableMapping,
    Sequence,
    cast,
    overload,
)

import numpy as np
import pandas as pd

import xarray as xr

from ..coding.calendar_ops import convert_calendar, interp_calendar
from ..coding.cftimeindex import CFTimeIndex, _parse_array_of_cftime_strings
from ..plot.dataset_plot import _Dataset_PlotMethods
from . import (
    alignment,
    dtypes,
    duck_array_ops,
    formatting,
    formatting_html,
    groupby,
    ops,
    resample,
    rolling,
    utils,
    weighted,
)
from .alignment import _broadcast_helper, _get_broadcast_dims_map_common_coords, align
from .arithmetic import DatasetArithmetic
from .common import DataWithCoords, _contains_datetime_like_objects, get_chunksizes
from .computation import unify_chunks
from .coordinates import (
    DatasetCoordinates,
    assert_coordinate_consistent,
    remap_label_indexers,
)
from .duck_array_ops import datetime_to_numeric
from .indexes import (
    Index,
    Indexes,
    PandasIndex,
    PandasMultiIndex,
    default_indexes,
    isel_variable_and_index,
    propagate_indexes,
    remove_unused_levels_categories,
    roll_index,
)
from .indexing import is_fancy_indexer
from .merge import (
    dataset_merge_method,
    dataset_update_method,
    merge_coordinates_without_align,
    merge_data_and_coords,
)
from .missing import get_clean_interp_index
from .options import OPTIONS, _get_keep_attrs
from .pycompat import is_duck_dask_array, sparse_array_type
from .utils import (
    Default,
    Frozen,
    HybridMappingProxy,
    OrderedSet,
    _default,
    decode_numpy_dict_values,
    drop_dims_from_indexers,
    either_dict_or_kwargs,
    hashable,
    infix_dims,
    is_dict_like,
    is_scalar,
    maybe_wrap_array,
)
from .variable import (
    IndexVariable,
    Variable,
    as_variable,
    assert_unique_multiindex_level_names,
    broadcast_variables,
)

if TYPE_CHECKING:
    from ..backends import AbstractDataStore, ZarrStore
    from .dataarray import DataArray
    from .merge import CoercibleMapping
    from .types import T_Xarray

    try:
        from dask.delayed import Delayed
    except ImportError:
        Delayed = None


# list of attributes of pd.DatetimeIndex that are ndarrays of time info
_DATETIMEINDEX_COMPONENTS = [
    "year",
    "month",
    "day",
    "hour",
    "minute",
    "second",
    "microsecond",
    "nanosecond",
    "date",
    "time",
    "dayofyear",
    "weekofyear",
    "dayofweek",
    "quarter",
]


def _get_virtual_variable(
    variables, key: Hashable, level_vars: Mapping = None, dim_sizes: Mapping = None
) -> tuple[Hashable, Hashable, Variable]:
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

    split_key = key.split(".", 1)
    var_name: str | None
    if len(split_key) == 2:
        ref_name, var_name = split_key
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


def calculate_dimensions(variables: Mapping[Any, Variable]) -> dict[Hashable, int]:
    """Calculate the dimensions corresponding to a set of variables.

    Returns dictionary mapping from dimension names to sizes. Raises ValueError
    if any of the dimension sizes conflict.
    """
    dims: dict[Hashable, int] = {}
    last_used = {}
    scalar_vars = {k for k, v in variables.items() if not v.dims}
    for k, var in variables.items():
        for dim, size in zip(var.dims, var.shape):
            if dim in scalar_vars:
                raise ValueError(
                    f"dimension {dim!r} already exists as a scalar variable"
                )
            if dim not in dims:
                dims[dim] = size
                last_used[dim] = k
            elif dims[dim] != size:
                raise ValueError(
                    f"conflicting sizes for dimension {dim!r}: "
                    f"length {size} on {k!r} and length {dims[dim]} on {last_used!r}"
                )
    return dims


def merge_indexes(
    indexes: Mapping[Any, Hashable | Sequence[Hashable]],
    variables: Mapping[Any, Variable],
    coord_names: set[Hashable],
    append: bool = False,
) -> tuple[dict[Hashable, Variable], set[Hashable]]:
    """Merge variables into multi-indexes.

    Not public API. Used in Dataset and DataArray set_index
    methods.
    """
    vars_to_replace: dict[Hashable, Variable] = {}
    vars_to_remove: list[Hashable] = []
    dims_to_replace: dict[Hashable, Hashable] = {}
    error_msg = "{} is not the name of an existing variable."

    for dim, var_names in indexes.items():
        if isinstance(var_names, str) or not isinstance(var_names, Sequence):
            var_names = [var_names]

        names: list[Hashable] = []
        codes: list[list[int]] = []
        levels: list[list[int]] = []
        current_index_variable = variables.get(dim)

        for n in var_names:
            try:
                var = variables[n]
            except KeyError:
                raise ValueError(error_msg.format(n))
            if (
                current_index_variable is not None
                and var.dims != current_index_variable.dims
            ):
                raise ValueError(
                    f"dimension mismatch between {dim!r} {current_index_variable.dims} and {n!r} {var.dims}"
                )

        if current_index_variable is not None and append:
            current_index = current_index_variable.to_index()
            if isinstance(current_index, pd.MultiIndex):
                names.extend(current_index.names)
                codes.extend(current_index.codes)
                levels.extend(current_index.levels)
            else:
                names.append(f"{dim}_level_0")
                cat = pd.Categorical(current_index.values, ordered=True)
                codes.append(cat.codes)
                levels.append(cat.categories)

        if not len(names) and len(var_names) == 1:
            idx = pd.Index(variables[var_names[0]].values)

        else:  # MultiIndex
            for n in var_names:
                try:
                    var = variables[n]
                except KeyError:
                    raise ValueError(error_msg.format(n))
                names.append(n)
                cat = pd.Categorical(var.values, ordered=True)
                codes.append(cat.codes)
                levels.append(cat.categories)

            idx = pd.MultiIndex(levels, codes, names=names)
            for n in names:
                dims_to_replace[n] = dim

        vars_to_replace[dim] = IndexVariable(dim, idx)
        vars_to_remove.extend(var_names)

    new_variables = {k: v for k, v in variables.items() if k not in vars_to_remove}
    new_variables.update(vars_to_replace)

    # update dimensions if necessary, GH: 3512
    for k, v in new_variables.items():
        if any(d in dims_to_replace for d in v.dims):
            new_dims = [dims_to_replace.get(d, d) for d in v.dims]
            new_variables[k] = v._replace(dims=new_dims)
    new_coord_names = coord_names | set(vars_to_replace)
    new_coord_names -= set(vars_to_remove)
    return new_variables, new_coord_names


def split_indexes(
    dims_or_levels: Hashable | Sequence[Hashable],
    variables: Mapping[Any, Variable],
    coord_names: set[Hashable],
    level_coords: Mapping[Any, Hashable],
    drop: bool = False,
) -> tuple[dict[Hashable, Variable], set[Hashable]]:
    """Extract (multi-)indexes (levels) as variables.

    Not public API. Used in Dataset and DataArray reset_index
    methods.
    """
    if isinstance(dims_or_levels, str) or not isinstance(dims_or_levels, Sequence):
        dims_or_levels = [dims_or_levels]

    dim_levels: DefaultDict[Any, list[Hashable]] = defaultdict(list)
    dims = []
    for k in dims_or_levels:
        if k in level_coords:
            dim_levels[level_coords[k]].append(k)
        else:
            dims.append(k)

    vars_to_replace = {}
    vars_to_create: dict[Hashable, Variable] = {}
    vars_to_remove = []

    for d in dims:
        index = variables[d].to_index()
        if isinstance(index, pd.MultiIndex):
            dim_levels[d] = index.names
        else:
            vars_to_remove.append(d)
            if not drop:
                vars_to_create[str(d) + "_"] = Variable(d, index, variables[d].attrs)

    for d, levs in dim_levels.items():
        index = variables[d].to_index()
        if len(levs) == index.nlevels:
            vars_to_remove.append(d)
        else:
            vars_to_replace[d] = IndexVariable(d, index.droplevel(levs))

        if not drop:
            for lev in levs:
                idx = index.get_level_values(lev)
                vars_to_create[idx.name] = Variable(d, idx, variables[d].attrs)

    new_variables = dict(variables)
    for v in set(vars_to_remove):
        del new_variables[v]
    new_variables.update(vars_to_replace)
    new_variables.update(vars_to_create)
    new_coord_names = (coord_names | set(vars_to_create)) - set(vars_to_remove)

    return new_variables, new_coord_names


def _assert_empty(args: tuple, msg: str = "%s") -> None:
    if args:
        raise ValueError(msg % args)


def _check_chunks_compatibility(var, chunks, preferred_chunks):
    for dim in var.dims:
        if dim not in chunks or (dim not in preferred_chunks):
            continue

        preferred_chunks_dim = preferred_chunks.get(dim)
        chunks_dim = chunks.get(dim)

        if isinstance(chunks_dim, int):
            chunks_dim = (chunks_dim,)
        else:
            chunks_dim = chunks_dim[:-1]

        if any(s % preferred_chunks_dim for s in chunks_dim):
            warnings.warn(
                f"Specified Dask chunks {chunks[dim]} would separate "
                f"on disks chunk shape {preferred_chunks[dim]} for dimension {dim}. "
                "This could degrade performance. "
                "Consider rechunking after loading instead.",
                stacklevel=2,
            )


def _get_chunk(var, chunks):
    # chunks need to be explicity computed to take correctly into accout
    # backend preferred chunking
    import dask.array as da

    if isinstance(var, IndexVariable):
        return {}

    if isinstance(chunks, int) or (chunks == "auto"):
        chunks = dict.fromkeys(var.dims, chunks)

    preferred_chunks = var.encoding.get("preferred_chunks", {})
    preferred_chunks_list = [
        preferred_chunks.get(dim, shape) for dim, shape in zip(var.dims, var.shape)
    ]

    chunks_list = [
        chunks.get(dim, None) or preferred_chunks.get(dim, None) for dim in var.dims
    ]

    output_chunks_list = da.core.normalize_chunks(
        chunks_list,
        shape=var.shape,
        dtype=var.dtype,
        previous_chunks=preferred_chunks_list,
    )

    output_chunks = dict(zip(var.dims, output_chunks_list))
    _check_chunks_compatibility(var, output_chunks, preferred_chunks)

    return output_chunks


def _maybe_chunk(
    name,
    var,
    chunks,
    token=None,
    lock=None,
    name_prefix="xarray-",
    overwrite_encoded_chunks=False,
):
    from dask.base import tokenize

    if chunks is not None:
        chunks = {dim: chunks[dim] for dim in var.dims if dim in chunks}
    if var.ndim:
        # when rechunking by different amounts, make sure dask names change
        # by provinding chunks as an input to tokenize.
        # subtle bugs result otherwise. see GH3350
        token2 = tokenize(name, token if token else var._data, chunks)
        name2 = f"{name_prefix}{name}-{token2}"
        var = var.chunk(chunks, name=name2, lock=lock)

        if overwrite_encoded_chunks and var.chunks is not None:
            var.encoding["chunks"] = tuple(x[0] for x in var.chunks)
        return var
    else:
        return var


def as_dataset(obj: Any) -> Dataset:
    """Cast the given object to a Dataset.

    Handles Datasets, DataArrays and dictionaries of variables. A new Dataset
    object is only created if the provided object is not already one.
    """
    if hasattr(obj, "to_dataset"):
        obj = obj.to_dataset()
    if not isinstance(obj, Dataset):
        obj = Dataset(obj)
    return obj


def _get_func_args(func, param_names):
    """Use `inspect.signature` to try accessing `func` args. Otherwise, ensure
    they are provided by user.
    """
    try:
        func_args = inspect.signature(func).parameters
    except ValueError:
        func_args = {}
        if not param_names:
            raise ValueError(
                "Unable to inspect `func` signature, and `param_names` was not provided."
            )
    if param_names:
        params = param_names
    else:
        params = list(func_args)[1:]
        if any(
            [(p.kind in [p.VAR_POSITIONAL, p.VAR_KEYWORD]) for p in func_args.values()]
        ):
            raise ValueError(
                "`param_names` must be provided because `func` takes variable length arguments."
            )
    return params, func_args


def _initialize_curvefit_params(params, p0, bounds, func_args):
    """Set initial guess and bounds for curvefit.
    Priority: 1) passed args 2) func signature 3) scipy defaults
    """

    def _initialize_feasible(lb, ub):
        # Mimics functionality of scipy.optimize.minpack._initialize_feasible
        lb_finite = np.isfinite(lb)
        ub_finite = np.isfinite(ub)
        p0 = np.nansum(
            [
                0.5 * (lb + ub) * int(lb_finite & ub_finite),
                (lb + 1) * int(lb_finite & ~ub_finite),
                (ub - 1) * int(~lb_finite & ub_finite),
            ]
        )
        return p0

    param_defaults = {p: 1 for p in params}
    bounds_defaults = {p: (-np.inf, np.inf) for p in params}
    for p in params:
        if p in func_args and func_args[p].default is not func_args[p].empty:
            param_defaults[p] = func_args[p].default
        if p in bounds:
            bounds_defaults[p] = tuple(bounds[p])
            if param_defaults[p] < bounds[p][0] or param_defaults[p] > bounds[p][1]:
                param_defaults[p] = _initialize_feasible(bounds[p][0], bounds[p][1])
        if p in p0:
            param_defaults[p] = p0[p]
    return param_defaults, bounds_defaults


class DataVariables(Mapping[Any, "DataArray"]):
    __slots__ = ("_dataset",)

    def __init__(self, dataset: Dataset):
        self._dataset = dataset

    def __iter__(self) -> Iterator[Hashable]:
        return (
            key
            for key in self._dataset._variables
            if key not in self._dataset._coord_names
        )

    def __len__(self) -> int:
        return len(self._dataset._variables) - len(self._dataset._coord_names)

    def __contains__(self, key: Hashable) -> bool:
        return key in self._dataset._variables and key not in self._dataset._coord_names

    def __getitem__(self, key: Hashable) -> DataArray:
        if key not in self._dataset._coord_names:
            return cast("DataArray", self._dataset[key])
        raise KeyError(key)

    def __repr__(self) -> str:
        return formatting.data_vars_repr(self)

    @property
    def variables(self) -> Mapping[Hashable, Variable]:
        all_variables = self._dataset.variables
        return Frozen({k: all_variables[k] for k in self})

    def _ipython_key_completions_(self):
        """Provide method for the key-autocompletions in IPython."""
        return [
            key
            for key in self._dataset._ipython_key_completions_()
            if key not in self._dataset._coord_names
        ]


class _LocIndexer:
    __slots__ = ("dataset",)

    def __init__(self, dataset: Dataset):
        self.dataset = dataset

    def __getitem__(self, key: Mapping[Any, Any]) -> Dataset:
        if not utils.is_dict_like(key):
            raise TypeError("can only lookup dictionaries from Dataset.loc")
        return self.dataset.sel(key)

    def __setitem__(self, key, value) -> None:
        if not utils.is_dict_like(key):
            raise TypeError(
                "can only set locations defined by dictionaries from Dataset.loc."
                f" Got: {key}"
            )

        # set new values
        pos_indexers, _ = remap_label_indexers(self.dataset, key)
        self.dataset[pos_indexers] = value


class Dataset(DataWithCoords, DatasetArithmetic, Mapping):
    """A multi-dimensional, in memory, array database.

    A dataset resembles an in-memory representation of a NetCDF file,
    and consists of variables, coordinates and attributes which
    together form a self describing dataset.

    Dataset implements the mapping interface with keys given by variable
    names and values given by DataArray objects for each variable name.

    One dimensional variables with name equal to their dimension are
    index coordinates used for label based indexing.

    To load data from a file or file-like object, use the `open_dataset`
    function.

    Parameters
    ----------
    data_vars : dict-like, optional
        A mapping from variable names to :py:class:`~xarray.DataArray`
        objects, :py:class:`~xarray.Variable` objects or to tuples of
        the form ``(dims, data[, attrs])`` which can be used as
        arguments to create a new ``Variable``. Each dimension must
        have the same length in all variables in which it appears.

        The following notations are accepted:

        - mapping {var name: DataArray}
        - mapping {var name: Variable}
        - mapping {var name: (dimension name, array-like)}
        - mapping {var name: (tuple of dimension names, array-like)}
        - mapping {dimension name: array-like}
          (it will be automatically moved to coords, see below)

        Each dimension must have the same length in all variables in
        which it appears.
    coords : dict-like, optional
        Another mapping in similar form as the `data_vars` argument,
        except the each item is saved on the dataset as a "coordinate".
        These variables have an associated meaning: they describe
        constant/fixed/independent quantities, unlike the
        varying/measured/dependent quantities that belong in
        `variables`. Coordinates values may be given by 1-dimensional
        arrays or scalars, in which case `dims` do not need to be
        supplied: 1D arrays will be assumed to give index values along
        the dimension with the same name.

        The following notations are accepted:

        - mapping {coord name: DataArray}
        - mapping {coord name: Variable}
        - mapping {coord name: (dimension name, array-like)}
        - mapping {coord name: (tuple of dimension names, array-like)}
        - mapping {dimension name: array-like}
          (the dimension name is implicitly set to be the same as the
          coord name)

        The last notation implies that the coord name is the same as
        the dimension name.

    attrs : dict-like, optional
        Global attributes to save on this dataset.

    Examples
    --------
    Create data:

    >>> np.random.seed(0)
    >>> temperature = 15 + 8 * np.random.randn(2, 2, 3)
    >>> precipitation = 10 * np.random.rand(2, 2, 3)
    >>> lon = [[-99.83, -99.32], [-99.79, -99.23]]
    >>> lat = [[42.25, 42.21], [42.63, 42.59]]
    >>> time = pd.date_range("2014-09-06", periods=3)
    >>> reference_time = pd.Timestamp("2014-09-05")

    Initialize a dataset with multiple dimensions:

    >>> ds = xr.Dataset(
    ...     data_vars=dict(
    ...         temperature=(["x", "y", "time"], temperature),
    ...         precipitation=(["x", "y", "time"], precipitation),
    ...     ),
    ...     coords=dict(
    ...         lon=(["x", "y"], lon),
    ...         lat=(["x", "y"], lat),
    ...         time=time,
    ...         reference_time=reference_time,
    ...     ),
    ...     attrs=dict(description="Weather related data."),
    ... )
    >>> ds
    <xarray.Dataset>
    Dimensions:         (x: 2, y: 2, time: 3)
    Coordinates:
        lon             (x, y) float64 -99.83 -99.32 -99.79 -99.23
        lat             (x, y) float64 42.25 42.21 42.63 42.59
      * time            (time) datetime64[ns] 2014-09-06 2014-09-07 2014-09-08
        reference_time  datetime64[ns] 2014-09-05
    Dimensions without coordinates: x, y
    Data variables:
        temperature     (x, y, time) float64 29.11 18.2 22.83 ... 18.28 16.15 26.63
        precipitation   (x, y, time) float64 5.68 9.256 0.7104 ... 7.992 4.615 7.805
    Attributes:
        description:  Weather related data.

    Find out where the coldest temperature was and what values the
    other variables had:

    >>> ds.isel(ds.temperature.argmin(...))
    <xarray.Dataset>
    Dimensions:         ()
    Coordinates:
        lon             float64 -99.32
        lat             float64 42.21
        time            datetime64[ns] 2014-09-08
        reference_time  datetime64[ns] 2014-09-05
    Data variables:
        temperature     float64 7.182
        precipitation   float64 8.326
    Attributes:
        description:  Weather related data.
    """

    _attrs: dict[Hashable, Any] | None
    _cache: dict[str, Any]
    _coord_names: set[Hashable]
    _dims: dict[Hashable, int]
    _encoding: dict[Hashable, Any] | None
    _close: Callable[[], None] | None
    _indexes: dict[Hashable, Index] | None
    _variables: dict[Hashable, Variable]

    __slots__ = (
        "_attrs",
        "_cache",
        "_coord_names",
        "_dims",
        "_encoding",
        "_close",
        "_indexes",
        "_variables",
        "__weakref__",
    )

    _groupby_cls = groupby.DatasetGroupBy
    _rolling_cls = rolling.DatasetRolling
    _coarsen_cls = rolling.DatasetCoarsen
    _resample_cls = resample.DatasetResample
    _weighted_cls = weighted.DatasetWeighted

    def __init__(
        self,
        # could make a VariableArgs to use more generally, and refine these
        # categories
        data_vars: Mapping[Any, Any] = None,
        coords: Mapping[Any, Any] = None,
        attrs: Mapping[Any, Any] = None,
    ):
        # TODO(shoyer): expose indexes as a public argument in __init__

        if data_vars is None:
            data_vars = {}
        if coords is None:
            coords = {}

        both_data_and_coords = set(data_vars) & set(coords)
        if both_data_and_coords:
            raise ValueError(
                f"variables {both_data_and_coords!r} are found in both data_vars and coords"
            )

        if isinstance(coords, Dataset):
            coords = coords.variables

        variables, coord_names, dims, indexes, _ = merge_data_and_coords(
            data_vars, coords, compat="broadcast_equals"
        )

        self._attrs = dict(attrs) if attrs is not None else None
        self._close = None
        self._encoding = None
        self._variables = variables
        self._coord_names = coord_names
        self._dims = dims
        self._indexes = indexes

    @classmethod
    def load_store(cls, store, decoder=None) -> Dataset:
        """Create a new dataset from the contents of a backends.*DataStore
        object
        """
        variables, attributes = store.load()
        if decoder:
            variables, attributes = decoder(variables, attributes)
        obj = cls(variables, attrs=attributes)
        obj.set_close(store.close)
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
    def attrs(self) -> dict[Hashable, Any]:
        """Dictionary of global attributes on this dataset"""
        if self._attrs is None:
            self._attrs = {}
        return self._attrs

    @attrs.setter
    def attrs(self, value: Mapping[Any, Any]) -> None:
        self._attrs = dict(value)

    @property
    def encoding(self) -> dict:
        """Dictionary of global encoding attributes on this dataset"""
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
        return Frozen(self._dims)

    @property
    def sizes(self) -> Mapping[Hashable, int]:
        """Mapping from dimension names to lengths.

        Cannot be modified directly, but is updated when adding new variables.

        This is an alias for `Dataset.dims` provided for the benefit of
        consistency with `DataArray.sizes`.

        See Also
        --------
        DataArray.sizes
        """
        return self.dims

    def load(self, **kwargs) -> Dataset:
        """Manually trigger loading and/or computation of this dataset's data
        from disk or a remote source into memory and return this dataset.
        Unlike compute, the original dataset is modified and returned.

        Normally, it should not be necessary to call this method in user code,
        because all xarray functions should either work on deferred data or
        load data automatically. However, this method can be necessary when
        working with many file objects on disk.

        Parameters
        ----------
        **kwargs : dict
            Additional keyword arguments passed on to ``dask.compute``.

        See Also
        --------
        dask.compute
        """
        # access .data to coerce everything to numpy or dask arrays
        lazy_data = {
            k: v._data for k, v in self.variables.items() if is_duck_dask_array(v._data)
        }
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

    def __dask_tokenize__(self):
        from dask.base import normalize_token

        return normalize_token(
            (type(self), self._variables, self._coord_names, self._attrs)
        )

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

        return [
            v.__dask_keys__()
            for v in self.variables.values()
            if dask.is_dask_collection(v)
        ]

    def __dask_layers__(self):
        import dask

        return sum(
            (
                v.__dask_layers__()
                for v in self.variables.values()
                if dask.is_dask_collection(v)
            ),
            (),
        )

    @property
    def __dask_optimize__(self):
        import dask.array as da

        return da.Array.__dask_optimize__

    @property
    def __dask_scheduler__(self):
        import dask.array as da

        return da.Array.__dask_scheduler__

    def __dask_postcompute__(self):
        return self._dask_postcompute, ()

    def __dask_postpersist__(self):
        return self._dask_postpersist, ()

    def _dask_postcompute(self, results: Iterable[Variable]) -> Dataset:
        import dask

        variables = {}
        results_iter = iter(results)

        for k, v in self._variables.items():
            if dask.is_dask_collection(v):
                rebuild, args = v.__dask_postcompute__()
                v = rebuild(next(results_iter), *args)
            variables[k] = v

        return Dataset._construct_direct(
            variables,
            self._coord_names,
            self._dims,
            self._attrs,
            self._indexes,
            self._encoding,
            self._close,
        )

    def _dask_postpersist(
        self, dsk: Mapping, *, rename: Mapping[str, str] = None
    ) -> Dataset:
        from dask import is_dask_collection
        from dask.highlevelgraph import HighLevelGraph
        from dask.optimization import cull

        variables = {}

        for k, v in self._variables.items():
            if not is_dask_collection(v):
                variables[k] = v
                continue

            if isinstance(dsk, HighLevelGraph):
                # dask >= 2021.3
                # __dask_postpersist__() was called by dask.highlevelgraph.
                # Don't use dsk.cull(), as we need to prevent partial layers:
                # https://github.com/dask/dask/issues/7137
                layers = v.__dask_layers__()
                if rename:
                    layers = [rename.get(k, k) for k in layers]
                dsk2 = dsk.cull_layers(layers)
            elif rename:  # pragma: nocover
                # At the moment of writing, this is only for forward compatibility.
                # replace_name_in_key requires dask >= 2021.3.
                from dask.base import flatten, replace_name_in_key

                keys = [
                    replace_name_in_key(k, rename) for k in flatten(v.__dask_keys__())
                ]
                dsk2, _ = cull(dsk, keys)
            else:
                # __dask_postpersist__() was called by dask.optimize or dask.persist
                dsk2, _ = cull(dsk, v.__dask_keys__())

            rebuild, args = v.__dask_postpersist__()
            # rename was added in dask 2021.3
            kwargs = {"rename": rename} if rename else {}
            variables[k] = rebuild(dsk2, *args, **kwargs)

        return Dataset._construct_direct(
            variables,
            self._coord_names,
            self._dims,
            self._attrs,
            self._indexes,
            self._encoding,
            self._close,
        )

    def compute(self, **kwargs) -> Dataset:
        """Manually trigger loading and/or computation of this dataset's data
        from disk or a remote source into memory and return a new dataset.
        Unlike load, the original dataset is left unaltered.

        Normally, it should not be necessary to call this method in user code,
        because all xarray functions should either work on deferred data or
        load data automatically. However, this method can be necessary when
        working with many file objects on disk.

        Parameters
        ----------
        **kwargs : dict
            Additional keyword arguments passed on to ``dask.compute``.

        See Also
        --------
        dask.compute
        """
        new = self.copy(deep=False)
        return new.load(**kwargs)

    def _persist_inplace(self, **kwargs) -> Dataset:
        """Persist all Dask arrays in memory"""
        # access .data to coerce everything to numpy or dask arrays
        lazy_data = {
            k: v._data for k, v in self.variables.items() if is_duck_dask_array(v._data)
        }
        if lazy_data:
            import dask

            # evaluate all the dask arrays simultaneously
            evaluated_data = dask.persist(*lazy_data.values(), **kwargs)

            for k, data in zip(lazy_data, evaluated_data):
                self.variables[k].data = data

        return self

    def persist(self, **kwargs) -> Dataset:
        """Trigger computation, keeping data as dask arrays

        This operation can be used to trigger computation on underlying dask
        arrays, similar to ``.compute()`` or ``.load()``.  However this
        operation keeps the data as dask arrays. This is particularly useful
        when using the dask.distributed scheduler and you want to load a large
        amount of data into distributed memory.

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
    def _construct_direct(
        cls,
        variables: dict[Any, Variable],
        coord_names: set[Hashable],
        dims: dict[Any, int] = None,
        attrs: dict = None,
        indexes: dict[Any, Index] = None,
        encoding: dict = None,
        close: Callable[[], None] = None,
    ) -> Dataset:
        """Shortcut around __init__ for internal use when we want to skip
        costly validation
        """
        if dims is None:
            dims = calculate_dimensions(variables)
        obj = object.__new__(cls)
        obj._variables = variables
        obj._coord_names = coord_names
        obj._dims = dims
        obj._indexes = indexes
        obj._attrs = attrs
        obj._close = close
        obj._encoding = encoding
        return obj

    def _replace(
        self,
        variables: dict[Hashable, Variable] = None,
        coord_names: set[Hashable] = None,
        dims: dict[Any, int] = None,
        attrs: dict[Hashable, Any] | None | Default = _default,
        indexes: dict[Hashable, Index] | None | Default = _default,
        encoding: dict | None | Default = _default,
        inplace: bool = False,
    ) -> Dataset:
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
            if attrs is not _default:
                self._attrs = attrs
            if indexes is not _default:
                self._indexes = indexes
            if encoding is not _default:
                self._encoding = encoding
            obj = self
        else:
            if variables is None:
                variables = self._variables.copy()
            if coord_names is None:
                coord_names = self._coord_names.copy()
            if dims is None:
                dims = self._dims.copy()
            if attrs is _default:
                attrs = copy.copy(self._attrs)
            if indexes is _default:
                indexes = copy.copy(self._indexes)
            if encoding is _default:
                encoding = copy.copy(self._encoding)
            obj = self._construct_direct(
                variables, coord_names, dims, attrs, indexes, encoding
            )
        return obj

    def _replace_with_new_dims(
        self,
        variables: dict[Hashable, Variable],
        coord_names: set = None,
        attrs: dict[Hashable, Any] | None | Default = _default,
        indexes: dict[Hashable, Index] | None | Default = _default,
        inplace: bool = False,
    ) -> Dataset:
        """Replace variables with recalculated dimensions."""
        dims = calculate_dimensions(variables)
        return self._replace(
            variables, coord_names, dims, attrs, indexes, inplace=inplace
        )

    def _replace_vars_and_dims(
        self,
        variables: dict[Hashable, Variable],
        coord_names: set = None,
        dims: dict[Hashable, int] = None,
        attrs: dict[Hashable, Any] | None | Default = _default,
        inplace: bool = False,
    ) -> Dataset:
        """Deprecated version of _replace_with_new_dims().

        Unlike _replace_with_new_dims(), this method always recalculates
        indexes from variables.
        """
        if dims is None:
            dims = calculate_dimensions(variables)
        return self._replace(
            variables, coord_names, dims, attrs, indexes=None, inplace=inplace
        )

    def _overwrite_indexes(self, indexes: Mapping[Any, Index]) -> Dataset:
        if not indexes:
            return self

        variables = self._variables.copy()
        new_indexes = dict(self.xindexes)
        for name, idx in indexes.items():
            variables[name] = IndexVariable(name, idx.to_pandas_index())
            new_indexes[name] = idx
        obj = self._replace(variables, indexes=new_indexes)

        # switch from dimension to level names, if necessary
        dim_names: dict[Hashable, str] = {}
        for dim, idx in indexes.items():
            pd_idx = idx.to_pandas_index()
            if not isinstance(pd_idx, pd.MultiIndex) and pd_idx.name != dim:
                dim_names[dim] = pd_idx.name
        if dim_names:
            obj = obj.rename(dim_names)
        return obj

    def copy(self, deep: bool = False, data: Mapping = None) -> Dataset:
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
        >>> ds = xr.Dataset(
        ...     {"foo": da, "bar": ("x", [-1, 2])},
        ...     coords={"x": ["one", "two"]},
        ... )
        >>> ds.copy()
        <xarray.Dataset>
        Dimensions:  (dim_0: 2, dim_1: 3, x: 2)
        Coordinates:
          * x        (x) <U3 'one' 'two'
        Dimensions without coordinates: dim_0, dim_1
        Data variables:
            foo      (dim_0, dim_1) float64 1.764 0.4002 0.9787 2.241 1.868 -0.9773
            bar      (x) int64 -1 2

        >>> ds_0 = ds.copy(deep=False)
        >>> ds_0["foo"][0, 0] = 7
        >>> ds_0
        <xarray.Dataset>
        Dimensions:  (dim_0: 2, dim_1: 3, x: 2)
        Coordinates:
          * x        (x) <U3 'one' 'two'
        Dimensions without coordinates: dim_0, dim_1
        Data variables:
            foo      (dim_0, dim_1) float64 7.0 0.4002 0.9787 2.241 1.868 -0.9773
            bar      (x) int64 -1 2

        >>> ds
        <xarray.Dataset>
        Dimensions:  (dim_0: 2, dim_1: 3, x: 2)
        Coordinates:
          * x        (x) <U3 'one' 'two'
        Dimensions without coordinates: dim_0, dim_1
        Data variables:
            foo      (dim_0, dim_1) float64 7.0 0.4002 0.9787 2.241 1.868 -0.9773
            bar      (x) int64 -1 2

        Changing the data using the ``data`` argument maintains the
        structure of the original object, but with the new data. Original
        object is unaffected.

        >>> ds.copy(data={"foo": np.arange(6).reshape(2, 3), "bar": ["a", "b"]})
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
            foo      (dim_0, dim_1) float64 7.0 0.4002 0.9787 2.241 1.868 -0.9773
            bar      (x) int64 -1 2

        See Also
        --------
        pandas.DataFrame.copy
        """
        if data is None:
            variables = {k: v.copy(deep=deep) for k, v in self._variables.items()}
        elif not utils.is_dict_like(data):
            raise ValueError("Data must be dict-like")
        else:
            var_keys = set(self.data_vars.keys())
            data_keys = set(data.keys())
            keys_not_in_vars = data_keys - var_keys
            if keys_not_in_vars:
                raise ValueError(
                    "Data must only contain variables in original "
                    "dataset. Extra variables: {}".format(keys_not_in_vars)
                )
            keys_missing_from_data = var_keys - data_keys
            if keys_missing_from_data:
                raise ValueError(
                    "Data must contain all variables in original "
                    "dataset. Data is missing {}".format(keys_missing_from_data)
                )
            variables = {
                k: v.copy(deep=deep, data=data.get(k))
                for k, v in self._variables.items()
            }

        attrs = copy.deepcopy(self._attrs) if deep else copy.copy(self._attrs)

        return self._replace(variables, attrs=attrs)

    def as_numpy(self: Dataset) -> Dataset:
        """
        Coerces wrapped data and coordinates into numpy arrays, returning a Dataset.

        See also
        --------
        DataArray.as_numpy
        DataArray.to_numpy : Returns only the data as a numpy.ndarray object.
        """
        numpy_variables = {k: v.as_numpy() for k, v in self.variables.items()}
        return self._replace(variables=numpy_variables)

    @property
    def _level_coords(self) -> dict[str, Hashable]:
        """Return a mapping of all MultiIndex levels and their corresponding
        coordinate name.
        """
        level_coords: dict[str, Hashable] = {}
        for name, index in self.xindexes.items():
            # TODO: benbovy - flexible indexes: update when MultIndex has its own xarray class.
            pd_index = index.to_pandas_index()
            if isinstance(pd_index, pd.MultiIndex):
                level_names = pd_index.names
                (dim,) = self.variables[name].dims
                level_coords.update({lname: dim for lname in level_names})
        return level_coords

    def _copy_listed(self, names: Iterable[Hashable]) -> Dataset:
        """Create a new Dataset with the listed variables from this dataset and
        the all relevant coordinates. Skips all validation.
        """
        variables: dict[Hashable, Variable] = {}
        coord_names = set()
        indexes: dict[Hashable, Index] = {}

        for name in names:
            try:
                variables[name] = self._variables[name]
            except KeyError:
                ref_name, var_name, var = _get_virtual_variable(
                    self._variables, name, self._level_coords, self.dims
                )
                variables[var_name] = var
                if ref_name in self._coord_names or ref_name in self.dims:
                    coord_names.add(var_name)
                if (var_name,) == var.dims:
                    indexes[var_name] = var._to_xindex()

        needed_dims: OrderedSet[Hashable] = OrderedSet()
        for v in variables.values():
            needed_dims.update(v.dims)

        dims = {k: self.dims[k] for k in needed_dims}

        # preserves ordering of coordinates
        for k in self._variables:
            if k not in self._coord_names:
                continue

            if set(self.variables[k].dims) <= needed_dims:
                variables[k] = self._variables[k]
                coord_names.add(k)
                if k in self.xindexes:
                    indexes[k] = self.xindexes[k]

        return self._replace(variables, coord_names, dims, indexes=indexes)

    def _construct_dataarray(self, name: Hashable) -> DataArray:
        """Construct a DataArray by indexing this dataset"""
        from .dataarray import DataArray

        try:
            variable = self._variables[name]
        except KeyError:
            _, name, variable = _get_virtual_variable(
                self._variables, name, self._level_coords, self.dims
            )

        needed_dims = set(variable.dims)

        coords: dict[Hashable, Variable] = {}
        # preserve ordering
        for k in self._variables:
            if k in self._coord_names and set(self.variables[k].dims) <= needed_dims:
                coords[k] = self.variables[k]

        if self._indexes is None:
            indexes = None
        else:
            indexes = {k: v for k, v in self._indexes.items() if k in coords}

        return DataArray(variable, coords, name=name, indexes=indexes, fastpath=True)

    def __copy__(self) -> Dataset:
        return self.copy(deep=False)

    def __deepcopy__(self, memo=None) -> Dataset:
        # memo does nothing but is required for compatibility with
        # copy.deepcopy
        return self.copy(deep=True)

    @property
    def _attr_sources(self) -> Iterable[Mapping[Hashable, Any]]:
        """Places to look-up items for attribute-style access"""
        yield from self._item_sources
        yield self.attrs

    @property
    def _item_sources(self) -> Iterable[Mapping[Hashable, Any]]:
        """Places to look-up items for key-completion"""
        yield self.data_vars
        yield HybridMappingProxy(keys=self._coord_names, mapping=self.coords)

        # virtual coordinates
        yield HybridMappingProxy(keys=self.dims, mapping=self)

        # uses empty dict -- everything here can already be found in self.coords.
        yield HybridMappingProxy(keys=self._level_coords, mapping={})

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
        raise TypeError(
            "cannot directly convert an xarray.Dataset into a "
            "numpy array. Instead, create an xarray.DataArray "
            "first, either with indexing on the Dataset or by "
            "invoking the `to_array()` method."
        )

    @property
    def nbytes(self) -> int:
        return sum(v.nbytes for v in self.variables.values())

    @property
    def loc(self) -> _LocIndexer:
        """Attribute for location based indexing. Only supports __getitem__,
        and only when the key is a dict of the form {dim: labels}.
        """
        return _LocIndexer(self)

    # FIXME https://github.com/python/mypy/issues/7328
    @overload
    def __getitem__(self, key: Mapping) -> Dataset:  # type: ignore[misc]
        ...

    @overload
    def __getitem__(self, key: Hashable) -> DataArray:  # type: ignore[misc]
        ...

    @overload
    def __getitem__(self, key: Any) -> Dataset:
        ...

    def __getitem__(self, key):
        """Access variables or coordinates this dataset as a
        :py:class:`~xarray.DataArray`.

        Indexing with a list of names will return a new ``Dataset`` object.
        """
        if utils.is_dict_like(key):
            return self.isel(**cast(Mapping, key))

        if hashable(key):
            return self._construct_dataarray(key)
        else:
            return self._copy_listed(key)

    def __setitem__(self, key: Hashable | list[Hashable] | Mapping, value) -> None:
        """Add an array to this dataset.
        Multiple arrays can be added at the same time, in which case each of
        the following operations is applied to the respective value.

        If key is a dictionary, update all variables in the dataset
        one by one with the given value at the given location.
        If the given value is also a dataset, select corresponding variables
        in the given value and in the dataset to be changed.

        If value is a `DataArray`, call its `select_vars()` method, rename it
        to `key` and merge the contents of the resulting dataset into this
        dataset.

        If value is a `Variable` object (or tuple of form
        ``(dims, data[, attrs])``), add it to this dataset as a new
        variable.
        """
        if utils.is_dict_like(key):
            # check for consistency and convert value to dataset
            value = self._setitem_check(key, value)
            # loop over dataset variables and set new values
            processed = []
            for name, var in self.items():
                try:
                    var[key] = value[name]
                    processed.append(name)
                except Exception as e:
                    if processed:
                        raise RuntimeError(
                            "An error occured while setting values of the"
                            f" variable '{name}'. The following variables have"
                            f" been successfully updated:\n{processed}"
                        ) from e
                    else:
                        raise e

        elif isinstance(key, list):
            if len(key) == 0:
                raise ValueError("Empty list of variables to be set")
            if len(key) == 1:
                self.update({key[0]: value})
            else:
                if len(key) != len(value):
                    raise ValueError(
                        f"Different lengths of variables to be set "
                        f"({len(key)}) and data used as input for "
                        f"setting ({len(value)})"
                    )
                if isinstance(value, Dataset):
                    self.update(dict(zip(key, value.data_vars.values())))
                elif isinstance(value, xr.DataArray):
                    raise ValueError("Cannot assign single DataArray to multiple keys")
                else:
                    self.update(dict(zip(key, value)))

        else:
            if isinstance(value, Dataset):
                raise TypeError(
                    "Cannot assign a Dataset to a single key - only a DataArray or Variable object can be stored under"
                    "a single key."
                )
            self.update({key: value})

    def _setitem_check(self, key, value):
        """Consistency check for __setitem__

        When assigning values to a subset of a Dataset, do consistency check beforehand
        to avoid leaving the dataset in a partially updated state when an error occurs.
        """
        from .dataarray import DataArray

        if isinstance(value, Dataset):
            missing_vars = [
                name for name in value.data_vars if name not in self.data_vars
            ]
            if missing_vars:
                raise ValueError(
                    f"Variables {missing_vars} in new values"
                    f" not available in original dataset:\n{self}"
                )
        elif not any([isinstance(value, t) for t in [DataArray, Number, str]]):
            raise TypeError(
                "Dataset assignment only accepts DataArrays, Datasets, and scalars."
            )

        new_value = xr.Dataset()
        for name, var in self.items():
            # test indexing
            try:
                var_k = var[key]
            except Exception as e:
                raise ValueError(
                    f"Variable '{name}': indexer {key} not available"
                ) from e

            if isinstance(value, Dataset):
                val = value[name]
            else:
                val = value

            if isinstance(val, DataArray):
                # check consistency of dimensions
                for dim in val.dims:
                    if dim not in var_k.dims:
                        raise KeyError(
                            f"Variable '{name}': dimension '{dim}' appears in new values "
                            f"but not in the indexed original data"
                        )
                dims = tuple(dim for dim in var_k.dims if dim in val.dims)
                if dims != val.dims:
                    raise ValueError(
                        f"Variable '{name}': dimension order differs between"
                        f" original and new data:\n{dims}\nvs.\n{val.dims}"
                    )
            else:
                val = np.array(val)

            # type conversion
            new_value[name] = val.astype(var_k.dtype, copy=False)

        # check consistency of dimension sizes and dimension coordinates
        if isinstance(value, DataArray) or isinstance(value, Dataset):
            xr.align(self[key], value, join="exact", copy=False)

        return new_value

    def __delitem__(self, key: Hashable) -> None:
        """Remove a variable from this dataset."""
        del self._variables[key]
        self._coord_names.discard(key)
        if key in self.xindexes:
            assert self._indexes is not None
            del self._indexes[key]
        self._dims = calculate_dimensions(self._variables)

    # mutable objects should not be hashable
    # https://github.com/python/mypy/issues/4266
    __hash__ = None  # type: ignore[assignment]

    def _all_compat(self, other: Dataset, compat_str: str) -> bool:
        """Helper function for equals and identical"""

        # some stores (e.g., scipy) do not seem to preserve order, so don't
        # require matching order for equality
        def compat(x: Variable, y: Variable) -> bool:
            return getattr(x, compat_str)(y)

        return self._coord_names == other._coord_names and utils.dict_equiv(
            self._variables, other._variables, compat=compat
        )

    def broadcast_equals(self, other: Dataset) -> bool:
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
            return self._all_compat(other, "broadcast_equals")
        except (TypeError, AttributeError):
            return False

    def equals(self, other: Dataset) -> bool:
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
            return self._all_compat(other, "equals")
        except (TypeError, AttributeError):
            return False

    def identical(self, other: Dataset) -> bool:
        """Like equals, but also checks all dataset attributes and the
        attributes on all variables and coordinates.

        See Also
        --------
        Dataset.broadcast_equals
        Dataset.equals
        """
        try:
            return utils.dict_equiv(self.attrs, other.attrs) and self._all_compat(
                other, "identical"
            )
        except (TypeError, AttributeError):
            return False

    @property
    def indexes(self) -> Indexes:
        """Mapping of pandas.Index objects used for label based indexing.

        Raises an error if this Dataset has indexes that cannot be coerced
        to pandas.Index objects.

        See Also
        --------
        Dataset.xindexes

        """
        return Indexes({k: idx.to_pandas_index() for k, idx in self.xindexes.items()})

    @property
    def xindexes(self) -> Indexes:
        """Mapping of xarray Index objects used for label based indexing."""
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
        """Dictionary of DataArray objects corresponding to data variables"""
        return DataVariables(self)

    def set_coords(self, names: Hashable | Iterable[Hashable]) -> Dataset:
        """Given names of one or more variables, set them as coordinates

        Parameters
        ----------
        names : hashable or iterable of hashable
            Name(s) of variables in this dataset to convert into coordinates.

        Returns
        -------
        Dataset

        See Also
        --------
        Dataset.swap_dims
        """
        # TODO: allow inserting new coordinates with this method, like
        # DataFrame.set_index?
        # nb. check in self._variables, not self.data_vars to insure that the
        # operation is idempotent
        if isinstance(names, str) or not isinstance(names, Iterable):
            names = [names]
        else:
            names = list(names)
        self._assert_all_in_dataset(names)
        obj = self.copy()
        obj._coord_names.update(names)
        return obj

    def reset_coords(
        self,
        names: Hashable | Iterable[Hashable] | None = None,
        drop: bool = False,
    ) -> Dataset:
        """Given names of coordinates, reset them to become variables

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
        Dataset
        """
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
                    f"cannot remove index coordinates with reset_coords: {bad_coords}"
                )
        obj = self.copy()
        obj._coord_names.difference_update(names)
        if drop:
            for name in names:
                del obj._variables[name]
        return obj

    def dump_to_store(self, store: AbstractDataStore, **kwargs) -> None:
        """Store dataset contents to a backends.*DataStore object."""
        from ..backends.api import dump_to_store

        # TODO: rename and/or cleanup this method to make it more consistent
        # with to_netcdf()
        dump_to_store(self, store, **kwargs)

    def to_netcdf(
        self,
        path=None,
        mode: str = "w",
        format: str = None,
        group: str = None,
        engine: str = None,
        encoding: Mapping = None,
        unlimited_dims: Iterable[Hashable] = None,
        compute: bool = True,
        invalid_netcdf: bool = False,
    ) -> bytes | Delayed | None:
        """Write dataset contents to a netCDF file.

        Parameters
        ----------
        path : str, path-like or file-like, optional
            Path to which to save this dataset. File-like objects are only
            supported by the scipy engine. If no path is provided, this
            function returns the resulting netCDF file as bytes; in this case,
            we need to use scipy, which does not support netCDF version 4 (the
            default format becomes NETCDF3_64BIT).
        mode : {"w", "a"}, default: "w"
            Write ('w') or append ('a') mode. If mode='w', any existing file at
            this location will be overwritten. If mode='a', existing variables
            will be overwritten.
        format : {"NETCDF4", "NETCDF4_CLASSIC", "NETCDF3_64BIT", \
                  "NETCDF3_CLASSIC"}, optional
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
        engine : {"netcdf4", "scipy", "h5netcdf"}, optional
            Engine to use when writing netCDF files. If not provided, the
            default engine is chosen based on available dependencies, with a
            preference for 'netcdf4' if writing to a file on disk.
        encoding : dict, optional
            Nested dictionary with variable names as keys and dictionaries of
            variable specific encodings as values, e.g.,
            ``{"my_variable": {"dtype": "int16", "scale_factor": 0.1,
            "zlib": True}, ...}``

            The `h5netcdf` engine supports both the NetCDF4-style compression
            encoding parameters ``{"zlib": True, "complevel": 9}`` and the h5py
            ones ``{"compression": "gzip", "compression_opts": 9}``.
            This allows using any compression plugin installed in the HDF5
            library, e.g. LZF.

        unlimited_dims : iterable of hashable, optional
            Dimension(s) that should be serialized as unlimited dimensions.
            By default, no dimensions are treated as unlimited dimensions.
            Note that unlimited_dims may also be set via
            ``dataset.encoding["unlimited_dims"]``.
        compute: bool, default: True
            If true compute immediately, otherwise return a
            ``dask.delayed.Delayed`` object that can be computed later.
        invalid_netcdf: bool, default: False
            Only valid along with ``engine="h5netcdf"``. If True, allow writing
            hdf5 files which are invalid netcdf as described in
            https://github.com/shoyer/h5netcdf.
        """
        if encoding is None:
            encoding = {}
        from ..backends.api import to_netcdf

        return to_netcdf(
            self,
            path,
            mode,
            format=format,
            group=group,
            engine=engine,
            encoding=encoding,
            unlimited_dims=unlimited_dims,
            compute=compute,
            invalid_netcdf=invalid_netcdf,
        )

    def to_zarr(
        self,
        store: MutableMapping | str | PathLike | None = None,
        chunk_store: MutableMapping | str | PathLike | None = None,
        mode: str = None,
        synchronizer=None,
        group: str = None,
        encoding: Mapping = None,
        compute: bool = True,
        consolidated: bool | None = None,
        append_dim: Hashable = None,
        region: Mapping[str, slice] = None,
        safe_chunks: bool = True,
        storage_options: dict[str, str] = None,
    ) -> ZarrStore:
        """Write dataset contents to a zarr group.

        Zarr chunks are determined in the following way:

        - From the ``chunks`` attribute in each variable's ``encoding``
        - If the variable is a Dask array, from the dask chunks
        - If neither Dask chunks nor encoding chunks are present, chunks will
          be determined automatically by Zarr
        - If both Dask chunks and encoding chunks are present, encoding chunks
          will be used, provided that there is a many-to-one relationship between
          encoding chunks and dask chunks (i.e. Dask chunks are bigger than and
          evenly divide encoding chunks); otherwise raise a ``ValueError``.
          This restriction ensures that no synchronization / locks are required
          when writing. To disable this restriction, use ``safe_chunks=False``.

        Parameters
        ----------
        store : MutableMapping, str or path-like, optional
            Store or path to directory in local or remote file system.
        chunk_store : MutableMapping, str or path-like, optional
            Store or path to directory in local or remote file system only for Zarr
            array chunks. Requires zarr-python v2.4.0 or later.
        mode : {"w", "w-", "a", "r+", None}, optional
            Persistence mode: "w" means create (overwrite if exists);
            "w-" means create (fail if exists);
            "a" means override existing variables (create if does not exist);
            "r+" means modify existing array *values* only (raise an error if
            any metadata or shapes would change).
            The default mode is "a" if ``append_dim`` is set. Otherwise, it is
            "r+" if ``region`` is set and ``w-`` otherwise.
        synchronizer : object, optional
            Zarr array synchronizer.
        group : str, optional
            Group path. (a.k.a. `path` in zarr terminology.)
        encoding : dict, optional
            Nested dictionary with variable names as keys and dictionaries of
            variable specific encodings as values, e.g.,
            ``{"my_variable": {"dtype": "int16", "scale_factor": 0.1,}, ...}``
        compute : bool, optional
            If True write array data immediately, otherwise return a
            ``dask.delayed.Delayed`` object that can be computed to write
            array data later. Metadata is always updated eagerly.
        consolidated : bool, optional
            If True, apply zarr's `consolidate_metadata` function to the store
            after writing metadata and read existing stores with consolidated
            metadata; if False, do not. The default (`consolidated=None`) means
            write consolidated metadata and attempt to read consolidated
            metadata for existing stores (falling back to non-consolidated).
        append_dim : hashable, optional
            If set, the dimension along which the data will be appended. All
            other dimensions on overriden variables must remain the same size.
        region : dict, optional
            Optional mapping from dimension names to integer slices along
            dataset dimensions to indicate the region of existing zarr array(s)
            in which to write this dataset's data. For example,
            ``{'x': slice(0, 1000), 'y': slice(10000, 11000)}`` would indicate
            that values should be written to the region ``0:1000`` along ``x``
            and ``10000:11000`` along ``y``.

            Two restrictions apply to the use of ``region``:

            - If ``region`` is set, _all_ variables in a dataset must have at
              least one dimension in common with the region. Other variables
              should be written in a separate call to ``to_zarr()``.
            - Dimensions cannot be included in both ``region`` and
              ``append_dim`` at the same time. To create empty arrays to fill
              in with ``region``, use a separate call to ``to_zarr()`` with
              ``compute=False``. See "Appending to existing Zarr stores" in
              the reference documentation for full details.
        safe_chunks : bool, optional
            If True, only allow writes to when there is a many-to-one relationship
            between Zarr chunks (specified in encoding) and Dask chunks.
            Set False to override this restriction; however, data may become corrupted
            if Zarr arrays are written in parallel. This option may be useful in combination
            with ``compute=False`` to initialize a Zarr from an existing
            Dataset with aribtrary chunk structure.
        storage_options : dict, optional
            Any additional parameters for the storage backend (ignored for local
            paths).

        References
        ----------
        https://zarr.readthedocs.io/

        Notes
        -----
        Zarr chunking behavior:
            If chunks are found in the encoding argument or attribute
            corresponding to any DataArray, those chunks are used.
            If a DataArray is a dask array, it is written with those chunks.
            If not other chunks are found, Zarr uses its own heuristics to
            choose automatic chunk sizes.

        encoding:
            The encoding attribute (if exists) of the DataArray(s) will be
            used. Override any existing encodings by providing the ``encoding`` kwarg.

        See Also
        --------
        :ref:`io.zarr`
            The I/O user guide, with more details and examples.
        """
        from ..backends.api import to_zarr

        if encoding is None:
            encoding = {}

        return to_zarr(
            self,
            store=store,
            chunk_store=chunk_store,
            storage_options=storage_options,
            mode=mode,
            synchronizer=synchronizer,
            group=group,
            encoding=encoding,
            compute=compute,
            consolidated=consolidated,
            append_dim=append_dim,
            region=region,
            safe_chunks=safe_chunks,
        )

    def __repr__(self) -> str:
        return formatting.dataset_repr(self)

    def _repr_html_(self):
        if OPTIONS["display_style"] == "text":
            return f"<pre>{escape(repr(self))}</pre>"
        return formatting_html.dataset_repr(self)

    def info(self, buf=None) -> None:
        """
        Concise summary of a Dataset variables and attributes.

        Parameters
        ----------
        buf : file-like, default: sys.stdout
            writable buffer

        See Also
        --------
        pandas.DataFrame.assign
        ncdump : netCDF's ncdump
        """
        if buf is None:  # pragma: no cover
            buf = sys.stdout

        lines = []
        lines.append("xarray.Dataset {")
        lines.append("dimensions:")
        for name, size in self.dims.items():
            lines.append(f"\t{name} = {size} ;")
        lines.append("\nvariables:")
        for name, da in self.variables.items():
            dims = ", ".join(da.dims)
            lines.append(f"\t{da.dtype} {name}({dims}) ;")
            for k, v in da.attrs.items():
                lines.append(f"\t\t{name}:{k} = {v} ;")
        lines.append("\n// global attributes:")
        for k, v in self.attrs.items():
            lines.append(f"\t:{k} = {v} ;")
        lines.append("}")

        buf.write("\n".join(lines))

    @property
    def chunks(self) -> Mapping[Hashable, tuple[int, ...]]:
        """
        Mapping from dimension names to block lengths for this dataset's data, or None if
        the underlying data is not a dask array.
        Cannot be modified directly, but can be modified by calling .chunk().

        Same as Dataset.chunksizes, but maintained for backwards compatibility.

        See Also
        --------
        Dataset.chunk
        Dataset.chunksizes
        xarray.unify_chunks
        """
        return get_chunksizes(self.variables.values())

    @property
    def chunksizes(self) -> Mapping[Any, tuple[int, ...]]:
        """
        Mapping from dimension names to block lengths for this dataset's data, or None if
        the underlying data is not a dask array.
        Cannot be modified directly, but can be modified by calling .chunk().

        Same as Dataset.chunks.

        See Also
        --------
        Dataset.chunk
        Dataset.chunks
        xarray.unify_chunks
        """
        return get_chunksizes(self.variables.values())

    def chunk(
        self,
        chunks: (
            int | Literal["auto"] | Mapping[Any, None | int | str | tuple[int, ...]]
        ) = {},  # {} even though it's technically unsafe, is being used intentionally here (#4667)
        name_prefix: str = "xarray-",
        token: str = None,
        lock: bool = False,
    ) -> Dataset:
        """Coerce all arrays in this dataset into dask arrays with the given
        chunks.

        Non-dask arrays in this dataset will be converted to dask arrays. Dask
        arrays will be rechunked to the given chunk sizes.

        If neither chunks is not provided for one or more dimensions, chunk
        sizes along that dimension will not be updated; non-dask arrays will be
        converted into dask arrays with a single block.

        Parameters
        ----------
        chunks : int, "auto" or mapping of hashable to int, optional
            Chunk sizes along each dimension, e.g., ``5``, ``"auto"``, or
            ``{"x": 5, "y": 5}``.
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

        See Also
        --------
        Dataset.chunks
        Dataset.chunksizes
        xarray.unify_chunks
        """
        if chunks is None:
            warnings.warn(
                "None value for 'chunks' is deprecated. "
                "It will raise an error in the future. Use instead '{}'",
                category=FutureWarning,
            )
            chunks = {}

        if isinstance(chunks, (Number, str, int)):
            chunks = dict.fromkeys(self.dims, chunks)

        bad_dims = chunks.keys() - self.dims.keys()
        if bad_dims:
            raise ValueError(
                f"some chunks keys are not dimensions on this object: {bad_dims}"
            )

        variables = {
            k: _maybe_chunk(k, v, chunks, token, lock, name_prefix)
            for k, v in self.variables.items()
        }
        return self._replace(variables)

    def _validate_indexers(
        self, indexers: Mapping[Any, Any], missing_dims: str = "raise"
    ) -> Iterator[tuple[Hashable, int | slice | np.ndarray | Variable]]:
        """Here we make sure
        + indexer has a valid keys
        + indexer is in a valid data type
        + string indexers are cast to the appropriate date type if the
          associated index is a DatetimeIndex or CFTimeIndex
        """
        from .dataarray import DataArray

        indexers = drop_dims_from_indexers(indexers, self.dims, missing_dims)

        # all indexers should be int, slice, np.ndarrays, or Variable
        for k, v in indexers.items():
            if isinstance(v, (int, slice, Variable)):
                yield k, v
            elif isinstance(v, DataArray):
                yield k, v.variable
            elif isinstance(v, tuple):
                yield k, as_variable(v)
            elif isinstance(v, Dataset):
                raise TypeError("cannot use a Dataset as an indexer")
            elif isinstance(v, Sequence) and len(v) == 0:
                yield k, np.empty((0,), dtype="int64")
            else:
                v = np.asarray(v)

                if v.dtype.kind in "US":
                    # TODO: benbovy - flexible indexes
                    # update when CFTimeIndex has its own xarray index class
                    index = self.xindexes[k].to_pandas_index()
                    if isinstance(index, pd.DatetimeIndex):
                        v = v.astype("datetime64[ns]")
                    elif isinstance(index, xr.CFTimeIndex):
                        v = _parse_array_of_cftime_strings(v, index.date_type)

                if v.ndim > 1:
                    raise IndexError(
                        "Unlabeled multi-dimensional array cannot be "
                        "used for indexing: {}".format(k)
                    )
                yield k, v

    def _validate_interp_indexers(
        self, indexers: Mapping[Any, Any]
    ) -> Iterator[tuple[Hashable, Variable]]:
        """Variant of _validate_indexers to be used for interpolation"""
        for k, v in self._validate_indexers(indexers):
            if isinstance(v, Variable):
                if v.ndim == 1:
                    yield k, v.to_index_variable()
                else:
                    yield k, v
            elif isinstance(v, int):
                yield k, Variable((), v, attrs=self.coords[k].attrs)
            elif isinstance(v, np.ndarray):
                if v.ndim == 0:
                    yield k, Variable((), v, attrs=self.coords[k].attrs)
                elif v.ndim == 1:
                    yield k, IndexVariable((k,), v, attrs=self.coords[k].attrs)
                else:
                    raise AssertionError()  # Already tested by _validate_indexers
            else:
                raise TypeError(type(v))

    def _get_indexers_coords_and_indexes(self, indexers):
        """Extract coordinates and indexes from indexers.

        Only coordinate with a name different from any of self.variables will
        be attached.
        """
        from .dataarray import DataArray

        coords_list = []
        for k, v in indexers.items():
            if isinstance(v, DataArray):
                if v.dtype.kind == "b":
                    if v.ndim != 1:  # we only support 1-d boolean array
                        raise ValueError(
                            "{:d}d-boolean array is used for indexing along "
                            "dimension {!r}, but only 1d boolean arrays are "
                            "supported.".format(v.ndim, k)
                        )
                    # Make sure in case of boolean DataArray, its
                    # coordinate also should be indexed.
                    v_coords = v[v.values.nonzero()[0]].coords
                else:
                    v_coords = v.coords
                coords_list.append(v_coords)

        # we don't need to call align() explicitly or check indexes for
        # alignment, because merge_variables already checks for exact alignment
        # between dimension coordinates
        coords, indexes = merge_coordinates_without_align(coords_list)
        assert_coordinate_consistent(self, coords)

        # silently drop the conflicted variables.
        attached_coords = {k: v for k, v in coords.items() if k not in self._variables}
        attached_indexes = {
            k: v for k, v in indexes.items() if k not in self._variables
        }
        return attached_coords, attached_indexes

    def isel(
        self,
        indexers: Mapping[Any, Any] = None,
        drop: bool = False,
        missing_dims: str = "raise",
        **indexers_kwargs: Any,
    ) -> Dataset:
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
        missing_dims : {"raise", "warn", "ignore"}, default: "raise"
            What to do if dimensions that should be selected from are not present in the
            Dataset:
            - "raise": raise an exception
            - "warning": raise a warning, and ignore the missing dimensions
            - "ignore": ignore the missing dimensions
        **indexers_kwargs : {dim: indexer, ...}, optional
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
        indexers = either_dict_or_kwargs(indexers, indexers_kwargs, "isel")
        if any(is_fancy_indexer(idx) for idx in indexers.values()):
            return self._isel_fancy(indexers, drop=drop, missing_dims=missing_dims)

        # Much faster algorithm for when all indexers are ints, slices, one-dimensional
        # lists, or zero or one-dimensional np.ndarray's
        indexers = drop_dims_from_indexers(indexers, self.dims, missing_dims)

        variables = {}
        dims: dict[Hashable, int] = {}
        coord_names = self._coord_names.copy()
        indexes = self._indexes.copy() if self._indexes is not None else None

        for var_name, var_value in self._variables.items():
            var_indexers = {k: v for k, v in indexers.items() if k in var_value.dims}
            if var_indexers:
                var_value = var_value.isel(var_indexers)
                if drop and var_value.ndim == 0 and var_name in coord_names:
                    coord_names.remove(var_name)
                    if indexes:
                        indexes.pop(var_name, None)
                    continue
                if indexes and var_name in indexes:
                    if var_value.ndim == 1:
                        indexes[var_name] = var_value._to_xindex()
                    else:
                        del indexes[var_name]
            variables[var_name] = var_value
            dims.update(zip(var_value.dims, var_value.shape))

        return self._construct_direct(
            variables=variables,
            coord_names=coord_names,
            dims=dims,
            attrs=self._attrs,
            indexes=indexes,
            encoding=self._encoding,
            close=self._close,
        )

    def _isel_fancy(
        self,
        indexers: Mapping[Any, Any],
        *,
        drop: bool,
        missing_dims: str = "raise",
    ) -> Dataset:
        # Note: we need to preserve the original indexers variable in order to merge the
        # coords below
        indexers_list = list(self._validate_indexers(indexers, missing_dims))

        variables: dict[Hashable, Variable] = {}
        indexes: dict[Hashable, Index] = {}

        for name, var in self.variables.items():
            var_indexers = {k: v for k, v in indexers_list if k in var.dims}
            if drop and name in var_indexers:
                continue  # drop this variable

            if name in self.xindexes:
                new_var, new_index = isel_variable_and_index(
                    name, var, self.xindexes[name], var_indexers
                )
                if new_index is not None:
                    indexes[name] = new_index
            elif var_indexers:
                new_var = var.isel(indexers=var_indexers)
            else:
                new_var = var.copy(deep=False)

            variables[name] = new_var

        coord_names = self._coord_names & variables.keys()
        selected = self._replace_with_new_dims(variables, coord_names, indexes)

        # Extract coordinates from indexers
        coord_vars, new_indexes = selected._get_indexers_coords_and_indexes(indexers)
        variables.update(coord_vars)
        indexes.update(new_indexes)
        coord_names = self._coord_names & variables.keys() | coord_vars.keys()
        return self._replace_with_new_dims(variables, coord_names, indexes=indexes)

    def sel(
        self,
        indexers: Mapping[Any, Any] = None,
        method: str = None,
        tolerance: Number = None,
        drop: bool = False,
        **indexers_kwargs: Any,
    ) -> Dataset:
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
        indexers = either_dict_or_kwargs(indexers, indexers_kwargs, "sel")
        pos_indexers, new_indexes = remap_label_indexers(
            self, indexers=indexers, method=method, tolerance=tolerance
        )
        # TODO: benbovy - flexible indexes: also use variables returned by Index.query
        # (temporary dirty fix).
        new_indexes = {k: v[0] for k, v in new_indexes.items()}

        result = self.isel(indexers=pos_indexers, drop=drop)
        return result._overwrite_indexes(new_indexes)

    def head(
        self,
        indexers: Mapping[Any, int] | int | None = None,
        **indexers_kwargs: Any,
    ) -> Dataset:
        """Returns a new dataset with the first `n` values of each array
        for the specified dimension(s).

        Parameters
        ----------
        indexers : dict or int, default: 5
            A dict with keys matching dimensions and integer values `n`
            or a single integer `n` applied over all dimensions.
            One of indexers or indexers_kwargs must be provided.
        **indexers_kwargs : {dim: n, ...}, optional
            The keyword arguments form of ``indexers``.
            One of indexers or indexers_kwargs must be provided.

        See Also
        --------
        Dataset.tail
        Dataset.thin
        DataArray.head
        """
        if not indexers_kwargs:
            if indexers is None:
                indexers = 5
            if not isinstance(indexers, int) and not is_dict_like(indexers):
                raise TypeError("indexers must be either dict-like or a single integer")
        if isinstance(indexers, int):
            indexers = {dim: indexers for dim in self.dims}
        indexers = either_dict_or_kwargs(indexers, indexers_kwargs, "head")
        for k, v in indexers.items():
            if not isinstance(v, int):
                raise TypeError(
                    "expected integer type indexer for "
                    f"dimension {k!r}, found {type(v)!r}"
                )
            elif v < 0:
                raise ValueError(
                    "expected positive integer as indexer "
                    f"for dimension {k!r}, found {v}"
                )
        indexers_slices = {k: slice(val) for k, val in indexers.items()}
        return self.isel(indexers_slices)

    def tail(
        self,
        indexers: Mapping[Any, int] | int | None = None,
        **indexers_kwargs: Any,
    ) -> Dataset:
        """Returns a new dataset with the last `n` values of each array
        for the specified dimension(s).

        Parameters
        ----------
        indexers : dict or int, default: 5
            A dict with keys matching dimensions and integer values `n`
            or a single integer `n` applied over all dimensions.
            One of indexers or indexers_kwargs must be provided.
        **indexers_kwargs : {dim: n, ...}, optional
            The keyword arguments form of ``indexers``.
            One of indexers or indexers_kwargs must be provided.

        See Also
        --------
        Dataset.head
        Dataset.thin
        DataArray.tail
        """
        if not indexers_kwargs:
            if indexers is None:
                indexers = 5
            if not isinstance(indexers, int) and not is_dict_like(indexers):
                raise TypeError("indexers must be either dict-like or a single integer")
        if isinstance(indexers, int):
            indexers = {dim: indexers for dim in self.dims}
        indexers = either_dict_or_kwargs(indexers, indexers_kwargs, "tail")
        for k, v in indexers.items():
            if not isinstance(v, int):
                raise TypeError(
                    "expected integer type indexer for "
                    f"dimension {k!r}, found {type(v)!r}"
                )
            elif v < 0:
                raise ValueError(
                    "expected positive integer as indexer "
                    f"for dimension {k!r}, found {v}"
                )
        indexers_slices = {
            k: slice(-val, None) if val != 0 else slice(val)
            for k, val in indexers.items()
        }
        return self.isel(indexers_slices)

    def thin(
        self,
        indexers: Mapping[Any, int] | int | None = None,
        **indexers_kwargs: Any,
    ) -> Dataset:
        """Returns a new dataset with each array indexed along every `n`-th
        value for the specified dimension(s)

        Parameters
        ----------
        indexers : dict or int
            A dict with keys matching dimensions and integer values `n`
            or a single integer `n` applied over all dimensions.
            One of indexers or indexers_kwargs must be provided.
        **indexers_kwargs : {dim: n, ...}, optional
            The keyword arguments form of ``indexers``.
            One of indexers or indexers_kwargs must be provided.

        See Also
        --------
        Dataset.head
        Dataset.tail
        DataArray.thin
        """
        if (
            not indexers_kwargs
            and not isinstance(indexers, int)
            and not is_dict_like(indexers)
        ):
            raise TypeError("indexers must be either dict-like or a single integer")
        if isinstance(indexers, int):
            indexers = {dim: indexers for dim in self.dims}
        indexers = either_dict_or_kwargs(indexers, indexers_kwargs, "thin")
        for k, v in indexers.items():
            if not isinstance(v, int):
                raise TypeError(
                    "expected integer type indexer for "
                    f"dimension {k!r}, found {type(v)!r}"
                )
            elif v < 0:
                raise ValueError(
                    "expected positive integer as indexer "
                    f"for dimension {k!r}, found {v}"
                )
            elif v == 0:
                raise ValueError("step cannot be zero")
        indexers_slices = {k: slice(None, None, val) for k, val in indexers.items()}
        return self.isel(indexers_slices)

    def broadcast_like(
        self, other: Dataset | DataArray, exclude: Iterable[Hashable] = None
    ) -> Dataset:
        """Broadcast this DataArray against another Dataset or DataArray.
        This is equivalent to xr.broadcast(other, self)[1]

        Parameters
        ----------
        other : Dataset or DataArray
            Object against which to broadcast this array.
        exclude : iterable of hashable, optional
            Dimensions that must not be broadcasted

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
        other: Dataset | DataArray,
        method: str = None,
        tolerance: int | float | Iterable[int | float] | None = None,
        copy: bool = True,
        fill_value: Any = dtypes.NA,
    ) -> Dataset:
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
            Method to use for filling index values from other not found in this
            dataset:

            * None (default): don't fill gaps
            * pad / ffill: propagate last valid index value forward
            * backfill / bfill: propagate next valid index value backward
            * nearest: use nearest valid index value
        tolerance : optional
            Maximum distance between original and new labels for inexact
            matches. The values of the index at the matching locations must
            satisfy the equation ``abs(index[indexer] - target) <= tolerance``.
            Tolerance may be a scalar value, which applies the same tolerance
            to all values, or list-like, which applies variable tolerance per
            element. List-like must be the same size as the index and its dtype
            must exactly match the index’s type.
        copy : bool, optional
            If ``copy=True``, data in the return value is always copied. If
            ``copy=False`` and reindexing is unnecessary, or can be performed
            with only slice operations, then the output may share memory with
            the input. In either case, a new xarray object is always returned.
        fill_value : scalar or dict-like, optional
            Value to use for newly missing values. If a dict-like maps
            variable names to fill values.

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
        return self.reindex(
            indexers=indexers,
            method=method,
            copy=copy,
            fill_value=fill_value,
            tolerance=tolerance,
        )

    def reindex(
        self,
        indexers: Mapping[Any, Any] = None,
        method: str = None,
        tolerance: int | float | Iterable[int | float] | None = None,
        copy: bool = True,
        fill_value: Any = dtypes.NA,
        **indexers_kwargs: Any,
    ) -> Dataset:
        """Conform this object onto a new set of indexes, filling in
        missing values with ``fill_value``. The default fill value is NaN.

        Parameters
        ----------
        indexers : dict, optional
            Dictionary with keys given by dimension names and values given by
            arrays of coordinates tick labels. Any mis-matched coordinate
            values will be filled in with NaN, and any mis-matched dimension
            names will simply be ignored.
            One of indexers or indexers_kwargs must be provided.
        method : {None, "nearest", "pad", "ffill", "backfill", "bfill"}, optional
            Method to use for filling index values in ``indexers`` not found in
            this dataset:

            * None (default): don't fill gaps
            * pad / ffill: propagate last valid index value forward
            * backfill / bfill: propagate next valid index value backward
            * nearest: use nearest valid index value
        tolerance : optional
            Maximum distance between original and new labels for inexact
            matches. The values of the index at the matching locations must
            satisfy the equation ``abs(index[indexer] - target) <= tolerance``.
            Tolerance may be a scalar value, which applies the same tolerance
            to all values, or list-like, which applies variable tolerance per
            element. List-like must be the same size as the index and its dtype
            must exactly match the index’s type.
        copy : bool, optional
            If ``copy=True``, data in the return value is always copied. If
            ``copy=False`` and reindexing is unnecessary, or can be performed
            with only slice operations, then the output may share memory with
            the input. In either case, a new xarray object is always returned.
        fill_value : scalar or dict-like, optional
            Value to use for newly missing values. If a dict-like,
            maps variable names (including coordinates) to fill values.
        sparse : bool, default: False
            use sparse-array.
        **indexers_kwargs : {dim: indexer, ...}, optional
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

        Examples
        --------
        Create a dataset with some fictional data.

        >>> x = xr.Dataset(
        ...     {
        ...         "temperature": ("station", 20 * np.random.rand(4)),
        ...         "pressure": ("station", 500 * np.random.rand(4)),
        ...     },
        ...     coords={"station": ["boston", "nyc", "seattle", "denver"]},
        ... )
        >>> x
        <xarray.Dataset>
        Dimensions:      (station: 4)
        Coordinates:
          * station      (station) <U7 'boston' 'nyc' 'seattle' 'denver'
        Data variables:
            temperature  (station) float64 10.98 14.3 12.06 10.9
            pressure     (station) float64 211.8 322.9 218.8 445.9
        >>> x.indexes
        station: Index(['boston', 'nyc', 'seattle', 'denver'], dtype='object', name='station')

        Create a new index and reindex the dataset. By default values in the new index that
        do not have corresponding records in the dataset are assigned `NaN`.

        >>> new_index = ["boston", "austin", "seattle", "lincoln"]
        >>> x.reindex({"station": new_index})
        <xarray.Dataset>
        Dimensions:      (station: 4)
        Coordinates:
          * station      (station) <U7 'boston' 'austin' 'seattle' 'lincoln'
        Data variables:
            temperature  (station) float64 10.98 nan 12.06 nan
            pressure     (station) float64 211.8 nan 218.8 nan

        We can fill in the missing values by passing a value to the keyword `fill_value`.

        >>> x.reindex({"station": new_index}, fill_value=0)
        <xarray.Dataset>
        Dimensions:      (station: 4)
        Coordinates:
          * station      (station) <U7 'boston' 'austin' 'seattle' 'lincoln'
        Data variables:
            temperature  (station) float64 10.98 0.0 12.06 0.0
            pressure     (station) float64 211.8 0.0 218.8 0.0

        We can also use different fill values for each variable.

        >>> x.reindex(
        ...     {"station": new_index}, fill_value={"temperature": 0, "pressure": 100}
        ... )
        <xarray.Dataset>
        Dimensions:      (station: 4)
        Coordinates:
          * station      (station) <U7 'boston' 'austin' 'seattle' 'lincoln'
        Data variables:
            temperature  (station) float64 10.98 0.0 12.06 0.0
            pressure     (station) float64 211.8 100.0 218.8 100.0

        Because the index is not monotonically increasing or decreasing, we cannot use arguments
        to the keyword method to fill the `NaN` values.

        >>> x.reindex({"station": new_index}, method="nearest")
        Traceback (most recent call last):
        ...
            raise ValueError('index must be monotonic increasing or decreasing')
        ValueError: index must be monotonic increasing or decreasing

        To further illustrate the filling functionality in reindex, we will create a
        dataset with a monotonically increasing index (for example, a sequence of dates).

        >>> x2 = xr.Dataset(
        ...     {
        ...         "temperature": (
        ...             "time",
        ...             [15.57, 12.77, np.nan, 0.3081, 16.59, 15.12],
        ...         ),
        ...         "pressure": ("time", 500 * np.random.rand(6)),
        ...     },
        ...     coords={"time": pd.date_range("01/01/2019", periods=6, freq="D")},
        ... )
        >>> x2
        <xarray.Dataset>
        Dimensions:      (time: 6)
        Coordinates:
          * time         (time) datetime64[ns] 2019-01-01 2019-01-02 ... 2019-01-06
        Data variables:
            temperature  (time) float64 15.57 12.77 nan 0.3081 16.59 15.12
            pressure     (time) float64 481.8 191.7 395.9 264.4 284.0 462.8

        Suppose we decide to expand the dataset to cover a wider date range.

        >>> time_index2 = pd.date_range("12/29/2018", periods=10, freq="D")
        >>> x2.reindex({"time": time_index2})
        <xarray.Dataset>
        Dimensions:      (time: 10)
        Coordinates:
          * time         (time) datetime64[ns] 2018-12-29 2018-12-30 ... 2019-01-07
        Data variables:
            temperature  (time) float64 nan nan nan 15.57 ... 0.3081 16.59 15.12 nan
            pressure     (time) float64 nan nan nan 481.8 ... 264.4 284.0 462.8 nan

        The index entries that did not have a value in the original data frame (for example, `2018-12-29`)
        are by default filled with NaN. If desired, we can fill in the missing values using one of several options.

        For example, to back-propagate the last valid value to fill the `NaN` values,
        pass `bfill` as an argument to the `method` keyword.

        >>> x3 = x2.reindex({"time": time_index2}, method="bfill")
        >>> x3
        <xarray.Dataset>
        Dimensions:      (time: 10)
        Coordinates:
          * time         (time) datetime64[ns] 2018-12-29 2018-12-30 ... 2019-01-07
        Data variables:
            temperature  (time) float64 15.57 15.57 15.57 15.57 ... 16.59 15.12 nan
            pressure     (time) float64 481.8 481.8 481.8 481.8 ... 284.0 462.8 nan

        Please note that the `NaN` value present in the original dataset (at index value `2019-01-03`)
        will not be filled by any of the value propagation schemes.

        >>> x2.where(x2.temperature.isnull(), drop=True)
        <xarray.Dataset>
        Dimensions:      (time: 1)
        Coordinates:
          * time         (time) datetime64[ns] 2019-01-03
        Data variables:
            temperature  (time) float64 nan
            pressure     (time) float64 395.9
        >>> x3.where(x3.temperature.isnull(), drop=True)
        <xarray.Dataset>
        Dimensions:      (time: 2)
        Coordinates:
          * time         (time) datetime64[ns] 2019-01-03 2019-01-07
        Data variables:
            temperature  (time) float64 nan nan
            pressure     (time) float64 395.9 nan

        This is because filling while reindexing does not look at dataset values, but only compares
        the original and desired indexes. If you do want to fill in the `NaN` values present in the
        original dataset, use the :py:meth:`~Dataset.fillna()` method.

        """
        return self._reindex(
            indexers,
            method,
            tolerance,
            copy,
            fill_value,
            sparse=False,
            **indexers_kwargs,
        )

    def _reindex(
        self,
        indexers: Mapping[Any, Any] = None,
        method: str = None,
        tolerance: int | float | Iterable[int | float] | None = None,
        copy: bool = True,
        fill_value: Any = dtypes.NA,
        sparse: bool = False,
        **indexers_kwargs: Any,
    ) -> Dataset:
        """
        same to _reindex but support sparse option
        """
        indexers = utils.either_dict_or_kwargs(indexers, indexers_kwargs, "reindex")

        bad_dims = [d for d in indexers if d not in self.dims]
        if bad_dims:
            raise ValueError(f"invalid reindex dimensions: {bad_dims}")

        variables, indexes = alignment.reindex_variables(
            self.variables,
            self.sizes,
            self.xindexes,
            indexers,
            method,
            tolerance,
            copy=copy,
            fill_value=fill_value,
            sparse=sparse,
        )
        coord_names = set(self._coord_names)
        coord_names.update(indexers)
        return self._replace_with_new_dims(variables, coord_names, indexes=indexes)

    def interp(
        self,
        coords: Mapping[Any, Any] = None,
        method: str = "linear",
        assume_sorted: bool = False,
        kwargs: Mapping[str, Any] = None,
        method_non_numeric: str = "nearest",
        **coords_kwargs: Any,
    ) -> Dataset:
        """Multidimensional interpolation of Dataset.

        Parameters
        ----------
        coords : dict, optional
            Mapping from dimension names to the new coordinates.
            New coordinate can be a scalar, array-like or DataArray.
            If DataArrays are passed as new coordinates, their dimensions are
            used for the broadcasting. Missing values are skipped.
        method : str, optional
            {"linear", "nearest"} for multidimensional array,
            {"linear", "nearest", "zero", "slinear", "quadratic", "cubic"}
            for 1-dimensional array. "linear" is used by default.
        assume_sorted : bool, optional
            If False, values of coordinates that are interpolated over can be
            in any order and they are sorted first. If True, interpolated
            coordinates are assumed to be an array of monotonically increasing
            values.
        kwargs : dict, optional
            Additional keyword arguments passed to scipy's interpolator. Valid
            options and their behavior depend on if 1-dimensional or
            multi-dimensional interpolation is used.
        method_non_numeric : {"nearest", "pad", "ffill", "backfill", "bfill"}, optional
            Method for non-numeric types. Passed on to :py:meth:`Dataset.reindex`.
            ``"nearest"`` is used by default.
        **coords_kwargs : {dim: coordinate, ...}, optional
            The keyword arguments form of ``coords``.
            One of coords or coords_kwargs must be provided.

        Returns
        -------
        interpolated : Dataset
            New dataset on the new coordinates.

        Notes
        -----
        scipy is required.

        See Also
        --------
        scipy.interpolate.interp1d
        scipy.interpolate.interpn

        Examples
        --------
        >>> ds = xr.Dataset(
        ...     data_vars={
        ...         "a": ("x", [5, 7, 4]),
        ...         "b": (
        ...             ("x", "y"),
        ...             [[1, 4, 2, 9], [2, 7, 6, np.nan], [6, np.nan, 5, 8]],
        ...         ),
        ...     },
        ...     coords={"x": [0, 1, 2], "y": [10, 12, 14, 16]},
        ... )
        >>> ds
        <xarray.Dataset>
        Dimensions:  (x: 3, y: 4)
        Coordinates:
          * x        (x) int64 0 1 2
          * y        (y) int64 10 12 14 16
        Data variables:
            a        (x) int64 5 7 4
            b        (x, y) float64 1.0 4.0 2.0 9.0 2.0 7.0 6.0 nan 6.0 nan 5.0 8.0

        1D interpolation with the default method (linear):

        >>> ds.interp(x=[0, 0.75, 1.25, 1.75])
        <xarray.Dataset>
        Dimensions:  (x: 4, y: 4)
        Coordinates:
          * y        (y) int64 10 12 14 16
          * x        (x) float64 0.0 0.75 1.25 1.75
        Data variables:
            a        (x) float64 5.0 6.5 6.25 4.75
            b        (x, y) float64 1.0 4.0 2.0 nan 1.75 6.25 ... nan 5.0 nan 5.25 nan

        1D interpolation with a different method:

        >>> ds.interp(x=[0, 0.75, 1.25, 1.75], method="nearest")
        <xarray.Dataset>
        Dimensions:  (x: 4, y: 4)
        Coordinates:
          * y        (y) int64 10 12 14 16
          * x        (x) float64 0.0 0.75 1.25 1.75
        Data variables:
            a        (x) float64 5.0 7.0 7.0 4.0
            b        (x, y) float64 1.0 4.0 2.0 9.0 2.0 7.0 ... 6.0 nan 6.0 nan 5.0 8.0

        1D extrapolation:

        >>> ds.interp(
        ...     x=[1, 1.5, 2.5, 3.5],
        ...     method="linear",
        ...     kwargs={"fill_value": "extrapolate"},
        ... )
        <xarray.Dataset>
        Dimensions:  (x: 4, y: 4)
        Coordinates:
          * y        (y) int64 10 12 14 16
          * x        (x) float64 1.0 1.5 2.5 3.5
        Data variables:
            a        (x) float64 7.0 5.5 2.5 -0.5
            b        (x, y) float64 2.0 7.0 6.0 nan 4.0 nan ... 4.5 nan 12.0 nan 3.5 nan

        2D interpolation:

        >>> ds.interp(x=[0, 0.75, 1.25, 1.75], y=[11, 13, 15], method="linear")
        <xarray.Dataset>
        Dimensions:  (x: 4, y: 3)
        Coordinates:
          * x        (x) float64 0.0 0.75 1.25 1.75
          * y        (y) int64 11 13 15
        Data variables:
            a        (x) float64 5.0 6.5 6.25 4.75
            b        (x, y) float64 2.5 3.0 nan 4.0 5.625 nan nan nan nan nan nan nan
        """
        from . import missing

        if kwargs is None:
            kwargs = {}

        coords = either_dict_or_kwargs(coords, coords_kwargs, "interp")
        indexers = dict(self._validate_interp_indexers(coords))

        if coords:
            # This avoids broadcasting over coordinates that are both in
            # the original array AND in the indexing array. It essentially
            # forces interpolation along the shared coordinates.
            sdims = (
                set(self.dims)
                .intersection(*[set(nx.dims) for nx in indexers.values()])
                .difference(coords.keys())
            )
            indexers.update({d: self.variables[d] for d in sdims})

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
            if _contains_datetime_like_objects(
                x
            ) and not _contains_datetime_like_objects(new_x):
                raise TypeError(
                    "When interpolating over a datetime-like "
                    "coordinate, the coordinates to "
                    "interpolate to must be either datetime "
                    "strings or datetimes. "
                    "Instead got\n{}".format(new_x)
                )
            return x, new_x

        validated_indexers = {
            k: _validate_interp_indexer(maybe_variable(obj, k), v)
            for k, v in indexers.items()
        }

        # optimization: subset to coordinate range of the target index
        if method in ["linear", "nearest"]:
            for k, v in validated_indexers.items():
                obj, newidx = missing._localize(obj, {k: v})
                validated_indexers[k] = newidx[k]

        # optimization: create dask coordinate arrays once per Dataset
        # rather than once per Variable when dask.array.unify_chunks is called later
        # GH4739
        if obj.__dask_graph__():
            dask_indexers = {
                k: (index.to_base_variable().chunk(), dest.to_base_variable().chunk())
                for k, (index, dest) in validated_indexers.items()
            }

        variables: dict[Hashable, Variable] = {}
        to_reindex: dict[Hashable, Variable] = {}
        for name, var in obj._variables.items():
            if name in indexers:
                continue

            if is_duck_dask_array(var.data):
                use_indexers = dask_indexers
            else:
                use_indexers = validated_indexers

            dtype_kind = var.dtype.kind
            if dtype_kind in "uifc":
                # For normal number types do the interpolation:
                var_indexers = {k: v for k, v in use_indexers.items() if k in var.dims}
                variables[name] = missing.interp(var, var_indexers, method, **kwargs)
            elif dtype_kind in "ObU" and (use_indexers.keys() & var.dims):
                # For types that we do not understand do stepwise
                # interpolation to avoid modifying the elements.
                # Use reindex_variables instead because it supports
                # booleans and objects and retains the dtype but inside
                # this loop there might be some duplicate code that slows it
                # down, therefore collect these signals and run it later:
                to_reindex[name] = var
            elif all(d not in indexers for d in var.dims):
                # For anything else we can only keep variables if they
                # are not dependent on any coords that are being
                # interpolated along:
                variables[name] = var

        if to_reindex:
            # Reindex variables:
            variables_reindex = alignment.reindex_variables(
                variables=to_reindex,
                sizes=obj.sizes,
                indexes=obj.xindexes,
                indexers={k: v[-1] for k, v in validated_indexers.items()},
                method=method_non_numeric,
            )[0]
            variables.update(variables_reindex)

        # Get the coords that also exist in the variables:
        coord_names = obj._coord_names & variables.keys()
        # Get the indexes that are not being interpolated along:
        indexes = {k: v for k, v in obj.xindexes.items() if k not in indexers}
        selected = self._replace_with_new_dims(
            variables.copy(), coord_names, indexes=indexes
        )

        # Attach indexer as coordinate
        variables.update(indexers)
        for k, v in indexers.items():
            assert isinstance(v, Variable)
            if v.dims == (k,):
                indexes[k] = v._to_xindex()

        # Extract coordinates from indexers
        coord_vars, new_indexes = selected._get_indexers_coords_and_indexes(coords)
        variables.update(coord_vars)
        indexes.update(new_indexes)

        coord_names = obj._coord_names & variables.keys() | coord_vars.keys()
        return self._replace_with_new_dims(variables, coord_names, indexes=indexes)

    def interp_like(
        self,
        other: Dataset | DataArray,
        method: str = "linear",
        assume_sorted: bool = False,
        kwargs: Mapping[str, Any] = None,
        method_non_numeric: str = "nearest",
    ) -> Dataset:
        """Interpolate this object onto the coordinates of another object,
        filling the out of range values with NaN.

        Parameters
        ----------
        other : Dataset or DataArray
            Object with an 'indexes' attribute giving a mapping from dimension
            names to an 1d array-like, which provides coordinates upon
            which to index the variables in this dataset. Missing values are skipped.
        method : str, optional
            {"linear", "nearest"} for multidimensional array,
            {"linear", "nearest", "zero", "slinear", "quadratic", "cubic"}
            for 1-dimensional array. 'linear' is used by default.
        assume_sorted : bool, optional
            If False, values of coordinates that are interpolated over can be
            in any order and they are sorted first. If True, interpolated
            coordinates are assumed to be an array of monotonically increasing
            values.
        kwargs : dict, optional
            Additional keyword passed to scipy's interpolator.
        method_non_numeric : {"nearest", "pad", "ffill", "backfill", "bfill"}, optional
            Method for non-numeric types. Passed on to :py:meth:`Dataset.reindex`.
            ``"nearest"`` is used by default.

        Returns
        -------
        interpolated : Dataset
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

        numeric_coords: dict[Hashable, pd.Index] = {}
        object_coords: dict[Hashable, pd.Index] = {}
        for k, v in coords.items():
            if v.dtype.kind in "uifcMm":
                numeric_coords[k] = v
            else:
                object_coords[k] = v

        ds = self
        if object_coords:
            # We do not support interpolation along object coordinate.
            # reindex instead.
            ds = self.reindex(object_coords)
        return ds.interp(
            coords=numeric_coords,
            method=method,
            assume_sorted=assume_sorted,
            kwargs=kwargs,
            method_non_numeric=method_non_numeric,
        )

    # Helper methods for rename()
    def _rename_vars(self, name_dict, dims_dict):
        variables = {}
        coord_names = set()
        for k, v in self.variables.items():
            var = v.copy(deep=False)
            var.dims = tuple(dims_dict.get(dim, dim) for dim in v.dims)
            name = name_dict.get(k, k)
            if name in variables:
                raise ValueError(f"the new name {name!r} conflicts")
            variables[name] = var
            if k in self._coord_names:
                coord_names.add(name)
        return variables, coord_names

    def _rename_dims(self, name_dict):
        return {name_dict.get(k, k): v for k, v in self.dims.items()}

    def _rename_indexes(self, name_dict, dims_set):
        # TODO: benbovy - flexible indexes: https://github.com/pydata/xarray/issues/5645
        if self._indexes is None:
            return None
        indexes = {}
        for k, v in self.indexes.items():
            new_name = name_dict.get(k, k)
            if new_name not in dims_set:
                continue
            if isinstance(v, pd.MultiIndex):
                new_names = [name_dict.get(k, k) for k in v.names]
                indexes[new_name] = PandasMultiIndex(
                    v.rename(names=new_names), new_name
                )
            else:
                indexes[new_name] = PandasIndex(v.rename(new_name), new_name)
        return indexes

    def _rename_all(self, name_dict, dims_dict):
        variables, coord_names = self._rename_vars(name_dict, dims_dict)
        dims = self._rename_dims(dims_dict)
        indexes = self._rename_indexes(name_dict, dims.keys())
        return variables, coord_names, dims, indexes

    def rename(
        self,
        name_dict: Mapping[Any, Hashable] = None,
        **names: Hashable,
    ) -> Dataset:
        """Returns a new object with renamed variables and dimensions.

        Parameters
        ----------
        name_dict : dict-like, optional
            Dictionary whose keys are current variable or dimension names and
            whose values are the desired names.
        **names : optional
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
        name_dict = either_dict_or_kwargs(name_dict, names, "rename")
        for k in name_dict.keys():
            if k not in self and k not in self.dims:
                raise ValueError(
                    f"cannot rename {k!r} because it is not a "
                    "variable or dimension in this dataset"
                )

        variables, coord_names, dims, indexes = self._rename_all(
            name_dict=name_dict, dims_dict=name_dict
        )
        assert_unique_multiindex_level_names(variables)
        return self._replace(variables, coord_names, dims=dims, indexes=indexes)

    def rename_dims(
        self, dims_dict: Mapping[Any, Hashable] = None, **dims: Hashable
    ) -> Dataset:
        """Returns a new object with renamed dimensions only.

        Parameters
        ----------
        dims_dict : dict-like, optional
            Dictionary whose keys are current dimension names and
            whose values are the desired names. The desired names must
            not be the name of an existing dimension or Variable in the Dataset.
        **dims : optional
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
        dims_dict = either_dict_or_kwargs(dims_dict, dims, "rename_dims")
        for k, v in dims_dict.items():
            if k not in self.dims:
                raise ValueError(
                    f"cannot rename {k!r} because it is not a "
                    "dimension in this dataset"
                )
            if v in self.dims or v in self:
                raise ValueError(
                    f"Cannot rename {k} to {v} because {v} already exists. "
                    "Try using swap_dims instead."
                )

        variables, coord_names, sizes, indexes = self._rename_all(
            name_dict={}, dims_dict=dims_dict
        )
        return self._replace(variables, coord_names, dims=sizes, indexes=indexes)

    def rename_vars(
        self, name_dict: Mapping[Any, Hashable] = None, **names: Hashable
    ) -> Dataset:
        """Returns a new object with renamed variables including coordinates

        Parameters
        ----------
        name_dict : dict-like, optional
            Dictionary whose keys are current variable or coordinate names and
            whose values are the desired names.
        **names : optional
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
        name_dict = either_dict_or_kwargs(name_dict, names, "rename_vars")
        for k in name_dict:
            if k not in self:
                raise ValueError(
                    f"cannot rename {k!r} because it is not a "
                    "variable or coordinate in this dataset"
                )
        variables, coord_names, dims, indexes = self._rename_all(
            name_dict=name_dict, dims_dict={}
        )
        return self._replace(variables, coord_names, dims=dims, indexes=indexes)

    def swap_dims(
        self, dims_dict: Mapping[Any, Hashable] = None, **dims_kwargs
    ) -> Dataset:
        """Returns a new object with swapped dimensions.

        Parameters
        ----------
        dims_dict : dict-like
            Dictionary whose keys are current dimension names and whose values
            are new names.
        **dims_kwargs : {existing_dim: new_dim, ...}, optional
            The keyword arguments form of ``dims_dict``.
            One of dims_dict or dims_kwargs must be provided.

        Returns
        -------
        swapped : Dataset
            Dataset with swapped dimensions.

        Examples
        --------
        >>> ds = xr.Dataset(
        ...     data_vars={"a": ("x", [5, 7]), "b": ("x", [0.1, 2.4])},
        ...     coords={"x": ["a", "b"], "y": ("x", [0, 1])},
        ... )
        >>> ds
        <xarray.Dataset>
        Dimensions:  (x: 2)
        Coordinates:
          * x        (x) <U1 'a' 'b'
            y        (x) int64 0 1
        Data variables:
            a        (x) int64 5 7
            b        (x) float64 0.1 2.4

        >>> ds.swap_dims({"x": "y"})
        <xarray.Dataset>
        Dimensions:  (y: 2)
        Coordinates:
            x        (y) <U1 'a' 'b'
          * y        (y) int64 0 1
        Data variables:
            a        (y) int64 5 7
            b        (y) float64 0.1 2.4

        >>> ds.swap_dims({"x": "z"})
        <xarray.Dataset>
        Dimensions:  (z: 2)
        Coordinates:
            x        (z) <U1 'a' 'b'
            y        (z) int64 0 1
        Dimensions without coordinates: z
        Data variables:
            a        (z) int64 5 7
            b        (z) float64 0.1 2.4

        See Also
        --------
        Dataset.rename
        DataArray.swap_dims
        """
        # TODO: deprecate this method in favor of a (less confusing)
        # rename_dims() method that only renames dimensions.

        dims_dict = either_dict_or_kwargs(dims_dict, dims_kwargs, "swap_dims")
        for k, v in dims_dict.items():
            if k not in self.dims:
                raise ValueError(
                    f"cannot swap from dimension {k!r} because it is "
                    "not an existing dimension"
                )
            if v in self.variables and self.variables[v].dims != (k,):
                raise ValueError(
                    f"replacement dimension {v!r} is not a 1D "
                    f"variable along the old dimension {k!r}"
                )

        result_dims = {dims_dict.get(dim, dim) for dim in self.dims}

        coord_names = self._coord_names.copy()
        coord_names.update({dim for dim in dims_dict.values() if dim in self.variables})

        variables: dict[Hashable, Variable] = {}
        indexes: dict[Hashable, Index] = {}
        for k, v in self.variables.items():
            dims = tuple(dims_dict.get(dim, dim) for dim in v.dims)
            if k in result_dims:
                var = v.to_index_variable()
                if k in self.xindexes:
                    indexes[k] = self.xindexes[k]
                else:
                    new_index = var.to_index()
                    if new_index.nlevels == 1:
                        # make sure index name matches dimension name
                        new_index = new_index.rename(k)
                    if isinstance(new_index, pd.MultiIndex):
                        indexes[k] = PandasMultiIndex(new_index, k)
                    else:
                        indexes[k] = PandasIndex(new_index, k)
            else:
                var = v.to_base_variable()
            var.dims = dims
            variables[k] = var

        return self._replace_with_new_dims(variables, coord_names, indexes=indexes)

    def expand_dims(
        self,
        dim: None | Hashable | Sequence[Hashable] | Mapping[Any, Any] = None,
        axis: None | int | Sequence[int] = None,
        **dim_kwargs: Any,
    ) -> Dataset:
        """Return a new object with an additional axis (or axes) inserted at
        the corresponding position in the array shape.  The new object is a
        view into the underlying array, not a copy.

        If dim is already a scalar coordinate, it will be promoted to a 1D
        coordinate consisting of a single value.

        Parameters
        ----------
        dim : hashable, sequence of hashable, mapping, or None
            Dimensions to include on the new variable. If provided as hashable
            or sequence of hashable, then dimensions are inserted with length
            1. If provided as a mapping, then the keys are the new dimensions
            and the values are either integers (giving the length of the new
            dimensions) or array-like (giving the coordinates of the new
            dimensions).
        axis : int, sequence of int, or None
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
        if dim is None:
            pass
        elif isinstance(dim, Mapping):
            # We're later going to modify dim in place; don't tamper with
            # the input
            dim = dict(dim)
        elif isinstance(dim, int):
            raise TypeError(
                "dim should be hashable or sequence of hashables or mapping"
            )
        elif isinstance(dim, str) or not isinstance(dim, Sequence):
            dim = {dim: 1}
        elif isinstance(dim, Sequence):
            if len(dim) != len(set(dim)):
                raise ValueError("dims should not contain duplicate values.")
            dim = {d: 1 for d in dim}

        dim = either_dict_or_kwargs(dim, dim_kwargs, "expand_dims")
        assert isinstance(dim, MutableMapping)

        if axis is None:
            axis = list(range(len(dim)))
        elif not isinstance(axis, Sequence):
            axis = [axis]

        if len(dim) != len(axis):
            raise ValueError("lengths of dim and axis should be identical.")
        for d in dim:
            if d in self.dims:
                raise ValueError(f"Dimension {d} already exists.")
            if d in self._variables and not utils.is_scalar(self._variables[d]):
                raise ValueError(
                    "{dim} already exists as coordinate or"
                    " variable name.".format(dim=d)
                )

        variables: dict[Hashable, Variable] = {}
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
                raise TypeError(
                    "The value of new dimension {k} must be "
                    "an iterable or an int".format(k=k)
                )

        for k, v in self._variables.items():
            if k not in dim:
                if k in coord_names:  # Do not change coordinates
                    variables[k] = v
                else:
                    result_ndim = len(v.dims) + len(axis)
                    for a in axis:
                        if a < -result_ndim or result_ndim - 1 < a:
                            raise IndexError(
                                f"Axis {a} of variable {k} is out of bounds of the "
                                f"expanded dimension size {result_ndim}"
                            )

                    axis_pos = [a if a >= 0 else result_ndim + a for a in axis]
                    if len(axis_pos) != len(set(axis_pos)):
                        raise ValueError("axis should not contain duplicate values")
                    # We need to sort them to make sure `axis` equals to the
                    # axis positions of the result array.
                    zip_axis_dim = sorted(zip(axis_pos, dim.items()))

                    all_dims = list(zip(v.dims, v.shape))
                    for d, c in zip_axis_dim:
                        all_dims.insert(d, c)
                    variables[k] = v.set_dims(dict(all_dims))
            else:
                # If dims includes a label of a non-dimension coordinate,
                # it will be promoted to a 1D coordinate with a single value.
                variables[k] = v.set_dims(k).to_index_variable()

        new_dims = self._dims.copy()
        new_dims.update(dim)

        return self._replace_vars_and_dims(
            variables, dims=new_dims, coord_names=coord_names
        )

    def set_index(
        self,
        indexes: Mapping[Any, Hashable | Sequence[Hashable]] = None,
        append: bool = False,
        **indexes_kwargs: Hashable | Sequence[Hashable],
    ) -> Dataset:
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
        **indexes_kwargs : optional
            The keyword arguments form of ``indexes``.
            One of indexes or indexes_kwargs must be provided.

        Returns
        -------
        obj : Dataset
            Another dataset, with this dataset's data but replaced coordinates.

        Examples
        --------
        >>> arr = xr.DataArray(
        ...     data=np.ones((2, 3)),
        ...     dims=["x", "y"],
        ...     coords={"x": range(2), "y": range(3), "a": ("x", [3, 4])},
        ... )
        >>> ds = xr.Dataset({"v": arr})
        >>> ds
        <xarray.Dataset>
        Dimensions:  (x: 2, y: 3)
        Coordinates:
          * x        (x) int64 0 1
          * y        (y) int64 0 1 2
            a        (x) int64 3 4
        Data variables:
            v        (x, y) float64 1.0 1.0 1.0 1.0 1.0 1.0
        >>> ds.set_index(x="a")
        <xarray.Dataset>
        Dimensions:  (x: 2, y: 3)
        Coordinates:
          * x        (x) int64 3 4
          * y        (y) int64 0 1 2
        Data variables:
            v        (x, y) float64 1.0 1.0 1.0 1.0 1.0 1.0

        See Also
        --------
        Dataset.reset_index
        Dataset.swap_dims
        """
        indexes = either_dict_or_kwargs(indexes, indexes_kwargs, "set_index")
        variables, coord_names = merge_indexes(
            indexes, self._variables, self._coord_names, append=append
        )
        return self._replace_vars_and_dims(variables, coord_names=coord_names)

    def reset_index(
        self,
        dims_or_levels: Hashable | Sequence[Hashable],
        drop: bool = False,
    ) -> Dataset:
        """Reset the specified index(es) or multi-index level(s).

        Parameters
        ----------
        dims_or_levels : str or list
            Name(s) of the dimension(s) and/or multi-index level(s) that will
            be reset.
        drop : bool, optional
            If True, remove the specified indexes and/or multi-index levels
            instead of extracting them as new coordinates (default: False).

        Returns
        -------
        obj : Dataset
            Another dataset, with this dataset's data but replaced coordinates.

        See Also
        --------
        Dataset.set_index
        """
        variables, coord_names = split_indexes(
            dims_or_levels,
            self._variables,
            self._coord_names,
            cast(Mapping[Hashable, Hashable], self._level_coords),
            drop=drop,
        )
        return self._replace_vars_and_dims(variables, coord_names=coord_names)

    def reorder_levels(
        self,
        dim_order: Mapping[Any, Sequence[int]] = None,
        **dim_order_kwargs: Sequence[int],
    ) -> Dataset:
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
        obj : Dataset
            Another dataset, with this dataset's data but replaced
            coordinates.
        """
        dim_order = either_dict_or_kwargs(dim_order, dim_order_kwargs, "reorder_levels")
        variables = self._variables.copy()
        indexes = dict(self.xindexes)
        for dim, order in dim_order.items():
            coord = self._variables[dim]
            # TODO: benbovy - flexible indexes: update when MultiIndex
            # has its own class inherited from xarray.Index
            index = self.xindexes[dim].to_pandas_index()
            if not isinstance(index, pd.MultiIndex):
                raise ValueError(f"coordinate {dim} has no MultiIndex")
            new_index = index.reorder_levels(order)
            variables[dim] = IndexVariable(coord.dims, new_index)
            indexes[dim] = PandasMultiIndex(new_index, dim)

        return self._replace(variables, indexes=indexes)

    def _stack_once(self, dims, new_dim):
        if dims == ...:
            raise ValueError("Please use [...] for dims, rather than just ...")
        if ... in dims:
            dims = list(infix_dims(dims, self.dims))
        variables = {}
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
        idx = utils.multiindex_from_product_levels(levels, names=dims)
        variables[new_dim] = IndexVariable(new_dim, idx)

        coord_names = set(self._coord_names) - set(dims) | {new_dim}

        indexes = {k: v for k, v in self.xindexes.items() if k not in dims}
        indexes[new_dim] = PandasMultiIndex(idx, new_dim)

        return self._replace_with_new_dims(
            variables, coord_names=coord_names, indexes=indexes
        )

    def stack(
        self,
        dimensions: Mapping[Any, Sequence[Hashable]] = None,
        **dimensions_kwargs: Sequence[Hashable],
    ) -> Dataset:
        """
        Stack any number of existing dimensions into a single new dimension.

        New dimensions will be added at the end, and the corresponding
        coordinate variables will be combined into a MultiIndex.

        Parameters
        ----------
        dimensions : mapping of hashable to sequence of hashable
            Mapping of the form `new_name=(dim1, dim2, ...)`. Names of new
            dimensions, and the existing dimensions that they replace. An
            ellipsis (`...`) will be replaced by all unlisted dimensions.
            Passing a list containing an ellipsis (`stacked_dim=[...]`) will stack over
            all dimensions.
        **dimensions_kwargs
            The keyword arguments form of ``dimensions``.
            One of dimensions or dimensions_kwargs must be provided.

        Returns
        -------
        stacked : Dataset
            Dataset with stacked data.

        See Also
        --------
        Dataset.unstack
        """
        dimensions = either_dict_or_kwargs(dimensions, dimensions_kwargs, "stack")
        result = self
        for new_dim, dims in dimensions.items():
            result = result._stack_once(dims, new_dim)
        return result

    def to_stacked_array(
        self,
        new_dim: Hashable,
        sample_dims: Collection,
        variable_dim: Hashable = "variable",
        name: Hashable = None,
    ) -> DataArray:
        """Combine variables of differing dimensionality into a DataArray
        without broadcasting.

        This method is similar to Dataset.to_array but does not broadcast the
        variables.

        Parameters
        ----------
        new_dim : hashable
            Name of the new stacked coordinate
        sample_dims : Collection of hashables
            List of dimensions that **will not** be stacked. Each array in the
            dataset must share these dimensions. For machine learning
            applications, these define the dimensions over which samples are
            drawn.
        variable_dim : hashable, optional
            Name of the level in the stacked coordinate which corresponds to
            the variables.
        name : hashable, optional
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
        >>> data = xr.Dataset(
        ...     data_vars={
        ...         "a": (("x", "y"), [[0, 1, 2], [3, 4, 5]]),
        ...         "b": ("x", [6, 7]),
        ...     },
        ...     coords={"y": ["u", "v", "w"]},
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

        >>> data.to_stacked_array("z", sample_dims=["x"])
        <xarray.DataArray 'a' (x: 2, z: 4)>
        array([[0, 1, 2, 6],
               [3, 4, 5, 7]])
        Coordinates:
          * z         (z) MultiIndex
          - variable  (z) object 'a' 'a' 'a' 'b'
          - y         (z) object 'u' 'v' 'w' nan
        Dimensions without coordinates: x

        """
        stacking_dims = tuple(dim for dim in self.dims if dim not in sample_dims)

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

            return (
                val.assign_coords(**assign_coords)
                .expand_dims(expand_dims)
                .stack({new_dim: (variable_dim,) + stacking_dims})
            )

        # concatenate the arrays
        stackable_vars = [ensure_stackable(self[key]) for key in self.data_vars]
        data_array = xr.concat(stackable_vars, dim=new_dim)

        # coerce the levels of the MultiIndex to have the same type as the
        # input dimensions. This code is messy, so it might be better to just
        # input a dummy value for the singleton dimension.
        # TODO: benbovy - flexible indexes: update when MultIndex has its own
        # class inheriting from xarray.Index
        idx = data_array.xindexes[new_dim].to_pandas_index()
        levels = [idx.levels[0]] + [
            level.astype(self[level.name].dtype) for level in idx.levels[1:]
        ]
        new_idx = idx.set_levels(levels)
        data_array[new_dim] = IndexVariable(new_dim, new_idx)

        if name is not None:
            data_array.name = name

        return data_array

    def _unstack_once(self, dim: Hashable, fill_value, sparse: bool = False) -> Dataset:
        index = self.get_index(dim)
        index = remove_unused_levels_categories(index)

        variables: dict[Hashable, Variable] = {}
        indexes = {k: v for k, v in self.xindexes.items() if k != dim}

        for name, var in self.variables.items():
            if name != dim:
                if dim in var.dims:
                    if isinstance(fill_value, Mapping):
                        fill_value_ = fill_value[name]
                    else:
                        fill_value_ = fill_value

                    variables[name] = var._unstack_once(
                        index=index, dim=dim, fill_value=fill_value_, sparse=sparse
                    )
                else:
                    variables[name] = var

        for name, lev in zip(index.names, index.levels):
            idx, idx_vars = PandasIndex.from_pandas_index(lev, name)
            variables[name] = idx_vars[name]
            indexes[name] = idx

        coord_names = set(self._coord_names) - {dim} | set(index.names)

        return self._replace_with_new_dims(
            variables, coord_names=coord_names, indexes=indexes
        )

    def _unstack_full_reindex(self, dim: Hashable, fill_value, sparse: bool) -> Dataset:
        index = self.get_index(dim)
        index = remove_unused_levels_categories(index)
        full_idx = pd.MultiIndex.from_product(index.levels, names=index.names)

        # take a shortcut in case the MultiIndex was not modified.
        if index.equals(full_idx):
            obj = self
        else:
            obj = self._reindex(
                {dim: full_idx}, copy=False, fill_value=fill_value, sparse=sparse
            )

        new_dim_names = index.names
        new_dim_sizes = [lev.size for lev in index.levels]

        variables: dict[Hashable, Variable] = {}
        indexes = {k: v for k, v in self.xindexes.items() if k != dim}

        for name, var in obj.variables.items():
            if name != dim:
                if dim in var.dims:
                    new_dims = dict(zip(new_dim_names, new_dim_sizes))
                    variables[name] = var.unstack({dim: new_dims})
                else:
                    variables[name] = var

        for name, lev in zip(new_dim_names, index.levels):
            idx, idx_vars = PandasIndex.from_pandas_index(lev, name)
            variables[name] = idx_vars[name]
            indexes[name] = idx

        coord_names = set(self._coord_names) - {dim} | set(new_dim_names)

        return self._replace_with_new_dims(
            variables, coord_names=coord_names, indexes=indexes
        )

    def unstack(
        self,
        dim: Hashable | Iterable[Hashable] = None,
        fill_value: Any = dtypes.NA,
        sparse: bool = False,
    ) -> Dataset:
        """
        Unstack existing dimensions corresponding to MultiIndexes into
        multiple new dimensions.

        New dimensions will be added at the end.

        Parameters
        ----------
        dim : hashable or iterable of hashable, optional
            Dimension(s) over which to unstack. By default unstacks all
            MultiIndexes.
        fill_value : scalar or dict-like, default: nan
            value to be filled. If a dict-like, maps variable names to
            fill values. If not provided or if the dict-like does not
            contain all variables, the dtype's NA value will be used.
        sparse : bool, default: False
            use sparse-array if True

        Returns
        -------
        unstacked : Dataset
            Dataset with unstacked data.

        See Also
        --------
        Dataset.stack
        """
        if dim is None:
            dims = [
                d for d in self.dims if isinstance(self.get_index(d), pd.MultiIndex)
            ]
        else:
            if isinstance(dim, str) or not isinstance(dim, Iterable):
                dims = [dim]
            else:
                dims = list(dim)

            missing_dims = [d for d in dims if d not in self.dims]
            if missing_dims:
                raise ValueError(
                    f"Dataset does not contain the dimensions: {missing_dims}"
                )

            non_multi_dims = [
                d for d in dims if not isinstance(self.get_index(d), pd.MultiIndex)
            ]
            if non_multi_dims:
                raise ValueError(
                    "cannot unstack dimensions that do not "
                    f"have a MultiIndex: {non_multi_dims}"
                )

        result = self.copy(deep=False)

        # we want to avoid allocating an object-dtype ndarray for a MultiIndex,
        # so we can't just access self.variables[v].data for every variable.
        # We only check the non-index variables.
        # https://github.com/pydata/xarray/issues/5902
        nonindexes = [
            self.variables[k] for k in set(self.variables) - set(self.xindexes)
        ]
        # Notes for each of these cases:
        # 1. Dask arrays don't support assignment by index, which the fast unstack
        #    function requires.
        #    https://github.com/pydata/xarray/pull/4746#issuecomment-753282125
        # 2. Sparse doesn't currently support (though we could special-case it)
        #    https://github.com/pydata/sparse/issues/422
        # 3. pint requires checking if it's a NumPy array until
        #    https://github.com/pydata/xarray/pull/4751 is resolved,
        #    Once that is resolved, explicitly exclude pint arrays.
        #    pint doesn't implement `np.full_like` in a way that's
        #    currently compatible.
        needs_full_reindex = any(
            is_duck_dask_array(v.data)
            or isinstance(v.data, sparse_array_type)
            or not isinstance(v.data, np.ndarray)
            for v in nonindexes
        )

        for dim in dims:
            if needs_full_reindex:
                result = result._unstack_full_reindex(dim, fill_value, sparse)
            else:
                result = result._unstack_once(dim, fill_value, sparse)
        return result

    def update(self, other: CoercibleMapping) -> Dataset:
        """Update this dataset's variables with those from another dataset.

        Just like :py:meth:`dict.update` this is a in-place operation.
        For a non-inplace version, see :py:meth:`Dataset.merge`.

        Parameters
        ----------
        other : Dataset or mapping
            Variables with which to update this dataset. One of:

            - Dataset
            - mapping {var name: DataArray}
            - mapping {var name: Variable}
            - mapping {var name: (dimension name, array-like)}
            - mapping {var name: (tuple of dimension names, array-like)}

        Returns
        -------
        updated : Dataset
            Updated dataset. Note that since the update is in-place this is the input
            dataset.

            It is deprecated since version 0.17 and scheduled to be removed in 0.21.

        Raises
        ------
        ValueError
            If any dimensions would have inconsistent sizes in the updated
            dataset.

        See Also
        --------
        Dataset.assign
        Dataset.merge
        """
        merge_result = dataset_update_method(self, other)
        return self._replace(inplace=True, **merge_result._asdict())

    def merge(
        self,
        other: CoercibleMapping | DataArray,
        overwrite_vars: Hashable | Iterable[Hashable] = frozenset(),
        compat: str = "no_conflicts",
        join: str = "outer",
        fill_value: Any = dtypes.NA,
        combine_attrs: str = "override",
    ) -> Dataset:
        """Merge the arrays of two datasets into a single dataset.

        This method generally does not allow for overriding data, with the
        exception of attributes, which are ignored on the second dataset.
        Variables with the same name are checked for conflicts via the equals
        or identical methods.

        Parameters
        ----------
        other : Dataset or mapping
            Dataset or variables to merge with this dataset.
        overwrite_vars : hashable or iterable of hashable, optional
            If provided, update variables of these name(s) without checking for
            conflicts in this dataset.
        compat : {"broadcast_equals", "equals", "identical", \
                  "no_conflicts"}, optional
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
        join : {"outer", "inner", "left", "right", "exact"}, optional
            Method for joining ``self`` and ``other`` along shared dimensions:

            - 'outer': use the union of the indexes
            - 'inner': use the intersection of the indexes
            - 'left': use indexes from ``self``
            - 'right': use indexes from ``other``
            - 'exact': error instead of aligning non-equal indexes
        fill_value : scalar or dict-like, optional
            Value to use for newly missing values. If a dict-like, maps
            variable names (including coordinates) to fill values.
        combine_attrs : {"drop", "identical", "no_conflicts", "drop_conflicts", \
                        "override"}, default: "override"
            String indicating how to combine attrs of the objects being merged:

            - "drop": empty attrs on returned Dataset.
            - "identical": all attrs must be the same on every object.
            - "no_conflicts": attrs from all objects are combined, any that have
              the same name must also have the same value.
            - "drop_conflicts": attrs from all objects are combined, any that have
              the same name but different values are dropped.
            - "override": skip comparing and copy attrs from the first dataset to
              the result.

        Returns
        -------
        merged : Dataset
            Merged dataset.

        Raises
        ------
        MergeError
            If any variables conflict (see ``compat``).

        See Also
        --------
        Dataset.update
        """
        other = other.to_dataset() if isinstance(other, xr.DataArray) else other
        merge_result = dataset_merge_method(
            self,
            other,
            overwrite_vars=overwrite_vars,
            compat=compat,
            join=join,
            fill_value=fill_value,
            combine_attrs=combine_attrs,
        )
        return self._replace(**merge_result._asdict())

    def _assert_all_in_dataset(
        self, names: Iterable[Hashable], virtual_okay: bool = False
    ) -> None:
        bad_names = set(names) - set(self._variables)
        if virtual_okay:
            bad_names -= self.virtual_variables
        if bad_names:
            raise ValueError(
                "One or more of the specified variables "
                "cannot be found in this dataset"
            )

    def drop_vars(
        self, names: Hashable | Iterable[Hashable], *, errors: str = "raise"
    ) -> Dataset:
        """Drop variables from this dataset.

        Parameters
        ----------
        names : hashable or iterable of hashable
            Name(s) of variables to drop.
        errors : {"raise", "ignore"}, optional
            If 'raise' (default), raises a ValueError error if any of the variable
            passed are not in the dataset. If 'ignore', any given names that are in the
            dataset are dropped and no error is raised.

        Returns
        -------
        dropped : Dataset

        """
        # the Iterable check is required for mypy
        if is_scalar(names) or not isinstance(names, Iterable):
            names = {names}
        else:
            names = set(names)
        if errors == "raise":
            self._assert_all_in_dataset(names)

        variables = {k: v for k, v in self._variables.items() if k not in names}
        coord_names = {k for k in self._coord_names if k in variables}
        indexes = {k: v for k, v in self.xindexes.items() if k not in names}
        return self._replace_with_new_dims(
            variables, coord_names=coord_names, indexes=indexes
        )

    def drop(self, labels=None, dim=None, *, errors="raise", **labels_kwargs):
        """Backward compatible method based on `drop_vars` and `drop_sel`

        Using either `drop_vars` or `drop_sel` is encouraged

        See Also
        --------
        Dataset.drop_vars
        Dataset.drop_sel
        """
        if errors not in ["raise", "ignore"]:
            raise ValueError('errors must be either "raise" or "ignore"')

        if is_dict_like(labels) and not isinstance(labels, dict):
            warnings.warn(
                "dropping coordinates using `drop` is be deprecated; use drop_vars.",
                FutureWarning,
                stacklevel=2,
            )
            return self.drop_vars(labels, errors=errors)

        if labels_kwargs or isinstance(labels, dict):
            if dim is not None:
                raise ValueError("cannot specify dim and dict-like arguments.")
            labels = either_dict_or_kwargs(labels, labels_kwargs, "drop")

        if dim is None and (is_scalar(labels) or isinstance(labels, Iterable)):
            warnings.warn(
                "dropping variables using `drop` will be deprecated; using drop_vars is encouraged.",
                PendingDeprecationWarning,
                stacklevel=2,
            )
            return self.drop_vars(labels, errors=errors)
        if dim is not None:
            warnings.warn(
                "dropping labels using list-like labels is deprecated; using "
                "dict-like arguments with `drop_sel`, e.g. `ds.drop_sel(dim=[labels]).",
                DeprecationWarning,
                stacklevel=2,
            )
            return self.drop_sel({dim: labels}, errors=errors, **labels_kwargs)

        warnings.warn(
            "dropping labels using `drop` will be deprecated; using drop_sel is encouraged.",
            PendingDeprecationWarning,
            stacklevel=2,
        )
        return self.drop_sel(labels, errors=errors)

    def drop_sel(self, labels=None, *, errors="raise", **labels_kwargs):
        """Drop index labels from this dataset.

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
        dropped : Dataset

        Examples
        --------
        >>> data = np.arange(6).reshape(2, 3)
        >>> labels = ["a", "b", "c"]
        >>> ds = xr.Dataset({"A": (["x", "y"], data), "y": labels})
        >>> ds
        <xarray.Dataset>
        Dimensions:  (x: 2, y: 3)
        Coordinates:
          * y        (y) <U1 'a' 'b' 'c'
        Dimensions without coordinates: x
        Data variables:
            A        (x, y) int64 0 1 2 3 4 5
        >>> ds.drop_sel(y=["a", "c"])
        <xarray.Dataset>
        Dimensions:  (x: 2, y: 1)
        Coordinates:
          * y        (y) <U1 'b'
        Dimensions without coordinates: x
        Data variables:
            A        (x, y) int64 1 4
        >>> ds.drop_sel(y="b")
        <xarray.Dataset>
        Dimensions:  (x: 2, y: 2)
        Coordinates:
          * y        (y) <U1 'a' 'c'
        Dimensions without coordinates: x
        Data variables:
            A        (x, y) int64 0 2 3 5
        """
        if errors not in ["raise", "ignore"]:
            raise ValueError('errors must be either "raise" or "ignore"')

        labels = either_dict_or_kwargs(labels, labels_kwargs, "drop_sel")

        ds = self
        for dim, labels_for_dim in labels.items():
            # Don't cast to set, as it would harm performance when labels
            # is a large numpy array
            if utils.is_scalar(labels_for_dim):
                labels_for_dim = [labels_for_dim]
            labels_for_dim = np.asarray(labels_for_dim)
            try:
                index = self.get_index(dim)
            except KeyError:
                raise ValueError(f"dimension {dim!r} does not have coordinate labels")
            new_index = index.drop(labels_for_dim, errors=errors)
            ds = ds.loc[{dim: new_index}]
        return ds

    def drop_isel(self, indexers=None, **indexers_kwargs):
        """Drop index positions from this Dataset.

        Parameters
        ----------
        indexers : mapping of hashable to Any
            Index locations to drop
        **indexers_kwargs : {dim: position, ...}, optional
            The keyword arguments form of ``dim`` and ``positions``

        Returns
        -------
        dropped : Dataset

        Raises
        ------
        IndexError

        Examples
        --------
        >>> data = np.arange(6).reshape(2, 3)
        >>> labels = ["a", "b", "c"]
        >>> ds = xr.Dataset({"A": (["x", "y"], data), "y": labels})
        >>> ds
        <xarray.Dataset>
        Dimensions:  (x: 2, y: 3)
        Coordinates:
          * y        (y) <U1 'a' 'b' 'c'
        Dimensions without coordinates: x
        Data variables:
            A        (x, y) int64 0 1 2 3 4 5
        >>> ds.drop_isel(y=[0, 2])
        <xarray.Dataset>
        Dimensions:  (x: 2, y: 1)
        Coordinates:
          * y        (y) <U1 'b'
        Dimensions without coordinates: x
        Data variables:
            A        (x, y) int64 1 4
        >>> ds.drop_isel(y=1)
        <xarray.Dataset>
        Dimensions:  (x: 2, y: 2)
        Coordinates:
          * y        (y) <U1 'a' 'c'
        Dimensions without coordinates: x
        Data variables:
            A        (x, y) int64 0 2 3 5
        """

        indexers = either_dict_or_kwargs(indexers, indexers_kwargs, "drop_isel")

        ds = self
        dimension_index = {}
        for dim, pos_for_dim in indexers.items():
            # Don't cast to set, as it would harm performance when labels
            # is a large numpy array
            if utils.is_scalar(pos_for_dim):
                pos_for_dim = [pos_for_dim]
            pos_for_dim = np.asarray(pos_for_dim)
            index = self.get_index(dim)
            new_index = index.delete(pos_for_dim)
            dimension_index[dim] = new_index
        ds = ds.loc[dimension_index]
        return ds

    def drop_dims(
        self, drop_dims: Hashable | Iterable[Hashable], *, errors: str = "raise"
    ) -> Dataset:
        """Drop dimensions and associated variables from this dataset.

        Parameters
        ----------
        drop_dims : hashable or iterable of hashable
            Dimension or dimensions to drop.
        errors : {"raise", "ignore"}, default: "raise"
            If 'raise', raises a ValueError error if any of the
            dimensions passed are not in the dataset. If 'ignore', any given
            dimensions that are in the dataset are dropped and no error is raised.

        Returns
        -------
        obj : Dataset
            The dataset without the given dimensions (or any variables
            containing those dimensions).
        """
        if errors not in ["raise", "ignore"]:
            raise ValueError('errors must be either "raise" or "ignore"')

        if isinstance(drop_dims, str) or not isinstance(drop_dims, Iterable):
            drop_dims = {drop_dims}
        else:
            drop_dims = set(drop_dims)

        if errors == "raise":
            missing_dims = drop_dims - set(self.dims)
            if missing_dims:
                raise ValueError(
                    f"Dataset does not contain the dimensions: {missing_dims}"
                )

        drop_vars = {k for k, v in self._variables.items() if set(v.dims) & drop_dims}
        return self.drop_vars(drop_vars)

    def transpose(
        self,
        *dims: Hashable,
        missing_dims: str = "raise",
    ) -> Dataset:
        """Return a new Dataset object with all array dimensions transposed.

        Although the order of dimensions on each array will change, the dataset
        dimensions themselves will remain in fixed (sorted) order.

        Parameters
        ----------
        *dims : hashable, optional
            By default, reverse the dimensions on each array. Otherwise,
            reorder the dimensions to this order.
        missing_dims : {"raise", "warn", "ignore"}, default: "raise"
            What to do if dimensions that should be selected from are not present in the
            Dataset:
            - "raise": raise an exception
            - "warn": raise a warning, and ignore the missing dimensions
            - "ignore": ignore the missing dimensions

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
        # Use infix_dims to check once for missing dimensions
        if len(dims) != 0:
            _ = list(infix_dims(dims, self.dims, missing_dims))

        ds = self.copy()
        for name, var in self._variables.items():
            var_dims = tuple(dim for dim in dims if dim in (var.dims + (...,)))
            ds._variables[name] = var.transpose(*var_dims)
        return ds

    def dropna(
        self,
        dim: Hashable,
        how: str = "any",
        thresh: int = None,
        subset: Iterable[Hashable] = None,
    ):
        """Returns a new dataset with dropped labels for missing values along
        the provided dimension.

        Parameters
        ----------
        dim : hashable
            Dimension along which to drop missing values. Dropping along
            multiple dimensions simultaneously is not yet supported.
        how : {"any", "all"}, default: "any"
            * any : if any NA values are present, drop that label
            * all : if all values are NA, drop that label
        thresh : int, default: None
            If supplied, require this many non-NA values.
        subset : iterable of hashable, optional
            Which variables to check for missing values. By default, all
            variables in the dataset are checked.

        Returns
        -------
        Dataset
        """
        # TODO: consider supporting multiple dimensions? Or not, given that
        # there are some ugly edge cases, e.g., pandas's dropna differs
        # depending on the order of the supplied axes.

        if dim not in self.dims:
            raise ValueError(f"{dim} must be a single dataset dimension")

        if subset is None:
            subset = iter(self.data_vars)

        count = np.zeros(self.dims[dim], dtype=np.int64)
        size = np.int_(0)  # for type checking

        for k in subset:
            array = self._variables[k]
            if dim in array.dims:
                dims = [d for d in array.dims if d != dim]
                count += np.asarray(array.count(dims))  # type: ignore[attr-defined]
                size += np.prod([self.dims[d] for d in dims])

        if thresh is not None:
            mask = count >= thresh
        elif how == "any":
            mask = count == size
        elif how == "all":
            mask = count > 0
        elif how is not None:
            raise ValueError(f"invalid how option: {how}")
        else:
            raise TypeError("must specify how or thresh")

        return self.isel({dim: mask})

    def fillna(self, value: Any) -> Dataset:
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

        Examples
        --------
        >>> ds = xr.Dataset(
        ...     {
        ...         "A": ("x", [np.nan, 2, np.nan, 0]),
        ...         "B": ("x", [3, 4, np.nan, 1]),
        ...         "C": ("x", [np.nan, np.nan, np.nan, 5]),
        ...         "D": ("x", [np.nan, 3, np.nan, 4]),
        ...     },
        ...     coords={"x": [0, 1, 2, 3]},
        ... )
        >>> ds
        <xarray.Dataset>
        Dimensions:  (x: 4)
        Coordinates:
          * x        (x) int64 0 1 2 3
        Data variables:
            A        (x) float64 nan 2.0 nan 0.0
            B        (x) float64 3.0 4.0 nan 1.0
            C        (x) float64 nan nan nan 5.0
            D        (x) float64 nan 3.0 nan 4.0

        Replace all `NaN` values with 0s.

        >>> ds.fillna(0)
        <xarray.Dataset>
        Dimensions:  (x: 4)
        Coordinates:
          * x        (x) int64 0 1 2 3
        Data variables:
            A        (x) float64 0.0 2.0 0.0 0.0
            B        (x) float64 3.0 4.0 0.0 1.0
            C        (x) float64 0.0 0.0 0.0 5.0
            D        (x) float64 0.0 3.0 0.0 4.0

        Replace all `NaN` elements in column ‘A’, ‘B’, ‘C’, and ‘D’, with 0, 1, 2, and 3 respectively.

        >>> values = {"A": 0, "B": 1, "C": 2, "D": 3}
        >>> ds.fillna(value=values)
        <xarray.Dataset>
        Dimensions:  (x: 4)
        Coordinates:
          * x        (x) int64 0 1 2 3
        Data variables:
            A        (x) float64 0.0 2.0 0.0 0.0
            B        (x) float64 3.0 4.0 1.0 1.0
            C        (x) float64 2.0 2.0 2.0 5.0
            D        (x) float64 3.0 3.0 3.0 4.0
        """
        if utils.is_dict_like(value):
            value_keys = getattr(value, "data_vars", value).keys()
            if not set(value_keys) <= set(self.data_vars.keys()):
                raise ValueError(
                    "all variables in the argument to `fillna` "
                    "must be contained in the original dataset"
                )
        out = ops.fillna(self, value)
        return out

    def interpolate_na(
        self,
        dim: Hashable = None,
        method: str = "linear",
        limit: int = None,
        use_coordinate: bool | Hashable = True,
        max_gap: (
            int | float | str | pd.Timedelta | np.timedelta64 | datetime.timedelta
        ) = None,
        **kwargs: Any,
    ) -> Dataset:
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
        use_coordinate : bool, str, default: True
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
        max_gap : int, float, str, pandas.Timedelta, numpy.timedelta64, datetime.timedelta, default: None
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
        **kwargs : dict, optional
            parameters passed verbatim to the underlying interpolation function

        Returns
        -------
        interpolated: Dataset
            Filled in Dataset.

        See Also
        --------
        numpy.interp
        scipy.interpolate

        Examples
        --------
        >>> ds = xr.Dataset(
        ...     {
        ...         "A": ("x", [np.nan, 2, 3, np.nan, 0]),
        ...         "B": ("x", [3, 4, np.nan, 1, 7]),
        ...         "C": ("x", [np.nan, np.nan, np.nan, 5, 0]),
        ...         "D": ("x", [np.nan, 3, np.nan, -1, 4]),
        ...     },
        ...     coords={"x": [0, 1, 2, 3, 4]},
        ... )
        >>> ds
        <xarray.Dataset>
        Dimensions:  (x: 5)
        Coordinates:
          * x        (x) int64 0 1 2 3 4
        Data variables:
            A        (x) float64 nan 2.0 3.0 nan 0.0
            B        (x) float64 3.0 4.0 nan 1.0 7.0
            C        (x) float64 nan nan nan 5.0 0.0
            D        (x) float64 nan 3.0 nan -1.0 4.0

        >>> ds.interpolate_na(dim="x", method="linear")
        <xarray.Dataset>
        Dimensions:  (x: 5)
        Coordinates:
          * x        (x) int64 0 1 2 3 4
        Data variables:
            A        (x) float64 nan 2.0 3.0 1.5 0.0
            B        (x) float64 3.0 4.0 2.5 1.0 7.0
            C        (x) float64 nan nan nan 5.0 0.0
            D        (x) float64 nan 3.0 1.0 -1.0 4.0

        >>> ds.interpolate_na(dim="x", method="linear", fill_value="extrapolate")
        <xarray.Dataset>
        Dimensions:  (x: 5)
        Coordinates:
          * x        (x) int64 0 1 2 3 4
        Data variables:
            A        (x) float64 1.0 2.0 3.0 1.5 0.0
            B        (x) float64 3.0 4.0 2.5 1.0 7.0
            C        (x) float64 20.0 15.0 10.0 5.0 0.0
            D        (x) float64 5.0 3.0 1.0 -1.0 4.0
        """
        from .missing import _apply_over_vars_with_dim, interp_na

        new = _apply_over_vars_with_dim(
            interp_na,
            self,
            dim=dim,
            method=method,
            limit=limit,
            use_coordinate=use_coordinate,
            max_gap=max_gap,
            **kwargs,
        )
        return new

    def ffill(self, dim: Hashable, limit: int = None) -> Dataset:
        """Fill NaN values by propogating values forward

        *Requires bottleneck.*

        Parameters
        ----------
        dim : Hashable
            Specifies the dimension along which to propagate values when
            filling.
        limit : int, default: None
            The maximum number of consecutive NaN values to forward fill. In
            other words, if there is a gap with more than this number of
            consecutive NaNs, it will only be partially filled. Must be greater
            than 0 or None for no limit. Must be None or greater than or equal
            to axis length if filling along chunked axes (dimensions).

        Returns
        -------
        Dataset
        """
        from .missing import _apply_over_vars_with_dim, ffill

        new = _apply_over_vars_with_dim(ffill, self, dim=dim, limit=limit)
        return new

    def bfill(self, dim: Hashable, limit: int = None) -> Dataset:
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
            than 0 or None for no limit. Must be None or greater than or equal
            to axis length if filling along chunked axes (dimensions).

        Returns
        -------
        Dataset
        """
        from .missing import _apply_over_vars_with_dim, bfill

        new = _apply_over_vars_with_dim(bfill, self, dim=dim, limit=limit)
        return new

    def combine_first(self, other: Dataset) -> Dataset:
        """Combine two Datasets, default to data_vars of self.

        The new coordinates follow the normal broadcasting and alignment rules
        of ``join='outer'``.  Vacant cells in the expanded coordinates are
        filled with np.nan.

        Parameters
        ----------
        other : Dataset
            Used to fill all matching missing values in this array.

        Returns
        -------
        Dataset
        """
        out = ops.fillna(self, other, join="outer", dataset_join="outer")
        return out

    def reduce(
        self,
        func: Callable,
        dim: Hashable | Iterable[Hashable] = None,
        keep_attrs: bool = None,
        keepdims: bool = False,
        numeric_only: bool = False,
        **kwargs: Any,
    ) -> Dataset:
        """Reduce this dataset by applying `func` along some dimension(s).

        Parameters
        ----------
        func : callable
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
        keepdims : bool, default: False
            If True, the dimensions which are reduced are left in the result
            as dimensions of size one. Coordinates that use these dimensions
            are removed.
        numeric_only : bool, optional
            If True, only apply ``func`` to variables with a numeric dtype.
        **kwargs : Any
            Additional keyword arguments passed on to ``func``.

        Returns
        -------
        reduced : Dataset
            Dataset with this object's DataArrays replaced with new DataArrays
            of summarized data and the indicated dimension(s) removed.
        """
        if "axis" in kwargs:
            raise ValueError(
                "passing 'axis' to Dataset reduce methods is ambiguous."
                " Please use 'dim' instead."
            )

        if dim is None or dim is ...:
            dims = set(self.dims)
        elif isinstance(dim, str) or not isinstance(dim, Iterable):
            dims = {dim}
        else:
            dims = set(dim)

        missing_dimensions = [d for d in dims if d not in self.dims]
        if missing_dimensions:
            raise ValueError(
                f"Dataset does not contain the dimensions: {missing_dimensions}"
            )

        if keep_attrs is None:
            keep_attrs = _get_keep_attrs(default=False)

        variables: dict[Hashable, Variable] = {}
        for name, var in self._variables.items():
            reduce_dims = [d for d in var.dims if d in dims]
            if name in self.coords:
                if not reduce_dims:
                    variables[name] = var
            else:
                if (
                    # Some reduction functions (e.g. std, var) need to run on variables
                    # that don't have the reduce dims: PR5393
                    not reduce_dims
                    or not numeric_only
                    or np.issubdtype(var.dtype, np.number)
                    or (var.dtype == np.bool_)
                ):
                    if len(reduce_dims) == 1:
                        # unpack dimensions for the benefit of functions
                        # like np.argmin which can't handle tuple arguments
                        (reduce_dims,) = reduce_dims
                    elif len(reduce_dims) == var.ndim:
                        # prefer to aggregate over axis=None rather than
                        # axis=(0, 1) if they will be equivalent, because
                        # the former is often more efficient
                        reduce_dims = None  # type: ignore[assignment]
                    variables[name] = var.reduce(
                        func,
                        dim=reduce_dims,
                        keep_attrs=keep_attrs,
                        keepdims=keepdims,
                        **kwargs,
                    )

        coord_names = {k for k in self.coords if k in variables}
        indexes = {k: v for k, v in self.xindexes.items() if k in variables}
        attrs = self.attrs if keep_attrs else None
        return self._replace_with_new_dims(
            variables, coord_names=coord_names, attrs=attrs, indexes=indexes
        )

    def map(
        self,
        func: Callable,
        keep_attrs: bool = None,
        args: Iterable[Any] = (),
        **kwargs: Any,
    ) -> Dataset:
        """Apply a function to each variable in this dataset

        Parameters
        ----------
        func : callable
            Function which can be called in the form `func(x, *args, **kwargs)`
            to transform each DataArray `x` in this dataset into another
            DataArray.
        keep_attrs : bool, optional
            If True, the dataset's attributes (`attrs`) will be copied from
            the original object to the new one. If False, the new object will
            be returned without attributes.
        args : tuple, optional
            Positional arguments passed on to `func`.
        **kwargs : Any
            Keyword arguments passed on to `func`.

        Returns
        -------
        applied : Dataset
            Resulting dataset from applying ``func`` to each data variable.

        Examples
        --------
        >>> da = xr.DataArray(np.random.randn(2, 3))
        >>> ds = xr.Dataset({"foo": da, "bar": ("x", [-1, 2])})
        >>> ds
        <xarray.Dataset>
        Dimensions:  (dim_0: 2, dim_1: 3, x: 2)
        Dimensions without coordinates: dim_0, dim_1, x
        Data variables:
            foo      (dim_0, dim_1) float64 1.764 0.4002 0.9787 2.241 1.868 -0.9773
            bar      (x) int64 -1 2
        >>> ds.map(np.fabs)
        <xarray.Dataset>
        Dimensions:  (dim_0: 2, dim_1: 3, x: 2)
        Dimensions without coordinates: dim_0, dim_1, x
        Data variables:
            foo      (dim_0, dim_1) float64 1.764 0.4002 0.9787 2.241 1.868 0.9773
            bar      (x) float64 1.0 2.0
        """
        if keep_attrs is None:
            keep_attrs = _get_keep_attrs(default=False)
        variables = {
            k: maybe_wrap_array(v, func(v, *args, **kwargs))
            for k, v in self.data_vars.items()
        }
        if keep_attrs:
            for k, v in variables.items():
                v._copy_attrs_from(self.data_vars[k])
        attrs = self.attrs if keep_attrs else None
        return type(self)(variables, attrs=attrs)

    def apply(
        self,
        func: Callable,
        keep_attrs: bool = None,
        args: Iterable[Any] = (),
        **kwargs: Any,
    ) -> Dataset:
        """
        Backward compatible implementation of ``map``

        See Also
        --------
        Dataset.map
        """
        warnings.warn(
            "Dataset.apply may be deprecated in the future. Using Dataset.map is encouraged",
            PendingDeprecationWarning,
            stacklevel=2,
        )
        return self.map(func, keep_attrs, args, **kwargs)

    def assign(
        self, variables: Mapping[Any, Any] = None, **variables_kwargs: Hashable
    ) -> Dataset:
        """Assign new data variables to a Dataset, returning a new object
        with all the original variables in addition to the new ones.

        Parameters
        ----------
        variables : mapping of hashable to Any
            Mapping from variables names to the new values. If the new values
            are callable, they are computed on the Dataset and assigned to new
            data variables. If the values are not callable, (e.g. a DataArray,
            scalar, or array), they are simply assigned.
        **variables_kwargs
            The keyword arguments form of ``variables``.
            One of variables or variables_kwargs must be provided.

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

        Examples
        --------
        >>> x = xr.Dataset(
        ...     {
        ...         "temperature_c": (
        ...             ("lat", "lon"),
        ...             20 * np.random.rand(4).reshape(2, 2),
        ...         ),
        ...         "precipitation": (("lat", "lon"), np.random.rand(4).reshape(2, 2)),
        ...     },
        ...     coords={"lat": [10, 20], "lon": [150, 160]},
        ... )
        >>> x
        <xarray.Dataset>
        Dimensions:        (lat: 2, lon: 2)
        Coordinates:
          * lat            (lat) int64 10 20
          * lon            (lon) int64 150 160
        Data variables:
            temperature_c  (lat, lon) float64 10.98 14.3 12.06 10.9
            precipitation  (lat, lon) float64 0.4237 0.6459 0.4376 0.8918

        Where the value is a callable, evaluated on dataset:

        >>> x.assign(temperature_f=lambda x: x.temperature_c * 9 / 5 + 32)
        <xarray.Dataset>
        Dimensions:        (lat: 2, lon: 2)
        Coordinates:
          * lat            (lat) int64 10 20
          * lon            (lon) int64 150 160
        Data variables:
            temperature_c  (lat, lon) float64 10.98 14.3 12.06 10.9
            precipitation  (lat, lon) float64 0.4237 0.6459 0.4376 0.8918
            temperature_f  (lat, lon) float64 51.76 57.75 53.7 51.62

        Alternatively, the same behavior can be achieved by directly referencing an existing dataarray:

        >>> x.assign(temperature_f=x["temperature_c"] * 9 / 5 + 32)
        <xarray.Dataset>
        Dimensions:        (lat: 2, lon: 2)
        Coordinates:
          * lat            (lat) int64 10 20
          * lon            (lon) int64 150 160
        Data variables:
            temperature_c  (lat, lon) float64 10.98 14.3 12.06 10.9
            precipitation  (lat, lon) float64 0.4237 0.6459 0.4376 0.8918
            temperature_f  (lat, lon) float64 51.76 57.75 53.7 51.62

        """
        variables = either_dict_or_kwargs(variables, variables_kwargs, "assign")
        data = self.copy()
        # do all calculations first...
        results = data._calc_assign_results(variables)
        # ... and then assign
        data.update(results)
        return data

    def to_array(self, dim="variable", name=None):
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
        indexes = propagate_indexes(self._indexes)

        dims = (dim,) + broadcast_vars[0].dims

        return DataArray(
            data, coords, dims, attrs=self.attrs, name=name, indexes=indexes
        )

    def _normalize_dim_order(
        self, dim_order: list[Hashable] = None
    ) -> dict[Hashable, int]:
        """
        Check the validity of the provided dimensions if any and return the mapping
        between dimension name and their size.

        Parameters
        ----------
        dim_order
            Dimension order to validate (default to the alphabetical order if None).

        Returns
        -------
        result
            Validated dimensions mapping.

        """
        if dim_order is None:
            dim_order = list(self.dims)
        elif set(dim_order) != set(self.dims):
            raise ValueError(
                "dim_order {} does not match the set of dimensions of this "
                "Dataset: {}".format(dim_order, list(self.dims))
            )

        ordered_dims = {k: self.dims[k] for k in dim_order}

        return ordered_dims

    def to_pandas(self) -> pd.Series | pd.DataFrame:
        """Convert this dataset into a pandas object without changing the number of dimensions.

        The type of the returned object depends on the number of Dataset
        dimensions:

        * 0D -> `pandas.Series`
        * 1D -> `pandas.DataFrame`

        Only works for Datasets with 1 or fewer dimensions.
        """
        if len(self.dims) == 0:
            return pd.Series({k: v.item() for k, v in self.items()})
        if len(self.dims) == 1:
            return self.to_dataframe()
        raise ValueError(
            "cannot convert Datasets with %s dimensions into "
            "pandas objects without changing the number of dimensions. "
            "Please use Dataset.to_dataframe() instead." % len(self.dims)
        )

    def _to_dataframe(self, ordered_dims: Mapping[Any, int]):
        columns = [k for k in self.variables if k not in self.dims]
        data = [
            self._variables[k].set_dims(ordered_dims).values.reshape(-1)
            for k in columns
        ]
        index = self.coords.to_index([*ordered_dims])
        return pd.DataFrame(dict(zip(columns, data)), index=index)

    def to_dataframe(self, dim_order: list[Hashable] = None) -> pd.DataFrame:
        """Convert this dataset into a pandas.DataFrame.

        Non-index variables in this dataset form the columns of the
        DataFrame. The DataFrame is indexed by the Cartesian product of
        this dataset's indices.

        Parameters
        ----------
        dim_order
            Hierarchical dimension order for the resulting dataframe. All
            arrays are transposed to this order and then written out as flat
            vectors in contiguous order, so the last dimension in this list
            will be contiguous in the resulting DataFrame. This has a major
            influence on which operations are efficient on the resulting
            dataframe.

            If provided, must include all dimensions of this dataset. By
            default, dimensions are sorted alphabetically.

        Returns
        -------
        result
            Dataset as a pandas DataFrame.

        """

        ordered_dims = self._normalize_dim_order(dim_order=dim_order)

        return self._to_dataframe(ordered_dims=ordered_dims)

    def _set_sparse_data_from_dataframe(
        self, idx: pd.Index, arrays: list[tuple[Hashable, np.ndarray]], dims: tuple
    ) -> None:
        from sparse import COO

        if isinstance(idx, pd.MultiIndex):
            coords = np.stack([np.asarray(code) for code in idx.codes], axis=0)
            is_sorted = idx.is_monotonic_increasing
            shape = tuple(lev.size for lev in idx.levels)
        else:
            coords = np.arange(idx.size).reshape(1, -1)
            is_sorted = True
            shape = (idx.size,)

        for name, values in arrays:
            # In virtually all real use cases, the sparse array will now have
            # missing values and needs a fill_value. For consistency, don't
            # special case the rare exceptions (e.g., dtype=int without a
            # MultiIndex).
            dtype, fill_value = dtypes.maybe_promote(values.dtype)
            values = np.asarray(values, dtype=dtype)

            data = COO(
                coords,
                values,
                shape,
                has_duplicates=False,
                sorted=is_sorted,
                fill_value=fill_value,
            )
            self[name] = (dims, data)

    def _set_numpy_data_from_dataframe(
        self, idx: pd.Index, arrays: list[tuple[Hashable, np.ndarray]], dims: tuple
    ) -> None:
        if not isinstance(idx, pd.MultiIndex):
            for name, values in arrays:
                self[name] = (dims, values)
            return

        # NB: similar, more general logic, now exists in
        # variable.unstack_once; we could consider combining them at some
        # point.

        shape = tuple(lev.size for lev in idx.levels)
        indexer = tuple(idx.codes)

        # We already verified that the MultiIndex has all unique values, so
        # there are missing values if and only if the size of output arrays is
        # larger that the index.
        missing_values = np.prod(shape) > idx.shape[0]

        for name, values in arrays:
            # NumPy indexing is much faster than using DataFrame.reindex() to
            # fill in missing values:
            # https://stackoverflow.com/a/35049899/809705
            if missing_values:
                dtype, fill_value = dtypes.maybe_promote(values.dtype)
                data = np.full(shape, fill_value, dtype)
            else:
                # If there are no missing values, keep the existing dtype
                # instead of promoting to support NA, e.g., keep integer
                # columns as integers.
                # TODO: consider removing this special case, which doesn't
                # exist for sparse=True.
                data = np.zeros(shape, values.dtype)
            data[indexer] = values
            self[name] = (dims, data)

    @classmethod
    def from_dataframe(cls, dataframe: pd.DataFrame, sparse: bool = False) -> Dataset:
        """Convert a pandas.DataFrame into an xarray.Dataset

        Each column will be converted into an independent variable in the
        Dataset. If the dataframe's index is a MultiIndex, it will be expanded
        into a tensor product of one-dimensional indices (filling in missing
        values with NaN). This method will produce a Dataset very similar to
        that on which the 'to_dataframe' method was called, except with
        possibly redundant dimensions (since all dataset variables will have
        the same dimensionality)

        Parameters
        ----------
        dataframe : DataFrame
            DataFrame from which to copy data and indices.
        sparse : bool, default: False
            If true, create a sparse arrays instead of dense numpy arrays. This
            can potentially save a large amount of memory if the DataFrame has
            a MultiIndex. Requires the sparse package (sparse.pydata.org).

        Returns
        -------
        New Dataset.

        See Also
        --------
        xarray.DataArray.from_series
        pandas.DataFrame.to_xarray
        """
        # TODO: Add an option to remove dimensions along which the variables
        # are constant, to enable consistent serialization to/from a dataframe,
        # even if some variables have different dimensionality.

        if not dataframe.columns.is_unique:
            raise ValueError("cannot convert DataFrame with non-unique columns")

        idx = remove_unused_levels_categories(dataframe.index)

        if isinstance(idx, pd.MultiIndex) and not idx.is_unique:
            raise ValueError(
                "cannot convert a DataFrame with a non-unique MultiIndex into xarray"
            )

        # Cast to a NumPy array first, in case the Series is a pandas Extension
        # array (which doesn't have a valid NumPy dtype)
        # TODO: allow users to control how this casting happens, e.g., by
        # forwarding arguments to pandas.Series.to_numpy?
        arrays = [(k, np.asarray(v)) for k, v in dataframe.items()]

        obj = cls()

        if isinstance(idx, pd.MultiIndex):
            dims = tuple(
                name if name is not None else "level_%i" % n
                for n, name in enumerate(idx.names)
            )
            for dim, lev in zip(dims, idx.levels):
                obj[dim] = (dim, lev)
        else:
            index_name = idx.name if idx.name is not None else "index"
            dims = (index_name,)
            obj[index_name] = (dims, idx)

        if sparse:
            obj._set_sparse_data_from_dataframe(idx, arrays, dims)
        else:
            obj._set_numpy_data_from_dataframe(idx, arrays, dims)
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

            If provided, must include all dimensions of this dataset. By
            default, dimensions are sorted alphabetically.
        set_index : bool, optional
            If set_index=True, the dask DataFrame is indexed by this dataset's
            coordinate. Since dask DataFrames do not support multi-indexes,
            set_index only works if the dataset only contains one dimension.

        Returns
        -------
        dask.dataframe.DataFrame
        """

        import dask.array as da
        import dask.dataframe as dd

        ordered_dims = self._normalize_dim_order(dim_order=dim_order)

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
            dim_order = [*ordered_dims]

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
        Useful for converting to json. To avoid datetime incompatibility
        use decode_times=False kwarg in xarrray.open_dataset.

        Parameters
        ----------
        data : bool, optional
            Whether to include the actual data in the dictionary. When set to
            False, returns just the schema.

        See Also
        --------
        Dataset.from_dict
        """
        d = {
            "coords": {},
            "attrs": decode_numpy_dict_values(self.attrs),
            "dims": dict(self.dims),
            "data_vars": {},
        }
        for k in self.coords:
            d["coords"].update({k: self[k].variable.to_dict(data=data)})
        for k in self.data_vars:
            d["data_vars"].update({k: self[k].variable.to_dict(data=data)})
        return d

    @classmethod
    def from_dict(cls, d):
        """
        Convert a dictionary into an xarray.Dataset.

        Input dict can take several forms:

        .. code:: python

            d = {
                "t": {"dims": ("t"), "data": t},
                "a": {"dims": ("t"), "data": x},
                "b": {"dims": ("t"), "data": y},
            }

            d = {
                "coords": {"t": {"dims": "t", "data": t, "attrs": {"units": "s"}}},
                "attrs": {"title": "air temperature"},
                "dims": "t",
                "data_vars": {
                    "a": {"dims": "t", "data": x},
                    "b": {"dims": "t", "data": y},
                },
            }

        where "t" is the name of the dimesion, "a" and "b" are names of data
        variables and t, x, and y are lists, numpy.arrays or pandas objects.

        Parameters
        ----------
        d : dict-like
            Mapping with a minimum structure of
                ``{"var_0": {"dims": [..], "data": [..]}, \
                            ...}``

        Returns
        -------
        obj : xarray.Dataset

        See also
        --------
        Dataset.to_dict
        DataArray.from_dict
        """

        if not {"coords", "data_vars"}.issubset(set(d)):
            variables = d.items()
        else:
            import itertools

            variables = itertools.chain(
                d.get("coords", {}).items(), d.get("data_vars", {}).items()
            )
        try:
            variable_dict = {
                k: (v["dims"], v["data"], v.get("attrs")) for k, v in variables
            }
        except KeyError as e:
            raise ValueError(
                "cannot convert dict without the key "
                "'{dims_data}'".format(dims_data=str(e.args[0]))
            )
        obj = cls(variable_dict)

        # what if coords aren't dims?
        coords = set(d.get("coords", {})) - set(d.get("dims", {}))
        obj = obj.set_coords(coords)

        obj.attrs.update(d.get("attrs", {}))

        return obj

    def _unary_op(self, f, *args, **kwargs):
        variables = {}
        keep_attrs = kwargs.pop("keep_attrs", None)
        if keep_attrs is None:
            keep_attrs = _get_keep_attrs(default=True)
        for k, v in self._variables.items():
            if k in self._coord_names:
                variables[k] = v
            else:
                variables[k] = f(v, *args, **kwargs)
                if keep_attrs:
                    variables[k].attrs = v._attrs
        attrs = self._attrs if keep_attrs else None
        return self._replace_with_new_dims(variables, attrs=attrs)

    def _binary_op(self, other, f, reflexive=False, join=None):
        from .dataarray import DataArray

        if isinstance(other, groupby.GroupBy):
            return NotImplemented
        align_type = OPTIONS["arithmetic_join"] if join is None else join
        if isinstance(other, (DataArray, Dataset)):
            self, other = align(self, other, join=align_type, copy=False)
        g = f if not reflexive else lambda x, y: f(y, x)
        ds = self._calculate_binary_op(g, other, join=align_type)
        return ds

    def _inplace_binary_op(self, other, f):
        from .dataarray import DataArray

        if isinstance(other, groupby.GroupBy):
            raise TypeError(
                "in-place operations between a Dataset and "
                "a grouped object are not permitted"
            )
        # we don't actually modify arrays in-place with in-place Dataset
        # arithmetic -- this lets us automatically align things
        if isinstance(other, (DataArray, Dataset)):
            other = other.reindex_like(self, copy=False)
        g = ops.inplace_to_noninplace_op(f)
        ds = self._calculate_binary_op(g, other, inplace=True)
        self._replace_with_new_dims(
            ds._variables,
            ds._coord_names,
            attrs=ds._attrs,
            indexes=ds._indexes,
            inplace=True,
        )
        return self

    def _calculate_binary_op(self, f, other, join="inner", inplace=False):
        def apply_over_both(lhs_data_vars, rhs_data_vars, lhs_vars, rhs_vars):
            if inplace and set(lhs_data_vars) != set(rhs_data_vars):
                raise ValueError(
                    "datasets must have the same data variables "
                    f"for in-place arithmetic operations: {list(lhs_data_vars)}, {list(rhs_data_vars)}"
                )

            dest_vars = {}

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
            new_data_vars = apply_over_both(
                self.data_vars, other, self.data_vars, other
            )
            return Dataset(new_data_vars)

        other_coords = getattr(other, "coords", None)
        ds = self.coords.merge(other_coords)

        if isinstance(other, Dataset):
            new_vars = apply_over_both(
                self.data_vars, other.data_vars, self.variables, other.variables
            )
        else:
            other_variable = getattr(other, "variable", other)
            new_vars = {k: f(self.variables[k], other_variable) for k in self.data_vars}
        ds._variables.update(new_vars)
        ds._dims = calculate_dimensions(ds._variables)
        return ds

    def _copy_attrs_from(self, other):
        self.attrs = other.attrs
        for v in other.variables:
            if v in self.variables:
                self.variables[v].attrs = other.variables[v].attrs

    def diff(self, dim, n=1, label="upper"):
        """Calculate the n-th order discrete difference along given axis.

        Parameters
        ----------
        dim : str
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

        Notes
        -----
        `n` matches numpy's behavior and is different from pandas' first argument named
        `periods`.

        Examples
        --------
        >>> ds = xr.Dataset({"foo": ("x", [5, 5, 6, 6])})
        >>> ds.diff("x")
        <xarray.Dataset>
        Dimensions:  (x: 3)
        Dimensions without coordinates: x
        Data variables:
            foo      (x) int64 0 1 0
        >>> ds.diff("x", 2)
        <xarray.Dataset>
        Dimensions:  (x: 2)
        Dimensions without coordinates: x
        Data variables:
            foo      (x) int64 1 -1

        See Also
        --------
        Dataset.differentiate
        """
        if n == 0:
            return self
        if n < 0:
            raise ValueError(f"order `n` must be non-negative but got {n}")

        # prepare slices
        kwargs_start = {dim: slice(None, -1)}
        kwargs_end = {dim: slice(1, None)}

        # prepare new coordinate
        if label == "upper":
            kwargs_new = kwargs_end
        elif label == "lower":
            kwargs_new = kwargs_start
        else:
            raise ValueError("The 'label' argument has to be either 'upper' or 'lower'")

        variables = {}

        for name, var in self.variables.items():
            if dim in var.dims:
                if name in self.data_vars:
                    variables[name] = var.isel(**kwargs_end) - var.isel(**kwargs_start)
                else:
                    variables[name] = var.isel(**kwargs_new)
            else:
                variables[name] = var

        indexes = dict(self.xindexes)
        if dim in indexes:
            if isinstance(indexes[dim], PandasIndex):
                # maybe optimize? (pandas index already indexed above with var.isel)
                new_index = indexes[dim].index[kwargs_new[dim]]
                if isinstance(new_index, pd.MultiIndex):
                    indexes[dim] = PandasMultiIndex(new_index, dim)
                else:
                    indexes[dim] = PandasIndex(new_index, dim)

        difference = self._replace_with_new_dims(variables, indexes=indexes)

        if n > 1:
            return difference.diff(dim, n - 1)
        else:
            return difference

    def shift(
        self,
        shifts: Mapping[Hashable, int] = None,
        fill_value: Any = dtypes.NA,
        **shifts_kwargs: int,
    ) -> Dataset:

        """Shift this dataset by an offset along one or more dimensions.

        Only data variables are moved; coordinates stay in place. This is
        consistent with the behavior of ``shift`` in pandas.

        Values shifted from beyond array bounds will appear at one end of
        each dimension, which are filled according to `fill_value`. For periodic
        offsets instead see `roll`.

        Parameters
        ----------
        shifts : mapping of hashable to int
            Integer offset to shift along each of the given dimensions.
            Positive offsets shift to the right; negative offsets shift to the
            left.
        fill_value : scalar or dict-like, optional
            Value to use for newly missing values. If a dict-like, maps
            variable names (including coordinates) to fill values.
        **shifts_kwargs
            The keyword arguments form of ``shifts``.
            One of shifts or shifts_kwargs must be provided.

        Returns
        -------
        shifted : Dataset
            Dataset with the same coordinates and attributes but shifted data
            variables.

        See Also
        --------
        roll

        Examples
        --------
        >>> ds = xr.Dataset({"foo": ("x", list("abcde"))})
        >>> ds.shift(x=2)
        <xarray.Dataset>
        Dimensions:  (x: 5)
        Dimensions without coordinates: x
        Data variables:
            foo      (x) object nan nan 'a' 'b' 'c'
        """
        shifts = either_dict_or_kwargs(shifts, shifts_kwargs, "shift")
        invalid = [k for k in shifts if k not in self.dims]
        if invalid:
            raise ValueError(f"dimensions {invalid!r} do not exist")

        variables = {}
        for name, var in self.variables.items():
            if name in self.data_vars:
                fill_value_ = (
                    fill_value.get(name, dtypes.NA)
                    if isinstance(fill_value, dict)
                    else fill_value
                )

                var_shifts = {k: v for k, v in shifts.items() if k in var.dims}
                variables[name] = var.shift(fill_value=fill_value_, shifts=var_shifts)
            else:
                variables[name] = var

        return self._replace(variables)

    def roll(
        self,
        shifts: Mapping[Hashable, int] = None,
        roll_coords: bool = False,
        **shifts_kwargs: int,
    ) -> Dataset:
        """Roll this dataset by an offset along one or more dimensions.

        Unlike shift, roll treats the given dimensions as periodic, so will not
        create any missing values to be filled.

        Also unlike shift, roll may rotate all variables, including coordinates
        if specified. The direction of rotation is consistent with
        :py:func:`numpy.roll`.

        Parameters
        ----------
        shifts : mapping of hashable to int, optional
            A dict with keys matching dimensions and values given
            by integers to rotate each of the given dimensions. Positive
            offsets roll to the right; negative offsets roll to the left.
        roll_coords : bool, default: False
            Indicates whether to roll the coordinates by the offset too.
        **shifts_kwargs : {dim: offset, ...}, optional
            The keyword arguments form of ``shifts``.
            One of shifts or shifts_kwargs must be provided.

        Returns
        -------
        rolled : Dataset
            Dataset with the same attributes but rolled data and coordinates.

        See Also
        --------
        shift

        Examples
        --------
        >>> ds = xr.Dataset({"foo": ("x", list("abcde"))}, coords={"x": np.arange(5)})
        >>> ds.roll(x=2)
        <xarray.Dataset>
        Dimensions:  (x: 5)
        Coordinates:
          * x        (x) int64 0 1 2 3 4
        Data variables:
            foo      (x) <U1 'd' 'e' 'a' 'b' 'c'

        >>> ds.roll(x=2, roll_coords=True)
        <xarray.Dataset>
        Dimensions:  (x: 5)
        Coordinates:
          * x        (x) int64 3 4 0 1 2
        Data variables:
            foo      (x) <U1 'd' 'e' 'a' 'b' 'c'

        """
        shifts = either_dict_or_kwargs(shifts, shifts_kwargs, "roll")
        invalid = [k for k in shifts if k not in self.dims]
        if invalid:
            raise ValueError(f"dimensions {invalid!r} do not exist")

        unrolled_vars = () if roll_coords else self.coords

        variables = {}
        for k, var in self.variables.items():
            if k not in unrolled_vars:
                variables[k] = var.roll(
                    shifts={k: s for k, s in shifts.items() if k in var.dims}
                )
            else:
                variables[k] = var

        if roll_coords:
            indexes: dict[Hashable, Index] = {}
            idx: pd.Index
            for k, idx in self.xindexes.items():
                (dim,) = self.variables[k].dims
                if dim in shifts:
                    indexes[k] = roll_index(idx, shifts[dim])
                else:
                    indexes[k] = idx
        else:
            indexes = dict(self.xindexes)

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
        variables : str, DataArray, or list of str or DataArray
            1D DataArray objects or name(s) of 1D variable(s) in
            coords/data_vars whose values are used to sort the dataset.
        ascending : bool, optional
            Whether to sort by ascending or descending order.

        Returns
        -------
        sorted : Dataset
            A new dataset where all the specified dims are sorted by dim
            labels.

        See Also
        --------
        DataArray.sortby
        numpy.sort
        pandas.sort_values
        pandas.sort_index

        Examples
        --------
        >>> ds = xr.Dataset(
        ...     {
        ...         "A": (("x", "y"), [[1, 2], [3, 4]]),
        ...         "B": (("x", "y"), [[5, 6], [7, 8]]),
        ...     },
        ...     coords={"x": ["b", "a"], "y": [1, 0]},
        ... )
        >>> ds.sortby("x")
        <xarray.Dataset>
        Dimensions:  (x: 2, y: 2)
        Coordinates:
          * x        (x) <U1 'a' 'b'
          * y        (y) int64 1 0
        Data variables:
            A        (x, y) int64 3 4 1 2
            B        (x, y) int64 7 8 5 6
        """
        from .dataarray import DataArray

        if not isinstance(variables, list):
            variables = [variables]
        else:
            variables = variables
        variables = [v if isinstance(v, DataArray) else self[v] for v in variables]
        aligned_vars = align(self, *variables, join="left")
        aligned_self = aligned_vars[0]
        aligned_other_vars = aligned_vars[1:]
        vars_by_dim = defaultdict(list)
        for data_array in aligned_other_vars:
            if data_array.ndim != 1:
                raise ValueError("Input DataArray is not 1-D.")
            (key,) = data_array.dims
            vars_by_dim[key].append(data_array)

        indices = {}
        for key, arrays in vars_by_dim.items():
            order = np.lexsort(tuple(reversed(arrays)))
            indices[key] = order if ascending else order[::-1]
        return aligned_self.isel(**indices)

    def quantile(
        self,
        q,
        dim=None,
        interpolation="linear",
        numeric_only=False,
        keep_attrs=None,
        skipna=True,
    ):
        """Compute the qth quantile of the data along the specified dimension.

        Returns the qth quantiles(s) of the array elements for each variable
        in the Dataset.

        Parameters
        ----------
        q : float or array-like of float
            Quantile to compute, which must be between 0 and 1 inclusive.
        dim : str or sequence of str, optional
            Dimension(s) over which to apply quantile.
        interpolation : {"linear", "lower", "higher", "midpoint", "nearest"}, default: "linear"
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
        skipna : bool, optional
            Whether to skip missing values when aggregating.

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
        numpy.nanquantile, numpy.quantile, pandas.Series.quantile, DataArray.quantile

        Examples
        --------
        >>> ds = xr.Dataset(
        ...     {"a": (("x", "y"), [[0.7, 4.2, 9.4, 1.5], [6.5, 7.3, 2.6, 1.9]])},
        ...     coords={"x": [7, 9], "y": [1, 1.5, 2, 2.5]},
        ... )
        >>> ds.quantile(0)  # or ds.quantile(0, dim=...)
        <xarray.Dataset>
        Dimensions:   ()
        Coordinates:
            quantile  float64 0.0
        Data variables:
            a         float64 0.7
        >>> ds.quantile(0, dim="x")
        <xarray.Dataset>
        Dimensions:   (y: 4)
        Coordinates:
          * y         (y) float64 1.0 1.5 2.0 2.5
            quantile  float64 0.0
        Data variables:
            a         (y) float64 0.7 4.2 2.6 1.5
        >>> ds.quantile([0, 0.5, 1])
        <xarray.Dataset>
        Dimensions:   (quantile: 3)
        Coordinates:
          * quantile  (quantile) float64 0.0 0.5 1.0
        Data variables:
            a         (quantile) float64 0.7 3.4 9.4
        >>> ds.quantile([0, 0.5, 1], dim="x")
        <xarray.Dataset>
        Dimensions:   (quantile: 3, y: 4)
        Coordinates:
          * y         (y) float64 1.0 1.5 2.0 2.5
          * quantile  (quantile) float64 0.0 0.5 1.0
        Data variables:
            a         (quantile, y) float64 0.7 4.2 2.6 1.5 3.6 ... 1.7 6.5 7.3 9.4 1.9
        """

        if isinstance(dim, str):
            dims = {dim}
        elif dim in [None, ...]:
            dims = set(self.dims)
        else:
            dims = set(dim)

        _assert_empty(
            [d for d in dims if d not in self.dims],
            "Dataset does not contain the dimensions: %s",
        )

        q = np.asarray(q, dtype=np.float64)

        variables = {}
        for name, var in self.variables.items():
            reduce_dims = [d for d in var.dims if d in dims]
            if reduce_dims or not var.dims:
                if name not in self.coords:
                    if (
                        not numeric_only
                        or np.issubdtype(var.dtype, np.number)
                        or var.dtype == np.bool_
                    ):
                        if len(reduce_dims) == var.ndim:
                            # prefer to aggregate over axis=None rather than
                            # axis=(0, 1) if they will be equivalent, because
                            # the former is often more efficient
                            reduce_dims = None
                        variables[name] = var.quantile(
                            q,
                            dim=reduce_dims,
                            interpolation=interpolation,
                            keep_attrs=keep_attrs,
                            skipna=skipna,
                        )

            else:
                variables[name] = var

        # construct the new dataset
        coord_names = {k for k in self.coords if k in variables}
        indexes = {k: v for k, v in self.xindexes.items() if k in variables}
        if keep_attrs is None:
            keep_attrs = _get_keep_attrs(default=False)
        attrs = self.attrs if keep_attrs else None
        new = self._replace_with_new_dims(
            variables, coord_names=coord_names, attrs=attrs, indexes=indexes
        )
        return new.assign_coords(quantile=q)

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
        if not OPTIONS["use_bottleneck"]:
            raise RuntimeError(
                "rank requires bottleneck to be enabled."
                " Call `xr.set_options(use_bottleneck=True)` to enable it."
            )

        if dim not in self.dims:
            raise ValueError(f"Dataset does not contain the dimension: {dim}")

        variables = {}
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
        coord : str
            The coordinate to be used to compute the gradient.
        edge_order : {1, 2}, default: 1
            N-th order accurate differences at the boundaries.
        datetime_unit : None or {"Y", "M", "W", "D", "h", "m", "s", "ms", \
            "us", "ns", "ps", "fs", "as"}, default: None
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
            raise ValueError(f"Coordinate {coord} does not exist.")

        coord_var = self[coord].variable
        if coord_var.ndim != 1:
            raise ValueError(
                "Coordinate {} must be 1 dimensional but is {}"
                " dimensional".format(coord, coord_var.ndim)
            )

        dim = coord_var.dims[0]
        if _contains_datetime_like_objects(coord_var):
            if coord_var.dtype.kind in "mM" and datetime_unit is None:
                datetime_unit, _ = np.datetime_data(coord_var.dtype)
            elif datetime_unit is None:
                datetime_unit = "s"  # Default to seconds for cftime objects
            coord_var = coord_var._to_numeric(datetime_unit=datetime_unit)

        variables = {}
        for k, v in self.variables.items():
            if k in self.data_vars and dim in v.dims and k not in self.coords:
                if _contains_datetime_like_objects(v):
                    v = v._to_numeric(datetime_unit=datetime_unit)
                grad = duck_array_ops.gradient(
                    v.data,
                    coord_var.data,
                    edge_order=edge_order,
                    axis=v.get_axis_num(dim),
                )
                variables[k] = Variable(v.dims, grad)
            else:
                variables[k] = v
        return self._replace(variables)

    def integrate(
        self,
        coord: Hashable | Sequence[Hashable],
        datetime_unit: str = None,
    ) -> Dataset:
        """Integrate along the given coordinate using the trapezoidal rule.

        .. note::
            This feature is limited to simple cartesian geometry, i.e. coord
            must be one dimensional.

        Parameters
        ----------
        coord : hashable, or sequence of hashable
            Coordinate(s) used for the integration.
        datetime_unit : {'Y', 'M', 'W', 'D', 'h', 'm', 's', 'ms', 'us', 'ns', \
                        'ps', 'fs', 'as'}, optional
            Specify the unit if datetime coordinate is used.

        Returns
        -------
        integrated : Dataset

        See also
        --------
        DataArray.integrate
        numpy.trapz : corresponding numpy function

        Examples
        --------
        >>> ds = xr.Dataset(
        ...     data_vars={"a": ("x", [5, 5, 6, 6]), "b": ("x", [1, 2, 1, 0])},
        ...     coords={"x": [0, 1, 2, 3], "y": ("x", [1, 7, 3, 5])},
        ... )
        >>> ds
        <xarray.Dataset>
        Dimensions:  (x: 4)
        Coordinates:
          * x        (x) int64 0 1 2 3
            y        (x) int64 1 7 3 5
        Data variables:
            a        (x) int64 5 5 6 6
            b        (x) int64 1 2 1 0
        >>> ds.integrate("x")
        <xarray.Dataset>
        Dimensions:  ()
        Data variables:
            a        float64 16.5
            b        float64 3.5
        >>> ds.integrate("y")
        <xarray.Dataset>
        Dimensions:  ()
        Data variables:
            a        float64 20.0
            b        float64 4.0
        """
        if not isinstance(coord, (list, tuple)):
            coord = (coord,)
        result = self
        for c in coord:
            result = result._integrate_one(c, datetime_unit=datetime_unit)
        return result

    def _integrate_one(self, coord, datetime_unit=None, cumulative=False):
        from .variable import Variable

        if coord not in self.variables and coord not in self.dims:
            raise ValueError(f"Coordinate {coord} does not exist.")

        coord_var = self[coord].variable
        if coord_var.ndim != 1:
            raise ValueError(
                "Coordinate {} must be 1 dimensional but is {}"
                " dimensional".format(coord, coord_var.ndim)
            )

        dim = coord_var.dims[0]
        if _contains_datetime_like_objects(coord_var):
            if coord_var.dtype.kind in "mM" and datetime_unit is None:
                datetime_unit, _ = np.datetime_data(coord_var.dtype)
            elif datetime_unit is None:
                datetime_unit = "s"  # Default to seconds for cftime objects
            coord_var = coord_var._replace(
                data=datetime_to_numeric(coord_var.data, datetime_unit=datetime_unit)
            )

        variables = {}
        coord_names = set()
        for k, v in self.variables.items():
            if k in self.coords:
                if dim not in v.dims or cumulative:
                    variables[k] = v
                    coord_names.add(k)
            else:
                if k in self.data_vars and dim in v.dims:
                    if _contains_datetime_like_objects(v):
                        v = datetime_to_numeric(v, datetime_unit=datetime_unit)
                    if cumulative:
                        integ = duck_array_ops.cumulative_trapezoid(
                            v.data, coord_var.data, axis=v.get_axis_num(dim)
                        )
                        v_dims = v.dims
                    else:
                        integ = duck_array_ops.trapz(
                            v.data, coord_var.data, axis=v.get_axis_num(dim)
                        )
                        v_dims = list(v.dims)
                        v_dims.remove(dim)
                    variables[k] = Variable(v_dims, integ)
                else:
                    variables[k] = v
        indexes = {k: v for k, v in self.xindexes.items() if k in variables}
        return self._replace_with_new_dims(
            variables, coord_names=coord_names, indexes=indexes
        )

    def cumulative_integrate(
        self,
        coord: Hashable | Sequence[Hashable],
        datetime_unit: str = None,
    ) -> Dataset:
        """Integrate along the given coordinate using the trapezoidal rule.

        .. note::
            This feature is limited to simple cartesian geometry, i.e. coord
            must be one dimensional.

            The first entry of the cumulative integral of each variable is always 0, in
            order to keep the length of the dimension unchanged between input and
            output.

        Parameters
        ----------
        coord : hashable, or sequence of hashable
            Coordinate(s) used for the integration.
        datetime_unit : {'Y', 'M', 'W', 'D', 'h', 'm', 's', 'ms', 'us', 'ns', \
                        'ps', 'fs', 'as'}, optional
            Specify the unit if datetime coordinate is used.

        Returns
        -------
        integrated : Dataset

        See also
        --------
        DataArray.cumulative_integrate
        scipy.integrate.cumulative_trapezoid : corresponding scipy function

        Examples
        --------
        >>> ds = xr.Dataset(
        ...     data_vars={"a": ("x", [5, 5, 6, 6]), "b": ("x", [1, 2, 1, 0])},
        ...     coords={"x": [0, 1, 2, 3], "y": ("x", [1, 7, 3, 5])},
        ... )
        >>> ds
        <xarray.Dataset>
        Dimensions:  (x: 4)
        Coordinates:
          * x        (x) int64 0 1 2 3
            y        (x) int64 1 7 3 5
        Data variables:
            a        (x) int64 5 5 6 6
            b        (x) int64 1 2 1 0
        >>> ds.cumulative_integrate("x")
        <xarray.Dataset>
        Dimensions:  (x: 4)
        Coordinates:
          * x        (x) int64 0 1 2 3
            y        (x) int64 1 7 3 5
        Data variables:
            a        (x) float64 0.0 5.0 10.5 16.5
            b        (x) float64 0.0 1.5 3.0 3.5
        >>> ds.cumulative_integrate("y")
        <xarray.Dataset>
        Dimensions:  (x: 4)
        Coordinates:
          * x        (x) int64 0 1 2 3
            y        (x) int64 1 7 3 5
        Data variables:
            a        (x) float64 0.0 30.0 8.0 20.0
            b        (x) float64 0.0 9.0 3.0 4.0
        """
        if not isinstance(coord, (list, tuple)):
            coord = (coord,)
        result = self
        for c in coord:
            result = result._integrate_one(
                c, datetime_unit=datetime_unit, cumulative=True
            )
        return result

    @property
    def real(self):
        return self.map(lambda x: x.real, keep_attrs=True)

    @property
    def imag(self):
        return self.map(lambda x: x.imag, keep_attrs=True)

    plot = utils.UncachedAccessor(_Dataset_PlotMethods)

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
        **kwargs
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
        >>> temp = 15 + 8 * np.random.randn(2, 2, 3)
        >>> precip = 10 * np.random.rand(2, 2, 3)
        >>> lon = [[-99.83, -99.32], [-99.79, -99.23]]
        >>> lat = [[42.25, 42.21], [42.63, 42.59]]
        >>> dims = ["x", "y", "time"]
        >>> temp_attr = dict(standard_name="air_potential_temperature")
        >>> precip_attr = dict(standard_name="convective_precipitation_flux")

        >>> ds = xr.Dataset(
        ...     dict(
        ...         temperature=(dims, temp, temp_attr),
        ...         precipitation=(dims, precip, precip_attr),
        ...     ),
        ...     coords=dict(
        ...         lon=(["x", "y"], lon),
        ...         lat=(["x", "y"], lat),
        ...         time=pd.date_range("2014-09-06", periods=3),
        ...         reference_time=pd.Timestamp("2014-09-05"),
        ...     ),
        ... )

        Get variables matching a specific standard_name:

        >>> ds.filter_by_attrs(standard_name="convective_precipitation_flux")
        <xarray.Dataset>
        Dimensions:         (x: 2, y: 2, time: 3)
        Coordinates:
            lon             (x, y) float64 -99.83 -99.32 -99.79 -99.23
            lat             (x, y) float64 42.25 42.21 42.63 42.59
          * time            (time) datetime64[ns] 2014-09-06 2014-09-07 2014-09-08
            reference_time  datetime64[ns] 2014-09-05
        Dimensions without coordinates: x, y
        Data variables:
            precipitation   (x, y, time) float64 5.68 9.256 0.7104 ... 7.992 4.615 7.805

        Get all variables that have a standard_name attribute:

        >>> standard_name = lambda v: v is not None
        >>> ds.filter_by_attrs(standard_name=standard_name)
        <xarray.Dataset>
        Dimensions:         (x: 2, y: 2, time: 3)
        Coordinates:
            lon             (x, y) float64 -99.83 -99.32 -99.79 -99.23
            lat             (x, y) float64 42.25 42.21 42.63 42.59
          * time            (time) datetime64[ns] 2014-09-06 2014-09-07 2014-09-08
            reference_time  datetime64[ns] 2014-09-05
        Dimensions without coordinates: x, y
        Data variables:
            temperature     (x, y, time) float64 29.11 18.2 22.83 ... 18.28 16.15 26.63
            precipitation   (x, y, time) float64 5.68 9.256 0.7104 ... 7.992 4.615 7.805

        """
        selection = []
        for var_name, variable in self.variables.items():
            has_value_flag = False
            for attr_name, pattern in kwargs.items():
                attr_value = variable.attrs.get(attr_name)
                if (callable(pattern) and pattern(attr_value)) or attr_value == pattern:
                    has_value_flag = True
                else:
                    has_value_flag = False
                    break
            if has_value_flag is True:
                selection.append(var_name)
        return self[selection]

    def unify_chunks(self) -> Dataset:
        """Unify chunk size along all chunked dimensions of this Dataset.

        Returns
        -------
        Dataset with consistent chunk sizes for all dask-array variables

        See Also
        --------
        dask.array.core.unify_chunks
        """

        return unify_chunks(self)[0]

    def map_blocks(
        self,
        func: Callable[..., T_Xarray],
        args: Sequence[Any] = (),
        kwargs: Mapping[str, Any] = None,
        template: DataArray | Dataset | None = None,
    ) -> T_Xarray:
        """
        Apply a function to each block of this Dataset.

        .. warning::
            This method is experimental and its signature may change.

        Parameters
        ----------
        func : callable
            User-provided function that accepts a Dataset as its first
            parameter. The function will receive a subset or 'block' of this Dataset (see below),
            corresponding to one chunk along each chunked dimension. ``func`` will be
            executed as ``func(subset_dataset, *subset_args, **kwargs)``.

            This function must return either a single DataArray or a single Dataset.

            This function cannot add a new chunked dimension.
        args : sequence
            Passed to func after unpacking and subsetting any xarray objects by blocks.
            xarray objects in args must be aligned with obj, otherwise an error is raised.
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
        subset to each block. Each block is loaded into memory. In the more common case where
        ``func`` can work on numpy arrays, it is recommended to use ``apply_ufunc``.

        If none of the variables in this object is backed by dask arrays, calling this function is
        equivalent to calling ``func(obj, *args, **kwargs)``.

        See Also
        --------
        dask.array.map_blocks, xarray.apply_ufunc, xarray.Dataset.map_blocks
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
        ...
        >>> time = xr.cftime_range("1990-01", "1992-01", freq="M")
        >>> month = xr.DataArray(time.month, coords={"time": time}, dims=["time"])
        >>> np.random.seed(123)
        >>> array = xr.DataArray(
        ...     np.random.rand(len(time)),
        ...     dims=["time"],
        ...     coords={"time": time, "month": month},
        ... ).chunk()
        >>> ds = xr.Dataset({"a": array})
        >>> ds.map_blocks(calculate_anomaly, template=ds).compute()
        <xarray.Dataset>
        Dimensions:  (time: 24)
        Coordinates:
          * time     (time) object 1990-01-31 00:00:00 ... 1991-12-31 00:00:00
            month    (time) int64 1 2 3 4 5 6 7 8 9 10 11 12 1 2 3 4 5 6 7 8 9 10 11 12
        Data variables:
            a        (time) float64 0.1289 0.1132 -0.0856 ... 0.2287 0.1906 -0.05901

        Note that one must explicitly use ``args=[]`` and ``kwargs={}`` to pass arguments
        to the function being applied in ``xr.map_blocks()``:

        >>> ds.map_blocks(
        ...     calculate_anomaly,
        ...     kwargs={"groupby_type": "time.year"},
        ...     template=ds,
        ... )
        <xarray.Dataset>
        Dimensions:  (time: 24)
        Coordinates:
          * time     (time) object 1990-01-31 00:00:00 ... 1991-12-31 00:00:00
            month    (time) int64 dask.array<chunksize=(24,), meta=np.ndarray>
        Data variables:
            a        (time) float64 dask.array<chunksize=(24,), meta=np.ndarray>
        """
        from .parallel import map_blocks

        return map_blocks(func, self, args, kwargs, template)

    def polyfit(
        self,
        dim: Hashable,
        deg: int,
        skipna: bool = None,
        rcond: float = None,
        w: Hashable | Any = None,
        full: bool = False,
        cov: bool | str = False,
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
        w : hashable or Any, optional
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
            A single dataset which contains (for each "var" in the input dataset):

            [var]_polyfit_coefficients
                The coefficients of the best fit for each variable in this dataset.
            [var]_polyfit_residuals
                The residuals of the least-square computation for each variable (only included if `full=True`)
                When the matrix rank is deficient, np.nan is returned.
            [dim]_matrix_rank
                The effective rank of the scaled Vandermonde coefficient matrix (only included if `full=True`)
                The rank is computed ignoring the NaN values that might be skipped.
            [dim]_singular_values
                The singular values of the scaled Vandermonde coefficient matrix (only included if `full=True`)
            [var]_polyfit_covariance
                The covariance matrix of the polynomial coefficient estimates (only included if `full=False` and `cov=True`)

        Warns
        -----
        RankWarning
            The rank of the coefficient matrix in the least-squares fit is deficient.
            The warning is not raised with in-memory (not dask) data and `full=True`.

        See Also
        --------
        numpy.polyfit
        numpy.polyval
        xarray.polyval
        """
        variables = {}
        skipna_da = skipna

        x = get_clean_interp_index(self, dim, strict=False)
        xname = f"{self[dim].name}_"
        order = int(deg) + 1
        lhs = np.vander(x, order)

        if rcond is None:
            rcond = (
                x.shape[0] * np.core.finfo(x.dtype).eps  # type: ignore[attr-defined]
            )

        # Weights:
        if w is not None:
            if isinstance(w, Hashable):
                w = self.coords[w]
            w = np.asarray(w)
            if w.ndim != 1:
                raise TypeError("Expected a 1-d array for weights.")
            if w.shape[0] != lhs.shape[0]:
                raise TypeError(f"Expected w and {dim} to have the same length")
            lhs *= w[:, np.newaxis]

        # Scaling
        scale = np.sqrt((lhs * lhs).sum(axis=0))
        lhs /= scale

        degree_dim = utils.get_temp_dimname(self.dims, "degree")

        rank = np.linalg.matrix_rank(lhs)

        if full:
            rank = xr.DataArray(rank, name=xname + "matrix_rank")
            variables[rank.name] = rank
            _sing = np.linalg.svd(lhs, compute_uv=False)
            sing = xr.DataArray(
                _sing,
                dims=(degree_dim,),
                coords={degree_dim: np.arange(rank - 1, -1, -1)},
                name=xname + "singular_values",
            )
            variables[sing.name] = sing

        for name, da in self.data_vars.items():
            if dim not in da.dims:
                continue

            if is_duck_dask_array(da.data) and (
                rank != order or full or skipna is None
            ):
                # Current algorithm with dask and skipna=False neither supports
                # deficient ranks nor does it output the "full" info (issue dask/dask#6516)
                skipna_da = True
            elif skipna is None:
                skipna_da = bool(np.any(da.isnull()))

            dims_to_stack = [dimname for dimname in da.dims if dimname != dim]
            stacked_coords: dict[Hashable, DataArray] = {}
            if dims_to_stack:
                stacked_dim = utils.get_temp_dimname(dims_to_stack, "stacked")
                rhs = da.transpose(dim, *dims_to_stack).stack(
                    {stacked_dim: dims_to_stack}
                )
                stacked_coords = {stacked_dim: rhs[stacked_dim]}
                scale_da = scale[:, np.newaxis]
            else:
                rhs = da
                scale_da = scale

            if w is not None:
                rhs *= w[:, np.newaxis]

            with warnings.catch_warnings():
                if full:  # Copy np.polyfit behavior
                    warnings.simplefilter("ignore", np.RankWarning)
                else:  # Raise only once per variable
                    warnings.simplefilter("once", np.RankWarning)

                coeffs, residuals = duck_array_ops.least_squares(
                    lhs, rhs.data, rcond=rcond, skipna=skipna_da
                )

            if isinstance(name, str):
                name = f"{name}_"
            else:
                # Thus a ReprObject => polyfit was called on a DataArray
                name = ""

            coeffs = xr.DataArray(
                coeffs / scale_da,
                dims=[degree_dim] + list(stacked_coords.keys()),
                coords={degree_dim: np.arange(order)[::-1], **stacked_coords},
                name=name + "polyfit_coefficients",
            )
            if dims_to_stack:
                coeffs = coeffs.unstack(stacked_dim)
            variables[coeffs.name] = coeffs

            if full or (cov is True):
                residuals = xr.DataArray(
                    residuals if dims_to_stack else residuals.squeeze(),
                    dims=list(stacked_coords.keys()),
                    coords=stacked_coords,
                    name=name + "polyfit_residuals",
                )
                if dims_to_stack:
                    residuals = residuals.unstack(stacked_dim)
                variables[residuals.name] = residuals

            if cov:
                Vbase = np.linalg.inv(np.dot(lhs.T, lhs))
                Vbase /= np.outer(scale, scale)
                if cov == "unscaled":
                    fac = 1
                else:
                    if x.shape[0] <= order:
                        raise ValueError(
                            "The number of data points must exceed order to scale the covariance matrix."
                        )
                    fac = residuals / (x.shape[0] - order)
                covariance = xr.DataArray(Vbase, dims=("cov_i", "cov_j")) * fac
                variables[name + "polyfit_covariance"] = covariance

        return Dataset(data_vars=variables, attrs=self.attrs.copy())

    def pad(
        self,
        pad_width: Mapping[Any, int | tuple[int, int]] = None,
        mode: str = "constant",
        stat_length: int
        | tuple[int, int]
        | Mapping[Any, tuple[int, int]]
        | None = None,
        constant_values: (
            int | tuple[int, int] | Mapping[Any, tuple[int, int]] | None
        ) = None,
        end_values: int | tuple[int, int] | Mapping[Any, tuple[int, int]] | None = None,
        reflect_type: str = None,
        **pad_width_kwargs: Any,
    ) -> Dataset:
        """Pad this dataset along one or more dimensions.

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
            One of the following string values (taken from numpy docs).

            - constant: Pads with a constant value.
            - edge: Pads with the edge values of array.
            - linear_ramp: Pads with the linear ramp between end_value and the
              array edge value.
            - maximum: Pads with the maximum value of all or part of the
              vector along each axis.
            - mean: Pads with the mean value of all or part of the
              vector along each axis.
            - median: Pads with the median value of all or part of the
              vector along each axis.
            - minimum: Pads with the minimum value of all or part of the
              vector along each axis.
            - reflect: Pads with the reflection of the vector mirrored on
              the first and last values of the vector along each axis.
            - symmetric: Pads with the reflection of the vector mirrored
              along the edge of the array.
            - wrap: Pads with the wrap of the vector along the axis.
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
        padded : Dataset
            Dataset with the padded coordinates and data.

        See Also
        --------
        Dataset.shift, Dataset.roll, Dataset.bfill, Dataset.ffill, numpy.pad, dask.array.pad

        Notes
        -----
        By default when ``mode="constant"`` and ``constant_values=None``, integer types will be
        promoted to ``float`` and padded with ``np.nan``. To avoid type promotion
        specify ``constant_values=np.nan``

        Examples
        --------
        >>> ds = xr.Dataset({"foo": ("x", range(5))})
        >>> ds.pad(x=(1, 2))
        <xarray.Dataset>
        Dimensions:  (x: 8)
        Dimensions without coordinates: x
        Data variables:
            foo      (x) float64 nan 0.0 1.0 2.0 3.0 4.0 nan nan
        """
        pad_width = either_dict_or_kwargs(pad_width, pad_width_kwargs, "pad")

        if mode in ("edge", "reflect", "symmetric", "wrap"):
            coord_pad_mode = mode
            coord_pad_options = {
                "stat_length": stat_length,
                "constant_values": constant_values,
                "end_values": end_values,
                "reflect_type": reflect_type,
            }
        else:
            coord_pad_mode = "constant"
            coord_pad_options = {}

        variables = {}
        for name, var in self.variables.items():
            var_pad_width = {k: v for k, v in pad_width.items() if k in var.dims}
            if not var_pad_width:
                variables[name] = var
            elif name in self.data_vars:
                variables[name] = var.pad(
                    pad_width=var_pad_width,
                    mode=mode,
                    stat_length=stat_length,
                    constant_values=constant_values,
                    end_values=end_values,
                    reflect_type=reflect_type,
                )
            else:
                variables[name] = var.pad(
                    pad_width=var_pad_width,
                    mode=coord_pad_mode,
                    **coord_pad_options,  # type: ignore[arg-type]
                )

        return self._replace_vars_and_dims(variables)

    def idxmin(
        self,
        dim: Hashable = None,
        skipna: bool = None,
        fill_value: Any = dtypes.NA,
        keep_attrs: bool = None,
    ) -> Dataset:
        """Return the coordinate label of the minimum value along a dimension.

        Returns a new `Dataset` named after the dimension with the values of
        the coordinate labels along that dimension corresponding to minimum
        values along that dimension.

        In comparison to :py:meth:`~Dataset.argmin`, this returns the
        coordinate label while :py:meth:`~Dataset.argmin` returns the index.

        Parameters
        ----------
        dim : str, optional
            Dimension over which to apply `idxmin`.  This is optional for 1D
            variables, but required for variables with 2 or more dimensions.
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
        reduced : Dataset
            New `Dataset` object with `idxmin` applied to its data and the
            indicated dimension removed.

        See Also
        --------
        DataArray.idxmin, Dataset.idxmax, Dataset.min, Dataset.argmin

        Examples
        --------
        >>> array1 = xr.DataArray(
        ...     [0, 2, 1, 0, -2], dims="x", coords={"x": ["a", "b", "c", "d", "e"]}
        ... )
        >>> array2 = xr.DataArray(
        ...     [
        ...         [2.0, 1.0, 2.0, 0.0, -2.0],
        ...         [-4.0, np.NaN, 2.0, np.NaN, -2.0],
        ...         [np.NaN, np.NaN, 1.0, np.NaN, np.NaN],
        ...     ],
        ...     dims=["y", "x"],
        ...     coords={"y": [-1, 0, 1], "x": ["a", "b", "c", "d", "e"]},
        ... )
        >>> ds = xr.Dataset({"int": array1, "float": array2})
        >>> ds.min(dim="x")
        <xarray.Dataset>
        Dimensions:  (y: 3)
        Coordinates:
          * y        (y) int64 -1 0 1
        Data variables:
            int      int64 -2
            float    (y) float64 -2.0 -4.0 1.0
        >>> ds.argmin(dim="x")
        <xarray.Dataset>
        Dimensions:  (y: 3)
        Coordinates:
          * y        (y) int64 -1 0 1
        Data variables:
            int      int64 4
            float    (y) int64 4 0 2
        >>> ds.idxmin(dim="x")
        <xarray.Dataset>
        Dimensions:  (y: 3)
        Coordinates:
          * y        (y) int64 -1 0 1
        Data variables:
            int      <U1 'e'
            float    (y) object 'e' 'a' 'c'
        """
        return self.map(
            methodcaller(
                "idxmin",
                dim=dim,
                skipna=skipna,
                fill_value=fill_value,
                keep_attrs=keep_attrs,
            )
        )

    def idxmax(
        self,
        dim: Hashable = None,
        skipna: bool = None,
        fill_value: Any = dtypes.NA,
        keep_attrs: bool = None,
    ) -> Dataset:
        """Return the coordinate label of the maximum value along a dimension.

        Returns a new `Dataset` named after the dimension with the values of
        the coordinate labels along that dimension corresponding to maximum
        values along that dimension.

        In comparison to :py:meth:`~Dataset.argmax`, this returns the
        coordinate label while :py:meth:`~Dataset.argmax` returns the index.

        Parameters
        ----------
        dim : str, optional
            Dimension over which to apply `idxmax`.  This is optional for 1D
            variables, but required for variables with 2 or more dimensions.
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
        reduced : Dataset
            New `Dataset` object with `idxmax` applied to its data and the
            indicated dimension removed.

        See Also
        --------
        DataArray.idxmax, Dataset.idxmin, Dataset.max, Dataset.argmax

        Examples
        --------
        >>> array1 = xr.DataArray(
        ...     [0, 2, 1, 0, -2], dims="x", coords={"x": ["a", "b", "c", "d", "e"]}
        ... )
        >>> array2 = xr.DataArray(
        ...     [
        ...         [2.0, 1.0, 2.0, 0.0, -2.0],
        ...         [-4.0, np.NaN, 2.0, np.NaN, -2.0],
        ...         [np.NaN, np.NaN, 1.0, np.NaN, np.NaN],
        ...     ],
        ...     dims=["y", "x"],
        ...     coords={"y": [-1, 0, 1], "x": ["a", "b", "c", "d", "e"]},
        ... )
        >>> ds = xr.Dataset({"int": array1, "float": array2})
        >>> ds.max(dim="x")
        <xarray.Dataset>
        Dimensions:  (y: 3)
        Coordinates:
          * y        (y) int64 -1 0 1
        Data variables:
            int      int64 2
            float    (y) float64 2.0 2.0 1.0
        >>> ds.argmax(dim="x")
        <xarray.Dataset>
        Dimensions:  (y: 3)
        Coordinates:
          * y        (y) int64 -1 0 1
        Data variables:
            int      int64 1
            float    (y) int64 0 2 2
        >>> ds.idxmax(dim="x")
        <xarray.Dataset>
        Dimensions:  (y: 3)
        Coordinates:
          * y        (y) int64 -1 0 1
        Data variables:
            int      <U1 'b'
            float    (y) object 'a' 'c' 'c'
        """
        return self.map(
            methodcaller(
                "idxmax",
                dim=dim,
                skipna=skipna,
                fill_value=fill_value,
                keep_attrs=keep_attrs,
            )
        )

    def argmin(self, dim=None, **kwargs):
        """Indices of the minima of the member variables.

        If there are multiple minima, the indices of the first one found will be
        returned.

        Parameters
        ----------
        dim : str, optional
            The dimension over which to find the minimum. By default, finds minimum over
            all dimensions - for now returning an int for backward compatibility, but
            this is deprecated, in future will be an error, since DataArray.argmin will
            return a dict with indices for all dimensions, which does not make sense for
            a Dataset.
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
        result : Dataset

        See Also
        --------
        DataArray.argmin
        """
        if dim is None:
            warnings.warn(
                "Once the behaviour of DataArray.argmin() and Variable.argmin() without "
                "dim changes to return a dict of indices of each dimension, for "
                "consistency it will be an error to call Dataset.argmin() with no argument,"
                "since we don't return a dict of Datasets.",
                DeprecationWarning,
                stacklevel=2,
            )
        if (
            dim is None
            or (not isinstance(dim, Sequence) and dim is not ...)
            or isinstance(dim, str)
        ):
            # Return int index if single dimension is passed, and is not part of a
            # sequence
            argmin_func = getattr(duck_array_ops, "argmin")
            return self.reduce(argmin_func, dim=dim, **kwargs)
        else:
            raise ValueError(
                "When dim is a sequence or ..., DataArray.argmin() returns a dict. "
                "dicts cannot be contained in a Dataset, so cannot call "
                "Dataset.argmin() with a sequence or ... for dim"
            )

    def argmax(self, dim=None, **kwargs):
        """Indices of the maxima of the member variables.

        If there are multiple maxima, the indices of the first one found will be
        returned.

        Parameters
        ----------
        dim : str, optional
            The dimension over which to find the maximum. By default, finds maximum over
            all dimensions - for now returning an int for backward compatibility, but
            this is deprecated, in future will be an error, since DataArray.argmax will
            return a dict with indices for all dimensions, which does not make sense for
            a Dataset.
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
        result : Dataset

        See Also
        --------
        DataArray.argmax

        """
        if dim is None:
            warnings.warn(
                "Once the behaviour of DataArray.argmin() and Variable.argmin() without "
                "dim changes to return a dict of indices of each dimension, for "
                "consistency it will be an error to call Dataset.argmin() with no argument,"
                "since we don't return a dict of Datasets.",
                DeprecationWarning,
                stacklevel=2,
            )
        if (
            dim is None
            or (not isinstance(dim, Sequence) and dim is not ...)
            or isinstance(dim, str)
        ):
            # Return int index if single dimension is passed, and is not part of a
            # sequence
            argmax_func = getattr(duck_array_ops, "argmax")
            return self.reduce(argmax_func, dim=dim, **kwargs)
        else:
            raise ValueError(
                "When dim is a sequence or ..., DataArray.argmin() returns a dict. "
                "dicts cannot be contained in a Dataset, so cannot call "
                "Dataset.argmin() with a sequence or ... for dim"
            )

    def query(
        self,
        queries: Mapping[Any, Any] = None,
        parser: str = "pandas",
        engine: str = None,
        missing_dims: str = "raise",
        **queries_kwargs: Any,
    ) -> Dataset:
        """Return a new dataset with each array indexed along the specified
        dimension(s), where the indexers are given as strings containing
        Python expressions to be evaluated against the data variables in the
        dataset.

        Parameters
        ----------
        queries : dict, optional
            A dict with keys matching dimensions and values given by strings
            containing Python expressions to be evaluated against the data variables
            in the dataset. The expressions will be evaluated using the pandas
            eval() function, and can contain any valid Python expressions but cannot
            contain any Python statements.
        parser : {"pandas", "python"}, default: "pandas"
            The parser to use to construct the syntax tree from the expression.
            The default of 'pandas' parses code slightly different than standard
            Python. Alternatively, you can parse an expression using the 'python'
            parser to retain strict Python semantics.
        engine : {"python", "numexpr", None}, default: None
            The engine used to evaluate the expression. Supported engines are:

            - None: tries to use numexpr, falls back to python
            - "numexpr": evaluates expressions using numexpr
            - "python": performs operations as if you had eval’d in top level python

        missing_dims : {"raise", "warn", "ignore"}, default: "raise"
            What to do if dimensions that should be selected from are not present in the
            Dataset:

            - "raise": raise an exception
            - "warning": raise a warning, and ignore the missing dimensions
            - "ignore": ignore the missing dimensions

        **queries_kwargs : {dim: query, ...}, optional
            The keyword arguments form of ``queries``.
            One of queries or queries_kwargs must be provided.

        Returns
        -------
        obj : Dataset
            A new Dataset with the same contents as this dataset, except each
            array and dimension is indexed by the results of the appropriate
            queries.

        See Also
        --------
        Dataset.isel
        pandas.eval

        Examples
        --------
        >>> a = np.arange(0, 5, 1)
        >>> b = np.linspace(0, 1, 5)
        >>> ds = xr.Dataset({"a": ("x", a), "b": ("x", b)})
        >>> ds
        <xarray.Dataset>
        Dimensions:  (x: 5)
        Dimensions without coordinates: x
        Data variables:
            a        (x) int64 0 1 2 3 4
            b        (x) float64 0.0 0.25 0.5 0.75 1.0
        >>> ds.query(x="a > 2")
        <xarray.Dataset>
        Dimensions:  (x: 2)
        Dimensions without coordinates: x
        Data variables:
            a        (x) int64 3 4
            b        (x) float64 0.75 1.0
        """

        # allow queries to be given either as a dict or as kwargs
        queries = either_dict_or_kwargs(queries, queries_kwargs, "query")

        # check queries
        for dim, expr in queries.items():
            if not isinstance(expr, str):
                msg = f"expr for dim {dim} must be a string to be evaluated, {type(expr)} given"
                raise ValueError(msg)

        # evaluate the queries to create the indexers
        indexers = {
            dim: pd.eval(expr, resolvers=[self], parser=parser, engine=engine)
            for dim, expr in queries.items()
        }

        # apply the selection
        return self.isel(indexers, missing_dims=missing_dims)

    def curvefit(
        self,
        coords: str | DataArray | Iterable[str | DataArray],
        func: Callable[..., Any],
        reduce_dims: Hashable | Iterable[Hashable] = None,
        skipna: bool = True,
        p0: dict[str, Any] = None,
        bounds: dict[str, Any] = None,
        param_names: Sequence[str] = None,
        kwargs: dict[str, Any] = None,
    ):
        """
        Curve fitting optimization for arbitrary functions.

        Wraps `scipy.optimize.curve_fit` with `apply_ufunc`.

        Parameters
        ----------
        coords : hashable, DataArray, or sequence of hashable or DataArray
            Independent coordinate(s) over which to perform the curve fitting. Must share
            at least one dimension with the calling object. When fitting multi-dimensional
            functions, supply `coords` as a sequence in the same order as arguments in
            `func`. To fit along existing dimensions of the calling object, `coords` can
            also be specified as a str or sequence of strs.
        func : callable
            User specified function in the form `f(x, *params)` which returns a numpy
            array of length `len(x)`. `params` are the fittable parameters which are optimized
            by scipy curve_fit. `x` can also be specified as a sequence containing multiple
            coordinates, e.g. `f((x0, x1), *params)`.
        reduce_dims : hashable or sequence of hashable
            Additional dimension(s) over which to aggregate while fitting. For example,
            calling `ds.curvefit(coords='time', reduce_dims=['lat', 'lon'], ...)` will
            aggregate all lat and lon points and fit the specified function along the
            time dimension.
        skipna : bool, optional
            Whether to skip missing values when fitting. Default is True.
        p0 : dict-like, optional
            Optional dictionary of parameter names to initial guesses passed to the
            `curve_fit` `p0` arg. If none or only some parameters are passed, the rest will
            be assigned initial values following the default scipy behavior.
        bounds : dict-like, optional
            Optional dictionary of parameter names to bounding values passed to the
            `curve_fit` `bounds` arg. If none or only some parameters are passed, the rest
            will be unbounded following the default scipy behavior.
        param_names : sequence of hashable, optional
            Sequence of names for the fittable parameters of `func`. If not supplied,
            this will be automatically determined by arguments of `func`. `param_names`
            should be manually supplied when fitting a function that takes a variable
            number of parameters.
        **kwargs : optional
            Additional keyword arguments to passed to scipy curve_fit.

        Returns
        -------
        curvefit_results : Dataset
            A single dataset which contains:

            [var]_curvefit_coefficients
                The coefficients of the best fit.
            [var]_curvefit_covariance
                The covariance matrix of the coefficient estimates.

        See Also
        --------
        Dataset.polyfit
        scipy.optimize.curve_fit
        """
        from scipy.optimize import curve_fit

        if p0 is None:
            p0 = {}
        if bounds is None:
            bounds = {}
        if kwargs is None:
            kwargs = {}

        if not reduce_dims:
            reduce_dims_ = []
        elif isinstance(reduce_dims, str) or not isinstance(reduce_dims, Iterable):
            reduce_dims_ = [reduce_dims]
        else:
            reduce_dims_ = list(reduce_dims)

        if (
            isinstance(coords, str)
            or isinstance(coords, xr.DataArray)
            or not isinstance(coords, Iterable)
        ):
            coords = [coords]
        coords_ = [self[coord] if isinstance(coord, str) else coord for coord in coords]

        # Determine whether any coords are dims on self
        for coord in coords_:
            reduce_dims_ += [c for c in self.dims if coord.equals(self[c])]
        reduce_dims_ = list(set(reduce_dims_))
        preserved_dims = list(set(self.dims) - set(reduce_dims_))
        if not reduce_dims_:
            raise ValueError(
                "No arguments to `coords` were identified as a dimension on the calling "
                "object, and no dims were supplied to `reduce_dims`. This would result "
                "in fitting on scalar data."
            )

        # Broadcast all coords with each other
        coords_ = xr.broadcast(*coords_)
        coords_ = [
            coord.broadcast_like(self, exclude=preserved_dims) for coord in coords_
        ]

        params, func_args = _get_func_args(func, param_names)
        param_defaults, bounds_defaults = _initialize_curvefit_params(
            params, p0, bounds, func_args
        )
        n_params = len(params)
        kwargs.setdefault("p0", [param_defaults[p] for p in params])
        kwargs.setdefault(
            "bounds",
            [
                [bounds_defaults[p][0] for p in params],
                [bounds_defaults[p][1] for p in params],
            ],
        )

        def _wrapper(Y, *coords_, **kwargs):
            # Wrap curve_fit with raveled coordinates and pointwise NaN handling
            x = np.vstack([c.ravel() for c in coords_])
            y = Y.ravel()
            if skipna:
                mask = np.all([np.any(~np.isnan(x), axis=0), ~np.isnan(y)], axis=0)
                x = x[:, mask]
                y = y[mask]
                if not len(y):
                    popt = np.full([n_params], np.nan)
                    pcov = np.full([n_params, n_params], np.nan)
                    return popt, pcov
            x = np.squeeze(x)
            popt, pcov = curve_fit(func, x, y, **kwargs)
            return popt, pcov

        result = xr.Dataset()
        for name, da in self.data_vars.items():
            if name is xr.core.dataarray._THIS_ARRAY:
                name = ""
            else:
                name = f"{str(name)}_"

            popt, pcov = xr.apply_ufunc(
                _wrapper,
                da,
                *coords_,
                vectorize=True,
                dask="parallelized",
                input_core_dims=[reduce_dims_ for d in range(len(coords_) + 1)],
                output_core_dims=[["param"], ["cov_i", "cov_j"]],
                dask_gufunc_kwargs={
                    "output_sizes": {
                        "param": n_params,
                        "cov_i": n_params,
                        "cov_j": n_params,
                    },
                },
                output_dtypes=(np.float64, np.float64),
                exclude_dims=set(reduce_dims_),
                kwargs=kwargs,
            )
            result[name + "curvefit_coefficients"] = popt
            result[name + "curvefit_covariance"] = pcov

        result = result.assign_coords(
            {"param": params, "cov_i": params, "cov_j": params}
        )
        result.attrs = self.attrs.copy()

        return result

    def convert_calendar(
        self,
        calendar: str,
        dim: str = "time",
        align_on: str | None = None,
        missing: Any | None = None,
        use_cftime: bool | None = None,
    ) -> Dataset:
        """Convert the Dataset to another calendar.

        Only converts the individual timestamps, does not modify any data except
        in dropping invalid/surplus dates or inserting missing dates.

        If the source and target calendars are either no_leap, all_leap or a
        standard type, only the type of the time array is modified.
        When converting to a leap year from a non-leap year, the 29th of February
        is removed from the array. In the other direction the 29th of February
        will be missing in the output, unless `missing` is specified,
        in which case that value is inserted.

        For conversions involving `360_day` calendars, see Notes.

        This method is safe to use with sub-daily data as it doesn't touch the
        time part of the timestamps.

        Parameters
        ---------
        calendar : str
            The target calendar name.
        dim : str
            Name of the time coordinate.
        align_on : {None, 'date', 'year'}
            Must be specified when either source or target is a `360_day` calendar,
            ignored otherwise. See Notes.
        missing : Optional[any]
            By default, i.e. if the value is None, this method will simply attempt
            to convert the dates in the source calendar to the same dates in the
            target calendar, and drop any of those that are not possible to
            represent.  If a value is provided, a new time coordinate will be
            created in the target calendar with the same frequency as the original
            time coordinate; for any dates that are not present in the source, the
            data will be filled with this value.  Note that using this mode requires
            that the source data have an inferable frequency; for more information
            see :py:func:`xarray.infer_freq`.  For certain frequency, source, and
            target calendar combinations, this could result in many missing values, see notes.
        use_cftime : boolean, optional
            Whether to use cftime objects in the output, only used if `calendar`
            is one of {"proleptic_gregorian", "gregorian" or "standard"}.
            If True, the new time axis uses cftime objects.
            If None (default), it uses :py:class:`numpy.datetime64` values if the
            date range permits it, and :py:class:`cftime.datetime` objects if not.
            If False, it uses :py:class:`numpy.datetime64`  or fails.

        Returns
        -------
        Dataset
            Copy of the dataarray with the time coordinate converted to the
            target calendar. If 'missing' was None (default), invalid dates in
            the new calendar are dropped, but missing dates are not inserted.
            If `missing` was given, the new data is reindexed to have a time axis
            with the same frequency as the source, but in the new calendar; any
            missing datapoints are filled with `missing`.

        Notes
        -----
        Passing a value to `missing` is only usable if the source's time coordinate as an
        inferrable frequencies (see :py:func:`~xarray.infer_freq`) and is only appropriate
        if the target coordinate, generated from this frequency, has dates equivalent to the
        source. It is usually **not** appropriate to use this mode with:

        - Period-end frequencies : 'A', 'Y', 'Q' or 'M', in opposition to 'AS' 'YS', 'QS' and 'MS'
        - Sub-monthly frequencies that do not divide a day evenly : 'W', 'nD' where `N != 1`
            or 'mH' where 24 % m != 0).

        If one of the source or target calendars is `"360_day"`, `align_on` must
        be specified and two options are offered.

        - "year"
            The dates are translated according to their relative position in the year,
            ignoring their original month and day information, meaning that the
            missing/surplus days are added/removed at regular intervals.

            From a `360_day` to a standard calendar, the output will be missing the
            following dates (day of year in parentheses):

            To a leap year:
                January 31st (31), March 31st (91), June 1st (153), July 31st (213),
                September 31st (275) and November 30th (335).
            To a non-leap year:
                February 6th (36), April 19th (109), July 2nd (183),
                September 12th (255), November 25th (329).

            From a standard calendar to a `"360_day"`, the following dates in the
            source array will be dropped:

            From a leap year:
                January 31st (31), April 1st (92), June 1st (153), August 1st (214),
                September 31st (275), December 1st (336)
            From a non-leap year:
                February 6th (37), April 20th (110), July 2nd (183),
                September 13th (256), November 25th (329)

            This option is best used on daily and subdaily data.

        - "date"
            The month/day information is conserved and invalid dates are dropped
            from the output. This means that when converting from a `"360_day"` to a
            standard calendar, all 31st (Jan, March, May, July, August, October and
            December) will be missing as there is no equivalent dates in the
            `"360_day"` calendar and the 29th (on non-leap years) and 30th of February
            will be dropped as there are no equivalent dates in a standard calendar.

            This option is best used with data on a frequency coarser than daily.
        """
        return convert_calendar(
            self,
            calendar,
            dim=dim,
            align_on=align_on,
            missing=missing,
            use_cftime=use_cftime,
        )

    def interp_calendar(
        self,
        target: pd.DatetimeIndex | CFTimeIndex | DataArray,
        dim: str = "time",
    ) -> Dataset:
        """Interpolates the Dataset to another calendar based on decimal year measure.

        Each timestamp in `source` and `target` are first converted to their decimal
        year equivalent then `source` is interpolated on the target coordinate.
        The decimal year of a timestamp is its year plus its sub-year component
        converted to the fraction of its year. For example "2000-03-01 12:00" is
        2000.1653 in a standard calendar or 2000.16301 in a `"noleap"` calendar.

        This method should only be used when the time (HH:MM:SS) information of
        time coordinate is not important.

        Parameters
        ----------
        target: DataArray or DatetimeIndex or CFTimeIndex
            The target time coordinate of a valid dtype
            (np.datetime64 or cftime objects)
        dim : str
            The time coordinate name.

        Return
        ------
        DataArray
            The source interpolated on the decimal years of target,
        """
        return interp_calendar(self, target, dim=dim)
