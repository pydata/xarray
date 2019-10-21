try:
    import dask
    import dask.array
    from dask.highlevelgraph import HighLevelGraph
    from .dask_array_compat import meta_from_array

except ImportError:
    pass

import itertools
import operator
from typing import (
    Any,
    Callable,
    Dict,
    Hashable,
    Mapping,
    Sequence,
    Tuple,
    TypeVar,
    Union,
)

import numpy as np

from .dataarray import DataArray
from .dataset import Dataset

T_DSorDA = TypeVar("T_DSorDA", DataArray, Dataset)


def dataset_to_dataarray(obj: Dataset) -> DataArray:
    if not isinstance(obj, Dataset):
        raise TypeError("Expected Dataset, got %s" % type(obj))

    if len(obj.data_vars) > 1:
        raise TypeError(
            "Trying to convert Dataset with more than one data variable to DataArray"
        )

    return next(iter(obj.data_vars.values()))


def make_meta(obj):
    """If obj is a DataArray or Dataset, return a new object of the same type and with
    the same variables and dtypes, but where all variables have size 0 and numpy
    backend.
    If obj is neither a DataArray nor Dataset, return it unaltered.
    """
    if isinstance(obj, DataArray):
        obj_array = obj
        obj = obj._to_temp_dataset()
    elif isinstance(obj, Dataset):
        obj_array = None
    else:
        return obj

    meta = Dataset()
    for name, variable in obj.variables.items():
        meta_obj = meta_from_array(variable.data, ndim=variable.ndim)
        meta[name] = (variable.dims, meta_obj, variable.attrs)
    meta.attrs = obj.attrs
    meta = meta.set_coords(obj.coords)

    if obj_array is not None:
        return obj_array._from_temp_dataset(meta)
    return meta


def infer_template(
    func: Callable[..., T_DSorDA], obj: Union[DataArray, Dataset], *args, **kwargs
) -> T_DSorDA:
    """Infer return object by running the function on meta objects.
    """
    meta_args = [make_meta(arg) for arg in (obj,) + args]

    try:
        template = func(*meta_args, **kwargs)
    except Exception as e:
        raise Exception(
            "Cannot infer object returned from running user provided function."
        ) from e

    if not isinstance(template, (Dataset, DataArray)):
        raise TypeError(
            "Function must return an xarray DataArray or Dataset. Instead it returned "
            f"{type(template)}"
        )

    return template


def make_dict(x: Union[DataArray, Dataset]) -> Dict[Hashable, Any]:
    """Map variable name to numpy(-like) data
    (Dataset.to_dict() is too complicated).
    """
    if isinstance(x, DataArray):
        x = x._to_temp_dataset()

    return {k: v.data for k, v in x.variables.items()}


def map_blocks(
    func: Callable[..., T_DSorDA],
    obj: Union[DataArray, Dataset],
    args: Sequence[Any] = (),
    kwargs: Mapping[str, Any] = None,
) -> T_DSorDA:
    """Apply a function to each chunk of a DataArray or Dataset. This function is
    experimental and its signature may change.

    Parameters
    ----------
    func: callable
        User-provided function that accepts a DataArray or Dataset as its first
        parameter. The function will receive a subset of 'obj' (see below),
        corresponding to one chunk along each chunked dimension. ``func`` will be
        executed as ``func(obj_subset, *args, **kwargs)``.

        The function will be first run on mocked-up data, that looks like 'obj' but
        has sizes 0, to determine properties of the returned object such as dtype,
        variable names, new dimensions and new indexes (if any).

        This function must return either a single DataArray or a single Dataset.

        This function cannot change size of existing dimensions, or add new chunked
        dimensions.
    obj: DataArray, Dataset
        Passed to the function as its first argument, one dask chunk at a time.
    args: Sequence
        Passed verbatim to func after unpacking, after the sliced obj. xarray objects,
        if any, will not be split by chunks. Passing dask collections is not allowed.
    kwargs: Mapping
        Passed verbatim to func after unpacking. xarray objects, if any, will not be
        split by chunks. Passing dask collections is not allowed.

    Returns
    -------
    A single DataArray or Dataset with dask backend, reassembled from the outputs of the
    function.

    Notes
    -----
    This function is designed for when one needs to manipulate a whole xarray object
    within each chunk. In the more common case where one can work on numpy arrays, it is
    recommended to use apply_ufunc.

    If none of the variables in obj is backed by dask, calling this function is
    equivalent to calling ``func(obj, *args, **kwargs)``.

    See Also
    --------
    dask.array.map_blocks, xarray.apply_ufunc, xarray.Dataset.map_blocks,
    xarray.DataArray.map_blocks
    """

    def _wrapper(func, obj, to_array, args, kwargs):
        if to_array:
            obj = dataset_to_dataarray(obj)

        result = func(obj, *args, **kwargs)

        for name, index in result.indexes.items():
            if name in obj.indexes:
                if len(index) != len(obj.indexes[name]):
                    raise ValueError(
                        "Length of the %r dimension has changed. This is not allowed."
                        % name
                    )

        return make_dict(result)

    if not isinstance(args, Sequence):
        raise TypeError("args must be a sequence (for example, a list or tuple).")
    if kwargs is None:
        kwargs = {}
    elif not isinstance(kwargs, Mapping):
        raise TypeError("kwargs must be a mapping (for example, a dict)")

    for value in list(args) + list(kwargs.values()):
        if dask.is_dask_collection(value):
            raise TypeError(
                "Cannot pass dask collections in args or kwargs yet. Please compute or "
                "load values before passing to map_blocks."
            )

    if not dask.is_dask_collection(obj):
        return func(obj, *args, **kwargs)

    if isinstance(obj, DataArray):
        # only using _to_temp_dataset would break
        # func = lambda x: x.to_dataset()
        # since that relies on preserving name.
        if obj.name is None:
            dataset = obj._to_temp_dataset()
        else:
            dataset = obj.to_dataset()
        input_is_array = True
    else:
        dataset = obj
        input_is_array = False

    input_chunks = dataset.chunks

    template: Union[DataArray, Dataset] = infer_template(func, obj, *args, **kwargs)
    if isinstance(template, DataArray):
        result_is_array = True
        template_name = template.name
        template = template._to_temp_dataset()
    elif isinstance(template, Dataset):
        result_is_array = False
    else:
        raise TypeError(
            f"func output must be DataArray or Dataset; got {type(template)}"
        )

    template_indexes = set(template.indexes)
    dataset_indexes = set(dataset.indexes)
    preserved_indexes = template_indexes & dataset_indexes
    new_indexes = template_indexes - dataset_indexes
    indexes = {dim: dataset.indexes[dim] for dim in preserved_indexes}
    indexes.update({k: template.indexes[k] for k in new_indexes})

    graph: Dict[Any, Any] = {}
    gname = "{}-{}".format(
        dask.utils.funcname(func), dask.base.tokenize(dataset, args, kwargs)
    )

    # map dims to list of chunk indexes
    ichunk = {dim: range(len(chunks_v)) for dim, chunks_v in input_chunks.items()}
    # mapping from chunk index to slice bounds
    chunk_index_bounds = {
        dim: np.cumsum((0,) + chunks_v) for dim, chunks_v in input_chunks.items()
    }

    # iterate over all possible chunk combinations
    for v in itertools.product(*ichunk.values()):
        chunk_index_dict = dict(zip(dataset.dims, v))

        # this will become [[name1, variable1],
        #                   [name2, variable2],
        #                   ...]
        # which is passed to dict and then to Dataset
        data_vars = []
        coords = []

        for name, variable in dataset.variables.items():
            # make a task that creates tuple of (dims, chunk)
            if dask.is_dask_collection(variable.data):
                # recursively index into dask_keys nested list to get chunk
                chunk = variable.__dask_keys__()
                for dim in variable.dims:
                    chunk = chunk[chunk_index_dict[dim]]

                chunk_variable_task = (f"{gname}-{chunk[0]}",) + v
                graph[chunk_variable_task] = (
                    tuple,
                    [variable.dims, chunk, variable.attrs],
                )
            else:
                # non-dask array with possibly chunked dimensions
                # index into variable appropriately
                subsetter = {}
                for dim in variable.dims:
                    if dim in chunk_index_dict:
                        which_chunk = chunk_index_dict[dim]
                        subsetter[dim] = slice(
                            chunk_index_bounds[dim][which_chunk],
                            chunk_index_bounds[dim][which_chunk + 1],
                        )

                subset = variable.isel(subsetter)
                chunk_variable_task = (
                    "{}-{}".format(gname, dask.base.tokenize(subset)),
                ) + v
                graph[chunk_variable_task] = (
                    tuple,
                    [subset.dims, subset, subset.attrs],
                )

            # this task creates dict mapping variable name to above tuple
            if name in dataset._coord_names:
                coords.append([name, chunk_variable_task])
            else:
                data_vars.append([name, chunk_variable_task])

        from_wrapper = (gname,) + v
        graph[from_wrapper] = (
            _wrapper,
            func,
            (Dataset, (dict, data_vars), (dict, coords), dataset.attrs),
            input_is_array,
            args,
            kwargs,
        )

        # mapping from variable name to dask graph key
        var_key_map: Dict[Hashable, str] = {}
        for name, variable in template.variables.items():
            if name in indexes:
                continue
            gname_l = f"{gname}-{name}"
            var_key_map[name] = gname_l

            key: Tuple[Any, ...] = (gname_l,)
            for dim in variable.dims:
                if dim in chunk_index_dict:
                    key += (chunk_index_dict[dim],)
                else:
                    # unchunked dimensions in the input have one chunk in the result
                    key += (0,)

            graph[key] = (operator.getitem, from_wrapper, name)

    graph = HighLevelGraph.from_collections(gname, graph, dependencies=[dataset])

    result = Dataset(coords=indexes, attrs=template.attrs)
    for name, gname_l in var_key_map.items():
        dims = template[name].dims
        var_chunks = []
        for dim in dims:
            if dim in input_chunks:
                var_chunks.append(input_chunks[dim])
            elif dim in indexes:
                var_chunks.append((len(indexes[dim]),))

        data = dask.array.Array(
            graph, name=gname_l, chunks=var_chunks, dtype=template[name].dtype
        )
        result[name] = (dims, data, template[name].attrs)

    result = result.set_coords(template._coord_names)

    if result_is_array:
        da = dataset_to_dataarray(result)
        da.name = template_name
        return da  # type: ignore
    return result  # type: ignore
