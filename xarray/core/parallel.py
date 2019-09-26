try:
    import dask
    import dask.array
    from dask.highlevelgraph import HighLevelGraph
    from dask.array.utils import meta_from_array

except ImportError:
    pass

import itertools
import numpy as np
import operator

from .dataarray import DataArray
from .dataset import Dataset

from typing import Sequence, Mapping


def dataset_to_dataarray(obj: Dataset) -> DataArray:
    if not isinstance(obj, Dataset):
        raise TypeError("Expected Dataset, got %s" % type(obj))

    if len(obj.data_vars) > 1:
        raise TypeError(
            "Trying to convert Dataset with more than one data variable to DataArray"
        )

    return next(iter(obj.data_vars.values()))


def make_meta(obj):

    if isinstance(obj, DataArray):
        to_array = True
        obj_array = obj.copy()
        obj = obj._to_temp_dataset()
    else:
        to_array = False

    if isinstance(obj, Dataset):
        meta = Dataset()
        for name, variable in obj.variables.items():
            meta_obj = meta_from_array(variable.data)
            meta[name] = (variable.dims, meta_obj)
    else:
        meta = obj

    if isinstance(obj, Dataset):
        meta = meta.set_coords(obj.coords)

    if to_array:
        meta = obj_array._from_temp_dataset(meta)

    return meta


def infer_template(func, obj, *args, **kwargs):
    """ Infer return object by running the function on meta objects. """
    meta_args = [make_meta(arg) for arg in (obj,) + args]

    try:
        template = func(*meta_args, **kwargs)
    except Exception as e:
        raise Exception(
            "Cannot infer object returned from running user provided function."
        ) from e

    if not isinstance(template, (Dataset, DataArray)):
        raise TypeError(
            "Function must return an xarray DataArray or Dataset. Instead it returned %r"
            % type(template)
        )

    return template


def make_dict(x):
    # Dataset.to_dict() is too complicated
    # maps variable name to numpy array
    if isinstance(x, DataArray):
        x = x._to_temp_dataset()

    return {k: v.data for k, v in x.variables.items()}


def map_blocks(func, obj, args=(), kwargs=None):
    """
    Apply a function to each chunk of a DataArray or Dataset. This function is experimental
    and its signature may change.

    Parameters
    ----------
    func: callable
        User-provided function that should accept xarray objects.
        This function will receive a subset of this dataset, corresponding to one chunk along
        each chunked dimension.
        To determine properties of the returned object such as type (DataArray or Dataset), dtypes,
        and new/removed dimensions and/or variables, the function will be run on dummy data
        with the same variables, dimension names, and data types as this DataArray, but zero-sized
        dimensions.

        This function must
        - return either a single DataArray or a single Dataset

        This function cannot
        - change size of existing dimensions.
        - add new chunked dimensions.

        This function should work with whole xarray objects. If your function can be applied
        to numpy or dask arrays (e.g. it doesn't need additional metadata such as dimension names,
        variable names, etc.), you should consider using :py:func:`~xarray.apply_ufunc` instead.
    obj: DataArray, Dataset
        Chunks of this object will be provided to 'func'.
    args: list
        Passed on to func after unpacking. xarray objects, if any, will not be split by chunks.
    kwargs: dict
        Passed on to func after unpacking. xarray objects, if any, will not be split by chunks.

    Returns
    -------
    A single DataArray or Dataset

    Notes
    -----

    This function is designed to work with dask-backed xarray objects. See apply_ufunc for
    a similar function that works with numpy arrays.

    See Also
    --------
    dask.array.map_blocks, xarray.apply_ufunc, xarray.Dataset.map_blocks, xarray.DataArray.map_blocks
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

        to_return = make_dict(result)

        return to_return

    if not isinstance(args, Sequence):
        raise TypeError("args must be a sequence (for example, a list).")

    if kwargs is None:
        kwargs = {}

    elif not isinstance(kwargs, Mapping):
        raise TypeError("kwargs must be a mapping (for example, a dict)")

    if not dask.is_dask_collection(obj):
        raise TypeError(
            "map_blocks can only be used with dask-backed DataArrays. Use .chunk() to convert to a Dask array."
        )

    if isinstance(obj, DataArray):
        dataset = obj._to_temp_dataset()
        input_is_array = True
    else:
        dataset = obj
        input_is_array = False

    template = infer_template(func, obj, *args, **kwargs)
    if isinstance(template, DataArray):
        result_is_array = True
        template = template._to_temp_dataset()
    elif isinstance(template, Dataset):
        result_is_array = False

    input_chunks = dataset.chunks
    indexes = {dim: dataset.indexes[dim] for dim in template.dims}
    graph = {}
    gname = "%s-%s" % (dask.utils.funcname(func), dask.base.tokenize(dataset))

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

                chunk_variable_task = ("tuple-" + dask.base.tokenize(chunk),) + v
                graph[chunk_variable_task] = (tuple, [variable.dims, chunk])
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
                chunk_variable_task = (name + dask.base.tokenize(subset),) + v
                graph[chunk_variable_task] = (tuple, [subset.dims, subset])

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
        var_key_map = {}
        for name, variable in template.variables.items():
            if name in indexes:
                continue
            gname_l = "%s-%s" % (gname, name)
            var_key_map[name] = gname_l

            key = (gname_l,)
            for dim in variable.dims:
                if dim in chunk_index_dict:
                    key += (chunk_index_dict[dim],)
                else:
                    # unchunked dimensions in the input have one chunk in the result
                    key += (0,)

            graph[key] = (operator.getitem, from_wrapper, name)

    graph = HighLevelGraph.from_collections(name, graph, dependencies=[dataset])

    result = Dataset(coords=indexes)
    for name, key in var_key_map.items():
        dims = template[name].dims
        var_chunks = []
        for dim in dims:
            if dim in input_chunks:
                var_chunks.append(input_chunks[dim])
            elif dim in indexes:
                var_chunks.append((len(indexes[dim]),))

        data = dask.array.Array(
            graph, name=key, chunks=var_chunks, dtype=template[name].dtype
        )
        result[name] = (dims, data)

    result = result.set_coords(template._coord_names)

    if result_is_array:
        result = dataset_to_dataarray(result)

    return result
