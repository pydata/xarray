try:
    import dask
    import dask.array
    from dask.highlevelgraph import HighLevelGraph

except ImportError:
    pass

import itertools
import numpy as np

from .dataarray import DataArray
from .dataset import Dataset


def map_blocks(func, obj, *args, dtype=None, **kwargs):
    """
    Apply a function to each chunk of a DataArray or Dataset.

    Parameters
    ----------
    func: callable
        User-provided function that should accept DataArrays corresponding to one chunk.
    obj: DataArray, Dataset
        Chunks of this object will be provided to 'func'. The function must not change
        shape of the provided DataArray.
    args:
        Passed on to func.
    dtype:
        dtype of the DataArray returned by func.
    kwargs:
        Passed on to func.


    Returns
    -------
    DataArray

    See Also
    --------
    dask.array.map_blocks
    """

    def _wrapper(func, obj, to_array, args, kwargs):
        if to_array:
            # this should be easier
            obj = obj.to_array().squeeze().drop("variable")

        result = func(obj, *args, **kwargs)

        if not isinstance(result, type(obj)):
            raise ValueError("Result is not the same type as input.")
        if result.shape != obj.shape:
            raise ValueError("Result does not have the same shape as input.")

        return result

    if not isinstance(obj, DataArray):
        raise ValueError("map_blocks can only be used with DataArrays at present.")

    if not dask.is_dask_collection(obj):
        raise ValueError(
            "map_blocks can only be used with dask-backed DataArrays. Use .chunk() to convert to a Dask array."
        )

    try:
        meta_array = DataArray(obj.data._meta, dims=obj.dims)
        result_meta = func(meta_array, *args, **kwargs)
        if dtype is None:
            dtype = result_meta.dtype
    except ValueError:
        raise ValueError("Cannot infer return type from user-provided function.")

    if isinstance(obj, DataArray):
        dataset = obj._to_temp_dataset()
        to_array = True
    else:
        dataset = obj
        to_array = False

    dataset_dims = list(dataset.dims)

    graph = {}
    gname = "map-%s-%s" % (dask.utils.funcname(func), dask.base.tokenize(dataset))

    # map dims to list of chunk indexes
    # If two different variables have different chunking along the same dim
    # .chunks will raise an error.
    chunks = dataset.chunks
    ichunk = {dim: range(len(chunks[dim])) for dim in chunks}
    # mapping from chunk index to slice bounds
    chunk_index_bounds = {dim: np.cumsum((0,) + chunks[dim]) for dim in chunks}

    # iterate over all possible chunk combinations
    for v in itertools.product(*ichunk.values()):
        chunk_index_dict = dict(zip(dataset_dims, v))

        # this will become [[name1, variable1],
        #                   [name2, variable2],
        #                   ...]
        # which is passed to dict and then to Dataset
        data_vars = []
        coords = []

        for name, variable in dataset.variables.items():
            # make a task that creates tuple of (dims, chunk)
            if dask.is_dask_collection(variable.data):
                var_dask_keys = variable.__dask_keys__()

                # recursively index into dask_keys nested list to get chunk
                chunk = var_dask_keys
                for dim in variable.dims:
                    chunk = chunk[chunk_index_dict[dim]]

                task_name = ("tuple-" + dask.base.tokenize(chunk),) + v
                graph[task_name] = (tuple, [variable.dims, chunk])
            else:
                # numpy array with possibly chunked dimensions
                # index into variable appropriately
                subsetter = dict()
                for dim in variable.dims:
                    if dim in chunk_index_dict:
                        which_chunk = chunk_index_dict[dim]
                        subsetter[dim] = slice(
                            chunk_index_bounds[dim][which_chunk],
                            chunk_index_bounds[dim][which_chunk + 1],
                        )

                subset = variable.isel(subsetter)
                task_name = (name + dask.base.tokenize(subset),) + v
                graph[task_name] = (tuple, [subset.dims, subset])

            # this task creates dict mapping variable name to above tuple
            if name in dataset.data_vars:
                data_vars.append([name, task_name])
            if name in dataset.coords:
                coords.append([name, task_name])

        graph[(gname,) + v] = (
            _wrapper,
            func,
            (Dataset, (dict, data_vars), (dict, coords), dataset.attrs),
            to_array,
            args,
            kwargs,
        )

    final_graph = HighLevelGraph.from_collections(name, graph, dependencies=[dataset])

    if isinstance(obj, DataArray):
        result = DataArray(
            dask.array.Array(
                final_graph, name=gname, chunks=obj.data.chunks, meta=result_meta
            ),
            dims=obj.dims,
            coords=obj.coords,
        )

    return result
