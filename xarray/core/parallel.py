try:
    import dask
    import dask.array
    from dask.highlevelgraph import HighLevelGraph

except ImportError:
    pass

import numpy as np
import operator

from .dataarray import DataArray
from .dataset import Dataset


def _to_dataset(obj):
    if obj.name is not None:
        dataset = obj.to_dataset()
    else:
        dataset = obj._to_temp_dataset()

    return dataset


def _to_array(obj):
    if not isinstance(obj, Dataset):
        raise ValueError("Trying to convert DataArray to DataArray!")

    if len(obj.data_vars) > 1:
        raise ValueError(
            "Trying to convert Dataset with more than one variable to DataArray"
        )

    name = list(obj.data_vars)[0]
    da = obj.to_array().squeeze().drop("variable")
    da.name = name
    return da


def make_meta(obj):
    if isinstance(obj, DataArray):
        meta = DataArray(obj.data._meta, dims=obj.dims)

    if isinstance(obj, Dataset):
        meta = Dataset()
        for name, variable in obj.variables.items():
            if dask.is_dask_collection(variable):
                meta[name] = DataArray(obj[name].data._meta, dims=obj[name].dims)
            else:
                continue

    return meta


def _make_dict(x):
    # Dataset.to_dict() is too complicated
    # maps variable name to numpy array
    if isinstance(x, DataArray):
        x = _to_dataset(x)

    to_return = dict()
    for var in x.variables:
        # if var not in x:
        #    raise ValueError("Variable %r not found in returned object." % var)
        to_return[var] = x[var].values

    return to_return


def map_blocks(func, obj, *args, template=None, **kwargs):
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
    template:
        template object representing result
    kwargs:
        Passed on to func.


    Returns
    -------
    DataArray or Dataset

    See Also
    --------
    dask.array.map_blocks
    """

    import itertools

    def _wrapper(func, obj, to_array, args, kwargs):
        if to_array:
            # this should be easier
            obj = _to_array(obj)

        result = func(obj, *args, **kwargs)

        # if isinstance(result, DataArray):
        #    if result.shape != obj.shape:
        #        raise ValueError("Result does not have the same shape as input.")

        to_return = _make_dict(result)

        return to_return

    if not dask.is_dask_collection(obj):
        raise ValueError(
            "map_blocks can only be used with dask-backed DataArrays. Use .chunk() to convert to a Dask array."
        )

    if isinstance(obj, DataArray):
        dataset = _to_dataset(obj)
        input_is_array = True
    else:
        dataset = obj
        input_is_array = False

    dataset_dims = list(dataset.dims)

    # infer template / meta information here
    if template is None:
        try:
            meta = make_meta(obj)
            result_meta = func(meta, *args, **kwargs)
        except ValueError:
            raise ValueError("Cannot infer return type from user-provided function.")

        template = result_meta

    if isinstance(template, DataArray):
        result_is_array = True
        template = _to_dataset(template)
    else:
        result_is_array = False

    template_vars = list(template.variables)

    graph = {}
    gname = "%s-%s" % (dask.utils.funcname(func), dask.base.tokenize(dataset))

    # If two different variables have different chunking along the same dim
    # .chunks will raise an error.
    chunks = dataset.chunks
    # map dims to list of chunk indexes
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
        for var in template_vars:
            var_dims = template.variables[var].dims
            gname_l = gname + dask.base.tokenize(var)
            var_key_map[var] = gname_l

            key = (gname_l,)
            for dim in var_dims:
                if dim in chunk_index_dict:
                    key += (chunk_index_dict[dim],)
                else:
                    # unchunked dimensions in the input have one chunk in the result
                    key += (0,)

            # this is a list [name, values, dims, attrs]
            graph[key] = (operator.getitem, from_wrapper, var)

    graph = HighLevelGraph.from_collections(name, graph, dependencies=[dataset])

    result = Dataset()
    for var, key in var_key_map.items():
        # indexes need to be known
        # otherwise compute is called when DataArray is created
        if var in template.indexes:
            result[var] = template[var]
            continue

        name = var
        dims = template[var].dims
        chunks = [
            template.chunks[dim] if dim in template.chunks else (len(template[dim]),)
            for dim in dims
        ]
        dtype = template[var].dtype

        data = dask.array.Array(graph, name=key, chunks=chunks, dtype=dtype)
        result[name] = DataArray(data=data, dims=dims, name=name)

    result = Dataset(result)
    if result_is_array:
        result = _to_array(result)

    return result
