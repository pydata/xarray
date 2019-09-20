try:
    import dask
    import dask.array
    from dask.highlevelgraph import HighLevelGraph

except ImportError:
    pass

import itertools
import numpy as np
import operator

from .dataarray import DataArray
from .dataset import Dataset


def _to_array(obj):
    if not isinstance(obj, Dataset):
        raise ValueError("Trying to convert DataArray to DataArray!")

    if len(obj.data_vars) > 1:
        raise ValueError(
            "Trying to convert Dataset with more than one variable to DataArray"
        )

    name = list(obj.data_vars)[0]
    # this should be easier
    da = obj.to_array().squeeze().drop("variable")
    da.name = name
    return da


def make_meta(obj):

    from dask.array.utils import meta_from_array

    if isinstance(obj, DataArray):
        to_array = True
        obj_array = obj.copy()
        obj = obj._to_temp_dataset()
    else:
        to_array = False

    if isinstance(obj, Dataset):
        meta = Dataset()
        for name, variable in obj.variables.items():
            if dask.is_dask_collection(variable):
                meta_obj = obj[name].data._meta
            else:
                meta_obj = meta_from_array(obj[name].data)
            meta[name] = DataArray(meta_obj, dims=obj[name].dims)
            # meta[name] = DataArray(obj[name].dims, meta_obj)
    else:
        meta = obj

    if isinstance(obj, Dataset):
        for coord_name in set(obj.coords) - set(obj.dims):
            meta = meta.set_coords(coord_name)

    if to_array:
        meta = obj_array._from_temp_dataset(meta)

    return meta


def infer_template(func, obj, *args, **kwargs):
    """ Infer return object by running the function on meta objects. """
    meta_args = []
    for arg in (obj,) + args:
        meta_args.append(make_meta(arg))

    try:
        template = func(*meta_args, **kwargs)
    except ValueError:
        raise ValueError("Cannot infer object returned by user-provided function.")

    return template


def make_dict(x):
    # Dataset.to_dict() is too complicated
    # maps variable name to numpy array
    if isinstance(x, DataArray):
        x = x._to_temp_dataset()

    to_return = dict()
    for var in x.variables:
        # if var not in x:
        #    raise ValueError("Variable %r not found in returned object." % var)
        to_return[var] = x[var].values

    return to_return


def map_blocks(func, obj, args=[], kwargs={}):
    """
    Apply a function to each chunk of a DataArray or Dataset. This function is experimental
    and its signature may change.

    Parameters
    ----------
    func: callable
        User-provided function that should accept DataArrays corresponding to one chunk.
        The function will be run on a small piece of data that looks like 'obj' to determine
        properties of the returned object such as dtype, variable names,
        new dimensions and new indexes (if any).

        This function must
        - return either a DataArray or a Dataset

        This function cannot
        - change size of existing dimensions.
        - add new chunked dimensions.

    obj: DataArray, Dataset
        Chunks of this object will be provided to 'func'. The function must not change
        shape of the provided DataArray.
    args: list
        Passed on to func after unpacking. xarray objects, if any, will not be split by chunks.
    kwargs: dict
        Passed on to func after unpacking. xarray objects, if any, will not be split by chunks.

    Returns
    -------
    DataArray or Dataset

    Notes
    -----

    This function is designed to work with dask-backed xarray objects. See apply_ufunc for
    a similar function that works with numpy arrays.

    See Also
    --------
    dask.array.map_blocks, xarray.apply_ufunc
    """

    def _wrapper(func, obj, to_array, args, kwargs):
        if to_array:
            obj = _to_array(obj)

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

    if not isinstance(args, list):
        raise ValueError("args must be a list.")

    if not isinstance(kwargs, dict):
        raise ValueError("kwargs must be a dictionary.")

    if not dask.is_dask_collection(obj):
        raise ValueError(
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
    else:
        raise ValueError(
            "Function must return an xarray DataArray or Dataset. Instead it returned %r"
            % type(template)
        )

    # If two different variables have different chunking along the same dim
    # fix that by "unifying chunks"
    dataset = dataset.unify_chunks()
    input_chunks = dataset.chunks

    # TODO: add a test that fails when template and dataset are switched
    indexes = dict(template.indexes)
    indexes.update(dataset.indexes)

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
                subsetter = dict()
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
            # cannot tokenize "name" because the hash of ReprObject (<this-array>)
            # is a function of its value. This happens when the user function does not
            # set a name on the returned DataArray
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

    result = Dataset()
    # a quicker way to assign indexes?
    # indexes need to be known
    # otherwise compute is called when DataArray is created
    for name in template.dims:
        result[name] = indexes[name]
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
        result = _to_array(result)

    return result
