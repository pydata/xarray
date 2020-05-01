try:
    import dask
    import dask.array
    from dask.highlevelgraph import HighLevelGraph
    from .dask_array_compat import meta_from_array

except ImportError:
    pass

import collections
import itertools
import operator
from typing import (
    Any,
    Callable,
    DefaultDict,
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


def check_result_variables(
    result: Union[DataArray, Dataset], expected: Mapping[str, Any], kind: str
):

    if kind == "coords":
        nice_str = "coordinate"
    elif kind == "data_vars":
        nice_str = "data"

    # check that coords and data variables are as expected
    missing = expected[kind] - set(getattr(result, kind))
    if missing:
        raise ValueError(
            "Result from applying user function does not contain "
            f"{nice_str} variables {missing}."
        )
    extra = set(getattr(result, kind)) - expected[kind]
    if extra:
        raise ValueError(
            "Result from applying user function has unexpected "
            f"{nice_str} variables {extra}."
        )


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
            "Cannot infer object returned from running user provided function. "
            "Please supply the 'template' kwarg to map_blocks."
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


def _get_chunk_slicer(dim: Hashable, chunk_index: Mapping, chunk_bounds: Mapping):
    if dim in chunk_index:
        which_chunk = chunk_index[dim]
        return slice(chunk_bounds[dim][which_chunk], chunk_bounds[dim][which_chunk + 1])
    return slice(None)


def map_blocks(
    func: Callable[..., T_DSorDA],
    obj: Union[DataArray, Dataset],
    args: Sequence[Any] = (),
    kwargs: Mapping[str, Any] = None,
    template: Union[DataArray, Dataset] = None,
) -> T_DSorDA:
    """Apply a function to each block of a DataArray or Dataset.

    .. warning::
        This function is experimental and its signature may change.

    Parameters
    ----------
    func: callable
        User-provided function that accepts a DataArray or Dataset as its first
        parameter. The function will receive a subset of 'obj' (see below),
        corresponding to one chunk along each chunked dimension. ``func`` will be
        executed as ``func(obj_subset, *args, **kwargs)``.

        This function must return either a single DataArray or a single Dataset.

        This function cannot add a new chunked dimension.

    obj: DataArray, Dataset
        Passed to the function as its first argument, one dask chunk at a time.
    args: Sequence
        Passed verbatim to func after unpacking, after the sliced obj. xarray objects,
        if any, will not be split by chunks. Passing dask collections is not allowed.
    kwargs: Mapping
        Passed verbatim to func after unpacking. xarray objects, if any, will not be
        split by chunks. Passing dask collections is not allowed.
    template: (optional) DataArray, Dataset
        xarray object representing the final result after compute is called. If not provided,
        the function will be first run on mocked-up data, that looks like 'obj' but
        has sizes 0, to determine properties of the returned object such as dtype,
        variable names, new dimensions and new indexes (if any).
        'template' must be provided if the function changes the size of existing dimensions.

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

    Examples
    --------

    Calculate an anomaly from climatology using ``.groupby()``. Using
    ``xr.map_blocks()`` allows for parallel operations with knowledge of ``xarray``,
    its indices, and its methods like ``.groupby()``.

    >>> def calculate_anomaly(da, groupby_type="time.month"):
    ...     # Necessary workaround to xarray's check with zero dimensions
    ...     # https://github.com/pydata/xarray/issues/3575
    ...     if sum(da.shape) == 0:
    ...         return da
    ...     gb = da.groupby(groupby_type)
    ...     clim = gb.mean(dim="time")
    ...     return gb - clim
    >>> time = xr.cftime_range("1990-01", "1992-01", freq="M")
    >>> np.random.seed(123)
    >>> array = xr.DataArray(
    ...     np.random.rand(len(time)), dims="time", coords=[time]
    ... ).chunk()
    >>> xr.map_blocks(calculate_anomaly, array).compute()
    <xarray.DataArray (time: 24)>
    array([ 0.12894847,  0.11323072, -0.0855964 , -0.09334032,  0.26848862,
            0.12382735,  0.22460641,  0.07650108, -0.07673453, -0.22865714,
           -0.19063865,  0.0590131 , -0.12894847, -0.11323072,  0.0855964 ,
            0.09334032, -0.26848862, -0.12382735, -0.22460641, -0.07650108,
            0.07673453,  0.22865714,  0.19063865, -0.0590131 ])
    Coordinates:
      * time     (time) object 1990-01-31 00:00:00 ... 1991-12-31 00:00:00

    Note that one must explicitly use ``args=[]`` and ``kwargs={}`` to pass arguments
    to the function being applied in ``xr.map_blocks()``:

    >>> xr.map_blocks(
    ...     calculate_anomaly, array, kwargs={"groupby_type": "time.year"},
    ... )
    <xarray.DataArray (time: 24)>
    array([ 0.15361741, -0.25671244, -0.31600032,  0.008463  ,  0.1766172 ,
           -0.11974531,  0.43791243,  0.14197797, -0.06191987, -0.15073425,
           -0.19967375,  0.18619794, -0.05100474, -0.42989909, -0.09153273,
            0.24841842, -0.30708526, -0.31412523,  0.04197439,  0.0422506 ,
            0.14482397,  0.35985481,  0.23487834,  0.12144652])
    Coordinates:
        * time     (time) object 1990-01-31 00:00:00 ... 1991-12-31 00:00:00
    """

    def _wrapper(func, obj, to_array, args, kwargs, expected):
        check_shapes = dict(obj.dims)
        check_shapes.update(expected["shapes"])

        if to_array:
            obj = dataset_to_dataarray(obj)

        result = func(obj, *args, **kwargs)

        # check all dims are present
        missing_dimensions = set(expected["shapes"]) - set(result.sizes)
        if missing_dimensions:
            raise ValueError(
                f"Dimensions {missing_dimensions} missing on returned object."
            )

        # check that index lengths and values are as expected
        for name, index in result.indexes.items():
            if name in check_shapes:
                if len(index) != check_shapes[name]:
                    raise ValueError(
                        f"Received dimension {name!r} of length {len(index)}. Expected length {check_shapes[name]}."
                    )
            if name in expected["indexes"]:
                expected_index = expected["indexes"][name]
                if not index.equals(expected_index):
                    raise ValueError(
                        f"Expected index {name!r} to be {expected_index!r}. Received {index!r} instead."
                    )

        # check that all expected variables were returned
        check_result_variables(result, expected, "coords")
        if isinstance(result, Dataset):
            check_result_variables(result, expected, "data_vars")

        return make_dict(result)

    if template is not None and not isinstance(template, (DataArray, Dataset)):
        raise TypeError(
            f"template must be a DataArray or Dataset. Received {type(template).__name__} instead."
        )
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
    dataset_indexes = set(dataset.indexes)
    if template is None:
        # infer template by providing zero-shaped arrays
        template = infer_template(func, obj, *args, **kwargs)
        template_indexes = set(template.indexes)
        preserved_indexes = template_indexes & dataset_indexes
        new_indexes = template_indexes - dataset_indexes
        indexes = {dim: dataset.indexes[dim] for dim in preserved_indexes}
        indexes.update({k: template.indexes[k] for k in new_indexes})
        output_chunks = {
            dim: input_chunks[dim] for dim in template.dims if dim in input_chunks
        }

    else:
        # template xarray object has been provided with proper sizes and chunk shapes
        template_indexes = set(template.indexes)
        indexes = {dim: dataset.indexes[dim] for dim in dataset_indexes}
        indexes.update({k: template.indexes[k] for k in template_indexes})
        if isinstance(template, DataArray):
            output_chunks = dict(zip(template.dims, template.chunks))  # type: ignore
        else:
            output_chunks = template.chunks  # type: ignore

    for dim in output_chunks:
        if dim in input_chunks and len(input_chunks[dim]) != len(output_chunks[dim]):
            raise ValueError(
                "map_blocks requires that one block of the input maps to one block of output. "
                f"Expected number of output chunks along dimension {dim!r} to be {len(input_chunks[dim])}. "
                f"Received {len(output_chunks[dim])} instead. Please provide template if not provided, or "
                "fix the provided template."
            )

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

    # We're building a new HighLevelGraph hlg. We'll have one new layer
    # for each variable in the dataset, which is the result of the
    # func applied to the values.

    graph: Dict[Any, Any] = {}
    new_layers: DefaultDict[str, Dict[Any, Any]] = collections.defaultdict(dict)
    gname = "{}-{}".format(
        dask.utils.funcname(func), dask.base.tokenize(dataset, args, kwargs)
    )

    # map dims to list of chunk indexes
    ichunk = {dim: range(len(chunks_v)) for dim, chunks_v in input_chunks.items()}
    # mapping from chunk index to slice bounds
    input_chunk_bounds = {
        dim: np.cumsum((0,) + chunks_v) for dim, chunks_v in input_chunks.items()
    }
    output_chunk_bounds = {
        dim: np.cumsum((0,) + chunks_v) for dim, chunks_v in output_chunks.items()
    }

    # iterate over all possible chunk combinations
    for v in itertools.product(*ichunk.values()):
        chunk_index = dict(zip(dataset.dims, v))

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
                    chunk = chunk[chunk_index[dim]]

                chunk_variable_task = (f"{gname}-{name}-{chunk[0]}",) + v
                graph[chunk_variable_task] = (
                    tuple,
                    [variable.dims, chunk, variable.attrs],
                )
            else:
                # non-dask array with possibly chunked dimensions
                # index into variable appropriately
                subsetter = {
                    dim: _get_chunk_slicer(dim, chunk_index, input_chunk_bounds)
                    for dim in variable.dims
                }
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

        # expected["shapes", "coords", "data_vars", "indexes"] are used to raise nice error messages in _wrapper
        expected = {}
        # input chunk 0 along a dimension maps to output chunk 0 along the same dimension
        # even if length of dimension is changed by the applied function
        expected["shapes"] = {
            k: output_chunks[k][v] for k, v in chunk_index.items() if k in output_chunks
        }
        expected["data_vars"] = set(template.data_vars.keys())  # type: ignore
        expected["coords"] = set(template.coords.keys())  # type: ignore
        expected["indexes"] = {
            dim: indexes[dim][_get_chunk_slicer(dim, chunk_index, output_chunk_bounds)]
            for dim in indexes
        }

        from_wrapper = (gname,) + v
        graph[from_wrapper] = (
            _wrapper,
            func,
            (Dataset, (dict, data_vars), (dict, coords), dataset.attrs),
            input_is_array,
            args,
            kwargs,
            expected,
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
                if dim in chunk_index:
                    key += (chunk_index[dim],)
                else:
                    # unchunked dimensions in the input have one chunk in the result
                    # output can have new dimensions with exactly one chunk
                    key += (0,)

            # We're adding multiple new layers to the graph:
            # The first new layer is the result of the computation on
            # the array.
            # Then we add one layer per variable, which extracts the
            # result for that variable, and depends on just the first new
            # layer.
            new_layers[gname_l][key] = (operator.getitem, from_wrapper, name)

    hlg = HighLevelGraph.from_collections(gname, graph, dependencies=[dataset])

    for gname_l, layer in new_layers.items():
        # This adds in the getitems for each variable in the dataset.
        hlg.dependencies[gname_l] = {gname}
        hlg.layers[gname_l] = layer

    result = Dataset(coords=indexes, attrs=template.attrs)
    for name, gname_l in var_key_map.items():
        dims = template[name].dims
        var_chunks = []
        for dim in dims:
            if dim in output_chunks:
                var_chunks.append(output_chunks[dim])
            elif dim in indexes:
                var_chunks.append((len(indexes[dim]),))
            elif dim in template.dims:
                # new unindexed dimension
                var_chunks.append((template.sizes[dim],))

        data = dask.array.Array(
            hlg, name=gname_l, chunks=var_chunks, dtype=template[name].dtype
        )
        result[name] = (dims, data, template[name].attrs)

    result = result.set_coords(template._coord_names)

    if result_is_array:
        da = dataset_to_dataarray(result)
        da.name = template_name
        return da  # type: ignore
    return result  # type: ignore
