try:
    import dask
    import dask.array
    from dask.highlevelgraph import HighLevelGraph

except ImportError:
    pass

from .dataarray import DataArray


def get_chunk_slices(dataset):
    chunk_slices = {}
    for dim, chunks in dataset.chunks.items():
        slices = []
        start = 0
        for chunk in chunks:
            stop = start + chunk
            slices.append(slice(start, stop))
            start = stop
        chunk_slices[dim] = slices

    return chunk_slices


def map_blocks(func, darray):
    """
    A version of dask's map_blocks for DataArrays.

    Parameters
    ----------
    func: callable
        User-provided function that should accept DataArrays corresponding to one chunk.
    darray: DataArray
        Chunks of this array will be provided to 'func'. The function must not change
        shape of the provided DataArray.

    Returns
    -------
    DataArray

    See Also
    --------
    dask.array.map_blocks
    """

    def _wrapper(darray):
        result = func(darray)
        if not isinstance(result, type(darray)):
            raise ValueError("Result is not the same type as input.")
        if result.shape != darray.shape:
            raise ValueError("Result does not have the same shape as input.")
        return result

    meta_array = DataArray(darray.data._meta, dims=darray.dims)
    result_meta = func(meta_array)

    name = "%s-%s" % (darray.name or func.__name__, dask.base.tokenize(darray))

    slicers = get_chunk_slices(darray._to_temp_dataset())
    dask_keys = list(dask.core.flatten(darray.__dask_keys__()))

    graph = {
        (name,)
        + (*key[1:],): (
            _wrapper,
            (
                DataArray,
                key,
                {
                    dim_name: darray[dim_name][slicers[dim_name][index]]
                    for dim_name, index in zip(darray.dims, key[1:])
                },
                darray.dims,
            ),
        )
        for key in dask_keys
    }

    graph = HighLevelGraph.from_collections(name, graph, dependencies=[darray])

    return DataArray(
        dask.array.Array(graph, name, chunks=darray.chunks, meta=result_meta),
        dims=darray.dims,
        coords=darray.coords,
    )
