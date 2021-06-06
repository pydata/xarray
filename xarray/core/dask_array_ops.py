from typing import TYPE_CHECKING, Tuple, TypeVar

from . import dtypes, nputils

if TYPE_CHECKING:
    from .dataarray import DataArray
    from .dataset import Dataset

    T_DSorDA = TypeVar("T_DSorDA", "DataArray", "Dataset")


def dask_rolling_wrapper(moving_func, a, window, min_count=None, axis=-1):
    """Wrapper to apply bottleneck moving window funcs on dask arrays"""
    import dask.array as da

    dtype, fill_value = dtypes.maybe_promote(a.dtype)
    a = a.astype(dtype)
    # inputs for overlap
    if axis < 0:
        axis = a.ndim + axis
    depth = {d: 0 for d in range(a.ndim)}
    depth[axis] = (window + 1) // 2
    boundary = {d: fill_value for d in range(a.ndim)}
    # Create overlap array.
    ag = da.overlap.overlap(a, depth=depth, boundary=boundary)
    # apply rolling func
    out = da.map_blocks(
        moving_func, ag, window, min_count=min_count, axis=axis, dtype=a.dtype
    )
    # trim array
    result = da.overlap.trim_internal(out, depth)
    return result


def least_squares(lhs, rhs, rcond=None, skipna=False):
    import dask.array as da

    lhs_da = da.from_array(lhs, chunks=(rhs.chunks[0], lhs.shape[1]))
    if skipna:
        added_dim = rhs.ndim == 1
        if added_dim:
            rhs = rhs.reshape(rhs.shape[0], 1)
        results = da.apply_along_axis(
            nputils._nanpolyfit_1d,
            0,
            rhs,
            lhs_da,
            dtype=float,
            shape=(lhs.shape[1] + 1,),
            rcond=rcond,
        )
        coeffs = results[:-1, ...]
        residuals = results[-1, ...]
        if added_dim:
            coeffs = coeffs.reshape(coeffs.shape[0])
            residuals = residuals.reshape(residuals.shape[0])
    else:
        # Residuals here are (1, 1) but should be (K,) as rhs is (N, K)
        # See issue dask/dask#6516
        coeffs, residuals, _, _ = da.linalg.lstsq(lhs_da, rhs)
    return coeffs, residuals


def push(array, n, axis):
    """
    Dask-aware bottleneck.push
    """
    from bottleneck import push

    if len(array.chunks[axis]) > 1 and n is not None and n < array.shape[axis]:
        raise NotImplementedError(
            "Cannot fill along a chunked axis when limit is not None."
            "Either rechunk to a single chunk along this axis or call .compute() or .load() first."
        )
    if all(c == 1 for c in array.chunks[axis]):
        array = array.rechunk({axis: 2})
    pushed = array.map_blocks(push, axis=axis, n=n)
    if len(array.chunks[axis]) > 1:
        pushed = pushed.map_overlap(
            push, axis=axis, n=n, depth={axis: (1, 0)}, boundary="none"
        )
    return pushed


def unify_chunks(*objects: "T_DSorDA") -> Tuple["T_DSorDA", ...]:
    """
    Given any number of Dataset and/or DataArray objects, returns
    new objects with unified chunk size along all chunked dimensions.

    Returns
    -------
    unified (DataArray or Dataset) – Tuple of objects with the same type as
    *objects with consistent chunk sizes for all dask-array variables

    See Also
    --------
    dask.array.core.unify_chunks
    """
    from .dataarray import DataArray

    # Convert chunked dataarrays to datasets
    datasets = []
    are_chunked = []
    for i, obj in enumerate(objects):
        ds = obj._to_temp_dataset() if isinstance(obj, DataArray) else obj.copy()
        datasets.append(ds)
        try:
            are_chunked.append(True if obj.chunks else False)
        except ValueError:  # "inconsistent chunks"
            are_chunked.append(True)

    # Return input objects if no object is chunked
    if not any(are_chunked):
        return objects

    # Unify chunks using dask.array.core.unify_chunks
    import dask.array

    dask_unify_args = []
    for ds, is_chunked in zip(datasets, are_chunked):
        if not is_chunked:
            continue
        dims_pos_map = {dim: index for index, dim in enumerate(ds.dims)}
        for variable in ds.variables.values():
            if isinstance(variable.data, dask.array.Array):
                dims_tuple = [dims_pos_map[dim] for dim in variable.dims]
                dask_unify_args.append(variable.data)
                dask_unify_args.append(dims_tuple)
    _, rechunked_arrays = dask.array.core.unify_chunks(*dask_unify_args)

    # Substitute rechunked variables
    unified = []
    rechunked_arrays = list(rechunked_arrays)
    for obj, ds, is_chunked in zip(objects, datasets, are_chunked):
        if not is_chunked:
            unified.append(obj)
        else:
            for name, variable in ds.variables.items():
                if isinstance(variable.data, dask.array.Array):
                    ds.variables[name]._data = rechunked_arrays.pop(0)
            unified.append(
                obj._from_temp_dataset(ds) if isinstance(obj, DataArray) else ds
            )

    return tuple(unified)
