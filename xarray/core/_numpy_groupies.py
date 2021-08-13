import itertools
import os
import typing

import numpy as np

XARRAY_NUMPY_GROUPIES = os.environ.get("XARRAY_NUMPY_GROUPIES", "False").lower() in (
    "true",
    "1",
)

if XARRAY_NUMPY_GROUPIES:
    try:
        import dask_groupby.aggregations
        import dask_groupby.core
    except ImportError:
        raise ImportError(
            "Using numpy_groupies with xarray requires the `dask-groupby` package "
            "to be installed. To install, run `python -m pip install dask_groupby`."
        )
    import dask

    def xarray_reduce(
        obj,
        *by,
        func,
        expected_groups=None,
        bins=None,
        dim=None,
        split_out=1,
        fill_value=None,
        blockwise=False,
    ):
        """Reduce a DataArray or Dataset using dask_groupby.xarray."""

        from .alignment import broadcast
        from .computation import apply_ufunc
        from .dataarray import DataArray

        by: typing.Tuple[DataArray] = tuple(obj[g] if isinstance(g, str) else g for g in by)  # type: ignore
        if len(by) > 1 and any(dask.is_dask_collection(by_) for by_ in by):
            raise ValueError(
                "Grouping by multiple variables will call compute dask variables."
            )

        grouper_dims = set(itertools.chain(*tuple(g.dims for g in by)))
        obj, *by = broadcast(obj, *by, exclude=set(obj.dims) - grouper_dims)
        obj = obj.transpose(..., *by[0].dims)

        dim = by[0].dims if dim is None else dask_groupby.aggregations._atleast_1d(dim)
        assert isinstance(obj, DataArray)
        axis = tuple(obj.get_axis_num(d) for d in dim)
        group_names = tuple(g.name for g in by)
        if len(by) > 1:
            (
                group_idx,
                expected_groups,
                group_shape,
                _,
                _,
                _,
            ) = dask_groupby.core.factorize_(
                tuple(g.data for g in by), expected_groups, bins
            )
            to_group = DataArray(
                group_idx, dims=dim, coords={d: by[0][d] for d in by[0].indexes}
            )
        else:
            if expected_groups is None and isinstance(by[0].data, np.ndarray):
                expected_groups = (np.unique(by[0].data),)
            if expected_groups is None:
                raise NotImplementedError(
                    "Please provided expected_groups if not grouping by a numpy-backed DataArray"
                )
            group_shape = (len(expected_groups[0]),)
            to_group = by[0]

        group_sizes = dict(zip(group_names, group_shape))
        indims = tuple(obj.dims)
        otherdims = tuple(d for d in indims if d not in dim)
        result_dims = otherdims + group_names

        def wrapper(*args, **kwargs):
            result, groups = dask_groupby.core.groupby_reduce(*args, **kwargs)
            if len(by) > 1:
                # all groups need not be present. reindex here
                # TODO: add test
                reindexed = dask_groupby.core.reindex_(
                    result,
                    from_=groups,
                    to=np.arange(np.prod(group_shape)),
                    fill_value=fill_value,
                    axis=-1,
                )
                result = reindexed.reshape(result.shape[:-1] + group_shape)
            return result

        actual = apply_ufunc(
            wrapper,
            obj,
            to_group,
            input_core_dims=[indims, dim],
            dask="allowed",
            output_core_dims=[result_dims],
            dask_gufunc_kwargs=dict(output_sizes=group_sizes),
            kwargs={
                "func": func,
                "axis": axis,
                "split_out": split_out,
                "fill_value": fill_value,
                "blockwise": blockwise,
            },
        )

        for name, expect in zip(group_names, expected_groups):
            actual[name] = expect

        return actual
