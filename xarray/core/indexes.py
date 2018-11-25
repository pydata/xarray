from __future__ import absolute_import, division, print_function

import numpy as np
import pandas as pd


def normalize_indexes(indexes, coords, sizes):
    """Normalize indexes for Dataset/DataArray.

    Validates that all indexes are pd.Index instances (or at least satisfy
    the Index API we need for xarray). Creates default indexes for variables
    whose name matches their sole dimension.

    Eventually: consider combining indexes along the same dimension into a
    MultiIndex.

    Parameters
    ----------
    indexes : Optional[Dict[Any, pandas.Index]]
        Explicitly supplied indexes, if any.
    coords : Mapping[Any, xarray.Variable]
        Coordinate variables from which to draw default indexes.
    dim_sizes : Mapping[Any, int]
        Integer sizes for each Dataset/DataArray dimension.

    Returns
    -------
    Mapping[Any, pandas.Index] mapping indexing keys (levels/dimension names)
    to indexes used for indexing along that dimension.
    """
    indexes = {} if indexes is None else dict(indexes)

    # default indexes
    for key in sizes:
        if key not in indexes:
            if key in coords:
                indexes[key] = coords[key].to_index()
            else:
                # need to ensure dtype=int64 in case range is empty on Python 2
                indexes[key] = pd.Index(
                    range(sizes[key]), name=key, dtype=np.int64)

    # TODO: merge logic to combine indexes along the same dimension into a
    # MultiIndex

    return indexes


def combine_indexes(input_indexes, output_coords, unsafe=True):
    """Combine indexes from inputs into indexes for an operation result.

    Drops indexes corresponding to dropped coordinates.

    Eventually: consider combining indexes along the same dimension into a
    MultiIndex.

    Parameters
    ----------
    input_indexes : Sequence[Mapping[Any, pandas.Index]]
        Sequence of mappings of indexes to combine.
    output_coords : Optional[Sequence[Mapping[Any, pandas.Variable]]]
        Optional sequence of mappings provided output coordinates.
    unsafe : bool, optional
        Whether it's OK to skip compatibility checks for input indexes.

    Returns
    -------
    List[Mapping[Any, pandas.Index]] mapping variable names to indexes,
    for each requested mapping of output coordinates.
    """
    if not unsafe:
        raise NotImplementedError('safe index combining not supported yet')
    output_indexes = []
    for output_coords_item in output_coords:
        indexes = {}
        for input_indexes_item in input_indexes:
            for k, v in input_indexes_item.items():
                if k in output_coords_item:
                    indexes[k] = v
        output_indexes.append(indexes)
