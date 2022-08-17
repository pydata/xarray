
from typing import Any, Hashable, List, Mapping, Optional, Set, Tuple, Union

import hypothesis.extra.numpy as npst
import hypothesis.strategies as st
import numpy as np
from hypothesis import assume

import xarray as xr

__all__ = [
    "valid_dtypes",
    "np_arrays",
    "names",
    "dimension_names",
    "dimension_sizes",
    "variables",
    "coordinate_variables",
    "dataarrays",
    "data_variables",
    "datasets",
]

# required to exclude weirder dtypes e.g. unicode, byte_string, array, or nested dtypes.
valid_dtypes: st.SearchStrategy[np.dtype] = (
    npst.integer_dtypes()
    | npst.unsigned_integer_dtypes()
    | npst.floating_dtypes()
    | npst.complex_number_dtypes()
)
valid_dtypes.__doc__ = """Generates only those numpy dtypes which xarray can handle."""


def elements(dtype) -> st.SearchStrategy[Any]:
    """
    Generates scalar elements to go in a numpy-like array.

    Requires the hypothesis package to be installed.
    """
    max_value = 100
    min_value = 0 if dtype.kind == "u" else -max_value

    return npst.from_dtype(
        dtype, allow_infinity=False, min_value=min_value, max_value=max_value
    )


@st.composite
def np_arrays(
    draw: st.DrawFn,
    shape: Union[Tuple[int], st.SearchStrategy[Tuple[int]]] = None,
    dtype: Union[np.dtype, st.SearchStrategy[np.dtype]] = None,
) -> st.SearchStrategy[np.ndarray]:
    """
    Generates arbitrary numpy arrays with xarray-compatible dtypes.

    Requires the hypothesis package to be installed.

    Parameters
    ----------
    shape
    dtype
        Default is to use any of the valid_dtypes defined for xarray.
    """
    if shape is None:
        shape = draw(npst.array_shapes())
    elif isinstance(shape, st.SearchStrategy):
        shape = draw(shape)

    if dtype is None:
        dtype = draw(valid_dtypes)
    elif isinstance(dtype, st.SearchStrategy):
        dtype = draw(dtype)

    return draw(npst.arrays(dtype=dtype, shape=shape, elements=elements(dtype)))


names = st.text(st.characters(), min_size=1)
names.__doc__ = """Generates arbitrary string names for dimensions / variables."""


def dimension_names(
    min_ndims: int = 0,
    max_ndims: int = 3,
) -> st.SearchStrategy[List[str]]:
    """
    Generates an arbitrary list of valid dimension names.

    Requires the hypothesis package to be installed.

    Parameters
    ----------
    min_ndims
        Minimum number of dimensions in generated list.
    max_ndims
        Maximum number of dimensions in generated list.
    """

    return st.lists(
        elements=names,
        min_size=min_ndims,
        max_size=max_ndims,
        unique=True,
    )


def dimension_sizes(
    min_ndims: int = 0,
    max_ndims: int = 3,
    min_length: int = 1,
    max_length: int = None,
) -> st.SearchStrategy[Mapping[str, int]]:
    """
    Generates an arbitrary mapping from dimension names to lengths.

    Requires the hypothesis package to be installed.

    Parameters
    ----------
    min_ndims: int, optional
        Minimum number of dimensions in generated list.
        Default is 1.
    max_ndims: int, optional
        Maximum number of dimensions in generated list.
        Default is 3.
    min_length: int, optional
        Minimum size of a dimension.
        Default is 1.
    max_length: int, optional
        Minimum size of a dimension.
        Default is `min_length` + 5.
    """

    if max_length is None:
        max_length = min_length + 5

    return st.dictionaries(
        keys=names,
        values=st.integers(min_value=min_length, max_value=max_length),
        min_size=min_ndims,
        max_size=max_ndims,
    )


# Is there a way to do this in general?
# Could make a Protocol...
T_Array = Any


@st.composite
def variables(
    draw: st.DrawFn,
    data: st.SearchStrategy[T_Array] = None,
    dims: Union[
        st.SearchStrategy[List[str]], st.SearchStrategy[Mapping[str, int]]
    ] = None,
    attrs: st.SearchStrategy[Mapping] = None,
) -> st.SearchStrategy[xr.Variable]:
    """
    Generates arbitrary xarray.Variable objects.

    Follows the signature of the xarray.Variable constructor, but you can also pass alternative strategies to generate
    either numpy-like array data or dimension names.

    Passing nothing will generate a completely arbitrary Variable (backed by a numpy array).

    Requires the hypothesis package to be installed.

    Parameters
    ----------
    data: Strategy generating array-likes, optional
        Default is to generate numpy data of arbitrary shape, values and dtype.
    dims: Strategy for generating the dimensions, optional
        Can either be a strategy for generating a list of string dimension names,
        or a strategy for generating a mapping of string dimension names to integer lengths along each dimension.
        If provided in the former form the lengths of the returned Variable will either be determined from the
        data argument if given or arbitrarily generated if not.
        Default is to generate arbitrary dimension names for each axis in data.
    attrs: Strategy which generates dicts, optional
    """

    if any(
        not isinstance(arg, st.SearchStrategy) and arg is not None
        for arg in [data, dims, attrs]
    ):
        raise TypeError(
            "Contents must be provided as a hypothesis.strategies.SearchStrategy object (or None)."
            "To specify fixed contents, use hypothesis.strategies.just()."
        )

    if data is not None and dims is None:
        # no dims -> generate dims to match data
        data = draw(data)
        dims = draw(dimension_names(min_ndims=data.ndim, max_ndims=data.ndim))

    elif dims is not None and data is None:
        # no data -> generate data to match dims
        dims = draw(dims)
        if isinstance(dims, List):
            valid_shapes = npst.array_shapes(min_dims=len(dims), max_dims=len(dims))
            data = draw(np_arrays(shape=draw(valid_shapes)))
        else:
            # should be a mapping of form {dim_names: lengths}
            shape = tuple(dims.values())
            data = draw(np_arrays(shape=shape))

    elif data is not None and dims is not None:
        # both data and dims provided -> check drawn examples are compatible
        dims = draw(dims)

        # TODO is there another way to enforce these assumptions? This is very like to fail hypothesis' health checks
        # TODO how do I write a test that checks that the hypothesis Unsatisfiable error will be raised?
        # TODO or we could just raise in this case?
        if isinstance(dims, List):
            data = draw(data)
            assume(data.ndim == len(dims))
        else:
            # should be a mapping of form {dim_names: lengths}
            data = draw(data)
            shape = tuple(dims.values())
            assume(data.shape == shape)

    else:
        # nothing provided, so generate everything consistently by drawing dims to match data
        data = draw(np_arrays())
        dims = draw(dimension_names(min_ndims=data.ndim, max_ndims=data.ndim))

    if isinstance(attrs, st.SearchStrategy):
        attrs = draw(attrs)
    elif attrs is None:
        # TODO autogenerate some attributes
        ...

    return xr.Variable(dims=dims, data=data, attrs=attrs)


@st.composite
def _unique_subset_of(
    draw: st.DrawFn, d: Mapping[Hashable, Any]
) -> st.SearchStrategy[Mapping[Hashable, Any]]:
    subset_keys = draw(st.lists(st.sampled_from(list(d.keys())), unique=True))
    return {k: d[k] for k in subset_keys}


@st.composite
def _alignable_variables(
    draw: st.DrawFn,
    dim_sizes: Mapping[str, int],
) -> st.SearchStrategy[List[xr.Variable]]:
    """Generates lists of variables with compatible (i.e. alignable) dimensions and sizes."""

    alignable_dim_sizes = draw(_unique_subset_of(dim_sizes)) if dim_sizes else {}

    # TODO don't hard code max number of variables
    return draw(st.lists(variables(dims=st.just(alignable_dim_sizes)), max_size=3))


@st.composite
def coordinate_variables(
    draw: st.DrawFn,
    dim_sizes: Mapping[str, int],
) -> st.SearchStrategy[Mapping[str, xr.Variable]]:
    """
    Generates dicts of alignable Variable objects for use as coordinates.

    Differs from data_variables strategy in that it deliberately creates dimension coordinates
    (i.e. 1D variables with the same name as a dimension) as well as non-dimension coordinates.

    Requires the hypothesis package to be installed.

    Parameters
    ----------
    dim_sizes: Mapping of str to int
        Sizes of dimensions to use for coordinates.
    """
    dim_names = list(dim_sizes.keys())

    all_coords = {}

    # Possibly generate 1D "dimension coordinates" - explicit possibility not to include amy helps with shrinking
    if dim_names and st.booleans():
        # first generate subset of dimension names - these set which dimension coords will be included
        dim_coord_names_and_lengths = draw(_unique_subset_of(dim_sizes))

        # then generate 1D variables for each name
        dim_coords = {
            n: draw(variables(dims=st.just({n: l})))
            for n, l in dim_coord_names_and_lengths.items()
        }
        all_coords.update(dim_coords)

    # Possibly generate ND "non-dimension coordinates" - explicit possibility not to include any helps with shrinking
    if st.booleans():
        non_dim_coord_vars = draw(_alignable_variables(dim_sizes=dim_sizes))

        # can't have same name as a dimension
        valid_non_dim_coord_names = names.filter(lambda n: n not in dim_names)
        non_dim_coord_names = draw(
            st.lists(
                valid_non_dim_coord_names,
                min_size=len(non_dim_coord_vars),
                max_size=len(non_dim_coord_vars),
                unique=True,
            )
        )

        non_dim_coords = {n: v for n, v in zip(non_dim_coord_names, non_dim_coord_vars)}
        all_coords.update(non_dim_coords)

    return all_coords


@st.composite
def dataarrays(
    draw: st.DrawFn,
    data: st.SearchStrategy[T_Array] = None,
    coords: Mapping[str, xr.Variable] = None,
    dims: Union[
        st.SearchStrategy[List[str]], st.SearchStrategy[Mapping[str, int]]
    ] = None,
    name: st.SearchStrategy[Union[str, None]] = None,
    attrs: st.SearchStrategy[Mapping] = None,
) -> st.SearchStrategy[xr.DataArray]:
    """
    Generates arbitrary xarray.DataArray objects.

    Follows the basic signature of the xarray.DataArray constructor, but you can also pass alternative strategies to
    generate either numpy-like array data, dimensions, or coordinates.

    Passing nothing will generate a completely arbitrary DataArray (backed by a numpy array).

    Requires the hypothesis package to be installed.

    Parameters
    ----------
    data: Strategy generating array-likes, optional
        Default is to generate numpy data of arbitrary shape, values and dtypes.
    coords: Strategy generating mappings from coordinate names to xr.Variables objects, optional
        Default is to generate an arbitrary combination of both dimension and non-dimension coordinates,
        with sizes matching data and/or dims, but arbitrary names, dtypes, and values.
    dims: Strategy for generating the dimensions, optional
        Can either be a strategy for generating a list of string dimension names,
        or a strategy for generating a mapping of string dimension names to integer lengths along each dimension.
        If provided in the former form the lengths of the returned Variable will either be determined from the
        data argument if given or arbitrarily generated if not.
        Default is to generate arbitrary dimension sizes, or arbitrary dimension names for each axis in data.
    name: Strategy for generating a string name, optional
        Default is to use the `names` strategy, or to create an unnamed DataArray.
    attrs: Strategy which generates dicts, optional
    """

    if name is None:
        name = draw(st.none() | names)

    if coords is not None:
        raise NotImplementedError()

    if data is not None and dims is None:
        # no dims -> generate dims to match data
        data = draw(data)
        dim_names = draw(dimension_names(min_ndims=data.ndim, max_ndims=data.ndim))
        dim_sizes = {n: l for n, l in zip(dim_names, data.shape)}
        coords = draw(coordinate_variables(dim_sizes=dim_sizes))

    elif data is None and dims is not None:
        # no data -> generate data to match dims
        dims = draw(dims)
        if isinstance(dims, List):
            dim_names = dims
            valid_shapes = npst.array_shapes(min_dims=len(dims), max_dims=len(dims))
            data = draw(np_arrays(shape=draw(valid_shapes)))
            dim_sizes = {n: l for n, l in zip(dims, data.shape)}
            coords = draw(coordinate_variables(dim_sizes=dim_sizes))

        else:
            # should be a mapping of form {dim_names: lengths}
            dim_names, shape = list(dims.keys()), tuple(dims.values())
            data = draw(np_arrays(shape=shape))
            coords = draw(coordinate_variables(dim_sizes=dims))

    elif data is not None and dims is not None:
        # both data and dims provided -> check drawn examples are compatible
        dims = draw(dims)
        if isinstance(dims, List):
            dim_names = dims
            data = draw(data)
            assume(data.ndim == len(dims))
            dim_sizes = {n: l for n, l in zip(dims, data.shape)}
        else:
            # should be a mapping of form {dim_names: lengths}
            data = draw(data)
            dim_sizes = dims
            dim_names, shape = list(dims.keys()), tuple(dims.values())
            assume(data.shape == shape)

        coords = draw(coordinate_variables(dim_sizes=dim_sizes))

    else:
        # nothing provided, so generate everything consistently by drawing dims to match data, and coords to match both
        data = draw(np_arrays())
        dim_names = draw(dimension_names(min_ndims=data.ndim, max_ndims=data.ndim))
        dim_sizes = {n: l for n, l in zip(dim_names, data.shape)}
        coords = draw(coordinate_variables(dim_sizes=dim_sizes))

    return xr.DataArray(
        data=data,
        coords=coords,
        name=name,
        dims=dim_names,
        attrs=attrs,
    )


@st.composite
def data_variables(
    draw: st.DrawFn,
    dim_sizes: Mapping[str, int],
    allowed_names: st.SearchStrategy[str] = None,
) -> st.SearchStrategy[Mapping[str, xr.Variable]]:
    """
    Generates dicts of alignable Variable objects for use as Dataset data variables.

    Requires the hypothesis package to be installed.

    Parameters
    ----------
    dim_sizes: Mapping of str to int
        Sizes of dimensions to use for variables.
    allowed_names: Strategy generating strings
        Allowed names for data variables. Needed to avoid conflict with names of coordinate variables & dimensions.
    """
    # TODO these shouldn't have the same name as any dimensions or any coordinates...
    vars = draw(_alignable_variables(dim_sizes=dim_sizes))
    dim_names = list(dim_sizes.keys())

    # can't have same name as a dimension
    # TODO this is also used in coordinate_variables so refactor it out into separate function
    valid_var_names = allowed_names.filter(lambda n: n not in dim_names)
    # TODO do I actually need to draw from st.lists for this?
    var_names = draw(
        st.lists(
            valid_var_names,
            min_size=len(vars),
            max_size=len(vars),
            unique=True,
        )
    )

    data_vars = {n: v for n, v in zip(var_names, vars)}
    return data_vars


@st.composite
def datasets(
    draw: st.DrawFn,
    data_vars: st.SearchStrategy[Mapping[str, xr.Variable]] = None,
    coords: Mapping[str, xr.Variable] = None,
    dims: Union[
        st.SearchStrategy[List[str]], st.SearchStrategy[Mapping[str, int]]
    ] = None,
    attrs: st.SearchStrategy[Mapping] = None,
) -> st.SearchStrategy[xr.Dataset]:
    """
    Generates arbitrary xarray.Dataset objects.

    Follows the basic signature of the xarray.Dataset constructor, but you can also pass alternative strategies to
    generate either numpy-like array data variables, dimensions, or coordinates.

    Passing nothing will generate a completely arbitrary Dataset (backed by numpy arrays).

    Requires the hypothesis package to be installed.

    Parameters
    ----------
    data_vars: Strategy generating mappings from variable names to xr.Variable objects, optional
        Default is to generate an arbitrary combination of compatible variables with sizes matching dims,
        but arbitrary names, dtypes, and values.
    coords: Strategy generating mappings from coordinate names to xr.Variable objects, optional
        Default is to generate an arbitrary combination of both dimension and non-dimension coordinates,
        with sizes matching data_vars and/or dims, but arbitrary names, dtypes, and values.
    dims: Strategy for generating the dimensions, optional
        Can either be a strategy for generating a list of string dimension names,
        or a strategy for generating a mapping of string dimension names to integer lengths along each dimension.
        If provided in the former form the lengths of the returned Variable will either be determined from the
        data argument if given or arbitrarily generated if not.
        Default is to generate arbitrary dimension sizes.
    attrs: Strategy which generates dicts, optional
    """

    if any(arg is not None for arg in [data_vars, coords, dims, attrs]):
        raise NotImplementedError()
    else:
        # nothing provided, so generate everything consistently by drawing dims to match data, and coords to match both
        dim_sizes = draw(dimension_sizes())
        # TODO allow for no coordinate variables
        coords = draw(coordinate_variables(dim_sizes=dim_sizes))
        coord_names = list(coords.keys())
        data_var_names = names.filter(lambda n: n not in coord_names)
        # TODO allow for no data variables
        data_vars = draw(
            data_variables(dim_sizes=dim_sizes, allowed_names=data_var_names)
        )

    return xr.Dataset(data_vars=data_vars, coords=coords, attrs=attrs)


@st.composite
def block_lengths(
    draw: st.DrawFn,
    ax_length: int,
    min_chunk_length: int = 1,
    max_chunk_length: Optional[int] = None,
) -> st.SearchStrategy[Tuple[int, ...]]:
    """Generate different chunking patterns along one dimension of an array."""

    chunks = []
    remaining_length = ax_length
    while remaining_length > 0:
        _max_chunk_length = (
            min(remaining_length, max_chunk_length)
            if max_chunk_length
            else remaining_length
        )

        if min_chunk_length > _max_chunk_length:
            # if we are at the end of the array we have no choice but to use a smaller chunk
            chunk = remaining_length
        else:
            chunk = draw(
                st.integers(min_value=min_chunk_length, max_value=_max_chunk_length)
            )

        chunks.append(chunk)
        remaining_length = remaining_length - chunk

    return tuple(chunks)


# TODO we could remove this once dask/9374 is merged upstream
@st.composite
def chunks(
    draw: st.DrawFn,
    shape: Tuple[int, ...],
    axes: Optional[Union[int, Tuple[int, ...]]] = None,
    min_chunk_length: int = 1,
    max_chunk_length: Optional[int] = None,
) -> st.SearchStrategy[Tuple[Tuple[int, ...], ...]]:
    """
    Generates different chunking patterns for an N-D array with a given shape.

    Returns chunking structure as a tuple of tuples of ints, with each inner tuple containing
    the block lengths along one dimension of the array.

    You can limit chunking to specific axes using the `axes` kwarg, and specify minimum and
    maximum block lengths.

    Requires the hypothesis package to be installed.

    Parameters
    ----------
    shape : tuple of ints
        Shape of the array for which you want to generate a chunking pattern.
    axes : None or int or tuple of ints, optional
        ...
    min_chunk_length : int, default is 1
        Minimum chunk length to use along all axes.
    max_chunk_length: int, optional
        Maximum chunk length to use along all axes.
        Default is that the chunk can be as long as the length of the array along that axis.

    Examples
    --------
    Chunking along all axes by default

    >>> chunks(shape=(2, 3)).example()
    ((1, 1), (1, 2))

    Chunking only along the second axis

    >>> chunks(shape=(2, 3), axis=1).example()
    ((2,), (1, 1, 1))

    Minimum size chunks of length 2 along all axes

    >>> chunks(shape=(2, 3), min_chunk_length=2).example()
    ((2,), (2, 1))

    Smallest possible chunks along all axes

    >>> chunks(shape=(2, 3), max_chunk_length=1).example()
    ((1, 1), (1, 1, 1))

    Maximum size chunks along all axes

    >>> chunks(shape=(2, 3), axes=()).example()
    ((2,), (3,))

    See Also
    --------
    testing.strategies.chunks
    DataArray.chunk
    DataArray.chunks
    """

    if min_chunk_length < 1 or not isinstance(min_chunk_length, int):
        raise ValueError("min_chunk_length must be an integer >= 1")

    if max_chunk_length:
        if max_chunk_length < 1 or not isinstance(min_chunk_length, int):
            raise ValueError("max_chunk_length must be an integer >= 1")

    if axes is None:
        axes = tuple(range(len(shape)))
    elif isinstance(axes, int):
        axes = (axes,)

    chunks = []
    for axis, ax_length in enumerate(shape):

        _max_chunk_length = (
            min(max_chunk_length, ax_length) if max_chunk_length else ax_length
        )

        if axes is not None and axis in axes:
            block_lengths_along_ax = draw(
                block_lengths(
                    ax_length,
                    min_chunk_length=min_chunk_length,
                    max_chunk_length=_max_chunk_length,
                )
            )
        else:
            # don't chunk along this dimension
            block_lengths_along_ax = (ax_length,)

        chunks.append(block_lengths_along_ax)

    return tuple(chunks)


@st.composite
def chunksizes(
    draw: st.DrawFn,
    sizes: Mapping[str, int],
    dims: Set[str] = None,
    min_chunk_length: int = 1,
    max_chunk_length: int = None,
) -> st.SearchStrategy[Mapping[str, Tuple[int, ...]]]:
    """
    Generate different chunking patterns for an xarray object with given sizes.

    Returns chunking structure as a mapping of dimension names to tuples of ints,
    with each tuple containing the block lengths along one dimension of the object.

    You can limit chunking to specific dimensions given by the `dim` kwarg.

    Requires the hypothesis package to be installed.

    Parameters
    ----------
    sizes : mapping of dimension names to ints
        Size of the object for which you want to generate a chunking pattern.
    dims : set of str, optional
        Dimensions to chunk along. Default is to chunk along all dimensions.
    min_chunk_length : int, default is 1
        Minimum chunk length to use along all dimensions.
    max_chunk_length: int, optional
        Maximum chunk length to use along all dimensions.
        Default is that the chunk can be as long as the length of the array along that dimension.

    See Also
    --------
    testing.strategies.chunks
    DataArray.chunk
    DataArray.chunksizes
    DataArray.sizes
    """
    shape = tuple(sizes.values())
    axes = tuple(list(sizes.keys()).index(d) for d in dims) if dims else None
    _chunks = draw(
        chunks(
            shape=shape,
            axes=axes,
            min_chunk_length=min_chunk_length,
            max_chunk_length=max_chunk_length,
        )
    )

    return {d: c for d, c in zip(list(sizes.keys()), _chunks)}
