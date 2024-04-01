from collections.abc import Hashable, Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, Protocol, Union, overload

try:
    import hypothesis.strategies as st
except ImportError as e:
    raise ImportError(
        "`xarray.testing.strategies` requires `hypothesis` to be installed."
    ) from e

import hypothesis.extra.numpy as npst
import numpy as np
from hypothesis.errors import InvalidArgument

import xarray as xr
from xarray.core.types import T_DuckArray

if TYPE_CHECKING:
    from xarray.core.types import _DTypeLikeNested, _ShapeLike


__all__ = [
    "supported_dtypes",
    "names",
    "dimension_names",
    "dimension_sizes",
    "attrs",
    "variables",
    "coordinate_variables",
    "dataarrays",
    "data_variables",
    "datasets",
    "unique_subset_of",
]


class ArrayStrategyFn(Protocol[T_DuckArray]):
    def __call__(
        self,
        *,
        shape: "_ShapeLike",
        dtype: "_DTypeLikeNested",
    ) -> st.SearchStrategy[T_DuckArray]: ...


def supported_dtypes() -> st.SearchStrategy[np.dtype]:
    """
    Generates only those numpy dtypes which xarray can handle.

    Use instead of hypothesis.extra.numpy.scalar_dtypes in order to exclude weirder dtypes such as unicode, byte_string, array, or nested dtypes.
    Also excludes datetimes, which dodges bugs with pandas non-nanosecond datetime overflows.

    Requires the hypothesis package to be installed.

    See Also
    --------
    :ref:`testing.hypothesis`_
    """
    # TODO should this be exposed publicly?
    # We should at least decide what the set of numpy dtypes that xarray officially supports is.
    return (
        npst.integer_dtypes()
        | npst.unsigned_integer_dtypes()
        | npst.floating_dtypes()
        | npst.complex_number_dtypes()
    )


# TODO Generalize to all valid unicode characters once formatting bugs in xarray's reprs are fixed + docs can handle it.
_readable_characters = st.characters(
    categories=["L", "N"], max_codepoint=0x017F
)  # only use characters within the "Latin Extended-A" subset of unicode


def names() -> st.SearchStrategy[str]:
    """
    Generates arbitrary string names for dimensions / variables.

    Requires the hypothesis package to be installed.

    See Also
    --------
    :ref:`testing.hypothesis`_
    """
    return st.text(
        _readable_characters,
        min_size=1,
        max_size=5,
    )


def dimension_names(
    *,
    min_dims: int = 0,
    max_dims: int = 3,
) -> st.SearchStrategy[list[Hashable]]:
    """
    Generates an arbitrary list of valid dimension names.

    Requires the hypothesis package to be installed.

    Parameters
    ----------
    min_dims
        Minimum number of dimensions in generated list.
    max_dims
        Maximum number of dimensions in generated list.
    """

    return st.lists(
        elements=names(),
        min_size=min_dims,
        max_size=max_dims,
        unique=True,
    )


def dimension_sizes(
    *,
    dim_names: st.SearchStrategy[Hashable] = names(),
    min_dims: int = 0,
    max_dims: int = 3,
    min_side: int = 1,
    max_side: Union[int, None] = None,
) -> st.SearchStrategy[Mapping[Hashable, int]]:
    """
    Generates an arbitrary mapping from dimension names to lengths.

    Requires the hypothesis package to be installed.

    Parameters
    ----------
    dim_names: strategy generating strings, optional
        Strategy for generating dimension names.
        Defaults to the `names` strategy.
    min_dims: int, optional
        Minimum number of dimensions in generated list.
        Default is 1.
    max_dims: int, optional
        Maximum number of dimensions in generated list.
        Default is 3.
    min_side: int, optional
        Minimum size of a dimension.
        Default is 1.
    max_side: int, optional
        Minimum size of a dimension.
        Default is `min_length` + 5.

    See Also
    --------
    :ref:`testing.hypothesis`_
    """

    if max_side is None:
        max_side = min_side + 3

    return st.dictionaries(
        keys=dim_names,
        values=st.integers(min_value=min_side, max_value=max_side),
        min_size=min_dims,
        max_size=max_dims,
    )


_readable_strings = st.text(
    _readable_characters,
    max_size=5,
)
_attr_keys = _readable_strings
_small_arrays = npst.arrays(
    shape=npst.array_shapes(
        max_side=2,
        max_dims=2,
    ),
    dtype=npst.scalar_dtypes(),
)
_attr_values = st.none() | st.booleans() | _readable_strings | _small_arrays


def attrs() -> st.SearchStrategy[Mapping[Hashable, Any]]:
    """
    Generates arbitrary valid attributes dictionaries for xarray objects.

    The generated dictionaries can potentially be recursive.

    Requires the hypothesis package to be installed.

    See Also
    --------
    :ref:`testing.hypothesis`_
    """
    return st.recursive(
        st.dictionaries(_attr_keys, _attr_values),
        lambda children: st.dictionaries(_attr_keys, children),
        max_leaves=3,
    )


@st.composite
def variables(
    draw: st.DrawFn,
    *,
    array_strategy_fn: Union[ArrayStrategyFn, None] = None,
    dims: Union[
        st.SearchStrategy[Union[Sequence[Hashable], Mapping[Hashable, int]]],
        None,
    ] = None,
    dtype: st.SearchStrategy[np.dtype] = supported_dtypes(),
    attrs: st.SearchStrategy[Mapping] = attrs(),
) -> xr.Variable:
    """
    Generates arbitrary xarray.Variable objects.

    Follows the basic signature of the xarray.Variable constructor, but allows passing alternative strategies to
    generate either numpy-like array data or dimensions. Also allows specifying the shape or dtype of the wrapped array
    up front.

    Passing nothing will generate a completely arbitrary Variable (containing a numpy array).

    Requires the hypothesis package to be installed.

    Parameters
    ----------
    array_strategy_fn: Callable which returns a strategy generating array-likes, optional
        Callable must only accept shape and dtype kwargs, and must generate results consistent with its input.
        If not passed the default is to generate a small numpy array with one of the supported_dtypes.
    dims: Strategy for generating the dimensions, optional
        Can either be a strategy for generating a sequence of string dimension names,
        or a strategy for generating a mapping of string dimension names to integer lengths along each dimension.
        If provided as a mapping the array shape will be passed to array_strategy_fn.
        Default is to generate arbitrary dimension names for each axis in data.
    dtype: Strategy which generates np.dtype objects, optional
        Will be passed in to array_strategy_fn.
        Default is to generate any scalar dtype using supported_dtypes.
        Be aware that this default set of dtypes includes some not strictly allowed by the array API standard.
    attrs: Strategy which generates dicts, optional
        Default is to generate a nested attributes dictionary containing arbitrary strings, booleans, integers, Nones,
        and numpy arrays.

    Returns
    -------
    variable_strategy
        Strategy for generating xarray.Variable objects.

    Raises
    ------
    ValueError
        If a custom array_strategy_fn returns a strategy which generates an example array inconsistent with the shape
        & dtype input passed to it.

    Examples
    --------
    Generate completely arbitrary Variable objects backed by a numpy array:

    >>> variables().example()  # doctest: +SKIP
    <xarray.Variable (żō: 3)>
    array([43506,   -16,  -151], dtype=int32)
    >>> variables().example()  # doctest: +SKIP
    <xarray.Variable (eD: 4, ğŻżÂĕ: 2, T: 2)>
    array([[[-10000000., -10000000.],
            [-10000000., -10000000.]],
           [[-10000000., -10000000.],
            [        0., -10000000.]],
           [[        0., -10000000.],
            [-10000000.,        inf]],
           [[       -0., -10000000.],
            [-10000000.,        -0.]]], dtype=float32)
    Attributes:
        śřĴ:      {'ĉ': {'iĥf': array([-30117,  -1740], dtype=int16)}}

    Generate only Variable objects with certain dimension names:

    >>> variables(dims=st.just(["a", "b"])).example()  # doctest: +SKIP
    <xarray.Variable (a: 5, b: 3)>
    array([[       248, 4294967295, 4294967295],
           [2412855555, 3514117556, 4294967295],
           [       111, 4294967295, 4294967295],
           [4294967295, 1084434988,      51688],
           [     47714,        252,      11207]], dtype=uint32)

    Generate only Variable objects with certain dimension names and lengths:

    >>> variables(dims=st.just({"a": 2, "b": 1})).example()  # doctest: +SKIP
    <xarray.Variable (a: 2, b: 1)>
    array([[-1.00000000e+007+3.40282347e+038j],
           [-2.75034266e-225+2.22507386e-311j]])

    See Also
    --------
    :ref:`testing.hypothesis`_
    """

    if not isinstance(dims, st.SearchStrategy) and dims is not None:
        raise InvalidArgument(
            f"dims must be provided as a hypothesis.strategies.SearchStrategy object (or None), but got type {type(dims)}. "
            "To specify fixed contents, use hypothesis.strategies.just()."
        )
    if not isinstance(dtype, st.SearchStrategy) and dtype is not None:
        raise InvalidArgument(
            f"dtype must be provided as a hypothesis.strategies.SearchStrategy object (or None), but got type {type(dtype)}. "
            "To specify fixed contents, use hypothesis.strategies.just()."
        )
    if not isinstance(attrs, st.SearchStrategy) and attrs is not None:
        raise InvalidArgument(
            f"attrs must be provided as a hypothesis.strategies.SearchStrategy object (or None), but got type {type(attrs)}. "
            "To specify fixed contents, use hypothesis.strategies.just()."
        )

    _array_strategy_fn: ArrayStrategyFn
    if array_strategy_fn is None:
        # For some reason if I move the default value to the function signature definition mypy incorrectly says the ignore is no longer necessary, making it impossible to satisfy mypy
        _array_strategy_fn = npst.arrays  # type: ignore[assignment]  # npst.arrays has extra kwargs that we aren't using later
    elif not callable(array_strategy_fn):
        raise InvalidArgument(
            "array_strategy_fn must be a Callable that accepts the kwargs dtype and shape and returns a hypothesis "
            "strategy which generates corresponding array-like objects."
        )
    else:
        _array_strategy_fn = (
            array_strategy_fn  # satisfy mypy that this new variable cannot be None
        )

    _dtype = draw(dtype)

    if dims is not None:
        # generate dims first then draw data to match
        _dims = draw(dims)
        if isinstance(_dims, Sequence):
            dim_names = list(_dims)
            valid_shapes = npst.array_shapes(min_dims=len(_dims), max_dims=len(_dims))
            _shape = draw(valid_shapes)
            array_strategy = _array_strategy_fn(shape=_shape, dtype=_dtype)
        elif isinstance(_dims, (Mapping, dict)):
            # should be a mapping of form {dim_names: lengths}
            dim_names, _shape = list(_dims.keys()), tuple(_dims.values())
            array_strategy = _array_strategy_fn(shape=_shape, dtype=_dtype)
        else:
            raise InvalidArgument(
                f"Invalid type returned by dims strategy - drew an object of type {type(dims)}"
            )
    else:
        # nothing provided, so generate everything consistently
        # We still generate the shape first here just so that we always pass shape to array_strategy_fn
        _shape = draw(npst.array_shapes())
        array_strategy = _array_strategy_fn(shape=_shape, dtype=_dtype)
        dim_names = draw(dimension_names(min_dims=len(_shape), max_dims=len(_shape)))

    _data = draw(array_strategy)

    if _data.shape != _shape:
        raise ValueError(
            "array_strategy_fn returned an array object with a different shape than it was passed."
            f"Passed {_shape}, but returned {_data.shape}."
            "Please either specify a consistent shape via the dims kwarg or ensure the array_strategy_fn callable "
            "obeys the shape argument passed to it."
        )
    if _data.dtype != _dtype:
        raise ValueError(
            "array_strategy_fn returned an array object with a different dtype than it was passed."
            f"Passed {_dtype}, but returned {_data.dtype}"
            "Please either specify a consistent dtype via the dtype kwarg or ensure the array_strategy_fn callable "
            "obeys the dtype argument passed to it."
        )

    return xr.Variable(dims=dim_names, data=_data, attrs=draw(attrs))


@overload
def unique_subset_of(
    objs: Sequence[Hashable],
    *,
    min_size: int = 0,
    max_size: Union[int, None] = None,
) -> st.SearchStrategy[Sequence[Hashable]]: ...


@overload
def unique_subset_of(
    objs: Mapping[Hashable, Any],
    *,
    min_size: int = 0,
    max_size: Union[int, None] = None,
) -> st.SearchStrategy[Mapping[Hashable, Any]]: ...


@st.composite
def unique_subset_of(
    draw: st.DrawFn,
    objs: Union[Sequence[Hashable], Mapping[Hashable, Any]],
    *,
    min_size: int = 0,
    max_size: Union[int, None] = None,
) -> Union[Sequence[Hashable], Mapping[Hashable, Any]]:
    """
    Return a strategy which generates a unique subset of the given objects.

    Each entry in the output subset will be unique (if input was a sequence) or have a unique key (if it was a mapping).

    Requires the hypothesis package to be installed.

    Parameters
    ----------
    objs: Union[Sequence[Hashable], Mapping[Hashable, Any]]
        Objects from which to sample to produce the subset.
    min_size: int, optional
        Minimum size of the returned subset. Default is 0.
    max_size: int, optional
        Maximum size of the returned subset. Default is the full length of the input.
        If set to 0 the result will be an empty mapping.

    Returns
    -------
    unique_subset_strategy
        Strategy generating subset of the input.

    Examples
    --------
    >>> unique_subset_of({"x": 2, "y": 3}).example()  # doctest: +SKIP
    {'y': 3}
    >>> unique_subset_of(["x", "y"]).example()  # doctest: +SKIP
    ['x']

    See Also
    --------
    :ref:`testing.hypothesis`_
    """
    if not isinstance(objs, Iterable):
        raise TypeError(
            f"Object to sample from must be an Iterable or a Mapping, but received type {type(objs)}"
        )

    if len(objs) == 0:
        raise ValueError("Can't sample from a length-zero object.")

    keys = list(objs.keys()) if isinstance(objs, Mapping) else objs

    subset_keys = draw(
        st.lists(
            st.sampled_from(keys),
            unique=True,
            min_size=min_size,
            max_size=max_size,
        )
    )

    return (
        {k: objs[k] for k in subset_keys} if isinstance(objs, Mapping) else subset_keys
    )


@st.composite
def _alignable_variables(
    draw: st.DrawFn,
    *,
    var_names: st.SearchStrategy[str],
    dim_sizes: Mapping[Hashable, int],
) -> Mapping[Hashable, xr.Variable]:
    """
    Generates dicts of names mapping to variables with compatible (i.e. alignable) dimensions and sizes.
    """

    alignable_dim_sizes = draw(unique_subset_of(dim_sizes)) if dim_sizes else {}

    vars = variables(dims=st.just(alignable_dim_sizes))
    # TODO don't hard code max number of variables
    return draw(st.dictionaries(var_names, vars, max_size=3))


@st.composite
def coordinate_variables(
    draw: st.DrawFn,
    *,
    dim_sizes: Mapping[Hashable, int],
    coord_names: st.SearchStrategy[Hashable] = names(),
) -> Mapping[Hashable, xr.Variable]:
    """
    Generates dicts of alignable Variable objects for use as coordinates.

    Differs from data_variables strategy in that it deliberately creates dimension coordinates
    (i.e. 1D variables with the same name as a dimension) as well as non-dimension coordinates.

    Requires the hypothesis package to be installed.

    Parameters
    ----------
    dim_sizes: Mapping of str to int
        Sizes of dimensions to use for coordinates.
    coord_names: Strategy generating strings, optional
        Allowed names for non-dimension coordinates. Defaults to `names` strategy.
    """

    all_coords = {}

    if draw(
        st.booleans()
    ):  # Allow for no coordinate variables - explicit possibility not to helps with shrinking
        dim_names = list(dim_sizes.keys())

        # Possibly generate 1D "dimension coordinates" - explicit possibility not to helps with shrinking
        if len(dim_names) > 0 and draw(st.booleans()):
            # first generate subset of dimension names - these set which dimension coords will be included
            dim_coord_names_and_lengths = draw(unique_subset_of(dim_sizes))

            # then generate 1D variables for each name
            dim_coords = {
                n: draw(variables(dims=st.just({n: length})))
                for n, length in dim_coord_names_and_lengths.items()
            }
            all_coords.update(dim_coords)

        # Possibly generate ND "non-dimension coordinates" - explicit possibility not to helps with shrinking
        if draw(st.booleans()):
            # can't have same name as a dimension
            valid_non_dim_coord_names = coord_names.filter(lambda n: n not in dim_names)
            non_dim_coords = draw(
                _alignable_variables(
                    var_names=valid_non_dim_coord_names, dim_sizes=dim_sizes
                )
            )
            all_coords.update(non_dim_coords)

    return all_coords


def _sizes_from_dim_names(
    dims: Sequence[Hashable],
) -> st.SearchStrategy[dict[Hashable, int]]:
    size_along_dim = st.integers(min_value=1, max_value=6)
    return st.fixed_dictionaries({d: size_along_dim for d in dims})


@st.composite
def dataarrays(
    draw: st.DrawFn,
    *,
    data: st.SearchStrategy[T_DuckArray] = None,
    dims: st.SearchStrategy[Union[Sequence[Hashable], Mapping[Hashable, int]]] = None,
    name: st.SearchStrategy[Union[Hashable, None]] = names(),
    attrs: st.SearchStrategy[Mapping] = attrs(),
) -> xr.DataArray:
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
    dims: Strategy for generating the dimensions, optional
        Can either be a strategy for generating a sequence of string dimension names,
        or a strategy for generating a mapping of string dimension names to integer lengths along each dimension.
        If provided in the former form the lengths of the returned Variable will either be determined from the
        data argument if given or arbitrarily generated if not.
        Default is to generate arbitrary dimension sizes, or arbitrary dimension names for each axis in data.
    name: Strategy for generating a string name, optional
        Default is to use the `names` strategy, or to create an unnamed DataArray.
    attrs: Strategy which generates dicts, optional

    Raises
    ------
    hypothesis.errors.InvalidArgument
        If custom strategies passed try to draw examples which together cannot create a valid DataArray.
    """

    _name = draw(st.none() | name)

    # TODO add a coords argument?

    if data is not None and dims is None:
        # no dims -> generate dims to match data
        _data = draw(data)
        dim_names = draw(dimension_names(min_dims=_data.ndim, max_dims=_data.ndim))
        dim_sizes: Mapping[Hashable, int] = {
            n: length for n, length in zip(dim_names, _data.shape)
        }
        coords = draw(coordinate_variables(dim_sizes=dim_sizes))

    elif data is None and dims is not None:
        # no data -> generate data to match dims
        _dims = draw(dims)
        if isinstance(_dims, Sequence):
            dim_sizes = draw(_sizes_from_dim_names(_dims))
        elif isinstance(_dims, Mapping):
            # should be a mapping of form {dim_names: lengths}
            dim_sizes = _dims
        else:
            raise ValueError(f"Invalid type for dims argument - got type {type(_dims)}")

        dim_names, shape = list(dim_sizes.keys()), tuple(dim_sizes.values())
        _data = draw(np_arrays(shape=shape))
        coords = draw(coordinate_variables(dim_sizes=dim_sizes))

    elif data is not None and dims is not None:
        # both data and dims provided -> check drawn examples are compatible
        _dims = draw(dims)
        _data = draw(data)
        if isinstance(_dims, Sequence):
            dim_names = list(_dims)
            if _data.ndim != len(_dims):
                raise InvalidArgument(
                    f"Strategy attempting to generate data with {_data.ndim} dims but {len(_dims)} "
                    "unique dimension names. Please only pass strategies which are guaranteed to "
                    "draw compatible examples for data and dims."
                )
            dim_sizes = {n: length for n, length in zip(_dims, _data.shape)}
        elif isinstance(_dims, Mapping):
            # should be a mapping of form {dim_names: lengths}
            dim_sizes = _dims
            dim_names, shape = list(dim_sizes.keys()), tuple(dim_sizes.values())
            if _data.shape != shape:
                raise InvalidArgument(
                    f"Strategy attempting to generate data with shape {_data.shape} dims but dimension "
                    f"sizes implying shape {shape}. Please only pass strategies which are guaranteed to "
                    "draw compatible examples for data and dims."
                )
        else:
            raise ValueError(f"Invalid type for dims argument - got type {type(_dims)}")

        coords = draw(coordinate_variables(dim_sizes=dim_sizes))

    else:
        # nothing provided, so generate everything consistently by drawing dims to match data, and coords to match both
        _data = draw(np_arrays())
        dim_names = draw(dimension_names(min_dims=_data.ndim, max_dims=_data.ndim))
        dim_sizes = {n: length for n, length in zip(dim_names, _data.shape)}
        coords = draw(coordinate_variables(dim_sizes=dim_sizes))

    return xr.DataArray(
        data=_data,
        coords=coords,
        name=_name,
        dims=dim_names,
        attrs=draw(attrs),
    )


@st.composite
def data_variables(
    draw: st.DrawFn,
    *,
    dim_sizes: Mapping[Hashable, int],
    var_names: st.SearchStrategy[Hashable] = names(),
) -> Mapping[Hashable, xr.Variable]:
    """
    Generates dicts of alignable Variable objects for use as Dataset data variables.

    Requires the hypothesis package to be installed.

    Parameters
    ----------
    dim_sizes: Mapping of str to int
        Sizes of dimensions to use for variables.
    var_names: Strategy generating strings
        Allowed names for data variables. Needed to avoid conflict with names of coordinate variables & dimensions.
    """
    if draw(
        st.booleans()
    ):  # Allow for no coordinate variables - explicit possibility not to helps with shrinking
        dim_names = list(dim_sizes.keys())

        # can't have same name as a dimension
        # TODO this is also used in coordinate_variables so refactor it out into separate function
        valid_var_names = var_names.filter(lambda n: n not in dim_names)
        data_vars = draw(
            _alignable_variables(var_names=valid_var_names, dim_sizes=dim_sizes)
        )
    else:
        data_vars = {}

    return data_vars


@st.composite
def datasets(
    draw: st.DrawFn,
    *,
    data_vars: st.SearchStrategy[Mapping[Hashable, xr.Variable]] = None,
    dims: st.SearchStrategy[Union[Sequence[Hashable], Mapping[Hashable, int]]] = None,
    attrs: st.SearchStrategy[Mapping] = attrs(),
) -> xr.Dataset:
    """
    Generates arbitrary xarray.Dataset objects.

    Follows the basic signature of the xarray.Dataset constructor, but you can also pass alternative strategies to
    generate either numpy-like array data variables or dimensions.

    Passing nothing will generate a completely arbitrary Dataset (backed by numpy arrays).

    Requires the hypothesis package to be installed.

    Parameters
    ----------
    data_vars: Strategy generating mappings from variable names to xr.Variable objects, optional
        Default is to generate an arbitrary combination of compatible variables with sizes matching dims,
        but arbitrary names, dtypes, and values.
    dims: Strategy for generating the dimensions, optional
        Can either be a strategy for generating a sequence of string dimension names,
        or a strategy for generating a mapping of string dimension names to integer lengths along each dimension.
        If provided in the former form the lengths of the returned Variable will either be determined from the
        data argument if given or arbitrarily generated if not.
        Default is to generate arbitrary dimension sizes.
    attrs: Strategy which generates dicts, optional

    Raises
    ------
    hypothesis.errors.InvalidArgument
        If custom strategies passed try to draw examples which together cannot create a valid DataArray.
    """

    # TODO add a coords argument?

    if data_vars is not None and dims is None:
        # no dims -> generate dims to match data
        _data_vars = draw(data_vars)
        dim_sizes = _find_overall_sizes(_data_vars)
        # only draw coordinate variables whose names don't conflict with data variables
        allowed_coord_names = names().filter(lambda n: n not in list(_data_vars.keys()))
        coords = draw(
            coordinate_variables(coord_names=allowed_coord_names, dim_sizes=dim_sizes)
        )

    elif data_vars is None and dims is not None:
        # no data -> generate data to match dims
        _dims = draw(dims)
        if isinstance(_dims, Sequence):
            dim_sizes = draw(_sizes_from_dim_names(_dims))
        elif isinstance(_dims, Mapping):
            # should be a mapping of form {dim_names: lengths}
            dim_sizes = _dims
        else:
            raise ValueError(f"Invalid type for dims argument - got type {type(_dims)}")

        coords = draw(coordinate_variables(dim_sizes=dim_sizes))
        coord_names = list(coords.keys())
        allowed_data_var_names = names().filter(lambda n: n not in coord_names)
        _data_vars = draw(
            data_variables(dim_sizes=dim_sizes, var_names=allowed_data_var_names)
        )

    elif data_vars is not None and dims is not None:
        # both data and dims provided -> check drawn examples are compatible
        _dims = draw(dims)
        if isinstance(_dims, Sequence):
            # TODO support dims as list too?
            raise NotImplementedError()
        elif isinstance(_dims, Mapping):
            # should be a mapping of form {dim_names: lengths}
            dim_sizes = _dims
            _data_vars = draw(data_vars)
            _check_compatible_sizes(_data_vars, dim_sizes)
        else:
            raise ValueError(f"Invalid type for dims argument - got type {type(_dims)}")

        # only draw coordinate variables whose names don't conflict with data variables
        allowed_coord_names = names().filter(lambda n: n not in list(_data_vars.keys()))
        coords = draw(
            coordinate_variables(coord_names=allowed_coord_names, dim_sizes=dim_sizes)
        )

    else:
        # nothing provided, so generate everything consistently by drawing data to match dims, and coords to match both
        dim_sizes = draw(dimension_sizes())
        coords = draw(coordinate_variables(dim_sizes=dim_sizes))
        allowed_data_var_names = names().filter(lambda n: n not in list(coords.keys()))
        _data_vars = draw(
            data_variables(dim_sizes=dim_sizes, var_names=allowed_data_var_names)
        )

    return xr.Dataset(data_vars=_data_vars, coords=coords, attrs=draw(attrs))


def _find_overall_sizes(vars: Mapping[Hashable, xr.Variable]) -> Mapping[Hashable, int]:
    """Given a set of variables, find their common sizes."""
    # TODO raise an error if inconsistent (i.e. if different values appear under same key)
    # TODO narrow type by checking if values are not ints
    sizes_dicts = [v.sizes for v in vars.values()]
    dim_sizes = {d: s for dim_sizes in sizes_dicts for d, s in dim_sizes.items()}
    return dim_sizes


def _check_compatible_sizes(
    vars: Mapping[Hashable, xr.Variable], dim_sizes: Mapping[Hashable, int]
):
    """Check set of variables have sizes compatible with given dim_sizes. If not raise InvalidArgument error."""

    for name, v in vars.items():
        if not set(v.sizes.items()).issubset(set(dim_sizes.items())):
            raise InvalidArgument(
                f"Strategy attempting to generate object with dimension sizes {dim_sizes} but drawn "
                f"variable {name} has sizes {v.sizes}, which is incompatible."
                "Please only pass strategies which are guaranteed to draw compatible examples for data "
                "and dims."
            )
