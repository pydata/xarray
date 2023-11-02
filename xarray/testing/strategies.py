from collections.abc import Hashable, Mapping, Sequence
from typing import Any, Protocol, Union

import hypothesis.extra.numpy as npst
import hypothesis.strategies as st
import numpy as np

import xarray as xr
from xarray.core.types import T_DuckArray

__all__ = [
    "numeric_dtypes",
    "names",
    "dimension_names",
    "dimension_sizes",
    "attrs",
    "variables",
]


class ArrayStrategyFn(Protocol):
    def __call__(
        self, *, shape: tuple[int, ...] = None, dtype: np.dtype = None, **kwargs
    ) -> st.SearchStrategy[T_DuckArray]:
        ...


# required to exclude weirder dtypes e.g. unicode, byte_string, array, or nested dtypes.
def numeric_dtypes() -> st.SearchStrategy[np.dtype]:
    """
    Generates only those numpy dtypes which xarray can handle.

    Requires the hypothesis package to be installed.
    """

    return (
        npst.integer_dtypes()
        | npst.unsigned_integer_dtypes()
        | npst.floating_dtypes()
        | npst.complex_number_dtypes()
    )


def np_arrays(
    *,
    shape: Union[
        tuple[int, ...], st.SearchStrategy[tuple[int, ...]]
    ] = npst.array_shapes(max_side=4),
    dtype: Union[np.dtype, st.SearchStrategy[np.dtype]] = numeric_dtypes(),
) -> st.SearchStrategy[np.ndarray]:
    """
    Generates arbitrary numpy arrays with xarray-compatible dtypes.

    Requires the hypothesis package to be installed.

    Parameters
    ----------
    shape
    dtype
        Default is to use any of the numeric_dtypes defined for xarray.
    """

    return npst.arrays(dtype=dtype, shape=shape)


# TODO Generalize to all valid unicode characters once formatting bugs in xarray's reprs are fixed + docs can handle it.
_readable_characters = st.characters(
    categories=["L", "N"], max_codepoint=0x017F
)  # only use characters within the "Latin Extended-A" subset of unicode


def names() -> st.SearchStrategy[str]:
    """
    Generates arbitrary string names for dimensions / variables.

    Requires the hypothesis package to be installed.
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
    max_side: int = None,
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
_small_arrays = np_arrays(
    shape=npst.array_shapes(
        max_side=2,
        max_dims=2,
    )
)
_attr_values = st.none() | st.booleans() | _readable_strings | _small_arrays


def attrs() -> st.SearchStrategy[Mapping[Hashable, Any]]:
    """
    Generates arbitrary valid attributes dictionaries for xarray objects.

    The generated dictionaries can potentially be recursive.

    Requires the hypothesis package to be installed.
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
    array_strategy_fn: ArrayStrategyFn = None,
    dims: st.SearchStrategy[Union[Sequence[Hashable], Mapping[Hashable, int]]] = None,
    dtype: st.SearchStrategy[np.dtype] = numeric_dtypes(),
    attrs: st.SearchStrategy[Mapping] = attrs(),
) -> xr.Variable:
    """
    Generates arbitrary xarray.Variable objects.

    Follows the basic signature of the xarray.Variable constructor, but allows passing alternative strategies to
    generate either numpy-like array data or dimensions.

    Passing nothing will generate a completely arbitrary Variable (containing a numpy array).

    Requires the hypothesis package to be installed.

    Parameters
    ----------
    array_strategy_fn: Callable which returns a strategy generating array-likes, optional
        Callable must accept shape and dtype kwargs if passed.
        If array_strategy_fn is not passed then the shapes will be derived from the dims kwarg.
        Default is to generate numpy data of arbitrary shape, values and dtype.
    dims: Strategy for generating the dimensions, optional
        Can either be a strategy for generating a sequence of string dimension names,
        or a strategy for generating a mapping of string dimension names to integer lengths along each dimension.
        If provided in the former form the lengths of the returned Variable will either be determined from the
        data argument if given or arbitrarily generated if not.
        Default is to generate arbitrary dimension names for each axis in data.
    dtype: Strategy which generates np.dtype objects, optional
    attrs: Strategy which generates dicts, optional

    Returns
    -------
    variable_strategy
        Strategy for generating xarray.Variable objects.

    Raises
    ------
    TypeError
        If custom strategies passed try to draw examples which are not of the correct type.

    Examples
    --------
    Generate completely arbitrary Variable objects backed by a numpy array:

    >>> variables().example()
    <xarray.Variable (żō: 3)>
    array([43506,   -16,  -151], dtype=int32)
    >>> variables().example()
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
        śřĴ:      {'ĉ': {'iĥſ': array([-30117,  -1740], dtype=int16)}}

    Generate only Variable objects with certain dimension names:

    >>> variables(dims=st.just(["a", "b"])).example()
    <xarray.Variable (a: 5, b: 3)>
    array([[       248, 4294967295, 4294967295],
           [2412855555, 3514117556, 4294967295],
           [       111, 4294967295, 4294967295],
           [4294967295, 1084434988,      51688],
           [     47714,        252,      11207]], dtype=uint32)

    Generate only Variable objects with certain dimension names and lengths:

    >>> variables(dims=st.just({"a": 2, "b": 1})).example()
    <xarray.Variable (a: 2, b: 1)>
    array([[-1.00000000e+007+3.40282347e+038j],
           [-2.75034266e-225+2.22507386e-311j]])

    Generate completely arbitrary Variable objects backed by a sparse array:

    >>> from hypothesis.extra.array_api import make_strategies_namespace
    >>> import cupy as cp
    >>> cupy_strategy_fn = make_strategies_namespace(cp).arrays
    >>> cupy_da = variables(array_strategy_fn=cupy_strategy_fn).example()
    >>> cupy_da
    <xarray.Variable (c: 3, d: 1)>
    array([[ 0.,  1.,  2.],
           [ 3.,  4.,  5.]], dtype=float32)
    >>> cupy_da.data.device
    <CUDA Device 0>
    """

    if any(
        not isinstance(arg, st.SearchStrategy) and arg is not None
        for arg in [dims, dtype, attrs]
    ):
        raise TypeError(
            "Contents must be provided as a hypothesis.strategies.SearchStrategy object (or None)."
            "To specify fixed contents, use hypothesis.strategies.just()."
        )

    if not array_strategy_fn:
        array_strategy_fn = np_arrays

    if dims is not None:
        # generate dims first then draw data to match
        _dims = draw(dims)
        if isinstance(_dims, Sequence):
            dim_names = list(_dims)
            valid_shapes = npst.array_shapes(min_dims=len(_dims), max_dims=len(_dims))
            array_strategy = array_strategy_fn(
                shape=draw(valid_shapes), dtype=draw(dtype)
            )
        elif isinstance(_dims, Mapping):
            # should be a mapping of form {dim_names: lengths}
            dim_names, shape = list(_dims.keys()), tuple(_dims.values())
            array_strategy = array_strategy_fn(shape=shape, dtype=draw(dtype))
        else:
            raise TypeError(
                f"Invalid type returned by dims strategy - drew an object of type {type(dims)}"
            )
        _data = draw(array_strategy)

    else:
        # nothing provided, so generate everything consistently by drawing dims to match data
        array_strategy = array_strategy_fn(dtype=draw(dtype))
        _data = draw(array_strategy)
        dim_names = draw(dimension_names(min_dims=_data.ndim, max_dims=_data.ndim))

    return xr.Variable(dims=dim_names, data=_data, attrs=draw(attrs))
