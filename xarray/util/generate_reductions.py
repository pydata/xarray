"""Generate module and stub file for arithmetic operators of various xarray classes.

For internal xarray development use only.

Usage:
    python xarray/util/generate_reductions.py > xarray/core/_reductions.py
    pytest --doctest-modules xarray/core/_reductions.py --accept || true
    pytest --doctest-modules xarray/core/_reductions.py --accept

This requires [pytest-accept](https://github.com/max-sixty/pytest-accept).
The second run of pytest is deliberate, since the first will return an error
while replacing the doctests.

"""

import collections
import textwrap
from functools import partial
from typing import Callable, Optional

MODULE_PREAMBLE = '''\
"""Mixin classes with reduction operations."""
# This file was generated using xarray.util.generate_reductions. Do not edit manually.

import sys
from typing import Any, Callable, Hashable, Optional, Sequence, Union

from . import duck_array_ops
from .types import T_DataArray, T_Dataset

if sys.version_info >= (3, 8):
    from typing import Protocol
else:
    from typing_extensions import Protocol'''

OBJ_PREAMBLE = """

class {obj}Reduce(Protocol):
    def reduce(
        self,
        func: Callable[..., Any],
        dim: Union[None, Hashable, Sequence[Hashable]] = None,
        axis: Union[None, int, Sequence[int]] = None,
        keep_attrs: bool = None,
        keepdims: bool = False,
        **kwargs: Any,
    ) -> T_{obj}:
        ..."""


CLASS_PREAMBLE = """

class {obj}{cls}Reductions:
    __slots__ = ()"""

_SKIPNA_DOCSTRING = """
skipna : bool, optional
    If True, skip missing values (as marked by NaN). By default, only
    skips missing values for float dtypes; other dtypes either do not
    have a sentinel missing value (int) or skipna=True has not been
    implemented (object, datetime64 or timedelta64)."""

_MINCOUNT_DOCSTRING = """
min_count : int, default: None
    The required number of valid values to perform the operation. If
    fewer than min_count non-NA values are present the result will be
    NA. Only used if skipna is set to True or defaults to True for the
    array's dtype. Changed in version 0.17.0: if specified on an integer
    array and skipna=True, the result will be a float array."""


BOOL_REDUCE_METHODS = ["all", "any"]
NAN_REDUCE_METHODS = [
    "max",
    "min",
    "mean",
    "prod",
    "sum",
    "std",
    "var",
    "median",
]
NAN_CUM_METHODS = ["cumsum", "cumprod"]
MIN_COUNT_METHODS = ["prod", "sum"]
NUMERIC_ONLY_METHODS = [
    "mean",
    "std",
    "var",
    "sum",
    "prod",
    "median",
    "cumsum",
    "cumprod",
]

TEMPLATE_REDUCTION = '''
    def {method}(
        self: {obj}Reduce,
        dim: Union[None, Hashable, Sequence[Hashable]] = None,{skip_na.kwarg}{min_count.kwarg}
        keep_attrs: bool = None,
        **kwargs,
    ) -> T_{obj}:
        """
        Reduce this {obj}'s data by applying ``{method}`` along some dimension(s).

        Parameters
        ----------
        dim : hashable or iterable of hashable, optional
            Name of dimension[s] along which to apply ``{method}``. For e.g. ``dim="x"``
            or ``dim=["x", "y"]``. {extra_dim}{extra_args}{skip_na.docs}{min_count.docs}
        keep_attrs : bool, optional
            If True, ``attrs`` will be copied from the original
            object to the new one.  If False (default), the new object will be
            returned without attributes.
        **kwargs : dict
            Additional keyword arguments passed on to the appropriate array
            function for calculating ``{method}`` on this object's data.

        Returns
        -------
        reduced : {obj}
            New {obj} with ``{method}`` applied to its data and the
            indicated dimension(s) removed

        Examples
        --------{example}

        See Also
        --------
        numpy.{method}
        {obj}.{method}
        :ref:`{docref}`
            User guide on {docref} operations.
        """
        return self.reduce(
            duck_array_ops.{array_method},
            dim=dim,{skip_na.call}{min_count.call}{numeric_only_call}
            keep_attrs=keep_attrs,
            **kwargs,
        )'''


def generate_groupby_example(obj: str, cls: str, method: str):
    """Generate examples for method."""
    dx = "ds" if obj == "Dataset" else "da"
    if cls == "Resample":
        calculation = f'{dx}.resample(time="3M").{method}'
    elif cls == "GroupBy":
        calculation = f'{dx}.groupby("labels").{method}'
    else:
        raise ValueError

    if method in BOOL_REDUCE_METHODS:
        np_array = """
        ...     np.array([True, True, True, True, True, False], dtype=bool),"""

    else:
        np_array = """
        ...     np.array([1, 2, 3, 1, 2, np.nan]),"""

    create_da = f"""
        >>> da = xr.DataArray({np_array}
        ...     dims="time",
        ...     coords=dict(
        ...         time=("time", pd.date_range("01-01-2001", freq="M", periods=6)),
        ...         labels=("time", np.array(["a", "b", "c", "c", "b", "a"])),
        ...     ),
        ... )"""

    if obj == "Dataset":
        maybe_dataset = """
        >>> ds = xr.Dataset(dict(da=da))
        >>> ds"""
    else:
        maybe_dataset = """
        >>> da"""

    if method in NAN_REDUCE_METHODS:
        maybe_skipna = f"""

        Use ``skipna`` to control whether NaNs are ignored.

        >>> {calculation}(skipna=False)"""
    else:
        maybe_skipna = ""

    if method in MIN_COUNT_METHODS:
        maybe_mincount = f"""

        Specify ``min_count`` for finer control over when NaNs are ignored.

        >>> {calculation}(skipna=True, min_count=2)"""
    else:
        maybe_mincount = ""

    return f"""{create_da}{maybe_dataset}

        >>> {calculation}(){maybe_skipna}{maybe_mincount}"""


def generate_method(
    obj: str,
    docref: str,
    method: str,
    skipna: bool,
    example_generator: Callable,
    array_method: Optional[str] = None,
):
    if not array_method:
        array_method = method

    if obj == "Dataset":
        if method in NUMERIC_ONLY_METHODS:
            numeric_only_call = "\n            numeric_only=True,"
        else:
            numeric_only_call = "\n            numeric_only=False,"
    else:
        numeric_only_call = ""

    kwarg = collections.namedtuple("kwarg", "docs kwarg call")
    if skipna:
        skip_na = kwarg(
            docs=textwrap.indent(_SKIPNA_DOCSTRING, "        "),
            kwarg="\n        skipna: bool = True,",
            call="\n            skipna=skipna,",
        )
    else:
        skip_na = kwarg(docs="", kwarg="", call="")

    if method in MIN_COUNT_METHODS:
        min_count = kwarg(
            docs=textwrap.indent(_MINCOUNT_DOCSTRING, "        "),
            kwarg="\n        min_count: Optional[int] = None,",
            call="\n            min_count=min_count,",
        )
    else:
        min_count = kwarg(docs="", kwarg="", call="")

    return TEMPLATE_REDUCTION.format(
        obj=obj,
        docref=docref,
        method=method,
        array_method=array_method,
        extra_dim="""If ``None``, will reduce over all dimensions
            present in the grouped variable.""",
        extra_args="",
        skip_na=skip_na,
        min_count=min_count,
        numeric_only_call=numeric_only_call,
        example=example_generator(obj=obj, method=method),
    )


def render(obj: str, cls: str, docref: str, example_generator: Callable):
    yield CLASS_PREAMBLE.format(obj=obj, cls=cls)
    yield generate_method(
        obj,
        method="count",
        docref=docref,
        skipna=False,
        example_generator=example_generator,
    )
    for method in BOOL_REDUCE_METHODS:
        yield generate_method(
            obj,
            method=method,
            docref=docref,
            skipna=False,
            array_method=f"array_{method}",
            example_generator=example_generator,
        )
    for method in NAN_REDUCE_METHODS:
        yield generate_method(
            obj,
            method=method,
            docref=docref,
            skipna=True,
            example_generator=example_generator,
        )


if __name__ == "__main__":
    print(MODULE_PREAMBLE)
    for obj in ["Dataset", "DataArray"]:
        print(OBJ_PREAMBLE.format(obj=obj))
        for cls, docref in (
            ("GroupBy", "groupby"),
            ("Resample", "resampling"),
        ):
            for line in render(
                obj=obj,
                cls=cls,
                docref=docref,
                example_generator=partial(generate_groupby_example, cls=cls),
            ):
                print(line)
