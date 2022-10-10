"""Generate module and stub file for arithmetic operators of various xarray classes.

For internal xarray development use only.

Usage:
    python xarray/util/generate_cumulatives.py
    pytest --doctest-modules xarray/core/_cumulatives.py --accept || true
    pytest --doctest-modules xarray/core/_cumulatives.py

This requires [pytest-accept](https://github.com/max-sixty/pytest-accept).
The second run of pytest is deliberate, since the first will return an error
while replacing the doctests.

"""
import textwrap
from dataclasses import dataclass

MODULE_PREAMBLE = '''\
"""Mixin classes with cumulative operations."""
# This file was generated using xarray.util.generate_cumulatives. Do not edit manually.

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Sequence

from . import duck_array_ops
from .options import OPTIONS
from .types import Dims
from .utils import contains_only_dask_or_numpy

if TYPE_CHECKING:
    from .dataarray import DataArray
    from .dataset import Dataset

try:
    import flox
except ImportError:
    flox = None  # type: ignore'''


DEFAULT_PREAMBLE = """

class {obj}{cls}Cumulatives:
    __slots__ = ()

    def reduce(
        self,
        func: Callable[..., Any],
        dim: Dims = None,
        *,
        axis: int | Sequence[int] | None = None,
        keep_attrs: bool | None = None,
        keepdims: bool = False,
        **kwargs: Any,
    ) -> {obj}:
        raise NotImplementedError()"""


GROUPBY_PREAMBLE = """

class {obj}{cls}Cumulatives:
    _obj: {obj}

    def reduce(
        self,
        func: Callable[..., Any],
        dim: Dims | ellipsis = None,
        *,
        axis: int | Sequence[int] | None = None,
        skipna: bool | None = None,
        keep_attrs: bool | None = None,
        keepdims: bool = False,
        **kwargs: Any,
    ) -> {obj}:
        raise NotImplementedError()

    def _flox_reduce(
        self,
        dim: Dims | ellipsis,
        **kwargs: Any,
    ) -> {obj}:
        raise NotImplementedError()"""


TEMPLATE_REDUCTION_SIGNATURE = '''
    def {method}(
        self,
        dim: Dims = None,
        *,
        axis: int | Sequence[int] | None = None,
        skipna: bool | None = None,
        keep_attrs: bool | None = None,
        **kwargs: Any,
    ) -> {obj}:
        """
        Apply ``{method}`` along some dimension of {obj}{cls}.

        Parameters
        ----------'''

TEMPLATE_RETURNS = """
        Returns
        -------
        cumvalue : {obj}
            New {obj} object with `{method}` applied to its data along the
            indicated dimension.
        """

TEMPLATE_SEE_ALSO = """
        See Also
        --------
        numpy.{method}
        dask.array.{method}
        {see_also_obj}.{method}"""

TEMPLATE_NOTES = """
        Notes
        -----
        {notes}"""

_DIM_DOCSTRING = """dim: str or sequence of str, optional
    Dimension over which to apply `{method}`."""

_AXIS_DOCSTRING = """axis: int or sequence of int, optional
    Axis over which to apply `{method}`. Only one of the ‘dim‘ and ‘axis’ arguments can be supplied."""

_SKIPNA_DOCSTRING = """skipna : bool, optional
    If True, skip missing values (as marked by NaN). By default, only
    skips missing values for float dtypes; other dtypes either do not
    have a sentinel missing value (int) or skipna=True has not been
    implemented (object, datetime64 or timedelta64)."""

_KEEP_ATTRS_DOCSTRING = """keep_attrs : bool, optional
    If True, the attributes (`attrs`) will be copied from the original
    object to the new one.  If False (default), the new object will be
    returned without attributes."""

_KWARGS_DOCSTRING = """**kwargs : Any
    Additional keyword arguments passed on to `{method}`."""


class Method:
    def __init__(
        self,
        name,
    ):
        self.name = name
        self.array_method = name


class ReductionGenerator:

    _dim_docstring = _DIM_DOCSTRING
    _axis_docstring = _AXIS_DOCSTRING
    _template_signature = TEMPLATE_REDUCTION_SIGNATURE

    def __init__(
        self,
        cls,
        datastructure,
        methods,
        example_call_preamble,
        definition_preamble,
        see_also_obj=None,
    ):
        self.datastructure = datastructure
        self.cls = cls
        self.methods = methods
        self.example_call_preamble = example_call_preamble
        self.preamble = definition_preamble.format(obj=datastructure.name, cls=cls)
        if not see_also_obj:
            self.see_also_obj = self.datastructure.name
        else:
            self.see_also_obj = see_also_obj

    def generate_methods(self):
        yield [self.preamble]
        for method in self.methods:
            yield self.generate_method(method)

    def generate_method(self, method):
        template_kwargs = dict(obj=self.datastructure.name, method=method.name)
        yield self._template_signature.format(
            **template_kwargs,
            cls=self.cls,
        )

        for text in [
            self._dim_docstring.format(method=method.name),
            self._axis_docstring.format(method=method.name),
            _SKIPNA_DOCSTRING,
            _KEEP_ATTRS_DOCSTRING,
            _KWARGS_DOCSTRING.format(method=method.name),
        ]:
            if text:
                yield textwrap.indent(text, 8 * " ")
        yield TEMPLATE_RETURNS.format(**template_kwargs)

        yield TEMPLATE_SEE_ALSO.format(
            **template_kwargs,
            see_also_obj=self.see_also_obj,
        )

        yield textwrap.indent(self.generate_example(method=method), "")

        yield '        """'

        yield self.generate_code(method)

    def generate_example(self, method):
        create_da = """
        >>> temperature = np.arange(1.0, 17.0).reshape(4, 4)
        >>> temperature[2, 2] = np.nan
        >>> da = xr.DataArray(
        ...     temperature,
        ...     dims=["x", "y"],
        ...     coords=dict(
        ...         lon=("x", np.arange(10, 30, 5)),
        ...         lat=("y", np.arange(40, 60, 5)),
        ...         labels=("y", ["a", "a", "b", "c"]),
        ...     ),
        ... )
        """

        calculation = f"{self.datastructure.example_var_name}{self.example_call_preamble}.{method.name}"
        calculation_ds = f"{self.datastructure.example_var_name}{self.example_call_preamble}.{method.name}(){self.datastructure.example_var_key}"

        return f"""
        Examples
        --------
        {create_da}{self.datastructure.docstring_create}

        >>> {calculation}()
        >>> {calculation_ds}
        """


class GroupByReductionGenerator(ReductionGenerator):
    _dim_docstring = _DIM_DOCSTRING
    _axis_docstring = _AXIS_DOCSTRING
    _template_signature = TEMPLATE_REDUCTION_SIGNATURE

    def generate_code(self, method):

        return f"""\
        if flox and OPTIONS["use_flox"] and contains_only_dask_or_numpy(self._obj):
            return self._flox_reduce(
                func="{method.name}",
                dim=dim,
                # fill_value=fill_value,
                keep_attrs=keep_attrs,
                **kwargs,
            )
        else:
            return self.reduce(
                func=duck_array_ops.{method.array_method},
                dim=dim,
                axis=axis,
                skipna=skipna,
                keep_attrs=keep_attrs,
                **kwargs,
            )"""


class GenericReductionGenerator(ReductionGenerator):
    def generate_code(self, method):

        return f"""\
        return self.reduce(
            func=duck_array_ops.{method.array_method},
            dim=dim,
            axis=axis,
            skipna=skipna,
            keep_attrs=keep_attrs,
            **kwargs,
        )"""


CUM_METHODS = (
    Method(
        "cumsum",
    ),
    Method(
        "cumprod",
    ),
)

CUM_METHODS_GROUPBY = (
    Method(
        "cumsum",
    ),
)


@dataclass
class DataStructure:
    name: str
    docstring_create: str
    example_var_name: str
    example_var_key: str


DATASET_OBJECT = DataStructure(
    name="Dataset",
    docstring_create="""
        >>> ds = xr.Dataset(dict(da=da))
        >>> ds
        """,
    example_var_name="ds",
    example_var_key="['da']",
)

DATAARRAY_OBJECT = DataStructure(
    name="DataArray",
    docstring_create="""
        >>> da
        """,
    example_var_name="da",
    example_var_key="",
)

DATASET_GENERATOR = GenericReductionGenerator(
    cls="",
    datastructure=DATASET_OBJECT,
    methods=CUM_METHODS,
    example_call_preamble="",
    see_also_obj="DataArray",
    definition_preamble=DEFAULT_PREAMBLE,
)
DATAARRAY_GENERATOR = GenericReductionGenerator(
    cls="",
    datastructure=DATAARRAY_OBJECT,
    methods=CUM_METHODS,
    example_call_preamble="",
    see_also_obj="Dataset",
    definition_preamble=DEFAULT_PREAMBLE,
)
DATAARRAY_GROUPBY_GENERATOR = GroupByReductionGenerator(
    cls="GroupBy",
    datastructure=DATAARRAY_OBJECT,
    methods=CUM_METHODS_GROUPBY,
    example_call_preamble='.groupby("labels")',
    definition_preamble=GROUPBY_PREAMBLE,
)
DATASET_GROUPBY_GENERATOR = GroupByReductionGenerator(
    cls="GroupBy",
    datastructure=DATASET_OBJECT,
    methods=CUM_METHODS_GROUPBY,
    example_call_preamble='.groupby("labels")',
    definition_preamble=GROUPBY_PREAMBLE,
)


if __name__ == "__main__":
    import os
    from pathlib import Path

    p = Path(os.getcwd())
    filepath = p.parent / "xarray" / "xarray" / "core" / "_cumulatives.py"
    # filepath = p.parent / "core" / "_cumulatives.py"  # Run from script location
    with open(filepath, mode="w", encoding="utf-8") as f:
        f.write(MODULE_PREAMBLE + "\n")
        for gen in [
            DATASET_GENERATOR,
            DATAARRAY_GENERATOR,
            DATASET_GROUPBY_GENERATOR,
            DATAARRAY_GROUPBY_GENERATOR,
        ]:
            for lines in gen.generate_methods():
                for line in lines:
                    f.write(line + "\n")
