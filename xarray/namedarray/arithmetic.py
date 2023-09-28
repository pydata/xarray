from __future__ import annotations

import textwrap
import typing


class ImplementsArrayReduce:
    __slots__ = ()

    @classmethod
    def _reduce_method(
        cls, func: typing.Callable, include_skipna: bool, numeric_only: bool
    ):
        if include_skipna:

            def wrapped_func(self, dim=None, axis=None, skipna=None, **kwargs):
                return self.reduce(func, dim, axis, skipna=skipna, **kwargs)

        else:

            def wrapped_func(self, dim=None, axis=None, **kwargs):  # type: ignore[misc]
                return self.reduce(func, dim, axis, **kwargs)

        return wrapped_func

    _reduce_extra_args_docstring = textwrap.dedent(
        """\
        dim : str or sequence of str, optional
            Dimension(s) over which to apply `{name}`.
        axis : int or sequence of int, optional
            Axis(es) over which to apply `{name}`. Only one of the 'dim'
            and 'axis' arguments can be supplied. If neither are supplied, then
            `{name}` is calculated over axes."""
    )

    _cum_extra_args_docstring = textwrap.dedent(
        """\
        dim : str or sequence of str, optional
            Dimension over which to apply `{name}`.
        axis : int or sequence of int, optional
            Axis over which to apply `{name}`. Only one of the 'dim'
            and 'axis' arguments can be supplied."""
    )


class IncludeReduceMethods:
    __slots__ = ()


class IncludeCumMethods:
    ...


class IncludeNumpySameMethods:
    ...


class SupportsArithmetic:
    ...


class NamedArrayOpsMixin:
    ...


class NamedArrayArithmetic(
    # ImplementsArrayReduce,
    # IncludeReduceMethods,
    IncludeCumMethods,
    IncludeNumpySameMethods,
    SupportsArithmetic,
    NamedArrayOpsMixin,
):
    __slots__ = ()
    # prioritize our operations over those of numpy.ndarray (priority=0)
    __array_priority__ = 50
