# The StringAccessor class defined below is an adaptation of the
# pandas string methods source code (see pd.core.strings)

# For reference, here is a copy of the pandas copyright notice:

# (c) 2011-2012, Lambda Foundry, Inc. and PyData Development Team
# All rights reserved.

# Copyright (c) 2008-2011 AQR Capital Management, LLC
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:

#     * Redistributions of source code must retain the above copyright
#        notice, this list of conditions and the following disclaimer.

#     * Redistributions in binary form must reproduce the above
#        copyright notice, this list of conditions and the following
#        disclaimer in the documentation and/or other materials provided
#        with the distribution.

#     * Neither the name of the copyright holder nor the names of any
#        contributors may be used to endorse or promote products derived
#        from this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import codecs
import re
import textwrap
from functools import reduce
from operator import or_ as set_union
from typing import (
    Any,
    Callable,
    Hashable,
    Mapping,
    Optional,
    Pattern,
    Tuple,
    Type,
    Union,
)
from unicodedata import normalize

import numpy as np

from .computation import apply_ufunc

_cpython_optimized_encoders = (
    "utf-8",
    "utf8",
    "latin-1",
    "latin1",
    "iso-8859-1",
    "mbcs",
    "ascii",
)
_cpython_optimized_decoders = _cpython_optimized_encoders + ("utf-16", "utf-32")


def _contains_obj_type(*, pat: Any, checker: Any) -> bool:
    """Determine if the object fits some rule or is array of objects that do so."""
    if isinstance(checker, type):
        targtype = checker
        checker = lambda x: isinstance(x, targtype)

    if checker(pat):
        return True

    # If it is not an object array it can't contain compiled re
    if getattr(pat, "dtype", "no") != np.object_:
        return False

    return _apply_str_ufunc(func=checker, obj=pat).all()


def _contains_str_like(pat: Any) -> bool:
    """Determine if the object is a str-like or array of str-like."""
    if isinstance(pat, (str, bytes)):
        return True

    if not hasattr(pat, "dtype"):
        return False

    return pat.dtype.kind in ["U", "S"]


def _contains_compiled_re(pat: Any) -> bool:
    """Determine if the object is a compiled re or array of compiled re."""
    return _contains_obj_type(pat=pat, checker=re.Pattern)


def _contains_callable(pat: Any) -> bool:
    """Determine if the object is a callable or array of callables."""
    return _contains_obj_type(pat=pat, checker=callable)


def _apply_str_ufunc(
    *,
    func: Callable,
    obj: Any,
    dtype: Union[str, np.dtype, Type] = None,
    output_core_dims: Union[list, tuple] = ((),),
    output_sizes: Mapping[Any, int] = None,
    func_args: Tuple = (),
    func_kwargs: Mapping = {},
) -> Any:
    # TODO handling of na values ?
    if dtype is None:
        dtype = obj.dtype

    dask_gufunc_kwargs = dict()
    if output_sizes is not None:
        dask_gufunc_kwargs["output_sizes"] = output_sizes

    return apply_ufunc(
        func,
        obj,
        *func_args,
        vectorize=True,
        dask="parallelized",
        output_dtypes=[dtype],
        output_core_dims=output_core_dims,
        dask_gufunc_kwargs=dask_gufunc_kwargs,
        **func_kwargs,
    )


class StringAccessor:
    r"""Vectorized string functions for string-like arrays.

    Similar to pandas, fields can be accessed through the `.str` attribute
    for applicable DataArrays.

        >>> da = xr.DataArray(["some", "text", "in", "an", "array"])
        >>> da.str.len()
        <xarray.DataArray (dim_0: 5)>
        array([4, 4, 2, 2, 5])
        Dimensions without coordinates: dim_0

    It also implements ``+``, ``*``, and ``%``, which operate as elementwise
    versions of the corresponding ``str`` methods. These will automatically
    broadcast for array-like inputs.

        >>> da1 = xr.DataArray(["first", "second", "third"], dims=["X"])
        >>> da2 = xr.DataArray([1, 2, 3], dims=["Y"])
        >>> da1.str + da2
        <xarray.DataArray (X: 3, Y: 3)>
        array([['first1', 'first2', 'first3'],
               ['second1', 'second2', 'second3'],
               ['third1', 'third2', 'third3']], dtype='<U7')
        Dimensions without coordinates: X, Y

        >>> da1 = xr.DataArray(["a", "b", "c", "d"], dims=["X"])
        >>> reps = xr.DataArray([3, 4], dims=["Y"])
        >>> da1.str * reps
        <xarray.DataArray (X: 4, Y: 2)>
        array([['aaa', 'aaaa'],
               ['bbb', 'bbbb'],
               ['ccc', 'cccc'],
               ['ddd', 'dddd']], dtype='<U4')
        Dimensions without coordinates: X, Y

        >>> da1 = xr.DataArray(["%s_%s", "%s-%s", "%s|%s"], dims=["X"])
        >>> da2 = xr.DataArray([1, 2], dims=["Y"])
        >>> da3 = xr.DataArray([0.1, 0.2], dims=["Z"])
        >>> da1.str % (da2, da3)
        <xarray.DataArray (X: 3, Y: 2, Z: 2)>
        array([[['1_0.1', '1_0.2'],
                ['2_0.1', '2_0.2']],
        <BLANKLINE>
               [['1-0.1', '1-0.2'],
                ['2-0.1', '2-0.2']],
        <BLANKLINE>
               [['1|0.1', '1|0.2'],
                ['2|0.1', '2|0.2']]], dtype='<U5')
        Dimensions without coordinates: X, Y, Z

    .. note::
        When using ``%`` formatting with a dict, the values are always used as a
        single value, they are not applied elementwise.

            >>> da1 = xr.DataArray(["%(a)s"], dims=["X"])
            >>> da2 = xr.DataArray([1, 2, 3], dims=["Y"])
            >>> da1 % {"a": da2}
            <xarray.DataArray (X: 1)>
            array(['<xarray.DataArray (Y: 3)>\narray([1, 2, 3])\nDimensions without coordinates: Y'],
                  dtype=object)
            Dimensions without coordinates: X
    """

    __slots__ = ("_obj",)

    def __init__(self, obj):
        self._obj = obj

    def _stringify(
        self,
        invar: Any,
    ) -> Union[str, bytes, Any]:
        """
        Convert a string-like to the correct string/bytes type.

        This is mostly here to tell mypy a pattern is a str/bytes not a re.Pattern.
        """
        if hasattr(invar, "astype"):
            return invar.astype(self._obj.dtype.kind)
        else:
            return self._obj.dtype.type(invar)

    def _apply(
        self,
        *,
        func: Callable,
        dtype: Union[str, np.dtype, Type] = None,
        output_core_dims: Union[list, tuple] = ((),),
        output_sizes: Mapping[Any, int] = None,
        func_args: Tuple = (),
        func_kwargs: Mapping = {},
    ) -> Any:
        return _apply_str_ufunc(
            obj=self._obj,
            func=func,
            dtype=dtype,
            output_core_dims=output_core_dims,
            output_sizes=output_sizes,
            func_args=func_args,
            func_kwargs=func_kwargs,
        )

    def _re_compile(
        self,
        *,
        pat: Union[str, bytes, Pattern, Any],
        flags: int = 0,
        case: bool = None,
    ) -> Union[Pattern, Any]:
        is_compiled_re = isinstance(pat, re.Pattern)

        if is_compiled_re and flags != 0:
            raise ValueError("Flags cannot be set when pat is a compiled regex.")

        if is_compiled_re and case is not None:
            raise ValueError("Case cannot be set when pat is a compiled regex.")

        if is_compiled_re:
            # no-op, needed to tell mypy this isn't a string
            return re.compile(pat)

        if case is None:
            case = True

        # The case is handled by the re flags internally.
        # Add it to the flags if necessary.
        if not case:
            flags |= re.IGNORECASE

        if getattr(pat, "dtype", None) != np.object_:
            pat = self._stringify(pat)

        def func(x):
            return re.compile(x, flags=flags)

        if isinstance(pat, np.ndarray):
            # apply_ufunc doesn't work for numpy arrays with output object dtypes
            func_ = np.vectorize(func)
            return func_(pat)
        else:
            return _apply_str_ufunc(func=func, obj=pat, dtype=np.object_)

    def len(self) -> Any:
        """
        Compute the length of each string in the array.

        Returns
        -------
        lengths array : array of int
        """
        return self._apply(func=len, dtype=int)

    def __getitem__(
        self,
        key: Union[int, slice],
    ) -> Any:
        if isinstance(key, slice):
            return self.slice(start=key.start, stop=key.stop, step=key.step)
        else:
            return self.get(key)

    def __add__(
        self,
        other: Any,
    ) -> Any:
        return self.cat(other, sep="")

    def __mul__(
        self,
        num: Union[int, Any],
    ) -> Any:
        return self.repeat(num)

    def __mod__(
        self,
        other: Any,
    ) -> Any:
        if isinstance(other, dict):
            other = {key: self._stringify(val) for key, val in other.items()}
            return self._apply(func=lambda x: x % other)
        elif isinstance(other, tuple):
            other = tuple(self._stringify(x) for x in other)
            return self._apply(func=lambda x, *y: x % y, func_args=other)
        else:
            return self._apply(func=lambda x, y: x % y, func_args=(other,))

    def get(
        self,
        i: Union[int, Any],
        default: Union[str, bytes] = "",
    ) -> Any:
        """
        Extract character number `i` from each string in the array.

        If `i` is array-like, they are broadcast against the array and
        applied elementwise.

        Parameters
        ----------
        i : int or array-like of int
            Position of element to extract.
            If array-like, it is broadcast.
        default : optional
            Value for out-of-range index. If not specified (None) defaults to
            an empty string.

        Returns
        -------
        items : array of object
        """

        def f(x, iind):
            islice = slice(-1, None) if iind == -1 else slice(iind, iind + 1)
            item = x[islice]

            return item if item else default

        return self._apply(func=f, func_args=(i,))

    def slice(
        self,
        start: Union[int, Any] = None,
        stop: Union[int, Any] = None,
        step: Union[int, Any] = None,
    ) -> Any:
        """
        Slice substrings from each string in the array.

        If `start`, `stop`, or 'step` is array-like, they are broadcast
        against the array and applied elementwise.

        Parameters
        ----------
        start : int or array-like of int, optional
            Start position for slice operation.
            If array-like, it is broadcast.
        stop : int or array-like of int, optional
            Stop position for slice operation.
            If array-like, it is broadcast.
        step : int or array-like of int, optional
            Step size for slice operation.
            If array-like, it is broadcast.

        Returns
        -------
        sliced strings : same type as values
        """
        f = lambda x, istart, istop, istep: x[slice(istart, istop, istep)]
        return self._apply(func=f, func_args=(start, stop, step))

    def slice_replace(
        self,
        start: Union[int, Any] = None,
        stop: Union[int, Any] = None,
        repl: Union[str, bytes, Any] = "",
    ) -> Any:
        """
        Replace a positional slice of a string with another value.

        If `start`, `stop`, or 'repl` is array-like, they are broadcast
        against the array and applied elementwise.

        Parameters
        ----------
        start : int or array-like of int, optional
            Left index position to use for the slice. If not specified (None),
            the slice is unbounded on the left, i.e. slice from the start
            of the string. If array-like, it is broadcast.
        stop : int or array-like of int, optional
            Right index position to use for the slice. If not specified (None),
            the slice is unbounded on the right, i.e. slice until the
            end of the string. If array-like, it is broadcast.
        repl : str or array-like of str, optional
            String for replacement. If not specified, the sliced region
            is replaced with an empty string. If array-like, it is broadcast.

        Returns
        -------
        replaced : same type as values
        """
        repl = self._stringify(repl)

        def func(x, istart, istop, irepl):
            if len(x[istart:istop]) == 0:
                local_stop = istart
            else:
                local_stop = istop
            y = self._stringify("")
            if istart is not None:
                y += x[:istart]
            y += irepl
            if istop is not None:
                y += x[local_stop:]
            return y

        return self._apply(func=func, func_args=(start, stop, repl))

    def cat(
        self,
        *others,
        sep: Union[str, bytes, Any] = "",
    ) -> Any:
        """
        Concatenate strings elementwise in the DataArray with other strings.

        The other strings can either be string scalars or other array-like.
        Dimensions are automatically broadcast together.

        An optional separator `sep` can also be specified. If `sep` is
        array-like, it is broadcast against the array and applied elementwise.

        Parameters
        ----------
        *others : str or array-like of str
            Strings or array-like of strings to concatenate elementwise with
            the current DataArray.
        sep : str or array-like of str, default: "".
            Separator to use between strings.
            It is broadcast in the same way as the other input strings.
            If array-like, its dimensions will be placed at the end of the output array dimensions.

        Returns
        -------
        concatenated : same type as values

        Examples
        --------
        Create a string array

        >>> myarray = xr.DataArray(
        ...     ["11111", "4"],
        ...     dims=["X"],
        ... )

        Create some arrays to concatenate with it

        >>> values_1 = xr.DataArray(
        ...     ["a", "bb", "cccc"],
        ...     dims=["Y"],
        ... )
        >>> values_2 = np.array(3.4)
        >>> values_3 = ""
        >>> values_4 = np.array("test", dtype=np.unicode_)

        Determine the separator to use

        >>> seps = xr.DataArray(
        ...     [" ", ", "],
        ...     dims=["ZZ"],
        ... )

        Concatenate the arrays using the separator

        >>> myarray.str.cat(values_1, values_2, values_3, values_4, sep=seps)
        <xarray.DataArray (X: 2, Y: 3, ZZ: 2)>
        array([[['11111 a 3.4  test', '11111, a, 3.4, , test'],
                ['11111 bb 3.4  test', '11111, bb, 3.4, , test'],
                ['11111 cccc 3.4  test', '11111, cccc, 3.4, , test']],
        <BLANKLINE>
               [['4 a 3.4  test', '4, a, 3.4, , test'],
                ['4 bb 3.4  test', '4, bb, 3.4, , test'],
                ['4 cccc 3.4  test', '4, cccc, 3.4, , test']]], dtype='<U24')
        Dimensions without coordinates: X, Y, ZZ

        See Also
        --------
        pandas.Series.str.cat
        str.join
        """
        sep = self._stringify(sep)
        others = tuple(self._stringify(x) for x in others)
        others = others + (sep,)

        # sep will go at the end of the input arguments.
        func = lambda *x: x[-1].join(x[:-1])

        return self._apply(
            func=func,
            func_args=others,
            dtype=self._obj.dtype.kind,
        )

    def join(
        self,
        dim: Hashable = None,
        sep: Union[str, bytes, Any] = "",
    ) -> Any:
        """
        Concatenate strings in a DataArray along a particular dimension.

        An optional separator `sep` can also be specified. If `sep` is
        array-like, it is broadcast against the array and applied elementwise.

        Parameters
        ----------
        dim : hashable, optional
            Dimension along which the strings should be concatenated.
            Only one dimension is allowed at a time.
            Optional for 0D or 1D DataArrays, required for multidimensional DataArrays.
        sep : str or array-like, default: "".
            Separator to use between strings.
            It is broadcast in the same way as the other input strings.
            If array-like, its dimensions will be placed at the end of the output array dimensions.

        Returns
        -------
        joined : same type as values

        Examples
        --------
        Create an array

        >>> values = xr.DataArray(
        ...     [["a", "bab", "abc"], ["abcd", "", "abcdef"]],
        ...     dims=["X", "Y"],
        ... )

        Determine the separator

        >>> seps = xr.DataArray(
        ...     ["-", "_"],
        ...     dims=["ZZ"],
        ... )

        Join the strings along a given dimension

        >>> values.str.join(dim="Y", sep=seps)
        <xarray.DataArray (X: 2, ZZ: 2)>
        array([['a-bab-abc', 'a_bab_abc'],
               ['abcd--abcdef', 'abcd__abcdef']], dtype='<U12')
        Dimensions without coordinates: X, ZZ

        See Also
        --------
        pandas.Series.str.join
        str.join
        """
        if self._obj.ndim > 1 and dim is None:
            raise ValueError("Dimension must be specified for multidimensional arrays.")

        if self._obj.ndim > 1:
            # Move the target dimension to the start and split along it
            dimshifted = list(self._obj.transpose(dim, ...))
        elif self._obj.ndim == 1:
            dimshifted = list(self._obj)
        else:
            dimshifted = [self._obj]

        start, *others = dimshifted

        # concatenate the resulting arrays
        return start.str.cat(*others, sep=sep)

    def format(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """
        Perform python string formatting on each element of the DataArray.

        This is equivalent to calling `str.format` on every element of the
        DataArray. The replacement values can either be a string-like
        scalar or array-like of string-like values. If array-like,
        the values will be broadcast and applied elementwiseto the input
        DataArray.

        .. note::
            Array-like values provided as `*args` will have their
            dimensions added even if those arguments are not used in any
            string formatting.

        .. warning::
            Array-like arguments are only applied elementwise for `*args`.
            For `**kwargs`, values are used as-is.

        Parameters
        ----------
        *args : str or bytes or array-like of str or bytes
            Values for positional formatting.
            If array-like, the values are broadcast and applied elementwise.
            The dimensions will be placed at the end of the output array dimensions
            in the order they are provided.
        **kwargs : str or bytes or array-like of str or bytes
            Values for keyword-based formatting.
            These are **not** broadcast or applied elementwise.

        Returns
        -------
        formatted : same type as values

        Examples
        --------
        Create an array to format.

        >>> values = xr.DataArray(
        ...     ["{} is {adj0}", "{} and {} are {adj1}"],
        ...     dims=["X"],
        ... )

        Set the values to fill.

        >>> noun0 = xr.DataArray(
        ...     ["spam", "egg"],
        ...     dims=["Y"],
        ... )
        >>> noun1 = xr.DataArray(
        ...     ["lancelot", "arthur"],
        ...     dims=["ZZ"],
        ... )
        >>> adj0 = "unexpected"
        >>> adj1 = "like a duck"

        Insert the values into the array

        >>> values.str.format(noun0, noun1, adj0=adj0, adj1=adj1)
        <xarray.DataArray (X: 2, Y: 2, ZZ: 2)>
        array([[['spam is unexpected', 'spam is unexpected'],
                ['egg is unexpected', 'egg is unexpected']],
        <BLANKLINE>
               [['spam and lancelot are like a duck',
                 'spam and arthur are like a duck'],
                ['egg and lancelot are like a duck',
                 'egg and arthur are like a duck']]], dtype='<U33')
        Dimensions without coordinates: X, Y, ZZ

        See Also
        --------
        str.format
        """
        args = tuple(self._stringify(x) for x in args)
        kwargs = {key: self._stringify(val) for key, val in kwargs.items()}
        func = lambda x, *args, **kwargs: self._obj.dtype.type.format(
            x, *args, **kwargs
        )
        return self._apply(func=func, func_args=args, func_kwargs={"kwargs": kwargs})

    def capitalize(self) -> Any:
        """
        Convert strings in the array to be capitalized.

        Returns
        -------
        capitalized : same type as values
        """
        return self._apply(func=lambda x: x.capitalize())

    def lower(self) -> Any:
        """
        Convert strings in the array to lowercase.

        Returns
        -------
        lowerd : same type as values
        """
        return self._apply(func=lambda x: x.lower())

    def swapcase(self) -> Any:
        """
        Convert strings in the array to be swapcased.

        Returns
        -------
        swapcased : same type as values
        """
        return self._apply(func=lambda x: x.swapcase())

    def title(self) -> Any:
        """
        Convert strings in the array to titlecase.

        Returns
        -------
        titled : same type as values
        """
        return self._apply(func=lambda x: x.title())

    def upper(self) -> Any:
        """
        Convert strings in the array to uppercase.

        Returns
        -------
        uppered : same type as values
        """
        return self._apply(func=lambda x: x.upper())

    def casefold(self) -> Any:
        """
        Convert strings in the array to be casefolded.

        Casefolding is similar to converting to lowercase,
        but removes all case distinctions.
        This is important in some languages that have more complicated
        cases and case conversions.

        Returns
        -------
        casefolded : same type as values
        """
        return self._apply(func=lambda x: x.casefold())

    def normalize(
        self,
        form: str,
    ) -> Any:
        """
        Return the Unicode normal form for the strings in the datarray.

        For more information on the forms, see the documentation for
        :func:`unicodedata.normalize`.

        Parameters
        ----------
        form : {"NFC", "NFKC", "NFD", "NFKD"}
            Unicode form.

        Returns
        -------
        normalized : same type as values

        """
        return self._apply(func=lambda x: normalize(form, x))

    def isalnum(self) -> Any:
        """
        Check whether all characters in each string are alphanumeric.

        Returns
        -------
        isalnum : array of bool
            Array of boolean values with the same shape as the original array.
        """
        return self._apply(func=lambda x: x.isalnum(), dtype=bool)

    def isalpha(self) -> Any:
        """
        Check whether all characters in each string are alphabetic.

        Returns
        -------
        isalpha : array of bool
            Array of boolean values with the same shape as the original array.
        """
        return self._apply(func=lambda x: x.isalpha(), dtype=bool)

    def isdecimal(self) -> Any:
        """
        Check whether all characters in each string are decimal.

        Returns
        -------
        isdecimal : array of bool
            Array of boolean values with the same shape as the original array.
        """
        return self._apply(func=lambda x: x.isdecimal(), dtype=bool)

    def isdigit(self) -> Any:
        """
        Check whether all characters in each string are digits.

        Returns
        -------
        isdigit : array of bool
            Array of boolean values with the same shape as the original array.
        """
        return self._apply(func=lambda x: x.isdigit(), dtype=bool)

    def islower(self) -> Any:
        """
        Check whether all characters in each string are lowercase.

        Returns
        -------
        islower : array of bool
            Array of boolean values with the same shape as the original array.
        """
        return self._apply(func=lambda x: x.islower(), dtype=bool)

    def isnumeric(self) -> Any:
        """
        Check whether all characters in each string are numeric.

        Returns
        -------
        isnumeric : array of bool
            Array of boolean values with the same shape as the original array.
        """
        return self._apply(func=lambda x: x.isnumeric(), dtype=bool)

    def isspace(self) -> Any:
        """
        Check whether all characters in each string are spaces.

        Returns
        -------
        isspace : array of bool
            Array of boolean values with the same shape as the original array.
        """
        return self._apply(func=lambda x: x.isspace(), dtype=bool)

    def istitle(self) -> Any:
        """
        Check whether all characters in each string are titlecase.

        Returns
        -------
        istitle : array of bool
            Array of boolean values with the same shape as the original array.
        """
        return self._apply(func=lambda x: x.istitle(), dtype=bool)

    def isupper(self) -> Any:
        """
        Check whether all characters in each string are uppercase.

        Returns
        -------
        isupper : array of bool
            Array of boolean values with the same shape as the original array.
        """
        return self._apply(func=lambda x: x.isupper(), dtype=bool)

    def count(
        self,
        pat: Union[str, bytes, Pattern, Any],
        flags: int = 0,
        case: bool = None,
    ) -> Any:
        """
        Count occurrences of pattern in each string of the array.

        This function is used to count the number of times a particular regex
        pattern is repeated in each of the string elements of the
        :class:`~xarray.DataArray`.

        The pattern `pat` can either be a single ``str`` or `re.Pattern` or
        array-like of ``str`` or `re.Pattern`. If array-like, it is broadcast
        against the array and applied elementwise.

        Parameters
        ----------
        pat : str or re.Pattern or array-like of str or re.Pattern
            A string containing a regular expression or a compiled regular
            expression object. If array-like, it is broadcast.
        flags : int, default: 0
            Flags to pass through to the re module, e.g. `re.IGNORECASE`.
            see `compilation-flags <https://docs.python.org/3/howto/regex.html#compilation-flags>`_.
            ``0`` means no flags. Flags can be combined with the bitwise or operator ``|``.
            Cannot be set if `pat` is a compiled regex.
        case : bool, default: True
            If True, case sensitive.
            Cannot be set if `pat` is a compiled regex.
            Equivalent to setting the `re.IGNORECASE` flag.

        Returns
        -------
        counts : array of int
        """
        pat = self._re_compile(pat=pat, flags=flags, case=case)

        func = lambda x, ipat: len(ipat.findall(x))
        return self._apply(func=func, func_args=(pat,), dtype=int)

    def startswith(
        self,
        pat: Union[str, bytes, Any],
    ) -> Any:
        """
        Test if the start of each string in the array matches a pattern.

        The pattern `pat` can either be a ``str`` or array-like of ``str``.
        If array-like, it will be broadcast and applied elementwise.

        Parameters
        ----------
        pat : str
            Character sequence. Regular expressions are not accepted.
            If array-like, it is broadcast.

        Returns
        -------
        startswith : array of bool
            An array of booleans indicating whether the given pattern matches
            the start of each string element.
        """
        pat = self._stringify(pat)
        func = lambda x, y: x.startswith(y)
        return self._apply(func=func, func_args=(pat,), dtype=bool)

    def endswith(
        self,
        pat: Union[str, bytes, Any],
    ) -> Any:
        """
        Test if the end of each string in the array matches a pattern.

        The pattern `pat` can either be a ``str`` or array-like of ``str``.
        If array-like, it will be broadcast and applied elementwise.

        Parameters
        ----------
        pat : str
            Character sequence. Regular expressions are not accepted.
            If array-like, it is broadcast.

        Returns
        -------
        endswith : array of bool
            A Series of booleans indicating whether the given pattern matches
            the end of each string element.
        """
        pat = self._stringify(pat)
        func = lambda x, y: x.endswith(y)
        return self._apply(func=func, func_args=(pat,), dtype=bool)

    def pad(
        self,
        width: Union[int, Any],
        side: str = "left",
        fillchar: Union[str, bytes, Any] = " ",
    ) -> Any:
        """
        Pad strings in the array up to width.

        If `width` or 'fillchar` is array-like, they are broadcast
        against the array and applied elementwise.

        Parameters
        ----------
        width : int or array-like of int
            Minimum width of resulting string; additional characters will be
            filled with character defined in ``fillchar``.
            If array-like, it is broadcast.
        side : {"left", "right", "both"}, default: "left"
            Side from which to fill resulting string.
        fillchar : str or array-like of str, default: " "
            Additional character for filling, default is a space.
            If array-like, it is broadcast.

        Returns
        -------
        filled : same type as values
            Array with a minimum number of char in each element.
        """
        if side == "left":
            func = self.rjust
        elif side == "right":
            func = self.ljust
        elif side == "both":
            func = self.center
        else:  # pragma: no cover
            raise ValueError("Invalid side")

        return func(width=width, fillchar=fillchar)

    def _padder(
        self,
        *,
        func: Callable,
        width: Union[int, Any],
        fillchar: Union[str, bytes, Any] = " ",
    ) -> Any:
        """
        Wrapper function to handle padding operations
        """
        fillchar = self._stringify(fillchar)

        def overfunc(x, iwidth, ifillchar):
            if len(ifillchar) != 1:
                raise TypeError("fillchar must be a character, not str")
            return func(x, int(iwidth), ifillchar)

        return self._apply(func=overfunc, func_args=(width, fillchar))

    def center(
        self,
        width: Union[int, Any],
        fillchar: Union[str, bytes, Any] = " ",
    ) -> Any:
        """
        Pad left and right side of each string in the array.

        If `width` or 'fillchar` is array-like, they are broadcast
        against the array and applied elementwise.

        Parameters
        ----------
        width : int or array-like of int
            Minimum width of resulting string; additional characters will be
            filled with ``fillchar``. If array-like, it is broadcast.
        fillchar : str or array-like of str, default: " "
            Additional character for filling, default is a space.
            If array-like, it is broadcast.

        Returns
        -------
        filled : same type as values
        """
        func = self._obj.dtype.type.center
        return self._padder(func=func, width=width, fillchar=fillchar)

    def ljust(
        self,
        width: Union[int, Any],
        fillchar: Union[str, bytes, Any] = " ",
    ) -> Any:
        """
        Pad right side of each string in the array.

        If `width` or 'fillchar` is array-like, they are broadcast
        against the array and applied elementwise.

        Parameters
        ----------
        width : int or array-like of int
            Minimum width of resulting string; additional characters will be
            filled with ``fillchar``. If array-like, it is broadcast.
        fillchar : str or array-like of str, default: " "
            Additional character for filling, default is a space.
            If array-like, it is broadcast.

        Returns
        -------
        filled : same type as values
        """
        func = self._obj.dtype.type.ljust
        return self._padder(func=func, width=width, fillchar=fillchar)

    def rjust(
        self,
        width: Union[int, Any],
        fillchar: Union[str, bytes, Any] = " ",
    ) -> Any:
        """
        Pad left side of each string in the array.

        If `width` or 'fillchar` is array-like, they are broadcast
        against the array and applied elementwise.

        Parameters
        ----------
        width : int or array-like of int
            Minimum width of resulting string; additional characters will be
            filled with ``fillchar``. If array-like, it is broadcast.
        fillchar : str or array-like of str, default: " "
            Additional character for filling, default is a space.
            If array-like, it is broadcast.

        Returns
        -------
        filled : same type as values
        """
        func = self._obj.dtype.type.rjust
        return self._padder(func=func, width=width, fillchar=fillchar)

    def zfill(self, width: Union[int, Any]) -> Any:
        """
        Pad each string in the array by prepending '0' characters.

        Strings in the array are padded with '0' characters on the
        left of the string to reach a total string length  `width`. Strings
        in the array with length greater or equal to `width` are unchanged.

        If `width` is array-like, it is broadcast against the array and applied
        elementwise.

        Parameters
        ----------
        width : int or array-like of int
            Minimum length of resulting string; strings with length less
            than `width` be prepended with '0' characters. If array-like, it is broadcast.

        Returns
        -------
        filled : same type as values
        """
        return self.rjust(width, fillchar="0")

    def contains(
        self,
        pat: Union[str, bytes, Pattern, Any],
        case: bool = None,
        flags: int = 0,
        regex: bool = True,
    ) -> Any:
        """
        Test if pattern or regex is contained within each string of the array.

        Return boolean array based on whether a given pattern or regex is
        contained within a string of the array.

        The pattern `pat` can either be a single ``str`` or `re.Pattern` or
        array-like of ``str`` or `re.Pattern`. If array-like, it is broadcast
        against the array and applied elementwise.

        Parameters
        ----------
        pat : str or re.Pattern or array-like of str or re.Pattern
            Character sequence, a string containing a regular expression,
            or a compiled regular expression object. If array-like, it is broadcast.
        case : bool, default: True
            If True, case sensitive.
            Cannot be set if `pat` is a compiled regex.
            Equivalent to setting the `re.IGNORECASE` flag.
        flags : int, default: 0
            Flags to pass through to the re module, e.g. `re.IGNORECASE`.
            see `compilation-flags <https://docs.python.org/3/howto/regex.html#compilation-flags>`_.
            ``0`` means no flags. Flags can be combined with the bitwise or operator ``|``.
            Cannot be set if `pat` is a compiled regex.
        regex : bool, default: True
            If True, assumes the pat is a regular expression.
            If False, treats the pat as a literal string.
            Cannot be set to `False` if `pat` is a compiled regex.

        Returns
        -------
        contains : array of bool
            An array of boolean values indicating whether the
            given pattern is contained within the string of each element
            of the array.
        """
        is_compiled_re = _contains_compiled_re(pat)
        if is_compiled_re and not regex:
            raise ValueError(
                "Must use regular expression matching for regular expression object."
            )

        if regex:
            if not is_compiled_re:
                pat = self._re_compile(pat=pat, flags=flags, case=case)

            def func(x, ipat):
                if ipat.groups > 0:  # pragma: no cover
                    raise ValueError("This pattern has match groups.")
                return bool(ipat.search(x))

        else:
            pat = self._stringify(pat)
            if case or case is None:
                func = lambda x, ipat: ipat in x
            elif self._obj.dtype.char == "U":
                uppered = self._obj.str.casefold()
                uppat = StringAccessor(pat).casefold()
                return uppered.str.contains(uppat, regex=False)
            else:
                uppered = self._obj.str.upper()
                uppat = StringAccessor(pat).upper()
                return uppered.str.contains(uppat, regex=False)

        return self._apply(func=func, func_args=(pat,), dtype=bool)

    def match(
        self,
        pat: Union[str, bytes, Pattern, Any],
        case: bool = None,
        flags: int = 0,
    ) -> Any:
        """
        Determine if each string in the array matches a regular expression.

        The pattern `pat` can either be a single ``str`` or `re.Pattern` or
        array-like of ``str`` or `re.Pattern`. If array-like, it is broadcast
        against the array and applied elementwise.

        Parameters
        ----------
        pat : str or re.Pattern or array-like of str or re.Pattern
            A string containing a regular expression or
            a compiled regular expression object. If array-like, it is broadcast.
        case : bool, default: True
            If True, case sensitive.
            Cannot be set if `pat` is a compiled regex.
            Equivalent to setting the `re.IGNORECASE` flag.
        flags : int, default: 0
            Flags to pass through to the re module, e.g. `re.IGNORECASE`.
            see `compilation-flags <https://docs.python.org/3/howto/regex.html#compilation-flags>`_.
            ``0`` means no flags. Flags can be combined with the bitwise or operator ``|``.
            Cannot be set if `pat` is a compiled regex.

        Returns
        -------
        matched : array of bool
        """
        pat = self._re_compile(pat=pat, flags=flags, case=case)

        func = lambda x, ipat: bool(ipat.match(x))
        return self._apply(func=func, func_args=(pat,), dtype=bool)

    def strip(
        self,
        to_strip: Union[str, bytes, Any] = None,
        side: str = "both",
    ) -> Any:
        """
        Remove leading and trailing characters.

        Strip whitespaces (including newlines) or a set of specified characters
        from each string in the array from left and/or right sides.

        `to_strip` can either be a ``str`` or array-like of ``str``.
        If array-like, it will be broadcast and applied elementwise.

        Parameters
        ----------
        to_strip : str or array-like of str or None, default: None
            Specifying the set of characters to be removed.
            All combinations of this set of characters will be stripped.
            If None then whitespaces are removed. If array-like, it is broadcast.
        side : {"left", "right", "both"}, default: "both"
            Side from which to strip.

        Returns
        -------
        stripped : same type as values
        """
        if to_strip is not None:
            to_strip = self._stringify(to_strip)

        if side == "both":
            func = lambda x, y: x.strip(y)
        elif side == "left":
            func = lambda x, y: x.lstrip(y)
        elif side == "right":
            func = lambda x, y: x.rstrip(y)
        else:  # pragma: no cover
            raise ValueError("Invalid side")

        return self._apply(func=func, func_args=(to_strip,))

    def lstrip(
        self,
        to_strip: Union[str, bytes, Any] = None,
    ) -> Any:
        """
        Remove leading characters.

        Strip whitespaces (including newlines) or a set of specified characters
        from each string in the array from the left side.

        `to_strip` can either be a ``str`` or array-like of ``str``.
        If array-like, it will be broadcast and applied elementwise.

        Parameters
        ----------
        to_strip : str or array-like of str or None, default: None
            Specifying the set of characters to be removed.
            All combinations of this set of characters will be stripped.
            If None then whitespaces are removed. If array-like, it is broadcast.

        Returns
        -------
        stripped : same type as values
        """
        return self.strip(to_strip, side="left")

    def rstrip(
        self,
        to_strip: Union[str, bytes, Any] = None,
    ) -> Any:
        """
        Remove trailing characters.

        Strip whitespaces (including newlines) or a set of specified characters
        from each string in the array from the right side.

        `to_strip` can either be a ``str`` or array-like of ``str``.
        If array-like, it will be broadcast and applied elementwise.

        Parameters
        ----------
        to_strip : str or array-like of str or None, default: None
            Specifying the set of characters to be removed.
            All combinations of this set of characters will be stripped.
            If None then whitespaces are removed. If array-like, it is broadcast.

        Returns
        -------
        stripped : same type as values
        """
        return self.strip(to_strip, side="right")

    def wrap(
        self,
        width: Union[int, Any],
        **kwargs,
    ) -> Any:
        """
        Wrap long strings in the array in paragraphs with length less than `width`.

        This method has the same keyword parameters and defaults as
        :class:`textwrap.TextWrapper`.

        If `width` is array-like, it is broadcast against the array and applied
        elementwise.

        Parameters
        ----------
        width : int or array-like of int
            Maximum line-width.
            If array-like, it is broadcast.
        **kwargs
            keyword arguments passed into :class:`textwrap.TextWrapper`.

        Returns
        -------
        wrapped : same type as values
        """
        ifunc = lambda x: textwrap.TextWrapper(width=x, **kwargs)
        tw = StringAccessor(width)._apply(func=ifunc, dtype=np.object_)
        func = lambda x, itw: "\n".join(itw.wrap(x))
        return self._apply(func=func, func_args=(tw,))

    def translate(
        self,
        table: Mapping[Union[str, bytes], Union[str, bytes]],
    ) -> Any:
        """
        Map characters of each string through the given mapping table.

        Parameters
        ----------
        table : dict
            A a mapping of Unicode ordinals to Unicode ordinals, strings,
            or None. Unmapped characters are left untouched. Characters mapped
            to None are deleted. :meth:`str.maketrans` is a helper function for
            making translation tables.

        Returns
        -------
        translated : same type as values
        """
        func = lambda x: x.translate(table)
        return self._apply(func=func)

    def repeat(
        self,
        repeats: Union[int, Any],
    ) -> Any:
        """
        Repeat each string in the array.

        If `repeats` is array-like, it is broadcast against the array and applied
        elementwise.

        Parameters
        ----------
        repeats : int or array-like of int
            Number of repetitions.
            If array-like, it is broadcast.

        Returns
        -------
        repeated : same type as values
            Array of repeated string objects.
        """
        func = lambda x, y: x * y
        return self._apply(func=func, func_args=(repeats,))

    def find(
        self,
        sub: Union[str, bytes, Any],
        start: Union[int, Any] = 0,
        end: Union[int, Any] = None,
        side: str = "left",
    ) -> Any:
        """
        Return lowest or highest indexes in each strings in the array
        where the substring is fully contained between [start:end].
        Return -1 on failure.

        If `start`, `end`, or 'sub` is array-like, they are broadcast
        against the array and applied elementwise.

        Parameters
        ----------
        sub : str or array-like of str
            Substring being searched.
            If array-like, it is broadcast.
        start : int or array-like of int
            Left edge index.
            If array-like, it is broadcast.
        end : int or array-like of int
            Right edge index.
            If array-like, it is broadcast.
        side : {"left", "right"}, default: "left"
            Starting side for search.

        Returns
        -------
        found : array of int
        """
        sub = self._stringify(sub)

        if side == "left":
            method = "find"
        elif side == "right":
            method = "rfind"
        else:  # pragma: no cover
            raise ValueError("Invalid side")

        func = lambda x, isub, istart, iend: getattr(x, method)(isub, istart, iend)
        return self._apply(func=func, func_args=(sub, start, end), dtype=int)

    def rfind(
        self,
        sub: Union[str, bytes, Any],
        start: Union[int, Any] = 0,
        end: Union[int, Any] = None,
    ) -> Any:
        """
        Return highest indexes in each strings in the array
        where the substring is fully contained between [start:end].
        Return -1 on failure.

        If `start`, `end`, or 'sub` is array-like, they are broadcast
        against the array and applied elementwise.

        Parameters
        ----------
        sub : str or array-like of str
            Substring being searched.
            If array-like, it is broadcast.
        start : int or array-like of int
            Left edge index.
            If array-like, it is broadcast.
        end : int or array-like of int
            Right edge index.
            If array-like, it is broadcast.

        Returns
        -------
        found : array of int
        """
        return self.find(sub, start=start, end=end, side="right")

    def index(
        self,
        sub: Union[str, bytes, Any],
        start: Union[int, Any] = 0,
        end: Union[int, Any] = None,
        side: str = "left",
    ) -> Any:
        """
        Return lowest or highest indexes in each strings where the substring is
        fully contained between [start:end]. This is the same as
        ``str.find`` except instead of returning -1, it raises a ValueError
        when the substring is not found.

        If `start`, `end`, or 'sub` is array-like, they are broadcast
        against the array and applied elementwise.

        Parameters
        ----------
        sub : str or array-like of str
            Substring being searched.
            If array-like, it is broadcast.
        start : int or array-like of int
            Left edge index.
            If array-like, it is broadcast.
        end : int or array-like of int
            Right edge index.
            If array-like, it is broadcast.
        side : {"left", "right"}, default: "left"
            Starting side for search.

        Returns
        -------
        found : array of int

        Raises
        ------
        ValueError
            substring is not found
        """
        sub = self._stringify(sub)

        if side == "left":
            method = "index"
        elif side == "right":
            method = "rindex"
        else:  # pragma: no cover
            raise ValueError("Invalid side")

        func = lambda x, isub, istart, iend: getattr(x, method)(isub, istart, iend)
        return self._apply(func=func, func_args=(sub, start, end), dtype=int)

    def rindex(
        self,
        sub: Union[str, bytes, Any],
        start: Union[int, Any] = 0,
        end: Union[int, Any] = None,
    ) -> Any:
        """
        Return highest indexes in each strings where the substring is
        fully contained between [start:end]. This is the same as
        ``str.rfind`` except instead of returning -1, it raises a ValueError
        when the substring is not found.

        If `start`, `end`, or 'sub` is array-like, they are broadcast
        against the array and applied elementwise.

        Parameters
        ----------
        sub : str or array-like of str
            Substring being searched.
            If array-like, it is broadcast.
        start : int or array-like of int
            Left edge index.
            If array-like, it is broadcast.
        end : int or array-like of int
            Right edge index.
            If array-like, it is broadcast.

        Returns
        -------
        found : array of int

        Raises
        ------
        ValueError
            substring is not found
        """
        return self.index(sub, start=start, end=end, side="right")

    def replace(
        self,
        pat: Union[str, bytes, Pattern, Any],
        repl: Union[str, bytes, Callable, Any],
        n: Union[int, Any] = -1,
        case: bool = None,
        flags: int = 0,
        regex: bool = True,
    ) -> Any:
        """
        Replace occurrences of pattern/regex in the array with some string.

        If `pat`, `repl`, or 'n` is array-like, they are broadcast
        against the array and applied elementwise.

        Parameters
        ----------
        pat : str or re.Pattern or array-like of str or re.Pattern
            String can be a character sequence or regular expression.
            If array-like, it is broadcast.
        repl : str or callable or array-like of str or callable
            Replacement string or a callable. The callable is passed the regex
            match object and must return a replacement string to be used.
            See :func:`re.sub`.
            If array-like, it is broadcast.
        n : int or array of int, default: -1
            Number of replacements to make from start. Use ``-1`` to replace all.
            If array-like, it is broadcast.
        case : bool, default: True
            If True, case sensitive.
            Cannot be set if `pat` is a compiled regex.
            Equivalent to setting the `re.IGNORECASE` flag.
        flags : int, default: 0
            Flags to pass through to the re module, e.g. `re.IGNORECASE`.
            see `compilation-flags <https://docs.python.org/3/howto/regex.html#compilation-flags>`_.
            ``0`` means no flags. Flags can be combined with the bitwise or operator ``|``.
            Cannot be set if `pat` is a compiled regex.
        regex : bool, default: True
            If True, assumes the passed-in pattern is a regular expression.
            If False, treats the pattern as a literal string.
            Cannot be set to False if `pat` is a compiled regex or `repl` is
            a callable.

        Returns
        -------
        replaced : same type as values
            A copy of the object with all matching occurrences of `pat`
            replaced by `repl`.
        """
        if _contains_str_like(repl):
            repl = self._stringify(repl)
        elif not _contains_callable(repl):  # pragma: no cover
            raise TypeError("repl must be a string or callable")

        is_compiled_re = _contains_compiled_re(pat)
        if not regex and is_compiled_re:
            raise ValueError(
                "Cannot use a compiled regex as replacement pattern with regex=False"
            )

        if not regex and callable(repl):
            raise ValueError("Cannot use a callable replacement when regex=False")

        if regex:
            pat = self._re_compile(pat=pat, flags=flags, case=case)
            func = lambda x, ipat, irepl, i_n: ipat.sub(
                repl=irepl, string=x, count=i_n if i_n >= 0 else 0
            )
        else:
            pat = self._stringify(pat)
            func = lambda x, ipat, irepl, i_n: x.replace(ipat, irepl, i_n)
        return self._apply(func=func, func_args=(pat, repl, n))

    def extract(
        self,
        pat: Union[str, bytes, Pattern, Any],
        dim: Hashable,
        case: bool = None,
        flags: int = 0,
    ) -> Any:
        r"""
        Extract the first match of capture groups in the regex pat as a new
        dimension in a DataArray.

        For each string in the DataArray, extract groups from the first match
        of regular expression pat.

        If `pat` is array-like, it is broadcast against the array and applied
        elementwise.

        Parameters
        ----------
        pat : str or re.Pattern or array-like of str or re.Pattern
            A string containing a regular expression or a compiled regular
            expression object. If array-like, it is broadcast.
        dim : hashable or None
            Name of the new dimension to store the captured strings in.
            If None, the pattern must have only one capture group and the
            resulting DataArray will have the same size as the original.
        case : bool, default: True
            If True, case sensitive.
            Cannot be set if `pat` is a compiled regex.
            Equivalent to setting the `re.IGNORECASE` flag.
        flags : int, default: 0
            Flags to pass through to the re module, e.g. `re.IGNORECASE`.
            see `compilation-flags <https://docs.python.org/3/howto/regex.html#compilation-flags>`_.
            ``0`` means no flags. Flags can be combined with the bitwise or operator ``|``.
            Cannot be set if `pat` is a compiled regex.

        Returns
        -------
        extracted : same type as values or object array

        Raises
        ------
        ValueError
            `pat` has no capture groups.
        ValueError
            `dim` is None and there is more than one capture group.
        ValueError
            `case` is set when `pat` is a compiled regular expression.
        KeyError
            The given dimension is already present in the DataArray.

        Examples
        --------
        Create a string array

        >>> value = xr.DataArray(
        ...     [
        ...         [
        ...             "a_Xy_0",
        ...             "ab_xY_10-bab_Xy_110-baab_Xy_1100",
        ...             "abc_Xy_01-cbc_Xy_2210",
        ...         ],
        ...         [
        ...             "abcd_Xy_-dcd_Xy_33210-dccd_Xy_332210",
        ...             "",
        ...             "abcdef_Xy_101-fef_Xy_5543210",
        ...         ],
        ...     ],
        ...     dims=["X", "Y"],
        ... )

        Extract matches

        >>> value.str.extract(r"(\w+)_Xy_(\d*)", dim="match")
        <xarray.DataArray (X: 2, Y: 3, match: 2)>
        array([[['a', '0'],
                ['bab', '110'],
                ['abc', '01']],
        <BLANKLINE>
               [['abcd', ''],
                ['', ''],
                ['abcdef', '101']]], dtype='<U6')
        Dimensions without coordinates: X, Y, match

        See Also
        --------
        DataArray.str.extractall
        DataArray.str.findall
        re.compile
        re.search
        pandas.Series.str.extract
        """
        pat = self._re_compile(pat=pat, flags=flags, case=case)

        if isinstance(pat, re.Pattern):
            maxgroups = pat.groups
        else:
            maxgroups = (
                _apply_str_ufunc(obj=pat, func=lambda x: x.groups, dtype=np.int_)
                .max()
                .data.tolist()
            )

        if maxgroups == 0:
            raise ValueError("No capture groups found in pattern.")

        if dim is None and maxgroups != 1:
            raise ValueError(
                "Dimension must be specified if more than one capture group is given."
            )

        if dim is not None and dim in self._obj.dims:
            raise KeyError(f"Dimension '{dim}' already present in DataArray.")

        def _get_res_single(val, pat):
            match = pat.search(val)
            if match is None:
                return ""
            res = match.group(1)
            if res is None:
                res = ""
            return res

        def _get_res_multi(val, pat):
            match = pat.search(val)
            if match is None:
                return np.array([""], val.dtype)
            match = match.groups()
            match = [grp if grp is not None else "" for grp in match]
            return np.array(match, val.dtype)

        if dim is None:
            return self._apply(func=_get_res_single, func_args=(pat,))
        else:
            # dtype MUST be object or strings can be truncated
            # See: https://github.com/numpy/numpy/issues/8352
            return self._apply(
                func=_get_res_multi,
                func_args=(pat,),
                dtype=np.object_,
                output_core_dims=[[dim]],
                output_sizes={dim: maxgroups},
            ).astype(self._obj.dtype.kind)

    def extractall(
        self,
        pat: Union[str, bytes, Pattern, Any],
        group_dim: Hashable,
        match_dim: Hashable,
        case: bool = None,
        flags: int = 0,
    ) -> Any:
        r"""
        Extract all matches of capture groups in the regex pat as new
        dimensions in a DataArray.

        For each string in the DataArray, extract groups from all matches
        of regular expression pat.
        Equivalent to applying re.findall() to all the elements in the DataArray
        and splitting the results across dimensions.

        If `pat` is array-like, it is broadcast against the array and applied
        elementwise.

        Parameters
        ----------
        pat : str or re.Pattern
            A string containing a regular expression or a compiled regular
            expression object. If array-like, it is broadcast.
        group_dim : hashable
            Name of the new dimensions corresponding to the capture groups.
            This dimension is added to the new DataArray first.
        match_dim : hashable
            Name of the new dimensions corresponding to the matches for each group.
            This dimension is added to the new DataArray second.
        case : bool, default: True
            If True, case sensitive.
            Cannot be set if `pat` is a compiled regex.
            Equivalent to setting the `re.IGNORECASE` flag.
        flags : int, default: 0
            Flags to pass through to the re module, e.g. `re.IGNORECASE`.
            see `compilation-flags <https://docs.python.org/3/howto/regex.html#compilation-flags>`_.
            ``0`` means no flags. Flags can be combined with the bitwise or operator ``|``.
            Cannot be set if `pat` is a compiled regex.

        Returns
        -------
        extracted : same type as values or object array

        Raises
        ------
        ValueError
            `pat` has no capture groups.
        ValueError
            `case` is set when `pat` is a compiled regular expression.
        KeyError
            Either of the given dimensions is already present in the DataArray.
        KeyError
            The given dimensions names are the same.

        Examples
        --------
        Create a string array

        >>> value = xr.DataArray(
        ...     [
        ...         [
        ...             "a_Xy_0",
        ...             "ab_xY_10-bab_Xy_110-baab_Xy_1100",
        ...             "abc_Xy_01-cbc_Xy_2210",
        ...         ],
        ...         [
        ...             "abcd_Xy_-dcd_Xy_33210-dccd_Xy_332210",
        ...             "",
        ...             "abcdef_Xy_101-fef_Xy_5543210",
        ...         ],
        ...     ],
        ...     dims=["X", "Y"],
        ... )

        Extract matches

        >>> value.str.extractall(
        ...     r"(\w+)_Xy_(\d*)", group_dim="group", match_dim="match"
        ... )
        <xarray.DataArray (X: 2, Y: 3, group: 3, match: 2)>
        array([[[['a', '0'],
                 ['', ''],
                 ['', '']],
        <BLANKLINE>
                [['bab', '110'],
                 ['baab', '1100'],
                 ['', '']],
        <BLANKLINE>
                [['abc', '01'],
                 ['cbc', '2210'],
                 ['', '']]],
        <BLANKLINE>
        <BLANKLINE>
               [[['abcd', ''],
                 ['dcd', '33210'],
                 ['dccd', '332210']],
        <BLANKLINE>
                [['', ''],
                 ['', ''],
                 ['', '']],
        <BLANKLINE>
                [['abcdef', '101'],
                 ['fef', '5543210'],
                 ['', '']]]], dtype='<U7')
        Dimensions without coordinates: X, Y, group, match

        See Also
        --------
        DataArray.str.extract
        DataArray.str.findall
        re.compile
        re.findall
        pandas.Series.str.extractall
        """
        pat = self._re_compile(pat=pat, flags=flags, case=case)

        if group_dim in self._obj.dims:
            raise KeyError(
                f"Group dimension '{group_dim}' already present in DataArray."
            )

        if match_dim in self._obj.dims:
            raise KeyError(
                f"Match dimension '{match_dim}' already present in DataArray."
            )

        if group_dim == match_dim:
            raise KeyError(
                f"Group dimension '{group_dim}' is the same as match dimension '{match_dim}'."
            )

        _get_count = lambda x, ipat: len(ipat.findall(x))
        maxcount = (
            self._apply(func=_get_count, func_args=(pat,), dtype=np.int_)
            .max()
            .data.tolist()
        )

        if isinstance(pat, re.Pattern):
            maxgroups = pat.groups
        else:
            maxgroups = (
                _apply_str_ufunc(obj=pat, func=lambda x: x.groups, dtype=np.int_)
                .max()
                .data.tolist()
            )

        def _get_res(val, ipat, imaxcount=maxcount, dtype=self._obj.dtype):
            if ipat.groups == 0:
                raise ValueError("No capture groups found in pattern.")
            matches = ipat.findall(val)
            res = np.zeros([maxcount, ipat.groups], dtype)

            if ipat.groups == 1:
                for imatch, match in enumerate(matches):
                    res[imatch, 0] = match
            else:
                for imatch, match in enumerate(matches):
                    for jmatch, submatch in enumerate(match):
                        res[imatch, jmatch] = submatch

            return res

        return self._apply(
            # dtype MUST be object or strings can be truncated
            # See: https://github.com/numpy/numpy/issues/8352
            func=_get_res,
            func_args=(pat,),
            dtype=np.object_,
            output_core_dims=[[group_dim, match_dim]],
            output_sizes={group_dim: maxgroups, match_dim: maxcount},
        ).astype(self._obj.dtype.kind)

    def findall(
        self,
        pat: Union[str, bytes, Pattern, Any],
        case: bool = None,
        flags: int = 0,
    ) -> Any:
        r"""
        Find all occurrences of pattern or regular expression in the DataArray.

        Equivalent to applying re.findall() to all the elements in the DataArray.
        Results in an object array of lists.
        If there is only one capture group, the lists will be a sequence of matches.
        If there are multiple capture groups, the lists will be a sequence of lists,
        each of which contains a sequence of matches.

        If `pat` is array-like, it is broadcast against the array and applied
        elementwise.

        Parameters
        ----------
        pat : str or re.Pattern
            A string containing a regular expression or a compiled regular
            expression object. If array-like, it is broadcast.
        case : bool, default: True
            If True, case sensitive.
            Cannot be set if `pat` is a compiled regex.
            Equivalent to setting the `re.IGNORECASE` flag.
        flags : int, default: 0
            Flags to pass through to the re module, e.g. `re.IGNORECASE`.
            see `compilation-flags <https://docs.python.org/3/howto/regex.html#compilation-flags>`_.
            ``0`` means no flags. Flags can be combined with the bitwise or operator ``|``.
            Cannot be set if `pat` is a compiled regex.

        Returns
        -------
        extracted : object array

        Raises
        ------
        ValueError
            `pat` has no capture groups.
        ValueError
            `case` is set when `pat` is a compiled regular expression.

        Examples
        --------
        Create a string array

        >>> value = xr.DataArray(
        ...     [
        ...         [
        ...             "a_Xy_0",
        ...             "ab_xY_10-bab_Xy_110-baab_Xy_1100",
        ...             "abc_Xy_01-cbc_Xy_2210",
        ...         ],
        ...         [
        ...             "abcd_Xy_-dcd_Xy_33210-dccd_Xy_332210",
        ...             "",
        ...             "abcdef_Xy_101-fef_Xy_5543210",
        ...         ],
        ...     ],
        ...     dims=["X", "Y"],
        ... )

        Extract matches

        >>> value.str.findall(r"(\w+)_Xy_(\d*)")
        <xarray.DataArray (X: 2, Y: 3)>
        array([[list([('a', '0')]), list([('bab', '110'), ('baab', '1100')]),
                list([('abc', '01'), ('cbc', '2210')])],
               [list([('abcd', ''), ('dcd', '33210'), ('dccd', '332210')]),
                list([]), list([('abcdef', '101'), ('fef', '5543210')])]],
              dtype=object)
        Dimensions without coordinates: X, Y

        See Also
        --------
        DataArray.str.extract
        DataArray.str.extractall
        re.compile
        re.findall
        pandas.Series.str.findall
        """
        pat = self._re_compile(pat=pat, flags=flags, case=case)

        def func(x, ipat):
            if ipat.groups == 0:
                raise ValueError("No capture groups found in pattern.")

            return ipat.findall(x)

        return self._apply(func=func, func_args=(pat,), dtype=np.object_)

    def _partitioner(
        self,
        *,
        func: Callable,
        dim: Hashable,
        sep: Optional[Union[str, bytes, Any]],
    ) -> Any:
        """
        Implements logic for `partition` and `rpartition`.
        """
        sep = self._stringify(sep)

        if dim is None:
            listfunc = lambda x, isep: list(func(x, isep))
            return self._apply(func=listfunc, func_args=(sep,), dtype=np.object_)

        # _apply breaks on an empty array in this case
        if not self._obj.size:
            return self._obj.copy().expand_dims({dim: 0}, axis=-1)

        arrfunc = lambda x, isep: np.array(func(x, isep), dtype=self._obj.dtype)

        # dtype MUST be object or strings can be truncated
        # See: https://github.com/numpy/numpy/issues/8352
        return self._apply(
            func=arrfunc,
            func_args=(sep,),
            dtype=np.object_,
            output_core_dims=[[dim]],
            output_sizes={dim: 3},
        ).astype(self._obj.dtype.kind)

    def partition(
        self,
        dim: Optional[Hashable],
        sep: Union[str, bytes, Any] = " ",
    ) -> Any:
        """
        Split the strings in the DataArray at the first occurrence of separator `sep`.

        This method splits the string at the first occurrence of `sep`,
        and returns 3 elements containing the part before the separator,
        the separator itself, and the part after the separator.
        If the separator is not found, return 3 elements containing the string itself,
        followed by two empty strings.

        If `sep` is array-like, it is broadcast against the array and applied
        elementwise.

        Parameters
        ----------
        dim : hashable or None
            Name for the dimension to place the 3 elements in.
            If `None`, place the results as list elements in an object DataArray.
        sep : str, default: " "
            String to split on.
            If array-like, it is broadcast.

        Returns
        -------
        partitioned : same type as values or object array

        See Also
        --------
        DataArray.str.rpartition
        str.partition
        pandas.Series.str.partition
        """
        return self._partitioner(func=self._obj.dtype.type.partition, dim=dim, sep=sep)

    def rpartition(
        self,
        dim: Optional[Hashable],
        sep: Union[str, bytes, Any] = " ",
    ) -> Any:
        """
        Split the strings in the DataArray at the last occurrence of separator `sep`.

        This method splits the string at the last occurrence of `sep`,
        and returns 3 elements containing the part before the separator,
        the separator itself, and the part after the separator.
        If the separator is not found, return 3 elements containing two empty strings,
        followed by the string itself.

        If `sep` is array-like, it is broadcast against the array and applied
        elementwise.

        Parameters
        ----------
        dim : hashable or None
            Name for the dimension to place the 3 elements in.
            If `None`, place the results as list elements in an object DataArray.
        sep : str, default: " "
            String to split on.
            If array-like, it is broadcast.

        Returns
        -------
        rpartitioned : same type as values or object array

        See Also
        --------
        DataArray.str.partition
        str.rpartition
        pandas.Series.str.rpartition
        """
        return self._partitioner(func=self._obj.dtype.type.rpartition, dim=dim, sep=sep)

    def _splitter(
        self,
        *,
        func: Callable,
        pre: bool,
        dim: Hashable,
        sep: Optional[Union[str, bytes, Any]],
        maxsplit: int,
    ) -> Any:
        """
        Implements logic for `split` and `rsplit`.
        """
        if sep is not None:
            sep = self._stringify(sep)

        if dim is None:
            f_none = lambda x, isep: func(x, isep, maxsplit)
            return self._apply(func=f_none, func_args=(sep,), dtype=np.object_)

        # _apply breaks on an empty array in this case
        if not self._obj.size:
            return self._obj.copy().expand_dims({dim: 0}, axis=-1)

        f_count = lambda x, isep: max(len(func(x, isep, maxsplit)), 1)
        maxsplit = (
            self._apply(func=f_count, func_args=(sep,), dtype=np.int_).max().data.item()
            - 1
        )

        def _dosplit(mystr, sep, maxsplit=maxsplit, dtype=self._obj.dtype):
            res = func(mystr, sep, maxsplit)
            if len(res) < maxsplit + 1:
                pad = [""] * (maxsplit + 1 - len(res))
                if pre:
                    res += pad
                else:
                    res = pad + res
            return np.array(res, dtype=dtype)

        # dtype MUST be object or strings can be truncated
        # See: https://github.com/numpy/numpy/issues/8352
        return self._apply(
            func=_dosplit,
            func_args=(sep,),
            dtype=np.object_,
            output_core_dims=[[dim]],
            output_sizes={dim: maxsplit},
        ).astype(self._obj.dtype.kind)

    def split(
        self,
        dim: Optional[Hashable],
        sep: Union[str, bytes, Any] = None,
        maxsplit: int = -1,
    ) -> Any:
        r"""
        Split strings in a DataArray around the given separator/delimiter `sep`.

        Splits the string in the DataArray from the beginning,
        at the specified delimiter string.

        If `sep` is array-like, it is broadcast against the array and applied
        elementwise.

        Parameters
        ----------
        dim : hashable or None
            Name for the dimension to place the results in.
            If `None`, place the results as list elements in an object DataArray.
        sep : str, default: None
            String to split on. If ``None`` (the default), split on any whitespace.
            If array-like, it is broadcast.
        maxsplit : int, default: -1
            Limit number of splits in output, starting from the beginning.
            If -1 (the default), return all splits.

        Returns
        -------
        splitted : same type as values or object array

        Examples
        --------
        Create a string DataArray

        >>> values = xr.DataArray(
        ...     [
        ...         ["abc def", "spam\t\teggs\tswallow", "red_blue"],
        ...         ["test0\ntest1\ntest2\n\ntest3", "", "abra  ka\nda\tbra"],
        ...     ],
        ...     dims=["X", "Y"],
        ... )

        Split once and put the results in a new dimension

        >>> values.str.split(dim="splitted", maxsplit=1)
        <xarray.DataArray (X: 2, Y: 3, splitted: 2)>
        array([[['abc', 'def'],
                ['spam', 'eggs\tswallow'],
                ['red_blue', '']],
        <BLANKLINE>
               [['test0', 'test1\ntest2\n\ntest3'],
                ['', ''],
                ['abra', 'ka\nda\tbra']]], dtype='<U18')
        Dimensions without coordinates: X, Y, splitted

        Split as many times as needed and put the results in a new dimension

        >>> values.str.split(dim="splitted")
        <xarray.DataArray (X: 2, Y: 3, splitted: 4)>
        array([[['abc', 'def', '', ''],
                ['spam', 'eggs', 'swallow', ''],
                ['red_blue', '', '', '']],
        <BLANKLINE>
               [['test0', 'test1', 'test2', 'test3'],
                ['', '', '', ''],
                ['abra', 'ka', 'da', 'bra']]], dtype='<U8')
        Dimensions without coordinates: X, Y, splitted

        Split once and put the results in lists

        >>> values.str.split(dim=None, maxsplit=1)
        <xarray.DataArray (X: 2, Y: 3)>
        array([[list(['abc', 'def']), list(['spam', 'eggs\tswallow']),
                list(['red_blue'])],
               [list(['test0', 'test1\ntest2\n\ntest3']), list([]),
                list(['abra', 'ka\nda\tbra'])]], dtype=object)
        Dimensions without coordinates: X, Y

        Split as many times as needed and put the results in a list

        >>> values.str.split(dim=None)
        <xarray.DataArray (X: 2, Y: 3)>
        array([[list(['abc', 'def']), list(['spam', 'eggs', 'swallow']),
                list(['red_blue'])],
               [list(['test0', 'test1', 'test2', 'test3']), list([]),
                list(['abra', 'ka', 'da', 'bra'])]], dtype=object)
        Dimensions without coordinates: X, Y

        Split only on spaces

        >>> values.str.split(dim="splitted", sep=" ")
        <xarray.DataArray (X: 2, Y: 3, splitted: 3)>
        array([[['abc', 'def', ''],
                ['spam\t\teggs\tswallow', '', ''],
                ['red_blue', '', '']],
        <BLANKLINE>
               [['test0\ntest1\ntest2\n\ntest3', '', ''],
                ['', '', ''],
                ['abra', '', 'ka\nda\tbra']]], dtype='<U24')
        Dimensions without coordinates: X, Y, splitted

        See Also
        --------
        DataArray.str.rsplit
        str.split
        pandas.Series.str.split
        """
        return self._splitter(
            func=self._obj.dtype.type.split,
            pre=True,
            dim=dim,
            sep=sep,
            maxsplit=maxsplit,
        )

    def rsplit(
        self,
        dim: Optional[Hashable],
        sep: Union[str, bytes, Any] = None,
        maxsplit: Union[int, Any] = -1,
    ) -> Any:
        r"""
        Split strings in a DataArray around the given separator/delimiter `sep`.

        Splits the string in the DataArray from the end,
        at the specified delimiter string.

        If `sep` is array-like, it is broadcast against the array and applied
        elementwise.

        Parameters
        ----------
        dim : hashable or None
            Name for the dimension to place the results in.
            If `None`, place the results as list elements in an object DataArray
        sep : str, default: None
            String to split on. If ``None`` (the default), split on any whitespace.
            If array-like, it is broadcast.
        maxsplit : int, default: -1
            Limit number of splits in output, starting from the end.
            If -1 (the default), return all splits.
            The final number of split values may be less than this if there are no
            DataArray elements with that many values.

        Returns
        -------
        rsplitted : same type as values or object array

        Examples
        --------
        Create a string DataArray

        >>> values = xr.DataArray(
        ...     [
        ...         ["abc def", "spam\t\teggs\tswallow", "red_blue"],
        ...         ["test0\ntest1\ntest2\n\ntest3", "", "abra  ka\nda\tbra"],
        ...     ],
        ...     dims=["X", "Y"],
        ... )

        Split once and put the results in a new dimension

        >>> values.str.rsplit(dim="splitted", maxsplit=1)
        <xarray.DataArray (X: 2, Y: 3, splitted: 2)>
        array([[['abc', 'def'],
                ['spam\t\teggs', 'swallow'],
                ['', 'red_blue']],
        <BLANKLINE>
               [['test0\ntest1\ntest2', 'test3'],
                ['', ''],
                ['abra  ka\nda', 'bra']]], dtype='<U17')
        Dimensions without coordinates: X, Y, splitted

        Split as many times as needed and put the results in a new dimension

        >>> values.str.rsplit(dim="splitted")
        <xarray.DataArray (X: 2, Y: 3, splitted: 4)>
        array([[['', '', 'abc', 'def'],
                ['', 'spam', 'eggs', 'swallow'],
                ['', '', '', 'red_blue']],
        <BLANKLINE>
               [['test0', 'test1', 'test2', 'test3'],
                ['', '', '', ''],
                ['abra', 'ka', 'da', 'bra']]], dtype='<U8')
        Dimensions without coordinates: X, Y, splitted

        Split once and put the results in lists

        >>> values.str.rsplit(dim=None, maxsplit=1)
        <xarray.DataArray (X: 2, Y: 3)>
        array([[list(['abc', 'def']), list(['spam\t\teggs', 'swallow']),
                list(['red_blue'])],
               [list(['test0\ntest1\ntest2', 'test3']), list([]),
                list(['abra  ka\nda', 'bra'])]], dtype=object)
        Dimensions without coordinates: X, Y

        Split as many times as needed and put the results in a list

        >>> values.str.rsplit(dim=None)
        <xarray.DataArray (X: 2, Y: 3)>
        array([[list(['abc', 'def']), list(['spam', 'eggs', 'swallow']),
                list(['red_blue'])],
               [list(['test0', 'test1', 'test2', 'test3']), list([]),
                list(['abra', 'ka', 'da', 'bra'])]], dtype=object)
        Dimensions without coordinates: X, Y

        Split only on spaces

        >>> values.str.rsplit(dim="splitted", sep=" ")
        <xarray.DataArray (X: 2, Y: 3, splitted: 3)>
        array([[['', 'abc', 'def'],
                ['', '', 'spam\t\teggs\tswallow'],
                ['', '', 'red_blue']],
        <BLANKLINE>
               [['', '', 'test0\ntest1\ntest2\n\ntest3'],
                ['', '', ''],
                ['abra', '', 'ka\nda\tbra']]], dtype='<U24')
        Dimensions without coordinates: X, Y, splitted

        See Also
        --------
        DataArray.str.split
        str.rsplit
        pandas.Series.str.rsplit
        """
        return self._splitter(
            func=self._obj.dtype.type.rsplit,
            pre=False,
            dim=dim,
            sep=sep,
            maxsplit=maxsplit,
        )

    def get_dummies(
        self,
        dim: Hashable,
        sep: Union[str, bytes, Any] = "|",
    ) -> Any:
        """
        Return DataArray of dummy/indicator variables.

        Each string in the DataArray is split at `sep`.
        A new dimension is created with coordinates for each unique result,
        and the corresponding element of that dimension is `True` if
        that result is present and `False` if not.

        If `sep` is array-like, it is broadcast against the array and applied
        elementwise.

        Parameters
        ----------
        dim : hashable
            Name for the dimension to place the results in.
        sep : str, default: "|".
            String to split on.
            If array-like, it is broadcast.

        Returns
        -------
        dummies : array of bool

        Examples
        --------
        Create a string array

        >>> values = xr.DataArray(
        ...     [
        ...         ["a|ab~abc|abc", "ab", "a||abc|abcd"],
        ...         ["abcd|ab|a", "abc|ab~abc", "|a"],
        ...     ],
        ...     dims=["X", "Y"],
        ... )

        Extract dummy values

        >>> values.str.get_dummies(dim="dummies")
        <xarray.DataArray (X: 2, Y: 3, dummies: 5)>
        array([[[ True, False,  True, False,  True],
                [False,  True, False, False, False],
                [ True, False,  True,  True, False]],
        <BLANKLINE>
               [[ True,  True, False,  True, False],
                [False, False,  True, False,  True],
                [ True, False, False, False, False]]])
        Coordinates:
          * dummies  (dummies) <U6 'a' 'ab' 'abc' 'abcd' 'ab~abc'
        Dimensions without coordinates: X, Y

        See Also
        --------
        pandas.Series.str.get_dummies
        """
        # _apply breaks on an empty array in this case
        if not self._obj.size:
            return self._obj.copy().expand_dims({dim: 0}, axis=-1)

        sep = self._stringify(sep)
        f_set = lambda x, isep: set(x.split(isep)) - {self._stringify("")}
        setarr = self._apply(func=f_set, func_args=(sep,), dtype=np.object_)
        vals = sorted(reduce(set_union, setarr.data.ravel()))

        func = lambda x: np.array([val in x for val in vals], dtype=np.bool_)
        res = _apply_str_ufunc(
            func=func,
            obj=setarr,
            output_core_dims=[[dim]],
            output_sizes={dim: len(vals)},
            dtype=np.bool_,
        )
        res.coords[dim] = vals
        return res

    def decode(
        self,
        encoding: str,
        errors: str = "strict",
    ) -> Any:
        """
        Decode character string in the array using indicated encoding.

        Parameters
        ----------
        encoding : str
            The encoding to use.
            Please see the Python documentation `codecs standard encoders <https://docs.python.org/3/library/codecs.html#standard-encodings>`_
            section for a list of encodings handlers.
        errors : str, optional
            The handler for encoding errors.
            Please see the Python documentation `codecs error handlers <https://docs.python.org/3/library/codecs.html#error-handlers>`_
            for a list of error handlers.

        Returns
        -------
        decoded : same type as values
        """
        if encoding in _cpython_optimized_decoders:
            func = lambda x: x.decode(encoding, errors)
        else:
            decoder = codecs.getdecoder(encoding)
            func = lambda x: decoder(x, errors)[0]
        return self._apply(func=func, dtype=np.str_)

    def encode(
        self,
        encoding: str,
        errors: str = "strict",
    ) -> Any:
        """
        Encode character string in the array using indicated encoding.

        Parameters
        ----------
        encoding : str
            The encoding to use.
            Please see the Python documentation `codecs standard encoders <https://docs.python.org/3/library/codecs.html#standard-encodings>`_
            section for a list of encodings handlers.
        errors : str, optional
            The handler for encoding errors.
            Please see the Python documentation `codecs error handlers <https://docs.python.org/3/library/codecs.html#error-handlers>`_
            for a list of error handlers.

        Returns
        -------
        encoded : same type as values
        """
        if encoding in _cpython_optimized_encoders:
            func = lambda x: x.encode(encoding, errors)
        else:
            encoder = codecs.getencoder(encoding)
            func = lambda x: encoder(x, errors)[0]
        return self._apply(func=func, dtype=np.bytes_)
