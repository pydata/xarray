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


def _is_str_like(x):
    return isinstance(x, str) or isinstance(x, bytes)


class StringAccessor:
    """Vectorized string functions for string-like arrays.

    Similar to pandas, fields can be accessed through the `.str` attribute
    for applicable DataArrays.

        >>> da = xr.DataArray(["some", "text", "in", "an", "array"])
        >>> da.str.len()
        <xarray.DataArray (dim_0: 5)>
        array([4, 4, 2, 2, 5])
        Dimensions without coordinates: dim_0

    """

    __slots__ = ("_obj",)

    def __init__(self, obj):
        self._obj = obj

    def _apply(self, f, dtype=None):
        # TODO handling of na values ?
        if dtype is None:
            dtype = self._obj.dtype

        g = np.vectorize(f, otypes=[dtype])
        return apply_ufunc(g, self._obj, dask="parallelized", output_dtypes=[dtype])

    def len(self):
        """
        Compute the length of each string in the array.

        Returns
        -------
        lengths array : array of int
        """
        return self._apply(len, dtype=int)

    def __getitem__(self, key):
        if isinstance(key, slice):
            return self.slice(start=key.start, stop=key.stop, step=key.step)
        else:
            return self.get(key)

    def get(self, i, default=""):
        """
        Extract character number `i` from each string in the array.

        Parameters
        ----------
        i : int
            Position of element to extract.
        default : optional
            Value for out-of-range index. If not specified (None) defaults to
            an empty string.

        Returns
        -------
        items : array of object
        """
        s = slice(-1, None) if i == -1 else slice(i, i + 1)

        def f(x):
            item = x[s]

            return item if item else default

        return self._apply(f)

    def slice(self, start=None, stop=None, step=None):
        """
        Slice substrings from each string in the array.

        Parameters
        ----------
        start : int, optional
            Start position for slice operation.
        stop : int, optional
            Stop position for slice operation.
        step : int, optional
            Step size for slice operation.

        Returns
        -------
        sliced strings : same type as values
        """
        s = slice(start, stop, step)
        f = lambda x: x[s]
        return self._apply(f)

    def slice_replace(self, start=None, stop=None, repl=""):
        """
        Replace a positional slice of a string with another value.

        Parameters
        ----------
        start : int, optional
            Left index position to use for the slice. If not specified (None),
            the slice is unbounded on the left, i.e. slice from the start
            of the string.
        stop : int, optional
            Right index position to use for the slice. If not specified (None),
            the slice is unbounded on the right, i.e. slice until the
            end of the string.
        repl : str, optional
            String for replacement. If not specified, the sliced region
            is replaced with an empty string.

        Returns
        -------
        replaced : same type as values
        """
        repl = self._obj.dtype.type(repl)

        def f(x):
            if len(x[start:stop]) == 0:
                local_stop = start
            else:
                local_stop = stop
            y = self._obj.dtype.type("")
            if start is not None:
                y += x[:start]
            y += repl
            if stop is not None:
                y += x[local_stop:]
            return y

        return self._apply(f)

    def capitalize(self):
        """
        Convert strings in the array to be capitalized.

        Returns
        -------
        capitalized : same type as values
        """
        return self._apply(lambda x: x.capitalize())

    def lower(self):
        """
        Convert strings in the array to lowercase.

        Returns
        -------
        lowerd : same type as values
        """
        return self._apply(lambda x: x.lower())

    def swapcase(self):
        """
        Convert strings in the array to be swapcased.

        Returns
        -------
        swapcased : same type as values
        """
        return self._apply(lambda x: x.swapcase())

    def title(self):
        """
        Convert strings in the array to titlecase.

        Returns
        -------
        titled : same type as values
        """
        return self._apply(lambda x: x.title())

    def upper(self):
        """
        Convert strings in the array to uppercase.

        Returns
        -------
        uppered : same type as values
        """
        return self._apply(lambda x: x.upper())

    def isalnum(self):
        """
        Check whether all characters in each string are alphanumeric.

        Returns
        -------
        isalnum : array of bool
            Array of boolean values with the same shape as the original array.
        """
        return self._apply(lambda x: x.isalnum(), dtype=bool)

    def isalpha(self):
        """
        Check whether all characters in each string are alphabetic.

        Returns
        -------
        isalpha : array of bool
            Array of boolean values with the same shape as the original array.
        """
        return self._apply(lambda x: x.isalpha(), dtype=bool)

    def isdecimal(self):
        """
        Check whether all characters in each string are decimal.

        Returns
        -------
        isdecimal : array of bool
            Array of boolean values with the same shape as the original array.
        """
        return self._apply(lambda x: x.isdecimal(), dtype=bool)

    def isdigit(self):
        """
        Check whether all characters in each string are digits.

        Returns
        -------
        isdigit : array of bool
            Array of boolean values with the same shape as the original array.
        """
        return self._apply(lambda x: x.isdigit(), dtype=bool)

    def islower(self):
        """
        Check whether all characters in each string are lowercase.

        Returns
        -------
        islower : array of bool
            Array of boolean values with the same shape as the original array.
        """
        return self._apply(lambda x: x.islower(), dtype=bool)

    def isnumeric(self):
        """
        Check whether all characters in each string are numeric.

        Returns
        -------
        isnumeric : array of bool
            Array of boolean values with the same shape as the original array.
        """
        return self._apply(lambda x: x.isnumeric(), dtype=bool)

    def isspace(self):
        """
        Check whether all characters in each string are spaces.

        Returns
        -------
        isspace : array of bool
            Array of boolean values with the same shape as the original array.
        """
        return self._apply(lambda x: x.isspace(), dtype=bool)

    def istitle(self):
        """
        Check whether all characters in each string are titlecase.

        Returns
        -------
        istitle : array of bool
            Array of boolean values with the same shape as the original array.
        """
        return self._apply(lambda x: x.istitle(), dtype=bool)

    def isupper(self):
        """
        Check whether all characters in each string are uppercase.

        Returns
        -------
        isupper : array of bool
            Array of boolean values with the same shape as the original array.
        """
        return self._apply(lambda x: x.isupper(), dtype=bool)

    def count(self, pat, flags=0):
        """
        Count occurrences of pattern in each string of the array.

        This function is used to count the number of times a particular regex
        pattern is repeated in each of the string elements of the
        :class:`~xarray.DataArray`.

        Parameters
        ----------
        pat : str
            Valid regular expression.
        flags : int, default: 0
            Flags for the `re` module. Use 0 for no flags. For a complete list,
            `see here <https://docs.python.org/3/howto/regex.html#compilation-flags>`_.

        Returns
        -------
        counts : array of int
        """
        pat = self._obj.dtype.type(pat)
        regex = re.compile(pat, flags=flags)
        f = lambda x: len(regex.findall(x))
        return self._apply(f, dtype=int)

    def startswith(self, pat):
        """
        Test if the start of each string in the array matches a pattern.

        Parameters
        ----------
        pat : str
            Character sequence. Regular expressions are not accepted.

        Returns
        -------
        startswith : array of bool
            An array of booleans indicating whether the given pattern matches
            the start of each string element.
        """
        pat = self._obj.dtype.type(pat)
        f = lambda x: x.startswith(pat)
        return self._apply(f, dtype=bool)

    def endswith(self, pat):
        """
        Test if the end of each string in the array matches a pattern.

        Parameters
        ----------
        pat : str
            Character sequence. Regular expressions are not accepted.

        Returns
        -------
        endswith : array of bool
            A Series of booleans indicating whether the given pattern matches
            the end of each string element.
        """
        pat = self._obj.dtype.type(pat)
        f = lambda x: x.endswith(pat)
        return self._apply(f, dtype=bool)

    def pad(self, width, side="left", fillchar=" "):
        """
        Pad strings in the array up to width.

        Parameters
        ----------
        width : int
            Minimum width of resulting string; additional characters will be
            filled with character defined in `fillchar`.
        side : {"left", "right", "both"}, default: "left"
            Side from which to fill resulting string.
        fillchar : str, default: " "
            Additional character for filling, default is whitespace.

        Returns
        -------
        filled : same type as values
            Array with a minimum number of char in each element.
        """
        width = int(width)
        fillchar = self._obj.dtype.type(fillchar)
        if len(fillchar) != 1:
            raise TypeError("fillchar must be a character, not str")

        if side == "left":
            f = lambda s: s.rjust(width, fillchar)
        elif side == "right":
            f = lambda s: s.ljust(width, fillchar)
        elif side == "both":
            f = lambda s: s.center(width, fillchar)
        else:  # pragma: no cover
            raise ValueError("Invalid side")

        return self._apply(f)

    def center(self, width, fillchar=" "):
        """
        Pad left and right side of each string in the array.

        Parameters
        ----------
        width : int
            Minimum width of resulting string; additional characters will be
            filled with ``fillchar``
        fillchar : str, default: " "
            Additional character for filling, default is whitespace

        Returns
        -------
        filled : same type as values
        """
        return self.pad(width, side="both", fillchar=fillchar)

    def ljust(self, width, fillchar=" "):
        """
        Pad right side of each string in the array.

        Parameters
        ----------
        width : int
            Minimum width of resulting string; additional characters will be
            filled with ``fillchar``
        fillchar : str, default: " "
            Additional character for filling, default is whitespace

        Returns
        -------
        filled : same type as values
        """
        return self.pad(width, side="right", fillchar=fillchar)

    def rjust(self, width, fillchar=" "):
        """
        Pad left side of each string in the array.

        Parameters
        ----------
        width : int
            Minimum width of resulting string; additional characters will be
            filled with ``fillchar``
        fillchar : str, default: " "
            Additional character for filling, default is whitespace

        Returns
        -------
        filled : same type as values
        """
        return self.pad(width, side="left", fillchar=fillchar)

    def zfill(self, width):
        """
        Pad each string in the array by prepending '0' characters.

        Strings in the array are padded with '0' characters on the
        left of the string to reach a total string length  `width`. Strings
        in the array with length greater or equal to `width` are unchanged.

        Parameters
        ----------
        width : int
            Minimum length of resulting string; strings with length less
            than `width` be prepended with '0' characters.

        Returns
        -------
        filled : same type as values
        """
        return self.pad(width, side="left", fillchar="0")

    def contains(self, pat, case=True, flags=0, regex=True):
        """
        Test if pattern or regex is contained within each string of the array.

        Return boolean array based on whether a given pattern or regex is
        contained within a string of the array.

        Parameters
        ----------
        pat : str
            Character sequence or regular expression.
        case : bool, default: True
            If True, case sensitive.
        flags : int, default: 0
            Flags to pass through to the re module, e.g. re.IGNORECASE.
            ``0`` means no flags.
        regex : bool, default: True
            If True, assumes the pat is a regular expression.
            If False, treats the pat as a literal string.

        Returns
        -------
        contains : array of bool
            An array of boolean values indicating whether the
            given pattern is contained within the string of each element
            of the array.
        """
        pat = self._obj.dtype.type(pat)
        if regex:
            if not case:
                flags |= re.IGNORECASE

            regex = re.compile(pat, flags=flags)

            if regex.groups > 0:  # pragma: no cover
                raise ValueError("This pattern has match groups.")

            f = lambda x: bool(regex.search(x))
        else:
            if case:
                f = lambda x: pat in x
            else:
                uppered = self._obj.str.upper()
                return uppered.str.contains(pat.upper(), regex=False)

        return self._apply(f, dtype=bool)

    def match(self, pat, case=True, flags=0):
        """
        Determine if each string in the array matches a regular expression.

        Parameters
        ----------
        pat : str
            Character sequence or regular expression
        case : bool, default: True
            If True, case sensitive
        flags : int, default: 0
            re module flags, e.g. re.IGNORECASE. ``0`` means no flags

        Returns
        -------
        matched : array of bool
        """
        if not case:
            flags |= re.IGNORECASE

        pat = self._obj.dtype.type(pat)
        regex = re.compile(pat, flags=flags)
        f = lambda x: bool(regex.match(x))
        return self._apply(f, dtype=bool)

    def strip(self, to_strip=None, side="both"):
        """
        Remove leading and trailing characters.

        Strip whitespaces (including newlines) or a set of specified characters
        from each string in the array from left and/or right sides.

        Parameters
        ----------
        to_strip : str or None, default: None
            Specifying the set of characters to be removed.
            All combinations of this set of characters will be stripped.
            If None then whitespaces are removed.
        side : {"left", "right", "both"}, default: "left"
            Side from which to strip.

        Returns
        -------
        stripped : same type as values
        """
        if to_strip is not None:
            to_strip = self._obj.dtype.type(to_strip)

        if side == "both":
            f = lambda x: x.strip(to_strip)
        elif side == "left":
            f = lambda x: x.lstrip(to_strip)
        elif side == "right":
            f = lambda x: x.rstrip(to_strip)
        else:  # pragma: no cover
            raise ValueError("Invalid side")

        return self._apply(f)

    def lstrip(self, to_strip=None):
        """
        Remove leading characters.

        Strip whitespaces (including newlines) or a set of specified characters
        from each string in the array from the left side.

        Parameters
        ----------
        to_strip : str or None, default: None
            Specifying the set of characters to be removed.
            All combinations of this set of characters will be stripped.
            If None then whitespaces are removed.

        Returns
        -------
        stripped : same type as values
        """
        return self.strip(to_strip, side="left")

    def rstrip(self, to_strip=None):
        """
        Remove trailing characters.

        Strip whitespaces (including newlines) or a set of specified characters
        from each string in the array from the right side.

        Parameters
        ----------
        to_strip : str or None, default: None
            Specifying the set of characters to be removed.
            All combinations of this set of characters will be stripped.
            If None then whitespaces are removed.

        Returns
        -------
        stripped : same type as values
        """
        return self.strip(to_strip, side="right")

    def wrap(self, width, **kwargs):
        """
        Wrap long strings in the array in paragraphs with length less than `width`.

        This method has the same keyword parameters and defaults as
        :class:`textwrap.TextWrapper`.

        Parameters
        ----------
        width : int
            Maximum line-width
        **kwargs
            keyword arguments passed into :class:`textwrap.TextWrapper`.

        Returns
        -------
        wrapped : same type as values
        """
        tw = textwrap.TextWrapper(width=width, **kwargs)
        f = lambda x: "\n".join(tw.wrap(x))
        return self._apply(f)

    def translate(self, table):
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
        f = lambda x: x.translate(table)
        return self._apply(f)

    def repeat(self, repeats):
        """
        Duplicate each string in the array.

        Parameters
        ----------
        repeats : int
            Number of repetitions.

        Returns
        -------
        repeated : same type as values
            Array of repeated string objects.
        """
        f = lambda x: repeats * x
        return self._apply(f)

    def find(self, sub, start=0, end=None, side="left"):
        """
        Return lowest or highest indexes in each strings in the array
        where the substring is fully contained between [start:end].
        Return -1 on failure.

        Parameters
        ----------
        sub : str
            Substring being searched
        start : int
            Left edge index
        end : int
            Right edge index
        side : {"left", "right"}, default: "left"
            Starting side for search.

        Returns
        -------
        found : array of int
        """
        sub = self._obj.dtype.type(sub)

        if side == "left":
            method = "find"
        elif side == "right":
            method = "rfind"
        else:  # pragma: no cover
            raise ValueError("Invalid side")

        if end is None:
            f = lambda x: getattr(x, method)(sub, start)
        else:
            f = lambda x: getattr(x, method)(sub, start, end)

        return self._apply(f, dtype=int)

    def rfind(self, sub, start=0, end=None):
        """
        Return highest indexes in each strings in the array
        where the substring is fully contained between [start:end].
        Return -1 on failure.

        Parameters
        ----------
        sub : str
            Substring being searched
        start : int
            Left edge index
        end : int
            Right edge index

        Returns
        -------
        found : array of int
        """
        return self.find(sub, start=start, end=end, side="right")

    def index(self, sub, start=0, end=None, side="left"):
        """
        Return lowest or highest indexes in each strings where the substring is
        fully contained between [start:end]. This is the same as
        ``str.find`` except instead of returning -1, it raises a ValueError
        when the substring is not found.

        Parameters
        ----------
        sub : str
            Substring being searched
        start : int
            Left edge index
        end : int
            Right edge index
        side : {"left", "right"}, default: "left"
            Starting side for search.

        Returns
        -------
        found : array of int
        """
        sub = self._obj.dtype.type(sub)

        if side == "left":
            method = "index"
        elif side == "right":
            method = "rindex"
        else:  # pragma: no cover
            raise ValueError("Invalid side")

        if end is None:
            f = lambda x: getattr(x, method)(sub, start)
        else:
            f = lambda x: getattr(x, method)(sub, start, end)

        return self._apply(f, dtype=int)

    def rindex(self, sub, start=0, end=None):
        """
        Return highest indexes in each strings where the substring is
        fully contained between [start:end]. This is the same as
        ``str.rfind`` except instead of returning -1, it raises a ValueError
        when the substring is not found.

        Parameters
        ----------
        sub : str
            Substring being searched
        start : int
            Left edge index
        end : int
            Right edge index

        Returns
        -------
        found : array of int
        """
        return self.index(sub, start=start, end=end, side="right")

    def replace(self, pat, repl, n=-1, case=None, flags=0, regex=True):
        """
        Replace occurrences of pattern/regex in the array with some string.

        Parameters
        ----------
        pat : str or re.Pattern
            String can be a character sequence or regular expression.
        repl : str or callable
            Replacement string or a callable. The callable is passed the regex
            match object and must return a replacement string to be used.
            See :func:`re.sub`.
        n : int, default: -1
            Number of replacements to make from start. Use ``-1`` to replace all.
        case : bool, default: None
            - If True, case sensitive (the default if `pat` is a string)
            - Set to False for case insensitive
            - Cannot be set if `pat` is a compiled regex
        flags : int, default: 0
            - re module flags, e.g. re.IGNORECASE. Use ``0`` for no flags.
            - Cannot be set if `pat` is a compiled regex
        regex : bool, default: True
            - If True, assumes the passed-in pattern is a regular expression.
            - If False, treats the pattern as a literal string
            - Cannot be set to False if `pat` is a compiled regex or `repl` is
              a callable.

        Returns
        -------
        replaced : same type as values
            A copy of the object with all matching occurrences of `pat`
            replaced by `repl`.
        """
        if not (_is_str_like(repl) or callable(repl)):  # pragma: no cover
            raise TypeError("repl must be a string or callable")

        if _is_str_like(pat):
            pat = self._obj.dtype.type(pat)

        if _is_str_like(repl):
            repl = self._obj.dtype.type(repl)

        is_compiled_re = isinstance(pat, type(re.compile("")))
        if regex:
            if is_compiled_re:
                if (case is not None) or (flags != 0):
                    raise ValueError(
                        "case and flags cannot be set when pat is a compiled regex"
                    )
            else:
                # not a compiled regex
                # set default case
                if case is None:
                    case = True

                # add case flag, if provided
                if case is False:
                    flags |= re.IGNORECASE
            if is_compiled_re or len(pat) > 1 or flags or callable(repl):
                n = n if n >= 0 else 0
                compiled = re.compile(pat, flags=flags)
                f = lambda x: compiled.sub(repl=repl, string=x, count=n)
            else:
                f = lambda x: x.replace(pat, repl, n)
        else:
            if is_compiled_re:
                raise ValueError(
                    "Cannot use a compiled regex as replacement "
                    "pattern with regex=False"
                )
            if callable(repl):
                raise ValueError("Cannot use a callable replacement when regex=False")
            f = lambda x: x.replace(pat, repl, n)
        return self._apply(f)

    def decode(self, encoding, errors="strict"):
        """
        Decode character string in the array using indicated encoding.

        Parameters
        ----------
        encoding : str
        errors : str, optional

        Returns
        -------
        decoded : same type as values
        """
        if encoding in _cpython_optimized_decoders:
            f = lambda x: x.decode(encoding, errors)
        else:
            decoder = codecs.getdecoder(encoding)
            f = lambda x: decoder(x, errors)[0]
        return self._apply(f, dtype=np.str_)

    def encode(self, encoding, errors="strict"):
        """
        Encode character string in the array using indicated encoding.

        Parameters
        ----------
        encoding : str
        errors : str, optional

        Returns
        -------
        encoded : same type as values
        """
        if encoding in _cpython_optimized_encoders:
            f = lambda x: x.encode(encoding, errors)
        else:
            encoder = codecs.getencoder(encoding)
            f = lambda x: encoder(x, errors)[0]
        return self._apply(f, dtype=np.bytes_)
