# Tests for the `str` accessor are derived from the original
# pandas string accessor tests.

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

import re

import numpy as np
import pytest

import xarray as xr

from . import assert_equal, requires_dask


@pytest.fixture(params=[np.str_, np.bytes_])
def dtype(request):
    return request.param


@requires_dask
def test_dask():
    import dask.array as da

    arr = da.from_array(["a", "b", "c"], chunks=-1)
    xarr = xr.DataArray(arr)

    result = xarr.str.len().compute()
    expected = xr.DataArray([1, 1, 1])
    assert_equal(result, expected)


def test_count(dtype):
    values = xr.DataArray(["foo", "foofoo", "foooofooofommmfoo"]).astype(dtype)
    result = values.str.count("f[o]+")
    expected = xr.DataArray([1, 2, 4])
    assert_equal(result, expected)


def test_contains(dtype):
    values = xr.DataArray(["Foo", "xYz", "fOOomMm__fOo", "MMM_"]).astype(dtype)
    # case insensitive using regex
    result = values.str.contains("FOO|mmm", case=False)
    expected = xr.DataArray([True, False, True, True])
    assert_equal(result, expected)
    # case insensitive without regex
    result = values.str.contains("foo", regex=False, case=False)
    expected = xr.DataArray([True, False, True, False])
    assert_equal(result, expected)


def test_starts_ends_with(dtype):
    values = xr.DataArray(["om", "foo_nom", "nom", "bar_foo", "foo"]).astype(dtype)
    result = values.str.startswith("foo")
    expected = xr.DataArray([False, True, False, False, True])
    assert_equal(result, expected)
    result = values.str.endswith("foo")
    expected = xr.DataArray([False, False, False, True, True])
    assert_equal(result, expected)


def test_case(dtype):
    da = xr.DataArray(["SOme word"]).astype(dtype)
    capitalized = xr.DataArray(["Some word"]).astype(dtype)
    lowered = xr.DataArray(["some word"]).astype(dtype)
    swapped = xr.DataArray(["soME WORD"]).astype(dtype)
    titled = xr.DataArray(["Some Word"]).astype(dtype)
    uppered = xr.DataArray(["SOME WORD"]).astype(dtype)
    assert_equal(da.str.capitalize(), capitalized)
    assert_equal(da.str.lower(), lowered)
    assert_equal(da.str.swapcase(), swapped)
    assert_equal(da.str.title(), titled)
    assert_equal(da.str.upper(), uppered)


def test_replace(dtype):
    values = xr.DataArray(["fooBAD__barBAD"]).astype(dtype)
    result = values.str.replace("BAD[_]*", "")
    expected = xr.DataArray(["foobar"]).astype(dtype)
    assert_equal(result, expected)

    result = values.str.replace("BAD[_]*", "", n=1)
    expected = xr.DataArray(["foobarBAD"]).astype(dtype)
    assert_equal(result, expected)

    s = xr.DataArray(["A", "B", "C", "Aaba", "Baca", "", "CABA", "dog", "cat"]).astype(
        dtype
    )
    result = s.str.replace("A", "YYY")
    expected = xr.DataArray(
        ["YYY", "B", "C", "YYYaba", "Baca", "", "CYYYBYYY", "dog", "cat"]
    ).astype(dtype)
    assert_equal(result, expected)

    result = s.str.replace("A", "YYY", case=False)
    expected = xr.DataArray(
        ["YYY", "B", "C", "YYYYYYbYYY", "BYYYcYYY", "", "CYYYBYYY", "dog", "cYYYt"]
    ).astype(dtype)
    assert_equal(result, expected)

    result = s.str.replace("^.a|dog", "XX-XX ", case=False)
    expected = xr.DataArray(
        ["A", "B", "C", "XX-XX ba", "XX-XX ca", "", "XX-XX BA", "XX-XX ", "XX-XX t"]
    ).astype(dtype)
    assert_equal(result, expected)


def test_replace_callable():
    values = xr.DataArray(["fooBAD__barBAD"])
    # test with callable
    repl = lambda m: m.group(0).swapcase()
    result = values.str.replace("[a-z][A-Z]{2}", repl, n=2)
    exp = xr.DataArray(["foObaD__baRbaD"])
    assert_equal(result, exp)
    # test regex named groups
    values = xr.DataArray(["Foo Bar Baz"])
    pat = r"(?P<first>\w+) (?P<middle>\w+) (?P<last>\w+)"
    repl = lambda m: m.group("middle").swapcase()
    result = values.str.replace(pat, repl)
    exp = xr.DataArray(["bAR"])
    assert_equal(result, exp)


def test_replace_unicode():
    # flags + unicode
    values = xr.DataArray([b"abcd,\xc3\xa0".decode("utf-8")])
    expected = xr.DataArray([b"abcd, \xc3\xa0".decode("utf-8")])
    pat = re.compile(r"(?<=\w),(?=\w)", flags=re.UNICODE)
    result = values.str.replace(pat, ", ")
    assert_equal(result, expected)


def test_replace_compiled_regex(dtype):
    values = xr.DataArray(["fooBAD__barBAD"]).astype(dtype)
    # test with compiled regex
    pat = re.compile(dtype("BAD[_]*"))
    result = values.str.replace(pat, "")
    expected = xr.DataArray(["foobar"]).astype(dtype)
    assert_equal(result, expected)

    result = values.str.replace(pat, "", n=1)
    expected = xr.DataArray(["foobarBAD"]).astype(dtype)
    assert_equal(result, expected)

    # case and flags provided to str.replace will have no effect
    # and will produce warnings
    values = xr.DataArray(["fooBAD__barBAD__bad"]).astype(dtype)
    pat = re.compile(dtype("BAD[_]*"))

    with pytest.raises(ValueError, match="case and flags cannot be"):
        result = values.str.replace(pat, "", flags=re.IGNORECASE)

    with pytest.raises(ValueError, match="case and flags cannot be"):
        result = values.str.replace(pat, "", case=False)

    with pytest.raises(ValueError, match="case and flags cannot be"):
        result = values.str.replace(pat, "", case=True)

    # test with callable
    values = xr.DataArray(["fooBAD__barBAD"]).astype(dtype)
    repl = lambda m: m.group(0).swapcase()
    pat = re.compile(dtype("[a-z][A-Z]{2}"))
    result = values.str.replace(pat, repl, n=2)
    expected = xr.DataArray(["foObaD__baRbaD"]).astype(dtype)
    assert_equal(result, expected)


def test_replace_literal(dtype):
    # GH16808 literal replace (regex=False vs regex=True)
    values = xr.DataArray(["f.o", "foo"]).astype(dtype)
    expected = xr.DataArray(["bao", "bao"]).astype(dtype)
    result = values.str.replace("f.", "ba")
    assert_equal(result, expected)

    expected = xr.DataArray(["bao", "foo"]).astype(dtype)
    result = values.str.replace("f.", "ba", regex=False)
    assert_equal(result, expected)

    # Cannot do a literal replace if given a callable repl or compiled
    # pattern
    callable_repl = lambda m: m.group(0).swapcase()
    compiled_pat = re.compile("[a-z][A-Z]{2}")

    msg = "Cannot use a callable replacement when regex=False"
    with pytest.raises(ValueError, match=msg):
        values.str.replace("abc", callable_repl, regex=False)

    msg = "Cannot use a compiled regex as replacement pattern with regex=False"
    with pytest.raises(ValueError, match=msg):
        values.str.replace(compiled_pat, "", regex=False)


def test_repeat(dtype):
    values = xr.DataArray(["a", "b", "c", "d"]).astype(dtype)
    result = values.str.repeat(3)
    expected = xr.DataArray(["aaa", "bbb", "ccc", "ddd"]).astype(dtype)
    assert_equal(result, expected)


def test_match(dtype):
    # New match behavior introduced in 0.13
    values = xr.DataArray(["fooBAD__barBAD", "foo"]).astype(dtype)
    result = values.str.match(".*(BAD[_]+).*(BAD)")
    expected = xr.DataArray([True, False])
    assert_equal(result, expected)

    values = xr.DataArray(["fooBAD__barBAD", "foo"]).astype(dtype)
    result = values.str.match(".*BAD[_]+.*BAD")
    expected = xr.DataArray([True, False])
    assert_equal(result, expected)


def test_empty_str_methods():
    empty = xr.DataArray(np.empty(shape=(0,), dtype="U"))
    empty_str = empty
    empty_int = xr.DataArray(np.empty(shape=(0,), dtype=int))
    empty_bool = xr.DataArray(np.empty(shape=(0,), dtype=bool))
    empty_bytes = xr.DataArray(np.empty(shape=(0,), dtype="S"))

    assert_equal(empty_str, empty.str.title())
    assert_equal(empty_int, empty.str.count("a"))
    assert_equal(empty_bool, empty.str.contains("a"))
    assert_equal(empty_bool, empty.str.startswith("a"))
    assert_equal(empty_bool, empty.str.endswith("a"))
    assert_equal(empty_str, empty.str.lower())
    assert_equal(empty_str, empty.str.upper())
    assert_equal(empty_str, empty.str.replace("a", "b"))
    assert_equal(empty_str, empty.str.repeat(3))
    assert_equal(empty_bool, empty.str.match("^a"))
    assert_equal(empty_int, empty.str.len())
    assert_equal(empty_int, empty.str.find("a"))
    assert_equal(empty_int, empty.str.rfind("a"))
    assert_equal(empty_str, empty.str.pad(42))
    assert_equal(empty_str, empty.str.center(42))
    assert_equal(empty_str, empty.str.slice(stop=1))
    assert_equal(empty_str, empty.str.slice(step=1))
    assert_equal(empty_str, empty.str.strip())
    assert_equal(empty_str, empty.str.lstrip())
    assert_equal(empty_str, empty.str.rstrip())
    assert_equal(empty_str, empty.str.wrap(42))
    assert_equal(empty_str, empty.str.get(0))
    assert_equal(empty_str, empty_bytes.str.decode("ascii"))
    assert_equal(empty_bytes, empty.str.encode("ascii"))
    assert_equal(empty_str, empty.str.isalnum())
    assert_equal(empty_str, empty.str.isalpha())
    assert_equal(empty_str, empty.str.isdigit())
    assert_equal(empty_str, empty.str.isspace())
    assert_equal(empty_str, empty.str.islower())
    assert_equal(empty_str, empty.str.isupper())
    assert_equal(empty_str, empty.str.istitle())
    assert_equal(empty_str, empty.str.isnumeric())
    assert_equal(empty_str, empty.str.isdecimal())
    assert_equal(empty_str, empty.str.capitalize())
    assert_equal(empty_str, empty.str.swapcase())
    table = str.maketrans("a", "b")
    assert_equal(empty_str, empty.str.translate(table))


def test_ismethods(dtype):
    values = ["A", "b", "Xy", "4", "3A", "", "TT", "55", "-", "  "]
    str_s = xr.DataArray(values).astype(dtype)
    alnum_e = [True, True, True, True, True, False, True, True, False, False]
    alpha_e = [True, True, True, False, False, False, True, False, False, False]
    digit_e = [False, False, False, True, False, False, False, True, False, False]
    space_e = [False, False, False, False, False, False, False, False, False, True]
    lower_e = [False, True, False, False, False, False, False, False, False, False]
    upper_e = [True, False, False, False, True, False, True, False, False, False]
    title_e = [True, False, True, False, True, False, False, False, False, False]

    assert_equal(str_s.str.isalnum(), xr.DataArray(alnum_e))
    assert_equal(str_s.str.isalpha(), xr.DataArray(alpha_e))
    assert_equal(str_s.str.isdigit(), xr.DataArray(digit_e))
    assert_equal(str_s.str.isspace(), xr.DataArray(space_e))
    assert_equal(str_s.str.islower(), xr.DataArray(lower_e))
    assert_equal(str_s.str.isupper(), xr.DataArray(upper_e))
    assert_equal(str_s.str.istitle(), xr.DataArray(title_e))


def test_isnumeric():
    # 0x00bc: ¼ VULGAR FRACTION ONE QUARTER
    # 0x2605: ★ not number
    # 0x1378: ፸ ETHIOPIC NUMBER SEVENTY
    # 0xFF13: ３ Em 3
    values = ["A", "3", "¼", "★", "፸", "３", "four"]
    s = xr.DataArray(values)
    numeric_e = [False, True, True, False, True, True, False]
    decimal_e = [False, True, False, False, False, True, False]
    assert_equal(s.str.isnumeric(), xr.DataArray(numeric_e))
    assert_equal(s.str.isdecimal(), xr.DataArray(decimal_e))


def test_len(dtype):
    values = ["foo", "fooo", "fooooo", "fooooooo"]
    result = xr.DataArray(values).astype(dtype).str.len()
    expected = xr.DataArray([len(x) for x in values])
    assert_equal(result, expected)


def test_find(dtype):
    values = xr.DataArray(["ABCDEFG", "BCDEFEF", "DEFGHIJEF", "EFGHEF", "XXX"])
    values = values.astype(dtype)
    result = values.str.find("EF")
    assert_equal(result, xr.DataArray([4, 3, 1, 0, -1]))
    expected = xr.DataArray([v.find(dtype("EF")) for v in values.values])
    assert_equal(result, expected)

    result = values.str.rfind("EF")
    assert_equal(result, xr.DataArray([4, 5, 7, 4, -1]))
    expected = xr.DataArray([v.rfind(dtype("EF")) for v in values.values])
    assert_equal(result, expected)

    result = values.str.find("EF", 3)
    assert_equal(result, xr.DataArray([4, 3, 7, 4, -1]))
    expected = xr.DataArray([v.find(dtype("EF"), 3) for v in values.values])
    assert_equal(result, expected)

    result = values.str.rfind("EF", 3)
    assert_equal(result, xr.DataArray([4, 5, 7, 4, -1]))
    expected = xr.DataArray([v.rfind(dtype("EF"), 3) for v in values.values])
    assert_equal(result, expected)

    result = values.str.find("EF", 3, 6)
    assert_equal(result, xr.DataArray([4, 3, -1, 4, -1]))
    expected = xr.DataArray([v.find(dtype("EF"), 3, 6) for v in values.values])
    assert_equal(result, expected)

    result = values.str.rfind("EF", 3, 6)
    assert_equal(result, xr.DataArray([4, 3, -1, 4, -1]))
    xp = xr.DataArray([v.rfind(dtype("EF"), 3, 6) for v in values.values])
    assert_equal(result, xp)


def test_index(dtype):
    s = xr.DataArray(["ABCDEFG", "BCDEFEF", "DEFGHIJEF", "EFGHEF"]).astype(dtype)

    result = s.str.index("EF")
    assert_equal(result, xr.DataArray([4, 3, 1, 0]))

    result = s.str.rindex("EF")
    assert_equal(result, xr.DataArray([4, 5, 7, 4]))

    result = s.str.index("EF", 3)
    assert_equal(result, xr.DataArray([4, 3, 7, 4]))

    result = s.str.rindex("EF", 3)
    assert_equal(result, xr.DataArray([4, 5, 7, 4]))

    result = s.str.index("E", 4, 8)
    assert_equal(result, xr.DataArray([4, 5, 7, 4]))

    result = s.str.rindex("E", 0, 5)
    assert_equal(result, xr.DataArray([4, 3, 1, 4]))

    with pytest.raises(ValueError):
        result = s.str.index("DE")


def test_pad(dtype):
    values = xr.DataArray(["a", "b", "c", "eeeee"]).astype(dtype)

    result = values.str.pad(5, side="left")
    expected = xr.DataArray(["    a", "    b", "    c", "eeeee"]).astype(dtype)
    assert_equal(result, expected)

    result = values.str.pad(5, side="right")
    expected = xr.DataArray(["a    ", "b    ", "c    ", "eeeee"]).astype(dtype)
    assert_equal(result, expected)

    result = values.str.pad(5, side="both")
    expected = xr.DataArray(["  a  ", "  b  ", "  c  ", "eeeee"]).astype(dtype)
    assert_equal(result, expected)


def test_pad_fillchar(dtype):
    values = xr.DataArray(["a", "b", "c", "eeeee"]).astype(dtype)

    result = values.str.pad(5, side="left", fillchar="X")
    expected = xr.DataArray(["XXXXa", "XXXXb", "XXXXc", "eeeee"]).astype(dtype)
    assert_equal(result, expected)

    result = values.str.pad(5, side="right", fillchar="X")
    expected = xr.DataArray(["aXXXX", "bXXXX", "cXXXX", "eeeee"]).astype(dtype)
    assert_equal(result, expected)

    result = values.str.pad(5, side="both", fillchar="X")
    expected = xr.DataArray(["XXaXX", "XXbXX", "XXcXX", "eeeee"]).astype(dtype)
    assert_equal(result, expected)

    msg = "fillchar must be a character, not str"
    with pytest.raises(TypeError, match=msg):
        result = values.str.pad(5, fillchar="XY")


def test_translate():
    values = xr.DataArray(["abcdefg", "abcc", "cdddfg", "cdefggg"])
    table = str.maketrans("abc", "cde")
    result = values.str.translate(table)
    expected = xr.DataArray(["cdedefg", "cdee", "edddfg", "edefggg"])
    assert_equal(result, expected)


def test_center_ljust_rjust(dtype):
    values = xr.DataArray(["a", "b", "c", "eeeee"]).astype(dtype)

    result = values.str.center(5)
    expected = xr.DataArray(["  a  ", "  b  ", "  c  ", "eeeee"]).astype(dtype)
    assert_equal(result, expected)

    result = values.str.ljust(5)
    expected = xr.DataArray(["a    ", "b    ", "c    ", "eeeee"]).astype(dtype)
    assert_equal(result, expected)

    result = values.str.rjust(5)
    expected = xr.DataArray(["    a", "    b", "    c", "eeeee"]).astype(dtype)
    assert_equal(result, expected)


def test_center_ljust_rjust_fillchar(dtype):
    values = xr.DataArray(["a", "bb", "cccc", "ddddd", "eeeeee"]).astype(dtype)
    result = values.str.center(5, fillchar="X")
    expected = xr.DataArray(["XXaXX", "XXbbX", "Xcccc", "ddddd", "eeeeee"])
    assert_equal(result, expected.astype(dtype))

    result = values.str.ljust(5, fillchar="X")
    expected = xr.DataArray(["aXXXX", "bbXXX", "ccccX", "ddddd", "eeeeee"])
    assert_equal(result, expected.astype(dtype))

    result = values.str.rjust(5, fillchar="X")
    expected = xr.DataArray(["XXXXa", "XXXbb", "Xcccc", "ddddd", "eeeeee"])
    assert_equal(result, expected.astype(dtype))

    # If fillchar is not a charatter, normal str raises TypeError
    # 'aaa'.ljust(5, 'XY')
    # TypeError: must be char, not str
    template = "fillchar must be a character, not {dtype}"

    with pytest.raises(TypeError, match=template.format(dtype="str")):
        values.str.center(5, fillchar="XY")

    with pytest.raises(TypeError, match=template.format(dtype="str")):
        values.str.ljust(5, fillchar="XY")

    with pytest.raises(TypeError, match=template.format(dtype="str")):
        values.str.rjust(5, fillchar="XY")


def test_zfill(dtype):
    values = xr.DataArray(["1", "22", "aaa", "333", "45678"]).astype(dtype)

    result = values.str.zfill(5)
    expected = xr.DataArray(["00001", "00022", "00aaa", "00333", "45678"])
    assert_equal(result, expected.astype(dtype))

    result = values.str.zfill(3)
    expected = xr.DataArray(["001", "022", "aaa", "333", "45678"])
    assert_equal(result, expected.astype(dtype))


def test_slice(dtype):
    arr = xr.DataArray(["aafootwo", "aabartwo", "aabazqux"]).astype(dtype)

    result = arr.str.slice(2, 5)
    exp = xr.DataArray(["foo", "bar", "baz"]).astype(dtype)
    assert_equal(result, exp)

    for start, stop, step in [(0, 3, -1), (None, None, -1), (3, 10, 2), (3, 0, -1)]:
        try:
            result = arr.str[start:stop:step]
            expected = xr.DataArray([s[start:stop:step] for s in arr.values])
            assert_equal(result, expected.astype(dtype))
        except IndexError:
            print(f"failed on {start}:{stop}:{step}")
            raise


def test_slice_replace(dtype):
    da = lambda x: xr.DataArray(x).astype(dtype)
    values = da(["short", "a bit longer", "evenlongerthanthat", ""])

    expected = da(["shrt", "a it longer", "evnlongerthanthat", ""])
    result = values.str.slice_replace(2, 3)
    assert_equal(result, expected)

    expected = da(["shzrt", "a zit longer", "evznlongerthanthat", "z"])
    result = values.str.slice_replace(2, 3, "z")
    assert_equal(result, expected)

    expected = da(["shzort", "a zbit longer", "evzenlongerthanthat", "z"])
    result = values.str.slice_replace(2, 2, "z")
    assert_equal(result, expected)

    expected = da(["shzort", "a zbit longer", "evzenlongerthanthat", "z"])
    result = values.str.slice_replace(2, 1, "z")
    assert_equal(result, expected)

    expected = da(["shorz", "a bit longez", "evenlongerthanthaz", "z"])
    result = values.str.slice_replace(-1, None, "z")
    assert_equal(result, expected)

    expected = da(["zrt", "zer", "zat", "z"])
    result = values.str.slice_replace(None, -2, "z")
    assert_equal(result, expected)

    expected = da(["shortz", "a bit znger", "evenlozerthanthat", "z"])
    result = values.str.slice_replace(6, 8, "z")
    assert_equal(result, expected)

    expected = da(["zrt", "a zit longer", "evenlongzerthanthat", "z"])
    result = values.str.slice_replace(-10, 3, "z")
    assert_equal(result, expected)


def test_strip_lstrip_rstrip(dtype):
    values = xr.DataArray(["  aa   ", " bb \n", "cc  "]).astype(dtype)

    result = values.str.strip()
    expected = xr.DataArray(["aa", "bb", "cc"]).astype(dtype)
    assert_equal(result, expected)

    result = values.str.lstrip()
    expected = xr.DataArray(["aa   ", "bb \n", "cc  "]).astype(dtype)
    assert_equal(result, expected)

    result = values.str.rstrip()
    expected = xr.DataArray(["  aa", " bb", "cc"]).astype(dtype)
    assert_equal(result, expected)


def test_strip_lstrip_rstrip_args(dtype):
    values = xr.DataArray(["xxABCxx", "xx BNSD", "LDFJH xx"]).astype(dtype)

    rs = values.str.strip("x")
    xp = xr.DataArray(["ABC", " BNSD", "LDFJH "]).astype(dtype)
    assert_equal(rs, xp)

    rs = values.str.lstrip("x")
    xp = xr.DataArray(["ABCxx", " BNSD", "LDFJH xx"]).astype(dtype)
    assert_equal(rs, xp)

    rs = values.str.rstrip("x")
    xp = xr.DataArray(["xxABC", "xx BNSD", "LDFJH "]).astype(dtype)
    assert_equal(rs, xp)


def test_wrap():
    # test values are: two words less than width, two words equal to width,
    # two words greater than width, one word less than width, one word
    # equal to width, one word greater than width, multiple tokens with
    # trailing whitespace equal to width
    values = xr.DataArray(
        [
            "hello world",
            "hello world!",
            "hello world!!",
            "abcdefabcde",
            "abcdefabcdef",
            "abcdefabcdefa",
            "ab ab ab ab ",
            "ab ab ab ab a",
            "\t",
        ]
    )

    # expected values
    xp = xr.DataArray(
        [
            "hello world",
            "hello world!",
            "hello\nworld!!",
            "abcdefabcde",
            "abcdefabcdef",
            "abcdefabcdef\na",
            "ab ab ab ab",
            "ab ab ab ab\na",
            "",
        ]
    )

    rs = values.str.wrap(12, break_long_words=True)
    assert_equal(rs, xp)

    # test with pre and post whitespace (non-unicode), NaN, and non-ascii
    # Unicode
    values = xr.DataArray(["  pre  ", "\xac\u20ac\U00008000 abadcafe"])
    xp = xr.DataArray(["  pre", "\xac\u20ac\U00008000 ab\nadcafe"])
    rs = values.str.wrap(6)
    assert_equal(rs, xp)


def test_get(dtype):
    values = xr.DataArray(["a_b_c", "c_d_e", "f_g_h"]).astype(dtype)

    result = values.str[2]
    expected = xr.DataArray(["b", "d", "g"]).astype(dtype)
    assert_equal(result, expected)

    # bounds testing
    values = xr.DataArray(["1_2_3_4_5", "6_7_8_9_10", "11_12"]).astype(dtype)

    # positive index
    result = values.str[5]
    expected = xr.DataArray(["_", "_", ""]).astype(dtype)
    assert_equal(result, expected)

    # negative index
    result = values.str[-6]
    expected = xr.DataArray(["_", "8", ""]).astype(dtype)
    assert_equal(result, expected)


def test_encode_decode():
    data = xr.DataArray(["a", "b", "a\xe4"])
    encoded = data.str.encode("utf-8")
    decoded = encoded.str.decode("utf-8")
    assert_equal(data, decoded)


def test_encode_decode_errors():
    encodeBase = xr.DataArray(["a", "b", "a\x9d"])

    msg = (
        r"'charmap' codec can't encode character '\\x9d' in position 1:"
        " character maps to <undefined>"
    )
    with pytest.raises(UnicodeEncodeError, match=msg):
        encodeBase.str.encode("cp1252")

    f = lambda x: x.encode("cp1252", "ignore")
    result = encodeBase.str.encode("cp1252", "ignore")
    expected = xr.DataArray([f(x) for x in encodeBase.values.tolist()])
    assert_equal(result, expected)

    decodeBase = xr.DataArray([b"a", b"b", b"a\x9d"])

    msg = (
        "'charmap' codec can't decode byte 0x9d in position 1:"
        " character maps to <undefined>"
    )
    with pytest.raises(UnicodeDecodeError, match=msg):
        decodeBase.str.decode("cp1252")

    f = lambda x: x.decode("cp1252", "ignore")
    result = decodeBase.str.decode("cp1252", "ignore")
    expected = xr.DataArray([f(x) for x in decodeBase.values.tolist()])
    assert_equal(result, expected)
