# -*- coding: utf-8 -*-

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
    assert result.dtype == expected.dtype
    assert_equal(result, expected)


def test_count(dtype):
    values = xr.DataArray(["foo", "foofoo", "foooofooofommmfoo"]).astype(dtype)
    result = values.str.count("f[o]+")
    expected = xr.DataArray([1, 2, 4])
    assert result.dtype == expected.dtype
    assert_equal(result, expected)


def test_contains(dtype):
    values = xr.DataArray(["Foo", "xYz", "fOOomMm__fOo", "MMM_"]).astype(dtype)

    # case insensitive using regex
    result = values.str.contains("FOO|mmm", case=False)
    expected = xr.DataArray([True, False, True, True])
    assert result.dtype == expected.dtype
    assert_equal(result, expected)

    # case insensitive without regex
    result = values.str.contains("foo", regex=False, case=False)
    expected = xr.DataArray([True, False, True, False])
    assert result.dtype == expected.dtype
    assert_equal(result, expected)


def test_starts_ends_with(dtype):
    values = xr.DataArray(["om", "foo_nom", "nom", "bar_foo", "foo"]).astype(dtype)

    result = values.str.startswith("foo")
    expected = xr.DataArray([False, True, False, False, True])
    assert result.dtype == expected.dtype
    assert_equal(result, expected)

    result = values.str.endswith("foo")
    expected = xr.DataArray([False, False, False, True, True])
    assert result.dtype == expected.dtype
    assert_equal(result, expected)


def test_case_bytes(dtype):
    dtype = np.bytes_
    value = xr.DataArray(["SOme wOrd"]).astype(dtype)

    exp_capitalized = xr.DataArray(["Some word"]).astype(dtype)
    exp_lowered = xr.DataArray(["some word"]).astype(dtype)
    exp_swapped = xr.DataArray(["soME WoRD"]).astype(dtype)
    exp_titled = xr.DataArray(["Some Word"]).astype(dtype)
    exp_uppered = xr.DataArray(["SOME WORD"]).astype(dtype)

    res_capitalized = value.str.capitalize()
    res_lowered = value.str.lower()
    res_swapped = value.str.swapcase()
    res_titled = value.str.title()
    res_uppered = value.str.upper()

    assert res_capitalized.dtype == exp_capitalized.dtype
    assert res_lowered.dtype == exp_lowered.dtype
    assert res_swapped.dtype == exp_swapped.dtype
    assert res_titled.dtype == exp_titled.dtype
    assert res_uppered.dtype == exp_uppered.dtype

    assert_equal(value.str.capitalize(), exp_capitalized)
    assert_equal(value.str.lower(), exp_lowered)
    assert_equal(value.str.swapcase(), exp_swapped)
    assert_equal(value.str.title(), exp_titled)
    assert_equal(value.str.upper(), exp_uppered)


def test_case_str():
    dtype = np.str_

    # This string includes some unicode characters
    # that are common case management corner cases
    value = xr.DataArray(["SOme wOrd Ǆ ß ᾛ ΣΣ ﬃ⁵Å Ç Ⅰ"]).astype(dtype)

    exp_capitalized = xr.DataArray(["Some word ǆ ß ᾓ σς ﬃ⁵å ç ⅰ"]).astype(dtype)
    exp_lowered = xr.DataArray(["some word ǆ ß ᾓ σς ﬃ⁵å ç ⅰ"]).astype(dtype)
    exp_swapped = xr.DataArray(["soME WoRD ǆ SS ᾛ σς FFI⁵å ç ⅰ"]).astype(dtype)
    exp_titled = xr.DataArray(["Some Word ǅ Ss ᾛ Σς Ffi⁵Å Ç Ⅰ"]).astype(dtype)
    exp_uppered = xr.DataArray(["SOME WORD Ǆ SS ἫΙ ΣΣ FFI⁵Å Ç Ⅰ"]).astype(dtype)
    exp_casefolded = xr.DataArray(["some word ǆ ss ἣι σσ ffi⁵å ç ⅰ"]).astype(dtype)

    exp_norm_nfc = xr.DataArray(["SOme wOrd Ǆ ß ᾛ ΣΣ ﬃ⁵Å Ç Ⅰ"]).astype(dtype)
    exp_norm_nfkc = xr.DataArray(["SOme wOrd DŽ ß ᾛ ΣΣ ffi5Å Ç I"]).astype(dtype)
    exp_norm_nfd = xr.DataArray(["SOme wOrd Ǆ ß ᾛ ΣΣ ﬃ⁵Å Ç Ⅰ"]).astype(dtype)
    exp_norm_nfkd = xr.DataArray(["SOme wOrd DŽ ß ᾛ ΣΣ ffi5Å Ç I"]).astype(dtype)

    res_capitalized = value.str.capitalize()
    res_casefolded = value.str.casefold()
    res_lowered = value.str.lower()
    res_swapped = value.str.swapcase()
    res_titled = value.str.title()
    res_uppered = value.str.upper()

    res_norm_nfc = value.str.normalize("NFC")
    res_norm_nfd = value.str.normalize("NFD")
    res_norm_nfkc = value.str.normalize("NFKC")
    res_norm_nfkd = value.str.normalize("NFKD")

    assert res_capitalized.dtype == exp_capitalized.dtype
    assert res_casefolded.dtype == exp_casefolded.dtype
    assert res_lowered.dtype == exp_lowered.dtype
    assert res_swapped.dtype == exp_swapped.dtype
    assert res_titled.dtype == exp_titled.dtype
    assert res_uppered.dtype == exp_uppered.dtype

    assert res_norm_nfc.dtype == exp_norm_nfc.dtype
    assert res_norm_nfd.dtype == exp_norm_nfd.dtype
    assert res_norm_nfkc.dtype == exp_norm_nfkc.dtype
    assert res_norm_nfkd.dtype == exp_norm_nfkd.dtype

    assert_equal(res_capitalized, exp_capitalized)
    assert_equal(res_casefolded, exp_casefolded)
    assert_equal(res_lowered, exp_lowered)
    assert_equal(res_swapped, exp_swapped)
    assert_equal(res_titled, exp_titled)
    assert_equal(res_uppered, exp_uppered)

    assert_equal(res_norm_nfc, exp_norm_nfc)
    assert_equal(res_norm_nfd, exp_norm_nfd)
    assert_equal(res_norm_nfkc, exp_norm_nfkc)
    assert_equal(res_norm_nfkd, exp_norm_nfkd)


def test_replace(dtype):
    values = xr.DataArray(["fooBAD__barBAD"]).astype(dtype)
    result = values.str.replace("BAD[_]*", "")
    expected = xr.DataArray(["foobar"]).astype(dtype)
    assert result.dtype == expected.dtype
    assert_equal(result, expected)

    result = values.str.replace("BAD[_]*", "", n=1)
    expected = xr.DataArray(["foobarBAD"]).astype(dtype)
    assert result.dtype == expected.dtype
    assert_equal(result, expected)

    s = xr.DataArray(["A", "B", "C", "Aaba", "Baca", "", "CABA", "dog", "cat"]).astype(
        dtype
    )
    result = s.str.replace("A", "YYY")
    expected = xr.DataArray(
        ["YYY", "B", "C", "YYYaba", "Baca", "", "CYYYBYYY", "dog", "cat"]
    ).astype(dtype)
    assert result.dtype == expected.dtype
    assert_equal(result, expected)

    result = s.str.replace("A", "YYY", case=False)
    expected = xr.DataArray(
        ["YYY", "B", "C", "YYYYYYbYYY", "BYYYcYYY", "", "CYYYBYYY", "dog", "cYYYt"]
    ).astype(dtype)
    assert result.dtype == expected.dtype
    assert_equal(result, expected)

    result = s.str.replace("^.a|dog", "XX-XX ", case=False)
    expected = xr.DataArray(
        ["A", "B", "C", "XX-XX ba", "XX-XX ca", "", "XX-XX BA", "XX-XX ", "XX-XX t"]
    ).astype(dtype)
    assert result.dtype == expected.dtype
    assert_equal(result, expected)


def test_replace_callable():
    values = xr.DataArray(["fooBAD__barBAD"])

    # test with callable
    repl = lambda m: m.group(0).swapcase()
    result = values.str.replace("[a-z][A-Z]{2}", repl, n=2)
    exp = xr.DataArray(["foObaD__baRbaD"])
    assert result.dtype == exp.dtype
    assert_equal(result, exp)

    # test regex named groups
    values = xr.DataArray(["Foo Bar Baz"])
    pat = r"(?P<first>\w+) (?P<middle>\w+) (?P<last>\w+)"
    repl = lambda m: m.group("middle").swapcase()
    result = values.str.replace(pat, repl)
    exp = xr.DataArray(["bAR"])
    assert result.dtype == exp.dtype
    assert_equal(result, exp)


def test_replace_unicode():
    # flags + unicode
    values = xr.DataArray([b"abcd,\xc3\xa0".decode("utf-8")])
    expected = xr.DataArray([b"abcd, \xc3\xa0".decode("utf-8")])
    pat = re.compile(r"(?<=\w),(?=\w)", flags=re.UNICODE)
    result = values.str.replace(pat, ", ")
    assert result.dtype == expected.dtype
    assert_equal(result, expected)


def test_replace_compiled_regex(dtype):
    values = xr.DataArray(["fooBAD__barBAD"]).astype(dtype)
    # test with compiled regex
    pat = re.compile(dtype("BAD[_]*"))
    result = values.str.replace(pat, "")
    expected = xr.DataArray(["foobar"]).astype(dtype)
    assert result.dtype == expected.dtype
    assert_equal(result, expected)

    result = values.str.replace(pat, "", n=1)
    expected = xr.DataArray(["foobarBAD"]).astype(dtype)
    assert result.dtype == expected.dtype
    assert_equal(result, expected)

    # case and flags provided to str.replace will have no effect
    # and will produce warnings
    values = xr.DataArray(["fooBAD__barBAD__bad"]).astype(dtype)
    pat = re.compile(dtype("BAD[_]*"))

    with pytest.raises(ValueError, match="flags cannot be set"):
        result = values.str.replace(pat, "", flags=re.IGNORECASE)

    with pytest.raises(ValueError, match="case cannot be set"):
        result = values.str.replace(pat, "", case=False)

    with pytest.raises(ValueError, match="case cannot be set"):
        result = values.str.replace(pat, "", case=True)

    # test with callable
    values = xr.DataArray(["fooBAD__barBAD"]).astype(dtype)
    repl = lambda m: m.group(0).swapcase()
    pat = re.compile(dtype("[a-z][A-Z]{2}"))
    result = values.str.replace(pat, repl, n=2)
    expected = xr.DataArray(["foObaD__baRbaD"]).astype(dtype)
    assert result.dtype == expected.dtype
    assert_equal(result, expected)


def test_replace_literal(dtype):
    # GH16808 literal replace (regex=False vs regex=True)
    values = xr.DataArray(["f.o", "foo"]).astype(dtype)
    expected = xr.DataArray(["bao", "bao"]).astype(dtype)
    result = values.str.replace("f.", "ba")
    assert result.dtype == expected.dtype
    assert_equal(result, expected)

    expected = xr.DataArray(["bao", "foo"]).astype(dtype)
    result = values.str.replace("f.", "ba", regex=False)
    assert result.dtype == expected.dtype
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


def test_extract_extractall_findall_empty_raises(dtype):
    pat_str = r"a_\w+_b_\d+_c_.*"
    pat_re = re.compile(pat_str)

    value = xr.DataArray(
        [
            ["a_first_b_1_c_de", "a_second_b_22_c_efh", "a_third_b_333_c_hijk"],
            [
                "a_fourth_b_4444_c_klmno",
                "a_fifth_b_5555_c_opqr",
                "a_sixth_b_66666_c_rst",
            ],
        ],
        dims=["X", "Y"],
    ).astype(dtype)

    with pytest.raises(ValueError):
        value.str.extract(pat=pat_str, dim="ZZ")

    with pytest.raises(ValueError):
        value.str.extract(pat=pat_re, dim="ZZ")

    with pytest.raises(ValueError):
        value.str.extractall(pat=pat_str, group_dim="XX", match_dim="YY")

    with pytest.raises(ValueError):
        value.str.extractall(pat=pat_re, group_dim="XX", match_dim="YY")

    with pytest.raises(ValueError):
        value.str.findall(pat=pat_str)

    with pytest.raises(ValueError):
        value.str.findall(pat=pat_re)


def test_extract_multi_None_raises(dtype):
    pat_str = r"a_(\w+)_b_(\d+)_c_.*"
    pat_re = re.compile(pat_str)

    value = xr.DataArray(
        [
            ["a_first_b_1_c_de", "a_second_b_22_c_efh", "a_third_b_333_c_hijk"],
            [
                "a_fourth_b_4444_c_klmno",
                "a_fifth_b_5555_c_opqr",
                "a_sixth_b_66666_c_rst",
            ],
        ],
        dims=["X", "Y"],
    ).astype(dtype)

    with pytest.raises(ValueError):
        value.str.extract(pat=pat_str, dim=None)

    with pytest.raises(ValueError):
        value.str.extract(pat=pat_re, dim=None)


def test_extract_extractall_findall_case_re_raises(dtype):
    pat_str = r"a_\w+_b_\d+_c_.*"
    pat_re = re.compile(pat_str)

    value = xr.DataArray(
        [
            ["a_first_b_1_c_de", "a_second_b_22_c_efh", "a_third_b_333_c_hijk"],
            [
                "a_fourth_b_4444_c_klmno",
                "a_fifth_b_5555_c_opqr",
                "a_sixth_b_66666_c_rst",
            ],
        ],
        dims=["X", "Y"],
    ).astype(dtype)

    with pytest.raises(ValueError):
        value.str.extract(pat=pat_re, case=True, dim="ZZ")

    with pytest.raises(ValueError):
        value.str.extract(pat=pat_re, case=False, dim="ZZ")

    with pytest.raises(ValueError):
        value.str.extractall(pat=pat_re, case=True, group_dim="XX", match_dim="YY")

    with pytest.raises(ValueError):
        value.str.extractall(pat=pat_re, case=False, group_dim="XX", match_dim="YY")

    with pytest.raises(ValueError):
        value.str.findall(pat=pat_re, case=True)

    with pytest.raises(ValueError):
        value.str.findall(pat=pat_re, case=False)


def test_extract_extractall_name_collision_raises(dtype):
    pat_str = r"a_(\w+)_b_\d+_c_.*"
    pat_re = re.compile(pat_str)

    value = xr.DataArray(
        [
            ["a_first_b_1_c_de", "a_second_b_22_c_efh", "a_third_b_333_c_hijk"],
            [
                "a_fourth_b_4444_c_klmno",
                "a_fifth_b_5555_c_opqr",
                "a_sixth_b_66666_c_rst",
            ],
        ],
        dims=["X", "Y"],
    ).astype(dtype)

    with pytest.raises(KeyError):
        value.str.extract(pat=pat_str, dim="X")

    with pytest.raises(KeyError):
        value.str.extract(pat=pat_re, dim="X")

    with pytest.raises(KeyError):
        value.str.extractall(pat=pat_str, group_dim="X", match_dim="ZZ")

    with pytest.raises(KeyError):
        value.str.extractall(pat=pat_re, group_dim="X", match_dim="YY")

    with pytest.raises(KeyError):
        value.str.extractall(pat=pat_str, group_dim="XX", match_dim="Y")

    with pytest.raises(KeyError):
        value.str.extractall(pat=pat_re, group_dim="XX", match_dim="Y")

    with pytest.raises(KeyError):
        value.str.extractall(pat=pat_str, group_dim="ZZ", match_dim="ZZ")

    with pytest.raises(KeyError):
        value.str.extractall(pat=pat_re, group_dim="ZZ", match_dim="ZZ")


def test_extract_single_case(dtype):
    pat_str = r"(\w+)_Xy_\d*"
    conv = {np.unicode_: str, np.bytes_: lambda x: bytes(x, encoding="UTF-8")}[dtype]
    pat_re = re.compile(conv(pat_str))

    value = xr.DataArray(
        [
            ["a_Xy_0", "ab_xY_10-bab_Xy_110-baab_Xy_1100", "abc_Xy_01-cbc_Xy_2210"],
            [
                "abcd_Xy_-dcd_Xy_33210-dccd_Xy_332210",
                "",
                "abcdef_Xy_101-fef_Xy_5543210",
            ],
        ],
        dims=["X", "Y"],
    ).astype(dtype)

    targ_none = xr.DataArray(
        [["a", "bab", "abc"], ["abcd", "", "abcdef"]], dims=["X", "Y"]
    ).astype(dtype)
    targ_dim = xr.DataArray(
        [[["a"], ["bab"], ["abc"]], [["abcd"], [""], ["abcdef"]]], dims=["X", "Y", "XX"]
    ).astype(dtype)

    res_str_none = value.str.extract(pat=pat_str, dim=None)
    res_str_dim = value.str.extract(pat=pat_str, dim="XX")
    res_str_none_case = value.str.extract(pat=pat_str, dim=None, case=True)
    res_str_dim_case = value.str.extract(pat=pat_str, dim="XX", case=True)
    res_re_none = value.str.extract(pat=pat_re, dim=None)
    res_re_dim = value.str.extract(pat=pat_re, dim="XX")

    assert res_str_none.dtype == targ_none.dtype
    assert res_str_dim.dtype == targ_dim.dtype
    assert res_str_none_case.dtype == targ_none.dtype
    assert res_str_dim_case.dtype == targ_dim.dtype
    assert res_re_none.dtype == targ_none.dtype
    assert res_re_dim.dtype == targ_dim.dtype

    assert_equal(res_str_none, targ_none)
    assert_equal(res_str_dim, targ_dim)
    assert_equal(res_str_none_case, targ_none)
    assert_equal(res_str_dim_case, targ_dim)
    assert_equal(res_re_none, targ_none)
    assert_equal(res_re_dim, targ_dim)


def test_extract_single_nocase(dtype):
    pat_str = r"(\w+)_Xy_\d*"
    conv = {np.unicode_: str, np.bytes_: lambda x: bytes(x, encoding="UTF-8")}[dtype]
    pat_re = re.compile(conv(pat_str), flags=re.IGNORECASE)

    value = xr.DataArray(
        [
            ["a_Xy_0", "ab_xY_10-bab_Xy_110-baab_Xy_1100", "abc_Xy_01-cbc_Xy_2210"],
            [
                "abcd_Xy_-dcd_Xy_33210-dccd_Xy_332210",
                "",
                "abcdef_Xy_101-fef_Xy_5543210",
            ],
        ],
        dims=["X", "Y"],
    ).astype(dtype)

    targ_none = xr.DataArray(
        [["a", "ab", "abc"], ["abcd", "", "abcdef"]], dims=["X", "Y"]
    ).astype(dtype)
    targ_dim = xr.DataArray(
        [[["a"], ["ab"], ["abc"]], [["abcd"], [""], ["abcdef"]]], dims=["X", "Y", "XX"]
    ).astype(dtype)

    res_str_none = value.str.extract(pat=pat_str, dim=None, case=False)
    res_str_dim = value.str.extract(pat=pat_str, dim="XX", case=False)
    res_re_none = value.str.extract(pat=pat_re, dim=None)
    res_re_dim = value.str.extract(pat=pat_re, dim="XX")

    assert res_re_dim.dtype == targ_none.dtype
    assert res_str_dim.dtype == targ_dim.dtype
    assert res_re_none.dtype == targ_none.dtype
    assert res_re_dim.dtype == targ_dim.dtype

    assert_equal(res_str_none, targ_none)
    assert_equal(res_str_dim, targ_dim)
    assert_equal(res_re_none, targ_none)
    assert_equal(res_re_dim, targ_dim)


def test_extract_multi_case(dtype):
    pat_str = r"(\w+)_Xy_(\d*)"
    conv = {np.unicode_: str, np.bytes_: lambda x: bytes(x, encoding="UTF-8")}[dtype]
    pat_re = re.compile(conv(pat_str))

    value = xr.DataArray(
        [
            ["a_Xy_0", "ab_xY_10-bab_Xy_110-baab_Xy_1100", "abc_Xy_01-cbc_Xy_2210"],
            [
                "abcd_Xy_-dcd_Xy_33210-dccd_Xy_332210",
                "",
                "abcdef_Xy_101-fef_Xy_5543210",
            ],
        ],
        dims=["X", "Y"],
    ).astype(dtype)

    targ = xr.DataArray(
        [
            [["a", "0"], ["bab", "110"], ["abc", "01"]],
            [["abcd", ""], ["", ""], ["abcdef", "101"]],
        ],
        dims=["X", "Y", "XX"],
    ).astype(dtype)

    res_str = value.str.extract(pat=pat_str, dim="XX")
    res_re = value.str.extract(pat=pat_re, dim="XX")
    res_str_case = value.str.extract(pat=pat_str, dim="XX", case=True)

    assert res_str.dtype == targ.dtype
    assert res_re.dtype == targ.dtype
    assert res_str_case.dtype == targ.dtype

    assert_equal(res_str, targ)
    assert_equal(res_re, targ)
    assert_equal(res_str_case, targ)


def test_extract_multi_nocase(dtype):
    pat_str = r"(\w+)_Xy_(\d*)"
    conv = {np.unicode_: str, np.bytes_: lambda x: bytes(x, encoding="UTF-8")}[dtype]
    pat_re = re.compile(conv(pat_str), flags=re.IGNORECASE)

    value = xr.DataArray(
        [
            ["a_Xy_0", "ab_xY_10-bab_Xy_110-baab_Xy_1100", "abc_Xy_01-cbc_Xy_2210"],
            [
                "abcd_Xy_-dcd_Xy_33210-dccd_Xy_332210",
                "",
                "abcdef_Xy_101-fef_Xy_5543210",
            ],
        ],
        dims=["X", "Y"],
    ).astype(dtype)

    targ = xr.DataArray(
        [
            [["a", "0"], ["ab", "10"], ["abc", "01"]],
            [["abcd", ""], ["", ""], ["abcdef", "101"]],
        ],
        dims=["X", "Y", "XX"],
    ).astype(dtype)

    res_str = value.str.extract(pat=pat_str, dim="XX", case=False)
    res_re = value.str.extract(pat=pat_re, dim="XX")

    assert res_str.dtype == targ.dtype
    assert res_re.dtype == targ.dtype

    assert_equal(res_str, targ)
    assert_equal(res_re, targ)


def test_extractall_single_single_case(dtype):
    pat_str = r"(\w+)_Xy_\d*"
    conv = {np.unicode_: str, np.bytes_: lambda x: bytes(x, encoding="UTF-8")}[dtype]
    pat_re = re.compile(conv(pat_str))

    value = xr.DataArray(
        [["a_Xy_0", "ab_xY_10", "abc_Xy_01"], ["abcd_Xy_", "", "abcdef_Xy_101"]],
        dims=["X", "Y"],
    ).astype(dtype)

    targ = xr.DataArray(
        [[[["a"]], [[""]], [["abc"]]], [[["abcd"]], [[""]], [["abcdef"]]]],
        dims=["X", "Y", "XX", "YY"],
    ).astype(dtype)

    res_str = value.str.extractall(pat=pat_str, group_dim="XX", match_dim="YY")
    res_re = value.str.extractall(pat=pat_re, group_dim="XX", match_dim="YY")
    res_str_case = value.str.extractall(
        pat=pat_str, group_dim="XX", match_dim="YY", case=True
    )

    assert res_str.dtype == targ.dtype
    assert res_re.dtype == targ.dtype
    assert res_str_case.dtype == targ.dtype

    assert_equal(res_str, targ)
    assert_equal(res_re, targ)
    assert_equal(res_str_case, targ)


def test_extractall_single_single_nocase(dtype):
    pat_str = r"(\w+)_Xy_\d*"
    conv = {np.unicode_: str, np.bytes_: lambda x: bytes(x, encoding="UTF-8")}[dtype]
    pat_re = re.compile(conv(pat_str), flags=re.I)

    value = xr.DataArray(
        [["a_Xy_0", "ab_xY_10", "abc_Xy_01"], ["abcd_Xy_", "", "abcdef_Xy_101"]],
        dims=["X", "Y"],
    ).astype(dtype)

    targ = xr.DataArray(
        [[[["a"]], [["ab"]], [["abc"]]], [[["abcd"]], [[""]], [["abcdef"]]]],
        dims=["X", "Y", "XX", "YY"],
    ).astype(dtype)

    res_str = value.str.extractall(
        pat=pat_str, group_dim="XX", match_dim="YY", case=False
    )
    res_re = value.str.extractall(pat=pat_re, group_dim="XX", match_dim="YY")

    assert res_str.dtype == targ.dtype
    assert res_re.dtype == targ.dtype

    assert_equal(res_str, targ)
    assert_equal(res_re, targ)


def test_extractall_single_multi_case(dtype):
    pat_str = r"(\w+)_Xy_\d*"
    conv = {np.unicode_: str, np.bytes_: lambda x: bytes(x, encoding="UTF-8")}[dtype]
    pat_re = re.compile(conv(pat_str))

    value = xr.DataArray(
        [
            ["a_Xy_0", "ab_xY_10-bab_Xy_110-baab_Xy_1100", "abc_Xy_01-cbc_Xy_2210"],
            [
                "abcd_Xy_-dcd_Xy_33210-dccd_Xy_332210",
                "",
                "abcdef_Xy_101-fef_Xy_5543210",
            ],
        ],
        dims=["X", "Y"],
    ).astype(dtype)

    targ = xr.DataArray(
        [
            [[["a"], [""], [""]], [["bab"], ["baab"], [""]], [["abc"], ["cbc"], [""]]],
            [
                [["abcd"], ["dcd"], ["dccd"]],
                [[""], [""], [""]],
                [["abcdef"], ["fef"], [""]],
            ],
        ],
        dims=["X", "Y", "XX", "YY"],
    ).astype(dtype)

    res_str = value.str.extractall(pat=pat_str, group_dim="XX", match_dim="YY")
    res_re = value.str.extractall(pat=pat_re, group_dim="XX", match_dim="YY")
    res_str_case = value.str.extractall(
        pat=pat_str, group_dim="XX", match_dim="YY", case=True
    )

    assert res_str.dtype == targ.dtype
    assert res_re.dtype == targ.dtype
    assert res_str_case.dtype == targ.dtype

    assert_equal(res_str, targ)
    assert_equal(res_re, targ)
    assert_equal(res_str_case, targ)


def test_extractall_single_multi_nocase(dtype):
    pat_str = r"(\w+)_Xy_\d*"
    conv = {np.unicode_: str, np.bytes_: lambda x: bytes(x, encoding="UTF-8")}[dtype]
    pat_re = re.compile(conv(pat_str), flags=re.I)

    value = xr.DataArray(
        [
            ["a_Xy_0", "ab_xY_10-bab_Xy_110-baab_Xy_1100", "abc_Xy_01-cbc_Xy_2210"],
            [
                "abcd_Xy_-dcd_Xy_33210-dccd_Xy_332210",
                "",
                "abcdef_Xy_101-fef_Xy_5543210",
            ],
        ],
        dims=["X", "Y"],
    ).astype(dtype)

    targ = xr.DataArray(
        [
            [
                [["a"], [""], [""]],
                [["ab"], ["bab"], ["baab"]],
                [["abc"], ["cbc"], [""]],
            ],
            [
                [["abcd"], ["dcd"], ["dccd"]],
                [[""], [""], [""]],
                [["abcdef"], ["fef"], [""]],
            ],
        ],
        dims=["X", "Y", "XX", "YY"],
    ).astype(dtype)

    res_str = value.str.extractall(
        pat=pat_str, group_dim="XX", match_dim="YY", case=False
    )
    res_re = value.str.extractall(pat=pat_re, group_dim="XX", match_dim="YY")

    assert res_str.dtype == targ.dtype
    assert res_re.dtype == targ.dtype

    assert_equal(res_str, targ)
    assert_equal(res_re, targ)


def test_extractall_multi_single_case(dtype):
    pat_str = r"(\w+)_Xy_(\d*)"
    conv = {np.unicode_: str, np.bytes_: lambda x: bytes(x, encoding="UTF-8")}[dtype]
    pat_re = re.compile(conv(pat_str))

    value = xr.DataArray(
        [["a_Xy_0", "ab_xY_10", "abc_Xy_01"], ["abcd_Xy_", "", "abcdef_Xy_101"]],
        dims=["X", "Y"],
    ).astype(dtype)

    targ = xr.DataArray(
        [
            [[["a", "0"]], [["", ""]], [["abc", "01"]]],
            [[["abcd", ""]], [["", ""]], [["abcdef", "101"]]],
        ],
        dims=["X", "Y", "XX", "YY"],
    ).astype(dtype)

    res_str = value.str.extractall(pat=pat_str, group_dim="XX", match_dim="YY")
    res_re = value.str.extractall(pat=pat_re, group_dim="XX", match_dim="YY")
    res_str_case = value.str.extractall(
        pat=pat_str, group_dim="XX", match_dim="YY", case=True
    )

    assert res_str.dtype == targ.dtype
    assert res_re.dtype == targ.dtype
    assert res_str_case.dtype == targ.dtype

    assert_equal(res_str, targ)
    assert_equal(res_re, targ)
    assert_equal(res_str_case, targ)


def test_extractall_multi_single_nocase(dtype):
    pat_str = r"(\w+)_Xy_(\d*)"
    conv = {np.unicode_: str, np.bytes_: lambda x: bytes(x, encoding="UTF-8")}[dtype]
    pat_re = re.compile(conv(pat_str), flags=re.I)

    value = xr.DataArray(
        [["a_Xy_0", "ab_xY_10", "abc_Xy_01"], ["abcd_Xy_", "", "abcdef_Xy_101"]],
        dims=["X", "Y"],
    ).astype(dtype)

    targ = xr.DataArray(
        [
            [[["a", "0"]], [["ab", "10"]], [["abc", "01"]]],
            [[["abcd", ""]], [["", ""]], [["abcdef", "101"]]],
        ],
        dims=["X", "Y", "XX", "YY"],
    ).astype(dtype)

    res_str = value.str.extractall(
        pat=pat_str, group_dim="XX", match_dim="YY", case=False
    )
    res_re = value.str.extractall(pat=pat_re, group_dim="XX", match_dim="YY")

    assert res_str.dtype == targ.dtype
    assert res_re.dtype == targ.dtype

    assert_equal(res_str, targ)
    assert_equal(res_re, targ)


def test_extractall_multi_multi_case(dtype):
    pat_str = r"(\w+)_Xy_(\d*)"
    conv = {np.unicode_: str, np.bytes_: lambda x: bytes(x, encoding="UTF-8")}[dtype]
    pat_re = re.compile(conv(pat_str))

    value = xr.DataArray(
        [
            ["a_Xy_0", "ab_xY_10-bab_Xy_110-baab_Xy_1100", "abc_Xy_01-cbc_Xy_2210"],
            [
                "abcd_Xy_-dcd_Xy_33210-dccd_Xy_332210",
                "",
                "abcdef_Xy_101-fef_Xy_5543210",
            ],
        ],
        dims=["X", "Y"],
    ).astype(dtype)

    targ = xr.DataArray(
        [
            [
                [["a", "0"], ["", ""], ["", ""]],
                [["bab", "110"], ["baab", "1100"], ["", ""]],
                [["abc", "01"], ["cbc", "2210"], ["", ""]],
            ],
            [
                [["abcd", ""], ["dcd", "33210"], ["dccd", "332210"]],
                [["", ""], ["", ""], ["", ""]],
                [["abcdef", "101"], ["fef", "5543210"], ["", ""]],
            ],
        ],
        dims=["X", "Y", "XX", "YY"],
    ).astype(dtype)

    res_str = value.str.extractall(pat=pat_str, group_dim="XX", match_dim="YY")
    res_re = value.str.extractall(pat=pat_re, group_dim="XX", match_dim="YY")
    res_str_case = value.str.extractall(
        pat=pat_str, group_dim="XX", match_dim="YY", case=True
    )

    assert res_str.dtype == targ.dtype
    assert res_re.dtype == targ.dtype
    assert res_str_case.dtype == targ.dtype

    assert_equal(res_str, targ)
    assert_equal(res_re, targ)
    assert_equal(res_str_case, targ)


def test_extractall_multi_multi_nocase(dtype):
    pat_str = r"(\w+)_Xy_(\d*)"
    conv = {np.unicode_: str, np.bytes_: lambda x: bytes(x, encoding="UTF-8")}[dtype]
    pat_re = re.compile(conv(pat_str), flags=re.I)

    value = xr.DataArray(
        [
            ["a_Xy_0", "ab_xY_10-bab_Xy_110-baab_Xy_1100", "abc_Xy_01-cbc_Xy_2210"],
            [
                "abcd_Xy_-dcd_Xy_33210-dccd_Xy_332210",
                "",
                "abcdef_Xy_101-fef_Xy_5543210",
            ],
        ],
        dims=["X", "Y"],
    ).astype(dtype)

    targ = xr.DataArray(
        [
            [
                [["a", "0"], ["", ""], ["", ""]],
                [["ab", "10"], ["bab", "110"], ["baab", "1100"]],
                [["abc", "01"], ["cbc", "2210"], ["", ""]],
            ],
            [
                [["abcd", ""], ["dcd", "33210"], ["dccd", "332210"]],
                [["", ""], ["", ""], ["", ""]],
                [["abcdef", "101"], ["fef", "5543210"], ["", ""]],
            ],
        ],
        dims=["X", "Y", "XX", "YY"],
    ).astype(dtype)

    res_str = value.str.extractall(
        pat=pat_str, group_dim="XX", match_dim="YY", case=False
    )
    res_re = value.str.extractall(pat=pat_re, group_dim="XX", match_dim="YY")

    assert res_str.dtype == targ.dtype
    assert res_re.dtype == targ.dtype

    assert_equal(res_str, targ)
    assert_equal(res_re, targ)


def test_findall_single_single_case(dtype):
    pat_str = r"(\w+)_Xy_\d*"
    conv = {np.unicode_: str, np.bytes_: lambda x: bytes(x, encoding="UTF-8")}[dtype]
    pat_re = re.compile(conv(pat_str))

    value = xr.DataArray(
        [["a_Xy_0", "ab_xY_10", "abc_Xy_01"], ["abcd_Xy_", "", "abcdef_Xy_101"]],
        dims=["X", "Y"],
    ).astype(dtype)

    targ = [[["a"], [], ["abc"]], [["abcd"], [], ["abcdef"]]]
    targ = [[[conv(x) for x in y] for y in z] for z in targ]
    targ = np.array(targ, dtype=np.object_)
    targ = xr.DataArray(targ, dims=["X", "Y"])

    res_str = value.str.findall(pat=pat_str)
    res_re = value.str.findall(pat=pat_re)
    res_str_case = value.str.findall(pat=pat_str, case=True)

    assert res_str.dtype == targ.dtype
    assert res_re.dtype == targ.dtype
    assert res_str_case.dtype == targ.dtype

    assert_equal(res_str, targ)
    assert_equal(res_re, targ)
    assert_equal(res_str_case, targ)


def test_findall_single_single_nocase(dtype):
    pat_str = r"(\w+)_Xy_\d*"
    conv = {np.unicode_: str, np.bytes_: lambda x: bytes(x, encoding="UTF-8")}[dtype]
    pat_re = re.compile(conv(pat_str), flags=re.I)

    value = xr.DataArray(
        [["a_Xy_0", "ab_xY_10", "abc_Xy_01"], ["abcd_Xy_", "", "abcdef_Xy_101"]],
        dims=["X", "Y"],
    ).astype(dtype)

    targ = [[["a"], ["ab"], ["abc"]], [["abcd"], [], ["abcdef"]]]
    targ = [[[conv(x) for x in y] for y in z] for z in targ]
    targ = np.array(targ, dtype=np.object_)
    print(targ)
    targ = xr.DataArray(targ, dims=["X", "Y"])

    res_str = value.str.findall(pat=pat_str, case=False)
    res_re = value.str.findall(pat=pat_re)

    assert res_str.dtype == targ.dtype
    assert res_re.dtype == targ.dtype

    assert_equal(res_str, targ)
    assert_equal(res_re, targ)


def test_findall_single_multi_case(dtype):
    pat_str = r"(\w+)_Xy_\d*"
    conv = {np.unicode_: str, np.bytes_: lambda x: bytes(x, encoding="UTF-8")}[dtype]
    pat_re = re.compile(conv(pat_str))

    value = xr.DataArray(
        [
            ["a_Xy_0", "ab_xY_10-bab_Xy_110-baab_Xy_1100", "abc_Xy_01-cbc_Xy_2210"],
            [
                "abcd_Xy_-dcd_Xy_33210-dccd_Xy_332210",
                "",
                "abcdef_Xy_101-fef_Xy_5543210",
            ],
        ],
        dims=["X", "Y"],
    ).astype(dtype)

    targ = [
        [["a"], ["bab", "baab"], ["abc", "cbc"]],
        [
            ["abcd", "dcd", "dccd"],
            [],
            ["abcdef", "fef"],
        ],
    ]
    targ = [[[conv(x) for x in y] for y in z] for z in targ]
    targ = np.array(targ, dtype=np.object_)
    targ = xr.DataArray(targ, dims=["X", "Y"])

    res_str = value.str.findall(pat=pat_str)
    res_re = value.str.findall(pat=pat_re)
    res_str_case = value.str.findall(pat=pat_str, case=True)

    assert res_str.dtype == targ.dtype
    assert res_re.dtype == targ.dtype
    assert res_str_case.dtype == targ.dtype

    assert_equal(res_str, targ)
    assert_equal(res_re, targ)
    assert_equal(res_str_case, targ)


def test_findall_single_multi_nocase(dtype):
    pat_str = r"(\w+)_Xy_\d*"
    conv = {np.unicode_: str, np.bytes_: lambda x: bytes(x, encoding="UTF-8")}[dtype]
    pat_re = re.compile(conv(pat_str), flags=re.I)

    value = xr.DataArray(
        [
            ["a_Xy_0", "ab_xY_10-bab_Xy_110-baab_Xy_1100", "abc_Xy_01-cbc_Xy_2210"],
            [
                "abcd_Xy_-dcd_Xy_33210-dccd_Xy_332210",
                "",
                "abcdef_Xy_101-fef_Xy_5543210",
            ],
        ],
        dims=["X", "Y"],
    ).astype(dtype)

    targ = [
        [
            ["a"],
            ["ab", "bab", "baab"],
            ["abc", "cbc"],
        ],
        [
            ["abcd", "dcd", "dccd"],
            [],
            ["abcdef", "fef"],
        ],
    ]
    targ = [[[conv(x) for x in y] for y in z] for z in targ]
    targ = np.array(targ, dtype=np.object_)
    targ = xr.DataArray(targ, dims=["X", "Y"])

    res_str = value.str.findall(pat=pat_str, case=False)
    res_re = value.str.findall(pat=pat_re)

    assert res_str.dtype == targ.dtype
    assert res_re.dtype == targ.dtype

    assert_equal(res_str, targ)
    assert_equal(res_re, targ)


def test_findall_multi_single_case(dtype):
    pat_str = r"(\w+)_Xy_(\d*)"
    conv = {np.unicode_: str, np.bytes_: lambda x: bytes(x, encoding="UTF-8")}[dtype]
    pat_re = re.compile(conv(pat_str))

    value = xr.DataArray(
        [["a_Xy_0", "ab_xY_10", "abc_Xy_01"], ["abcd_Xy_", "", "abcdef_Xy_101"]],
        dims=["X", "Y"],
    ).astype(dtype)

    targ = [
        [[["a", "0"]], [], [["abc", "01"]]],
        [[["abcd", ""]], [], [["abcdef", "101"]]],
    ]
    targ = [[[tuple(conv(x) for x in y) for y in z] for z in w] for w in targ]
    targ = np.array(targ, dtype=np.object_)
    targ = xr.DataArray(targ, dims=["X", "Y"])

    res_str = value.str.findall(pat=pat_str)
    res_re = value.str.findall(pat=pat_re)
    res_str_case = value.str.findall(pat=pat_str, case=True)

    assert res_str.dtype == targ.dtype
    assert res_re.dtype == targ.dtype
    assert res_str_case.dtype == targ.dtype

    assert_equal(res_str, targ)
    assert_equal(res_re, targ)
    assert_equal(res_str_case, targ)


def test_findall_multi_single_nocase(dtype):
    pat_str = r"(\w+)_Xy_(\d*)"
    conv = {np.unicode_: str, np.bytes_: lambda x: bytes(x, encoding="UTF-8")}[dtype]
    pat_re = re.compile(conv(pat_str), flags=re.I)

    value = xr.DataArray(
        [["a_Xy_0", "ab_xY_10", "abc_Xy_01"], ["abcd_Xy_", "", "abcdef_Xy_101"]],
        dims=["X", "Y"],
    ).astype(dtype)

    targ = [
        [[["a", "0"]], [["ab", "10"]], [["abc", "01"]]],
        [[["abcd", ""]], [], [["abcdef", "101"]]],
    ]
    targ = [[[tuple(conv(x) for x in y) for y in z] for z in w] for w in targ]
    targ = np.array(targ, dtype=np.object_)
    targ = xr.DataArray(targ, dims=["X", "Y"])

    res_str = value.str.findall(pat=pat_str, case=False)
    res_re = value.str.findall(pat=pat_re)

    assert res_str.dtype == targ.dtype
    assert res_re.dtype == targ.dtype

    assert_equal(res_str, targ)
    assert_equal(res_re, targ)


def test_findall_multi_multi_case(dtype):
    pat_str = r"(\w+)_Xy_(\d*)"
    conv = {np.unicode_: str, np.bytes_: lambda x: bytes(x, encoding="UTF-8")}[dtype]
    pat_re = re.compile(conv(pat_str))

    value = xr.DataArray(
        [
            ["a_Xy_0", "ab_xY_10-bab_Xy_110-baab_Xy_1100", "abc_Xy_01-cbc_Xy_2210"],
            [
                "abcd_Xy_-dcd_Xy_33210-dccd_Xy_332210",
                "",
                "abcdef_Xy_101-fef_Xy_5543210",
            ],
        ],
        dims=["X", "Y"],
    ).astype(dtype)

    targ = [
        [
            [["a", "0"]],
            [["bab", "110"], ["baab", "1100"]],
            [["abc", "01"], ["cbc", "2210"]],
        ],
        [
            [["abcd", ""], ["dcd", "33210"], ["dccd", "332210"]],
            [],
            [["abcdef", "101"], ["fef", "5543210"]],
        ],
    ]
    targ = [[[tuple(conv(x) for x in y) for y in z] for z in w] for w in targ]
    targ = np.array(targ, dtype=np.object_)
    targ = xr.DataArray(targ, dims=["X", "Y"])

    res_str = value.str.findall(pat=pat_str)
    res_re = value.str.findall(pat=pat_re)
    res_str_case = value.str.findall(pat=pat_str, case=True)

    assert res_str.dtype == targ.dtype
    assert res_re.dtype == targ.dtype
    assert res_str_case.dtype == targ.dtype

    assert_equal(res_str, targ)
    assert_equal(res_re, targ)
    assert_equal(res_str_case, targ)


def test_findall_multi_multi_nocase(dtype):
    pat_str = r"(\w+)_Xy_(\d*)"
    conv = {np.unicode_: str, np.bytes_: lambda x: bytes(x, encoding="UTF-8")}[dtype]
    pat_re = re.compile(conv(pat_str), flags=re.I)

    value = xr.DataArray(
        [
            ["a_Xy_0", "ab_xY_10-bab_Xy_110-baab_Xy_1100", "abc_Xy_01-cbc_Xy_2210"],
            [
                "abcd_Xy_-dcd_Xy_33210-dccd_Xy_332210",
                "",
                "abcdef_Xy_101-fef_Xy_5543210",
            ],
        ],
        dims=["X", "Y"],
    ).astype(dtype)

    targ = [
        [
            [["a", "0"]],
            [["ab", "10"], ["bab", "110"], ["baab", "1100"]],
            [["abc", "01"], ["cbc", "2210"]],
        ],
        [
            [["abcd", ""], ["dcd", "33210"], ["dccd", "332210"]],
            [],
            [["abcdef", "101"], ["fef", "5543210"]],
        ],
    ]
    targ = [[[tuple(conv(x) for x in y) for y in z] for z in w] for w in targ]
    targ = np.array(targ, dtype=np.object_)
    targ = xr.DataArray(targ, dims=["X", "Y"])

    res_str = value.str.findall(pat=pat_str, case=False)
    res_re = value.str.findall(pat=pat_re)

    assert res_str.dtype == targ.dtype
    assert res_re.dtype == targ.dtype

    assert_equal(res_str, targ)
    assert_equal(res_re, targ)


def test_repeat(dtype):
    values = xr.DataArray(["a", "b", "c", "d"]).astype(dtype)
    result = values.str.repeat(3)
    expected = xr.DataArray(["aaa", "bbb", "ccc", "ddd"]).astype(dtype)
    assert result.dtype == expected.dtype
    assert_equal(result, expected)


def test_match(dtype):
    # New match behavior introduced in 0.13
    values = xr.DataArray(["fooBAD__barBAD", "foo"]).astype(dtype)
    result = values.str.match(".*(BAD[_]+).*(BAD)")
    expected = xr.DataArray([True, False])
    assert result.dtype == expected.dtype
    assert_equal(result, expected)

    values = xr.DataArray(["fooBAD__barBAD", "foo"]).astype(dtype)
    result = values.str.match(".*BAD[_]+.*BAD")
    expected = xr.DataArray([True, False])
    assert result.dtype == expected.dtype
    assert_equal(result, expected)


def test_empty_str_methods():
    empty = xr.DataArray(np.empty(shape=(0,), dtype="U"))
    empty_str = empty
    empty_int = xr.DataArray(np.empty(shape=(0,), dtype=int))
    empty_bool = xr.DataArray(np.empty(shape=(0,), dtype=bool))
    empty_bytes = xr.DataArray(np.empty(shape=(0,), dtype="S"))

    # TODO: Determine why U and S dtype sizes don't match and figure
    # out a reliable way to predict what they should be

    assert empty_bool.dtype == empty.str.contains("a").dtype
    assert empty_bool.dtype == empty.str.endswith("a").dtype
    assert empty_bool.dtype == empty.str.match("^a").dtype
    assert empty_bool.dtype == empty.str.startswith("a").dtype
    assert empty_bool.dtype == empty.str.isalnum().dtype
    assert empty_bool.dtype == empty.str.isalpha().dtype
    assert empty_bool.dtype == empty.str.isdecimal().dtype
    assert empty_bool.dtype == empty.str.isdigit().dtype
    assert empty_bool.dtype == empty.str.islower().dtype
    assert empty_bool.dtype == empty.str.isnumeric().dtype
    assert empty_bool.dtype == empty.str.isspace().dtype
    assert empty_bool.dtype == empty.str.istitle().dtype
    assert empty_bool.dtype == empty.str.isupper().dtype
    assert empty_bytes.dtype.kind == empty.str.encode("ascii").dtype.kind
    assert empty_int.dtype.kind == empty.str.count("a").dtype.kind
    assert empty_int.dtype.kind == empty.str.find("a").dtype.kind
    assert empty_int.dtype.kind == empty.str.len().dtype.kind
    assert empty_int.dtype.kind == empty.str.rfind("a").dtype.kind
    assert empty_str.dtype.kind == empty.str.capitalize().dtype.kind
    assert empty_str.dtype.kind == empty.str.center(42).dtype.kind
    assert empty_str.dtype.kind == empty.str.get(0).dtype.kind
    assert empty_str.dtype.kind == empty.str.lower().dtype.kind
    assert empty_str.dtype.kind == empty.str.lstrip().dtype.kind
    assert empty_str.dtype.kind == empty.str.pad(42).dtype.kind
    assert empty_str.dtype.kind == empty.str.repeat(3).dtype.kind
    assert empty_str.dtype.kind == empty.str.rstrip().dtype.kind
    assert empty_str.dtype.kind == empty.str.slice(step=1).dtype.kind
    assert empty_str.dtype.kind == empty.str.slice(stop=1).dtype.kind
    assert empty_str.dtype.kind == empty.str.strip().dtype.kind
    assert empty_str.dtype.kind == empty.str.swapcase().dtype.kind
    assert empty_str.dtype.kind == empty.str.title().dtype.kind
    assert empty_str.dtype.kind == empty.str.upper().dtype.kind
    assert empty_str.dtype.kind == empty.str.wrap(42).dtype.kind
    assert empty_str.dtype.kind == empty_bytes.str.decode("ascii").dtype.kind

    assert_equal(empty_bool, empty.str.contains("a"))
    assert_equal(empty_bool, empty.str.endswith("a"))
    assert_equal(empty_bool, empty.str.match("^a"))
    assert_equal(empty_bool, empty.str.startswith("a"))
    assert_equal(empty_bool, empty.str.isalnum())
    assert_equal(empty_bool, empty.str.isalpha())
    assert_equal(empty_bool, empty.str.isdecimal())
    assert_equal(empty_bool, empty.str.isdigit())
    assert_equal(empty_bool, empty.str.islower())
    assert_equal(empty_bool, empty.str.isnumeric())
    assert_equal(empty_bool, empty.str.isspace())
    assert_equal(empty_bool, empty.str.istitle())
    assert_equal(empty_bool, empty.str.isupper())
    assert_equal(empty_bytes, empty.str.encode("ascii"))
    assert_equal(empty_int, empty.str.count("a"))
    assert_equal(empty_int, empty.str.find("a"))
    assert_equal(empty_int, empty.str.len())
    assert_equal(empty_int, empty.str.rfind("a"))
    assert_equal(empty_str, empty.str.capitalize())
    assert_equal(empty_str, empty.str.center(42))
    assert_equal(empty_str, empty.str.get(0))
    assert_equal(empty_str, empty.str.lower())
    assert_equal(empty_str, empty.str.lstrip())
    assert_equal(empty_str, empty.str.pad(42))
    assert_equal(empty_str, empty.str.repeat(3))
    assert_equal(empty_str, empty.str.replace("a", "b"))
    assert_equal(empty_str, empty.str.rstrip())
    assert_equal(empty_str, empty.str.slice(step=1))
    assert_equal(empty_str, empty.str.slice(stop=1))
    assert_equal(empty_str, empty.str.strip())
    assert_equal(empty_str, empty.str.swapcase())
    assert_equal(empty_str, empty.str.title())
    assert_equal(empty_str, empty.str.upper())
    assert_equal(empty_str, empty.str.wrap(42))
    assert_equal(empty_str, empty_bytes.str.decode("ascii"))

    table = str.maketrans("a", "b")
    assert empty_str.dtype.kind == empty.str.translate(table).dtype.kind
    assert_equal(empty_str, empty.str.translate(table))


def test_ismethods(dtype):
    values = ["A", "b", "Xy", "4", "3A", "", "TT", "55", "-", "  "]

    exp_alnum = [True, True, True, True, True, False, True, True, False, False]
    exp_alpha = [True, True, True, False, False, False, True, False, False, False]
    exp_digit = [False, False, False, True, False, False, False, True, False, False]
    exp_space = [False, False, False, False, False, False, False, False, False, True]
    exp_lower = [False, True, False, False, False, False, False, False, False, False]
    exp_upper = [True, False, False, False, True, False, True, False, False, False]
    exp_title = [True, False, True, False, True, False, False, False, False, False]

    values = xr.DataArray(values).astype(dtype)

    exp_alnum = xr.DataArray(exp_alnum)
    exp_alpha = xr.DataArray(exp_alpha)
    exp_digit = xr.DataArray(exp_digit)
    exp_space = xr.DataArray(exp_space)
    exp_lower = xr.DataArray(exp_lower)
    exp_upper = xr.DataArray(exp_upper)
    exp_title = xr.DataArray(exp_title)

    res_alnum = values.str.isalnum()
    res_alpha = values.str.isalpha()
    res_digit = values.str.isdigit()
    res_lower = values.str.islower()
    res_space = values.str.isspace()
    res_title = values.str.istitle()
    res_upper = values.str.isupper()

    assert res_alnum.dtype == exp_alnum.dtype
    assert res_alpha.dtype == exp_alpha.dtype
    assert res_digit.dtype == exp_digit.dtype
    assert res_lower.dtype == exp_lower.dtype
    assert res_space.dtype == exp_space.dtype
    assert res_title.dtype == exp_title.dtype
    assert res_upper.dtype == exp_upper.dtype

    assert_equal(res_alnum, exp_alnum)
    assert_equal(res_alpha, exp_alpha)
    assert_equal(res_digit, exp_digit)
    assert_equal(res_lower, exp_lower)
    assert_equal(res_space, exp_space)
    assert_equal(res_title, exp_title)
    assert_equal(res_upper, exp_upper)


def test_isnumeric():
    # 0x00bc: ¼ VULGAR FRACTION ONE QUARTER
    # 0x2605: ★ not number
    # 0x1378: ፸ ETHIOPIC NUMBER SEVENTY
    # 0xFF13: ３ Em 3
    values = ["A", "3", "¼", "★", "፸", "３", "four"]
    exp_numeric = [False, True, True, False, True, True, False]
    exp_decimal = [False, True, False, False, False, True, False]

    values = xr.DataArray(values)
    exp_numeric = xr.DataArray(exp_numeric)
    exp_decimal = xr.DataArray(exp_decimal)

    res_numeric = values.str.isnumeric()
    res_decimal = values.str.isdecimal()

    assert res_numeric.dtype == exp_numeric.dtype
    assert res_decimal.dtype == exp_decimal.dtype

    assert_equal(res_numeric, exp_numeric)
    assert_equal(res_decimal, exp_decimal)


def test_len(dtype):
    values = ["foo", "fooo", "fooooo", "fooooooo"]
    result = xr.DataArray(values).astype(dtype).str.len()
    expected = xr.DataArray([len(x) for x in values])
    assert result.dtype == expected.dtype
    assert_equal(result, expected)


def test_find(dtype):
    values = xr.DataArray(["ABCDEFG", "BCDEFEF", "DEFGHIJEF", "EFGHEF", "XXX"])
    values = values.astype(dtype)
    result = values.str.find("EF")
    expected = xr.DataArray([4, 3, 1, 0, -1])
    assert result.dtype == expected.dtype
    assert_equal(result, expected)
    expected = xr.DataArray([v.find(dtype("EF")) for v in values.values])
    assert result.dtype == expected.dtype
    assert_equal(result, expected)

    result = values.str.rfind("EF")
    expected = xr.DataArray([4, 5, 7, 4, -1])
    assert result.dtype == expected.dtype
    assert_equal(result, expected)
    expected = xr.DataArray([v.rfind(dtype("EF")) for v in values.values])
    assert result.dtype == expected.dtype
    assert_equal(result, expected)

    result = values.str.find("EF", 3)
    expected = xr.DataArray([4, 3, 7, 4, -1])
    assert result.dtype == expected.dtype
    assert_equal(result, expected)
    expected = xr.DataArray([v.find(dtype("EF"), 3) for v in values.values])
    assert result.dtype == expected.dtype
    assert_equal(result, expected)

    result = values.str.rfind("EF", 3)
    expected = xr.DataArray([4, 5, 7, 4, -1])
    assert result.dtype == expected.dtype
    assert_equal(result, expected)
    expected = xr.DataArray([v.rfind(dtype("EF"), 3) for v in values.values])
    assert result.dtype == expected.dtype
    assert_equal(result, expected)

    result = values.str.find("EF", 3, 6)
    expected = xr.DataArray([4, 3, -1, 4, -1])
    assert result.dtype == expected.dtype
    assert_equal(result, expected)
    expected = xr.DataArray([v.find(dtype("EF"), 3, 6) for v in values.values])
    assert result.dtype == expected.dtype
    assert_equal(result, expected)

    result = values.str.rfind("EF", 3, 6)
    expected = xr.DataArray([4, 3, -1, 4, -1])
    assert result.dtype == expected.dtype
    assert_equal(result, expected)
    xp = xr.DataArray([v.rfind(dtype("EF"), 3, 6) for v in values.values])
    assert result.dtype == xp.dtype
    assert_equal(result, xp)


def test_index(dtype):
    s = xr.DataArray(["ABCDEFG", "BCDEFEF", "DEFGHIJEF", "EFGHEF"]).astype(dtype)

    result = s.str.index("EF")
    expected = xr.DataArray([4, 3, 1, 0])
    assert result.dtype == expected.dtype
    assert_equal(result, expected)

    result = s.str.rindex("EF")
    expected = xr.DataArray([4, 5, 7, 4])
    assert result.dtype == expected.dtype
    assert_equal(result, expected)

    result = s.str.index("EF", 3)
    expected = xr.DataArray([4, 3, 7, 4])
    assert result.dtype == expected.dtype
    assert_equal(result, expected)

    result = s.str.rindex("EF", 3)
    expected = xr.DataArray([4, 5, 7, 4])
    assert result.dtype == expected.dtype
    assert_equal(result, expected)

    result = s.str.index("E", 4, 8)
    expected = xr.DataArray([4, 5, 7, 4])
    assert result.dtype == expected.dtype
    assert_equal(result, expected)

    result = s.str.rindex("E", 0, 5)
    expected = xr.DataArray([4, 3, 1, 4])
    assert result.dtype == expected.dtype
    assert_equal(result, expected)

    with pytest.raises(ValueError):
        result = s.str.index("DE")


def test_pad(dtype):
    values = xr.DataArray(["a", "b", "c", "eeeee"]).astype(dtype)

    result = values.str.pad(5, side="left")
    expected = xr.DataArray(["    a", "    b", "    c", "eeeee"]).astype(dtype)
    assert result.dtype == expected.dtype
    assert_equal(result, expected)

    result = values.str.pad(5, side="right")
    expected = xr.DataArray(["a    ", "b    ", "c    ", "eeeee"]).astype(dtype)
    assert result.dtype == expected.dtype
    assert_equal(result, expected)

    result = values.str.pad(5, side="both")
    expected = xr.DataArray(["  a  ", "  b  ", "  c  ", "eeeee"]).astype(dtype)
    assert result.dtype == expected.dtype
    assert_equal(result, expected)


def test_pad_fillchar(dtype):
    values = xr.DataArray(["a", "b", "c", "eeeee"]).astype(dtype)

    result = values.str.pad(5, side="left", fillchar="X")
    expected = xr.DataArray(["XXXXa", "XXXXb", "XXXXc", "eeeee"]).astype(dtype)
    assert result.dtype == expected.dtype
    assert_equal(result, expected)

    result = values.str.pad(5, side="right", fillchar="X")
    expected = xr.DataArray(["aXXXX", "bXXXX", "cXXXX", "eeeee"]).astype(dtype)
    assert result.dtype == expected.dtype
    assert_equal(result, expected)

    result = values.str.pad(5, side="both", fillchar="X")
    expected = xr.DataArray(["XXaXX", "XXbXX", "XXcXX", "eeeee"]).astype(dtype)
    assert result.dtype == expected.dtype
    assert_equal(result, expected)

    msg = "fillchar must be a character, not str"
    with pytest.raises(TypeError, match=msg):
        result = values.str.pad(5, fillchar="XY")


def test_translate():
    values = xr.DataArray(["abcdefg", "abcc", "cdddfg", "cdefggg"])
    table = str.maketrans("abc", "cde")
    result = values.str.translate(table)
    expected = xr.DataArray(["cdedefg", "cdee", "edddfg", "edefggg"])
    assert result.dtype == expected.dtype
    assert_equal(result, expected)


def test_center_ljust_rjust(dtype):
    values = xr.DataArray(["a", "b", "c", "eeeee"]).astype(dtype)

    result = values.str.center(5)
    expected = xr.DataArray(["  a  ", "  b  ", "  c  ", "eeeee"]).astype(dtype)
    assert result.dtype == expected.dtype
    assert_equal(result, expected)

    result = values.str.ljust(5)
    expected = xr.DataArray(["a    ", "b    ", "c    ", "eeeee"]).astype(dtype)
    assert result.dtype == expected.dtype
    assert_equal(result, expected)

    result = values.str.rjust(5)
    expected = xr.DataArray(["    a", "    b", "    c", "eeeee"]).astype(dtype)
    assert result.dtype == expected.dtype
    assert_equal(result, expected)


def test_center_ljust_rjust_fillchar(dtype):
    values = xr.DataArray(["a", "bb", "cccc", "ddddd", "eeeeee"]).astype(dtype)
    result = values.str.center(5, fillchar="X")
    expected = xr.DataArray(["XXaXX", "XXbbX", "Xcccc", "ddddd", "eeeeee"])
    assert result.dtype == expected.astype(dtype).dtype
    assert_equal(result, expected.astype(dtype))

    result = values.str.ljust(5, fillchar="X")
    expected = xr.DataArray(["aXXXX", "bbXXX", "ccccX", "ddddd", "eeeeee"])
    assert result.dtype == expected.astype(dtype).dtype
    assert_equal(result, expected.astype(dtype))

    result = values.str.rjust(5, fillchar="X")
    expected = xr.DataArray(["XXXXa", "XXXbb", "Xcccc", "ddddd", "eeeeee"])
    assert result.dtype == expected.astype(dtype).dtype
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
    assert result.dtype == expected.astype(dtype).dtype
    assert_equal(result, expected.astype(dtype))

    result = values.str.zfill(3)
    expected = xr.DataArray(["001", "022", "aaa", "333", "45678"])
    assert result.dtype == expected.astype(dtype).dtype
    assert_equal(result, expected.astype(dtype))


def test_slice(dtype):
    arr = xr.DataArray(["aafootwo", "aabartwo", "aabazqux"]).astype(dtype)

    result = arr.str.slice(2, 5)
    exp = xr.DataArray(["foo", "bar", "baz"]).astype(dtype)
    assert result.dtype == exp.dtype
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
    assert result.dtype == expected.dtype
    assert_equal(result, expected)

    expected = da(["shzrt", "a zit longer", "evznlongerthanthat", "z"])
    result = values.str.slice_replace(2, 3, "z")
    assert result.dtype == expected.dtype
    assert_equal(result, expected)

    expected = da(["shzort", "a zbit longer", "evzenlongerthanthat", "z"])
    result = values.str.slice_replace(2, 2, "z")
    assert result.dtype == expected.dtype
    assert_equal(result, expected)

    expected = da(["shzort", "a zbit longer", "evzenlongerthanthat", "z"])
    result = values.str.slice_replace(2, 1, "z")
    assert result.dtype == expected.dtype
    assert_equal(result, expected)

    expected = da(["shorz", "a bit longez", "evenlongerthanthaz", "z"])
    result = values.str.slice_replace(-1, None, "z")
    assert result.dtype == expected.dtype
    assert_equal(result, expected)

    expected = da(["zrt", "zer", "zat", "z"])
    result = values.str.slice_replace(None, -2, "z")
    assert result.dtype == expected.dtype
    assert_equal(result, expected)

    expected = da(["shortz", "a bit znger", "evenlozerthanthat", "z"])
    result = values.str.slice_replace(6, 8, "z")
    assert result.dtype == expected.dtype
    assert_equal(result, expected)

    expected = da(["zrt", "a zit longer", "evenlongzerthanthat", "z"])
    result = values.str.slice_replace(-10, 3, "z")
    assert result.dtype == expected.dtype
    assert_equal(result, expected)


def test_strip_lstrip_rstrip(dtype):
    values = xr.DataArray(["  aa   ", " bb \n", "cc  "]).astype(dtype)

    result = values.str.strip()
    expected = xr.DataArray(["aa", "bb", "cc"]).astype(dtype)
    assert result.dtype == expected.dtype
    assert_equal(result, expected)

    result = values.str.lstrip()
    expected = xr.DataArray(["aa   ", "bb \n", "cc  "]).astype(dtype)
    assert result.dtype == expected.dtype
    assert_equal(result, expected)

    result = values.str.rstrip()
    expected = xr.DataArray(["  aa", " bb", "cc"]).astype(dtype)
    assert result.dtype == expected.dtype
    assert_equal(result, expected)


def test_strip_lstrip_rstrip_args(dtype):
    values = xr.DataArray(["xxABCxx", "xx BNSD", "LDFJH xx"]).astype(dtype)

    rs = values.str.strip("x")
    xp = xr.DataArray(["ABC", " BNSD", "LDFJH "]).astype(dtype)
    assert rs.dtype == xp.dtype
    assert_equal(rs, xp)

    rs = values.str.lstrip("x")
    xp = xr.DataArray(["ABCxx", " BNSD", "LDFJH xx"]).astype(dtype)
    assert rs.dtype == xp.dtype
    assert_equal(rs, xp)

    rs = values.str.rstrip("x")
    xp = xr.DataArray(["xxABC", "xx BNSD", "LDFJH "]).astype(dtype)
    assert rs.dtype == xp.dtype
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
    expected = xr.DataArray(
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

    result = values.str.wrap(12, break_long_words=True)
    assert result.dtype == expected.dtype
    assert_equal(result, expected)

    # test with pre and post whitespace (non-unicode), NaN, and non-ascii
    # Unicode
    values = xr.DataArray(["  pre  ", "\xac\u20ac\U00008000 abadcafe"])
    expected = xr.DataArray(["  pre", "\xac\u20ac\U00008000 ab\nadcafe"])
    result = values.str.wrap(6)
    assert result.dtype == expected.dtype
    assert_equal(result, expected)


def test_wrap_kwargs_passed():
    # GH4334

    values = xr.DataArray("  hello world  ")

    result = values.str.wrap(7)
    expected = xr.DataArray("  hello\nworld")
    assert result.dtype == expected.dtype
    assert_equal(result, expected)

    result = values.str.wrap(7, drop_whitespace=False)
    expected = xr.DataArray("  hello\n world\n  ")
    assert result.dtype == expected.dtype
    assert_equal(result, expected)


def test_get(dtype):
    values = xr.DataArray(["a_b_c", "c_d_e", "f_g_h"]).astype(dtype)

    result = values.str[2]
    expected = xr.DataArray(["b", "d", "g"]).astype(dtype)
    assert result.dtype == expected.dtype
    assert_equal(result, expected)

    # bounds testing
    values = xr.DataArray(["1_2_3_4_5", "6_7_8_9_10", "11_12"]).astype(dtype)

    # positive index
    result = values.str[5]
    expected = xr.DataArray(["_", "_", ""]).astype(dtype)
    assert result.dtype == expected.dtype
    assert_equal(result, expected)

    # negative index
    result = values.str[-6]
    expected = xr.DataArray(["_", "8", ""]).astype(dtype)
    assert result.dtype == expected.dtype
    assert_equal(result, expected)


def test_get_default(dtype):
    # GH4334
    values = xr.DataArray(["a_b", "c", ""]).astype(dtype)

    result = values.str.get(2, "default")
    expected = xr.DataArray(["b", "default", "default"]).astype(dtype)
    assert result.dtype == expected.dtype
    assert_equal(result, expected)


def test_encode_decode():
    data = xr.DataArray(["a", "b", "a\xe4"])
    encoded = data.str.encode("utf-8")
    decoded = encoded.str.decode("utf-8")
    assert data.dtype == decoded.dtype
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

    assert result.dtype == expected.dtype
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

    assert result.dtype == expected.dtype
    assert_equal(result, expected)


def test_partition_whitespace(dtype):
    values = xr.DataArray(
        [
            ["abc def", "spam eggs swallow", "red_blue"],
            ["test0 test1 test2 test3", "", "abra ka da bra"],
        ],
        dims=["X", "Y"],
    ).astype(dtype)

    exp_part_dim = [
        [
            ["abc", " ", "def"],
            ["spam", " ", "eggs swallow"],
            ["red_blue", "", ""],
        ],
        [
            ["test0", " ", "test1 test2 test3"],
            ["", "", ""],
            ["abra", " ", "ka da bra"],
        ],
    ]

    exp_rpart_dim = [
        [
            ["abc", " ", "def"],
            ["spam eggs", " ", "swallow"],
            ["", "", "red_blue"],
        ],
        [
            ["test0 test1 test2", " ", "test3"],
            ["", "", ""],
            ["abra ka da", " ", "bra"],
        ],
    ]

    exp_part_dim = xr.DataArray(exp_part_dim, dims=["X", "Y", "ZZ"]).astype(dtype)
    exp_rpart_dim = xr.DataArray(exp_rpart_dim, dims=["X", "Y", "ZZ"]).astype(dtype)

    res_part_dim = values.str.partition(dim="ZZ")
    res_rpart_dim = values.str.rpartition(dim="ZZ")

    assert res_part_dim.dtype == exp_part_dim.dtype
    assert res_rpart_dim.dtype == exp_rpart_dim.dtype

    assert_equal(res_part_dim, exp_part_dim)
    assert_equal(res_rpart_dim, exp_rpart_dim)


def test_partition_comma(dtype):
    values = xr.DataArray(
        [
            ["abc, def", "spam, eggs, swallow", "red_blue"],
            ["test0, test1, test2, test3", "", "abra, ka, da, bra"],
        ],
        dims=["X", "Y"],
    ).astype(dtype)

    exp_part_dim = [
        [
            ["abc", ", ", "def"],
            ["spam", ", ", "eggs, swallow"],
            ["red_blue", "", ""],
        ],
        [
            ["test0", ", ", "test1, test2, test3"],
            ["", "", ""],
            ["abra", ", ", "ka, da, bra"],
        ],
    ]

    exp_rpart_dim = [
        [
            ["abc", ", ", "def"],
            ["spam, eggs", ", ", "swallow"],
            ["", "", "red_blue"],
        ],
        [
            ["test0, test1, test2", ", ", "test3"],
            ["", "", ""],
            ["abra, ka, da", ", ", "bra"],
        ],
    ]

    exp_part_dim = xr.DataArray(exp_part_dim, dims=["X", "Y", "ZZ"]).astype(dtype)
    exp_rpart_dim = xr.DataArray(exp_rpart_dim, dims=["X", "Y", "ZZ"]).astype(dtype)

    res_part_dim = values.str.partition(sep=", ", dim="ZZ")
    res_rpart_dim = values.str.rpartition(sep=", ", dim="ZZ")

    assert res_part_dim.dtype == exp_part_dim.dtype
    assert res_rpart_dim.dtype == exp_rpart_dim.dtype

    assert_equal(res_part_dim, exp_part_dim)
    assert_equal(res_rpart_dim, exp_rpart_dim)


def test_split_whitespace(dtype):
    values = xr.DataArray(
        [
            ["abc def", "spam\t\teggs\tswallow", "red_blue"],
            ["test0\ntest1\ntest2\n\ntest3", "", "abra  ka\nda\tbra"],
        ],
        dims=["X", "Y"],
    ).astype(dtype)

    exp_split_dim_full = [
        [
            ["abc", "def", "", ""],
            ["spam", "eggs", "swallow", ""],
            ["red_blue", "", "", ""],
        ],
        [
            ["test0", "test1", "test2", "test3"],
            ["", "", "", ""],
            ["abra", "ka", "da", "bra"],
        ],
    ]

    exp_rsplit_dim_full = [
        [
            ["", "", "abc", "def"],
            ["", "spam", "eggs", "swallow"],
            ["", "", "", "red_blue"],
        ],
        [
            ["test0", "test1", "test2", "test3"],
            ["", "", "", ""],
            ["abra", "ka", "da", "bra"],
        ],
    ]

    exp_split_dim_1 = [
        [["abc", "def"], ["spam", "eggs\tswallow"], ["red_blue", ""]],
        [["test0", "test1\ntest2\n\ntest3"], ["", ""], ["abra", "ka\nda\tbra"]],
    ]

    exp_rsplit_dim_1 = [
        [["abc", "def"], ["spam\t\teggs", "swallow"], ["", "red_blue"]],
        [["test0\ntest1\ntest2", "test3"], ["", ""], ["abra  ka\nda", "bra"]],
    ]

    exp_split_none_full = [
        [["abc", "def"], ["spam", "eggs", "swallow"], ["red_blue"]],
        [["test0", "test1", "test2", "test3"], [], ["abra", "ka", "da", "bra"]],
    ]

    exp_rsplit_none_full = [
        [["abc", "def"], ["spam", "eggs", "swallow"], ["red_blue"]],
        [["test0", "test1", "test2", "test3"], [], ["abra", "ka", "da", "bra"]],
    ]

    exp_split_none_1 = [
        [["abc", "def"], ["spam", "eggs\tswallow"], ["red_blue"]],
        [["test0", "test1\ntest2\n\ntest3"], [], ["abra", "ka\nda\tbra"]],
    ]

    exp_rsplit_none_1 = [
        [["abc", "def"], ["spam\t\teggs", "swallow"], ["red_blue"]],
        [["test0\ntest1\ntest2", "test3"], [], ["abra  ka\nda", "bra"]],
    ]

    conv = {np.unicode_: str, np.bytes_: lambda x: bytes(x, encoding="UTF-8")}[dtype]

    exp_split_none_full = [
        [[conv(x) for x in y] for y in z] for z in exp_split_none_full
    ]
    exp_rsplit_none_full = [
        [[conv(x) for x in y] for y in z] for z in exp_rsplit_none_full
    ]
    exp_split_none_1 = [[[conv(x) for x in y] for y in z] for z in exp_split_none_1]
    exp_rsplit_none_1 = [[[conv(x) for x in y] for y in z] for z in exp_rsplit_none_1]

    exp_split_none_full = np.array(exp_split_none_full, dtype=np.object_)
    exp_rsplit_none_full = np.array(exp_rsplit_none_full, dtype=np.object_)
    exp_split_none_1 = np.array(exp_split_none_1, dtype=np.object_)
    exp_rsplit_none_1 = np.array(exp_rsplit_none_1, dtype=np.object_)

    exp_split_dim_full = xr.DataArray(exp_split_dim_full, dims=["X", "Y", "ZZ"]).astype(
        dtype
    )
    exp_rsplit_dim_full = xr.DataArray(
        exp_rsplit_dim_full, dims=["X", "Y", "ZZ"]
    ).astype(dtype)
    exp_split_dim_1 = xr.DataArray(exp_split_dim_1, dims=["X", "Y", "ZZ"]).astype(dtype)
    exp_rsplit_dim_1 = xr.DataArray(exp_rsplit_dim_1, dims=["X", "Y", "ZZ"]).astype(
        dtype
    )

    exp_split_none_full = xr.DataArray(exp_split_none_full, dims=["X", "Y"])
    exp_rsplit_none_full = xr.DataArray(exp_rsplit_none_full, dims=["X", "Y"])
    exp_split_none_1 = xr.DataArray(exp_split_none_1, dims=["X", "Y"])
    exp_rsplit_none_1 = xr.DataArray(exp_rsplit_none_1, dims=["X", "Y"])

    res_split_dim_full = values.str.split(dim="ZZ")
    res_rsplit_dim_full = values.str.rsplit(dim="ZZ")
    res_split_dim_1 = values.str.split(dim="ZZ", maxsplit=1)
    res_rsplit_dim_1 = values.str.rsplit(dim="ZZ", maxsplit=1)
    res_split_dim_10 = values.str.split(dim="ZZ", maxsplit=10)
    res_rsplit_dim_10 = values.str.rsplit(dim="ZZ", maxsplit=10)

    res_split_none_full = values.str.split(dim=None)
    res_rsplit_none_full = values.str.rsplit(dim=None)
    res_split_none_1 = values.str.split(dim=None, maxsplit=1)
    res_rsplit_none_1 = values.str.rsplit(dim=None, maxsplit=1)
    res_split_none_10 = values.str.split(dim=None, maxsplit=10)
    res_rsplit_none_10 = values.str.rsplit(dim=None, maxsplit=10)

    assert res_split_dim_full.dtype == exp_split_dim_full.dtype
    assert res_rsplit_dim_full.dtype == exp_rsplit_dim_full.dtype
    assert res_split_dim_1.dtype == exp_split_dim_1.dtype
    assert res_rsplit_dim_1.dtype == exp_rsplit_dim_1.dtype
    assert res_split_dim_10.dtype == exp_split_dim_full.dtype
    assert res_rsplit_dim_10.dtype == exp_rsplit_dim_full.dtype

    assert res_split_none_full.dtype == exp_split_none_full.dtype
    assert res_rsplit_none_full.dtype == exp_rsplit_none_full.dtype
    assert res_split_none_1.dtype == exp_split_none_1.dtype
    assert res_rsplit_none_1.dtype == exp_rsplit_none_1.dtype
    assert res_split_none_10.dtype == exp_split_none_full.dtype
    assert res_rsplit_none_10.dtype == exp_rsplit_none_full.dtype

    assert_equal(res_split_dim_full, exp_split_dim_full)
    assert_equal(res_rsplit_dim_full, exp_rsplit_dim_full)
    assert_equal(res_split_dim_1, exp_split_dim_1)
    assert_equal(res_rsplit_dim_1, exp_rsplit_dim_1)
    assert_equal(res_split_dim_10, exp_split_dim_full)
    assert_equal(res_rsplit_dim_10, exp_rsplit_dim_full)

    assert_equal(res_split_none_full, exp_split_none_full)
    assert_equal(res_rsplit_none_full, exp_rsplit_none_full)
    assert_equal(res_split_none_1, exp_split_none_1)
    assert_equal(res_rsplit_none_1, exp_rsplit_none_1)
    assert_equal(res_split_none_10, exp_split_none_full)
    assert_equal(res_rsplit_none_10, exp_rsplit_none_full)


def test_split_comma(dtype):
    values = xr.DataArray(
        [
            ["abc,def", "spam,,eggs,swallow", "red_blue"],
            ["test0,test1,test2,test3", "", "abra,ka,da,bra"],
        ],
        dims=["X", "Y"],
    ).astype(dtype)

    exp_split_dim_full = [
        [
            ["abc", "def", "", ""],
            ["spam", "", "eggs", "swallow"],
            ["red_blue", "", "", ""],
        ],
        [
            ["test0", "test1", "test2", "test3"],
            ["", "", "", ""],
            ["abra", "ka", "da", "bra"],
        ],
    ]

    exp_rsplit_dim_full = [
        [
            ["", "", "abc", "def"],
            ["spam", "", "eggs", "swallow"],
            ["", "", "", "red_blue"],
        ],
        [
            ["test0", "test1", "test2", "test3"],
            ["", "", "", ""],
            ["abra", "ka", "da", "bra"],
        ],
    ]

    exp_split_dim_1 = [
        [["abc", "def"], ["spam", ",eggs,swallow"], ["red_blue", ""]],
        [["test0", "test1,test2,test3"], ["", ""], ["abra", "ka,da,bra"]],
    ]

    exp_rsplit_dim_1 = [
        [["abc", "def"], ["spam,,eggs", "swallow"], ["", "red_blue"]],
        [["test0,test1,test2", "test3"], ["", ""], ["abra,ka,da", "bra"]],
    ]

    exp_split_none_full = [
        [["abc", "def"], ["spam", "", "eggs", "swallow"], ["red_blue"]],
        [["test0", "test1", "test2", "test3"], [""], ["abra", "ka", "da", "bra"]],
    ]

    exp_rsplit_none_full = [
        [["abc", "def"], ["spam", "", "eggs", "swallow"], ["red_blue"]],
        [["test0", "test1", "test2", "test3"], [""], ["abra", "ka", "da", "bra"]],
    ]

    exp_split_none_1 = [
        [["abc", "def"], ["spam", ",eggs,swallow"], ["red_blue"]],
        [["test0", "test1,test2,test3"], [""], ["abra", "ka,da,bra"]],
    ]

    exp_rsplit_none_1 = [
        [["abc", "def"], ["spam,,eggs", "swallow"], ["red_blue"]],
        [["test0,test1,test2", "test3"], [""], ["abra,ka,da", "bra"]],
    ]

    conv = {np.unicode_: str, np.bytes_: lambda x: bytes(x, encoding="UTF-8")}[dtype]

    exp_split_none_full = [
        [[conv(x) for x in y] for y in z] for z in exp_split_none_full
    ]
    exp_rsplit_none_full = [
        [[conv(x) for x in y] for y in z] for z in exp_rsplit_none_full
    ]
    exp_split_none_1 = [[[conv(x) for x in y] for y in z] for z in exp_split_none_1]
    exp_rsplit_none_1 = [[[conv(x) for x in y] for y in z] for z in exp_rsplit_none_1]

    exp_split_none_full = np.array(exp_split_none_full, dtype=np.object_)
    exp_rsplit_none_full = np.array(exp_rsplit_none_full, dtype=np.object_)
    exp_split_none_1 = np.array(exp_split_none_1, dtype=np.object_)
    exp_rsplit_none_1 = np.array(exp_rsplit_none_1, dtype=np.object_)

    exp_split_dim_full = xr.DataArray(exp_split_dim_full, dims=["X", "Y", "ZZ"]).astype(
        dtype
    )
    exp_rsplit_dim_full = xr.DataArray(
        exp_rsplit_dim_full, dims=["X", "Y", "ZZ"]
    ).astype(dtype)
    exp_split_dim_1 = xr.DataArray(exp_split_dim_1, dims=["X", "Y", "ZZ"]).astype(dtype)
    exp_rsplit_dim_1 = xr.DataArray(exp_rsplit_dim_1, dims=["X", "Y", "ZZ"]).astype(
        dtype
    )

    exp_split_none_full = xr.DataArray(exp_split_none_full, dims=["X", "Y"])
    exp_rsplit_none_full = xr.DataArray(exp_rsplit_none_full, dims=["X", "Y"])
    exp_split_none_1 = xr.DataArray(exp_split_none_1, dims=["X", "Y"])
    exp_rsplit_none_1 = xr.DataArray(exp_rsplit_none_1, dims=["X", "Y"])

    res_split_dim_full = values.str.split(sep=",", dim="ZZ")
    res_rsplit_dim_full = values.str.rsplit(sep=",", dim="ZZ")
    res_split_dim_1 = values.str.split(sep=",", dim="ZZ", maxsplit=1)
    res_rsplit_dim_1 = values.str.rsplit(sep=",", dim="ZZ", maxsplit=1)
    res_split_dim_10 = values.str.split(sep=",", dim="ZZ", maxsplit=10)
    res_rsplit_dim_10 = values.str.rsplit(sep=",", dim="ZZ", maxsplit=10)

    res_split_none_full = values.str.split(sep=",", dim=None)
    res_rsplit_none_full = values.str.rsplit(sep=",", dim=None)
    res_split_none_1 = values.str.split(sep=",", dim=None, maxsplit=1)
    res_rsplit_none_1 = values.str.rsplit(sep=",", dim=None, maxsplit=1)
    res_split_none_10 = values.str.split(sep=",", dim=None, maxsplit=10)
    res_rsplit_none_10 = values.str.rsplit(sep=",", dim=None, maxsplit=10)

    assert res_split_dim_full.dtype == exp_split_dim_full.dtype
    assert res_rsplit_dim_full.dtype == exp_rsplit_dim_full.dtype
    assert res_split_dim_1.dtype == exp_split_dim_1.dtype
    assert res_rsplit_dim_1.dtype == exp_rsplit_dim_1.dtype
    assert res_split_dim_10.dtype == exp_split_dim_full.dtype
    assert res_rsplit_dim_10.dtype == exp_rsplit_dim_full.dtype

    assert res_split_none_full.dtype == exp_split_none_full.dtype
    assert res_rsplit_none_full.dtype == exp_rsplit_none_full.dtype
    assert res_split_none_1.dtype == exp_split_none_1.dtype
    assert res_rsplit_none_1.dtype == exp_rsplit_none_1.dtype
    assert res_split_none_10.dtype == exp_split_none_full.dtype
    assert res_rsplit_none_10.dtype == exp_rsplit_none_full.dtype

    assert_equal(res_split_dim_full, exp_split_dim_full)
    assert_equal(res_rsplit_dim_full, exp_rsplit_dim_full)
    assert_equal(res_split_dim_1, exp_split_dim_1)
    assert_equal(res_rsplit_dim_1, exp_rsplit_dim_1)
    assert_equal(res_split_dim_10, exp_split_dim_full)
    assert_equal(res_rsplit_dim_10, exp_rsplit_dim_full)

    assert_equal(res_split_none_full, exp_split_none_full)
    assert_equal(res_rsplit_none_full, exp_rsplit_none_full)
    assert_equal(res_split_none_1, exp_split_none_1)
    assert_equal(res_rsplit_none_1, exp_rsplit_none_1)
    assert_equal(res_split_none_10, exp_split_none_full)
    assert_equal(res_rsplit_none_10, exp_rsplit_none_full)


def test_get_dummies(dtype):
    values_line = xr.DataArray(
        [["a|ab~abc|abc", "ab", "a||abc|abcd"], ["abcd|ab|a", "abc|ab~abc", "|a"]],
        dims=["X", "Y"],
    ).astype(dtype)
    values_comma = xr.DataArray(
        [["a~ab|abc~~abc", "ab", "a~abc~abcd"], ["abcd~ab~a", "abc~ab|abc", "~a"]],
        dims=["X", "Y"],
    ).astype(dtype)

    vals_line = np.array(["a", "ab", "abc", "abcd", "ab~abc"]).astype(dtype)
    vals_comma = np.array(["a", "ab", "abc", "abcd", "ab|abc"]).astype(dtype)
    targ = [
        [
            [True, False, True, False, True],
            [False, True, False, False, False],
            [True, False, True, True, False],
        ],
        [
            [True, True, False, True, False],
            [False, False, True, False, True],
            [True, False, False, False, False],
        ],
    ]
    targ = np.array(targ)
    targ = xr.DataArray(targ, dims=["X", "Y", "ZZ"])
    targ_line = targ.copy()
    targ_comma = targ.copy()
    targ_line.coords["ZZ"] = vals_line
    targ_comma.coords["ZZ"] = vals_comma

    res_default = values_line.str.get_dummies(dim="ZZ")
    res_line = values_line.str.get_dummies(dim="ZZ", sep="|")
    res_comma = values_comma.str.get_dummies(dim="ZZ", sep="~")

    assert res_default.dtype == targ_line.dtype
    assert res_line.dtype == targ_line.dtype
    assert res_comma.dtype == targ_comma.dtype

    assert_equal(res_default, targ_line)
    assert_equal(res_line, targ_line)
    assert_equal(res_comma, targ_comma)


def test_splitters_empty_str(dtype):
    values = xr.DataArray(
        [["", "", ""], ["", "", ""]],
        dims=["X", "Y"],
    ).astype(dtype)

    conv = {np.unicode_: str, np.bytes_: lambda x: bytes(x, encoding="UTF-8")}[dtype]

    targ_partition_dim = xr.DataArray(
        [
            [["", "", ""], ["", "", ""], ["", "", ""]],
            [["", "", ""], ["", "", ""], ["", "", ""]],
        ],
        dims=["X", "Y", "ZZ"],
    ).astype(dtype)

    targ_partition_none = [
        [["", "", ""], ["", "", ""], ["", "", ""]],
        [["", "", ""], ["", "", ""], ["", "", "", ""]],
    ]
    targ_partition_none = [
        [[conv(x) for x in y] for y in z] for z in targ_partition_none
    ]
    targ_partition_none = np.array(targ_partition_none, dtype=np.object_)
    del targ_partition_none[-1, -1][-1]
    targ_partition_none = xr.DataArray(
        targ_partition_none,
        dims=["X", "Y"],
    )

    targ_split_dim = xr.DataArray(
        [[[""], [""], [""]], [[""], [""], [""]]],
        dims=["X", "Y", "ZZ"],
    ).astype(dtype)
    targ_split_none = xr.DataArray(
        np.array([[[], [], []], [[], [], [""]]], dtype=np.object_),
        dims=["X", "Y"],
    )
    del targ_split_none.data[-1, -1][-1]

    res_partition_dim = values.str.partition(dim="ZZ")
    res_rpartition_dim = values.str.rpartition(dim="ZZ")
    res_partition_none = values.str.partition(dim=None)
    res_rpartition_none = values.str.rpartition(dim=None)

    res_split_dim = values.str.split(dim="ZZ")
    res_rsplit_dim = values.str.rsplit(dim="ZZ")
    res_split_none = values.str.split(dim=None)
    res_rsplit_none = values.str.rsplit(dim=None)

    res_dummies = values.str.rsplit(dim="ZZ")

    assert res_partition_dim.dtype == targ_partition_dim.dtype
    assert res_rpartition_dim.dtype == targ_partition_dim.dtype
    assert res_partition_none.dtype == targ_partition_none.dtype
    assert res_rpartition_none.dtype == targ_partition_none.dtype

    assert res_split_dim.dtype == targ_split_dim.dtype
    assert res_rsplit_dim.dtype == targ_split_dim.dtype
    assert res_split_none.dtype == targ_split_none.dtype
    assert res_rsplit_none.dtype == targ_split_none.dtype

    assert res_dummies.dtype == targ_split_dim.dtype

    assert_equal(res_partition_dim, targ_partition_dim)
    assert_equal(res_rpartition_dim, targ_partition_dim)
    assert_equal(res_partition_none, targ_partition_none)
    assert_equal(res_rpartition_none, targ_partition_none)

    assert_equal(res_split_dim, targ_split_dim)
    assert_equal(res_rsplit_dim, targ_split_dim)
    assert_equal(res_split_none, targ_split_none)
    assert_equal(res_rsplit_none, targ_split_none)

    assert_equal(res_dummies, targ_split_dim)


def test_splitters_empty_array(dtype):
    values = xr.DataArray(
        [[], []],
        dims=["X", "Y"],
    ).astype(dtype)

    targ_dim = xr.DataArray(
        np.empty([2, 0, 0]),
        dims=["X", "Y", "ZZ"],
    ).astype(dtype)
    targ_none = xr.DataArray(
        np.empty([2, 0]),
        dims=["X", "Y"],
    ).astype(np.object_)

    res_part_dim = values.str.partition(dim="ZZ")
    res_rpart_dim = values.str.rpartition(dim="ZZ")
    res_part_none = values.str.partition(dim=None)
    res_rpart_none = values.str.rpartition(dim=None)

    res_split_dim = values.str.split(dim="ZZ")
    res_rsplit_dim = values.str.rsplit(dim="ZZ")
    res_split_none = values.str.split(dim=None)
    res_rsplit_none = values.str.rsplit(dim=None)

    res_dummies = values.str.get_dummies(dim="ZZ")

    assert res_part_dim.dtype == targ_dim.dtype
    assert res_rpart_dim.dtype == targ_dim.dtype
    assert res_part_none.dtype == targ_none.dtype
    assert res_rpart_none.dtype == targ_none.dtype

    assert res_split_dim.dtype == targ_dim.dtype
    assert res_rsplit_dim.dtype == targ_dim.dtype
    assert res_split_none.dtype == targ_none.dtype
    assert res_rsplit_none.dtype == targ_none.dtype

    assert res_dummies.dtype == targ_dim.dtype

    assert_equal(res_part_dim, targ_dim)
    assert_equal(res_rpart_dim, targ_dim)
    assert_equal(res_part_none, targ_none)
    assert_equal(res_rpart_none, targ_none)

    assert_equal(res_split_dim, targ_dim)
    assert_equal(res_rsplit_dim, targ_dim)
    assert_equal(res_split_none, targ_none)
    assert_equal(res_rsplit_none, targ_none)

    assert_equal(res_dummies, targ_dim)
