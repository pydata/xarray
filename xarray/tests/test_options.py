from __future__ import absolute_import, division, print_function

import pytest

import xarray
from xarray.core.options import OPTIONS
from xarray.backends.file_manager import FILE_CACHE


def test_invalid_option_raises():
    with pytest.raises(ValueError):
        xarray.set_options(not_a_valid_options=True)


def test_display_width():
    with pytest.raises(ValueError):
        xarray.set_options(display_width=0)
    with pytest.raises(ValueError):
        xarray.set_options(display_width=-10)
    with pytest.raises(ValueError):
        xarray.set_options(display_width=3.5)


def test_arithmetic_join():
    with pytest.raises(ValueError):
        xarray.set_options(arithmetic_join='invalid')
    with xarray.set_options(arithmetic_join='exact'):
        assert OPTIONS['arithmetic_join'] == 'exact'


def test_enable_cftimeindex():
    with pytest.raises(ValueError):
        xarray.set_options(enable_cftimeindex=None)
    with xarray.set_options(enable_cftimeindex=True):
        assert OPTIONS['enable_cftimeindex']
    with pytest.warns(FutureWarning):
        with xarray.set_options(enable_cftimeindex=True):
            pass


def test_file_cache_maxsize():
    with pytest.raises(ValueError):
        xarray.set_options(file_cache_maxsize=0)
    original_size = FILE_CACHE.maxsize
    with xarray.set_options(file_cache_maxsize=123):
        assert FILE_CACHE.maxsize == 123
    assert FILE_CACHE.maxsize == original_size


def test_nested_options():
    original = OPTIONS['display_width']
    with xarray.set_options(display_width=1):
        assert OPTIONS['display_width'] == 1
        with xarray.set_options(display_width=2):
            assert OPTIONS['display_width'] == 2
        assert OPTIONS['display_width'] == 1
    assert OPTIONS['display_width'] == original
