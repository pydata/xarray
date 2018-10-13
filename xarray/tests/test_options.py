from __future__ import absolute_import, division, print_function

import pytest

import xarray
from xarray.core.options import OPTIONS, _get_keep_attrs
from xarray.backends.file_manager import FILE_CACHE
from xarray.tests.test_dataset import create_test_data


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


def test_file_cache_maxsize():
    with pytest.raises(ValueError):
        xarray.set_options(file_cache_maxsize=0)
    original_size = FILE_CACHE.maxsize
    with xarray.set_options(file_cache_maxsize=123):
        assert FILE_CACHE.maxsize == 123
    assert FILE_CACHE.maxsize == original_size


def test_keep_attrs():
    with pytest.raises(ValueError):
        xarray.set_options(keep_attrs='invalid_str')
    with xarray.set_options(keep_attrs=True):
        assert OPTIONS['keep_attrs']
    with xarray.set_options(keep_attrs=False):
        assert not OPTIONS['keep_attrs']
    with xarray.set_options(keep_attrs='default'):
        assert _get_keep_attrs(default=True)
        assert not _get_keep_attrs(default=False)


def create_test_data_attrs(seed=0):
    ds = create_test_data(seed)
    ds.attrs = {'attr1': 5, 'attr2': 'history',
                'attr3': {'nested': 'more_info'}}
    return ds


def test_attr_retention():
    ds = create_test_data_attrs()
    original_attrs = ds.attrs

    # Test default behaviour
    result = ds.mean()
    assert result.attrs == {}
    with xarray.set_options(keep_attrs='default'):
        result = ds.mean()
        assert result.attrs == {}

    with xarray.set_options(keep_attrs=True):
        result = ds.mean()
        assert result.attrs == original_attrs

    with xarray.set_options(keep_attrs=False):
        result = ds.mean()
        assert result.attrs == {}


def test_nested_options():
    original = OPTIONS['display_width']
    with xarray.set_options(display_width=1):
        assert OPTIONS['display_width'] == 1
        with xarray.set_options(display_width=2):
            assert OPTIONS['display_width'] == 2
        assert OPTIONS['display_width'] == 1
    assert OPTIONS['display_width'] == original
