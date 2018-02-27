from __future__ import absolute_import, division, print_function

import pytest

import xarray
from xarray.core.options import OPTIONS


def test_invalid_option_raises():
    with pytest.raises(ValueError):
        xarray.set_options(not_a_valid_options=True)


def test_nested_options():
    original = OPTIONS['display_width']
    with xarray.set_options(display_width=1):
        assert OPTIONS['display_width'] == 1
        with xarray.set_options(display_width=2):
            assert OPTIONS['display_width'] == 2
        assert OPTIONS['display_width'] == 1
    assert OPTIONS['display_width'] == original
