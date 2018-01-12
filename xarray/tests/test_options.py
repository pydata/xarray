from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import xarray
import pytest

from xarray.core.options import OPTIONS, _AVAILABLE_IO_ENGINES


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


@pytest.mark.skipif(not _AVAILABLE_IO_ENGINES,
                    reason='No available I/O engines')
def test_set_io_engines():
    with xarray.set_options(io_engines=_AVAILABLE_IO_ENGINES[0]):
        assert OPTIONS['io_engines'] == [_AVAILABLE_IO_ENGINES[0]]


def test_invalid_io_engine_raises():
    with pytest.raises(ValueError):
        xarray.set_options(io_engines='foo')
