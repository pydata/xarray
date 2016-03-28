import xarray as xr

from . import TestCase


class TestAccessor(TestCase):
    def test_register(self):

        @xr.register_dataset_accessor('demo')
        @xr.register_dataarray_accessor('demo')
        class DemoAccessor(object):
            def __init__(self, xarray_obj):
                self._obj = xarray_obj

            @property
            def foo(self):
                return 'bar'

        ds = xr.Dataset()
        assert ds.demo.foo == 'bar'

        da = xr.DataArray(0)
        assert da.demo.foo == 'bar'

        del xr.Dataset.demo
        assert not hasattr(ds, 'demo')

        with self.assertRaises(xr.core.extensions.AccessorRegistrationError):
            @xr.register_dataarray_accessor('demo')
            class Foo(object):
                pass
