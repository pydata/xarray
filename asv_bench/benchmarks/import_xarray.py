class ImportXarray:
    def setup(self, *args, **kwargs):
        def import_xr():
            import xarray  # noqa: F401

        self._import_xr = import_xr

    def import_xarray(self):
        self._import_xr()
