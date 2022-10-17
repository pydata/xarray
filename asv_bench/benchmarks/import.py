class Import:
    """Benchmark importing xarray"""

    def timeraw_import_xarray(self):
        return """
        import xarray
        """

    def timeraw_import_xarray_plot(self):
        return """
        import xarray.plot
        """

    def timeraw_import_xarray_backends(self):
        return """
        from xarray.backends import list_engines
        list_engines()
        """
