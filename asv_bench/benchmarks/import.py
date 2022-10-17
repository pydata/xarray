class Import:
    """Benchmark importing xarray"""

    def timeraw_import_xarray():
        return """
        import xarray
        """

    def timeraw_import_xarray_plot():
        return """
        import xarray.plot
        """

    def timeraw_import_xarray_backends():
        return """
        from xarray.backends import list_engines
        list_engines()
        """
