class Import:
    """Benchmark importing xarray"""

    def timeraw_import_xarray():
        return """
        import xarray
        """

    def timeraw_import_xarray_plot():
        return """
        import xarray.plot as xplt
        """
