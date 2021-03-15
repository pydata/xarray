#%%
def testing(filename, parse_coordinates=None, chunks=None, cache=None, lock=None):
    """Open a file with rasterio (experimental).

    This should work with any file that rasterio can open (most often:
    geoTIFF). The x and y coordinates are generated automatically from the
    file's geoinformation, shifted to the center of each pixel (see
    `"PixelIsArea" Raster Space
    <http://web.archive.org/web/20160326194152/http://remotesensing.org/geotiff/spec/geotiff2.5.html#2.5.2>`_
    for more information).

    You can generate 2D coordinates from the file's attributes with::

        >>> from affine import Affine
        >>> import xarray as xr
        >>> import numpy as np
        >>> da = xr.open_rasterio('https://github.com/mapbox/rasterio/raw/master/tests/data/RGB.byte.tif')
        >>> da
        <xarray.DataArray (band: 3, y: 718, x: 791)>
        [1703814 values with dtype=uint8]
        Coordinates:
          * band     (band) int64 1 2 3
          * y        (y) float64 2.827e+06 2.826e+06 2.826e+06 ... 2.612e+06 2.612e+06
          * x        (x) float64 1.021e+05 1.024e+05 1.027e+05 ... 3.389e+05 3.392e+05
        Attributes:
            transform:      (300.0379266750948, 0.0, 101985.0, 0.0, -300.041782729805...
            crs:            +init=epsg:32618
            res:            (300.0379266750948, 300.041782729805)
            is_tiled:       0
            nodatavals:     (0.0, 0.0, 0.0)
            scales:         (1.0, 1.0, 1.0)
            offsets:        (0.0, 0.0, 0.0)
            AREA_OR_POINT:  Area
        >>> transform = Affine(*da.transform)
        >>> transform
        Affine(300.0379266750948, 0.0, 101985.0,
               0.0, -300.041782729805, 2826915.0)
        >>> nx, ny = da.sizes['x'], da.sizes['y']
        >>> x, y = transform * np.meshgrid(np.arange(nx)+0.5, np.arange(ny)+0.5)
        >>> x
        array([[102135.01896334, 102435.05689001, 102735.09481669, ...,
                338564.90518331, 338864.94310999, 339164.98103666],
               [102135.01896334, 102435.05689001, 102735.09481669, ...,
                338564.90518331, 338864.94310999, 339164.98103666],
               [102135.01896334, 102435.05689001, 102735.09481669, ...,
                338564.90518331, 338864.94310999, 339164.98103666],
               ...,
               [102135.01896334, 102435.05689001, 102735.09481669, ...,
                338564.90518331, 338864.94310999, 339164.98103666],
               [102135.01896334, 102435.05689001, 102735.09481669, ...,
                338564.90518331, 338864.94310999, 339164.98103666],
               [102135.01896334, 102435.05689001, 102735.09481669, ...,
                338564.90518331, 338864.94310999, 339164.98103666]])

    Parameters
    ----------
    filename : str, rasterio.DatasetReader, or rasterio.WarpedVRT
        Path to the file to open. Or already open rasterio dataset.
    parse_coordinates : bool, optional
        Whether to parse the x and y coordinates out of the file's
        ``transform`` attribute or not. The default is to automatically
        parse the coordinates only if they are rectilinear (1D).
        It can be useful to set ``parse_coordinates=False``
        if your files are very large or if you don't need the coordinates.
    chunks : int, tuple or dict, optional
        Chunk sizes along each dimension, e.g., ``5``, ``(5, 5)`` or
        ``{'x': 5, 'y': 5}``. If chunks is provided, it used to load the new
        DataArray into a dask array.
    cache : bool, optional
        If True, cache data loaded from the underlying datastore in memory as
        NumPy arrays when accessed to avoid reading from the underlying data-
        store multiple times. Defaults to True unless you specify the `chunks`
        argument to use dask, in which case it defaults to False.
    lock : False, True or threading.Lock, optional
        If chunks is provided, this argument is passed on to
        :py:func:`dask.array.from_array`. By default, a global lock is
        used to avoid issues with concurrent access to the same file when using
        dask's multithreaded backend.

    Returns
    -------
    data : DataArray
        The newly created DataArray.
    """
    import rasterio
    from rasterio.vrt import WarpedVRT

if __name__ == "__main__":
    import doctest
    doctest.testmod()