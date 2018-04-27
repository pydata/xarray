from __future__ import absolute_import, division, print_function


def interpolate_at(obj, method='liner', **dest):
    """
    Interpolate DataArray or Dataset.

    Parameters
    ----------
    obj: DataArray or Dataset
        Object to be interpolated
    method: string
        interpolation method
    dest: mapping from dimensional coordinate name to the new coordinateself.

    Returns
    -------
    interpolated: DataArray or Dataset
        Interpolated object

    Note
    ----
    Non-dimensional coordinates those in interpolated dimensions are dropped
    by this method.
    """
    pass


def interp1d(array, old_coord, new_coord, method):
    f = interpolate.interp1d(array, old_coord, kind=method)
    return f(new_coord)


def ghosted_interp1d(array, old_coord, new_coord, method):
    import dask.array as da

    assert isinstance(array, da.ndarray)
    


def interp_func(array, old_coords, new_coords, method):
    """
    array: np.ndarray or da.ndarray. Should not contain nan.
    old_coords, new_coords: list of np.ndarrays.
    """
    from scipy import interpolate



def ghosted_interp_func(array, old_coords, new_coords, method):
    """
    Interpolation with ghosting
    """
    return
