import numpy as np

from . import utils


# Use as a sentinel value to indicate a dtype appropriate NA value.
NA = utils.ReprObject('<NA>')


def maybe_promote(dtype):
    """Simpler equivalent of pandas.core.common._maybe_promote

    Parameters
    ----------
    dtype : np.dtype

    Returns
    -------
    dtype : Promoted dtype that can hold missing values.
    fill_value : Valid missing value for the promoted dtype.
    """
    # N.B. these casting rules should match pandas
    if np.issubdtype(dtype, np.floating):
        fill_value = np.nan
    elif np.issubdtype(dtype, np.integer):
        if dtype.itemsize <= 2:
            dtype = np.float32
        else:
            dtype = np.float64
        fill_value = np.nan
    elif np.issubdtype(dtype, np.complexfloating):
        fill_value = np.nan + np.nan * 1j
    elif np.issubdtype(dtype, np.datetime64):
        fill_value = np.datetime64('NaT')
    elif np.issubdtype(dtype, np.timedelta64):
        fill_value = np.timedelta64('NaT')
    else:
        dtype = object
        fill_value = np.nan
    return np.dtype(dtype), fill_value


def get_fill_value(dtype):
    """Return an appropriate fill value for this dtype.

    Parameters
    ----------
    dtype : np.dtype

    Returns
    -------
    fill_value : Missing value corresponding to this dtype.
    """
    _, fill_value = maybe_promote(dtype)
    return fill_value


def is_datetime_like(dtype):
    """Check if a dtype is a subclass of the numpy datetime types
    """
    return (np.issubdtype(dtype, np.datetime64) or
            np.issubdtype(dtype, np.timedelta64))


def result_type(*arrays_and_dtypes):
    """Like np.result_type, but number + string -> object (not string).

    Parameters
    ----------
    *arrays_and_dtypes : list of arrays and dtypes
        The dtype is extracted from both numpy and dask arrays.

    Returns
    -------
    numpy.dtype for the result.
    """
    types = [np.result_type(t).type for t in arrays_and_dtypes]

    # For reference, see the NumPy type hierarchy:
    # https://docs.scipy.org/doc/numpy-1.13.0/reference/arrays.scalars.html
    if (any(issubclass(t, (np.number, np.bool_)) for t in types) and
            any(issubclass(t, np.character) for t in types)):
        return np.dtype(object)
    else:
        return np.result_type(*arrays_and_dtypes)
