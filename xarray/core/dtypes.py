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
    if np.issubdtype(dtype, float):
        fill_value = np.nan
    elif np.issubdtype(dtype, int):
        # convert to floating point so NaN is valid
        dtype = float
        fill_value = np.nan
    elif np.issubdtype(dtype, complex):
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
