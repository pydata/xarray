from collections.abc import Mapping
from typing import Literal

from xarray.coding.core import Coder
from xarray.coding.times import CFDatetimeCoder, CFTimedeltaCoder
from xarray.core.dataset import Dataset
from xarray.core.variable import Variable


def cf_coders(
    mask_and_scale: bool | Mapping[str, bool] | None = None,
    decode_times: (
        bool | CFDatetimeCoder | Mapping[str, bool | CFDatetimeCoder] | None
    ) = None,
    decode_timedelta: (
        bool | CFTimedeltaCoder | Mapping[str, bool | CFTimedeltaCoder] | None
    ) = None,
    concat_characters: bool | Mapping[str, bool] | None = None,
    decode_coords: Literal["coordinates", "all"] | bool | None = None,
) -> tuple[list[Coder[Dataset]], list[Coder[Variable]]]:
    """Create the default cf coders

    Parameters
    ----------
    mask_and_scale : bool or mapping of str to bool, optional
        If True, replace array values equal to `_FillValue` with NA and scale
        values according to the formula `original_values * scale_factor +
        add_offset`, where `_FillValue`, `scale_factor` and `add_offset` are
        taken from variable attributes (if they exist).  If the `_FillValue` or
        `missing_value` attribute contains multiple values a warning will be
        issued and all array values matching one of the multiple values will
        be replaced by NA. Pass a mapping, e.g. ``{"my_variable": False}``,
        to toggle this feature per-variable individually.
    decode_times : bool, CFDatetimeCoder or dict-like, optional
        If True, decode times encoded in the standard NetCDF datetime format
        into datetime objects. Otherwise, use :py:class:`coders.CFDatetimeCoder` or leave them
        encoded as numbers.
        Pass a mapping, e.g. ``{"my_variable": False}``,
        to toggle this feature per-variable individually.
    decode_timedelta : bool, CFTimedeltaCoder, or dict-like, optional
        If True, decode variables and coordinates with time units in
        {"days", "hours", "minutes", "seconds", "milliseconds", "microseconds"}
        into timedelta objects. If False, leave them encoded as numbers.
        If None (default), assume the same value of ``decode_times``; if
        ``decode_times`` is a :py:class:`coders.CFDatetimeCoder` instance, this
        takes the form of a :py:class:`coders.CFTimedeltaCoder` instance with a
        matching ``time_unit``.
        Pass a mapping, e.g. ``{"my_variable": False}``,
        to toggle this feature per-variable individually.
    concat_characters : bool or dict-like, optional
        If True, concatenate along the last dimension of character arrays to
        form string arrays. Dimensions will only be concatenated over (and
        removed) if they have no corresponding variable and if they are only
        used as the last dimension of character arrays.
        Pass a mapping, e.g. ``{"my_variable": False}``,
        to toggle this feature per-variable individually.
        This keyword may not be supported by all the backends.
    decode_coords : bool or {"coordinates", "all"}, optional
        Controls which variables are set as coordinate variables:

        - "coordinates" or True: Set variables referred to in the
          ``'coordinates'`` attribute of the datasets or individual variables
          as coordinate variables.
        - "all": Set variables referred to in  ``'grid_mapping'``, ``'bounds'`` and
          other attributes as coordinate variables.

        Only existing variables can be set as coordinates. Missing variables
        will be silently ignored.

    Returns
    -------
    dataset_coders : list of Coder
        The constructed dataset coders.
    variable_coders : list of Coder
        The constructed variable coders.

    See Also
    --------
    decode, encode
    decode_cf
    """


def encode(
    obj: Dataset,
    *,
    dataset_coders: list[Coder[Dataset]] | None = None,
    variable_coders: list[Coder[Variable]] | None = None,
) -> Dataset:
    """Encode a dataset using the given coders

    Parameters
    ----------
    obj : xarray.Dataset
        The dataset to encode.
    dataset_coders : list of Coder, optional
        The dataset coders to apply.
    variable_coders : list of Coder, optional
        The variable coders to apply.

    Returns
    -------
    xarray.Dataset
        The encoded dataset after applying all coders.

    Notes
    -----
    Coders that return `NotImplemented` will be skipped.

    See Also
    --------
    decode, decode_cf
    """
    encoded = obj
    for coder in dataset_coders:
        encoded = coder.encode(encoded)

    # by applying the dataset coders first there are no coordinates anymore (though we may want to check that)
    for coder in variable_coders:
        encoded = encoded.map(coder.encode)

    return encoded


def decode(
    obj: Dataset,
    *,
    dataset_coders: list[Coder[Dataset]] | None = None,
    variable_coders: list[Coder[Variable]] | None = None,
) -> Dataset:
    """Decode a dataset using the given coders

    Parameters
    ----------
    obj : xarray.Dataset
        The dataset to decode.
    dataset_coders : list of Coder, optional
        The dataset coders to apply.
    variable_coders : list of Coder, optional
        The variable coders to apply.

    Returns
    -------
    xarray.Dataset
        The decoded dataset after applying all coders.

    Notes
    -----
    Coders that return `NotImplemented` will be skipped.

    See Also
    --------
    encode, decode_cf
    """
    decoded = obj
    for coder in variable_coders:
        decoded = decoded.map(coder.decode)

    for coder in dataset_coders:
        decoded = coder.decode(decoded)

    return decoded
