"""String formatting routines for __repr__.
"""
import contextlib
import functools
from datetime import datetime, timedelta
from itertools import zip_longest

import numpy as np
import pandas as pd
from pandas.errors import OutOfBoundsDatetime

from .duck_array_ops import array_equiv
from .options import OPTIONS
from .pycompat import dask_array_type, sparse_array_type


def pretty_print(x, numchars):
    """Given an object `x`, call `str(x)` and format the returned string so
    that it is numchars long, padding with trailing spaces or truncating with
    ellipses as necessary
    """
    s = maybe_truncate(x, numchars)
    return s + " " * max(numchars - len(s), 0)


def maybe_truncate(obj, maxlen=500):
    s = str(obj)
    if len(s) > maxlen:
        s = s[: (maxlen - 3)] + "..."
    return s


def wrap_indent(text, start="", length=None):
    if length is None:
        length = len(start)
    indent = "\n" + " " * length
    return start + indent.join(x for x in text.splitlines())


def _get_indexer_at_least_n_items(shape, n_desired, from_end):
    assert 0 < n_desired <= np.prod(shape)
    cum_items = np.cumprod(shape[::-1])
    n_steps = np.argmax(cum_items >= n_desired)
    stop = int(np.ceil(float(n_desired) / np.r_[1, cum_items][n_steps]))
    indexer = (
        ((-1 if from_end else 0),) * (len(shape) - 1 - n_steps)
        + ((slice(-stop, None) if from_end else slice(stop)),)
        + (slice(None),) * n_steps
    )
    return indexer


def first_n_items(array, n_desired):
    """Returns the first n_desired items of an array"""
    # Unfortunately, we can't just do array.flat[:n_desired] here because it
    # might not be a numpy.ndarray. Moreover, access to elements of the array
    # could be very expensive (e.g. if it's only available over DAP), so go out
    # of our way to get them in a single call to __getitem__ using only slices.
    if n_desired < 1:
        raise ValueError("must request at least one item")

    if array.size == 0:
        # work around for https://github.com/numpy/numpy/issues/5195
        return []

    if n_desired < array.size:
        indexer = _get_indexer_at_least_n_items(array.shape, n_desired, from_end=False)
        array = array[indexer]
    return np.asarray(array).flat[:n_desired]


def last_n_items(array, n_desired):
    """Returns the last n_desired items of an array"""
    # Unfortunately, we can't just do array.flat[-n_desired:] here because it
    # might not be a numpy.ndarray. Moreover, access to elements of the array
    # could be very expensive (e.g. if it's only available over DAP), so go out
    # of our way to get them in a single call to __getitem__ using only slices.
    if (n_desired == 0) or (array.size == 0):
        return []

    if n_desired < array.size:
        indexer = _get_indexer_at_least_n_items(array.shape, n_desired, from_end=True)
        array = array[indexer]
    return np.asarray(array).flat[-n_desired:]


def last_item(array):
    """Returns the last item of an array in a list or an empty list."""
    if array.size == 0:
        # work around for https://github.com/numpy/numpy/issues/5195
        return []

    indexer = (slice(-1, None),) * array.ndim
    return np.ravel(np.asarray(array[indexer])).tolist()


def format_timestamp(t):
    """Cast given object to a Timestamp and return a nicely formatted string"""
    # Timestamp is only valid for 1678 to 2262
    try:
        datetime_str = str(pd.Timestamp(t))
    except OutOfBoundsDatetime:
        datetime_str = str(t)

    try:
        date_str, time_str = datetime_str.split()
    except ValueError:
        # catch NaT and others that don't split nicely
        return datetime_str
    else:
        if time_str == "00:00:00":
            return date_str
        else:
            return f"{date_str}T{time_str}"


def format_timedelta(t, timedelta_format=None):
    """Cast given object to a Timestamp and return a nicely formatted string"""
    timedelta_str = str(pd.Timedelta(t))
    try:
        days_str, time_str = timedelta_str.split(" days ")
    except ValueError:
        # catch NaT and others that don't split nicely
        return timedelta_str
    else:
        if timedelta_format == "date":
            return days_str + " days"
        elif timedelta_format == "time":
            return time_str
        else:
            return timedelta_str


def format_item(x, timedelta_format=None, quote_strings=True):
    """Returns a succinct summary of an object as a string"""
    if isinstance(x, (np.datetime64, datetime)):
        return format_timestamp(x)
    if isinstance(x, (np.timedelta64, timedelta)):
        return format_timedelta(x, timedelta_format=timedelta_format)
    elif isinstance(x, (str, bytes)):
        return repr(x) if quote_strings else x
    elif isinstance(x, (float, np.float)):
        return f"{x:.4}"
    else:
        return str(x)


def format_items(x):
    """Returns a succinct summaries of all items in a sequence as strings"""
    x = np.asarray(x)
    timedelta_format = "datetime"
    if np.issubdtype(x.dtype, np.timedelta64):
        x = np.asarray(x, dtype="timedelta64[ns]")
        day_part = x[~pd.isnull(x)].astype("timedelta64[D]").astype("timedelta64[ns]")
        time_needed = x[~pd.isnull(x)] != day_part
        day_needed = day_part != np.timedelta64(0, "ns")
        if np.logical_not(day_needed).all():
            timedelta_format = "time"
        elif np.logical_not(time_needed).all():
            timedelta_format = "date"

    formatted = [format_item(xi, timedelta_format) for xi in x]
    return formatted


def format_array_flat(array, max_width):
    """Return a formatted string for as many items in the flattened version of
    array that will fit within max_width characters.
    """
    # every item will take up at least two characters, but we always want to
    # print at least first and last items
    max_possibly_relevant = min(
        max(array.size, 1), max(int(np.ceil(max_width / 2.0)), 2)
    )
    relevant_front_items = format_items(
        first_n_items(array, (max_possibly_relevant + 1) // 2)
    )
    relevant_back_items = format_items(last_n_items(array, max_possibly_relevant // 2))
    # interleave relevant front and back items:
    #     [a, b, c] and [y, z] -> [a, z, b, y, c]
    relevant_items = sum(
        zip_longest(relevant_front_items, reversed(relevant_back_items)), ()
    )[:max_possibly_relevant]

    cum_len = np.cumsum([len(s) + 1 for s in relevant_items]) - 1
    if (array.size > 2) and (
        (max_possibly_relevant < array.size) or (cum_len > max_width).any()
    ):
        padding = " ... "
        count = min(
            array.size, max(np.argmax(cum_len + len(padding) - 1 > max_width), 2)
        )
    else:
        count = array.size
        padding = "" if (count <= 1) else " "

    num_front = (count + 1) // 2
    num_back = count - num_front
    # note that num_back is 0 <--> array.size is 0 or 1
    #                         <--> relevant_back_items is []
    pprint_str = (
        " ".join(relevant_front_items[:num_front])
        + padding
        + " ".join(relevant_back_items[-num_back:])
    )
    return pprint_str


_KNOWN_TYPE_REPRS = {np.ndarray: "np.ndarray"}
with contextlib.suppress(ImportError):
    import sparse

    _KNOWN_TYPE_REPRS[sparse.COO] = "sparse.COO"


def inline_dask_repr(array):
    """Similar to dask.array.DataArray.__repr__, but without
    redundant information that's already printed by the repr
    function of the xarray wrapper.
    """
    assert isinstance(array, dask_array_type), array

    chunksize = tuple(c[0] for c in array.chunks)

    if hasattr(array, "_meta"):
        meta = array._meta
        if type(meta) in _KNOWN_TYPE_REPRS:
            meta_repr = _KNOWN_TYPE_REPRS[type(meta)]
        else:
            meta_repr = type(meta).__name__
        meta_string = f", meta={meta_repr}"
    else:
        meta_string = ""

    return f"dask.array<chunksize={chunksize}{meta_string}>"


def inline_sparse_repr(array):
    """Similar to sparse.COO.__repr__, but without the redundant shape/dtype."""
    assert isinstance(array, sparse_array_type), array
    return "<{}: nnz={:d}, fill_value={!s}>".format(
        type(array).__name__, array.nnz, array.fill_value
    )


def inline_variable_array_repr(var, max_width):
    """Build a one-line summary of a variable's data."""
    if var._in_memory:
        return format_array_flat(var, max_width)
    elif isinstance(var._data, dask_array_type):
        return inline_dask_repr(var.data)
    elif isinstance(var._data, sparse_array_type):
        return inline_sparse_repr(var.data)
    elif hasattr(var._data, "__array_function__"):
        return maybe_truncate(repr(var._data).replace("\n", " "), max_width)
    else:
        # internal xarray array type
        return "..."


def summarize_variable(name, var, col_width, marker=" ", max_width=None):
    """Summarize a variable in one line, e.g., for the Dataset.__repr__."""
    if max_width is None:
        max_width = OPTIONS["display_width"]
    first_col = pretty_print(f"  {marker} {name} ", col_width)
    if var.dims:
        dims_str = "({}) ".format(", ".join(map(str, var.dims)))
    else:
        dims_str = ""
    front_str = f"{first_col}{dims_str}{var.dtype} "

    values_width = max_width - len(front_str)
    values_str = inline_variable_array_repr(var, values_width)

    return front_str + values_str


def _summarize_coord_multiindex(coord, col_width, marker):
    first_col = pretty_print(f"  {marker} {coord.name} ", col_width)
    return "{}({}) MultiIndex".format(first_col, str(coord.dims[0]))


def _summarize_coord_levels(coord, col_width, marker="-"):
    return "\n".join(
        [
            summarize_variable(
                lname, coord.get_level_variable(lname), col_width, marker=marker
            )
            for lname in coord.level_names
        ]
    )


def summarize_datavar(name, var, col_width):
    return summarize_variable(name, var.variable, col_width)


def summarize_coord(name, var, col_width):
    is_index = name in var.dims
    marker = "*" if is_index else " "
    if is_index:
        coord = var.variable.to_index_variable()
        if coord.level_names is not None:
            return "\n".join(
                [
                    _summarize_coord_multiindex(coord, col_width, marker),
                    _summarize_coord_levels(coord, col_width),
                ]
            )
    return summarize_variable(name, var.variable, col_width, marker)


def summarize_attr(key, value, col_width=None):
    """Summary for __repr__ - use ``X.attrs[key]`` for full value."""
    # Indent key and add ':', then right-pad if col_width is not None
    k_str = f"    {key}:"
    if col_width is not None:
        k_str = pretty_print(k_str, col_width)
    # Replace tabs and newlines, so we print on one line in known width
    v_str = str(value).replace("\t", "\\t").replace("\n", "\\n")
    # Finally, truncate to the desired display width
    return maybe_truncate(f"{k_str} {v_str}", OPTIONS["display_width"])


EMPTY_REPR = "    *empty*"


def _get_col_items(mapping):
    """Get all column items to format, including both keys of `mapping`
    and MultiIndex levels if any.
    """
    from .variable import IndexVariable

    col_items = []
    for k, v in mapping.items():
        col_items.append(k)
        var = getattr(v, "variable", v)
        if isinstance(var, IndexVariable):
            level_names = var.to_index_variable().level_names
            if level_names is not None:
                col_items += list(level_names)
    return col_items


def _calculate_col_width(col_items):
    max_name_length = max(len(str(s)) for s in col_items) if col_items else 0
    col_width = max(max_name_length, 7) + 6
    return col_width


def _mapping_repr(mapping, title, summarizer, col_width=None):
    if col_width is None:
        col_width = _calculate_col_width(mapping)
    summary = [f"{title}:"]
    if mapping:
        summary += [summarizer(k, v, col_width) for k, v in mapping.items()]
    else:
        summary += [EMPTY_REPR]
    return "\n".join(summary)


data_vars_repr = functools.partial(
    _mapping_repr, title="Data variables", summarizer=summarize_datavar
)


attrs_repr = functools.partial(
    _mapping_repr, title="Attributes", summarizer=summarize_attr
)


def coords_repr(coords, col_width=None):
    if col_width is None:
        col_width = _calculate_col_width(_get_col_items(coords))
    return _mapping_repr(
        coords, title="Coordinates", summarizer=summarize_coord, col_width=col_width
    )


def indexes_repr(indexes):
    summary = []
    for k, v in indexes.items():
        summary.append(wrap_indent(repr(v), f"{k}: "))
    return "\n".join(summary)


def dim_summary(obj):
    elements = [f"{k}: {v}" for k, v in obj.sizes.items()]
    return ", ".join(elements)


def unindexed_dims_repr(dims, coords):
    unindexed_dims = [d for d in dims if d not in coords]
    if unindexed_dims:
        dims_str = ", ".join(f"{d}" for d in unindexed_dims)
        return "Dimensions without coordinates: " + dims_str
    else:
        return None


@contextlib.contextmanager
def set_numpy_options(*args, **kwargs):
    original = np.get_printoptions()
    np.set_printoptions(*args, **kwargs)
    try:
        yield
    finally:
        np.set_printoptions(**original)


def short_numpy_repr(array):
    array = np.asarray(array)

    # default to lower precision so a full (abbreviated) line can fit on
    # one line with the default display_width
    options = {"precision": 6, "linewidth": OPTIONS["display_width"], "threshold": 200}
    if array.ndim < 3:
        edgeitems = 3
    elif array.ndim == 3:
        edgeitems = 2
    else:
        edgeitems = 1
    options["edgeitems"] = edgeitems
    with set_numpy_options(**options):
        return repr(array)


def short_data_repr(array):
    """Format "data" for DataArray and Variable."""
    internal_data = getattr(array, "variable", array)._data
    if isinstance(array, np.ndarray):
        return short_numpy_repr(array)
    elif hasattr(internal_data, "__array_function__") or isinstance(
        internal_data, dask_array_type
    ):
        return repr(array.data)
    elif array._in_memory or array.size < 1e5:
        return short_numpy_repr(array)
    else:
        # internal xarray array type
        return f"[{array.size} values with dtype={array.dtype}]"


def array_repr(arr):
    # used for DataArray, Variable and IndexVariable
    if hasattr(arr, "name") and arr.name is not None:
        name_str = f"{arr.name!r} "
    else:
        name_str = ""

    summary = [
        "<xarray.{} {}({})>".format(type(arr).__name__, name_str, dim_summary(arr)),
        short_data_repr(arr),
    ]

    if hasattr(arr, "coords"):
        if arr.coords:
            summary.append(repr(arr.coords))

        unindexed_dims_str = unindexed_dims_repr(arr.dims, arr.coords)
        if unindexed_dims_str:
            summary.append(unindexed_dims_str)

    if arr.attrs:
        summary.append(attrs_repr(arr.attrs))

    return "\n".join(summary)


def dataset_repr(ds):
    summary = ["<xarray.{}>".format(type(ds).__name__)]

    col_width = _calculate_col_width(_get_col_items(ds.variables))

    dims_start = pretty_print("Dimensions:", col_width)
    summary.append("{}({})".format(dims_start, dim_summary(ds)))

    if ds.coords:
        summary.append(coords_repr(ds.coords, col_width=col_width))

    unindexed_dims_str = unindexed_dims_repr(ds.dims, ds.coords)
    if unindexed_dims_str:
        summary.append(unindexed_dims_str)

    summary.append(data_vars_repr(ds.data_vars, col_width=col_width))

    if ds.attrs:
        summary.append(attrs_repr(ds.attrs))

    return "\n".join(summary)


def diff_dim_summary(a, b):
    if a.dims != b.dims:
        return "Differing dimensions:\n    ({}) != ({})".format(
            dim_summary(a), dim_summary(b)
        )
    else:
        return ""


def _diff_mapping_repr(a_mapping, b_mapping, compat, title, summarizer, col_width=None):
    def extra_items_repr(extra_keys, mapping, ab_side):
        extra_repr = [summarizer(k, mapping[k], col_width) for k in extra_keys]
        if extra_repr:
            header = f"{title} only on the {ab_side} object:"
            return [header] + extra_repr
        else:
            return []

    a_keys = set(a_mapping)
    b_keys = set(b_mapping)

    summary = []

    diff_items = []

    for k in a_keys & b_keys:
        try:
            # compare xarray variable
            compatible = getattr(a_mapping[k], compat)(b_mapping[k])
            is_variable = True
        except AttributeError:
            # compare attribute value
            compatible = a_mapping[k] == b_mapping[k]
            is_variable = False

        if not compatible:
            temp = [
                summarizer(k, vars[k], col_width) for vars in (a_mapping, b_mapping)
            ]

            if compat == "identical" and is_variable:
                attrs_summary = []

                for m in (a_mapping, b_mapping):
                    attr_s = "\n".join(
                        [summarize_attr(ak, av) for ak, av in m[k].attrs.items()]
                    )
                    attrs_summary.append(attr_s)

                temp = [
                    "\n".join([var_s, attr_s]) if attr_s else var_s
                    for var_s, attr_s in zip(temp, attrs_summary)
                ]

            diff_items += [ab_side + s[1:] for ab_side, s in zip(("L", "R"), temp)]

    if diff_items:
        summary += ["Differing {}:".format(title.lower())] + diff_items

    summary += extra_items_repr(a_keys - b_keys, a_mapping, "left")
    summary += extra_items_repr(b_keys - a_keys, b_mapping, "right")

    return "\n".join(summary)


diff_coords_repr = functools.partial(
    _diff_mapping_repr, title="Coordinates", summarizer=summarize_coord
)


diff_data_vars_repr = functools.partial(
    _diff_mapping_repr, title="Data variables", summarizer=summarize_datavar
)


diff_attrs_repr = functools.partial(
    _diff_mapping_repr, title="Attributes", summarizer=summarize_attr
)


def _compat_to_str(compat):
    if compat == "equals":
        return "equal"
    else:
        return compat


def diff_array_repr(a, b, compat):
    # used for DataArray, Variable and IndexVariable
    summary = [
        "Left and right {} objects are not {}".format(
            type(a).__name__, _compat_to_str(compat)
        )
    ]

    summary.append(diff_dim_summary(a, b))

    if not array_equiv(a.data, b.data):
        temp = [wrap_indent(short_numpy_repr(obj), start="    ") for obj in (a, b)]
        diff_data_repr = [
            ab_side + "\n" + ab_data_repr
            for ab_side, ab_data_repr in zip(("L", "R"), temp)
        ]
        summary += ["Differing values:"] + diff_data_repr

    if hasattr(a, "coords"):
        col_width = _calculate_col_width(set(a.coords) | set(b.coords))
        summary.append(
            diff_coords_repr(a.coords, b.coords, compat, col_width=col_width)
        )

    if compat == "identical":
        summary.append(diff_attrs_repr(a.attrs, b.attrs, compat))

    return "\n".join(summary)


def diff_dataset_repr(a, b, compat):
    summary = [
        "Left and right {} objects are not {}".format(
            type(a).__name__, _compat_to_str(compat)
        )
    ]

    col_width = _calculate_col_width(
        set(_get_col_items(a.variables) + _get_col_items(b.variables))
    )

    summary.append(diff_dim_summary(a, b))
    summary.append(diff_coords_repr(a.coords, b.coords, compat, col_width=col_width))
    summary.append(
        diff_data_vars_repr(a.data_vars, b.data_vars, compat, col_width=col_width)
    )

    if compat == "identical":
        summary.append(diff_attrs_repr(a.attrs, b.attrs, compat))

    return "\n".join(summary)
