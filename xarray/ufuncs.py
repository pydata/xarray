"""xarray specific universal functions."""

import textwrap

import numpy as np

import xarray as xr
from xarray.core.groupby import GroupBy


def _walk_array_namespaces(obj, namespaces):
    if isinstance(obj, xr.DataTree):
        # TODO: DataTree doesn't actually support ufuncs yet
        for node in obj.subtree:
            _walk_array_namespaces(node.dataset, namespaces)
    elif isinstance(obj, xr.Dataset):
        for name in obj.data_vars:
            _walk_array_namespaces(obj[name], namespaces)
    elif isinstance(obj, GroupBy):
        _walk_array_namespaces(next(iter(obj))[1], namespaces)
    elif isinstance(obj, xr.DataArray | xr.Variable):
        _walk_array_namespaces(obj.data, namespaces)
    else:
        namespace = getattr(obj, "__array_namespace__", None)
        if namespace is not None:
            namespaces.add(namespace())

    return namespaces


def get_array_namespace(*args):
    xps = set()
    for arg in args:
        _walk_array_namespaces(arg, xps)

    xps.discard(np)
    if len(xps) > 1:
        names = [module.__name__ for module in xps]
        raise ValueError(f"Mixed array types {names} are not supported.")

    return next(iter(xps)) if len(xps) else np


class _UnaryUfunc:
    """Wrapper for dispatching unary ufuncs."""

    def __init__(self, name):
        self._name = name

    def __call__(self, x, **kwargs):
        xp = get_array_namespace(x)
        func = getattr(xp, self._name)
        return xr.apply_ufunc(func, x, dask="allowed", **kwargs)


class _BinaryUfunc:
    """Wrapper for dispatching binary ufuncs."""

    def __init__(self, name):
        self._name = name

    def __call__(self, x, y, **kwargs):
        xp = get_array_namespace(x, y)
        func = getattr(xp, self._name)
        return xr.apply_ufunc(func, x, y, dask="allowed", **kwargs)


class _UnavailableUfunc:
    """Wrapper for unimplemented ufuncs in older numpy versions."""

    def __init__(self, name):
        self._name = name

    def __call__(self, *args, **kwargs):
        raise NotImplementedError(
            f"Ufunc {self._name} is not available in numpy {np.__version__}."
        )


def _skip_signature(doc, name):
    if not isinstance(doc, str):
        return doc

    if doc.startswith(name):
        signature_end = doc.find("\n\n")
        doc = doc[signature_end + 2 :]

    return doc


def _remove_unused_reference_labels(doc):
    if not isinstance(doc, str):
        return doc

    max_references = 5
    for num in range(max_references):
        label = f".. [{num}]"
        reference = f"[{num}]_"
        index = f"{num}.    "

        if label not in doc or reference in doc:
            continue

        doc = doc.replace(label, index)

    return doc


def _dedent(doc):
    if not isinstance(doc, str):
        return doc

    return textwrap.dedent(doc)


def _create_op(name):
    if not hasattr(np, name):
        # handle older numpy versions with missing array api standard aliases
        if np.lib.NumpyVersion(np.__version__) < "2.0.0":
            return _UnavailableUfunc(name)
        raise ValueError(f"'{name}' is not a valid numpy function")

    np_func = getattr(np, name)
    if hasattr(np_func, "nin") and np_func.nin == 2:
        func = _BinaryUfunc(name)
    else:
        func = _UnaryUfunc(name)

    func.__name__ = name
    doc = getattr(np, name).__doc__

    doc = _remove_unused_reference_labels(_skip_signature(_dedent(doc), name))

    func.__doc__ = (
        f"xarray specific variant of numpy.{name}. Handles "
        "xarray objects by dispatching to the appropriate "
        "function for the underlying array type.\n\n"
        f"Documentation from numpy:\n\n{doc}"
    )
    return func


# These can be auto-generated from the public numpy ufuncs:
# {name for name in dir(np) if isinstance(getattr(np, name), np.ufunc)}

# Ufuncs that use core dimensions or product multiple output arrays are
# not currently supported, and left commented below.

# UNARY
abs = _create_op("abs")
absolute = _create_op("absolute")
acos = _create_op("acos")
acosh = _create_op("acosh")
arccos = _create_op("arccos")
arccosh = _create_op("arccosh")
arcsin = _create_op("arcsin")
arcsinh = _create_op("arcsinh")
arctan = _create_op("arctan")
arctanh = _create_op("arctanh")
asin = _create_op("asin")
asinh = _create_op("asinh")
atan = _create_op("atan")
atanh = _create_op("atanh")
bitwise_count = _create_op("bitwise_count")
bitwise_invert = _create_op("bitwise_invert")
bitwise_not = _create_op("bitwise_not")
cbrt = _create_op("cbrt")
ceil = _create_op("ceil")
conj = _create_op("conj")
conjugate = _create_op("conjugate")
cos = _create_op("cos")
cosh = _create_op("cosh")
deg2rad = _create_op("deg2rad")
degrees = _create_op("degrees")
exp = _create_op("exp")
exp2 = _create_op("exp2")
expm1 = _create_op("expm1")
fabs = _create_op("fabs")
floor = _create_op("floor")
# frexp = _create_op("frexp")
invert = _create_op("invert")
isfinite = _create_op("isfinite")
isinf = _create_op("isinf")
isnan = _create_op("isnan")
isnat = _create_op("isnat")
log = _create_op("log")
log10 = _create_op("log10")
log1p = _create_op("log1p")
log2 = _create_op("log2")
logical_not = _create_op("logical_not")
# modf = _create_op("modf")
negative = _create_op("negative")
positive = _create_op("positive")
rad2deg = _create_op("rad2deg")
radians = _create_op("radians")
reciprocal = _create_op("reciprocal")
rint = _create_op("rint")
sign = _create_op("sign")
signbit = _create_op("signbit")
sin = _create_op("sin")
sinh = _create_op("sinh")
spacing = _create_op("spacing")
sqrt = _create_op("sqrt")
square = _create_op("square")
tan = _create_op("tan")
tanh = _create_op("tanh")
trunc = _create_op("trunc")

# BINARY
add = _create_op("add")
arctan2 = _create_op("arctan2")
atan2 = _create_op("atan2")
bitwise_and = _create_op("bitwise_and")
bitwise_left_shift = _create_op("bitwise_left_shift")
bitwise_or = _create_op("bitwise_or")
bitwise_right_shift = _create_op("bitwise_right_shift")
bitwise_xor = _create_op("bitwise_xor")
copysign = _create_op("copysign")
divide = _create_op("divide")
# divmod = _create_op("divmod")
equal = _create_op("equal")
float_power = _create_op("float_power")
floor_divide = _create_op("floor_divide")
fmax = _create_op("fmax")
fmin = _create_op("fmin")
fmod = _create_op("fmod")
gcd = _create_op("gcd")
greater = _create_op("greater")
greater_equal = _create_op("greater_equal")
heaviside = _create_op("heaviside")
hypot = _create_op("hypot")
lcm = _create_op("lcm")
ldexp = _create_op("ldexp")
left_shift = _create_op("left_shift")
less = _create_op("less")
less_equal = _create_op("less_equal")
logaddexp = _create_op("logaddexp")
logaddexp2 = _create_op("logaddexp2")
logical_and = _create_op("logical_and")
logical_or = _create_op("logical_or")
logical_xor = _create_op("logical_xor")
# matmul = _create_op("matmul")
maximum = _create_op("maximum")
minimum = _create_op("minimum")
mod = _create_op("mod")
multiply = _create_op("multiply")
nextafter = _create_op("nextafter")
not_equal = _create_op("not_equal")
pow = _create_op("pow")
power = _create_op("power")
remainder = _create_op("remainder")
right_shift = _create_op("right_shift")
subtract = _create_op("subtract")
true_divide = _create_op("true_divide")
# vecdot = _create_op("vecdot")

# elementwise non-ufunc
angle = _create_op("angle")
isreal = _create_op("isreal")
iscomplex = _create_op("iscomplex")


__all__ = [
    "abs",
    "absolute",
    "acos",
    "acosh",
    "arccos",
    "arccosh",
    "arcsin",
    "arcsinh",
    "arctan",
    "arctanh",
    "asin",
    "asinh",
    "atan",
    "atanh",
    "bitwise_count",
    "bitwise_invert",
    "bitwise_not",
    "cbrt",
    "ceil",
    "conj",
    "conjugate",
    "cos",
    "cosh",
    "deg2rad",
    "degrees",
    "exp",
    "exp2",
    "expm1",
    "fabs",
    "floor",
    "invert",
    "isfinite",
    "isinf",
    "isnan",
    "isnat",
    "log",
    "log10",
    "log1p",
    "log2",
    "logical_not",
    "negative",
    "positive",
    "rad2deg",
    "radians",
    "reciprocal",
    "rint",
    "sign",
    "signbit",
    "sin",
    "sinh",
    "spacing",
    "sqrt",
    "square",
    "tan",
    "tanh",
    "trunc",
    "add",
    "arctan2",
    "atan2",
    "bitwise_and",
    "bitwise_left_shift",
    "bitwise_or",
    "bitwise_right_shift",
    "bitwise_xor",
    "copysign",
    "divide",
    "equal",
    "float_power",
    "floor_divide",
    "fmax",
    "fmin",
    "fmod",
    "gcd",
    "greater",
    "greater_equal",
    "heaviside",
    "hypot",
    "lcm",
    "ldexp",
    "left_shift",
    "less",
    "less_equal",
    "logaddexp",
    "logaddexp2",
    "logical_and",
    "logical_or",
    "logical_xor",
    "maximum",
    "minimum",
    "mod",
    "multiply",
    "nextafter",
    "not_equal",
    "pow",
    "power",
    "remainder",
    "right_shift",
    "subtract",
    "true_divide",
    "angle",
    "isreal",
    "iscomplex",
]
