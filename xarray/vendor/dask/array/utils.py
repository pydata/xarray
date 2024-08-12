import numbers
import numpy

from packaging.version import Version


if Version(numpy.__version__).release >= (2, 0):
    from numpy.exceptions import AxisError
else:
    from numpy import AxisError  # type: ignore[attr-defined, no-redef]


def validate_axis(axis: int, ndim: int) -> int:
    """Validate an input to axis= keywords"""
    if isinstance(axis, (tuple, list)):
        return tuple(validate_axis(ax, ndim) for ax in axis)
    if not isinstance(axis, numbers.Integral):
        raise TypeError(f"Axis value must be an integer, got {axis}")
    if axis < -ndim or axis >= ndim:
        raise AxisError(f"Axis {axis} is out of bounds for array of dimension {ndim}")
    if axis < 0:
        axis += ndim
    return axis
