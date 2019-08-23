from distutils.version import LooseVersion

import dask.array as da
import numpy as np
from dask import __version__ as dask_version

try:
    blockwise = da.blockwise
except AttributeError:
    blockwise = da.atop


try:
    from dask.array import isin
except ImportError:  # pragma: no cover
    # Copied from dask v0.17.3.
    # Used under the terms of Dask's license, see licenses/DASK_LICENSE.

    def _isin_kernel(element, test_elements, assume_unique=False):
        values = np.in1d(element.ravel(), test_elements, assume_unique=assume_unique)
        return values.reshape(element.shape + (1,) * test_elements.ndim)

    def isin(element, test_elements, assume_unique=False, invert=False):
        element = da.asarray(element)
        test_elements = da.asarray(test_elements)
        element_axes = tuple(range(element.ndim))
        test_axes = tuple(i + element.ndim for i in range(test_elements.ndim))
        mapped = blockwise(
            _isin_kernel,
            element_axes + test_axes,
            element,
            element_axes,
            test_elements,
            test_axes,
            adjust_chunks={axis: lambda _: 1 for axis in test_axes},
            dtype=bool,
            assume_unique=assume_unique,
        )
        result = mapped.any(axis=test_axes)
        if invert:
            result = ~result
        return result


if LooseVersion(dask_version) > LooseVersion("0.19.2"):
    gradient = da.gradient

else:  # pragma: no cover
    # Copied from dask v0.19.2
    # Used under the terms of Dask's license, see licenses/DASK_LICENSE.
    import math
    from numbers import Integral, Real

    try:
        AxisError = np.AxisError
    except AttributeError:
        try:
            np.array([0]).sum(axis=5)
        except Exception as e:
            AxisError = type(e)

    def validate_axis(axis, ndim):
        """ Validate an input to axis= keywords """
        if isinstance(axis, (tuple, list)):
            return tuple(validate_axis(ax, ndim) for ax in axis)
        if not isinstance(axis, Integral):
            raise TypeError("Axis value must be an integer, got %s" % axis)
        if axis < -ndim or axis >= ndim:
            raise AxisError(
                "Axis %d is out of bounds for array of dimension " "%d" % (axis, ndim)
            )
        if axis < 0:
            axis += ndim
        return axis

    def _gradient_kernel(x, block_id, coord, axis, array_locs, grad_kwargs):
        """
        x: nd-array
            array of one block
        coord: 1d-array or scalar
            coordinate along which the gradient is computed.
        axis: int
            axis along which the gradient is computed
        array_locs:
            actual location along axis. None if coordinate is scalar
        grad_kwargs:
            keyword to be passed to np.gradient
        """
        block_loc = block_id[axis]
        if array_locs is not None:
            coord = coord[array_locs[0][block_loc] : array_locs[1][block_loc]]
        grad = np.gradient(x, coord, axis=axis, **grad_kwargs)
        return grad

    def gradient(f, *varargs, axis=None, **kwargs):
        f = da.asarray(f)

        kwargs["edge_order"] = math.ceil(kwargs.get("edge_order", 1))
        if kwargs["edge_order"] > 2:
            raise ValueError("edge_order must be less than or equal to 2.")

        drop_result_list = False
        if axis is None:
            axis = tuple(range(f.ndim))
        elif isinstance(axis, Integral):
            drop_result_list = True
            axis = (axis,)

        axis = validate_axis(axis, f.ndim)

        if len(axis) != len(set(axis)):
            raise ValueError("duplicate axes not allowed")

        axis = tuple(ax % f.ndim for ax in axis)

        if varargs == ():
            varargs = (1,)
        if len(varargs) == 1:
            varargs = len(axis) * varargs
        if len(varargs) != len(axis):
            raise TypeError(
                "Spacing must either be a single scalar, or a scalar / "
                "1d-array per axis"
            )

        if issubclass(f.dtype.type, (np.bool8, Integral)):
            f = f.astype(float)
        elif issubclass(f.dtype.type, Real) and f.dtype.itemsize < 4:
            f = f.astype(float)

        results = []
        for i, ax in enumerate(axis):
            for c in f.chunks[ax]:
                if np.min(c) < kwargs["edge_order"] + 1:
                    raise ValueError(
                        "Chunk size must be larger than edge_order + 1. "
                        "Minimum chunk for aixs {} is {}. Rechunk to "
                        "proceed.".format(np.min(c), ax)
                    )

            if np.isscalar(varargs[i]):
                array_locs = None
            else:
                if isinstance(varargs[i], da.Array):
                    raise NotImplementedError(
                        "dask array coordinated is not supported."
                    )
                # coordinate position for each block taking overlap into
                # account
                chunk = np.array(f.chunks[ax])
                array_loc_stop = np.cumsum(chunk) + 1
                array_loc_start = array_loc_stop - chunk - 2
                array_loc_stop[-1] -= 1
                array_loc_start[0] = 0
                array_locs = (array_loc_start, array_loc_stop)

            results.append(
                f.map_overlap(
                    _gradient_kernel,
                    dtype=f.dtype,
                    depth={j: 1 if j == ax else 0 for j in range(f.ndim)},
                    boundary="none",
                    coord=varargs[i],
                    axis=ax,
                    array_locs=array_locs,
                    grad_kwargs=kwargs,
                )
            )

        if drop_result_list:
            results = results[0]

        return results
