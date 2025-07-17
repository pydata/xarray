.. currentmodule:: xarray

Universal functions
===================

These functions are equivalent to their NumPy versions, but for xarray
objects backed by non-NumPy array types (e.g. ``cupy``, ``sparse``, or ``jax``),
they will ensure that the computation is dispatched to the appropriate
backend. You can find them in the ``xarray.ufuncs`` module:

.. autosummary::
   :toctree: ../generated/

   ufuncs.abs
   ufuncs.absolute
   ufuncs.acos
   ufuncs.acosh
   ufuncs.arccos
   ufuncs.arccosh
   ufuncs.arcsin
   ufuncs.arcsinh
   ufuncs.arctan
   ufuncs.arctanh
   ufuncs.asin
   ufuncs.asinh
   ufuncs.atan
   ufuncs.atanh
   ufuncs.bitwise_count
   ufuncs.bitwise_invert
   ufuncs.bitwise_not
   ufuncs.cbrt
   ufuncs.ceil
   ufuncs.conj
   ufuncs.conjugate
   ufuncs.cos
   ufuncs.cosh
   ufuncs.deg2rad
   ufuncs.degrees
   ufuncs.exp
   ufuncs.exp2
   ufuncs.expm1
   ufuncs.fabs
   ufuncs.floor
   ufuncs.invert
   ufuncs.isfinite
   ufuncs.isinf
   ufuncs.isnan
   ufuncs.isnat
   ufuncs.log
   ufuncs.log10
   ufuncs.log1p
   ufuncs.log2
   ufuncs.logical_not
   ufuncs.negative
   ufuncs.positive
   ufuncs.rad2deg
   ufuncs.radians
   ufuncs.reciprocal
   ufuncs.rint
   ufuncs.sign
   ufuncs.signbit
   ufuncs.sin
   ufuncs.sinh
   ufuncs.spacing
   ufuncs.sqrt
   ufuncs.square
   ufuncs.tan
   ufuncs.tanh
   ufuncs.trunc
   ufuncs.add
   ufuncs.arctan2
   ufuncs.atan2
   ufuncs.bitwise_and
   ufuncs.bitwise_left_shift
   ufuncs.bitwise_or
   ufuncs.bitwise_right_shift
   ufuncs.bitwise_xor
   ufuncs.copysign
   ufuncs.divide
   ufuncs.equal
   ufuncs.float_power
   ufuncs.floor_divide
   ufuncs.fmax
   ufuncs.fmin
   ufuncs.fmod
   ufuncs.gcd
   ufuncs.greater
   ufuncs.greater_equal
   ufuncs.heaviside
   ufuncs.hypot
   ufuncs.lcm
   ufuncs.ldexp
   ufuncs.left_shift
   ufuncs.less
   ufuncs.less_equal
   ufuncs.logaddexp
   ufuncs.logaddexp2
   ufuncs.logical_and
   ufuncs.logical_or
   ufuncs.logical_xor
   ufuncs.maximum
   ufuncs.minimum
   ufuncs.mod
   ufuncs.multiply
   ufuncs.nextafter
   ufuncs.not_equal
   ufuncs.pow
   ufuncs.power
   ufuncs.remainder
   ufuncs.right_shift
   ufuncs.subtract
   ufuncs.true_divide
   ufuncs.angle
   ufuncs.isreal
   ufuncs.iscomplex
