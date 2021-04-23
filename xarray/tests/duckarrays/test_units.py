import pytest

pytest.importorskip("hypothesis")

import hypothesis.strategies as st
import numpy as np
from hypothesis import note

from .. import assert_identical
from ..test_units import assert_units_equal, attach_units, strip_units
from . import base
from .base import strategies, utils

pint = pytest.importorskip("pint")
unit_registry = pint.UnitRegistry(force_ndarray_like=True)
Quantity = unit_registry.Quantity

pytestmark = [pytest.mark.filterwarnings("error::pint.UnitStrippedWarning")]


all_units = st.sampled_from(["m", "mm", "s", "dimensionless"])


def apply_func(op, var, *args, **kwargs):
    dim = kwargs.pop("dim", None)
    if dim in var.dims:
        axis = utils.valid_axes_from_dims(var.dims, dim)
    else:
        axis = None
    kwargs["axis"] = axis

    arr = var.data
    func_name = f"nan{op}" if arr.dtype.kind in "fc" else op
    func = getattr(np, func_name, getattr(np, op))
    with utils.suppress_warning(RuntimeWarning):
        result = func(arr, *args, **kwargs)

    return getattr(result, "units", None)


@pytest.mark.apply_marks(
    {
        "test_reduce": {
            "[prod]": pytest.mark.skip(reason="inconsistent implementation in pint"),
        }
    }
)
class TestVariableReduceMethods(base.VariableReduceTests):
    @st.composite
    @staticmethod
    def create(draw, op, shape):
        if op in ("cumprod",):
            units = st.just("dimensionless")
        else:
            units = all_units

        return Quantity(draw(strategies.numpy_array(shape)), draw(units))

    def compute_expected(self, obj, op, *args, **kwargs):
        without_units = strip_units(obj)
        expected = getattr(without_units, op)(*args, **kwargs)

        units = apply_func(op, obj, *args, **kwargs)
        return attach_units(expected, {None: units})

    def check_reduce(self, obj, op, *args, **kwargs):
        if (
            op in ("cumprod",)
            and obj.data.size > 1
            and obj.data.units != unit_registry.dimensionless
        ):
            with pytest.raises(pint.DimensionalityError):
                getattr(obj, op)(*args, **kwargs)
        else:
            actual = getattr(obj, op)(*args, **kwargs)

            note(f"actual:\n{actual}")

            expected = self.compute_expected(obj, op, *args, **kwargs)

            note(f"expected:\n{expected}")

            assert_units_equal(actual, expected)
            assert_identical(actual, expected)


@pytest.mark.apply_marks(
    {
        "test_reduce": {
            "[prod]": pytest.mark.skip(reason="inconsistent implementation in pint"),
        }
    }
)
class TestDataArrayReduceMethods(base.DataArrayReduceTests):
    @st.composite
    @staticmethod
    def create(draw, op, shape):
        if op in ("cumprod",):
            units = st.just("dimensionless")
        else:
            units = all_units

        return Quantity(draw(strategies.numpy_array(shape)), draw(units))

    def compute_expected(self, obj, op, *args, **kwargs):
        without_units = strip_units(obj)
        expected = getattr(without_units, op)(*args, **kwargs)
        units = apply_func(op, obj.variable, *args, **kwargs)

        return attach_units(expected, {obj.name: units})

    def check_reduce(self, obj, op, *args, **kwargs):
        if (
            op in ("cumprod",)
            and obj.data.size > 1
            and obj.data.units != unit_registry.dimensionless
        ):
            with pytest.raises(pint.DimensionalityError):
                getattr(obj, op)(*args, **kwargs)
        else:
            actual = getattr(obj, op)(*args, **kwargs)

            note(f"actual:\n{actual}")

            expected = self.compute_expected(obj, op, *args, **kwargs)

            note(f"expected:\n{expected}")

            assert_units_equal(actual, expected)
            assert_identical(actual, expected)


@pytest.mark.apply_marks(
    {
        "test_reduce": {
            "[prod]": pytest.mark.skip(reason="inconsistent implementation in pint"),
        }
    }
)
class TestDatasetReduceMethods(base.DatasetReduceTests):
    @st.composite
    @staticmethod
    def create(draw, op, shape):
        if op in ("cumprod",):
            units = st.just("dimensionless")
        else:
            units = all_units

        return Quantity(draw(strategies.numpy_array(shape)), draw(units))

    def compute_expected(self, obj, op, *args, **kwargs):
        without_units = strip_units(obj)
        result_without_units = getattr(without_units, op)(*args, **kwargs)
        units = {
            name: apply_func(op, var, *args, **kwargs)
            for name, var in obj.variables.items()
        }
        attached = attach_units(result_without_units, units)
        return attached

    def check_reduce(self, obj, op, *args, **kwargs):
        if op in ("cumprod",) and any(
            var.size > 1
            and getattr(var.data, "units", None) != unit_registry.dimensionless
            for var in obj.variables.values()
        ):
            with pytest.raises(pint.DimensionalityError):
                getattr(obj, op)(*args, **kwargs)
        else:
            actual = getattr(obj, op)(*args, **kwargs)

            note(f"actual:\n{actual}")

            expected = self.compute_expected(obj, op, *args, **kwargs)

            note(f"expected:\n{expected}")

            assert_units_equal(actual, expected)
            assert_identical(actual, expected)
