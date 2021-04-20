import hypothesis.strategies as st
import numpy as np
import pytest
from hypothesis import note

from .. import assert_identical
from ..test_units import assert_units_equal
from . import base
from .base import utils

pint = pytest.importorskip("pint")
unit_registry = pint.UnitRegistry(force_ndarray_like=True)
Quantity = unit_registry.Quantity

pytestmark = [pytest.mark.filterwarnings("error::pint.UnitStrippedWarning")]


all_units = st.sampled_from(["m", "mm", "s", "dimensionless"])


class TestVariableReduceMethods(base.VariableReduceTests):
    @st.composite
    @staticmethod
    def create(draw, op):
        if op in ("cumprod",):
            units = st.just("dimensionless")
        else:
            units = all_units

        return Quantity(draw(utils.numpy_array), draw(units))

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

            without_units = obj.copy(data=obj.data.magnitude)
            expected = getattr(without_units, op)(*args, **kwargs)

            func_name = f"nan{op}" if obj.dtype.kind in "fc" else op
            func = getattr(np, func_name, getattr(np, op))
            func_kwargs = kwargs.copy()
            dim = func_kwargs.pop("dim", None)
            axis = utils.valid_axes_from_dims(obj.dims, dim)
            func_kwargs["axis"] = axis
            with utils.suppress_warning(RuntimeWarning):
                result = func(obj.data, *args, **func_kwargs)
            units = getattr(result, "units", None)
            if units is not None:
                expected = expected.copy(data=Quantity(expected.data, units))

            note(f"expected:\n{expected}")

            assert_units_equal(actual, expected)
            assert_identical(actual, expected)
