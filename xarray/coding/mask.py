from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, ClassVar

from xarray.compat.array_api_compat import get_array_namespace

if TYPE_CHECKING:
    import numpy as np

    from xarray.coding.core import CoderKind
    from xarray.core.variable import Variable


class MaskValueDefaults(StrEnum):
    ValueMax = "ValueMax"
    ValueMin = "ValueMin"


ValueMin = MaskValueDefaults.ValueMin
ValueMax = MaskValueDefaults.ValueMax

ValueType = int | float | bool
DefaultValueType = ValueType | MaskValueDefaults


def _mask_valid_min(values: np.ndarray, valid_min: ValueType) -> np.ndarray:
    return values < valid_min


def _mask_valid_max(values: np.ndarray, valid_max: ValueType) -> np.ndarray:
    return values > valid_max


def _mask_valid_range(
    values: np.ndarray, range: np.ndarray | tuple[ValueType, ValueType]
) -> np.ndarray:
    return _mask_valid_min(values, range[0]) | _mask_valid_max(values, range[1])


def _mask_value(values: np.ndarray, value: ValueType) -> np.ndarray:
    return values == value


_masking_functions = {
    "missing_value": _mask_value,
    "valid_min": _mask_valid_min,
    "valid_max": _mask_valid_max,
    "valid_range": _mask_valid_range,
}


@dataclass
class CfMarrayCoder:
    kind: ClassVar[CoderKind] = "variable"
    missing_value_keys: tuple[str, ...] = (
        "missing_value",
        "valid_min",
        "valid_max",
        "valid_range",
    )
    defaults: tuple[ValueType | tuple[ValueType, ValueType], ...] = (
        ValueMin,
        ValueMin,
        ValueMax,
        (ValueMin, ValueMax),
    )

    def decode(self, var: Variable) -> Variable:
        import marray

        masking_attrs = {
            key: var.attrs[key] for key in self.missing_value_keys if key in var.attrs
        }
        if not masking_attrs:
            return var

        method, value = next(iter(masking_attrs.items()))
        data = var._data

        xp = get_array_namespace(data)
        mxp = marray.masked_namespace(xp)

        mask = _masking_functions[method](data, value)

        masked_data = mxp.asarray(data, mask=mask)
        print(masked_data)

        masked_variable = var.copy(data=masked_data)
        masked_variable.encoding["mask"] = masking_attrs
        for key in masking_attrs:
            del masked_variable.attrs[key]

        return masked_variable

    def encode(self, var: Variable) -> Variable:
        # TODO: how does a user control the encoding per variable? Usually that's the
        # encoding dict on the object or passed to the to_* method. However, that means
        # that the coding pipeline needs a way to be configured, and this coder to have a
        # way to store this.
        pass
