from __future__ import annotations

from typing import TYPE_CHECKING

from xarray.namedarray._array_api._dtypes import (
    bool,
    complex64,
    complex128,
    float32,
    float64,
    int8,
    int16,
    int32,
    int64,
    uint8,
    uint16,
    uint32,
    uint64,
)

if TYPE_CHECKING:
    from types import ModuleType

    from xarray.namedarray._typing import (
        _Capabilities,
        _DataTypes,
        _DefaultDataTypes,
        _Device,
    )


def __array_namespace_info__() -> ModuleType:
    import xarray.namedarray._array_api._info

    return xarray.namedarray._array_api._info


def capabilities() -> _Capabilities:
    return {
        "boolean indexing": False,
        "data-dependent shapes": False,
        "max rank": int | None,
    }


def default_device() -> _Device:
    from xarray.namedarray._array_api._utils import _maybe_default_namespace

    xp = _maybe_default_namespace()
    info = xp.__array_namespace_info__()
    return info.default_device()


def default_dtypes(
    *,
    device: _Device | None = None,
) -> _DefaultDataTypes:
    return {
        "real floating": float64,
        "complex floating": complex128,
        "integral": int64,
        "indexing": int64,
    }


def dtypes(
    *,
    device: _Device | None = None,
    kind: str | tuple[str, ...] | None = None,
) -> _DataTypes:
    if kind is None:
        return {
            "bool": bool,
            "int8": int8,
            "int16": int16,
            "int32": int32,
            "int64": int64,
            "uint8": uint8,
            "uint16": uint16,
            "uint32": uint32,
            "uint64": uint64,
            "float32": float32,
            "float64": float64,
            "complex64": complex64,
            "complex128": complex128,
        }
    if kind == "bool":
        return {"bool": bool}
    if kind == "signed integer":
        return {
            "int8": int8,
            "int16": int16,
            "int32": int32,
            "int64": int64,
        }
    if kind == "unsigned integer":
        return {
            "uint8": uint8,
            "uint16": uint16,
            "uint32": uint32,
            "uint64": uint64,
        }
    if kind == "integral":
        return {
            "int8": int8,
            "int16": int16,
            "int32": int32,
            "int64": int64,
            "uint8": uint8,
            "uint16": uint16,
            "uint32": uint32,
            "uint64": uint64,
        }
    if kind == "real floating":
        return {
            "float32": float32,
            "float64": float64,
        }
    if kind == "complex floating":
        return {
            "complex64": complex64,
            "complex128": complex128,
        }
    if kind == "numeric":
        return {
            "int8": int8,
            "int16": int16,
            "int32": int32,
            "int64": int64,
            "uint8": uint8,
            "uint16": uint16,
            "uint32": uint32,
            "uint64": uint64,
            "float32": float32,
            "float64": float64,
            "complex64": complex64,
            "complex128": complex128,
        }
    if isinstance(kind, tuple):
        res: _DataTypes = {}
        for k in kind:
            res.update(dtypes(kind=k))
        return res
    raise ValueError(f"unsupported kind: {kind!r}")


def devices() -> list[_Device]:
    return [default_device()]


__all__ = [
    "capabilities",
    "default_device",
    "default_dtypes",
    "devices",
    "dtypes",
]
