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
    from xarray.namedarray._typing import (
        _Capabilities,
        _DataTypes,
        _DefaultDataTypes,
        _Device,
    )


class __array_namespace_info__:
    def capabilities(self) -> _Capabilities:
        """
        Returns a dictionary of array library capabilities.

        Examples
        --------
        >>> __array_namespace_info__().capabilities()
        {'boolean indexing': True, 'data-dependent shapes': True, 'max dimensions': None}
        """
        from xarray.namedarray._array_api._utils import _maybe_default_namespace

        xp = _maybe_default_namespace()
        info = xp.__array_namespace_info__()

        # Default capabilities:
        out = {
            "boolean indexing": False,
            "data-dependent shapes": False,
            "max dimensions": None,
        }

        # Update with the default namespace (guarantees correct format):
        out.update(info.capabilities())

        return out

    def default_device(self) -> _Device:
        """
        Returns the default device.

        Examples
        --------
        >>> __array_namespace_info__().default_device()
        'cpu'
        """
        from xarray.namedarray._array_api._utils import _maybe_default_namespace

        xp = _maybe_default_namespace()
        info = xp.__array_namespace_info__()
        return info.default_device()

    def default_dtypes(
        self,
        *,
        device: _Device | None = None,  # TODO: not used?
    ) -> _DefaultDataTypes:
        """
        Returns a dictionary containing default data types.

        Examples
        --------
        >>> __array_namespace_info__().default_dtypes()
        {'real floating': <class 'numpy.float64'>, 'complex floating': <class 'numpy.complex128'>, 'integral': <class 'numpy.int64'>, 'indexing': <class 'numpy.int64'>}

        """
        return {
            "real floating": float64,
            "complex floating": complex128,
            "integral": int64,
            "indexing": int64,
        }

    def dtypes(
        self,
        *,
        device: _Device | None = None,
        kind: str | tuple[str, ...] | None = None,
    ) -> _DataTypes:
        """
        Returns a dictionary of supported Array API data types.

        Examples
        --------
        >>> __array_namespace_info__().dtypes()
        {'bool': <class 'numpy.bool'>, 'int8': <class 'numpy.int8'>, 'int16': <class 'numpy.int16'>, 'int32': <class 'numpy.int32'>, 'int64': <class 'numpy.int64'>, 'uint8': <class 'numpy.uint8'>, 'uint16': <class 'numpy.uint16'>, 'uint32': <class 'numpy.uint32'>, 'uint64': <class 'numpy.uint64'>, 'float32': <class 'numpy.float32'>, 'float64': <class 'numpy.float64'>, 'complex64': <class 'numpy.complex64'>, 'complex128': <class 'numpy.complex128'>}
        """
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
                res.update(self.dtypes(kind=k))
            return res
        raise ValueError(f"unsupported kind: {kind!r}")

    def devices(self) -> list[_Device]:
        """
        Returns a list of supported devices which are available at runtime.

        Examples
        --------
        >>> __array_namespace_info__().devices()
        ['cpu']
        """
        return [self.default_device()]


if __name__ == "__main__":
    import doctest

    doctest.testmod()
