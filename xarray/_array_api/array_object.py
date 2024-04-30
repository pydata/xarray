# Vendored from https://github.com/data-apis/array-api/pull/589
from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, TypeVar

if TYPE_CHECKING:
    from xarray._array_api._types import (
        Any,
        Enum,
        PyCapsule,
        ellipsis,
    )
    from xarray._array_api._types import (
        device as Device,
    )
    from xarray._array_api.data_types import DType

array = TypeVar("array", bound="Array")
# NOTE: when working with py3.11+ this can be ``typing.array``.


class Array(Protocol):
    def __init__(self) -> None:
        """Initialize the attributes for the array object class."""
        ...

    @property
    def dtype(self) -> DType:
        """
        Data type of the array elements.

        Returns
        -------
        out: dtype
            array data type.
        """
        ...

    @property
    def device(self) -> Device:  # type: ignore[type-var]
        """
        Hardware device the array data resides on.

        Returns
        -------
        out: device
            a ``device`` object (see :ref:`device-support`).
        """
        ...

    @property
    def mT(self: array) -> array:
        """
        Transpose of a matrix (or a stack of matrices).

        If an array instance has fewer than two dimensions, an error should be raised.

        Returns
        -------
        out: array
            array whose last two dimensions (axes) are permuted in reverse order relative to original array (i.e., for an array instance having shape ``(..., M, N)``, the returned array must have shape ``(..., N, M)``). The returned array must have the same data type as the original array.
        """
        ...

    @property
    def ndim(self) -> int:
        """
        Number of array dimensions (axes).

        Returns
        -------
        out: int
            number of array dimensions (axes).
        """
        ...

    @property
    def shape(self) -> tuple[int | None, ...]:
        """
        Array dimensions.

        Returns
        -------
        out: Tuple[Optional[int], ...]
            array dimensions. An array dimension must be ``None`` if and only if a dimension is unknown.


        .. note::
           For array libraries having graph-based computational models, array dimensions may be unknown due to data-dependent operations (e.g., boolean indexing; ``A[:, B > 0]``) and thus cannot be statically resolved without knowing array contents.

        .. note::
           The returned value should be a tuple; however, where warranted, an array library may choose to return a custom shape object. If an array library returns a custom shape object, the object must be immutable, must support indexing for dimension retrieval, and must behave similarly to a tuple.
        """
        ...

    @property
    def size(self) -> int | None:
        """
        Number of elements in an array.

        .. note::
           This must equal the product of the array's dimensions.

        Returns
        -------
        out: Optional[int]
            number of elements in an array. The returned value must be ``None`` if and only if one or more array dimensions are unknown.


        .. note::
           For array libraries having graph-based computational models, an array may have unknown dimensions due to data-dependent operations.
        """
        ...

    @property
    def T(self: array) -> array:
        """
        Transpose of the array.

        The array instance must be two-dimensional. If the array instance is not two-dimensional, an error should be raised.

        Returns
        -------
        out: array
            two-dimensional array whose first and last dimensions (axes) are permuted in reverse order relative to original array. The returned array must have the same data type as the original array.


        .. note::
           Limiting the transpose to two-dimensional arrays (matrices) deviates from the NumPy et al practice of reversing all axes for arrays having more than two-dimensions. This is intentional, as reversing all axes was found to be problematic (e.g., conflicting with the mathematical definition of a transpose which is limited to matrices; not operating on batches of matrices; et cetera). In order to reverse all axes, one is recommended to use the functional ``permute_dims`` interface found in this specification.
        """
        ...

    def __abs__(self: array, /) -> array:
        """
        Calculates the absolute value for each element of an array instance.

        For real-valued input arrays, the element-wise result has the same magnitude as the respective element in ``x`` but has positive sign.

        .. note::
           For signed integer data types, the absolute value of the minimum representable integer is implementation-dependent.

        Parameters
        ----------
        self: array
            array instance. Should have a numeric data type.

        Returns
        -------
        out: array
            an array containing the element-wise absolute value. If ``self`` has a real-valued data type, the returned array must have the same data type as ``self``. If ``self`` has a complex floating-point data type, the returned arrayed must have a real-valued floating-point data type whose precision matches the precision of ``self`` (e.g., if ``self`` is ``complex128``, then the returned array must have a ``float64`` data type).

        Notes
        -----

        .. note::
           Element-wise results, including special cases, must equal the results returned by the equivalent element-wise function :func:`~array_api.abs`.

        .. versionchanged:: 2022.12
            Added complex data type support.
        """
        ...

    def __add__(self: array, other: int | float | array, /) -> array:
        """
        Calculates the sum for each element of an array instance with the respective element of the array ``other``.

        Parameters
        ----------
        self: array
            array instance (augend array). Should have a numeric data type.
        other: Union[int, float, array]
            addend array. Must be compatible with ``self`` (see :ref:`broadcasting`). Should have a numeric data type.

        Returns
        -------
        out: array
            an array containing the element-wise sums. The returned array must have a data type determined by :ref:`type-promotion`.

        Notes
        -----

        .. note::
           Element-wise results, including special cases, must equal the results returned by the equivalent element-wise function :func:`~array_api.add`.

        .. versionchanged:: 2022.12
            Added complex data type support.
        """
        ...

    def __and__(self: array, other: int | bool | array, /) -> array:
        """
        Evaluates ``self_i & other_i`` for each element of an array instance with the respective element of the array ``other``.

        Parameters
        ----------
        self: array
            array instance. Should have an integer or boolean data type.
        other: Union[int, bool, array]
            other array. Must be compatible with ``self`` (see :ref:`broadcasting`). Should have an integer or boolean data type.

        Returns
        -------
        out: array
            an array containing the element-wise results. The returned array must have a data type determined by :ref:`type-promotion`.


        .. note::
           Element-wise results must equal the results returned by the equivalent element-wise function :func:`~array_api.bitwise_and`.
        """
        ...

    def __array_namespace__(self, /, *, api_version: str | None = None) -> Any:
        """
        Returns an object that has all the array API functions on it.

        Parameters
        ----------
        self: array
            array instance.
        api_version: Optional[str]
            string representing the version of the array API specification to be returned, in ``'YYYY.MM'`` form, for example, ``'2020.10'``. If it is ``None``, it should return the namespace corresponding to latest version of the array API specification.  If the given version is invalid or not implemented for the given module, an error should be raised. Default: ``None``.

        Returns
        -------
        out: Any
            an object representing the array API namespace. It should have every top-level function defined in the specification as an attribute. It may contain other public names as well, but it is recommended to only include those names that are part of the specification.
        """
        ...

    def __bool__(self, /) -> bool:
        """
        Converts a zero-dimensional array to a Python ``bool`` object.

        Parameters
        ----------
        self: array
            zero-dimensional array instance.

        Returns
        -------
        out: bool
            a Python ``bool`` object representing the single element of the array.

        Notes
        -----

        **Special cases**

        For real-valued floating-point operands,

        - If ``self`` is ``NaN``, the result is ``True``.
        - If ``self`` is either ``+infinity`` or ``-infinity``, the result is ``True``.
        - If ``self`` is either ``+0`` or ``-0``, the result is ``False``.

        For complex floating-point operands, special cases must be handled as if the operation is implemented as the logical AND of ``bool(real(self))`` and ``bool(imag(self))``.

        .. versionchanged:: 2022.12
            Added boolean and complex data type support.
        """
        ...

    def __complex__(self, /) -> complex:
        """
        Converts a zero-dimensional array to a Python ``complex`` object.

        Parameters
        ----------
        self: array
            zero-dimensional array instance.

        Returns
        -------
        out: complex
            a Python ``complex`` object representing the single element of the array instance.

        Notes
        -----

        **Special cases**

        For boolean operands,

        - If ``self`` is ``True``, the result is ``1+0j``.
        - If ``self`` is ``False``, the result is ``0+0j``.

        For real-valued floating-point operands,

        - If ``self`` is ``NaN``, the result is ``NaN + NaN j``.
        - If ``self`` is ``+infinity``, the result is ``+infinity + 0j``.
        - If ``self`` is ``-infinity``, the result is ``-infinity + 0j``.
        - If ``self`` is a finite number, the result is ``self + 0j``.

        .. versionadded:: 2022.12
        """
        ...

    def __dlpack__(
        self, /, *, stream: int | Any | None = None
    ) -> PyCapsule:  # type: ignore[type-var]
        """
        Exports the array for consumption by :func:`~array_api.from_dlpack` as a DLPack capsule.

        Parameters
        ----------
        self: array
            array instance.
        stream: Optional[Union[int, Any]]
            for CUDA and ROCm, a Python integer representing a pointer to a stream, on devices that support streams. ``stream`` is provided by the consumer to the producer to instruct the producer to ensure that operations can safely be performed on the array (e.g., by inserting a dependency between streams via "wait for event"). The pointer must be a positive integer or ``-1``. If ``stream`` is ``-1``, the value may be used by the consumer to signal "producer must not perform any synchronization". The ownership of the stream stays with the consumer. On CPU and other device types without streams, only ``None`` is accepted.

            For other device types which do have a stream, queue or similar synchronization mechanism, the most appropriate type to use for ``stream`` is not yet determined. E.g., for SYCL one may want to use an object containing an in-order ``cl::sycl::queue``. This is allowed when libraries agree on such a convention, and may be standardized in a future version of this API standard.


        .. note::
            Support for a ``stream`` value other than ``None`` is optional and implementation-dependent.


        Device-specific notes:


        .. admonition:: CUDA
            :class: note

            - ``None``: producer must assume the legacy default stream (default).
            - ``1``: the legacy default stream.
            - ``2``: the per-thread default stream.
            - ``> 2``: stream number represented as a Python integer.
            - ``0`` is disallowed due to its ambiguity: ``0`` could mean either ``None``, ``1``, or ``2``.


        .. admonition:: ROCm
            :class: note

            - ``None``: producer must assume the legacy default stream (default).
            - ``0``: the default stream.
            - ``> 2``: stream number represented as a Python integer.
            - Using ``1`` and ``2`` is not supported.


        .. admonition:: Tip
            :class: important

            It is recommended that implementers explicitly handle streams. If
            they use the legacy default stream, specifying ``1`` (CUDA) or ``0``
            (ROCm) is preferred. ``None`` is a safe default for developers who do
            not want to think about stream handling at all, potentially at the
            cost of more synchronization than necessary.

        Returns
        -------
        capsule: PyCapsule
            a DLPack capsule for the array. See :ref:`data-interchange` for details.

        Raises
        ------
        BufferError
            Implementations should raise ``BufferError`` when the data cannot
            be exported as DLPack (e.g., incompatible dtype or strides). Other
            errors are raised when export fails for other reasons (e.g., incorrect
            arguments passed or out of memory).

        Notes
        -----

        .. versionchanged:: 2022.12
            Added BufferError.
        """
        ...

    def __dlpack_device__(self, /) -> tuple[Enum, int]:
        """
        Returns device type and device ID in DLPack format. Meant for use within :func:`~array_api.from_dlpack`.

        Parameters
        ----------
        self: array
            array instance.

        Returns
        -------
        device: Tuple[Enum, int]
            a tuple ``(device_type, device_id)`` in DLPack format. Valid device type enum members are:

            ::

              CPU = 1
              CUDA = 2
              CPU_PINNED = 3
              OPENCL = 4
              VULKAN = 7
              METAL = 8
              VPI = 9
              ROCM = 10
        """
        ...

    # Note that __eq__ returns an array while `object.__eq__` returns a bool.
    # Hence Mypy will complain that this violates the Liskov substitution
    # principle - ignore that.
    def __eq__(self: array, other: int | float | bool | array, /) -> array:  # type: ignore[override]
        r"""
        Computes the truth value of ``self_i == other_i`` for each element of an array instance with the respective element of the array ``other``.

        Parameters
        ----------
        self: array
            array instance. May have any data type.
        other: Union[int, float, bool, array]
            other array. Must be compatible with ``self`` (see :ref:`broadcasting`). May have any data type.

        Returns
        -------
        out: array
            an array containing the element-wise results. The returned array must have a data type of ``bool``.


        .. note::
           Element-wise results, including special cases, must equal the results returned by the equivalent element-wise function :func:`~array_api.equal`.
        """
        ...

    def __float__(self, /) -> float:
        """
        Converts a zero-dimensional array to a Python ``float`` object.

        .. note::
           Casting integer values outside the representable bounds of Python's float type is not specified and is implementation-dependent.

        Parameters
        ----------
        self: array
            zero-dimensional array instance. Should have a real-valued or boolean data type. If ``self`` has a complex floating-point data type, the function must raise a ``TypeError``.

        Returns
        -------
        out: float
            a Python ``float`` object representing the single element of the array instance.

        Notes
        -----

        **Special cases**

        For boolean operands,

        - If ``self`` is ``True``, the result is ``1``.
        - If ``self`` is ``False``, the result is ``0``.

        .. versionchanged:: 2022.12
            Added boolean and complex data type support.
        """
        ...

    def __floordiv__(self: array, other: int | float | array, /) -> array:
        """
        Evaluates ``self_i // other_i`` for each element of an array instance with the respective element of the array ``other``.

        .. note::
           For input arrays which promote to an integer data type, the result of division by zero is unspecified and thus implementation-defined.

        Parameters
        ----------
        self: array
            array instance. Should have a real-valued data type.
        other: Union[int, float, array]
            other array. Must be compatible with ``self`` (see :ref:`broadcasting`). Should have a real-valued data type.

        Returns
        -------
        out: array
            an array containing the element-wise results. The returned array must have a data type determined by :ref:`type-promotion`.


        .. note::
           Element-wise results, including special cases, must equal the results returned by the equivalent element-wise function :func:`~array_api.floor_divide`.
        """
        ...

    def __ge__(self: array, other: int | float | array, /) -> array:
        """
        Computes the truth value of ``self_i >= other_i`` for each element of an array instance with the respective element of the array ``other``.

        .. note::
           For backward compatibility, conforming implementations may support complex numbers; however, inequality comparison of complex numbers is unspecified and thus implementation-dependent (see :ref:`complex-number-ordering`).

        Parameters
        ----------
        self: array
            array instance. Should have a real-valued data type.
        other: Union[int, float, array]
            other array. Must be compatible with ``self`` (see :ref:`broadcasting`). Should have a real-valued data type.

        Returns
        -------
        out: array
            an array containing the element-wise results. The returned array must have a data type of ``bool``.


        .. note::
           Element-wise results must equal the results returned by the equivalent element-wise function :func:`~array_api.greater_equal`.
        """
        ...

    def __getitem__(
        self: array,
        key: int | slice | ellipsis | tuple[int | slice | ellipsis, ...] | array,
        /,
    ) -> array:
        """
        Returns ``self[key]``.

        Parameters
        ----------
        self: array
            array instance.
        key: Union[int, slice, ellipsis, Tuple[Union[int, slice, ellipsis], ...], array]
            index key.

        Returns
        -------
        out: array
            an array containing the accessed value(s). The returned array must have the same data type as ``self``.
        """
        ...

    def __gt__(self: array, other: int | float | array, /) -> array:
        """
        Computes the truth value of ``self_i > other_i`` for each element of an array instance with the respective element of the array ``other``.

        .. note::
           For backward compatibility, conforming implementations may support complex numbers; however, inequality comparison of complex numbers is unspecified and thus implementation-dependent (see :ref:`complex-number-ordering`).

        Parameters
        ----------
        self: array
            array instance. Should have a real-valued data type.
        other: Union[int, float, array]
            other array. Must be compatible with ``self`` (see :ref:`broadcasting`). Should have a real-valued data type.

        Returns
        -------
        out: array
            an array containing the element-wise results. The returned array must have a data type of ``bool``.


        .. note::
           Element-wise results must equal the results returned by the equivalent element-wise function :func:`~array_api.greater`.
        """
        ...

    def __index__(self, /) -> int:
        """
        Converts a zero-dimensional integer array to a Python ``int`` object.

        .. note::
           This method is called to implement `operator.index() <https://docs.python.org/3/reference/datamodel.html#object.__index__>`_. See also `PEP 357 <https://www.python.org/dev/peps/pep-0357/>`_.

        Parameters
        ----------
        self: array
            zero-dimensional array instance. Should have an integer data type. If ``self`` has a floating-point data type, the function must raise a ``TypeError``.

        Returns
        -------
        out: int
            a Python ``int`` object representing the single element of the array instance.
        """
        ...

    def __int__(self, /) -> int:
        """
        Converts a zero-dimensional array to a Python ``int`` object.

        Parameters
        ----------
        self: array
            zero-dimensional array instance. Should have a real-valued or boolean data type. If ``self`` has a complex floating-point data type, the function must raise a ``TypeError``.

        Returns
        -------
        out: int
            a Python ``int`` object representing the single element of the array instance.

        Notes
        -----

        **Special cases**

        For boolean operands,

        - If ``self`` is ``True``, the result is ``1``.
        - If ``self`` is ``False``, the result is ``0``.

        For floating-point operands,

        - If ``self`` is a finite number, the result is the integer part of ``self``.
        - If ``self`` is ``-0``, the result is ``0``.

        **Raises**

        For floating-point operands,

        - If ``self`` is either ``+infinity`` or ``-infinity``, raise ``OverflowError``.
        - If ``self`` is ``NaN``, raise ``ValueError``.

        .. versionchanged:: 2022.12
            Added boolean and complex data type support.
        """
        ...

    def __invert__(self: array, /) -> array:
        """
        Evaluates ``~self_i`` for each element of an array instance.

        Parameters
        ----------
        self: array
            array instance. Should have an integer or boolean data type.

        Returns
        -------
        out: array
            an array containing the element-wise results. The returned array must have the same data type as `self`.


        .. note::
           Element-wise results must equal the results returned by the equivalent element-wise function :func:`~array_api.bitwise_invert`.
        """
        ...

    def __le__(self: array, other: int | float | array, /) -> array:
        """
        Computes the truth value of ``self_i <= other_i`` for each element of an array instance with the respective element of the array ``other``.

        .. note::
           For backward compatibility, conforming implementations may support complex numbers; however, inequality comparison of complex numbers is unspecified and thus implementation-dependent (see :ref:`complex-number-ordering`).

        Parameters
        ----------
        self: array
            array instance. Should have a real-valued data type.
        other: Union[int, float, array]
            other array. Must be compatible with ``self`` (see :ref:`broadcasting`). Should have a real-valued data type.

        Returns
        -------
        out: array
            an array containing the element-wise results. The returned array must have a data type of ``bool``.


        .. note::
           Element-wise results must equal the results returned by the equivalent element-wise function :func:`~array_api.less_equal`.
        """
        ...

    def __lshift__(self: array, other: int | array, /) -> array:
        """
        Evaluates ``self_i << other_i`` for each element of an array instance with the respective element  of the array ``other``.

        Parameters
        ----------
        self: array
            array instance. Should have an integer data type.
        other: Union[int, array]
            other array. Must be compatible with ``self`` (see :ref:`broadcasting`). Should have an integer data type. Each element must be greater than or equal to ``0``.

        Returns
        -------
        out: array
            an array containing the element-wise results. The returned array must have the same data type as ``self``.


        .. note::
           Element-wise results must equal the results returned by the equivalent element-wise function :func:`~array_api.bitwise_left_shift`.
        """
        ...

    def __lt__(self: array, other: int | float | array, /) -> array:
        """
        Computes the truth value of ``self_i < other_i`` for each element of an array instance with the respective element of the array ``other``.

        .. note::
           For backward compatibility, conforming implementations may support complex numbers; however, inequality comparison of complex numbers is unspecified and thus implementation-dependent (see :ref:`complex-number-ordering`).

        Parameters
        ----------
        self: array
            array instance. Should have a real-valued data type.
        other: Union[int, float, array]
            other array. Must be compatible with ``self`` (see :ref:`broadcasting`). Should have a real-valued data type.

        Returns
        -------
        out: array
            an array containing the element-wise results. The returned array must have a data type of ``bool``.


        .. note::
           Element-wise results must equal the results returned by the equivalent element-wise function :func:`~array_api.less`.
        """
        ...

    def __matmul__(self: array, other: array, /) -> array:
        """
        Computes the matrix product.

        .. note::
           The ``matmul`` function must implement the same semantics as the built-in ``@`` operator (see `PEP 465 <https://www.python.org/dev/peps/pep-0465>`_).

        Parameters
        ----------
        self: array
            array instance. Should have a numeric data type. Must have at least one dimension. If ``self`` is one-dimensional having shape ``(M,)`` and ``other`` has more than one dimension, ``self`` must be promoted to a two-dimensional array by prepending ``1`` to its dimensions (i.e., must have shape ``(1, M)``). After matrix multiplication, the prepended dimensions in the returned array must be removed. If ``self`` has more than one dimension (including after vector-to-matrix promotion), ``shape(self)[:-2]`` must be compatible with ``shape(other)[:-2]`` (after vector-to-matrix promotion) (see :ref:`broadcasting`). If ``self`` has shape ``(..., M, K)``, the innermost two dimensions form matrices on which to perform matrix multiplication.
        other: array
            other array. Should have a numeric data type. Must have at least one dimension. If ``other`` is one-dimensional having shape ``(N,)`` and ``self`` has more than one dimension, ``other`` must be promoted to a two-dimensional array by appending ``1`` to its dimensions (i.e., must have shape ``(N, 1)``). After matrix multiplication, the appended dimensions in the returned array must be removed. If ``other`` has more than one dimension (including after vector-to-matrix promotion), ``shape(other)[:-2]`` must be compatible with ``shape(self)[:-2]`` (after vector-to-matrix promotion) (see :ref:`broadcasting`). If ``other`` has shape ``(..., K, N)``, the innermost two dimensions form matrices on which to perform matrix multiplication.


        .. note::
           If either ``x1`` or ``x2`` has a complex floating-point data type, neither argument must be complex-conjugated or transposed. If conjugation and/or transposition is desired, these operations should be explicitly performed prior to computing the matrix product.

        Returns
        -------
        out: array
            -   if both ``self`` and ``other`` are one-dimensional arrays having shape ``(N,)``, a zero-dimensional array containing the inner product as its only element.
            -   if ``self`` is a two-dimensional array having shape ``(M, K)`` and ``other`` is a two-dimensional array having shape ``(K, N)``, a two-dimensional array containing the `conventional matrix product <https://en.wikipedia.org/wiki/Matrix_multiplication>`_ and having shape ``(M, N)``.
            -   if ``self`` is a one-dimensional array having shape ``(K,)`` and ``other`` is an array having shape ``(..., K, N)``, an array having shape ``(..., N)`` (i.e., prepended dimensions during vector-to-matrix promotion must be removed) and containing the `conventional matrix product <https://en.wikipedia.org/wiki/Matrix_multiplication>`_.
            -   if ``self`` is an array having shape ``(..., M, K)`` and ``other`` is a one-dimensional array having shape ``(K,)``, an array having shape ``(..., M)`` (i.e., appended dimensions during vector-to-matrix promotion must be removed) and containing the `conventional matrix product <https://en.wikipedia.org/wiki/Matrix_multiplication>`_.
            -   if ``self`` is a two-dimensional array having shape ``(M, K)`` and ``other`` is an array having shape ``(..., K, N)``, an array having shape ``(..., M, N)`` and containing the `conventional matrix product <https://en.wikipedia.org/wiki/Matrix_multiplication>`_ for each stacked matrix.
            -   if ``self`` is an array having shape ``(..., M, K)`` and ``other`` is a two-dimensional array having shape ``(K, N)``, an array having shape ``(..., M, N)`` and containing the `conventional matrix product <https://en.wikipedia.org/wiki/Matrix_multiplication>`_ for each stacked matrix.
            -   if either ``self`` or ``other`` has more than two dimensions, an array having a shape determined by :ref:`broadcasting` ``shape(self)[:-2]`` against ``shape(other)[:-2]`` and containing the `conventional matrix product <https://en.wikipedia.org/wiki/Matrix_multiplication>`_ for each stacked matrix.
            -   The returned array must have a data type determined by :ref:`type-promotion`.

        Notes
        -----

        .. note::
           Results must equal the results returned by the equivalent function :func:`~array_api.matmul`.

        **Raises**

        - if either ``self`` or ``other`` is a zero-dimensional array.
        - if ``self`` is a one-dimensional array having shape ``(K,)``, ``other`` is a one-dimensional array having shape ``(L,)``, and ``K != L``.
        - if ``self`` is a one-dimensional array having shape ``(K,)``, ``other`` is an array having shape ``(..., L, N)``, and ``K != L``.
        - if ``self`` is an array having shape ``(..., M, K)``, ``other`` is a one-dimensional array having shape ``(L,)``, and ``K != L``.
        - if ``self`` is an array having shape ``(..., M, K)``, ``other`` is an array having shape ``(..., L, N)``, and ``K != L``.

        .. versionchanged:: 2022.12
            Added complex data type support.
        """
        ...

    def __mod__(self: array, other: int | float | array, /) -> array:
        """
        Evaluates ``self_i % other_i`` for each element of an array instance with the respective element of the array ``other``.

        .. note::
           For input arrays which promote to an integer data type, the result of division by zero is unspecified and thus implementation-defined.

        Parameters
        ----------
        self: array
            array instance. Should have a real-valued data type.
        other: Union[int, float, array]
            other array. Must be compatible with ``self`` (see :ref:`broadcasting`). Should have a real-valued data type.

        Returns
        -------
        out: array
            an array containing the element-wise results. Each element-wise result must have the same sign as the respective element ``other_i``. The returned array must have a real-valued floating-point data type determined by :ref:`type-promotion`.


        .. note::
           Element-wise results, including special cases, must equal the results returned by the equivalent element-wise function :func:`~array_api.remainder`.
        """
        ...

    def __mul__(self: array, other: int | float | array, /) -> array:
        r"""
        Calculates the product for each element of an array instance with the respective element of the array ``other``.

        .. note::
           Floating-point multiplication is not always associative due to finite precision.

        Parameters
        ----------
        self: array
            array instance. Should have a numeric data type.
        other: Union[int, float, array]
            other array. Must be compatible with ``self`` (see :ref:`broadcasting`). Should have a numeric data type.

        Returns
        -------
        out: array
            an array containing the element-wise products. The returned array must have a data type determined by :ref:`type-promotion`.

        Notes
        -----

        .. note::
           Element-wise results, including special cases, must equal the results returned by the equivalent element-wise function :func:`~array_api.multiply`.

        .. versionchanged:: 2022.12
            Added complex data type support.
        """
        ...

    # See note above __eq__ method for explanation of the `type: ignore`
    def __ne__(self: array, other: int | float | bool | array, /) -> array:  # type: ignore[override]
        """
        Computes the truth value of ``self_i != other_i`` for each element of an array instance with the respective element of the array ``other``.

        Parameters
        ----------
        self: array
            array instance. May have any data type.
        other: Union[int, float, bool, array]
            other array. Must be compatible with ``self`` (see :ref:`broadcasting`). May have any data type.

        Returns
        -------
        out: array
            an array containing the element-wise results. The returned array must have a data type of ``bool`` (i.e., must be a boolean array).


        Notes
        -----

        .. note::
           Element-wise results, including special cases, must equal the results returned by the equivalent element-wise function :func:`~array_api.not_equal`.

        .. versionchanged:: 2022.12
            Added complex data type support.
        """
        ...

    def __neg__(self: array, /) -> array:
        """
        Evaluates ``-self_i`` for each element of an array instance.

        .. note::
           For signed integer data types, the numerical negative of the minimum representable integer is implementation-dependent.

        .. note::
           If ``self`` has a complex floating-point data type, both the real and imaginary components for each ``self_i`` must be negated (a result which follows from the rules of complex number multiplication).

        Parameters
        ----------
        self: array
            array instance. Should have a numeric data type.

        Returns
        -------
        out: array
            an array containing the evaluated result for each element in ``self``. The returned array must have a data type determined by :ref:`type-promotion`.

        Notes
        -----

        .. note::
           Element-wise results must equal the results returned by the equivalent element-wise function :func:`~array_api.negative`.

        .. versionchanged:: 2022.12
            Added complex data type support.
        """
        ...

    def __or__(self: array, other: int | bool | array, /) -> array:
        """
        Evaluates ``self_i | other_i`` for each element of an array instance with the respective element of the array ``other``.

        Parameters
        ----------
        self: array
            array instance. Should have an integer or boolean data type.
        other: Union[int, bool, array]
            other array. Must be compatible with ``self`` (see :ref:`broadcasting`). Should have an integer or boolean data type.

        Returns
        -------
        out: array
            an array containing the element-wise results. The returned array must have a data type determined by :ref:`type-promotion`.


        .. note::
           Element-wise results must equal the results returned by the equivalent element-wise function :func:`~array_api.bitwise_or`.
        """
        ...

    def __pos__(self: array, /) -> array:
        """
        Evaluates ``+self_i`` for each element of an array instance.

        Parameters
        ----------
        self: array
            array instance. Should have a numeric data type.

        Returns
        -------
        out: array
            an array containing the evaluated result for each element. The returned array must have the same data type as ``self``.

        Notes
        -----

        .. note::
           Element-wise results must equal the results returned by the equivalent element-wise function :func:`~array_api.positive`.

        .. versionchanged:: 2022.12
            Added complex data type support.
        """
        ...

    def __pow__(self: array, other: int | float | array, /) -> array:
        r"""
        Calculates an implementation-dependent approximation of exponentiation by raising each element (the base) of an array instance to the power of ``other_i`` (the exponent), where ``other_i`` is the corresponding element of the array ``other``.

        .. note::
           If both ``self`` and ``other`` have integer data types, the result of ``__pow__`` when `other_i` is negative (i.e., less than zero) is unspecified and thus implementation-dependent.

           If ``self`` has an integer data type and ``other`` has a floating-point data type, behavior is implementation-dependent, as type promotion between data type "kinds" (e.g., integer versus floating-point) is unspecified.

        Parameters
        ----------
        self: array
            array instance whose elements correspond to the exponentiation base. Should have a numeric data type.
        other: Union[int, float, array]
            other array whose elements correspond to the exponentiation exponent. Must be compatible with ``self`` (see :ref:`broadcasting`). Should have a numeric data type.

        Returns
        -------
        out: array
            an array containing the element-wise results. The returned array must have a data type determined by :ref:`type-promotion`.

        Notes
        -----

        .. note::
           Element-wise results, including special cases, must equal the results returned by the equivalent element-wise function :func:`~array_api.pow`.

        .. versionchanged:: 2022.12
            Added complex data type support.
        """
        ...

    def __rshift__(self: array, other: int | array, /) -> array:
        """
        Evaluates ``self_i >> other_i`` for each element of an array instance with the respective element of the array ``other``.

        Parameters
        ----------
        self: array
            array instance. Should have an integer data type.
        other: Union[int, array]
            other array. Must be compatible with ``self`` (see :ref:`broadcasting`). Should have an integer data type. Each element must be greater than or equal to ``0``.

        Returns
        -------
        out: array
            an array containing the element-wise results. The returned array must have the same data type as ``self``.


        .. note::
           Element-wise results must equal the results returned by the equivalent element-wise function :func:`~array_api.bitwise_right_shift`.
        """
        ...

    def __setitem__(
        self: array,
        key: int | slice | ellipsis | tuple[int | slice | ellipsis, ...] | array,
        value: int | float | bool | array,
        /,
    ) -> None:
        """
        Sets ``self[key]`` to ``value``.

        Parameters
        ----------
        self: array
            array instance.
        key: Union[int, slice, ellipsis, Tuple[Union[int, slice, ellipsis], ...], array]
            index key.
        value: Union[int, float, bool, array]
            value(s) to set. Must be compatible with ``self[key]`` (see :ref:`broadcasting`).


        .. note::

           Setting array values must not affect the data type of ``self``.

           When ``value`` is a Python scalar (i.e., ``int``, ``float``, ``bool``), behavior must follow specification guidance on mixing arrays with Python scalars (see :ref:`type-promotion`).

           When ``value`` is an ``array`` of a different data type than ``self``, how values are cast to the data type of ``self`` is implementation defined.
        """
        ...

    def __sub__(self: array, other: int | float | array, /) -> array:
        """
        Calculates the difference for each element of an array instance with the respective element of the array ``other``.

        The result of ``self_i - other_i`` must be the same as ``self_i + (-other_i)`` and must be governed by the same floating-point rules as addition (see :meth:`array.__add__`).

        Parameters
        ----------
        self: array
            array instance (minuend array). Should have a numeric data type.
        other: Union[int, float, array]
            subtrahend array. Must be compatible with ``self`` (see :ref:`broadcasting`). Should have a numeric data type.

        Returns
        -------
        out: array
            an array containing the element-wise differences. The returned array must have a data type determined by :ref:`type-promotion`.

        Notes
        -----

        .. note::
           Element-wise results must equal the results returned by the equivalent element-wise function :func:`~array_api.subtract`.

        .. versionchanged:: 2022.12
            Added complex data type support.
        """
        ...

    def __truediv__(self: array, other: int | float | array, /) -> array:
        r"""
        Evaluates ``self_i / other_i`` for each element of an array instance with the respective element of the array ``other``.

        .. note::
           If one or both of ``self`` and ``other`` have integer data types, the result is implementation-dependent, as type promotion between data type "kinds" (e.g., integer versus floating-point) is unspecified.

           Specification-compliant libraries may choose to raise an error or return an array containing the element-wise results. If an array is returned, the array must have a real-valued floating-point data type.

        Parameters
        ----------
        self: array
            array instance. Should have a numeric data type.
        other: Union[int, float, array]
            other array. Must be compatible with ``self`` (see :ref:`broadcasting`). Should have a numeric data type.

        Returns
        -------
        out: array
            an array containing the element-wise results. The returned array should have a floating-point data type determined by :ref:`type-promotion`.

        Notes
        -----

        .. note::
           Element-wise results, including special cases, must equal the results returned by the equivalent element-wise function :func:`~array_api.divide`.

        .. versionchanged:: 2022.12
            Added complex data type support.
        """
        ...

    def __xor__(self: array, other: int | bool | array, /) -> array:
        """
        Evaluates ``self_i ^ other_i`` for each element of an array instance with the respective element of the array ``other``.

        Parameters
        ----------
        self: array
            array instance. Should have an integer or boolean data type.
        other: Union[int, bool, array]
            other array. Must be compatible with ``self`` (see :ref:`broadcasting`). Should have an integer or boolean data type.

        Returns
        -------
        out: array
            an array containing the element-wise results. The returned array must have a data type determined by :ref:`type-promotion`.


        .. note::
           Element-wise results must equal the results returned by the equivalent element-wise function :func:`~array_api.bitwise_xor`.
        """
        ...

    def to_device(
        self: array, device: Device, /, *, stream: int | Any | None = None
    ) -> array:
        """
        Copy the array from the device on which it currently resides to the specified ``device``.

        Parameters
        ----------
        self: array
            array instance.
        device: device
            a ``device`` object (see :ref:`device-support`).
        stream: Optional[Union[int, Any]]
            stream object to use during copy. In addition to the types supported in :meth:`array.__dlpack__`, implementations may choose to support any library-specific stream object with the caveat that any code using such an object would not be portable.

        Returns
        -------
        out: array
            an array with the same data and data type as ``self`` and located on the specified ``device``.


        .. note::
           If ``stream`` is given, the copy operation should be enqueued on the provided ``stream``; otherwise, the copy operation should be enqueued on the default stream/queue. Whether the copy is performed synchronously or asynchronously is implementation-dependent. Accordingly, if synchronization is required to guarantee data safety, this must be clearly explained in a conforming library's documentation.
        """
        ...


__all__ = ["Array"]
