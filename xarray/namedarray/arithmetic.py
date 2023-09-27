class ImplementsArrayReduce:
    ...


class IncludeReduceMethods:
    ...


class IncludeCumMethods:
    ...


class IncludeNumpySameMethods:
    ...


class SupportsArithmetic:
    ...


class NamedArrayOpsMixin:
    ...


class NamedArrayArithmetic(
    ImplementsArrayReduce,
    IncludeReduceMethods,
    IncludeCumMethods,
    IncludeNumpySameMethods,
    SupportsArithmetic,
    NamedArrayOpsMixin,
):
    __slots__ = ()
    # prioritize our operations over those of numpy.ndarray (priority=0)
    __array_priority__ = 50
