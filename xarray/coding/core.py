from typing import ClassVar, Generic, Literal, Protocol, TypeVar

CoderKind = Literal["variable", "dataset", "datatree"]
T = TypeVar("T")


class Coder(Protocol, Generic[T]):
    kind: ClassVar[CoderKind]

    def decode(self, obj: T) -> T: ...

    def encode(self, obj: T) -> T: ...
