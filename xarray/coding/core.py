from typing import ClassVar, Literal, Protocol, Self

from xarray.core.dataset import Dataset
from xarray.core.datatree import DataTree
from xarray.core.variable import Variable

CoderKind = Literal["variable", "dataset", "datatree"]

XarrayObj = Variable | Dataset | DataTree


class Coder[T: XarrayObj](Protocol):
    kind: ClassVar[CoderKind]

    def configure(self, **settings) -> Self: ...

    def decode(self, obj: T) -> T: ...

    def encode(self, obj: T) -> T: ...
