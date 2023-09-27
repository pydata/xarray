import typing
from collections.abc import Hashable, Iterable

if typing.TYPE_CHECKING:
    from xarray.namedarray.core import NamedArray

T_NamedArray = typing.TypeVar("T_NamedArray", bound="NamedArray")
DimsInput = typing.Union[str, Iterable[Hashable]]
Dims = tuple[Hashable, ...]


# temporary placeholder for indicating an array api compliant type.
# hopefully in the future we can narrow this down more
T_DuckArray = typing.TypeVar("T_DuckArray", bound=typing.Any)
