"""Module for the TileDB array backend."""

from collections import defaultdict
from typing import Any, Mapping, Optional, Tuple, Union

try:
    from typing import Protocol  # type: ignore
except ImportError:
    from typing_extensions import Protocol  # type: ignore

import numpy as np


class ExternalBackendArray(Protocol):
    """An example a a BackendArray protocol

    This example uses and index_support integer flag to define how the indexing
    is supported. An alterantive would be to define multiple  protocols for the
    different types of indexing.
    """

    @property
    def dtype(self) -> np.dtype:
        raise NotImplementedError

    @property
    def index_support(self) -> int:
        raise NotImplementedError

    @property
    def shape(self) -> Tuple[int, ...]:
        raise NotImplementedError

    def __getitem__(self, key):
        raise NotImplementedError


class RawVariable(Protocol):
    @property
    def dimensions(self) -> Tuple[str, ...]:
        raise NotImplementedError

    @property
    def data(self) -> ExternalBackendArray:
        raise NotImplementedError

    @property
    def attributes(self) -> Optional[Mapping[str, Any]]:
        raise NotImplementedError

    @property
    def encoding(self) -> Optional[Mapping[str, Any]]:
        raise NotImplementedError


class DataStore(Protocol):
    def get_dimensions(self) -> Mapping[str, int]:
        raise NotImplementedError

    def get_attrs(self) -> Mapping[str, Any]:
        raise NotImplementedError

    def get_variables(self) -> Mapping[str, RawVariable]:
        raise NotImplementedError

    def get_encoding(self) -> Mapping[str, Any]:
        raise NotImplementedError

    def close(self):
        raise NotImplementedError

