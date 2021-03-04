
from ..core.indexing import (
    ExplicitIndexer,
    IndexingSupport,
    LazilyOuterIndexedArray,
    explicit_indexing_adapter,
)
from ..core.utils import FrozenDict
from ..core.variable import Variable
from .backend_protocol import ExternalBackendArray, DataStore
from .common import AbstractDataStore, BackendArray


class BackendArrayPlugin(BackendArray):
    def __init__(self, raw_array: ExternalBackendArray):
        self._array = raw_array
        self._indexing_support = IndexingSupport(self._array.indexing_support)

    @property
    def dtype(self):
        return self._array.dtype

    @property
    def shape(self):
        return self._array.shape

    def __getitem__(self, key):
        return explicit_indexing_adapter(
            key,
            self.shape,
            self._indexing_support,
            self._array.__getitem__,
        )


class LazyDataStore(AbstractDataStore):
    """Data store for reading TileDB arrays."""

    def __init__(self, data_store: DataStore):
        self._data_store = data_store

    def get_dimensions(self):
        return FrozenDict(**self._data_store.get_dimensions())

    def get_attrs(self):
        return FrozenDict(**self._data_store.get_attrs())

    def get_variables(self):
        return FrozenDict(
            (
                name,
                Variable(
                    raw_variable.dimensions,
                    LazilyOuterIndexedArray(BackendArrayPlugin(raw_variable.data)),
                    raw_variable.attributes,
                    raw_variable.encoding,
                ),
            )
            for name, raw_variable in self._data_store.get_variables().items()
        )

    def close(self):
        self._data_store.close()
