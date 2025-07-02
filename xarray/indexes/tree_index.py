from __future__ import annotations

import abc
from collections.abc import Hashable, Iterable, Mapping
from typing import TYPE_CHECKING, Any, Generic, TypeVar

import numpy as np

from xarray.core.dataarray import DataArray
from xarray.core.indexes import Index
from xarray.core.indexing import IndexSelResult
from xarray.core.variable import Variable

if TYPE_CHECKING:
    from scipy.spatial import KDTree

    from xarray.core.types import Self


class TreeAdapter(abc.ABC):
    """Lightweight adapter abstract class for plugging in 3rd-party structures
    like :py:class:`scipy.spatial.KDTree` or :py:class:`sklearn.neighbors.KDTree`
    into :py:class:`~xarray.indexes.TreeIndex`.

    """

    @abc.abstractmethod
    def __init__(self, points: np.ndarray, *, options: Mapping[str, Any]):
        """
        Parameters
        ----------
        points : ndarray of shape (n_points, n_coordinates)
            Two-dimensional array of points/samples (rows) and their
            corresponding coordinate labels (columns) to index.
        """
        ...

    @abc.abstractmethod
    def query(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Query points.

        Parameters
        ----------
        points: ndarray of shape (n_points, n_coordinates)
            Two-dimensional array of points/samples (rows) and their
            corresponding coordinate labels (columns) to query.

        Returns
        -------
        distances : ndarray of shape (n_points)
            Distances to the nearest neighbors.
        indices : ndarray of shape (n_points)
            Indices of the nearest neighbors in the array of the indexed
            points.
        """
        ...

    def equals(self, other: Self) -> bool:
        """Check equality with another TreeAdapter of the same kind.

        Parameters
        ----------
        other :
            The other TreeAdapter object to compare with this object.

        """
        raise NotImplementedError


class ScipyKDTreeAdapter(TreeAdapter):
    """:py:class:`scipy.spatial.KDTree` adapter for :py:class:`~xarray.indexes.TreeIndex`."""

    _kdtree: KDTree

    def __init__(self, points: np.ndarray, options: Mapping[str, Any]):
        from scipy.spatial import KDTree

        self._kdtree = KDTree(points, **options)

    def query(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return self._kdtree.query(points)

    def equals(self, other: Self) -> bool:
        return np.array_equal(self._kdtree.data, other._kdtree.data)


def get_points(coords: Iterable[Variable | Any]) -> np.ndarray:
    """Re-arrange data from a sequence of xarray coordinate variables or
    labels into a 2-d array of shape (n_points, n_coordinates).

    """
    data = [c.values if isinstance(c, Variable | DataArray) else c for c in coords]
    return np.stack([np.ravel(d) for d in data]).T


T_TreeAdapter = TypeVar("T_TreeAdapter", bound=TreeAdapter)


class TreeIndex(Index, Generic[T_TreeAdapter]):
    """Xarray index for irregular, n-dimensional data.

    This index may be associated with a set of coordinate variables representing
    the location of the data points in an n-dimensional space. All coordinates
    must have the same shape and dimensions. The number of associated coordinate
    variables must correspond to the number of dimensions of the space.

    This index supports label-based selection (nearest neighbor lookup). It also
    has limited support for alignment.

    By default, this index relies on :py:class:`scipy.spatial.KDTree` for fast
    lookup.

    Examples
    --------
    TODO

    """

    _tree_obj: T_TreeAdapter
    _coord_names: tuple[Hashable, ...]
    _dims: tuple[Hashable, ...]
    _shape: tuple[int, ...]

    def __init__(
        self,
        tree_obj: T_TreeAdapter,
        *,
        coord_names: tuple[Hashable, ...],
        dims: tuple[Hashable, ...],
        shape: tuple[int, ...],
    ):
        # this constructor is "private"
        assert isinstance(tree_obj, TreeAdapter)
        self._tree_obj = tree_obj

        assert len(coord_names) == len(dims) == len(shape)
        self._coord_names = coord_names
        self._dims = dims
        self._shape = shape

    @classmethod
    def from_variables(
        cls,
        variables: Mapping[Any, Variable],
        *,
        options: Mapping[str, Any],
    ) -> Self:
        if len(set([var.dims for var in variables.values()])) > 1:
            var_names = ",".join(vn for vn in variables)
            raise ValueError(
                f"variables {var_names} must all have the same dimensions and the same shape"
            )

        var0 = next(iter(variables.values()))

        if len(variables) != len(var0.dims):
            raise ValueError(
                f"the number of variables {len(variables)} doesn't match the number of dimensions {len(var0.dims)}"
            )

        opts = dict(options)

        tree_adapter_cls: type[T_TreeAdapter] = opts.pop("tree_adapter_cls", None)
        if tree_adapter_cls is None:
            tree_adapter_cls = ScipyKDTreeAdapter

        points = get_points(variables.values())

        return cls(
            tree_adapter_cls(points, options=opts),
            coord_names=tuple(variables),
            dims=var0.dims,
            shape=var0.shape,
        )

    def equals(
        self, other: Index, *, exclude: frozenset[Hashable] | None = None
    ) -> bool:
        if not isinstance(other, TreeIndex):
            return False
        if type(self._tree_obj) is not type(other._tree_obj):
            return False
        return self._tree_obj.equals(other._tree_obj)

    def _get_dim_indexers(
        self,
        indices: np.ndarray,
        label_dims: tuple[Hashable, ...],
        label_shape: tuple[int, ...],
    ) -> dict[Hashable, Variable]:
        """Returns dimension indexers based on the query results (indices) and
        the original label dimensions and shape.

        1. Unravel the flat indices returned from the query
        2. Reshape the unraveled indices according to indexers shapes
        3. Wrap the indices in xarray.Variable objects.

        """
        dim_indexers = {}

        u_indices = list(np.unravel_index(indices.ravel(), self._shape))

        for dim, ind in zip(self._dims, u_indices, strict=False):
            dim_indexers[dim] = Variable(label_dims, ind.reshape(label_shape))

        return dim_indexers

    def sel(
        self, labels: dict[Any, Any], method=None, tolerance=None
    ) -> IndexSelResult:
        if method != "nearest":
            raise ValueError("TreeIndex only supports selection with method='nearest'")

        missing_labels = set(self._coord_names) - set(labels)
        if missing_labels:
            missing_labels_str = ",".join([f"{name}" for name in missing_labels])
            raise ValueError(f"missing labels for coordinate(s): {missing_labels_str}.")

        if any(not isinstance(lbl, Variable | DataArray) for lbl in labels.values()):
            raise TypeError(
                "TreeIndex only supports advanced (point-wise) indexing "
                "with either xarray.DataArray or xarray.Variable objects."
            )

        if len(set([var.dims for var in labels.values()])) > 1:
            raise ValueError(
                "TreeIndex only supports advanced (point-wise) indexing "
                "with xarray.DataArray or xarray.Variable objects of matching dimensions."
            )

        label0: DataArray | Variable = next(iter(labels.values()))

        points = get_points(labels[name] for name in self._coord_names)
        _, indices = self._tree_obj.query(points)

        dim_indexers = self._get_dim_indexers(indices, label0.dims, label0.shape)

        return IndexSelResult(dim_indexers=dim_indexers)

    def rename(
        self,
        name_dict: Mapping[Any, Hashable],
        dims_dict: Mapping[Any, Hashable],
    ) -> Self:
        if not set(self._coord_names) & set(name_dict) and not set(self._dims) & set(
            dims_dict
        ):
            return self

        new_coord_names = tuple(name_dict.get(n, n) for n in self._coord_names)
        new_dims = tuple(dims_dict.get(d, d) for d in self._dims)

        return type(self)(
            self._tree_obj,
            coord_names=new_coord_names,
            dims=new_dims,
            shape=self._shape,
        )
