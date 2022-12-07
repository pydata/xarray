from __future__ import annotations

import copy
import itertools
from collections import OrderedDict
from html import escape
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Generic,
    Hashable,
    Iterable,
    Iterator,
    Mapping,
    MutableMapping,
    Optional,
    Set,
    Tuple,
    Union,
    overload,
)

import pandas as pd
from xarray.core import utils
from xarray.core.coordinates import DatasetCoordinates
from xarray.core.dataarray import DataArray
from xarray.core.dataset import Dataset, DataVariables
from xarray.core.indexes import Index, Indexes
from xarray.core.merge import dataset_update_method
from xarray.core.options import OPTIONS as XR_OPTS
from xarray.core.utils import Default, Frozen, _default
from xarray.core.variable import Variable, calculate_dimensions

from . import formatting, formatting_html
from .mapping import TreeIsomorphismError, check_isomorphic, map_over_subtree
from .ops import (
    DataTreeArithmeticMixin,
    MappedDatasetMethodsMixin,
    MappedDataWithCoords,
)
from .render import RenderTree
from .treenode import NamedNode, NodePath, Tree

if TYPE_CHECKING:
    from xarray.core.merge import CoercibleValue

# """
# DEVELOPERS' NOTE
# ----------------
# The idea of this module is to create a `DataTree` class which inherits the tree structure from TreeNode, and also copies
# the entire API of `xarray.Dataset`, but with certain methods decorated to instead map the dataset function over every
# node in the tree. As this API is copied without directly subclassing `xarray.Dataset` we instead create various Mixin
# classes (in ops.py) which each define part of `xarray.Dataset`'s extensive API.
#
# Some of these methods must be wrapped to map over all nodes in the subtree. Others are fine to inherit unaltered
# (normally because they (a) only call dataset properties and (b) don't return a dataset that should be nested into a new
# tree) and some will get overridden by the class definition of DataTree.
# """


T_Path = Union[str, NodePath]


def _coerce_to_dataset(data: Dataset | DataArray | None) -> Dataset:
    if isinstance(data, DataArray):
        ds = data.to_dataset()
    elif isinstance(data, Dataset):
        ds = data
    elif data is None:
        ds = Dataset()
    else:
        raise TypeError(
            f"data object is not an xarray Dataset, DataArray, or None, it is of type {type(data)}"
        )
    return ds


def _check_for_name_collisions(
    children: Iterable[str], variables: Iterable[Hashable]
) -> None:
    colliding_names = set(children).intersection(set(variables))
    if colliding_names:
        raise KeyError(
            f"Some names would collide between variables and children: {list(colliding_names)}"
        )


class DatasetView(Dataset):
    """
    An immutable Dataset-like view onto the data in a single DataTree node.

    In-place operations modifying this object should raise an AttributeError.

    Operations returning a new result will return a new xarray.Dataset object.
    This includes all API on Dataset, which will be inherited.

    This requires overriding all inherited private constructors.
    """

    # TODO what happens if user alters (in-place) a DataArray they extracted from this object?

    __slots__ = (
        "_attrs",
        "_cache",
        "_coord_names",
        "_dims",
        "_encoding",
        "_close",
        "_indexes",
        "_variables",
    )

    def __init__(
        self,
        data_vars: Optional[Mapping[Any, Any]] = None,
        coords: Optional[Mapping[Any, Any]] = None,
        attrs: Optional[Mapping[Any, Any]] = None,
    ):
        raise AttributeError("DatasetView objects are not to be initialized directly")

    @classmethod
    def _from_node(
        cls,
        wrapping_node: DataTree,
    ) -> DatasetView:
        """Constructor, using dataset attributes from wrapping node"""

        obj: DatasetView = object.__new__(cls)
        obj._variables = wrapping_node._variables
        obj._coord_names = wrapping_node._coord_names
        obj._dims = wrapping_node._dims
        obj._indexes = wrapping_node._indexes
        obj._attrs = wrapping_node._attrs
        obj._close = wrapping_node._close
        obj._encoding = wrapping_node._encoding

        return obj

    def __setitem__(self, key, val) -> None:
        raise AttributeError(
            "Mutation of the DatasetView is not allowed, please use __setitem__ on the wrapping DataTree node, "
            "or use `DataTree.to_dataset()` if you want a mutable dataset"
        )

    def update(self, other) -> None:
        raise AttributeError(
            "Mutation of the DatasetView is not allowed, please use .update on the wrapping DataTree node, "
            "or use `DataTree.to_dataset()` if you want a mutable dataset"
        )

    # FIXME https://github.com/python/mypy/issues/7328
    @overload
    def __getitem__(self, key: Mapping) -> Dataset:  # type: ignore[misc]
        ...

    @overload
    def __getitem__(self, key: Hashable) -> DataArray:  # type: ignore[misc]
        ...

    @overload
    def __getitem__(self, key: Any) -> Dataset:
        ...

    def __getitem__(self, key) -> DataArray:
        # TODO call the `_get_item` method of DataTree to allow path-like access to contents of other nodes
        # For now just call Dataset.__getitem__
        return Dataset.__getitem__(self, key)

    @classmethod
    def _construct_direct(
        cls,
        variables: dict[Any, Variable],
        coord_names: set[Hashable],
        dims: Optional[dict[Any, int]] = None,
        attrs: Optional[dict] = None,
        indexes: Optional[dict[Any, Index]] = None,
        encoding: Optional[dict] = None,
        close: Optional[Callable[[], None]] = None,
    ) -> Dataset:
        """
        Overriding this method (along with ._replace) and modifying it to return a Dataset object
        should hopefully ensure that the return type of any method on this object is a Dataset.
        """
        if dims is None:
            dims = calculate_dimensions(variables)
        if indexes is None:
            indexes = {}
        obj = object.__new__(Dataset)
        obj._variables = variables
        obj._coord_names = coord_names
        obj._dims = dims
        obj._indexes = indexes
        obj._attrs = attrs
        obj._close = close
        obj._encoding = encoding
        return obj

    def _replace(
        self,
        variables: Optional[dict[Hashable, Variable]] = None,
        coord_names: Optional[set[Hashable]] = None,
        dims: Optional[dict[Any, int]] = None,
        attrs: dict[Hashable, Any] | None | Default = _default,
        indexes: Optional[dict[Hashable, Index]] = None,
        encoding: dict | None | Default = _default,
        inplace: bool = False,
    ) -> Dataset:
        """
        Overriding this method (along with ._construct_direct) and modifying it to return a Dataset object
        should hopefully ensure that the return type of any method on this object is a Dataset.
        """

        if inplace:
            raise AttributeError("In-place mutation of the DatasetView is not allowed")

        return Dataset._replace(
            self,
            variables=variables,
            coord_names=coord_names,
            dims=dims,
            attrs=attrs,
            indexes=indexes,
            encoding=encoding,
            inplace=inplace,
        )


class DataTree(
    NamedNode,
    MappedDatasetMethodsMixin,
    MappedDataWithCoords,
    DataTreeArithmeticMixin,
    Generic[Tree],
    Mapping,
):
    """
    A tree-like hierarchical collection of xarray objects.

    Attempts to present an API like that of xarray.Dataset, but methods are wrapped to also update all the tree's child nodes.
    """

    # TODO attribute-like access for both vars and child nodes (by inheriting from xarray.core.common.AttrsAccessMixin?)

    # TODO ipython autocomplete for child nodes

    # TODO Some way of sorting children by depth

    # TODO Consistency in copying vs updating objects

    # TODO do we need a watch out for if methods intended only for root nodes are called on non-root nodes?

    # TODO dataset methods which should not or cannot act over the whole tree, such as .to_array

    # TODO del and delitem methods

    # TODO .loc, __contains__, __iter__, __array__, __len__

    # TODO a lot of properties like .variables could be defined in a DataMapping class which both Dataset and DataTree inherit from

    # TODO __slots__

    # TODO all groupby classes

    _name: Optional[str]
    _parent: Optional[DataTree]
    _children: OrderedDict[str, DataTree]
    _attrs: Optional[Dict[Hashable, Any]]
    _cache: Dict[str, Any]
    _coord_names: Set[Hashable]
    _dims: Dict[Hashable, int]
    _encoding: Optional[Dict[Hashable, Any]]
    _close: Optional[Callable[[], None]]
    _indexes: Dict[Hashable, Index]
    _variables: Dict[Hashable, Variable]

    __slots__ = (
        "_attrs",
        "_cache",
        "_coord_names",
        "_dims",
        "_encoding",
        "_close",
        "_indexes",
        "_variables",
    )

    def __init__(
        self,
        data: Optional[Dataset | DataArray] = None,
        parent: Optional[DataTree] = None,
        children: Optional[Mapping[str, DataTree]] = None,
        name: Optional[str] = None,
    ):
        """
        Create a single node of a DataTree.

        The node may optionally contain data in the form of data and coordinate variables, stored in the same way as
        data is stored in an xarray.Dataset.

        Parameters
        ----------
        data : Dataset, DataArray, or None, optional
            Data to store under the .ds attribute of this node. DataArrays will be promoted to Datasets.
            Default is None.
        parent : DataTree, optional
            Parent node to this node. Default is None.
        children : Mapping[str, DataTree], optional
            Any child nodes of this node. Default is None.
        name : str, optional
            Name for this node of the tree. Default is None.

        Returns
        -------
        DataTree

        See Also
        --------
        DataTree.from_dict
        """

        # validate input
        if children is None:
            children = {}
        ds = _coerce_to_dataset(data)
        _check_for_name_collisions(children, ds.variables)

        # set tree attributes
        super().__init__(children=children)
        self.name = name
        self.parent = parent

        # set data attributes
        self._replace(
            inplace=True,
            variables=ds._variables,
            coord_names=ds._coord_names,
            dims=ds._dims,
            indexes=ds._indexes,
            attrs=ds._attrs,
            encoding=ds._encoding,
        )
        self._close = ds._close

    @property
    def parent(self: DataTree) -> DataTree | None:
        """Parent of this node."""
        return self._parent

    @parent.setter
    def parent(self: DataTree, new_parent: DataTree) -> None:
        if new_parent and self.name is None:
            raise ValueError("Cannot set an unnamed node as a child of another node")
        self._set_parent(new_parent, self.name)

    @property
    def ds(self) -> DatasetView:
        """
        An immutable Dataset-like view onto the data in this node.

        For a mutable Dataset containing the same data as in this node, use `.to_dataset()` instead.

        See Also
        --------
        DataTree.to_dataset
        """
        return DatasetView._from_node(self)

    @ds.setter
    def ds(self, data: Optional[Union[Dataset, DataArray]] = None) -> None:

        ds = _coerce_to_dataset(data)

        _check_for_name_collisions(self.children, ds.variables)

        self._replace(
            inplace=True,
            variables=ds._variables,
            coord_names=ds._coord_names,
            dims=ds._dims,
            indexes=ds._indexes,
            attrs=ds._attrs,
            encoding=ds._encoding,
        )
        self._close = ds._close

    def _pre_attach(self: DataTree, parent: DataTree) -> None:
        """
        Method which superclass calls before setting parent, here used to prevent having two
        children with duplicate names (or a data variable with the same name as a child).
        """
        super()._pre_attach(parent)
        if self.name in list(parent.ds.variables):
            raise KeyError(
                f"parent {parent.name} already contains a data variable named {self.name}"
            )

    def to_dataset(self) -> Dataset:
        """
        Return the data in this node as a new xarray.Dataset object.

        See Also
        --------
        DataTree.ds
        """
        return Dataset._construct_direct(
            self._variables,
            self._coord_names,
            self._dims,
            self._attrs,
            self._indexes,
            self._encoding,
            self._close,
        )

    @property
    def has_data(self):
        """Whether or not there are any data variables in this node."""
        return len(self._variables) > 0

    @property
    def has_attrs(self) -> bool:
        """Whether or not there are any metadata attributes in this node."""
        return len(self.attrs.keys()) > 0

    @property
    def is_empty(self) -> bool:
        """False if node contains any data or attrs. Does not look at children."""
        return not (self.has_data or self.has_attrs)

    @property
    def variables(self) -> Mapping[Hashable, Variable]:
        """Low level interface to node contents as dict of Variable objects.

        This ordered dictionary is frozen to prevent mutation that could
        violate Dataset invariants. It contains all variable objects
        constituting this DataTree node, including both data variables and
        coordinates.
        """
        return Frozen(self._variables)

    @property
    def attrs(self) -> Dict[Hashable, Any]:
        """Dictionary of global attributes on this node object."""
        if self._attrs is None:
            self._attrs = {}
        return self._attrs

    @attrs.setter
    def attrs(self, value: Mapping[Any, Any]) -> None:
        self._attrs = dict(value)

    @property
    def encoding(self) -> Dict:
        """Dictionary of global encoding attributes on this node object."""
        if self._encoding is None:
            self._encoding = {}
        return self._encoding

    @encoding.setter
    def encoding(self, value: Mapping) -> None:
        self._encoding = dict(value)

    @property
    def dims(self) -> Mapping[Hashable, int]:
        """Mapping from dimension names to lengths.

        Cannot be modified directly, but is updated when adding new variables.

        Note that type of this object differs from `DataArray.dims`.
        See `DataTree.sizes`, `Dataset.sizes`, and `DataArray.sizes` for consistently named
        properties.
        """
        return Frozen(self._dims)

    @property
    def sizes(self) -> Mapping[Hashable, int]:
        """Mapping from dimension names to lengths.

        Cannot be modified directly, but is updated when adding new variables.

        This is an alias for `DataTree.dims` provided for the benefit of
        consistency with `DataArray.sizes`.

        See Also
        --------
        DataArray.sizes
        """
        return self.dims

    def __contains__(self, key: object) -> bool:
        """The 'in' operator will return true or false depending on whether
        'key' is either an array stored in the datatree or a child node, or neither.
        """
        return key in self.variables or key in self.children

    def __bool__(self) -> bool:
        return bool(self.ds.data_vars) or bool(self.children)

    def __iter__(self) -> Iterator[Hashable]:
        return itertools.chain(self.ds.data_vars, self.children)

    def __repr__(self) -> str:
        return formatting.datatree_repr(self)

    def __str__(self) -> str:
        return formatting.datatree_repr(self)

    def _repr_html_(self):
        """Make html representation of datatree object"""
        if XR_OPTS["display_style"] == "text":
            return f"<pre>{escape(repr(self))}</pre>"
        return formatting_html.datatree_repr(self)

    @classmethod
    def _construct_direct(
        cls,
        variables: dict[Any, Variable],
        coord_names: set[Hashable],
        dims: Optional[dict[Any, int]] = None,
        attrs: Optional[dict] = None,
        indexes: Optional[dict[Any, Index]] = None,
        encoding: Optional[dict] = None,
        name: str | None = None,
        parent: DataTree | None = None,
        children: Optional[OrderedDict[str, DataTree]] = None,
        close: Optional[Callable[[], None]] = None,
    ) -> DataTree:
        """Shortcut around __init__ for internal use when we want to skip costly validation."""

        # data attributes
        if dims is None:
            dims = calculate_dimensions(variables)
        if indexes is None:
            indexes = {}
        if children is None:
            children = OrderedDict()

        obj: DataTree = object.__new__(cls)
        obj._variables = variables
        obj._coord_names = coord_names
        obj._dims = dims
        obj._indexes = indexes
        obj._attrs = attrs
        obj._close = close
        obj._encoding = encoding

        # tree attributes
        obj._name = name
        obj._children = children
        obj._parent = parent

        return obj

    def _replace(
        self: DataTree,
        variables: Optional[dict[Hashable, Variable]] = None,
        coord_names: Optional[set[Hashable]] = None,
        dims: Optional[dict[Any, int]] = None,
        attrs: dict[Hashable, Any] | None | Default = _default,
        indexes: Optional[dict[Hashable, Index]] = None,
        encoding: dict | None | Default = _default,
        name: str | None | Default = _default,
        parent: DataTree | None = _default,
        children: Optional[OrderedDict[str, DataTree]] = None,
        inplace: bool = False,
    ) -> DataTree:
        """
        Fastpath constructor for internal use.

        Returns an object with optionally replaced attributes.

        Explicitly passed arguments are *not* copied when placed on the new
        datatree. It is up to the caller to ensure that they have the right type
        and are not used elsewhere.
        """
        if inplace:
            if variables is not None:
                self._variables = variables
            if coord_names is not None:
                self._coord_names = coord_names
            if dims is not None:
                self._dims = dims
            if attrs is not _default:
                self._attrs = attrs
            if indexes is not None:
                self._indexes = indexes
            if encoding is not _default:
                self._encoding = encoding
            if name is not _default:
                self._name = name
            if parent is not _default:
                self._parent = parent
            if children is not None:
                self._children = children
            obj = self
        else:
            if variables is None:
                variables = self._variables.copy()
            if coord_names is None:
                coord_names = self._coord_names.copy()
            if dims is None:
                dims = self._dims.copy()
            if attrs is _default:
                attrs = copy.copy(self._attrs)
            if indexes is None:
                indexes = self._indexes.copy()
            if encoding is _default:
                encoding = copy.copy(self._encoding)
            if name is _default:
                name = self._name  # no need to copy str objects or None
            if parent is _default:
                parent = copy.copy(self._parent)
            if children is _default:
                children = copy.copy(self._children)
            obj = self._construct_direct(
                variables,
                coord_names,
                dims,
                attrs,
                indexes,
                encoding,
                name,
                parent,
                children,
            )
        return obj

    def get(
        self: DataTree, key: str, default: Optional[DataTree | DataArray] = None
    ) -> Optional[DataTree | DataArray]:
        """
        Access child nodes, variables, or coordinates stored in this node.

        Returned object will be either a DataTree or DataArray object depending on whether the key given points to a
        child or variable.

        Parameters
        ----------
        key : str
            Name of variable / child within this node. Must lie in this immediate node (not elsewhere in the tree).
        default : DataTree | DataArray, optional
            A value to return if the specified key does not exist. Default return value is None.
        """
        if key in self.children:
            return self.children[key]
        elif key in self.ds:
            return self.ds[key]
        else:
            return default

    def __getitem__(self: DataTree, key: str) -> DataTree | DataArray:
        """
        Access child nodes, variables, or coordinates stored anywhere in this tree.

        Returned object will be either a DataTree or DataArray object depending on whether the key given points to a
        child or variable.

        Parameters
        ----------
        key : str
            Name of variable / child within this node, or unix-like path to variable / child within another node.

        Returns
        -------
        Union[DataTree, DataArray]
        """

        # Either:
        if utils.is_dict_like(key):

            # dict-like indexing
            raise NotImplementedError("Should this index over whole tree?")
        elif isinstance(key, str):
            # TODO should possibly deal with hashables in general?
            # path-like: a name of a node/variable, or path to a node/variable
            path = NodePath(key)
            return self._get_item(path)
        elif utils.is_list_like(key):
            # iterable of variable names
            raise NotImplementedError(
                "Selecting via tags is deprecated, and selecting multiple items should be "
                "implemented via .subset"
            )
        else:
            raise ValueError(f"Invalid format for key: {key}")

    def _set(self, key: str, val: DataTree | CoercibleValue) -> None:
        """
        Set the child node or variable with the specified key to value.

        Counterpart to the public .get method, and also only works on the immediate node, not other nodes in the tree.
        """
        if isinstance(val, DataTree):
            val.name = key
            val.parent = self
        else:
            if not isinstance(val, (DataArray, Variable)):
                # accommodate other types that can be coerced into Variables
                val = DataArray(val)

            self.update({key: val})

    def __setitem__(
        self,
        key: str,
        value: Any,
    ) -> None:
        """
        Add either a child node or an array to the tree, at any position.

        Data can be added anywhere, and new nodes will be created to cross the path to the new location if necessary.

        If there is already a node at the given location, then if value is a Node class or Dataset it will overwrite the
        data already present at that node, and if value is a single array, it will be merged with it.
        """
        # TODO xarray.Dataset accepts other possibilities, how do we exactly replicate all the behaviour?
        if utils.is_dict_like(key):
            raise NotImplementedError
        elif isinstance(key, str):
            # TODO should possibly deal with hashables in general?
            # path-like: a name of a node/variable, or path to a node/variable
            path = NodePath(key)
            return self._set_item(path, value, new_nodes_along_path=True)
        else:
            raise ValueError("Invalid format for key")

    def update(self, other: Dataset | Mapping[str, DataTree | DataArray]) -> None:
        """
        Update this node's children and / or variables.

        Just like `dict.update` this is an in-place operation.
        """
        # TODO separate by type
        new_children = {}
        new_variables = {}
        for k, v in other.items():
            if isinstance(v, DataTree):
                new_children[k] = v
            elif isinstance(v, (DataArray, Variable)):
                # TODO this should also accommodate other types that can be coerced into Variables
                new_variables[k] = v
            else:
                raise TypeError(f"Type {type(v)} cannot be assigned to a DataTree")

        vars_merge_result = dataset_update_method(self.to_dataset(), new_variables)
        # TODO are there any subtleties with preserving order of children like this?
        merged_children = OrderedDict(**self.children, **new_children)
        self._replace(
            inplace=True, children=merged_children, **vars_merge_result._asdict()
        )

    @classmethod
    def from_dict(
        cls,
        d: MutableMapping[str, Dataset | DataArray | DataTree | None],
        name: Optional[str] = None,
    ) -> DataTree:
        """
        Create a datatree from a dictionary of data objects, organised by paths into the tree.

        Parameters
        ----------
        d : dict-like
            A mapping from path names to xarray.Dataset, xarray.DataArray, or DataTree objects.

            Path names are to be given as unix-like path. If path names containing more than one part are given, new
            tree nodes will be constructed as necessary.

            To assign data to the root node of the tree use "/" as the path.
        name : Hashable, optional
            Name for the root node of the tree. Default is None.

        Returns
        -------
        DataTree

        Notes
        -----
        If your dictionary is nested you will need to flatten it before using this method.
        """

        # First create the root node
        root_data = d.pop("/", None)
        obj = cls(name=name, data=root_data, parent=None, children=None)

        if d:
            # Populate tree with children determined from data_objects mapping
            for path, data in d.items():
                # Create and set new node
                node_name = NodePath(path).name
                if isinstance(data, cls):
                    # TODO ignoring type error only needed whilst .copy() method is copied from Dataset.copy().
                    new_node = data.copy()  # type: ignore[attr-defined]
                    new_node.orphan()
                else:
                    new_node = cls(name=node_name, data=data)
                obj._set_item(
                    path,
                    new_node,
                    allow_overwrite=False,
                    new_nodes_along_path=True,
                )

        return obj

    def to_dict(self) -> Dict[str, Dataset]:
        """
        Create a dictionary mapping of absolute node paths to the data contained in those nodes.

        Returns
        -------
        Dict[str, Dataset]
        """
        return {node.path: node.to_dataset() for node in self.subtree}

    @property
    def nbytes(self) -> int:
        return sum(node.to_dataset().nbytes for node in self.subtree)

    def __len__(self) -> int:
        return len(self.children) + len(self.data_vars)

    @property
    def indexes(self) -> Indexes[pd.Index]:
        """Mapping of pandas.Index objects used for label based indexing.
        Raises an error if this DataTree node has indexes that cannot be coerced
        to pandas.Index objects.

        See Also
        --------
        DataTree.xindexes
        """
        return self.xindexes.to_pandas_indexes()

    @property
    def xindexes(self) -> Indexes[Index]:
        """Mapping of xarray Index objects used for label based indexing."""
        return Indexes(self._indexes, {k: self._variables[k] for k in self._indexes})

    @property
    def coords(self) -> DatasetCoordinates:
        """Dictionary of xarray.DataArray objects corresponding to coordinate
        variables
        """
        return DatasetCoordinates(self.to_dataset())

    @property
    def data_vars(self) -> DataVariables:
        """Dictionary of DataArray objects corresponding to data variables"""
        return DataVariables(self.to_dataset())

    def isomorphic(
        self,
        other: DataTree,
        from_root: bool = False,
        strict_names: bool = False,
    ) -> bool:
        """
        Two DataTrees are considered isomorphic if every node has the same number of children.

        Nothing about the data in each node is checked.

        Isomorphism is a necessary condition for two trees to be used in a nodewise binary operation,
        such as ``tree1 + tree2``.

        By default this method does not check any part of the tree above the given node.
        Therefore this method can be used as default to check that two subtrees are isomorphic.

        Parameters
        ----------
        other : DataTree
            The other tree object to compare to.
        from_root : bool, optional, default is False
            Whether or not to first traverse to the root of the two trees before checking for isomorphism.
            If neither tree has a parent then this has no effect.
        strict_names : bool, optional, default is False
            Whether or not to also check that every node in the tree has the same name as its counterpart in the other
            tree.

        See Also
        --------
        DataTree.equals
        DataTree.identical
        """
        try:
            check_isomorphic(
                self,
                other,
                require_names_equal=strict_names,
                check_from_root=from_root,
            )
            return True
        except (TypeError, TreeIsomorphismError):
            return False

    def equals(self, other: DataTree, from_root: bool = True) -> bool:
        """
        Two DataTrees are equal if they have isomorphic node structures, with matching node names,
        and if they have matching variables and coordinates, all of which are equal.

        By default this method will check the whole tree above the given node.

        Parameters
        ----------
        other : DataTree
            The other tree object to compare to.
        from_root : bool, optional, default is True
            Whether or not to first traverse to the root of the two trees before checking for isomorphism.
            If neither tree has a parent then this has no effect.

        See Also
        --------
        Dataset.equals
        DataTree.isomorphic
        DataTree.identical
        """
        if not self.isomorphic(other, from_root=from_root, strict_names=True):
            return False

        return all(
            [
                node.ds.equals(other_node.ds)
                for node, other_node in zip(self.subtree, other.subtree)
            ]
        )

    def identical(self, other: DataTree, from_root=True) -> bool:
        """
        Like equals, but will also check all dataset attributes and the attributes on
        all variables and coordinates.

        By default this method will check the whole tree above the given node.

        Parameters
        ----------
        other : DataTree
            The other tree object to compare to.
        from_root : bool, optional, default is True
            Whether or not to first traverse to the root of the two trees before checking for isomorphism.
            If neither tree has a parent then this has no effect.

        See Also
        --------
        Dataset.identical
        DataTree.isomorphic
        DataTree.equals
        """
        if not self.isomorphic(other, from_root=from_root, strict_names=True):
            return False

        return all(
            node.ds.identical(other_node.ds)
            for node, other_node in zip(self.subtree, other.subtree)
        )

    def map_over_subtree(
        self,
        func: Callable,
        *args: Iterable[Any],
        **kwargs: Any,
    ) -> DataTree | Tuple[DataTree]:
        """
        Apply a function to every dataset in this subtree, returning a new tree which stores the results.

        The function will be applied to any dataset stored in this node, as well as any dataset stored in any of the
        descendant nodes. The returned tree will have the same structure as the original subtree.

        func needs to return a Dataset in order to rebuild the subtree.

        Parameters
        ----------
        func : callable
            Function to apply to datasets with signature:
            `func(node.ds, *args, **kwargs) -> Dataset`.

            Function will not be applied to any nodes without datasets.
        *args : tuple, optional
            Positional arguments passed on to `func`.
        **kwargs : Any
            Keyword arguments passed on to `func`.

        Returns
        -------
        subtrees : DataTree, Tuple of DataTrees
            One or more subtrees containing results from applying ``func`` to the data at each node.
        """
        # TODO this signature means that func has no way to know which node it is being called upon - change?

        # TODO fix this typing error
        return map_over_subtree(func)(self, *args, **kwargs)  # type: ignore[operator]

    def map_over_subtree_inplace(
        self,
        func: Callable,
        *args: Iterable[Any],
        **kwargs: Any,
    ) -> None:
        """
        Apply a function to every dataset in this subtree, updating data in place.

        Parameters
        ----------
        func : callable
            Function to apply to datasets with signature:
            `func(node.ds, *args, **kwargs) -> Dataset`.

            Function will not be applied to any nodes without datasets,
        *args : tuple, optional
            Positional arguments passed on to `func`.
        **kwargs : Any
            Keyword arguments passed on to `func`.
        """

        # TODO if func fails on some node then the previous nodes will still have been updated...

        for node in self.subtree:
            if node.has_data:
                node.ds = func(node.ds, *args, **kwargs)

    def pipe(
        self, func: Callable | tuple[Callable, str], *args: Any, **kwargs: Any
    ) -> Any:
        """Apply ``func(self, *args, **kwargs)``

        This method replicates the pandas method of the same name.

        Parameters
        ----------
        func : callable
            function to apply to this xarray object (Dataset/DataArray).
            ``args``, and ``kwargs`` are passed into ``func``.
            Alternatively a ``(callable, data_keyword)`` tuple where
            ``data_keyword`` is a string indicating the keyword of
            ``callable`` that expects the xarray object.
        *args
            positional arguments passed into ``func``.
        **kwargs
            a dictionary of keyword arguments passed into ``func``.

        Returns
        -------
        object : Any
            the return type of ``func``.

        Notes
        -----
        Use ``.pipe`` when chaining together functions that expect
        xarray or pandas objects, e.g., instead of writing

        .. code:: python

            f(g(h(dt), arg1=a), arg2=b, arg3=c)

        You can write

        .. code:: python

            (dt.pipe(h).pipe(g, arg1=a).pipe(f, arg2=b, arg3=c))

        If you have a function that takes the data as (say) the second
        argument, pass a tuple indicating which keyword expects the
        data. For example, suppose ``f`` takes its data as ``arg2``:

        .. code:: python

            (dt.pipe(h).pipe(g, arg1=a).pipe((f, "arg2"), arg1=a, arg3=c))

        """
        if isinstance(func, tuple):
            func, target = func
            if target in kwargs:
                raise ValueError(
                    f"{target} is both the pipe target and a keyword argument"
                )
            kwargs[target] = self
        else:
            args = (self,) + args
        return func(*args, **kwargs)

    def render(self):
        """Print tree structure, including any data stored at each node."""
        for pre, fill, node in RenderTree(self):
            print(f"{pre}DataTree('{self.name}')")
            for ds_line in repr(node.ds)[1:]:
                print(f"{fill}{ds_line}")

    def merge(self, datatree: DataTree) -> DataTree:
        """Merge all the leaves of a second DataTree into this one."""
        raise NotImplementedError

    def merge_child_nodes(self, *paths, new_path: T_Path) -> DataTree:
        """Merge a set of child nodes into a single new node."""
        raise NotImplementedError

    # TODO some kind of .collapse() or .flatten() method to merge a subtree

    def as_array(self) -> DataArray:
        return self.ds.as_dataarray()

    @property
    def groups(self):
        """Return all netCDF4 groups in the tree, given as a tuple of path-like strings."""
        return tuple(node.path for node in self.subtree)

    def to_netcdf(
        self, filepath, mode: str = "w", encoding=None, unlimited_dims=None, **kwargs
    ):
        """
        Write datatree contents to a netCDF file.

        Parameters
        ----------
        filepath : str or Path
            Path to which to save this datatree.
        mode : {"w", "a"}, default: "w"
            Write ('w') or append ('a') mode. If mode='w', any existing file at
            this location will be overwritten. If mode='a', existing variables
            will be overwritten. Only appies to the root group.
        encoding : dict, optional
            Nested dictionary with variable names as keys and dictionaries of
            variable specific encodings as values, e.g.,
            ``{"root/set1": {"my_variable": {"dtype": "int16", "scale_factor": 0.1,
            "zlib": True}, ...}, ...}``. See ``xarray.Dataset.to_netcdf`` for available
            options.
        unlimited_dims : dict, optional
            Mapping of unlimited dimensions per group that that should be serialized as unlimited dimensions.
            By default, no dimensions are treated as unlimited dimensions.
            Note that unlimited_dims may also be set via
            ``dataset.encoding["unlimited_dims"]``.
        kwargs :
            Addional keyword arguments to be passed to ``xarray.Dataset.to_netcdf``
        """
        from .io import _datatree_to_netcdf

        _datatree_to_netcdf(
            self,
            filepath,
            mode=mode,
            encoding=encoding,
            unlimited_dims=unlimited_dims,
            **kwargs,
        )

    def to_zarr(
        self, store, mode: str = "w", encoding=None, consolidated: bool = True, **kwargs
    ):
        """
        Write datatree contents to a Zarr store.

        Parameters
        ----------
        store : MutableMapping, str or Path, optional
            Store or path to directory in file system
        mode : {{"w", "w-", "a", "r+", None}, default: "w"
            Persistence mode: “w” means create (overwrite if exists); “w-” means create (fail if exists);
            “a” means override existing variables (create if does not exist); “r+” means modify existing
            array values only (raise an error if any metadata or shapes would change). The default mode
            is “a” if append_dim is set. Otherwise, it is “r+” if region is set and w- otherwise.
        encoding : dict, optional
            Nested dictionary with variable names as keys and dictionaries of
            variable specific encodings as values, e.g.,
            ``{"root/set1": {"my_variable": {"dtype": "int16", "scale_factor": 0.1}, ...}, ...}``.
            See ``xarray.Dataset.to_zarr`` for available options.
        consolidated : bool
            If True, apply zarr's `consolidate_metadata` function to the store
            after writing metadata for all groups.
        kwargs :
            Additional keyword arguments to be passed to ``xarray.Dataset.to_zarr``
        """
        from .io import _datatree_to_zarr

        _datatree_to_zarr(
            self,
            store,
            mode=mode,
            encoding=encoding,
            consolidated=consolidated,
            **kwargs,
        )

    def plot(self):
        raise NotImplementedError
