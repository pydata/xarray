from __future__ import annotations

from collections import OrderedDict
from html import escape
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Generic,
    Iterable,
    Mapping,
    MutableMapping,
    Optional,
    Tuple,
    Union,
)

from xarray import DataArray, Dataset
from xarray.core import utils
from xarray.core.options import OPTIONS as XR_OPTS
from xarray.core.variable import Variable

from . import formatting, formatting_html
from .mapping import TreeIsomorphismError, check_isomorphic, map_over_subtree
from .ops import (
    DataTreeArithmeticMixin,
    MappedDatasetMethodsMixin,
    MappedDataWithCoords,
)
from .render import RenderTree
from .treenode import NodePath, Tree, TreeNode

if TYPE_CHECKING:
    from xarray.core.merge import CoercibleValue

# """
# DEVELOPERS' NOTE
# ----------------
# The idea of this module is to create a `DataTree` class which inherits the tree structure from TreeNode, and also copies
# the entire API of `xarray.Dataset`, but with certain methods decorated to instead map the dataset function over every
# node in the tree. As this API is copied without directly subclassing `xarray.Dataset` we instead create various Mixin
# classes (in ops.py) which each define part of `xarray.Dataset`'s extensive API.

# Some of these methods must be wrapped to map over all nodes in the subtree. Others are fine to inherit unaltered
# (normally because they (a) only call dataset properties and (b) don't return a dataset that should be nested into a new
# tree) and some will get overridden by the class definition of DataTree.
# """


T_Path = Union[str, NodePath]


class DataTree(
    TreeNode,
    MappedDatasetMethodsMixin,
    MappedDataWithCoords,
    DataTreeArithmeticMixin,
    Generic[Tree],
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

    _name: Optional[str]
    _parent: Optional[Tree]
    _children: OrderedDict[str, Tree]
    _ds: Dataset

    def __init__(
        self,
        data: Optional[Dataset | DataArray] = None,
        parent: DataTree = None,
        children: Mapping[str, DataTree] = None,
        name: str = None,
    ):
        """
        Create a single node of a DataTree, which optionally contains data in the form of an xarray.Dataset.

        Parameters
        ----------
        data : Dataset, DataArray, Variable or None, optional
            Data to store under the .ds attribute of this node. DataArrays and Variables will be promoted to Datasets.
            Default is None.
        parent : DataTree, optional
            Parent node to this node. Default is None.
        children : Mapping[str, DataTree], optional
            Any child nodes of this node. Default is None.
        name : str, optional
            Name for the root node of the tree.

        Returns
        -------
        node :  DataTree

        See Also
        --------
        DataTree.from_dict
        """

        super().__init__(children=children)
        self.name = name
        self.parent = parent
        self.ds = data  # type: ignore[assignment]

    @property
    def name(self) -> str | None:
        """The name of this node."""
        return self._name

    @name.setter
    def name(self, name: str | None) -> None:
        self._name = name

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
    def ds(self) -> Dataset:
        """The data in this node, returned as a Dataset."""
        return self._ds

    @ds.setter
    def ds(self, data: Union[Dataset, DataArray] = None) -> None:
        if not isinstance(data, (Dataset, DataArray)) and data is not None:
            raise TypeError(
                f"{type(data)} object is not an xarray Dataset, DataArray, or None"
            )

        if isinstance(data, DataArray):
            data = data.to_dataset()
        elif data is None:
            data = Dataset()

        for var in list(data.variables):
            if var in self.children:
                raise KeyError(
                    f"Cannot add variable named {var}: node already has a child named {var}"
                )

        self._ds = data

    @property
    def has_data(self) -> bool:
        """Whether or not there are any data variables in this node."""
        return len(self.ds.variables) > 0

    @property
    def has_attrs(self) -> bool:
        """Whether or not there are any metadata attributes in this node."""
        return len(self.ds.attrs.keys()) > 0

    @property
    def is_empty(self) -> bool:
        """False if node contains any data or attrs. Does not look at children."""
        return not (self.has_data or self.has_attrs)

    def _pre_attach(self: DataTree, parent: DataTree) -> None:
        """
        Method which superclass calls before setting parent, here used to prevent having two
        children with duplicate names (or a data variable with the same name as a child).
        """
        super()._pre_attach(parent)
        if parent.has_data and self.name in list(parent.ds.variables):
            raise KeyError(
                f"parent {parent.name} already contains a data variable named {self.name}"
            )

    def __repr__(self) -> str:
        return formatting.datatree_repr(self)

    def __str__(self) -> str:
        return formatting.datatree_repr(self)

    def _repr_html_(self):
        """Make html representation of datatree object"""
        if XR_OPTS["display_style"] == "text":
            return f"<pre>{escape(repr(self))}</pre>"
        return formatting_html.datatree_repr(self)

    def get(
        self: DataTree, key: str, default: Optional[DataTree | DataArray] = None
    ) -> Optional[DataTree | DataArray]:
        """
        Access child nodes stored in this node as a DataTree or variables or coordinates stored in this node as a
        DataArray.

        Parameters
        ----------
        key : str
            Name of variable / node item, which must lie in this immediate node (not elsewhere in the tree).
        default : DataTree | DataArray, optional
            A value to return if the specified key does not exist.
            Default value is None.
        """
        if key in self.children:
            return self.children[key]
        elif key in self.ds:
            return self.ds[key]
        else:
            return default

    def __getitem__(self: DataTree, key: str) -> DataTree | DataArray:
        """
        Access child nodes stored in this tree as a DataTree or variables or coordinates stored in this tree as a
        DataArray.

        Parameters
        ----------
        key : str
            Name of variable / node, or unix-like path to variable / node.
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
        elif isinstance(val, (DataArray, Variable)):
            # TODO this should also accomodate other types that can be coerced into Variables
            self.ds[key] = val
        else:
            raise TypeError(f"Type {type(val)} cannot be assigned to a DataTree")

    def __setitem__(
        self, key: str, value: DataTree | Dataset | DataArray | Variable
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

        super().update(new_children)
        self.ds.update(new_variables)

    @classmethod
    def from_dict(
        cls,
        d: MutableMapping[str, DataTree | Dataset | DataArray],
        name: str = None,
    ) -> DataTree:
        """
        Create a datatree from a dictionary of data objects, labelled by paths into the tree.

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
        """

        # First create the root node
        # TODO there is a real bug here where what if root_data is of type DataTree?
        root_data = d.pop("/", None)
        obj = cls(name=name, data=root_data, parent=None, children=None)  # type: ignore[arg-type]

        if d:
            # Populate tree with children determined from data_objects mapping
            for path, data in d.items():
                # Create and set new node
                node_name = NodePath(path).name
                new_node = cls(name=node_name, data=data)  # type: ignore[arg-type]
                obj._set_item(
                    path,
                    new_node,
                    allow_overwrite=False,
                    new_nodes_along_path=True,
                )
        return obj

    @property
    def nbytes(self) -> int:
        return sum(node.ds.nbytes if node.has_data else 0 for node in self.subtree)

    def __len__(self) -> int:
        if self.children:
            n_children = len(self.children)
        else:
            n_children = 0
        return n_children + len(self.ds)

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
        such as tree1 + tree2.

        By default this method does not check any part of the tree above the given node.
        Therefore this method can be used as default to check that two subtrees are isomorphic.

        Parameters
        ----------
        other : DataTree
            The tree object to compare to.
        from_root : bool, optional, default is False
            Whether or not to first traverse to the root of the trees before checking for isomorphism.
            If a & b have no parents then this has no effect.
        strict_names : bool, optional, default is False
            Whether or not to also check that each node has the same name as its counterpart.

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
            The tree object to compare to.
        from_root : bool, optional, default is True
            Whether or not to first traverse to the root of the trees before checking.
            If a & b have no parents then this has no effect.

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
            The tree object to compare to.
        from_root : bool, optional, default is True
            Whether or not to first traverse to the root of the trees before checking.
            If a & b have no parents then this has no effect.

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
