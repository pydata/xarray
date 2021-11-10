from collections.abc import MutableMapping
from typing import Dict, Hashable, Iterator, Mapping, Sequence

from xarray.core.variable import Variable

# from xarray.tree.datatree import DataTree


class DataTree:
    """Purely for type hinting purposes for now (and to avoid a circular import)"""

    ...


class DataManifest(MutableMapping):
    """
    Stores variables like a dict, but also stores children alongside in a hidden manner, to check against.

    Acts like a dict of keys to variables, but prevents setting variables to same key as any children. It prevents name
    collisions by acting as a common record of stored items for both the DataTree instance and its wrapped Dataset instance.
    """

    def __init__(
        self,
        variables: Dict[Hashable, Variable] = {},
        children: Dict[Hashable, DataTree] = {},
    ):
        if variables and children:
            keys_in_both = set(variables.keys()) & set(children.keys())
            if keys_in_both:
                raise KeyError(
                    f"The keys {keys_in_both} exist in both the variables and child nodes"
                )

        self._variables = variables
        self._children = children

    @property
    def children(self) -> Dict[Hashable, DataTree]:
        """Stores list of the node's children"""
        return self._children

    @children.setter
    def children(self, children: Dict[Hashable, DataTree]):
        for key, child in children.items():
            if key in self.keys():
                raise KeyError(
                    f"Cannot add child under key {key} because a variable is already stored under that key"
                )

            if not isinstance(child, DataTree):
                raise TypeError(
                    f"child nodes must be of type DataTree, not {type(child)}"
                )

        self._children = children

    def __getitem__(self, key: Hashable) -> Variable:
        """Forward to the variables here so the manifest acts like a normal dict of variables"""
        return self._variables[key]

    def __setitem__(self, key: Hashable, value: Variable):
        """Allow adding new variables, but first check if they conflict with any children"""

        if key in self._children:
            raise KeyError(
                f"key {key} already in use to denote a child"
                "node in wrapping DataTree node"
            )

        if isinstance(value, Variable):
            self._variables[key] = value
        else:
            raise TypeError(f"Cannot store object of type {type(value)}")

    def __delitem__(self, key: Hashable):
        """Forward to the variables here so the manifest acts like a normal dict of variables"""
        if key in self._variables:
            del self._variables[key]
        elif key in self.children:
            # TODO might be better not to del children here?
            del self._children[key]
        else:
            raise KeyError(f"Cannot remove item because nothing is stored under {key}")

    def __contains__(self, item: object) -> bool:
        """Forward to the variables here so the manifest acts like a normal dict of variables"""
        return item in self._variables

    def __iter__(self) -> Iterator:
        """Forward to the variables here so the manifest acts like a normal dict of variables"""
        return iter(self._variables)

    def __len__(self) -> int:
        """Forward to the variables here so the manifest acts like a normal dict of variables"""
        return len(self._variables)

    def copy(self) -> "DataManifest":
        """Required for consistency with dict"""
        return DataManifest(
            variables=self._variables.copy(), children=self._children.copy()
        )

    # TODO __repr__
