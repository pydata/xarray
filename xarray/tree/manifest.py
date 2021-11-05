from collections.abc import MutableMapping
from typing import Dict, Hashable, Mapping, Union

from xarray.core.variable import Variable

from .datatree import DataTree


class DataManifest(MutableMapping):
    """
    Stores variables and/or child tree nodes.

    When stored inside a DataTree node it prevents name collisions by acting as a common
    record of stored items for both the DataTree instance and its wrapped Dataset instance.

    When stored inside a lone Dataset it acts merely like a dictionary of Variables.
    """

    def __init__(
        self,
        variables: Dict[Hashable, Variable] = {},
        children: Dict[Hashable, DataTree] = {},
    ):
        self._variables = variables
        self._children = children

    @property
    def variables(self) -> Mapping[Hashable, Variable]:
        return self._variables

    @variables.setter
    def variables(self, variables):
        for key in variables:
            if key in self.children:
                raise KeyError(
                    f"Cannot set variable under name {key} because a child node "
                    "with that name already exists"
                )
        self._variables = variables

    @property
    def children(self) -> Mapping[Hashable, DataTree]:
        return self._children

    def __getitem__(self, key: Hashable) -> Union[Variable, DataTree]:
        if key in self._variables:
            return self._variables[key]
        elif key in self._children:
            return self._children[key]
        else:
            raise KeyError(f"{key} is not present")

    def __setitem__(self, key, value):
        raise NotImplementedError

    def __delitem__(self, key: Hashable):
        if key in self.variables:
            del self._variables[key]
        elif key in self.children:
            del self._children[key]
        else:
            raise KeyError(f"Cannot remove item because nothing is stored under {key}")

    def __iter__(self):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError
