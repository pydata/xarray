from abc import abstractmethod
from collections import abc
from typing import Callable, Iterator, List, Optional

from .treenode import Tree

"""These iterators are copied from anytree.iterators, with minor modifications."""


class AbstractIter(abc.Iterator):
    def __init__(
        self,
        node: Tree,
        filter_: Optional[Callable] = None,
        stop: Optional[Callable] = None,
        maxlevel: Optional[int] = None,
    ):
        """
        Iterate over tree starting at `node`.
        Base class for all iterators.
        Keyword Args:
            filter_: function called with every `node` as argument, `node` is returned if `True`.
            stop: stop iteration at `node` if `stop` function returns `True` for `node`.
            maxlevel (int): maximum descending in the node hierarchy.
        """
        self.node = node
        self.filter_ = filter_
        self.stop = stop
        self.maxlevel = maxlevel
        self.__iter = None

    def __init(self):
        node = self.node
        maxlevel = self.maxlevel
        filter_ = self.filter_ or AbstractIter.__default_filter
        stop = self.stop or AbstractIter.__default_stop
        children = (
            []
            if AbstractIter._abort_at_level(1, maxlevel)
            else AbstractIter._get_children([node], stop)
        )
        return self._iter(children, filter_, stop, maxlevel)

    @staticmethod
    def __default_filter(node):
        return True

    @staticmethod
    def __default_stop(node):
        return False

    def __iter__(self) -> Iterator[Tree]:
        return self

    def __next__(self) -> Iterator[Tree]:
        if self.__iter is None:
            self.__iter = self.__init()
        item = next(self.__iter)  # type: ignore[call-overload]
        return item

    @staticmethod
    @abstractmethod
    def _iter(children: List[Tree], filter_, stop, maxlevel) -> Iterator[Tree]:
        ...

    @staticmethod
    def _abort_at_level(level, maxlevel):
        return maxlevel is not None and level > maxlevel

    @staticmethod
    def _get_children(children: List[Tree], stop) -> List[Tree]:
        return [child for child in children if not stop(child)]


class PreOrderIter(AbstractIter):
    """
    Iterate over tree applying pre-order strategy starting at `node`.
    Start at root and go-down until reaching a leaf node.
    Step upwards then, and search for the next leafs.
    """

    @staticmethod
    def _iter(children, filter_, stop, maxlevel):
        for child_ in children:
            if stop(child_):
                continue
            if filter_(child_):
                yield child_
            if not AbstractIter._abort_at_level(2, maxlevel):
                descendantmaxlevel = maxlevel - 1 if maxlevel else None
                for descendant_ in PreOrderIter._iter(
                    list(child_.children.values()), filter_, stop, descendantmaxlevel
                ):
                    yield descendant_


class LevelOrderIter(AbstractIter):
    """
    Iterate over tree applying level-order strategy starting at `node`.
    """

    @staticmethod
    def _iter(children, filter_, stop, maxlevel):
        level = 1
        while children:
            next_children = []
            for child in children:
                if filter_(child):
                    yield child
                next_children += AbstractIter._get_children(
                    list(child.children.values()), stop
                )
            children = next_children
            level += 1
            if AbstractIter._abort_at_level(level, maxlevel):
                break
