"""
String Tree Rendering. Copied from anytree.
"""

import collections
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .datatree import DataTree

Row = collections.namedtuple("Row", ("pre", "fill", "node"))


class AbstractStyle(object):
    def __init__(self, vertical, cont, end):
        """
        Tree Render Style.
        Args:
            vertical: Sign for vertical line.
            cont: Chars for a continued branch.
            end: Chars for the last branch.
        """
        super(AbstractStyle, self).__init__()
        self.vertical = vertical
        self.cont = cont
        self.end = end
        assert (
            len(cont) == len(vertical) == len(end)
        ), f"'{vertical}', '{cont}' and '{end}' need to have equal length"

    @property
    def empty(self):
        """Empty string as placeholder."""
        return " " * len(self.end)

    def __repr__(self):
        return f"{self.__class__.__name__}()"


class ContStyle(AbstractStyle):
    def __init__(self):
        """
        Continued style, without gaps.

        >>> from anytree import Node, RenderTree
        >>> root = Node("root")
        >>> s0 = Node("sub0", parent=root)
        >>> s0b = Node("sub0B", parent=s0)
        >>> s0a = Node("sub0A", parent=s0)
        >>> s1 = Node("sub1", parent=root)
        >>> print(RenderTree(root, style=ContStyle()))

        Node('/root')
        ├── Node('/root/sub0')
        │   ├── Node('/root/sub0/sub0B')
        │   └── Node('/root/sub0/sub0A')
        └── Node('/root/sub1')
        """
        super(ContStyle, self).__init__(
            "\u2502   ", "\u251c\u2500\u2500 ", "\u2514\u2500\u2500 "
        )


class RenderTree(object):
    def __init__(
        self, node: "DataTree", style=ContStyle(), childiter=list, maxlevel=None
    ):
        """
        Render tree starting at `node`.
        Keyword Args:
            style (AbstractStyle): Render Style.
            childiter: Child iterator.
            maxlevel: Limit rendering to this depth.
        :any:`RenderTree` is an iterator, returning a tuple with 3 items:
        `pre`
            tree prefix.
        `fill`
            filling for multiline entries.
        `node`
            :any:`NodeMixin` object.
        It is up to the user to assemble these parts to a whole.
        >>> from anytree import Node, RenderTree
        >>> root = Node("root", lines=["c0fe", "c0de"])
        >>> s0 = Node("sub0", parent=root, lines=["ha", "ba"])
        >>> s0b = Node("sub0B", parent=s0, lines=["1", "2", "3"])
        >>> s0a = Node("sub0A", parent=s0, lines=["a", "b"])
        >>> s1 = Node("sub1", parent=root, lines=["Z"])
        Simple one line:
        >>> for pre, _, node in RenderTree(root):
        ...     print("%s%s" % (pre, node.name))
        ...
        root
        ├── sub0
        │   ├── sub0B
        │   └── sub0A
        └── sub1
        Multiline:
        >>> for pre, fill, node in RenderTree(root):
        ...     print("%s%s" % (pre, node.lines[0]))
        ...     for line in node.lines[1:]:
        ...         print("%s%s" % (fill, line))
        ...
        c0fe
        c0de
        ├── ha
        │   ba
        │   ├── 1
        │   │   2
        │   │   3
        │   └── a
        │       b
        └── Z
        `maxlevel` limits the depth of the tree:
        >>> print(RenderTree(root, maxlevel=2))
        Node('/root', lines=['c0fe', 'c0de'])
        ├── Node('/root/sub0', lines=['ha', 'ba'])
        └── Node('/root/sub1', lines=['Z'])
        The `childiter` is responsible for iterating over child nodes at the
        same level. An reversed order can be achived by using `reversed`.
        >>> for row in RenderTree(root, childiter=reversed):
        ...     print("%s%s" % (row.pre, row.node.name))
        ...
        root
        ├── sub1
        └── sub0
            ├── sub0A
            └── sub0B
        Or writing your own sort function:
        >>> def mysort(items):
        ...     return sorted(items, key=lambda item: item.name)
        ...
        >>> for row in RenderTree(root, childiter=mysort):
        ...     print("%s%s" % (row.pre, row.node.name))
        ...
        root
        ├── sub0
        │   ├── sub0A
        │   └── sub0B
        └── sub1
        :any:`by_attr` simplifies attribute rendering and supports multiline:
        >>> print(RenderTree(root).by_attr())
        root
        ├── sub0
        │   ├── sub0B
        │   └── sub0A
        └── sub1
        >>> print(RenderTree(root).by_attr("lines"))
        c0fe
        c0de
        ├── ha
        │   ba
        │   ├── 1
        │   │   2
        │   │   3
        │   └── a
        │       b
        └── Z
        And can be a function:
        >>> print(RenderTree(root).by_attr(lambda n: " ".join(n.lines)))
        c0fe c0de
        ├── ha ba
        │   ├── 1 2 3
        │   └── a b
        └── Z
        """
        if not isinstance(style, AbstractStyle):
            style = style()
        self.node = node
        self.style = style
        self.childiter = childiter
        self.maxlevel = maxlevel

    def __iter__(self):
        return self.__next(self.node, tuple())

    def __next(self, node, continues, level=0):
        yield RenderTree.__item(node, continues, self.style)
        children = node.children.values()
        level += 1
        if children and (self.maxlevel is None or level < self.maxlevel):
            children = self.childiter(children)
            for child, is_last in _is_last(children):
                for grandchild in self.__next(
                    child, continues + (not is_last,), level=level
                ):
                    yield grandchild

    @staticmethod
    def __item(node, continues, style):
        if not continues:
            return Row("", "", node)
        else:
            items = [style.vertical if cont else style.empty for cont in continues]
            indent = "".join(items[:-1])
            branch = style.cont if continues[-1] else style.end
            pre = indent + branch
            fill = "".join(items)
            return Row(pre, fill, node)

    def __str__(self):
        lines = ["%s%r" % (pre, node) for pre, _, node in self]
        return "\n".join(lines)

    def __repr__(self):
        classname = self.__class__.__name__
        args = [
            repr(self.node),
            "style=%s" % repr(self.style),
            "childiter=%s" % repr(self.childiter),
        ]
        return "%s(%s)" % (classname, ", ".join(args))

    def by_attr(self, attrname="name"):
        """
        Return rendered tree with node attribute `attrname`.
        >>> from anytree import AnyNode, RenderTree
        >>> root = AnyNode(id="root")
        >>> s0 = AnyNode(id="sub0", parent=root)
        >>> s0b = AnyNode(id="sub0B", parent=s0, foo=4, bar=109)
        >>> s0a = AnyNode(id="sub0A", parent=s0)
        >>> s1 = AnyNode(id="sub1", parent=root)
        >>> s1a = AnyNode(id="sub1A", parent=s1)
        >>> s1b = AnyNode(id="sub1B", parent=s1, bar=8)
        >>> s1c = AnyNode(id="sub1C", parent=s1)
        >>> s1ca = AnyNode(id="sub1Ca", parent=s1c)
        >>> print(RenderTree(root).by_attr("id"))
        root
        ├── sub0
        │   ├── sub0B
        │   └── sub0A
        └── sub1
            ├── sub1A
            ├── sub1B
            └── sub1C
                └── sub1Ca
        """

        def get():
            for pre, fill, node in self:
                attr = (
                    attrname(node)
                    if callable(attrname)
                    else getattr(node, attrname, "")
                )
                if isinstance(attr, (list, tuple)):
                    lines = attr
                else:
                    lines = str(attr).split("\n")
                yield "%s%s" % (pre, lines[0])
                for line in lines[1:]:
                    yield "%s%s" % (fill, line)

        return "\n".join(get())


def _is_last(iterable):
    iter_ = iter(iterable)
    try:
        nextitem = next(iter_)
    except StopIteration:
        pass
    else:
        item = nextitem
        while True:
            try:
                nextitem = next(iter_)
                yield item, False
            except StopIteration:
                yield nextitem, True
                break
            item = nextitem
