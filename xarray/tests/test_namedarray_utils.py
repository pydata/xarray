from __future__ import annotations

from collections.abc import Hashable

from xarray.namedarray import utils


def test_repr_object():
    obj = utils.ReprObject("foo")
    assert repr(obj) == "foo"
    assert isinstance(obj, Hashable)
    assert not isinstance(obj, str)


def test_repr_object_magic_methods():
    o1 = utils.ReprObject("foo")
    o2 = utils.ReprObject("foo")
    o3 = utils.ReprObject("bar")
    o4 = "foo"
    assert o1 == o2
    assert o1 != o3
    assert o1 != o4
    assert hash(o1) == hash(o2)
    assert hash(o1) != hash(o3)
    assert hash(o1) != hash(o4)
