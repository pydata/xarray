from typing import Any, ClassVar, Literal

import pytest

from xarray.coding.pipeline import ReorderableDict
from xarray.core.dataset import Dataset


class DummyCoder:
    kind: ClassVar[Literal["dataset"]] = "dataset"

    value: int
    name: str

    def __init__(self, value: int):
        self.value = value
        self.name = f"dummy{value}"

    def to_dict(self) -> dict[str, Any]:
        return {"value": self.value}

    def decode(self, ds: Dataset) -> Dataset:
        new = ds.copy(deep=False)
        coders = new.attrs.setdefault("coders", {})
        coders[self.name] = self.to_dict()

        return new

    def encode(self, ds: Dataset) -> Dataset:
        new = ds.copy(deep=False)
        new.attrs.get("coders", {}).pop(self.name, {})

        return new


@pytest.fixture
def coder_mapping():
    coders = (DummyCoder(value=index) for index in range(5))
    return ReorderableDict({coder.name: coder for coder in coders})


class TestReorderableDict:
    def test_init(self) -> None:
        d = ReorderableDict({"dummy": DummyCoder(value=2)})

        assert isinstance(d, ReorderableDict)
        assert "dummy" in d

    @pytest.mark.parametrize("coder_index", range(5))
    def test_index_of(self, coder_mapping, coder_index) -> None:
        name = f"dummy{coder_index}"

        actual = coder_mapping.index_of(name)
        assert actual == coder_index

    @pytest.mark.parametrize("key", ("unknown", "dummy10"))
    def test_index_of_unknown(self, coder_mapping, key) -> None:
        with pytest.raises(ValueError, match=f"unknown key: {key}"):
            coder_mapping.index_of(key)

    @pytest.mark.parametrize("at", [0, 2, 4, 5, 100, -1, -3, -5, -6, -20])
    def test_insert(self, coder_mapping, at) -> None:
        if at in range(len(coder_mapping) + 1):
            expected = at
        elif at > len(coder_mapping):
            expected = len(coder_mapping)
        elif at < 0:
            at_ = at + len(coder_mapping)
            expected = max(at_, 0)

        coder = DummyCoder(value=5)
        coder_mapping.insert(coder.name, coder, at=at)

        assert coder_mapping.get(coder.name) is coder
        assert coder_mapping.index_of(coder.name) == expected

    @pytest.mark.parametrize(
        ["name", "at"],
        (
            ("dummy1", 5),
            ("dummy3", 0),
            ("dummy4", 3),
        ),
    )
    def test_move(self, coder_mapping, name, at):
        coder = coder_mapping[name]

        expected = list(coder_mapping)
        expected.remove(name)
        expected.insert(at, name)

        coder_mapping.move(name, at)

        assert list(coder_mapping) == expected
        assert coder_mapping.get(name) is coder
