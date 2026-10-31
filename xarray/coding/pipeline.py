from __future__ import annotations

import copy
from collections.abc import Mapping
from dataclasses import dataclass
from typing import ClassVar, Literal, Self, cast, overload

from xarray.coding.core import Coder
from xarray.core.dataset import Dataset
from xarray.core.datatree import DataTree
from xarray.core.variable import Variable

XarrayObj = DataTree | Dataset | Variable


class _Missing:
    _instance: ClassVar[_Missing | None] = None

    def __init__(self):
        raise TypeError("cannot create new instances")

    @classmethod
    def _instance(cls) -> Self:
        if cls._instance is None:
            cls._instance = cls.__new__(cls)

        return cls._instance


MISSING = _Missing._instance()


def _exactly_one_parameter[T](func_name: str, **kwargs):
    given_parameters = {k: v for k, v in kwargs.items() if v is not MISSING}

    n_given = len(given_parameters)
    if n_given != 1:
        raise TypeError(
            f"Must pass exactly one of the keyword parameters of {func_name!r}."
            f" Got {', '.join(map(repr, sorted(given_parameters.keys())))}."
        )


def _at_most_one_parameter[T](func_name: str, **kwargs):
    given_parameters = {k: v for k, v in kwargs.items() if v is not MISSING}

    n_given = len(given_parameters)
    if n_given > 1:
        raise TypeError(
            f"Must pass at most one of the keyword parameters of {func_name!r}."
            f" Got {', '.join(map(repr, sorted(given_parameters.keys())))}."
        )


class ReorderableDict[K, V](dict[K, V]):
    def index_of(self, name: K) -> int:
        keys = list(self.keys())
        try:
            index = keys.index(name)
        except ValueError:
            raise ValueError(f"unknown key: {name}") from None

        return index

    def insert(self, name: K, value: V, at: int) -> None:
        if name in self:
            raise ValueError(f"duplicate key: {name}")

        if at < 0:
            # clip to 0 to prepend
            at = max(len(self) + at, 0)

        new_keys = list(self.keys())
        new_keys.insert(at, name)

        self[name] = value

        # no builtin sort, so we replace self with the new one
        new = dict(sorted(self.items(), key=lambda it: new_keys.index(it[0])))
        self.clear()
        self.update(new)

    def move(self, name: K, at: int) -> None:
        value = self.pop(name, MISSING)
        if value is MISSING:
            raise KeyError(f"unknown key: {name}")

        self.insert(name, value, at)


@overload
def _encode(coders: dict[str, Coder[DataTree]], obj: DataTree) -> DataTree: ...


@overload
def _encode(coders: dict[str, Coder[Dataset]], obj: Dataset) -> Dataset: ...


@overload
def _encode(coders: dict[str, Coder[Variable]], obj: Variable) -> Variable: ...


def _encode[T: XarrayObj](coders: dict[str, Coder[T]], obj: T) -> T:
    encoded = obj
    for coder in coders.values():
        encoded = coder.encode(encoded)

    return encoded


@overload
def _decode(coders: dict[str, Coder[DataTree]], obj: DataTree) -> DataTree: ...


@overload
def _decode(coders: dict[str, Coder[Dataset]], obj: Dataset) -> Dataset: ...


@overload
def _decode(coders: dict[str, Coder[Variable]], obj: Variable) -> Variable: ...


def _decode[T: XarrayObj](coders: dict[str, Coder[T]], obj: T) -> T:
    decoded = obj
    for coder in coders.values():
        decoded = coder.decode(decoded)

    return decoded


@overload
def _extract_coder(
    coders: Mapping[str, Coder[DataTree]], kind: Literal["datatree"]
) -> dict[str, Coder[DataTree]]: ...


@overload
def _extract_coder(
    coders: Mapping[str, Coder[Dataset]], kind: Literal["dataset"]
) -> dict[str, Coder[Dataset]]: ...


@overload
def _extract_coder(
    coders: Mapping[str, Coder[Variable]], kind: Literal["variable"]
) -> dict[str, Coder[Variable]]: ...


def _extract_coder[T: XarrayObj](
    coders: Mapping[str, Coder[T]], kind: Literal["datatree", "dataset", "variable"]
) -> dict[str, T]:
    return {name: coder for name, coder in coders.items() if coder.kind == kind}


@dataclass
class CoderPipeline[T: XarrayObj]:
    coders: ReorderableDict[str, Coder[T]]

    def __post_init__(self):
        self.coders = ReorderableDict(self.coders)

    def _replace(self, coders: ReorderableDict[str, Coder[T]]) -> Self:
        cls = type(self)
        new = cls.__new__(cls)
        new.coders = coders

        return new

    def insert(
        self,
        name: str,
        coder: Coder[T],
        *,
        before: str | MISSING = MISSING,
        after: str | MISSING = MISSING,
        at: int | MISSING = MISSING,
    ) -> Self:
        """Insert a new coder.

        If none of ``begin``, ``after``, or ``at`` was given, appends at the end.

        Parameters
        ----------
        name : str
            The name of the coder. Must be unique.
        coder : Coder
            The coder to insert.
        before : str, optional
            Insert before this coder. Must exist in the existing coders.
        after : str, optional
            Insert after this coder. Must exist in the existing coders.
        at : int, optional
            Insert at this position.
        """
        _at_most_one_parameter(
            "CoderPipeline.insert", before=before, after=after, at=at
        )

        if name in self.coders:
            raise ValueError(f"Coder {name} already exists.")

        if before is not MISSING:
            at = self.coders.index_of(before)
        elif after is not MISSING:
            at = self.coders.index_of(after) + 1
        elif at is MISSING:
            # append by default
            at = len(self.coders)

        self.coders.insert(name, coder, at)

        return self

    def remove(self, name: str) -> Self:
        new = copy.copy(self.coders)
        result = new.pop(name, MISSING)
        if result is MISSING:
            raise KeyError(f"unknown coder: {name}")

        return self._replace(new)

    def move(
        self,
        name: str,
        *,
        before: str | MISSING = MISSING,
        after: str | MISSING = MISSING,
        at: int | MISSING = MISSING,
    ) -> Self:
        _exactly_one_parameter("CoderPipeline.move", before=before, after=after, at=at)

        if before is not MISSING:
            at = self.coders.index_of(before)
        elif after is not MISSING:
            at = self.coders.index_of(after) + 1

        self.coders.move(name, at)

        return self

    def chain(self, other: Self) -> Self:
        return self._replace(self.coders | other.coders)

    @property
    def datatree_coders(self) -> dict[str, Coder[DataTree]]:
        return {
            name: cast(Coder[DataTree], coder)
            for name, coder in self.coders.items()
            if coder.kind == "datatree"
        }

    @property
    def dataset_coders(self) -> dict[str, Coder[Dataset]]:
        return {
            name: cast(Coder[Dataset], coder)
            for name, coder in self.coders.items()
            if coder.kind == "dataset"
        }

    @property
    def variable_coders(self) -> dict[str, Coder[Variable]]:
        return {
            name: cast(Coder[Variable], coder)
            for name, coder in self.coders.items()
            if coder.kind == "variable"
        }

    def _decode_datatree(self, tree: DataTree) -> DataTree:
        decoded = _decode(self.datatree_coders, tree)

        return decoded.map_over_datasets(self._decode_dataset)

    def _decode_dataset(self, ds: Dataset) -> Dataset:
        decoded = _decode(self.dataset_coders, ds)
        return decoded._replace(
            variables={
                name: self._decode_variable(var)
                for name, var in decoded.variables.items()
            }
        )

    def _decode_variable(self, var: Variable) -> Variable:
        return _decode(self.variable_coders, var)

    @overload
    def decode(self, obj: DataTree) -> DataTree: ...

    @overload
    def decode(self, obj: Dataset) -> Dataset: ...

    @overload
    def decode(self, obj: Variable) -> Variable: ...

    def decode(self, obj: XarrayObj) -> XarrayObj:
        if isinstance(obj, DataTree):
            return self._decode_datatree(obj)
        elif isinstance(obj, Dataset):
            return self._decode_dataset(obj)
        elif isinstance(obj, Variable):
            return self._decode_variable(obj)
        else:
            raise TypeError(f"unknown object type: {type(obj)}")

    def _encode_datatree(self, tree: DataTree) -> DataTree:
        encoded = _encode(self.datatree_coders, tree)

        return encoded.map_over_datasets(self._encode_dataset)

    def _encode_dataset(self, ds: Dataset) -> Dataset:
        encoded = _encode(self.dataset_coders, ds)
        return encoded._replace(
            variables={
                name: self._encode_variable(var)
                for name, var in encoded.variables.items()
            }
        )

    def _encode_variable(self, var: Variable) -> Variable:
        return _encode(self.variable_coders, var)

    @overload
    def encode(self, obj: DataTree) -> DataTree: ...

    @overload
    def encode(self, obj: Dataset) -> Dataset: ...

    @overload
    def encode(self, obj: Variable) -> Variable: ...

    def encode(self, obj: XarrayObj) -> XarrayObj:
        if isinstance(obj, DataTree):
            return self._encode_datatree(obj)
        elif isinstance(obj, Dataset):
            return self._encode_dataset(obj)
        elif isinstance(obj, Variable):
            return self._encode_variable(obj)
        else:
            raise TypeError(f"unknown object type: {type(obj)}")
