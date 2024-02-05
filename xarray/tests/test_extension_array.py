from collections.abc import Sequence

import numpy as np
import pandas as pd
import pytest

from xarray.core.duck_array_ops import (
    ExtensionDuckArray,
    __extension_duck_array__broadcast,
    __extension_duck_array__concatenate,
    __extension_duck_array__where,
)
from xarray.tests import requires_plum


@pytest.fixture
def categorical1():
    return pd.Categorical(["cat1", "cat2", "cat2", "cat1", "cat2"])


@pytest.fixture
def categorical2():
    return pd.Categorical(["cat2", "cat1", "cat2", "cat3", "cat1"])


@pytest.fixture
def int1():
    return pd.arrays.IntegerArray(
        np.array([1, 2, 3, 4, 5]), np.array([True, False, False, True, True])
    )


@pytest.fixture
def int2():
    return pd.arrays.IntegerArray(
        np.array([6, 7, 8, 9, 10]), np.array([True, True, False, True, False])
    )


@__extension_duck_array__concatenate.dispatch
def _(arrays: Sequence[pd.arrays.IntegerArray], axis: int = 0, out=None):
    values = np.concatenate(arrays)
    mask = np.isnan(values)
    values = values.astype("int8")
    return pd.arrays.IntegerArray(values, mask)


@requires_plum
def test_where_all_categoricals(categorical1, categorical2):
    assert (
        __extension_duck_array__where(
            np.array([True, False, True, False, False]), categorical1, categorical2
        )
        == pd.Categorical(["cat1", "cat1", "cat2", "cat3", "cat1"])
    ).all()


@requires_plum
def test_where_drop_categoricals(categorical1, categorical2):
    assert (
        __extension_duck_array__where(
            np.array([False, True, True, False, True]), categorical1, categorical2
        ).remove_unused_categories()
        == pd.Categorical(["cat2", "cat2", "cat2", "cat3", "cat2"])
    ).all()


@requires_plum
def test_broadcast_to_categorical(categorical1):
    with pytest.raises(NotImplementedError):
        __extension_duck_array__broadcast(categorical1, (5, 2))


@requires_plum
def test_broadcast_to_same_categorical(categorical1):
    assert (__extension_duck_array__broadcast(categorical1, (5,)) == categorical1).all()


@requires_plum
def test_concategorical_categorical(categorical1, categorical2):
    assert (
        __extension_duck_array__concatenate([categorical1, categorical2])
        == type(categorical1)._concat_same_type((categorical1, categorical2))
    ).all()


@requires_plum
def test_integer_array_register_concatenate(int1, int2):
    assert (
        __extension_duck_array__concatenate([int1, int2])
        == type(int1)._concat_same_type((int1, int2))
    ).all()


def test_duck_extension_array_equality(categorical1, int1):
    int_duck_array = ExtensionDuckArray(int1)
    categorical_duck_array = ExtensionDuckArray(categorical1)
    assert (int_duck_array != categorical_duck_array).all()
    assert (categorical_duck_array == categorical1).all()
    assert (int1[0:2] == int_duck_array[0:2]).all()


def test_duck_extension_array_repr(int1):
    int_duck_array = ExtensionDuckArray(int1)
    assert repr(int1) in repr(int_duck_array)


def test_duck_extension_array_attr(int1):
    int_duck_array = ExtensionDuckArray(int1)
    assert (~int_duck_array.fillna(10)).all()
