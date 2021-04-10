import pytest

from . import base
from .base import utils

sparse = pytest.importorskip("sparse")


def create(op):
    return utils.numpy_array.map(sparse.COO.from_numpy)


class TestReduceMethods(base.ReduceMethodTests):
    @staticmethod
    def create(op):
        return create(op)
