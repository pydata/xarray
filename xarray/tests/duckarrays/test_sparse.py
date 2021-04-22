import pytest

from .. import assert_allclose
from . import base
from .base import utils

sparse = pytest.importorskip("sparse")


def create(op):
    def convert(arr):
        if arr.ndim == 0:
            return arr

        return sparse.COO.from_numpy(arr)

    return utils.numpy_array.map(convert)


@pytest.mark.apply_marks(
    {"test_reduce": pytest.mark.skip(reason="sparse times out on the first call")}
)
class TestVariableReduceMethods(base.VariableReduceTests):
    @staticmethod
    def create(op):
        return create(op)

    def check_reduce(self, obj, op, *args, **kwargs):
        actual = getattr(obj, op)(*args, **kwargs)

        if isinstance(actual.data, sparse.COO):
            actual = actual.copy(data=actual.data.todense())

        dense = obj.copy(data=obj.data.todense())
        expected = getattr(dense, op)(*args, **kwargs)

        assert_allclose(actual, expected)
