import pytest

pytest.importorskip("hypothesis")

from .. import assert_allclose
from . import base
from .base import strategies

sparse = pytest.importorskip("sparse")


def create(op, shape):
    def convert(arr):
        if arr.ndim == 0:
            return arr

        return sparse.COO.from_numpy(arr)

    return strategies.numpy_array(shape).map(convert)


@pytest.mark.apply_marks(
    {
        "test_reduce": {
            "[cumprod]": pytest.mark.skip(reason="cumprod not implemented by sparse"),
            "[cumsum]": pytest.mark.skip(reason="cumsum not implemented by sparse"),
            "[median]": pytest.mark.skip(reason="median not implemented by sparse"),
            "[std]": pytest.mark.skip(reason="nanstd not implemented by sparse"),
            "[var]": pytest.mark.skip(reason="nanvar not implemented by sparse"),
        }
    }
)
class TestVariableReduceMethods(base.VariableReduceTests):
    @staticmethod
    def create(op, shape):
        return create(op, shape)

    def check_reduce(self, obj, op, *args, **kwargs):
        actual = getattr(obj, op)(*args, **kwargs)

        if isinstance(actual.data, sparse.COO):
            actual = actual.copy(data=actual.data.todense())

        dense = obj.copy(data=obj.data.todense())
        expected = getattr(dense, op)(*args, **kwargs)

        assert_allclose(actual, expected)
