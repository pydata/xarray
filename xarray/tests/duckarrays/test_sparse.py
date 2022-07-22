import numpy as np
import pytest

from xarray import DataArray, Dataset, Variable

from .. import assert_allclose
from . import base
from .base import strategies

pytest.importorskip("hypothesis")
sparse = pytest.importorskip("sparse")


@pytest.fixture(autouse=True)
def disable_bottleneck():
    from xarray import set_options

    with set_options(use_bottleneck=False):
        yield


def create(op, shape, dtypes):
    def convert(arr):
        if arr.ndim == 0:
            return arr
        # sparse doesn't support float16
        if np.issubdtype(arr.dtype, np.float16):
            return arr

        return sparse.COO.from_numpy(arr)

    return strategies.numpy_array(shape, dtypes).map(convert)


def as_dense(obj):
    if isinstance(obj, Variable) and isinstance(obj.data, sparse.COO):
        new_obj = obj.copy(data=obj.data.todense())
    elif isinstance(obj, DataArray):
        ds = obj._to_temp_dataset()
        dense = as_dense(ds)
        new_obj = obj._from_temp_dataset(dense)
    elif isinstance(obj, Dataset):
        variables = {name: as_dense(var) for name, var in obj.variables.items()}
        coords = {
            name: var for name, var in variables.items() if name in obj._coord_names
        }
        data_vars = {
            name: var for name, var in variables.items() if name not in obj._coord_names
        }

        new_obj = Dataset(coords=coords, data_vars=data_vars, attrs=obj.attrs)
    else:
        new_obj = obj

    return new_obj


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
class TestSparseVariableReduceMethods(base.VariableReduceTests):
    @staticmethod
    def create(op, shape, dtypes):
        return create(op, shape, dtypes)

    def check_reduce(self, obj, op, *args, **kwargs):
        actual = as_dense(getattr(obj, op)(*args, **kwargs))
        expected = getattr(as_dense(obj), op)(*args, **kwargs)

        assert_allclose(actual, expected)


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
class TestSparseDataArrayReduceMethods(base.DataArrayReduceTests):
    @staticmethod
    def create(op, shape, dtypes):
        return create(op, shape, dtypes)

    def check_reduce(self, obj, op, *args, **kwargs):
        actual = as_dense(getattr(obj, op)(*args, **kwargs))
        expected = getattr(as_dense(obj), op)(*args, **kwargs)

        assert_allclose(actual, expected)


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
class TestSparseDatasetReduceMethods(base.DatasetReduceTests):
    @staticmethod
    def create(op, shape, dtypes):
        return create(op, shape, dtypes)

    def check_reduce(self, obj, op, *args, **kwargs):
        actual = as_dense(getattr(obj, op)(*args, **kwargs))
        expected = getattr(as_dense(obj), op)(*args, **kwargs)

        assert_allclose(actual, expected)
