import pytest

from . import requires_dask


@pytest.fixture(params=["numpy", pytest.param("dask", marks=requires_dask)])
def backend(request):
    return request.param
