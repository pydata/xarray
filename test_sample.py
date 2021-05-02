import pytest


@pytest.mark.parametrize("a", [1, 2])
@pytest.mark.parametrize("b", ["x"])
@pytest.mark.parametrize("c", [5, 10])
def test_hello(a, b, c):
    assert 2 == 2
    print(a, b, c)
