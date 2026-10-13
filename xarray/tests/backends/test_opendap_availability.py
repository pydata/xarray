"""Unit tests for OPeNDAP network-test outage detection (no network needed)."""

from __future__ import annotations

from contextlib import nullcontext
from unittest.mock import patch
from urllib.error import HTTPError, URLError

import pytest

from xarray.tests.backends.conftest import _require_opendap_endpoint


def test_available_opendap_endpoint_is_not_skipped() -> None:
    with patch("xarray.tests.backends.conftest.urlopen", return_value=nullcontext()):
        _require_opendap_endpoint("https://test.opendap.org/test")


@pytest.mark.parametrize("status_code", [502, 503, 504])
def test_opendap_server_outages_are_skipped(status_code: int) -> None:
    error = HTTPError("https://test.opendap.org/test", status_code, "outage", None, None)
    with patch("xarray.tests.backends.conftest.urlopen", side_effect=error):
        with pytest.raises(pytest.skip.Exception, match=f"HTTP {status_code}"):
            _require_opendap_endpoint("https://test.opendap.org/test")


@pytest.mark.parametrize("status_code", [401, 404, 500])
def test_other_http_errors_remain_failures(status_code: int) -> None:
    error = HTTPError("https://test.opendap.org/test", status_code, "error", None, None)
    with patch("xarray.tests.backends.conftest.urlopen", side_effect=error):
        with pytest.raises(HTTPError) as raised:
            _require_opendap_endpoint("https://test.opendap.org/test")
    assert raised.value.code == status_code


@pytest.mark.parametrize("error", [URLError("connection refused"), TimeoutError("timeout")])
def test_connection_failures_are_skipped(error: Exception) -> None:
    with patch("xarray.tests.backends.conftest.urlopen", side_effect=error):
        with pytest.raises(pytest.skip.Exception, match="unreachable"):
            _require_opendap_endpoint("https://test.opendap.org/test")
