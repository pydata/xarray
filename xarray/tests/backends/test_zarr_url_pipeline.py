"""Tests for Zarr URL pipeline strings (ZEP 8).

xarray turns a string that contains ``|`` and that zarr's parser accepts into a
``zarr.storage.URLPipeline`` (``as_zarr_url_pipeline``); every other value keeps
its meaning. The end-to-end tests register a small test adapter over a
``file:`` root, so they need nothing but zarr's pipeline core.
"""

from __future__ import annotations

import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import xarray as xr
from xarray.backends import _url_pipeline
from xarray.backends._url_pipeline import as_zarr_url_pipeline
from xarray.backends.common import _find_absolute_paths, _normalize_path
from xarray.backends.zarr import _get_mappers
from xarray.tests import (
    assert_identical,
    requires_dask,
    requires_zarr,
    requires_zarr_url_pipeline,
)

SCHEME = "xarray-test.sub"

skip_on_windows = pytest.mark.skipif(
    sys.platform == "win32", reason="POSIX paths; '|' is invalid in Windows paths"
)


def _dataset() -> xr.Dataset:
    return xr.Dataset(
        {"a": (("x", "y"), np.arange(6.0).reshape(3, 2))},
        coords={"x": [10, 20, 30]},
        attrs={"title": "pipeline"},
    )


def _url(path: Path, body: str = "") -> str:
    # "file://" makes the root look remote, which exercises engine guessing
    return f"file://{path.as_posix()}|{SCHEME}:{body}"


def _as_form(url: str, form: str) -> Any:
    from zarr.storage import URLPipeline  # type: ignore[attr-defined,unused-ignore]

    return url if form == "string" else URLPipeline.from_url(url)


@pytest.fixture
def sub_adapter() -> Iterator[str]:
    """Register ``...|xarray-test.sub:<path>``, which descends into ``<path>``."""
    from zarr.abc.url_pipeline import (  # type: ignore[import-not-found,unused-ignore]
        URLPipelineAdapter,
    )
    from zarr.registry import (  # type: ignore[attr-defined,unused-ignore]
        _url_adapter_registry,
        register_url_adapter,
    )
    from zarr.storage import StorePath

    class SubAdapter(URLPipelineAdapter):  # type: ignore[misc,unused-ignore]
        @classmethod
        async def open_pipeline_segment(cls, segment: Any, context: Any) -> StorePath:
            preceding = await context.resolve_preceding()
            parts = (preceding.path.strip("/"), segment.body.strip("/"))
            return StorePath(preceding.store, "/".join(p for p in parts if p))

    register_url_adapter(SCHEME, SubAdapter)
    yield SCHEME
    _url_adapter_registry.pop(SCHEME, None)


form = pytest.mark.parametrize("form", ["string", "object"])


# --- the helper --------------------------------------------------------------


@requires_zarr_url_pipeline
@pytest.mark.parametrize(
    "url",
    [
        "s3://bucket/repo|icechunk://tag.v1/",
        "file:///data/x.zip|zip:inner|zarr3:",
        "file:/data/x|zarr3:",
        "memory:x|xarray-test.sub:a",
    ],
)
def test_pipeline_string_becomes_url_pipeline(url: str) -> None:
    from zarr.storage import URLPipeline  # type: ignore[attr-defined,unused-ignore]

    # parsing loads no adapter, so the adapters need not be installed
    result = as_zarr_url_pipeline(url)
    assert isinstance(result, URLPipeline)
    assert result == URLPipeline.from_url(url)


@requires_zarr
@pytest.mark.parametrize(
    "value",
    [
        None,
        1,
        b"memory:|zarr3:",
        Path("a|b"),
        "plain/path.zarr",
        "s3://bucket/data.zarr",
        pytest.param("a|b", marks=skip_on_windows, id="relative"),
        pytest.param("/abs/a|b.zarr", marks=skip_on_windows, id="absolute"),
        pytest.param("~/a|b", marks=skip_on_windows, id="home"),
        pytest.param("C:\\x|y", id="windows-drive"),
        pytest.param("1bad://x|zarr3:", id="invalid-scheme"),
        pytest.param(" s3://bucket/x|zarr3:", id="whitespace"),
        pytest.param("s3://bucket/x#frag|zarr3:", id="fragment"),
    ],
)
def test_non_pipelines_pass_through(value: object) -> None:
    assert as_zarr_url_pipeline(value) is value


@requires_zarr_url_pipeline
def test_url_pipeline_object_passes_through() -> None:
    from zarr.storage import URLPipeline  # type: ignore[attr-defined,unused-ignore]

    pipeline = URLPipeline.from_url("memory:|zarr3:")
    assert as_zarr_url_pipeline(pipeline) is pipeline


@requires_zarr
def test_noop_without_zarr_url_pipelines(monkeypatch: pytest.MonkeyPatch) -> None:
    # zarr without URL pipelines: the helper must not even import zarr's modules
    monkeypatch.setattr(_url_pipeline, "_has_url_pipeline", lambda: False)
    monkeypatch.setitem(sys.modules, "zarr.errors", None)
    monkeypatch.setitem(sys.modules, "zarr.storage", None)
    url = "s3://bucket/repo|icechunk://tag.v1/"
    assert as_zarr_url_pipeline(url) is url


@requires_zarr_url_pipeline
def test_normalize_path_leaves_pipelines_alone() -> None:
    url = "file:/data/x|zarr3:"  # not a "remote" URI to _normalize_path
    assert _normalize_path(url) == url
    assert _find_absolute_paths("file:///data/*.zarr|zarr3:") == [
        "file:///data/*.zarr|zarr3:"
    ]


# --- end to end --------------------------------------------------------------


@requires_zarr_url_pipeline
@form
@pytest.mark.parametrize("engine", [None, "zarr", "open_zarr"])
def test_open(tmp_path: Path, sub_adapter: str, form: str, engine: str | None) -> None:
    expected = _dataset()
    # no "zarr" in the path, which would steer engine guessing on its own
    expected.to_zarr(tmp_path / "data")
    store = _as_form(_url(tmp_path / "data"), form)
    if engine == "open_zarr":
        actual = xr.open_zarr(store)
    else:
        # without engine=, zarr is guessed for a pipeline string and object
        actual = xr.open_dataset(store, engine=engine)
    with actual:
        assert_identical(actual, expected)


@requires_zarr_url_pipeline
@form
def test_to_zarr(tmp_path: Path, sub_adapter: str, form: str) -> None:
    expected = _dataset()
    # the pipeline addresses a node below the store's root, where to_zarr
    # doesn't consolidate (see #11835), so don't use consolidated metadata
    expected.to_zarr(
        _as_form(_url(tmp_path / "data.zarr", "nested"), form), consolidated=False
    )
    with xr.open_zarr(
        tmp_path / "data.zarr", group="nested", consolidated=False
    ) as actual:
        assert_identical(actual, expected)


@requires_zarr_url_pipeline
@form
def test_datatree_round_trip(tmp_path: Path, sub_adapter: str, form: str) -> None:
    expected = xr.DataTree.from_dict({"/": _dataset(), "/child": _dataset()})
    store = _as_form(_url(tmp_path / "tree.zarr"), form)
    expected.to_zarr(store)
    # zarr resolved the pipeline, rather than writing to a path with "|" in it
    assert [p.name for p in tmp_path.iterdir()] == ["tree.zarr"]
    with xr.open_datatree(store, engine="zarr") as actual:
        xr.testing.assert_identical(actual, expected)


@requires_zarr_url_pipeline
@requires_dask
def test_open_mfdataset(tmp_path: Path, sub_adapter: str) -> None:
    expected = _dataset()
    expected.isel(x=[0]).to_zarr(tmp_path / "a.zarr")
    expected.isel(x=[1, 2]).to_zarr(tmp_path / "b.zarr")
    urls = [_url(tmp_path / "a.zarr"), _url(tmp_path / "b.zarr")]
    with xr.open_mfdataset(urls, engine="zarr", combine="by_coords") as actual:
        assert_identical(actual, expected)
    # a pipeline is one node: "*" in it is not a glob pattern
    with pytest.raises(FileNotFoundError):
        xr.open_mfdataset(_url(tmp_path / "*.zarr"), engine="zarr")


@requires_zarr_url_pipeline
@form
def test_to_zarr_accepts_storage_options(form: str) -> None:
    url = "s3://bucket/repo|icechunk://tag.v1/"
    kwargs, mapper = _get_mappers(
        storage_options={"anon": True}, store=_as_form(url, form)
    )
    assert kwargs == {"storage_options": {"anon": True}}
    assert mapper == as_zarr_url_pipeline(url)


@requires_zarr
@skip_on_windows
def test_local_path_with_pipe(tmp_path: Path) -> None:
    expected = _dataset()
    path = str(tmp_path / "a|b.zarr")
    expected.to_zarr(path)
    with xr.open_zarr(path) as actual:
        assert_identical(actual, expected)
