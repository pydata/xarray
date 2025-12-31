import numpy as np
import pytest

pytest.importorskip("hypothesis")
# isort: split

from hypothesis import given

import xarray.testing.strategies as xrst
from xarray.namedarray.parallelcompat import ChunkManagerEntrypoint


class TestPreserveChunks:
    @given(xrst.shape_and_chunks())
    def test_preserve_all_chunks(
        self, shape_and_chunks: tuple[tuple[int, ...], tuple[int, ...]]
    ) -> None:
        shape, previous_chunks = shape_and_chunks
        typesize = 8
        target = 1024 * 1024

        actual = ChunkManagerEntrypoint.preserve_chunks(
            chunks=("preserve",) * len(shape),
            shape=shape,
            target=target,
            typesize=typesize,
            previous_chunks=previous_chunks,
        )
        for i, chunk in enumerate(actual):
            if chunk != shape[i]:
                assert chunk >= previous_chunks[i]
                assert chunk % previous_chunks[i] == 0
                assert chunk <= shape[i]

        if actual != shape:
            assert np.prod(actual) * typesize >= 0.5 * target

    @pytest.mark.parametrize("first_chunk", [-1, (), 1])
    @given(xrst.shape_and_chunks(min_dims=2))
    def test_preserve_some_chunks(
        self,
        first_chunk: int | tuple[int, ...],
        shape_and_chunks: tuple[tuple[int, ...], tuple[int, ...]],
    ) -> None:
        shape, previous_chunks = shape_and_chunks
        typesize = 4
        target = 2 * 1024 * 1024

        actual = ChunkManagerEntrypoint.preserve_chunks(
            chunks=(first_chunk, *["preserve" for _ in range(len(shape) - 1)]),
            shape=shape,
            target=target,
            typesize=typesize,
            previous_chunks=previous_chunks,
        )
        for i, chunk in enumerate(actual):
            if i == 0:
                if first_chunk == 1:
                    assert chunk == 1
                elif first_chunk == -1:
                    assert chunk == shape[i]
                elif first_chunk == ():
                    assert chunk == previous_chunks[i]
            elif chunk != shape[i]:
                assert chunk >= previous_chunks[i]
                assert chunk % previous_chunks[i] == 0
                assert chunk <= shape[i]

        # if we have more than one chunk, make sure the chunks are big enough
        if actual[1:] != shape[1:]:
            assert np.prod(actual) * typesize >= 0.5 * target
