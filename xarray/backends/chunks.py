import itertools

import numpy as np

from xarray.core.datatree import Variable


def align_nd_chunks(
    nd_v_chunks: tuple[tuple[int, ...], ...],
    nd_backend_chunks: tuple[tuple[int, ...], ...],
) -> tuple[tuple[int, ...], ...]:
    if len(nd_backend_chunks) != len(nd_v_chunks):
        raise ValueError(
            "The number of dimensions on the backend and the variable must be the same."
        )

    nd_aligned_chunks: list[tuple[int, ...]] = []
    for backend_chunks, v_chunks in zip(nd_backend_chunks, nd_v_chunks, strict=True):
        # Validate that they have the same number of elements
        if sum(backend_chunks) != sum(v_chunks):
            raise ValueError(
                "The number of elements in the backend does not "
                "match the number of elements in the variable. "
                "This inconsistency should never occur at this stage."
            )

        # Validate if the backend_chunks satisfy the condition that all the values
        # excluding the borders are equal
        if len(set(backend_chunks[1:-1])) > 1:
            raise ValueError(
                f"This function currently supports aligning chunks "
                f"only when backend chunks are of uniform size, excluding borders. "
                f"If you encounter this error, please report it—this scenario should never occur "
                f"unless there is an internal misuse. "
                f"Backend chunks: {backend_chunks}"
            )

        # The algorithm assumes that there are always two borders on the
        # Backend and the Array if not, the result is going to be the same
        # as the input, and there is nothing to optimize
        if len(backend_chunks) == 1:
            nd_aligned_chunks.append(backend_chunks)
            continue

        if len(v_chunks) == 1:
            nd_aligned_chunks.append(v_chunks)
            continue

        # Size of the chunk on the backend
        fixed_chunk = max(backend_chunks)

        # The ideal size of the chunks is the maximum of the two; this would avoid
        # that we use more memory than expected
        max_chunk = max(fixed_chunk, *v_chunks)

        # The algorithm assumes that the chunks on this array are aligned except the last one
        # because it can be considered a partial one
        aligned_chunks: list[int] = []

        # For simplicity of the algorithm, let's transform the Array chunks in such a way that
        # we remove the partial chunks. To achieve this, we add artificial data to the borders
        t_v_chunks = list(v_chunks)
        t_v_chunks[0] += fixed_chunk - backend_chunks[0]
        t_v_chunks[-1] += fixed_chunk - backend_chunks[-1]

        # The unfilled_size is the amount of space that has not been filled on the last
        # processed chunk; this is equivalent to the amount of data that would need to be
        # added to a partial Zarr chunk to fill it up to the fixed_chunk size
        unfilled_size = 0

        for v_chunk in t_v_chunks:
            # Ideally, we should try to preserve the original Dask chunks, but this is only
            # possible if the last processed chunk was aligned (unfilled_size == 0)
            ideal_chunk = v_chunk
            if unfilled_size:
                # If that scenario is not possible, the best option is to merge the chunks
                ideal_chunk = v_chunk + aligned_chunks[-1]

            while ideal_chunk:
                if not unfilled_size:
                    # If the previous chunk is filled, let's add a new chunk
                    # of size 0 that will be used on the merging step to simplify the algorithm
                    aligned_chunks.append(0)

                if ideal_chunk > max_chunk:
                    # If the ideal_chunk is bigger than the max_chunk,
                    # we need to increase the last chunk as much as possible
                    # but keeping it aligned, and then add a new chunk
                    max_increase = max_chunk - aligned_chunks[-1]
                    max_increase = (
                        max_increase - (max_increase - unfilled_size) % fixed_chunk
                    )
                    aligned_chunks[-1] += max_increase
                else:
                    # Perfect scenario where the chunks can be merged without any split.
                    aligned_chunks[-1] = ideal_chunk

                ideal_chunk -= aligned_chunks[-1]
                unfilled_size = (
                    fixed_chunk - aligned_chunks[-1] % fixed_chunk
                ) % fixed_chunk

        # Now we have to remove the artificial data added to the borders
        for order in [-1, 1]:
            border_size = fixed_chunk - backend_chunks[::order][0]
            aligned_chunks = aligned_chunks[::order]
            aligned_chunks[0] -= border_size
            t_v_chunks = t_v_chunks[::order]
            t_v_chunks[0] -= border_size
            if (
                len(aligned_chunks) >= 2
                and aligned_chunks[0] + aligned_chunks[1] <= max_chunk
                and aligned_chunks[0] != t_v_chunks[0]
            ):
                # The artificial data added to the border can introduce inefficient chunks
                # on the borders, for that reason, we will check if we can merge them or not
                # Example:
                # backend_chunks = [6, 6, 1]
                # v_chunks = [6, 7]
                # t_v_chunks = [6, 12]
                # The ideal output should preserve the same v_chunks, but the previous loop
                # is going to produce aligned_chunks = [6, 6, 6]
                # And after removing the artificial data, we will end up with aligned_chunks = [6, 6, 1]
                # which is not ideal and can be merged into a single chunk
                aligned_chunks[1] += aligned_chunks[0]
                aligned_chunks = aligned_chunks[1:]

            t_v_chunks = t_v_chunks[::order]
            aligned_chunks = aligned_chunks[::order]

        nd_aligned_chunks.append(tuple(aligned_chunks))

    return tuple(nd_aligned_chunks)


def build_grid_chunks(
    size: int,
    chunk_size: int | tuple[int, ...],
    region: slice | None = None,
) -> tuple[int, ...]:
    if isinstance(chunk_size, (list, tuple)):
        return _build_rectilinear_grid_chunks(chunk_size, region)

    if region is None:
        region = slice(0, size)

    region_start = region.start or 0
    # Generate the zarr chunks inside the region of this dim
    chunks_on_region = [chunk_size - (region_start % chunk_size)]
    if chunks_on_region[0] >= size:
        # This is useful for the scenarios where the chunk_size are bigger
        # than the variable chunks, which can happens when the user specifies
        # the enc_chunks manually.
        return (size,)
    chunks_on_region.extend([chunk_size] * ((size - chunks_on_region[0]) // chunk_size))
    if (size - chunks_on_region[0]) % chunk_size != 0:
        chunks_on_region.append((size - chunks_on_region[0]) % chunk_size)
    return tuple(chunks_on_region)


def _build_rectilinear_grid_chunks(
    chunk_sizes: tuple[int, ...],
    region: slice | None = None,
) -> tuple[int, ...]:
    """Build grid chunks for a rectilinear dimension within a region."""
    if region is None or region == slice(None):
        return tuple(chunk_sizes)

    region_start = region.start or 0
    region_stop = region.stop or sum(chunk_sizes)

    boundaries = [0]
    for cs in chunk_sizes:
        boundaries.append(boundaries[-1] + cs)

    result = []
    for i in range(len(chunk_sizes)):
        chunk_start = boundaries[i]
        chunk_end = boundaries[i + 1]

        if chunk_end <= region_start or chunk_start >= region_stop:
            continue

        effective_start = max(chunk_start, region_start)
        effective_end = min(chunk_end, region_stop)
        result.append(effective_end - effective_start)

    return tuple(result)


def grid_rechunk(
    v: Variable,
    encoding_chunks: tuple[int, ...] | tuple[int | tuple[int, ...], ...],
    region: tuple[slice, ...],
) -> Variable:
    nd_v_chunks = v.chunks
    if not nd_v_chunks:
        return v

    nd_grid_chunks = tuple(
        build_grid_chunks(
            v_size,
            region=interval,
            chunk_size=chunk_size,
        )
        for v_size, chunk_size, interval in zip(
            v.shape, encoding_chunks, region, strict=True
        )
    )

    nd_aligned_chunks = align_nd_chunks(
        nd_v_chunks=nd_v_chunks,
        nd_backend_chunks=nd_grid_chunks,
    )
    v = v.chunk(dict(zip(v.dims, nd_aligned_chunks, strict=True)))
    return v


def _validate_rectilinear_chunk_alignment(
    dask_chunks: tuple[int, ...],
    encoding_chunks: tuple[int, ...],
    axis: int,
    name: str,
    region: slice = slice(None),
) -> None:
    """Validate dask chunks align with rectilinear encoding chunk boundaries."""
    encoding_stops = set(itertools.accumulate(encoding_chunks))
    region_start = region.start or 0
    dask_stops = {region_start + s for s in itertools.accumulate(dask_chunks)}
    # The final stop (total size) always matches — exclude it
    total = sum(encoding_chunks)
    encoding_stops.discard(total)
    dask_stops.discard(total)
    bad = dask_stops - encoding_stops
    if bad:
        raise ValueError(
            f"Specified rectilinear encoding chunks {encoding_chunks!r} for variable "
            f"named {name!r} would overlap multiple Dask chunks on axis {axis}. "
            f"Dask chunk boundaries at positions {sorted(bad)} do not align with "
            f"encoding chunk boundaries at {sorted(encoding_stops)}. "
            "Writing this array in parallel with Dask could lead to corrupted data. "
            "Consider rechunking using `chunk()` or setting `safe_chunks=False`."
        )


def validate_grid_chunks_alignment(
    nd_v_chunks: tuple[tuple[int, ...], ...] | None,
    enc_chunks: tuple[int | tuple[int, ...], ...],
    backend_shape: tuple[int, ...],
    region: tuple[slice, ...],
    allow_partial_chunks: bool,
    name: str,
):
    if nd_v_chunks is None:
        return
    base_error = (
        "Specified Zarr chunks encoding['chunks']={enc_chunks!r} for "
        "variable named {name!r} would overlap multiple Dask chunks. "
        "Please check the Dask chunks at position {v_chunk_pos} and "
        "{v_chunk_pos_next}, on axis {axis}, they are overlapped "
        "on the same Zarr chunk in the region {region}. "
        "Writing this array in parallel with Dask could lead to corrupted data. "
        "To resolve this issue, consider one of the following options: "
        "- Rechunk the array using `chunk()`. "
        "- Modify or delete `encoding['chunks']`. "
        "- Set `safe_chunks=False`. "
        "- Enable automatic chunks alignment with `align_chunks=True`."
    )

    for axis, enc_chunk, v_chunks, interval, size in zip(
        range(len(enc_chunks)),
        enc_chunks,
        nd_v_chunks,
        region,
        backend_shape,
        strict=True,
    ):
        if isinstance(enc_chunk, (list, tuple)):
            # Rectilinear dimension — use boundary-based validation
            _validate_rectilinear_chunk_alignment(
                dask_chunks=v_chunks,
                encoding_chunks=enc_chunk,
                axis=axis,
                name=name,
                region=interval,
            )
            continue

        # Regular dimension — existing validation logic
        chunk_size = enc_chunk
        for i, chunk in enumerate(v_chunks[1:-1]):
            if chunk % chunk_size:
                raise ValueError(
                    base_error.format(
                        v_chunk_pos=i + 1,
                        v_chunk_pos_next=i + 2,
                        v_chunk_size=chunk,
                        axis=axis,
                        name=name,
                        chunk_size=chunk_size,
                        region=interval,
                        enc_chunks=enc_chunks,
                    )
                )

        interval_start = interval.start or 0

        if len(v_chunks) > 1:
            # The first border size is the amount of data that needs to be updated on the
            # first chunk taking into account the region slice.
            first_border_size = chunk_size
            if allow_partial_chunks:
                first_border_size = chunk_size - interval_start % chunk_size

            if (v_chunks[0] - first_border_size) % chunk_size:
                raise ValueError(
                    base_error.format(
                        v_chunk_pos=0,
                        v_chunk_pos_next=0,
                        v_chunk_size=v_chunks[0],
                        axis=axis,
                        name=name,
                        chunk_size=chunk_size,
                        region=interval,
                        enc_chunks=enc_chunks,
                    )
                )

        if not allow_partial_chunks:
            region_stop = interval.stop or size

            error_on_last_chunk = base_error.format(
                v_chunk_pos=len(v_chunks) - 1,
                v_chunk_pos_next=len(v_chunks) - 1,
                v_chunk_size=v_chunks[-1],
                axis=axis,
                name=name,
                chunk_size=chunk_size,
                region=interval,
                enc_chunks=enc_chunks,
            )
            if interval_start % chunk_size:
                # The last chunk which can also be the only one is a partial chunk
                # if it is not aligned at the beginning
                raise ValueError(error_on_last_chunk)

            if np.ceil(region_stop / chunk_size) == np.ceil(size / chunk_size):
                # If the region is covering the last chunk then check
                # if the reminder with the default chunk size
                # is equal to the size of the last chunk
                if v_chunks[-1] % chunk_size != size % chunk_size:
                    raise ValueError(error_on_last_chunk)
            elif v_chunks[-1] % chunk_size:
                raise ValueError(error_on_last_chunk)
