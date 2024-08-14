from __future__ import annotations

import math
from numbers import Number

import numpy as np

from xarray.vendor.dask.utils import is_integer, parse_bytes
from xarray.vendor.toolz.itertoolz import frequencies

unknown_chunk_message = (
    "\n\n"
    "A possible solution: "
    "https://docs.dask.org/en/latest/array-chunks.html#unknown-chunks\n"
    "Summary: to compute chunks sizes, use\n\n"
    "   x.compute_chunk_sizes()  # for Dask Array `x`\n"
    "   ddf.to_dask_array(lengths=True)  # for Dask DataFrame `ddf`"
)


def blockdims_from_blockshape(shape, chunks):
    """

    >>> blockdims_from_blockshape((10, 10), (4, 3))
    ((4, 4, 2), (3, 3, 3, 1))
    >>> blockdims_from_blockshape((10, 0), (4, 0))
    ((4, 4, 2), (0,))
    """
    if chunks is None:
        raise TypeError("Must supply chunks= keyword argument")
    if shape is None:
        raise TypeError("Must supply shape= keyword argument")
    if np.isnan(sum(shape)) or np.isnan(sum(chunks)):
        raise ValueError(
            f"Array chunk sizes are unknown. shape: {shape}, chunks: {chunks}{unknown_chunk_message}"
        )
    if not all(map(is_integer, chunks)):
        raise ValueError("chunks can only contain integers.")
    if not all(map(is_integer, shape)):
        raise ValueError("shape can only contain integers.")
    shape = tuple(map(int, shape))
    chunks = tuple(map(int, chunks))
    return tuple(
        ((bd,) * (d // bd) + ((d % bd,) if d % bd else ()) if d else (0,))
        for d, bd in zip(shape, chunks)
    )


CHUNKS_NONE_ERROR_MESSAGE = """
You must specify a chunks= keyword argument.
This specifies the chunksize of your array blocks.

See the following documentation page for details:
  https://docs.dask.org/en/latest/array-creation.html#chunks
""".strip()


def normalize_chunks(chunks, shape=None, limit=None, dtype=None, previous_chunks=None):
    """Normalize chunks to tuple of tuples

    This takes in a variety of input types and information and produces a full
    tuple-of-tuples result for chunks, suitable to be passed to Array or
    rechunk or any other operation that creates a Dask array.

    Parameters
    ----------
    chunks: tuple, int, dict, or string
        The chunks to be normalized.  See examples below for more details
    shape: Tuple[int]
        The shape of the array
    limit: int (optional)
        The maximum block size to target in bytes,
        if freedom is given to choose
    dtype: np.dtype
    previous_chunks: Tuple[Tuple[int]] optional
        Chunks from a previous array that we should use for inspiration when
        rechunking auto dimensions.  If not provided but auto-chunking exists
        then auto-dimensions will prefer square-like chunk shapes.

    Examples
    --------
    Specify uniform chunk sizes

    >>> from dask.array.core import normalize_chunks
    >>> normalize_chunks((2, 2), shape=(5, 6))
    ((2, 2, 1), (2, 2, 2))

    Also passes through fully explicit tuple-of-tuples

    >>> normalize_chunks(((2, 2, 1), (2, 2, 2)), shape=(5, 6))
    ((2, 2, 1), (2, 2, 2))

    Cleans up lists to tuples

    >>> normalize_chunks([[2, 2], [3, 3]])
    ((2, 2), (3, 3))

    Expands integer inputs 10 -> (10, 10)

    >>> normalize_chunks(10, shape=(30, 5))
    ((10, 10, 10), (5,))

    Expands dict inputs

    >>> normalize_chunks({0: 2, 1: 3}, shape=(6, 6))
    ((2, 2, 2), (3, 3))

    The values -1 and None get mapped to full size

    >>> normalize_chunks((5, -1), shape=(10, 10))
    ((5, 5), (10,))

    Use the value "auto" to automatically determine chunk sizes along certain
    dimensions.  This uses the ``limit=`` and ``dtype=`` keywords to
    determine how large to make the chunks.  The term "auto" can be used
    anywhere an integer can be used.  See array chunking documentation for more
    information.

    >>> normalize_chunks(("auto",), shape=(20,), limit=5, dtype="uint8")
    ((5, 5, 5, 5),)

    You can also use byte sizes (see :func:`dask.utils.parse_bytes`) in place of
    "auto" to ask for a particular size

    >>> normalize_chunks("1kiB", shape=(2000,), dtype="float32")
    ((256, 256, 256, 256, 256, 256, 256, 208),)

    Respects null dimensions

    >>> normalize_chunks((), shape=(0, 0))
    ((0,), (0,))
    """
    if dtype and not isinstance(dtype, np.dtype):
        dtype = np.dtype(dtype)
    if chunks is None:
        raise ValueError(CHUNKS_NONE_ERROR_MESSAGE)
    if isinstance(chunks, list):
        chunks = tuple(chunks)
    if isinstance(chunks, (Number, str)):
        chunks = (chunks,) * len(shape)
    if isinstance(chunks, dict):
        chunks = tuple(chunks.get(i, None) for i in range(len(shape)))
    if isinstance(chunks, np.ndarray):
        chunks = chunks.tolist()
    if not chunks and shape and all(s == 0 for s in shape):
        chunks = ((0,),) * len(shape)

    if (
        shape
        and len(shape) == 1
        and len(chunks) > 1
        and all(isinstance(c, (Number, str)) for c in chunks)
    ):
        chunks = (chunks,)

    if shape and len(chunks) != len(shape):
        raise ValueError(
            "Chunks and shape must be of the same length/dimension. "
            f"Got chunks={chunks}, shape={shape}"
        )
    if -1 in chunks or None in chunks:
        chunks = tuple(s if c == -1 or c is None else c for c, s in zip(chunks, shape))

    # If specifying chunk size in bytes, use that value to set the limit.
    # Verify there is only one consistent value of limit or chunk-bytes used.
    for c in chunks:
        if isinstance(c, str) and c != "auto":
            parsed = parse_bytes(c)
            if limit is None:
                limit = parsed
            elif parsed != limit:
                raise ValueError(
                    "Only one consistent value of limit or chunk is allowed."
                    f"Used {parsed} != {limit}"
                )
    # Substitute byte limits with 'auto' now that limit is set.
    chunks = tuple("auto" if isinstance(c, str) and c != "auto" else c for c in chunks)

    if any(c == "auto" for c in chunks):
        chunks = auto_chunks(chunks, shape, limit, dtype, previous_chunks)

    if shape is not None:
        chunks = tuple(c if c not in {None, -1} else s for c, s in zip(chunks, shape))

    if chunks and shape is not None:
        chunks = sum(
            (
                (
                    blockdims_from_blockshape((s,), (c,))
                    if not isinstance(c, (tuple, list))
                    else (c,)
                )
                for s, c in zip(shape, chunks)
            ),
            (),
        )
    for c in chunks:
        if not c:
            raise ValueError(
                "Empty tuples are not allowed in chunks. Express "
                "zero length dimensions with 0(s) in chunks"
            )

    if shape is not None:
        if len(chunks) != len(shape):
            raise ValueError(
                "Input array has %d dimensions but the supplied "
                "chunks has only %d dimensions" % (len(shape), len(chunks))
            )
        if not all(
            c == s or (math.isnan(c) or math.isnan(s))
            for c, s in zip(map(sum, chunks), shape)
        ):
            raise ValueError(
                "Chunks do not add up to shape. " f"Got chunks={chunks}, shape={shape}"
            )

    return tuple(
        tuple(int(x) if not math.isnan(x) else np.nan for x in c) for c in chunks
    )


def _compute_multiplier(limit: int, dtype, largest_block: int, result):
    """
    Utility function for auto_chunk, to fin how much larger or smaller the ideal
    chunk size is relative to what we have now.
    """
    return (
        limit
        / dtype.itemsize
        / largest_block
        / math.prod(r for r in result.values() if r)
    )


def auto_chunks(chunks, shape, limit, dtype, previous_chunks=None):
    """Determine automatic chunks

    This takes in a chunks value that contains ``"auto"`` values in certain
    dimensions and replaces those values with concrete dimension sizes that try
    to get chunks to be of a certain size in bytes, provided by the ``limit=``
    keyword.  If multiple dimensions are marked as ``"auto"`` then they will
    all respond to meet the desired byte limit, trying to respect the aspect
    ratio of their dimensions in ``previous_chunks=``, if given.

    Parameters
    ----------
    chunks: Tuple
        A tuple of either dimensions or tuples of explicit chunk dimensions
        Some entries should be "auto"
    shape: Tuple[int]
    limit: int, str
        The maximum allowable size of a chunk in bytes
    previous_chunks: Tuple[Tuple[int]]

    See also
    --------
    normalize_chunks: for full docstring and parameters
    """
    if previous_chunks is not None:
        previous_chunks = tuple(
            c if isinstance(c, tuple) else (c,) for c in previous_chunks
        )
    chunks = list(chunks)

    autos = {i for i, c in enumerate(chunks) if c == "auto"}
    if not autos:
        return tuple(chunks)

    if limit is None:
        limit = "128MiB"  # config.get("array.chunk-size")
    if isinstance(limit, str):
        limit = parse_bytes(limit)

    if dtype is None:
        raise TypeError("dtype must be known for auto-chunking")

    if dtype.hasobject:
        raise NotImplementedError(
            "Can not use auto rechunking with object dtype. "
            "We are unable to estimate the size in bytes of object data"
        )

    for x in tuple(chunks) + tuple(shape):
        if (
            isinstance(x, Number)
            and np.isnan(x)
            or isinstance(x, tuple)
            and np.isnan(x).any()
        ):
            raise ValueError(
                "Can not perform automatic rechunking with unknown "
                f"(nan) chunk sizes.{unknown_chunk_message}"
            )

    limit = max(1, limit)

    largest_block = math.prod(
        cs if isinstance(cs, Number) else max(cs) for cs in chunks if cs != "auto"
    )

    if previous_chunks:
        # Base ideal ratio on the median chunk size of the previous chunks
        result = {a: np.median(previous_chunks[a]) for a in autos}

        ideal_shape = []
        for i, s in enumerate(shape):
            chunk_frequencies = frequencies(previous_chunks[i])
            mode, count = max(chunk_frequencies.items(), key=lambda kv: kv[1])
            if mode > 1 and count >= len(previous_chunks[i]) / 2:
                ideal_shape.append(mode)
            else:
                ideal_shape.append(s)

        # How much larger or smaller the ideal chunk size is relative to what we have now
        multiplier = _compute_multiplier(limit, dtype, largest_block, result)

        last_multiplier = 0
        last_autos = set()
        while (
            multiplier != last_multiplier or autos != last_autos
        ):  # while things change
            last_multiplier = multiplier  # record previous values
            last_autos = set(autos)  # record previous values

            # Expand or contract each of the dimensions appropriately
            for a in sorted(autos):
                if ideal_shape[a] == 0:
                    result[a] = 0
                    continue
                proposed = result[a] * multiplier ** (1 / len(autos))
                if proposed > shape[a]:  # we've hit the shape boundary
                    autos.remove(a)
                    largest_block *= shape[a]
                    chunks[a] = shape[a]
                    del result[a]
                else:
                    result[a] = round_to(proposed, ideal_shape[a])

            # recompute how much multiplier we have left, repeat
            multiplier = _compute_multiplier(limit, dtype, largest_block, result)

        for k, v in result.items():
            chunks[k] = v
        return tuple(chunks)

    else:
        # Check if dtype.itemsize is greater than 0
        if dtype.itemsize == 0:
            raise ValueError(
                "auto-chunking with dtype.itemsize == 0 is not supported, please pass in `chunks` explicitly"
            )
        size = (limit / dtype.itemsize / largest_block) ** (1 / len(autos))
        small = [i for i in autos if shape[i] < size]
        if small:
            for i in small:
                chunks[i] = (shape[i],)
            return auto_chunks(chunks, shape, limit, dtype)

        for i in autos:
            chunks[i] = round_to(size, shape[i])

        return tuple(chunks)


def round_to(c, s):
    """Return a chunk dimension that is close to an even multiple or factor

    We want values for c that are nicely aligned with s.

    If c is smaller than s we use the original chunk size and accept an
    uneven chunk at the end.

    If c is larger than s then we want the largest multiple of s that is still
    smaller than c.
    """
    if c <= s:
        return max(1, int(c))
    else:
        return c // s * s
