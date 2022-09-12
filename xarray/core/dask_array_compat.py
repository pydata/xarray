from __future__ import annotations

import numpy as np

try:
    import dask.array as da

    sliding_window_view = da.lib.stride_tricks.sliding_window_view
except ImportError:
    sliding_window_view = None
