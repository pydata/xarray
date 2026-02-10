from typing import Literal

import numpy as np
import pandas as pd

from xarray.core.dataset import Dataset
from xarray.core.extensions import (
    register_dataset_accessor,
)


@register_dataset_accessor("interval")
class DatasetIntervalAccessor:
    def __init__(self, obj):
        self._obj = obj

    def decode_bounds(
        self, name, *, closed: Literal["left", "right", "both", "neither"] | None = None
    ) -> Dataset:
        """Add a pandas IntervalIndex for indexing along a dimension.

        This function uses the CF conventions on "bounds" variables to detect
        the appropriate intervals to use. The ``"bounds"`` attribute on variable
        ``name`` should link to a second variable that must be present. This 'bounds'
        variable must be 2-dimensional with one of the dimensions being ``name``. The name
        of the other dimension does not matter, but it must be of size 2.
        A pandas IntervalIndex is created for the intervals defined by the bounds variable.

        Parameters
        ----------
        name: Hashable
            Name of the dimension coordinate for which an IntervalIndex is desired.
        closed: {"left", "right", "both", "neither"}, optional
            Which edge of the interval is closed. Must be specified if the ``closed``
            attribute is not present. Note that the CF conventions do not dictate
            how ``closed`` must be recorded. Usage of an attribute named ``'closed'``
            is an Xarray-specific convention.

        Returns
        -------
        Dataset
        """
        if not (bname := self._obj[name].attrs.get("bounds", None)):
            raise ValueError(f"Attribute named 'bounds' not found on variable {name!r}")

        boundsvar = self._obj[bname]
        assert boundsvar.ndim == 2
        (bdim,) = [dim for dim in boundsvar.dims if dim != name]
        closed = closed or boundsvar.attrs.pop("closed")
        assert self._obj.sizes[bdim] == 2
        # TODO: error for cftime
        index = pd.IntervalIndex.from_arrays(
            boundsvar.isel({bdim: 0}), boundsvar.isel({bdim: 1}), closed=closed
        )
        return self._obj.drop_vars([name, boundsvar.name]).assign_coords(
            {name: (name, index, boundsvar.attrs)}
        )

    def encode_bounds(
        self,
        name: str,
        *,
        suffix="bounds",
        which: Literal["left", "mid", "right"] = "mid",
    ) -> Dataset:
        """Encode an IntervalIndex using the CF conventions.

        Intervals are recorded using two variables:
        - a 1D variable named ``'name'`` containing the "central value" (chosen by ``which``), and
        - a 2D 'bounds' variable named ``'{name}_bounds'`` variable containing the left and right values
          of the intervals stacked along a new dimension named ``'bounds'``.

        For convenience, a new `PandasIndex` for the central values is assigned for the
        dimension coordinate ``name``.

        Parameters
        ----------
        name: Hashable
            Name of the dimension coordinate for which an IntervalIndex is desired.
        suffix: str
            Suffix for the newly created bounds variable.
        closed: {"left", "right", "both", "neither"}
            Which edge of the interval is closed. The CF conventions do not dictate
            how ``closed`` must be recorded, so the user is expected to provide it.

        Returns
        -------
        Dataset
        """
        intarray = self._obj[name].data
        bname = f"{name}_{suffix}"
        bvar = xr.DataArray(
            name=bname,
            dims=(name, "bounds"),
            data=np.stack([intarray.left, intarray.right], axis=-1),
        )
        bvar.attrs["closed"] = intarray.closed
        newvar = self._obj[name].copy(data=getattr(self._obj[name].data, which))
        newvar.attrs["bounds"] = bname
        return self._obj.drop_vars(name).assign_coords({name: newvar, bname: bvar})
