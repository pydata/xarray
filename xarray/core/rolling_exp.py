from typing import TYPE_CHECKING, Generic, Hashable, Mapping, Optional, TypeVar

import numpy as np

from .options import _get_keep_attrs
from .pdcompat import count_not_none
from .pycompat import is_duck_dask_array

if TYPE_CHECKING:
    from .dataarray import DataArray  # noqa: F401
    from .dataset import Dataset  # noqa: F401

T_DSorDA = TypeVar("T_DSorDA", "DataArray", "Dataset")


def _get_alpha(com=None, span=None, halflife=None, alpha=None):
    # pandas defines in terms of com (converting to alpha in the algo)
    # so use its function to get a com and then convert to alpha

    com = _get_center_of_mass(com, span, halflife, alpha)
    return 1 / (1 + com)


def move_exp_nanmean(array, *, axis, alpha):
    if is_duck_dask_array(array):
        raise TypeError("rolling_exp is not currently support for dask-like arrays")
    import numbagg

    if axis == ():
        return array.astype(np.float64)
    else:
        return numbagg.move_exp_nanmean(array, axis=axis, alpha=alpha)


def _get_center_of_mass(comass, span, halflife, alpha):
    """
    Vendored from pandas.core.window.common._get_center_of_mass

    See licenses/PANDAS_LICENSE for the function's license
    """
    valid_count = count_not_none(comass, span, halflife, alpha)
    if valid_count > 1:
        raise ValueError("comass, span, halflife, and alpha are mutually exclusive")

    # Convert to center of mass; domain checks ensure 0 < alpha <= 1
    if comass is not None:
        if comass < 0:
            raise ValueError("comass must satisfy: comass >= 0")
    elif span is not None:
        if span < 1:
            raise ValueError("span must satisfy: span >= 1")
        comass = (span - 1) / 2.0
    elif halflife is not None:
        if halflife <= 0:
            raise ValueError("halflife must satisfy: halflife > 0")
        decay = 1 - np.exp(np.log(0.5) / halflife)
        comass = 1 / decay - 1
    elif alpha is not None:
        if alpha <= 0 or alpha > 1:
            raise ValueError("alpha must satisfy: 0 < alpha <= 1")
        comass = (1.0 - alpha) / alpha
    else:
        raise ValueError("Must pass one of comass, span, halflife, or alpha")

    return float(comass)


class RollingExp(Generic[T_DSorDA]):
    """
    Exponentially-weighted moving window object.
    Similar to EWM in pandas

    Parameters
    ----------
    obj : Dataset or DataArray
        Object to window.
    windows : mapping of hashable to int
        A mapping from the name of the dimension to create the rolling
        exponential window along (e.g. `time`) to the size of the moving window.
    window_type : {"span", "com", "halflife", "alpha"}, default: "span"
        The format of the previously supplied window. Each is a simple
        numerical transformation of the others. Described in detail:
        https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.ewm.html

    Returns
    -------
    RollingExp : type of input argument
    """

    def __init__(
        self,
        obj: T_DSorDA,
        windows: Mapping[Hashable, int],
        window_type: str = "span",
    ):
        self.obj: T_DSorDA = obj
        dim, window = next(iter(windows.items()))
        self.dim = dim
        self.alpha = _get_alpha(**{window_type: window})

    def mean(self, keep_attrs: Optional[bool] = None) -> T_DSorDA:
        """
        Exponentially weighted moving average

        Parameters
        ----------
        keep_attrs : bool, default: None
            If True, the attributes (``attrs``) will be copied from the original
            object to the new one. If False, the new object will be returned
            without attributes. If None uses the global default.

        Examples
        --------
        >>> da = xr.DataArray([1, 1, 2, 2, 2], dims="x")
        >>> da.rolling_exp(x=2, window_type="span").mean()
        <xarray.DataArray (x: 5)>
        array([1.        , 1.        , 1.69230769, 1.9       , 1.96694215])
        Dimensions without coordinates: x
        """

        if keep_attrs is None:
            keep_attrs = _get_keep_attrs(default=True)

        return self.obj.reduce(
            move_exp_nanmean, dim=self.dim, alpha=self.alpha, keep_attrs=keep_attrs
        )
