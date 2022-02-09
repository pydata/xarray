from __future__ import annotations

from typing import TYPE_CHECKING, Generic, Hashable, Iterable, cast

import numpy as np

from . import duck_array_ops
from .computation import dot
from .pycompat import is_duck_dask_array
from .types import T_Xarray

_WEIGHTED_REDUCE_DOCSTRING_TEMPLATE = """
    Reduce this {cls}'s data by a weighted ``{fcn}`` along some dimension(s).

    Parameters
    ----------
    dim : str or sequence of str, optional
        Dimension(s) over which to apply the weighted ``{fcn}``.
    skipna : bool, optional
        If True, skip missing values (as marked by NaN). By default, only
        skips missing values for float dtypes; other dtypes either do not
        have a sentinel missing value (int) or skipna=True has not been
        implemented (object, datetime64 or timedelta64).
    keep_attrs : bool, optional
        If True, the attributes (``attrs``) will be copied from the original
        object to the new one.  If False (default), the new object will be
        returned without attributes.

    Returns
    -------
    reduced : {cls}
        New {cls} object with weighted ``{fcn}`` applied to its data and
        the indicated dimension(s) removed.

    Notes
    -----
        Returns {on_zero} if the ``weights`` sum to 0.0 along the reduced
        dimension(s).
    """

_SUM_OF_WEIGHTS_DOCSTRING = """
    Calculate the sum of weights, accounting for missing values in the data.

    Parameters
    ----------
    dim : str or sequence of str, optional
        Dimension(s) over which to sum the weights.
    keep_attrs : bool, optional
        If True, the attributes (``attrs``) will be copied from the original
        object to the new one.  If False (default), the new object will be
        returned without attributes.

    Returns
    -------
    reduced : {cls}
        New {cls} object with the sum of the weights over the given dimension.
    """


if TYPE_CHECKING:
    from .dataarray import DataArray
    from .dataset import Dataset


class Weighted(Generic[T_Xarray]):
    """An object that implements weighted operations.

    You should create a Weighted object by using the ``DataArray.weighted`` or
    ``Dataset.weighted`` methods.

    See Also
    --------
    Dataset.weighted
    DataArray.weighted
    """

    __slots__ = ("obj", "weights")

    def __init__(self, obj: T_Xarray, weights: DataArray):
        """
        Create a Weighted object

        Parameters
        ----------
        obj : DataArray or Dataset
            Object over which the weighted reduction operation is applied.
        weights : DataArray
            An array of weights associated with the values in the obj.
            Each value in the obj contributes to the reduction operation
            according to its associated weight.

        Notes
        -----
        ``weights`` must be a ``DataArray`` and cannot contain missing values.
        Missing values can be replaced by ``weights.fillna(0)``.
        """

        from .dataarray import DataArray

        if not isinstance(weights, DataArray):
            raise ValueError("`weights` must be a DataArray")

        def _weight_check(w):
            # Ref https://github.com/pydata/xarray/pull/4559/files#r515968670
            if duck_array_ops.isnull(w).any():
                raise ValueError(
                    "`weights` cannot contain missing values. "
                    "Missing values can be replaced by `weights.fillna(0)`."
                )
            return w

        if is_duck_dask_array(weights.data):
            # assign to copy - else the check is not triggered
            weights = weights.copy(
                data=weights.data.map_blocks(_weight_check, dtype=weights.dtype),
                deep=False,
            )

        else:
            _weight_check(weights.data)

        self.obj: T_Xarray = obj
        self.weights: DataArray = weights

    def _check_dim(self, dim: Hashable | Iterable[Hashable] | None):
        """raise an error if any dimension is missing"""

        if isinstance(dim, str) or not isinstance(dim, Iterable):
            dims = [dim] if dim else []
        else:
            dims = list(dim)
        missing_dims = set(dims) - set(self.obj.dims) - set(self.weights.dims)
        if missing_dims:
            raise ValueError(
                f"{self.__class__.__name__} does not contain the dimensions: {missing_dims}"
            )

    @staticmethod
    def _reduce(
        da: DataArray,
        weights: DataArray,
        dim: Hashable | Iterable[Hashable] | None = None,
        skipna: bool | None = None,
    ) -> DataArray:
        """reduce using dot; equivalent to (da * weights).sum(dim, skipna)

        for internal use only
        """

        # need to infer dims as we use `dot`
        if dim is None:
            dim = ...

        # need to mask invalid values in da, as `dot` does not implement skipna
        if skipna or (skipna is None and da.dtype.kind in "cfO"):
            da = da.fillna(0.0)

        # `dot` does not broadcast arrays, so this avoids creating a large
        # DataArray (if `weights` has additional dimensions)
        return dot(da, weights, dims=dim)

    def _sum_of_weights(
        self, da: DataArray, dim: Hashable | Iterable[Hashable] | None = None
    ) -> DataArray:
        """Calculate the sum of weights, accounting for missing values"""

        # we need to mask data values that are nan; else the weights are wrong
        mask = da.notnull()

        # bool -> int, because ``xr.dot([True, True], [True, True])`` -> True
        # (and not 2); GH4074
        if self.weights.dtype == bool:
            sum_of_weights = self._reduce(
                mask, self.weights.astype(int), dim=dim, skipna=False
            )
        else:
            sum_of_weights = self._reduce(mask, self.weights, dim=dim, skipna=False)

        # 0-weights are not valid
        valid_weights = sum_of_weights != 0.0

        return sum_of_weights.where(valid_weights)

    def _sum_of_squares(
        self,
        da: DataArray,
        dim: Hashable | Iterable[Hashable] | None = None,
        skipna: bool | None = None,
    ) -> DataArray:
        """Reduce a DataArray by a weighted ``sum_of_squares`` along some dimension(s)."""

        demeaned = da - da.weighted(self.weights).mean(dim=dim)

        return self._reduce((demeaned**2), self.weights, dim=dim, skipna=skipna)

    def _weighted_sum(
        self,
        da: DataArray,
        dim: Hashable | Iterable[Hashable] | None = None,
        skipna: bool | None = None,
    ) -> DataArray:
        """Reduce a DataArray by a weighted ``sum`` along some dimension(s)."""

        return self._reduce(da, self.weights, dim=dim, skipna=skipna)

    def _weighted_mean(
        self,
        da: DataArray,
        dim: Hashable | Iterable[Hashable] | None = None,
        skipna: bool | None = None,
    ) -> DataArray:
        """Reduce a DataArray by a weighted ``mean`` along some dimension(s)."""

        weighted_sum = self._weighted_sum(da, dim=dim, skipna=skipna)

        sum_of_weights = self._sum_of_weights(da, dim=dim)

        return weighted_sum / sum_of_weights

    def _weighted_var(
        self,
        da: DataArray,
        dim: Hashable | Iterable[Hashable] | None = None,
        skipna: bool | None = None,
    ) -> DataArray:
        """Reduce a DataArray by a weighted ``var`` along some dimension(s)."""

        sum_of_squares = self._sum_of_squares(da, dim=dim, skipna=skipna)

        sum_of_weights = self._sum_of_weights(da, dim=dim)

        return sum_of_squares / sum_of_weights

    def _weighted_std(
        self,
        da: DataArray,
        dim: Hashable | Iterable[Hashable] | None = None,
        skipna: bool | None = None,
    ) -> DataArray:
        """Reduce a DataArray by a weighted ``std`` along some dimension(s)."""

        return cast("DataArray", np.sqrt(self._weighted_var(da, dim, skipna)))

    def _implementation(self, func, dim, **kwargs):

        raise NotImplementedError("Use `Dataset.weighted` or `DataArray.weighted`")

    def sum_of_weights(
        self,
        dim: Hashable | Iterable[Hashable] | None = None,
        keep_attrs: bool | None = None,
    ) -> T_Xarray:

        return self._implementation(
            self._sum_of_weights, dim=dim, keep_attrs=keep_attrs
        )

    def sum_of_squares(
        self,
        dim: Hashable | Iterable[Hashable] | None = None,
        skipna: bool | None = None,
        keep_attrs: bool | None = None,
    ) -> T_Xarray:

        return self._implementation(
            self._sum_of_squares, dim=dim, skipna=skipna, keep_attrs=keep_attrs
        )

    def sum(
        self,
        dim: Hashable | Iterable[Hashable] | None = None,
        skipna: bool | None = None,
        keep_attrs: bool | None = None,
    ) -> T_Xarray:

        return self._implementation(
            self._weighted_sum, dim=dim, skipna=skipna, keep_attrs=keep_attrs
        )

    def mean(
        self,
        dim: Hashable | Iterable[Hashable] | None = None,
        skipna: bool | None = None,
        keep_attrs: bool | None = None,
    ) -> T_Xarray:

        return self._implementation(
            self._weighted_mean, dim=dim, skipna=skipna, keep_attrs=keep_attrs
        )

    def var(
        self,
        dim: Hashable | Iterable[Hashable] | None = None,
        skipna: bool | None = None,
        keep_attrs: bool | None = None,
    ) -> T_Xarray:

        return self._implementation(
            self._weighted_var, dim=dim, skipna=skipna, keep_attrs=keep_attrs
        )

    def std(
        self,
        dim: Hashable | Iterable[Hashable] | None = None,
        skipna: bool | None = None,
        keep_attrs: bool | None = None,
    ) -> T_Xarray:

        return self._implementation(
            self._weighted_std, dim=dim, skipna=skipna, keep_attrs=keep_attrs
        )

    def __repr__(self):
        """provide a nice str repr of our Weighted object"""

        klass = self.__class__.__name__
        weight_dims = ", ".join(self.weights.dims)
        return f"{klass} with weights along dimensions: {weight_dims}"


class DataArrayWeighted(Weighted["DataArray"]):
    def _implementation(self, func, dim, **kwargs) -> DataArray:

        self._check_dim(dim)

        dataset = self.obj._to_temp_dataset()
        dataset = dataset.map(func, dim=dim, **kwargs)
        return self.obj._from_temp_dataset(dataset)


class DatasetWeighted(Weighted["Dataset"]):
    def _implementation(self, func, dim, **kwargs) -> Dataset:

        self._check_dim(dim)

        return self.obj.map(func, dim=dim, **kwargs)


def _inject_docstring(cls, cls_name):

    cls.sum_of_weights.__doc__ = _SUM_OF_WEIGHTS_DOCSTRING.format(cls=cls_name)

    cls.sum.__doc__ = _WEIGHTED_REDUCE_DOCSTRING_TEMPLATE.format(
        cls=cls_name, fcn="sum", on_zero="0"
    )

    cls.mean.__doc__ = _WEIGHTED_REDUCE_DOCSTRING_TEMPLATE.format(
        cls=cls_name, fcn="mean", on_zero="NaN"
    )

    cls.sum_of_squares.__doc__ = _WEIGHTED_REDUCE_DOCSTRING_TEMPLATE.format(
        cls=cls_name, fcn="sum_of_squares", on_zero="0"
    )

    cls.var.__doc__ = _WEIGHTED_REDUCE_DOCSTRING_TEMPLATE.format(
        cls=cls_name, fcn="var", on_zero="NaN"
    )

    cls.std.__doc__ = _WEIGHTED_REDUCE_DOCSTRING_TEMPLATE.format(
        cls=cls_name, fcn="std", on_zero="NaN"
    )


_inject_docstring(DataArrayWeighted, "DataArray")
_inject_docstring(DatasetWeighted, "Dataset")
