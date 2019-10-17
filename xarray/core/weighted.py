from .computation import where, dot
from typing import TYPE_CHECKING, Hashable, Iterable, Optional, Tuple, Union, overload

if TYPE_CHECKING:
    from .dataarray import DataArray, Dataset

_WEIGHTED_REDUCE_DOCSTRING_TEMPLATE = """
    Reduce this {cls}'s data by a weighted `{fcn}` along some dimension(s).

    Parameters
    ----------
    dim : str or sequence of str, optional
        Dimension(s) over which to apply the weighted `{fcn}`.
    axis : int or sequence of int, optional
        Axis(es) over which to apply the weighted `{fcn}`. Only one of the
        'dim' and 'axis' arguments can be supplied. If neither are supplied,
        then the weighted `{fcn}` is calculated over all axes.
    skipna : bool, optional
        If True, skip missing values (as marked by NaN). By default, only
        skips missing values for float dtypes; other dtypes either do not
        have a sentinel missing value (int) or skipna=True has not been
        implemented (object, datetime64 or timedelta64).
        Note: Missing values in the weights are replaced with 0 (i.e. no
        weight).
    keep_attrs : bool, optional
        If True, the attributes (`attrs`) will be copied from the original
        object to the new one.  If False (default), the new object will be
        returned without attributes.
    **kwargs : dict
        Additional keyword arguments passed on to the appropriate array
        function for calculating `{fcn}` on this object's data.

    Returns
    -------
    reduced : {cls}
        New {cls} object with weighted `{fcn}` applied to its data and
        the indicated dimension(s) removed.
    """

_SUM_OF_WEIGHTS_DOCSTRING = """
    Calcualte the sum of weights, accounting for missing values

    Parameters
    ----------
    dim : str or sequence of str, optional
        Dimension(s) over which to sum the weights.
    axis : int or sequence of int, optional
        Axis(es) over which to sum the weights. Only one of the 'dim' and
        'axis' arguments can be supplied. If neither are supplied, then
        the weights are summed over all axes.
    
    Returns
    -------
    reduced : {cls}
        New {cls} object with the sum of the weights over the given dimension.


    """


# functions for weighted operations for one DataArray
# NOTE: weights must not contain missing values (this is taken care of in the
# DataArrayWeighted and DatasetWeighted cls)


def _maybe_get_all_dims(
    dims: Optional[Union[Hashable, Iterable[Hashable]]], dims1: Tuple[Hashable, ...], dims2: Tuple[Hashable, ...]
):
    """ the union of all dimensions

    `dims=None` behaves differently in `dot` and `sum`, so we have to apply
    `dot` over the union of the dimensions

    """

    if dims is None:
        dims = set(dims1) | set(dims2)

    return dims


def _sum_of_weights(
    da: "DataArray",
    weights: "DataArray",
    dim: Optional[Union[Hashable, Iterable[Hashable]]] = None,
    axis=None,
) -> "DataArray":
    """ Calcualte the sum of weights, accounting for missing values """

    # we need to mask DATA values that are nan; else the weights are wrong
    mask = where(da.notnull(), 1, 0)  # binary mask

    # need to infer dims as we use `dot`
    dims = _maybe_get_all_dims(dim, da.dims, weights.dims)

    # use `dot` to avoid creating large da's
    sum_of_weights = dot(mask, weights, dims=dims)

    # find all weights that are valid (not 0)
    valid_weights = sum_of_weights != 0.0

    # set invalid weights to nan
    return sum_of_weights.where(valid_weights)


def _weighted_sum(
    da: "DataArray",
    weights: "DataArray",
    dim: Optional[Union[Hashable, Iterable[Hashable]]] = None,
    axis=None,
    skipna: Optional[bool] = None,
    **kwargs
) -> "DataArray":
    """Reduce a DataArray by a by a weighted `sum` along some dimension(s)."""

    # need to infer dims as we use `dot`
    dims = _maybe_get_all_dims(dim, da.dims, weights.dims)

    # use `dot` to avoid creating large da's

    # need to mask invalid DATA as dot does not implement skipna
    if skipna or skipna is None:
        return where(da.isnull(), 0.0, da).dot(weights, dims=dims)

    return dot(da, weights, dims=dims)


def _weighted_mean(
    da: "DataArray",
    weights: "DataArray",
    dim: Optional[Union[Hashable, Iterable[Hashable]]] = None,
    axis=None,
    skipna: Optional[bool] = None,
    **kwargs
) -> "DataArray":
    """Reduce a DataArray by a weighted `mean` along some dimension(s)."""

    # get weighted sum
    weighted_sum = _weighted_sum(
        da, weights, dim=dim, axis=axis, skipna=skipna, **kwargs
    )

    # get the sum of weights
    sum_of_weights = _sum_of_weights(da, weights, dim=dim, axis=axis)

    # calculate weighted mean
    return weighted_sum / sum_of_weights


class Weighted:
    """A object that implements weighted operations.

    You should create a Weighted object by using the `DataArray.weighted` or
    `Dataset.weighted` methods.

    See Also
    --------
    Dataset.weighted
    DataArray.weighted
    """

    __slots__ = ("obj", "weights")

    @overload
    def __init__(self, obj: "DataArray", weights: "DataArray") -> None:
        ...

    @overload
    def __init__(self, obj: "Dataset", weights: "DataArray") -> None:
        ...

    def __init__(self, obj, weights) -> None:
        """
        Weighted operations for DataArray.

        Parameters
        ----------
        obj : DataArray or Dataset
            Object over which the weighted reduction operation is applied.
        weights : DataArray
            An array of weights associated with the values in this obj.
            Each value in the obj contributes to the reduction operation
            according to its associated weight.

        Note
        ----
        Missing values in the weights are replaced with 0 (i.e. no weight).

        """

        from .dataarray import DataArray

        msg = "'weights' must be a DataArray"
        assert isinstance(weights, DataArray), msg

        self.obj = obj
        self.weights = weights.fillna(0)

    def __repr__(self):
        """provide a nice str repr of our weighted object"""

        msg = "{klass} with weights along dimensions: {weight_dims}"
        return msg.format(
            klass=self.__class__.__name__, weight_dims=", ".join(self.weights.dims)
        )


class DataArrayWeighted(Weighted):
    def sum_of_weights(
        self, dim: Optional[Union[Hashable, Iterable[Hashable]]] = None, axis=None
    ) -> "DataArray":

        return _sum_of_weights(self.obj, self.weights, dim=dim, axis=axis)

    def sum(
        self,
        dim: Optional[Union[Hashable, Iterable[Hashable]]] = None,
        axis=None,
        skipna: Optional[bool] = None,
        **kwargs
    ) -> "DataArray":

        return _weighted_sum(
            self.obj, self.weights, dim=dim, axis=axis, skipna=skipna, **kwargs
        )

    def mean(
        self,
        dim: Optional[Union[Hashable, Iterable[Hashable]]] = None,
        axis=None,
        skipna: Optional[bool] = None,
        **kwargs
    ) -> "DataArray":

        return _weighted_mean(
            self.obj, self.weights, dim=dim, axis=axis, skipna=skipna, **kwargs
        )


# add docstrings
DataArrayWeighted.sum_of_weights.__doc__ = _SUM_OF_WEIGHTS_DOCSTRING.format(
    cls="DataArray"
)
DataArrayWeighted.mean.__doc__ = _WEIGHTED_REDUCE_DOCSTRING_TEMPLATE.format(
    cls="DataArray", fcn="mean"
)
DataArrayWeighted.sum.__doc__ = _WEIGHTED_REDUCE_DOCSTRING_TEMPLATE.format(
    cls="DataArray", fcn="sum"
)


class DatasetWeighted(Weighted):
    def _dataset_implementation(self, func, **kwargs) -> "Dataset":
        
        from .dataset import Dataset

        weighted = {}
        for key, da in self.obj.data_vars.items():

            weighted[key] = func(da, self.weights, **kwargs)

        return Dataset(weighted, coords=self.obj.coords)

    def sum_of_weights(
        self,
        dim: Optional[Union[Hashable, Iterable[Hashable]]] = None,
        axis=None,
        skipna: Optional[bool] = None,
    ) -> "Dataset":

        return self._dataset_implementation(_sum_of_weights, dim=dim, axis=axis)

    def sum(
        self,
        dim: Optional[Union[Hashable, Iterable[Hashable]]] = None,
        axis=None,
        skipna: Optional[bool] = None,
        **kwargs
    ) -> "Dataset":

        return self._dataset_implementation(
            _weighted_sum, dim=dim, axis=axis, skipna=skipna, **kwargs
        )

    def mean(
        self,
        dim: Optional[Union[Hashable, Iterable[Hashable]]] = None,
        axis=None,
        skipna: Optional[bool] = None,
        **kwargs
    ) -> "Dataset":

        return self._dataset_implementation(
            _weighted_mean, dim=dim, axis=axis, skipna=skipna, **kwargs
        )


# add docstring
DatasetWeighted.sum_of_weights.__doc__ = _SUM_OF_WEIGHTS_DOCSTRING.format(cls="Dataset")
DatasetWeighted.mean.__doc__ = _WEIGHTED_REDUCE_DOCSTRING_TEMPLATE.format(
    cls="Dataset", fcn="mean"
)
DatasetWeighted.sum.__doc__ = _WEIGHTED_REDUCE_DOCSTRING_TEMPLATE.format(
    cls="Dataset", fcn="sum"
)
