from typing import TYPE_CHECKING, Hashable, Iterable, Optional, Union, overload

from .computation import dot

if TYPE_CHECKING:
    from .dataarray import DataArray, Dataset

_WEIGHTED_REDUCE_DOCSTRING_TEMPLATE = """
    Reduce this {cls}'s data by a weighted `{fcn}` along some dimension(s).

    Parameters
    ----------
    dim : str or sequence of str, optional
        Dimension(s) over which to apply the weighted `{fcn}`.
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
    Calculate the sum of weights, accounting for missing values

    Parameters
    ----------
    dim : str or sequence of str, optional
        Dimension(s) over which to sum the weights.

    Returns
    -------
    reduced : {cls}
        New {cls} object with the sum of the weights over the given dimension.


    """


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

    @overload  # noqa: F811
    def __init__(self, obj: "Dataset", weights: "DataArray") -> None:
        ...

    def __init__(self, obj, weights):  # noqa: F811
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

        Note
        ----
        Weights can not contain missing values.

        """

        from .dataarray import DataArray

        msg = "'weights' must be a DataArray"
        assert isinstance(weights, DataArray), msg

        self.obj = obj

        if weights.isnull().any():
            raise ValueError("`weights` cannot contain missing values.")

        self.weights = weights

    def _sum_of_weights(
        self,
        da: "DataArray",
        dim: Optional[Union[Hashable, Iterable[Hashable]]] = None,
    ) -> "DataArray":
        """ Calculate the sum of weights, accounting for missing values """

        # we need to mask data values that are nan; else the weights are wrong
        mask = da.notnull()

        # need to infer dims as we use `dot`
        if dim is None:
            dim = ...

        # use `dot` to avoid creating large DataArrays (if da and weights do not share all dims)
        sum_of_weights = dot(mask, self.weights, dims=dim)

        # find all weights that are valid (not 0)
        valid_weights = sum_of_weights != 0.0

        # set invalid weights to nan
        return sum_of_weights.where(valid_weights)

    def _weighted_sum(
        self,
        da: "DataArray",
        dim: Optional[Union[Hashable, Iterable[Hashable]]] = None,
        skipna: Optional[bool] = None,
    ) -> "DataArray":
        """Reduce a DataArray by a by a weighted `sum` along some dimension(s)."""

        # need to infer dims as we use `dot`
        if dim is None:
            dim = ...

        # use `dot` to avoid creating large DataArrays

        # need to mask invalid DATA as dot does not implement skipna
        if skipna or (skipna is None and da.dtype.kind in "cfO"):
            return dot(da.fillna(0.0), self.weights, dims=dim)

        return dot(da, self.weights, dims=dim)

    def _weighted_mean(
        self,
        da: "DataArray",
        dim: Optional[Union[Hashable, Iterable[Hashable]]] = None,
        skipna: Optional[bool] = None,
    ) -> "DataArray":
        """Reduce a DataArray by a weighted `mean` along some dimension(s)."""

        # get weighted sum
        weighted_sum = self._weighted_sum(da, dim=dim, skipna=skipna)

        # get the sum of weights
        sum_of_weights = self._sum_of_weights(da, dim=dim)

        # calculate weighted mean
        return weighted_sum / sum_of_weights

    def _implementation(self, func, **kwargs):

        msg = "Use 'Dataset.weighted' or 'DataArray.weighted'"
        raise NotImplementedError(msg)

    def sum_of_weights(
        self, dim: Optional[Union[Hashable, Iterable[Hashable]]] = None
    ) -> Union["DataArray", "Dataset"]:

        return self._implementation(self._sum_of_weights, dim=dim)

    def sum(
        self,
        dim: Optional[Union[Hashable, Iterable[Hashable]]] = None,
        skipna: Optional[bool] = None,
    ) -> Union["DataArray", "Dataset"]:

        return self._implementation(self._weighted_sum, dim=dim, skipna=skipna)

    def mean(
        self,
        dim: Optional[Union[Hashable, Iterable[Hashable]]] = None,
        skipna: Optional[bool] = None,
    ) -> Union["DataArray", "Dataset"]:

        return self._implementation(self._weighted_mean, dim=dim, skipna=skipna)

    def __repr__(self):
        """provide a nice str repr of our Weighted object"""

        msg = "{klass} with weights along dimensions: {weight_dims}"
        return msg.format(
            klass=self.__class__.__name__, weight_dims=", ".join(self.weights.dims),
        )


class DataArrayWeighted(Weighted):
    def _implementation(self, func, **kwargs):

        return func(self.obj, **kwargs)


class DatasetWeighted(Weighted):
    def _implementation(self, func, **kwargs) -> "Dataset":

        from .dataset import Dataset

        weighted = {}
        for key, da in self.obj.data_vars.items():

            weighted[key] = func(da, **kwargs)

        return Dataset(weighted, coords=self.obj.coords)


def _inject_docstring(cls, cls_name):

    cls.sum_of_weights.__doc__ = _SUM_OF_WEIGHTS_DOCSTRING.format(cls=cls_name)

    for operator in ["sum", "mean"]:
        getattr(cls, operator).__doc__ = _WEIGHTED_REDUCE_DOCSTRING_TEMPLATE.format(
            cls=cls_name, fcn=operator
        )


_inject_docstring(DataArrayWeighted, "DataArray")
_inject_docstring(DatasetWeighted, "Dataset")
