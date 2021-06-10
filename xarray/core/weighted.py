from typing import TYPE_CHECKING, Generic, Hashable, Iterable, Tuple, List, Optional, TypeVar, Union

from . import duck_array_ops
from .computation import dot
from .pycompat import is_duck_dask_array

if TYPE_CHECKING:
    from .common import DataWithCoords  # noqa: F401
    from .dataarray import DataArray, Dataset
    from .computation import _ALLOWED_BINS_TYPES

T_DataWithCoords = TypeVar("T_DataWithCoords", bound="DataWithCoords")


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
    Calculate the sum of weights, accounting for missing values in the data

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

_WEIGHTED_HIST_DOCSTRING = """
    Weighted histogram applied along specified dimensions.

    If the supplied arguments are chunked dask arrays it will use
    `dask.array.blockwise` internally to parallelize over all chunks.

    Parameters
    ----------
    dim : tuple of strings, optional
        Dimensions over which which the histogram is computed. The default is to
        compute the histogram of the flattened {cls}. i.e. over all dimensions.
    bins :  int or array_like or a list of ints or arrays, or list of DataArrays, optional
        If a list, there should be one entry for each item in ``args``.
        The bin specification:

          * If int, the number of bins for all arguments in ``args``.
          * If array_like, the bin edges for all arguments in ``args``.
          * If a list of ints, the number of bins  for every argument in ``args``.
          * If a list arrays, the bin edges for each argument in ``args``
            (required format for Dask inputs).
          * A combination [int, array] or [array, int], where int
            is the number of bins and array is the bin edges.
          * If a list of DataArrays, the bins for each argument in ``args``
            The DataArrays can be multidimensional, but must not have any
            dimensions shared with the `dim` argument.

        When bin edges are specified, all but the last (righthand-most) bin include
        the left edge and exclude the right edge. The last bin includes both edges.

        A ``TypeError`` will be raised if ``args`` contains dask arrays and
        ``bins`` are not specified explicitly as a list of arrays.
    density : bool, optional
        If ``False``, the result will contain the number of samples in
        each bin. If ``True``, the result is the value of the
        probability *density* function at the bin, normalized such that
        the *integral* over the range is 1. Note that the sum of the
        histogram values will not be equal to 1 unless bins of unit
        width are chosen; it is not a probability *mass* function.
    keep_attrs : bool, optional
        If True, the attributes (``attrs``) will be copied from the original
        object to the new one.  If False (default), the new object will be
        returned without attributes.

    Returns
    -------
    hist : xarray.DataArray
        A xarray.DataArray object which contains the values of the histogram. See
        `density` and `weights` for a description of the possible semantics.

        The returned dataarray will have one additional coordinate for each
        variable supplied, named as `var_bins`, which contains the positions
        of the centres of each bin.

    Examples
    --------

    See Also
    --------
    xarray.hist
    DataArray.hist
    Dataset.hist
    numpy.histogramdd
    dask.array.blockwise
    """


class Weighted(Generic[T_DataWithCoords]):
    """An object that implements weighted operations.

    You should create a Weighted object by using the ``DataArray.weighted`` or
    ``Dataset.weighted`` methods.

    See Also
    --------
    Dataset.weighted
    DataArray.weighted
    """

    __slots__ = ("obj", "weights")

    def __init__(self, obj: T_DataWithCoords, weights: "DataArray"):
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

        self.obj: T_DataWithCoords = obj
        self.weights: "DataArray" = weights

    def _check_dim(self, dim: Optional[Union[Hashable, Iterable[Hashable]]]):
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
        da: "DataArray",
        weights: "DataArray",
        dim: Optional[Union[Hashable, Iterable[Hashable]]] = None,
        skipna: Optional[bool] = None,
    ) -> "DataArray":
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
        self, da: "DataArray", dim: Optional[Union[Hashable, Iterable[Hashable]]] = None
    ) -> "DataArray":
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

    def _weighted_sum(
        self,
        da: "DataArray",
        dim: Optional[Union[Hashable, Iterable[Hashable]]] = None,
        skipna: Optional[bool] = None,
    ) -> "DataArray":
        """Reduce a DataArray by a by a weighted ``sum`` along some dimension(s)."""

        return self._reduce(da, self.weights, dim=dim, skipna=skipna)

    def _weighted_mean(
        self,
        da: "DataArray",
        dim: Optional[Union[Hashable, Iterable[Hashable]]] = None,
        skipna: Optional[bool] = None,
    ) -> "DataArray":
        """Reduce a DataArray by a weighted ``mean`` along some dimension(s)."""

        weighted_sum = self._weighted_sum(da, dim=dim, skipna=skipna)

        sum_of_weights = self._sum_of_weights(da, dim=dim)

        return weighted_sum / sum_of_weights

    def _implementation(self, func, dim, **kwargs):

        raise NotImplementedError("Use `Dataset.weighted` or `DataArray.weighted`")

    def sum_of_weights(
        self,
        dim: Optional[Union[Hashable, Iterable[Hashable]]] = None,
        keep_attrs: Optional[bool] = None,
    ) -> T_DataWithCoords:

        return self._implementation(
            self._sum_of_weights, dim=dim, keep_attrs=keep_attrs
        )

    def sum(
        self,
        dim: Optional[Union[Hashable, Iterable[Hashable]]] = None,
        skipna: Optional[bool] = None,
        keep_attrs: Optional[bool] = None,
    ) -> T_DataWithCoords:

        return self._implementation(
            self._weighted_sum, dim=dim, skipna=skipna, keep_attrs=keep_attrs
        )

    def mean(
        self,
        dim: Optional[Union[Hashable, Iterable[Hashable]]] = None,
        skipna: Optional[bool] = None,
        keep_attrs: Optional[bool] = None,
    ) -> T_DataWithCoords:

        return self._implementation(
            self._weighted_mean, dim=dim, skipna=skipna, keep_attrs=keep_attrs
        )

    def hist(
        self,
        dim : Union[Hashable, Iterable[Hashable]] = None,
        bins : Union[_ALLOWED_BINS_TYPES, List[_ALLOWED_BINS_TYPES]] = None,
        range : Union[Tuple[float, float], List[Tuple[float, float]]] = None,
        density : bool = False,
        keep_attrs : bool = None,
    ) -> T_DataWithCoords:
        return self.obj.hist(
            dim=dim,
            bins=bins,
            range=range,
            weights=self.weights,
            density=density,
            keep_attrs=keep_attrs,
        )

    def __repr__(self):
        """provide a nice str repr of our Weighted object"""

        klass = self.__class__.__name__
        weight_dims = ", ".join(self.weights.dims)
        return f"{klass} with weights along dimensions: {weight_dims}"


class DataArrayWeighted(Weighted["DataArray"]):
    def _implementation(self, func, dim, **kwargs) -> "DataArray":

        self._check_dim(dim)

        dataset = self.obj._to_temp_dataset()
        dataset = dataset.map(func, dim=dim, **kwargs)
        return self.obj._from_temp_dataset(dataset)


class DatasetWeighted(Weighted["Dataset"]):
    def _implementation(self, func, dim, **kwargs) -> "Dataset":

        self._check_dim(dim)

        return self.obj.map(func, dim=dim, **kwargs)


def _inject_docstring(cls, cls_name):

    cls.sum_of_weights.__doc__ = _SUM_OF_WEIGHTS_DOCSTRING.format(cls=cls_name)

    cls.hist.__doc__ = _WEIGHTED_HIST_DOCSTRING.format(cls=cls_name)

    cls.sum.__doc__ = _WEIGHTED_REDUCE_DOCSTRING_TEMPLATE.format(
        cls=cls_name, fcn="sum", on_zero="0"
    )

    cls.mean.__doc__ = _WEIGHTED_REDUCE_DOCSTRING_TEMPLATE.format(
        cls=cls_name, fcn="mean", on_zero="NaN"
    )


_inject_docstring(DataArrayWeighted, "DataArray")
_inject_docstring(DatasetWeighted, "Dataset")
