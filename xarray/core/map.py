from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Optional
)

from .options import _get_keep_attrs
from .utils import (
    maybe_wrap_array,
)
from .. import  Dataset, DataArray

def map(
    datasets: Iterable[Any],
    func: Callable,
    keep_attrs: Optional[int] = None,
    args: Iterable[Any] = (),
    kwargs: Dict = None,
    ) -> "Dataset":
    """Apply a function to each variable in the provided dataset(s).

    The function may take several DataArrays as inputs. The number of DataArrays
    passed to the function will be equal to the length of the datasets variable.

    It is assumed that the Datasets in the datasets variable share common data variable names.
    If the same variable name is present in all Datasets, then the function will be performed on
    those DataArrays

    Parameters
    ----------
    datasets : sequence of Datasets
        The Dataset whose variables will be the input DataArrays of the function
    func : callable
        Function which can be called in the form `func(x,y,z, ..., *args, **kwargs)`
        to transform each sequence of DataArrays `x`, `y`, `z` in the datasets into another
        DataArray.
    keep_attrs : int or bool, optional
        If False, the new object will be returned without attributes.
        If is an integer between 0 and len(datasets-1), it will give the index of the Dataset in
        datasets parameter whose attributes needs to be copied
    args : tuple, optional
        Positional arguments passed on to `func`.
    kwargs : dict, optional
        Keyword arguments passed on to `func`.

    Returns
    -------
    applied : Dataset
        Resulting dataset from applying ``func`` to each tuple of data variables.

    Examples
    --------
    >>> da = xr.DataArray(np.random.randn(2, 3))
    >>> ds1 = xr.Dataset({"foo": da, "bar": ("x", [-1, 2])})
    >>> ds2 = xr.Dataset({"foo": da+1, "bar": ("x", [-1, 2])})
    >>> ds1
    <xarray.Dataset>
    Dimensions:  (dim_0: 2, dim_1: 3, x: 2)
    Dimensions without coordinates: dim_0, dim_1, x
    Data variables:
        foo      (dim_0, dim_1) float64 1.764 0.4002 0.9787 2.241 1.868 -0.9773
        bar      (x) int64 -1 2
    >>> ds2
    <xarray.Dataset>
    Dimensions:  (dim_0: 2, dim_1: 3, x: 2)
    Dimensions without coordinates: dim_0, dim_1, x
    Data variables:
        foo      (dim_0, dim_1) float64 2.764 1.4002 1.9787 3.241 2.868 0.0227
        bar      (x) int64 -1 2
    >>> f = lambda a, b: b-a
    >>> map([ds1, ds2], f)
    <xarray.Dataset>
    Dimensions:  (dim_0: 2, dim_1: 3, x: 2)
    Dimensions without coordinates: dim_0, dim_1, x
    Data variables:
        foo      (dim_0, dim_1) float64 1.0 1.0 1.0 1.0 1.0 1.0
        bar      (x) float64 0.0 0.0


    See Also
    --------
    Dataset.map
    """
    if kwargs is None:
        kwargs = {}
    variables = {}
    if len(datasets):
        shared_variable_names = set.intersection(*(set(ds.data_vars) for ds in datasets))
        for k in shared_variable_names:
            data_arrays  = [d[k] for d in datasets]
            v = maybe_wrap_array(datasets[0][k], func(*(data_arrays+list(args)), **kwargs))
            variables[k] = v

    if keep_attrs is None:
        keep_attrs = _get_keep_attrs(default=False)
        if keep_attrs:
            keep_attrs = 0

    if keep_attrs is not False:
        attrs = datasets[keep_attrs].attrs
    else:
        attrs = None

    return Dataset(variables, attrs=attrs)


