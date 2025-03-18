"""Mixin class to add Dataset methods to DataTree"""

# This file was generated using xarray.util.generate_datatree_methods. Do not edit manually.

from __future__ import annotations

from collections.abc import Hashable, Iterable
from functools import wraps
from typing import Literal, Self

from xarray.core.dataset import Dataset
from xarray.core.datatree_mapping import map_over_datasets
from xarray.core.types import ErrorOptionsWithWarn


def _wrap_dataset_method(to_apply):
    def wrap_method(func):
        @wraps(func)
        def inner(self, *args, **kwargs):
            return map_over_datasets(to_apply, self, *args, kwargs=kwargs)

        return inner

    return wrap_method


class TreeMethodsMixin:
    __slots__ = ()

    @_wrap_dataset_method(Dataset.argmax)
    def argmax(self, dim: Hashable | None = None, **kwargs) -> Self:
        """Indices of the maxima of the member variables.

        If there are multiple maxima, the indices of the first one found will be
        returned.

        Parameters
        ----------
        dim : str, optional
            The dimension over which to find the maximum. By default, finds maximum over
            all dimensions - for now returning an int for backward compatibility, but
            this is deprecated, in future will be an error, since DataArray.argmax will
            return a dict with indices for all dimensions, which does not make sense for
            a Dataset.
        keep_attrs : bool, optional
            If True, the attributes (`attrs`) will be copied from the original
            object to the new one.  If False (default), the new object will be
            returned without attributes.
        skipna : bool, optional
            If True, skip missing values (as marked by NaN). By default, only
            skips missing values for float dtypes; other dtypes either do not
            have a sentinel missing value (int) or skipna=True has not been
            implemented (object, datetime64 or timedelta64).

        Returns
        -------
        result : Dataset

        Examples
        --------

        >>> dataset = xr.Dataset(
        ...     {
        ...         "math_scores": (
        ...             ["student", "test"],
        ...             [[90, 85, 92], [78, 80, 85], [95, 92, 98]],
        ...         ),
        ...         "english_scores": (
        ...             ["student", "test"],
        ...             [[88, 90, 92], [75, 82, 79], [93, 96, 91]],
        ...         ),
        ...     },
        ...     coords={
        ...         "student": ["Alice", "Bob", "Charlie"],
        ...         "test": ["Test 1", "Test 2", "Test 3"],
        ...     },
        ... )

        # Indices of the maximum values along the 'student' dimension are calculated

        >>> argmax_indices = dataset.argmax(dim="test")

        >>> argmax_indices
        <xarray.Dataset> Size: 132B
        Dimensions:         (student: 3)
        Coordinates:
          * student         (student) <U7 84B 'Alice' 'Bob' 'Charlie'
        Data variables:
            math_scores     (student) int64 24B 2 2 2
            english_scores  (student) int64 24B 2 1 1

        See Also
        --------
        DataArray.argmax

        """
        # NOTE: the method is executed in the wrapper
        pass

    @_wrap_dataset_method(Dataset.dropna)
    def dropna(
        self,
        dim: Hashable,
        *,
        how: Literal["any", "all"] = "any",
        thresh: int | None = None,
        subset: Iterable[Hashable] | None = None,
    ) -> Self:
        """Returns a new dataset with dropped labels for missing values along
        the provided dimension.

        Parameters
        ----------
        dim : hashable
            Dimension along which to drop missing values. Dropping along
            multiple dimensions simultaneously is not yet supported.
        how : {"any", "all"}, default: "any"
            - any : if any NA values are present, drop that label
            - all : if all values are NA, drop that label

        thresh : int or None, optional
            If supplied, require this many non-NA values (summed over all the subset variables).
        subset : iterable of hashable or None, optional
            Which variables to check for missing values. By default, all
            variables in the dataset are checked.

        Examples
        --------
        >>> dataset = xr.Dataset(
        ...     {
        ...         "temperature": (
        ...             ["time", "location"],
        ...             [[23.4, 24.1], [np.nan, 22.1], [21.8, 24.2], [20.5, 25.3]],
        ...         )
        ...     },
        ...     coords={"time": [1, 2, 3, 4], "location": ["A", "B"]},
        ... )
        >>> dataset
        <xarray.Dataset> Size: 104B
        Dimensions:      (time: 4, location: 2)
        Coordinates:
          * time         (time) int64 32B 1 2 3 4
          * location     (location) <U1 8B 'A' 'B'
        Data variables:
            temperature  (time, location) float64 64B 23.4 24.1 nan ... 24.2 20.5 25.3

        Drop NaN values from the dataset

        >>> dataset.dropna(dim="time")
        <xarray.Dataset> Size: 80B
        Dimensions:      (time: 3, location: 2)
        Coordinates:
          * time         (time) int64 24B 1 3 4
          * location     (location) <U1 8B 'A' 'B'
        Data variables:
            temperature  (time, location) float64 48B 23.4 24.1 21.8 24.2 20.5 25.3

        Drop labels with any NaN values

        >>> dataset.dropna(dim="time", how="any")
        <xarray.Dataset> Size: 80B
        Dimensions:      (time: 3, location: 2)
        Coordinates:
          * time         (time) int64 24B 1 3 4
          * location     (location) <U1 8B 'A' 'B'
        Data variables:
            temperature  (time, location) float64 48B 23.4 24.1 21.8 24.2 20.5 25.3

        Drop labels with all NAN values

        >>> dataset.dropna(dim="time", how="all")
        <xarray.Dataset> Size: 104B
        Dimensions:      (time: 4, location: 2)
        Coordinates:
          * time         (time) int64 32B 1 2 3 4
          * location     (location) <U1 8B 'A' 'B'
        Data variables:
            temperature  (time, location) float64 64B 23.4 24.1 nan ... 24.2 20.5 25.3

        Drop labels with less than 2 non-NA values

        >>> dataset.dropna(dim="time", thresh=2)
        <xarray.Dataset> Size: 80B
        Dimensions:      (time: 3, location: 2)
        Coordinates:
          * time         (time) int64 24B 1 3 4
          * location     (location) <U1 8B 'A' 'B'
        Data variables:
            temperature  (time, location) float64 48B 23.4 24.1 21.8 24.2 20.5 25.3

        Returns
        -------
        Dataset

        See Also
        --------
        DataArray.dropna
        pandas.DataFrame.dropna
        """
        # NOTE: the method is executed in the wrapper
        pass

    @_wrap_dataset_method(Dataset.transpose)
    def transpose(
        self, *dim: Hashable, missing_dims: ErrorOptionsWithWarn = "raise"
    ) -> Self:
        """Return a new Dataset object with all array dimensions transposed.

        Although the order of dimensions on each array will change, the dataset
        dimensions themselves will remain in fixed (sorted) order.

        Parameters
        ----------
        *dim : hashable, optional
            By default, reverse the dimensions on each array. Otherwise,
            reorder the dimensions to this order.
        missing_dims : {"raise", "warn", "ignore"}, default: "raise"
            What to do if dimensions that should be selected from are not present in the
            Dataset:
            - "raise": raise an exception
            - "warn": raise a warning, and ignore the missing dimensions
            - "ignore": ignore the missing dimensions

        Returns
        -------
        transposed : Dataset
            Each array in the dataset (including) coordinates will be
            transposed to the given order.

        Notes
        -----
        This operation returns a view of each array's data. It is
        lazy for dask-backed DataArrays but not for numpy-backed DataArrays
        -- the data will be fully loaded into memory.

        See Also
        --------
        numpy.transpose
        DataArray.transpose
        """
        # NOTE: the method is executed in the wrapper
        pass
