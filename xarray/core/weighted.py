

_doc_ = """
    Reduce this DataArray's data by a weighted `{fcn}` along some dimension(s).

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
    reduced : DataArray
        New DataArray object with weighted `{fcn}` applied to its data and
        the indicated dimension(s) removed.
    """


class DataArrayWeighted(object):
    def __init__(self, obj, weights):
        """
        Weighted operations for DataArray.

        Parameters
        ----------
        obj : DataArray
            Object over which the weighted reduction operation is applied.
        weights : DataArray
            An array of weights associated with the values in this Dataset.
            Each value in the DataArray contributes to the reduction operation
            according to its associated weight.

        Note
        ----
        Missing values in the weights are replaced with 0 (i.e. no weight).

        """

        super(DataArrayWeighted, self).__init__()

        from .dataarray import DataArray

        msg = "'weights' must be a DataArray"
        assert isinstance(weights, DataArray), msg

        self.obj = obj
        self.weights = weights.fillna(0)

    def sum_of_weights(self, dim=None, axis=None):
        """
        Calcualte the sum of weights, accounting for missing values

        Parameters
        ----------
        dim : str or sequence of str, optional
            Dimension(s) over which to sum the weights.
        axis : int or sequence of int, optional
            Axis(es) over which to sum the weights. Only one of the 'dim' and
            'axis' arguments can be supplied. If neither are supplied, then
            the weights are summed over all axes.

        """

        # we need to mask DATA values that are nan; else the weights are wrong
        masked_weights = self.weights.where(self.obj.notnull())

        sum_of_weights = masked_weights.sum(dim=dim, axis=axis, skipna=True)

        # find all weights that are valid (not 0)
        valid_weights = sum_of_weights != 0.

        # set invalid weights to nan
        return sum_of_weights.where(valid_weights)

    def sum(self, dim=None, axis=None, skipna=None, **kwargs):

        # calculate weighted sum
        return (self.obj * self.weights).sum(dim, axis=axis, skipna=skipna,
                                             **kwargs)

    def mean(self, dim=None, axis=None, skipna=None, **kwargs):

        # get the sum of weights
        sum_of_weights = self.sum_of_weights(dim=dim, axis=axis)

        # get weighted sum
        weighted_sum = self.sum(dim=dim, axis=axis, skipna=skipna, **kwargs)

        # calculate weighted mean
        return weighted_sum / sum_of_weights

    def __repr__(self):
        """provide a nice str repr of our weighted object"""

        msg = "{klass} with weights along dimensions: {weight_dims}"
        return msg.format(klass=self.__class__.__name__,
                          weight_dims=", ".join(self.weights.dims))


# add docstrings
DataArrayWeighted.mean.__doc__ = _doc_.format(fcn='mean')
DataArrayWeighted.sum.__doc__ = _doc_.format(fcn='sum')
