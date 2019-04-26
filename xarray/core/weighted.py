
class DataArrayWeighted(object):  
    def __init__(self, obj, weights):
        """
        Weighted operations for DataArray.

        Parameters
        ----------
        obj : DataArray
            Object to window.
        weights : DataArray
            An array of weights associated with the values in this Dataset.
            Each value in a contributes to the average according to its
            associated weight.

        Note
        ----
        Missing values in the weights are treated as 0 (i.e. no weight).

        """
        
        super(DataArrayWeighted, self).__init__()
        
        from .dataarray import DataArray

        msg = "'weights' must be a DataArray"
        assert isinstance(weights, DataArray)

        self.obj = obj
        self.weights = weights

    def sum_of_weights(self, dim=None, axis=None):
        """
        Calcualte the sum of weights accounting for missing values

        Parameters
        ----------
        dim : str or sequence of str, optional
            Dimension(s) over which to sum the weights.
        axis : int or sequence of int, optional
            Axis(es) over which to sum the weights. Only one of the 'dim' and
            'axis' arguments can be supplied. If neither are supplied, then
            the weights are summed over all axes.

        """

        # we need to mask values that are nan; else the weights are wrong
        notnull = self.obj.notnull()
        
        return self.weights.where(notnull).sum(dim=dim, axis=axis, skipna=True)
        

    def mean(self, dim=None, axis=None, skipna=None, **kwargs):
        """
        Reduce this DataArray's data by a weighted `mean` along some dimension(s).

        Parameters
        ----------
        dim : str or sequence of str, optional
            Dimension(s) over which to apply the weighted `mean`.
        axis : int or sequence of int, optional
            Axis(es) over which to apply the weighted `mean`. Only one of the
            'dim'and 'axis' arguments can be supplied. If neither are supplied,
            then the weighted `mean` is calculated over all axes.
        skipna : bool, optional
            If True, skip missing values (as marked by NaN). By default, only
            skips missing values for float dtypes; other dtypes either do not
            have a sentinel missing value (int) or skipna=True has not been
            implemented (object, datetime64 or timedelta64).
            Note: Missing values in the weights are always skipped.
        keep_attrs : bool, optional
            If True, the attributes (`attrs`) will be copied from the original
            object to the new one.  If False (default), the new object will be
            returned without attributes.
        **kwargs : dict
            Additional keyword arguments passed on to the appropriate array
            function for calculating `mean` on this object's data.

        Returns
        -------
        reduced : DataArray
            New DataArray object with weighted `mean` applied to its data and
            the indicated dimension(s) removed.
        """

        # get the sum of weights of the dims
        sum_of_weights = self.sum_of_weights(dim=dim, axis=axis)

        # normalize weights to 1
        w = self.weights / sum_of_weights
        
        obj = self.obj

        # check if invalid values are masked by weights that are 0
        # e.g. values = [1 NaN]; weights = [1, 0], should return 1
        # if not skipna:
        #     # w = w.fillna(0)
        #     sel = ((w.isnull()) & (obj.isnull()))
        #     if sel.any():
        #         obj = obj.where(sel, 0)


        w = w.fillna(0)

        # calculate weighted mean
        weighted = (obj * w).sum(dim, axis=axis, skipna=skipna, **kwargs)

        # set to NaN if sum_of_weights is zero
        invalid_weights = sum_of_weights == 0
        return weighted.where(~ invalid_weights)
