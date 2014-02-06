
class _DataWrapperMixin(object):
    @property
    def dtype(self):
        return self._data.dtype

    @property
    def shape(self):
        return self._data.shape

    @property
    def size(self):
        return self._data.size

    @property
    def ndim(self):
        return self._data.ndim

    def __len__(self):
        return len(self._data)

    def __nonzero__(self):
        return bool(self._data)

    def __float__(self):
        return float(self._data)

    def __int__(self):
        return int(self._data)

    def __complex__(self):
        return complex(self._data)

    def __long__(self):
        return long(self._data)

    # adapted from pandas.NDFrame
    # https://github.com/pydata/pandas/blob/master/pandas/core/generic.py#L699

    def __array__(self, dtype=None):
        return self.data

    # @property
    # def __array_interface__(self):
    #    data = self.data
    #    return dict(typestr=data.dtype.str, shape=data.shape, data=data)

    _collapse_method_docstring = \
        """Collapse this {cls}'s data' by applying `{name}` along some
        dimension(s)

        Parameters
        ----------
        dimension : str or sequence of str, optional
            Dimension(s) over which to repeatedly apply `{name}`.
        axis : int or sequence of int, optional
            Axis(es) over which to repeatedly apply `{name}`. Only one of the
            'dimension' and 'axis' arguments can be supplied. If neither are
            supplied, then `{name}` is calculated over the flattened array
            (by calling `{name}(x)` without an axis argument).
        **kwargs : dict
            Additional keyword arguments passed on to `{name}`.

        Note
        ----
        If this method is called with multiple dimensions (or axes, which are
        converted into dimensions), then `{name}` is performed repeatedly along
        each dimension in turn from left to right.

        Returns
        -------
        collapsed : {cls}
            New {cls} object with `{name}` applied to its data and the
            indicated dimension(s) removed.
        """

    @classmethod
    def _collapse_method(cls, f, name=None, module=None):
        def func(self, dimension=None, axis=None, **kwargs):
            return self.collapse(f, dimension, axis, **kwargs)
        if name is None:
            name = f.__name__
        func.__name__ = name
        func.__doc__ = cls._collapse_method_docstring.format(
            name=('' if module is None else module + '.') + name,
            cls=cls.__name__)
        return func

    # we want something like this, right?
    # def apply(self, func, dimension=None, axis=None, **kwargs):
    #     if dimension is not None and axis is not None:
    #         raise ValueError("cannot supply both 'axis' and 'dimension' "
    #                          "arguments")
    #     if axis is None:
    #         axis = self.dimensions.index(dimension)
    #     f = self._unary_op(partial(func, axis=axis, **kwargs))
    #     return f(self)

