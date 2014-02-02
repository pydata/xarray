import numpy as np


class _DataWrapperMixin(object):
    @property
    def data(self):
        """
        The variable's data as a numpy.ndarray
        """
        if not isinstance(self._data, np.ndarray):
            self._data = np.asarray(self._data[...])
        return self._data

    @data.setter
    def data(self, value):
        value = np.asarray(value)
        if value.shape != self.shape:
            raise ValueError("replacement data must match the Variable's "
                             "shape")
        self._data = value

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

